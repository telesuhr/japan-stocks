"""通過17銘柄の「いくら以下で買うか」を、各社固有の過去5年バリュエーション分布から逆算する。

考え方: 絶対的な「割安な株価」は存在しない。あるのは
  (A) その銘柄自身の平常レンジに対して安いか  ← 過去5年PBR/PERの分位
  (B) キャッシュフロー(配当)に対して安いか    ← 予想利回りアンカー
  (C) 需給的に踏まれにくい水準か              ← 200日線・52週安値
の3つ。3本の逆算価格を並べ、現値からの乖離まで出して指値の現実性を示す。

【計算の要点】調整後価格空間で組むと分割補正が自動で効く:
  shares(t) = shares_disc * r(t)/r_disc  (run.py と同じ関係)
  → PBR(t)  = adj_close(t) * (shares_disc/r_disc) / 自己資本
  → PER(t)  = adj_close(t) / (FEPS * r_disc)
  → 利回り(t)= FDivAnn * r_disc / adj_close(t) * 100
各定数は開示ごとに階段状に切り替わる。自己資本は TA×EqAR（Eq=純資産合計は非支配株主持分込みで誤り）。
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
from pathlib import Path
import numpy as np
import pandas as pd
from jstock import db

HERE = Path(__file__).resolve().parent
YEARS = 5
CODES = (pd.concat([pd.read_csv(HERE / "candidates_nonfin.csv"),
                    pd.read_csv(HERE / "candidates_fin.csv")])
         ["code"].astype(str).str.zfill(5).tolist())

print(f"[1] 価格 ({len(CODES)}銘柄)...")
px = db.read_sql("""
    SELECT code, date, close, adj_close FROM stocks_daily
    WHERE code = ANY(%(c)s) AND date >= %(s)s AND close > 0 AND adj_close > 0
""", {"c": CODES, "s": f"{2026 - YEARS - 1}-01-01"})
px["date"] = pd.to_datetime(px["date"])
AC = px.pivot(index="date", columns="code", values="adj_close").sort_index()
CL = px.pivot(index="date", columns="code", values="close").sort_index()
R = AC / CL
LAST = AC.index[-1]
WIN = LAST - pd.DateOffset(years=YEARS)

print("[2] fin_summary...")
fin = db.read_sql("""
    SELECT code, disc_date,
           NULLIF(payload->>'TA','')::float      ta,
           NULLIF(payload->>'EqAR','')::float    eqar,
           NULLIF(payload->>'Eq','')::float      eq,
           NULLIF(payload->>'ShOutFY','')::float shout,
           NULLIF(payload->>'TrShFY','')::float  trsh,
           NULLIF(payload->>'FEPS','')::float    feps,
           NULLIF(payload->>'FDivAnn','')::float fdiv
    FROM fin_summary
    WHERE code = ANY(%(c)s) AND doc_type LIKE '%%FinancialStatements%%'
      AND disc_date >= %(s)s
""", {"c": CODES, "s": f"{2026 - YEARS - 2}-01-01"})
fin["disc_date"] = pd.to_datetime(fin["disc_date"])
own = fin["ta"] * fin["eqar"]
fin["eq_own"] = np.where(np.isfinite(own) & (own > 0) & (own <= fin["eq"]), own, fin["eq"])
fin["sh"] = fin["shout"] - fin["trsh"].fillna(0)
fin = fin[(fin["eq_own"] > 0) & (fin["sh"] > 0)].sort_values("disc_date")

sm = db.read_sql("SELECT code5 code, name_ja FROM symbol_master").set_index("code")["name_ja"]
cal = AC.index

rows = []
for c in CODES:
    f = fin[fin["code"] == c]
    if f.empty or c not in AC.columns:
        continue
    r_at = R[c].reindex(cal).ffill()
    # 開示日時点の r（開示日以降で最初に値のある営業日）
    r_disc = r_at.reindex(f["disc_date"], method="bfill").values
    d = pd.DataFrame({
        "date": f["disc_date"].values,
        "s_adj": f["sh"].values / r_disc,          # PBR用の定数
        "eq_own": f["eq_own"].values,
        "e_adj": f["feps"].values * r_disc,        # PER用の定数
        "d_adj": f["fdiv"].values * r_disc,        # 利回り用の定数
    }).dropna(subset=["date"]).set_index("date").sort_index()
    d = d[~d.index.duplicated(keep="last")].reindex(cal, method="ffill")

    a = AC[c].reindex(cal)
    pbr = a * d["s_adj"] / d["eq_own"]
    per = a / d["e_adj"].where(d["e_adj"] > 0)
    yld = d["d_adj"] / a * 100

    m = (cal >= WIN) & pbr.notna()
    if m.sum() < 250:
        continue
    p_now, a_now = CL[c].iloc[-1], a.iloc[-1]
    # 逆算: 目標バリュエーションに一致する「現在の株価」= 現値 × (目標/現在値)
    def price_at(series, target):
        cur = series.iloc[-1]
        return np.nan if not np.isfinite(cur) or cur <= 0 else p_now * target / cur

    pb_hist, pe_hist = pbr[m], per[m]
    # 直近2年窓: 東証の資本コスト要請による構造的な再評価「後」のレンジ。
    # 5年窓は2021-22のディープバリュー期を含み、指値が非現実的に低く出るため両方見る。
    m2 = (cal >= LAST - pd.DateOffset(years=2)) & pbr.notna()
    pb2, pe2 = pbr[m2], per[m2]
    rows.append(dict(
        code=c, name=sm.get(c, c), price=p_now,
        pbr=pbr.iloc[-1], pbr_med=pb_hist.median(), per=per.iloc[-1], yld=yld.iloc[-1],
        pctile=(pb_hist < pbr.iloc[-1]).mean() * 100,      # 自社5年PBRの何%タイルに居るか
        pct2=(pb2 < pbr.iloc[-1]).mean() * 100,            # 直近2年での位置
        A=price_at(pbr, pb_hist.quantile(.25)),            # 自社5年PBR 25%タイル相当
        A2=price_at(per, pe_hist.quantile(.25)),           # 自社5年予想PER 25%タイル相当
        B=price_at(yld, 4.0) if yld.iloc[-1] > 0 else np.nan,  # 予想利回り4.0%相当
        # 実務用アンカー: 再評価後(2年)レンジの下位1/3
        P2=price_at(pbr, pb2.quantile(1 / 3)), E2=price_at(per, pe2.quantile(1 / 3)),
        sma200=CL[c].tail(200).mean(), low52=CL[c].tail(252).min(),
    ))

T = pd.DataFrame(rows)
# 【厳格】5年アンカー3本の中央値。ディープバリュー期を含むので基本的に届かない水準。
T["strict"] = T[["A", "A2", "B"]].median(axis=1)
# 【実務】再評価後2年のPBR/PER下位1/3 と 利回り4% の中央値。こちらを指値の主軸にする。
T["target"] = T[["P2", "E2", "B"]].median(axis=1)
T["disc%"] = (T["target"] / T["price"] - 1) * 100
T = T.sort_values("pctile")

O = pd.DataFrame({
    "コード": T["code"].str[:4], "銘柄": T["name"],
    "現値": T["price"].round(0).astype(int), "現PBR": T["pbr"].round(2),
    "5年PBR中央": T["pbr_med"].round(2),
    "5年内位置%": T["pctile"].round(0), "2年内位置%": T["pct2"].round(0),
    "利回4%相当": T["B"].round(-1), "200日線": T["sma200"].round(-1),
    "52週安値": T["low52"].round(0).astype(int),
    "指値(実務)": T["target"].round(-1), "現値比%": T["disc%"].round(1),
    "指値(厳格5y)": T["strict"].round(-1),
})
print("\n" + "=" * 140)
print(f"買い指値の逆算（基準 {LAST.date()} 終値）")
print("  5年内位置% = 自社の過去5年PBR分布での現在地。100に近いほど『自社史上まれに見る高評価』")
print("=" * 140)
print(O.to_string(index=False))
O.to_csv(HERE / "entry_levels.csv", index=False, encoding="utf-8-sig")
print(f"\n自社5年PBRで下位30%タイル以下＝本当に出遅れている銘柄: "
      f"{', '.join(T.loc[T['pctile'] <= 30, 'name'])}")
print("saved entry_levels.csv")
