"""
中長期・現物向け 割安株スクリーン（2026-08-17時点）。

既存知見の上に積む:
  - 20260719_pbr_below1_market_wide: PBR<1は「買い」(対市場+4.3%/年 t2.25)、東証要請後に効果拡大
  - 20260729_value_rotation_laggards: 足元は割安>割高のローテーション局面
  - 20260615_regional_bank_value_rate_hike: 質(ROE)オーバーレイがバリュートラップを回避

本スクリーンの追加点:
  (1) BPS を FY開示待ちにせず Eq/(ShOutFY-TrShFY) で四半期ごとに更新
  (2) 株式分割の補正（開示時点と現在の adj_close/close 比で1株当たり値を換算）
  (3) 会社予想(FEPS/FNP/FDivAnn)ベースの予想PER・予想ROE・予想利回り
  (4) 過去5期の最終赤字回数・増益率で「安いだけの罠」を除外
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
from pathlib import Path
import numpy as np
import pandas as pd
from jstock import db

HERE = Path(__file__).resolve().parent
FIN_SEC = ("銀行業", "保険業", "証券･商品先物取引業", "その他金融業")

# ---------------- 価格 ----------------
print("[1] 価格...")
px = db.read_sql("""
    SELECT code, date, close, adj_close, turnover_value
    FROM stocks_daily WHERE date >= '2025-07-01' AND close > 0 AND adj_close > 0
""")
px["date"] = pd.to_datetime(px["date"])
px = px.sort_values(["code", "date"])
AC = px.pivot(index="date", columns="code", values="adj_close")
last_date = px["date"].max()
print(f"  最終営業日 {last_date.date()}")

g = px.groupby("code")
snap = pd.DataFrame({
    "close": g["close"].last(),
    "adjc": g["adj_close"].last(),
    "adv": g["turnover_value"].apply(lambda s: s.tail(60).mean()),
})
snap["r_now"] = snap["adjc"] / snap["close"]
for lbl, d in [("ret1m", 21), ("ret3m", 63), ("ret6m", 126), ("ret12m", 252)]:
    snap[lbl] = AC.iloc[-1] / AC.shift(d).iloc[-1] - 1

# 開示日時点の adj/close 比（株式分割の補正用）
rr = px.assign(r=px["adj_close"] / px["close"])[["code", "date", "r"]]

# ---------------- 財務 ----------------
print("[2] fin_summary...")
fin = db.read_sql("""
    SELECT code, disc_date, cur_per_type, cur_per_en,
           NULLIF(payload->>'Eq','')::float      eq,
           NULLIF(payload->>'TA','')::float      ta,
           NULLIF(payload->>'ShOutFY','')::float shout,
           NULLIF(payload->>'TrShFY','')::float  trsh,
           NULLIF(payload->>'EqAR','')::float    eqar,
           NULLIF(payload->>'NP','')::float      np,
           NULLIF(payload->>'OP','')::float      op,
           NULLIF(payload->>'Sales','')::float   sales,
           NULLIF(payload->>'CashEq','')::float  cash,
           NULLIF(payload->>'FEPS','')::float    feps,
           NULLIF(payload->>'FNP','')::float     fnp,
           NULLIF(payload->>'FOP','')::float     fop,
           NULLIF(payload->>'FDivAnn','')::float fdiv,
           NULLIF(payload->>'EPS','')::float     eps
    FROM fin_summary
    WHERE doc_type LIKE '%%FinancialStatements%%' AND disc_date >= '2019-01-01'
""")
fin["disc_date"] = pd.to_datetime(fin["disc_date"])
fin = fin.sort_values("disc_date")

# 最新開示（予想・純資産の最新値）
latest = fin.groupby("code").tail(1).set_index("code")
# 直近FY実績（増益率の分母）＋ 過去5期の最終赤字回数
fy = fin[fin["cur_per_type"] == "FY"].copy()
fy_last = fy.groupby("code").tail(1).set_index("code")
hist = fy.groupby("code").tail(5)
q = hist.groupby("code").agg(loss_yrs=("np", lambda s: int((s < 0).sum())),
                             fy_n=("np", "size"),
                             np_5y_avg=("np", "mean"))

df = latest.join(q, how="left").join(
    fy_last[["np", "op", "sales"]].rename(columns={"np": "np_fy", "op": "op_fy", "sales": "sales_fy"}),
    how="left")
df = df.join(snap, how="inner")

# 分割補正: 開示日時点の r と現在の r の比で1株当たり値を換算
rj = pd.merge_asof(df.reset_index()[["code", "disc_date"]].sort_values("disc_date"),
                   rr.sort_values("date"), by="code",
                   left_on="disc_date", right_on="date", direction="forward")
df["r_disc"] = rj.set_index("code")["r"].reindex(df.index)
df["k"] = df["r_now"] / df["r_disc"]
df.loc[~np.isfinite(df["k"]) | (df["k"] <= 0), "k"] = 1.0

df["shares"] = (df["shout"] - df["trsh"].fillna(0)) * df["k"]   # 現在株数ベース
# 【重要】payload の Eq は「純資産合計」で非支配株主持分を含む。株主に帰属するのは自己資本
# ＝ TA × EqAR（開示ベースの自己資本比率）。これを使わないとPBRが系統的に過小になる
# （いすゞ7202: Eq基準0.93 → 自己資本基準1.03。外部公表値1.08と整合するのは後者）。
own = df["ta"] * df["eqar"]
df["eq_own"] = np.where(np.isfinite(own) & (own > 0) & (own <= df["eq"]), own, df["eq"])
df["bps"] = np.where(df["shares"] > 0, df["eq_own"] / df["shares"], np.nan)
df["feps_adj"] = df["feps"] / df["k"]
df["fdiv_adj"] = df["fdiv"] / df["k"]

df["mktcap"] = df["close"] * df["shares"]
df["pbr"] = df["close"] / df["bps"]
df["per_f"] = np.where(df["feps_adj"] > 0, df["close"] / df["feps_adj"], np.nan)
df["yield_f"] = df["fdiv_adj"] / df["close"] * 100
df["roe_f"] = np.where(df["eq_own"] > 0, df["fnp"] / df["eq_own"] * 100, np.nan)
df["growth"] = np.where(df["np_fy"] > 0, df["fnp"] / df["np_fy"] - 1, np.nan) * 100
df["netcash_r"] = df["cash"] / df["mktcap"]      # 現金/時価総額（負債未控除の粗い指標）
# 一過性チェック: 今期予想が過去5期平均利益に対して膨らみすぎていないか
df["fnp_vs5y"] = np.where(df["np_5y_avg"] > 0, df["fnp"] / df["np_5y_avg"], np.nan)
df["payout"] = np.where(df["feps_adj"] > 0, df["fdiv_adj"] / df["feps_adj"] * 100, np.nan)
# 予想純利益/予想営業利益。通常0.6〜0.75。1.0超は特別利益・税効果・持分法など
# 一過性で純利益が膨らんでいる疑い＝表面PERが実力より低く出る（例: TSI 77億/75億）
df["np_op"] = np.where(df["fop"] > 0, df["fnp"] / df["fop"], np.nan)
df["eqar_pct"] = df["eqar"] * 100

sm = db.read_sql("SELECT code5 code, name_ja, sector33_nm, market_nm FROM symbol_master")
df = df.join(sm.set_index("code"), how="left")
df["disc_days"] = (last_date - df["disc_date"]).dt.days
print(f"  対象 {len(df):,}銘柄 / 最新開示の中央値経過日数 {df['disc_days'].median():.0f}日")

# ---------------- スクリーン ----------------
BASE = ((df["market_nm"].isin(["プライム", "スタンダード"]))
        & (df["adv"] >= 2e8) & (df["mktcap"] >= 3e10)
        & (df["disc_days"] <= 120)
        & df["pbr"].between(0.05, 10) & df["per_f"].between(0, 100))

VALUE = (df["pbr"] <= 1.0) & (df["per_f"] <= 13) & (df["yield_f"] >= 3.0)
QUAL = (df["roe_f"] >= 8.0) & (df["loss_yrs"].fillna(9) == 0) & (df["fy_n"].fillna(0) >= 3)
TRAP = ((df["growth"] >= 0) & (df["ret3m"] >= -0.20)
        & (df["fnp_vs5y"].between(0.7, 2.5))          # 予想が5期平均から乖離しすぎない
        & (df["payout"].fillna(50).between(10, 90)))  # 配当が無理な水準でない
SAFE = (df["eqar_pct"] >= 40) | df["sector33_nm"].isin(FIN_SEC)

nonfin = ~df["sector33_nm"].isin(FIN_SEC)
sel = df[BASE & VALUE & QUAL & TRAP & SAFE & nonfin].copy()
selfin = df[BASE & (df["pbr"] <= 1.0) & (df["per_f"] <= 13) & (df["yield_f"] >= 3.0)
            & (df["roe_f"] >= 7.0) & (df["loss_yrs"].fillna(9) == 0)
            & (df["growth"] >= 0) & (df["ret3m"] >= -0.20)
            & (df["fnp_vs5y"].between(0.7, 2.5)) & (df["payout"].fillna(50).between(10, 90))
            & df["sector33_nm"].isin(FIN_SEC)].copy()

# 総合スコア: 割安3指標の順位平均 × 質
for d in (sel, selfin):
    d["score"] = (d["pbr"].rank(pct=True) + d["per_f"].rank(pct=True)
                  + (1 - d["yield_f"].rank(pct=True))) / 3
    d.sort_values("score", inplace=True)

COLS = ["name_ja", "sector33_nm", "close", "pbr", "per_f", "yield_f", "roe_f",
        "eqar_pct", "growth", "fnp_vs5y", "np_op", "payout", "ret3m", "ret12m", "mktcap", "adv"]


def fmt(d):
    o = d[COLS].copy()
    o["close"] = o["close"].round(0).astype(int)
    o["pbr"] = o["pbr"].round(2)
    for c in ["per_f", "yield_f", "roe_f", "eqar_pct", "growth", "payout"]:
        o[c] = o[c].round(1)
    o["fnp_vs5y"] = o["fnp_vs5y"].round(2)
    o["np_op"] = o["np_op"].round(2)
    o["ret3m"] = (o["ret3m"] * 100).round(1)
    o["ret12m"] = (o["ret12m"] * 100).round(1)
    o["時価総額億"] = (o["mktcap"] / 1e8).round(0).astype(int)
    o["ADV億"] = (o["adv"] / 1e8).round(1)
    return o.drop(columns=["mktcap", "adv"]).rename(columns={
        "name_ja": "銘柄", "sector33_nm": "業種", "close": "株価", "pbr": "PBR",
        "per_f": "予想PER", "yield_f": "予想利回%", "roe_f": "予想ROE%",
        "eqar_pct": "自己資本比%", "growth": "予想増益%", "fnp_vs5y": "予想/5期平均",
        "np_op": "純利/営利", "payout": "配当性向%", "ret3m": "3M%", "ret12m": "1Y%"})


print("\n" + "=" * 100)
print(f"【本命】非金融 割安×質×増益  (PBR≤1.0 / 予想PER≤13 / 予想利回り≥3% /"
      f" 予想ROE≥8% / 5期赤字なし / 増益 / 自己資本≥40% / ADV≥2億 / 時価総額≥300億)")
print("=" * 100)
print(f"該当 {len(sel)}銘柄。割安スコア順 上位30:")
print(fmt(sel).head(30).to_string())
print("\n業種分布:")
print(sel["sector33_nm"].value_counts().head(10).to_string())

print("\n" + "=" * 100)
print(f"【金融】銀行・保険等（自己資本比率の基準を外し ROE≥7%）  該当 {len(selfin)}銘柄 上位15:")
print("=" * 100)
print(fmt(selfin).head(15).to_string())

fmt(sel).to_csv(HERE / "candidates_nonfin.csv", encoding="utf-8-sig")
fmt(selfin).to_csv(HERE / "candidates_fin.csv", encoding="utf-8-sig")

print("\n※ バリュー・ローテーションの現況は rotation_check.py（PIT形成→前向きリターン）を参照。")
print("saved candidates_nonfin.csv / candidates_fin.csv")
