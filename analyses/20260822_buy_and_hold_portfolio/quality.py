"""「優良銘柄（高ROE・財務健全）を買って放置」は成立するか。高配当と別ファクターか。

既出チェック: analyses/README.md・SUMMARY.md に quality/ROE ファクターの検証は無い（未検証）。

事前登録した仮説（教訓5: グリッドサーチ前に仮説）:
  H1 予想ROE上位（財務健全スクリーン後）は市場EWを上回る            棄却条件: 市場を下回る
  H2 優良銘柄は放置に向く（5年保有で負けにくい）                    棄却条件: 5年保有の負け率が高配当より高い
  H3 質は高配当と別ファクター＝併せ持つ意味がある                    棄却条件: 月次相関が高く分散にならない

H3 が本題。20260818 で高配当は「バリューの一表現」と判明したので、
もう一本の軸が要る。質が独立なら「高配当30＋優良30」の二本立てに意味がある。
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
from pathlib import Path
import numpy as np
import pandas as pd
from jstock import db

HERE = Path(__file__).resolve().parent
LIQ, START, COST_BPS = 3e8, "2015-06-01", 2.0

px = db.read_sql("""SELECT code, date, close, adj_close, turnover_value
                    FROM stocks_daily WHERE date >= %(s)s AND close>0 AND adj_close>0""", {"s": START})
px["date"] = pd.to_datetime(px["date"])
AC = px.pivot(index="date", columns="code", values="adj_close").sort_index()
CL = px.pivot(index="date", columns="code", values="close").sort_index()
TV = px.pivot(index="date", columns="code", values="turnover_value").sort_index()
del px
cal, R = AC.index, AC / CL

fin = db.read_sql("""
    SELECT code, disc_date, NULLIF(payload->>'FDivAnn','')::float fdiv,
           NULLIF(payload->>'FEPS','')::float feps, NULLIF(payload->>'FNP','')::float fnp,
           NULLIF(payload->>'NP','')::float np, NULLIF(payload->>'TA','')::float ta,
           NULLIF(payload->>'EqAR','')::float eqar, NULLIF(payload->>'Eq','')::float eq
    FROM fin_summary WHERE doc_type LIKE '%%FinancialStatements%%' AND disc_date >= '2013-01-01'
""")
fin["disc_date"] = pd.to_datetime(fin["disc_date"])
own = fin["ta"] * fin["eqar"]
fin["eq_own"] = np.where(np.isfinite(own) & (own > 0) & (own <= fin["eq"]), own, fin["eq"])
fin = fin.sort_values("disc_date")
# 過去の赤字回数（実績純利益ベース・開示時点までの累積＝PIT）
fin["loss"] = (fin["np"] < 0).astype(float)
fin["loss_cum"] = fin.groupby("code")["loss"].transform(lambda s: s.rolling(20, min_periods=4).sum())
fin["n_disc"] = fin.groupby("code").cumcount() + 1

cols = AC.columns
def blank(): return pd.DataFrame(np.nan, index=cal, columns=cols)
D, E, Nf, Qe, EQAR, LOSS, NDISC = blank(), blank(), blank(), blank(), blank(), blank(), blank()
for c, f in fin.groupby("code"):
    if c not in R.columns:
        continue
    r_at = R[c].reindex(cal).ffill().bfill()
    rd = r_at.reindex(f["disc_date"], method="bfill").values
    d = pd.DataFrame({"date": f["disc_date"].values, "d": f["fdiv"].values * rd,
                      "e": f["feps"].values * rd, "n": f["fnp"].values, "q": f["eq_own"].values,
                      "ea": f["eqar"].values, "lo": f["loss_cum"].values,
                      "nd": f["n_disc"].values}).set_index("date").sort_index()
    d = d[~d.index.duplicated(keep="last")].reindex(cal, method="ffill")
    D[c], E[c], Nf[c], Qe[c], EQAR[c], LOSS[c], NDISC[c] = d["d"], d["e"], d["n"], d["q"], d["ea"], d["lo"], d["nd"]

YLD, PAYOUT = D / AC * 100, (D / E.where(E > 0)) * 100
ROE = (Nf / Qe.where(Qe > 0)) * 100
ADV = TV.rolling(60, min_periods=40).mean().shift(1)
sm = db.read_sql("SELECT code5 code, name_ja, sector33_nm, market_nm FROM symbol_master").set_index("code")
LISTED = sm["market_nm"].reindex(cols).isin(["プライム", "スタンダード"])

me = pd.DatetimeIndex(pd.Series(cal, index=cal).groupby([cal.year, cal.month]).last().values)
me = me[(me >= cal[0] + pd.Timedelta(days=200)) & (me < cal[-1])]


def tr(d0, d1, names):
    pr = AC.loc[d1, names] / AC.loc[d0, names] - 1
    div = (YLD.loc[d0, names] / 100) * ((d1 - d0).days / 365.25)
    return (pr + div.fillna(0)).mean()


def stats(s, lb):
    s = s.dropna()
    ann = (1 + s).prod() ** (12 / len(s)) - 1
    cum = (1 + s).cumprod()
    return dict(構成=lb, 年率=ann * 100, Sharpe=s.mean() / s.std() * np.sqrt(12),
                MDD=(cum / cum.cummax() - 1).min() * 100, 勝率=(s > 0).mean() * 100, N月=len(s))


def base_univ(d0):
    """財務健全スクリーン: 自己資本比率≥40% / 過去赤字ゼロ / 開示履歴4期以上 / 予想最終黒字"""
    return ((ADV.loc[d0] >= LIQ) & AC.loc[d0].notna() & LISTED
            & (EQAR.loc[d0] >= 0.40) & (LOSS.loc[d0] == 0) & (NDISC.loc[d0] >= 4)
            & (Nf.loc[d0] > 0))


def pick_quality(d0, n=30):
    ok = base_univ(d0)
    q = ROE.loc[d0].where(ok)
    q = q[q.notna() & (q > 0) & (q < 100)]      # 極端値除外
    return list(q.nlargest(n).index) if len(q) >= 60 else None


def pick_yield(d0, n=30):
    ok = (ADV.loc[d0] >= LIQ) & AC.loc[d0].notna() & LISTED
    y = YLD.loc[d0].where(ok)
    ok2 = y.notna() & (y > 0) & (y < 20) & PAYOUT.loc[d0].between(20, 80) & (ROE.loc[d0] >= 8.0) & (Nf.loc[d0] > 0)
    y = y[ok2]
    return list(y.nlargest(n).index) if len(y) >= 60 else None


def sim(picker, rebal=12, phase=0):
    prev, out, held, nform = set(), {}, None, 0
    for i in range(len(me) - 1):
        d0, d1 = me[i], me[i + 1]
        if held is None or (i - phase) % rebal == 0:
            cand = picker(d0)
            if cand is None:
                continue
            held, nform = cand, len(cand)
            to = 1.0 if not prev else len(set(held) - prev) / len(held)
            prev = set(held)
        else:
            to = 0.0
        held = [c for c in held if pd.notna(AC.loc[d1, c]) and pd.notna(AC.loc[d0, c])]
        if nform == 0 or len(held) < max(5, int(nform * 0.7)):
            continue
        out[d1] = tr(d0, d1, held) - to * 2 * COST_BPS / 1e4
    return pd.Series(out).sort_index()


# ---------------- H1: ROE5分位 ----------------
print("=" * 96)
print("H1: 予想ROE 5分位（財務健全スクリーン後・月次入替・配当込み・コスト込み）")
print("=" * 96)
qs, prev = {f"Q{k+1}": {} for k in range(5)}, {}
mkt = {}
for i in range(len(me) - 1):
    d0, d1 = me[i], me[i + 1]
    ok = base_univ(d0)
    q = ROE.loc[d0].where(ok)
    q = q[q.notna() & (q > 0) & (q < 100)]
    if len(q) < 150:
        continue
    lab = pd.qcut(q.rank(method="first"), 5, labels=[f"Q{k+1}" for k in range(5)])
    for k in qs:
        nm = list(lab[lab == k].index)
        to = 1.0 if k not in prev else len(set(nm) - prev[k]) / len(nm)
        prev[k] = set(nm)
        qs[k][d1] = tr(d0, d1, nm) - to * 2 * COST_BPS / 1e4
    u = (ADV.loc[d0] >= LIQ) & AC.loc[d0].notna() & LISTED
    mkt[d1] = tr(d0, d1, list(AC.columns[u.fillna(False)]))
rows = [stats(pd.Series(v).sort_index(), f"{k}{' 低ROE' if k=='Q1' else ' 高ROE' if k=='Q5' else ''}")
        for k, v in qs.items()]
rows.append(stats(pd.Series(mkt).sort_index(), "市場EW"))
Q = pd.DataFrame(rows)
print(Q.to_string(index=False, float_format=lambda v: f"{v:7.2f}"))

# ---------------- 年1回入替での質 vs 高配当 ----------------
print("\n" + "=" * 96)
print("年1回入替・上位30銘柄: 優良(高ROE) vs 高配当")
print("=" * 96)
sq, sy = sim(pick_quality), sim(pick_yield)
common = sq.index.intersection(sy.index)
comp = pd.DataFrame([stats(sq[common], "優良30(高ROE・財務健全)"), stats(sy[common], "高配当30")])
print(comp.to_string(index=False, float_format=lambda v: f"{v:7.2f}"))
for lb, s in [("優良30", sq), ("高配当30", sy)]:
    a = s[s.index < "2021-01-01"]; b = s[s.index >= "2021-01-01"]
    print(f"  {lb:<10} IS(〜2020) 年率{stats(a,'')['年率']:6.2f}% Sh{stats(a,'')['Sharpe']:5.2f} / "
          f"OOS(2021〜) 年率{stats(b,'')['年率']:6.2f}% Sh{stats(b,'')['Sharpe']:5.2f}")

# ---------------- H3: 相関と重複 ----------------
print("\n" + "=" * 96)
print("H3: 質は高配当と別ファクターか")
print("=" * 96)
print(f"  月次リターン相関: {sq[common].corr(sy[common]):.3f}")
ov = []
for d0 in me[::6]:
    a, b = pick_quality(d0), pick_yield(d0)
    if a and b:
        ov.append(len(set(a) & set(b)) / 30 * 100)
print(f"  銘柄重複率（30銘柄中・半年ごと{len(ov)}時点の平均）: {np.mean(ov):.1f}%")
half = ((sq[common] + sy[common]) / 2)
print("\n  50:50 合成:")
print(pd.DataFrame([stats(sq[common], "優良30のみ"), stats(sy[common], "高配当30のみ"),
                    stats(half, "50:50 合成")]).to_string(index=False, float_format=lambda v: f"{v:7.2f}"))

# ---------------- H2: 放置年数 ----------------
def hold_return(d0, years, names):
    d1c = d0 + pd.DateOffset(years=years)
    fut = cal[cal <= d1c]
    if len(fut) == 0 or fut[-1] < d1c - pd.Timedelta(days=15):
        return None
    d1, rets = fut[-1], []
    for c in names:
        s = AC.loc[d0:d1, c].dropna()
        if len(s) < 2:
            continue
        pr = s.iloc[-1] / AC.loc[d0, c] - 1
        gone = s.index[-1] < d1 - pd.Timedelta(days=10)
        y0 = YLD.loc[d0, c]
        pr += (y0 / 100) * ((s.index[-1] - d0).days / 365.25) if np.isfinite(y0) else 0.0
        if gone:
            pr -= 0.02
        rets.append(pr)
    return np.mean(rets) if len(rets) >= len(names) * 0.8 else None


print("\n" + "=" * 96)
print("H2: 買って放置（優良30 vs 高配当30 vs 50:50）")
print("=" * 96)
rows = []
for years in [1, 3, 5]:
    rec = {"放置年数": years}
    for lb, pk in [("優良30", pick_quality), ("高配当30", pick_yield)]:
        ann = []
        for d0 in me:
            nm = pk(d0)
            if nm is None:
                continue
            r = hold_return(d0, years, nm)
            if r is not None:
                ann.append((1 + r) ** (1 / years) - 1)
        ann = np.array(ann)
        rec[f"{lb}_年率中央%"] = np.median(ann) * 100
        rec[f"{lb}_負け%"] = (ann < 0).mean() * 100
        rec[f"{lb}_n"] = len(ann)
    rows.append(rec)
H = pd.DataFrame(rows)
print(H.to_string(index=False, float_format=lambda v: f"{v:8.2f}"))

pd.DataFrame({"優良30": sq, "高配当30": sy}).to_csv(HERE / "quality_vs_yield_monthly.csv")
Q.to_csv(HERE / "roe_quintiles.csv", index=False)
H.to_csv(HERE / "quality_hold.csv", index=False)

# ---------------- 最新の優良30 ----------------
d0 = me[-1]
nm = pick_quality(d0)
L = pd.DataFrame({
    "コード": [c[:4] for c in nm], "銘柄": sm["name_ja"].reindex(nm).values,
    "業種": sm["sector33_nm"].reindex(nm).values,
    "株価": CL.loc[cal[-1], nm].round(0).values,
    "予想ROE%": ROE.loc[d0, nm].round(1).values,
    "自己資本比率%": (EQAR.loc[d0, nm] * 100).round(0).values,
    "予想利回%": YLD.loc[d0, nm].round(2).values,
})
print("\n" + "=" * 96)
print(f"最新の優良30銘柄（形成日 {d0.date()} / 株価は {cal[-1].date()}）")
print("=" * 96)
print(L.to_string(index=False))
L.to_csv(HERE / "quality_list.csv", index=False, encoding="utf-8-sig")
print("\nsaved roe_quintiles.csv / quality_vs_yield_monthly.csv / quality_hold.csv / quality_list.csv")
