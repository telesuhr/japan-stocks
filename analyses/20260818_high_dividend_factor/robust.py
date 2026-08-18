"""run.py の結果に対する3つの検算。

(1) 質フィルタの比較が不公平 — 素Q5は114ヶ月・質Q5は85ヶ月で母数が違う。共通月で揃えて再比較。
(2) Q5は約128銘柄。個人が実際に持てる本数(20/30/50)に絞っても効くか。
(3) 年1回リバランス（実運用の現実解）でも残るか。月次入替のコストと手間を避けられるか。
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
from pathlib import Path
import numpy as np
import pandas as pd
from jstock import db

HERE = Path(__file__).resolve().parent
START = "2015-06-01"
LIQ = 3e8
COST_BPS = 2.0

px = db.read_sql("""SELECT code, date, close, adj_close, turnover_value
                    FROM stocks_daily WHERE date >= %(s)s AND close>0 AND adj_close>0""",
                 {"s": START})
px["date"] = pd.to_datetime(px["date"])
AC = px.pivot(index="date", columns="code", values="adj_close").sort_index()
CL = px.pivot(index="date", columns="code", values="close").sort_index()
TV = px.pivot(index="date", columns="code", values="turnover_value").sort_index()
del px
cal = AC.index
R = AC / CL

fin = db.read_sql("""
    SELECT code, disc_date, NULLIF(payload->>'FDivAnn','')::float fdiv,
           NULLIF(payload->>'FEPS','')::float feps, NULLIF(payload->>'FNP','')::float fnp,
           NULLIF(payload->>'NP','')::float np, NULLIF(payload->>'TA','')::float ta,
           NULLIF(payload->>'EqAR','')::float eqar, NULLIF(payload->>'Eq','')::float eq
    FROM fin_summary
    WHERE doc_type LIKE '%%FinancialStatements%%' AND disc_date >= '2014-01-01'
""")
fin["disc_date"] = pd.to_datetime(fin["disc_date"])
own = fin["ta"] * fin["eqar"]
fin["eq_own"] = np.where(np.isfinite(own) & (own > 0) & (own <= fin["eq"]), own, fin["eq"])
fin = fin.sort_values("disc_date")

cols = AC.columns
D = pd.DataFrame(np.nan, index=cal, columns=cols)
E = pd.DataFrame(np.nan, index=cal, columns=cols)
Nf = pd.DataFrame(np.nan, index=cal, columns=cols)
Qe = pd.DataFrame(np.nan, index=cal, columns=cols)
for c, f in fin.groupby("code"):
    if c not in R.columns:
        continue
    r_at = R[c].reindex(cal).ffill().bfill()
    rd = r_at.reindex(f["disc_date"], method="bfill").values
    d = pd.DataFrame({"date": f["disc_date"].values, "d": f["fdiv"].values * rd,
                      "e": f["feps"].values * rd, "n": f["fnp"].values,
                      "q": f["eq_own"].values}).set_index("date").sort_index()
    d = d[~d.index.duplicated(keep="last")].reindex(cal, method="ffill")
    D[c], E[c], Nf[c], Qe[c] = d["d"], d["e"], d["n"], d["q"]

YLD = D / AC * 100
PAYOUT = (D / E.where(E > 0)) * 100
ROE = (Nf / Qe.where(Qe > 0)) * 100
ADV = TV.rolling(60, min_periods=40).mean().shift(1)

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
    return dict(戦略=lb, 年率=ann * 100, Sharpe=s.mean() / s.std() * np.sqrt(12),
                MDD=(cum / cum.cummax() - 1).min() * 100, 勝率=(s > 0).mean() * 100, N月=len(s))


def sim(top=None, quality=False, rebal_months=1, phase=0, label=""):
    """top=保有本数(Noneなら上位20%全部) / rebal_months=リバランス間隔 / phase=入替を始める月"""
    prev, out, held, nform = set(), {}, None, 0
    for i in range(len(me) - 1):
        d0, d1 = me[i], me[i + 1]
        if held is None or (i - phase) % rebal_months == 0:
            univ = (ADV.loc[d0] >= LIQ) & AC.loc[d0].notna()
            y = YLD.loc[d0].where(univ)
            ok = y.notna() & (y > 0) & (y < 20)
            if quality:
                ok &= PAYOUT.loc[d0].between(20, 80) & (ROE.loc[d0] >= 8.0) & (Nf.loc[d0] > 0)
            y = y[ok]
            if len(y) < 150:
                continue
            n = top if top else max(20, int(len(y) * 0.2))
            held = list(y.nlargest(n).index)
            nform = len(held)
            to = 1.0 if not prev else len(set(held) - prev) / len(held)
            prev = set(held)
        else:
            to = 0.0
        held = [c for c in held if pd.notna(AC.loc[d1, c]) and pd.notna(AC.loc[d0, c])]
        if nform == 0 or len(held) < max(5, int(nform * 0.7)):
            continue
        out[d1] = tr(d0, d1, held) - to * 2 * COST_BPS / 1e4
    return pd.Series(out).sort_index()


print("=" * 92)
print("(1) 質フィルタの公平比較 — 共通月だけで揃える")
print("=" * 92)
plain = sim(label="素")
qual = sim(quality=True)
common = plain.index.intersection(qual.index)
print(pd.DataFrame([stats(plain[common], "素の高配当 上位20%"),
                    stats(qual[common], "高配当×質 上位20%")]
                   ).to_string(index=False, float_format=lambda v: f"{v:7.2f}"))
print(f"  ※共通{len(common)}ヶ月 ({common[0].date()}〜{common[-1].date()}) で比較")

print("\n" + "=" * 92)
print("(2) 実際に持てる本数に絞る（月次リバランス）")
print("=" * 92)
rows = []
res = {}
for n in [10, 20, 30, 50, None]:
    s = sim(top=n)
    res[n] = s
    rows.append(stats(s, f"高配当 上位{n}銘柄" if n else "高配当 上位20%(約128)"))
print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:7.2f}"))

print("\n" + "=" * 92)
print("(3) リバランス頻度を落とす（30銘柄固定）")
print("=" * 92)
rows = []
for m, lb in [(1, "毎月"), (3, "四半期"), (6, "半年"), (12, "年1回")]:
    rows.append(stats(sim(top=30, rebal_months=m), f"上位30銘柄 / {lb}入替"))
print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:7.2f}"))

print("\n" + "=" * 92)
print("(4) 上位30銘柄・年1回入替 の IS/OOS と年次")
print("=" * 92)
s30 = sim(top=30, rebal_months=12)
mkt = sim(top=None)  # 参考
for seg, ss in [("IS(〜2020)", s30[s30.index < "2021-01-01"]), ("OOS(2021〜)", s30[s30.index >= "2021-01-01"])]:
    st = stats(ss, seg)
    print(f"  {seg:<12} 年率{st['年率']:6.2f}%  Sharpe{st['Sharpe']:5.2f}  MDD{st['MDD']:6.1f}%  N={st['N月']}")
yr = s30.groupby(s30.index.year).apply(lambda x: ((1 + x).prod() - 1) * 100)
print("\n年次%:", "  ".join(f"{y}:{v:+.1f}" for y, v in yr.items()))

pd.DataFrame({f"top{k or 'all'}": v for k, v in res.items()}).to_csv(HERE / "by_count.csv")
s30.to_frame("top30_annual").to_csv(HERE / "top30_annual.csv")
print("\nsaved by_count.csv / top30_annual.csv")

print("\n" + "=" * 92)
print("(5) 年1回入替は『何月に入替えるか』に依存しないか（フェーズ12通り）")
print("=" * 92)
ph = []
for k in range(12):
    st = stats(sim(top=30, rebal_months=12, phase=k), f"phase{k}")
    ph.append(st)
P = pd.DataFrame(ph)
print(f"  年率: 最小{P['年率'].min():.2f}%  中央{P['年率'].median():.2f}%  最大{P['年率'].max():.2f}%")
print(f"  Sharpe: 最小{P['Sharpe'].min():.2f}  中央{P['Sharpe'].median():.2f}  最大{P['Sharpe'].max():.2f}")
print(f"  → 12通り全てで年率プラス: {(P['年率']>0).all()} / 全てSharpe>0.5: {(P['Sharpe']>0.5).all()}")
P.to_csv(HERE / "annual_phase.csv", index=False)
