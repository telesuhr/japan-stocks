"""build.py で制約が年率を下げた（A 12.87% → D 11.10%）。
制約は「保険」として正当化するつもりだったので、守るはずのリスクを実際に減らしているかを直接測る。
減らしていないなら、ただ利回りを捨てているだけなので外す。

 (a) 一過性利益フィルタ → 1年後の減配を実際に減らすか（FDivAnn の前年比で直接測定）
 (b) 業種上限     → 悪い年（2018/2020）のドローダウンを実際に減らすか
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
from pathlib import Path
import numpy as np
import pandas as pd
from jstock import db

HERE = Path(__file__).resolve().parent

# ---------------- (a) 一過性利益は減配を予測するか ----------------
fin = db.read_sql("""
    SELECT code, disc_date, NULLIF(payload->>'FDivAnn','')::float fdiv,
           NULLIF(payload->>'FNP','')::float fnp, NULLIF(payload->>'FOP','')::float fop,
           NULLIF(payload->>'FEPS','')::float feps
    FROM fin_summary
    WHERE doc_type LIKE '%%FinancialStatements%%' AND disc_date >= '2015-01-01'
""")
fin["disc_date"] = pd.to_datetime(fin["disc_date"])
fin = fin.dropna(subset=["fdiv", "fnp", "fop"]).query("fdiv > 0 and fop > 0")
fin["oneoff"] = fin["fnp"] / fin["fop"]
fin = fin.sort_values("disc_date")

# 各開示の「約1年後(300〜430日)の最初の開示」の予想配当と比較
rows = []
for c, f in fin.groupby("code"):
    f = f.drop_duplicates("disc_date").set_index("disc_date")
    for dt, r in f.iterrows():
        fut = f.loc[(f.index >= dt + pd.Timedelta(days=300)) & (f.index <= dt + pd.Timedelta(days=430))]
        if fut.empty:
            continue
        rows.append((c, dt, r["oneoff"], r["fdiv"], fut.iloc[0]["fdiv"], r["feps"]))
E = pd.DataFrame(rows, columns=["code", "date", "oneoff", "div0", "div1", "feps"])
E["chg"] = E["div1"] / E["div0"] - 1
E = E[E["chg"].between(-0.95, 3.0)]          # 分割等の異常値を除外
E["payout"] = np.where(E["feps"] > 0, E["div0"] / E["feps"] * 100, np.nan)

print("=" * 92)
print("(a) 予想純利益/予想営業利益（一過性利益の代理）と 1年後の減配")
print("=" * 92)
E["bucket"] = pd.cut(E["oneoff"], [-np.inf, 0.5, 0.7, 0.85, 1.0, np.inf],
                     labels=["<0.5", "0.5-0.7", "0.7-0.85", "0.85-1.0", "≥1.0(一過性疑い)"])
g = E.groupby("bucket", observed=True).agg(N=("chg", "size"), 減配率=("chg", lambda s: (s < -0.01).mean() * 100),
                                           大幅減配率=("chg", lambda s: (s < -0.20).mean() * 100),
                                           配当変化中央=("chg", lambda s: s.median() * 100))
print(g.to_string(float_format=lambda v: f"{v:8.2f}"))

hi = E[E["oneoff"] >= 1.0]["chg"]
lo = E[E["oneoff"] < 1.0]["chg"]
from scipy import stats as sps
t, p = sps.ttest_ind(hi, lo, equal_var=False)
print(f"\n  ≥1.0 (n={len(hi)}) 減配率 {(hi<-0.01).mean()*100:.1f}% / 中央 {hi.median()*100:+.2f}%")
print(f"  <1.0 (n={len(lo)}) 減配率 {(lo<-0.01).mean()*100:.1f}% / 中央 {lo.median()*100:+.2f}%")
print(f"  差の t = {t:.2f}  p = {p:.4f}")

# 高利回り銘柄に限定（実際にフィルタが適用される母集団）
HY = E[(E["payout"].between(20, 80))].copy()
HY = HY[HY["div0"] / HY["feps"].where(HY["feps"] > 0) > 0]
hi2, lo2 = HY[HY["oneoff"] >= 1.0]["chg"], HY[HY["oneoff"] < 1.0]["chg"]
if len(hi2) > 30:
    t2, p2 = sps.ttest_ind(hi2, lo2, equal_var=False)
    print(f"\n  【質フィルタ通過後の母集団】≥1.0 (n={len(hi2)}) 減配率 {(hi2<-0.01).mean()*100:.1f}%"
          f" vs <1.0 (n={len(lo2)}) {(lo2<-0.01).mean()*100:.1f}%   t={t2:.2f} p={p2:.4f}")

# ---------------- (b) 業種上限は悪い年を守ったか ----------------
print("\n" + "=" * 92)
print("(b) 業種上限は悪い年のドローダウンを減らしたか")
print("=" * 92)
M = pd.read_csv(HERE / "monthly.csv", index_col=0, parse_dates=True)
yr = M.groupby(M.index.year).apply(lambda x: ((1 + x).prod() - 1) * 100)
print(yr.to_string(float_format=lambda v: f"{v:7.1f}"))
print("\n  悪い年の比較（A=制約なし / C=業種上限あり・同30銘柄）:")
for y in [2018, 2020, 2026]:
    if y in yr.index:
        print(f"    {y}: A {yr.loc[y,'A']:+6.1f}%   C {yr.loc[y,'C']:+6.1f}%   差 {yr.loc[y,'C']-yr.loc[y,'A']:+5.1f}pt")
dd = {}
for k in M.columns:
    s = M[k].dropna()
    cum = (1 + s).cumprod()
    dd[k] = (cum / cum.cummax() - 1).min() * 100
print("\n  最大DD:", "  ".join(f"{k} {v:.1f}%" for k, v in dd.items()))
print("  月次相関 A-D:", f"{M['A'].corr(M['D']):.3f}")
E.to_csv(HERE / "dividend_cut_test.csv", index=False)
print("\nsaved dividend_cut_test.csv")
