"""「高配当銘柄を買っておけばいいのか」を10年で検証する。

【この検証で絶対に外せない点】
1. **adj_close は配当調整されていない**（JQuantsのAdjustmentFactorは分割・併合・権利落ち用）。
   価格リターンだけで比較すると、配当を出す分だけ高配当銘柄が構造的に不利になる＝
   高配当戦略を検証しているつもりで逆のものを測ることになる。→ 配当を足し戻す。
2. **生存者バイアス排除** — delisted_at で母集団を切らない。PITの流動性のみで決める
   （高配当の罠＝減配・経営悪化銘柄は退場する。そこを除くと戦略が過大評価される）。
3. **PIT形成 → 前向きリターン**。今日の利回りで分位を作って過去を見るのは同語反復。
4. **コスト込み**（教訓2）。分位ポートフォリオの入替分にのみ課金。

仮説:
  H1 高配当(Q5)は市場EWを上回る                      棄却: 年率で市場を下回る
  H2 単純な高配当は「罠」を拾う。質フィルタ(配当性向・ROE・赤字なし)で改善する
                                                      棄却: フィルタ後も改善しない
  H3 高配当の優位は低PBRの優位と別物（重複していない）  棄却: 高配当Q5がほぼ低PBR銘柄
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
from pathlib import Path
import numpy as np
import pandas as pd
from jstock import db

HERE = Path(__file__).resolve().parent
START = "2015-06-01"
LIQ = 3e8            # PIT 60日平均売買代金
COST_BPS = 2.0       # 片道。入替分に往復で課金
FIN_SEC = ("銀行業", "保険業", "証券･商品先物取引業", "その他金融業")

print("[1] 価格...")
px = db.read_sql("""
    SELECT code, date, close, adj_close, turnover_value
    FROM stocks_daily WHERE date >= %(s)s AND close > 0 AND adj_close > 0
""", {"s": START})
px["date"] = pd.to_datetime(px["date"])
AC = px.pivot(index="date", columns="code", values="adj_close").sort_index()
TV = px.pivot(index="date", columns="code", values="turnover_value").sort_index()
del px
cal = AC.index
print(f"  {AC.shape[0]}営業日 × {AC.shape[1]}銘柄  ({cal[0].date()}〜{cal[-1].date()})")

print("[2] fin_summary (PIT)...")
fin = db.read_sql("""
    SELECT code, disc_date,
           NULLIF(payload->>'FDivAnn','')::float fdiv,
           NULLIF(payload->>'FEPS','')::float    feps,
           NULLIF(payload->>'FNP','')::float     fnp,
           NULLIF(payload->>'NP','')::float      np,
           NULLIF(payload->>'TA','')::float      ta,
           NULLIF(payload->>'EqAR','')::float    eqar,
           NULLIF(payload->>'Eq','')::float      eq,
           NULLIF(payload->>'ShOutFY','')::float shout,
           NULLIF(payload->>'TrShFY','')::float  trsh
    FROM fin_summary
    WHERE doc_type LIKE '%%FinancialStatements%%' AND disc_date >= '2014-01-01'
""")
fin["disc_date"] = pd.to_datetime(fin["disc_date"])
own = fin["ta"] * fin["eqar"]
fin["eq_own"] = np.where(np.isfinite(own) & (own > 0) & (own <= fin["eq"]), own, fin["eq"])
fin["sh"] = fin["shout"] - fin["trsh"].fillna(0)
fin = fin.sort_values("disc_date")

sm = db.read_sql("SELECT code5 code, sector33_nm FROM symbol_master").set_index("code")["sector33_nm"]

# ---- 開示テーブルを日次パネルに展開（調整後空間で定数化） ----
# DPS(t) = fdiv * r_disc / r(t)  →  利回り(t) = fdiv*r_disc / adj_close(t)
CL = None
R = None
print("[3] PITパネル構築...")
cl = db.read_sql("SELECT code, date, close FROM stocks_daily WHERE date >= %(s)s AND close > 0",
                 {"s": START})
cl["date"] = pd.to_datetime(cl["date"])
CL = cl.pivot(index="date", columns="code", values="close").sort_index()
del cl
R = (AC / CL).reindex(cal)

cols = AC.columns
D = pd.DataFrame(np.nan, index=cal, columns=cols)   # fdiv * r_disc
E = pd.DataFrame(np.nan, index=cal, columns=cols)   # feps * r_disc
S = pd.DataFrame(np.nan, index=cal, columns=cols)   # sh / r_disc  (BPS/PBR用)
Q = pd.DataFrame(np.nan, index=cal, columns=cols)   # eq_own
N = pd.DataFrame(np.nan, index=cal, columns=cols)   # fnp

for c, f in fin.groupby("code"):
    if c not in R.columns:
        continue
    r_at = R[c].reindex(cal).ffill().bfill()
    rd = r_at.reindex(f["disc_date"], method="bfill").values
    d = pd.DataFrame({"date": f["disc_date"].values,
                      "d": f["fdiv"].values * rd, "e": f["feps"].values * rd,
                      "s": f["sh"].values / rd, "q": f["eq_own"].values,
                      "n": f["fnp"].values}).set_index("date").sort_index()
    d = d[~d.index.duplicated(keep="last")].reindex(cal, method="ffill")
    D[c], E[c], S[c], Q[c], N[c] = d["d"], d["e"], d["s"], d["q"], d["n"]

YLD = D / AC * 100                       # 予想配当利回り %
PBR = AC * S / Q                         # PIT PBR（自己資本ベース）
PAYOUT = (D / E.where(E > 0)) * 100      # 予想配当性向 %
ROE = (N / Q.where(Q > 0)) * 100         # 予想ROE %
ADV = TV.rolling(60, min_periods=40).mean().shift(1)

# ---- 月末形成 → 翌月保有 ----
me = pd.Series(cal, index=cal).groupby([cal.year, cal.month]).last().values
me = pd.DatetimeIndex(me)
me = me[(me >= cal[0] + pd.Timedelta(days=200)) & (me < cal[-1])]
print(f"  形成月 {len(me)}回 ({me[0].date()}〜{me[-1].date()})")


def month_ret(d0, d1, names):
    """d0→d1 の price return に配当の期間按分を足したトータルリターン"""
    pr = AC.loc[d1, names] / AC.loc[d0, names] - 1
    days = (d1 - d0).days
    div = (YLD.loc[d0, names] / 100) * (days / 365.25)   # 期間按分（平滑近似）
    return pr + div.fillna(0)


def run(label, extra_filter=None, nq=5):
    prev = {}
    rows = []
    for i in range(len(me) - 1):
        d0, d1 = me[i], me[i + 1]
        univ = (ADV.loc[d0] >= LIQ) & AC.loc[d0].notna() & AC.loc[d1].notna()
        y = YLD.loc[d0].where(univ)
        ok = y.notna() & (y >= 0) & (y < 20)          # 異常利回りは除外
        if extra_filter is not None:
            ok &= extra_filter(d0)
        y = y[ok]
        if len(y) < 200:
            continue
        pos = y[y > 0]
        zero = y[y <= 0].index                         # 無配
        lab = pd.qcut(pos.rank(method="first"), nq, labels=[f"Q{k+1}" for k in range(nq)])
        grp = {f"Q{k+1}": pos.index[lab == f"Q{k+1}"] for k in range(nq)}
        grp["無配"] = zero
        r = {}
        for k, names in grp.items():
            if len(names) < 15:
                r[k] = np.nan
                continue
            gross = month_ret(d0, d1, list(names)).mean()
            p = prev.get(k, set())
            to = 1.0 if not p else len(set(names) - p) / len(names)
            r[k] = gross - to * 2 * COST_BPS / 1e4     # 入替分に往復コスト
            prev[k] = set(names)
        r["市場EW"] = month_ret(d0, d1, list(y.index)).mean()
        r["date"] = d1
        r["N"] = len(y)
        rows.append(r)
    R_ = pd.DataFrame(rows).set_index("date")
    print(f"\n--- {label}  (形成{len(R_)}回 / 平均{R_['N'].mean():.0f}銘柄) ---")
    out = []
    for k in [f"Q{j+1}" for j in range(nq)] + ["無配", "市場EW"]:
        s = R_[k].dropna()
        if len(s) < 24:
            continue
        ann = (1 + s).prod() ** (12 / len(s)) - 1
        sh = s.mean() / s.std() * np.sqrt(12)
        cum = (1 + s).cumprod()
        mdd = (cum / cum.cummax() - 1).min()
        out.append(dict(分位=k, 年率=ann * 100, Sharpe=sh, MDD=mdd * 100,
                        勝率=(s > 0).mean() * 100, N月=len(s)))
    O = pd.DataFrame(out)
    print(O.to_string(index=False, float_format=lambda v: f"{v:7.2f}"))
    return R_, O


print("\n" + "=" * 88)
print("H1: 素の高配当（Q5=最高利回り）は市場に勝つか  ※コスト込み・配当込み・PIT")
print("=" * 88)
R1, O1 = run("素の予想配当利回り 5分位")

print("\n" + "=" * 88)
print("H2: 質フィルタ（配当性向20-80% / 予想ROE≥8% / 直近FY黒字）を掛けると改善するか")
print("=" * 88)


def qual(d0):
    return (PAYOUT.loc[d0].between(20, 80) & (ROE.loc[d0] >= 8.0) & (N.loc[d0] > 0))


R2, O2 = run("高配当 × 質フィルタ", extra_filter=qual)

print("\n" + "=" * 88)
print("H3: 高配当は低PBRと別物か（Q5構成銘柄のPBR分位の分布）")
print("=" * 88)
ov = []
for d0 in me[:-1]:
    univ = (ADV.loc[d0] >= LIQ) & AC.loc[d0].notna()
    y = YLD.loc[d0].where(univ)
    p = PBR.loc[d0].where(univ & PBR.loc[d0].between(0.05, 20))
    both = y[(y > 0) & y.notna()].index.intersection(p.dropna().index)
    if len(both) < 200:
        continue
    yq = pd.qcut(y[both].rank(method="first"), 5, labels=False)
    pq = pd.qcut(p[both].rank(method="first"), 5, labels=False)
    hi = yq == 4
    ov.append(dict(date=d0, 低PBR_Q1に居る率=(pq[hi] == 0).mean() * 100,
                   PBR下位2割まで=(pq[hi] <= 1).mean() * 100,
                   平均PBR分位=pq[hi].mean() + 1))
OV = pd.DataFrame(ov)
print(f"高配当Q5の銘柄のうち、PBR最安Q1に居る割合: 平均 {OV['低PBR_Q1に居る率'].mean():.1f}%")
print(f"                        PBR下位2分位まで: 平均 {OV['PBR下位2割まで'].mean():.1f}%")
print(f"高配当Q5銘柄の平均PBR分位(1=最安,5=最高): {OV['平均PBR分位'].mean():.2f}")

# ---- 年次 ----
print("\n" + "=" * 88)
print("年次リターン %（素のQ5=高配当 / 質フィルタ後Q5 / 市場EW）")
print("=" * 88)
yr = pd.DataFrame({
    "高配当Q5": R1["Q5"].groupby(R1.index.year).apply(lambda s: ((1 + s).prod() - 1) * 100),
    "高配当Q5×質": R2["Q5"].groupby(R2.index.year).apply(lambda s: ((1 + s).prod() - 1) * 100),
    "低配当Q1": R1["Q1"].groupby(R1.index.year).apply(lambda s: ((1 + s).prod() - 1) * 100),
    "無配": R1["無配"].groupby(R1.index.year).apply(lambda s: ((1 + s).prod() - 1) * 100),
    "市場EW": R1["市場EW"].groupby(R1.index.year).apply(lambda s: ((1 + s).prod() - 1) * 100),
})
print(yr.to_string(float_format=lambda v: f"{v:7.1f}"))

# ---- IS/OOS ----
print("\n" + "=" * 88)
print("IS(〜2020) / OOS(2021〜) 分割")
print("=" * 88)
for nm, RR in [("素の高配当Q5", R1["Q5"]), ("高配当Q5×質", R2["Q5"]), ("市場EW", R1["市場EW"])]:
    for seg, s in [("IS", RR[RR.index < "2021-01-01"].dropna()),
                   ("OOS", RR[RR.index >= "2021-01-01"].dropna())]:
        ann = (1 + s).prod() ** (12 / len(s)) - 1
        print(f"  {nm:<12} {seg:<4} 年率{ann*100:6.2f}%  Sharpe{s.mean()/s.std()*np.sqrt(12):5.2f}  N={len(s)}")

R1.to_csv(HERE / "monthly_plain.csv")
R2.to_csv(HERE / "monthly_quality.csv")
yr.to_csv(HERE / "yearly.csv")
O1.to_csv(HERE / "summary_plain.csv", index=False)
O2.to_csv(HERE / "summary_quality.csv", index=False)
print("\nsaved csv")
