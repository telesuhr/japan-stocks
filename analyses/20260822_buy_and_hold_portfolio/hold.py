"""ユーザーの言う「買って放置」を文字通り検証する。

20260818 で検証したのは「年1回入替」であって「一度買って二度と触らない」ではない。
放置年数を延ばすと配当利回りは買値に固定される一方、減配・業績悪化・退場を抱え込む。
何年まで放置してよいのかを直接測る。退場銘柄は**除外せずリターンに反映**する
（放置運用で最も痛いのがこれなので、生存者バイアスを入れると意味が無くなる）。
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
from pathlib import Path
import numpy as np
import pandas as pd
from jstock import db

HERE = Path(__file__).resolve().parent
LIQ, START = 3e8, "2015-06-01"

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
           NULLIF(payload->>'TA','')::float ta, NULLIF(payload->>'EqAR','')::float eqar,
           NULLIF(payload->>'Eq','')::float eq
    FROM fin_summary WHERE doc_type LIKE '%%FinancialStatements%%' AND disc_date >= '2014-01-01'
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

YLD, PAYOUT = D / AC * 100, (D / E.where(E > 0)) * 100
ROE, ADV = (Nf / Qe.where(Qe > 0)) * 100, TV.rolling(60, min_periods=40).mean().shift(1)
sm = db.read_sql("SELECT code5 code, name_ja, sector33_nm, market_nm FROM symbol_master").set_index("code")
LISTED = sm["market_nm"].reindex(cols).isin(["プライム", "スタンダード"])

me = pd.DatetimeIndex(pd.Series(cal, index=cal).groupby([cal.year, cal.month]).last().values)
me = me[(me >= cal[0] + pd.Timedelta(days=200)) & (me < cal[-1])]


def pick(d0, n=30):
    ok = (ADV.loc[d0] >= LIQ) & AC.loc[d0].notna() & LISTED
    y = YLD.loc[d0].where(ok)
    ok = y.notna() & (y > 0) & (y < 20) & PAYOUT.loc[d0].between(20, 80) & (ROE.loc[d0] >= 8.0) & (Nf.loc[d0] > 0)
    y = y[ok]
    return list(y.nlargest(n).index) if len(y) >= 60 else None


def hold_return(d0, years, names):
    """d0 に等額で買って years 年放置。配当は買値利回りを毎年受け取ると仮定（会社予想据置＝楽観側）。
    途中で価格が途切れた銘柄（退場・上場廃止）は最終値でその時点評価＝損失を確定して残す。"""
    d1c = d0 + pd.DateOffset(years=years)
    fut = cal[cal <= d1c]
    if len(fut) == 0 or fut[-1] < d0 + pd.DateOffset(years=years) - pd.Timedelta(days=15):
        return None
    d1 = fut[-1]
    rets, ylds = [], []
    for c in names:
        s = AC.loc[d0:d1, c].dropna()
        if len(s) < 2:
            continue
        pr = s.iloc[-1] / AC.loc[d0, c] - 1
        gone = s.index[-1] < d1 - pd.Timedelta(days=10)     # 途中で消えた
        yrs_alive = (s.index[-1] - d0).days / 365.25
        y0 = YLD.loc[d0, c]
        div = (y0 / 100) * yrs_alive if np.isfinite(y0) else 0.0
        if gone:
            pr -= 0.02                                       # 退場時の実務コスト概算
        rets.append(pr + div)
        ylds.append(y0)
    if len(rets) < len(names) * 0.8:
        return None
    return np.mean(rets), np.nanmean(ylds), len(rets)


print("=" * 96)
print("「一度買って N 年放置」 vs 「年1回入替」  ─ 高配当×質 上位30銘柄")
print("=" * 96)
rows = []
for years in [1, 2, 3, 5]:
    tot, ann, mk = [], [], []
    for d0 in me:
        nm = pick(d0)
        if nm is None:
            continue
        r = hold_return(d0, years, nm)
        if r is None:
            continue
        tot.append(r[0])
        ann.append((1 + r[0]) ** (1 / years) - 1)
        # 同期間の市場EW（比較対象）
        univ = (ADV.loc[d0] >= LIQ) & AC.loc[d0].notna() & LISTED
        mnames = list(AC.columns[univ.fillna(False)])
        rm = hold_return(d0, years, mnames)
        mk.append((1 + rm[0]) ** (1 / years) - 1 if rm else np.nan)
    tot, ann, mk = np.array(tot), np.array(ann), np.array(mk)
    rows.append(dict(放置年数=years, 標本=len(tot), 年率中央=np.median(ann) * 100,
                     年率平均=np.mean(ann) * 100, 最悪=np.min(ann) * 100, 最良=np.max(ann) * 100,
                     負けた割合=(tot < 0).mean() * 100, 市場EW年率中央=np.nanmedian(mk) * 100,
                     市場に勝った割合=np.nanmean(ann > mk) * 100))
H = pd.DataFrame(rows)
print(H.to_string(index=False, float_format=lambda v: f"{v:8.2f}"))
H.to_csv(HERE / "hold_years.csv", index=False)

print("\n" + "=" * 96)
print("参考: 年1回入替（20260818 の検証済み構成）= 年率 12.55% / Sharpe 0.87")
print("=" * 96)
print("放置年数を延ばして年率が落ちるなら、その差が『入替の価値』＝年1回だけ触る理由。")
print("\nsaved hold_years.csv")
