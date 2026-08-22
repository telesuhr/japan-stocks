"""勤務先の制約（不動産・銀行を保有不可）を課したときに、検証済みの成績が保たれるか。

20260822 本編で「業種上限を掛けると成績が悪化する」ことが分かっている以上、
業種を丸ごと落とす制約も同様に劣化させる可能性が高い。実額を確かめてから最終リストを出す。

除外は2段階で測る:
  A: 不動産業 + 銀行業（ユーザーの申告どおり厳密に）
  B: A + 証券･商品先物取引業 + その他金融業 + 保険業（金融を広く取る解釈）
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
from pathlib import Path
import numpy as np
import pandas as pd
from jstock import db

HERE = Path(__file__).resolve().parent
LIQ, START, COST_BPS = 3e8, "2015-06-01", 2.0
EX_A = ["不動産業", "銀行業"]
EX_B = EX_A + ["証券･商品先物取引業", "その他金融業", "保険業"]

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
def blank(): return pd.DataFrame(np.nan, index=cal, columns=cols)
D, E, Nf, Qe = blank(), blank(), blank(), blank()
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
ROE = (Nf / Qe.where(Qe > 0)) * 100
ADV = TV.rolling(60, min_periods=40).mean().shift(1)
sm = db.read_sql("SELECT code5 code, name_ja, sector33_nm, market_nm FROM symbol_master").set_index("code")
SEC = sm["sector33_nm"].reindex(cols)
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


def pick(d0, n=30, exclude=()):
    ok = (ADV.loc[d0] >= LIQ) & AC.loc[d0].notna() & LISTED
    if exclude:
        ok &= ~SEC.isin(exclude)
    y = YLD.loc[d0].where(ok)
    ok2 = y.notna() & (y > 0) & (y < 20) & PAYOUT.loc[d0].between(20, 80) & (ROE.loc[d0] >= 8.0) & (Nf.loc[d0] > 0)
    y = y[ok2]
    return list(y.nlargest(n).index) if len(y) >= 50 else None


def sim(exclude=(), n=30, rebal=12, phase=0):
    prev, out, held, nform = set(), {}, None, 0
    for i in range(len(me) - 1):
        d0, d1 = me[i], me[i + 1]
        if held is None or (i - phase) % rebal == 0:
            cand = pick(d0, n, exclude)
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


VAR = [("制約なし（本編の採用構成）", ()), ("A: 不動産・銀行を除外", tuple(EX_A)),
       ("B: A + 証券・その他金融・保険も除外", tuple(EX_B))]

print("=" * 96)
print("勤務先制約を課したときの成績（年1回入替・上位30銘柄）")
print("=" * 96)
S = {lb: sim(exclude=ex) for lb, ex in VAR}
common = None
for s in S.values():
    common = s.index if common is None else common.intersection(s.index)
print(pd.DataFrame([stats(S[lb][common], lb) for lb, _ in VAR]).to_string(index=False, float_format=lambda v: f"{v:7.2f}"))
print(f"  ※共通{len(common)}ヶ月")

print("\nIS/OOS:")
for lb, _ in VAR:
    s = S[lb]
    a, b = s[s.index < "2021-01-01"], s[s.index >= "2021-01-01"]
    print(f"  {lb:<34} IS {stats(a,'')['年率']:6.2f}% / OOS {stats(b,'')['年率']:6.2f}%")

print("\n" + "=" * 96)
print("入替月への依存（12フェーズ）")
print("=" * 96)
for lb, ex in VAR:
    a = np.array([stats(sim(exclude=ex, phase=k), "")["年率"] for k in range(12)])
    sh = np.array([stats(sim(exclude=ex, phase=k), "")["Sharpe"] for k in range(12)])
    print(f"  {lb:<34} 年率 {a.min():5.2f}〜{a.max():5.2f} (中央{np.median(a):5.2f}) / "
          f"Sh中央{np.median(sh):4.2f} / 全て正 {bool((a>0).all())}")

# 放置年数
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
print("買って放置（年率中央% / 負けた割合%）")
print("=" * 96)
rows = []
for years in [1, 3, 5]:
    rec = {"放置年数": years}
    for lb, ex in VAR:
        ann = []
        for d0 in me:
            nm = pick(d0, 30, ex)
            if nm is None:
                continue
            r = hold_return(d0, years, nm)
            if r is not None:
                ann.append((1 + r) ** (1 / years) - 1)
        ann = np.array(ann)
        key = lb.split(":")[0].split("（")[0]
        rec[f"{key}_年率"] = np.median(ann) * 100
        rec[f"{key}_負け%"] = (ann < 0).mean() * 100
    rows.append(rec)
print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:7.2f}"))
pd.DataFrame(rows).to_csv(HERE / "restricted_hold.csv", index=False)
pd.DataFrame(S).to_csv(HERE / "restricted_monthly.csv")

# ---------------- 最新リスト ----------------
snap_last = cal[-1]
for tag, ex in [("A", tuple(EX_A)), ("B", tuple(EX_B))]:
    d0 = me[-1]
    nm = pick(d0, 30, ex)
    L = pd.DataFrame({
        "コード": [c[:4] for c in nm], "銘柄": sm["name_ja"].reindex(nm).values,
        "業種": sm["sector33_nm"].reindex(nm).values,
        "株価": CL.loc[snap_last, nm].round(0).values,
        "予想利回%": YLD.loc[d0, nm].round(2).values,
        "予想ROE%": ROE.loc[d0, nm].round(1).values,
        "配当性向%": PAYOUT.loc[d0, nm].round(0).values,
    })
    L.index = range(1, len(L) + 1)
    print("\n" + "=" * 96)
    print(f"【{tag}】最新30銘柄（形成 {d0.date()} / 株価 {snap_last.date()}） 除外業種: {', '.join(ex)}")
    print("=" * 96)
    print(L.to_string())
    print(f"  平均予想利回り {L['予想利回%'].mean():.2f}% / 業種数 {L['業種'].nunique()} / "
          f"単元合計 {(L['株価']*100).sum():,.0f}円")
    L.to_csv(HERE / f"restricted_list_{tag}.csv", encoding="utf-8-sig")
print("\nsaved restricted_monthly.csv / restricted_hold.csv / restricted_list_A.csv / restricted_list_B.csv")
