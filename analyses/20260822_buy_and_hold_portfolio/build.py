"""買って放置する現物ポートフォリオの構築と、その制約が性能を壊さないかの検証。

20260818 で「高配当×質・上位30銘柄・年1回入替」は検証済み(年率12.55%/Sharpe0.87)。
だが「放置」は月次入替の検証で無視できた3つのリスクを顕在化させる:

  (a) 一過性利益で表面利回りが高く見える銘柄 → 翌期に減配（20260817 の FNP/FOP 検出列）
  (b) 業種偏り（先週の30銘柄は不動産4/卸売4/建設4/サービス4 = 16/30 が内需・景気敏感）
  (c) 自社史で見て既に再評価され切った銘柄（20260817 追補で自分の結論がひっくり返った件）

制約を足すと普通は性能が落ちる。落ちないことを12フェーズの年1回入替で確認してから
最終リストを出す。落ちるなら制約を外す。
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
SECTOR_CAP = 3          # 1業種あたりの上限銘柄数
ONE_OFF_MAX = 1.00      # 予想純利益/予想営業利益 の上限（超＝一過性利益の疑い）

# ---------------- パネル構築（20260818 robust.py と同じ土台） ----------------
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
           NULLIF(payload->>'FOP','')::float fop, NULLIF(payload->>'TA','')::float ta,
           NULLIF(payload->>'EqAR','')::float eqar, NULLIF(payload->>'Eq','')::float eq
    FROM fin_summary
    WHERE doc_type LIKE '%%FinancialStatements%%' AND disc_date >= '2014-01-01'
""")
fin["disc_date"] = pd.to_datetime(fin["disc_date"])
# 教訓(20260817): payload の Eq は純資産合計＝非支配株主持分込み。株主帰属分は TA×EqAR
own = fin["ta"] * fin["eqar"]
fin["eq_own"] = np.where(np.isfinite(own) & (own > 0) & (own <= fin["eq"]), own, fin["eq"])
fin = fin.sort_values("disc_date")

cols = AC.columns
D = pd.DataFrame(np.nan, index=cal, columns=cols)   # 予想年間配当（調整後価格空間）
E = pd.DataFrame(np.nan, index=cal, columns=cols)   # 予想EPS（同）
Nf = pd.DataFrame(np.nan, index=cal, columns=cols)  # 予想純利益
Qe = pd.DataFrame(np.nan, index=cal, columns=cols)  # 自己資本
Op = pd.DataFrame(np.nan, index=cal, columns=cols)  # 予想営業利益
for c, f in fin.groupby("code"):
    if c not in R.columns:
        continue
    r_at = R[c].reindex(cal).ffill().bfill()
    rd = r_at.reindex(f["disc_date"], method="bfill").values
    d = pd.DataFrame({"date": f["disc_date"].values, "d": f["fdiv"].values * rd,
                      "e": f["feps"].values * rd, "n": f["fnp"].values,
                      "q": f["eq_own"].values, "o": f["fop"].values}).set_index("date").sort_index()
    d = d[~d.index.duplicated(keep="last")].reindex(cal, method="ffill")
    D[c], E[c], Nf[c], Qe[c], Op[c] = d["d"], d["e"], d["n"], d["q"], d["o"]

YLD = D / AC * 100
PAYOUT = (D / E.where(E > 0)) * 100
ROE = (Nf / Qe.where(Qe > 0)) * 100
ONEOFF = Nf / Op.where(Op > 0)
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


def pick(d0, n, quality, oneoff, sector_cap):
    """形成日 d0 のユニバースから n 銘柄を選ぶ。"""
    ok = (ADV.loc[d0] >= LIQ) & AC.loc[d0].notna() & LISTED
    y = YLD.loc[d0].where(ok)
    ok = y.notna() & (y > 0) & (y < 20)
    if quality:
        ok &= PAYOUT.loc[d0].between(20, 80) & (ROE.loc[d0] >= 8.0) & (Nf.loc[d0] > 0)
    if oneoff:
        ok &= (ONEOFF.loc[d0] < ONE_OFF_MAX)
    y = y[ok].sort_values(ascending=False)
    if len(y) < 60:
        return None
    if not sector_cap:
        return list(y.index[:n])
    held, cnt = [], {}
    for c in y.index:                      # 利回り順に、1業種 sector_cap 銘柄まで
        s = SEC.get(c, "不明")
        if cnt.get(s, 0) >= sector_cap:
            continue
        held.append(c)
        cnt[s] = cnt.get(s, 0) + 1
        if len(held) >= n:
            break
    return held if len(held) >= n * 0.8 else None


def sim(n=30, quality=True, oneoff=False, sector_cap=0, rebal=12, phase=0):
    prev, out, held, nform = set(), {}, None, 0
    for i in range(len(me) - 1):
        d0, d1 = me[i], me[i + 1]
        if held is None or (i - phase) % rebal == 0:
            cand = pick(d0, n, quality, oneoff, sector_cap)
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


VARIANTS = [
    ("A: 高配当×質 30銘柄（20260818の検証済み構成）", dict(n=30, oneoff=False, sector_cap=0)),
    ("B: A + 一過性利益フィルタ",                      dict(n=30, oneoff=True, sector_cap=0)),
    ("C: B + 業種上限3銘柄",                           dict(n=30, oneoff=True, sector_cap=SECTOR_CAP)),
    ("D: C を20銘柄に絞る（実際に持つ本数）",           dict(n=20, oneoff=True, sector_cap=SECTOR_CAP)),
]

print("=" * 100)
print("検証1: 制約を足すと性能は落ちるか（年1回入替・phase=0）")
print("=" * 100)
rows, series = [], {}
for lb, kw in VARIANTS:
    s = sim(**kw)
    series[lb[0]] = s
    rows.append(stats(s, lb))
base_idx = series["A"].index
common = base_idx
for k in series:
    common = common.intersection(series[k].index)
print("  ※共通%d ヶ月で比較 (%s〜%s)" % (len(common), common[0].date(), common[-1].date()))
rows = [stats(series[lb[0]][common], lb) for lb, _ in VARIANTS]
T = pd.DataFrame(rows)
print(T.to_string(index=False, float_format=lambda v: f"{v:7.2f}"))

print("\n" + "=" * 100)
print("検証2: 入替月への依存（12フェーズ）— 放置運用は『いつ買ったか』で決まってはいけない")
print("=" * 100)
ph_rows = []
for lb, kw in VARIANTS:
    a = [stats(sim(phase=k, **kw), "x") for k in range(12)]
    ann = np.array([x["年率"] for x in a])
    sh = np.array([x["Sharpe"] for x in a])
    ph_rows.append(dict(構成=lb, 年率最小=ann.min(), 年率中央=np.median(ann), 年率最大=ann.max(),
                        Sh最小=sh.min(), Sh中央=np.median(sh), 全て正=bool((ann > 0).all())))
P = pd.DataFrame(ph_rows)
print(P.to_string(index=False, float_format=lambda v: f"{v:7.2f}"))

print("\n" + "=" * 100)
print("検証3: IS/OOS（採用構成）")
print("=" * 100)
best = series["D"]
for seg, ss in [("IS(〜2020)", best[best.index < "2021-01-01"]),
                ("OOS(2021〜)", best[best.index >= "2021-01-01"])]:
    st = stats(ss, seg)
    print(f"  {seg:<12} 年率{st['年率']:6.2f}%  Sharpe{st['Sharpe']:5.2f}  MDD{st['MDD']:6.1f}%  N={st['N月']}")
yr = best.groupby(best.index.year).apply(lambda x: ((1 + x).prod() - 1) * 100)
print("  年次%:", "  ".join(f"{y}:{v:+.1f}" for y, v in yr.items()))

T.to_csv(HERE / "variants.csv", index=False)
P.to_csv(HERE / "phase.csv", index=False)
pd.DataFrame(series).to_csv(HERE / "monthly.csv")
yr.to_frame("年次%").to_csv(HERE / "yearly.csv")
print("\nsaved variants.csv / phase.csv / monthly.csv / yearly.csv")
