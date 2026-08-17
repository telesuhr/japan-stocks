"""
バリュー・ローテーションが足元も続いているかの正しい検証。

【重要】現在のPBRで分位を作って"過去"リターンを見るのは同語反復
（上がった銘柄ほど今のPBRが高い）。必ず
  過去の一時点でPIT PBR分位を形成 → そこからの"前向き"リターン
で測る。20260729分析と同じ作法。
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
import pandas as pd
from jstock import db

FIN_SEC = ("銀行業", "保険業", "証券･商品先物取引業", "その他金融業")
LIQ = 3e8

px = db.read_sql("""
    SELECT code, date, close, adj_close, turnover_value
    FROM stocks_daily WHERE date >= '2024-10-01' AND close > 0 AND adj_close > 0
""")
px["date"] = pd.to_datetime(px["date"])
fin = db.read_sql("""
    SELECT code, disc_date,
           NULLIF(payload->>'Eq','')::float      eq,
           NULLIF(payload->>'ShOutFY','')::float shout,
           NULLIF(payload->>'TrShFY','')::float  trsh
    FROM fin_summary
    WHERE doc_type LIKE '%%FinancialStatements%%' AND disc_date >= '2022-01-01'
      AND NULLIF(payload->>'Eq','')::float > 0
""")
fin["disc_date"] = pd.to_datetime(fin["disc_date"])
sm = db.read_sql("SELECT code5 code, name_ja, sector33_nm FROM symbol_master").set_index("code")

AC = px.pivot(index="date", columns="code", values="adj_close").sort_index()
CL = px.pivot(index="date", columns="code", values="close").sort_index()
TVm = px.pivot(index="date", columns="code", values="turnover_value").sort_index()
R = AC / CL                       # 分割補正用の比率
cal = AC.index
LAST = cal[-1]

# 各月末営業日
me = pd.DatetimeIndex(pd.Series(cal, index=cal).groupby([cal.year, cal.month]).last().values)
FORM = [d for d in me if pd.Timestamp("2025-08-01") <= d < LAST]


def pit_pbr(d):
    """d時点で"開示済み"の最新Eq/株数からPIT BPSを作り、PBRを返す"""
    f = fin[fin["disc_date"] <= d].sort_values("disc_date").groupby("code").tail(1)
    f = f.set_index("code")
    sh = (f["shout"] - f["trsh"].fillna(0))
    sh = sh[sh > 0]
    # 開示時点→d の分割補正
    r_at = R.reindex(index=cal).ffill()
    r_d = r_at.loc[d]
    r_disc = pd.Series({c: (r_at.loc[:f.at[c, "disc_date"]].iloc[-1][c]
                            if c in r_at.columns and
                            len(r_at.loc[:f.at[c, "disc_date"]]) else np.nan)
                        for c in sh.index})
    k = (r_d.reindex(sh.index) / r_disc).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    k[k <= 0] = 1.0
    shares = sh * k
    bps = f.loc[sh.index, "eq"] / shares
    price = CL.loc[d].reindex(sh.index)
    pbr = price / bps
    adv = TVm.loc[:d].tail(60).mean().reindex(sh.index)
    out = pd.DataFrame({"pbr": pbr, "adv": adv})
    out["sector"] = sm["sector33_nm"].reindex(out.index)
    return out[(out.pbr.between(0.1, 10)) & (out.adv >= LIQ) & (~out.sector.isin(FIN_SEC))]


print("=" * 92)
print("PIT PBR分位を形成 → そこからの『前向き』EWリターン（非金融・ADV≥3億・全市場）")
print("=" * 92)
print(f"{'形成日':<12}{'N':>6}  {'Q1割安':>8}{'Q2':>8}{'Q3':>8}{'Q4':>8}{'Q5割高':>8}"
      f"{'Q1-Q5':>9}{'市場EW':>9}")
rows = []
for d in FORM:
    g = pit_pbr(d)
    fwd = (AC.loc[LAST] / AC.loc[d] - 1).reindex(g.index)
    g = g.assign(fwd=fwd).dropna(subset=["fwd"])
    if len(g) < 200:
        continue
    g["q"] = pd.qcut(g["pbr"], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
    qr = g.groupby("q", observed=True)["fwd"].mean() * 100
    sp = qr["Q1"] - qr["Q5"]
    rows.append(dict(form=d.date(), N=len(g), **{k: v for k, v in qr.items()},
                     spread=sp, mkt=g["fwd"].mean() * 100))
    print(f"{str(d.date()):<12}{len(g):>6}  " + "".join(f"{v:>8.1f}" for v in qr.values)
          + f"{sp:>9.1f}{g['fwd'].mean()*100:>9.1f}")
pd.DataFrame(rows).to_csv("rotation_pit.csv", index=False)
print("\n※ 各行とも『形成日 → 最終営業日(%s)』までのリターン。期間が異なるので縦比較は"
      "「割安が勝っているか(Q1-Q5の符号)」で見る。" % LAST.date())
print("saved rotation_pit.csv")
