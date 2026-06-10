#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全面ギャップダウン日の「寄り押し目買い」検証
仮説: 市場が広範にギャップダウンして寄り付いた日に、主力半導体・非鉄を
      寄り(open)で等金額買うと、当日引け/翌日にかけてコスト後でもプラスになる
      （= パニック寄り→リバの取り込み）。
注意(教訓1): 同日の値動きは「同時反応」。予測力があるかは、無条件(全日)の
      ベースラインと比較して上乗せがあるかで判定する。
"""
import sys, os
sys.stdout.reconfigure(line_buffering=True)
import psycopg2
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

OUT = os.path.dirname(os.path.abspath(__file__))

# ---- 日本語フォント ----
def setup_jp_font():
    cands = [
        "/mnt/c/Windows/Fonts/meiryo.ttc",
        "/mnt/c/Windows/Fonts/YuGothM.ttc",
        "/mnt/c/Windows/Fonts/YuGothR.ttc",
        "/mnt/c/Windows/Fonts/msgothic.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJKjp-Regular.otf",
    ]
    for p in cands:
        if os.path.exists(p):
            try:
                font_manager.fontManager.addfont(p)
                name = font_manager.FontProperties(fname=p).get_name()
                plt.rcParams["font.family"] = name
                plt.rcParams["axes.unicode_minus"] = False
                print("JP font:", name, p)
                return
            except Exception as e:
                print("font fail", p, e)
    print("WARN: JP font not found, labels may be tofu")
setup_jp_font()

PG = dict(host=os.environ.get("PGHOST","localhost"), port=5432,
          user="postgres", dbname="market_data")

# ---- ユニバース（主力 半導体/AI + 非鉄、5桁、特殊状況の新光電工は除外）----
SEMI = {
 "99840":"ソフトバンクG","80350":"東京エレク","68570":"アドバンテスト","61460":"ディスコ",
 "69200":"レーザーテック","69630":"ローム","65260":"ソシオネクスト","34360":"SUMCO",
 "67230":"ルネサス","77350":"SCREEN","65250":"KOKUSAI","40630":"信越化学"}
NONFE = {
 "57130":"住友鉱山","57110":"三菱マテリアル","57060":"三井金属","58010":"古河電工",
 "58020":"住友電工","58030":"フジクラ","57140":"DOWA","57070":"東邦亜鉛"}
UNIV = {**SEMI, **NONFE}
codes = list(UNIV.keys())

conn = psycopg2.connect(**PG)

# ---- 市場ギャップ（流動株の寄りギャップ中央値）を日次で ----
sql_mkt = """
WITH liq AS (
  SELECT code, date, open,
         LAG(close) OVER (PARTITION BY code ORDER BY date) AS prev_close,
         turnover_value
  FROM stocks_daily
)
SELECT date, percentile_cont(0.5) WITHIN GROUP (ORDER BY (open/prev_close-1.0)) AS mgap
FROM liq
WHERE prev_close>0 AND open>0 AND turnover_value > 5e8
GROUP BY date HAVING COUNT(*)>=100
ORDER BY date;
"""
mkt = pd.read_sql(sql_mkt, conn)
mkt["date"] = pd.to_datetime(mkt["date"])
mkt = mkt.set_index("date")["mgap"].astype(float)
print("market days:", len(mkt))

# ---- ユニバースの日次OHLC ----
sql_u = "SELECT code,date,open,close FROM stocks_daily WHERE code = ANY(%s) ORDER BY code,date;"
u = pd.read_sql(sql_u, conn, params=(codes,))
conn.close()
u["date"] = pd.to_datetime(u["date"])
for c in ["open","close"]:
    u[c] = u[c].astype(float)
u = u.sort_values(["code","date"])
g = u.groupby("code", group_keys=False)
u["prev_close"] = g["close"].shift(1)
u["next_open"]  = g["open"].shift(-1)
u["next_close"] = g["close"].shift(-1)
u["self_gap"] = u["open"]/u["prev_close"]-1.0
u = u.merge(mkt.rename("mgap"), left_on="date", right_index=True, how="left")

# ---- リターン定義（entry=寄りopen, 3つのexit）----
u["ret_intraday"]  = u["close"]/u["open"]-1.0       # 当日引け
u["ret_nextopen"]  = u["next_open"]/u["open"]-1.0   # 翌寄り
u["ret_nextclose"] = u["next_close"]/u["open"]-1.0  # 翌引け（=仮説の「翌日まで」）

EXITS = {"当日引け":"ret_intraday","翌寄り":"ret_nextopen","翌引け":"ret_nextclose"}

def perf(df, retcol, rt_cost, period_per_year):
    """df: 1行=1トレード(銘柄×イベント日)。イベント日ごとに等金額バスケット化して評価。"""
    d = df.dropna(subset=[retcol]).copy()
    if len(d)==0: return None
    d["net"] = d[retcol] - rt_cost
    # 銘柄レベル指標
    wins = d.loc[d.net>0,"net"].sum(); loss = -d.loc[d.net<0,"net"].sum()
    pf = wins/loss if loss>0 else np.inf
    winrate = (d.net>0).mean()
    # イベント日バスケット（等金額平均）系列
    bask = d.groupby("date")["net"].mean().sort_index()
    n = len(bask)
    mean = bask.mean(); std = bask.std(ddof=1)
    tstat = mean/(std/np.sqrt(n)) if std>0 and n>1 else np.nan
    sharpe_ann = (mean/std*np.sqrt(period_per_year)) if std>0 else np.nan
    eq = (1+bask).cumprod()
    total = eq.iloc[-1]-1
    dd = (eq/eq.cummax()-1).min()
    return dict(n_events=n, n_trades=len(d), mean_trade=d.net.mean(),
                winrate=winrate, pf=pf, tstat=tstat, sharpe_ann=sharpe_ann,
                total=total, maxdd=dd, basket=bask, equity=eq)

# 年間イベント数（年率換算用）
years = (u["date"].max()-u["date"].min()).days/365.25

print("\n" + "="*72)
print("仮説検証: 市場ギャップ閾値 × Exit × コスト")
print("="*72)

THRESH = [-0.010, -0.015, -0.020]
COSTS  = [0.001, 0.002, 0.003, 0.005]  # 往復コスト(0.1%/0.2%/0.3%/0.5%)

# メインテーブル（往復0.2%固定で 閾値×Exit）
RT_MAIN = 0.002
rows=[]
for th in THRESH:
    days = mkt.index[mkt<=th]
    sub = u[u["date"].isin(days)]
    ppy = len(days)/years
    for ename, col in EXITS.items():
        r = perf(sub, col, RT_MAIN, ppy)
        if r: rows.append(dict(閾値=f"{th*100:.1f}%", Exit=ename, **{k:r[k] for k in
              ["n_events","n_trades","mean_trade","winrate","pf","tstat","sharpe_ann","total","maxdd"]}))
res = pd.DataFrame(rows)
pd.set_option("display.width",200, "display.float_format", lambda x:f"{x:,.4f}")
print(f"\n[往復コスト {RT_MAIN*100:.1f}% 固定]")
print(res.to_string(index=False))

# ---- ベースライン（無条件・全日、entry=open→翌引け）: 予測力の有無 ----
base = perf(u, "ret_nextclose", RT_MAIN, 252)
print("\n[ベースライン] 全日・寄り→翌引け（コスト後, バスケット平均）:")
print(f"  平均トレード={base['mean_trade']*100:.3f}%  勝率={base['winrate']*100:.1f}%  "
      f"PF={base['pf']:.2f}  n_trades={base['n_trades']}")
gd = perf(u[u['date'].isin(mkt.index[mkt<=-0.015])], "ret_nextclose", RT_MAIN, 1)
print(f"[ギャップ日(-1.5%)] 寄り→翌引け（同上）:")
print(f"  平均トレード={gd['mean_trade']*100:.3f}%  勝率={gd['winrate']*100:.1f}%  "
      f"PF={gd['pf']:.2f}  n_trades={gd['n_trades']}")
print(f"  → 上乗せ(エッジ)= {(gd['mean_trade']-base['mean_trade'])*100:+.3f}%/トレード")

# ---- コスト感応度（閾値-1.5%、翌引け）----
print("\n[コスト感応度] 閾値-1.5% / 寄り→翌引け")
days15 = mkt.index[mkt<=-0.015]; sub15 = u[u["date"].isin(days15)]; ppy15=len(days15)/years
for rt in COSTS:
    r = perf(sub15, "ret_nextclose", rt, ppy15)
    print(f"  往復{rt*100:.1f}%: 平均={r['mean_trade']*100:+.3f}%  勝率={r['winrate']*100:.1f}%  "
          f"PF={r['pf']:.2f}  累積={r['total']*100:+.1f}%  t={r['tstat']:.2f}")

# ---- 半導体 vs 非鉄 の切り分け（閾値-1.5%、翌引け、コスト0.2%）----
print("\n[グループ別] 閾値-1.5% / 寄り→翌引け / 往復0.2%")
for gname, gset in [("半導体/AI",SEMI),("非鉄",NONFE)]:
    r = perf(sub15[sub15["code"].isin(gset)], "ret_nextclose", RT_MAIN, ppy15)
    print(f"  {gname}: 平均={r['mean_trade']*100:+.3f}%  勝率={r['winrate']*100:.1f}%  PF={r['pf']:.2f}  n={r['n_trades']}")

# ---- 可視化: 閾値-1.5%、3 Exit のイベント連結エクイティ ----
fig, ax = plt.subplots(figsize=(12,6.75), dpi=100)
colors={"当日引け":"#1f77b4","翌寄り":"#ff7f0e","翌引け":"#d62728"}
for ename,col in EXITS.items():
    r = perf(sub15, col, RT_MAIN, ppy15)
    eq = r["equity"]
    ax.plot(eq.index, (eq.values-1)*100, label=f"{ename} (累積{r['total']*100:+.1f}%, 勝率{r['winrate']*100:.0f}%, PF{r['pf']:.2f})",
            color=colors[ename], lw=1.8)
ax.axhline(0, color="gray", lw=0.8, ls="--")
ax.set_title(f"全面ギャップダウン日(市場中央値≤-1.5%, {len(days15)}日)の寄り押し目買い\n"
             f"主力半導体/AI+非鉄20銘柄・等金額バスケット・往復コスト0.2%控除後", fontsize=13)
ax.set_xlabel("イベント発生日"); ax.set_ylabel("累積リターン (%)")
ax.legend(loc="best", fontsize=10); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT,"result.png"), dpi=100, bbox_inches="tight")
print("\nsaved:", os.path.join(OUT,"result.png"))
