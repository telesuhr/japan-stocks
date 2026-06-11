"""SOX大幅高の翌日本営業日、寄→引で取れる業種はあるか（探索的・多重検定注意）。

半導体は寄りギャップで取り切られる（20260612_sox_surge_jp_nextday で確認済み）。
では寄ってから引けまでプラスの業種が他にあるか。17業種を横断スクリーニング。
教訓5: これは仮説先行でなく探索なので、多重検定(17業種)に厳しく見る。Bonferroni α=0.05/17≈0.003。
"""
import os, sys
sys.stdout.reconfigure(line_buffering=True)
import numpy as np, pandas as pd, psycopg2
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

PG = {"host": os.environ.get("PGHOST","localhost"), "port": int(os.environ.get("PGPORT",5432)),
      "user": os.environ.get("PGUSER","postgres"), "dbname": os.environ.get("PGDATABASE","market_data")}
conn = psycopg2.connect(**PG)

# 日次・業種等加重の gap/intraday（プライムのみ）
sec = pd.read_sql("""
WITH px AS (
  SELECT sd.date, sm.sector17_nm AS sec, sd.open, sd.close,
         lag(sd.close) OVER (PARTITION BY sd.code ORDER BY sd.date) AS prev
  FROM stocks_daily sd JOIN symbol_master sm ON sm.code5=sd.code
  WHERE sm.market_nm LIKE '%%プライム%%' AND sd.date>='2020-01-01' AND sd.open>0)
SELECT date, sec,
       avg((close/open-1)*1e4) AS intraday,
       avg((open/prev-1)*1e4)  AS gap,
       count(*) AS n
FROM px WHERE prev>0 GROUP BY date, sec ORDER BY date""", conn)
sox = pd.read_sql("SELECT trade_date, close FROM macro.daily_ohlcv WHERE symbol='.SOX' AND trade_date>='2020-01-01' ORDER BY trade_date", conn)
conn.close()

sec["date"] = pd.to_datetime(sec["date"])
sox["trade_date"] = pd.to_datetime(sox["trade_date"]); sox["ret"] = sox["close"].pct_change()*100

jp_dates = pd.DatetimeIndex(sorted(sec["date"].unique()))
def next_jp(d):
    i = jp_dates.searchsorted(d, side="right")
    return jp_dates[i] if i < len(jp_dates) else pd.NaT

ev_days = sox[sox["ret"]>=3.0]["trade_date"].map(next_jp).dropna().unique()
print(f"SOX+3%翌日本営業日: {len(ev_days)}日")

rows = []
for s, g in sec.groupby("sec"):
    g = g.set_index("date")
    base = g["intraday"].dropna()
    ev = g.reindex(ev_days)["intraday"].dropna()
    if len(ev) < 20: continue
    t, p = stats.ttest_1samp(ev, 0)
    rows.append({"sector": s, "n": len(ev),
                 "intraday_bp": round(ev.mean(),1),
                 "base_bp": round(base.mean(),1),
                 "excess_bp": round(ev.mean()-base.mean(),1),
                 "gap_bp": round(g.reindex(ev_days)["gap"].mean(),1),
                 "t": round(t,2), "p": round(p,4),
                 "win%": round((ev>0).mean()*100,1)})
res = pd.DataFrame(rows).sort_values("intraday_bp", ascending=False)
BONF = 0.05/len(res)
print(f"\n=== SOX+3%翌日の業種別 寄→引(bp)  [Bonferroni α={BONF:.4f}] ===")
res["有意"] = np.where(res["p"]<BONF, "★", np.where(res["p"]<0.05, "(弱)", ""))
print(res.to_string(index=False))
all_ev = sec[sec["date"].isin(ev_days)].groupby("date")["intraday"].mean()
print(f"\nプライム全体 寄→引 平均(SOX+3%翌日): {all_ev.mean():.1f}bp")
res.to_csv("results.csv", index=False)

# ── 可視化 ──
import matplotlib.font_manager as fm
for _f in ["/mnt/c/Windows/Fonts/YuGothM.ttc","/mnt/c/Windows/Fonts/meiryo.ttc"]:
    if os.path.exists(_f): fm.fontManager.addfont(_f); plt.rcParams["font.family"]=fm.FontProperties(fname=_f).get_name(); break
plt.rcParams["axes.unicode_minus"]=False
fig, ax = plt.subplots(figsize=(12,6.75),facecolor="white")
r = res.sort_values("intraday_bp")
colors = ["#e34a33" if v=="★" else ("#fdbb84" if v=="(弱)" else "#9ecae1") for v in r["有意"]]
ax.barh(r["sector"], r["intraday_bp"], color=colors)
ax.axvline(0, color="gray", lw=0.8)
ax.set_xlabel("SOX+3%翌日の 寄→引 平均リターン (bp)")
ax.set_title(f"SOX大幅高の翌日、寄ってから取れる業種はあるか (n≈{int(res['n'].median())}/業種)\n"
             f"赤=Bonferroni有意 橙=p<0.05(弱) 青=非有意 → 大半が非有意=データマイニング注意", fontsize=11)
fig.text(0.99,0.01,"データ: 2020-2026 / stocks_daily プライム × .SOX / 寄→引 コスト未控除",ha="right",va="bottom",fontsize=8,color="gray")
plt.tight_layout(); fig.savefig("result.png",dpi=100,bbox_inches="tight",facecolor="white")
print("saved result.png / results.csv")
