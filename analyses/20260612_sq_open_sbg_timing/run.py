"""(A) メジャーSQ日の寄り方向バイアス  (B) SBGを寄りで売るか待つか(寄り後プロファイル)。

仮説(事前固定):
 A: メジャーSQの寄りギャップに一貫した下方バイアスは無い（SQ値は清算値で需給は事後）。検証で確認。
 B: ギャップアップ日のSBGは寄り後に巻き戻す（SOX検証と整合）→寄りで売る方が有利、待つと不利。
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

MAJOR = ['2016-09-09','2016-12-09','2017-03-10','2017-06-09','2017-09-08','2017-12-08','2018-03-09',
 '2018-06-08','2018-09-14','2018-12-14','2019-03-08','2019-06-14','2019-09-13','2019-12-13','2020-03-13',
 '2020-06-12','2020-09-11','2020-12-11','2021-03-12','2021-06-11','2021-09-10','2021-12-10','2022-03-11',
 '2022-06-10','2022-09-09','2022-12-09','2023-03-10','2023-06-09','2023-09-08','2023-12-08','2024-03-08',
 '2024-06-14','2024-09-13','2024-12-13','2025-03-14','2025-06-13','2025-09-12','2025-12-12','2026-03-13']
major = pd.to_datetime(MAJOR)

# ── (A) メジャーSQ日 1321 ──
nk = pd.read_sql("SELECT date, open, high, low, close FROM stocks_daily WHERE code='13210' AND date>='2016-01-01' ORDER BY date", conn)
nk["date"] = pd.to_datetime(nk["date"]); nk = nk.set_index("date").astype(float)
nk["prev_close"] = nk["close"].shift(1)
nk["gap"] = (nk["open"]/nk["prev_close"]-1)*1e4
nk["intraday"] = (nk["close"]/nk["open"]-1)*1e4
nk["dow"] = nk.index.dayofweek
sq = nk.reindex(major).dropna()
fri = nk[(nk["dow"]==4) & (~nk.index.isin(major))].dropna()
alld = nk.dropna()

print("=== (A) メジャーSQ日 vs 通常金曜 vs 全営業日 (bp) ===")
for name, d in [("メジャーSQ日", sq), ("通常の金曜", fri), ("全営業日", alld)]:
    gt, gp = stats.ttest_1samp(d["gap"], 0)
    print(f"{name:10s} n={len(d):4d}  寄りギャップ {d['gap'].mean():+6.1f}bp(t={gt:+.2f})  "
          f"寄→引 {d['intraday'].mean():+6.1f}bp  ギャップ<0率 {(d['gap']<0).mean()*100:.0f}%")

# ── (B) SBG 寄り後プロファイル ──
sbg = pd.read_sql("SELECT ts, open, close FROM stocks_intraday WHERE code='99840' ORDER BY ts", conn)
sbg["ts"] = pd.to_datetime(sbg["ts"]); sbg["date"] = sbg["ts"].dt.date
sbg[["open","close"]] = sbg[["open","close"]].astype(float)
# 日次gap（前日終値→当日寄り）でグループ分け
dl = pd.read_sql("SELECT date, open, close FROM stocks_daily WHERE code='99840' ORDER BY date", conn)
conn.close()
dl["date"] = pd.to_datetime(dl["date"]); dl = dl.set_index("date").astype(float)
dl["gap"] = (dl["open"]/dl["close"].shift(1)-1)*100

CHECK = ["09:30","10:00","10:30","11:00","11:30","12:30","13:00","13:30","14:00","14:30","15:00"]
def profile(days):
    acc = {t: [] for t in CHECK}
    for d, g in sbg.groupby("date"):
        if pd.Timestamp(d) not in days: continue
        g = g.set_index(g["ts"].dt.strftime("%H:%M"))
        op = g.between_time("09:00","09:05")["open"] if False else None
        o = g["open"].iloc[0]
        for t in CHECK:
            sub = g[g.index <= t]
            if len(sub): acc[t].append((sub["close"].iloc[-1]/o-1)*1e4)
    return {t: np.mean(v) if v else np.nan for t,v in acc.items()}, len(set(d for d,_ in sbg.groupby("date") if pd.Timestamp(d) in days))

gap_up = set(dl[dl["gap"]>=1.0].index)   # ギャップアップ+1%以上で始まった日
all_days = set(dl.index)
prof_up, n_up = profile(gap_up)
prof_all, n_all = profile(all_days)

print(f"\n=== (B) SBG 寄り基準の時刻別累積リターン(bp) ===")
print(f"{'時刻':6s} {'ギャップUP日':>12s}(n={n_up}) {'全日':>10s}(n={n_all})")
for t in CHECK:
    print(f"{t:6s} {prof_up[t]:>12.1f} {prof_all[t]:>16.1f}")

# ── 可視化 ──
import matplotlib.font_manager as fm
for _f in ["/mnt/c/Windows/Fonts/YuGothM.ttc","/mnt/c/Windows/Fonts/meiryo.ttc"]:
    if os.path.exists(_f): fm.fontManager.addfont(_f); plt.rcParams["font.family"]=fm.FontProperties(fname=_f).get_name(); break
plt.rcParams["axes.unicode_minus"]=False
fig, axes = plt.subplots(1,2,figsize=(12,6.75),facecolor="white")
cats=["メジャーSQ日","通常の金曜","全営業日"]; ds=[sq,fri,alld]
axes[0].bar(np.arange(3)-0.2,[d["gap"].mean() for d in ds],0.4,label="寄りギャップ",color="#2b8cbe")
axes[0].bar(np.arange(3)+0.2,[d["intraday"].mean() for d in ds],0.4,label="寄→引",color="#e34a33")
axes[0].set_xticks(range(3),cats); axes[0].axhline(0,color="gray",lw=0.8); axes[0].legend(fontsize=9)
axes[0].set_ylabel("bp"); axes[0].set_title("(A) メジャーSQ日に寄り下げバイアスは無い")
xs=range(len(CHECK))
axes[1].plot(xs,[prof_up[t] for t in CHECK],marker="o",label=f"ギャップUP日(n={n_up})",color="#e34a33")
axes[1].plot(xs,[prof_all[t] for t in CHECK],marker="s",label=f"全日(n={n_all})",color="#2b8cbe")
axes[1].set_xticks(xs,CHECK,rotation=45,fontsize=8); axes[1].axhline(0,color="gray",lw=0.8); axes[1].legend(fontsize=9)
axes[1].set_ylabel("寄り基準 累積リターン(bp)"); axes[1].set_title("(B) SBG: ギャップUP日は寄り後ずるずる下げ")
fig.suptitle("メジャーSQの寄り方向 と SBGを寄りで売るか待つか", fontsize=13)
fig.text(0.99,0.01,"データ: 1321 2016-2026 / SBG分足 2024-05〜2026-06 (JQuants)",ha="right",va="bottom",fontsize=8,color="gray")
plt.tight_layout(rect=[0,0.02,1,0.95]); fig.savefig("result.png",dpi=100,bbox_inches="tight",facecolor="white")
print("saved result.png")
