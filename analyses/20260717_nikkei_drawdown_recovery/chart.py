"""N225 水面下(ドローダウン)チャート + 現在地。"""
import sys; sys.stdout.reconfigure(line_buffering=True)
from pathlib import Path
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from jstock import db

HERE = Path(__file__).resolve().parent
d = db.read_sql("SELECT date, close FROM index_daily WHERE code='N225' ORDER BY date", [])
d["date"] = pd.to_datetime(d["date"])
d["peak"] = d["close"].cummax(); d["dd"] = (d["close"]/d["peak"]-1)*100

try:
    import matplotlib.font_manager as fm
    fp = fm.FontProperties(fname="/root/.fonts/NotoSansJP.ttf"); plt.rcParams["font.family"]=fp.get_name()
except Exception: pass

fig, ax = plt.subplots(figsize=(12, 6.75), facecolor="white")
ax.fill_between(d["date"], d["dd"], 0, color="#c0392b", alpha=0.35)
ax.plot(d["date"], d["dd"], color="#c0392b", lw=0.8)
ax.axhline(-7.64, color="#2980b9", ls="--", lw=1.5)
ax.text(d["date"].iloc[10], -7.64, " 現在 -7.6%(7/16)", color="#2980b9", va="bottom", fontsize=11)
# 主要トラフ注記
notes = [("2020-03-19",-31.8,"コロナ -32%"),("2025-04-07",-26.3,"関税 -26%"),
         ("2022-03-09",-19.4,"22年 -19%"),("2018-03-23",-14.5,"18年 -15%")]
for dt,val,lab in notes:
    ax.annotate(lab, xy=(pd.Timestamp(dt),val), fontsize=9, color="#555",
                xytext=(pd.Timestamp(dt),val-2.5), ha="center")
ax.set_title("日経225 ピークからの下落率(水面下チャート) 2016–2026  現在は浅い調整局面", fontsize=15)
ax.set_ylabel("ピーク比 (%)"); ax.grid(alpha=0.3); ax.set_ylim(-36, 2)
fig.text(0.99, 0.01, "データ: JQuants index_daily N225 / 2026-07-16時点。今日7/17の下落は未反映",
         ha="right", va="bottom", fontsize=8, color="gray")
fig.savefig(HERE/"result.png", dpi=100, bbox_inches="tight", facecolor="white")
print("saved result.png")
