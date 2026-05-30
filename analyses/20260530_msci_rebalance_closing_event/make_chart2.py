"""result2.png: MSCI採用銘柄の公表前ドリフト vs 公表後（選択バイアス注記つき）。"""
import sys; sys.stdout.reconfigure(line_buffering=True)
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np, pandas as pd

OUT=Path(__file__).parent
fp="/root/.fonts/NotoSansJP.ttf"
if Path(fp).exists():
    font_manager.fontManager.addfont(fp)
    plt.rcParams["font.family"]=font_manager.FontProperties(fname=fp).get_name()
plt.rcParams["axes.unicode_minus"]=False

r=pd.read_csv(OUT/"additions_returns.csv")
phases=["P20_公表前20d","P10_公表前10d","R1_公表後ドリフト","R2_公表→発効引","R3_発効当日"]
labels=["公表前\n20営業日","公表前\n10営業日","公表後→\n発効前日","公表→\n発効引け","発効当日\n寄→引"]
means=[r[p].mean() for p in phases]
wins=[(r[p]>0).mean()*100 for p in phases]
colors=["#27ae60","#27ae60","#c0392b","#c0392b","#7f8c8d"]

fig,ax=plt.subplots(figsize=(12,6.75),facecolor="white")
ax.bar(range(len(phases)),means,color=colors,alpha=0.85)
ax.axhline(0,color="#333",lw=1)
ax.set_xticks(range(len(phases))); ax.set_xticklabels(labels,fontsize=9)
ax.set_ylabel("TOPIX超過リターン 平均(%)",fontsize=11)
ax.grid(axis="y",alpha=0.3)
for i,(m,w) in enumerate(zip(means,wins)):
    ax.text(i,m+(0.8 if m>=0 else -0.8),f"{m:+.1f}%\n勝率{w:.0f}%",ha="center",
            va="bottom" if m>=0 else "top",fontsize=9)
fig.suptitle("MSCI採用エッジは『公表前』に集中 — だが選択バイアスに注意",fontsize=14,fontweight="bold",y=0.98)
ax.set_title("採用銘柄は公表10日前まで100%上昇(+16.5%)、公表後は失速(-7.6%)。"
             "ただし『上がったから採用された』選択バイアス→事前予測が無いと取れない",
             fontsize=9.5,color="#444",pad=8)
fig.text(0.99,0.01,"データ: 2025-02〜2026-05 MSCIスタンダード採用9銘柄(Web収集) / TOPIX超過 / n=9・暫定",
         ha="right",va="bottom",fontsize=7.5,color="gray")
plt.tight_layout(rect=[0,0.02,1,0.93])
plt.savefig(OUT/"result2.png",dpi=100,bbox_inches="tight",facecolor="white")
print("saved result2.png")
