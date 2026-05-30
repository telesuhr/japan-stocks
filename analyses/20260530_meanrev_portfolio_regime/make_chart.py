"""result.png (1200x675): 平均回帰のαは市場ベータの幻だったことを可視化。"""
import sys; sys.stdout.reconfigure(line_buffering=True)
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd

OUT=Path(__file__).parent
fp="/root/.fonts/NotoSansJP.ttf"
if Path(fp).exists():
    font_manager.fontManager.addfont(fp)
    plt.rcParams["font.family"]=font_manager.FontProperties(fname=fp).get_name()
plt.rcParams["axes.unicode_minus"]=False

summ=pd.read_csv(OUT/"portfolio_summary.csv")
yr=pd.read_csv(OUT/"by_year.csv")

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,6.75),facecolor="white",gridspec_kw={"width_ratios":[1,1.25]})

# 左: 生 vs ベータ中立 Sharpe (ALL/IS/OOS)
labels=["ALL","IS\n(16-21)","OOS\n(22-26)"]
raw=[summ[summ.label=="RAW_ALL"].sharpe.iloc[0], summ[summ.label=="RAW_IS"].sharpe.iloc[0], summ[summ.label=="RAW_OOS"].sharpe.iloc[0]]
neu=[summ[summ.label.str.startswith("NEUTRAL_ALL")].sharpe.iloc[0], summ[summ.label=="NEUTRAL_IS"].sharpe.iloc[0], summ[summ.label=="NEUTRAL_OOS"].sharpe.iloc[0]]
import numpy as np
x=np.arange(3); w=0.38
ax1.bar(x-w/2,raw,w,label="生リターン",color="#7f8c8d",alpha=0.85)
ax1.bar(x+w/2,neu,w,label="ベータ中立(真のα)",color="#c0392b",alpha=0.85)
ax1.axhline(0,color="#333",lw=1); ax1.set_xticks(x); ax1.set_xticklabels(labels,fontsize=9)
ax1.set_ylabel("年率 Sharpe",fontsize=11); ax1.legend(fontsize=9); ax1.grid(axis="y",alpha=0.3)
ax1.set_title("ベータ中立化で全期間マイナス\n= 平均回帰のαは市場ベータの幻",fontsize=11,pad=8)
for i,(r,n) in enumerate(zip(raw,neu)):
    ax1.text(i-w/2,r+(0.02 if r>=0 else -0.02),f"{r:.2f}",ha="center",va="bottom" if r>=0 else "top",fontsize=8)
    ax1.text(i+w/2,n+(0.02 if n>=0 else -0.02),f"{n:.2f}",ha="center",va="bottom" if n>=0 else "top",fontsize=8)

# 右: 年別 生Sharpe（バラつき=レジーム依存）
yr=yr.rename(columns={yr.columns[0]:"year"})
colors=["#2980b9" if v>=0 else "#c0392b" for v in yr["sharpe_raw"]]
ax2.bar(yr["year"].astype(str),yr["sharpe_raw"],color=colors,alpha=0.85)
ax2.axhline(0,color="#333",lw=1); ax2.grid(axis="y",alpha=0.3)
ax2.set_ylabel("年率 Sharpe (生)",fontsize=11)
ax2.set_title("年別Sharpeは+2.0〜−1.3で乱高下\n= 安定したエッジではない",fontsize=11,pad=8)
ax2.tick_params(axis="x",rotation=90,labelsize=8)

fig.suptitle("平均回帰(RSI25逆張り)を全上場でポートフォリオ化 → 真のαは無し（10年・流動性≥10億円）",
             fontsize=13,fontweight="bold",y=0.99)
fig.text(0.99,0.01,"データ: 2016-05〜2026-05 / 日本株日足(JQuants) / 平均34銘柄保有 / TOPIXベータ0.85",
         ha="right",va="bottom",fontsize=7.5,color="gray")
plt.tight_layout(rect=[0,0.02,1,0.94])
plt.savefig(OUT/"result.png",dpi=100,bbox_inches="tight",facecolor="white")
print("saved result.png")
