"""result.png (1200x675): MSCIリバランス引けイベントの月末集中と翌日リターン。"""
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

ev=pd.read_csv(OUT/"events.csv",parse_dates=["dt"])
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,6.75),facecolor="white",gridspec_kw={"width_ratios":[1.1,1]})

# 左: 月別イベント件数（MSCI月が突出）
ev["m"]=ev["dt"].dt.month
cnt=ev.groupby("m").size().reindex(range(1,13),fill_value=0)
colors=["#c0392b" if m in (2,5,8,11) else "#bdc3c7" for m in range(1,13)]
ax1.bar(range(1,13),cnt.values,color=colors)
ax1.set_xticks(range(1,13)); ax1.set_xticklabels([f"{m}月" for m in range(1,13)],fontsize=8)
ax1.set_ylabel("引けスパイク・イベント件数",fontsize=10)
ax1.set_title("赤=MSCIリバランス月(2/5/8/11月)に集中\n→ 引け出来高スパイク=リバランス痕跡",fontsize=10.5,pad=8)
ax1.grid(axis="y",alpha=0.3)

# 右: MSCI窓イベントの翌日リターン(引け方向別)
m=ev[ev["dt"].dt.month.isin([2,5,8,11])&(ev["dt"].dt.day>=24)].dropna(subset=["fwd_oc_bps"])
up=m[m["close_jump_bps"]>0]; dn=m[m["close_jump_bps"]<0]
cats=["引け上昇\nイベント\n(買われた)","引け下落\nイベント\n(売られた)"]
oc=[up["fwd_oc_bps"].mean(),dn["fwd_oc_bps"].mean()]
wr=[(up["fwd_oc_bps"]>0).mean()*100,(dn["fwd_oc_bps"]>0).mean()*100]
x=np.arange(2)
b=ax2.bar(x,oc,0.5,color=["#27ae60","#2980b9"],alpha=0.85)
ax2.set_xticks(x); ax2.set_xticklabels(cats,fontsize=9)
ax2.set_ylabel("翌日 寄→引 平均リターン (bps, コスト前)",fontsize=10)
ax2.axhline(0,color="#333",lw=1); ax2.grid(axis="y",alpha=0.3)
for i,(v,w,n) in enumerate(zip(oc,wr,[len(up),len(dn)])):
    ax2.text(i,v+1,f"+{v:.0f}bps\n勝率{w:.0f}%\nn={n}",ha="center",va="bottom",fontsize=9)
ax2.set_title("翌日リターンは勝率48%以下=コイン投げ\n現象は実在も事前に取れるエッジは未確認",fontsize=10.5,pad=8)

fig.suptitle("MSCIリバランス『引けイベント』は毎度起きる — が、取れるエッジは確認できず",
             fontsize=14,fontweight="bold",y=0.99)
fig.text(0.99,0.01,"データ: 2024-05〜2026-05 / 日本株1分足(JQuants) / 引け出来高比≥8%&中央値比≥4倍 / 流動性≥10億円 / コスト前",
         ha="right",va="bottom",fontsize=7,color="gray")
plt.tight_layout(rect=[0,0.02,1,0.94])
plt.savefig(OUT/"result.png",dpi=100,bbox_inches="tight",facecolor="white")
print("saved result.png")
