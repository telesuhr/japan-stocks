"""result.png: 採用銘柄は事前ランク上昇(検証1)も前向きバスケットは無エッジ(検証2)。"""
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

cp=pd.read_csv(OUT/"case_rank_path.csv")
fb=pd.read_csv(OUT/"forward_basket.csv")

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,6.75),facecolor="white")

# 左: 採用銘柄のランク推移（検証1）
xs=[-6,-3,-1,0]
for _,r in cp.iterrows():
    ys=[r["rank_-6m"],r["rank_-3m"],r["rank_-1m"],r["rank_pub"]]
    if any(pd.isna(ys)): continue
    ax1.plot(xs,ys,marker="o",alpha=0.7,label=r["name"])
ax1.invert_yaxis()
ax1.set_xlabel("公表までの月数",fontsize=10); ax1.set_ylabel("時価総額ランク(小さいほど大型)",fontsize=10)
ax1.set_title("検証1: 採用銘柄は公表3-6ヶ月前から\nランク上昇=事前に兆候はある",fontsize=10.5,pad=8)
ax1.legend(fontsize=7,ncol=2); ax1.grid(alpha=0.3)

# 右: 前向きバスケット vs ベンチ（検証2）
h=fb["hold_m"].astype(str)+"ヶ月"
x=np.arange(len(fb)); w=0.38
ax2.bar(x-w/2,fb["cand_mean_ex%"],w,label="rising候補バスケット",color="#2980b9",alpha=0.85)
ax2.bar(x+w/2,fb["band_mean_ex%"],w,label="同帯ベンチ(全銘柄)",color="#bdc3c7",alpha=0.85)
ax2.axhline(0,color="#333",lw=1); ax2.set_xticks(x); ax2.set_xticklabels(h,fontsize=9)
ax2.set_ylabel("TOPIX超過リターン 平均(%)",fontsize=10)
ax2.legend(fontsize=9); ax2.grid(axis="y",alpha=0.3)
ax2.set_title("検証2(本命): 候補を機械的に買っても\nベンチと差ゼロ=エッジ無し",fontsize=10.5,pad=8)

fig.suptitle("MSCI採用前ドリフト(+16.5%)は選択バイアス — 事前スクリーニングでは取れない",
             fontsize=13.5,fontweight="bold",y=0.98)
fig.text(0.99,0.01,"データ: 2022-01〜2026-05 月末時価総額ランク(JQuants ShOutFY) / 前向き678シグナル / MSCIリスト不使用",
         ha="right",va="bottom",fontsize=7.5,color="gray")
plt.tight_layout(rect=[0,0.02,1,0.93])
plt.savefig(OUT/"result.png",dpi=100,bbox_inches="tight",facecolor="white")
print("saved result.png")
