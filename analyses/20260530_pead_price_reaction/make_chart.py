"""result.png: 決算反応の分位別ドリフト(リバーサル)とL/S Sharpe。"""
import sys; sys.stdout.reconfigure(line_buffering=True)
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np, pandas as pd
OUT=Path(__file__).parent
fp="/root/.fonts/NotoSansJP.ttf"
if Path(fp).exists():
    font_manager.fontManager.addfont(fp); plt.rcParams["font.family"]=font_manager.FontProperties(fname=fp).get_name()
plt.rcParams["axes.unicode_minus"]=False
qt=pd.read_csv(OUT/"quantile_drift.csv"); rs=pd.read_csv(OUT/"pead_summary.csv")
a=qt[qt.label=="ALL"]

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,6.75),facecolor="white")
# 左: 分位別20日ドリフト(単調リバーサル)
colors=["#27ae60" if v>=0 else "#c0392b" for v in a["d20_bps"]]
ax1.bar(a["q"],a["d20_bps"],color=colors,alpha=0.85)
ax1.axhline(0,color="#333",lw=1); ax1.set_xticks(range(10))
ax1.set_xlabel("決算反応リターンの分位 (q0=暴落 → q9=急騰)",fontsize=10)
ax1.set_ylabel("決算翌日以降20営業日のTOPIX超過(bps)",fontsize=10)
ax1.set_title("決算で売られた株は戻し、買われた株は萎む\n=オーバーリアクションの単調リバーサル",fontsize=10.5,pad=8)
ax1.grid(axis="y",alpha=0.3)
# 右: リバーサルL/S Sharpe (保有別, IS/OOS)
piv=rs.pivot_table(index="hold",columns="label",values="ann_sharpe")
x=np.arange(len(piv)); w=0.26
for i,lab in enumerate(["ALL","IS","OOS"]):
    if lab in piv.columns: ax2.bar(x+(i-1)*w,piv[lab],w,label=lab,alpha=0.85)
ax2.axhline(2.0,color="#999",ls="--",lw=1); ax2.text(len(piv)-0.5,2.03,"昇格基準2.0",fontsize=8,color="#777",ha="right")
ax2.axhline(0,color="#333",lw=1); ax2.set_xticks(x); ax2.set_xticklabels([f"{int(h)}日保有" for h in piv.index],fontsize=9)
ax2.set_ylabel("年率Sharpe(リバーサルL/S, コスト20bps後)",fontsize=10)
ax2.legend(fontsize=9,title="期間"); ax2.grid(axis="y",alpha=0.3)
ax2.set_title("損失銘柄Long/勝者Shortで20日Sharpe~1.0\nIS/OOS一貫(今セッション初の生存シグナル)",fontsize=10.5,pad=8)
fig.suptitle("決算オーバーリアクション・リバーサル — 全上場・ベータ中立L/S・5年",fontsize=14,fontweight="bold",y=0.98)
fig.text(0.99,0.01,"データ: 2021-2026 決算17.4万件(JQuants fin_summary) / 反応翌日エントリー / 流動性≥10億円 / TOPIX超過",
         ha="right",va="bottom",fontsize=7.5,color="gray")
plt.tight_layout(rect=[0,0.02,1,0.94])
plt.savefig(OUT/"result.png",dpi=100,bbox_inches="tight",facecolor="white")
print("saved")
