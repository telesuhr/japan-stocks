"""result.png (1200x675): 信用残ファクターのLSスプレッド（符号反転=実行可能方向）。"""
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

r=pd.read_csv(OUT/"factor_summary.csv")
# 実行方向 = 符号反転（Q10-Q1がマイナスなので、低分位Long/高分位Short）
r["sharpe_actionable"]=-r["LS_sharpe_ann"]
jp={"margin_ratio":"信用倍率\n(買残/売残)","long_chg":"信用買残\n増加","short_chg":"信用売残\n増加",
    "short_ratio":"売残比率","net_chg":"需給変化\n(買-売)"}
order=["margin_ratio","long_chg","net_chg","short_ratio","short_chg"]

fig,ax=plt.subplots(figsize=(12,6.75),facecolor="white")
x=np.arange(len(order)); w=0.26
for i,(lab,col) in enumerate([("ALL","#16a085"),("IS","#2980b9"),("OOS","#8e44ad")]):
    vals=[r[(r.label==lab)&(r.factor==f)]["sharpe_actionable"].iloc[0] for f in order]
    ax.bar(x+(i-1)*w,vals,w,label=lab,color=col,alpha=0.85)
ax.axhline(0,color="#333",lw=1)
ax.axhline(2.0,color="#c0392b",ls="--",lw=1,alpha=0.6)
ax.text(len(order)-0.5,2.02,"昇格基準 Sharpe=2.0",color="#c0392b",fontsize=8,ha="right",va="bottom")
ax.set_xticks(x); ax.set_xticklabels([jp[f] for f in order],fontsize=9)
ax.set_ylabel("年率 Sharpe（実行方向・ベータ中立LS・コスト40bps/週控除後）",fontsize=10)
ax.legend(fontsize=10,title="期間"); ax.grid(axis="y",alpha=0.3)
fig.suptitle("信用残ファクターに本物のエッジ — 「信用買いの重し」効果（全上場・週次LS・5年）",
             fontsize=13.5,fontweight="bold",y=0.98)
ax.set_title("信用倍率の低い銘柄をLong/高い銘柄をShort → 年率Sharpe ~1.6（IS/OOS一貫, OOSで強化）",
             fontsize=10,color="#444",pad=8)
fig.text(0.99,0.01,"データ: 2021-01〜2026-05 / 日本株週次信用残(JQuants jquants_margin_interest) / 流動性≥10億円・746銘柄",
         ha="right",va="bottom",fontsize=7.5,color="gray")
plt.tight_layout(rect=[0,0.02,1,0.94])
plt.savefig(OUT/"result.png",dpi=100,bbox_inches="tight",facecolor="white")
print("saved result.png")
