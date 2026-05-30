"""result.png (1200x675): 信用残ファクター — コスト前は弱いシグナル、コスト後は消える。"""
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

df=pd.read_csv(OUT/"factor_panel.csv",low_memory=False)
df["date"]=pd.to_datetime(df["date"],errors="coerce")
N_Q=10; WK=52
def lss(f):
    sub=df.dropna(subset=[f]).copy(); sub=sub[np.isfinite(sub[f])]
    def qw(x):
        if x[f].nunique()<N_Q: return pd.Series(np.nan,index=x.index)
        try: return pd.qcut(x[f].rank(method="first"),N_Q,labels=False)
        except: return pd.Series(np.nan,index=x.index)
    sub["q"]=sub.groupby("date",group_keys=False).apply(qw)
    sub=sub.dropna(subset=["q"])
    wk=sub.groupby(["date","q"])["fwd_ret"].mean().unstack("q")
    ls=(wk[N_Q-1]-wk[0]).dropna()
    return ls.mean()/ls.std()*np.sqrt(WK)

factors=["margin_ratio","long_chg","net_chg","short_chg","short_ratio"]
jp={"margin_ratio":"信用倍率","long_chg":"信用買残増","net_chg":"需給変化","short_chg":"信用売残増","short_ratio":"売残比率"}
gross=[abs(lss(f)) for f in factors]   # 実行方向の絶対値（コスト前）
# コスト後: 週次40bps を実行方向に1回引く
import pandas as pd2
def net_after(f):
    sub=df.dropna(subset=[f]).copy(); sub=sub[np.isfinite(sub[f])]
    def qw(x):
        if x[f].nunique()<N_Q: return pd.Series(np.nan,index=x.index)
        try: return pd.qcut(x[f].rank(method="first"),N_Q,labels=False)
        except: return pd.Series(np.nan,index=x.index)
    sub["q"]=sub.groupby("date",group_keys=False).apply(qw)
    sub=sub.dropna(subset=["q"])
    wk=sub.groupby(["date","q"])["fwd_ret"].mean().unstack("q")
    ls=(wk[N_Q-1]-wk[0]).dropna()
    ex=ls*np.sign(ls.mean())            # 実行方向に揃える
    ex=ex-40/1e4
    return ex.mean()/ex.std()*np.sqrt(WK)
net=[net_after(f) for f in factors]

fig,ax=plt.subplots(figsize=(12,6.75),facecolor="white")
x=np.arange(len(factors)); w=0.38
ax.bar(x-w/2,gross,w,label="コスト前（実行方向）",color="#16a085",alpha=0.85)
ax.bar(x+w/2,net,w,label="週40bpsコスト控除後",color="#c0392b",alpha=0.85)
ax.axhline(0,color="#333",lw=1)
ax.axhline(2.0,color="#999",ls="--",lw=1); ax.text(len(factors)-0.5,2.03,"昇格基準2.0",fontsize=8,color="#777",ha="right")
ax.set_xticks(x); ax.set_xticklabels([jp[f] for f in factors],fontsize=10)
ax.set_ylabel("年率 Sharpe（週次LS分位スプレッド・ベータ中立）",fontsize=10)
ax.legend(fontsize=10); ax.grid(axis="y",alpha=0.3)
fig.suptitle("信用残ファクター: 弱いシグナルはあるがコストで消える（全上場・週次LS・5年）",
             fontsize=13.5,fontweight="bold",y=0.98)
ax.set_title("コスト前ですら最大Sharpe0.4（信用倍率）。週次リバランスのコスト40bpsで全ファクター負け＝実行不可",
             fontsize=9.5,color="#444",pad=8)
fig.text(0.99,0.01,"データ: 2021-01〜2026-05 / 日本株週次信用残(JQuants) / 流動性≥10億円・1170銘柄・256週",
         ha="right",va="bottom",fontsize=7.5,color="gray")
plt.tight_layout(rect=[0,0.02,1,0.94])
plt.savefig(OUT/"result.png",dpi=100,bbox_inches="tight",facecolor="white")
print("saved result.png")
