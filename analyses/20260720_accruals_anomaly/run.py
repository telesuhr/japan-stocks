"""
会計発生高アノマリー（Sloan 1996）全市場検証。
accr=(NP-CFO)/平均TA。FY本決算のみ・PIT・月次評価・コスト後・IS/OOS・セクター中立・既存系列と相関。
"""
import sys; sys.stdout.reconfigure(line_buffering=True)
from pathlib import Path
import numpy as np, pandas as pd
from jstock import db, costs, stats

HERE=Path(__file__).resolve().parent
LIQ=1e8; SPLIT="2021-01-01"
FIN=("銀行業","保険業","証券･商品先物取引業","その他金融業")

# ---- 月末パネル ----
px=db.read_sql("""
  WITH m AS (SELECT code,date,close,adj_close,turnover_value,
      date_trunc('month',date)::date mo,
      row_number() OVER (PARTITION BY code,date_trunc('month',date) ORDER BY date DESC) rn,
      avg(turnover_value) OVER (PARTITION BY code,date_trunc('month',date)) tv_avg
    FROM stocks_daily WHERE date>='2016-01-01' AND adj_close>0)
  SELECT code,mo,date me_date,close rawc,adj_close adjc,tv_avg FROM m WHERE rn=1 ORDER BY code,mo""",[])
# ---- FY本決算のみ: NP, CFO, TA, BPS(バリュー用) ----
fin=db.read_sql("""
  SELECT code, disc_date,
         NULLIF(payload->>'NP','')::float np,
         NULLIF(payload->>'CFO','')::float cfo,
         NULLIF(payload->>'TA','')::float ta,
         NULLIF(payload->>'BPS','')::float bps
  FROM fin_summary
  WHERE doc_type LIKE 'FY%%FinancialStatements%%'
    AND NULLIF(payload->>'CFO','') IS NOT NULL AND NULLIF(payload->>'TA','') IS NOT NULL
  ORDER BY code, disc_date""",[])
sm=db.read_sql("SELECT code5 code, sector33_nm FROM symbol_master",[])
print(f"月末パネル {len(px):,} / FY財務(CFO&TA有) {len(fin):,} / master {len(sm):,}")

px["me_date"]=pd.to_datetime(px["me_date"]); px["mo"]=pd.to_datetime(px["mo"])
fin["disc_date"]=pd.to_datetime(fin["disc_date"])
for c in ["rawc","adjc","tv_avg"]: px[c]=px[c].astype(float)

# 発生高: accr=(NP-CFO)/平均TA（当期TAと前期TAの平均）
fin=fin.sort_values(["code","disc_date"])
fin["ta_prev"]=fin.groupby("code")["ta"].shift(1)
fin["avg_ta"]=fin[["ta","ta_prev"]].mean(axis=1)
fin["accr"]=(fin["np"]-fin["cfo"])/fin["avg_ta"]
fin=fin[fin["avg_ta"]>0]

# PIT結合（月末に直近FY）
px=px.sort_values("me_date"); fin=fin.sort_values("disc_date")
m=pd.merge_asof(px,fin[["code","disc_date","accr","bps"]],by="code",
                left_on="me_date",right_on="disc_date",direction="backward").merge(sm,on="code",how="left")
m["pbr"]=m["rawc"]/m["bps"]
m=m.sort_values(["code","mo"])
m["fwd"]=m.groupby("code")["adjc"].shift(-1)/m["adjc"]-1
m["ret6m"]=m["adjc"]/m.groupby("code")["adjc"].shift(6)-1
m=m[~m["sector33_nm"].isin(FIN)]
base=m[(m["tv_avg"]>=LIQ)&m["fwd"].notna()&m["accr"].notna()].copy()
# 発生高の極端外れ値を除去（比率なので±1にクリップ相当のwinsor）
lo,hi=base["accr"].quantile([0.01,0.99]); base=base[(base["accr"]>=lo)&(base["accr"]<=hi)]

def msum(s,ls=True,rt=1.0):
    d=costs.net_returns(pd.Series(s).dropna(), round_trips=rt, ls=ls); r=pd.Series(d).dropna()
    if len(r)<3: return dict(n=len(r),ann=np.nan,sh=np.nan,t=np.nan)
    return dict(n=len(r),ann=r.mean()*12*100,sh=stats.sharpe(r,ann=12),t=stats.t_stat(r))

# ---- H1/H2 分位（accr昇順: Q1=低発生高=高クオリティ） ----
qser={q:{} for q in ["Q1","Q2","Q3","Q4","Q5"]}; ls={}; mkt={}
accr_ls_sn={}   # セクター中立L/S
val_ls={}; mom_ls={}
for mo,g in base.groupby("mo"):
    if len(g)<50: continue
    g=g.copy(); g["q"]=pd.qcut(g["accr"],5,labels=["Q1","Q2","Q3","Q4","Q5"],duplicates="drop")
    for q in ["Q1","Q2","Q3","Q4","Q5"]: qser[q][mo]=g.loc[g["q"]==q,"fwd"].mean()
    ls[mo]=g.loc[g["q"]=="Q1","fwd"].mean()-g.loc[g["q"]=="Q5","fwd"].mean()
    mkt[mo]=g["fwd"].mean()
    # セクター中立: 各sector33内でaccrを標準化しトップ/ボトム3分位相当（sector内 low-high）
    parts=[]
    for _,gs in g.groupby("sector33_nm"):
        if len(gs)<6: continue
        gs=gs.copy(); gs["r"]=gs["accr"].rank(pct=True)
        parts.append(gs.loc[gs["r"]<=0.33,"fwd"].mean()-gs.loc[gs["r"]>=0.67,"fwd"].mean())
    if parts: accr_ls_sn[mo]=np.nanmean(parts)
    # バリューL/S(割安-割高) / 6MモメンタムL/S(勝-負) 同月
    gv=g[g["pbr"].notna()&(g["pbr"]>0)&(g["pbr"]<10)]
    if len(gv)>=50:
        gv=gv.copy(); gv["vq"]=pd.qcut(gv["pbr"],5,labels=[1,2,3,4,5],duplicates="drop")
        val_ls[mo]=gv.loc[gv["vq"]==1,"fwd"].mean()-gv.loc[gv["vq"]==5,"fwd"].mean()
    gm=g[g["ret6m"].notna()]
    if len(gm)>=50:
        gm=gm.copy(); gm["mq"]=pd.qcut(gm["ret6m"],5,labels=[1,2,3,4,5],duplicates="drop")
        mom_ls[mo]=gm.loc[gm["mq"]==5,"fwd"].mean()-gm.loc[gm["mq"]==1,"fwd"].mean()

Q={q:pd.Series(qser[q]).sort_index() for q in qser}
LSs=pd.Series(ls).sort_index(); SN=pd.Series(accr_ls_sn).sort_index()
V=pd.Series(val_ls).sort_index(); M=pd.Series(mom_ls).sort_index()
n_mo=len(LSs)
print(f"\n有効月数 {n_mo}  ({LSs.index.min().date()}〜{LSs.index.max().date()})  月平均 {base.groupby('mo').size().mean():.0f}銘柄")

print("\n[H1] 発生高5分位 等加重 long-only（コスト後・年率）  Q1=低発生高(高クオリティ) … Q5=高発生高")
for q in ["Q1","Q2","Q3","Q4","Q5"]:
    d=msum(Q[q],ls=False,rt=1.0); print(f"  {q}: ann={d['ann']:6.2f}%  Sharpe={d['sh']:5.2f}  t={d['t']:5.2f}")
dm=msum(pd.Series(mkt),ls=False,rt=1.0); print(f"  市場EW: ann={dm['ann']:6.2f}%  Sharpe={dm['sh']:5.2f}")

print("\n[H2] 低発生高Q1 Long − 高発生高Q5 Short L/S（コスト後8bp/月）")
d=msum(LSs,ls=True); print(f"  全期間: ann={d['ann']:6.2f}%  Sharpe={d['sh']:5.2f}  t={d['t']:5.2f}  月数{d['n']}")
for nm,mask in [("IS(〜2020-12)",LSs.index<SPLIT),("OOS(2021-01〜)",LSs.index>=SPLIT)]:
    d=msum(LSs[mask],ls=True); print(f"  {nm:16} ann={d['ann']:6.2f}%  Sharpe={d['sh']:5.2f}  t={d['t']:5.2f}  月数{d['n']}")

print("\n[H3] セクター中立版（sector33内 低-高発生高 L/S・コスト後）")
d=msum(SN,ls=True); print(f"  全期間: ann={d['ann']:6.2f}%  Sharpe={d['sh']:5.2f}  t={d['t']:5.2f}")

print("\n[H4] 既存系列との相関（gross月次）＝分散寄与")
df=pd.concat([LSs.rename("accr"),V.rename("value"),M.rename("mom")],axis=1).dropna()
print(f"  accr×value ρ={df['accr'].corr(df['value']):+.3f} | accr×momentum ρ={df['accr'].corr(df['mom']):+.3f}  (共通{len(df)}ヶ月)")
combo=(df["accr"]+df["value"])/2
dc=msum(combo,ls=True); print(f"  合成 accr+value 50/50: Sharpe={dc['sh']:5.2f}  ann={dc['ann']:6.2f}%")

# ---- 可視化 ----
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.font_manager as fm
try:
    fm.fontManager.addfont("/root/.fonts/NotoSansJP.ttf"); plt.rcParams["font.family"]="Noto Sans JP"
    plt.rcParams["axes.unicode_minus"]=False
except Exception: pass
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(14,6.75),facecolor="white",gridspec_kw={"width_ratios":[1.5,1]})
cum=(1+costs.net_returns(LSs,ls=True)).cumprod()
ax1.plot(cum.index,cum.values,color="#1f6feb",lw=2,label="低発生高−高発生高 L/S")
ax1.axhline(1,color="k",lw=0.8); ax1.axvline(pd.Timestamp(SPLIT),color="gray",ls=":")
ax1.set_title("発生高L/S 累積(コスト後・金融除外)"); ax1.set_ylabel("成長率(倍)"); ax1.legend(); ax1.grid(alpha=0.3)
anns=[msum(Q[q],ls=False,rt=1.0)["ann"] for q in ["Q1","Q2","Q3","Q4","Q5"]]
ax2.bar(["Q1低","Q2","Q3","Q4","Q5高"],anns,color=["#2ca02c","#7ac07a","#bbb","#e08a8a","#d62728"])
ax2.axhline(0,color="k",lw=0.8); ax2.set_title("発生高分位別 年率(コスト後%)\nQ1=低発生高(高質) → Q5=高発生高")
for i,v in enumerate(anns): ax2.text(i,v,f"{v:.1f}",ha="center",va="bottom" if v>=0 else "top",fontsize=9)
ax2.grid(alpha=0.3,axis="y")
fig.suptitle("会計発生高アノマリー（全市場・非金融・2016-2026）",fontsize=15); fig.tight_layout()
fig.savefig(HERE/"result.png",dpi=100,bbox_inches="tight",facecolor="white")
print("\nsaved result.png")
