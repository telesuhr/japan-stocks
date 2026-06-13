"""機関ショート積み増し→ドリフト検証。サイズ調整ベンチ＋モメンタム制御。
仮説(informed-short): 積み増し→アンダー/カバー→アウト。結果は逆だったので、モメンタム代理かを制御で確認。
翌寄りエントリ(先読み排除)・コスト往復20bp。
"""
import os, sys
sys.stdout.reconfigure(line_buffering=True)
import numpy as np, pandas as pd, psycopg2

PG = {"host":os.environ.get("PGHOST","localhost"),"port":int(os.environ.get("PGPORT",5432)),
      "user":os.environ.get("PGUSER","postgres"),"password":os.environ.get("PGPASSWORD","postgres"),
      "dbname":os.environ.get("PGDATABASE","market_data")}
HORIZONS=[1,3,5,10,20]; COST_RT=0.0020; OOS_START=pd.Timestamp("2024-01-01"); HOLD=20

conn=psycopg2.connect(**PG)
ev=pd.read_sql("""SELECT code, disc_date, sum(shrt_pos_to_so - COALESCE(prev_rpt_ratio,0)) AS net_chg
                  FROM jquants_short_sale_report GROUP BY code, disc_date""",conn)
sd=pd.read_sql("SELECT code,date,adj_open,adj_close FROM stocks_daily WHERE code=ANY(%s) AND date>='2020-09-01'",conn,params=[list(ev.code.unique())])
idx=pd.read_sql("SELECT code,date,open,close FROM index_daily WHERE code IN ('0040','0041','0043','0045') AND date>='2020-09-01'",conn)
wt=pd.read_sql("SELECT code5,size_class FROM public.topix_weights WHERE ref_date='2026-04-30'",conn)
conn.close()

SIZE_MAP={"TOPIX Core30":"0040","TOPIX Large70":"0041","TOPIX Mid400":"0043","TOPIX Small 1":"0045","TOPIX Small 2":"0045"}
code_size={r.code5:SIZE_MAP.get(r.size_class,"0045") for r in wt.itertuples()}
idx["date"]=pd.to_datetime(idx["date"]); IDX={}
for c,g in idx.groupby("code"):
    g=g.sort_values("date")
    IDX[c]=({d:i for i,d in enumerate(g["date"].values.astype("datetime64[D]"))},g["open"].astype(float).values,g["close"].astype(float).values)
cal=IDX["0045"][0]
sd["date"]=pd.to_datetime(sd["date"]); sd=sd.sort_values(["code","date"])
carr={c:(g["date"].values.astype("datetime64[D]"),g["adj_open"].astype(float).values,g["adj_close"].astype(float).values) for c,g in sd.groupby("code")}

def idx_ret(code,ed,xd):
    m,o,c=IDX[code]; ti=m.get(ed); xi=m.get(xd)
    return (c[xi]/o[ti]-1) if (ti is not None and xi is not None) else None

ev["disc_date"]=pd.to_datetime(ev["disc_date"]); rows=[]
for r in ev.itertuples():
    a=carr.get(r.code)
    if a is None: continue
    cd,co,cc=a
    ei=np.searchsorted(cd,np.datetime64(r.disc_date.date(),"D"),side="right")
    if ei<21 or ei>=len(cd): continue
    ed=cd[ei]
    if ed not in cal or co[ei]<=0: continue
    szc=code_size.get(r.code,"0045"); eo=co[ei]
    pm_idx=idx_ret(szc,cd[ei-20],cd[ei-1])
    prior_mom=(cc[ei-1]/co[ei-20]-1)-pm_idx if pm_idx is not None else np.nan   # 直前20日 size調整
    rec={"code":r.code,"entry_date":pd.Timestamp(ed),"net":float(r.net_chg),"mom":prior_mom}
    for h in HORIZONS:
        xi=ei+h
        if xi>=len(cd): rec[f"d{h}"]=np.nan; continue
        szi=idx_ret(szc,ed,cd[xi])
        rec[f"d{h}"]=(cc[xi]/eo-1)-szi if szi is not None else np.nan
    rows.append(rec)

df=pd.DataFrame(rows).dropna(subset=["d20","mom"])
print(f"events n={len(df)}  {df.entry_date.min().date()}~{df.entry_date.max().date()}")
df["q"]=pd.qcut(df["net"],5,labels=[1,2,3,4,5])

print("\n=== net_short_change 五分位 → サイズ調整ドリフト(bp) [Q1=大幅カバー … Q5=大幅積み増し] ===")
agg=df.groupby("q")[[f"d{h}" for h in HORIZONS]].mean()*1e4; agg["n"]=df.groupby("q").size()
print(agg.round(1).to_string())

# モメンタム制御 OLS: d20 ~ z(net) + z(mom)
def z(s): return (s-s.mean())/s.std()
def ols(y,X,names):
    X=np.column_stack([np.ones(len(X))]+[X[c].values for c in X.columns]); names=["const"]+names
    b,*_=np.linalg.lstsq(X,y,rcond=None); e=y-X@b; dof=len(y)-X.shape[1]
    se=np.sqrt(np.diag((e@e)/dof*np.linalg.inv(X.T@X)))
    return pd.DataFrame({"coef_bp":b*1e4,"t":b/se},index=names)
reg=df.assign(znet=z(df.net),zmom=z(df.mom))
print("\n=== OLS: d20 ~ z(net_short_change) + z(prior_mom20)  [モメンタム制御で net が残るか] ===")
print(ols(reg["d20"].values, reg[["znet","zmom"]].rename(columns={"znet":"znet","zmom":"zmom"}), ["znet","zmom"]).round(3).to_string())
print("単独 d20~z(net):")
print(ols(reg["d20"].values, reg[["znet"]], ["znet"]).round(3).to_string())

print(f"\n=== 戦略(翌寄り{HOLD}日・往復20bp) ===")
for name,sgnQ5,sgnQ1 in [("仮説どおり(Q5ショート/Q1ロング)",-1,+1),("逆=データ順(Q5ロング/Q1ショート)",+1,-1)]:
    for label,d in [("全",df),("IS",df[df.entry_date<OOS_START]),("OOS",df[df.entry_date>=OOS_START])]:
        ls=pd.concat([sgnQ5*d[d.q==5][f"d{HOLD}"]-COST_RT, sgnQ1*d[d.q==1][f"d{HOLD}"]-COST_RT])
        if label=="全": print(f"  {name}: 全 net={ls.mean()*1e4:6.1f}bp Sh={ls.mean()/ls.std()*np.sqrt(252/HOLD):+.2f}", end="")
        else: print(f" | {label} {ls.mean()*1e4:6.1f}bp", end="")
    print()

# 可視化
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
try:
    import matplotlib.font_manager as fm
    fm.fontManager.addfont("/root/.fonts/NotoSansJP.ttf"); plt.rcParams["font.family"]="Noto Sans JP"; plt.rcParams["axes.unicode_minus"]=False
except Exception: pass
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(13.5,5.2)); qs=[1,2,3,4,5]
for h,c in [(1,"#bdc3c7"),(5,"#2980b9"),(20,"#27ae60")]:
    ax1.plot(qs,[agg.loc[q,f"d{h}"] for q in qs],"o-",label=f"{h}日",color=c,lw=1.8)
ax1.axhline(0,color="k",lw=.8); ax1.set_xlabel("net_short_change 五分位 (1=大幅カバー … 5=大幅積み増し)")
ax1.set_ylabel("サイズ調整 ドリフト (bp)"); ax1.set_title(f"機関ショート増減 → その後ドリフト (n={len(df)})\n仮説の逆: 積み増し→上昇/カバー→下落"); ax1.legend(title="保有",fontsize=9)
# 逆=データ順の戦略(Q5ロング+Q1ショート)
ls=pd.concat([df[df.q==5].assign(p=lambda x:x[f"d{HOLD}"]-COST_RT)[["entry_date","p"]],
              df[df.q==1].assign(p=lambda x:-x[f"d{HOLD}"]-COST_RT)[["entry_date","p"]]]).sort_values("entry_date")
ax2.plot(ls["entry_date"],ls["p"].cumsum()*100,color="#27ae60",lw=1.3,label="Q5ロング+Q1ショート(データ順)")
ax2.axvline(OOS_START,color="red",ls="--",lw=1,alpha=.7); ax2.axhline(0,color="k",lw=.8)
ax2.set_ylabel("累積ネット (%, コスト込)"); ax2.set_title(f"戦略P&L(データ順・中立{HOLD}日・往復20bp)"); ax2.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(os.path.dirname(__file__),"result.png"),dpi=100,bbox_inches="tight")
print("\nsaved result.png")
