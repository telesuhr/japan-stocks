"""EDINET大量保有 機関集積→ドリフト。サイズ調整ベンチ＋翌寄りエントリ＋コスト往復20bp。
仮説: 積み増し銘柄アウト/減らす銘柄アンダー(機関ショート検証の買い手側の鏡)。
"""
import os, sys
sys.stdout.reconfigure(line_buffering=True)
import numpy as np, pandas as pd, psycopg2

PG={"host":os.environ.get("PGHOST","localhost"),"port":int(os.environ.get("PGPORT",5432)),
    "user":os.environ.get("PGUSER","postgres"),"password":os.environ.get("PGPASSWORD","postgres"),
    "dbname":os.environ.get("PGDATABASE","market_data")}
HORIZONS=[5,10,20,40]; COST_RT=0.0020; OOS=pd.Timestamp("2025-03-01"); HOLD=20

conn=psycopg2.connect(**PG)
ev=pd.read_sql("""SELECT issuer_code5 code, submit_date,
                    sum(COALESCE(ratio_change, holding_ratio)) net_chg,
                    bool_or(prev_holding_ratio IS NULL) has_new
                  FROM public.edinet_large_holdings
                  WHERE submit_date BETWEEN '2024-06-01' AND '2025-10-31' AND issuer_code5 IS NOT NULL
                  GROUP BY 1,2""",conn)
sd=pd.read_sql("SELECT code,date,adj_open,adj_close FROM stocks_daily WHERE code=ANY(%s) AND date>='2024-04-01'",conn,params=[list(ev.code.unique())])
idx=pd.read_sql("SELECT code,date,open,close FROM index_daily WHERE code IN ('0040','0041','0043','0045') AND date>='2024-04-01'",conn)
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

ev["submit_date"]=pd.to_datetime(ev["submit_date"]); rows=[]
for r in ev.itertuples():
    a=carr.get(r.code)
    if a is None: continue
    cd,co,cc=a
    ei=np.searchsorted(cd,np.datetime64(r.submit_date.date(),"D"),side="right")  # 翌営業日
    if ei<1 or ei>=len(cd): continue
    ed=cd[ei]
    if ed not in cal or co[ei]<=0: continue
    szc=code_size.get(r.code,"0045"); eo=co[ei]
    rec={"code":r.code,"entry_date":pd.Timestamp(ed),"net":float(r.net_chg),"new":bool(r.has_new)}
    for h in HORIZONS:
        xi=ei+h
        if xi>=len(cd): rec[f"d{h}"]=np.nan; continue
        szi=idx_ret(szc,ed,cd[xi])
        rec[f"d{h}"]=(cc[xi]/eo-1)-szi if szi is not None else np.nan
    rows.append(rec)

df=pd.DataFrame(rows).dropna(subset=["d20"])
print(f"events n={len(df)}  {df.entry_date.min().date()}~{df.entry_date.max().date()}")
df["q"]=pd.qcut(df["net"],5,labels=[1,2,3,4,5],duplicates="drop")

print("\n=== net_holding_change 五分位 → サイズ調整ドリフト(bp) [Q1=大幅減 … Q5=大幅積み増し] ===")
agg=df.groupby("q")[[f"d{h}" for h in HORIZONS]].mean()*1e4; agg["n"]=df.groupby("q").size()
print(agg.round(1).to_string())

print("\n=== 新規5%超(has_new) のみ → ドリフト(bp) ===")
nw=df[df.new]
print(f"  n={len(nw)}  d5={nw.d5.mean()*1e4:.1f}  d10={nw.d10.mean()*1e4:.1f}  d20={nw.d20.mean()*1e4:.1f}  d40={nw.d40.mean()*1e4:.1f}")

print(f"\n=== 戦略: Q5積み増し中立ロング/Q1減 中立ショート, 翌寄り{HOLD}日, 往復{COST_RT*1e4:.0f}bp ===")
for label,d in [("全",df),("IS(〜2025-02)",df[df.entry_date<OOS]),("OOS(2025-03〜)",df[df.entry_date>=OOS])]:
    lo=d[d.q==5][f"d{HOLD}"]-COST_RT; sh=-d[d.q==1][f"d{HOLD}"]-COST_RT
    ls=pd.concat([lo,sh])
    print(f"  {label:13} ロングQ5: n={len(lo):4d} net={lo.mean()*1e4:6.1f}bp | ショートQ1: n={len(sh):4d} net={sh.mean()*1e4:6.1f}bp | "
          f"LS net={ls.mean()*1e4:6.1f}bp Sh={ls.mean()/ls.std()*np.sqrt(252/HOLD):+.2f}")

# 可視化
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
try:
    import matplotlib.font_manager as fm
    fm.fontManager.addfont("/root/.fonts/NotoSansJP.ttf"); plt.rcParams["font.family"]="Noto Sans JP"; plt.rcParams["axes.unicode_minus"]=False
except Exception: pass
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(13.5,5.2)); qs=list(agg.index)
for h,c in [(5,"#bdc3c7"),(20,"#2980b9"),(40,"#27ae60")]:
    ax1.plot(qs,[agg.loc[q,f"d{h}"] for q in qs],"o-",label=f"{h}日",color=c,lw=1.8)
ax1.axhline(0,color="k",lw=.8); ax1.set_xlabel("net_holding_change 五分位 (1=大幅減 … 5=大幅積み増し)")
ax1.set_ylabel("サイズ調整 ドリフト (bp)"); ax1.set_title(f"EDINET機関集積 → その後ドリフト (n={len(df)}, 2024-06〜2025-10)\n積み増し→上昇/減→下落か"); ax1.legend(title="保有",fontsize=9)
lo=df[df.q==5].assign(p=lambda x:x[f"d{HOLD}"]-COST_RT)[["entry_date","p"]]
sh=df[df.q==1].assign(p=lambda x:-x[f"d{HOLD}"]-COST_RT)[["entry_date","p"]]
ls=pd.concat([lo,sh]).sort_values("entry_date")
ax2.plot(ls["entry_date"],ls["p"].cumsum()*100,color="#2980b9",lw=1.3)
ax2.axvline(OOS,color="red",ls="--",lw=1,alpha=.7); ax2.axhline(0,color="k",lw=.8)
ax2.set_ylabel("累積ネット (%, コスト込)"); ax2.set_title(f"戦略P&L: Q5ロング+Q1ショート(中立)\n翌寄り{HOLD}日・往復20bp")
fig.tight_layout(); fig.savefig(os.path.join(os.path.dirname(__file__),"result.png"),dpi=100,bbox_inches="tight")
print("\nsaved result.png")
