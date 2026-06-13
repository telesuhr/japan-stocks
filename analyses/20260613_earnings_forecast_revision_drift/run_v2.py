"""v2: サイズ交絡を除去して再検証。
v1で寄り後ドリフトが全分位一律マイナス→修正銘柄が中小型偏重で大型加重TOPIXにsize負けの疑い。
2つの size-clean ベンチで再評価:
  (A) 横断de-mean: 各イベントのドリフト − 同月の全修正銘柄の平均ドリフト (100%カバー・先読み無)
  (B) サイズ別指数: TOPIX Core30/Large70/Mid400/Small 指数を銘柄サイズに合わせて控除 (分類可能分)
"""
import os, sys, datetime
sys.stdout.reconfigure(line_buffering=True)
import numpy as np, pandas as pd, psycopg2
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

PG = {"host": os.environ.get("PGHOST","localhost"),"port":int(os.environ.get("PGPORT",5432)),
      "user":os.environ.get("PGUSER","postgres"),"password":os.environ.get("PGPASSWORD","postgres"),
      "dbname":os.environ.get("PGDATABASE","market_data")}
HORIZONS=[1,3,5,10]; COST_RT=0.0020; OOS_START=pd.Timestamp("2023-01-01"); HOLD=5

conn=psycopg2.connect(**PG)
rev=pd.read_sql("SELECT code,disc_date,disc_time,rev_op_pct FROM public.earnings_forecast_revisions WHERE rev_op_pct IS NOT NULL",conn)
sd=pd.read_sql("SELECT code,date,adj_open,adj_close FROM stocks_daily WHERE code=ANY(%s) AND date>='2016-01-01'",conn,params=[list(rev.code.unique())])
idx=pd.read_sql("SELECT code,date,open,close FROM index_daily WHERE code IN ('0000','0040','0041','0043','0045') AND date>='2016-01-01'",conn)
wt=pd.read_sql("SELECT code5,size_class FROM public.topix_weights WHERE ref_date='2026-04-30'",conn)
conn.close()

# サイズ分類 -> 指数コード (非TOPIXはSmall 0045で代用)
SIZE_MAP={"TOPIX Core30":"0040","TOPIX Large70":"0041","TOPIX Mid400":"0043","TOPIX Small 1":"0045","TOPIX Small 2":"0045"}
code_size={r.code5: SIZE_MAP.get(r.size_class,"0045") for r in wt.itertuples()}

# 指数: code -> (date_idx, open[], close[])
idx["date"]=pd.to_datetime(idx["date"])
IDX={}
for c,g in idx.groupby("code"):
    g=g.sort_values("date")
    IDX[c]=({d:i for i,d in enumerate(g["date"].values.astype("datetime64[D]"))},
            g["open"].astype(float).values, g["close"].astype(float).values)
top_idx=IDX["0000"][0]

sd["date"]=pd.to_datetime(sd["date"]); sd=sd.sort_values(["code","date"])
carr={c:(g["date"].values.astype("datetime64[D]"),g["adj_open"].astype(float).values,g["adj_close"].astype(float).values) for c,g in sd.groupby("code")}

def idx_ret(code,entry_date,exit_date):
    m,o,c=IDX[code]; ti=m.get(entry_date); xi=m.get(exit_date)
    if ti is None or xi is None: return None
    return c[xi]/o[ti]-1

rev["disc_date"]=pd.to_datetime(rev["disc_date"])
rows=[]
for r in rev.itertuples():
    a=carr.get(r.code)
    if a is None: continue
    cd,co,cc=a
    disc=np.datetime64(r.disc_date.date(),"D")
    preopen=(r.disc_time is not None) and (r.disc_time<datetime.time(9,0))
    ei=np.searchsorted(cd,disc,side=("left" if preopen else "right"))
    if ei<1 or ei>=len(cd): continue
    entry_date=cd[ei]
    if entry_date not in top_idx or top_idx[entry_date]<1: continue
    eo=co[ei]
    if eo<=0: continue
    szc=code_size.get(r.code,"0045")
    rec={"code":r.code,"entry_date":pd.Timestamp(entry_date),"month":pd.Timestamp(entry_date).to_period("M"),
         "op":float(r.rev_op_pct),"szc":szc}
    for h in HORIZONS:
        xi=ei+h
        if xi>=len(cd): rec[f"raw{h}"]=np.nan; continue
        exit_date=cd[xi]
        stock=cc[xi]/eo-1
        tpx=idx_ret("0000",entry_date,exit_date)
        szi=idx_ret(szc,entry_date,exit_date)
        rec[f"raw{h}"]=stock
        rec[f"vsTPX{h}"]=(stock-tpx) if tpx is not None else np.nan
        rec[f"vsSIZE{h}"]=(stock-szi) if szi is not None else np.nan
    rows.append(rec)

df=pd.DataFrame(rows).dropna(subset=["raw5"])
df["q"]=pd.qcut(df["op"],5,labels=[1,2,3,4,5])
print(f"events n={len(df)}  {df.entry_date.min().date()}~{df.entry_date.max().date()}")

# (A) 横断de-mean: raw - 同月平均raw
for h in HORIZONS:
    df[f"dm{h}"]=df[f"raw{h}"]-df.groupby("month")[f"raw{h}"].transform("mean")

def qtab(col):
    t=df.groupby("q")[col].mean()*1e4
    return [round(t.loc[q],1) for q in [1,2,3,4,5]]

print("\n=== 五分位 d5 超過(bp): ベンチ3種 比較 ===")
print(f"  {'Q':<3}{'vs TOPIX(v1)':>14}{'横断de-mean(A)':>16}{'vs サイズ指数(B)':>18}")
for i,q in enumerate([1,2,3,4,5]):
    a=df[df.q==q]["vsTPX5"].mean()*1e4
    b=df[df.q==q]["dm5"].mean()*1e4
    c=df[df.q==q]["vsSIZE5"].mean()*1e4
    print(f"  Q{q:<2}{a:>14.1f}{b:>16.1f}{c:>18.1f}")

# 戦略再評価: size-clean ベンチで Q5/Q1
def strat(col):
    out={}
    for label,d in [("全",df),("IS",df[df.entry_date<OOS_START]),("OOS",df[df.entry_date>=OOS_START])]:
        ls=pd.concat([d[d.q==5][col]*(+1)-COST_RT, d[d.q==1][col]*(-1)-COST_RT])
        l=d[d.q==5][col].mean()*1e4-COST_RT*1e4
        out[label]=(l, ls.mean()*1e4, ls.mean()/ls.std()*np.sqrt(252/HOLD))
    return out

print("\n=== 戦略(翌寄り5日・往復20bp) size-cleanベンチでの Q5ロング net / LS合算 net / Sharpe ===")
for name,col in [("vs TOPIX(v1)","vsTPX5"),("de-mean(A)","dm5"),("サイズ指数(B)","vsSIZE5")]:
    s=strat(col)
    print(f"  {name:14} 全:Q5L={s['全'][0]:6.1f} LS={s['全'][1]:6.1f}(Sh{s['全'][2]:+.2f}) | IS LS={s['IS'][1]:6.1f} | OOS LS={s['OOS'][1]:6.1f}")

# 可視化: ベンチ別 五分位 d5
try:
    import matplotlib.font_manager as fm
    fm.fontManager.addfont("/root/.fonts/NotoSansJP.ttf"); plt.rcParams["font.family"]="Noto Sans JP"; plt.rcParams["axes.unicode_minus"]=False
except Exception: pass
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(13.5,5.2))
qs=[1,2,3,4,5]
ax1.plot(qs,[df[df.q==q]["vsTPX5"].mean()*1e4 for q in qs],"o-",label="vs TOPIX (v1・交絡あり)",color="#95a5a6")
ax1.plot(qs,[df[df.q==q]["dm5"].mean()*1e4 for q in qs],"s-",label="横断de-mean (A)",color="#2980b9",lw=2)
ax1.plot(qs,[df[df.q==q]["vsSIZE5"].mean()*1e4 for q in qs],"^-",label="vs サイズ指数 (B)",color="#27ae60",lw=2)
ax1.axhline(0,color="k",lw=0.8); ax1.set_xlabel("rev_op_pct 五分位 (1=大幅下方 … 5=大幅上方)")
ax1.set_ylabel("寄り後5日 超過リターン (bp)"); ax1.set_title("サイズ交絡 除去後の5日ドリフト\n水準が消えるか/相対パターンが残るか"); ax1.legend(fontsize=8)
# de-mean 戦略 累積
ls=pd.concat([df[df.q==5].assign(p=lambda x:x["dm5"]-COST_RT)[["entry_date","p"]],
              df[df.q==1].assign(p=lambda x:-x["dm5"]-COST_RT)[["entry_date","p"]]]).sort_values("entry_date")
ax2.plot(ls["entry_date"],ls["p"].cumsum()*100,color="#2980b9",lw=1.3,label="de-mean基準 Q5L+Q1S")
ax2.axvline(OOS_START,color="red",ls="--",lw=1,alpha=.7); ax2.axhline(0,color="k",lw=.8)
ax2.set_ylabel("累積ネット (%, コスト込)"); ax2.set_title("size-cleanベンチでの戦略P&L"); ax2.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(os.path.dirname(__file__),"result_v2.png"),dpi=100,bbox_inches="tight")
print("\nsaved result_v2.png")
