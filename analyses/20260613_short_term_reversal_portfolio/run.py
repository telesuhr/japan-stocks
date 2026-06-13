"""20日短期リバーサルの分散ポートフォリオbacktest。敗者ロング/勝者ショート・市場中立・週次。
コスト命: 実ターンオーバー×片側bpを控除しネットSharpeで判定(教訓2)。IS/OOS・コスト感応度。
"""
import os, sys
sys.stdout.reconfigure(line_buffering=True)
import numpy as np, pandas as pd, psycopg2

PG={"host":os.environ.get("PGHOST","localhost"),"port":int(os.environ.get("PGPORT",5432)),
    "user":os.environ.get("PGUSER","postgres"),"password":os.environ.get("PGPASSWORD","postgres"),
    "dbname":os.environ.get("PGDATABASE","market_data")}
START="2017-01-01"; OOS=pd.Timestamp("2023-01-01")
NUNIV=500; FORM=20; SKIP=1; REBAL=5; QFRAC=0.2; PERSIDE_BP=10

conn=psycopg2.connect(**PG)
# 流動性上位~1200コードに限定して読み込み
top=pd.read_sql("""SELECT code FROM stocks_daily WHERE date>=%s GROUP BY code
                   HAVING avg(turnover_value)>0 ORDER BY avg(turnover_value) DESC NULLS LAST LIMIT 1200""",conn,params=[START])
codes=top.code.tolist()
df=pd.read_sql("SELECT code,date,adj_close,turnover_value FROM stocks_daily WHERE code=ANY(%s) AND date>=%s",conn,params=[codes,START])
conn.close()

df["date"]=pd.to_datetime(df["date"])
px=df.pivot(index="date",columns="code",values="adj_close").sort_index().astype(float)
tov=df.pivot(index="date",columns="code",values="turnover_value").sort_index().astype(float)
ret=px.pct_change()
dates=px.index
liq=tov.rolling(60,min_periods=40).mean()        # トレイル流動性
print(f"loaded {px.shape[1]} codes, {len(dates)} days ({dates[0].date()}~{dates[-1].date()})")

# backtest
pnl_gross=pd.Series(0.0,index=dates); cost_ser=pd.Series(0.0,index=dates); turn_ser=pd.Series(0.0,index=dates)
w_prev=pd.Series(dtype=float)
rebal_pts=range(FORM+SKIP+1, len(dates)-1, REBAL)
for r in rebal_pts:
    d=dates[r]
    # ユニバース: トレイル流動性 上位NUNIV
    lq=liq.iloc[r]
    univ=lq.dropna().nlargest(NUNIV).index
    # シグナル: トレイル20日リターン(直近SKIP日skip)  px[r-SKIP]/px[r-SKIP-FORM]-1
    sig=(px.iloc[r-SKIP]/px.iloc[r-SKIP-FORM]-1).reindex(univ).dropna()
    # 価格が当日有効な銘柄のみ
    sig=sig[px.iloc[r].reindex(sig.index).notna()]
    if len(sig)<100: continue
    k=int(len(sig)*QFRAC)
    losers=sig.nsmallest(k).index   # ロング(敗者)
    winners=sig.nlargest(k).index   # ショート(勝者)
    w=pd.Series(0.0,index=sig.index)
    w[losers]=1.0/len(losers); w[winners]=-1.0/len(winners)
    # ターンオーバー&コスト
    allc=w.index.union(w_prev.index)
    wn=w.reindex(allc).fillna(0.0); wp=w_prev.reindex(allc).fillna(0.0)
    turnover=0.5*(wn-wp).abs().sum()
    cost=turnover*PERSIDE_BP*1e-4
    cost_ser.iloc[r]+=cost; turn_ser.iloc[r]=turnover
    # 保有: r+1 .. r+REBAL の日次P&L
    end=min(r+REBAL,len(dates)-1)
    rr=ret.iloc[r+1:end+1]
    daily=(rr[losers].mean(axis=1)-rr[winners].mean(axis=1))
    pnl_gross.loc[daily.index]+=daily.values
    w_prev=w

net=pnl_gross-cost_ser
def stats(s,label):
    s=s[s.index>=dates[FORM+SKIP+1]]
    ann=s.mean()*252; sh=s.mean()/s.std()*np.sqrt(252) if s.std()>0 else 0
    eq=s.cumsum(); dd=(eq-eq.cummax()).min()
    print(f"  {label:18} 年率={ann*100:6.2f}%  Sharpe={sh:5.2f}  最大DD={dd*100:6.1f}%  勝率={(s>0).mean()*100:.0f}%")
    return sh
print("\n=== 全期間 ===")
stats(pnl_gross,"グロス"); stats(net,f"ネット({PERSIDE_BP}bp/片側)")
ann_turn=turn_ser.sum()/((dates[-1]-dates[FORM]).days/365)
print(f"  年間ターンオーバー≈{ann_turn:.1f}回  (1回転≈往復{2*PERSIDE_BP}bp)")
# 損益分岐
g_ann=pnl_gross[pnl_gross.index>=dates[FORM]].mean()*252
be=g_ann/(turn_ser.sum()/((dates[-1]-dates[FORM]).days/365))/1e-4 if turn_ser.sum()>0 else 0
print(f"  コスト損益分岐 ≈ 片側 {be:.1f}bp (これ未満ならネット黒字)")

print("\n=== IS(〜2022) / OOS(2023〜) ネット ===")
for label,mask in [("IS",net.index<OOS),("OOS",net.index>=OOS)]:
    stats(net[mask],label)

print("\n=== コスト感応度(ネットSharpe) ===")
for bp in [0,5,10,15,20]:
    n=pnl_gross-cost_ser*(bp/PERSIDE_BP)
    n=n[n.index>=dates[FORM+SKIP+1]]
    print(f"  片側{bp:2d}bp: Sharpe={n.mean()/n.std()*np.sqrt(252):5.2f}  年率={n.mean()*252*100:6.2f}%")

# 可視化
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
try:
    import matplotlib.font_manager as fm
    fm.fontManager.addfont("/root/.fonts/NotoSansJP.ttf"); plt.rcParams["font.family"]="Noto Sans JP"; plt.rcParams["axes.unicode_minus"]=False
except Exception: pass
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(13.5,5.2))
sl=pnl_gross[pnl_gross.index>=dates[FORM+SKIP+1]]
ax1.plot(sl.index,sl.cumsum()*100,label="グロス",color="#27ae60",lw=1.4)
ax1.plot(net[net.index>=dates[FORM+SKIP+1]].index,net[net.index>=dates[FORM+SKIP+1]].cumsum()*100,label=f"ネット({PERSIDE_BP}bp/片側)",color="#2980b9",lw=1.4)
ax1.axvline(OOS,color="red",ls="--",lw=1,alpha=.7); ax1.axhline(0,color="k",lw=.8)
ax1.set_ylabel("累積リターン (%)"); ax1.set_title("20日リバーサル 敗者ロング/勝者ショート\n週次・売買代金上位500・市場中立"); ax1.legend(fontsize=9)
bars=[0,5,10,15,20]; shs=[]
for bp in bars:
    n=(pnl_gross-cost_ser*(bp/PERSIDE_BP)); n=n[n.index>=dates[FORM+SKIP+1]]
    shs.append(n.mean()/n.std()*np.sqrt(252))
ax2.bar([str(b) for b in bars],shs,color=["#27ae60" if s>0 else "#c0392b" for s in shs])
ax2.axhline(0,color="k",lw=.8); ax2.set_xlabel("片側コスト (bp)"); ax2.set_ylabel("ネット Sharpe")
ax2.set_title("コスト感応度: 何bpまで黒字か")
fig.tight_layout(); fig.savefig(os.path.join(os.path.dirname(__file__),"result.png"),dpi=100,bbox_inches="tight")
print("\nsaved result.png")
