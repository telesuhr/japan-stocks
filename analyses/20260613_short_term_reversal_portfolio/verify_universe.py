"""流動性3層で20日リバーサルのグロス成績を比較=小型ほど効くか(マイクロストラクチャ起源か)を裏取り。"""
import os,sys; sys.stdout.reconfigure(line_buffering=True)
import numpy as np, pandas as pd, psycopg2
PG={"host":"localhost","port":5432,"user":"postgres","password":"postgres","dbname":"market_data"}
START="2017-01-01"; FORM=20; SKIP=1; REBAL=5; QFRAC=0.2
conn=psycopg2.connect(**PG)
top=pd.read_sql("""SELECT code, avg(turnover_value) av FROM stocks_daily WHERE date>=%s GROUP BY code
                   HAVING avg(turnover_value)>0 ORDER BY av DESC NULLS LAST LIMIT 1800""",conn,params=[START])
codes=top.code.tolist()
df=pd.read_sql("SELECT code,date,adj_close,turnover_value FROM stocks_daily WHERE code=ANY(%s) AND date>=%s",conn,params=[codes,START])
conn.close()
df["date"]=pd.to_datetime(df["date"])
px=df.pivot(index="date",columns="code",values="adj_close").sort_index().astype(float)
tov=df.pivot(index="date",columns="code",values="turnover_value").sort_index().astype(float)
ret=px.pct_change(); dates=px.index; liq=tov.rolling(60,min_periods=40).mean()

def backtest(rank_lo,rank_hi,label):
    pnl=pd.Series(0.0,index=dates)
    for r in range(FORM+SKIP+1,len(dates)-1,REBAL):
        lq=liq.iloc[r].dropna().sort_values(ascending=False)
        univ=lq.index[rank_lo:rank_hi]
        sig=(px.iloc[r-SKIP]/px.iloc[r-SKIP-FORM]-1).reindex(univ).dropna()
        sig=sig[px.iloc[r].reindex(sig.index).notna()]
        if len(sig)<60: continue
        k=int(len(sig)*QFRAC)
        lo=sig.nsmallest(k).index; wi=sig.nlargest(k).index
        end=min(r+REBAL,len(dates)-1); rr=ret.iloc[r+1:end+1]
        daily=rr[lo].mean(axis=1)-rr[wi].mean(axis=1)
        pnl.loc[daily.index]+=daily.values
    s=pnl[pnl.index>=dates[FORM+SKIP+1]]
    print(f"  {label:22} グロス年率={s.mean()*252*100:6.2f}%  Sharpe={s.mean()/s.std()*np.sqrt(252):5.2f}")

print("=== 流動性3層 20日リバーサル(グロス・敗者ロング/勝者ショート) ===")
backtest(0,300,"上位 1-300(超大型)")
backtest(300,700,"401-700(大中型)")
backtest(700,1300,"701-1300(中小型)")
