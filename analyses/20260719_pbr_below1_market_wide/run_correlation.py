"""
バリューL/S（割安Q1−割高Q5）と既存採用系列の相関測定。
同一月次パネルで 6MモメンタムL/S を再構築し、バリューL/Sとの相関を測る。
バリュー×モメンタムは教科書的に逆相関＝分散寄与を定量化。合成50/50バスケットのSharpeも。
"""
import sys; sys.stdout.reconfigure(line_buffering=True)
from pathlib import Path
import numpy as np, pandas as pd
from jstock import db, costs, stats

HERE=Path(__file__).resolve().parent
LIQ=1e8; TSE="2023-03-31"
FIN=("銀行業","保険業","証券･商品先物取引業","その他金融業")

px=db.read_sql("""
  WITH m AS (SELECT code,date,close,adj_close,turnover_value,
      date_trunc('month',date)::date mo,
      row_number() OVER (PARTITION BY code,date_trunc('month',date) ORDER BY date DESC) rn,
      avg(turnover_value) OVER (PARTITION BY code,date_trunc('month',date)) tv_avg
    FROM stocks_daily WHERE date>='2016-01-01' AND adj_close>0 AND close>0)
  SELECT code,mo,date me_date,close rawc,adj_close adjc,tv_avg FROM m WHERE rn=1 ORDER BY code,mo""",[])
fin=db.read_sql("""SELECT code,disc_date,NULLIF(payload->>'BPS','')::float bps
  FROM fin_summary WHERE NULLIF(payload->>'BPS','') IS NOT NULL AND (payload->>'BPS')::float>0
  ORDER BY disc_date""",[])
sm=db.read_sql("SELECT code5 code,sector33_nm FROM symbol_master",[])
px["me_date"]=pd.to_datetime(px["me_date"]); px["mo"]=pd.to_datetime(px["mo"])
fin["disc_date"]=pd.to_datetime(fin["disc_date"])
for c in ["rawc","adjc","tv_avg"]: px[c]=px[c].astype(float)
px=px.sort_values("me_date"); fin=fin.sort_values("disc_date")
m=pd.merge_asof(px,fin,by="code",left_on="me_date",right_on="disc_date",direction="backward").merge(sm,on="code",how="left")
m["pbr"]=m["rawc"]/m["bps"]
m=m.sort_values(["code","mo"])
m["fwd"]=m.groupby("code")["adjc"].shift(-1)/m["adjc"]-1
m["ret6m"]=m["adjc"]/m.groupby("code")["adjc"].shift(6)-1     # 過去6Mモメンタム
m=m[~m["sector33_nm"].isin(FIN)]
m=m[(m["pbr"]>0)&(m["pbr"]<10)&(m["tv_avg"]>=LIQ)&m["fwd"].notna()]

val=[]; mom=[]
for mo,g in m.groupby("mo"):
    if len(g)<50: continue
    g=g.copy()
    g["vq"]=pd.qcut(g["pbr"],5,labels=[1,2,3,4,5],duplicates="drop")
    val.append((mo, g.loc[g["vq"]==1,"fwd"].mean()-g.loc[g["vq"]==5,"fwd"].mean()))   # 割安-割高
    gm=g[g["ret6m"].notna()]
    if len(gm)>=50:
        gm=gm.copy(); gm["mq"]=pd.qcut(gm["ret6m"],5,labels=[1,2,3,4,5],duplicates="drop")
        mom.append((mo, gm.loc[gm["mq"]==5,"fwd"].mean()-gm.loc[gm["mq"]==1,"fwd"].mean()))  # 勝ち-負け
V=pd.Series(dict(val)).sort_index(); M=pd.Series(dict(mom)).sort_index()
df=pd.concat([V.rename("value"),M.rename("mom")],axis=1).dropna()

def sm_(s,label):
    d=costs.net_returns(s,ls=True); r=pd.Series(d).dropna()
    return f"{label:26} ann={r.mean()*12*100:6.2f}%  Sharpe={stats.sharpe(r,ann=12):5.2f}  t={stats.t_stat(r):5.2f}"

print(f"共通月数 {len(df)}  ({df.index.min().date()}〜{df.index.max().date()})\n")
print(sm_(df["value"],"バリューL/S(割安-割高)"))
print(sm_(df["mom"],"6MモメンタムL/S(勝-負)"))
print(f"\n★ バリューL/S × モメンタムL/S 相関(gross月次): {df['value'].corr(df['mom']):+.3f}")

# 合成50/50
combo=0.5*df["value"]+0.5*df["mom"]
print("\n"+sm_(combo,"合成50/50 (val+mom)"))
# 全期間 & OOS
for nm,mask in [("全期間",df.index>="2000-01-01"),("OOS(2023-04〜)",df.index>="2023-04-01")]:
    cv=costs.net_returns(df["value"][mask],ls=True); cm=costs.net_returns(df["mom"][mask],ls=True)
    cc=costs.net_returns(combo[mask],ls=True)
    print(f"  [{nm:12}] value Sh={stats.sharpe(pd.Series(cv).dropna(),ann=12):4.2f}  "
          f"mom Sh={stats.sharpe(pd.Series(cm).dropna(),ann=12):4.2f}  "
          f"合成 Sh={stats.sharpe(pd.Series(cc).dropna(),ann=12):4.2f}  ρ={df['value'][mask].corr(df['mom'][mask]):+.2f}")
