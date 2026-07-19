"""
低PBR効果のサイズ中立化。
時価総額 mcap = raw close × (Eq/BPS=発行済株数) をPIT推定し、
サイズ3分位 × PBR5分位の2wayソートで「純バリュー効果」を分離する。
- サイズ内でPBR分位を切る → サイズ中立な割安Q1−割高Q5 L/S
- 各サイズ帯で低PBRが効くか（小型だけの現象でないか）を確認
"""
import sys; sys.stdout.reconfigure(line_buffering=True)
from pathlib import Path
import numpy as np, pandas as pd
from jstock import db, costs, stats

HERE = Path(__file__).resolve().parent
TSE_REQUEST = "2023-03-31"; LIQ = 1e8
FIN_SECTORS = ("銀行業", "保険業", "証券･商品先物取引業", "その他金融業")

px = db.read_sql("""
  WITH m AS (
    SELECT code, date, close, adj_close, turnover_value,
           date_trunc('month',date)::date mo,
           row_number() OVER (PARTITION BY code, date_trunc('month',date) ORDER BY date DESC) rn,
           avg(turnover_value) OVER (PARTITION BY code, date_trunc('month',date)) tv_avg
    FROM stocks_daily WHERE date>='2016-01-01' AND adj_close>0 AND close>0)
  SELECT code, mo, date me_date, close rawc, adj_close adjc, tv_avg
  FROM m WHERE rn=1 ORDER BY code, mo""", [])
fin = db.read_sql("""
  SELECT code, disc_date, NULLIF(payload->>'BPS','')::float bps,
         NULLIF(payload->>'NP','')::float np, NULLIF(payload->>'Eq','')::float eq
  FROM fin_summary WHERE NULLIF(payload->>'BPS','') IS NOT NULL AND (payload->>'BPS')::float>0
  ORDER BY disc_date""", [])
sm = db.read_sql("SELECT code5 code, sector33_nm FROM symbol_master", [])

px["me_date"]=pd.to_datetime(px["me_date"]); px["mo"]=pd.to_datetime(px["mo"])
fin["disc_date"]=pd.to_datetime(fin["disc_date"])
for c in ["rawc","adjc","tv_avg"]: px[c]=px[c].astype(float)
px=px.sort_values("me_date"); fin=fin.sort_values("disc_date")
m=pd.merge_asof(px, fin, by="code", left_on="me_date", right_on="disc_date", direction="backward")
m=m.merge(sm, on="code", how="left")
m["pbr"]=m["rawc"]/m["bps"]
m["mcap"]=np.where(m["bps"]>0, m["rawc"]*m["eq"]/m["bps"], np.nan)   # = price×株数
m=m.sort_values(["code","mo"]); m["fwd"]=m.groupby("code")["adjc"].shift(-1)/m["adjc"]-1
m=m[~m["sector33_nm"].isin(FIN_SECTORS)]                            # 金融除外
m=m[(m["pbr"]>0)&(m["pbr"]<10)&(m["tv_avg"]>=LIQ)&m["fwd"].notna()&m["mcap"].notna()&(m["mcap"]>0)]

def msum(s):
    r=pd.Series(s).dropna()
    if len(r)<3: return dict(n=len(r),ann=np.nan,sh=np.nan,t=np.nan)
    return dict(n=len(r), ann=r.mean()*12*100, sh=stats.sharpe(r,ann=12), t=stats.t_stat(r))

# ---- サイズ3分位 × PBR5分位 2wayソート ----
grid_fwd={}   # (size,pbrq) -> monthly series
sn_ls=[]      # サイズ中立 割安-割高 の月次
naive_ls=[]   # 参考: サイズ無視の Q1-Q5
by_size_ls={"S":[],"M":[],"L":[]}
months=[]
for mo,g in m.groupby("mo"):
    if len(g)<60: continue
    g=g.copy()
    g["sz"]=pd.qcut(g["mcap"],3,labels=["S","M","L"])
    parts_cheap=[]; parts_rich=[]
    ok=True
    for sz in ["S","M","L"]:
        gs=g[g["sz"]==sz]
        if len(gs)<15: ok=False; break
        gs=gs.copy(); gs["vq"]=pd.qcut(gs["pbr"],5,labels=["Q1","Q2","Q3","Q4","Q5"],duplicates="drop")
        for vq in ["Q1","Q2","Q3","Q4","Q5"]:
            grid_fwd.setdefault((sz,vq),{})[mo]=gs.loc[gs["vq"]==vq,"fwd"].mean()
        c=gs.loc[gs["vq"]=="Q1","fwd"].mean(); r=gs.loc[gs["vq"]=="Q5","fwd"].mean()
        by_size_ls[sz].append((mo, c-r))
        parts_cheap.append(c); parts_rich.append(r)
    if not ok: continue
    # サイズ各帯を等加重した中立L/S
    sn_ls.append((mo, np.mean(parts_cheap)-np.mean(parts_rich)))
    # naive (全体でPBR分位)
    g["vq_all"]=pd.qcut(g["pbr"],5,labels=["Q1","Q2","Q3","Q4","Q5"],duplicates="drop")
    naive_ls.append((mo, g.loc[g["vq_all"]=="Q1","fwd"].mean()-g.loc[g["vq_all"]=="Q5","fwd"].mean()))
    months.append(mo)

sn=pd.Series(dict(sn_ls)).sort_index()
nv=pd.Series(dict(naive_ls)).sort_index()

print(f"月数 {len(sn)}  ({sn.index.min().date()}〜{sn.index.max().date()})")
print("\n=== 割安Q1−割高Q5 L/S（コスト後8bp/月・年率） ===")
for nm,s in [("① サイズ無視(naive・元検証)", nv), ("② サイズ中立(サイズ内でPBR分位→等加重)", sn)]:
    d=msum(costs.net_returns(s, ls=True))
    print(f"  {nm:38} ann={d['ann']:6.2f}%  Sharpe={d['sh']:5.2f}  t={d['t']:5.2f}")

print("\n=== 各サイズ帯での 割安−割高 L/S（純バリューが全サイズで効くか・コスト後） ===")
for sz,lab in [("S","小型"),("M","中型"),("L","大型")]:
    s=pd.Series(dict(by_size_ls[sz])).sort_index()
    d=msum(costs.net_returns(s, ls=True))
    print(f"  {lab}(mcap下位/中位/上位1/3内): ann={d['ann']:6.2f}%  Sharpe={d['sh']:5.2f}  t={d['t']:5.2f}  月数{d['n']}")

print("\n=== サイズ×PBR 3x5グリッド 平均月次リターン%（gross・行=サイズ 列=PBR分位） ===")
tbl=pd.DataFrame({(sz,vq): pd.Series(grid_fwd[(sz,vq)]) for sz in ["S","M","L"] for vq in ["Q1","Q2","Q3","Q4","Q5"]})
grid_mean=tbl.mean()*100
out=pd.DataFrame(index=["S小型","M中型","L大型"],columns=["Q1割安","Q2","Q3","Q4","Q5割高","Q1-Q5"])
for sz,lab in [("S","S小型"),("M","M中型"),("L","L大型")]:
    for vq,col in [("Q1","Q1割安"),("Q2","Q2"),("Q3","Q3"),("Q4","Q4"),("Q5","Q5割高")]:
        out.loc[lab,col]=f"{grid_mean[(sz,vq)]:+.2f}"
    out.loc[lab,"Q1-Q5"]=f"{grid_mean[(sz,'Q1')]-grid_mean[(sz,'Q5')]:+.2f}"
print(out.to_string())

# IS/OOS（サイズ中立L/S）
print("\n=== サイズ中立L/S の IS/OOS（東証要請2023-03） ===")
for nm,mask in [("IS(〜2023-03)", sn.index<=TSE_REQUEST), ("OOS(2023-04〜)", sn.index>="2023-04-01")]:
    d=msum(costs.net_returns(sn[mask], ls=True))
    print(f"  {nm:16} ann={d['ann']:6.2f}%  Sharpe={d['sh']:5.2f}  t={d['t']:5.2f}  月数{d['n']}")
