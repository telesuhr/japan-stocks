"""
v2: 半導体・非鉄バスケット + TOPIX で 7/17まで DD→回復分析。

理由: N225(price-weight)は値がさ半導体偏重でDDが深く出る一方、指数feedは7/16止まり。
stocks_dailyは7/17入り済みなので、ユーザーの実建玉に近い「半導体等加重・非鉄等加重バスケット」を
自前構築して7/17まで評価する。市場refはTOPIX(0000, 7/17まで)。
"""
import sys; sys.stdout.reconfigure(line_buffering=True)
from pathlib import Path
import numpy as np, pandas as pd
from jstock import db

HERE = Path(__file__).resolve().parent
SEMI = ["80350","68570","69200","77350","61460","77290","40630","34360","40620","67230","69630"]
def nf_codes():
    r = db.read_sql("SELECT code5 FROM symbol_master WHERE sector33_nm='非鉄金属' "
                    "AND market_nm='プライム' AND delisted_at IS NULL", [])
    return list(r["code5"])

def basket_index(codes):
    df = db.read_sql("SELECT code,date,adj_close FROM stocks_daily WHERE code=ANY(%s) "
                     "AND date>='2016-01-01' AND adj_close>0 ORDER BY date", [list(codes)])
    df["date"]=pd.to_datetime(df["date"])
    piv = df.pivot(index="date", columns="code", values="adj_close").sort_index()
    piv = piv.dropna(axis=1, thresh=int(len(piv)*0.5))     # 半分以上欠損の銘柄除外
    norm = piv / piv.bfill().iloc[0]                        # 各銘柄を初日=1
    idx = norm.mean(axis=1)                                 # 等加重
    return idx.dropna()

def analyze(name, idx):
    s = idx.copy()
    dates = s.index; close = s.values; N=len(s)
    peak = np.maximum.accumulate(close); dd = close/peak-1
    print(f"\n{'='*70}\n【{name}】{dates[0].date()}〜{dates[-1].date()} {N}営業日（等加重・adj_close）")
    print(f"現在DD(7/17): {dd[-1]*100:.1f}% / 直近10日 {(close[-1]/close[-11]-1)*100:.1f}% / 本日 {(close[-1]/close[-2]-1)*100:+.1f}%")
    # 回復日数 by depth
    rows=[]
    for D in [0.05,0.10,0.15,0.20,0.30]:
        rec=[]; ong=0; i=0
        while i<N:
            if dd[i]<=-D:
                pl=peak[i]; j=i
                while j<N and close[j]<pl: j+=1
                if j<N: rec.append(j-i)
                else: ong+=1
                i=j+1 if j<N else N
            else: i+=1
        if rec:
            a=np.array(rec)
            rows.append({"深度":f"-{int(D*100)}%","N":len(a),"中央値":int(np.median(a)),
                         "平均":int(a.mean()),"最長":int(a.max()),"未回復":ong})
        else:
            rows.append({"深度":f"-{int(D*100)}%","N":0,"中央値":None,"平均":None,"最長":None,"未回復":ong})
    R=pd.DataFrame(rows)
    print(R.to_string(index=False))
    # 条件付き先行リターン
    hz=[20,60,120,250]
    def fwd(mask,lab):
        ix=np.where(mask)[0]; ix=ix[ix<N-1]; o={"条件":lab,"日数":len(ix)}
        for h in hz:
            v=[close[k+h]/close[k]-1 for k in ix if k+h<N]
            o[f"+{h}d"]=f"{(np.array(v)>0).mean()*100:.0f}%/{np.median(v)*100:+.1f}" if v else "—"
        return o
    cr=[fwd(dd<=-x,f"DD≤-{int(x*100)}%") for x in [0.10,0.15,0.20,0.30]]
    cr.append(fwd(np.ones(N,bool),"無条件"))
    print(pd.DataFrame(cr).to_string(index=False))
    return dd[-1], R

# 市場ref TOPIX
t = db.read_sql("SELECT date, close FROM index_daily WHERE code='0000' ORDER BY date", [])
t["date"]=pd.to_datetime(t["date"]); t=t.set_index("date")["close"]
tdd = (t/t.cummax()-1).iloc[-1]
print(f"【TOPIX 参考】現在DD(7/17): {tdd*100:.1f}% / 本日 {(t.iloc[-1]/t.iloc[-2]-1)*100:+.1f}%")

semi_dd,_ = analyze("半導体バスケット(11銘柄・あなたの主戦場)", basket_index(SEMI))
nf_dd,_   = analyze("非鉄バスケット(プライム)", basket_index(nf_codes()))

print(f"\n{'='*70}\nまとめ: 7/17時点 現在DD  半導体 {semi_dd*100:.1f}% / 非鉄 {nf_dd*100:.1f}% / TOPIX {tdd*100:.1f}%")
