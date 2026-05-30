"""pead_obs.csv から分位ドリフトとL/S Sharpeを集計（単一プロセス・確定データ）。"""
import sys; sys.stdout.reconfigure(line_buffering=True)
from pathlib import Path
import numpy as np, pandas as pd
OUT=Path(__file__).parent
N_Q=10; HOLDS=[5,10,20]; COST_BPS=20.0; OOS_START="2024-01-01"

d=pd.read_csv(OUT/"pead_obs.csv")
d["entry_date"]=pd.to_datetime(d["entry_date"])
print(f"obs={len(d)} codes={d['code'].nunique()} days={d['entry_date'].nunique()}")

def assign_q(df):
    df=df.dropna(subset=["car0"]).copy()
    df["q"]=df.groupby("entry_date")["car0"].transform(
        lambda s: pd.qcut(s.rank(method="first"),N_Q,labels=False) if len(s)>=N_Q*2 else np.nan)
    return df.dropna(subset=["q"])

def quantile_table(df,label):
    df=assign_q(df); rows=[]
    for q in range(N_Q):
        x=df[df["q"]==q]
        rows.append({"label":label,"q":q,"n":len(x),"car0_%":round(x["car0"].mean()*100,2),
            "d5_bps":round(x["d5"].mean(),1),"d10_bps":round(x["d10"].mean(),1),"d20_bps":round(x["d20"].mean(),1)})
    return pd.DataFrame(rows)

def ls_eval(df,label):
    df=assign_q(df); out=[]
    for H in HOLDS:
        col=f"d{H}"; sub=df.dropna(subset=[col])
        wk=sub.groupby(["entry_date","q"])[col].mean().unstack("q")
        wk.columns=[int(c) for c in wk.columns]
        if 0 not in wk.columns or N_Q-1 not in wk.columns: continue
        ls=(wk[N_Q-1]-wk[0]).dropna()/1e4
        net=ls-COST_BPS/1e4
        ann=net.mean()/net.std()*np.sqrt(245/H) if net.std()>0 else np.nan
        out.append({"label":label,"hold":H,"n_days":len(ls),
            "LS_gross_bps":round(ls.mean()*1e4,1),"LS_net_bps":round(net.mean()*1e4,1),
            "LS_win":round((net>0).mean(),3),"ann_sharpe":round(ann,2)})
    return pd.DataFrame(out)

qt=pd.concat([quantile_table(d,"ALL"),quantile_table(d[d.entry_date>=OOS_START],"OOS")],ignore_index=True)
qt.to_csv(OUT/"quantile_drift.csv",index=False)
print("\n===== 分位別ドリフト(コスト前bps) car0=決算反応リターン 低(q0)→高(q9) =====")
print(qt.to_string(index=False))

res=pd.concat([ls_eval(d,"ALL"),ls_eval(d[d.entry_date<OOS_START],"IS"),
               ls_eval(d[d.entry_date>=OOS_START],"OOS")],ignore_index=True)
res.to_csv(OUT/"pead_summary.csv",index=False)
print("\n===== L/Sドリフト(Q9-Q0, コスト20bps後) =====")
print(res.to_string(index=False))
print("[DONE]")
