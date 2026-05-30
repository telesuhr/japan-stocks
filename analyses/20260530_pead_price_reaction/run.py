"""
価格反応ベースPEAD（決算後ドリフト）のクロスセクション検証
================================================================
今セッション6連敗（板/ブレイク/平均回帰/信用残/MSCI×2）の総括:
  「全員が同時に見るシグナル(テクニカル・指数イベント)は裁定で消える。
   取れるのは情報・反応に時間差があるもの」。決算は全銘柄同時に出ず消化に
   時間差→ドリフトが残る(PEAD)。既存昇格 earnings_pead(Sharpe2.19)は
   ファンダ・サプライズ(OP vs 会社予想)ベース。本分析は未開拓の
   **価格反応ベースPEAD**(Bernard-Thomas型)を全ユニバースで検証する。

着想:
  決算サプライズの大きさは「決算後の市場の初動リターン」が最も素直に織り込む
  (アナリストもbankerも見るがそれでも織り込みに時間差)。サプライズ数値を使わず、
  決算翌日の価格反応を10分位に分け、その後N日のドリフト継続をL/Sで測る。
  構造的ベータ中立(L/S)で、前段の「ベータの幻」を最初から回避。

データ: fin_summary(決算日 disc_date, disc_time) + stocks_daily(adj価格) + index_daily(TOPIX)
        2021-2026, 約18万決算イベント, 90%が引け後発表

イベント定義(look-ahead無し):
  - 各(code, disc_date)を1イベントに集約(同日複数開示は最初の1行)
  - 引け後発表(disc_time>=15:00): 反応日 react = disc_dateの翌営業日
    引け前発表(disc_time<15:00): 反応日 react = disc_date当日
  - 反応リターン CAR0 = react日の TOPIX超過(open→close, 寄りで織り込み済み部分も含むため
    保守的に react前日close → react close も併記)
  - エントリー = react の翌営業日(react+1)の寄り。ドリフト = react+1 open → react+1+H close
    (反応を「観測してから」入るので look-ahead無し)
  - 各「エントリー日」でCAR0を10分位 → 翌H日ドリフトのL/S(Q10-Q1)

評価: H=5/10/20営業日。L/Sドリフトの年率Sharpe(週次相当に補正)、平均bps、勝率。
      コスト往復20bps(L/Sなので両サイド)控除。IS(2021-2023)/OOS(2024-2026)。
"""
import sys; sys.stdout.reconfigure(line_buffering=True)
import os
from pathlib import Path
import numpy as np, pandas as pd
import psycopg2

OUT=Path(__file__).parent
PG=dict(host=os.environ.get("PGHOST","localhost"),port=int(os.environ.get("PGPORT",5432)),
        user=os.environ.get("PGUSER","postgres"),password=os.environ.get("PGPASSWORD","postgres"),
        dbname=os.environ.get("PGDATABASE","market_data"))

ADV_FLOOR=1e9
N_Q=10
HOLDS=[5,10,20]
COST_BPS=20.0
OOS_START="2024-01-01"
START="2021-01-01"

def load(conn):
    print("  loading earnings events...")
    ev=pd.read_sql("""SELECT DISTINCT ON (code, disc_date) code, disc_date, disc_time
        FROM fin_summary WHERE disc_date>=%(s)s ORDER BY code, disc_date, disc_time""",
        conn,params={"s":START})
    ev["disc_date"]=pd.to_datetime(ev["disc_date"])
    ev["after_close"]=ev["disc_time"].astype(str)>='15:00:00'
    print("  loading daily prices...")
    px=pd.read_sql("""SELECT code,date,adj_open,adj_close,turnover_value FROM stocks_daily
        WHERE date>=%(s)s AND code IN (SELECT code5 FROM symbol_master WHERE market_nm IN ('プライム','スタンダード','グロース'))
        ORDER BY code,date""",conn,params={"s":START})
    px["date"]=pd.to_datetime(px["date"])
    for c in ["adj_open","adj_close","turnover_value"]: px[c]=pd.to_numeric(px[c],errors="coerce")
    tp=pd.read_sql("SELECT date,open,close FROM index_daily WHERE code='0000' ORDER BY date",conn)
    tp["date"]=pd.to_datetime(tp["date"])
    return ev,px,tp

def build(ev,px,tp):
    px=px.sort_values(["code","date"]).reset_index(drop=True)
    g=px.groupby("code")
    px["adv60"]=g["turnover_value"].transform(lambda s:s.rolling(60,min_periods=40).mean())
    # 各銘柄の日付→行番号
    px["i"]=g.cumcount()
    tp=tp.set_index("date")
    tp_oc=(tp["close"]/tp["open"]-1)             # TOPIX当日 open→close
    tp_close=tp["close"]
    recs=[]
    for code,gx in px.groupby("code"):
        gx=gx.reset_index(drop=True)
        dmap={d:i for i,d in enumerate(gx["date"])}
        e=ev[ev["code"]==code]
        for _,r in e.iterrows():
            d=r["disc_date"]
            # 反応日インデックス
            if d in dmap:
                base=dmap[d]
            else:
                # disc_dateが非営業日: 次営業日を反応起点に
                fut=gx[gx["date"]>d]
                if fut.empty: continue
                base=fut.index[0]
            react = base+1 if r["after_close"] else base
            entry = react+1
            if entry>=len(gx): continue
            try:
                # 反応リターン(react: open→close), TOPIX超過
                rd=gx.loc[react,"date"]
                if rd not in tp_oc.index: continue
                car0=(gx.loc[react,"adj_close"]/gx.loc[react,"adj_open"]-1)-tp_oc.loc[rd]
                if not np.isfinite(car0): continue
                if gx.loc[entry,"adv60"]<ADV_FLOOR or pd.isna(gx.loc[entry,"adv60"]): continue
                rec={"code":code,"entry_date":gx.loc[entry,"date"],"car0":car0}
                e_open=gx.loc[entry,"adj_open"]; ed=gx.loc[entry,"date"]
                for H in HOLDS:
                    j=entry+H-1
                    if j>=len(gx): rec[f"d{H}"]=np.nan; continue
                    jd=gx.loc[j,"date"]
                    if ed not in tp_close.index or jd not in tp_close.index:
                        rec[f"d{H}"]=np.nan; continue
                    stock=gx.loc[j,"adj_close"]/e_open-1
                    tpx=tp_close.loc[jd]/tp_close.loc[ed]-1
                    rec[f"d{H}"]=(stock-tpx)*1e4   # TOPIX超過 bps
                recs.append(rec)
            except Exception: continue
    return pd.DataFrame(recs)

def assign_q(df):
    df=df.dropna(subset=["car0"]).copy()
    parts=[]
    for d,x in df.groupby("entry_date"):
        x=x.copy()
        if len(x)<N_Q*2:
            x["q"]=np.nan
        else:
            x["q"]=pd.qcut(x["car0"].rank(method="first"),N_Q,labels=False).astype(float)
        parts.append(x)
    return pd.concat(parts).dropna(subset=["q"])

def evaluate(df,label):
    out=[]
    df=assign_q(df)
    for H in HOLDS:
        col=f"d{H}"
        sub=df.dropna(subset=[col])
        wk=sub.groupby(["entry_date","q"])[col].mean().unstack("q")
        wk.columns=[int(c) for c in wk.columns]
        if wk.shape[1]<N_Q or 0 not in wk.columns or (N_Q-1) not in wk.columns: continue
        ls=(wk[N_Q-1]-wk[0]).dropna()/1e4
        if len(ls)<20: continue
        net=ls-COST_BPS/1e4
        ppt=net.mean()/net.std() if net.std()>0 else np.nan
        ann=ppt*np.sqrt(245/H) if net.std()>0 else np.nan
        out.append({"label":label,"hold":H,"n_days":len(ls),
            "q_lo_bps":round(wk[0].mean(),1),"q_hi_bps":round(wk[N_Q-1].mean(),1),
            "LS_net_bps":round(net.mean()*1e4,1),"LS_win":round((net>0).mean(),3),
            "ann_sharpe":round(ann,2)})
    return pd.DataFrame(out)

def quantile_table(df,label):
    """分位別の平均ドリフト(bps, コスト前) — 単調性を見る本質テーブル"""
    df=assign_q(df)
    rows=[]
    for q in range(N_Q):
        x=df[df["q"]==q]
        rows.append({"label":label,"q":q,"n":len(x),
            "car0_%":round(x["car0"].mean()*100,2),
            "d5_bps":round(x["d5"].mean(),1),"d10_bps":round(x["d10"].mean(),1),
            "d20_bps":round(x["d20"].mean(),1)})
    return pd.DataFrame(rows)

def main():
    conn=psycopg2.connect(**PG); print("[RUN] price-reaction PEAD")
    ev,px,tp=load(conn); conn.close()
    print(f"  events={len(ev)} px_rows={len(px)}")
    df=build(ev,px,tp)
    print(f"  PEAD obs={len(df)} codes={df['code'].nunique()} entry_days={df['entry_date'].nunique()}")
    df.to_csv(OUT/"pead_obs.csv",index=False)
    res=pd.concat([evaluate(df,"ALL"),evaluate(df[df['entry_date']<OOS_START],"IS"),
                   evaluate(df[df['entry_date']>=OOS_START],"OOS")],ignore_index=True)
    res.to_csv(OUT/"pead_summary.csv",index=False)
    print("\n===== 価格反応PEAD L/Sドリフト(コスト20bps後) =====")
    print(res.to_string(index=False))
    qt=pd.concat([quantile_table(df,"ALL"),quantile_table(df[df['entry_date']>=OOS_START],"OOS")],ignore_index=True)
    qt.to_csv(OUT/"quantile_drift.csv",index=False)
    print("\n===== 分位別ドリフト(コスト前, bps) car0=反応リターン低→高 =====")
    print(qt.to_string(index=False))

if __name__=="__main__":
    import traceback
    try: main(); print("[DONE]")
    except Exception:
        traceback.print_exc()
        with open(OUT/"error.log","w") as f: f.write(traceback.format_exc())
        raise
