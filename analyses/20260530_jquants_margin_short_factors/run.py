"""
JQuants信用残データのクロスセクション・ファクター検証
================================================================
データ: public.jquants_margin_interest（週次・信用残・全銘柄, 2021-01〜2026-05）
        + stocks_daily（フォワードリターン・流動性）, PostgreSQL@Omen

前段の教訓（生Sharpeはほぼ市場ベータ、中立化で消える）を踏まえ、最初から
**ロングショート分位スプレッド = 構造的にベータ中立**で検証する。

カラム（実スキーマ）:
  shrt_vol      : 信用売り残（株数）
  long_vol      : 信用買い残（株数）
  shrt_neg_vol  : 信用売り（一般）  long_neg_vol: 信用買い（一般）
  shrt_std_vol  : 信用売り（制度）  long_std_vol: 信用買い（制度）

検証ファクター（週次, 各週で流動性≥10億円の銘柄をクロスセクションでソート）:
  F1 margin_ratio  : long_vol / shrt_vol（信用倍率）。高=買い残過多→将来の戻り売り圧力(弱気)
  F2 long_chg      : long_vol 前週比（信用買い残の増加→上値重い予兆 or 強気?）
  F3 short_chg     : shrt_vol 前週比（信用売り残の増加→踏み上げ余地 or 弱気?）
  F4 short_ratio   : shrt_vol / (long_vol+shrt_vol)（売り残比率）
  F5 net_chg       : (long_chg - short_chg) 需給バランスの変化

注: 空売り残報告(short_sale_report)は0.5%超義務の疎データで全銘柄網羅せず→本検証では除外。

評価:
  - 各週、ファクターで10分位 → 翌週(5営業日)リターン
  - Long-Short(Q10−Q1)スプレッド週次系列の年率Sharpe（構造的ベータ中立）
  - 符号は両方向の含意を見る（高分位Longが正/負どちらでもLS_meanの符号で判断）
  - IS(2021-2023)/OOS(2024-2026)分割、コスト週40bps(両サイド往復)控除
"""
import sys; sys.stdout.reconfigure(line_buffering=True)
import os
from pathlib import Path
import numpy as np
import pandas as pd
import psycopg2

OUT=Path(__file__).parent
PG=dict(host=os.environ.get("PGHOST","localhost"),port=int(os.environ.get("PGPORT",5432)),
        user=os.environ.get("PGUSER","postgres"),password=os.environ.get("PGPASSWORD","postgres"),
        dbname=os.environ.get("PGDATABASE","market_data"))

ADV_FLOOR=1e9
N_Q=10
FWD_DAYS=5
LS_COST_BPS=40.0
OOS_START="2024-01-01"
WEEKS_PER_YEAR=52
START,END="2021-01-01","2026-05-29"
FACTORS=["margin_ratio","long_chg","short_chg","short_ratio","net_chg"]


def load(conn):
    print("  loading margin...")
    mg=pd.read_sql("SELECT code,date,shrt_vol,long_vol FROM jquants_margin_interest WHERE date>=%(s)s ORDER BY code,date",
                   conn,params={"s":START})
    mg["date"]=pd.to_datetime(mg["date"])
    for c in ["shrt_vol","long_vol"]: mg[c]=pd.to_numeric(mg[c],errors="coerce")
    print("  loading daily...")
    px=pd.read_sql("""SELECT code,date,adj_close,turnover_value FROM stocks_daily
        WHERE date>=%(s)s AND date<=%(e)s
          AND code IN (SELECT code5 FROM symbol_master WHERE market_nm IN ('プライム','スタンダード','グロース'))
        ORDER BY code,date""",conn,params={"s":START,"e":END})
    px["date"]=pd.to_datetime(px["date"])
    for c in ["adj_close","turnover_value"]: px[c]=pd.to_numeric(px[c],errors="coerce")
    return mg,px


def build(mg,px):
    px=px.sort_values(["code","date"]).reset_index(drop=True)
    g=px.groupby("code")
    px["adv60"]=g["turnover_value"].transform(lambda s:s.rolling(60,min_periods=40).mean())
    px["fwd_ret"]=g["adj_close"].transform(lambda s:s.shift(-FWD_DAYS))/px["adj_close"]-1.0

    mg=mg.sort_values(["code","date"]).reset_index(drop=True)
    gm=mg.groupby("code")
    mg["margin_ratio"]=mg["long_vol"]/mg["shrt_vol"].replace(0,np.nan)
    mg["short_ratio"]=mg["shrt_vol"]/(mg["long_vol"]+mg["shrt_vol"]).replace(0,np.nan)
    mg["long_chg"]=gm["long_vol"].transform(lambda s:s.pct_change())
    mg["short_chg"]=gm["shrt_vol"].transform(lambda s:s.pct_change())
    mg["net_chg"]=mg["long_chg"]-mg["short_chg"]

    pxi=px.dropna(subset=["adj_close"]).sort_values("date")
    rows=[]
    for code,m in mg.groupby("code"):
        p=pxi[pxi["code"]==code][["date","adv60","fwd_ret"]].sort_values("date")
        if p.empty: continue
        mm=pd.merge_asof(m.sort_values("date"),p,on="date",direction="backward",tolerance=pd.Timedelta("7D"))
        rows.append(mm)
    df=pd.concat(rows,ignore_index=True)
    df=df[df["adv60"]>=ADV_FLOOR].dropna(subset=["fwd_ret"])
    return df


def evaluate(df,label):
    out=[]
    for f in FACTORS:
        sub=df.dropna(subset=[f]).copy()
        sub=sub[np.isfinite(sub[f])]
        def qw(x):
            if x[f].nunique()<N_Q: return pd.Series(np.nan,index=x.index)
            try: return pd.qcut(x[f].rank(method="first"),N_Q,labels=False)
            except Exception: return pd.Series(np.nan,index=x.index)
        sub["q"]=sub.groupby("date",group_keys=False).apply(qw)
        sub=sub.dropna(subset=["q"])
        if sub.empty: continue
        wk=sub.groupby(["date","q"])["fwd_ret"].mean().unstack("q")
        if wk.shape[1]<N_Q: continue
        ls=(wk[N_Q-1]-wk[0]).dropna()-LS_COST_BPS/1e4
        if len(ls)<20: continue
        shp=ls.mean()/ls.std()*np.sqrt(WEEKS_PER_YEAR) if ls.std()>0 else np.nan
        out.append({"label":label,"factor":f,"n_weeks":len(ls),
            "q_lo_bps":round(wk[0].mean()*1e4,1),"q_hi_bps":round(wk[N_Q-1].mean()*1e4,1),
            "LS_mean_bps":round(ls.mean()*1e4,1),"LS_sharpe_ann":round(shp,2),
            "LS_win":round((ls>0).mean(),3)})
    return pd.DataFrame(out)


def main():
    conn=psycopg2.connect(**PG); print("[RUN] JQuants margin factor study")
    mg,px=load(conn); conn.close()
    print(f"  margin={len(mg)} px={len(px)}")
    df=build(mg,px)
    print(f"  factor rows={len(df)} codes={df['code'].nunique()} weeks={df['date'].nunique()}")
    df.to_csv(OUT/"factor_panel.csv",index=False)
    res=pd.concat([evaluate(df,"ALL"),evaluate(df[df["date"]<OOS_START],"IS"),
                   evaluate(df[df["date"]>=OOS_START],"OOS")],ignore_index=True)
    res.to_csv(OUT/"factor_summary.csv",index=False)
    print("\n===== 信用残ファクター LS分位スプレッド（週次, コスト40bps後, bps & 年率Sharpe） =====")
    with pd.option_context("display.width",200,"display.max_rows",None):
        print(res.sort_values(["label","LS_sharpe_ann"],ascending=[True,False]).to_string(index=False))


if __name__=="__main__":
    import traceback
    try: main(); print("[DONE]")
    except Exception:
        traceback.print_exc()
        with open(OUT/"error.log","w") as f: f.write(traceback.format_exc())
        raise
