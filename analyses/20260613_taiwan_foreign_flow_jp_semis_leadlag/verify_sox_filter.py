"""別セッションの本命発見の独立再現:
   SOX急落(オーバーナイト≤-1%)の翌日本リバウンドを、台湾外国人フローz の符号で条件分け。
   仮説: SOX下げ × 台湾は買い(z>0) = ダイバージェンス → 日本のリバウンド強い。
   ※ 設定は完全一致でなく方向性の収束を見る独立チェック。"""
import os, sys
sys.stdout.reconfigure(line_buffering=True)
import numpy as np, pandas as pd, psycopg2
PG = dict(host="localhost", port=5432, user="postgres", password="postgres", dbname="market_data")
conn = psycopg2.connect(**PG)
JP=["80350","68570","61460","69200","77350","77290","67230","69630","40630","34360","40620"]
START="2022-10-01"
jp=pd.read_sql("SELECT code,date,adj_open,adj_close FROM stocks_daily WHERE code=ANY(%s) AND date>=%s",conn,params=[JP,START])
top=pd.read_sql("SELECT date,open,close FROM index_daily WHERE code='0000' AND date>=%s",conn,params=[START])
tw=pd.read_sql("SELECT trade_date,foreign_net FROM macro.tw_foreign_flow WHERE code='2330' AND trade_date>=%s",conn,params=[START])
sox=pd.read_sql("SELECT trade_date,close FROM macro.daily_ohlcv WHERE symbol='.SOX' AND trade_date>=%s",conn,params=[START])
conn.close()
for d in (jp,top,tw,sox):
    c="date" if "date" in d.columns else "trade_date"; d[c]=pd.to_datetime(d[c])

# 各銘柄: D寄→D引(o2c), D寄→D+1引(2day) を作り等加重平均
jp=jp.sort_values(["code","date"]); jp[["adj_open","adj_close"]]=jp[["adj_open","adj_close"]].astype(float)
jp["oc"]=jp["adj_close"]/jp["adj_open"]-1
jp["o2"]=jp.groupby("code")["adj_close"].shift(-1)/jp["adj_open"]-1
b=jp.groupby("date")[["oc","o2"]].mean()
top=top.set_index("date").sort_index().astype(float)
top["t_oc"]=top["close"]/top["open"]-1
top["t_o2"]=top["close"].shift(-1)/top["open"]-1
P=pd.concat([b,top[["t_oc","t_o2"]]],axis=1).dropna(subset=["oc"]).reset_index().rename(columns={"index":"date"})
P["exc_oc"]=P["oc"]-P["t_oc"]; P["exc_o2"]=P["o2"]-P["t_o2"]

# シグナル(< D): TSMC z(60d), SOXオーバーナイト
tw=tw.sort_values("trade_date")
tw["z"]=(tw["foreign_net"]-tw["foreign_net"].rolling(60,min_periods=30).mean())/tw["foreign_net"].rolling(60,min_periods=30).std()
sox=sox.sort_values("trade_date"); sox["sox_ret"]=sox["close"].astype(float).pct_change()
P=P.sort_values("date")
P=pd.merge_asof(P,tw[["trade_date","z"]].dropna(),left_on="date",right_on="trade_date",direction="backward",allow_exact_matches=False)
P=pd.merge_asof(P.sort_values("date"),sox[["trade_date","sox_ret"]].dropna(),left_on="date",right_on="trade_date",direction="backward",allow_exact_matches=False,suffixes=("","_s"))
P=P.dropna(subset=["z","sox_ret"])

def rep(df,col,lab):
    print(f"  {lab:30} N={len(df):4d}  平均={df[col].mean()*100:+.2f}%  勝率={(df[col]>0).mean()*100:4.1f}%")

for thr in (-0.01,-0.015):
    sell=P[P["sox_ret"]<=thr]
    print(f"\n=== SOXオーバーナイト ≤ {thr*100:.1f}% の翌日本(寄エントリ) ===")
    print("[D寄→D+1引 2day・絶対リターン]")
    rep(sell,"o2","SOX売り 全体")
    rep(sell[sell.z<0],"o2","  +台湾も売り z<0")
    rep(sell[sell.z>0],"o2","  +台湾は買い z>0 ★")
    print("[D寄→D引 1day・TOPIX中立超過]")
    rep(sell,"exc_oc","SOX売り 全体")
    rep(sell[sell.z<0],"exc_oc","  +台湾も売り z<0")
    rep(sell[sell.z>0],"exc_oc","  +台湾は買い z>0 ★")
