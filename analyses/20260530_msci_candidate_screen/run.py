"""
MSCI採用候補の事前スクリーニング検証（選択バイアス排除の前向きテスト）
================================================================
前段 20260530_msci_rebalance_closing_event の核心:
  MSCI採用銘柄は公表"前"に+16.5%(TOPIX超過)上昇するが、これは「上がったから採用された」
  選択バイアスかもしれず、そのままでは取引不能。本当のエッジにするには
  「次に採用される銘柄を市場より早く予測」できる必要がある。

着想:
  MSCIスタンダードは時価総額の大きい銘柄を採用する。よって「時価総額ランクが上昇して
  大型株ゾーンに新規参入しつつある銘柄(=rising star)」を各月末の情報だけで抽出すれば、
  実際のMSCI採用リストを知らずとも候補を先回りできるのではないか。

データ: stocks_daily(close) × fin_summary(発行済株式数-自己株式) → 時価総額
        + index_daily(TOPIX) + symbol_master(scale_cat)

時価総額 = adj無しclose × (issued - treasury)  ※各日時点で最新のfin_summary値(as-of, look-ahead無し)

検証1 ケーススタディ:
  既知のMSCI採用9銘柄が、公表の何ヶ月前に時価総額ランクが「採用閾値帯」に入ったか

検証2 前向きバスケット(選択バイアス排除):
  各月末、時価総額ランクが直近6ヶ月で上昇し「候補帯(rank 150-400)」に新規参入した銘柄を
  バスケット化 → 翌1/3/6ヶ月のTOPIX超過リターン。MSCIリストは一切使わない。
  「rising star近MSCI cutoff」が実際に先のリターンを生むかを純粋に測る。

実行:
  /root/venvs/jpstocks/bin/python run.py
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

# 候補帯(時価総額ランク) と rising 判定窓
BAND_LO, BAND_HI = 150, 450     # MSCIスタンダード採用閾値帯の概算(大型〜中型上位)
RISE_LOOKBACK = 6               # 直近Nヶ月でランク上昇
HOLD_MONTHS = [1,3,6]
ADV_FLOOR = 1e9
START = "2022-01-01"

ADDITIONS=[  # (announce, code, name) 検証1用
    ("2026-05-12","58010","古河電工"),("2026-05-12","57060","三井金属"),
    ("2025-11-06","63610","荏原"),("2025-11-06","50160","JX金属"),
    ("2025-11-06","285A0","キオクシア"),("2025-11-06","90240","西武HD"),
    ("2025-08-08","70120","川崎重工"),("2025-08-08","74530","良品計画"),
    ("2025-02-11","90230","東京メトロ"),
]

def load(conn):
    # fin_summary の略名キー: ShOutFY=期末発行済株式数, TrShFY=自己株式数（FY決算, as-of）
    print("  loading shares (ShOutFY-TrShFY from fin_summary FY)...")
    sh=pd.read_sql("""SELECT code, disc_date AS date,
        (payload->>'ShOutFY')::numeric
          - COALESCE(NULLIF(payload->>'TrShFY','')::numeric,0) AS shares
        FROM fin_summary
        WHERE cur_per_type='FY' AND NULLIF(payload->>'ShOutFY','')::numeric > 0
        ORDER BY code, disc_date""", conn)
    sh["date"]=pd.to_datetime(sh["date"])
    sh=sh[(sh["shares"]>0)&np.isfinite(sh["shares"])]
    print("  loading daily close...")
    px=pd.read_sql("""SELECT code,date,close,turnover_value FROM stocks_daily
        WHERE date>=%(s)s AND code IN (SELECT code5 FROM symbol_master WHERE market_nm IN ('プライム','スタンダード','グロース'))
        ORDER BY code,date""",conn,params={"s":START})
    px["date"]=pd.to_datetime(px["date"])
    for c in ["close","turnover_value"]: px[c]=pd.to_numeric(px[c],errors="coerce")
    tp=pd.read_sql("SELECT date,close FROM index_daily WHERE code='0000' ORDER BY date",conn)
    tp["date"]=pd.to_datetime(tp["date"]); tp=tp.rename(columns={"close":"topix"})
    return sh,px,tp

def build_monthly(sh,px,tp):
    # as-of結合: 各px日に、その日以前最新のshares
    px=px.sort_values("date"); sh=sh.sort_values("date")
    rows=[]
    for code,g in px.groupby("code"):
        s=sh[sh["code"]==code][["date","shares"]].sort_values("date")
        if s.empty: continue
        m=pd.merge_asof(g.sort_values("date"),s,on="date",direction="backward")
        rows.append(m)
    px=pd.concat(rows,ignore_index=True)
    px=px.dropna(subset=["shares"])
    px["mktcap"]=px["close"]*px["shares"]
    px["adv60"]=px.groupby("code")["turnover_value"].transform(lambda x:x.rolling(60,min_periods=40).mean())
    # 月末スナップショット
    px["ym"]=px["date"].dt.to_period("M")
    me=px.sort_values("date").groupby(["code","ym"]).tail(1).copy()
    me=me.merge(tp,on="date",how="left")
    # 月末ごとに時価総額ランク(降順=1が最大)
    me["rank"]=me.groupby("ym")["mktcap"].rank(ascending=False,method="first")
    return me

def main():
    conn=psycopg2.connect(**PG)
    sh,px,tp=load(conn); conn.close()
    me=build_monthly(sh,px,tp)
    print(f"  monthly snapshots={len(me)} codes={me['code'].nunique()} months={me['ym'].nunique()}")
    me.to_csv(OUT/"monthly_mktcap.csv",index=False)

    # ---- 検証1: 採用9銘柄のランク推移 ----
    print("\n===== 検証1: MSCI採用銘柄の公表前ランク推移 =====")
    print("  (各銘柄の公表月とその6/3/1ヶ月前の時価総額ランク)")
    cs=[]
    for ann,code,name in ADDITIONS:
        a=pd.Period(pd.Timestamp(ann),"M")
        sub=me[me["code"]==code].set_index("ym")["rank"]
        def r(off):
            p=a-off
            return int(sub[p]) if p in sub.index else None
        row={"name":name,"code":code,"announce":ann,
             "rank_-6m":r(6),"rank_-3m":r(3),"rank_-1m":r(1),"rank_pub":r(0)}
        cs.append(row)
    csdf=pd.DataFrame(cs); print(csdf.to_string(index=False))
    csdf.to_csv(OUT/"case_rank_path.csv",index=False)
    inband=csdf["rank_-3m"].apply(lambda x: x is not None and BAND_LO<=x<=BAND_HI*1.5).mean()
    print(f"\n  公表3ヶ月前に候補帯({BAND_LO}-{int(BAND_HI*1.5)}位)にいた割合: {inband:.0%}")

    # ---- 検証2: 前向きバスケット(選択バイアス排除) ----
    print("\n===== 検証2: rising-starバスケットの前向きリターン(MSCIリスト不使用) =====")
    me=me.sort_values(["code","ym"])
    me["rank_prev"]=me.groupby("code")["rank"].shift(RISE_LOOKBACK)  # 6ヶ月前ランク
    # forward total return (月末close→Nヶ月後月末close), TOPIX超過
    for h in HOLD_MONTHS:
        me[f"fwd{h}"]=me.groupby("code")["close"].shift(-h)/me["close"]-1
        me[f"tfwd{h}"]=me.groupby("code")["topix"].shift(-h)/me["topix"]-1
        me[f"ex{h}"]=(me[f"fwd{h}"]-me[f"tfwd{h}"])*100
    # 候補: 流動性十分 & 現ランクが帯内 & ランクが直近6Mで上昇(rank数値が減少) & 6M前は帯外(下)
    cand=me[(me["adv60"]>=ADV_FLOOR)&(me["rank"]>=BAND_LO)&(me["rank"]<=BAND_HI)&
            (me["rank_prev"].notna())&(me["rank"]<me["rank_prev"]-20)&(me["rank_prev"]>BAND_HI)].copy()
    print(f"  候補シグナル総数={len(cand)} ユニーク銘柄={cand['code'].nunique()} 月数={cand['ym'].nunique()}")
    rows=[]
    # ベンチ: 同じ帯にいる全銘柄(rising条件なし)の平均超過
    base=me[(me["adv60"]>=ADV_FLOOR)&(me["rank"]>=BAND_LO)&(me["rank"]<=BAND_HI)]
    for h in HOLD_MONTHS:
        c=cand[f"ex{h}"].dropna(); b=base[f"ex{h}"].dropna()
        rows.append({"hold_m":h,"cand_n":len(c),"cand_mean_ex%":round(c.mean(),2),
            "cand_win":round((c>0).mean(),3),"cand_median%":round(c.median(),2),
            "band_mean_ex%":round(b.mean(),2),"diff%":round(c.mean()-b.mean(),2)})
    res=pd.DataFrame(rows); res.to_csv(OUT/"forward_basket.csv",index=False)
    print(res.to_string(index=False))

if __name__=="__main__":
    import traceback
    try: main(); print("[DONE]")
    except Exception:
        traceback.print_exc()
        with open(OUT/"error.log","w") as f: f.write(traceback.format_exc())
        raise
