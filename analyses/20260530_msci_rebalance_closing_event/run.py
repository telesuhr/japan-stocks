"""
MSCIリバランス「引けイベント」の検出と翌日リターン検証
================================================================
データ: public.stocks_intraday（1分足, 2024/05〜2026/05）+ stocks_daily（翌日リターン）

着想:
  2026-05-29 のMSCIリバランスで、イビデン/キオクシアが大引け(15:30 クロージング
  オークション)で巨大出来高を伴い跳ねた。パッシブファンドが発効日の引けで強制売買
  するため、「引け出来高が当日総出来高に占める比率(close_pct)」が異常スパイクする。
  実測: イビデン 5/29 = 27.7% vs 通常日 1.5-2.5%。

  MSCI構成銘柄データがDBに無いため、この **close_pct 異常スパイク = リバランス代理
  シグナル** として全銘柄・全期間からイベントを検出する。

検証:
  Q1. close_pct 異常日は MSCIリバランス発効日（2/5/8/11月末営業日）に集中するか？
      → 集中すれば「引けスパイク検出 = リバランス検出」が妥当と裏付け
  Q2. イベント日の引けでの値動き(pre_close 15:24 → close 15:30)の符号と大きさ
  Q3. 翌営業日リターン（リバランス需給の反転 or 継続）。これが取引可能なエッジか
      - 寄り→引け、寄り→翌々日 など

イベント定義:
  各(code,date)で close_pct = vol(15:30) / vol(当日全体)
  通常比の倍率 close_x = close_pct / 各銘柄のclose_pct中央値(過去60営業日)
  「イベント」= close_pct >= 8% AND close_x >= 4（流動性≥10億円の銘柄のみ）

実行:
  /root/venvs/jpstocks/bin/python run.py --build   # SQL集計してclose_events.csv生成(重い,数分)
  /root/venvs/jpstocks/bin/python run.py           # 分析（既存CSV使用）
"""
import sys; sys.stdout.reconfigure(line_buffering=True)
import os, argparse
from pathlib import Path
import numpy as np, pandas as pd
import psycopg2

OUT=Path(__file__).parent
PG=dict(host=os.environ.get("PGHOST","localhost"),port=int(os.environ.get("PGPORT",5432)),
        user=os.environ.get("PGUSER","postgres"),password=os.environ.get("PGPASSWORD","postgres"),
        dbname=os.environ.get("PGDATABASE","market_data"))

CLOSE_PCT_MIN=8.0   # 引け出来高比率の下限(%)
CLOSE_X_MIN=4.0     # 過去中央値に対する倍率
ADV_FLOOR=1e9

AGG_SQL = """
WITH base AS (
  SELECT code, ts::date AS dt, ts::time AS tm, close, volume
  FROM stocks_intraday
),
agg AS (
  SELECT code, dt,
    SUM(volume) AS total_vol,
    SUM(volume) FILTER (WHERE tm='15:30:00') AS close_vol,
    MAX(close)  FILTER (WHERE tm='15:30:00') AS close_px,
    MAX(close)  FILTER (WHERE tm='15:24:00') AS pre_px,
    MAX(close)  FILTER (WHERE tm='09:00:00') AS open_px
  FROM base GROUP BY code, dt
)
SELECT code, dt, total_vol, close_vol, close_px, pre_px, open_px,
  ROUND(100.0*close_vol/NULLIF(total_vol,0),3) AS close_pct
FROM agg
WHERE close_vol IS NOT NULL AND total_vol > 0
ORDER BY code, dt
"""

def build(conn):
    print("  集計SQL実行中(数分)...")
    df = pd.read_sql(AGG_SQL, conn)
    df.to_csv(OUT/"close_events_raw.csv", index=False)
    print(f"  saved close_events_raw.csv rows={len(df)}")
    return df

def load_daily(conn):
    # 翌日リターン用の日足（adj）+ 流動性
    px = pd.read_sql("""SELECT code,date,adj_open,adj_close,turnover_value FROM stocks_daily
        WHERE date>='2024-04-01' ORDER BY code,date""", conn)
    px["date"]=pd.to_datetime(px["date"])
    for c in ["adj_open","adj_close","turnover_value"]: px[c]=pd.to_numeric(px[c],errors="coerce")
    return px

def is_month_end_window(d):
    """その月の最終営業日±2日窓か（MSCIリバランス発効=月末営業日近辺）"""
    # 月末営業日近辺かは日付のday>=24 かつ 2/5/8/11月で近似（厳密なカレンダーは別途）
    return d.day >= 24

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--build",action="store_true"); a=ap.parse_args()
    conn=psycopg2.connect(**PG)
    raw_path=OUT/"close_events_raw.csv"
    if a.build or not raw_path.exists():
        df=build(conn)
    else:
        df=pd.read_csv(raw_path)
    df["dt"]=pd.to_datetime(df["dt"])
    px=load_daily(conn); conn.close()

    # 流動性フィルタ（60日平均売買代金）
    px=px.sort_values(["code","date"])
    px["adv60"]=px.groupby("code")["turnover_value"].transform(lambda s:s.rolling(60,min_periods=40).mean())
    px["fwd_open"]=px.groupby("code")["adj_open"].shift(-1)   # 翌日寄り
    px["fwd_close"]=px.groupby("code")["adj_close"].shift(-1) # 翌日引け
    px["fwd2_close"]=px.groupby("code")["adj_close"].shift(-2)
    liq=px[["code","date","adv60","adj_close","fwd_open","fwd_close","fwd2_close"]].rename(columns={"date":"dt"})

    df=df.merge(liq,on=["code","dt"],how="left")
    df=df[df["adv60"]>=ADV_FLOOR].copy().reset_index(drop=True)

    # close_pct の過去中央値(銘柄ごと60日)
    df=df.sort_values(["code","dt"])
    df["cp_med"]=df.groupby("code")["close_pct"].transform(lambda s:s.rolling(60,min_periods=20).median().shift(1))
    df["close_x"]=df["close_pct"]/df["cp_med"]

    # イベント検出
    ev=df[(df["close_pct"]>=CLOSE_PCT_MIN)&(df["close_x"]>=CLOSE_X_MIN)].copy().reset_index(drop=True)
    # 引けスパイク時の値動き
    ev["close_jump_bps"]=(ev["close_px"]/ev["pre_px"]-1)*1e4              # 15:24→15:30
    ev["fwd_oc_bps"]=(ev["fwd_close"]/ev["fwd_open"]-1)*1e4              # 翌日寄→引
    ev["fwd_co_bps"]=(ev["fwd_open"]/ev["close_px"]-1)*1e4              # 当日引→翌日寄(ギャップ)
    ev["fwd_cc_bps"]=(ev["fwd_close"]/ev["close_px"]-1)*1e4            # 当日引→翌日引
    ev.to_csv(OUT/"events.csv",index=False)
    print(f"\n検出イベント数={len(ev)}  銘柄={ev['code'].nunique()}  日数={ev['dt'].nunique()}")

    # Q1: イベント日の月内分布（月末集中度）
    ev["month"]=ev["dt"].dt.to_period("M").astype(str)
    ev["dom"]=ev["dt"].dt.day
    ev["is_msci_month"]=ev["dt"].dt.month.isin([2,5,8,11])
    ev["is_monthend"]=ev["dom"]>=24
    print("\n=== Q1: 月末集中度 ===")
    print(f"  月末窓(dom>=24)に入るイベント: {ev['is_monthend'].mean():.1%}")
    print(f"  MSCI該当月(2/5/8/11)のイベント: {ev['is_msci_month'].mean():.1%}")
    print(f"  MSCI該当月×月末窓: {(ev['is_msci_month']&ev['is_monthend']).mean():.1%}")
    print("\n  イベント件数 上位日:")
    top=ev.groupby("dt").size().sort_values(ascending=False).head(12)
    for d,n in top.items(): print(f"    {d.date()}  {n}件  (dom={d.day}, month={d.month})")

    # Q2/Q3: リターン統計（全イベント vs 月末MSCI窓イベント）
    def block(sub,label):
        v=sub.dropna(subset=["fwd_cc_bps"])
        if len(v)<5: return None
        return {"label":label,"n":len(v),
            "close_jump_bps":round(sub["close_jump_bps"].mean(),0),
            "翌寄ギャップ_bps":round(v["fwd_co_bps"].mean(),0),
            "翌日寄引_bps":round(v["fwd_oc_bps"].mean(),0),
            "翌日引引_bps":round(v["fwd_cc_bps"].mean(),0),
            "翌寄引勝率":round((v["fwd_oc_bps"]>0).mean(),3)}
    rows=[block(ev,"全イベント"),
          block(ev[ev["is_msci_month"]&ev["is_monthend"]],"MSCI月×月末窓"),
          block(ev[~(ev["is_msci_month"]&ev["is_monthend"])],"その他")]
    rows=[r for r in rows if r]
    res=pd.DataFrame(rows); res.to_csv(OUT/"return_summary.csv",index=False)
    print("\n=== Q2/Q3: 引けジャンプと翌日リターン(bps) ===")
    print(res.to_string(index=False))

    # 引けジャンプ方向別の翌日反転（買われた→翌日下げる？）
    print("\n=== 引けジャンプ方向別 翌日寄→引リターン ===")
    msci=ev[ev["is_msci_month"]&ev["is_monthend"]].dropna(subset=["fwd_oc_bps"])
    up=msci[msci["close_jump_bps"]>0]; dn=msci[msci["close_jump_bps"]<0]
    print(f"  引け上昇イベント(n={len(up)}): 翌日寄引 平均 {up['fwd_oc_bps'].mean():.0f}bps 勝率{(up['fwd_oc_bps']>0).mean():.1%}")
    print(f"  引け下落イベント(n={len(dn)}): 翌日寄引 平均 {dn['fwd_oc_bps'].mean():.0f}bps 勝率{(dn['fwd_oc_bps']>0).mean():.1%}")

if __name__=="__main__":
    import traceback
    try: main(); print("[DONE]")
    except Exception:
        traceback.print_exc()
        with open(OUT/"error.log","w") as f: f.write(traceback.format_exc())
        raise
