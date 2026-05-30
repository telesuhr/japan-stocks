"""
平均回帰(MR_rsi L25) の日次ポートフォリオ化 + ベータ中立 + レジーム分解
================================================================
前段 20260530_swing_breakout_meanrev_universe の続き。
MR_rsi L25（RSI(25)<=30 で翌日寄成Long, H日保有）が唯一 IS/OOS 両プラスだった。
これを以下で深掘りする:
  (a) 日次等加重バスケットの実ポートフォリオSharpe（同時保有・相関込み）
  (b) TOPIX に対するベータ中立化 → 純粋なαを分離
  (c) レジーム分解: 年別 / VXJ(日経VI)ボラ環境別

ポジション規則（pyramiding無し）:
  - 各銘柄で RSI(25)<=30 かつ 流動性≥10億円 かつ ストップ高でない日にシグナル
  - 翌営業日の寄りでエントリー、H営業日保有して引けで決済
  - 同一銘柄が保有中の追加シグナルは無視（in/outの2状態）
  - 各日、保有中の全銘柄を等加重。日次リターン=各銘柄のその日のリターン平均
    エントリー日は open->close、以降は前日close->close、決済日も close->close
  - コスト: 往復 COST_BPS をエントリー日とエントリー時点リターンから控除

実行:
  /root/venvs/jpstocks/bin/python run.py --smoke   # 上位200銘柄
  /root/venvs/jpstocks/bin/python run.py           # 全ユニバース
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
import argparse, os
from pathlib import Path
import numpy as np
import pandas as pd
import psycopg2

OUTDIR = Path(__file__).parent
PG = {"host": os.environ.get("PGHOST","localhost"), "port": int(os.environ.get("PGPORT",5432)),
      "user": os.environ.get("PGUSER","postgres"), "password": os.environ.get("PGPASSWORD","postgres"),
      "dbname": os.environ.get("PGDATABASE","market_data")}

RSI_L = 25
RSI_TH = 30.0
HOLD = 5                 # 前段で IS/OOS 両プラスだった保有日数
ADV_WIN, ADV_FLOOR = 60, 1e9
COST_BPS = 10.0
OOS_START = "2022-01-01"
TRADING_DAYS = 245
DATA_START, DATA_END = "2016-05-09", "2026-05-28"


def load_panel(conn, smoke=False):
    seg = "(SELECT code5 FROM symbol_master WHERE market_nm IN ('プライム','スタンダード','グロース'))"
    if smoke:
        codes = pd.read_sql(
            f"SELECT s.code FROM stocks_daily s WHERE s.date>='2024-01-01' AND s.code IN {seg} "
            "GROUP BY s.code ORDER BY AVG(s.turnover_value) DESC LIMIT 200", conn)["code"].tolist()
        flt = "AND code = ANY(%(codes)s)"; params={"codes":codes,"s":DATA_START,"e":DATA_END}
    else:
        flt = f"""AND code IN {seg} AND code IN (
            SELECT code FROM stocks_daily WHERE date>=%(s)s GROUP BY code
            HAVING MAX(turnover_value)>=%(f)s)"""
        params={"s":DATA_START,"e":DATA_END,"f":ADV_FLOOR}
    df = pd.read_sql(f"""
        SELECT code,date,adj_open,adj_high,adj_low,adj_close,turnover_value,upper_limit
        FROM stocks_daily WHERE date>=%(s)s AND date<=%(e)s {flt} ORDER BY code,date""", conn, params=params)
    df["date"]=pd.to_datetime(df["date"])
    for c in ["adj_open","adj_high","adj_low","adj_close","turnover_value"]:
        df[c]=pd.to_numeric(df[c],errors="coerce")
    return df


def build_daily_returns(df):
    """各銘柄の日次リターン列と保有マスクを作り、日次ポートフォリオ系列を返す。"""
    df = df.sort_values(["code","date"]).reset_index(drop=True)
    g = df.groupby("code")
    c = df["adj_close"]
    # RSI(25)
    up = g["adj_close"].transform(lambda s: s.diff().clip(lower=0).rolling(RSI_L,min_periods=RSI_L).mean())
    dn = g["adj_close"].transform(lambda s: (-s.diff().clip(upper=0)).rolling(RSI_L,min_periods=RSI_L).mean())
    df["rsi"] = 100 - 100/(1 + up/dn.replace(0,np.nan))
    # 流動性ゲート（前日まで）
    df["adv"] = g["turnover_value"].transform(lambda s: s.rolling(ADV_WIN,min_periods=ADV_WIN).mean().shift(1))
    df["liquid"] = df["adv"]>=ADV_FLOOR
    # 日次リターン（close-to-close）と entry日リターン（open->close）
    df["ret_cc"] = g["adj_close"].transform(lambda s: s.pct_change())
    df["ret_oc"] = df["adj_close"]/df["adj_open"] - 1.0
    # シグナル: rsi<=th & liquid & not upper_limit
    df["sig"] = (df["rsi"]<=RSI_TH) & df["liquid"] & ~df["upper_limit"].fillna(False)

    # 各銘柄ごとに、pyramiding無しの保有状態を構築
    # entry は シグナル日の翌営業日。保有は entry日(=シグナル+1)から H 日。
    # 保有中はシグナルを無視。
    parts=[]
    for code, x in df.groupby("code"):
        x = x.reset_index(drop=True)
        n=len(x)
        held = np.zeros(n, dtype=bool)
        entry_day = np.zeros(n, dtype=bool)
        exit_day = np.zeros(n, dtype=bool)
        i=0
        sig = x["sig"].values
        while i < n-1:
            if sig[i]:
                e = i+1                  # entry index (翌営業日)
                xend = min(e+HOLD-1, n-1)  # 保有最終日
                held[e:xend+1]=True
                entry_day[e]=True
                exit_day[xend]=True
                i = xend+1               # 決済まで次シグナル無視
            else:
                i+=1
        x["held"]=held; x["entry_day"]=entry_day; x["exit_day"]=exit_day
        parts.append(x)
    df = pd.concat(parts, ignore_index=True)

    # 各保有日の銘柄リターン: entry日は ret_oc、それ以外は ret_cc。コストはentry日に往復控除
    df["pos_ret"] = np.where(df["entry_day"], df["ret_oc"], df["ret_cc"])
    df.loc[df["entry_day"], "pos_ret"] -= COST_BPS/1e4
    df_held = df[df["held"]].copy()
    # 日次ポートフォリオ = 保有銘柄の等加重平均
    port = df_held.groupby("date")["pos_ret"].agg(["mean","count"]).rename(
        columns={"mean":"ret","count":"n_pos"})
    return port, df_held


def load_market(conn):
    # TOPIX は index_daily の code='0000'
    mk = pd.read_sql("SELECT date,close FROM index_daily WHERE code='0000' ORDER BY date", conn)
    mk["date"]=pd.to_datetime(mk["date"]); mk=mk.set_index("date")
    mk["mret"]=mk["close"].pct_change()
    # VXJ(日経VI)はDBに無いので、TOPIX実現ボラ(20日, 年率%)で代替
    mk["rv20"]=mk["mret"].rolling(20,min_periods=20).std()*np.sqrt(TRADING_DAYS)*100
    return mk[["mret","rv20","close"]]


def sharpe(r):
    r=r.dropna()
    if len(r)<20 or r.std()==0: return np.nan
    return r.mean()/r.std()*np.sqrt(TRADING_DAYS)


def maxdd(r):
    eq=(1+r.fillna(0)).cumprod()
    return (eq/eq.cummax()-1).min()


def stats_block(r, label):
    r=r.dropna()
    return {"label":label,"n_days":len(r),"ann_ret_pct":round(r.mean()*TRADING_DAYS*100,2),
            "ann_vol_pct":round(r.std()*np.sqrt(TRADING_DAYS)*100,2),
            "sharpe":round(sharpe(r),3),"maxdd_pct":round(maxdd(r)*100,2),
            "win_day":round((r>0).mean(),3)}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--smoke",action="store_true"); a=ap.parse_args()
    conn=psycopg2.connect(**PG)
    print("[SMOKE]" if a.smoke else "[FULL]","loading panel...")
    df=load_panel(conn, smoke=a.smoke)
    print(f"  rows={len(df)} codes={df['code'].nunique()}")
    port, held = build_daily_returns(df)
    mk=load_market(conn); conn.close()

    P=port.join(mk, how="left")
    P=P[P["n_pos"]>=5]   # 最低5銘柄保有日のみ（分散の体をなす）
    P.to_csv(OUTDIR/"daily_portfolio.csv")
    print(f"  portfolio days={len(P)}  avg positions/day={P['n_pos'].mean():.1f}")

    # (a) 生ポートフォリオ Sharpe（全/IS/OOS）
    rows=[]
    rows.append({**stats_block(P["ret"],"RAW_ALL")})
    rows.append({**stats_block(P.loc[P.index<OOS_START,"ret"],"RAW_IS")})
    rows.append({**stats_block(P.loc[P.index>=OOS_START,"ret"],"RAW_OOS")})

    # (b) ベータ中立: 日次 port_ret = α + β*mret + ε。残差(α+ε)を中立リターンとする
    reg=P.dropna(subset=["ret","mret"])
    beta=np.cov(reg["ret"],reg["mret"])[0,1]/np.var(reg["mret"])
    P["ret_neutral"]=P["ret"]-beta*P["mret"]
    print(f"  TOPIX beta={beta:.3f}")
    rows.append({**stats_block(P["ret_neutral"],f"NEUTRAL_ALL(beta={beta:.2f})")})
    rows.append({**stats_block(P.loc[P.index<OOS_START,"ret_neutral"],"NEUTRAL_IS")})
    rows.append({**stats_block(P.loc[P.index>=OOS_START,"ret_neutral"],"NEUTRAL_OOS")})
    summ=pd.DataFrame(rows)
    summ.to_csv(OUTDIR/"portfolio_summary.csv",index=False)

    # (c1) 年別 Sharpe（生 / 中立）
    yr=P.groupby(P.index.year, observed=True).apply(lambda x: pd.Series({
        "sharpe_raw":sharpe(x["ret"]),"sharpe_neutral":sharpe(x["ret_neutral"]),
        "ann_ret_pct":round(x["ret"].mean()*TRADING_DAYS*100,1),
        "avg_pos":round(x["n_pos"].mean(),0),"n_days":len(x)}))
    yr.to_csv(OUTDIR/"by_year.csv")

    # (c2) ボラ環境別: TOPIX実現ボラ(20日)の3分位（低/中/高）で生リターンSharpe
    P2=P.dropna(subset=["rv20"]).copy()
    P2["vol_bucket"]=pd.qcut(P2["rv20"],3,labels=["低ボラ","中ボラ","高ボラ"])
    vol=P2.groupby("vol_bucket",observed=True).apply(lambda x: pd.Series({
        "sharpe_raw":sharpe(x["ret"]),"sharpe_neutral":sharpe(x["ret_neutral"]),
        "ann_ret_pct":round(x["ret"].mean()*TRADING_DAYS*100,1),
        "rv20_range":f"{x['rv20'].min():.0f}-{x['rv20'].max():.0f}%","n_days":len(x)}))
    vol.to_csv(OUTDIR/"by_vol.csv")

    print("\n===== (a)(b) ポートフォリオ Sharpe =====")
    print(summ.to_string(index=False))
    print("\n===== (c1) 年別 =====")
    print(yr.round(3).to_string())
    print("\n===== (c2) VXJボラ環境別 =====")
    print(vol.round(3).to_string())


if __name__=="__main__":
    import traceback
    try: main(); print("[DONE]")
    except Exception:
        traceback.print_exc()
        with open(OUTDIR/"error.log","w") as f: f.write(traceback.format_exc())
        raise
