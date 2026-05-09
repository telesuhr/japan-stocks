#!/usr/bin/env python3
"""
Task 4: lasertec_ma25_support 重複除外 (クールダウン) 再バックテスト
同一局面での連続シグナルを除外: シグナル後 HOLD日間は新規エントリー禁止
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import psycopg2
from datetime import date, timedelta

PG_CONFIG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}
COST_BPS = 2.0; HOLD = 10; STOP_PCT = 10.0

def sharpe(arr):
    arr = np.array([x for x in arr if not np.isnan(x) and not np.isinf(x)])
    if len(arr)<2 or arr.std()==0: return np.nan
    return arr.mean()/arr.std()*np.sqrt(252)

conn = psycopg2.connect(**PG_CONFIG)
df = pd.read_sql(
    "SELECT symbol, DATE(trade_date) AS dt, open, high, low, close "
    "FROM daily_stats WHERE symbol='6920.T' ORDER BY trade_date", conn)
conn.close()
df['dt'] = pd.to_datetime(df['dt']).dt.date
df = df.set_index('dt').sort_index()
df = df.astype({c:float for c in ['open','high','low','close']})

df['ma25'] = df['close'].rolling(25).mean()
df['ma25_5d_ago'] = df['ma25'].shift(5)
df['hh20'] = df['close'].rolling(20).max()
df['dd20_pct'] = (df['close']/df['hh20']-1)*100
lo,hi,ma = df['low'],df['high'],df['ma25']
df['touched'] = (lo<=ma*1.01)&(hi>=ma*0.99)
df['downtrend'] = df['dd20_pct']<=-5.0
df['slope_up'] = df['ma25']>df['ma25_5d_ago']
df['signal'] = df['touched']&df['downtrend']&df['slope_up']&df['ma25'].notna()

def run_backtest(cooldown_days, label):
    trades = []; sig_dates = []; last_entry = None
    for idx in range(len(df)):
        if not df.iloc[idx]['signal']: continue
        sig_dt = df.index[idx]
        # クールダウン: 前回エントリーから cooldown_days 以内ならスキップ
        if cooldown_days>0 and last_entry is not None:
            if (sig_dt - last_entry).days < cooldown_days: continue
        if idx+1>=len(df): continue
        entry = float(df.iloc[idx+1]['open'])
        if entry<=0 or np.isnan(entry): continue
        stop_level = entry*(1-STOP_PCT/100)
        last_entry = df.index[idx+1]
        sig_dates.append(sig_dt)
        fut = df.iloc[idx+1:idx+2+HOLD]
        stop_hit=False; exit_px=None
        for _,r in fut.iloc[1:].iterrows():
            if r['low']<=stop_level:
                exit_px=stop_level; stop_hit=True; break
        if exit_px is None:
            exit_px=float(fut.iloc[HOLD]['close']) if len(fut)>HOLD else float(fut.iloc[-1]['close'])
        ret_bps = (exit_px/entry-1)*10000 - COST_BPS*2
        trades.append(ret_bps)

    arr = np.array(trades)
    n = len(arr)
    if n==0:
        print(f"\n[{label}] N=0"); return
    wr = (arr>0).mean()*100
    pos=arr[arr>0].sum(); neg=abs(arr[arr<=0].sum())
    pf = pos/neg if neg>0 and pos>0 else np.nan
    sh = sharpe(arr)
    print(f"\n[{label}]")
    print(f"  N={n}  WR={wr:.1f}%  PF={pf:.2f}  Sharpe={sh:.2f}")
    print(f"  平均={arr.mean():+.1f}bps  合計={arr.sum():+.0f}bps")
    print(f"  シグナル日: {[str(d) for d in sig_dates[:5]]}{'...' if len(sig_dates)>5 else ''}")
    return arr, sig_dates

print("="*65)
print("lasertec_ma25_support クールダウン感度分析")
print("="*65)

run_backtest(0,  "クールダウンなし (元のロジック)")
run_backtest(10, "クールダウン 10日 (HOLDと同じ)")
run_backtest(15, "クールダウン 15日")
run_backtest(20, "クールダウン 20日")

# 2025〜2026 年別
print("\n[年別内訳 (クールダウン10日)]")
trades_10=[]; dates_10=[]; last10=None
for idx in range(len(df)):
    if not df.iloc[idx]['signal']: continue
    sig_dt = df.index[idx]
    if last10 is not None and (sig_dt-last10).days<10: continue
    if idx+1>=len(df): continue
    entry=float(df.iloc[idx+1]['open'])
    if entry<=0 or np.isnan(entry): continue
    stop_level=entry*0.90; last10=df.index[idx+1]
    dates_10.append(sig_dt)
    fut=df.iloc[idx+1:idx+2+HOLD]
    exit_px=None
    for _,r in fut.iloc[1:].iterrows():
        if r['low']<=stop_level: exit_px=stop_level; break
    if exit_px is None:
        exit_px=float(fut.iloc[HOLD]['close']) if len(fut)>HOLD else float(fut.iloc[-1]['close'])
    trades_10.append((exit_px/entry-1)*10000-COST_BPS*2)

for yr in sorted(set(d.year for d in dates_10)):
    yt=[t for t,d in zip(trades_10,dates_10) if d.year==yr]
    if not yt: continue
    arr=np.array(yt)
    sh=sharpe(arr)
    print(f"  {yr}: N={len(arr)}  WR={(arr>0).mean()*100:.1f}%  "
          f"平均={arr.mean():+.1f}bps  Sharpe={sh:.2f}")

print("\n[README記載 Sharpe: +7.68 (H2 +12.96)]")
