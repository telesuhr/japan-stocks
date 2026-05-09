#!/usr/bin/env python3
"""
Task 3: eneos_vwap_trend Long/Short 分割再評価
Long (VWAP上方乖離 → 買い順張り) と Short (下方乖離 → 売り順張り) を個別評価
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import psycopg2
from datetime import date

PG_CONFIG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}
COST_BPS = 2.0

def sharpe(arr):
    arr = np.array([x for x in arr if not np.isnan(x) and not np.isinf(x)])
    if len(arr)<2 or arr.std()==0: return np.nan
    return arr.mean()/arr.std()*np.sqrt(252)

conn = psycopg2.connect(**PG_CONFIG)
df = pd.read_sql(
    "SELECT symbol,timestamp,open,high,low,close,volume FROM intraday_data "
    "WHERE symbol='5020.T' AND close IS NOT NULL AND timestamp>='2025-01-24' "
    "ORDER BY timestamp", conn)
conn.close()
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['jst'] = df['timestamp'] + pd.Timedelta(hours=9)
df['jst_date'] = df['jst'].dt.date

long_trades, short_trades = [], []
long_dates, short_dates = [], []
monthly_long, monthly_short = {}, {}

for dt, g in df.groupby('jst_date'):
    if isinstance(dt, str): dt = date.fromisoformat(dt)
    if dt.weekday() >= 5: continue
    g = g.sort_values('jst').copy()
    morning = g[g['jst'].dt.hour >= 9]
    if len(morning) < 5: continue

    vol = morning['volume'].fillna(0).clip(lower=1)
    cum_pv = (morning['close'] * vol).cumsum()
    cum_vol = vol.cumsum()
    vwap_s = cum_pv / cum_vol

    bar_930 = morning[(morning['jst'].dt.hour==9)&(morning['jst'].dt.minute==30)]
    if bar_930.empty: continue

    close_930 = float(bar_930['close'].iloc[-1])
    vwap_930 = float(vwap_s.loc[bar_930.index[-1]])
    if vwap_930<=0: continue
    dev_bps = (close_930/vwap_930-1)*10000
    if abs(dev_bps)<50: continue

    direction = 1 if dev_bps>=50 else -1
    hhmm = g['jst'].dt.hour*60 + g['jst'].dt.minute
    after = g[hhmm > 9*60+30]
    exit_bars = g[hhmm >= 15*60+20]
    if after.empty or exit_bars.empty: continue

    entry_px = float(after['close'].iloc[0])
    exit_px = float(exit_bars['close'].iloc[-1])
    if entry_px<=0 or exit_px<=0: continue

    ret_bps = direction*(exit_px/entry_px-1)*10000 - COST_BPS*2
    ym = f"{dt.year}-{dt.month:02d}"

    if direction==1:
        long_trades.append(ret_bps); long_dates.append(dt)
        if ym not in monthly_long: monthly_long[ym]=[]
        monthly_long[ym].append(ret_bps)
    else:
        short_trades.append(ret_bps); short_dates.append(dt)
        if ym not in monthly_short: monthly_short[ym]=[]
        monthly_short[ym].append(ret_bps)

print("="*65)
print("eneos_vwap_trend Long/Short 分割評価 (2025-01〜2026-05)")
print("="*65)

def show(label, trades, dates):
    arr = np.array(trades)
    n = len(arr)
    if n==0: print(f"\n[{label}] N=0"); return
    wr = (arr>0).mean()*100
    pos=arr[arr>0].sum(); neg=abs(arr[arr<=0].sum())
    pf = pos/neg if neg>0 and pos>0 else np.nan
    sh = sharpe(arr)
    print(f"\n[{label}]  N={n}  WR={wr:.1f}%  PF={pf:.2f}  Sharpe={sh:.2f}")
    print(f"  平均={arr.mean():+.1f}bps  合計={arr.sum():+.0f}bps  Std={arr.std():.1f}bps")

show("Long (dev≥+50bps → 買い)", long_trades, long_dates)
show("Short (dev≤-50bps → 売り)", short_trades, short_dates)

# 両方合計
all_trades = long_trades + short_trades
show("合計 (Long+Short)", all_trades, long_dates+short_dates)

# 月別内訳
print("\n[月別 Long/Short 件数・平均リターン]")
all_yms = sorted(set(list(monthly_long.keys())+list(monthly_short.keys())))
print(f"  {'月':>8} {'Long N':>7} {'Long avg':>9} {'Sh N':>7} {'Sh avg':>9}")
for ym in all_yms:
    lret = monthly_long.get(ym,[])
    sret = monthly_short.get(ym,[])
    la = f"{np.mean(lret):+.0f}" if lret else "  -"
    sa = f"{np.mean(sret):+.0f}" if sret else "  -"
    print(f"  {ym:>8} {len(lret):>7} {la:>9} {len(sret):>7} {sa:>9}")

# 2026年のみ
print("\n[2026年限定]")
l26 = [r for r,d in zip(long_trades,long_dates) if d.year==2026]
s26 = [r for r,d in zip(short_trades,short_dates) if d.year==2026]
show("Long 2026", l26, [])
show("Short 2026", s26, [])

print("\n[閾値感度 50/75/100bps]")
conn2 = psycopg2.connect(**PG_CONFIG)
df2 = pd.read_sql(
    "SELECT symbol,timestamp,open,high,low,close,volume FROM intraday_data "
    "WHERE symbol='5020.T' AND close IS NOT NULL AND timestamp>='2025-01-24' "
    "ORDER BY timestamp", conn2)
conn2.close()
df2['jst'] = pd.to_datetime(df2['timestamp']) + pd.Timedelta(hours=9)
df2['jst_date'] = df2['jst'].dt.date

for thresh in [50,75,100]:
    ts = []
    for dt, g in df2.groupby('jst_date'):
        if isinstance(dt, str): dt = date.fromisoformat(dt)
        if dt.weekday()>=5: continue
        g = g.sort_values('jst').copy()
        morning = g[g['jst'].dt.hour>=9]
        if len(morning)<5: continue
        vol = morning['volume'].fillna(0).clip(lower=1)
        vwap_s = (morning['close']*vol).cumsum() / vol.cumsum()
        bar = morning[(morning['jst'].dt.hour==9)&(morning['jst'].dt.minute==30)]
        if bar.empty: continue
        c = float(bar['close'].iloc[-1]); v = float(vwap_s.loc[bar.index[-1]])
        if v<=0: continue
        dev = (c/v-1)*10000
        if abs(dev)<thresh: continue
        d = 1 if dev>0 else -1
        hhmm = g['jst'].dt.hour*60+g['jst'].dt.minute
        after = g[hhmm>9*60+30]; ex = g[hhmm>=15*60+20]
        if after.empty or ex.empty: continue
        ep = float(after['close'].iloc[0]); xp = float(ex['close'].iloc[-1])
        if ep<=0 or xp<=0: continue
        ts.append(d*(xp/ep-1)*10000-COST_BPS*2)
    arr = np.array(ts)
    if len(arr)<2: continue
    sh = sharpe(arr)
    print(f"  閾値±{thresh}bps: N={len(arr)}  WR={(arr>0).mean()*100:.1f}%  "
          f"平均={arr.mean():+.1f}bps  Sharpe={sh:.2f}")
