#!/usr/bin/env python3
"""
Task 1: sox_overnight_short 再評価
.TOPX日足 (NAS MariaDB 2015〜) でShortリターンを計算する

シグナル: .SOX 前日 ret ≤ -2.0% AND ESc1 ret ≤ -1.0% AND VIX [15,35] AND 火曜除外
エントリー: 当日 .TOPX open でShort
決済:       当日 .TOPX close で買戻し
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import pymysql
import psycopg2
from datetime import date

MARIA = dict(host='100.92.181.92', port=3306, user='rfnews',
             password='Bleach@924', database='refinitiv_news', connect_timeout=10)
PG_CONFIG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}
COST_BPS = 2.0

def load_maria(sql):
    conn = pymysql.connect(**MARIA)
    df = pd.read_sql(sql, conn)
    conn.close()
    return df

def sharpe(arr):
    arr = np.array([x for x in arr if not np.isnan(x) and not np.isinf(x)], dtype=float)
    if len(arr) < 2 or arr.std() == 0: return np.nan
    return arr.mean() / arr.std() * np.sqrt(252)

# ── データ取得 ──
print("データ取得中...")
sox = load_maria("SELECT trade_date, close FROM daily_data WHERE symbol='.SOX' ORDER BY trade_date")
es  = load_maria("SELECT trade_date, close FROM daily_data WHERE symbol='ESc1'  ORDER BY trade_date")
vix = load_maria("SELECT trade_date, close FROM daily_data WHERE symbol='VXc1'  ORDER BY trade_date")
topx= load_maria("SELECT trade_date, open, close FROM daily_data WHERE symbol='.TOPX' ORDER BY trade_date")

for df in [sox, es, vix, topx]:
    df['trade_date'] = pd.to_datetime(df['trade_date']).dt.date

sox['ret'] = sox['close'].pct_change() * 100
es['ret']  = es['close'].pct_change() * 100

sox_s  = sox.set_index('trade_date')
es_s   = es.set_index('trade_date')
vix_s  = vix.set_index('trade_date')
topx_s = topx.set_index('trade_date')

# ── シグナル生成 & バックテスト ──
trades_strong = []   # SOX≤-2% AND ES≤-1%
trades_normal = []   # SOX≤-2% only (ES条件なし)
signal_dates_s = []
signal_dates_n = []

all_dates = sorted(set(sox_s.index) & set(topx_s.index))

for i, dt in enumerate(all_dates):
    sox_ret = float(sox_s.loc[dt, 'ret']) if dt in sox_s.index else np.nan
    if np.isnan(sox_ret) or sox_ret > -2.0:
        continue
    # 翌営業日 (シグナル日の翌日)
    future = [d for d in all_dates[i+1:] if d > dt]
    if not future:
        continue
    entry_dt = future[0]
    if isinstance(entry_dt, str):
        entry_dt = date.fromisoformat(entry_dt)
    # 火曜除外
    if entry_dt.weekday() == 1:
        continue
    # VIX フィルタ
    if dt in vix_s.index:
        vix_val = float(vix_s.loc[dt, 'close'])
        if vix_val < 15 or vix_val >= 35:
            continue
    # TOPX (1306.T代理) Short
    if entry_dt not in topx_s.index:
        continue
    op = float(topx_s.loc[entry_dt, 'open'])
    cl = float(topx_s.loc[entry_dt, 'close'])
    if op <= 0 or cl <= 0 or np.isnan(op) or np.isnan(cl):
        continue
    ret_bps = -(cl / op - 1) * 10000 - COST_BPS * 2

    # Normalトレード (SOX≤-2%のみ)
    trades_normal.append(ret_bps)
    signal_dates_n.append(entry_dt)

    # Strongトレード (AND ESc1≤-1%)
    if dt in es_s.index:
        es_ret = float(es_s.loc[dt, 'ret'])
        if es_ret <= -1.0:
            trades_strong.append(ret_bps)
            signal_dates_s.append(entry_dt)

# ── 結果 ──
print("\n" + "="*65)
print("sox_overnight_short 再評価 (.TOPX日足Short代理, 2015〜2026)")
print("="*65)

def show(label, trades, dates):
    arr = np.array([x for x in trades if not np.isnan(x)], dtype=float)
    n = len(arr)
    if n == 0:
        print(f"\n[{label}] N=0"); return
    wr  = (arr > 0).mean() * 100
    pos = arr[arr>0].sum(); neg = abs(arr[arr<=0].sum())
    pf  = pos/neg if neg>0 and pos>0 else np.nan
    sh  = sharpe(arr)
    print(f"\n[{label}]")
    print(f"  N={n}  WR={wr:.1f}%  PF={pf:.2f}  Sharpe={sh:.2f}")
    print(f"  平均={arr.mean():+.1f}bps  合計={arr.sum():+.0f}bps")
    # 年別
    s = pd.Series(arr, index=pd.to_datetime(dates))
    print("  年別:")
    for yr, grp in s.groupby(s.index.year):
        sh_yr = sharpe(grp.values)
        print(f"    {yr}: N={len(grp)}  平均={grp.mean():+.1f}bps  Sharpe={sh_yr:.2f}")

show("Normal (SOX≤-2%, VIX15-35, 火曜除外)", trades_normal, signal_dates_n)
show("Strong (AND ESc1≤-1%)", trades_strong, signal_dates_s)

# 最近2年 (2024〜2026)
recent_dates = [d for d in signal_dates_s if d.year >= 2024]
recent_trades = [t for t, d in zip(trades_strong, signal_dates_s) if d.year >= 2024]
if recent_trades:
    show("Strong 2024〜2026 (最新期間)", recent_trades, recent_dates)

print("\n[README記載 Sharpe: +2.11 (強シグナル +2.83)]")
