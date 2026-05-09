#!/usr/bin/env python3
"""
Task 2: lme_on_copper 詳細分析
  - 月別・年別リターン内訳
  - CORE5銘柄別リターン内訳
  - LMEシグナル強度別 (1〜2% / 2〜3% / 3%+)
  - CORE5入れ替え候補: 非鉄関連銘柄 (5714/5706/5713/7013/7012) の比較
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import psycopg2
from datetime import date, timedelta

PG_CONFIG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}
COST_BPS = 2.0
BST_PERIODS = [
    (date(2025, 3, 30), date(2025, 10, 26)),
    (date(2026, 3, 29), date(2026, 10, 25)),
]
CORE5 = ["5711.T","6501.T","7011.T","5016.T","4502.T"]
# 入れ替え候補: より非鉄・素材色が強い銘柄
CANDIDATES = ["5714.T","5706.T","5713.T","7012.T","7013.T","6367.T"]

def is_bst(d):
    for s,e in BST_PERIODS:
        if s<=d<=e: return True
    return False

def load_intraday(symbols, start='2025-03-01'):
    conn = psycopg2.connect(**PG_CONFIG)
    syms = "','".join(symbols)
    df = pd.read_sql(
        f"SELECT symbol,timestamp,open,high,low,close FROM intraday_data "
        f"WHERE symbol IN ('{syms}') AND close IS NOT NULL AND timestamp>='{start}' "
        f"ORDER BY symbol,timestamp", conn)
    conn.close()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['jst'] = df['timestamp'] + pd.Timedelta(hours=9)
    df['jst_date'] = df['jst'].dt.date
    return df

def load_daily(symbols, start='2025-03-01'):
    conn = psycopg2.connect(**PG_CONFIG)
    syms = "','".join(symbols)
    df = pd.read_sql(
        f"SELECT symbol, DATE(trade_date) AS dt, open, close FROM daily_stats "
        f"WHERE symbol IN ('{syms}') AND trade_date>='{start}' ORDER BY symbol,trade_date", conn)
    conn.close()
    df['dt'] = pd.to_datetime(df['dt']).dt.date
    return df

def check_available(candidates):
    conn = psycopg2.connect(**PG_CONFIG)
    syms = "','".join(candidates)
    df = pd.read_sql(
        f"SELECT symbol, DATE(MIN(trade_date)) AS from_dt, DATE(MAX(trade_date)) AS to_dt, COUNT(*) n "
        f"FROM daily_stats WHERE symbol IN ('{syms}') GROUP BY symbol", conn)
    conn.close()
    return df

# ── LMEシグナル取得 ──
print("LME銅データ取得中...")
lme = load_intraday(['CMCU3'], start='2025-03-20')
daily_core = load_daily(CORE5)
# 候補銘柄の可用性確認
avail = check_available(CANDIDATES)
avail_syms = list(avail['symbol']) if not avail.empty else []
print(f"入れ替え候補で利用可能: {avail_syms}")
if avail_syms:
    daily_cand = load_daily(avail_syms)

# ── シグナル発動日とLME変化率を記録 ──
signal_records = []
for dt, g in lme.groupby('jst_date'):
    if isinstance(dt, str): dt = date.fromisoformat(dt)
    if dt.weekday() == 3 or dt.weekday() >= 5: continue
    bst = is_bst(dt)
    g = g.sort_values('jst')
    ob = g[g['jst'].dt.hour == (9 if bst else 10)]
    if ob.empty: continue
    tokyo_open = float(ob['close'].iloc[0])
    hhmm = g['jst'].dt.hour*60 + g['jst'].dt.minute
    b1525 = g[hhmm <= 15*60+25]
    if b1525.empty: continue
    current = float(b1525['close'].iloc[-1])
    if tokyo_open <= 0: continue
    change_pct = (current/tokyo_open-1)*100
    if change_pct < 1.0: continue
    signal_records.append({'date': dt, 'lme_chg': change_pct})

print(f"\nシグナル発動: {len(signal_records)} 件")

# ── 銘柄別 ON リターン ──
def on_ret(daily_df, sym, signal_date):
    sd = daily_df[daily_df['symbol']==sym].set_index('dt')
    if signal_date not in sd.index: return np.nan
    entry = float(sd.loc[signal_date,'close'])
    future = sd[sd.index > signal_date]
    if future.empty: return np.nan
    exit_open = float(future.iloc[0]['open'])
    if entry<=0 or exit_open<=0: return np.nan
    return (exit_open/entry-1)*10000 - COST_BPS*2

print("\n" + "="*65)
print("lme_on_copper 詳細分析")
print("="*65)

# CORE5 銘柄別
print("\n[CORE5 銘柄別リターン]")
all_trades_by_sym = {}
for sym in CORE5:
    rets = []
    for rec in signal_records:
        r = on_ret(daily_core, sym, rec['date'])
        if not np.isnan(r): rets.append(r)
    arr = np.array(rets)
    if len(arr)>0:
        wr = (arr>0).mean()*100
        sh = arr.mean()/arr.std()*np.sqrt(252) if arr.std()>0 else np.nan
        print(f"  {sym}: N={len(arr)}  WR={wr:.1f}%  平均={arr.mean():+.1f}bps  Sharpe={sh:.2f}")
    all_trades_by_sym[sym] = rets

# 月別内訳
print("\n[月別 CORE5平均リターン]")
monthly = {}
for rec in signal_records:
    dt = rec['date']
    ym = f"{dt.year}-{dt.month:02d}"
    rets = []
    for sym in CORE5:
        r = on_ret(daily_core, sym, dt)
        if not np.isnan(r): rets.append(r)
    if rets:
        if ym not in monthly: monthly[ym] = []
        monthly[ym].append(np.mean(rets))
for ym in sorted(monthly):
    arr = np.array(monthly[ym])
    print(f"  {ym}: N={len(arr)}  平均={arr.mean():+.1f}bps  {'✅' if arr.mean()>0 else '❌'}")

# LME変化率帯別
print("\n[LME変化率帯別 CORE5平均リターン]")
bands = [(1.0,2.0,'1-2%'),(2.0,3.0,'2-3%'),(3.0,99,'3%+')]
for lo,hi,label in bands:
    rets = []
    for rec in signal_records:
        if not (lo<=rec['lme_chg']<hi): continue
        r_list = [on_ret(daily_core,sym,rec['date']) for sym in CORE5]
        r_list = [x for x in r_list if not np.isnan(x)]
        if r_list: rets.append(np.mean(r_list))
    if rets:
        arr = np.array(rets)
        sh = arr.mean()/arr.std()*np.sqrt(252) if arr.std()>0 else np.nan
        print(f"  {label}: N={len(arr)}  WR={(arr>0).mean()*100:.1f}%  平均={arr.mean():+.1f}bps  Sharpe={sh:.2f}")

# 入れ替え候補銘柄比較
if avail_syms:
    print(f"\n[CORE5入れ替え候補銘柄との比較]")
    print(f"  (シグナル日: {[str(r['date']) for r in signal_records]})")
    print(f"  {'銘柄':<10} {'N':>4} {'WR%':>6} {'平均bps':>9} {'Sharpe':>7}")
    for sym in avail_syms:
        rets = []
        for rec in signal_records:
            r = on_ret(daily_cand, sym, rec['date'])
            if not np.isnan(r): rets.append(r)
        if not rets: continue
        arr = np.array(rets)
        wr = (arr>0).mean()*100
        sh = arr.mean()/arr.std()*np.sqrt(252) if arr.std()>0 else np.nan
        print(f"  {sym:<10} {len(arr):>4} {wr:>5.1f}% {arr.mean():>+9.1f} {sh:>7.2f}")
    print("\n  参考 CORE5合計平均:")
    all_avg = []
    for rec in signal_records:
        rs = [on_ret(daily_core,s,rec['date']) for s in CORE5]
        rs = [x for x in rs if not np.isnan(x)]
        if rs: all_avg.append(np.mean(rs))
    if all_avg:
        arr = np.array(all_avg)
        sh = arr.mean()/arr.std()*np.sqrt(252) if arr.std()>0 else np.nan
        print(f"  CORE5平均   {len(arr):>4} {(arr>0).mean()*100:>5.1f}% {arr.mean():>+9.1f} {sh:>7.2f}")
else:
    print("\n[入れ替え候補] DB未登録のため比較不可")
    print(f"  候補銘柄: {CANDIDATES}")
    print("  → sync_mariadb_to_postgres.py で追加取得が必要")

print("\n[結論]")
print("  lme_on_copper の劣化要因と対応方針を上記から判断")
