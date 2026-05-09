#!/usr/bin/env python3
"""
Task 5: lme_on_copper 代替シグナル — LME累積 LB=10 +3% → CORE5 ON Long
候補戦略 (analyses/README.md 記載):
  LME銅の過去10営業日 (LB=10) 累積リターン ≥ +3% の翌日にCORE5 ON Long

既存のlme_on_copperとの重複日も確認する
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import psycopg2
from datetime import date, timedelta

PG_CONFIG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}
COST_BPS = 2.0
CORE5 = ["5711.T","6501.T","7011.T","5016.T","4502.T"]

def sharpe(arr):
    arr = np.array([x for x in arr if not np.isnan(x) and not np.isinf(x)])
    if len(arr)<2 or arr.std()==0: return np.nan
    return arr.mean()/arr.std()*np.sqrt(252)

conn = psycopg2.connect(**PG_CONFIG)
# LME日足が必要: CMCU3のdaily_statsを確認
lme_d = pd.read_sql(
    "SELECT DATE(trade_date) AS dt, close FROM daily_stats "
    "WHERE symbol='CMCU3' ORDER BY trade_date", conn)
# CORE5日足
core_d = pd.read_sql(
    "SELECT symbol, DATE(trade_date) AS dt, open, close FROM daily_stats "
    "WHERE symbol IN ('5711.T','6501.T','7011.T','5016.T','4502.T') ORDER BY symbol,trade_date", conn)
conn.close()

lme_d['dt'] = pd.to_datetime(lme_d['dt']).dt.date
core_d['dt'] = pd.to_datetime(core_d['dt']).dt.date

if lme_d.empty:
    # LME日足がない場合はイントラデイから日次終値を生成
    print("CMCU3 daily_stats なし → イントラデイから代替生成")
    conn2 = psycopg2.connect(**PG_CONFIG)
    lme_intra = pd.read_sql(
        "SELECT DATE(timestamp + INTERVAL '9 hours') AS dt, "
        "MAX(close) FILTER (WHERE EXTRACT(hour FROM timestamp+'9h') BETWEEN 14 AND 16) AS close_proxy "
        "FROM intraday_data WHERE symbol='CMCU3' AND close IS NOT NULL "
        "GROUP BY DATE(timestamp + INTERVAL '9 hours') ORDER BY dt", conn2)
    conn2.close()
    lme_d = lme_intra.rename(columns={'close_proxy':'close'})
    lme_d['dt'] = pd.to_datetime(lme_d['dt']).dt.date
    lme_d = lme_d.dropna(subset=['close'])
else:
    print(f"CMCU3 daily_stats: {len(lme_d)} 件")

lme_s = lme_d.set_index('dt').sort_index()
lme_s['ret10'] = lme_s['close'].pct_change(10) * 100  # 10日累積

# CORE5日足
def on_ret_core5(core_df, signal_date):
    rets = []
    for sym in CORE5:
        sd = core_df[core_df['symbol']==sym].set_index('dt')
        if signal_date not in sd.index: continue
        entry = float(sd.loc[signal_date,'close'])
        future = sd[sd.index>signal_date]
        if future.empty: continue
        exit_open = float(future.iloc[0]['open'])
        if entry<=0 or exit_open<=0: continue
        rets.append((exit_open/entry-1)*10000 - COST_BPS*2)
    return np.mean(rets) if rets else np.nan

print("\n" + "="*65)
print("LME累積シグナル (LB=10) バックテスト")
print("="*65)

# LB=10で閾値を変えてスキャン
for thresh in [2.0, 3.0, 4.0, 5.0]:
    trades = []; sig_dates = []
    for dt, row in lme_s.iterrows():
        if isinstance(dt, str): dt = date.fromisoformat(dt)
        ret10 = row['ret10']
        if np.isnan(ret10) or ret10 < thresh: continue
        if dt.weekday() == 3 or dt.weekday() >= 5: continue
        # 翌営業日にON Long
        future_dates = [d for d in lme_s.index if d > dt]
        if not future_dates: continue
        next_dt = future_dates[0]
        r = on_ret_core5(core_d, next_dt)
        if not np.isnan(r):
            trades.append(r); sig_dates.append(next_dt)
    if not trades: continue
    arr = np.array(trades)
    sh = sharpe(arr)
    print(f"\n  閾値 +{thresh:.0f}%: N={len(arr)}  WR={(arr>0).mean()*100:.1f}%  "
          f"平均={arr.mean():+.1f}bps  Sharpe={sh:.2f}")
    # 月別
    for yr in sorted(set(d.year for d in sig_dates)):
        yt = [t for t,d in zip(trades,sig_dates) if d.year==yr]
        if not yt: continue
        yt_arr = np.array(yt)
        print(f"    {yr}: N={len(yt)}  平均={yt_arr.mean():+.1f}bps  "
              f"Sharpe={sharpe(yt_arr):.2f}")

# 既存 lme_on_copper との重複確認 (1日変化率≥1% vs 累積≥3%)
print("\n[lme_on_copper (単日≥1%) との発動日重複確認]")
# シグナル日を再現 (簡易版: CMCU3の日次リターン≥1%を使用)
lme_s['daily_ret'] = lme_s['close'].pct_change()*100
single_sig = set(dt for dt, row in lme_s.iterrows()
                 if not np.isnan(row.get('daily_ret',np.nan)) and row.get('daily_ret',0)>=1.0
                 and isinstance(dt, date) and dt.weekday()!=3 and dt.weekday()<5)
cum_sig_3 = set(dt for dt, row in lme_s.iterrows()
                if not np.isnan(row.get('ret10',np.nan)) and row.get('ret10',0)>=3.0
                and isinstance(dt, date) and dt.weekday()!=3 and dt.weekday()<5)
overlap = single_sig & cum_sig_3
print(f"  単日シグナル発動: {len(single_sig)} 件")
print(f"  累積LB10≥3% 発動: {len(cum_sig_3)} 件")
print(f"  重複: {len(overlap)} 件")
print(f"  重複率: {len(overlap)/max(len(single_sig),1)*100:.1f}%")

print("\n[README記載 Sharpe: +7.55 (N=37)]")
