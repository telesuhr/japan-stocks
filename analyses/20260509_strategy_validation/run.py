#!/usr/bin/env python3
"""
全採用戦略 継続性バックテスト (2026-05-09 検証)

対象:
  1. lme_on_copper         (LME銅 東京時間 ≥+1% → CORE5 ON Long)
  2. topix_overnight       (TOPIX ギャップ ≥+0.3% → CORE5 ON Long)
  3. eneos_vwap_trend      (ENEOS 9:30 VWAP乖離 ≥±50bps → イントラ)
  4. vwap_morning_meanrevert (TEL/ディスコ/レーザー 10-11:30 VWAP 乖離≥275bps)
  5. orb_breakout_long     (ディスコ60分/三井金属30分 OR High ブレイク)
  6. lasertec_ma25_support (6920 MA25接触 dd20≤-5% スイング)
  7. sox_overnight_short   (MariaDB依存→別途検証)

コスト: ON戦略=2bps×往復, イントラ=2bps×往復
"""
import warnings
warnings.filterwarnings('ignore')

from datetime import date, datetime, timedelta
import numpy as np
import pandas as pd
import psycopg2

PG_CONFIG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}

BST_PERIODS = [
    (date(2024, 3, 31), date(2024, 10, 27)),
    (date(2025, 3, 30), date(2025, 10, 26)),
    (date(2026, 3, 29), date(2026, 10, 25)),
]
CORE5 = ["5711.T", "6501.T", "7011.T", "5016.T", "4502.T"]
COST_BPS = 2.0

def is_bst(d: date) -> bool:
    for s, e in BST_PERIODS:
        if s <= d <= e:
            return True
    return False

def load_intraday(symbols, start_date=None):
    conn = psycopg2.connect(**PG_CONFIG)
    syms = "','".join(symbols)
    where = f"WHERE symbol IN ('{syms}') AND close IS NOT NULL"
    if start_date:
        where += f" AND timestamp >= '{start_date}'"
    df = pd.read_sql(
        f"SELECT symbol, timestamp, open, high, low, close, volume "
        f"FROM intraday_data {where} ORDER BY symbol, timestamp",
        conn)
    conn.close()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['jst'] = df['timestamp'] + pd.Timedelta(hours=9)
    df['jst_date'] = df['jst'].dt.date
    return df

def load_daily(symbols, start_date=None):
    """daily_stats: open=寄付, close=引け (JST日付)"""
    conn = psycopg2.connect(**PG_CONFIG)
    syms = "','".join(symbols)
    where = f"WHERE symbol IN ('{syms}')"
    if start_date:
        where += f" AND trade_date >= '{start_date}'"
    df = pd.read_sql(
        f"SELECT symbol, DATE(trade_date) AS dt, open, high, low, close "
        f"FROM daily_stats {where} ORDER BY symbol, trade_date",
        conn)
    conn.close()
    df['dt'] = pd.to_datetime(df['dt']).dt.date
    return df

def sharpe(arr):
    arr = np.array([x for x in arr if not np.isnan(x) and not np.isinf(x)], dtype=float)
    if len(arr) < 2 or arr.std() == 0:
        return np.nan
    return arr.mean() / arr.std() * np.sqrt(252)

def summarize(name, trades):
    arr = np.array([x for x in trades if not np.isnan(x) and not np.isinf(x)], dtype=float)
    n = len(arr)
    if n == 0:
        print(f"\n{'='*60}")
        print(f"[{name}] N=0 (シグナル発動なし)")
        return dict(name=name, N=0)
    wr = (arr > 0).mean() * 100
    pos_sum = arr[arr > 0].sum()
    neg_sum = abs(arr[arr <= 0].sum())
    pf = (pos_sum / neg_sum) if neg_sum > 0 and pos_sum > 0 else np.nan
    sh = sharpe(arr)
    print(f"\n{'='*60}")
    print(f"[{name}]")
    print(f"  N={n}  WR={wr:.1f}%  PF={pf:.2f}  Sharpe={sh:.2f}")
    print(f"  平均={arr.mean():+.1f}bps  合計={arr.sum():+.0f}bps  Std={arr.std():.1f}bps")
    return dict(name=name, N=n, WR=wr, PF=pf, Sharpe=sh, mean_bps=arr.mean(), sum_bps=arr.sum())

def on_return_core5(daily_df, signal_date):
    """
    signal_date (引成エントリー日) のCORE5平均ONリターン (bps, コスト控除後)。
    エントリー: signal_date の close
    決済:        signal_date の翌営業日の open
    """
    rets = []
    for sym in CORE5:
        sd = daily_df[daily_df['symbol'] == sym].copy()
        sd = sd.set_index('dt')
        if signal_date not in sd.index:
            continue
        entry = sd.loc[signal_date, 'close']
        # 翌営業日を探す
        future = sd[sd.index > signal_date]
        if future.empty:
            continue
        exit_open = future.iloc[0]['open']
        if entry <= 0 or np.isnan(entry) or exit_open <= 0 or np.isnan(exit_open):
            continue
        rets.append((exit_open / entry - 1) * 10000 - COST_BPS * 2)
    return np.mean(rets) if rets else np.nan


# ─────────────────────────────────────────────────────────────
# 1. lme_on_copper
# ─────────────────────────────────────────────────────────────
def backtest_lme_on_copper():
    print("\n" + "="*60)
    print("1. lme_on_copper バックテスト")

    df_lme = load_intraday(['CMCU3'], start_date='2025-03-20')
    daily = load_daily(CORE5, start_date='2025-03-01')

    trades = []
    signal_dates = []

    for dt, g in df_lme.groupby('jst_date'):
        if isinstance(dt, str):
            dt = date.fromisoformat(dt)
        if dt.weekday() == 3 or dt.weekday() >= 5:
            continue

        bst = is_bst(dt)
        open_hour = 9 if bst else 10
        g = g.sort_values('jst')

        open_bars = g[g['jst'].dt.hour == open_hour]
        if open_bars.empty:
            continue
        tokyo_open = float(open_bars['close'].iloc[0])
        if np.isnan(tokyo_open) or tokyo_open <= 0:
            continue

        hhmm = g['jst'].dt.hour * 60 + g['jst'].dt.minute
        bars_1525 = g[hhmm <= 15 * 60 + 25]
        if bars_1525.empty:
            continue
        current = float(bars_1525['close'].iloc[-1])
        if np.isnan(current) or current <= 0:
            continue

        change_pct = (current / tokyo_open - 1) * 100
        if change_pct < 1.0:
            continue

        ret = on_return_core5(daily, dt)
        if not np.isnan(ret):
            trades.append(ret)
            signal_dates.append(dt)

    print(f"  シグナル発動日: {signal_dates}")
    return summarize("lme_on_copper", trades), trades, signal_dates


# ─────────────────────────────────────────────────────────────
# 2. topix_overnight
# ─────────────────────────────────────────────────────────────
def backtest_topix_overnight():
    print("\n" + "="*60)
    print("2. topix_overnight バックテスト")

    df_topix = load_intraday(['.TOPX'], start_date='2025-04-17')
    df_topix = df_topix[df_topix['close'].notna()].copy()
    daily_topix = load_daily(['.TOPX'], start_date='2025-04-01')
    daily_core = load_daily(CORE5, start_date='2025-04-01')

    trades = []
    signal_dates = []
    all_dates = sorted(df_topix['jst_date'].unique())

    # 前日終値を日足から取得
    topix_daily = daily_topix[daily_topix['symbol'] == '.TOPX'].set_index('dt')

    for dt in all_dates:
        if isinstance(dt, str):
            dt = date.fromisoformat(dt)
        if dt.weekday() == 3 or dt.weekday() >= 5:
            continue

        # 前日終値 (日足)
        prev = topix_daily[topix_daily.index < dt]
        if prev.empty:
            continue
        prev_close = float(prev.iloc[-1]['close'])
        if np.isnan(prev_close) or prev_close <= 0:
            continue

        # 当日9:00近辺 (イントラ)
        today_bars = df_topix[df_topix['jst_date'] == dt].sort_values('jst')
        morning = today_bars[today_bars['jst'].dt.hour <= 9]
        if morning.empty:
            continue
        today_open = float(morning['close'].iloc[0])
        if np.isnan(today_open) or today_open <= 0:
            continue

        gap_pct = (today_open / prev_close - 1) * 100
        if gap_pct < 0.3:
            continue

        ret = on_return_core5(daily_core, dt)
        if not np.isnan(ret):
            trades.append(ret)
            signal_dates.append(dt)

    return summarize("topix_overnight", trades), trades, signal_dates


# ─────────────────────────────────────────────────────────────
# 3. eneos_vwap_trend
# ─────────────────────────────────────────────────────────────
def backtest_eneos_vwap():
    print("\n" + "="*60)
    print("3. eneos_vwap_trend バックテスト")

    df = load_intraday(['5020.T'], start_date='2025-01-24')
    trades = []
    long_rets = []
    short_rets = []

    for dt, g in df.groupby('jst_date'):
        if isinstance(dt, str):
            dt = date.fromisoformat(dt)
        if dt.weekday() >= 5:
            continue

        g = g.sort_values('jst').copy()
        morning = g[g['jst'].dt.hour >= 9]
        if len(morning) < 5:
            continue

        vol = morning['volume'].fillna(0).clip(lower=1)
        cum_pv = (morning['close'] * vol).cumsum()
        cum_vol = vol.cumsum()
        vwap_s = cum_pv / cum_vol

        bar_930 = morning[(morning['jst'].dt.hour == 9) & (morning['jst'].dt.minute == 30)]
        if bar_930.empty:
            continue

        close_930 = float(bar_930['close'].iloc[-1])
        vwap_930 = float(vwap_s.loc[bar_930.index[-1]])
        if vwap_930 <= 0 or np.isnan(vwap_930):
            continue
        dev_bps = (close_930 / vwap_930 - 1) * 10000

        if abs(dev_bps) < 50:
            continue
        direction = 1 if dev_bps >= 50 else -1

        hhmm = g['jst'].dt.hour * 60 + g['jst'].dt.minute
        after_930 = g[hhmm > 9 * 60 + 30]
        exit_bars = g[hhmm >= 15 * 60 + 20]

        if after_930.empty or exit_bars.empty:
            continue

        entry_px = float(after_930['close'].iloc[0])
        exit_px = float(exit_bars['close'].iloc[-1])
        if entry_px <= 0 or exit_px <= 0:
            continue

        ret_bps = direction * (exit_px / entry_px - 1) * 10000 - COST_BPS * 2
        trades.append(ret_bps)
        if direction == 1:
            long_rets.append(ret_bps)
        else:
            short_rets.append(ret_bps)

    print(f"  Long: N={len(long_rets)}  Short: N={len(short_rets)}")
    return summarize("eneos_vwap_trend", trades), trades


# ─────────────────────────────────────────────────────────────
# 4. vwap_morning_meanrevert
# ─────────────────────────────────────────────────────────────
def backtest_vwap_meanrevert():
    print("\n" + "="*60)
    print("4. vwap_morning_meanrevert バックテスト")

    TARGETS = {'8035.T': 'TEL', '6146.T': 'ディスコ', '6920.T': 'レーザー'}
    THRESH = 275.0
    STOP_BPS = 400.0

    df_all = load_intraday(list(TARGETS.keys()), start_date='2024-11-14')
    trades = []
    per_sym = {s: [] for s in TARGETS}

    for sym in TARGETS:
        df = df_all[df_all['symbol'] == sym].copy()

        for dt, g in df.groupby('jst_date'):
            if isinstance(dt, str):
                dt = date.fromisoformat(dt)
            if dt.weekday() >= 5:
                continue

            g = g.sort_values('jst').copy()
            h = g['jst'].dt.hour; m = g['jst'].dt.minute
            session = g[((h == 9) | (h == 10) | ((h == 11) & (m <= 30)) |
                         ((h == 12) & (m >= 30)) | (h == 13) | (h == 14) |
                         ((h == 15) & (m <= 30)))]
            if len(session) < 10:
                continue

            vol = session['volume'].fillna(0).clip(lower=1)
            pv = (session['close'] * vol).cumsum()
            cv = vol.cumsum()
            session = session.copy()
            session['vwap'] = (pv / cv).values
            session['dev_bps'] = (session['close'] / session['vwap'] - 1) * 10000
            session['hhmm'] = session['jst'].dt.hour * 60 + session['jst'].dt.minute

            window = session[(session['hhmm'] >= 10 * 60) & (session['hhmm'] <= 11 * 60 + 30)]
            trigger = window[window['dev_bps'].abs() >= THRESH]
            if trigger.empty:
                continue

            sig_row = trigger.iloc[0]
            dev = float(sig_row['dev_bps'])
            direction = -1 if dev > 0 else 1
            entry_px = float(sig_row['close'])
            if entry_px <= 0 or np.isnan(entry_px):
                continue

            stop_price = entry_px * (1 + STOP_BPS/10000) if direction == -1 else entry_px * (1 - STOP_BPS/10000)
            after = session[session['jst'] >= sig_row['jst']]

            exit_px = None
            for _, bar in after.iterrows():
                if direction == -1 and bar['high'] >= stop_price:
                    exit_px = stop_price; break
                if direction == 1 and bar['low'] <= stop_price:
                    exit_px = stop_price; break

            if exit_px is None:
                ex = session[session['hhmm'] >= 15 * 60 + 20]
                exit_px = float(ex['close'].iloc[-1]) if not ex.empty else float(session['close'].iloc[-1])

            ret_bps = direction * (exit_px / entry_px - 1) * 10000 - COST_BPS * 2
            trades.append(ret_bps)
            per_sym[sym].append(ret_bps)

    for sym, name in TARGETS.items():
        arr = np.array(per_sym[sym])
        n = len(arr)
        if n > 0:
            wr = (arr > 0).mean() * 100
            sh = sharpe(arr)
            print(f"  {sym} {name}: N={n}  WR={wr:.1f}%  Sharpe={sh:.2f}  平均={arr.mean():+.1f}bps")

    return summarize("vwap_morning_meanrevert", trades), trades


# ─────────────────────────────────────────────────────────────
# 5. orb_breakout_long
# ─────────────────────────────────────────────────────────────
def backtest_orb():
    print("\n" + "="*60)
    print("5. orb_breakout_long バックテスト")

    TARGETS = [('5706.T', '三井金属', 30), ('6146.T', 'ディスコ', 60)]
    VWAP_THRESH = 275.0

    syms = [s for s, _, _ in TARGETS]
    df_all = load_intraday(syms, start_date='2024-11-14')
    trades = []
    per_sym = {}

    for sym, name, or_min in TARGETS:
        df = df_all[df_all['symbol'] == sym].copy()
        per_sym[sym] = []

        for dt, g in df.groupby('jst_date'):
            if isinstance(dt, str):
                dt = date.fromisoformat(dt)
            if dt.weekday() >= 5:
                continue

            g = g.sort_values('jst').copy()
            hhmm = g['jst'].dt.hour * 60 + g['jst'].dt.minute

            or_bars = g[hhmm < 9 * 60 + or_min]
            if len(or_bars) < max(3, or_min // 2):
                continue

            or_high = or_bars['high'].max()
            or_low = or_bars['low'].min()

            # ディスコ: vwap発動日はスキップ
            if sym == '6146.T':
                session = g[g['jst'].dt.hour >= 9]
                if not session.empty:
                    vol = session['volume'].fillna(0).clip(lower=1)
                    pv = (session['close'] * vol).cumsum()
                    cv = vol.cumsum()
                    vwap = pv / cv
                    dev = (session['close'] / vwap - 1) * 10000
                    wh = hhmm[(g['jst'].dt.hour >= 9)].values  # 同じインデックス
                    win_mask = (hhmm >= 10 * 60) & (hhmm <= 11 * 60 + 30)
                    if (dev[win_mask].abs() >= VWAP_THRESH).any():
                        continue

            # OR ブレイク
            post = g[hhmm >= 9 * 60 + or_min]
            entry_px = None
            for _, bar in post.iterrows():
                if bar['high'] > or_high:
                    entry_px = or_high
                    entry_time = bar['jst']
                    break

            if entry_px is None or entry_px <= 0:
                continue

            after = post[post['jst'] >= entry_time]
            exit_px = None
            for _, bar in after.iterrows():
                if bar['low'] <= or_low:
                    exit_px = or_low; break

            if exit_px is None:
                ex_hhmm = after['jst'].dt.hour * 60 + after['jst'].dt.minute
                ex = after[ex_hhmm >= 15 * 60 + 20]
                exit_px = float(ex['close'].iloc[0]) if not ex.empty else float(after['close'].iloc[-1])

            ret_bps = (exit_px / entry_px - 1) * 10000 - COST_BPS * 2
            trades.append(ret_bps)
            per_sym[sym].append(ret_bps)

    for sym, name, _ in TARGETS:
        arr = np.array(per_sym[sym])
        n = len(arr)
        if n > 0:
            wr = (arr > 0).mean() * 100
            sh = sharpe(arr)
            print(f"  {sym} {name}: N={n}  WR={wr:.1f}%  Sharpe={sh:.2f}  平均={arr.mean():+.1f}bps")

    return summarize("orb_breakout_long", trades), trades


# ─────────────────────────────────────────────────────────────
# 6. lasertec_ma25_support
# ─────────────────────────────────────────────────────────────
def backtest_lasertec_ma25():
    print("\n" + "="*60)
    print("6. lasertec_ma25_support バックテスト")

    daily = load_daily(['6920.T'])
    df = daily[daily['symbol'] == '6920.T'].copy()
    df = df.set_index('dt').sort_index()
    df = df.astype({c: float for c in ['open', 'high', 'low', 'close']})

    df['ma25'] = df['close'].rolling(25).mean()
    df['ma25_5d_ago'] = df['ma25'].shift(5)
    df['hh20'] = df['close'].rolling(20).max()
    df['dd20_pct'] = (df['close'] / df['hh20'] - 1) * 100

    lo, hi, ma = df['low'], df['high'], df['ma25']
    df['touched'] = (lo <= ma * 1.01) & (hi >= ma * 0.99)
    df['downtrend'] = df['dd20_pct'] <= -5.0
    df['slope_up'] = df['ma25'] > df['ma25_5d_ago']
    df['signal'] = df['touched'] & df['downtrend'] & df['slope_up'] & df['ma25'].notna()

    HOLD = 10
    trades = []
    sig_dates = []

    for idx in range(len(df)):
        if not df.iloc[idx]['signal']:
            continue
        if idx + 1 >= len(df):
            continue
        entry = float(df.iloc[idx + 1]['open'])
        if entry <= 0 or np.isnan(entry):
            continue
        stop_level = entry * 0.90
        sig_dates.append(df.index[idx])

        fut = df.iloc[idx + 1: idx + 2 + HOLD]
        stop_hit = False
        exit_px = None
        for _, r in fut.iloc[1:].iterrows():
            if r['low'] <= stop_level:
                exit_px = stop_level; stop_hit = True; break
        if exit_px is None and len(fut) > HOLD:
            exit_px = float(fut.iloc[HOLD]['close'])
        elif exit_px is None:
            exit_px = float(fut.iloc[-1]['close'])

        ret_bps = (exit_px / entry - 1) * 10000 - COST_BPS * 2
        trades.append(ret_bps)

    print(f"  シグナル日: {[str(d) for d in sig_dates]}")
    return summarize("lasertec_ma25_support", trades), trades


# ─────────────────────────────────────────────────────────────
# sox_overnight_short — MariaDB検証
# ─────────────────────────────────────────────────────────────
def check_sox_overnight():
    print("\n" + "="*60)
    print("7. sox_overnight_short バックテスト")
    print("   (NAS MariaDB daily_data の .SOX / ESc1 / VXc1 が必要)")
    try:
        import pymysql
        maria = dict(host='100.92.181.92', port=3306, user='rfnews',
                     password='Bleach@924', database='refinitiv_news', connect_timeout=5)
        conn = pymysql.connect(**maria)
        df = pd.read_sql(
            "SELECT symbol, MAX(trade_date) AS latest, COUNT(*) AS n "
            "FROM daily_data WHERE symbol IN ('.SOX','ESc1','VXc1','1306.T') "
            "GROUP BY symbol ORDER BY symbol",
            conn)
        conn.close()
        print("  MariaDB接続: ✅")
        print(df.to_string(index=False))

        # バックテスト: .SOX≤-2% (木曜除外 & VIX[15,35]) → 1306.Tをイントラ用1分足で代替確認
        # 1306.TはローカルDBにないため、リターン推定はTOPIX日足から代用
        conn2 = pymysql.connect(**maria)
        sox_df = pd.read_sql(
            "SELECT trade_date, close FROM daily_data WHERE symbol='.SOX' ORDER BY trade_date",
            conn2)
        es_df = pd.read_sql(
            "SELECT trade_date, close FROM daily_data WHERE symbol='ESc1' ORDER BY trade_date",
            conn2)
        vix_df = pd.read_sql(
            "SELECT trade_date, close FROM daily_data WHERE symbol='VXc1' ORDER BY trade_date",
            conn2)
        conn2.close()

        for d in [sox_df, es_df, vix_df]:
            d['trade_date'] = pd.to_datetime(d['trade_date']).dt.date

        sox_df['ret'] = sox_df['close'].pct_change() * 100
        es_df['ret'] = es_df['close'].pct_change() * 100

        sox_s = sox_df.set_index('trade_date')
        es_s = es_df.set_index('trade_date')
        vix_s = vix_df.set_index('trade_date')

        # 1306.T日足はないのでTOPIX日足を代用
        topix_d = load_daily(['.TOPX'])
        topix_s = topix_d[topix_d['symbol'] == '.TOPX'].set_index('dt')

        signal_dates = []
        for dt, row in sox_s.iterrows():
            sox_ret = row['ret']
            if pd.isna(sox_ret) or sox_ret > -2.0:
                continue
            if isinstance(dt, str):
                dt = date.fromisoformat(dt)
            if dt.weekday() == 1:  # 火曜除外
                continue
            # VIX [15, 35]
            if dt in vix_s.index:
                vix_val = float(vix_s.loc[dt, 'close'])
                if vix_val < 15 or vix_val >= 35:
                    continue
            # ESc1 AND条件
            if dt in es_s.index:
                es_ret = float(es_s.loc[dt, 'ret'])
                if es_ret > -1.0:
                    continue
            signal_dates.append(dt)

        # 翌日のTOPIXリターン (open→close) でShort代理
        trades = []
        for sig_dt in signal_dates:
            next_dates = topix_s[topix_s.index > sig_dt]
            if next_dates.empty:
                continue
            nxt = next_dates.iloc[0]
            if nxt['open'] <= 0 or np.isnan(nxt['open']):
                continue
            # Short: 寄付→引け の逆
            ret_bps = -(nxt['close'] / nxt['open'] - 1) * 10000 - COST_BPS * 2
            trades.append(ret_bps)

        print(f"\n  シグナル発動日 (AND条件込み): {len(signal_dates)} 件")
        print(f"  (TOPIX日足Short代理: open→close逆)")
        return summarize("sox_overnight_short (TOPIX代理)", trades), trades

    except Exception as e:
        print(f"  ❌ MariaDB接続失敗: {e}")
        print("  → strategies/sox_overnight_short/signal_check.py --verify-db で手動確認")
        return dict(name="sox_overnight_short", N="接続失敗"), []


# ─────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("全採用戦略 継続性バックテスト")
    print(f"実行日: {date.today()}")
    print("=" * 60)

    results = []

    r1, t1, sd1 = backtest_lme_on_copper()
    results.append(r1)

    r2, t2, sd2 = backtest_topix_overnight()
    results.append(r2)

    r3, t3 = backtest_eneos_vwap()
    results.append(r3)

    r4, t4 = backtest_vwap_meanrevert()
    results.append(r4)

    r5, t5 = backtest_orb()
    results.append(r5)

    r6, t6 = backtest_lasertec_ma25()
    results.append(r6)

    r7, t7 = check_sox_overnight()
    results.append(r7)

    # ─── サマリー表 ───
    print("\n\n" + "="*70)
    print("★ 全戦略サマリー (コスト2bps往復控除後)")
    print("="*70)
    print(f"{'戦略':<35} {'N':>5} {'WR%':>7} {'PF':>6} {'Sharpe':>8} {'平均bps':>9} {'判定':>6}")
    print("-"*70)

    README_SHARPE = {
        'lme_on_copper': 12.34,
        'topix_overnight': 4.79,
        'eneos_vwap_trend': 5.54,
        'vwap_morning_meanrevert': 6.11,
        'orb_breakout_long': 2.15,
        'lasertec_ma25_support': 7.68,
        'sox_overnight_short': 2.11,
    }

    for r in results:
        name = r['name']
        n = r.get('N', '-')
        base_name = name.split(' ')[0].replace(' (daily)', '')
        ref_sharpe = README_SHARPE.get(base_name, None)

        if isinstance(n, int) and n > 0:
            wr = f"{r.get('WR', 0):.1f}%"
            pf_v = r.get('PF', float('nan'))
            pf = f"{pf_v:.2f}" if not np.isnan(pf_v) else "  -"
            sh_v = r.get('Sharpe', float('nan'))
            sh = f"{sh_v:.2f}" if not np.isnan(sh_v) else "  -"
            mb = f"{r.get('mean_bps', 0):+.1f}"
            if ref_sharpe and not np.isnan(sh_v):
                if sh_v >= ref_sharpe * 0.7:
                    verdict = "✅継続"
                elif sh_v >= 2.0:
                    verdict = "⚠️低下"
                else:
                    verdict = "❌劣化"
            else:
                verdict = "  -"
        else:
            wr = pf = sh = mb = "  -"
            verdict = "  -"

        print(f"  {name:<33} {str(n):>5} {wr:>7} {pf:>6} {sh:>8} {mb:>9} {verdict:>6}")

    print("="*70)
    print("\n判定基準: README記載Sharpeの70%以上 → 継続 / Sharpe≥2.0 → 低下 / <2.0 → 劣化")
    print("(注) lme_on_copper はONトレード; SOXはTOPIX日足Short代理での評価")


if __name__ == "__main__":
    main()
