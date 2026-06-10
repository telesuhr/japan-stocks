"""
イントラデイ システムトレード 戦略検証
========================================
1分足データ(stocks_intraday_YYYYMM)を使って
繰り返し可能なイントラエッジを探す

検証戦略:
  1. ORB5   - 5分オープニングレンジブレイク
  2. ORB15  - 15分オープニングレンジブレイク
  3. GAP_GO - ギャップアップ後の前場継続
  4. GAP_FILL - ギャップアップ後の埋め戻し
  5. VWAP_REV - VWAP下方乖離からの回帰
  6. FIRST15_REV - 前場15分方向の逆張り(9:30以降)
  7. LUNCH_MOM - 昼休み後の方向性(12:30-13:00)
  8. CLOSE30  - 後場終盤14:30以降のモメンタム
  9. VOL_SPIKE - 出来高スパイク後の5分方向
 10. TIME_930  - 9:30起点の方向性バイアス

コスト: 片道0.05%(証券会社手数料+スリッページ想定)
実行: python3 intraday_strategies.py
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import pandas as pd
import numpy as np
import psycopg2
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

DB_URL  = "postgresql://postgres@localhost/market_data"
OUT_DIR = "/mnt/d/Root/ClaudeCode/01_Trading/japan-stocks/analyses/20260605_sector_edge_research"
COST    = 0.001   # 往復コスト 0.1% (片道0.05%)

# 対象銘柄（流動性高い主要銘柄に絞る）
TARGET_CODES = {
    '80350': '東京エレクトロン',
    '68570': 'アドバンテスト',
    '69200': 'レーザーテック',
    '61460': 'ディスコ',
    '99840': 'ソフトバンクG',
    '57130': '住友金属鉱山',
    '57110': '三菱マテリアル',
    '69810': '村田製作所',
    '83060': '三菱UFJ',
    '70110': '三菱重工',
}
CODES = list(TARGET_CODES.keys())

# ─── データロード ──────────────────────────────────────────────────────────
def get_conn(): return psycopg2.connect(DB_URL)

def load_intraday(codes, months=None):
    """
    直近N月分の1分足を取得。月次パーティションをUNION。
    lunch break(11:30-12:30)は除外済みデータとして扱う。
    """
    if months is None:
        months = ['202405','202406','202407','202408','202409','202410',
                  '202411','202412','202501','202502','202503','202504',
                  '202505','202506']
    code_list = ",".join(f"'{c}'" for c in codes)
    unions = " UNION ALL ".join(
        f"SELECT code, ts, open, high, low, close, volume "
        f"FROM public.stocks_intraday_{m} "
        f"WHERE code IN ({code_list})"
        for m in months
    )
    sql = f"SELECT * FROM ({unions}) t ORDER BY code, ts"
    with get_conn() as conn:
        df = pd.read_sql(sql, conn, parse_dates=['ts'])
    df = df.astype({'open':'float','high':'float','low':'float','close':'float','volume':'float'})
    return df

def load_daily_close(codes):
    """前日終値（ギャップ計算用）"""
    code_list = ",".join(f"'{c}'" for c in codes)
    sql = f"""
        SELECT code, date, adj_close AS prev_close
        FROM public.stocks_daily
        WHERE code IN ({code_list}) AND date >= '2024-04-01'
        ORDER BY code, date
    """
    with get_conn() as conn:
        df = pd.read_sql(sql, conn, parse_dates=['date'])
    return df

print("データロード中...")
intra = load_intraday(CODES)
daily_close = load_daily_close(CODES)
print(f"  1分足: {len(intra):,}行  銘柄: {intra['code'].nunique()}  期間: {intra['ts'].min().date()} ~ {intra['ts'].max().date()}")

# ─── 日次OHLC + 時間帯別データ構築 ──────────────────────────────────────
intra['date']    = intra['ts'].dt.date
intra['time']    = intra['ts'].dt.time
intra['minute']  = intra['ts'].dt.hour * 60 + intra['ts'].dt.minute
intra['is_am']   = intra['minute'] < 690   # 11:30 = 690
intra['is_pm']   = intra['minute'] >= 750  # 12:30 = 750

# 前日終値をマージ（ギャップ計算）
daily_close['date_next'] = daily_close['date'] + pd.Timedelta(days=1)
daily_close_shifted = daily_close.rename(columns={'date_next':'date_join','prev_close':'prev_close_val'})
intra['date_dt'] = pd.to_datetime(intra['date'])
# 前日終値は date-1 の close → 実際の取引日ベースで merge
# 簡便化: stocks_daily の前日closeをdate基準でマージ
dc = daily_close.copy()
dc['date'] = pd.to_datetime(dc['date'])
dc_shifted = dc.sort_values(['code','date']).copy()
dc_shifted['prev_close'] = dc_shifted.groupby('code')['prev_close'].shift(1)
dc_shifted = dc_shifted[['code','date','prev_close']].dropna()

intra['date_dt'] = pd.to_datetime(intra['date'])
intra = intra.merge(dc_shifted.rename(columns={'date':'date_dt'}), on=['code','date_dt'], how='left')

# ─── 統計計算 ─────────────────────────────────────────────────────────────
def calc_stats(returns, strategy_name, cost=COST):
    r = pd.Series(returns).dropna()
    r_net = r - cost
    n = len(r_net)
    if n < 10:
        return {'strategy': strategy_name, 'n': n, 'note': 'N不足'}
    wr   = (r_net > 0).mean()
    mu   = r_net.mean()
    sd   = r_net.std()
    t, p = stats.ttest_1samp(r_net, 0)
    sh   = mu / sd * np.sqrt(252 * 5) if sd > 0 else np.nan  # イントラ年率
    # 最大DD（累積）
    cum  = (1 + r_net).cumprod()
    mdd  = (cum / cum.cummax() - 1).min()
    return {
        'strategy': strategy_name,
        'n': n,
        'win_rate': wr,
        'mean_net': mu,
        'sharpe_annual': sh,
        't_stat': t,
        'p_value': p,
        'max_dd': mdd,
        'edge_class': classify(t, mu),
    }

def classify(t, mu):
    if pd.isna(t): return 'insufficient'
    if t >  2.5 and mu > 0: return 'strong_pos'
    if t >  1.8 and mu > 0: return 'weak_pos'
    if t < -2.5 and mu < 0: return 'strong_neg'
    if t < -1.8 and mu < 0: return 'weak_neg'
    return 'noise'

results = []

# ─── 1. ORB5 (5分オープニングレンジブレイク) ─────────────────────────────
# 9:00-9:05のhigh/lowを定義 → 9:06以降にブレイクした方向にエントリー
# 出口: 当日前場引け(11:30) または ストップ(ORB幅の1倍)
print("\n[1] ORB5 計算中...")

def orb_strategy(df, orb_end_min, target_end_min, stop_multiplier=1.5, name='ORB'):
    """
    orb_end_min: ORB定義終了分(例:5分→9:05=545)
    target_end_min: 出口時間(例:前場終了11:25=685)
    """
    OPEN_MIN = 540  # 9:00
    ORB_OPEN = OPEN_MIN
    ORB_CLOSE = OPEN_MIN + orb_end_min

    trade_rets = []
    for (code, date_val), gdf in df.groupby(['code','date']):
        gdf = gdf.sort_values('minute')
        # ORB期間
        orb_bars = gdf[(gdf['minute'] >= ORB_OPEN) & (gdf['minute'] < ORB_CLOSE)]
        if len(orb_bars) < 2: continue
        orb_high = orb_bars['high'].max()
        orb_low  = orb_bars['low'].min()
        orb_range = orb_high - orb_low
        if orb_range <= 0: continue

        # ORB後のバー
        post_bars = gdf[(gdf['minute'] >= ORB_CLOSE) & (gdf['minute'] <= target_end_min)]
        if len(post_bars) < 3: continue

        entered = False
        entry_price, direction, stop = None, None, None
        for _, bar in post_bars.iterrows():
            if not entered:
                if bar['high'] > orb_high:  # 上ブレイク
                    entry_price = orb_high
                    direction   = +1
                    stop        = orb_high - orb_range * stop_multiplier
                    entered     = True
                elif bar['low'] < orb_low:  # 下ブレイク
                    entry_price = orb_low
                    direction   = -1
                    stop        = orb_low + orb_range * stop_multiplier
                    entered     = True
            else:
                # ストップ確認
                if direction == +1 and bar['low'] < stop:
                    ret = (stop - entry_price) / entry_price * direction
                    trade_rets.append(ret)
                    break
                elif direction == -1 and bar['high'] > stop:
                    ret = (stop - entry_price) / entry_price * direction * -1
                    trade_rets.append(ret)  # ショートのリターン
                    break
        else:
            if entered and entry_price is not None:
                exit_price = post_bars.iloc[-1]['close']
                if direction == +1:
                    ret = (exit_price - entry_price) / entry_price
                else:
                    ret = (entry_price - exit_price) / entry_price
                trade_rets.append(ret)

    return trade_rets

orb5_rets  = orb_strategy(intra, 5,  685, name='ORB5')   # 9:05以降 → 11:25まで
orb15_rets = orb_strategy(intra, 15, 685, name='ORB15')  # 9:15以降 → 11:25まで
orb5_pm    = orb_strategy(intra, 5,  890, name='ORB5_PM') # 9:05以降 → 14:50まで

results.append(calc_stats(orb5_rets,  'ORB5  (9:05ブレイク→11:25出口)'))
results.append(calc_stats(orb15_rets, 'ORB15 (9:15ブレイク→11:25出口)'))
results.append(calc_stats(orb5_pm,   'ORB5  (9:05ブレイク→14:50出口)'))
print(f"  ORB5={len(orb5_rets)}, ORB15={len(orb15_rets)}, ORB5_PM={len(orb5_pm)}")

# ─── 2. ギャップ戦略 ─────────────────────────────────────────────────────
print("\n[2] ギャップ戦略...")

def gap_strategy(df, gap_threshold, go=True, hold_minutes=60):
    """
    go=True:  ギャップ方向に継続 (gap and go)
    go=False: ギャップを埋め方向 (gap fill)
    """
    trade_rets = []
    for (code, date_val), gdf in df.groupby(['code','date']):
        gdf = gdf.sort_values('minute')
        pc = gdf['prev_close'].iloc[0]
        if pd.isna(pc) or pc <= 0: continue
        first_bar = gdf[gdf['minute'] >= 540].iloc[0] if len(gdf[gdf['minute'] >= 540]) > 0 else None
        if first_bar is None: continue
        open_price = first_bar['open']
        gap = (open_price - pc) / pc
        if abs(gap) < gap_threshold: continue
        direction = np.sign(gap) if go else -np.sign(gap)

        # エントリー: 寄り付き open price
        entry_price = open_price
        entry_min   = first_bar['minute']
        exit_min    = entry_min + hold_minutes
        exit_bars   = gdf[gdf['minute'] >= exit_min]
        if len(exit_bars) == 0:
            exit_price = gdf.iloc[-1]['close']
        else:
            exit_price = exit_bars.iloc[0]['open']

        ret = direction * (exit_price - entry_price) / entry_price
        trade_rets.append(ret)

    return trade_rets

gap_go_2_rets   = gap_strategy(intra, 0.02, go=True,  hold_minutes=60)
gap_fill_2_rets = gap_strategy(intra, 0.02, go=False, hold_minutes=60)
gap_go_1_rets   = gap_strategy(intra, 0.01, go=True,  hold_minutes=30)
gap_fill_1_rets = gap_strategy(intra, 0.01, go=False, hold_minutes=30)

results.append(calc_stats(gap_go_2_rets,   'GAP_GO   (ギャップ2%超→1時間継続)'))
results.append(calc_stats(gap_fill_2_rets, 'GAP_FILL (ギャップ2%超→1時間逆張り)'))
results.append(calc_stats(gap_go_1_rets,   'GAP_GO   (ギャップ1%超→30分継続)'))
results.append(calc_stats(gap_fill_1_rets, 'GAP_FILL (ギャップ1%超→30分逆張り)'))
print(f"  GAP_GO2={len(gap_go_2_rets)}, GAP_FILL2={len(gap_fill_2_rets)}")
print(f"  GAP_GO1={len(gap_go_1_rets)}, GAP_FILL1={len(gap_fill_1_rets)}")

# ─── 3. VWAP乖離 平均回帰 ─────────────────────────────────────────────────
print("\n[3] VWAP平均回帰...")

def vwap_reversion(df, dev_threshold=0.005, hold_minutes=30, exit_at_vwap=True):
    """
    日次累積VWAPから dev_threshold 以上乖離したらエントリー
    出口: VWAP到達 または hold_minutes後
    """
    trade_rets = []
    for (code, date_val), gdf in df.groupby(['code','date']):
        gdf = gdf.sort_values('minute').copy()
        # VWAP計算
        gdf['tp']      = (gdf['high'] + gdf['low'] + gdf['close']) / 3
        gdf['cum_pv']  = (gdf['tp'] * gdf['volume']).cumsum()
        gdf['cum_vol'] = gdf['volume'].cumsum()
        gdf['vwap']    = gdf['cum_pv'] / gdf['cum_vol'].replace(0, np.nan)
        gdf['vwap_dev']= (gdf['close'] - gdf['vwap']) / gdf['vwap']

        in_trade = False
        entry_price, direction, entry_idx, vwap_at_entry = None, None, None, None

        for i, (_, bar) in enumerate(gdf.iterrows()):
            # 前場のみ（9:10~11:20）
            if not (550 <= bar['minute'] <= 680): continue
            if pd.isna(bar['vwap_dev']): continue

            if not in_trade:
                dev = bar['vwap_dev']
                if dev < -dev_threshold:    # VWAP下方乖離 → 買い
                    entry_price = bar['close']
                    direction   = +1
                    entry_idx   = i
                    vwap_at_entry = bar['vwap']
                    in_trade    = True
                elif dev > dev_threshold:   # VWAP上方乖離 → 売り
                    entry_price = bar['close']
                    direction   = -1
                    entry_idx   = i
                    vwap_at_entry = bar['vwap']
                    in_trade    = True
            else:
                elapsed = i - entry_idx
                reached_vwap = (direction == +1 and bar['close'] >= bar['vwap']) or \
                               (direction == -1 and bar['close'] <= bar['vwap'])
                if (exit_at_vwap and reached_vwap) or elapsed >= hold_minutes:
                    exit_price = bar['close']
                    ret = direction * (exit_price - entry_price) / entry_price
                    trade_rets.append(ret)
                    in_trade = False
    return trade_rets

vwap_05_rets = vwap_reversion(intra, dev_threshold=0.005, hold_minutes=30)
vwap_10_rets = vwap_reversion(intra, dev_threshold=0.010, hold_minutes=30)

results.append(calc_stats(vwap_05_rets, 'VWAP_REV (乖離0.5%→VWAP回帰, 最大30分)'))
results.append(calc_stats(vwap_10_rets, 'VWAP_REV (乖離1.0%→VWAP回帰, 最大30分)'))
print(f"  VWAP_REV0.5={len(vwap_05_rets)}, VWAP_REV1.0={len(vwap_10_rets)}")

# ─── 4. 前場15分方向の逆張り ─────────────────────────────────────────────
print("\n[4] 前場15分逆張り...")

def first_15min_reversal(df, entry_min=555, exit_min=680):
    """
    9:00-9:15の方向と逆に9:30からエントリー → 前場引けで出口
    """
    trade_rets = []
    for (code, date_val), gdf in df.groupby(['code','date']):
        gdf = gdf.sort_values('minute')
        first15 = gdf[(gdf['minute'] >= 540) & (gdf['minute'] < 555)]
        if len(first15) < 3: continue
        first15_ret = (first15.iloc[-1]['close'] - first15.iloc[0]['open']) / first15.iloc[0]['open']
        if abs(first15_ret) < 0.003: continue  # 方向が弱すぎる場合はスキップ

        entry_bars = gdf[gdf['minute'] >= entry_min]
        if len(entry_bars) == 0: continue
        entry_price = entry_bars.iloc[0]['open']
        direction   = -np.sign(first15_ret)  # 逆張り

        exit_bars = gdf[gdf['minute'] >= exit_min]
        exit_price = exit_bars.iloc[0]['close'] if len(exit_bars) > 0 else gdf.iloc[-1]['close']
        ret = direction * (exit_price - entry_price) / entry_price
        trade_rets.append(ret)
    return trade_rets

def first_15min_momentum(df, entry_min=555, exit_min=680):
    """逆張りではなくモメンタム追従"""
    trade_rets = []
    for (code, date_val), gdf in df.groupby(['code','date']):
        gdf = gdf.sort_values('minute')
        first15 = gdf[(gdf['minute'] >= 540) & (gdf['minute'] < 555)]
        if len(first15) < 3: continue
        first15_ret = (first15.iloc[-1]['close'] - first15.iloc[0]['open']) / first15.iloc[0]['open']
        if abs(first15_ret) < 0.003: continue

        entry_bars = gdf[gdf['minute'] >= entry_min]
        if len(entry_bars) == 0: continue
        entry_price = entry_bars.iloc[0]['open']
        direction   = np.sign(first15_ret)   # 順張り

        exit_bars = gdf[gdf['minute'] >= exit_min]
        exit_price = exit_bars.iloc[0]['close'] if len(exit_bars) > 0 else gdf.iloc[-1]['close']
        ret = direction * (exit_price - entry_price) / entry_price
        trade_rets.append(ret)
    return trade_rets

rev15_rets = first_15min_reversal(intra)
mom15_rets = first_15min_momentum(intra)
results.append(calc_stats(rev15_rets, 'FIRST15_REV (前場15分逆張り→前場引け)'))
results.append(calc_stats(mom15_rets, 'FIRST15_MOM (前場15分順張り→前場引け)'))
print(f"  REV={len(rev15_rets)}, MOM={len(mom15_rets)}")

# ─── 5. 昼休み後のモメンタム ─────────────────────────────────────────────
print("\n[5] 昼休み後モメンタム...")

def lunch_momentum(df, hold_minutes=30):
    """
    前場の方向が後場冒頭(12:30-13:00)も続くか
    """
    trade_rets = []
    for (code, date_val), gdf in df.groupby(['code','date']):
        gdf = gdf.sort_values('minute')
        am = gdf[gdf['is_am']]
        pm = gdf[gdf['is_pm']]
        if len(am) < 5 or len(pm) < 3: continue

        am_ret = (am.iloc[-1]['close'] - am.iloc[0]['open']) / am.iloc[0]['open']
        if abs(am_ret) < 0.002: continue  # 前場がほぼフラットはスキップ

        entry_price = pm.iloc[0]['open']
        direction   = np.sign(am_ret)

        exit_bars = pm[pm['minute'] >= 750 + hold_minutes]
        exit_price = exit_bars.iloc[0]['close'] if len(exit_bars) > 0 else pm.iloc[-1]['close']
        ret = direction * (exit_price - entry_price) / entry_price
        trade_rets.append(ret)
    return trade_rets

def lunch_reversal(df, hold_minutes=30):
    """前場方向の逆に後場エントリー"""
    trade_rets = []
    for (code, date_val), gdf in df.groupby(['code','date']):
        gdf = gdf.sort_values('minute')
        am = gdf[gdf['is_am']]
        pm = gdf[gdf['is_pm']]
        if len(am) < 5 or len(pm) < 3: continue
        am_ret = (am.iloc[-1]['close'] - am.iloc[0]['open']) / am.iloc[0]['open']
        if abs(am_ret) < 0.002: continue
        entry_price = pm.iloc[0]['open']
        direction   = -np.sign(am_ret)   # 逆張り
        exit_bars = pm[pm['minute'] >= 750 + hold_minutes]
        exit_price = exit_bars.iloc[0]['close'] if len(exit_bars) > 0 else pm.iloc[-1]['close']
        ret = direction * (exit_price - entry_price) / entry_price
        trade_rets.append(ret)
    return trade_rets

lunch_mom_rets = lunch_momentum(intra)
lunch_rev_rets = lunch_reversal(intra)
results.append(calc_stats(lunch_mom_rets, 'LUNCH_MOM (前場方向→後場30分継続)'))
results.append(calc_stats(lunch_rev_rets, 'LUNCH_REV (前場方向逆→後場30分)'))
print(f"  MOM={len(lunch_mom_rets)}, REV={len(lunch_rev_rets)}")

# ─── 6. 後場終盤モメンタム ────────────────────────────────────────────────
print("\n[6] 後場終盤モメンタム...")

def close_momentum(df, start_min=870, end_min=890):
    """
    14:30(870)以降の30分方向が引けまで続くか
    start_min: 方向確認開始, end_min: エントリー, 出口: 15:00(900)引け
    """
    trade_rets = []
    for (code, date_val), gdf in df.groupby(['code','date']):
        gdf = gdf.sort_values('minute')
        signal_bars = gdf[(gdf['minute'] >= start_min) & (gdf['minute'] < end_min)]
        exit_bars   = gdf[gdf['minute'] >= end_min]
        if len(signal_bars) < 3 or len(exit_bars) < 2: continue

        signal_ret  = (signal_bars.iloc[-1]['close'] - signal_bars.iloc[0]['open']) / signal_bars.iloc[0]['open']
        if abs(signal_ret) < 0.002: continue

        entry_price = exit_bars.iloc[0]['open']
        direction   = np.sign(signal_ret)
        exit_price  = exit_bars.iloc[-1]['close']
        ret = direction * (exit_price - entry_price) / entry_price
        trade_rets.append(ret)
    return trade_rets

close_mom_rets = close_momentum(intra)
results.append(calc_stats(close_mom_rets, 'CLOSE_MOM (14:30方向→15:00引け)'))
print(f"  CLOSE={len(close_mom_rets)}")

# ─── 7. 出来高スパイク後の方向 ───────────────────────────────────────────
print("\n[7] 出来高スパイク後方向...")

def vol_spike_momentum(df, spike_ratio=3.0, hold_minutes=10, rolling_window=20):
    """
    直近rolling_window分の平均出来高のspike_ratio倍以上の分 → 方向継続
    """
    trade_rets = []
    for (code, date_val), gdf in df.groupby(['code','date']):
        gdf = gdf.sort_values('minute').reset_index(drop=True)
        gdf['vol_ma'] = gdf['volume'].rolling(rolling_window, min_periods=5).mean().shift(1)
        gdf['bar_ret'] = (gdf['close'] - gdf['open']) / gdf['open'].replace(0, np.nan)

        in_trade = False
        for i, row in gdf.iterrows():
            if not (560 <= row['minute'] <= 860): continue  # 9:20-14:20
            if in_trade:
                in_trade = False
                continue
            if pd.isna(row['vol_ma']) or row['vol_ma'] <= 0: continue
            if row['volume'] < row['vol_ma'] * spike_ratio: continue
            if abs(row['bar_ret']) < 0.001: continue  # 方向が弱い

            direction   = np.sign(row['bar_ret'])
            entry_price = row['close']
            exit_idx    = min(i + hold_minutes, len(gdf) - 1)
            exit_price  = gdf.loc[exit_idx, 'close']
            ret = direction * (exit_price - entry_price) / entry_price
            trade_rets.append(ret)
            in_trade = True
    return trade_rets

vol_spike_rets = vol_spike_momentum(intra, spike_ratio=3.0, hold_minutes=10)
vol_spike_rets5 = vol_spike_momentum(intra, spike_ratio=2.0, hold_minutes=5)
results.append(calc_stats(vol_spike_rets,  'VOL_SPIKE (3x出来高→10分方向継続)'))
results.append(calc_stats(vol_spike_rets5, 'VOL_SPIKE (2x出来高→5分方向継続)'))
print(f"  SPIKE3x={len(vol_spike_rets)}, SPIKE2x={len(vol_spike_rets5)}")

# ─── 8. 時間帯別バイアス（9:30前後）────────────────────────────────────
print("\n[8] 時間帯別バイアス...")

def time_slot_bias(df, slot_start, slot_end, hold_minutes=15, direction=+1):
    """特定時間帯に direction でエントリー → hold_minutes後決済"""
    trade_rets = []
    for (code, date_val), gdf in df.groupby(['code','date']):
        gdf = gdf.sort_values('minute')
        slot = gdf[(gdf['minute'] >= slot_start) & (gdf['minute'] < slot_end)]
        if len(slot) == 0: continue
        entry_price = slot.iloc[0]['open']
        exit_min    = slot_start + hold_minutes
        exit_bars   = gdf[gdf['minute'] >= exit_min]
        if len(exit_bars) == 0: continue
        exit_price = exit_bars.iloc[0]['close']
        ret = direction * (exit_price - entry_price) / entry_price
        trade_rets.append(ret)
    return trade_rets

# 9:30(570)に無条件ロング → 15分後決済（統計的バイアスがあるか）
t930_long  = time_slot_bias(intra, 570, 572, hold_minutes=30, direction=+1)
# 11:00(660)にショート → 前場引けまで (売り圧力仮説)
t1100_short = time_slot_bias(intra, 660, 662, hold_minutes=25, direction=-1)
# 12:30(750)にロング → 30分保有
t1230_long  = time_slot_bias(intra, 750, 752, hold_minutes=30, direction=+1)

results.append(calc_stats(t930_long,   'TIME_930  (9:30無条件ロング→30分)'))
results.append(calc_stats(t1100_short, 'TIME_1100 (11:00無条件ショート→前引け)'))
results.append(calc_stats(t1230_long,  'TIME_1230 (12:30無条件ロング→30分)'))
print(f"  T930={len(t930_long)}, T1100={len(t1100_short)}, T1230={len(t1230_long)}")

# ─── 結果表示 ─────────────────────────────────────────────────────────────
MARKER = {
    'strong_pos': '★★ 強↑',
    'weak_pos':   '★  弱↑',
    'noise':      '   中立',
    'weak_neg':   '▼  弱↓',
    'strong_neg': '▼▼ 強↓',
    'insufficient':'   N不足',
}

print("\n" + "="*90)
print("  イントラデイ システム戦略 検証結果  (コスト控除後: 往復0.1%)")
print("="*90)
print(f"  {'戦略':<44} {'N':>5}  {'勝率':>6}  {'期待値':>8}  {'t値':>5}  {'Sharpe':>6}  {'最大DD':>7}  判定")
print("-"*90)

for r in results:
    if 'note' in r:
        print(f"  {r['strategy']:<44} {'N不足':>5}")
        continue
    n   = int(r['n'])
    wr  = f"{r['win_rate']*100:5.1f}%" if pd.notna(r.get('win_rate')) else '    –'
    mn  = f"{r['mean_net']*100:+.3f}%"  if pd.notna(r.get('mean_net')) else '     –'
    t   = f"{r['t_stat']:+.2f}"         if pd.notna(r.get('t_stat'))  else '    –'
    sh  = f"{r['sharpe_annual']:+.2f}"  if pd.notna(r.get('sharpe_annual')) else '    –'
    dd  = f"{r['max_dd']*100:.2f}%"     if pd.notna(r.get('max_dd'))  else '    –'
    ec  = MARKER.get(r.get('edge_class',''), r.get('edge_class',''))
    print(f"  {r['strategy']:<44} {n:>5}  {wr}  {mn:>8}  {t:>5}  {sh:>6}  {dd:>7}  {ec}")

# 有望戦略
print("\n" + "="*90)
print("  有望戦略 (t値>1.8, コスト控除後プラス)")
print("="*90)
promising = [r for r in results if 'note' not in r and
             r.get('edge_class') in ('strong_pos','weak_pos','strong_neg','weak_neg')]
if not promising:
    print("  なし")
else:
    for r in sorted(promising, key=lambda x: abs(x.get('t_stat',0)), reverse=True):
        n  = int(r['n'])
        wr = f"{r['win_rate']*100:.1f}%"
        mn = f"{r['mean_net']*100:+.3f}%"
        t  = f"{r['t_stat']:+.2f}"
        sh = f"{r['sharpe_annual']:+.2f}"
        dd = f"{r['max_dd']*100:.2f}%"
        ec = MARKER.get(r['edge_class'], r['edge_class'])
        print(f"  [{ec}] {r['strategy']}")
        print(f"         N={n}  勝率={wr}  期待値/トレード={mn}  t={t}  Sharpe={sh}  最大DD={dd}")
        # 年間期待収益（N回×期待値）
        annual_n = n / ((pd.Timestamp.now() - pd.Timestamp('2024-05-01')).days / 365)
        annual_ret = annual_n * r['mean_net']
        print(f"         年間約{annual_n:.0f}回 × {r['mean_net']*100:+.3f}% = 年率約{annual_ret*100:+.1f}% (10銘柄×資金×配分)")
        print()

# CSV
pd.DataFrame(results).to_csv(f"{OUT_DIR}/intraday_results.csv", index=False)
print(f"保存: {OUT_DIR}/intraday_results.csv")
print("完了")
