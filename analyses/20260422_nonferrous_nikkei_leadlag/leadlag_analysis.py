"""
非鉄金属(CMCU3/非鉄株) vs 日経先物(JNIc1) リードラグ分析
分析日: 2026-04-22

分析内容:
1. 日中1分足クロス相関（ラグ0-10分）: CMCU3 → JNIc1
2. オーバーナイト銅変化率 → 翌朝の日経・非鉄株の寄付リターン
3. 非鉄金属株 vs CMCU3 の1分足リードラグ
"""

import psycopg2
import pandas as pd
import numpy as np
from datetime import time

PG_CONFIG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}

SYMBOLS = {
    "CMCU3": "LME銅",
    "JNIc1": "日経先物",
    "5711.T": "三菱マテリアル",
    "5706.T": "三井金属",
    "5713.T": "住友金属鉱山",
    "5803.T": "フジクラ",
}


def load_data(sym):
    conn = psycopg2.connect(**PG_CONFIG)
    df = pd.read_sql(
        f"SELECT timestamp, open, high, low, close, volume FROM intraday_data "
        f"WHERE symbol = '{sym}' ORDER BY timestamp",
        conn
    )
    conn.close()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['jst'] = df['timestamp'] + pd.Timedelta(hours=9)
    df = df.dropna(subset=['close']).set_index('jst').sort_index()
    df['ret'] = df['close'].pct_change() * 100
    return df


def morning_session(df):
    """日本株前後場セッション (JST 9:00-15:30)"""
    return df[
        ((df.index.hour == 9) & (df.index.minute >= 0)) |
        ((df.index.hour >= 10) & (df.index.hour < 11)) |
        ((df.index.hour == 11) & (df.index.minute <= 30)) |
        ((df.index.hour == 12) & (df.index.minute >= 30)) |
        ((df.index.hour >= 13) & (df.index.hour < 15)) |
        ((df.index.hour == 15) & (df.index.minute <= 30))
    ]


# ─────────────────────────────────────────────────
# 1. 日中1分足クロス相関 (CMCU3 → JNIc1)
# ─────────────────────────────────────────────────
def cross_correlation_intraday(df_a, df_b, name_a, name_b, max_lag=10):
    """
    ラグ k で: corr( ret_a(t), ret_b(t+k) )
    k > 0 → a が b をリード
    """
    # 共通時間帯のみ（JST 9:00-15:30）
    a = morning_session(df_a)['ret'].dropna()
    b = morning_session(df_b)['ret'].dropna()

    # インデックスを分単位の整数に変換して align
    a_idx = pd.Series(a.values, index=a.index.floor('1min'))
    b_idx = pd.Series(b.values, index=b.index.floor('1min'))

    combined = pd.DataFrame({'a': a_idx, 'b': b_idx}).dropna()
    print(f"\n[日中クロス相関] {name_a} → {name_b}  (共通バー数: {len(combined)})")
    print(f"{'ラグ(分)':>8} {'相関係数':>10} {'解釈':>20}")
    print("-" * 45)

    # corr(a(t), b(t+lag)): lag>0→aがbをリード, lag<0→bがaをリード
    results = []
    for lag in range(-max_lag, max_lag + 1):
        corr = combined['a'].corr(combined['b'].shift(-lag))
        if lag < 0:
            label = f"{name_b}が{name_a}をリード{abs(lag)}分"
        elif lag == 0:
            label = "同時相関"
        else:
            label = f"{name_a}が{name_b}をリード{lag}分"
        results.append({'lag': lag, 'corr': corr, 'label': label})
        print(f"{lag:>8} {corr:>10.4f}  {label}")

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────
# 2. オーバーナイト銅変化率 → 翌朝の寄付リターン
# ─────────────────────────────────────────────────
def overnight_copper_vs_open(df_cu, df_target, target_name):
    """
    LME銅のロンドン引け(UTC 17:00 = JST 2:00頃)から
    前日終値までの変化率 → 翌朝9:01のリターンの予測力
    """
    # 銅: UTC 16:00-18:00 (JST 1:00-3:00) の平均 → 各日の「銅ロンドン引け」
    cu_london = df_cu[
        ((df_cu.index.hour >= 1) & (df_cu.index.hour < 3))
    ].copy()

    # 日付ごとに最終値を取得（その日のロンドン引け）
    cu_daily_close = cu_london.groupby(cu_london.index.date)['close'].last()

    # 翌日の日本株/日経の寄付リターン (9:00→9:01の変化率)
    open_rets = {}
    for dt, g in df_target.groupby(df_target.index.date):
        g_morning = g[g.index.hour == 9]
        if len(g_morning) >= 2:
            open_price = g_morning['close'].iloc[0]
            second_bar = g_morning['close'].iloc[1]
            if open_price > 0:
                open_rets[dt] = (second_bar / open_price - 1) * 100

    # VWAP-likeな寄付リターン: 9:00バーのretを使う
    open_ret_series = {}
    for dt, g in df_target.groupby(df_target.index.date):
        g_morning = g[(g.index.hour == 9) & (g.index.minute <= 5)]
        if len(g_morning) >= 1:
            # 最初の5分間の累積リターン
            first_close = g_morning['close'].iloc[-1]
            first_open = g_morning['open'].iloc[0]
            if first_open > 0:
                open_ret_series[dt] = (first_close / first_open - 1) * 100

    # アライン：銅の日付 → 翌日の寄付リターン
    pairs = []
    for dt, cu_close in cu_daily_close.items():
        # 翌営業日
        for offset in [1, 2, 3]:
            next_dt = pd.Timestamp(dt) + pd.Timedelta(days=offset)
            next_dt_date = next_dt.date()
            if next_dt_date in open_ret_series:
                # 銅の当日変化率（前日比）
                prev_dates = [d for d in cu_daily_close.index if d < dt]
                if prev_dates:
                    prev_close = cu_daily_close[prev_dates[-1]]
                    cu_chg = (cu_close / prev_close - 1) * 100
                    pairs.append({
                        'cu_date': dt,
                        'target_date': next_dt_date,
                        'cu_chg': cu_chg,
                        'target_open_ret': open_ret_series[next_dt_date]
                    })
                break

    if not pairs:
        print(f"\n[オーバーナイト] {target_name}: データ不足")
        return

    df_pairs = pd.DataFrame(pairs)
    corr = df_pairs['cu_chg'].corr(df_pairs['target_open_ret'])

    # 方向性一致率
    same_dir = ((df_pairs['cu_chg'] > 0) == (df_pairs['target_open_ret'] > 0)).mean()

    # 分位別平均リターン
    df_pairs['cu_quartile'] = pd.qcut(df_pairs['cu_chg'], 4, labels=['Q1(下落)', 'Q2', 'Q3', 'Q4(上昇)'])
    q_means = df_pairs.groupby('cu_quartile', observed=True)['target_open_ret'].mean()

    print(f"\n[オーバーナイト銅変化率 → 翌朝{target_name}寄付]  N={len(df_pairs)}")
    print(f"  相関係数: {corr:.4f}")
    print(f"  方向一致率: {same_dir*100:.1f}%")
    print(f"  銅変化率分位別の{target_name}寄付リターン平均:")
    for q, v in q_means.items():
        print(f"    {q}: {v:+.3f}%")

    return df_pairs


# ─────────────────────────────────────────────────
# 3. 非鉄株 vs CMCU3 の日中リードラグ
# ─────────────────────────────────────────────────
def stock_vs_copper_leadlag(df_cu, df_stock, stock_name, max_lag=5):
    """日中（9:00-15:30）の非鉄株とLME銅の相互リードラグ"""
    cu = morning_session(df_cu)['ret'].dropna()
    st = morning_session(df_stock)['ret'].dropna()

    cu_idx = pd.Series(cu.values, index=cu.index.floor('1min'))
    st_idx = pd.Series(st.values, index=st.index.floor('1min'))

    combined = pd.DataFrame({'cu': cu_idx, 'st': st_idx}).dropna()

    print(f"\n[{stock_name} vs LME銅] 日中リードラグ (N={len(combined)})")
    print(f"{'ラグ':>6} {'相関':>8} {'解釈':>25}")
    print("-" * 45)

    # corr(cu(t), st(t+lag)): lag>0→銅がリード, lag<0→株がリード
    for lag in range(-max_lag, max_lag + 1):
        corr = combined['cu'].corr(combined['st'].shift(-lag))
        if lag < 0:
            label = f"{stock_name}が銅をリード{abs(lag)}分"
        elif lag == 0:
            label = "同時"
        else:
            label = f"銅が{stock_name}をリード{lag}分"
        print(f"{lag:>6} {corr:>8.4f}  {label}")


# ─────────────────────────────────────────────────
# 4. ロンドン銅急騰シグナル → 翌朝寄付戦略バックテスト
# ─────────────────────────────────────────────────
def bt_copper_signal(df_cu, df_target, target_name, threshold=0.5):
    """
    ロンドン銅(JST 1:00-3:00)の変化率が+threshold%超 → 翌朝買い
    """
    cu_london = df_cu[
        ((df_cu.index.hour >= 1) & (df_cu.index.hour < 3))
    ].copy()

    cu_daily = cu_london.groupby(cu_london.index.date).agg(
        first_close=('close', 'first'),
        last_close=('close', 'last')
    )
    cu_daily['london_chg'] = (cu_daily['last_close'] / cu_daily['first_close'] - 1) * 100

    # 翌朝の寄付→引けリターン
    daily_rets = {}
    for dt, g in df_target.groupby(df_target.index.date):
        g_day = g[(g.index.hour >= 9) &
                  ((g.index.hour < 15) | ((g.index.hour == 15) & (g.index.minute <= 30)))]
        if len(g_day) >= 5:
            open_p = g_day['open'].iloc[0]
            close_p = g_day['close'].iloc[-1]
            open_5min = g_day['close'].iloc[min(4, len(g_day)-1)]
            if open_p > 0:
                daily_rets[dt] = {
                    'open_to_close': (close_p / open_p - 1) * 100,
                    'open5_to_close': (close_p / open_5min - 1) * 100 if open_5min > 0 else 0
                }

    trades = []
    for dt, row in cu_daily.iterrows():
        for offset in [1, 2, 3]:
            next_dt = (pd.Timestamp(dt) + pd.Timedelta(days=offset)).date()
            if next_dt in daily_rets:
                direction = 1 if row['london_chg'] > threshold else (-1 if row['london_chg'] < -threshold else 0)
                if direction != 0:
                    trades.append({
                        'date': next_dt,
                        'cu_signal': row['london_chg'],
                        'direction': direction,
                        'pnl': daily_rets[next_dt]['open_to_close'] * direction,
                        'pnl_5min': daily_rets[next_dt]['open5_to_close'] * direction
                    })
                break

    if not trades:
        print(f"\n[シグナル戦略] {target_name}: トレードなし")
        return

    df_tr = pd.DataFrame(trades)
    arr = df_tr['pnl'].values

    wr = (arr > 0).mean() * 100
    pf = arr[arr>0].sum() / abs(arr[arr<=0].sum()) if (arr<=0).any() else float('inf')
    sharpe = arr.mean() / arr.std() * np.sqrt(252) if arr.std() > 0 else 0
    avg_pnl = arr.mean()

    print(f"\n[銅ロンドンセッション急騰シグナル → 翌朝{target_name}]  閾値: ±{threshold}%")
    print(f"  トレード数: {len(df_tr)} (ロング:{(df_tr['direction']==1).sum()} / ショート:{(df_tr['direction']==-1).sum()})")
    print(f"  勝率: {wr:.1f}%")
    print(f"  PF: {pf:.2f}")
    print(f"  Sharpe: {sharpe:.2f}")
    print(f"  平均PnL: {avg_pnl:+.4f}%")

    return df_tr


# ─────────────────────────────────────────────────
# メイン実行
# ─────────────────────────────────────────────────
if __name__ == '__main__':
    print("=== データロード ===")
    data = {}
    for sym, name in SYMBOLS.items():
        print(f"  {sym} ({name}) をロード中...")
        data[sym] = load_data(sym)
        print(f"    → {len(data[sym])} バー ({data[sym].index.min().date()} ～ {data[sym].index.max().date()})")

    print("\n" + "="*60)
    print("1. 日中クロス相関: LME銅 vs 日経先物")
    print("="*60)
    cc_result = cross_correlation_intraday(
        data['CMCU3'], data['JNIc1'], "LME銅", "日経先物", max_lag=10
    )

    print("\n" + "="*60)
    print("2. オーバーナイト銅変化率 → 翌朝寄付リターン")
    print("="*60)
    for sym in ['JNIc1', '5711.T', '5706.T', '5713.T']:
        overnight_copper_vs_open(data['CMCU3'], data[sym], SYMBOLS[sym])

    print("\n" + "="*60)
    print("3. 非鉄株 vs LME銅 日中リードラグ")
    print("="*60)
    for sym in ['5711.T', '5706.T', '5713.T', '5803.T']:
        stock_vs_copper_leadlag(data['CMCU3'], data[sym], SYMBOLS[sym], max_lag=5)

    print("\n" + "="*60)
    print("4. ロンドン銅シグナル戦略バックテスト")
    print("="*60)
    for sym in ['JNIc1', '5711.T', '5713.T']:
        bt_copper_signal(data['CMCU3'], data[sym], SYMBOLS[sym], threshold=0.5)
        bt_copper_signal(data['CMCU3'], data[sym], SYMBOLS[sym], threshold=1.0)

    print("\n分析完了!")
