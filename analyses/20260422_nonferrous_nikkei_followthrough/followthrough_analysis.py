"""
日経先物上昇 → 非鉄金属株の追従性分析
分析日: 2026-04-22

分析内容:
1. 日中1分足リードラグ: JNIc1 → 非鉄株（ラグ0-10分）
2. 日次リターン相関とベータ（上昇日/下落日別）
3. 日経の動き幅別追従率（大幅高/小幅高/中立/小幅安/大幅安）
4. 日中の追従性タイムライン（日経が動いた後、各非鉄株が何分後に追いつくか）
5. 乖離戦略バックテスト（日経上昇後に遅れた非鉄株を買う）
"""

import psycopg2
import pandas as pd
import numpy as np

PG_CONFIG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}

NONFERROUS = {
    "5711.T": "三菱マテリアル",
    "5706.T": "三井金属",
    "5713.T": "住友鉱山",
    "5803.T": "フジクラ",
    "5802.T": "住友電工",
    "5801.T": "古河電工",
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


def trading_hours(df):
    """前後場のみ抽出 (JST 9:00-15:30)"""
    h = df.index.hour
    m = df.index.minute
    return df[
        ((h == 9)) |
        ((h >= 10) & (h < 11)) |
        ((h == 11) & (m <= 30)) |
        ((h == 12) & (m >= 30)) |
        ((h >= 13) & (h < 15)) |
        ((h == 15) & (m <= 30))
    ]


# ──────────────────────────────────────────────────────────
# 1. 日中1分足リードラグ: 日経先物 → 各非鉄株
# ──────────────────────────────────────────────────────────
def intraday_leadlag_jni_vs_stocks(df_jni, stocks_data, max_lag=10):
    print("="*60)
    print("1. 日中1分足リードラグ: 日経先物 → 各非鉄株")
    print("="*60)

    jni = trading_hours(df_jni)['ret'].dropna()
    jni_aligned = pd.Series(jni.values, index=jni.index.floor('1min'))

    print(f"\n{'銘柄':<12}", end="")
    for lag in range(0, max_lag + 1):
        print(f"  lag+{lag:>2}", end="")
    print()
    print("-" * (12 + 8 * (max_lag + 1)))

    results = {}
    for sym, name in NONFERROUS.items():
        if sym not in stocks_data:
            continue
        st = trading_hours(stocks_data[sym])['ret'].dropna()
        st_aligned = pd.Series(st.values, index=st.index.floor('1min'))
        combined = pd.DataFrame({'jni': jni_aligned, 'st': st_aligned}).dropna()

        corrs = {}
        for lag in range(-max_lag, max_lag + 1):
            # corr(jni(t), st(t+lag)): lag>0 → 日経が非鉄をリード
            corrs[lag] = combined['jni'].corr(combined['st'].shift(-lag))
        results[sym] = corrs

        print(f"{name:<12}", end="")
        for lag in range(0, max_lag + 1):
            print(f"  {corrs[lag]:>5.3f}", end="")
        print()

    # どのラグで相関が最大か
    print(f"\n{'銘柄':<12}  {'最大相関ラグ':>10}  {'最大相関値':>10}  {'lag0相関':>10}  {'lag1相関':>10}")
    print("-" * 55)
    for sym, name in NONFERROUS.items():
        if sym not in results:
            continue
        corrs = results[sym]
        # lag 0-5 で最大
        best_lag = max(range(0, 6), key=lambda k: corrs[k])
        print(f"{name:<12}  {best_lag:>10}  {corrs[best_lag]:>10.4f}  {corrs[0]:>10.4f}  {corrs[1]:>10.4f}")

    return results


# ──────────────────────────────────────────────────────────
# 2. 日次リターン相関・ベータ
# ──────────────────────────────────────────────────────────
def daily_return_beta(df_jni, stocks_data):
    print("\n" + "="*60)
    print("2. 日次リターン相関・ベータ（全体/上昇日/下落日）")
    print("="*60)

    # 日次リターン（前場寄付→後場引け）
    def daily_ret(df):
        rets = {}
        for dt, g in df.groupby(df.index.date):
            g_day = trading_hours(g)
            if len(g_day) < 10:
                continue
            op = g_day['open'].iloc[0]
            cl = g_day['close'].iloc[-1]
            if op > 0:
                rets[dt] = (cl / op - 1) * 100
        return pd.Series(rets)

    jni_daily = daily_ret(df_jni)

    print(f"\n{'銘柄':<12}  {'相関':>6}  {'β全体':>6}  {'β上昇日':>8}  {'β下落日':>8}  {'上昇追従率':>10}  {'下落追従率':>10}")
    print("-" * 72)

    for sym, name in NONFERROUS.items():
        if sym not in stocks_data:
            continue
        st_daily = daily_ret(stocks_data[sym])
        combined = pd.DataFrame({'jni': jni_daily, 'st': st_daily}).dropna()
        if len(combined) < 20:
            continue

        corr = combined['jni'].corr(combined['st'])
        beta = np.cov(combined['jni'], combined['st'])[0, 1] / np.var(combined['jni'])

        up_days = combined[combined['jni'] > 0]
        dn_days = combined[combined['jni'] < 0]

        beta_up = np.cov(up_days['jni'], up_days['st'])[0, 1] / np.var(up_days['jni']) if len(up_days) > 5 else np.nan
        beta_dn = np.cov(dn_days['jni'], dn_days['st'])[0, 1] / np.var(dn_days['jni']) if len(dn_days) > 5 else np.nan

        # 追従率：日経と同方向に動いた日の割合
        follow_up = ((up_days['st'] > 0).mean() * 100) if len(up_days) > 0 else np.nan
        follow_dn = ((dn_days['st'] < 0).mean() * 100) if len(dn_days) > 0 else np.nan

        print(f"{name:<12}  {corr:>6.3f}  {beta:>6.2f}  {beta_up:>8.2f}  {beta_dn:>8.2f}  {follow_up:>9.1f}%  {follow_dn:>9.1f}%")


# ──────────────────────────────────────────────────────────
# 3. 日経変化幅別・非鉄株の平均リターン
# ──────────────────────────────────────────────────────────
def nikkei_bin_vs_nonferrous(df_jni, stocks_data):
    print("\n" + "="*60)
    print("3. 日経変化幅別 → 非鉄株の平均日次リターン")
    print("="*60)

    def daily_ret(df):
        rets = {}
        for dt, g in df.groupby(df.index.date):
            g_day = trading_hours(g)
            if len(g_day) < 10:
                continue
            op = g_day['open'].iloc[0]
            cl = g_day['close'].iloc[-1]
            if op > 0:
                rets[dt] = (cl / op - 1) * 100
        return pd.Series(rets)

    jni_daily = daily_ret(df_jni)

    # ビン定義
    bins = [-np.inf, -2, -1, -0.3, 0.3, 1, 2, np.inf]
    labels = ['大幅安(<-2%)', '安(-2~-1%)', '小幅安(-1~-0.3%)',
              'フラット(±0.3%)', '小幅高(0.3~1%)', '高(1~2%)', '大幅高(>2%)']
    jni_bin = pd.cut(jni_daily, bins=bins, labels=labels)

    all_data = pd.DataFrame({'jni': jni_daily, 'bin': jni_bin})

    for sym, name in NONFERROUS.items():
        if sym not in stocks_data:
            continue
        st_daily = daily_ret(stocks_data[sym])
        combined = all_data.join(st_daily.rename('st')).dropna()
        if len(combined) < 20:
            continue

        bin_stats = combined.groupby('bin', observed=True)['st'].agg(['mean', 'count'])
        print(f"\n  {name}:")
        print(f"  {'日経レンジ':<20}  {'N':>4}  {'平均リターン':>12}")
        for b in labels:
            if b in bin_stats.index:
                row = bin_stats.loc[b]
                print(f"  {b:<20}  {int(row['count']):>4}  {row['mean']:>+11.3f}%")


# ──────────────────────────────────────────────────────────
# 4. 日中追従タイムライン（日経が大きく動いた後の累積追従）
# ──────────────────────────────────────────────────────────
def intraday_followthrough_timeline(df_jni, stocks_data, jni_threshold=0.3):
    """
    日経先物が30分で+jni_threshold%動いた後、
    非鉄株が何分後までに何%追いつくか
    """
    print("\n" + "="*60)
    print(f"4. 日経先物が30分で±{jni_threshold}%超動いた後の非鉄株追従タイムライン")
    print("="*60)

    jni_th = trading_hours(df_jni).copy()
    jni_th = jni_th.resample('1min').last().dropna(subset=['close'])

    # 日経先物の30分窓でのリターン
    jni_30m = jni_th['close'].pct_change(30) * 100

    # 上昇シグナルと下落シグナルを収集
    events_up = jni_30m[jni_30m > jni_threshold].index
    events_dn = jni_30m[jni_30m < -jni_threshold].index

    print(f"\n日経上昇シグナル数: {len(events_up)}, 下落シグナル数: {len(events_dn)}")

    for direction, events, label in [(1, events_up, "上昇"), (-1, events_dn, "下落")]:
        if len(events) == 0:
            continue

        print(f"\n  ---- 日経{label}後の追従 ----")
        print(f"  {'銘柄':<12}", end="")
        for min_fwd in [1, 2, 3, 5, 10, 15, 30]:
            print(f"  {min_fwd}分後", end="")
        print()
        print("  " + "-" * (12 + 8 * 7))

        for sym, name in NONFERROUS.items():
            if sym not in stocks_data:
                continue
            st_th = trading_hours(stocks_data[sym]).copy()
            st_th = st_th.resample('1min').last().dropna(subset=['close'])

            fwd_rets = {m: [] for m in [1, 2, 3, 5, 10, 15, 30]}

            for ev_t in events:
                for min_fwd in [1, 2, 3, 5, 10, 15, 30]:
                    t_end = ev_t + pd.Timedelta(minutes=min_fwd)
                    if ev_t in st_th.index and t_end in st_th.index:
                        ret = (st_th.loc[t_end, 'close'] / st_th.loc[ev_t, 'close'] - 1) * 100 * direction
                        fwd_rets[min_fwd].append(ret)

            print(f"  {name:<12}", end="")
            for min_fwd in [1, 2, 3, 5, 10, 15, 30]:
                vals = fwd_rets[min_fwd]
                if vals:
                    print(f"  {np.mean(vals):>+5.3f}", end="")
                else:
                    print(f"  {'N/A':>6}", end="")
            print()


# ──────────────────────────────────────────────────────────
# 5. 乖離戦略バックテスト
#    「日経が動いたのに非鉄株が遅れている → 非鉄株を買う」
# ──────────────────────────────────────────────────────────
def divergence_strategy_backtest(df_jni, stocks_data, jni_window=5, jni_min=0.3, hold_minutes=10):
    """
    日経が過去jni_window分でjni_min%以上上昇したが、
    非鉄株は同期間でほぼ動いていない（|ret| < 0.2%）→ 非鉄株を買う
    """
    print("\n" + "="*60)
    print(f"5. 乖離戦略バックテスト（日経{jni_window}分で±{jni_min}%超、非鉄株未追従）")
    print("="*60)

    POS = 10_000_000
    COST_BPS = 2  # 片道2bps

    jni_th = trading_hours(df_jni).copy()
    jni_th = jni_th.resample('1min').last().dropna(subset=['close'])
    jni_roll = jni_th['close'].pct_change(jni_window) * 100

    print(f"\n{'銘柄':<12}  {'N':>5}  {'勝率':>6}  {'PF':>6}  {'Sharpe':>7}  {'平均PnL':>9}  {'コスト後':>9}")
    print("-" * 68)

    for sym, name in NONFERROUS.items():
        if sym not in stocks_data:
            continue
        st_th = trading_hours(stocks_data[sym]).copy()
        st_th = st_th.resample('1min').last().dropna(subset=['close'])
        st_roll = st_th['close'].pct_change(jni_window) * 100

        combined = pd.DataFrame({'jni': jni_roll, 'st': st_roll}).dropna()

        trades = []
        last_trade_t = None

        for t in combined.index:
            if last_trade_t is not None and (t - last_trade_t).seconds < hold_minutes * 60:
                continue

            jni_r = combined.loc[t, 'jni']
            st_r = combined.loc[t, 'st']

            # 日経が大きく動いたが非鉄株が遅れている
            if jni_r > jni_min and abs(st_r) < 0.2:
                direction = 1
            elif jni_r < -jni_min and abs(st_r) < 0.2:
                direction = -1
            else:
                continue

            # hold_minutes後の非鉄株リターン
            t_exit = t + pd.Timedelta(minutes=hold_minutes)
            if t_exit not in st_th.index:
                continue

            entry = st_th.loc[t, 'close']
            exit_p = st_th.loc[t_exit, 'close']
            if entry <= 0:
                continue

            pnl_pct = (exit_p / entry - 1) * 100 * direction
            cost = COST_BPS * 0.0001 * 2 * 100  # 往復コスト（%換算）
            trades.append({'t': t, 'pnl': pnl_pct, 'pnl_net': pnl_pct - cost, 'direction': direction})
            last_trade_t = t

        if len(trades) < 5:
            print(f"{name:<12}  トレード数不足 ({len(trades)})")
            continue

        arr = np.array([t['pnl'] for t in trades])
        arr_net = np.array([t['pnl_net'] for t in trades])
        wr = (arr > 0).mean() * 100
        pf = arr[arr > 0].sum() / abs(arr[arr <= 0].sum()) if (arr <= 0).any() else float('inf')
        sharpe = arr.mean() / arr.std() * np.sqrt(252 * 6.5 * 60 / hold_minutes) if arr.std() > 0 else 0

        print(f"{name:<12}  {len(trades):>5}  {wr:>5.1f}%  {pf:>6.2f}  {sharpe:>7.2f}  {arr.mean():>+8.4f}%  {arr_net.mean():>+8.4f}%")


# ──────────────────────────────────────────────────────────
# メイン
# ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=== データロード ===")
    df_jni = load_data('JNIc1')
    print(f"  日経先物: {len(df_jni)} バー ({df_jni.index.min().date()} ～ {df_jni.index.max().date()})")

    stocks_data = {}
    for sym, name in NONFERROUS.items():
        try:
            stocks_data[sym] = load_data(sym)
            print(f"  {sym} ({name}): {len(stocks_data[sym])} バー")
        except Exception as e:
            print(f"  {sym}: ロード失敗 ({e})")

    # 分析実行
    intraday_leadlag_jni_vs_stocks(df_jni, stocks_data, max_lag=10)
    daily_return_beta(df_jni, stocks_data)
    nikkei_bin_vs_nonferrous(df_jni, stocks_data)
    intraday_followthrough_timeline(df_jni, stocks_data, jni_threshold=0.3)
    divergence_strategy_backtest(df_jni, stocks_data, jni_window=5, jni_min=0.3, hold_minutes=10)

    print("\n分析完了!")
