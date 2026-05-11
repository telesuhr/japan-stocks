"""
B-Improved: レジームフィルタ追加版

旧 B-Optimized は 5/11 で 107戦全敗 -40,700bps の大惨敗。
原因はダウントレンド日の「ナイフ落としを買い向かう失敗」だった。

改善: 当日の地合いを判定して、ダウントレンド日にエントリー停止する。

追加フィルタ:
  F1) GAP: 寄付ギャップ <= -200bps なら当日完全停止
  F2) MORN: シグナル時点で「寄付から-100bps以下」なら以降のエントリー停止
  F3) VOL_DYN: 動的99%ile (過去5営業日のローリング)

検証対象: SBG のみ, 2026-01-05 ~ 2026-05-11 (OOS含む)
"""
import sys
sys.path.insert(0, '/Users/Yusuke/claude-code/DataFetcher')
from src.ticks import _file_globs, _to_date, _norm5, _norm4

import duckdb
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
from pathlib import Path
import psycopg2
import warnings
warnings.filterwarnings('ignore')

COST_BPS = 4
CODE = '99840'
PG_CONFIG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}

CACHE = Path('sbg_signals_jan_to_may.parquet')


def fetch_sbg_ticks(start, end):
    """指定期間のSBGティック"""
    s = _to_date(start)
    e = _to_date(end)
    root = Path.home() / "Data" / "jquants_trades" / "equities" / "trades"
    files = _file_globs(root, s, e)
    files_arr = "[" + ", ".join(f"'{f}'" for f in files) + "]"
    csv_part = (
        f"read_csv({files_arr}, header=true, "
        "columns={'Date':'DATE','Code':'VARCHAR','Time':'VARCHAR',"
        "'SessionDistinction':'VARCHAR','Price':'DOUBLE','TradingVolume':'BIGINT',"
        "'TransactionId':'VARCHAR'})"
    )
    sql = f"""
        SELECT Date AS d, Time AS tm, Price AS price, TradingVolume AS vol,
               date_add(CAST(Date AS TIMESTAMP),
                        INTERVAL (
                            EXTRACT(HOUR FROM CAST(Time AS TIME)) * 3600
                          + EXTRACT(MINUTE FROM CAST(Time AS TIME)) * 60
                          + EXTRACT(SECOND FROM CAST(Time AS TIME))
                        ) SECOND) AS ts_sec,
               EXTRACT(MICROSECOND FROM CAST(Time AS TIME)) AS us
        FROM {csv_part}
        WHERE Date BETWEEN ? AND ? AND Code IN (?, ?)
        ORDER BY ts_sec
    """
    con = duckdb.connect()
    con.execute("PRAGMA threads=4")
    df = con.execute(sql, [s, e, _norm5(CODE), _norm4(CODE)]).df()
    con.close()
    return df


def get_daily_open_close():
    """SBG の日次open/closeを取得 (gap_ret計算用)"""
    conn = psycopg2.connect(**PG_CONFIG)
    df = pd.read_sql("""
        SELECT date, adj_open AS open, adj_close AS close
        FROM stocks_daily WHERE code = '99840' AND date >= '2025-12-01'
        ORDER BY date
    """, conn)
    conn.close()
    df['date'] = pd.to_datetime(df['date']).dt.date
    df['prev_close'] = df['close'].shift(1)
    df['gap_ret'] = (df['open'] / df['prev_close'] - 1) * 10000
    return df


def build_signals(ticks: pd.DataFrame, vol_threshold_static=8200, vol_threshold_dynamic=None):
    """大口買いシグナル抽出 + 各種特徴量計算"""
    ticks = ticks.sort_values('ts_sec').reset_index(drop=True)
    ticks['prev_price'] = ticks['price'].shift(1)
    ticks['uptick'] = ticks['price'] > ticks['prev_price']
    ticks['date'] = pd.to_datetime(ticks['d']).dt.date

    # 取引時間内
    hh = ticks['ts_sec'].dt.hour
    mm = ticks['ts_sec'].dt.minute
    ticks['in_morning_window'] = (hh == 9) | ((hh == 10) & (mm < 30))

    # 動的閾値 (ローリング5日 99%ile)
    if vol_threshold_dynamic is not None:
        ticks['vol_thresh'] = ticks['date'].map(vol_threshold_dynamic)
        sig = ticks[
            (ticks['vol'] >= ticks['vol_thresh']) &
            (ticks['uptick']) &
            (ticks['in_morning_window'])
        ].copy().reset_index(drop=True)
    else:
        sig = ticks[
            (ticks['vol'] >= vol_threshold_static) &
            (ticks['uptick']) &
            (ticks['in_morning_window'])
        ].copy().reset_index(drop=True)

    return sig, ticks


def compute_features_and_returns(sig, ticks, daily_oc):
    """各シグナルに以下を付与:
       - gap_ret: その日の寄付ギャップ
       - day_open: その日の寄付値
       - cum_ret_at_signal: 寄付からシグナル時点の累積騰落 (bps)
       - fwd_60min: 60分後のリターン (bps, +1秒レイテンシ前提)
    """
    # 全ティックの精密タイムスタンプ
    ts_sec_int = ticks['ts_sec'].astype('datetime64[s]').astype('int64').values
    ts_us_int = ticks['us'].astype('int64').values
    ts_full_us = ts_sec_int * 1_000_000 + ts_us_int
    prices = ticks['price'].values.astype(float)
    dates = ticks['date'].values

    # daily open のlookup
    daily_oc['day_str'] = daily_oc['date'].astype(str)
    daily_open = daily_oc.set_index('date')['open'].to_dict()
    gap_map = daily_oc.set_index('date')['gap_ret'].to_dict()

    sig['day_open'] = sig['date'].map(daily_open)
    sig['gap_ret'] = sig['date'].map(gap_map)
    sig['cum_ret_at_signal'] = (sig['price'] / sig['day_open'] - 1) * 10000

    sig['ts_us'] = sig['ts_sec'].astype('datetime64[s]').astype('int64') * 1_000_000 + sig['us'].astype('int64')

    fwd_60 = np.full(len(sig), np.nan)
    for i in range(len(sig)):
        t_entry = sig['ts_us'].iloc[i] + 1_000_000  # +1sec latency
        j_entry = np.searchsorted(ts_full_us, t_entry)
        if j_entry >= len(prices):
            continue
        sig_date = sig['date'].iloc[i]
        if dates[j_entry] != sig_date:
            continue
        entry_price = prices[j_entry]

        t_exit = ts_full_us[j_entry] + 60 * 60 * 1_000_000
        j_exit = np.searchsorted(ts_full_us, t_exit)
        if j_exit >= len(prices):
            # 同日最後のtickで決済
            last_same = j_entry
            while last_same + 1 < len(prices) and dates[last_same + 1] == sig_date:
                last_same += 1
            if last_same <= j_entry:
                continue
            exit_price = prices[last_same]
        else:
            if dates[j_exit] != sig_date:
                # 同日内最終
                last_same = j_entry
                while last_same + 1 < len(prices) and dates[last_same + 1] == sig_date:
                    last_same += 1
                if last_same <= j_entry:
                    continue
                exit_price = prices[last_same]
            else:
                exit_price = prices[j_exit]
        fwd_60[i] = (exit_price / entry_price - 1) * 10000

    sig['fwd_60min'] = fwd_60
    return sig


def stats_summary(arr, label='', cost=COST_BPS):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n < 30:
        return dict(label=label, N=n, mean_raw=np.nan, mean_net=np.nan,
                    t_stat=np.nan, win_rate=np.nan, sharpe=np.nan, total_net=np.nan)
    net = arr - cost
    t, p = stats.ttest_1samp(arr, 0)
    return dict(label=label, N=n,
                mean_raw=round(arr.mean(), 2), mean_net=round(net.mean(), 2),
                std=round(arr.std(), 2), t_stat=round(t, 2), p_val=round(p, 4),
                win_rate=round((arr > 0).mean() * 100, 1),
                sharpe=round(net.mean() / arr.std() * np.sqrt(252 * 60) if arr.std() > 0 else 0, 2),
                total_net=round(net.sum(), 1))


def main():
    import time
    t0 = time.time()

    # ---- データロード or 再構築 ----
    if CACHE.exists():
        print(f"キャッシュからロード: {CACHE}")
        sig = pd.read_parquet(CACHE)
    else:
        print("SBGティック取得 (2026-01-01 ~ 2026-05-11)...")
        ticks = fetch_sbg_ticks('2026-01-01', '2026-05-11')
        print(f"  {len(ticks):,} ticks")

        print("\n動的閾値 (5日ローリング 99%ile)計算...")
        ticks_day = ticks.copy()
        ticks_day['date'] = pd.to_datetime(ticks_day['d']).dt.date
        # 各日の99%ile出来高
        daily_p99 = ticks_day.groupby('date')['vol'].quantile(0.99)
        # 5日ローリング (前5営業日の平均)
        rolling_p99 = daily_p99.rolling(5, min_periods=1).mean()
        dynamic_thresh = rolling_p99.shift(1).fillna(8200).to_dict()  # その日のthreshはshift(1)で当日を含まない

        print("シグナル抽出 (静的閾値 8,200株)...")
        sig_static, ticks = build_signals(ticks, vol_threshold_static=8200)
        print(f"  静的閾値シグナル: {len(sig_static):,}件")

        print("日次open/close取得...")
        daily_oc = get_daily_open_close()
        # 5/11 の open を tick から (DBに未投入なので)
        first_tick_511 = ticks[ticks['date'] == pd.Timestamp('2026-05-11').date()].iloc[0]
        if pd.isna(daily_oc[daily_oc['date'] == pd.Timestamp('2026-05-11').date()]['open'].iloc[0] if (daily_oc['date'] == pd.Timestamp('2026-05-11').date()).any() else np.nan):
            pass
        # 5/11はstocks_dailyに未投入 → tickの寄付値で補完
        if not (daily_oc['date'] == pd.Timestamp('2026-05-11').date()).any():
            prev_close = daily_oc.iloc[-1]['close']  # 5/8 close
            day511 = pd.Timestamp('2026-05-11').date()
            new_row = pd.DataFrame([{
                'date': day511,
                'open': float(first_tick_511['price']),
                'close': np.nan,
                'prev_close': prev_close,
                'gap_ret': (float(first_tick_511['price']) / prev_close - 1) * 10000
            }])
            daily_oc = pd.concat([daily_oc, new_row], ignore_index=True)

        print(f"\n5/11 の地合い: 寄付={daily_oc[daily_oc['date']==pd.Timestamp('2026-05-11').date()]['open'].iloc[0]:.0f}円, "
              f"5/8 引け={daily_oc[daily_oc['date']==pd.Timestamp('2026-05-08').date()]['close'].iloc[0]:.0f}円, "
              f"ギャップ={daily_oc[daily_oc['date']==pd.Timestamp('2026-05-11').date()]['gap_ret'].iloc[0]:.0f}bps")

        print("\n特徴量＋60min リターン計算 (1秒レイテンシ)...")
        sig = compute_features_and_returns(sig_static, ticks, daily_oc)

        # 動的閾値判定もマーク
        sig['vol_thresh_dynamic'] = sig['date'].map(dynamic_thresh)
        sig['passes_dynamic'] = sig['vol'] >= sig['vol_thresh_dynamic']

        sig.to_parquet(CACHE)
        print(f"  キャッシュ保存: {CACHE}")

    print(f"\n総シグナル数: {len(sig):,}")
    print(f"期間: {sig['date'].min()} 〜 {sig['date'].max()}")
    print(f"営業日: {sig['date'].nunique()}日")

    # ---- 期間別の振り分け ----
    sig['date_dt'] = pd.to_datetime(sig['date'])
    sig['is_511'] = sig['date_dt'] == pd.Timestamp('2026-05-11')
    sig['is_oos'] = sig['date_dt'] >= pd.Timestamp('2026-05-01')

    print(f"\n  Train (1-4月): {(~sig['is_oos']).sum():,}件")
    print(f"  OOS (5月)   : {sig['is_oos'].sum():,}件 (うち5/11: {sig['is_511'].sum():,}件)")

    # =======================================================
    # フィルタなし vs フィルタあり 比較
    # =======================================================
    print("\n" + "="*70)
    print("【1】フィルタ単独効果 (全期間)")
    print("="*70)

    valid = sig.dropna(subset=['fwd_60min']).copy()
    results = []

    # Baseline
    results.append(stats_summary(valid['fwd_60min'].values, 'Baseline (フィルタなし)'))

    # F1: ギャップフィルタ
    f1_pass = valid[valid['gap_ret'] > -200]
    results.append(stats_summary(f1_pass['fwd_60min'].values, 'F1: gap > -200bps'))

    f1_strict = valid[valid['gap_ret'] > -100]
    results.append(stats_summary(f1_strict['fwd_60min'].values, 'F1strict: gap > -100bps'))

    # F2: 朝の累積騰落フィルタ
    f2_pass = valid[valid['cum_ret_at_signal'] > -100]
    results.append(stats_summary(f2_pass['fwd_60min'].values, 'F2: cum_ret > -100bps'))

    f2_strict = valid[valid['cum_ret_at_signal'] > -50]
    results.append(stats_summary(f2_strict['fwd_60min'].values, 'F2strict: cum_ret > -50bps'))

    # F3: 動的閾値
    if 'passes_dynamic' in valid.columns:
        f3_pass = valid[valid['passes_dynamic']]
        results.append(stats_summary(f3_pass['fwd_60min'].values, 'F3: 動的閾値 (5d rolling 99%ile)'))

    # 組み合わせ
    f12 = valid[(valid['gap_ret'] > -200) & (valid['cum_ret_at_signal'] > -100)]
    results.append(stats_summary(f12['fwd_60min'].values, 'F1+F2'))

    f12_strict = valid[(valid['gap_ret'] > -100) & (valid['cum_ret_at_signal'] > -50)]
    results.append(stats_summary(f12_strict['fwd_60min'].values, 'F1strict+F2strict'))

    res_df = pd.DataFrame(results)
    print(res_df[['label', 'N', 'mean_raw', 'mean_net', 't_stat', 'win_rate', 'sharpe', 'total_net']].to_string(index=False))

    # =======================================================
    # 期間別 (Train vs 5/11)
    # =======================================================
    print("\n" + "="*70)
    print("【2】期間別 (Train 1-4月 vs OOS 5/11)")
    print("="*70)

    period_results = []
    for label_period, subset_mask in [
        ('Train (1-4月) Baseline', ~valid['is_oos']),
        ('Train + F1+F2 (gap>-200, cum>-100)',
         (~valid['is_oos']) & (valid['gap_ret'] > -200) & (valid['cum_ret_at_signal'] > -100)),
        ('5/11 Baseline', valid['is_511']),
        ('5/11 + F1 (gap>-200)', valid['is_511'] & (valid['gap_ret'] > -200)),
        ('5/11 + F1+F2', valid['is_511'] & (valid['gap_ret'] > -200) & (valid['cum_ret_at_signal'] > -100)),
        ('5/11 + F1strict+F2strict',
         valid['is_511'] & (valid['gap_ret'] > -100) & (valid['cum_ret_at_signal'] > -50)),
        ('全期間 (Train+5/11) F1strict+F2strict',
         (valid['gap_ret'] > -100) & (valid['cum_ret_at_signal'] > -50)),
    ]:
        sub = valid[subset_mask]
        r = stats_summary(sub['fwd_60min'].values, label_period)
        period_results.append(r)
    period_df = pd.DataFrame(period_results)
    print(period_df[['label', 'N', 'mean_raw', 'mean_net', 't_stat', 'win_rate', 'sharpe', 'total_net']].to_string(index=False))

    # =======================================================
    # 5/11 ギャップ確認
    # =======================================================
    print(f"\n5/11 のギャップ: {valid[valid['is_511']]['gap_ret'].iloc[0]:.0f} bps")
    print(f"  → F1 (gap>-200) の判定: {'通過' if valid[valid['is_511']]['gap_ret'].iloc[0] > -200 else '✗停止'}")
    print(f"  → F1strict (gap>-100) の判定: {'通過' if valid[valid['is_511']]['gap_ret'].iloc[0] > -100 else '✗停止'}")

    # =======================================================
    # CSV保存
    # =======================================================
    res_df.to_csv('filter_results.csv', index=False)
    period_df.to_csv('period_results.csv', index=False)

    # =======================================================
    # 図
    # =======================================================
    fig = plt.figure(figsize=(15, 9), facecolor='white')
    plt.rcParams.update({
        'font.family': ['Hiragino Sans', 'IPAexGothic', 'sans-serif'],
        'axes.unicode_minus': False,
    })
    fig.suptitle('B-Improved: レジームフィルタ追加版 (SBG 2026-01~05/11 OOS含む)',
                 fontsize=13, fontweight='bold', y=0.99)

    # ---- 上: フィルタ別 Sharpe比較 ----
    ax1 = fig.add_axes([0.05, 0.58, 0.55, 0.34])
    plot_df = res_df.dropna(subset=['sharpe'])
    xs = range(len(plot_df))
    cols = ['#9E9E9E' if 'Baseline' in str(l) else
            ('#43A047' if v > 5 else '#FF9800' if v > 0 else '#E53935')
            for l, v in zip(plot_df['label'], plot_df['sharpe'])]
    ax1.bar(xs, plot_df['sharpe'].values, color=cols, alpha=0.85)
    for i, (_, row) in enumerate(plot_df.iterrows()):
        ax1.text(i, row['sharpe'] + (0.3 if row['sharpe']>=0 else -0.3),
                 f"Sh={row['sharpe']:.1f}\nN={row['N']:,}",
                 ha='center', fontsize=7.5,
                 va='bottom' if row['sharpe']>=0 else 'top')
    ax1.set_xticks(xs)
    ax1.set_xticklabels(plot_df['label'], rotation=20, ha='right', fontsize=7.5)
    ax1.axhline(0, color='black', lw=0.7)
    ax1.set_ylabel('Sharpe (年率)', fontsize=9)
    ax1.set_title('フィルタ単独効果 (全期間)', fontsize=10, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # ---- 上右: 5/11だけの効果 ----
    ax2 = fig.add_axes([0.65, 0.58, 0.32, 0.34])
    ax2.axis('off')
    df_511 = period_df[period_df['label'].str.contains('5/11')].copy()
    if len(df_511) > 0:
        df_511 = df_511[['label','N','mean_net','total_net']].copy()
        df_511['label'] = df_511['label'].str.replace('5/11', '').str.strip()
        df_511.columns = ['設定', 'N', 'net(bps)', '合計net']
        df_511 = df_511.fillna(0)
        for c in ['N']:
            df_511[c] = df_511[c].astype(int)
        for c in ['net(bps)', '合計net']:
            df_511[c] = df_511[c].round(1)

        table = ax2.table(cellText=df_511.values, colLabels=df_511.columns,
                          cellLoc='center', loc='upper center',
                          bbox=[0, 0.2, 1, 0.7])
        table.auto_set_font_size(False)
        table.set_fontsize(9.5)
        for (r, c), cell in table.get_celld().items():
            if r == 0:
                cell.set_facecolor('#1565C0')
                cell.set_text_props(color='white', fontweight='bold')
            elif r > 0 and c >= 2 and r-1 < len(df_511):
                v = df_511.iloc[r-1, c]
                try:
                    if float(v) > 0:
                        cell.set_facecolor('#C8E6C9')
                    elif float(v) < 0:
                        cell.set_facecolor('#FFCDD2')
                except:
                    pass
            cell.set_edgecolor('#BDBDBD')
        ax2.set_title('5/11 の救済効果', fontsize=10, fontweight='bold', y=0.96)

    # ---- 下: エクイティカーブ (Baseline vs Improved) ----
    ax3 = fig.add_axes([0.05, 0.07, 0.92, 0.42])
    daily_pnl_baseline = valid.groupby('date_dt').apply(lambda g: (g['fwd_60min'] - COST_BPS).sum())
    cum_baseline = daily_pnl_baseline.cumsum()

    improved_mask = (valid['gap_ret'] > -100) & (valid['cum_ret_at_signal'] > -50)
    valid_impr = valid[improved_mask].copy()
    daily_pnl_impr = valid_impr.groupby('date_dt').apply(lambda g: (g['fwd_60min'] - COST_BPS).sum())
    cum_impr = daily_pnl_impr.cumsum()

    ax3.plot(cum_baseline.index, cum_baseline.values, color='#E53935', lw=1.5,
             label=f'Baseline (フィルタなし) 最終={cum_baseline.iloc[-1]:.0f}bps')
    ax3.plot(cum_impr.index, cum_impr.values, color='#43A047', lw=1.5,
             label=f'B-Improved (F1+F2 strict) 最終={cum_impr.iloc[-1]:.0f}bps')
    ax3.axvline(pd.Timestamp('2026-05-11'), color='red', linestyle='--', lw=1, alpha=0.6, label='5/11 暴落日')
    ax3.axhline(0, color='black', lw=0.7)
    ax3.set_xlabel('日付', fontsize=9)
    ax3.set_ylabel('累積 net PnL (bps)', fontsize=9)
    ax3.set_title('エクイティカーブ Baseline vs Improved (OOS 5/11 含む)', fontsize=10, fontweight='bold')
    ax3.legend(loc='best', fontsize=10)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax3.grid(alpha=0.3)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    fig.text(0.99, 0.005,
             'データ: 2026-01-05 〜 2026-05-11 / SBG (99840) JQuantsティック | 1秒レイテンシ前提 | コスト4bps',
             ha='right', va='bottom', fontsize=7, color='gray')
    plt.savefig('result.png', dpi=100, bbox_inches='tight', facecolor='white')
    print(f"\nresult.png 保存完了 ({time.time()-t0:.1f}秒)")


if __name__ == '__main__':
    main()
