"""
B-Optimized Out-of-Sample 検証: 2026-05-11 (本日)

検証方針:
  - これまでのバックテストは 2026-01-01 〜 2026-04-30 (4ヶ月)
  - 本日 5/11 は完全な Out-of-Sample (未学習データ)
  - B-Optimized 条件で SBG の大口買いを抽出し、60分後リターンを実測

条件:
  - 銘柄: 99840 (SBG)
  - 時間帯: 9:00-10:30
  - シグナル: vol >= 8,200株 (SBG の 99%ile) かつ uptick
  - 保有: 60分
  - コスト: 4bps
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
import warnings
warnings.filterwarnings('ignore')

COST_BPS = 4
CODE = '99840'
DATE = '2026-05-11'

# 学習期間で求めた99%ile (SBG): 8,200株
VOL_99 = 8200


def fetch_sbg_ticks_today():
    """5/11 のSBGティックを取得"""
    root = Path.home() / "Data" / "jquants_trades" / "equities" / "trades"
    live_file = root / "live" / "equities_trades_20260511.csv.gz"
    if not live_file.exists():
        raise FileNotFoundError(f"5/11 のティックファイルが見つかりません: {live_file}")

    csv_part = (
        f"read_csv(['{live_file}'], "
        "header=true, "
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
        WHERE Code IN (?, ?)
        ORDER BY tm
    """
    con = duckdb.connect()
    df = con.execute(sql, [_norm5(CODE), _norm4(CODE)]).df()
    con.close()
    return df


def main():
    import time
    t0 = time.time()
    print("=" * 65)
    print(f"B-Optimized OOS検証: {DATE}")
    print("=" * 65)

    # ---- ティック取得 ----
    print("\n[1] 5/11のSBGティック取得...")
    ticks = fetch_sbg_ticks_today()
    print(f"  ticks: {len(ticks):,}")
    print(f"  時間範囲: {ticks['tm'].iloc[0]} 〜 {ticks['tm'].iloc[-1]}")
    print(f"  価格範囲: {ticks['price'].min():.0f} 〜 {ticks['price'].max():.0f}")
    print(f"  始値: {ticks['price'].iloc[0]:.0f}, 終値: {ticks['price'].iloc[-1]:.0f}")
    print(f"  日中変化: {(ticks['price'].iloc[-1] / ticks['price'].iloc[0] - 1)*100:.2f}%")

    # ---- 大口買いシグナル抽出 ----
    print("\n[2] 大口買いシグナル抽出...")
    ticks = ticks.sort_values('ts_sec').reset_index(drop=True)
    ticks['prev_price'] = ticks['price'].shift(1)
    ticks['uptick'] = ticks['price'] > ticks['prev_price']

    # 9:00-10:30 限定
    hh = ticks['ts_sec'].dt.hour
    mm = ticks['ts_sec'].dt.minute
    in_window = (hh == 9) | ((hh == 10) & (mm < 30))

    sig = ticks[
        (ticks['vol'] >= VOL_99) &
        (ticks['uptick']) &
        in_window
    ].copy().reset_index(drop=True)
    print(f"  シグナル数: {len(sig):,}件")

    if len(sig) == 0:
        print("  シグナルなし。終了")
        return

    print(f"  最初のシグナル: {sig.iloc[0]['ts_sec']} @{sig.iloc[0]['price']:.0f} (vol={sig.iloc[0]['vol']})")
    print(f"  最後のシグナル: {sig.iloc[-1]['ts_sec']} @{sig.iloc[-1]['price']:.0f} (vol={sig.iloc[-1]['vol']})")

    # ---- 60分後の価格を計算 (実行可能性想定でレイテンシ別) ----
    print("\n[3] レイテンシ別バックテスト...")

    # 全ティックの精密タイムスタンプ
    ts_sec_int = ticks['ts_sec'].astype('datetime64[s]').astype('int64').values
    ts_us = ticks['us'].astype('int64').values
    ts_full_us = ts_sec_int * 1_000_000 + ts_us
    prices = ticks['price'].values.astype(float)

    sig['ts_us'] = sig['ts_sec'].astype('datetime64[s]').astype('int64') * 1_000_000 + sig['us'].astype('int64')

    rows = []
    for lat_sec in [0, 1, 10, 60]:
        lat_us = int(lat_sec * 1_000_000)
        ret_list = []
        details = []
        for i, srow in sig.iterrows():
            t_signal = int(srow['ts_us'])
            t_entry = t_signal + lat_us
            j_entry = np.searchsorted(ts_full_us, t_entry)
            if j_entry >= len(prices):
                continue
            entry_price = prices[j_entry]

            t_exit = ts_full_us[j_entry] + 60 * 60 * 1_000_000
            j_exit = np.searchsorted(ts_full_us, t_exit)
            if j_exit >= len(prices):
                # 60分後ない → 同日内最終ティック
                j_exit = len(prices) - 1
            exit_price = prices[j_exit]

            ret_bps = (exit_price / entry_price - 1) * 10000
            ret_list.append(ret_bps)
            if lat_sec == 1:
                details.append({
                    'signal_ts': srow['ts_sec'],
                    'signal_price': srow['price'],
                    'signal_vol': srow['vol'],
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'ret_bps': ret_bps,
                    'net_bps': ret_bps - COST_BPS,
                })

        arr = np.array(ret_list)
        net = arr - COST_BPS
        n = len(arr)
        if n > 0:
            wr = (arr > 0).mean() * 100
            total_pnl = net.sum()
            rows.append({
                'latency_sec': lat_sec,
                'N': n,
                'mean_raw_bps': round(arr.mean(), 2),
                'mean_net_bps': round(net.mean(), 2),
                'std_bps': round(arr.std(), 2) if n > 1 else 0,
                'total_net_bps': round(total_pnl, 1),
                'win_rate': round(wr, 1),
                'best': round(arr.max(), 1),
                'worst': round(arr.min(), 1),
            })

        if lat_sec == 1:
            detail_df = pd.DataFrame(details)
            detail_df.to_csv('trades_1s_latency.csv', index=False)

    results = pd.DataFrame(rows)
    print(results.to_string(index=False))

    # ---- 比較: 期待値 (4ヶ月学習) ----
    print("\n[4] 学習期間 (1〜4月) との比較")
    train_summary = {
        '0sec': {'N': 11655, 'mean_net': 34.50, 'win_rate': 52.3},
        '1sec': {'N': 11655, 'mean_net': 34.31, 'win_rate': 52.4},
        '10sec': {'N': 11655, 'mean_net': 32.93, 'win_rate': 52.2},
        '60sec': {'N': 11655, 'mean_net': 26.85, 'win_rate': 51.5},
    }
    print("レイテンシ | 学習期 net | 学習期 WR | 5/11 net | 5/11 WR | 判定")
    for r in rows:
        ls = r['latency_sec']
        key = f'{ls}sec'
        if key in train_summary:
            t_ = train_summary[key]
            judge = '◎' if r['mean_net_bps'] > t_['mean_net'] * 0.5 else ('○' if r['mean_net_bps'] > 0 else '✗')
            print(f"  {ls:>2}秒    | {t_['mean_net']:>7.1f}bps | {t_['win_rate']:>5.1f}% | "
                  f"{r['mean_net_bps']:>6.1f}bps | {r['win_rate']:>5.1f}% | {judge}")

    # ---- 全ティック+シグナルの価格チャート ----
    print("\n[5] チャート描画...")
    fig = plt.figure(figsize=(15, 9), facecolor='white')
    plt.rcParams.update({
        'font.family': ['Hiragino Sans', 'IPAexGothic', 'sans-serif'],
        'axes.unicode_minus': False,
    })
    fig.suptitle(f'B-Optimized Out-of-Sample 検証: SBG (99840) {DATE}',
                 fontsize=13, fontweight='bold', y=0.99)

    # 上: 価格チャート + シグナル
    ax1 = fig.add_axes([0.06, 0.55, 0.90, 0.38])
    # 取引時間内のみ表示
    in_session = ticks['ts_sec'].dt.time.between(pd.Timestamp('09:00').time(), pd.Timestamp('15:00').time())
    plot_ticks = ticks[in_session]
    ax1.plot(plot_ticks['ts_sec'], plot_ticks['price'], color='#666', lw=0.5, alpha=0.7)

    # シグナル点
    ax1.scatter(sig['ts_sec'], sig['price'], c='blue', s=60, marker='^',
                edgecolor='black', lw=0.5, zorder=5, label=f'大口買い ({len(sig)}件)')

    # 9:00-10:30 帯
    today = pd.Timestamp(DATE)
    ax1.axvspan(today + pd.Timedelta('9h'), today + pd.Timedelta('10h30m'),
                alpha=0.08, color='blue', label='シグナル時間帯')

    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax1.set_ylabel('株価 (円)', fontsize=10)
    ax1.set_title(f'SBG ティック価格 + 大口買いシグナル', fontsize=10, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # 下左: トレード詳細表 (1秒レイテンシ)
    ax2 = fig.add_axes([0.06, 0.06, 0.45, 0.42])
    ax2.axis('off')
    if len(detail_df) > 0:
        disp = detail_df.copy()
        disp['signal_ts'] = pd.to_datetime(disp['signal_ts']).dt.strftime('%H:%M:%S')
        disp = disp[['signal_ts', 'signal_price', 'signal_vol', 'entry_price', 'exit_price', 'net_bps']]
        disp.columns = ['時刻', 'シグナル価', '出来高', 'エントリー', 'エグジット', 'net(bps)']
        # 数値整形
        disp['シグナル価'] = disp['シグナル価'].astype(int)
        disp['エントリー'] = disp['エントリー'].astype(int)
        disp['エグジット'] = disp['エグジット'].astype(int)
        disp['net(bps)'] = disp['net(bps)'].round(1)

        # 上位N行のみ表示 (N=10)
        if len(disp) > 12:
            disp_show = pd.concat([disp.head(6), pd.DataFrame([['...'] * 6], columns=disp.columns), disp.tail(5)])
        else:
            disp_show = disp

        table = ax2.table(cellText=disp_show.values, colLabels=disp_show.columns,
                          cellLoc='center', loc='upper center',
                          bbox=[0, 0.05, 1, 0.9])
        table.auto_set_font_size(False)
        table.set_fontsize(8.5)
        for (r, c), cell in table.get_celld().items():
            if r == 0:
                cell.set_facecolor('#1565C0')
                cell.set_text_props(color='white', fontweight='bold')
            elif r-1 < len(disp_show):
                try:
                    v = disp_show.iloc[r-1]['net(bps)']
                    if v == '...' or pd.isna(v):
                        cell.set_facecolor('#EEEEEE')
                    elif float(v) > 0:
                        cell.set_facecolor('#C8E6C9')
                    else:
                        cell.set_facecolor('#FFCDD2')
                except:
                    pass
            cell.set_edgecolor('#BDBDBD')
        ax2.set_title(f'1秒レイテンシ時の各トレード詳細', fontsize=10, fontweight='bold', y=1.0)

    # 下右: レイテンシ別サマリー
    ax3 = fig.add_axes([0.55, 0.06, 0.42, 0.42])
    ax3.axis('off')
    summary_df = pd.DataFrame(rows)
    summary_df.columns = ['遅延(秒)', 'N', 'raw(bps)', 'net(bps)', 'std', '合計net', '勝率%', '最高', '最低']
    table = ax3.table(cellText=summary_df.values, colLabels=summary_df.columns,
                      cellLoc='center', loc='upper center',
                      bbox=[0, 0.55, 1, 0.40])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor('#1565C0')
            cell.set_text_props(color='white', fontweight='bold')
        elif r % 2 == 0:
            cell.set_facecolor('#E3F2FD')
        cell.set_edgecolor('#BDBDBD')
    ax3.set_title('レイテンシ別サマリー (5/11)', fontsize=10, fontweight='bold', y=1.0)

    # 結論メッセージ
    best = max(rows, key=lambda x: x['mean_net_bps']) if rows else None
    if best:
        verdict = '✓ ワーク' if best['mean_net_bps'] > 0 else '✗ 機能せず'
        color = '#43A047' if best['mean_net_bps'] > 0 else '#E53935'
        ax3.text(0.5, 0.40, f"判定: {verdict}", ha='center', fontsize=14, fontweight='bold',
                 color=color, transform=ax3.transAxes)
        ax3.text(0.5, 0.30,
                 f"最良: 遅延{best['latency_sec']}秒 → net {best['mean_net_bps']:+.1f}bps × {best['N']}回\n"
                 f"合計 = {best['total_net_bps']:+.0f} bps\n"
                 f"勝率 = {best['win_rate']:.1f}% (学習期 52.3%)",
                 ha='center', fontsize=10, transform=ax3.transAxes,
                 bbox=dict(boxstyle='round', facecolor='#FFF9C4', alpha=0.8))

    fig.text(0.99, 0.005,
             f'データ: 2026-05-11 / SBG (99840) JQuants ティック / 5/11はOUT-OF-SAMPLE | コスト4bps',
             ha='right', va='bottom', fontsize=7, color='gray')
    plt.savefig('result.png', dpi=100, bbox_inches='tight', facecolor='white')

    results.to_csv('latency_results.csv', index=False)
    print(f"\nresult.png 保存完了 ({time.time()-t0:.1f}秒)")


if __name__ == '__main__':
    main()
