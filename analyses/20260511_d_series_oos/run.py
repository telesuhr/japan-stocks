"""
D系シグナル (新発見) の 5/11 OOS 検証

問: B-Optimized が 5/11 で 107戦全敗 -42,000bps の大敗を喫した中、
   D系 (大口クラスター・モメンタム継続) は独立して機能するのか?

検証戦略:
  D2: 1分内に大口プリント >= 5 件 → 30分後ロング
  D3: 直近5バーで累積大口プリント >= 10件 → 30分後ロング
  I1: 直近15分で+50bps上昇 → 30分後継続ロング (モメンタム)
  E2: 価格下落 × 出来高小 → 15分後反転ロング

期間: 2026-05-11 (OOS, B-Optimizedが完敗した日)
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
START = '2026-05-11'
END = '2026-05-11'


def build_511_bars():
    """5/11 のSBGティック → 1分バー + リッチ特徴量"""
    s = _to_date(START); e = _to_date(END)
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
                        ) SECOND) AS ts_sec
        FROM {csv_part}
        WHERE Date BETWEEN ? AND ? AND Code IN (?, ?)
        ORDER BY ts_sec
    """
    con = duckdb.connect()
    print("ティック取得...")
    ticks = con.execute(sql, [s, e, _norm5(CODE), _norm4(CODE)]).df()
    con.close()
    print(f"  {len(ticks):,} ticks")

    ticks = ticks.sort_values('ts_sec').reset_index(drop=True)
    ticks['diff'] = ticks['price'] - ticks['price'].shift(1)
    ticks['is_up'] = (ticks['diff'] > 0).astype(int)
    ticks['is_down'] = (ticks['diff'] < 0).astype(int)
    ticks['up_vol'] = ticks['vol'] * ticks['is_up']
    ticks['down_vol'] = ticks['vol'] * ticks['is_down']
    ticks['is_large'] = (ticks['vol'] >= 8200).astype(int)
    ticks['bar_ts'] = ticks['ts_sec'].dt.floor('1min')

    print("1分バー集計...")
    bars = ticks.groupby('bar_ts').agg(
        open=('price', 'first'),
        high=('price', 'max'),
        low=('price', 'min'),
        close=('price', 'last'),
        volume=('vol', 'sum'),
        tick_count=('vol', 'size'),
        up_count=('is_up', 'sum'),
        down_count=('is_down', 'sum'),
        up_volume=('up_vol', 'sum'),
        down_volume=('down_vol', 'sum'),
        large_count=('is_large', 'sum'),
        max_single_vol=('vol', 'max'),
    ).reset_index()

    # 取引時間内のみ
    h = bars['bar_ts'].dt.hour
    m = bars['bar_ts'].dt.minute
    in_session = (
        ((h == 9) | (h == 10) | ((h == 11) & (m <= 29))) |
        (((h == 12) & (m >= 30)) | (h == 13) | ((h == 14)) | ((h == 15) & (m == 0)))
    )
    bars = bars[in_session].copy().reset_index(drop=True)
    print(f"  {len(bars):,} bars")

    # 特徴量・ターゲット
    bars['ofi'] = (bars['up_volume'] - bars['down_volume']) / (bars['volume'] + 1)
    bars['avg_trade_size'] = bars['volume'] / (bars['tick_count'] + 1)
    bars['cum_large_5b'] = bars['large_count'].shift(1).rolling(5, min_periods=2).sum()
    for col in ['tick_count', 'volume', 'large_count']:
        ma = bars[col].shift(1).rolling(30, min_periods=10).mean()
        bars[f'{col}_ratio'] = bars[col] / (ma + 1e-6)
    bars['bar_ret'] = (bars['close'] / bars['open'] - 1) * 10000
    bars['cum_ret_5b'] = (bars['close'] / bars['close'].shift(5) - 1) * 10000
    bars['cum_ret_15b'] = (bars['close'] / bars['close'].shift(15) - 1) * 10000
    bars['minute_of_day'] = bars['bar_ts'].dt.hour * 60 + bars['bar_ts'].dt.minute

    # フォワードリターン
    for k, label in [(5, 'fwd_5min'), (15, 'fwd_15min'),
                     (30, 'fwd_30min'), (60, 'fwd_60min')]:
        bars[label] = (bars['close'].shift(-k) / bars['close'] - 1) * 10000

    return bars


def test_signal(bars, cond, target, label, direction=1):
    sub = bars[cond & bars[target].notna()]
    if len(sub) == 0:
        return dict(label=label, target=target, N=0,
                    mean_raw=np.nan, mean_net=np.nan, t_stat=np.nan,
                    win_rate=np.nan, total_net=np.nan, best=np.nan, worst=np.nan)
    arr = sub[target].values * direction
    net = arr - COST_BPS
    n = len(arr)
    if n >= 2:
        t, p = stats.ttest_1samp(arr, 0)
    else:
        t, p = np.nan, np.nan
    return dict(label=label, target=target, N=n,
                mean_raw=round(arr.mean(), 2),
                mean_net=round(net.mean(), 2),
                t_stat=round(t, 2) if pd.notna(t) else np.nan,
                p_val=round(p, 4) if pd.notna(p) else np.nan,
                win_rate=round((arr > 0).mean() * 100, 1),
                total_net=round(net.sum(), 1),
                best=round(arr.max(), 1),
                worst=round(arr.min(), 1),
                sharpe=round(net.mean()/arr.std() * np.sqrt(252*60), 2) if (n > 1 and arr.std() > 0) else np.nan)


def main():
    import time
    t0 = time.time()

    print("=" * 70)
    print("D系シグナル 5/11 OOS 検証")
    print("=" * 70)

    bars = build_511_bars()
    print(f"\n5/11 SBG: {len(bars)}分バー")
    print(f"  始値: {bars.iloc[0]['open']:.0f}円")
    print(f"  終値: {bars.iloc[-1]['close']:.0f}円")
    print(f"  日中: {(bars.iloc[-1]['close'] / bars.iloc[0]['open'] - 1) * 100:.2f}%")

    print("\n各シグナル検証:")
    results = []

    # D2: 1分内 大口プリント >= 5件 → 30分後ロング
    cond = bars['large_count'] >= 5
    r = test_signal(bars, cond, 'fwd_30min', 'D2: large_count≥5 → 30min')
    results.append(r); print(f"  D2: N={r['N']}, net={r['mean_net']}, WR={r['win_rate']}%")

    # D3: 累積大口 >= 10 → 30分後ロング
    cond = bars['cum_large_5b'] >= 10
    r = test_signal(bars, cond, 'fwd_30min', 'D3: cum_large_5b≥10 → 30min')
    results.append(r); print(f"  D3: N={r['N']}, net={r['mean_net']}, WR={r['win_rate']}%")

    # I1: 直近15分で+50bps → 30分後継続買い
    cond = bars['cum_ret_15b'] >= 50
    r = test_signal(bars, cond, 'fwd_30min', 'I1: cum_ret_15b≥50 → 30min')
    results.append(r); print(f"  I1: N={r['N']}, net={r['mean_net']}, WR={r['win_rate']}%")

    # E2: 価格下落 × 出来高小 → 15分後反転買い
    cond = (bars['bar_ret'] <= -30) & (bars['volume_ratio'] <= 0.7)
    r = test_signal(bars, cond, 'fwd_15min', 'E2: PriceDown × VolLow → 15min')
    results.append(r); print(f"  E2: N={r['N']}, net={r['mean_net']}, WR={r['win_rate']}%")

    # 比較: B-Optimized (再確認)
    # 大口買い uptick → 60分後 (1分bar levelで近似)
    # 厳密にはティックレベルだが、ここでは「1分中に大口プリントあり かつ そのバーがup」で近似
    cond = (bars['large_count'] >= 1) & (bars['bar_ret'] > 0)
    in_morn = (bars['minute_of_day'] >= 540) & (bars['minute_of_day'] < 630)
    cond = cond & in_morn
    r = test_signal(bars, cond, 'fwd_60min', 'B-Opt近似 (large≥1 × up × 9:00-10:30) → 60min')
    results.append(r); print(f"  B-Opt近似: N={r['N']}, net={r['mean_net']}, WR={r['win_rate']}%")

    # ===== 逆方向のテスト: ショート系 =====
    print("\nショート系 (ダウントレンド下の挙動):")

    # I2: 直近15分で-50bps以下 → 30分後ショート継続
    cond = bars['cum_ret_15b'] <= -50
    r = test_signal(bars, cond, 'fwd_30min', 'I2: cum_ret_15b≤-50 → 30min Short', direction=-1)
    results.append(r); print(f"  I2 (Short): N={r['N']}, net={r['mean_net']}, WR={r['win_rate']}%")

    # 大口"売り" (down tick + large) → 30min Short
    cond_down = (bars['large_count'] >= 1) & (bars['bar_ret'] < 0) & in_morn
    r = test_signal(bars, cond_down, 'fwd_30min', '大口売り (近似) → 30min Short', direction=-1)
    results.append(r); print(f"  大口売り近似: N={r['N']}, net={r['mean_net']}, WR={r['win_rate']}%")

    # 一斉売り (down ticks ≥ 5) → 30min Short
    # downtick の大口がある = 売り浴びせ
    cond_panic = (bars['large_count'] >= 5) & (bars['bar_ret'] < 0)
    r = test_signal(bars, cond_panic, 'fwd_30min', 'パニック売り (large≥5 × down) → 30min Short', direction=-1)
    results.append(r); print(f"  パニック売り: N={r['N']}, net={r['mean_net']}, WR={r['win_rate']}%")

    res_df = pd.DataFrame(results)
    print("\n=== 全結果 ===")
    print(res_df[['label', 'N', 'mean_raw', 'mean_net', 't_stat', 'win_rate', 'total_net']].to_string(index=False))

    # =============================
    # 学習期との比較
    # =============================
    train_summary = {
        'D2: large_count≥5 → 30min': {'net':7.93, 't':5.55, 'wr':52.2, 'Sh':8.19},
        'D3: cum_large_5b≥10 → 30min': {'net':4.93, 't':8.09, 'wr':51.0, 'Sh':5.83},
        'I1: cum_ret_15b≥50 → 30min': {'net':4.66, 't':4.80, 'wr':52.0, 'Sh':5.57},
        'E2: PriceDown × VolLow → 15min': {'net':28.89, 't':4.03, 'wr':61.9, 'Sh':38.98},
    }
    print("\n=== 学習期 (1-4月) との比較 ===")
    print(f"{'戦略':<35} {'学習期net':>10} {'5/11 net':>10} {'差分':>10}")
    for r in results:
        if r['label'] in train_summary:
            train = train_summary[r['label']]
            diff = r['mean_net'] - train['net']
            print(f"{r['label']:<35} {train['net']:>+10.2f} {r['mean_net']:>+10.2f} {diff:>+10.2f}")

    res_df.to_csv('results.csv', index=False)

    # =============================
    # 図
    # =============================
    fig = plt.figure(figsize=(15, 9), facecolor='white')
    plt.rcParams.update({
        'font.family': ['Hiragino Sans', 'IPAexGothic', 'sans-serif'],
        'axes.unicode_minus': False,
    })
    fig.suptitle('D系シグナル 5/11 OOS 検証 (SBG 暴落日)',
                 fontsize=13, fontweight='bold', y=0.99)

    # ---- 上: 価格チャート + シグナル ----
    ax1 = fig.add_axes([0.05, 0.55, 0.92, 0.38])
    ax1.plot(bars['bar_ts'], bars['close'], color='#666', lw=1, alpha=0.7)

    # D2シグナル点
    d2_sig = bars[bars['large_count'] >= 5]
    if len(d2_sig) > 0:
        ax1.scatter(d2_sig['bar_ts'], d2_sig['close'], c='#FF9800', s=60,
                    marker='^', edgecolor='black', lw=0.5, zorder=5,
                    label=f'D2: 大口5件以上 (N={len(d2_sig)})')

    # D3シグナル点
    d3_sig = bars[bars['cum_large_5b'] >= 10]
    if len(d3_sig) > 0:
        ax1.scatter(d3_sig['bar_ts'], d3_sig['close'], c='#9C27B0', s=30,
                    marker='o', alpha=0.7, edgecolor='none', zorder=4,
                    label=f'D3: 累積大口10件以上 (N={len(d3_sig)})')

    # I1シグナル点
    i1_sig = bars[bars['cum_ret_15b'] >= 50]
    if len(i1_sig) > 0:
        ax1.scatter(i1_sig['bar_ts'], i1_sig['close'], c='#43A047', s=80,
                    marker='*', edgecolor='black', lw=0.5, zorder=6,
                    label=f'I1: 15分+50bps以上 (N={len(i1_sig)})')

    # 寄付時間帯
    today = pd.Timestamp('2026-05-11')
    ax1.axvspan(today + pd.Timedelta('9h'), today + pd.Timedelta('10h30m'),
                alpha=0.08, color='blue', label='B-Opt時間帯')

    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax1.set_ylabel('株価', fontsize=10)
    ax1.set_title(f'SBG 5/11 ティック由来シグナル位置  (-7.65%急落日)',
                  fontsize=10, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # ---- 下左: 結果バー ----
    ax2 = fig.add_axes([0.05, 0.07, 0.40, 0.40])
    plot_res = res_df.copy()
    plot_res['short_label'] = plot_res['label'].str.split(':').str[0]
    xs = range(len(plot_res))
    cols = ['#43A047' if v >= 0 else '#E53935' for v in plot_res['mean_net'].fillna(0)]
    ax2.bar(xs, plot_res['mean_net'].fillna(0), color=cols, alpha=0.85)
    ax2.set_xticks(xs)
    ax2.set_xticklabels(plot_res['short_label'], rotation=20, ha='right', fontsize=8)
    for i, (_, r) in enumerate(plot_res.iterrows()):
        if pd.notna(r['mean_net']):
            ax2.text(i, r['mean_net'] + (3 if r['mean_net'] >= 0 else -3),
                     f"N={r['N']}\n{r['win_rate']:.0f}%",
                     ha='center', fontsize=7.5,
                     va='bottom' if r['mean_net'] >= 0 else 'top')
    ax2.axhline(0, color='black', lw=0.7)
    ax2.set_ylabel('5/11 mean_net (bps)', fontsize=9)
    ax2.set_title('5/11 のシグナル別 net期待値', fontsize=10, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # ---- 下右: テーブル ----
    ax3 = fig.add_axes([0.50, 0.07, 0.47, 0.40])
    ax3.axis('off')

    tbl_data = []
    for r in results:
        if r['N'] > 0:
            tbl_data.append([r['label'][:25], r['N'],
                             f"{r['mean_net']:.1f}",
                             f"{r['t_stat']:.1f}" if pd.notna(r['t_stat']) else '-',
                             f"{r['win_rate']:.0f}%",
                             f"{r['total_net']:.0f}"])

    tbl_df = pd.DataFrame(tbl_data, columns=['戦略','N','net(bps)','t値','勝率','合計net'])
    table = ax3.table(cellText=tbl_df.values, colLabels=tbl_df.columns,
                      cellLoc='center', loc='upper center',
                      bbox=[0, 0.1, 1, 0.85])
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor('#1565C0')
            cell.set_text_props(color='white', fontweight='bold')
        elif r > 0 and c == 5 and r-1 < len(tbl_df):
            try:
                v = float(tbl_df.iloc[r-1, c])
                if v > 0:
                    cell.set_facecolor('#C8E6C9')
                elif v < 0:
                    cell.set_facecolor('#FFCDD2')
            except:
                pass
        cell.set_edgecolor('#BDBDBD')
    ax3.set_title('5/11 各シグナル詳細', fontsize=10, fontweight='bold', y=0.97)

    fig.text(0.99, 0.005,
             'データ: 2026-05-11 / SBG (99840) | 学習期間: 1-4月 | OOS=完全未学習 | コスト4bps',
             ha='right', va='bottom', fontsize=7, color='gray')
    plt.savefig('result.png', dpi=100, bbox_inches='tight', facecolor='white')
    print(f"\nresult.png 保存完了 ({time.time()-t0:.1f}秒)")


if __name__ == '__main__':
    main()
