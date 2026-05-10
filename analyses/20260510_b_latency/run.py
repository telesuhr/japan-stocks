"""
B-Optimized 執行レイテンシ感度分析

問題:
  バックテストはティック検出と同時に約定する前提だが、現実は遅延がある。
  detection → algo処理 → 発注 → 約定 の各段階で時間が経過し、その間に値が動く。

検証:
  各レイテンシで「シグナル後 N ミリ秒の最良価格」をエントリー価格とし、
  60分後の最良価格をエグジット価格として、実効リターンを計算。

レイテンシ水準:
  0ms     : 理論値 (現状のバックテスト)
  100ms   : プロ水準 (低レイテンシ・コロケーション)
  500ms   : 中速 (クラウド+良いAPI)
  1秒     : 普通のシステム
  3秒     : リテール証券のREST API
  10秒    : 手動判断
  30秒    : ゆっくり手動
  60秒    : 完全に手動

対象: SBG (99840) のみ, 9:00-10:30, 大口買い (uptick, vol>=99%ile)
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
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

COST_BPS = 4
ENRICHED = Path('/Users/Yusuke/claude-code/japan-stocks/.claude/worktrees/vibrant-mccarthy-d4c865/analyses/20260510_b_deep_dive/lp_enriched.parquet')
SBG_TICKS_CACHE = Path('sbg_ticks.parquet')

CODE = '99840'
START = '2026-01-01'
END   = '2026-04-30'

LATENCIES_SEC = [0, 0.1, 0.5, 1.0, 3.0, 10.0, 30.0, 60.0]


def fetch_sbg_ticks():
    """SBGの全ティックを取得 (DuckDB経由)"""
    s = _to_date(START)
    e = _to_date(END)
    root = Path.home() / "Data" / "jquants_trades" / "equities" / "trades"
    files = _file_globs(root, s, e)
    files_arr = "[" + ", ".join(f"'{f}'" for f in files) + "]"
    csv_part = (
        f"read_csv({files_arr}, "
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
               -- マイクロ秒部分も保持
               EXTRACT(MICROSECOND FROM CAST(Time AS TIME)) AS us
        FROM {csv_part}
        WHERE Date BETWEEN ? AND ? AND Code IN (?, ?)
        ORDER BY d, tm
    """
    con = duckdb.connect()
    con.execute("PRAGMA threads=4")
    df = con.execute(sql, [s, e, _norm5(CODE), _norm4(CODE)]).df()
    con.close()
    return df


def load_or_fetch_ticks():
    if SBG_TICKS_CACHE.exists():
        print(f"SBGティック キャッシュロード: {SBG_TICKS_CACHE}")
        return pd.read_parquet(SBG_TICKS_CACHE)
    print("SBGティック取得中...")
    import time
    t0 = time.time()
    df = fetch_sbg_ticks()
    print(f"  {len(df):,}ティック ({time.time()-t0:.1f}秒)")
    df.to_parquet(SBG_TICKS_CACHE)
    return df


def stats_summary(returns, label='', cost=COST_BPS):
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n < 30:
        return dict(label=label, N=n, mean_raw=np.nan, mean_net=np.nan,
                    t_stat=np.nan, win_rate=np.nan, sharpe=np.nan)
    net = arr - cost
    t, p = stats.ttest_1samp(arr, 0)
    return dict(label=label, N=n,
                mean_raw=round(arr.mean(), 2), mean_net=round(net.mean(), 2),
                std=round(arr.std(), 2), t_stat=round(t, 2), p_val=round(p, 4),
                win_rate=round((arr > 0).mean() * 100, 1),
                sharpe=round(net.mean() / arr.std() * np.sqrt(252 * 60) if arr.std() > 0 else 0, 2))


def main():
    import time
    t0 = time.time()

    # ---- B-Optimized 条件のシグナルだけ抽出 (SBG, 9:00-10:30, 大口買い) ----
    print("enriched data ロード...")
    lp = pd.read_parquet(ENRICHED)
    lp['ts'] = pd.to_datetime(lp['ts'])
    sig = lp[
        (lp['code'] == CODE) &
        (lp['direction'] == 1) &
        (((lp['ts'].dt.hour == 9)) |
         ((lp['ts'].dt.hour == 10) & (lp['ts'].dt.minute < 30)))
    ].copy()
    print(f"  B-Optimized SBGシグナル: {len(sig):,}件")
    print(f"  期間: {sig['ts'].min()} 〜 {sig['ts'].max()}")

    # ---- SBG全ティック ----
    ticks = load_or_fetch_ticks()
    print(f"  SBGティック: {len(ticks):,}件")

    # ティックを numpy 配列に (高速検索用)
    # ts は datetime64[us] になっているので秒整数 + microsecond で精度保持
    ticks = ticks.sort_values('ts_sec').reset_index(drop=True)
    # 全ティックの精密タイムスタンプ (秒×1e6 + us)
    ts_sec_int = ticks['ts_sec'].astype('datetime64[s]').astype('int64').values
    ts_us = ticks['us'].astype('int64').values
    ts_full_us = ts_sec_int.astype('int64') * 1_000_000 + ts_us  # マイクロ秒精度
    prices = ticks['price'].values.astype(float)
    dates = ticks['ts_sec'].dt.date.values

    print(f"\nレイテンシ別バックテスト...")

    # シグナルtsをマイクロ秒精度に変換
    sig['ts_us'] = sig['ts'].astype('datetime64[us]').astype('int64')
    sig_ts_us = sig['ts_us'].values
    sig_dates = sig['ts'].dt.date.values
    sig_prices_orig = sig['price'].values

    results = []
    for lat_sec in LATENCIES_SEC:
        lat_us = int(lat_sec * 1_000_000)

        rets = []
        for i in range(len(sig)):
            t_signal = sig_ts_us[i]
            d_signal = sig_dates[i]

            # エントリー: t_signal + latency 以後の最初のティック
            t_entry_target = t_signal + lat_us
            j_entry = np.searchsorted(ts_full_us, t_entry_target)
            if j_entry >= len(prices):
                continue
            if dates[j_entry] != d_signal:
                continue
            entry_price = prices[j_entry]

            # エグジット: エントリー時刻 + 60min 以後の最初のティック
            t_exit_target = ts_full_us[j_entry] + 60 * 60 * 1_000_000
            j_exit = np.searchsorted(ts_full_us, t_exit_target)
            if j_exit >= len(prices):
                continue
            if dates[j_exit] != d_signal:
                # 同日に60min後がない → スキップ (または当日終値?)
                # ここでは同日内最後のティックを使う (引け持ち越しなし)
                # 同日内の最大idxを探す
                last_same = j_entry
                while (last_same + 1 < len(prices) and
                       dates[last_same + 1] == d_signal):
                    last_same += 1
                if last_same <= j_entry:
                    continue
                exit_price = prices[last_same]
            else:
                exit_price = prices[j_exit]

            ret_bps = (exit_price / entry_price - 1) * 10000
            rets.append(ret_bps)

        r = stats_summary(rets, f'{lat_sec*1000:.0f}ms')
        r['latency_sec'] = lat_sec
        results.append(r)
        print(f"  {lat_sec*1000:>6.0f}ms: N={r['N']:5}, "
              f"net={r['mean_net']:>6.2f}bps, t={r['t_stat']:>5.2f}, "
              f"WR={r['win_rate']:.1f}%, Sharpe={r['sharpe']:>5.2f}")

    res_df = pd.DataFrame(results)
    res_df.to_csv('latency_results.csv', index=False)

    # ---- 図 ----
    fig = plt.figure(figsize=(13, 8), facecolor='white')
    plt.rcParams.update({
        'font.family': ['Hiragino Sans', 'IPAexGothic', 'sans-serif'],
        'axes.unicode_minus': False,
    })
    fig.suptitle('B-Optimized 執行レイテンシ感度 (SBG 9:00-10:30 60min保有)',
                 fontsize=13, fontweight='bold', y=0.99)

    # ---- 左: Sharpe 減衰 ----
    ax1 = fig.add_axes([0.07, 0.58, 0.40, 0.34])
    xs = res_df['latency_sec'].values
    ax1.plot(xs, res_df['sharpe'].values, 'o-', color='#1565C0', lw=2, ms=8)
    ax1.fill_between(xs, 0, res_df['sharpe'].values,
                      where=res_df['sharpe'].values > 0, alpha=0.2, color='#1565C0')
    ax1.fill_between(xs, 0, res_df['sharpe'].values,
                      where=res_df['sharpe'].values <= 0, alpha=0.2, color='#E53935')
    for x, sh, n in zip(xs, res_df['sharpe'], res_df['N']):
        ax1.annotate(f'{sh:.1f}', (x, sh), textcoords='offset points',
                     xytext=(0, 8), ha='center', fontsize=8, fontweight='bold')
    ax1.axhline(0, color='black', lw=0.7)
    ax1.set_xscale('symlog', linthresh=0.05)
    ax1.set_xlabel('レイテンシ (秒, log)', fontsize=9)
    ax1.set_ylabel('Sharpe (年率)', fontsize=9)
    ax1.set_title('Sharpe レイテンシ感度', fontsize=10, fontweight='bold')
    ax1.grid(alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # ---- 右: net (bps) 減衰 ----
    ax2 = fig.add_axes([0.55, 0.58, 0.40, 0.34])
    cols = ['#43A047' if v >= 0 else '#E53935' for v in res_df['mean_net']]
    xpos = range(len(res_df))
    ax2.bar(xpos, res_df['mean_net'].values, color=cols, alpha=0.85)
    for i, (_, row) in enumerate(res_df.iterrows()):
        ax2.text(i, row['mean_net'] + (0.5 if row['mean_net'] >= 0 else -0.5),
                 f"{row['mean_net']:.1f}\nt={row['t_stat']:.1f}",
                 ha='center', fontsize=8,
                 va='bottom' if row['mean_net'] >= 0 else 'top')
    ax2.set_xticks(xpos)
    ax2.set_xticklabels(res_df['label'], rotation=20, ha='right', fontsize=8)
    ax2.axhline(0, color='black', lw=0.7)
    ax2.axhline(COST_BPS, color='gray', linestyle='--', lw=1,
                label=f'コスト={COST_BPS}bps')
    ax2.set_ylabel('mean_net (bps/トレード)', fontsize=9)
    ax2.set_title('レイテンシ別 net期待値', fontsize=10, fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(axis='y', alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # ---- 下: テーブル ----
    ax3 = fig.add_axes([0.07, 0.07, 0.88, 0.42])
    ax3.axis('off')

    tbl_data = []
    for _, row in res_df.iterrows():
        # 実用判定
        if row['sharpe'] > 5:
            judge = '◎ 余裕'
        elif row['sharpe'] > 2:
            judge = '○ 実用OK'
        elif row['sharpe'] > 0.5:
            judge = '△ ギリギリ'
        elif row['sharpe'] > 0:
            judge = '✗ 微妙'
        else:
            judge = '✗ 損失'

        # 想定環境
        env = {
            '0ms': '理論値 (バックテスト前提)',
            '100ms': 'プロ コロケーション',
            '500ms': 'クラウド + 高速API',
            '1000ms': '一般システム + REST API',
            '3000ms': 'リテール証券 (WebSocket)',
            '10000ms': '手動 (反射神経)',
            '30000ms': '手動 (考えてから)',
            '60000ms': '完全マニュアル',
        }.get(row['label'], '')

        tbl_data.append([row['label'], env,
                         f"{row['N']:,}",
                         f"{row['mean_raw']:.2f}",
                         f"{row['mean_net']:.2f}",
                         f"{row['t_stat']:.2f}",
                         f"{row['win_rate']:.1f}%",
                         f"{row['sharpe']:.2f}",
                         judge])

    table_df = pd.DataFrame(tbl_data, columns=[
        'レイテンシ', '想定環境', 'N', 'raw(bps)', 'net(bps)',
        't値', '勝率', 'Sharpe', '判定'
    ])

    table = ax3.table(cellText=table_df.values, colLabels=table_df.columns,
                      cellLoc='center', loc='upper center',
                      bbox=[0, 0.05, 1, 0.92])
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor('#1565C0')
            cell.set_text_props(color='white', fontweight='bold')
        else:
            v_str = table_df.iloc[r-1]['判定']
            if '◎' in v_str:
                cell.set_facecolor('#A5D6A7')
            elif '○' in v_str:
                cell.set_facecolor('#FFF59D')
            elif '△' in v_str:
                cell.set_facecolor('#FFCC80')
            else:
                cell.set_facecolor('#FFCDD2')
        cell.set_edgecolor('#999999')

    fig.text(0.99, 0.005,
             'データ: 2026/1〜2026/4 / SBG (99840) JQuantsティック / B-Optimized 60min保有 | コスト4bps往復',
             ha='right', va='bottom', fontsize=7, color='gray')
    plt.savefig('result.png', dpi=100, bbox_inches='tight', facecolor='white')
    print(f"\nresult.png 保存完了 ({time.time()-t0:.1f}秒)")


if __name__ == '__main__':
    main()
