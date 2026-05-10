"""
B戦略 (大口買いプリント追随) の深掘り分析

目的: B1 (+2.34bps net, t=16.89, Sharpe=3.14) を実戦投入できるレベルに磨く

検証項目 (推奨順):
  1. 保有時間最適化  : 30/60/90/120分/EOD 比較
  3. 銘柄別ロバストネス: 5銘柄ごとに同じエッジか
  4. A+B 統合シグナル : 大口買い × ティック数スパイク
  2. 時間帯フィルタ   : 寄付/前場/ランチ前/後場/大引前
  5. 損切り効果      : 30分中の最大DD で早期撤退
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
SYMBOL_NAMES = {'72030':'トヨタ','99840':'ソフトバンクG','83060':'MUFG',
                '68570':'アドバンテスト','67580':'ソニーG'}

LP_CACHE = Path('/Users/Yusuke/claude-code/japan-stocks/.claude/worktrees/vibrant-mccarthy-d4c865/analyses/20260510_large_prints/large_prints_cache.parquet')
BARS_CACHE = Path('/Users/Yusuke/claude-code/japan-stocks/.claude/worktrees/vibrant-mccarthy-d4c865/analyses/20260510_tick_spike_signals/bars_with_tickcount.parquet')
ENRICHED_CACHE = Path('lp_enriched.parquet')


def stats_summary(returns, label='', cost=COST_BPS):
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n < 30:
        return None
    net = arr - cost
    t, p = stats.ttest_1samp(arr, 0)
    return dict(label=label, N=n,
                mean_raw=round(arr.mean(), 2), mean_net=round(net.mean(), 2),
                std=round(arr.std(), 2), t_stat=round(t, 2), p_val=round(p, 4),
                win_rate=round((arr > 0).mean() * 100, 1),
                sharpe=round(net.mean() / arr.std() * np.sqrt(252 * 60) if arr.std() > 0 else 0, 2))


def enrich_large_prints():
    """
    大口プリントに以下を追加:
    - 各保有時間でのフォワードクローズ (30/60/90/120min/EOD)
    - 30分保有中の最低価格 (DD), 最高価格
    - 入った1分バーの tick_ratio
    """
    print(f"large_prints ロード: {LP_CACHE}")
    lp = pd.read_parquet(LP_CACHE)
    print(f"  {len(lp):,}件")

    print(f"bars_with_tickcount ロード: {BARS_CACHE}")
    bars = pd.read_parquet(BARS_CACHE)
    print(f"  {len(bars):,}バー")

    # 1分バーに tick_ratio を計算 (per code, per date)
    bars['date'] = bars['ts'].dt.date
    print("bars に tick_ratio を計算...")
    out = []
    for (code, d), g in bars.groupby(['code', 'date']):
        g = g.sort_values('ts').reset_index(drop=True)
        g['tick_ma30'] = g['tick_count'].shift(1).rolling(30, min_periods=10).mean()
        g['tick_ratio'] = g['tick_count'] / g['tick_ma30']
        out.append(g)
    bars = pd.concat(out, ignore_index=True)

    # 各銘柄の bars を辞書に (np配列で高速アクセス)
    bars_idx = {}
    for code, g in bars.groupby('code'):
        g = g.sort_values('ts').reset_index(drop=True)
        bars_idx[code] = {
            'ts_int': g['ts'].astype('datetime64[s]').astype('int64').values,
            'close': g['close'].values,
            'high': g['high'].values,
            'low': g['low'].values,
            'tick_ratio': g['tick_ratio'].values,
            'date': g['date'].values,
        }
    print(f"  bars_idx: {sum(len(v['ts_int']) for v in bars_idx.values()):,}行")

    # large_prints 拡張
    lp['bar_ts'] = pd.to_datetime(lp['ts']).dt.floor('1min')
    lp['date'] = pd.to_datetime(lp['ts']).dt.date

    # 結果列を初期化
    horizons = [30, 60, 90, 120]
    for h in horizons:
        lp[f'fwd_{h}min'] = np.nan
    lp['fwd_eod'] = np.nan
    lp['dd30_max'] = np.nan       # 30分保有中の最大ドローダウン (long)
    lp['mfe30_max'] = np.nan      # 30分保有中の最大利益 (long)
    lp['tick_ratio_at_entry'] = np.nan

    print("各大口プリントに保有期間別リターン・DD/MFE・tick_ratio を計算中...")
    lp_arr = lp.reset_index(drop=True)

    for i, row in lp_arr.iterrows():
        if i % 20000 == 0:
            print(f"  {i:,}/{len(lp_arr):,}")
        code = row['code']
        if code not in bars_idx:
            continue
        bidx = bars_idx[code]

        bar_ts_int = int(pd.Timestamp(row['bar_ts']).timestamp())
        # entry bar idx
        j = np.searchsorted(bidx['ts_int'], bar_ts_int)
        if j >= len(bidx['ts_int']) or bidx['ts_int'][j] != bar_ts_int:
            continue
        entry_close = bidx['close'][j]  # use bar close as entry approximation
        entry_price = row['price']
        same_date = bidx['date'][j]
        lp_arr.loc[i, 'tick_ratio_at_entry'] = bidx['tick_ratio'][j]

        # フォワードリターン: bar_ts + N分 のbar close で計算
        for h in horizons:
            target = bar_ts_int + h * 60
            k = np.searchsorted(bidx['ts_int'], target)
            if k >= len(bidx['ts_int']):
                continue
            if bidx['date'][k] != same_date:
                continue
            lp_arr.loc[i, f'fwd_{h}min'] = (bidx['close'][k] / entry_price - 1) * 10000

        # EOD: 同じ date の最後のバー
        last_idx = j
        while last_idx + 1 < len(bidx['ts_int']) and bidx['date'][last_idx + 1] == same_date:
            last_idx += 1
        if last_idx > j:
            lp_arr.loc[i, 'fwd_eod'] = (bidx['close'][last_idx] / entry_price - 1) * 10000

        # 30分保有中のDD/MFE: bar j+1 から 30分後までの low/high
        end_target = bar_ts_int + 30 * 60
        end_k = np.searchsorted(bidx['ts_int'], end_target)
        end_k = min(end_k, last_idx)  # 同日内に制限
        if end_k > j:
            lows = bidx['low'][j+1:end_k+1]
            highs = bidx['high'][j+1:end_k+1]
            if len(lows) > 0:
                # ロングポジションのDD: 最低 low - entry
                lp_arr.loc[i, 'dd30_max'] = (lows.min() / entry_price - 1) * 10000
                lp_arr.loc[i, 'mfe30_max'] = (highs.max() / entry_price - 1) * 10000

    print(f"完了。キャッシュ保存: {ENRICHED_CACHE}")
    lp_arr.to_parquet(ENRICHED_CACHE)
    return lp_arr


def time_bucket(ts):
    h, m = ts.hour, ts.minute
    if h == 9 and m < 30: return '寄付30分'
    if h == 9 or (h == 10 and m < 30): return '前場前半'
    if h == 10 or (h == 11 and m <= 30): return '前場後半'
    if h == 12 and m >= 30: return 'ランチ後'
    if h == 13 or (h == 14 and m < 30): return '後場前半'
    return '大引前30分'


def main():
    import time
    t0 = time.time()

    if ENRICHED_CACHE.exists():
        print(f"enriched キャッシュからロード: {ENRICHED_CACHE}")
        lp = pd.read_parquet(ENRICHED_CACHE)
    else:
        lp = enrich_large_prints()

    print(f"\n対象データ: {len(lp):,}件 / 銘柄: {lp['code'].nunique()}")
    bigbuy = lp[lp['direction'] == 1].copy()  # B1 用 (大口買い)
    print(f"  大口買い (direction=+1): {len(bigbuy):,}件")

    # =======================================================
    # 1. 保有時間最適化
    # =======================================================
    print("\n" + "="*60)
    print("【1】保有時間最適化 (大口買い → ロング)")
    print("="*60)
    holding_results = []
    for h_lab, col in [('5min','fwd_5min'),('15min','fwd_15min'),
                       ('30min','fwd_30min'),('60min','fwd_60min'),
                       ('90min','fwd_90min'),('120min','fwd_120min'),
                       ('EOD','fwd_eod')]:
        if col not in bigbuy.columns:
            continue
        r = stats_summary(bigbuy[col].dropna().values, h_lab)
        if r:
            holding_results.append(r)
    holding_df = pd.DataFrame(holding_results)
    print(holding_df[['label','N','mean_raw','mean_net','t_stat','win_rate','sharpe']].to_string(index=False))

    # =======================================================
    # 3. 銘柄別ロバストネス (30分保有を共通条件に)
    # =======================================================
    print("\n" + "="*60)
    print("【3】銘柄別ロバストネス (30分保有)")
    print("="*60)
    sym_results = []
    for code, g in bigbuy.groupby('code'):
        r = stats_summary(g['fwd_30min'].dropna().values, f"{code} {SYMBOL_NAMES.get(code, '')}")
        if r:
            sym_results.append(r)
    sym_df = pd.DataFrame(sym_results)
    print(sym_df[['label','N','mean_raw','mean_net','t_stat','win_rate','sharpe']].to_string(index=False))

    # =======================================================
    # 4. A+B 統合シグナル: 大口買い × ティック数スパイク
    # =======================================================
    print("\n" + "="*60)
    print("【4】A+B 統合シグナル (30分保有)")
    print("="*60)
    integrated_results = []
    # 段階的に閾値を上げる
    for thresh in [None, 1.5, 2.0, 3.0, 5.0]:
        if thresh is None:
            sub = bigbuy
            label = '全大口買い (フィルタなし)'
        else:
            sub = bigbuy[bigbuy['tick_ratio_at_entry'] >= thresh]
            label = f'tick_ratio≥{thresh}'
        r = stats_summary(sub['fwd_30min'].dropna().values, label)
        if r:
            integrated_results.append(r)
    integ_df = pd.DataFrame(integrated_results)
    print(integ_df[['label','N','mean_raw','mean_net','t_stat','win_rate','sharpe']].to_string(index=False))

    # =======================================================
    # 2. 時間帯フィルタ
    # =======================================================
    print("\n" + "="*60)
    print("【2】時間帯フィルタ (30分保有)")
    print("="*60)
    bigbuy['time_bucket'] = pd.to_datetime(bigbuy['ts']).apply(time_bucket)
    time_results = []
    bucket_order = ['寄付30分','前場前半','前場後半','ランチ後','後場前半','大引前30分']
    for tb in bucket_order:
        sub = bigbuy[bigbuy['time_bucket'] == tb]
        r = stats_summary(sub['fwd_30min'].dropna().values, tb)
        if r:
            time_results.append(r)
    time_df = pd.DataFrame(time_results)
    print(time_df[['label','N','mean_raw','mean_net','t_stat','win_rate','sharpe']].to_string(index=False))

    # =======================================================
    # 5. 損切り効果
    # =======================================================
    print("\n" + "="*60)
    print("【5】損切り効果 (30分保有 ベース)")
    print("="*60)
    # 30分保有中の最大DD分布
    valid = bigbuy.dropna(subset=['fwd_30min','dd30_max'])
    print(f"  分析対象: {len(valid):,}件")
    print(f"  30分保有中DD分布: 5%ile={np.percentile(valid['dd30_max'],5):.0f}bps, "
          f"25%ile={np.percentile(valid['dd30_max'],25):.0f}bps, "
          f"中央値={np.percentile(valid['dd30_max'],50):.0f}bps")

    sl_results = []
    # 損切り条件をシミュレート: dd30_max <= -X bps なら -X bps で撤退
    # それ以外は30分後に決済
    for sl_thresh in [None, -50, -30, -20, -15, -10, -5]:
        if sl_thresh is None:
            ret = valid['fwd_30min'].values
            label = '損切りなし'
        else:
            stopped = valid['dd30_max'] <= sl_thresh
            ret = np.where(stopped, sl_thresh, valid['fwd_30min'].values)
            label = f'SL={sl_thresh}bps'
        r = stats_summary(ret, label)
        if r:
            r['stopped_pct'] = round((valid['dd30_max'] <= (sl_thresh or -1e9)).mean()*100, 1) if sl_thresh else 0
            sl_results.append(r)
    sl_df = pd.DataFrame(sl_results)
    print(sl_df[['label','N','mean_net','t_stat','win_rate','sharpe','stopped_pct']].to_string(index=False))

    # CSV保存
    holding_df.to_csv('1_holding_time.csv', index=False)
    sym_df.to_csv('3_per_symbol.csv', index=False)
    integ_df.to_csv('4_integrated.csv', index=False)
    time_df.to_csv('2_time_bucket.csv', index=False)
    sl_df.to_csv('5_stoploss.csv', index=False)

    # =======================================================
    # 図 (1200x900 で6パネル)
    # =======================================================
    fig = plt.figure(figsize=(15, 10), facecolor='white')
    plt.rcParams.update({
        'font.family': ['Hiragino Sans', 'IPAexGothic', 'sans-serif'],
        'axes.unicode_minus': False,
    })
    fig.suptitle('B戦略深掘り: 大口買いプリント追随 (5銘柄 × 4ヶ月 / N=59,403)',
                 fontsize=13, fontweight='bold', y=0.99)

    def style_ax(ax):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(alpha=0.3)

    # ① 保有時間最適化
    ax1 = fig.add_axes([0.05, 0.55, 0.28, 0.36])
    xs = range(len(holding_df))
    cols = ['#43A047' if v >= 0 else '#E53935' for v in holding_df['mean_net']]
    ax1.bar(xs, holding_df['mean_net'], color=cols, alpha=0.85)
    ax1.set_xticks(xs)
    ax1.set_xticklabels(holding_df['label'], rotation=30, ha='right', fontsize=8)
    ax1.axhline(0, color='black', lw=0.7)
    for i, (_, row) in enumerate(holding_df.iterrows()):
        ax1.text(i, row['mean_net'] + (0.3 if row['mean_net'] >= 0 else -0.3),
                 f"t={row['t_stat']:.1f}", ha='center', fontsize=7,
                 va='bottom' if row['mean_net'] >= 0 else 'top')
    ax1.set_ylabel('mean_net (bps)', fontsize=9)
    ax1.set_title('① 保有時間最適化', fontsize=10, fontweight='bold')
    style_ax(ax1)

    # ③ 銘柄別
    ax3 = fig.add_axes([0.38, 0.55, 0.28, 0.36])
    xs = range(len(sym_df))
    cols = ['#43A047' if v >= 0 else '#E53935' for v in sym_df['mean_net']]
    ax3.bar(xs, sym_df['mean_net'], color=cols, alpha=0.85)
    short_lab = sym_df['label'].str.split(' ').str[1].fillna(sym_df['label'])
    ax3.set_xticks(xs)
    ax3.set_xticklabels(short_lab, rotation=20, ha='right', fontsize=8)
    ax3.axhline(0, color='black', lw=0.7)
    for i, (_, row) in enumerate(sym_df.iterrows()):
        ax3.text(i, row['mean_net'] + (0.3 if row['mean_net'] >= 0 else -0.3),
                 f"t={row['t_stat']:.1f}\nSh={row['sharpe']:.1f}",
                 ha='center', fontsize=6.5,
                 va='bottom' if row['mean_net'] >= 0 else 'top')
    ax3.set_ylabel('mean_net (bps)', fontsize=9)
    ax3.set_title('③ 銘柄別ロバストネス (30min)', fontsize=10, fontweight='bold')
    style_ax(ax3)

    # ④ A+B 統合
    ax4 = fig.add_axes([0.71, 0.55, 0.27, 0.36])
    xs = range(len(integ_df))
    cols = ['#1E88E5' if 'tick' in str(l) else '#9E9E9E' for l in integ_df['label']]
    ax4.bar(xs, integ_df['mean_net'], color=cols, alpha=0.85)
    ax4.set_xticks(xs)
    ax4.set_xticklabels(integ_df['label'], rotation=30, ha='right', fontsize=7.5)
    ax4.axhline(0, color='black', lw=0.7)
    for i, (_, row) in enumerate(integ_df.iterrows()):
        ax4.text(i, row['mean_net'] + 0.3, f"N={row['N']}\nt={row['t_stat']:.1f}",
                 ha='center', fontsize=6.5)
    ax4.set_ylabel('mean_net (bps)', fontsize=9)
    ax4.set_title('④ A+B 統合 (大口買い × tick_ratio)', fontsize=10, fontweight='bold')
    style_ax(ax4)

    # ② 時間帯
    ax2 = fig.add_axes([0.05, 0.07, 0.42, 0.38])
    xs = range(len(time_df))
    cols = ['#43A047' if v >= 0 else '#E53935' for v in time_df['mean_net']]
    ax2.bar(xs, time_df['mean_net'], color=cols, alpha=0.85)
    ax2.set_xticks(xs)
    ax2.set_xticklabels(time_df['label'], rotation=20, ha='right', fontsize=8)
    ax2.axhline(0, color='black', lw=0.7)
    for i, (_, row) in enumerate(time_df.iterrows()):
        ax2.text(i, row['mean_net'] + (0.3 if row['mean_net'] >= 0 else -0.3),
                 f"N={row['N']}\nt={row['t_stat']:.1f}",
                 ha='center', fontsize=7,
                 va='bottom' if row['mean_net'] >= 0 else 'top')
    ax2.set_ylabel('mean_net (bps)', fontsize=9)
    ax2.set_title('② 時間帯フィルタ (30min)', fontsize=10, fontweight='bold')
    style_ax(ax2)

    # ⑤ 損切り
    ax5 = fig.add_axes([0.55, 0.07, 0.42, 0.38])
    xs = range(len(sl_df))
    cols = ['#43A047' if v >= 0 else '#E53935' for v in sl_df['mean_net']]
    ax5.bar(xs, sl_df['mean_net'], color=cols, alpha=0.85)
    ax5.set_xticks(xs)
    ax5.set_xticklabels(sl_df['label'], rotation=20, ha='right', fontsize=8)
    ax5.axhline(0, color='black', lw=0.7)
    for i, (_, row) in enumerate(sl_df.iterrows()):
        ax5.text(i, row['mean_net'] + (0.3 if row['mean_net'] >= 0 else -0.3),
                 f"Sh={row['sharpe']:.1f}\n{row['stopped_pct']:.0f}%停",
                 ha='center', fontsize=6.5,
                 va='bottom' if row['mean_net'] >= 0 else 'top')
    ax5.set_ylabel('mean_net (bps)', fontsize=9)
    ax5.set_title('⑤ 損切り効果 (30min ベース)', fontsize=10, fontweight='bold')
    style_ax(ax5)

    fig.text(0.99, 0.005,
             'データ: 2026/1〜2026/4 / トヨタ・SBG・MUFG・ADTEST・ソニー / JQuantsティック | 大口=99%ile以上',
             ha='right', va='bottom', fontsize=7, color='gray')
    plt.savefig('result.png', dpi=100, bbox_inches='tight', facecolor='white')
    print(f"\nresult.png 保存完了 ({time.time()-t0:.1f}秒)")


if __name__ == '__main__':
    main()
