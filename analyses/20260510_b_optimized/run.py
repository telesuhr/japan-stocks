"""
B-Optimized: 全フィルタ統合最終形バックテスト

旧B1 (大口買い → 30min ロング) Sharpe=3.14 を起点に、
深掘り分析の知見を全て積み上げて統合バックテストを実施。

積上げ順:
  Base    : 全銘柄 / 全時間帯 / 30min保有 / フィルタなし
  +60min  : 保有時間 30→60min
  +Time   : 9:00-10:30 限定
  +Symbol : SBG のみ (or SBG+MUFG)
  +Quiet  : tick_ratio<1.5 フィルタ追加 ← 最終形

出力:
  - 各段階の累積効果テーブル
  - 推奨設定でのエクイティカーブ (日次累積)
  - 月別パフォーマンス
  - 銘柄別 (SBGのみ vs SBG+MUFG vs 全銘柄)
"""
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
ENRICHED = Path('/Users/Yusuke/claude-code/japan-stocks/.claude/worktrees/vibrant-mccarthy-d4c865/analyses/20260510_b_deep_dive/lp_enriched.parquet')

SYMBOL_NAMES = {'72030':'トヨタ','99840':'ソフトバンクG','83060':'MUFG',
                '68570':'アドバンテスト','67580':'ソニーG'}


def time_bucket(ts):
    h, m = ts.hour, ts.minute
    if h == 9 and m < 30: return '寄付30分'
    if h == 9 or (h == 10 and m < 30): return '前場前半'
    if h == 10 or (h == 11 and m <= 30): return '前場後半'
    if h == 12 and m >= 30: return 'ランチ後'
    if h == 13 or (h == 14 and m < 30): return '後場前半'
    return '大引前30分'


def stats_summary(returns, label='', cost=COST_BPS):
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n < 30:
        return dict(label=label, N=n, mean_raw=np.nan, mean_net=np.nan,
                    std=np.nan, t_stat=np.nan, p_val=np.nan,
                    win_rate=np.nan, sharpe=np.nan)
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

    print("enriched データロード...")
    lp = pd.read_parquet(ENRICHED)
    lp['ts'] = pd.to_datetime(lp['ts'])
    lp['date'] = lp['ts'].dt.date
    lp['time_bucket'] = lp['ts'].apply(time_bucket)
    print(f"  {len(lp):,}件")

    # 大口買い のみ
    bigbuy = lp[lp['direction'] == 1].copy()
    print(f"  大口買い (direction=+1): {len(bigbuy):,}件")

    # ===========================================================
    # フィルタ積上げ (全銘柄ベース、保有時間30/60で比較)
    # ===========================================================
    print("\n" + "="*65)
    print("フィルタ積上げ (各段階の累積効果)")
    print("="*65)

    buildup = []
    # Base 30min (旧B1)
    df = bigbuy.copy()
    buildup.append(stats_summary(df['fwd_30min'].values, 'Base (全条件) 30min'))

    # +60min
    df = bigbuy.copy()
    buildup.append(stats_summary(df['fwd_60min'].values, '+60min保有'))

    # +Time (9:00-10:30)
    df = bigbuy[bigbuy['time_bucket'].isin(['寄付30分','前場前半'])]
    buildup.append(stats_summary(df['fwd_60min'].values, '+9:00-10:30限定'))

    # +Symbol (SBG+MUFG)
    df = bigbuy[(bigbuy['code'].isin(['99840','83060'])) &
                (bigbuy['time_bucket'].isin(['寄付30分','前場前半']))]
    buildup.append(stats_summary(df['fwd_60min'].values, '+SBG+MUFG限定'))

    # +Symbol (SBG only)
    df = bigbuy[(bigbuy['code'] == '99840') &
                (bigbuy['time_bucket'].isin(['寄付30分','前場前半']))]
    buildup.append(stats_summary(df['fwd_60min'].values, '+SBG単独'))

    # +tick_ratio<1.5 (SBG only)
    df = bigbuy[(bigbuy['code'] == '99840') &
                (bigbuy['time_bucket'].isin(['寄付30分','前場前半'])) &
                (bigbuy['tick_ratio_at_entry'] < 1.5)]
    buildup.append(stats_summary(df['fwd_60min'].values, '+tick_ratio<1.5 (SBG最終)'))

    # +tick_ratio<1.5 (SBG+MUFG)
    df = bigbuy[(bigbuy['code'].isin(['99840','83060'])) &
                (bigbuy['time_bucket'].isin(['寄付30分','前場前半'])) &
                (bigbuy['tick_ratio_at_entry'] < 1.5)]
    buildup.append(stats_summary(df['fwd_60min'].values, '+tick_ratio<1.5 (SBG+MUFG最終)'))

    buildup_df = pd.DataFrame(buildup)
    print(buildup_df[['label','N','mean_raw','mean_net','t_stat','win_rate','sharpe']].to_string(index=False))

    # ===========================================================
    # 推奨最終条件 (B-Optimized): SBG+MUFG, 9:00-10:30, tick<1.5, 60min
    # ===========================================================
    print("\n" + "="*65)
    print("【推奨最終形】B-Optimized (SBG+MUFG)")
    print("="*65)

    final = bigbuy[
        (bigbuy['code'].isin(['99840','83060'])) &
        (bigbuy['time_bucket'].isin(['寄付30分','前場前半'])) &
        (bigbuy['tick_ratio_at_entry'] < 1.5)
    ].copy().dropna(subset=['fwd_60min'])

    print(f"\nトレード数: {len(final):,} ({final['date'].nunique()}営業日, "
          f"{len(final) / final['date'].nunique():.1f} トレード/日 / 銘柄合算)")

    final['pnl_net'] = final['fwd_60min'] - COST_BPS

    # 月別パフォーマンス
    final['ym'] = final['ts'].dt.to_period('M').astype(str)
    monthly = []
    for ym, g in final.groupby('ym'):
        r = stats_summary(g['fwd_60min'].values, ym)
        if r['N'] >= 30:
            monthly.append(r)
    monthly_df = pd.DataFrame(monthly)
    print("\n月別:")
    print(monthly_df[['label','N','mean_net','t_stat','win_rate','sharpe']].to_string(index=False))

    # 銘柄別 (最終条件下)
    print("\n銘柄別 (最終条件下):")
    sym_final = []
    for code, g in final.groupby('code'):
        r = stats_summary(g['fwd_60min'].values, f"{code} {SYMBOL_NAMES.get(code,'')}")
        if r:
            sym_final.append(r)
    sym_final_df = pd.DataFrame(sym_final)
    print(sym_final_df[['label','N','mean_net','t_stat','win_rate','sharpe']].to_string(index=False))

    # ===========================================================
    # SBG単独最終形
    # ===========================================================
    sbg_final = bigbuy[
        (bigbuy['code'] == '99840') &
        (bigbuy['time_bucket'].isin(['寄付30分','前場前半'])) &
        (bigbuy['tick_ratio_at_entry'] < 1.5)
    ].copy().dropna(subset=['fwd_60min'])
    print(f"\nSBG単独最終形: {len(sbg_final):,}トレード")
    sbg_stat = stats_summary(sbg_final['fwd_60min'].values, 'SBG単独')
    print(f"  N={sbg_stat['N']}, mean_net={sbg_stat['mean_net']}bps, "
          f"t={sbg_stat['t_stat']}, Sharpe={sbg_stat['sharpe']}")

    # ===========================================================
    # CSV保存
    # ===========================================================
    buildup_df.to_csv('buildup.csv', index=False)
    monthly_df.to_csv('monthly.csv', index=False)
    sym_final_df.to_csv('symbol_final.csv', index=False)

    # ===========================================================
    # 図 (4パネル)
    # ===========================================================
    fig = plt.figure(figsize=(15, 9.5), facecolor='white')
    plt.rcParams.update({
        'font.family': ['Hiragino Sans', 'IPAexGothic', 'sans-serif'],
        'axes.unicode_minus': False,
    })
    fig.suptitle('B-Optimized: 全フィルタ統合バックテスト (大口買い × 静かな環境 × 前場 × 60min)',
                 fontsize=13, fontweight='bold', y=0.99)

    # ---- ① フィルタ積上げ (Sharpe推移) ----
    ax1 = fig.add_axes([0.05, 0.58, 0.55, 0.34])
    xs = range(len(buildup_df))
    bars = ax1.bar(xs, buildup_df['sharpe'].values,
                   color=['#9E9E9E']*2 + ['#42A5F5']*2 + ['#1565C0']*2 + ['#0D47A1'],
                   alpha=0.85)
    for i, (_, row) in enumerate(buildup_df.iterrows()):
        if pd.notna(row['sharpe']):
            ax1.text(i, row['sharpe'] + 0.3,
                     f"Sh={row['sharpe']:.1f}\nN={row['N']:,}\nnet={row['mean_net']:.1f}bps",
                     ha='center', fontsize=7.5)
    ax1.set_xticks(xs)
    ax1.set_xticklabels(buildup_df['label'], rotation=25, ha='right', fontsize=8)
    ax1.axhline(0, color='black', lw=0.7)
    ax1.set_ylabel('Sharpe (年率, 1分bar基準)', fontsize=9)
    ax1.set_title('① フィルタ積上げ効果', fontsize=10, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # ---- ② エクイティカーブ (日次累積 net pnl) ----
    ax2 = fig.add_axes([0.65, 0.58, 0.32, 0.34])
    daily_pnl = final.groupby('date')['pnl_net'].sum()
    cum = daily_pnl.cumsum()
    ax2.plot(pd.to_datetime(cum.index), cum.values, color='#1565C0', lw=1.2)
    ax2.fill_between(pd.to_datetime(cum.index), 0, cum.values,
                      alpha=0.2, color='#1565C0')
    ax2.axhline(0, color='black', lw=0.7)
    ax2.set_ylabel('累積 net PnL (bps)', fontsize=9)
    ax2.set_title('② エクイティカーブ (B-Optimized)', fontsize=10, fontweight='bold')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax2.grid(alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # ---- ③ 月別パフォーマンス ----
    ax3 = fig.add_axes([0.05, 0.07, 0.42, 0.42])
    if len(monthly_df) > 0:
        xs = range(len(monthly_df))
        cols = ['#43A047' if v >= 0 else '#E53935' for v in monthly_df['mean_net']]
        ax3.bar(xs, monthly_df['mean_net'].fillna(0), color=cols, alpha=0.85)
        ax3.set_xticks(xs)
        ax3.set_xticklabels(monthly_df['label'], rotation=20, ha='right', fontsize=9)
        for i, (_, row) in enumerate(monthly_df.iterrows()):
            if pd.notna(row['mean_net']):
                ax3.text(i, row['mean_net'] + (0.5 if row['mean_net'] >= 0 else -0.5),
                         f"N={row['N']}\nSh={row['sharpe']:.1f}",
                         ha='center', fontsize=7.5,
                         va='bottom' if row['mean_net'] >= 0 else 'top')
        ax3.axhline(0, color='black', lw=0.7)
    ax3.set_ylabel('mean_net (bps)', fontsize=9)
    ax3.set_title('③ 月別パフォーマンス (B-Optimized)', fontsize=10, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    # ---- ④ サマリー表 ----
    ax4 = fig.add_axes([0.55, 0.07, 0.42, 0.42])
    ax4.axis('off')

    # 比較サマリー
    summary_data = [
        ['旧B1 (Base 30min)', f"{buildup_df.iloc[0]['N']:,}",
         f"{buildup_df.iloc[0]['mean_net']:.1f}",
         f"{buildup_df.iloc[0]['t_stat']:.1f}",
         f"{buildup_df.iloc[0]['sharpe']:.1f}"],
        ['+60min保有', f"{buildup_df.iloc[1]['N']:,}",
         f"{buildup_df.iloc[1]['mean_net']:.1f}",
         f"{buildup_df.iloc[1]['t_stat']:.1f}",
         f"{buildup_df.iloc[1]['sharpe']:.1f}"],
        ['+前場限定', f"{buildup_df.iloc[2]['N']:,}",
         f"{buildup_df.iloc[2]['mean_net']:.1f}",
         f"{buildup_df.iloc[2]['t_stat']:.1f}",
         f"{buildup_df.iloc[2]['sharpe']:.1f}"],
        ['+SBG+MUFG', f"{buildup_df.iloc[3]['N']:,}",
         f"{buildup_df.iloc[3]['mean_net']:.1f}",
         f"{buildup_df.iloc[3]['t_stat']:.1f}",
         f"{buildup_df.iloc[3]['sharpe']:.1f}"],
        ['+tick_ratio<1.5 (最終)', f"{buildup_df.iloc[6]['N']:,}",
         f"{buildup_df.iloc[6]['mean_net']:.1f}",
         f"{buildup_df.iloc[6]['t_stat']:.1f}",
         f"{buildup_df.iloc[6]['sharpe']:.1f}"],
        ['SBG単独 (最終)', f"{sbg_stat['N']:,}",
         f"{sbg_stat['mean_net']:.1f}",
         f"{sbg_stat['t_stat']:.1f}",
         f"{sbg_stat['sharpe']:.1f}"],
    ]
    summary_df_fig = pd.DataFrame(summary_data, columns=['設定', 'N', 'net(bps)', 't値', 'Sharpe'])

    table = ax4.table(cellText=summary_df_fig.values, colLabels=summary_df_fig.columns,
                      cellLoc='center', loc='upper center',
                      bbox=[0, 0.45, 1, 0.55])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor('#1565C0')
            cell.set_text_props(color='white', fontweight='bold')
        elif r in [4, 5]:  # 最終行を強調
            cell.set_facecolor('#FFF59D')
            cell.set_text_props(fontweight='bold')
        elif r % 2 == 0:
            cell.set_facecolor('#E3F2FD')
        cell.set_edgecolor('#BDBDBD')

    # 最終形のメトリクス
    final_sharpe = buildup_df.iloc[6]['sharpe']
    final_n = buildup_df.iloc[6]['N']
    final_mean = buildup_df.iloc[6]['mean_net']
    expected_trades_per_day = final_n / final['date'].nunique() if len(final) > 0 else 0
    expected_pnl_per_day = expected_trades_per_day * final_mean

    ax4.text(0.02, 0.40, '【B-Optimized 推奨設定】',
             fontsize=11, fontweight='bold', color='#0D47A1')
    ax4.text(0.02, 0.30,
             f"• 銘柄: SBG (99840) + MUFG (83060)\n"
             f"• 時間: 9:00 - 10:30 のみ\n"
             f"• 環境: tick_ratio < 1.5 (静かな相場)\n"
             f"• 保有: 60分\n"
             f"• 損切り: なし\n",
             fontsize=9, va='top')
    ax4.text(0.02, 0.10,
             f"期待トレード数: {expected_trades_per_day:.1f}/日 (合算)\n"
             f"期待PnL/日: {expected_pnl_per_day:.0f} bps × ロット",
             fontsize=9, va='top',
             bbox=dict(boxstyle='round', facecolor='#FFF9C4', alpha=0.8))

    fig.text(0.99, 0.005,
             'データ: 2026/1〜2026/4 / JQuants ティック (5銘柄, 大口買い 59,403件中の絞り込み) | コスト4bps往復',
             ha='right', va='bottom', fontsize=7, color='gray')
    plt.savefig('result.png', dpi=100, bbox_inches='tight', facecolor='white')
    print(f"\nresult.png 保存完了 ({time.time()-t0:.1f}秒)")


if __name__ == '__main__':
    main()
