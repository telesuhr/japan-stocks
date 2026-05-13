"""
A+B 合成戦略

Task A (投資部門別売買 = 市場全体のレジーム判定)
Task B (機関別空売り = 個別銘柄シグナル)

合成アイデア:
  Phase A: 市場が「強気レジーム」(海外/投信買い越し)の週に絞る
  Phase B: その週に、空売り急増銘柄をロング (ダブル強気)

OR

  Phase A: 市場が「弱気レジーム」(海外売り越し)の週に絞る
  Phase B: その週に、ショートカバー銘柄をショート (ダブル弱気)

仮説:
  H1: A強気 × B空売り急増(個別) → 銘柄レベルロング (squeeze + 市場tailwind)
  H2: A弱気 × Bショートカバー → 銘柄ショート (機関カバー後の下落)
  H3: A強気 × B空売りカバー → ロング (機関がカバー = 反転確認)
  H4: フィルタなし vs フィルタあり 比較
"""
import psycopg2
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

PG_CONFIG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}
COST_BPS = 8

A_CACHE = Path('/Users/Yusuke/claude-code/japan-stocks/.claude/worktrees/vibrant-mccarthy-d4c865/analyses/20260512_foreign_flow/flow_data.parquet')
B_CACHE = Path('/Users/Yusuke/claude-code/japan-stocks/.claude/worktrees/vibrant-mccarthy-d4c865/analyses/20260512_short_position/short_pos_data.parquet')


def stats_summary(arr, label=''):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n < 20:
        return dict(label=label, N=n, mean=np.nan, t_stat=np.nan, sharpe=np.nan,
                    win_rate=np.nan)
    t, p = stats.ttest_1samp(arr, 0)
    return dict(label=label, N=n,
                mean=round(arr.mean(), 1), std=round(arr.std(), 1),
                t_stat=round(t, 2), p_val=round(p, 4),
                win_rate=round((arr > 0).mean() * 100, 1),
                sharpe=round(arr.mean()/arr.std() * np.sqrt(52) if arr.std() > 0 else 0, 2))


def main():
    import time
    t0 = time.time()

    print("=" * 70)
    print("A+B 合成戦略")
    print("=" * 70)

    # ---- Task A: 投資部門別 ----
    print("\nTask A データロード...")
    a_df = pd.read_parquet(A_CACHE)
    a_df['pub_date'] = pd.to_datetime(a_df['pub_date'])
    if a_df['FrgnBal'].abs().max() > 100000:
        for c in [c for c in a_df.columns if c.endswith('Bal')]:
            a_df[c] = a_df[c] / 1e8
    # Z-score 計算
    for c in ['FrgnBal','IndBal','InvTrBal','PropBal','TrstBnkBal']:
        ma = a_df[c].shift(1).rolling(26, min_periods=8).mean()
        sd = a_df[c].shift(1).rolling(26, min_periods=8).std()
        a_df[f'{c}_z'] = (a_df[c] - ma) / sd.replace(0, np.nan)
    a_df['FvI_z'] = ((a_df['FrgnBal'] - a_df['IndBal'])
                     .pipe(lambda s: (s - s.shift(1).rolling(26).mean()) / s.shift(1).rolling(26).std().replace(0, np.nan)))

    # レジーム判定: 市場の地合い
    # 強気: 海外勢買い越し or 投信買い越し or 海外-個人スプレッド大
    a_df['regime_bull'] = (
        (a_df['FrgnBal_z'] > 1.0) |
        (a_df['InvTrBal_z'] > 1.0) |
        (a_df['FvI_z'] > 1.0)
    )
    # 弱気: 海外勢売り越し or 海外-個人スプレッド小
    a_df['regime_bear'] = (
        (a_df['FrgnBal_z'] < -1.0) |
        (a_df['FvI_z'] < -1.0)
    )
    print(f"  Aレジーム判定:")
    print(f"    強気週: {a_df['regime_bull'].sum()} / {len(a_df)} ({100*a_df['regime_bull'].mean():.1f}%)")
    print(f"    弱気週: {a_df['regime_bear'].sum()} / {len(a_df)} ({100*a_df['regime_bear'].mean():.1f}%)")
    print(f"    中立: {((~a_df['regime_bull']) & (~a_df['regime_bear'])).sum()}")

    # ---- Task B: 機関空売り ----
    print("\nTask B データロード...")
    b_df = pd.read_parquet(B_CACHE)
    b_df['disc_date'] = pd.to_datetime(b_df['disc_date'])
    print(f"  {len(b_df):,} reports")

    # 銘柄×日付集計
    print("\n銘柄×日付集計...")
    agg = b_df.groupby(['disc_date', 'code']).agg(
        n_reporters=('ss_name', 'count'),
        total_short=('shrt_pos_to_so', 'sum'),
        total_change=('change', 'sum'),
    ).reset_index()
    print(f"  {len(agg):,} 集計観測")

    # 株価
    print("株価取得...")
    conn = psycopg2.connect(**PG_CONFIG)
    daily = pd.read_sql("""
        SELECT code, date, adj_close FROM stocks_daily
        WHERE date >= '2021-01-01' AND adj_close IS NOT NULL
        ORDER BY code, date
    """, conn)
    conn.close()
    daily['date'] = pd.to_datetime(daily['date'])
    daily_dict = {c: g.sort_values('date').reset_index(drop=True) for c, g in daily.groupby('code')}

    # 翌5日リターン
    print("翌週リターン計算...")
    rows = []
    for _, row in agg.iterrows():
        c, d = row['code'], row['disc_date']
        if c not in daily_dict:
            continue
        g = daily_dict[c]
        idx_after = g[g['date'] > d].index
        if len(idx_after) < 6:
            continue
        start = idx_after[0]
        p0 = g.loc[start, 'adj_close']
        if start + 5 < len(g):
            p5 = g.loc[start + 5, 'adj_close']
            rows.append({
                'disc_date': d, 'code': c,
                'total_short': row['total_short'],
                'total_change': row['total_change'],
                'n_reporters': row['n_reporters'],
                'ret_5d': (p5/p0 - 1) * 10000,
            })

    sig_df = pd.DataFrame(rows)
    print(f"  {len(sig_df):,} 観測")

    # 市場ニュートラル化
    sig_df['ret_5d_mkt'] = sig_df.groupby('disc_date')['ret_5d'].transform('mean')
    sig_df['ret_5d_neutral'] = sig_df['ret_5d'] - sig_df['ret_5d_mkt']

    # クロスセクショナル ランク
    sig_df['change_rank'] = sig_df.groupby('disc_date')['total_change'].rank(pct=True)

    # Task A のレジーム判定を各日付にマッピング (merge_asof で高速)
    print("Aレジームと結合...")
    a_for_merge = a_df[['pub_date', 'regime_bull', 'regime_bear']].sort_values('pub_date')
    sig_df = sig_df.sort_values('disc_date').reset_index(drop=True)
    sig_df = pd.merge_asof(
        sig_df, a_for_merge,
        left_on='disc_date', right_on='pub_date',
        direction='backward',
        tolerance=pd.Timedelta(days=10),
    )
    sig_df['a_bull'] = sig_df['regime_bull']
    sig_df['a_bear'] = sig_df['regime_bear']

    # =======================================
    # 戦略比較
    # =======================================
    print("\n=== 戦略比較 (市場ニュートラル化 5日リターン) ===")

    results = []

    # === B 単独 ===
    sub = sig_df[sig_df['change_rank'] > 0.90]
    r = stats_summary(sub['ret_5d_neutral'].values, 'B単独: 空売り急増Top10% Long')
    if r: results.append(r)

    sub = sig_df[sig_df['change_rank'] < 0.10]
    r = stats_summary(-sub['ret_5d_neutral'].values, 'B単独: 空売りカバーTop10% Short')
    if r: results.append(r)

    # === A強気 × B (フィルタA: 市場強気の週に絞る) ===
    sub = sig_df[(sig_df['a_bull'] == True) & (sig_df['change_rank'] > 0.90)]
    r = stats_summary(sub['ret_5d_neutral'].values, 'A強気 × B空売り急増 Long (ダブル強気)')
    if r: results.append(r)

    sub = sig_df[(sig_df['a_bear'] == True) & (sig_df['change_rank'] < 0.10)]
    r = stats_summary(-sub['ret_5d_neutral'].values, 'A弱気 × Bカバー Short (ダブル弱気)')
    if r: results.append(r)

    # === A強気 のみ (市場全体) ===
    sub = sig_df[sig_df['a_bull'] == True]
    r = stats_summary(sub['ret_5d_neutral'].values, 'A強気のみ (全銘柄)')
    if r: results.append(r)

    sub = sig_df[sig_df['a_bear'] == True]
    r = stats_summary(-sub['ret_5d_neutral'].values, 'A弱気のみ (全銘柄) Short')
    if r: results.append(r)

    # === A 中立 × B (Aフィルタなしで純粋にBを使う) ===
    sub = sig_df[(sig_df['a_bull'] != True) & (sig_df['a_bear'] != True) & (sig_df['change_rank'] > 0.90)]
    r = stats_summary(sub['ret_5d_neutral'].values, 'A中立 × B空売り急増 Long')
    if r: results.append(r)

    # === A強気フィルタ × B (空売り増は不問、市場強気時) ===
    sub = sig_df[sig_df['a_bull'] == True]
    sub_top = sub[sub['change_rank'] > 0.80]
    r = stats_summary(sub_top['ret_5d_neutral'].values, 'A強気 × B空売り増Top20% Long')
    if r: results.append(r)

    # === LS両建て: 市場レジームに沿って ===
    # 強気週: B空売り急増ロング, 弱気週: Bカバー ショート
    long_strong = sig_df[(sig_df['a_bull'] == True) & (sig_df['change_rank'] > 0.90)]['ret_5d_neutral'].dropna().values
    short_weak = -sig_df[(sig_df['a_bear'] == True) & (sig_df['change_rank'] < 0.10)]['ret_5d_neutral'].dropna().values
    combined = np.concatenate([long_strong, short_weak])
    r = stats_summary(combined, 'LS両建て (A強気→空売増Long / A弱気→カバーShort)')
    if r: results.append(r)

    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))
    res_df.to_csv('signal_results.csv', index=False)

    # =====================================================================
    # エクイティカーブ (各戦略)
    # =====================================================================
    print("\n=== エクイティカーブ (週次平均) ===")
    sig_df_eq = sig_df.dropna(subset=['ret_5d_neutral'])

    # B単独
    b_only_long = sig_df_eq[sig_df_eq['change_rank'] > 0.90]
    b_only_daily = b_only_long.groupby('disc_date')['ret_5d_neutral'].mean()

    # A+B (強気)
    ab_long = sig_df_eq[(sig_df_eq['a_bull'] == True) & (sig_df_eq['change_rank'] > 0.90)]
    ab_daily = ab_long.groupby('disc_date')['ret_5d_neutral'].mean()

    # 全銘柄 (A強気時)
    a_strong = sig_df_eq[sig_df_eq['a_bull'] == True]
    a_strong_daily = a_strong.groupby('disc_date')['ret_5d_neutral'].mean()

    # LS両建て
    ls_long = sig_df_eq[(sig_df_eq['a_bull'] == True) & (sig_df_eq['change_rank'] > 0.90)]
    ls_short = sig_df_eq[(sig_df_eq['a_bear'] == True) & (sig_df_eq['change_rank'] < 0.10)]
    ls_daily_long = ls_long.groupby('disc_date')['ret_5d_neutral'].mean()
    ls_daily_short = -ls_short.groupby('disc_date')['ret_5d_neutral'].mean()
    ls_combined = pd.concat([ls_daily_long.rename('long'), ls_daily_short.rename('short')], axis=1).fillna(0)
    ls_combined['daily'] = (ls_combined['long'] + ls_combined['short']) / 2

    print(f"  B単独 N週: {len(b_only_daily)}, 累積: {b_only_daily.sum():.0f}bps")
    print(f"  A+B N週: {len(ab_daily)}, 累積: {ab_daily.sum():.0f}bps")
    print(f"  A強気 全銘柄 N週: {len(a_strong_daily)}, 累積: {a_strong_daily.sum():.0f}bps")
    print(f"  LS両建て N週: {len(ls_combined)}, 累積: {ls_combined['daily'].sum():.0f}bps")

    # =====================================================================
    # 図
    # =====================================================================
    fig = plt.figure(figsize=(15, 10), facecolor='white')
    plt.rcParams.update({
        'font.family': ['Hiragino Sans', 'IPAexGothic', 'sans-serif'],
        'axes.unicode_minus': False,
    })
    fig.suptitle('A+B 合成戦略 (市場レジームフィルタ × 個別銘柄シグナル)',
                 fontsize=13, fontweight='bold', y=0.99)

    # 上左: 戦略比較バー
    ax1 = fig.add_axes([0.05, 0.55, 0.55, 0.38])
    plot_df = res_df.dropna(subset=['mean']).copy()
    xs = range(len(plot_df))
    cols = ['#43A047' if (t > 1.96 and m > 0) else
            '#FF9800' if (t > 1.0 and m > 0) else
            '#E53935' for t, m in zip(plot_df['t_stat'], plot_df['mean'])]
    ax1.barh(list(xs), plot_df['mean'].values, color=cols, alpha=0.85)
    ax1.set_yticks(list(xs))
    ax1.set_yticklabels(plot_df['label'].str[:35], fontsize=8)
    for i, (_, row) in enumerate(plot_df.iterrows()):
        ax1.text(row['mean'] + (1 if row['mean']>=0 else -1), i,
                 f"N={row['N']}, t={row['t_stat']:.1f}, WR={row['win_rate']:.0f}%",
                 va='center', fontsize=7,
                 ha='left' if row['mean']>=0 else 'right')
    ax1.axvline(0, color='black', lw=0.6)
    ax1.set_xlabel('5日後 mean リターン (bps)', fontsize=9)
    ax1.set_title('戦略別パフォーマンス', fontsize=10, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # 上右: 改善度 (B単独 vs A+B)
    ax2 = fig.add_axes([0.65, 0.55, 0.32, 0.38])
    # B単独 vs A+B の比較
    b_only_row = plot_df[plot_df['label'].str.contains('B単独: 空売り急増')].iloc[0] if (plot_df['label'].str.contains('B単独: 空売り急増')).any() else None
    ab_row = plot_df[plot_df['label'].str.contains('A強気 × B空売り急増')].iloc[0] if (plot_df['label'].str.contains('A強気 × B空売り急増')).any() else None

    comparisons = []
    if b_only_row is not None:
        comparisons.append(('B単独 急増→Long', b_only_row['mean'], b_only_row['t_stat'], b_only_row['N']))
    if ab_row is not None:
        comparisons.append(('A強気×B Long', ab_row['mean'], ab_row['t_stat'], ab_row['N']))
    if (plot_df['label'].str.contains('A強気のみ')).any():
        a_row = plot_df[plot_df['label'].str.contains('A強気のみ')].iloc[0]
        comparisons.append(('A強気のみ', a_row['mean'], a_row['t_stat'], a_row['N']))
    if (plot_df['label'].str.contains('LS両建て')).any():
        ls_row = plot_df[plot_df['label'].str.contains('LS両建て')].iloc[0]
        comparisons.append(('LS両建て', ls_row['mean'], ls_row['t_stat'], ls_row['N']))

    if comparisons:
        labels = [c[0] for c in comparisons]
        means = [c[1] for c in comparisons]
        ts = [c[2] for c in comparisons]
        ns = [c[3] for c in comparisons]
        cols2 = ['#43A047' if t > 1.96 else '#FF9800' if t > 1.0 else '#9E9E9E' for t in ts]
        xs = range(len(labels))
        ax2.bar(xs, means, color=cols2, alpha=0.85)
        ax2.set_xticks(xs)
        ax2.set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
        for i, (m, t, n) in enumerate(zip(means, ts, ns)):
            ax2.text(i, m + (1.5 if m >= 0 else -1.5),
                     f"t={t:.1f}\nN={n}",
                     ha='center', fontsize=8,
                     va='bottom' if m>=0 else 'top')
        ax2.axhline(0, color='black', lw=0.6)
        ax2.set_ylabel('mean (bps)', fontsize=9)
        ax2.set_title('B単独 vs A+B 比較', fontsize=10, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # 下: エクイティカーブ
    ax3 = fig.add_axes([0.05, 0.07, 0.92, 0.40])
    if len(b_only_daily) > 0:
        ax3.plot(b_only_daily.index, b_only_daily.cumsum().values,
                 color='gray', lw=1, alpha=0.6, label=f'B単独 (Long) 累積={b_only_daily.sum():.0f}bps')
    if len(ab_daily) > 0:
        ax3.plot(ab_daily.index, ab_daily.cumsum().values,
                 color='#1565C0', lw=1.8, label=f'A+B 強気合成 累積={ab_daily.sum():.0f}bps')
    if len(ls_combined) > 0:
        ax3.plot(ls_combined.index, ls_combined['daily'].cumsum().values,
                 color='#43A047', lw=1.8, label=f'LS両建て 累積={ls_combined["daily"].sum():.0f}bps')
    ax3.axhline(0, color='black', lw=0.6)
    ax3.set_xlabel('日付', fontsize=9)
    ax3.set_ylabel('累積 net PnL (bps, 市場ニュートラル化済)', fontsize=9)
    ax3.set_title('A+B 合成戦略のエクイティカーブ比較 (2021-2026)',
                  fontsize=10, fontweight='bold')
    ax3.legend(fontsize=9, loc='best')
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%y/%m'))
    ax3.grid(alpha=0.3)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    fig.text(0.99, 0.005,
             'データ: 投資部門別 (週次) × 空売り報告 (個別) | Walk-forward Z-score 26週 | 5日リターン',
             ha='right', va='bottom', fontsize=7, color='gray')
    plt.savefig('result.png', dpi=100, bbox_inches='tight', facecolor='white')
    print(f"\nresult.png 保存完了 ({time.time()-t0:.1f}秒)")


if __name__ == '__main__':
    main()
