"""
Task A (投資部門別売買) OOS 検証

学習期: 2016-2022 (7年, 363週)
OOS期: 2023-2026 (3.4年, 159週)

検証項目:
  1. 学習期に発見したシグナル (海外勢, 投信, 海外-個人スプレッド)
     を学習期と完全分離したOOSで適用
  2. 在野での実トレード相当 (Walk-forward) で運用可能か
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

CACHE = Path('/Users/Yusuke/claude-code/japan-stocks/.claude/worktrees/vibrant-mccarthy-d4c865/analyses/20260512_foreign_flow/flow_data.parquet')


def stats_summary(arr, label=''):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n < 5:
        return dict(label=label, N=n, mean=np.nan, std=np.nan, t_stat=np.nan,
                    p_val=np.nan, win_rate=np.nan, sharpe=np.nan)
    t, p = stats.ttest_1samp(arr, 0)
    return dict(label=label, N=n,
                mean=round(arr.mean(), 1), std=round(arr.std(), 1),
                t_stat=round(t, 2), p_val=round(p, 4),
                win_rate=round((arr > 0).mean() * 100, 1),
                sharpe=round(arr.mean()/arr.std() * np.sqrt(52) if arr.std() > 0 else 0, 2))


def main():
    import time
    t0 = time.time()

    print("Task A データロード...")
    df = pd.read_parquet(CACHE)
    print(f"  {len(df)} 週")

    # Bal を 億円に変換
    bal_cols = [c for c in df.columns if c.endswith('Bal')]
    # 既に億円に変換されていない場合があるので、もし大きすぎたら変換
    if df['FrgnBal'].abs().max() > 100000:
        for c in bal_cols:
            df[c] = df[c] / 1e8

    df['pub_date'] = pd.to_datetime(df['pub_date'])

    # =====================================================================
    # 重要: Walk-forward (リアルな運用相当)
    # 各週時点で「過去26週」のローリングZ-scoreを使う (look-aheadなし)
    # =====================================================================
    for c in bal_cols:
        ma = df[c].shift(1).rolling(26, min_periods=8).mean()
        sd = df[c].shift(1).rolling(26, min_periods=8).std()
        df[f'{c}_z'] = (df[c] - ma) / sd.replace(0, np.nan)

    # 海外-個人スプレッド
    df['FvI'] = df['FrgnBal'] - df['IndBal']
    df['FvI_z'] = ((df['FvI'] - df['FvI'].shift(1).rolling(26).mean()) /
                   df['FvI'].shift(1).rolling(26).std().replace(0, np.nan))

    # ===== 期間分割 =====
    train_end = pd.Timestamp('2022-12-31')
    df['period'] = np.where(df['pub_date'] <= train_end, 'Train', 'OOS')

    train_df = df[df['period'] == 'Train'].copy()
    oos_df = df[df['period'] == 'OOS'].copy()

    print(f"\n  Train (2016-2022): {len(train_df)}週")
    print(f"  OOS (2023-2026.5): {len(oos_df)}週")

    # =====================================================================
    # 主要シグナル一覧 (Task A で発見したベスト)
    # =====================================================================
    signals = [
        ('海外勢 Z>1.5 Long', lambda d: d['FrgnBal_z'] > 1.5, 1, 'ret_5d'),
        ('海外勢 Z>1.0 Long (緩)', lambda d: d['FrgnBal_z'] > 1.0, 1, 'ret_5d'),
        ('投信 Z>1.5 Long', lambda d: d['InvTrBal_z'] > 1.5, 1, 'ret_5d'),
        ('信託銀行 Z>1.5 Long', lambda d: d['TrstBnkBal_z'] > 1.5, 1, 'ret_5d'),
        ('海外-個人 Z>1.5 Long', lambda d: d['FvI_z'] > 1.5, 1, 'ret_5d'),
        ('海外-個人 Z>2.0 Long', lambda d: d['FvI_z'] > 2.0, 1, 'ret_5d'),
        ('海外-個人 Z>1.0 Long (緩)', lambda d: d['FvI_z'] > 1.0, 1, 'ret_5d'),
        ('自己売買 Z<-1.5 Short', lambda d: d['PropBal_z'] < -1.5, -1, 'ret_5d'),
    ]

    print("\n" + "="*80)
    print("【OOS検証 結果】")
    print("="*80)
    print(f"{'シグナル':<32} | {'Train (2016-22)':<32} | {'OOS (2023-26)':<32}")
    print("-"*100)

    results = []
    for sig_name, sig_func, direction, target in signals:
        train_sub = train_df[sig_func(train_df)]
        oos_sub = oos_df[sig_func(oos_df)]

        train_ret = direction * train_sub[target].values
        oos_ret = direction * oos_sub[target].values

        train_stat = stats_summary(train_ret, sig_name + ' Train')
        oos_stat = stats_summary(oos_ret, sig_name + ' OOS')

        train_str = f"N={train_stat['N']:>3}, mean={train_stat['mean']:>+6.1f}, WR={train_stat['win_rate']:>4.1f}%"
        oos_str   = f"N={oos_stat['N']:>3}, mean={oos_stat['mean']:>+6.1f}, WR={oos_stat['win_rate']:>4.1f}%"
        print(f"{sig_name:<32} | {train_str:<32} | {oos_str:<32}")

        results.append({
            'signal': sig_name, 'direction': direction,
            'train_N': train_stat['N'], 'train_mean': train_stat['mean'],
            'train_t': train_stat['t_stat'], 'train_wr': train_stat['win_rate'],
            'train_sharpe': train_stat['sharpe'],
            'oos_N': oos_stat['N'], 'oos_mean': oos_stat['mean'],
            'oos_t': oos_stat['t_stat'], 'oos_wr': oos_stat['win_rate'],
            'oos_sharpe': oos_stat['sharpe'],
        })

    res_df = pd.DataFrame(results)
    res_df.to_csv('oos_results.csv', index=False)

    # =====================================================================
    # OOS 期間中のシグナル発生履歴 (海外勢 Z>1.5)
    # =====================================================================
    print("\n=== OOS期間中のシグナル発生履歴 (海外勢 Z>1.5 Long) ===")
    oos_signal_log = oos_df[oos_df['FrgnBal_z'] > 1.5][
        ['pub_date', 'FrgnBal', 'FrgnBal_z', 'ret_5d']
    ].copy()
    oos_signal_log['FrgnBal'] = oos_signal_log['FrgnBal'].round(0)
    oos_signal_log['FrgnBal_z'] = oos_signal_log['FrgnBal_z'].round(2)
    oos_signal_log['ret_5d'] = oos_signal_log['ret_5d'].round(0)
    oos_signal_log['pub_date'] = oos_signal_log['pub_date'].dt.strftime('%Y-%m-%d')
    oos_signal_log['result'] = oos_signal_log['ret_5d'].apply(
        lambda x: '✓ 勝ち' if x > 0 else '✗ 負け' if pd.notna(x) else '-')
    print(oos_signal_log.to_string(index=False))
    oos_signal_log.to_csv('oos_signal_log.csv', index=False)

    # =====================================================================
    # 直近6か月のシグナル (2025-11 ~ 2026-05)
    # =====================================================================
    print("\n=== 直近6ヶ月のシグナル (2025-11 ~ 2026-05) ===")
    recent = df[df['pub_date'] >= '2025-11-01']
    recent_sig = recent[
        (recent['FrgnBal_z'] > 1.5) |
        (recent['InvTrBal_z'] > 1.5) |
        (recent['FvI_z'] > 1.5)
    ].copy()
    recent_sig['signals'] = recent_sig.apply(lambda r:
        ('海外Z' if r['FrgnBal_z'] > 1.5 else '') +
        ('+投信' if r['InvTrBal_z'] > 1.5 else '') +
        ('+海個' if r['FvI_z'] > 1.5 else ''), axis=1)
    if len(recent_sig) > 0:
        cols_show = ['pub_date', 'FrgnBal_z', 'InvTrBal_z', 'FvI_z', 'ret_5d', 'signals']
        recent_sig['pub_date'] = recent_sig['pub_date'].dt.strftime('%Y-%m-%d')
        for c in ['FrgnBal_z', 'InvTrBal_z', 'FvI_z']:
            recent_sig[c] = recent_sig[c].round(2)
        recent_sig['ret_5d'] = recent_sig['ret_5d'].round(0)
        print(recent_sig[cols_show].to_string(index=False))

    # =====================================================================
    # エクイティカーブ (OOS期間)
    # =====================================================================
    oos_df_copy = oos_df.copy()
    # ベスト合成シグナル (海外勢 OR 投信 OR 海外-個人)
    oos_df_copy['signal'] = (
        (oos_df_copy['FrgnBal_z'] > 1.5) |
        (oos_df_copy['InvTrBal_z'] > 1.5) |
        (oos_df_copy['FvI_z'] > 1.5)
    )
    oos_df_copy['pnl'] = np.where(oos_df_copy['signal'], oos_df_copy['ret_5d'], 0)
    oos_df_copy = oos_df_copy.sort_values('pub_date').reset_index(drop=True)
    oos_df_copy['cum_pnl'] = oos_df_copy['pnl'].cumsum()
    oos_df_copy['cum_topix'] = oos_df_copy['ret_5d'].cumsum()
    oos_df_copy['cum_topix_normalized'] = (
        oos_df_copy['cum_topix'].iloc[-1] *
        oos_df_copy.index / len(oos_df_copy)
    )

    # シグナル週数
    sig_weeks = oos_df_copy['signal'].sum()
    total_pnl = oos_df_copy['pnl'].sum()
    print(f"\nOOS期間総括 (合成シグナル: 海外/投信/海外-個人 OR):")
    print(f"  シグナル発生週: {sig_weeks}/{len(oos_df_copy)} ({100*sig_weeks/len(oos_df_copy):.1f}%)")
    print(f"  合計PnL: {total_pnl:.0f}bps ({total_pnl/100:.1f}%)")
    if sig_weeks > 0:
        signal_returns = oos_df_copy[oos_df_copy['signal']]['ret_5d'].values
        signal_returns = signal_returns[np.isfinite(signal_returns)]
        sharpe = np.mean(signal_returns) / np.std(signal_returns) * np.sqrt(52) if np.std(signal_returns) > 0 else 0
        print(f"  シグナル週リターン: mean={np.mean(signal_returns):.1f}bps, "
              f"Sharpe={sharpe:.2f}, 勝率={(signal_returns > 0).mean()*100:.1f}%")

    # =====================================================================
    # 図
    # =====================================================================
    fig = plt.figure(figsize=(15, 10), facecolor='white')
    plt.rcParams.update({
        'font.family': ['Hiragino Sans', 'IPAexGothic', 'sans-serif'],
        'axes.unicode_minus': False,
    })
    fig.suptitle('Task A 投資部門別売買戦略 OOS検証 (Train: 2016-2022 / OOS: 2023-2026)',
                 fontsize=13, fontweight='bold', y=0.99)

    # 上: シグナル別 Train vs OOS 比較
    ax1 = fig.add_axes([0.06, 0.55, 0.55, 0.38])
    n_sigs = len(res_df)
    x = np.arange(n_sigs)
    width = 0.35
    bars1 = ax1.bar(x - width/2, res_df['train_mean'], width, color='#1565C0',
                    alpha=0.85, label='Train (2016-2022)')
    bars2 = ax1.bar(x + width/2, res_df['oos_mean'], width, color='#FF9800',
                    alpha=0.85, label='OOS (2023-2026.5)')
    for i, (_, row) in enumerate(res_df.iterrows()):
        ax1.text(i - width/2, row['train_mean'] + (2 if row['train_mean']>=0 else -2),
                 f"N={row['train_N']}\nWR={row['train_wr']}",
                 ha='center', fontsize=6.5,
                 va='bottom' if row['train_mean']>=0 else 'top')
        ax1.text(i + width/2, row['oos_mean'] + (2 if row['oos_mean']>=0 else -2),
                 f"N={row['oos_N']}\nWR={row['oos_wr']}",
                 ha='center', fontsize=6.5,
                 va='bottom' if row['oos_mean']>=0 else 'top')
    ax1.set_xticks(x)
    ax1.set_xticklabels([s.replace(' Long','').replace(' Short','↓').replace(' (緩)','').replace('海外勢','海外').replace('海外-個人','海個') for s in res_df['signal']],
                        rotation=20, ha='right', fontsize=8)
    ax1.axhline(0, color='black', lw=0.6)
    ax1.set_ylabel('mean リターン (bps/週)', fontsize=9)
    ax1.set_title('Train vs OOS — シグナル別パフォーマンス', fontsize=10, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(axis='y', alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # 右上: 勝率の Train vs OOS 比較
    ax2 = fig.add_axes([0.65, 0.55, 0.32, 0.38])
    ax2.plot(res_df['train_wr'], res_df['oos_wr'], 'o', color='#1565C0', markersize=10)
    for i, row in res_df.iterrows():
        ax2.annotate(row['signal'][:14], (row['train_wr'], row['oos_wr']),
                     textcoords='offset points', xytext=(5, 5), fontsize=7)
    ax2.plot([30, 90], [30, 90], 'k--', lw=0.5, alpha=0.5)
    ax2.axhline(50, color='gray', lw=0.5, linestyle=':')
    ax2.axvline(50, color='gray', lw=0.5, linestyle=':')
    ax2.set_xlabel('Train 勝率 (%)', fontsize=9)
    ax2.set_ylabel('OOS 勝率 (%)', fontsize=9)
    ax2.set_title('Train vs OOS 勝率の整合性', fontsize=10, fontweight='bold')
    ax2.grid(alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # 下: OOS エクイティカーブ
    ax3 = fig.add_axes([0.06, 0.07, 0.91, 0.40])
    ax3.plot(oos_df_copy['pub_date'], oos_df_copy['cum_pnl'], color='#1565C0',
             lw=1.5, label='合成シグナル戦略 (シグナル週のみTOPIX Long)')
    ax3.fill_between(oos_df_copy['pub_date'], 0, oos_df_copy['cum_pnl'].values,
                      alpha=0.15, color='#1565C0')
    ax3.plot(oos_df_copy['pub_date'], oos_df_copy['cum_topix'], color='gray',
             lw=1.0, alpha=0.6, label='TOPIX (常時Long参考)')
    ax3.axhline(0, color='black', lw=0.6)
    ax3.set_ylabel('累積リターン (bps, 週次)', fontsize=9)
    ax3.set_title(f'OOS エクイティカーブ (2023〜2026.5): 合成戦略 {total_pnl:.0f}bps vs TOPIX',
                  fontsize=10, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%y/%m'))
    ax3.grid(alpha=0.3)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    fig.text(0.99, 0.005,
             'データ: investor_types (TSEPrime/TSE1st) | Walk-forward Z-score (前26週基準, look-aheadなし)',
             ha='right', va='bottom', fontsize=7, color='gray')
    plt.savefig('result.png', dpi=100, bbox_inches='tight', facecolor='white')
    print(f"\nresult.png 保存完了 ({time.time()-t0:.1f}秒)")


if __name__ == '__main__':
    main()
