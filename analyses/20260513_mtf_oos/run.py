"""
MTF戦略 OOS 厳格検証

Train: 2024-05-10 〜 2025-12-31 (約20か月)
OOS:   2026-01-01 〜 2026-05-19 (約5か月)

検証項目:
  1. 22ルール全てを Train / OOS 別々に評価
  2. Sharpe / mean_net / 勝率の劣化を測定
  3. 過学習度: |Train_Sh - OOS_Sh|
  4. ベスト戦略の合成エクイティカーブ
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
CACHE_MTF = Path('/Users/Yusuke/claude-code/japan-stocks/.claude/worktrees/vibrant-mccarthy-d4c865/analyses/20260513_mtf_strategy/mtf_features.parquet')
TRAIN_END = pd.Timestamp('2025-12-31').date()


def stats_summary(arr, cost=COST_BPS):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n < 10:
        return dict(N=n, mean=np.nan, t=np.nan, sharpe=np.nan, wr=np.nan)
    net = arr - cost
    t, _ = stats.ttest_1samp(arr, 0)
    sh = net.mean() / arr.std() * np.sqrt(252) if arr.std() > 0 else 0
    return dict(N=n, mean=round(net.mean(), 1),
                t=round(t, 2), sharpe=round(sh, 2),
                wr=round((arr > 0).mean()*100, 1))


def evaluate_rule(df, cond_func, target_col, label, direction=1):
    """ある期間データでルール評価"""
    cond = cond_func(df)
    sub = df[cond & df[target_col].notna()]
    if len(sub) == 0:
        return None
    arr = sub[target_col].values * direction
    return stats_summary(arr)


# 22ルール定義
RULES = [
    # ↓↓↓ (条件関数, target, label, direction) ↓↓↓
    (lambda d: (d['week_ret'] > 200) & (d['daily_ret'] > 0) & (d['gap_pct'] > 50),
     'fwd_close_10:00:00', 'A1: 週up+日up+ギャップup → Long', 1),
    (lambda d: (d['week_ret'] > 300) & (d['cum_ret_10:00:00'] > 30),
     'fwd_close_10:00:00', 'A2: 週強up + 10:00 up → Long', 1),
    (lambda d: (d['week_ret'] > 200) & (d['daily_ret'] > 0) & (d['cum_ret_13:00:00'] > 0),
     'fwd_close_13:00:00', 'A3: 週up+日up+13:00 up → Long', 1),
    (lambda d: (d['week_ret'] < -200) & (d['daily_ret'] < 0) & (d['gap_pct'] < -50),
     'fwd_close_10:00:00', 'A4: 週down+日down+ギャップdown → Short', -1),

    (lambda d: (d['week_ret'] > 200) & (d['daily_ret'] < -150) & (d['cum_ret_10:00:00'] > 0),
     'fwd_close_10:00:00', 'B1: 週up + 前日下落 + 朝反発 → Long', 1),
    (lambda d: (d['week_ret'] > 200) & (d['gap_pct'] < -50),
     'fwd_close_10:00:00', 'B2: 週up + ギャップ↓ → Long', 1),
    (lambda d: (d['dist_ma50'] > 0) & (d['rsi14'] < 30) & (d['cum_ret_10:00:00'] > 0),
     'fwd_close_10:00:00', 'B3: 50日MA上 + RSI<30 + 朝反発 → Long', 1),

    (lambda d: (d['week_ret'] > 500) & (d['gap_pct'] > 100),
     'fwd_close_10:00:00', 'C1: 週過熱 + ギャップ↑ → Short', -1),
    (lambda d: (d['month_ret'] > 500) & (d['gap_pct'] > 100),
     'fwd_close_10:00:00', 'C2: 月過熱 + ギャップ↑ → Short', -1),
    (lambda d: (d['rsi14'] > 70) & (d['cum_ret_10:00:00'] > 50),
     'fwd_close_10:00:00', 'C3: RSI>70 + 朝強 → Short', -1),
    (lambda d: (d['week_ret'] < -500) & (d['gap_pct'] < -100),
     'fwd_close_10:00:00', 'C4: 週深売り + ギャップ↓ → Long (逆張)', 1),

    (lambda d: (d['week_ret'] < -200) & (d['daily_ret'] > 100) & (d['gap_pct'] > 50),
     'fwd_close_10:00:00', 'D1: 週下落 + 日反転 + ギャップ↑ → Long', 1),
    (lambda d: (d['month_ret'] < -500) & (d['daily_ret'] > 200),
     'fwd_close_10:00:00', 'D2: 月下落 + 日反転 → Long', 1),

    (lambda d: (d['week_ret'] > 200) & (d['daily_ret'] > 100),
     'next_day_full', 'E1: 週up + 日up → 翌日 Long', 1),
    (lambda d: (d['week_ret'] < -200) & (d['daily_ret'] < -100),
     'next_day_full', 'E2: 週down + 日down → 翌日 Short', -1),
    (lambda d: (d['week_ret'] > 200) & (d['daily_ret'] < -100),
     'next_day_full', 'E3: 週up + 日down → 翌日 Long', 1),
    (lambda d: (d['week_ret'] < -200) & (d['daily_ret'] > 100),
     'next_day_full', 'E4: 週down + 日up → 翌日 Short', -1),
    (lambda d: (d['dist_ma50'] > 0) & (d['rsi14'] < 30),
     'next_day_full', 'E5: 50日MA上 + RSI<30 → 翌日 Long', 1),

    (lambda d: (d['week_ret'] > 200) & (d['month_ret'] > 0) & (d['daily_ret'] < -100) & (d['cum_ret_10:00:00'] > 0),
     'fwd_close_10:00:00', 'F1: 週up+月up+日下落+朝反発 → Long', 1),
    (lambda d: (d['week_ret'] < -200) & (d['month_ret'] < 0) & (d['daily_ret'] > 100) & (d['cum_ret_10:00:00'] < 0),
     'fwd_close_10:00:00', 'F2: 週down+月down+日上昇+朝失速 → Short', -1),
]


def main():
    import time
    t0 = time.time()

    print(f"MTFキャッシュロード: {CACHE_MTF}")
    df = pd.read_parquet(CACHE_MTF)
    df['date'] = pd.to_datetime(df['date']).dt.date
    print(f"  {len(df):,} 観測")

    train = df[df['date'] <= TRAIN_END].copy()
    oos = df[df['date'] > TRAIN_END].copy()
    print(f"\n  Train: {len(train):,} 行 ({train['date'].nunique()}営業日)")
    print(f"  OOS  : {len(oos):,} 行 ({oos['date'].nunique()}営業日)")

    # 各ルールを Train / OOS で評価
    print(f"\n{'='*120}")
    print(f"{'Rule':<48} | {'Train':<32} | {'OOS':<32} | 過学習度")
    print('-'*120)

    rows = []
    for cond_func, target, label, direction in RULES:
        tr = evaluate_rule(train, cond_func, target, label, direction)
        os = evaluate_rule(oos, cond_func, target, label, direction)
        if tr is None or os is None:
            continue
        tr_str = f"Sh={tr['sharpe']:>5.2f}, N={tr['N']:>4}, t={tr['t']:>5.2f}"
        os_str = f"Sh={os['sharpe'] if pd.notna(os['sharpe']) else 0:>5.2f}, N={os['N']:>3}, t={os['t'] if pd.notna(os['t']) else 0:>5.2f}"
        decay = (tr['sharpe'] or 0) - (os['sharpe'] or 0)
        decay_str = f"{decay:+.2f}"
        print(f"{label:<48} | {tr_str:<32} | {os_str:<32} | {decay_str}")
        rows.append({
            'rule': label,
            'train_N': tr['N'], 'train_mean': tr['mean'], 'train_t': tr['t'],
            'train_sharpe': tr['sharpe'], 'train_wr': tr['wr'],
            'oos_N': os['N'], 'oos_mean': os['mean'], 'oos_t': os['t'],
            'oos_sharpe': os['sharpe'], 'oos_wr': os['wr'],
            'decay': decay,
        })

    res_df = pd.DataFrame(rows)
    res_df.to_csv('mtf_oos_results.csv', index=False)

    # ========================================
    # サマリー
    # ========================================
    print(f"\n{'='*60}")
    print("過学習度サマリー")
    print(f"{'='*60}")
    valid = res_df.dropna(subset=['train_sharpe', 'oos_sharpe'])
    print(f"  検証ルール数: {len(valid)}")
    print(f"  Train Sharpe 平均: {valid['train_sharpe'].mean():.2f}")
    print(f"  OOS Sharpe 平均:   {valid['oos_sharpe'].mean():.2f}")
    print(f"  劣化平均: {valid['decay'].mean():+.2f}")
    print(f"  Train Sh>2 で OOS Sh>1 維持: {((valid['train_sharpe'] > 2) & (valid['oos_sharpe'] > 1)).sum()}")
    print(f"  Train で勝者 + OOSで勝者: {((valid['train_sharpe'] > 0) & (valid['oos_sharpe'] > 0)).sum()}/{len(valid)}")
    print(f"  Train で勝者 + OOSで敗者 (過学習): {((valid['train_sharpe'] > 0) & (valid['oos_sharpe'] < 0)).sum()}")

    # ========================================
    # ベスト戦略のエクイティカーブ (OOS期間)
    # ========================================
    print("\nベスト3戦略のOOSエクイティカーブ計算...")
    best_strategies = res_df.dropna().sort_values('oos_sharpe', ascending=False).head(3)
    print("\n=== ベスト3 (OOS Sharpe降順) ===")
    print(best_strategies[['rule','train_sharpe','oos_sharpe','oos_N','oos_t']].to_string(index=False))

    # 各ベスト戦略のOOS PnL
    fig = plt.figure(figsize=(16, 12), facecolor='white')
    plt.rcParams.update({
        'font.family': ['Hiragino Sans', 'IPAexGothic', 'sans-serif'],
        'axes.unicode_minus': False,
    })
    fig.suptitle('MTF戦略 OOS厳格検証 (Train: 2024-05〜2025-12 / OOS: 2026-01〜2026-05)',
                 fontsize=13, fontweight='bold', y=0.99)

    # 1. Train vs OOS Sharpe 全戦略
    ax1 = fig.add_axes([0.05, 0.55, 0.55, 0.40])
    n_rules = len(res_df)
    x = np.arange(n_rules)
    width = 0.4
    sorted_df = res_df.sort_values('train_sharpe', ascending=False).reset_index(drop=True)
    ax1.bar(x - width/2, sorted_df['train_sharpe'], width, color='#1565C0', alpha=0.85, label='Train Sharpe')
    ax1.bar(x + width/2, sorted_df['oos_sharpe'].fillna(0), width, color='#FF9800', alpha=0.85, label='OOS Sharpe')
    ax1.set_xticks(x)
    ax1.set_xticklabels([s[:14] for s in sorted_df['rule']], rotation=80, ha='right', fontsize=6.5)
    ax1.axhline(0, color='black', lw=0.6)
    ax1.set_ylabel('Sharpe', fontsize=9)
    ax1.set_title('全ルール Train vs OOS Sharpe', fontsize=10, fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.grid(axis='y', alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # 2. スキャッター
    ax2 = fig.add_axes([0.66, 0.55, 0.30, 0.40])
    for _, row in valid.iterrows():
        if row['train_sharpe'] > 0 and row['oos_sharpe'] > 0:
            c, m, s = '#43A047', '*', 150
        elif row['train_sharpe'] > 0 and row['oos_sharpe'] < 0:
            c, m, s = '#E53935', 'x', 100
        elif row['train_sharpe'] < 0 and row['oos_sharpe'] < 0:
            c, m, s = '#9E9E9E', 'o', 50
        else:
            c, m, s = '#FF9800', 's', 80
        ax2.scatter(row['train_sharpe'], row['oos_sharpe'], color=c, marker=m, s=s,
                    alpha=0.85, edgecolor='black', lw=0.4)
    # 対角線
    minv, maxv = -2, 12
    ax2.plot([minv, maxv], [minv, maxv], 'k--', lw=0.5, alpha=0.5, label='Train = OOS')
    ax2.axhline(0, color='black', lw=0.5)
    ax2.axvline(0, color='black', lw=0.5)
    ax2.set_xlabel('Train Sharpe', fontsize=9)
    ax2.set_ylabel('OOS Sharpe', fontsize=9)
    ax2.set_title('Train vs OOS スキャッター', fontsize=10, fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # 3. ベスト戦略のOOS Equity
    ax3 = fig.add_axes([0.05, 0.05, 0.55, 0.40])
    colors = ['#1565C0', '#FF9800', '#43A047']
    for idx, (i, row) in enumerate(best_strategies.iterrows()):
        # ルール再評価して PnL を時系列取得
        label = row['rule']
        rule_idx = next(j for j, r in enumerate(RULES) if r[2] == label)
        cond_func, target, _, direction = RULES[rule_idx]

        sub = oos[cond_func(oos) & oos[target].notna()].copy()
        sub['date'] = pd.to_datetime(sub['date'])
        sub['ret'] = sub[target] * direction - COST_BPS
        sub = sub.sort_values('date')
        # 日次集計
        daily_pnl = sub.groupby('date')['ret'].sum()
        cum = daily_pnl.cumsum()
        if len(cum) > 0:
            ax3.plot(cum.index, cum.values, lw=1.5, color=colors[idx],
                     label=f'{label[:30]} (OOS Sh={row["oos_sharpe"]:.2f})')

    ax3.axhline(0, color='black', lw=0.6)
    ax3.set_ylabel('累積 net PnL (bps)', fontsize=9)
    ax3.set_title('OOS期間ベスト3戦略のエクイティカーブ', fontsize=10, fontweight='bold')
    ax3.legend(fontsize=9, loc='best')
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax3.grid(alpha=0.3)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    # 4. テーブル: Top10 OOS Sharpe
    ax4 = fig.add_axes([0.66, 0.05, 0.30, 0.40])
    ax4.axis('off')
    top = res_df.dropna().sort_values('oos_sharpe', ascending=False).head(10)[
        ['rule', 'train_sharpe', 'oos_sharpe', 'oos_N']
    ].copy()
    top.columns = ['ルール', 'Train Sh', 'OOS Sh', 'OOS N']
    top['ルール'] = top['ルール'].str[:24]
    table = ax4.table(cellText=top.values, colLabels=top.columns,
                      cellLoc='center', loc='upper center',
                      bbox=[0, 0.05, 1, 0.92])
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor('#1565C0')
            cell.set_text_props(color='white', fontweight='bold')
        elif r > 0 and c == 2:  # OOS Sharpe column
            try:
                v = float(top.iloc[r-1, c])
                if v > 2:
                    cell.set_facecolor('#A5D6A7')
                elif v > 0:
                    cell.set_facecolor('#FFF59D')
                else:
                    cell.set_facecolor('#FFCDD2')
            except: pass
        cell.set_edgecolor('#BDBDBD')
    ax4.set_title('OOS Sharpe Top10', fontsize=10, fontweight='bold', y=0.97)

    fig.text(0.99, 0.005,
             f'14銘柄 (非鉄7+AI半導体7) | Train: {train["date"].nunique()}日 / OOS: {oos["date"].nunique()}日',
             ha='right', va='bottom', fontsize=7, color='gray')
    plt.savefig('result.png', dpi=100, bbox_inches='tight', facecolor='white')
    print(f"\nresult.png 保存完了 ({time.time()-t0:.1f}秒)")


if __name__ == '__main__':
    main()
