"""
セクター LS / ペア戦略 OOS 厳格検証

Train: 2024-05-10 〜 2025-12-31 (20か月)
OOS:   2026-01-01 〜 2026-05-19 (5か月)

検証方法:
  1. Train期間で「各ペア/戦略の最適時刻」を求める
  2. 同じ設定を OOS期間に適用
  3. Train Sharpe vs OOS Sharpe を比較
  4. 過学習度を測定

検証対象:
  - 6ペア (前回特定)
  - Best-Worst (Reversion / Momentum)
"""
import psycopg2
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

PG_CONFIG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}
COST_BPS = 8

SECTORS = {
    '非鉄': {'57060': '三井金属', '57110': '三菱マテ', '57130': '住友鉱山',
            '57140': 'DOWA', '58010': '古河電工', '58020': '住友電工', '58030': 'フジクラ'},
    'AI半導体': {'61460': 'ディスコ', '65260': 'ソシオネクスト', '68570': 'アドバンテスト',
                '69200': 'レーザーテック', '69630': 'ローム', '69760': '太陽誘電', '80350': '東京エレクトロン'},
}

ENTRY_TIMES = []
for h, m in [(9, 10), (9, 20), (9, 30), (9, 40), (9, 50),
             (10, 0), (10, 10), (10, 20), (10, 30), (10, 40), (10, 50),
             (11, 0), (11, 10), (11, 20),
             (12, 40), (12, 50),
             (13, 0), (13, 10), (13, 20), (13, 30), (13, 40), (13, 50),
             (14, 0), (14, 10), (14, 20), (14, 30), (14, 40), (14, 50)]:
    ENTRY_TIMES.append(f'{h:02d}:{m:02d}:00')

CACHE_SNAPS = Path('/Users/Yusuke/claude-code/japan-stocks/.claude/worktrees/vibrant-mccarthy-d4c865/analyses/20260513_entry_time_grid/snapshots.parquet')

TRAIN_END = pd.Timestamp('2025-12-31').date()


def stats_summary(arr, cost=COST_BPS):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 10:
        return dict(N=len(arr), mean=np.nan, t=np.nan, sharpe=np.nan, wr=np.nan)
    net = arr - cost
    t, _ = stats.ttest_1samp(arr, 0)
    sh = net.mean() / arr.std() * np.sqrt(252) if arr.std() > 0 else 0
    return dict(N=len(arr),
                mean=round(net.mean(), 2),
                t=round(t, 2),
                sharpe=round(sh, 2),
                wr=round((arr > 0).mean() * 100, 1))


def compute_pair_returns(snaps, code_a, code_b, entry_t, threshold=50):
    pc = f'p_{entry_t}'
    sa = snaps[snaps['code'] == code_a][['date', pc, 'day_open', 'day_close']].copy()
    sb = snaps[snaps['code'] == code_b][['date', pc, 'day_open', 'day_close']].copy()
    sa['ret_a'] = (sa[pc] / sa['day_open'] - 1) * 10000
    sa['fwd_a'] = (sa['day_close'] / sa[pc] - 1) * 10000
    sb['ret_b'] = (sb[pc] / sb['day_open'] - 1) * 10000
    sb['fwd_b'] = (sb['day_close'] / sb[pc] - 1) * 10000
    m = sa[['date', 'ret_a', 'fwd_a']].merge(sb[['date', 'ret_b', 'fwd_b']], on='date').dropna()
    m['spread'] = m['ret_a'] - m['ret_b']
    sig = m[m['spread'].abs() >= threshold].copy()
    sig['ls_ret'] = np.where(sig['spread'] > 0, sig['fwd_b'] - sig['fwd_a'], sig['fwd_a'] - sig['fwd_b'])
    return sig[['date', 'ls_ret']]


def compute_bw_returns(sec_snaps, entry_t, direction='Reversion'):
    pc = f'p_{entry_t}'
    sub = sec_snaps.dropna(subset=[pc, 'day_open', 'day_close']).copy()
    sub['cum_ret'] = (sub[pc] / sub['day_open'] - 1) * 10000
    sub['fwd'] = (sub['day_close'] / sub[pc] - 1) * 10000
    sub['rank'] = sub.groupby('date')['cum_ret'].rank()
    sub['max_rank'] = sub.groupby('date')['rank'].transform('max')
    worst = sub[sub['rank'] == 1].set_index('date')['fwd']
    best = sub[sub['rank'] == sub['max_rank']].set_index('date')['fwd']
    if direction == 'Reversion':
        spread = (worst - best).dropna()
    else:
        spread = (best - worst).dropna()
    return spread.reset_index().rename(columns={'fwd': 'ls_ret'})


def main():
    import time
    t0 = time.time()

    print(f"snapshots ロード: {CACHE_SNAPS}")
    snaps = pd.read_parquet(CACHE_SNAPS)
    snaps['date'] = pd.to_datetime(snaps['date']).dt.date
    print(f"  {len(snaps):,} (date,code) snapshots")

    train_snaps = snaps[snaps['date'] <= TRAIN_END].copy()
    oos_snaps = snaps[snaps['date'] > TRAIN_END].copy()
    print(f"\n  Train: {train_snaps['date'].nunique()}営業日 ({train_snaps['date'].min()} 〜 {train_snaps['date'].max()})")
    print(f"  OOS  : {oos_snaps['date'].nunique()}営業日 ({oos_snaps['date'].min()} 〜 {oos_snaps['date'].max()})")

    # =====================================================================
    # ペア OOS検証
    # =====================================================================
    pair_targets = [
        ('AI半導体', '65260', '69630', 'ソシオネクスト-ローム'),
        ('AI半導体', '65260', '80350', 'ソシオネクスト-東エレ'),
        ('非鉄', '57110', '57130', '三菱マテ-住友鉱山'),
        ('AI半導体', '69630', '69760', 'ローム-太陽誘電'),
        ('非鉄', '57060', '57130', '三井金属-住友鉱山'),
        ('非鉄', '57130', '57140', '住友鉱山-DOWA'),
    ]

    print(f"\n{'='*100}")
    print("ペア OOS 検証")
    print(f"{'='*100}")
    print(f"{'ペア':<22} | {'Train最適時刻':<14} | {'Train':<27} | {'OOS同時刻':<27} | {'OOS最適':<23}")
    print("-"*120)

    pair_results = []
    for sec, ca, cb, label in pair_targets:
        # Train で最適時刻を探す
        best_train = None
        best_train_sh = -10
        for et in ENTRY_TIMES:
            arr = compute_pair_returns(train_snaps, ca, cb, et, threshold=50)
            if len(arr) < 30:
                continue
            r = stats_summary(arr['ls_ret'].values)
            if r['sharpe'] > best_train_sh:
                best_train_sh = r['sharpe']
                best_train = {'time': et, **r}

        # 同じ時刻で OOS をテスト
        if best_train is None:
            continue
        oos_arr = compute_pair_returns(oos_snaps, ca, cb, best_train['time'], threshold=50)
        oos_r = stats_summary(oos_arr['ls_ret'].values) if len(oos_arr) > 0 else dict(N=0, mean=np.nan, sharpe=np.nan, wr=np.nan, t=np.nan)

        # OOS 最適時刻 (参考)
        best_oos = None
        best_oos_sh = -10
        for et in ENTRY_TIMES:
            arr = compute_pair_returns(oos_snaps, ca, cb, et, threshold=50)
            if len(arr) < 10:
                continue
            r = stats_summary(arr['ls_ret'].values)
            if r['sharpe'] > best_oos_sh:
                best_oos_sh = r['sharpe']
                best_oos = {'time': et, **r}

        train_str = f"Sh={best_train['sharpe']:>5.2f}, N={best_train['N']:>3}, t={best_train['t']:>5.2f}"
        oos_str = f"Sh={oos_r['sharpe'] if pd.notna(oos_r['sharpe']) else 0:>5.2f}, N={oos_r['N']:>3}, t={oos_r['t'] if pd.notna(oos_r['t']) else 0:>5.2f}"
        oos_best_str = f"{best_oos['time']} Sh={best_oos['sharpe']:>5.2f}" if best_oos else 'N/A'
        print(f"{label:<22} | {best_train['time']:<14} | {train_str:<27} | {oos_str:<27} | {oos_best_str}")

        pair_results.append({
            'pair': label, 'sector': sec,
            'train_time': best_train['time'],
            'train_sharpe': best_train['sharpe'],
            'train_N': best_train['N'],
            'train_t': best_train['t'],
            'train_mean': best_train['mean'],
            'train_wr': best_train['wr'],
            'oos_sharpe': oos_r['sharpe'],
            'oos_N': oos_r['N'],
            'oos_t': oos_r['t'],
            'oos_mean': oos_r['mean'],
            'oos_wr': oos_r['wr'],
            'oos_best_time': best_oos['time'] if best_oos else None,
            'oos_best_sharpe': best_oos['sharpe'] if best_oos else np.nan,
        })

    pair_df = pd.DataFrame(pair_results)
    pair_df.to_csv('pair_oos.csv', index=False)

    # =====================================================================
    # Best-Worst OOS
    # =====================================================================
    print(f"\n{'='*100}")
    print("Best-Worst OOS 検証")
    print(f"{'='*100}")
    print(f"{'設定':<32} | {'Train最適時刻':<14} | {'Train':<27} | {'OOS同時刻':<27}")
    print("-"*110)

    bw_results = []
    for sec_name, codes_dict in SECTORS.items():
        train_sec = train_snaps[train_snaps['code'].isin(codes_dict.keys())]
        oos_sec = oos_snaps[oos_snaps['code'].isin(codes_dict.keys())]
        for direction in ['Reversion', 'Momentum']:
            # Train で最適時刻
            best_train = None
            best_sh = -10
            for et in ENTRY_TIMES:
                arr = compute_bw_returns(train_sec, et, direction)
                r = stats_summary(arr['ls_ret'].values)
                if pd.notna(r['sharpe']) and r['sharpe'] > best_sh:
                    best_sh = r['sharpe']
                    best_train = {'time': et, **r}
            if best_train is None:
                continue
            # OOS 同時刻
            oos_arr = compute_bw_returns(oos_sec, best_train['time'], direction)
            oos_r = stats_summary(oos_arr['ls_ret'].values)

            label = f'{sec_name} BW {direction}'
            train_str = f"Sh={best_train['sharpe']:>5.2f}, N={best_train['N']:>3}, t={best_train['t']:>5.2f}"
            oos_str = f"Sh={oos_r['sharpe'] if pd.notna(oos_r['sharpe']) else 0:>5.2f}, N={oos_r['N']:>3}, t={oos_r['t'] if pd.notna(oos_r['t']) else 0:>5.2f}"
            print(f"{label:<32} | {best_train['time']:<14} | {train_str:<27} | {oos_str:<27}")

            bw_results.append({
                'strategy': label, 'sector': sec_name, 'direction': direction,
                'train_time': best_train['time'],
                'train_sharpe': best_train['sharpe'], 'train_N': best_train['N'],
                'train_t': best_train['t'], 'train_mean': best_train['mean'],
                'oos_sharpe': oos_r['sharpe'], 'oos_N': oos_r['N'],
                'oos_t': oos_r['t'], 'oos_mean': oos_r['mean'],
            })

    bw_df = pd.DataFrame(bw_results)
    bw_df.to_csv('bw_oos.csv', index=False)

    # =====================================================================
    # 統計サマリー
    # =====================================================================
    print(f"\n{'='*60}")
    print("OOS 過学習度サマリー")
    print(f"{'='*60}")

    # 全ペア+BW で集計
    all_results = []
    for _, r in pair_df.iterrows():
        all_results.append({
            'name': r['pair'], 'type': 'Pair',
            'train_sh': r['train_sharpe'], 'oos_sh': r['oos_sharpe'],
        })
    for _, r in bw_df.iterrows():
        all_results.append({
            'name': r['strategy'], 'type': 'BW',
            'train_sh': r['train_sharpe'], 'oos_sh': r['oos_sharpe'],
        })
    all_df = pd.DataFrame(all_results).dropna()
    train_avg = all_df['train_sh'].mean()
    oos_avg = all_df['oos_sh'].mean()
    n_robust = ((all_df['oos_sh'] > 0.5)).sum()
    n_failed = (all_df['oos_sh'] < 0).sum()
    print(f"  Train Sharpe 平均: {train_avg:.2f}")
    print(f"  OOS Sharpe 平均:   {oos_avg:.2f}")
    print(f"  劣化幅: {oos_avg - train_avg:+.2f}")
    print(f"  OOS Sharpe > 0.5 (頑健): {n_robust}/{len(all_df)}")
    print(f"  OOS Sharpe < 0   (失敗): {n_failed}/{len(all_df)}")

    # =====================================================================
    # 図
    # =====================================================================
    fig = plt.figure(figsize=(15, 9), facecolor='white')
    plt.rcParams.update({
        'font.family': ['Hiragino Sans', 'IPAexGothic', 'sans-serif'],
        'axes.unicode_minus': False,
    })
    fig.suptitle('OOS 厳格検証: Train (2024-05〜2025-12) vs OOS (2026-01〜2026-05)',
                 fontsize=13, fontweight='bold', y=0.99)

    # 左上: ペア Train vs OOS Sharpe
    ax1 = fig.add_axes([0.05, 0.55, 0.55, 0.38])
    n = len(pair_df)
    x = np.arange(n)
    width = 0.35
    ax1.bar(x - width/2, pair_df['train_sharpe'], width, color='#1565C0',
            alpha=0.85, label='Train Sharpe')
    ax1.bar(x + width/2, pair_df['oos_sharpe'].fillna(0), width, color='#FF9800',
            alpha=0.85, label='OOS Sharpe (同時刻)')
    for i, row in pair_df.iterrows():
        ax1.text(i - width/2, row['train_sharpe'] + 0.05,
                 f"{row['train_sharpe']:.2f}\n@{row['train_time'][:5]}",
                 ha='center', fontsize=7)
        if pd.notna(row['oos_sharpe']):
            ax1.text(i + width/2, row['oos_sharpe'] + (0.05 if row['oos_sharpe'] >= 0 else -0.05),
                     f"{row['oos_sharpe']:.2f}\nN={row['oos_N']}",
                     ha='center', fontsize=7,
                     va='bottom' if row['oos_sharpe'] >= 0 else 'top')
    ax1.set_xticks(x)
    ax1.set_xticklabels(pair_df['pair'], rotation=20, ha='right', fontsize=8)
    ax1.axhline(0, color='black', lw=0.6)
    ax1.set_ylabel('Sharpe (年率)', fontsize=9)
    ax1.set_title('ペア戦略 Train vs OOS', fontsize=10, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(axis='y', alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # 右上: BW Train vs OOS
    ax2 = fig.add_axes([0.65, 0.55, 0.32, 0.38])
    n_bw = len(bw_df)
    x_bw = np.arange(n_bw)
    ax2.bar(x_bw - width/2, bw_df['train_sharpe'], width, color='#1565C0',
            alpha=0.85, label='Train Sharpe')
    ax2.bar(x_bw + width/2, bw_df['oos_sharpe'].fillna(0), width, color='#FF9800',
            alpha=0.85, label='OOS Sharpe')
    for i, row in bw_df.iterrows():
        ax2.text(i - width/2, row['train_sharpe'] + (0.05 if row['train_sharpe']>=0 else -0.05),
                 f"{row['train_sharpe']:.2f}", ha='center', fontsize=7,
                 va='bottom' if row['train_sharpe']>=0 else 'top')
        if pd.notna(row['oos_sharpe']):
            ax2.text(i + width/2, row['oos_sharpe'] + (0.05 if row['oos_sharpe']>=0 else -0.05),
                     f"{row['oos_sharpe']:.2f}", ha='center', fontsize=7,
                     va='bottom' if row['oos_sharpe']>=0 else 'top')
    ax2.set_xticks(x_bw)
    ax2.set_xticklabels([s.replace(' BW ', '\n') for s in bw_df['strategy']],
                        rotation=0, ha='center', fontsize=7.5)
    ax2.axhline(0, color='black', lw=0.6)
    ax2.set_ylabel('Sharpe', fontsize=9)
    ax2.set_title('Best-Worst Spread Train vs OOS', fontsize=10, fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(axis='y', alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # 下: Train Sharpe vs OOS Sharpe スキャッター (過学習度)
    ax3 = fig.add_axes([0.05, 0.07, 0.40, 0.40])
    colors_map = {'Pair': '#1565C0', 'BW': '#FF9800'}
    for _, row in all_df.iterrows():
        c = colors_map.get(row['type'], 'gray')
        ax3.scatter(row['train_sh'], row['oos_sh'], s=100, color=c, alpha=0.8,
                    edgecolor='black', lw=0.5)
        ax3.annotate(row['name'][:12], (row['train_sh'], row['oos_sh']),
                     textcoords='offset points', xytext=(5, 5), fontsize=7)
    # 対角線
    ax3.plot([-2, 2], [-2, 2], 'k--', lw=0.5, alpha=0.5, label='Train = OOS')
    ax3.axhline(0, color='black', lw=0.5)
    ax3.axvline(0, color='black', lw=0.5)
    ax3.axhline(0.5, color='green', lw=0.5, linestyle=':', alpha=0.5, label='OOS Sh=0.5')
    ax3.set_xlabel('Train Sharpe', fontsize=9)
    ax3.set_ylabel('OOS Sharpe', fontsize=9)
    ax3.set_title('Train vs OOS 過学習度', fontsize=10, fontweight='bold')
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    # 下右: サマリーテーブル
    ax4 = fig.add_axes([0.50, 0.07, 0.47, 0.40])
    ax4.axis('off')
    summary_df = pair_df.copy()
    summary_df = summary_df[['pair', 'train_time', 'train_sharpe', 'oos_sharpe', 'oos_N', 'oos_t']]
    summary_df.columns = ['ペア', '最適時刻', 'Train Sh', 'OOS Sh', 'OOS N', 'OOS t']
    table = ax4.table(cellText=summary_df.values, colLabels=summary_df.columns,
                      cellLoc='center', loc='upper center',
                      bbox=[0, 0.1, 1, 0.85])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor('#1565C0')
            cell.set_text_props(color='white', fontweight='bold')
        elif r > 0 and c == 3 and r-1 < len(summary_df):  # OOS Sharpe column
            try:
                v = float(summary_df.iloc[r-1, c])
                if v > 0.5:
                    cell.set_facecolor('#C8E6C9')
                elif v < 0:
                    cell.set_facecolor('#FFCDD2')
                else:
                    cell.set_facecolor('#FFF9C4')
            except: pass
        cell.set_edgecolor('#BDBDBD')
    ax4.set_title('OOS Sharpe による判定 (緑=頑健, 黄=境界, 赤=失敗)',
                  fontsize=10, fontweight='bold', y=0.96)

    fig.text(0.99, 0.005,
             f'Train: 2024-05〜2025-12 / OOS: 2026-01〜2026-05 | コスト8bps (LS往復) | spread閾値50bps',
             ha='right', va='bottom', fontsize=7, color='gray')
    plt.savefig('result.png', dpi=100, bbox_inches='tight', facecolor='white')
    print(f"\nresult.png 保存完了 ({time.time()-t0:.1f}秒)")


if __name__ == '__main__':
    main()
