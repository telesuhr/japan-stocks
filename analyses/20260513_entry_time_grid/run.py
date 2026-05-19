"""
判定時刻 10分刻み 全数探索

検証:
  - Best-Worst Spread (Reversion / Momentum) × 9:10〜14:50 の30個の時刻
  - 上位ペア (ソシオネクスト-ローム, 三菱マテ-住友鉱山) の判定時刻最適化
  - 出力: Sharpe ヒートマップ + 最適時刻表

セクター: 非鉄, AI半導体
保有: 判定時刻 → 大引け (15:30)
コスト: 8bps (LS往復)
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
    '非鉄': {
        '57060': '三井金属', '57110': '三菱マテ', '57130': '住友鉱山',
        '57140': 'DOWA', '58010': '古河電工', '58020': '住友電工', '58030': 'フジクラ',
    },
    'AI半導体': {
        '61460': 'ディスコ', '65260': 'ソシオネクスト', '68570': 'アドバンテスト',
        '69200': 'レーザーテック', '69630': 'ローム', '69760': '太陽誘電', '80350': '東京エレクトロン',
    },
}

# 10分刻みの判定時刻 (9:10 から 14:50 まで、ランチ11:30-12:30を除く)
ENTRY_TIMES = []
for h, m in [(9, 10), (9, 20), (9, 30), (9, 40), (9, 50),
             (10, 0), (10, 10), (10, 20), (10, 30), (10, 40), (10, 50),
             (11, 0), (11, 10), (11, 20),
             (12, 40), (12, 50),
             (13, 0), (13, 10), (13, 20), (13, 30), (13, 40), (13, 50),
             (14, 0), (14, 10), (14, 20), (14, 30), (14, 40), (14, 50)]:
    ENTRY_TIMES.append(f'{h:02d}:{m:02d}:00')

CACHE_BARS = Path('/Users/Yusuke/claude-code/japan-stocks/.claude/worktrees/vibrant-mccarthy-d4c865/analyses/20260513_sector_ls_intraday/intraday_bars.parquet')
CACHE_SNAPS = Path('snapshots.parquet')


def build_snapshots():
    """各 (date, code) × 各time × 寄付/引け を高速計算"""
    if CACHE_SNAPS.exists():
        print(f"snapshots キャッシュからロード: {CACHE_SNAPS}")
        return pd.read_parquet(CACHE_SNAPS)

    print(f"1分足ロード: {CACHE_BARS}")
    df = pd.read_parquet(CACHE_BARS)
    df['ts'] = pd.to_datetime(df['ts'])
    df['date'] = df['ts'].dt.date
    df['time_str'] = df['ts'].dt.strftime('%H:%M:%S')
    print(f"  {len(df):,} 行")

    # 各 (date, code) ごとに pivot: time_str → close
    print("snapshots 構築中...")
    # 単純に各 entry time の最新close (<=time_str)
    out = []
    for (d, code), g in df.groupby(['date', 'code']):
        g = g.sort_values('ts').reset_index(drop=True)
        if len(g) < 30:
            continue
        # 寄付
        morning = g[g['time_str'] >= '09:00:00']
        if morning.empty:
            continue
        day_open = morning.iloc[0]['open']
        day_close = g.iloc[-1]['close']
        rec = {'date': d, 'code': code, 'day_open': day_open, 'day_close': day_close}
        for t in ENTRY_TIMES:
            sub = g[g['time_str'] <= t]
            rec[f'p_{t}'] = sub.iloc[-1]['close'] if not sub.empty else np.nan
        out.append(rec)

    snaps = pd.DataFrame(out)
    snaps.to_parquet(CACHE_SNAPS)
    print(f"  保存: {CACHE_SNAPS} ({len(snaps):,} 行)")
    return snaps


def compute_bw_for_time(sec_snaps, entry_t, exit_col='day_close', direction='Reversion'):
    """Best-Worst spread at entry_t"""
    price_col = f'p_{entry_t}'
    sub = sec_snaps.dropna(subset=[price_col, 'day_open', exit_col]).copy()
    if len(sub) == 0:
        return None
    sub['cum_ret_to_entry'] = (sub[price_col] / sub['day_open'] - 1) * 10000
    sub['ret_entry_to_exit'] = (sub[exit_col] / sub[price_col] - 1) * 10000
    sub['rank'] = sub.groupby('date')['cum_ret_to_entry'].rank()
    # max rank per date
    sub['max_rank'] = sub.groupby('date')['rank'].transform('max')
    worsts = sub[sub['rank'] == 1].set_index('date')['ret_entry_to_exit']
    bests = sub[sub['rank'] == sub['max_rank']].set_index('date')['ret_entry_to_exit']
    if direction == 'Reversion':
        spreads = (worsts - bests).dropna()
    else:
        spreads = (bests - worsts).dropna()
    return spreads.values


def compute_pair_for_time(sec_snaps, code_a, code_b, entry_t, threshold=50):
    """Pair spread reversion at entry_t"""
    price_col = f'p_{entry_t}'
    sub_a = sec_snaps[sec_snaps['code'] == code_a][['date', price_col, 'day_open', 'day_close']].copy()
    sub_b = sec_snaps[sec_snaps['code'] == code_b][['date', price_col, 'day_open', 'day_close']].copy()
    sub_a['ret_a'] = (sub_a[price_col] / sub_a['day_open'] - 1) * 10000
    sub_a['fwd_a'] = (sub_a['day_close'] / sub_a[price_col] - 1) * 10000
    sub_b['ret_b'] = (sub_b[price_col] / sub_b['day_open'] - 1) * 10000
    sub_b['fwd_b'] = (sub_b['day_close'] / sub_b[price_col] - 1) * 10000
    merged = sub_a[['date', 'ret_a', 'fwd_a']].merge(sub_b[['date', 'ret_b', 'fwd_b']], on='date').dropna()
    if len(merged) < 20:
        return None
    merged['spread'] = merged['ret_a'] - merged['ret_b']
    sig = merged[merged['spread'].abs() >= threshold].copy()
    sig['ls_ret'] = np.where(sig['spread'] > 0,
                              sig['fwd_b'] - sig['fwd_a'],
                              sig['fwd_a'] - sig['fwd_b'])
    return sig['ls_ret'].values


def sharpe(arr, cost=COST_BPS):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 20:
        return np.nan, len(arr), np.nan, np.nan
    net = arr - cost
    sh = net.mean() / arr.std() * np.sqrt(252) if arr.std() > 0 else 0
    t, p = stats.ttest_1samp(arr, 0)
    return sh, len(arr), net.mean(), t


def main():
    import time
    t0 = time.time()

    snaps = build_snapshots()
    print(f"\n対象営業日: {snaps['date'].nunique()}")

    # =====================================================================
    # Best-Worst スキャン
    # =====================================================================
    print("\n=== Best-Worst Spread 時刻スキャン ===")
    bw_records = []
    for sec_name, codes_dict in SECTORS.items():
        sec_snaps = snaps[snaps['code'].isin(codes_dict.keys())].copy()
        for entry_t in ENTRY_TIMES:
            for direction in ['Reversion', 'Momentum']:
                arr = compute_bw_for_time(sec_snaps, entry_t, direction=direction)
                if arr is None:
                    continue
                sh, n, m, t_ = sharpe(arr)
                bw_records.append({
                    'sector': sec_name, 'entry': entry_t, 'direction': direction,
                    'N': n, 'mean_net': round(m, 2) if pd.notna(m) else np.nan,
                    't_stat': round(t_, 2) if pd.notna(t_) else np.nan,
                    'sharpe': round(sh, 2) if pd.notna(sh) else np.nan,
                })
    bw_df = pd.DataFrame(bw_records)
    bw_df.to_csv('bw_grid.csv', index=False)

    print("\nTop10 Best-Worst 設定 (Sharpe降順):")
    print(bw_df.dropna(subset=['sharpe']).sort_values('sharpe', ascending=False).head(10).to_string(index=False))

    # =====================================================================
    # ペアトレード スキャン (主要ペア)
    # =====================================================================
    pair_targets = [
        ('AI半導体', '65260', '69630', 'ソシオネクスト-ローム'),
        ('非鉄', '57110', '57130', '三菱マテ-住友鉱山'),
        ('AI半導体', '69630', '69760', 'ローム-太陽誘電'),
        ('非鉄', '57060', '57130', '三井金属-住友鉱山'),
        ('AI半導体', '65260', '80350', 'ソシオネクスト-東エレ'),
        ('非鉄', '57130', '57140', '住友鉱山-DOWA'),
    ]
    print("\n=== ペアトレード 時刻スキャン (主要6ペア) ===")
    pair_records = []
    for sec_name, code_a, code_b, label in pair_targets:
        sec_snaps = snaps[snaps['code'].isin([code_a, code_b])].copy()
        for entry_t in ENTRY_TIMES:
            arr = compute_pair_for_time(sec_snaps, code_a, code_b, entry_t, threshold=50)
            if arr is None:
                continue
            sh, n, m, t_ = sharpe(arr)
            pair_records.append({
                'pair': label, 'sector': sec_name, 'entry': entry_t,
                'N': n, 'mean_net': round(m, 2) if pd.notna(m) else np.nan,
                't_stat': round(t_, 2) if pd.notna(t_) else np.nan,
                'sharpe': round(sh, 2) if pd.notna(sh) else np.nan,
            })
    pair_df = pd.DataFrame(pair_records)
    pair_df.to_csv('pair_grid.csv', index=False)

    print("\nTop10 ペア×時刻 (Sharpe降順):")
    print(pair_df.dropna(subset=['sharpe']).sort_values('sharpe', ascending=False).head(10).to_string(index=False))

    # ベスト時刻ごとのペア
    print("\n各ペアの最適時刻:")
    for label in [p[3] for p in pair_targets]:
        sub = pair_df[pair_df['pair'] == label].dropna(subset=['sharpe']).sort_values('sharpe', ascending=False)
        if len(sub) > 0:
            best = sub.iloc[0]
            print(f"  {label:<20}: 最適={best['entry']}, Sharpe={best['sharpe']:.2f}, N={best['N']}, t={best['t_stat']:.2f}")

    # =====================================================================
    # 図 (4パネル)
    # =====================================================================
    fig = plt.figure(figsize=(16, 12), facecolor='white')
    plt.rcParams.update({
        'font.family': ['Hiragino Sans', 'IPAexGothic', 'sans-serif'],
        'axes.unicode_minus': False,
    })
    fig.suptitle('判定時刻 10分刻み 最適化 (非鉄 / AI半導体, 2024/5-2026/5)',
                 fontsize=13, fontweight='bold', y=0.99)

    # 1. 非鉄 Reversion/Momentum 折れ線
    ax1 = fig.add_axes([0.05, 0.56, 0.42, 0.36])
    sub = bw_df[bw_df['sector'] == '非鉄']
    for direction, color in [('Reversion', '#1565C0'), ('Momentum', '#E53935')]:
        d = sub[sub['direction'] == direction].sort_values('entry')
        ax1.plot(d['entry'], d['sharpe'], 'o-', color=color, lw=1.5, label=direction)
    ax1.axhline(0, color='black', lw=0.6)
    ax1.set_ylabel('Sharpe (年率)', fontsize=9)
    ax1.set_title('非鉄: Best-Worst Spread 時刻別 Sharpe', fontsize=10, fontweight='bold')
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=7)
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # 2. AI半導体 Reversion/Momentum 折れ線
    ax2 = fig.add_axes([0.55, 0.56, 0.42, 0.36])
    sub = bw_df[bw_df['sector'] == 'AI半導体']
    for direction, color in [('Reversion', '#1565C0'), ('Momentum', '#E53935')]:
        d = sub[sub['direction'] == direction].sort_values('entry')
        ax2.plot(d['entry'], d['sharpe'], 'o-', color=color, lw=1.5, label=direction)
    ax2.axhline(0, color='black', lw=0.6)
    ax2.set_ylabel('Sharpe (年率)', fontsize=9)
    ax2.set_title('AI半導体: Best-Worst Spread 時刻別 Sharpe', fontsize=10, fontweight='bold')
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right', fontsize=7)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # 3. ペアトレード 折れ線
    ax3 = fig.add_axes([0.05, 0.06, 0.92, 0.42])
    pair_colors = {
        'ソシオネクスト-ローム': '#1565C0',
        '三菱マテ-住友鉱山': '#FF9800',
        'ローム-太陽誘電': '#43A047',
        '三井金属-住友鉱山': '#9C27B0',
        'ソシオネクスト-東エレ': '#00BCD4',
        '住友鉱山-DOWA': '#E91E63',
    }
    for label, color in pair_colors.items():
        d = pair_df[pair_df['pair'] == label].sort_values('entry')
        ax3.plot(d['entry'], d['sharpe'], 'o-', color=color, lw=1.3, alpha=0.85, label=label)
    ax3.axhline(0, color='black', lw=0.6)
    ax3.set_xlabel('判定時刻 (JST)', fontsize=9)
    ax3.set_ylabel('Sharpe (年率)', fontsize=9)
    ax3.set_title('ペアトレード Reversion 時刻別 Sharpe (|spread|>50bps, →大引け持ち)',
                  fontsize=10, fontweight='bold')
    plt.setp(ax3.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    ax3.legend(fontsize=9, ncol=2, loc='best')
    ax3.grid(alpha=0.3)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    fig.text(0.99, 0.005,
             'データ: stocks_intraday 2024-05〜2026-05 / 28個の判定時刻 | コスト8bps (LS往復)',
             ha='right', va='bottom', fontsize=7, color='gray')
    plt.savefig('result.png', dpi=100, bbox_inches='tight', facecolor='white')
    print(f"\nresult.png 保存完了 ({time.time()-t0:.1f}秒)")


if __name__ == '__main__':
    main()
