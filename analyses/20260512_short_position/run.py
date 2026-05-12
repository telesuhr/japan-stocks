"""
B. 機関の個別空売り報告データを使ったクロスセクショナル戦略

データ: jquants_short_sale_report (5年, 893K reports)
  発行株数の 0.5% 以上の空売り建玉を持つ機関が報告義務

検証戦略:
  H1: 銘柄ごとの空売り建玉急増 → 翌週反転買い (Short Squeeze)
  H2: 空売り建玉減 (カバー進行) → 上昇継続
  H3: 「Smart Money 機関」(Citadel, MLP, ICS等)の空売り増 → 翌月ショート
  H4: 複数機関の空売り集中 (crowding) → 反転買い候補
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
CACHE = Path('short_pos_data.parquet')

# Smart Money 候補機関 (アルファ追求型ファンド)
SMART_MONEY = [
    'Integrated Core Strategies (Asia) Pte. Ltd.',  # Millennium spin-off
    'Marshall Wace LLP',                            # Quant
    'Maverick Capital Ltd.',
    'Citadel Equities GP, LLC',
    'XTX Markets Pte Ltd',                          # HFT
    'Flow Traders Asia Pte Ltd.',                   # Market maker
]


def load_data():
    if CACHE.exists():
        return pd.read_parquet(CACHE)

    conn = psycopg2.connect(**PG_CONFIG)
    print("空売り報告データ取得...")
    df = pd.read_sql("""
        SELECT s.disc_date, s.calc_date, s.code, s.ss_name,
               s.shrt_pos_to_so, s.shrt_pos_shares, s.prev_rpt_ratio
        FROM jquants_short_sale_report s
        WHERE s.disc_date >= '2021-01-01'
        ORDER BY s.disc_date, s.code
    """, conn)
    conn.close()
    print(f"  {len(df):,} reports")

    df['disc_date'] = pd.to_datetime(df['disc_date'])
    df['shrt_pos_to_so'] = df['shrt_pos_to_so'].astype(float)
    df['prev_rpt_ratio'] = df['prev_rpt_ratio'].astype(float).fillna(0)
    df['change'] = df['shrt_pos_to_so'] - df['prev_rpt_ratio']
    df['is_smart_money'] = df['ss_name'].isin(SMART_MONEY)

    df.to_parquet(CACHE)
    print(f"  キャッシュ保存: {CACHE}")
    return df


def aggregate_by_code_date(df):
    """各 (code, disc_date) で集計"""
    agg = df.groupby(['disc_date', 'code']).agg(
        n_reporters=('ss_name', 'count'),
        total_short=('shrt_pos_to_so', 'sum'),
        total_change=('change', 'sum'),
        n_smart=('is_smart_money', 'sum'),
        smart_short=('shrt_pos_to_so', lambda x: x[df.loc[x.index, 'is_smart_money']].sum()),
        smart_change=('change', lambda x: x[df.loc[x.index, 'is_smart_money']].sum()),
    ).reset_index()
    return agg


def get_stock_prices():
    conn = psycopg2.connect(**PG_CONFIG)
    daily = pd.read_sql("""
        SELECT code, date, adj_close, volume
        FROM stocks_daily
        WHERE date >= '2021-01-01' AND adj_close IS NOT NULL
        ORDER BY code, date
    """, conn)
    conn.close()
    daily['date'] = pd.to_datetime(daily['date'])
    return daily


def stats_summary(arr, label=''):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n < 20:
        return None
    t, p = stats.ttest_1samp(arr, 0)
    return dict(label=label, N=n,
                mean=round(arr.mean(), 1), std=round(arr.std(), 1),
                t_stat=round(t, 2), p_val=round(p, 4),
                win_rate=round((arr > 0).mean() * 100, 1),
                sharpe=round(arr.mean()/arr.std() * np.sqrt(52) if arr.std() > 0 else 0, 2))


def main():
    import time
    t0 = time.time()

    df = load_data()
    print(f"\n総報告数: {len(df):,}")
    print(f"  銘柄数: {df['code'].nunique()}")
    print(f"  Smart Money 報告: {df['is_smart_money'].sum():,}")

    # 銘柄×日付集計
    print("\n銘柄×日付集計...")
    agg = aggregate_by_code_date(df)
    print(f"  集計後: {len(agg):,} 観測")

    # 株価データ取得
    print("\n株価データ取得...")
    daily = get_stock_prices()
    print(f"  {len(daily):,} 日次レコード")

    # 各 (code, disc_date) の翌5日リターン計算
    print("\n翌週リターン計算...")
    # asof merge: 各 disc_date の最初の取引日を起点
    agg = agg.sort_values(['code', 'disc_date'])

    rows = []
    daily_dict = {c: g.sort_values('date').reset_index(drop=True)
                  for c, g in daily.groupby('code')}

    for _, row in agg.iterrows():
        c = row['code']
        d = row['disc_date']
        if c not in daily_dict:
            continue
        g = daily_dict[c]
        idx_after = g[g['date'] > d].index
        if len(idx_after) < 6:
            continue
        start = idx_after[0]
        p0 = g.loc[start, 'adj_close']
        # 5日後・10日後
        if start + 5 < len(g):
            p5 = g.loc[start + 5, 'adj_close']
            ret5 = (p5 / p0 - 1) * 10000
        else:
            ret5 = np.nan
        if start + 10 < len(g):
            p10 = g.loc[start + 10, 'adj_close']
            ret10 = (p10 / p0 - 1) * 10000
        else:
            ret10 = np.nan

        rows.append({
            'disc_date': d, 'code': c,
            'n_reporters': row['n_reporters'],
            'total_short': row['total_short'],
            'total_change': row['total_change'],
            'n_smart': row['n_smart'],
            'smart_change': row['smart_change'],
            'ret_5d': ret5, 'ret_10d': ret10,
        })

    sig_df = pd.DataFrame(rows)
    print(f"  {len(sig_df):,} 観測")

    # 市場ニュートラル化 (各日付内で平均を引く)
    sig_df['ret_5d_mkt'] = sig_df.groupby('disc_date')['ret_5d'].transform('mean')
    sig_df['ret_5d_neutral'] = sig_df['ret_5d'] - sig_df['ret_5d_mkt']
    sig_df['ret_10d_mkt'] = sig_df.groupby('disc_date')['ret_10d'].transform('mean')
    sig_df['ret_10d_neutral'] = sig_df['ret_10d'] - sig_df['ret_10d_mkt']

    # ランク (各日付内クロスセクション)
    sig_df['total_change_rank'] = sig_df.groupby('disc_date')['total_change'].rank(pct=True)
    sig_df['total_short_rank'] = sig_df.groupby('disc_date')['total_short'].rank(pct=True)
    sig_df['n_reporters_rank'] = sig_df.groupby('disc_date')['n_reporters'].rank(pct=True)
    sig_df['smart_change_rank'] = sig_df.groupby('disc_date')['smart_change'].rank(pct=True)

    # =====================================================================
    # シグナル検証
    # =====================================================================
    print("\n=== シグナル検証 (市場ニュートラル化 5/10日リターン) ===")
    results = []

    # H1: 空売り建玉急増 (合計+) → 翌週反転買い (squeeze)
    for pct in [0.05, 0.10, 0.20]:
        sub = sig_df[sig_df['total_change_rank'] > 1 - pct]
        r = stats_summary(sub['ret_5d_neutral'].values, f'H1: 空売り急増Top{int(pct*100)}% ロング (5d)')
        if r: results.append(r)
        r = stats_summary(sub['ret_10d_neutral'].values, f'H1: 空売り急増Top{int(pct*100)}% ロング (10d)')
        if r: results.append(r)

    # H2: 空売り建玉減少 (カバー) → 上昇継続
    for pct in [0.05, 0.10, 0.20]:
        sub = sig_df[sig_df['total_change_rank'] < pct]
        r = stats_summary(sub['ret_5d_neutral'].values, f'H2: 空売り減少Top{int(pct*100)}% ロング (5d)')
        if r: results.append(r)

    # H3: 空売り建玉合計が大きい (crowded short) → 翌週反転買い
    for pct in [0.05, 0.10, 0.20]:
        sub = sig_df[sig_df['total_short_rank'] > 1 - pct]
        r = stats_summary(sub['ret_5d_neutral'].values, f'H3: 空売り蓄積大Top{int(pct*100)}% ロング')
        if r: results.append(r)

    # H4: 多数の機関が空売り (n_reporters >= N) → squeeze
    for n_min in [3, 5, 8]:
        sub = sig_df[sig_df['n_reporters'] >= n_min]
        r = stats_summary(sub['ret_5d_neutral'].values, f'H4: {n_min}機関以上空売り ロング')
        if r: results.append(r)

    # H5: Smart Money 空売り増 → ショート (彼らが当てる前提)
    sub = sig_df[(sig_df['smart_change'] > 0) & (sig_df['n_smart'] >= 1)]
    r = stats_summary(-sub['ret_5d_neutral'].values, f'H5: Smart Money 空売り増 → ショート (5d)')
    if r: results.append(r)
    r = stats_summary(-sub['ret_10d_neutral'].values, f'H5: Smart Money 空売り増 → ショート (10d)')
    if r: results.append(r)

    # H5b: Smart Money 空売り減 (カバー) → ロング
    sub = sig_df[(sig_df['smart_change'] < 0) & (sig_df['n_smart'] >= 1)]
    r = stats_summary(sub['ret_5d_neutral'].values, f'H5b: Smart Money カバー → ロング')
    if r: results.append(r)

    # H6: Smart Money 空売り増 (上位 5%) → ショート
    for pct in [0.05, 0.10]:
        sub = sig_df[sig_df['smart_change_rank'] > 1 - pct]
        r = stats_summary(-sub['ret_5d_neutral'].values, f'H6: Smart空売り Top{int(pct*100)}% ショート')
        if r: results.append(r)

    res_df = pd.DataFrame(results).sort_values('t_stat', ascending=False)
    print(res_df.to_string(index=False))
    res_df.to_csv('signal_results.csv', index=False)

    # ===========================================
    # トップシグナルの年別ロバストネス
    # ===========================================
    print("\n=== 年別ロバストネス (主要戦略) ===")
    sig_df['year'] = sig_df['disc_date'].dt.year

    yearly = []
    for y in sorted(sig_df['year'].unique()):
        sub_y = sig_df[sig_df['year'] == y]
        # H1: 空売り急増 Top10% → ロング
        sub = sub_y[sub_y['total_change_rank'] > 0.90]
        arr = sub['ret_5d_neutral'].dropna().values
        if len(arr) > 10:
            yearly.append({'year': y, 'strategy': 'H1: 空売り急増Top10% Long',
                           'N': len(arr), 'mean': round(np.mean(arr), 1),
                           'wr': round((arr > 0).mean() * 100, 1)})
        # H3: 空売り蓄積大 Top10% → ロング
        sub = sub_y[sub_y['total_short_rank'] > 0.90]
        arr = sub['ret_5d_neutral'].dropna().values
        if len(arr) > 10:
            yearly.append({'year': y, 'strategy': 'H3: 空売り蓄積Top10% Long',
                           'N': len(arr), 'mean': round(np.mean(arr), 1),
                           'wr': round((arr > 0).mean() * 100, 1)})
        # H5: Smart Money short → ショート
        sub = sub_y[(sub_y['smart_change'] > 0) & (sub_y['n_smart'] >= 1)]
        arr = (-sub['ret_5d_neutral']).dropna().values
        if len(arr) > 10:
            yearly.append({'year': y, 'strategy': 'H5: Smart空売り増→Short',
                           'N': len(arr), 'mean': round(np.mean(arr), 1),
                           'wr': round((arr > 0).mean() * 100, 1)})

    yearly_df = pd.DataFrame(yearly)
    print(yearly_df.to_string(index=False))
    yearly_df.to_csv('yearly.csv', index=False)

    # ===========================================
    # エクイティカーブ (LS両建て: 空売り急増 long / smart空売り short)
    # ===========================================
    sig_df_eq = sig_df.dropna(subset=['ret_5d_neutral'])
    long_pos = sig_df_eq[sig_df_eq['total_change_rank'] > 0.90]
    short_pos = sig_df_eq[(sig_df_eq['smart_change'] > 0) & (sig_df_eq['n_smart'] >= 1)]

    long_daily = long_pos.groupby('disc_date')['ret_5d_neutral'].mean()
    short_daily = -short_pos.groupby('disc_date')['ret_5d_neutral'].mean()

    pnl_df = pd.concat([long_daily.rename('long'), short_daily.rename('short')], axis=1).fillna(0)
    pnl_df['portfolio'] = (pnl_df['long'] + pnl_df['short']) / 2 - 8  # 両建てコスト8bps
    pnl_df['cum'] = pnl_df['portfolio'].cumsum()
    pnl_df.to_csv('equity.csv')

    sharpe_p = pnl_df['portfolio'].mean() / pnl_df['portfolio'].std() * np.sqrt(52) if pnl_df['portfolio'].std() > 0 else 0
    print(f"\nLS両建てポートフォリオ:")
    print(f"  Sharpe (年率): {sharpe_p:.2f}")
    print(f"  累積: {pnl_df['cum'].iloc[-1]:.0f}bps")

    # =============================
    # 図
    # =============================
    fig = plt.figure(figsize=(15, 9), facecolor='white')
    plt.rcParams.update({
        'font.family': ['Hiragino Sans', 'IPAexGothic', 'sans-serif'],
        'axes.unicode_minus': False,
    })
    fig.suptitle('B. 機関別空売り報告データ × 翌週リターン (2021-2026)',
                 fontsize=13, fontweight='bold', y=0.99)

    # 上左: t値ランキング
    ax1 = fig.add_axes([0.05, 0.55, 0.55, 0.38])
    plot_df = res_df.head(20).iloc[::-1]
    xs = range(len(plot_df))
    cols = ['#43A047' if (t > 1.96) else '#E53935' if (t < -1.96) else '#9E9E9E'
            for t in plot_df['t_stat']]
    ax1.barh(list(xs), plot_df['t_stat'].values, color=cols, alpha=0.85)
    ax1.set_yticks(list(xs))
    ax1.set_yticklabels(plot_df['label'], fontsize=8)
    ax1.axvline(0, color='black', lw=0.6)
    ax1.axvline(1.96, color='gray', lw=0.6, linestyle='--', alpha=0.5)
    ax1.axvline(-1.96, color='gray', lw=0.6, linestyle='--', alpha=0.5)
    ax1.set_xlabel('t統計量', fontsize=9)
    ax1.set_title('シグナル別 t値', fontsize=10, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # 上右: 主要機関の報告数 (棒)
    ax2 = fig.add_axes([0.65, 0.55, 0.32, 0.38])
    counts = df['ss_name'].value_counts().head(10).iloc[::-1]
    ys = range(len(counts))
    ax2.barh(list(ys), counts.values, color='#FF9800', alpha=0.85)
    ax2.set_yticks(list(ys))
    ax2.set_yticklabels([n[:30] for n in counts.index], fontsize=7.5)
    ax2.set_xlabel('報告回数', fontsize=8)
    ax2.set_title('主要空売り機関 Top10', fontsize=10, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # 下: エクイティカーブ
    ax3 = fig.add_axes([0.05, 0.07, 0.92, 0.40])
    pnl_df_reset = pnl_df.reset_index()
    ax3.plot(pnl_df_reset['disc_date'], pnl_df_reset['cum'].values, color='#1565C0', lw=1.5,
             label=f'LS両建て (Long=空売り急増, Short=Smart空売り)')
    ax3.axhline(0, color='black', lw=0.6)
    ax3.set_ylabel('累積 net PnL (bps)', fontsize=9)
    ax3.set_title(f'LS両建てエクイティカーブ (Sharpe={sharpe_p:.2f}, '
                  f'累積={pnl_df["cum"].iloc[-1]:.0f}bps)',
                  fontsize=10, fontweight='bold')
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%y/%m'))
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    fig.text(0.99, 0.005,
             'データ: 2021-2026 機関別空売り報告 (jquants_short_sale_report) | コスト両建て8bps',
             ha='right', va='bottom', fontsize=7, color='gray')
    plt.savefig('result.png', dpi=100, bbox_inches='tight', facecolor='white')
    print(f"\nresult.png 保存完了 ({time.time()-t0:.1f}秒)")


if __name__ == '__main__':
    main()
