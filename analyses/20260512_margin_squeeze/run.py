"""
信用残・空売り残データを使った週次クロスセクショナル戦略

これまでの戦略は全て個別銘柄のティック解析。今回は完全に別の角度:
  - データソース: jquants_margin_interest (週次, 金曜公表)
  - 期間: 2021-2026 (5年)
  - 銘柄: 流動性のある2,000銘柄

仮説:
  H1: 信用買い残 急増 → 投機過熱 → 翌週反落 (空売り対象)
  H2: 信用売り残 急増 → 踏み上げ可能性 → 翌週上昇 (Short Squeeze 買い)
  H3: 信用倍率 (long/short) 高水準 → 過熱 → 反落
  H4: 信用倍率 低水準 (空売り過多) → 翌週上昇

検証方法:
  各金曜時点で:
    1. 信用残データの前週比変化率を計算
    2. 全銘柄を変化率でランク付け
    3. 各週で上位/下位の銘柄群を抽出
    4. 翌5営業日のリターンを測定
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
COST_BPS = 8  # 信用取引込み (ロングのみ +2bps × 往復 = 4bps, ショートだと +金利等で 8bps)
CACHE = Path('margin_data.parquet')


def load_data():
    """信用残データ × 株価日足を結合"""
    if CACHE.exists():
        return pd.read_parquet(CACHE)

    conn = psycopg2.connect(**PG_CONFIG)
    print("信用残データ取得 (2021-01-01 ~ 2026-05-01)...")
    sql = """
        WITH liquid AS (
            -- 流動性のある銘柄に絞る (直近1年で日次出来高 100万以上)
            SELECT DISTINCT code FROM stocks_daily
            WHERE date >= '2025-05-01' AND date < '2026-05-01'
            GROUP BY code
            HAVING AVG(volume) > 1000000
        )
        SELECT m.date, m.code,
               m.long_vol, m.shrt_vol,
               d.adj_close AS close, d.adj_open AS open, d.volume
        FROM jquants_margin_interest m
        JOIN liquid l ON l.code = m.code
        LEFT JOIN stocks_daily d ON d.code = m.code AND d.date = m.date
        WHERE m.date >= '2021-01-01'
        ORDER BY m.code, m.date
    """
    df = pd.read_sql(sql, conn)
    conn.close()
    print(f"  {len(df):,} rows, {df['code'].nunique()}銘柄")

    df['date'] = pd.to_datetime(df['date'])
    df['long_short_ratio'] = df['long_vol'] / df['shrt_vol'].replace(0, np.nan)

    # 前週比変化率
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    for col in ['long_vol', 'shrt_vol', 'long_short_ratio']:
        df[f'{col}_chg'] = df.groupby('code')[col].pct_change()

    # 過去N週分の絶対水準のZスコア (ボラ正規化)
    for col, name in [('long_vol', 'long_z'), ('shrt_vol', 'shrt_z'),
                      ('long_short_ratio', 'lsr_z')]:
        rolling_mean = df.groupby('code')[col].transform(
            lambda x: x.shift(1).rolling(26, min_periods=8).mean())
        rolling_std = df.groupby('code')[col].transform(
            lambda x: x.shift(1).rolling(26, min_periods=8).std())
        df[name] = (df[col] - rolling_mean) / rolling_std.replace(0, np.nan)

    print("翌週リターン計算 (信用残発表日 → 5営業日後)...")
    # 翌5営業日後の close (株価データから引っ張る)
    conn = psycopg2.connect(**PG_CONFIG)
    daily = pd.read_sql("""
        SELECT code, date, adj_close FROM stocks_daily
        WHERE date >= '2021-01-01' ORDER BY code, date
    """, conn)
    conn.close()
    daily['date'] = pd.to_datetime(daily['date'])

    # 各margin日(金曜)から5営業日後の close
    daily_sorted = daily.sort_values(['code', 'date']).reset_index(drop=True)
    daily_sorted['fwd5_close'] = daily_sorted.groupby('code')['adj_close'].shift(-5)
    daily_sorted['fwd1_close'] = daily_sorted.groupby('code')['adj_close'].shift(-1)
    daily_sorted['fwd10_close'] = daily_sorted.groupby('code')['adj_close'].shift(-10)
    daily_sorted = daily_sorted.rename(columns={'adj_close': 'close_d'})

    df = df.merge(
        daily_sorted[['code','date','close_d','fwd1_close','fwd5_close','fwd10_close']],
        on=['code', 'date'], how='left'
    )
    df['ret_1d'] = (df['fwd1_close'] / df['close_d'] - 1) * 10000
    df['ret_5d'] = (df['fwd5_close'] / df['close_d'] - 1) * 10000
    df['ret_10d'] = (df['fwd10_close'] / df['close_d'] - 1) * 10000

    df.to_parquet(CACHE)
    print(f"  キャッシュ保存: {CACHE}")
    return df


def stats_summary(arr, label='', cost=0):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n < 30:
        return None
    net = arr - cost
    t, p = stats.ttest_1samp(arr, 0)
    return dict(label=label, N=n,
                mean_raw=round(arr.mean(), 1),
                mean_net=round(net.mean(), 1),
                std=round(arr.std(), 1),
                t_stat=round(t, 2), p_val=round(p, 4),
                win_rate=round((arr > 0).mean() * 100, 1),
                sharpe=round(net.mean() / arr.std() * np.sqrt(52) if arr.std() > 0 else 0, 2))


def market_neutralize(df, group_col='date', ret_col='ret_5d'):
    """各日付内で平均リターンを差し引き (市場ニュートラル化)"""
    df = df.copy()
    df[f'{ret_col}_market'] = df.groupby(group_col)[ret_col].transform('mean')
    df[f'{ret_col}_neutral'] = df[ret_col] - df[f'{ret_col}_market']
    return df


def main():
    import time
    t0 = time.time()

    df = load_data()
    print(f"\n対象データ: {len(df):,} 行, {df['code'].nunique()} 銘柄")
    print(f"期間: {df['date'].min().date()} ~ {df['date'].max().date()}")

    # 各日にデータがある銘柄数
    weekly_n = df.groupby('date')['code'].nunique()
    print(f"\n週あたり平均銘柄数: {weekly_n.mean():.0f}")
    print(f"全週数: {len(weekly_n)}")

    # 市場ニュートラル化 (各週でcross-sectionに平均を引く)
    df = market_neutralize(df, 'date', 'ret_5d')
    df = market_neutralize(df, 'date', 'ret_1d')
    df = market_neutralize(df, 'date', 'ret_10d')

    # クロスセクショナルランク (各日付内で)
    for col in ['long_vol_chg', 'shrt_vol_chg', 'long_short_ratio_chg',
                'long_z', 'shrt_z', 'lsr_z']:
        df[f'{col}_rank'] = df.groupby('date')[col].rank(pct=True)

    print("\n=== シグナル検証 (市場ニュートラル化済 5日後リターン) ===")
    results = []

    # ===== H1: 信用買い残 急増 → 翌週ショート =====
    # Top 10%増の銘柄を空売り
    for pct in [0.05, 0.10, 0.20]:
        sub = df[df['long_vol_chg_rank'] > 1 - pct]
        r = stats_summary(-sub['ret_5d_neutral'].values, f'H1: 信用買増Top{int(pct*100)}% ショート')
        if r: results.append(r)

    # ===== H2: 信用売り残 急増 → 翌週ロング (Short Squeeze) =====
    for pct in [0.05, 0.10, 0.20]:
        sub = df[df['shrt_vol_chg_rank'] > 1 - pct]
        r = stats_summary(sub['ret_5d_neutral'].values, f'H2: 信用売増Top{int(pct*100)}% ロング')
        if r: results.append(r)

    # ===== H3: 信用倍率 (long/short) 高水準 → 翌週ショート =====
    for pct in [0.05, 0.10, 0.20]:
        sub = df[df['long_short_ratio_chg_rank'] > 1 - pct]
        r = stats_summary(-sub['ret_5d_neutral'].values, f'H3: 信用倍率増Top{int(pct*100)}% ショート')
        if r: results.append(r)

    # ===== H4: 信用倍率 低水準 → 翌週ロング =====
    for pct in [0.05, 0.10, 0.20]:
        sub = df[df['long_short_ratio_chg_rank'] < pct]
        r = stats_summary(sub['ret_5d_neutral'].values, f'H4: 信用倍率減Bottom{int(pct*100)}% ロング')
        if r: results.append(r)

    # ===== H5: Z-score ベース =====
    for thresh in [2.0, 3.0]:
        # 信用買い残 Z > thresh → ショート
        sub = df[df['long_z'] > thresh]
        r = stats_summary(-sub['ret_5d_neutral'].values, f'H5a: long_Z>{thresh} ショート')
        if r: results.append(r)
        # 信用売り残 Z > thresh → ロング (squeeze)
        sub = df[df['shrt_z'] > thresh]
        r = stats_summary(sub['ret_5d_neutral'].values, f'H5b: shrt_Z>{thresh} ロング')
        if r: results.append(r)
        # 信用倍率 Z < -thresh → ロング (ショート過多)
        sub = df[df['lsr_z'] < -thresh]
        r = stats_summary(sub['ret_5d_neutral'].values, f'H5c: lsr_Z<-{thresh} ロング')
        if r: results.append(r)
        # 信用倍率 Z > thresh → ショート
        sub = df[df['lsr_z'] > thresh]
        r = stats_summary(-sub['ret_5d_neutral'].values, f'H5d: lsr_Z>{thresh} ショート')
        if r: results.append(r)

    # ===== H6: Long-Short ポートフォリオ (両建て) =====
    for pct in [0.10, 0.20]:
        # Buy: 信用倍率最低 (空売り過多), Sell: 信用倍率最高
        long_side = df[df['long_short_ratio_chg_rank'] < pct]['ret_5d_neutral'].values
        short_side = -df[df['long_short_ratio_chg_rank'] > 1-pct]['ret_5d_neutral'].values
        combined = np.concatenate([long_side, short_side])
        r = stats_summary(combined, f'H6: LS両建て {int(pct*100)}% (買=squeeze候補,売=過熱)')
        if r: results.append(r)

    # 結果まとめ
    res_df = pd.DataFrame(results).sort_values('t_stat', ascending=False)
    print(res_df[['label','N','mean_raw','t_stat','p_val','win_rate','sharpe']].to_string(index=False))

    res_df.to_csv('signal_results.csv', index=False)

    # ===========================================================
    # 最強シグナル(t値最高)の年別・期間別パフォーマンス
    # ===========================================================
    best = res_df.iloc[0]
    print(f"\n=== Best シグナル: {best['label']} ===")
    print(f"  N={best['N']:,}, mean={best['mean_raw']}bps, t={best['t_stat']}, Sharpe={best['sharpe']}")

    # 年別
    print("\n=== 期間別 (年別ロバストネス) ===")
    df['year'] = df['date'].dt.year

    yearly = []
    for y in sorted(df['year'].unique()):
        sub_year = df[df['year'] == y]
        for label, mask_func, direction in [
            ('H4(LSR最低 Bottom10% Long)', lambda d: d['long_short_ratio_chg_rank'] < 0.10, 1),
            ('H1(信用買増Top10% Short)', lambda d: d['long_vol_chg_rank'] > 0.90, -1),
            ('H2(信用売増Top10% Long)', lambda d: d['shrt_vol_chg_rank'] > 0.90, 1),
        ]:
            arr = sub_year[mask_func(sub_year)]['ret_5d_neutral'].values * direction
            arr = arr[np.isfinite(arr)]
            if len(arr) > 30:
                t,p = stats.ttest_1samp(arr,0)
                yearly.append({
                    'year': y, 'strategy': label, 'N': len(arr),
                    'mean': round(arr.mean(),1), 't': round(t,2),
                    'wr': round((arr>0).mean()*100,1),
                })
    yearly_df = pd.DataFrame(yearly)
    print(yearly_df.to_string(index=False))
    yearly_df.to_csv('yearly_robustness.csv', index=False)

    # ===========================================================
    # エクイティカーブ作成 (LS両建て portfolio)
    # ===========================================================
    print("\n=== エクイティカーブ計算 ===")
    df_eq = df.copy()

    # ベスト戦略のlong/short両建てを週次で集計
    df_eq = df_eq.sort_values('date')
    portfolio_pnl = []
    for d, week in df_eq.groupby('date'):
        long_pos = week[week['long_short_ratio_chg_rank'] < 0.10]
        short_pos = week[week['long_short_ratio_chg_rank'] > 0.90]
        if len(long_pos) > 5 and len(short_pos) > 5:
            long_ret = long_pos['ret_5d_neutral'].mean()
            short_ret = -short_pos['ret_5d_neutral'].mean()
            pnl = (long_ret + short_ret) / 2 - COST_BPS  # 両建てなのでコスト2倍
            if pd.notna(pnl):
                portfolio_pnl.append({'date': d, 'pnl': pnl})

    pnl_df = pd.DataFrame(portfolio_pnl)
    pnl_df['cum'] = pnl_df['pnl'].cumsum()
    pnl_df.to_csv('equity_curve.csv', index=False)
    print(f"  週次N: {len(pnl_df)}, 累積: {pnl_df['cum'].iloc[-1]:.0f}bps")
    sharpe_p = pnl_df['pnl'].mean() / pnl_df['pnl'].std() * np.sqrt(52)
    print(f"  Sharpe (週次, 年率): {sharpe_p:.2f}")

    # ===========================================================
    # 図
    # ===========================================================
    fig = plt.figure(figsize=(15, 9), facecolor='white')
    plt.rcParams.update({
        'font.family': ['Hiragino Sans', 'IPAexGothic', 'sans-serif'],
        'axes.unicode_minus': False,
    })
    fig.suptitle('信用残データ駆動 週次クロスセクショナル戦略 (2021〜2026, 約2,000銘柄)',
                 fontsize=13, fontweight='bold', y=0.99)

    # 上左: シグナル別 t値
    ax1 = fig.add_axes([0.05, 0.55, 0.55, 0.38])
    plot_df = res_df.head(20).iloc[::-1]
    xs = range(len(plot_df))
    cols = ['#43A047' if (t > 1.96 and m > 0) else
            '#E53935' if (t < -1.96) else
            '#9E9E9E' for t, m in zip(plot_df['t_stat'], plot_df['mean_raw'])]
    ax1.barh(list(xs), plot_df['t_stat'].values, color=cols, alpha=0.85)
    ax1.set_yticks(list(xs))
    ax1.set_yticklabels(plot_df['label'], fontsize=8)
    ax1.axvline(0, color='black', lw=0.6)
    ax1.axvline(1.96, color='gray', lw=0.6, linestyle='--', alpha=0.5)
    ax1.axvline(-1.96, color='gray', lw=0.6, linestyle='--', alpha=0.5)
    ax1.set_xlabel('t統計量 (市場ニュートラル化済リターン)', fontsize=9)
    ax1.set_title('シグナル別 t値', fontsize=10, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # 上右: best 5 mean リターン
    ax2 = fig.add_axes([0.65, 0.55, 0.32, 0.38])
    top5 = res_df.head(5)
    xs = range(len(top5))
    ax2.bar(xs, top5['mean_raw'].values,
            color=['#43A047' if v > 0 else '#E53935' for v in top5['mean_raw']],
            alpha=0.85)
    ax2.set_xticks(xs)
    ax2.set_xticklabels(top5['label'].str[:18], rotation=20, ha='right', fontsize=7.5)
    for i, (_, row) in enumerate(top5.iterrows()):
        ax2.text(i, row['mean_raw'] + (1 if row['mean_raw'] >= 0 else -1),
                 f"t={row['t_stat']}\nWR={row['win_rate']}%",
                 ha='center', fontsize=7,
                 va='bottom' if row['mean_raw']>=0 else 'top')
    ax2.axhline(0, color='black', lw=0.6)
    ax2.set_ylabel('平均5日リターン (bps, ニュートラル化済)', fontsize=8)
    ax2.set_title('Top5 シグナル', fontsize=10, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # 下: エクイティカーブ
    ax3 = fig.add_axes([0.05, 0.07, 0.92, 0.40])
    ax3.plot(pnl_df['date'], pnl_df['cum'], color='#1565C0', lw=1.5)
    ax3.fill_between(pnl_df['date'], 0, pnl_df['cum'],
                      where=pnl_df['cum'] > 0, alpha=0.2, color='#43A047')
    ax3.fill_between(pnl_df['date'], 0, pnl_df['cum'],
                      where=pnl_df['cum'] <= 0, alpha=0.2, color='#E53935')
    ax3.axhline(0, color='black', lw=0.6)
    ax3.set_xlabel('日付', fontsize=9)
    ax3.set_ylabel('累積 net PnL (bps)', fontsize=9)
    ax3.set_title(f'LS両建てポートフォリオ エクイティカーブ '
                  f'(累積={pnl_df["cum"].iloc[-1]:.0f}bps, Sharpe={sharpe_p:.2f})',
                  fontsize=10, fontweight='bold')
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%y/%m'))
    ax3.grid(alpha=0.3)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    fig.text(0.99, 0.005,
             'データ: 2021〜2026 信用残データ / 流動性Top 2,213銘柄 / 5営業日後リターン市場ニュートラル化',
             ha='right', va='bottom', fontsize=7, color='gray')
    plt.savefig('result.png', dpi=100, bbox_inches='tight', facecolor='white')
    print(f"\nresult.png 保存完了 ({time.time()-t0:.1f}秒)")


if __name__ == '__main__':
    main()
