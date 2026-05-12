"""
A. 投資部門別売買データを使った週次マーケットタイミング戦略

データ: investor_types (10年, 522週)
  - Frgn (海外勢): 機関投資家の代表
  - Ind (個人投資家): 逆張り傾向
  - Prop (自己): 証券会社の自己売買
  - TrstBnk (信託銀行): 年金等の代理
  - InvTr (投信): 公募投信フロー

仮説:
  H1: 海外勢 大幅買い越し → 翌週TOPIX上昇 (Foreign Flow Momentum)
  H2: 個人 大幅買い越し → 翌週TOPIX下落 (Individual Sentiment Inverse)
  H3: 信託銀行 買い越し → 翌週TOPIX上昇 (Pension Fund Smart Money)
  H4: 海外 - 個人 のスプレッド → 強気弱気指標
  H5: 投信フローの変化 → 退場・参入のシグナル
  H6: 自己売買 (Prop) の方向 → HFT/アービトラージのバイアス

対象: TSEPrime (主市場)
予測: 翌週 (5営業日) TOPIX リターン
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
import json
import warnings
warnings.filterwarnings('ignore')

PG_CONFIG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}
CACHE = Path('flow_data.parquet')


def load_data():
    if CACHE.exists():
        return pd.read_parquet(CACHE)

    conn = psycopg2.connect(**PG_CONFIG)
    print("投資部門別データ取得 (TSEPrime中心) ...")
    # TSEPrime と古い TSE1st を結合 (2022年4月以降はTSEPrime, それ以前はTSE1st)
    df = pd.read_sql("""
        SELECT pub_date, st_date, en_date, section, payload
        FROM investor_types
        WHERE section IN ('TSEPrime', 'TSE1st', 'TokyoNagoya')
        ORDER BY pub_date, section
    """, conn)
    print(f"  {len(df)} 行")

    # payload を分解
    print("payload 分解中...")
    fields_balance = ['FrgnBal', 'IndBal', 'PropBal', 'TrstBnkBal', 'InvTrBal',
                      'InsCoBal', 'BankBal', 'BusCoBal', 'SecCoBal']
    fields_total = ['FrgnTot', 'IndTot', 'PropTot', 'TrstBnkTot', 'InvTrTot',
                    'InsCoTot', 'BankTot', 'BusCoTot', 'SecCoTot']

    records = []
    for _, row in df.iterrows():
        try:
            p = row['payload'] if isinstance(row['payload'], dict) else json.loads(row['payload'])
            rec = {'pub_date': row['pub_date'], 'st_date': row['st_date'],
                   'en_date': row['en_date'], 'section': row['section']}
            for f in fields_balance + fields_total:
                rec[f] = p.get(f, np.nan)
            records.append(rec)
        except:
            continue
    df = pd.DataFrame(records)

    # 各 section ごとに別の行 → wideに pivot
    # TSEPrime (2022/4以降) と TSE1st (それ以前) を同列扱い → 'main' に統一
    df['market'] = df['section'].map({'TSE1st':'main','TSEPrime':'main','TokyoNagoya':'all'})
    df_main = df[df['market'] == 'main'].copy()
    print(f"  main市場行数: {len(df_main)}")

    # TOPIX データ
    print("TOPIX データ取得...")
    conn2 = psycopg2.connect(**PG_CONFIG)
    topix = pd.read_sql("""
        SELECT date, close, open FROM index_daily WHERE code = '0000'
        ORDER BY date
    """, conn2)
    n225 = pd.read_sql("""
        SELECT date AS date_n, close AS n225_close FROM index_daily WHERE code = 'N225'
        ORDER BY date
    """, conn2)
    conn2.close()
    topix['date'] = pd.to_datetime(topix['date'])
    n225['date_n'] = pd.to_datetime(n225['date_n'])
    print(f"  TOPIX rows: {len(topix)}")

    # 翌週リターン計算 (pub_date は金曜 → 翌月曜から金曜 5営業日)
    df_main['pub_date'] = pd.to_datetime(df_main['pub_date'])
    df_main['en_date'] = pd.to_datetime(df_main['en_date'])

    # pub_date 直後の最初の営業日の close
    topix_sorted = topix.sort_values('date').reset_index(drop=True)

    def get_fwd_return(pub_d, days_ahead):
        # pub_d より後の最初のbusiness day を起点に5日後
        idx_after = topix_sorted[topix_sorted['date'] > pub_d].index
        if len(idx_after) < days_ahead + 1:
            return np.nan
        start_idx = idx_after[0]
        end_idx = start_idx + days_ahead - 1
        if end_idx >= len(topix_sorted):
            return np.nan
        return (topix_sorted.loc[end_idx, 'close'] / topix_sorted.loc[start_idx, 'open'] - 1) * 10000

    print("リターン計算中...")
    df_main['ret_5d'] = df_main['pub_date'].apply(lambda d: get_fwd_return(d, 5))
    df_main['ret_10d'] = df_main['pub_date'].apply(lambda d: get_fwd_return(d, 10))
    df_main['ret_20d'] = df_main['pub_date'].apply(lambda d: get_fwd_return(d, 20))

    df_main.to_parquet(CACHE)
    print(f"  キャッシュ保存: {CACHE}")
    return df_main


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
    print(f"\n対象: {len(df)} 週")
    print(f"期間: {df['pub_date'].min()} ~ {df['pub_date'].max()}")

    # Bal は買い-売り、正なら買い越し
    # 単位は円。億円に変換
    bal_cols = [c for c in df.columns if c.endswith('Bal')]
    for c in bal_cols:
        df[c] = df[c] / 1e8  # 億円

    # 26週ローリングの Z-score (異常値検出)
    for c in bal_cols:
        ma = df[c].shift(1).rolling(26, min_periods=8).mean()
        sd = df[c].shift(1).rolling(26, min_periods=8).std()
        df[f'{c}_z'] = (df[c] - ma) / sd.replace(0, np.nan)

    print(f"\n海外勢買い越しの統計 (億円):")
    print(df['FrgnBal'].describe(percentiles=[.1,.25,.5,.75,.9]))

    # ========== シグナル検証 ==========
    print("\n=== シグナル検証 (翌週TOPIX リターン bps) ===")
    results = []

    target = 'ret_5d'

    # H1: 海外勢買い越し
    for thresh in [1000, 2000, 5000]:  # 億円
        sub = df[df['FrgnBal'] > thresh]
        r = stats_summary(sub[target].values, f'H1: 海外買越し > {thresh}億')
        if r: results.append(r)

    # 海外勢売り越し → 翌週下落?
    for thresh in [-1000, -2000, -5000]:
        sub = df[df['FrgnBal'] < thresh]
        r = stats_summary(-sub[target].values, f'H1neg: 海外売越し<{thresh}億 ショート')
        if r: results.append(r)

    # Z-score ベース
    for z in [1.5, 2.0]:
        sub = df[df['FrgnBal_z'] > z]
        r = stats_summary(sub[target].values, f'H1z: 海外Bal Z>{z} ロング')
        if r: results.append(r)
        sub = df[df['FrgnBal_z'] < -z]
        r = stats_summary(-sub[target].values, f'H1z_neg: 海外Bal Z<-{z} ショート')
        if r: results.append(r)

    # H2: 個人逆張り
    for thresh in [500, 1000, 2000]:
        sub = df[df['IndBal'] > thresh]
        r = stats_summary(-sub[target].values, f'H2: 個人買越し>{thresh}億 ショート (逆張)')
        if r: results.append(r)
    for thresh in [-500, -1000, -2000]:
        sub = df[df['IndBal'] < thresh]
        r = stats_summary(sub[target].values, f'H2neg: 個人売越し<{thresh}億 ロング (逆張)')
        if r: results.append(r)

    # H3: 信託銀行 (年金代理)
    for z in [1.5, 2.0]:
        sub = df[df['TrstBnkBal_z'] > z]
        r = stats_summary(sub[target].values, f'H3: 信託Bal Z>{z} ロング')
        if r: results.append(r)
        sub = df[df['TrstBnkBal_z'] < -z]
        r = stats_summary(-sub[target].values, f'H3neg: 信託Bal Z<-{z} ショート')
        if r: results.append(r)

    # H4: 海外-個人 スプレッド
    df['FvI'] = df['FrgnBal'] - df['IndBal']
    df['FvI_z'] = ((df['FvI'] - df['FvI'].shift(1).rolling(26).mean()) /
                   df['FvI'].shift(1).rolling(26).std())
    for z in [1.5, 2.0]:
        sub = df[df['FvI_z'] > z]
        r = stats_summary(sub[target].values, f'H4: (海外-個人)Z>{z} ロング')
        if r: results.append(r)
        sub = df[df['FvI_z'] < -z]
        r = stats_summary(-sub[target].values, f'H4neg: (海外-個人)Z<-{z} ショート')
        if r: results.append(r)

    # H5: 投信フロー (買い越しは強気)
    for z in [1.5, 2.0]:
        sub = df[df['InvTrBal_z'] > z]
        r = stats_summary(sub[target].values, f'H5: 投信Bal Z>{z} ロング')
        if r: results.append(r)

    # H6: 自己売買 (Prop)
    for z in [1.5, 2.0]:
        sub = df[df['PropBal_z'] > z]
        r = stats_summary(sub[target].values, f'H6: Prop Z>{z} ロング')
        if r: results.append(r)
        sub = df[df['PropBal_z'] < -z]
        r = stats_summary(-sub[target].values, f'H6neg: Prop Z<-{z} ショート')
        if r: results.append(r)

    res_df = pd.DataFrame(results).sort_values('t_stat', ascending=False)
    print(res_df.to_string(index=False))
    res_df.to_csv('signal_results.csv', index=False)

    # ============================================================
    # ベスト戦略の年別ロバストネス
    # ============================================================
    print("\n=== 年別ロバストネス (主要戦略) ===")
    df['year'] = df['pub_date'].dt.year
    yearly = []
    for y in sorted(df['year'].unique()):
        sub_y = df[df['year'] == y]
        # 海外勢買い越し (Z>1.5)
        arr = sub_y[sub_y['FrgnBal_z'] > 1.5]['ret_5d'].values
        arr = arr[np.isfinite(arr)]
        if len(arr) > 3:
            yearly.append({'year': y, 'strategy': '海外Bal Z>1.5 Long',
                           'N': len(arr), 'mean': round(np.mean(arr), 1),
                           'wr': round((arr > 0).mean() * 100, 1)})
        # 個人売り越し (Z>1.5)
        arr = sub_y[sub_y['IndBal_z'] < -1.5]['ret_5d'].values
        arr = arr[np.isfinite(arr)]
        if len(arr) > 3:
            yearly.append({'year': y, 'strategy': '個人Bal Z<-1.5 Long',
                           'N': len(arr), 'mean': round(np.mean(arr), 1),
                           'wr': round((arr > 0).mean() * 100, 1)})
    yearly_df = pd.DataFrame(yearly)
    print(yearly_df.to_string(index=False))
    yearly_df.to_csv('yearly.csv', index=False)

    # ============================================================
    # メインシグナル: 海外Bal Z-scoreベース 連続戦略
    # 各週ごとにシグナルが出れば翌週ロング、出ない週は休む
    # ============================================================
    df['signal_long'] = (df['FrgnBal_z'] > 1.0).astype(int)
    df['signal_short'] = (df['FrgnBal_z'] < -1.0).astype(int)
    df['position'] = df['signal_long'] - df['signal_short']
    df['week_pnl'] = df['position'] * df['ret_5d']

    valid = df.dropna(subset=['week_pnl', 'ret_5d']).copy()
    valid['cum_pnl'] = valid['week_pnl'].cumsum()
    valid['cum_topix'] = valid['ret_5d'].cumsum()

    total_signals = (valid['position'] != 0).sum()
    print(f"\nシグナル発生週: {total_signals}/{len(valid)} ({100*total_signals/len(valid):.1f}%)")
    if total_signals > 30:
        sig_only = valid[valid['position'] != 0]
        r = stats_summary(sig_only['week_pnl'].values, 'シグナル週のみ')
        if r:
            print(f"  シグナル週のみ: N={r['N']}, mean={r['mean']}bps, t={r['t_stat']}, "
                  f"WR={r['win_rate']}%, Sharpe={r['sharpe']}")

    valid.to_csv('equity.csv', index=False)

    # ============================================================
    # 図
    # ============================================================
    fig = plt.figure(figsize=(15, 9), facecolor='white')
    plt.rcParams.update({
        'font.family': ['Hiragino Sans', 'IPAexGothic', 'sans-serif'],
        'axes.unicode_minus': False,
    })
    fig.suptitle('A. 投資部門別売買データ × 翌週TOPIX (2016-2026, 522週)',
                 fontsize=13, fontweight='bold', y=0.99)

    # 上左: t値ランキング
    ax1 = fig.add_axes([0.05, 0.55, 0.55, 0.38])
    plot_df = res_df.head(20).iloc[::-1]
    xs = range(len(plot_df))
    cols = ['#43A047' if (t > 1.96 and m > 0) else
            '#E53935' if (t < -1.96 or m < 0) else
            '#9E9E9E' for t, m in zip(plot_df['t_stat'], plot_df['mean'])]
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

    # 上右: 海外勢買い越しのヒストグラム
    ax2 = fig.add_axes([0.65, 0.55, 0.32, 0.38])
    vals = df['FrgnBal'].dropna().values
    ax2.hist(vals[np.abs(vals) < 10000], bins=50, color='#1565C0', alpha=0.7)
    ax2.axvline(0, color='black', lw=1)
    ax2.axvline(np.median(vals), color='red', linestyle='--', lw=1, label=f'中央値 {np.median(vals):.0f}億')
    ax2.set_xlabel('海外勢買い越し (億円)', fontsize=9)
    ax2.set_ylabel('週数', fontsize=9)
    ax2.set_title('海外勢買い越し分布', fontsize=10, fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # 下: エクイティカーブ
    ax3 = fig.add_axes([0.05, 0.07, 0.92, 0.40])
    ax3.plot(valid['pub_date'], valid['cum_pnl'].values, color='#1565C0', lw=1.5,
             label='海外フロー追随戦略 (Z>±1)')
    ax3.plot(valid['pub_date'], valid['cum_topix'].values, color='gray', lw=1.0,
             alpha=0.5, label='TOPIX (buy-and-hold)')
    ax3.axhline(0, color='black', lw=0.6)
    ax3.set_ylabel('累積リターン (bps, 週次)', fontsize=9)
    ax3.set_title(f'エクイティカーブ 海外フロー戦略 vs TOPIX', fontsize=10, fontweight='bold')
    ax3.legend(fontsize=9, loc='upper left')
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%y'))
    ax3.grid(alpha=0.3)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    fig.text(0.99, 0.005,
             'データ: 2016-05〜2026-05 / 投資部門別売買 (TSEPrime/TSE1st) × TOPIX 5日リターン',
             ha='right', va='bottom', fontsize=7, color='gray')
    plt.savefig('result.png', dpi=100, bbox_inches='tight', facecolor='white')
    print(f"\nresult.png 保存完了 ({time.time()-t0:.1f}秒)")


if __name__ == '__main__':
    main()
