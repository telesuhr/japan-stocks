"""
セクター内 イントラデイ ロングショート戦略

セクター:
  非鉄: 5706(三井金属), 5711(三菱マテ), 5713(住友鉱山), 5714(DOWA),
        5801(古河電工), 5802(住友電工), 5803(フジクラ)
  AI半導体: 6146(ディスコ), 6526(ソシオネクスト), 6857(アドバンテスト),
            6920(レーザーテック), 6963(ローム), 6976(太陽誘電),
            8035(東京エレクトロン)

戦略候補:
  S1. Best-Worst Spread Reversion: 朝~時点のbest順位を短, worst順位を長, 引けまで
  S2. Best-Worst Spread Momentum: 逆方向
  S3. ペアごとのスプレッドZ-score 反転 (中央銘柄ペアのみ)

判定時刻: 9:30, 10:00, 10:30, 11:00, 12:30, 13:00
保有: 判定時刻 〜 大引け
コスト: 往復8bps (2銘柄分の4bps×2)
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
COST_BPS = 8  # LS 2銘柄分往復

SECTORS = {
    '非鉄': {
        '57060': '三井金属', '57110': '三菱マテ', '57130': '住友鉱山',
        '57140': 'DOWA',
        '58010': '古河電工', '58020': '住友電工', '58030': 'フジクラ',
    },
    'AI半導体': {
        '61460': 'ディスコ', '65260': 'ソシオネクスト', '68570': 'アドバンテスト',
        '69200': 'レーザーテック', '69630': 'ローム', '69760': '太陽誘電',
        '80350': '東京エレクトロン',
    },
}

ENTRY_TIMES = ['09:30:00', '10:00:00', '10:30:00', '11:00:00', '12:30:00', '13:00:00']
CACHE = Path('intraday_bars.parquet')


def load_data():
    if CACHE.exists():
        print(f"キャッシュからロード: {CACHE}")
        return pd.read_parquet(CACHE)

    print("1分足データ取得中...")
    all_codes = sum([list(v.keys()) for v in SECTORS.values()], [])
    placeholders = ','.join(["%s"] * len(all_codes))
    conn = psycopg2.connect(**PG_CONFIG)
    df = pd.read_sql(f"""
        SELECT code, ts, open, high, low, close, volume
        FROM stocks_intraday
        WHERE code IN ({placeholders})
          AND ts >= '2024-05-10' AND ts <= '2026-05-19'
        ORDER BY code, ts
    """, conn, params=all_codes)
    conn.close()
    df['ts'] = pd.to_datetime(df['ts'])
    df['date'] = df['ts'].dt.date
    df['time_str'] = df['ts'].dt.strftime('%H:%M:%S')
    # 取引時間内
    h = df['ts'].dt.hour
    m = df['ts'].dt.minute
    in_session = (
        ((h == 9) | (h == 10) | ((h == 11) & (m <= 30))) |
        (((h == 12) & (m >= 30)) | (h == 13) | (h == 14) | ((h == 15) & (m <= 30)))
    )
    df = df[in_session].reset_index(drop=True)
    df.to_parquet(CACHE)
    print(f"  {len(df):,} 行, {df['code'].nunique()} 銘柄")
    return df


def build_daily_returns(df, entry_times):
    """各 (date, code) に対し: open, entry_times のclose, 大引けclose を抽出"""
    df = df.copy()
    df['time_str'] = df['ts'].dt.strftime('%H:%M:%S')

    # 9:00のopen (寄付値) ← 9:00:00 のバーの open
    # 各entry_time の close
    # 大引け (15:00-15:30の最後)

    out = []
    for (d, code), g in df.groupby(['date', 'code']):
        g = g.sort_values('ts').reset_index(drop=True)
        if len(g) < 30:
            continue

        # 寄付値 (9:00のopenが最良)
        morning_first = g[g['time_str'] >= '09:00:00']
        if morning_first.empty:
            continue
        day_open = morning_first.iloc[0]['open']

        # 大引け値 (最後のバーのclose, 15:30が望ましいが15:00でも可)
        day_close = g.iloc[-1]['close']

        rec = {'date': d, 'code': code, 'day_open': day_open, 'day_close': day_close}

        # 各entry_time のclose
        for t in entry_times:
            sub = g[g['time_str'] <= t]
            if sub.empty:
                rec[f'price_{t}'] = np.nan
            else:
                rec[f'price_{t}'] = sub.iloc[-1]['close']

        out.append(rec)

    return pd.DataFrame(out)


def stats_summary(arr, label='', cost=COST_BPS):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n < 20:
        return dict(label=label, N=n, mean=np.nan, t_stat=np.nan, sharpe=np.nan, win_rate=np.nan)
    net = arr - cost
    t, p = stats.ttest_1samp(arr, 0)
    return dict(label=label, N=n,
                mean_raw=round(arr.mean(), 2),
                mean_net=round(net.mean(), 2),
                std=round(arr.std(), 2),
                t_stat=round(t, 2), p_val=round(p, 4),
                win_rate=round((arr > 0).mean() * 100, 1),
                sharpe=round(net.mean()/arr.std() * np.sqrt(252) if arr.std() > 0 else 0, 2))


def main():
    import time
    t0 = time.time()

    df = load_data()
    print(f"\n総レコード: {len(df):,}")

    print("\n日次集計 (寄付/各時刻close/大引け)...")
    daily = build_daily_returns(df, ENTRY_TIMES)
    print(f"  {len(daily):,} (date, code) 観測")
    print(f"  営業日: {daily['date'].nunique()}")

    # =====================================================================
    # 戦略テスト
    # =====================================================================
    results = []
    pair_results = []

    for sector_name, codes_dict in SECTORS.items():
        codes = list(codes_dict.keys())
        sec_daily = daily[daily['code'].isin(codes)].copy()

        print(f"\n{'='*70}")
        print(f"{sector_name} セクター ({len(codes)}銘柄)")
        print(f"{'='*70}")

        # 各日のセクター内ランキング
        for entry_t in ENTRY_TIMES:
            price_col = f'price_{entry_t}'
            # 各 (date, code) で「寄付→entry時点」の累積リターン
            sub = sec_daily.dropna(subset=[price_col, 'day_open', 'day_close']).copy()
            sub['cum_ret_to_entry'] = (sub[price_col] / sub['day_open'] - 1) * 10000
            sub['cum_ret_to_close'] = (sub['day_close'] / sub['day_open'] - 1) * 10000
            sub['ret_entry_to_close'] = (sub['day_close'] / sub[price_col] - 1) * 10000

            # 各日ごとに ranking
            sub['rank'] = sub.groupby('date')['cum_ret_to_entry'].rank()

            # Best-Worst spread (rank=1 worst, rank=N best)
            n_codes = sub.groupby('date')['code'].count().mode()[0]  # typical N
            # Reversion: long worst (rank=1), short best (rank=N)
            # → spread = ret(worst entry→close) - ret(best entry→close)
            worsts = sub[sub['rank'] == 1].set_index('date')['ret_entry_to_close']
            bests = sub[sub['rank'] == sub.groupby('date')['rank'].transform('max')].set_index('date')['ret_entry_to_close']
            spreads_rev = (worsts - bests).dropna()
            r = stats_summary(spreads_rev.values, f'{sector_name} {entry_t} BW Reversion')
            r['sector'] = sector_name
            r['entry'] = entry_t
            r['type'] = 'Reversion (Long worst / Short best)'
            results.append(r)

            # Momentum: long best, short worst
            spreads_mom = (bests - worsts).dropna()
            r = stats_summary(spreads_mom.values, f'{sector_name} {entry_t} BW Momentum')
            r['sector'] = sector_name
            r['entry'] = entry_t
            r['type'] = 'Momentum (Long best / Short worst)'
            results.append(r)

        # 全ペア (固定: 10:00判定/引け持ち)
        print(f"\n{sector_name}: 各ペアの 10:00判定 → 引け Reversion")
        entry_t = '10:00:00'
        price_col = f'price_{entry_t}'
        sub = sec_daily.dropna(subset=[price_col, 'day_open', 'day_close']).copy()
        sub['cum_ret_to_entry'] = (sub[price_col] / sub['day_open'] - 1) * 10000
        sub['ret_entry_to_close'] = (sub['day_close'] / sub[price_col] - 1) * 10000

        # 各ペア
        for i, code_a in enumerate(codes):
            for code_b in codes[i+1:]:
                df_a = sub[sub['code'] == code_a][['date', 'cum_ret_to_entry', 'ret_entry_to_close']].rename(
                    columns={'cum_ret_to_entry': 'ret_a', 'ret_entry_to_close': 'fwd_a'})
                df_b = sub[sub['code'] == code_b][['date', 'cum_ret_to_entry', 'ret_entry_to_close']].rename(
                    columns={'cum_ret_to_entry': 'ret_b', 'ret_entry_to_close': 'fwd_b'})
                merged = df_a.merge(df_b, on='date')
                if len(merged) < 20:
                    continue
                # A > B (Aがリード) → spread short A, long B → fwd_b - fwd_a
                merged['spread_morning'] = merged['ret_a'] - merged['ret_b']
                # 大きいときに reversion
                threshold = 50  # bps
                rev_signal = merged[merged['spread_morning'].abs() >= threshold].copy()
                rev_signal['ls_ret'] = np.where(
                    rev_signal['spread_morning'] > 0,
                    rev_signal['fwd_b'] - rev_signal['fwd_a'],  # A高い→ Short A, Long B
                    rev_signal['fwd_a'] - rev_signal['fwd_b'],  # B高い→ Long A, Short B
                )
                r = stats_summary(
                    rev_signal['ls_ret'].values,
                    f'{sector_name} ({codes_dict[code_a]}-{codes_dict[code_b]})',
                )
                if r and r.get('N', 0) > 30:
                    r['sector'] = sector_name
                    r['pair'] = f"{code_a}-{code_b}"
                    r['pair_label'] = f"{codes_dict[code_a]}-{codes_dict[code_b]}"
                    pair_results.append(r)

    res_df = pd.DataFrame(results)
    pair_df = pd.DataFrame(pair_results)

    print("\n=== Best-Worst Spread サマリ (全戦略) ===")
    cols = ['sector','entry','type','N','mean_raw','mean_net','t_stat','win_rate','sharpe']
    cleaned = res_df[cols].copy()
    print(cleaned.to_string(index=False))

    res_df.to_csv('bw_results.csv', index=False)
    pair_df.to_csv('pair_results.csv', index=False)

    # =====================================================================
    # ベスト戦略のエクイティカーブ
    # =====================================================================
    print("\n=== ペアトレード結果 (10:00判定, スプレッド>50bps) ===")
    if len(pair_df) > 0:
        print(pair_df[['sector','pair_label','N','mean_raw','mean_net','t_stat','win_rate','sharpe']].to_string(index=False))

    # ベスト Best-Worst戦略 (sharpe高い) のエクイティカーブ
    best_strategy = res_df.dropna(subset=['sharpe']).sort_values('sharpe', ascending=False).head(1)
    print(f"\n=== ベスト戦略: {best_strategy.iloc[0]['label']} (Sharpe={best_strategy.iloc[0]['sharpe']:.2f}) ===")

    # 各 (sector, entry_time) でエクイティカーブを保存
    print("\n年別パフォーマンス:")
    for sec_name in SECTORS.keys():
        codes = list(SECTORS[sec_name].keys())
        sec_daily = daily[daily['code'].isin(codes)].copy()
        entry_t = '10:00:00'
        price_col = f'price_{entry_t}'
        sub = sec_daily.dropna(subset=[price_col, 'day_open', 'day_close']).copy()
        sub['cum_ret_to_entry'] = (sub[price_col] / sub['day_open'] - 1) * 10000
        sub['ret_entry_to_close'] = (sub['day_close'] / sub[price_col] - 1) * 10000
        sub['rank'] = sub.groupby('date')['cum_ret_to_entry'].rank()
        worsts = sub[sub['rank'] == 1].set_index('date')['ret_entry_to_close']
        bests = sub[sub['rank'] == sub.groupby('date')['rank'].transform('max')].set_index('date')['ret_entry_to_close']
        spreads_rev = (worsts - bests).dropna()
        spreads_rev_df = spreads_rev.reset_index()
        spreads_rev_df.columns = ['date', 'spread']
        spreads_rev_df['date'] = pd.to_datetime(spreads_rev_df['date'])
        spreads_rev_df['year'] = spreads_rev_df['date'].dt.year
        for y, g in spreads_rev_df.groupby('year'):
            arr = g['spread'].values - COST_BPS
            n = len(arr)
            print(f"  {sec_name} {y}: N={n}, mean_net={arr.mean():.1f}, "
                  f"WR={(arr > 0).mean() * 100:.1f}%, "
                  f"Sharpe={arr.mean()/arr.std() * np.sqrt(252) if arr.std() > 0 else 0:.2f}")

    # =====================================================================
    # 図
    # =====================================================================
    fig = plt.figure(figsize=(16, 11), facecolor='white')
    plt.rcParams.update({
        'font.family': ['Hiragino Sans', 'IPAexGothic', 'sans-serif'],
        'axes.unicode_minus': False,
    })
    fig.suptitle('セクター内イントラデイ ロングショート戦略 (非鉄 / AI半導体, 2024/5-2026/5)',
                 fontsize=13, fontweight='bold', y=0.99)

    # ====== Heatmap: 戦略×時間×セクター ======
    # 非鉄 Reversion
    ax1 = fig.add_axes([0.05, 0.58, 0.27, 0.35])
    df_hm = res_df[res_df['type'].str.contains('Reversion')].pivot_table(
        index='entry', columns='sector', values='sharpe')
    im = ax1.imshow(df_hm.values, aspect='auto', cmap='RdYlGn', vmin=-1.5, vmax=1.5)
    ax1.set_xticks(range(len(df_hm.columns))); ax1.set_xticklabels(df_hm.columns)
    ax1.set_yticks(range(len(df_hm.index))); ax1.set_yticklabels(df_hm.index, fontsize=8)
    for i in range(len(df_hm.index)):
        for j in range(len(df_hm.columns)):
            ax1.text(j, i, f"{df_hm.values[i,j]:.2f}", ha='center', va='center',
                     fontsize=9, color='black' if abs(df_hm.values[i,j]) < 0.8 else 'white')
    ax1.set_title('Reversion Sharpe (worst→close Long / best→close Short)',
                  fontsize=9, fontweight='bold')
    plt.colorbar(im, ax=ax1, label='Sharpe')

    # Momentum
    ax2 = fig.add_axes([0.36, 0.58, 0.27, 0.35])
    df_hm2 = res_df[res_df['type'].str.contains('Momentum')].pivot_table(
        index='entry', columns='sector', values='sharpe')
    im2 = ax2.imshow(df_hm2.values, aspect='auto', cmap='RdYlGn', vmin=-1.5, vmax=1.5)
    ax2.set_xticks(range(len(df_hm2.columns))); ax2.set_xticklabels(df_hm2.columns)
    ax2.set_yticks(range(len(df_hm2.index))); ax2.set_yticklabels(df_hm2.index, fontsize=8)
    for i in range(len(df_hm2.index)):
        for j in range(len(df_hm2.columns)):
            ax2.text(j, i, f"{df_hm2.values[i,j]:.2f}", ha='center', va='center',
                     fontsize=9, color='black' if abs(df_hm2.values[i,j]) < 0.8 else 'white')
    ax2.set_title('Momentum Sharpe (best→close Long / worst→close Short)',
                  fontsize=9, fontweight='bold')
    plt.colorbar(im2, ax=ax2, label='Sharpe')

    # ペア結果
    ax3 = fig.add_axes([0.69, 0.58, 0.28, 0.35])
    ax3.axis('off')
    if len(pair_df) > 0:
        pair_show = pair_df.sort_values('sharpe', ascending=False).head(8)
        tbl = pair_show[['pair_label', 'N', 'mean_net', 't_stat', 'sharpe']].copy()
        tbl.columns = ['ペア', 'N', 'net(bps)', 't値', 'Sharpe']
        tbl['ペア'] = tbl['ペア'].str[:18]
        table = ax3.table(cellText=tbl.values, colLabels=tbl.columns,
                          cellLoc='center', loc='upper center',
                          bbox=[0, 0.1, 1, 0.85])
        table.auto_set_font_size(False)
        table.set_fontsize(8.5)
        for (r, c), cell in table.get_celld().items():
            if r == 0:
                cell.set_facecolor('#1565C0')
                cell.set_text_props(color='white', fontweight='bold')
            elif r > 0 and c >= 4:
                try:
                    v = float(tbl.iloc[r-1, c])
                    if v > 1.0:
                        cell.set_facecolor('#C8E6C9')
                    elif v < -0.5:
                        cell.set_facecolor('#FFCDD2')
                except: pass
            cell.set_edgecolor('#BDBDBD')
        ax3.set_title('ペアトレード Top8 (10:00 spread>50bps)', fontsize=10,
                      fontweight='bold', y=0.95)

    # 下: ベスト戦略のエクイティカーブ
    # 各セクター、10:00 Reversion
    ax4 = fig.add_axes([0.05, 0.08, 0.91, 0.43])
    colors = {'非鉄': '#FF9800', 'AI半導体': '#1565C0'}
    for sec_name in SECTORS.keys():
        codes = list(SECTORS[sec_name].keys())
        sec_daily = daily[daily['code'].isin(codes)].copy()
        entry_t = '10:00:00'
        price_col = f'price_{entry_t}'
        sub = sec_daily.dropna(subset=[price_col, 'day_open', 'day_close']).copy()
        sub['cum_ret_to_entry'] = (sub[price_col] / sub['day_open'] - 1) * 10000
        sub['ret_entry_to_close'] = (sub['day_close'] / sub[price_col] - 1) * 10000
        sub['rank'] = sub.groupby('date')['cum_ret_to_entry'].rank()
        worsts = sub[sub['rank'] == 1].set_index('date')['ret_entry_to_close']
        bests = sub[sub['rank'] == sub.groupby('date')['rank'].transform('max')].set_index('date')['ret_entry_to_close']
        spreads_rev = (worsts - bests).dropna() - COST_BPS

        equity = spreads_rev.sort_index().cumsum()
        # Sharpe
        arr = spreads_rev.values
        sharpe = arr.mean()/arr.std() * np.sqrt(252) if arr.std() > 0 else 0
        ax4.plot(pd.to_datetime(equity.index), equity.values,
                 color=colors[sec_name], lw=1.5,
                 label=f'{sec_name} Reversion (10:00判定) Sharpe={sharpe:.2f}, 累積={arr.sum():.0f}bps')

    ax4.axhline(0, color='black', lw=0.6)
    ax4.set_ylabel('累積 net PnL (bps, コスト8bps差引後)', fontsize=9)
    ax4.set_title('セクター内 LS Reversion (10:00 判定 → 大引け) エクイティカーブ',
                  fontsize=10, fontweight='bold')
    ax4.legend(fontsize=10, loc='best')
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%y/%m'))
    ax4.grid(alpha=0.3)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)

    fig.text(0.99, 0.005,
             'データ: stocks_intraday 2024-05〜2026-05 / 非鉄7銘柄+AI半導体7銘柄 | コスト8bps (LS往復)',
             ha='right', va='bottom', fontsize=7, color='gray')
    plt.savefig('result.png', dpi=100, bbox_inches='tight', facecolor='white')
    print(f"\nresult.png 保存完了 ({time.time()-t0:.1f}秒)")


if __name__ == '__main__':
    main()
