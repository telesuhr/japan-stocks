"""
F4戦略ロバストネス検証
「前日-1%以下 × 当日ギャップ-0.2%以下 → 引け買い・翌朝売り (ON)」

検証項目:
  1. 関税ショック除外期間でのパフォーマンス
  2. 月別・期間別分解
  3. セクター別分解
  4. パラメータ感度 (条件閾値)
  5. エクイティカーブ
"""

import psycopg2
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

PG_CONFIG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}

SECTOR_MAP = {
    '6920.T': '半導体', '6857.T': '半導体', '8035.T': '半導体',
    '6146.T': '半導体', '6273.T': '半導体', '6861.T': '半導体',
    '6762.T': '電機', '6752.T': '電機', '6503.T': '電機',
    '6954.T': '電機', '6501.T': '電機', '6594.T': '電機',
    '6702.T': 'IT', '6981.T': '電機',
    '5713.T': '非鉄', '5711.T': '非鉄', '5706.T': '非鉄',
    '5714.T': '非鉄', '5801.T': '非鉄', '5802.T': '非鉄', '5803.T': '非鉄',
    '9104.T': '海運', '9107.T': '海運', '9101.T': '海運',
    '1605.T': 'エネルギー', '5020.T': 'エネルギー', '5016.T': 'エネルギー',
    '7203.T': '自動車', '7267.T': '自動車', '7201.T': '自動車', '7270.T': '自動車',
    '5401.T': '鉄鋼', '5411.T': '鉄鋼', '4063.T': '化学',
    '8306.T': '銀行', '8316.T': '銀行', '8411.T': '銀行',
    '8604.T': '証券', '8308.T': '銀行',
    '8053.T': '商社', '8058.T': '商社', '8001.T': '商社',
    '8015.T': '商社', '8002.T': '商社', '8031.T': '商社',
    '9432.T': '通信', '9433.T': '通信', '9434.T': '通信',
    '7011.T': '重工', '7012.T': '重工', '7013.T': '重工',
    '9984.T': 'IT', '9983.T': '小売',
    '2914.T': '食品', '2503.T': '食品', '2502.T': '食品', '2801.T': '食品',
    '4661.T': 'レジャー', '6098.T': 'サービス',
    '8113.T': '生活用品', '9023.T': '鉄道',
    '4502.T': '製薬', '4503.T': '製薬', '4523.T': '製薬',
}

# 関税ショック期間 (2025/4/2~4/25: 相互関税発表→一部猶予発表)
TARIFF_SHOCK_START = pd.Timestamp('2025-04-02')
TARIFF_SHOCK_END   = pd.Timestamp('2025-04-25')


def load_daily(start='2025-01-01'):
    conn = psycopg2.connect(**PG_CONFIG)
    q = f"""
        SELECT symbol, timestamp, open, high, low, close, volume
        FROM intraday_data
        WHERE interval = '1min' AND symbol LIKE '%.T'
          AND timestamp >= '{start}'
        ORDER BY symbol, timestamp
    """
    print("データ読み込み中...")
    raw = pd.read_sql(q, conn)
    conn.close()
    raw['timestamp'] = pd.to_datetime(raw['timestamp'])
    raw['jst'] = raw['timestamp'] + pd.Timedelta(hours=9)
    raw = raw.dropna(subset=['open', 'close'])
    print(f"  {len(raw):,}行, {raw['symbol'].nunique()}銘柄")
    return raw


def build_daily(raw):
    records = []
    for symbol, g in raw.groupby('symbol'):
        g = g.set_index('jst').sort_index()
        for date, day in g.groupby(g.index.date):
            morning = day[
                (day.index.hour >= 9) & (
                    (day.index.hour < 11) |
                    ((day.index.hour == 11) & (day.index.minute <= 30))
                )
            ]
            if len(morning) < 10:
                continue
            open_price = morning['open'].iloc[0]
            close_price = day['close'].iloc[-1]
            fullday_ret = (close_price / open_price - 1) * 10000 if open_price > 0 else np.nan
            records.append({
                'symbol': symbol,
                'date': pd.Timestamp(date),
                'open': open_price,
                'close': close_price,
                'fullday_ret': fullday_ret,
            })
    df = pd.DataFrame(records)
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)

    for sym, g in df.groupby('symbol'):
        idx = g.index
        df.loc[idx, 'prev_close'] = g['close'].shift(1).values
        df.loc[idx, 'prev_fullday_ret'] = g['fullday_ret'].shift(1).values
        # ON ret: 当日引け → 翌日寄付
        df.loc[idx, 'on_ret'] = (g['open'].shift(-1) / g['close'] - 1).values * 10000

    df['gap_ret'] = (df['open'] / df['prev_close'] - 1) * 10000
    df['sector'] = df['symbol'].map(SECTOR_MAP).fillna('その他')
    return df.dropna(subset=['prev_close', 'gap_ret', 'on_ret'])


def f4_trades(df, prev_thresh=-100, gap_thresh=-20):
    """F4条件でトレードを抽出"""
    cond = (df['prev_fullday_ret'] <= prev_thresh) & (df['gap_ret'] <= gap_thresh)
    return df[cond].copy()


def stats_summary(returns, label='', cost=4):
    arr = np.array(returns)
    net = arr - cost
    n = len(arr)
    if n < 10:
        return None
    t, p = stats.ttest_1samp(arr, 0)
    return {
        'label': label,
        'N': n,
        'mean_raw': round(arr.mean(), 1),
        'mean_net': round(net.mean(), 1),
        'std': round(arr.std(), 1),
        't_stat': round(t, 2),
        'p_val': round(p, 4),
        'win_rate': round((arr > 0).mean() * 100, 1),
        'sharpe': round(net.mean() / arr.std() * np.sqrt(252), 2) if arr.std() > 0 else 0,
    }


if __name__ == '__main__':
    raw = load_daily()
    print("日次フィーチャー構築中...")
    df = build_daily(raw)
    print(f"  有効レコード: {len(df):,}")

    trades_all = f4_trades(df)
    print(f"\nF4条件合致: {len(trades_all)}件 ({trades_all['symbol'].nunique()}銘柄)")

    # ---- 1. 期間別分解 ----
    shock = trades_all[trades_all['date'].between(TARIFF_SHOCK_START, TARIFF_SHOCK_END)]
    no_shock = trades_all[~trades_all['date'].between(TARIFF_SHOCK_START, TARIFF_SHOCK_END)]
    pre_shock  = trades_all[trades_all['date'] < TARIFF_SHOCK_START]
    post_shock = trades_all[trades_all['date'] > TARIFF_SHOCK_END]

    period_results = []
    for label, sub in [
        ('全期間 (2025/1~)', trades_all),
        ('関税ショック除外', no_shock),
        ('ショック前 (~4/1)', pre_shock),
        ('ショック期間 (4/2~4/25)', shock),
        ('ショック後 (4/26~)', post_shock),
    ]:
        r = stats_summary(sub['on_ret'], label)
        if r:
            period_results.append(r)

    period_df = pd.DataFrame(period_results)
    print("\n=== 期間別 ===")
    print(period_df[['label','N','mean_raw','mean_net','std','t_stat','p_val','win_rate','sharpe']].to_string(index=False))

    # ---- 2. 月別 ----
    trades_all['ym'] = trades_all['date'].dt.to_period('M')
    monthly = []
    for ym, g in trades_all.groupby('ym'):
        r = stats_summary(g['on_ret'], str(ym))
        if r:
            r['is_shock'] = (ym == pd.Period('2025-04', 'M'))
            monthly.append(r)
    monthly_df = pd.DataFrame(monthly)
    print("\n=== 月別 ===")
    print(monthly_df[['label','N','mean_raw','mean_net','t_stat','win_rate','sharpe','is_shock']].to_string(index=False))

    # ---- 3. セクター別 (ショック除外) ----
    sector_results = []
    for sec, g in no_shock.groupby('sector'):
        r = stats_summary(g['on_ret'], sec)
        if r:
            sector_results.append(r)
    sector_df = pd.DataFrame(sector_results).sort_values('sharpe', ascending=False)
    print("\n=== セクター別 (ショック除外) ===")
    print(sector_df[['label','N','mean_net','t_stat','win_rate','sharpe']].to_string(index=False))

    # ---- 4. パラメータ感度 ----
    print("\n=== パラメータ感度 (ショック除外) ===")
    param_results = []
    for prev_t in [-50, -80, -100, -150, -200]:
        for gap_t in [-10, -15, -20, -30, -50]:
            sub = f4_trades(df[~df['date'].between(TARIFF_SHOCK_START, TARIFF_SHOCK_END)],
                            prev_thresh=prev_t, gap_thresh=gap_t)
            r = stats_summary(sub['on_ret'], f'prev<={prev_t}, gap<={gap_t}')
            if r:
                r['prev_thresh'] = prev_t
                r['gap_thresh'] = gap_t
                param_results.append(r)
    param_df = pd.DataFrame(param_results)
    pivot = param_df.pivot_table(index='prev_thresh', columns='gap_thresh',
                                  values='mean_net', aggfunc='first')
    print("mean_net_bps (ショック除外):")
    print(pivot.round(1).to_string())
    pivot_t = param_df.pivot_table(index='prev_thresh', columns='gap_thresh',
                                    values='t_stat', aggfunc='first')
    print("\nt_stat:")
    print(pivot_t.round(2).to_string())

    # ---- 5. エクイティカーブ (日次累積) ----
    # 日付ベースで集計 (同日複数銘柄は平均)
    trades_all_sorted = trades_all.sort_values('date')
    daily_avg = trades_all_sorted.groupby('date')['on_ret'].mean()
    daily_avg_no_shock = no_shock.sort_values('date').groupby('date')['on_ret'].mean()

    # ---- 図作成 ----
    fig = plt.figure(figsize=(16, 10), facecolor='white')
    plt.rcParams.update({
        'font.family': ['Hiragino Sans', 'IPAexGothic', 'sans-serif'],
        'axes.unicode_minus': False,
    })
    fig.suptitle('F4戦略ロバストネス検証\n「前日-1%以下×ギャップ-0.2%以下 → 引け買い翌朝売り」',
                 fontsize=13, fontweight='bold', y=0.99)

    # ---- A. エクイティカーブ ----
    ax1 = fig.add_axes([0.05, 0.63, 0.55, 0.32])
    cum_all = (daily_avg - 4).cumsum()
    cum_no  = (daily_avg_no_shock - 4).cumsum()
    ax1.plot(cum_all.index, cum_all.values, color='#2196F3', lw=1.5, label='全期間')
    ax1.plot(cum_no.index, cum_no.values, color='#4CAF50', lw=1.5, label='ショック除外')
    # ショック期間をハイライト
    ax1.axvspan(TARIFF_SHOCK_START, TARIFF_SHOCK_END, alpha=0.15, color='red', label='関税ショック期')
    ax1.axhline(0, color='black', lw=0.8)
    ax1.set_ylabel('累積リターン (bps)', fontsize=9)
    ax1.set_title('エクイティカーブ (日次平均、コスト4bps差引)', fontsize=10, fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # ---- B. 月別棒グラフ ----
    ax2 = fig.add_axes([0.63, 0.63, 0.34, 0.32])
    colors_m = ['#FF5722' if s else '#42A5F5' for s in monthly_df['is_shock']]
    ax2.bar(range(len(monthly_df)), monthly_df['mean_net'], color=colors_m, alpha=0.8)
    ax2.axhline(0, color='black', lw=0.8)
    ax2.set_xticks(range(len(monthly_df)))
    ax2.set_xticklabels(monthly_df['label'], rotation=45, ha='right', fontsize=7)
    ax2.set_ylabel('月別 mean_net (bps)', fontsize=9)
    ax2.set_title('月別パフォーマンス\n(赤=ショック月)', fontsize=10, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # ---- C. パラメータ感度ヒートマップ ----
    ax3 = fig.add_axes([0.05, 0.10, 0.40, 0.45])
    data_hm = pivot.values
    im = ax3.imshow(data_hm, aspect='auto', cmap='RdYlGn',
                    vmin=-20, vmax=60)
    ax3.set_xticks(range(len(pivot.columns)))
    ax3.set_xticklabels([f'gap≤{c}' for c in pivot.columns], fontsize=8)
    ax3.set_yticks(range(len(pivot.index)))
    ax3.set_yticklabels([f'prev≤{r}' for r in pivot.index], fontsize=8)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = data_hm[i, j]
            t_v = pivot_t.values[i, j]
            if not np.isnan(v):
                color = 'white' if abs(v) > 30 else 'black'
                ax3.text(j, i, f'{v:.0f}\n(t={t_v:.1f})',
                         ha='center', va='center', fontsize=7, color=color)
    plt.colorbar(im, ax=ax3, label='mean_net (bps)')
    ax3.set_title('パラメータ感度 — mean_net bps (ショック除外)\n縦: 前日リターン閾値, 横: ギャップ閾値',
                  fontsize=9, fontweight='bold')

    # ---- D. 期間別サマリー表 ----
    ax4 = fig.add_axes([0.50, 0.10, 0.47, 0.45])
    ax4.axis('off')

    # 期間比較テーブル
    tbl_data = period_df[['label', 'N', 'mean_raw', 'mean_net', 't_stat', 'win_rate', 'sharpe']].copy()
    tbl_data.columns = ['期間', 'N', 'raw(bps)', 'net(bps)', 't値', '勝率%', 'Sharpe']

    table = ax4.table(
        cellText=tbl_data.values,
        colLabels=tbl_data.columns,
        cellLoc='center',
        loc='upper center',
        bbox=[0, 0.55, 1, 0.44]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor('#1565C0')
            cell.set_text_props(color='white', fontweight='bold')
        elif tbl_data.values[r-1][0] == 'ショック期間 (4/2~4/25)' if r > 0 else False:
            cell.set_facecolor('#FFCDD2')
        elif r % 2 == 0:
            cell.set_facecolor('#E3F2FD')
        cell.set_edgecolor('#BDBDBD')
    ax4.set_title('期間別パフォーマンス比較', fontsize=10, fontweight='bold', y=1.01)

    # セクター別グラフ (ショック除外、Sharpe降順)
    sec_top = sector_df.head(10)
    ys = range(len(sec_top))
    colors_s = ['#4CAF50' if v > 0 else '#F44336' for v in sec_top['mean_net']]
    ax4b_x = [0.50, 0.10, 0.47, 0.44]
    ax4b = fig.add_axes([0.50, 0.10, 0.47, 0.40])
    ax4b.axis('off')  # reuse as text area

    # bar chart for sectors
    ax4b.set_visible(False)
    ax5 = fig.add_axes([0.50, 0.10, 0.47, 0.40])
    bars = ax5.barh(list(ys), sec_top['mean_net'], color=colors_s, alpha=0.8, height=0.6)
    ax5.set_yticks(list(ys))
    ax5.set_yticklabels(sec_top['label'], fontsize=9)
    ax5.axvline(0, color='black', lw=0.8)
    for i, (_, row) in enumerate(sec_top.iterrows()):
        ax5.text(row['mean_net'] + (1 if row['mean_net'] >= 0 else -1),
                 i, f"t={row['t_stat']:.1f}", va='center', fontsize=7.5,
                 ha='left' if row['mean_net'] >= 0 else 'right')
    ax5.set_xlabel('mean_net (bps)', fontsize=9)
    ax5.set_title('セクター別 (ショック除外)', fontsize=10, fontweight='bold')
    ax5.grid(axis='x', alpha=0.3)
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)

    fig.text(0.99, 0.005,
             'データ: 2025-01-01〜2026-05-07 / 日本株117銘柄1分足 | F4: prev≤-100bps & gap≤-20bps → ON buy',
             ha='right', va='bottom', fontsize=7, color='gray')

    plt.savefig('result.png', dpi=100, bbox_inches='tight', facecolor='white')
    print("\nresult.png 保存完了")

    # CSV保存
    period_df.to_csv('period_results.csv', index=False)
    monthly_df.to_csv('monthly_results.csv', index=False)
    param_df.to_csv('param_sensitivity.csv', index=False)
    sector_df.to_csv('sector_results.csv', index=False)
