"""
ティック由来 新規シグナル広範探索

これまでに試した: 大口買い・ティック数スパイク・各種瞬間異常
これから試す未検証シグナル:

  A. Order Flow Imbalance (OFI): 上昇ティック数 vs 下降ティック数
  B. Average Trade Size の変化: 1約定の平均株数が増/減
  C. Trade Speed: ティック間隔の縮短 (高頻度化)
  D. Large Print Count: 大口プリント数の累積
  E. Price-Volume Divergence: 価格上昇 × 出来高減 (or 逆)
  F. Cumulative Buy-Sell Imbalance: アップティック株数 - ダウンティック株数
  G. Tick Variance: ティック単位の価格変動率の標準偏差
  H. Run Length: 連続up tick/down tickの長さ
  I. Spread Proxy: high-low / close (1分間)
  J. Volume Concentration: 1分間の最大単発 / 合計出来高

対象: SBG (99840), 2026-01-05 ~ 2026-04-30 (4ヶ月)
バー: 1分足にティック由来特徴量を集約
ターゲット: fwd_5min / fwd_15min / fwd_30min / fwd_60min
"""
import sys
sys.path.insert(0, '/Users/Yusuke/claude-code/DataFetcher')
from src.ticks import _file_globs, _to_date, _norm5, _norm4

import duckdb
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

COST_BPS = 4
CODE = '99840'
START = '2026-01-01'
END   = '2026-04-30'
CACHE = Path('sbg_rich_bars.parquet')


def build_rich_bars(start=START, end=END):
    """SBGの1分バー + リッチなティック由来特徴量
    各バーごとに:
      - OHLC, volume, turnover
      - tick_count
      - up_tick_count (uptick数)
      - down_tick_count (downtick数)
      - up_volume (upticks の合計vol)
      - down_volume (downticks の合計vol)
      - large_count (vol >= 8200 の約定数)
      - max_single_vol (最大単発)
    """
    s = _to_date(start)
    e = _to_date(end)
    root = Path.home() / "Data" / "jquants_trades" / "equities" / "trades"
    files = _file_globs(root, s, e)
    files_arr = "[" + ", ".join(f"'{f}'" for f in files) + "]"
    csv_part = (
        f"read_csv({files_arr}, header=true, "
        "columns={'Date':'DATE','Code':'VARCHAR','Time':'VARCHAR',"
        "'SessionDistinction':'VARCHAR','Price':'DOUBLE','TradingVolume':'BIGINT',"
        "'TransactionId':'VARCHAR'})"
    )
    # まずティックを取得 (DuckDB)
    sql = f"""
        SELECT Date AS d, Time AS tm, Price AS price, TradingVolume AS vol,
               date_add(CAST(Date AS TIMESTAMP),
                        INTERVAL (
                            EXTRACT(HOUR FROM CAST(Time AS TIME)) * 3600
                          + EXTRACT(MINUTE FROM CAST(Time AS TIME)) * 60
                          + EXTRACT(SECOND FROM CAST(Time AS TIME))
                        ) SECOND) AS ts_sec
        FROM {csv_part}
        WHERE Date BETWEEN ? AND ? AND Code IN (?, ?)
        ORDER BY ts_sec
    """
    con = duckdb.connect()
    con.execute("PRAGMA threads=4")
    print("ティック取得中...")
    ticks = con.execute(sql, [s, e, _norm5(CODE), _norm4(CODE)]).df()
    con.close()
    print(f"  {len(ticks):,} ticks")

    # uptick/downtick 判定
    ticks = ticks.sort_values('ts_sec').reset_index(drop=True)
    ticks['prev_price'] = ticks['price'].shift(1)
    ticks['diff'] = ticks['price'] - ticks['prev_price']
    ticks['is_up'] = (ticks['diff'] > 0).astype(int)
    ticks['is_down'] = (ticks['diff'] < 0).astype(int)
    ticks['up_vol'] = ticks['vol'] * ticks['is_up']
    ticks['down_vol'] = ticks['vol'] * ticks['is_down']
    ticks['is_large'] = (ticks['vol'] >= 8200).astype(int)
    ticks['bar_ts'] = ticks['ts_sec'].dt.floor('1min')

    print("1分バー集計中...")
    bars = ticks.groupby('bar_ts').agg(
        open=('price', 'first'),
        high=('price', 'max'),
        low=('price', 'min'),
        close=('price', 'last'),
        volume=('vol', 'sum'),
        tick_count=('vol', 'size'),
        up_count=('is_up', 'sum'),
        down_count=('is_down', 'sum'),
        up_volume=('up_vol', 'sum'),
        down_volume=('down_vol', 'sum'),
        large_count=('is_large', 'sum'),
        max_single_vol=('vol', 'max'),
    ).reset_index()

    bars['date'] = bars['bar_ts'].dt.date

    # 取引時間内のみ
    h = bars['bar_ts'].dt.hour
    m = bars['bar_ts'].dt.minute
    in_session = (
        ((h == 9) | (h == 10) | ((h == 11) & (m <= 29))) |
        (((h == 12) & (m >= 30)) | (h == 13) | ((h == 14)) | ((h == 15) & (m == 0)))
    )
    bars = bars[in_session].copy().reset_index(drop=True)
    print(f"  {len(bars):,} bars")

    return bars


def add_features_and_targets(bars):
    """バーごとに特徴量とフォワードリターンを計算"""
    out = []
    for d, g in bars.groupby('date'):
        g = g.sort_values('bar_ts').reset_index(drop=True)
        if len(g) < 30:
            continue

        # 各特徴量
        # === OFI系 ===
        g['ofi'] = (g['up_volume'] - g['down_volume']) / (g['volume'] + 1)  # -1〜1
        g['ofi_count'] = (g['up_count'] - g['down_count']) / (g['tick_count'] + 1)
        g['avg_trade_size'] = g['volume'] / (g['tick_count'] + 1)

        # === 累積系 (前バーまで) ===
        g['cum_ofi_5b'] = g['ofi'].shift(1).rolling(5, min_periods=2).sum()
        g['cum_ofi_15b'] = g['ofi'].shift(1).rolling(15, min_periods=5).sum()
        g['cum_large_5b'] = g['large_count'].shift(1).rolling(5, min_periods=2).sum()

        # === 比率系 (現バーの異常度) ===
        # 直前30バー平均
        for col in ['tick_count', 'volume', 'large_count', 'avg_trade_size']:
            ma = g[col].shift(1).rolling(30, min_periods=10).mean()
            g[f'{col}_ratio'] = g[col] / (ma + 1e-6)

        # === 価格変動・spread ===
        g['bar_ret'] = (g['close'] / g['open'] - 1) * 10000
        g['bar_range'] = (g['high'] - g['low']) / g['close'] * 10000
        g['bar_range_ma30'] = g['bar_range'].shift(1).rolling(30, min_periods=10).mean()
        g['range_ratio'] = g['bar_range'] / (g['bar_range_ma30'] + 1e-6)

        # === Volume concentration ===
        g['concentration'] = g['max_single_vol'] / (g['volume'] + 1)

        # === 直近の累積モメンタム ===
        g['cum_ret_5b'] = (g['close'] / g['close'].shift(5) - 1) * 10000
        g['cum_ret_15b'] = (g['close'] / g['close'].shift(15) - 1) * 10000

        # === 時刻 ===
        g['minute_of_day'] = g['bar_ts'].dt.hour * 60 + g['bar_ts'].dt.minute

        # === ターゲット ===
        for k, label in [(5, 'fwd_5min'), (15, 'fwd_15min'), (30, 'fwd_30min'), (60, 'fwd_60min')]:
            g[label] = (g['close'].shift(-k) / g['close'] - 1) * 10000

        out.append(g)

    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def test_rule(df, cond, target, label, direction=1):
    sub = df[cond & df[target].notna()]
    if len(sub) < 100:
        return None
    arr = sub[target].values * direction
    net = arr - COST_BPS
    n = len(arr)
    t, p = stats.ttest_1samp(arr, 0)
    sharpe = net.mean() / arr.std() * np.sqrt(252 * 60) if arr.std() > 0 else 0
    return dict(rule=label, target=target, dir=direction, N=n,
                mean_raw=round(arr.mean(), 2), mean_net=round(net.mean(), 2),
                t_stat=round(t, 2), p_val=round(p, 4),
                win_rate=round((arr > 0).mean() * 100, 1),
                sharpe=round(sharpe, 2),
                sig=(p < 0.05 and net.mean() > 0))


def main():
    import time
    t0 = time.time()

    if CACHE.exists():
        print(f"キャッシュからロード: {CACHE}")
        feats = pd.read_parquet(CACHE)
    else:
        bars = build_rich_bars()
        print("\n特徴量＋フォワードリターン計算...")
        feats = add_features_and_targets(bars)
        feats.to_parquet(CACHE)
        print(f"  キャッシュ保存: {CACHE} ({len(feats):,}行)")

    print(f"\n総レコード: {len(feats):,}")

    # =====================================================================
    # ルール検証 (合計 ~40ルール)
    # =====================================================================
    print("\n" + "="*70)
    print("ルール検証")
    print("="*70)

    res = []

    # ========== A. OFI (Order Flow Imbalance) ==========
    for k, t in [(5,'fwd_5min'),(15,'fwd_15min'),(30,'fwd_30min'),(60,'fwd_60min')]:
        # OFI 強買い → 継続
        cond = feats['ofi'] >= 0.3
        r = test_rule(feats, cond, t, f'A1_OFI≥0.3_Long_{k}min')
        if r: res.append(r)
        # OFI 強売り → 継続
        cond = feats['ofi'] <= -0.3
        r = test_rule(feats, cond, t, f'A2_OFI≤-0.3_Short_{k}min', direction=-1)
        if r: res.append(r)

    # OFI 累積
    cond = feats['cum_ofi_15b'] >= 5.0
    r = test_rule(feats, cond, 'fwd_30min', 'A3_CumOFI15b≥5_Long_30min')
    if r: res.append(r)
    cond = feats['cum_ofi_15b'] <= -5.0
    r = test_rule(feats, cond, 'fwd_30min', 'A4_CumOFI15b≤-5_Short_30min', direction=-1)
    if r: res.append(r)

    # ========== B. Average Trade Size ==========
    cond = feats['avg_trade_size_ratio'] >= 2.0
    r = test_rule(feats, cond, 'fwd_15min', 'B1_AvgTradeSize≥2x_Buy_15min')
    if r: res.append(r)
    r = test_rule(feats, cond, 'fwd_30min', 'B2_AvgTradeSize≥2x_Buy_30min')
    if r: res.append(r)

    cond = (feats['avg_trade_size_ratio'] >= 2.0) & (feats['bar_ret'] > 0)
    r = test_rule(feats, cond, 'fwd_30min', 'B3_AvgTradeSize≥2x_Up_Buy_30min')
    if r: res.append(r)
    cond = (feats['avg_trade_size_ratio'] >= 2.0) & (feats['bar_ret'] < 0)
    r = test_rule(feats, cond, 'fwd_30min', 'B4_AvgTradeSize≥2x_Down_Short_30min', direction=-1)
    if r: res.append(r)

    # ========== C. Trade Speed (tick_count_ratio) ==========
    cond = feats['tick_count_ratio'] >= 3.0
    r = test_rule(feats, cond, 'fwd_15min', 'C1_TickSpeed≥3x_15min')
    if r: res.append(r)
    cond = (feats['tick_count_ratio'] >= 3.0) & (feats['bar_ret'] > 0)
    r = test_rule(feats, cond, 'fwd_30min', 'C2_TickSpeed×Up_Buy_30min')
    if r: res.append(r)
    cond = (feats['tick_count_ratio'] >= 3.0) & (feats['bar_ret'] < 0)
    r = test_rule(feats, cond, 'fwd_30min', 'C3_TickSpeed×Down_Short_30min', direction=-1)
    if r: res.append(r)

    # ========== D. Large Print Count ==========
    cond = feats['large_count'] >= 3
    r = test_rule(feats, cond, 'fwd_15min', 'D1_Large≥3_Buy_15min')
    if r: res.append(r)
    cond = feats['large_count'] >= 5
    r = test_rule(feats, cond, 'fwd_30min', 'D2_Large≥5_Buy_30min')
    if r: res.append(r)
    cond = feats['cum_large_5b'] >= 10
    r = test_rule(feats, cond, 'fwd_30min', 'D3_CumLarge5b≥10_Buy_30min')
    if r: res.append(r)

    # ========== E. Price-Volume Divergence ==========
    # 価格上昇 × 出来高小 → 反転?
    cond = (feats['bar_ret'] >= 30) & (feats['volume_ratio'] <= 0.7)
    r = test_rule(feats, cond, 'fwd_15min', 'E1_PriceUp×VolLow_Reverse_Short', direction=-1)
    if r: res.append(r)
    # 価格下落 × 出来高小 → 反転?
    cond = (feats['bar_ret'] <= -30) & (feats['volume_ratio'] <= 0.7)
    r = test_rule(feats, cond, 'fwd_15min', 'E2_PriceDown×VolLow_Reverse_Buy')
    if r: res.append(r)

    # ========== F. Volatility Expansion ==========
    cond = feats['range_ratio'] >= 3.0
    r = test_rule(feats, cond, 'fwd_15min', 'F1_VolExpand_Any_15min')
    if r: res.append(r)
    cond = (feats['range_ratio'] >= 3.0) & (feats['bar_ret'] > 0)
    r = test_rule(feats, cond, 'fwd_30min', 'F2_VolExpand×Up_Buy_30min')
    if r: res.append(r)
    cond = (feats['range_ratio'] >= 3.0) & (feats['bar_ret'] < 0)
    r = test_rule(feats, cond, 'fwd_30min', 'F3_VolExpand×Down_Short_30min', direction=-1)
    if r: res.append(r)

    # ========== G. Volume Concentration ==========
    # 1分の単発が出来高の30%超 → 機関の大口?
    cond = feats['concentration'] >= 0.3
    r = test_rule(feats, cond, 'fwd_30min', 'G1_Concentration≥0.3_Buy_30min')
    if r: res.append(r)
    cond = (feats['concentration'] >= 0.3) & (feats['bar_ret'] > 0)
    r = test_rule(feats, cond, 'fwd_30min', 'G2_Concentration×Up_Buy_30min')
    if r: res.append(r)
    cond = (feats['concentration'] >= 0.3) & (feats['bar_ret'] < 0)
    r = test_rule(feats, cond, 'fwd_30min', 'G3_Concentration×Down_Short_30min', direction=-1)
    if r: res.append(r)

    # ========== H. Mean Reversion ==========
    # 直近15bar (15min) で急騰 → 反転
    cond = feats['cum_ret_15b'] >= 50
    r = test_rule(feats, cond, 'fwd_15min', 'H1_15min_Up50_Reverse_Short', direction=-1)
    if r: res.append(r)
    cond = feats['cum_ret_15b'] <= -50
    r = test_rule(feats, cond, 'fwd_15min', 'H2_15min_Down50_Reverse_Buy')
    if r: res.append(r)
    # 直近5min
    cond = feats['cum_ret_5b'] >= 30
    r = test_rule(feats, cond, 'fwd_5min', 'H3_5min_Up30_Reverse_Short', direction=-1)
    if r: res.append(r)
    cond = feats['cum_ret_5b'] <= -30
    r = test_rule(feats, cond, 'fwd_5min', 'H4_5min_Down30_Reverse_Buy')
    if r: res.append(r)

    # ========== I. Momentum (継続) ==========
    cond = feats['cum_ret_15b'] >= 50
    r = test_rule(feats, cond, 'fwd_30min', 'I1_15min_Up50_Continue_Buy')
    if r: res.append(r)
    cond = feats['cum_ret_15b'] <= -50
    r = test_rule(feats, cond, 'fwd_30min', 'I2_15min_Down50_Continue_Short', direction=-1)
    if r: res.append(r)

    # ========== J. OFI × その他 ==========
    cond = (feats['ofi'] >= 0.3) & (feats['volume_ratio'] >= 2.0)
    r = test_rule(feats, cond, 'fwd_30min', 'J1_OFI≥0.3×Vol≥2x_Buy_30min')
    if r: res.append(r)
    cond = (feats['ofi'] <= -0.3) & (feats['volume_ratio'] >= 2.0)
    r = test_rule(feats, cond, 'fwd_30min', 'J2_OFI≤-0.3×Vol≥2x_Short_30min', direction=-1)
    if r: res.append(r)

    # ========== K. 時間帯 × OFI ==========
    # 寄付30分 × OFI強買い
    open_mask = (feats['minute_of_day'] >= 540) & (feats['minute_of_day'] < 570)
    cond = open_mask & (feats['ofi'] >= 0.3)
    r = test_rule(feats, cond, 'fwd_30min', 'K1_Open30min×OFI≥0.3_Buy')
    if r: res.append(r)
    cond = open_mask & (feats['ofi'] <= -0.3)
    r = test_rule(feats, cond, 'fwd_30min', 'K2_Open30min×OFI≤-0.3_Short', direction=-1)
    if r: res.append(r)

    # ============================================================
    # 結果集計
    # ============================================================
    results = pd.DataFrame(res).sort_values('t_stat', ascending=False)
    sig = results[results['sig']]

    print(f"\n全ルール: {len(results)}")
    print(f"有意 (p<0.05, net>0): {len(sig)}件")

    print("\n=== 上位 (t値降順) ===")
    cols = ['rule','target','N','mean_raw','mean_net','t_stat','p_val','win_rate','sharpe','sig']
    print(results.head(20)[cols].to_string(index=False))

    if len(sig) > 0:
        print("\n=== 有望ルール ===")
        print(sig[cols].to_string(index=False))

    results.to_csv('results_all.csv', index=False)
    sig.to_csv('results_significant.csv', index=False)

    # ============================================================
    # 図
    # ============================================================
    fig = plt.figure(figsize=(15, 10), facecolor='white')
    plt.rcParams.update({
        'font.family': ['Hiragino Sans', 'IPAexGothic', 'sans-serif'],
        'axes.unicode_minus': False,
    })
    fig.suptitle('ティック由来 新規シグナル広範探索 (SBG 2026-01〜04 / 約40ルール)',
                 fontsize=13, fontweight='bold', y=0.99)

    def col(rule):
        if rule.startswith('A') or rule.startswith('J') or rule.startswith('K'): return '#1E88E5'  # OFI
        if rule.startswith('B'): return '#FF9800'  # AvgTradeSize
        if rule.startswith('C'): return '#43A047'  # TickSpeed
        if rule.startswith('D'): return '#9C27B0'  # Large
        if rule.startswith('E'): return '#00BCD4'  # Divergence
        if rule.startswith('F'): return '#E91E63'  # Vol Expand
        if rule.startswith('G'): return '#795548'  # Concentration
        if rule.startswith('H'): return '#3F51B5'  # MeanRev
        if rule.startswith('I'): return '#F44336'  # Momentum
        return 'gray'

    # ---- 左: 全ルールのt値ランキング ----
    ax1 = fig.add_axes([0.04, 0.05, 0.42, 0.90])
    sorted_r = results.sort_values('t_stat')
    ys = range(len(sorted_r))
    colors = [col(r) for r in sorted_r['rule']]
    ax1.barh(list(ys), sorted_r['t_stat'].values, color=colors, alpha=0.85, height=0.7)
    ax1.set_yticks(list(ys))
    ax1.set_yticklabels(sorted_r['rule'], fontsize=6.5)
    ax1.axvline(0, color='black', lw=0.6)
    ax1.axvline(1.96, color='gray', lw=0.6, linestyle='--', alpha=0.6)
    ax1.axvline(-1.96, color='gray', lw=0.6, linestyle='--', alpha=0.6)
    ax1.set_xlabel('t統計量', fontsize=9)
    ax1.set_title('全ルール t値', fontsize=10, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # ---- 右上: スキャッター ----
    ax2 = fig.add_axes([0.51, 0.55, 0.46, 0.40])
    for _, row in results.iterrows():
        c = col(row['rule'])
        m = '*' if row['sig'] else 'o'
        s = 110 if row['sig'] else 30
        ax2.scatter(row['t_stat'], row['mean_net'], c=c, marker=m, s=s,
                    alpha=0.85 if row['sig'] else 0.4,
                    zorder=5 if row['sig'] else 3,
                    edgecolor='black', lw=0.4)
    ax2.axhline(0, color='black', lw=0.7)
    ax2.axvline(0, color='black', lw=0.7)
    ax2.axvline(1.96, color='gray', lw=0.6, linestyle='--', alpha=0.6)
    ax2.set_xlabel('t統計量', fontsize=9)
    ax2.set_ylabel('mean_net (bps)', fontsize=9)
    ax2.set_title('t値 vs net期待値 (★=有意)', fontsize=10, fontweight='bold')

    # 凡例
    legend_groups = {
        'A,J,K: OFI系': '#1E88E5',
        'B: AvgTradeSize': '#FF9800',
        'C: TickSpeed': '#43A047',
        'D: LargePrint': '#9C27B0',
        'E: Divergence': '#00BCD4',
        'F: VolExpand': '#E91E63',
        'G: Concentration': '#795548',
        'H: MeanRev': '#3F51B5',
        'I: Momentum': '#F44336',
    }
    patches = [mpatches.Patch(color=v, label=k, alpha=0.85) for k, v in legend_groups.items()]
    ax2.legend(handles=patches, fontsize=7, loc='best')
    ax2.grid(alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # ---- 右下: 有望ルールテーブル ----
    ax3 = fig.add_axes([0.51, 0.05, 0.46, 0.42])
    ax3.axis('off')
    if len(sig) > 0:
        tbl = sig.head(10)[['rule','N','mean_net','t_stat','win_rate','sharpe']].copy()
        tbl.columns = ['ルール','N','net(bps)','t値','勝率%','Sharpe']
        tbl['ルール'] = tbl['ルール'].str[:30]
        table = ax3.table(cellText=tbl.values, colLabels=tbl.columns,
                          cellLoc='center', loc='upper center',
                          bbox=[0, 0.05, 1, 0.90])
        table.auto_set_font_size(False)
        table.set_fontsize(8.5)
        for (r, c), cell in table.get_celld().items():
            if r == 0:
                cell.set_facecolor('#1565C0')
                cell.set_text_props(color='white', fontweight='bold')
            elif r % 2 == 0:
                cell.set_facecolor('#E3F2FD')
            cell.set_edgecolor('#BDBDBD')
        ax3.set_title(f'有望ルール: {len(sig)}件 (p<0.05, net>0)',
                      fontsize=10, fontweight='bold', y=0.96)
    else:
        ax3.text(0.5, 0.5, '有望ルール: 0件', ha='center', va='center',
                 fontsize=14, fontweight='bold', color='red')

    fig.text(0.99, 0.005,
             'データ: 2026-01-01〜2026-04-30 / SBG (99840) JQuantsティック | コスト4bps',
             ha='right', va='bottom', fontsize=7, color='gray')
    plt.savefig('result.png', dpi=100, bbox_inches='tight', facecolor='white')
    print(f"\nresult.png 保存完了 ({time.time()-t0:.1f}秒)")


if __name__ == '__main__':
    main()
