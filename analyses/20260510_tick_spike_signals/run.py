"""
ティックスパイクシグナル検証
1分間のティック数(約定回数)が急増した時点でエントリーすると勝てるか?

仮説:
  H1) ティック数スパイク × 価格上昇 → モメンタム継続 (買い)
  H2) ティック数スパイク × 価格下落 → モメンタム継続 (売り)
  H3) ティック数スパイク × 大きな動き → 逆方向の反転

検証方法:
  - 1分バー集計時に tick_count を追加カラムとして保持
  - tick_ratio = 当該バーのtick_count / 直近30分のtick_count平均
  - スパイク = tick_ratio >= 3.0
  - フォワードリターン: 1min, 5min, 15min, 30min, EOD

データ:
  - 5銘柄 (7203, 9984, 8306, 6857, 6758)
  - 2026-01-01 ~ 2026-04-30 (4か月)
  - 取引時間内バーのみ
"""

import sys
sys.path.insert(0, '/Users/Yusuke/claude-code/DataFetcher')
from src.ticks import TickQuery, _file_globs, _to_date, _norm5, _norm4

import duckdb
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from pathlib import Path
from datetime import date
import warnings
warnings.filterwarnings('ignore')

COST_BPS = 4
DATA_CACHE = Path('bars_with_tickcount.parquet')

# 流動性の高い主要銘柄
SYMBOLS = {
    '72030': 'トヨタ',
    '99840': 'ソフトバンクG',
    '83060': 'MUFG',
    '68570': 'アドバンテスト',
    '67580': 'ソニーG',
}
START = '2026-01-01'
END   = '2026-04-30'


def fetch_bars_with_features(code: str, start: str, end: str) -> pd.DataFrame:
    """
    DuckDB で直接 read_csv してリッチな1分バーを構築する。
    通常の OHLCV に加え tick_count / max_single_volume / large_tick_count を持つ。
    """
    s = _to_date(start)
    e = _to_date(end)
    root = Path.home() / "Data" / "jquants_trades" / "equities" / "trades"
    files = _file_globs(root, s, e)
    if not files:
        return pd.DataFrame()

    files_arr = "[" + ", ".join(f"'{f}'" for f in files) + "]"
    csv_part = (
        f"read_csv({files_arr}, "
        "header=true, "
        "columns={'Date':'DATE','Code':'VARCHAR','Time':'VARCHAR',"
        "'SessionDistinction':'VARCHAR','Price':'DOUBLE','TradingVolume':'BIGINT',"
        "'TransactionId':'VARCHAR'})"
    )

    sql = f"""
    WITH t AS (
        SELECT Date, Time, Price, TradingVolume
        FROM {csv_part}
        WHERE Date BETWEEN ? AND ? AND Code IN (?, ?)
    ), tt AS (
        SELECT
            date_add(
                CAST(Date AS TIMESTAMP),
                INTERVAL (
                    FLOOR(
                        (EXTRACT(HOUR FROM CAST(Time AS TIME)) * 3600
                         + EXTRACT(MINUTE FROM CAST(Time AS TIME)) * 60
                         + EXTRACT(SECOND FROM CAST(Time AS TIME))) / 60
                    ) * 60
                ) SECOND
            ) AS bar_ts,
            Price, TradingVolume
        FROM t
    )
    SELECT
        bar_ts AS ts,
        FIRST(Price ORDER BY bar_ts) AS open,
        MAX(Price) AS high,
        MIN(Price) AS low,
        LAST(Price ORDER BY bar_ts) AS close,
        SUM(TradingVolume) AS volume,
        SUM(Price * TradingVolume) AS turnover_value,
        COUNT(*) AS tick_count,
        MAX(TradingVolume) AS max_single_vol
    FROM tt
    GROUP BY bar_ts
    ORDER BY bar_ts
    """
    con = duckdb.connect()
    con.execute("PRAGMA threads=4")
    df = con.execute(sql, [s, e, _norm5(code), _norm4(code)]).df()
    con.close()
    return df


def filter_session_bars(df):
    """取引時間内バーのみ (9:00-11:30, 12:30-15:00)"""
    h = df['ts'].dt.hour
    m = df['ts'].dt.minute
    sess = (
        ((h == 9) | (h == 10) | ((h == 11) & (m <= 29))) |
        (((h == 12) & (m >= 30)) | (h == 13) | ((h == 14)) | ((h == 15) & (m == 0)))
    )
    return df[sess].copy()


def add_features(df):
    """1銘柄1日 単位で特徴量を追加 (look-ahead なし)"""
    df = df.sort_values('ts').reset_index(drop=True)
    df['date'] = df['ts'].dt.date

    out = []
    for d, g in df.groupby('date'):
        g = g.sort_values('ts').reset_index(drop=True)
        if len(g) < 30:
            continue

        # 直近30バー(30分)の tick_count 平均
        g['tick_ma30'] = g['tick_count'].shift(1).rolling(30, min_periods=10).mean()
        g['tick_ratio'] = g['tick_count'] / g['tick_ma30']

        # 出来高の30分平均比
        g['vol_ma30'] = g['volume'].shift(1).rolling(30, min_periods=10).mean()
        g['vol_ratio'] = g['volume'] / g['vol_ma30']

        # 当該バー内の方向 (close vs open)
        g['bar_dir'] = np.sign(g['close'] - g['open'])
        g['bar_ret'] = (g['close'] / g['open'] - 1) * 10000  # bps

        # フォワードリターン (close から N分後close まで, bps)
        for k, label in [(1, 'fwd_1min'), (5, 'fwd_5min'),
                         (15, 'fwd_15min'), (30, 'fwd_30min')]:
            g[label] = (g['close'].shift(-k) / g['close'] - 1) * 10000

        # EODまで
        eod = g['close'].iloc[-1]
        g['fwd_eod'] = (eod / g['close'] - 1) * 10000

        out.append(g)

    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def test_rule(df, cond, target, label, direction=1):
    sub = df[cond & df[target].notna()]
    if len(sub) < 30:
        return None
    arr = sub[target].values * direction
    net = arr - COST_BPS
    t, p = stats.ttest_1samp(arr, 0)
    wr = (arr > 0).mean() * 100
    sharpe = net.mean() / arr.std() * np.sqrt(252 * 60) if arr.std() > 0 else 0
    return dict(rule=label, target=target, dir=direction, N=len(sub),
                mean_raw=round(arr.mean(), 2), mean_net=round(net.mean(), 2),
                std=round(arr.std(), 2), t_stat=round(t, 2), p_val=round(p, 4),
                win_rate=round(wr, 1), sharpe=round(sharpe, 2),
                sig=(p < 0.05 and net.mean() > 0))


# ============================================================
def main():
    import time
    t0 = time.time()

    # ---- データ取得 ----
    if DATA_CACHE.exists():
        print(f"キャッシュからロード: {DATA_CACHE}")
        all_df = pd.read_parquet(DATA_CACHE)
    else:
        print(f"ティックデータから1分バー生成中 (5銘柄 × 4か月)...")
        frames = []
        for code, name in SYMBOLS.items():
            print(f"  {code} {name}...")
            df = fetch_bars_with_features(code, START, END)
            df['code'] = code
            df = filter_session_bars(df)
            frames.append(df)
        all_df = pd.concat(frames, ignore_index=True)
        all_df.to_parquet(DATA_CACHE)
        print(f"  キャッシュ保存: {DATA_CACHE} ({len(all_df):,}バー)")

    print(f"  全バー: {len(all_df):,}, 銘柄数: {all_df['code'].nunique()}")
    print(f"  期間: {all_df['ts'].min()} 〜 {all_df['ts'].max()}")

    # ---- 特徴量追加 ----
    print("特徴量計算中...")
    feats_list = []
    for code, g in all_df.groupby('code'):
        f = add_features(g)
        f['code'] = code
        feats_list.append(f)
    feats = pd.concat(feats_list, ignore_index=True)
    feats = feats.dropna(subset=['tick_ratio'])
    print(f"  有効レコード: {len(feats):,}")

    # ---- 各種ルール検証 ----
    res = []
    def add(r):
        if r: res.append(r)

    # ===== A1. tick_ratio スパイクで方向性 =====
    # ティック数スパイク × 上昇 → モメンタム
    for k, target in [(1,'fwd_1min'),(5,'fwd_5min'),(15,'fwd_15min'),(30,'fwd_30min')]:
        cond = (feats['tick_ratio'] >= 3.0) & (feats['bar_ret'] > 0)
        add(test_rule(feats, cond, target, f'A1_TickSpike3x_Up_Buy_{k}min'))
        cond = (feats['tick_ratio'] >= 3.0) & (feats['bar_ret'] < 0)
        add(test_rule(feats, cond, target, f'A2_TickSpike3x_Down_Sell_{k}min', direction=-1))

    # ===== A3. tick_ratio 強スパイク (5x) =====
    for k, target in [(5,'fwd_5min'),(15,'fwd_15min')]:
        cond = (feats['tick_ratio'] >= 5.0) & (feats['bar_ret'] > 0)
        add(test_rule(feats, cond, target, f'A3_TickSpike5x_Up_Buy_{k}min'))
        cond = (feats['tick_ratio'] >= 5.0) & (feats['bar_ret'] < 0)
        add(test_rule(feats, cond, target, f'A4_TickSpike5x_Down_Sell_{k}min', direction=-1))

    # ===== A5. リバーサル仮説 =====
    # ティック数スパイク × 上昇 → 反転売り (5min)
    for k in [5, 15, 30]:
        cond = (feats['tick_ratio'] >= 3.0) & (feats['bar_ret'] > 0)
        add(test_rule(feats, cond, f'fwd_{k}min', f'A5_TickSpike_Up_Reversal_{k}min', direction=-1))
        cond = (feats['tick_ratio'] >= 3.0) & (feats['bar_ret'] < 0)
        add(test_rule(feats, cond, f'fwd_{k}min', f'A6_TickSpike_Down_Reversal_{k}min'))

    # ===== A7. ティック数 × 出来高比 同時条件 =====
    for k in [5, 15]:
        cond = (feats['tick_ratio'] >= 3.0) & (feats['vol_ratio'] >= 3.0) & (feats['bar_ret'] > 0)
        add(test_rule(feats, cond, f'fwd_{k}min', f'A7_TickAndVol_Up_Buy_{k}min'))
        cond = (feats['tick_ratio'] >= 3.0) & (feats['vol_ratio'] >= 3.0) & (feats['bar_ret'] < 0)
        add(test_rule(feats, cond, f'fwd_{k}min', f'A8_TickAndVol_Down_Sell_{k}min', direction=-1))

    # ===== A9. ティック数増 vs 出来高増 のダイバージェンス =====
    # ティック多いが出来高少ない (小口殺到) × 上昇
    cond = (feats['tick_ratio'] >= 3.0) & (feats['vol_ratio'] <= 1.5) & (feats['bar_ret'] > 0)
    add(test_rule(feats, cond, 'fwd_15min', 'A9_TickHigh_VolLow_Up_Reversal', direction=-1))
    cond = (feats['tick_ratio'] >= 3.0) & (feats['vol_ratio'] <= 1.5) & (feats['bar_ret'] < 0)
    add(test_rule(feats, cond, 'fwd_15min', 'A10_TickHigh_VolLow_Down_Reversal'))

    # 出来高多いがティック少ない (大口主導) × 上昇 → モメンタム
    cond = (feats['vol_ratio'] >= 3.0) & (feats['tick_ratio'] <= 2.0) & (feats['bar_ret'] > 0)
    add(test_rule(feats, cond, 'fwd_15min', 'A11_VolHigh_TickLow_Up_Buy'))
    cond = (feats['vol_ratio'] >= 3.0) & (feats['tick_ratio'] <= 2.0) & (feats['bar_ret'] < 0)
    add(test_rule(feats, cond, 'fwd_15min', 'A12_VolHigh_TickLow_Down_Sell', direction=-1))

    results = pd.DataFrame(res).sort_values('t_stat', ascending=False)
    sig = results[results['sig']]

    print("\n=== 全ルール (t値降順) ===")
    cols = ['rule', 'N', 'mean_raw', 'mean_net', 't_stat', 'p_val', 'win_rate', 'sharpe', 'sig']
    print(results[cols].to_string(index=False))

    print(f"\n=== 有望ルール (p<0.05 かつ net>0): {len(sig)}件 ===")
    if len(sig) > 0:
        print(sig[cols].to_string(index=False))

    results.to_csv('results_all.csv', index=False)
    sig.to_csv('results_significant.csv', index=False)

    # ---- 図 ----
    fig = plt.figure(figsize=(14, 8.5), facecolor='white')
    plt.rcParams.update({
        'font.family': ['Hiragino Sans', 'IPAexGothic', 'sans-serif'],
        'axes.unicode_minus': False,
    })
    fig.suptitle('A. ティックスパイクシグナル検証\n5銘柄 × 4ヶ月 (2026/1〜4) ティック由来1分バー',
                 fontsize=13, fontweight='bold', y=0.99)

    # Cat colors
    def col(rule):
        if 'TickAndVol' in rule: return '#FF9800'
        if 'TickHigh' in rule or 'VolHigh' in rule: return '#9C27B0'
        if 'TickSpike5x' in rule: return '#E53935'
        if 'TickSpike3x' in rule or 'TickSpike_' in rule: return '#1E88E5'
        return 'gray'

    # Top: t統計量バー
    ax1 = fig.add_axes([0.04, 0.10, 0.45, 0.82])
    sorted_r = results.sort_values('t_stat')
    ys = range(len(sorted_r))
    colors_b = [col(r) for r in sorted_r['rule']]
    ax1.barh(list(ys), sorted_r['t_stat'], color=colors_b, alpha=0.8, height=0.7)
    ax1.set_yticks(list(ys))
    ax1.set_yticklabels(sorted_r['rule'], fontsize=7)
    ax1.axvline(0, color='black', lw=0.8)
    ax1.axvline(1.96, color='gray', lw=0.8, linestyle='--', alpha=0.6)
    ax1.axvline(-1.96, color='gray', lw=0.8, linestyle='--', alpha=0.6)
    ax1.set_xlabel('t統計量', fontsize=9)
    ax1.set_title('全ルール t値', fontsize=10, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Right: scatter
    ax2 = fig.add_axes([0.55, 0.55, 0.42, 0.36])
    for _, row in results.iterrows():
        c = col(row['rule'])
        m = '*' if row['sig'] else 'o'
        s = 110 if row['sig'] else 35
        ax2.scatter(row['t_stat'], row['mean_net'], c=c, marker=m, s=s,
                    alpha=0.85 if row['sig'] else 0.4, zorder=5 if row['sig'] else 3,
                    edgecolor='black', lw=0.4)
    ax2.axhline(0, color='black', lw=0.8)
    ax2.axvline(0, color='black', lw=0.8)
    ax2.axvline(1.96, color='gray', lw=0.8, linestyle='--', alpha=0.6)
    ax2.set_xlabel('t統計量', fontsize=9)
    ax2.set_ylabel('コスト後 mean_net (bps)', fontsize=9)
    ax2.set_title('t値 vs net_bps (★=有意)', fontsize=10, fontweight='bold')
    ax2.grid(alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Bottom right: 結果テーブル
    ax3 = fig.add_axes([0.55, 0.10, 0.42, 0.36])
    ax3.axis('off')
    if len(sig) > 0:
        tbl = sig.head(7)[['rule', 'N', 'mean_net', 't_stat', 'win_rate', 'sharpe']].copy()
        tbl.columns = ['ルール', 'N', 'net(bps)', 't値', '勝率%', 'Sharpe']
        tbl['ルール'] = tbl['ルール'].str[:24]
        table = ax3.table(cellText=tbl.values, colLabels=tbl.columns,
                          cellLoc='center', loc='upper center',
                          bbox=[0, 0.2, 1, 0.78])
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        for (r, c), cell in table.get_celld().items():
            if r == 0:
                cell.set_facecolor('#1565C0')
                cell.set_text_props(color='white', fontweight='bold')
            elif r % 2 == 0:
                cell.set_facecolor('#E3F2FD')
            cell.set_edgecolor('#BDBDBD')
        ax3.set_title(f'有望ルール (p<0.05, net>0): {len(sig)}件', fontsize=10, fontweight='bold', y=1.02)
    else:
        ax3.text(0.5, 0.5, '有望ルール: 0件', ha='center', va='center',
                 fontsize=14, fontweight='bold', color='red')
        ax3.set_title('有望ルール', fontsize=10, fontweight='bold')

    fig.text(0.99, 0.005,
             'データ: 2026/1〜2026/4 / 主要5銘柄 / JQuantsティック由来1分バー | コスト4bps往復',
             ha='right', va='bottom', fontsize=7, color='gray')
    plt.savefig('result.png', dpi=100, bbox_inches='tight', facecolor='white')
    print(f"\nresult.png 保存完了 ({time.time()-t0:.1f}秒)")


if __name__ == '__main__':
    main()
