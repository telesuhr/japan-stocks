"""
B. 大口約定の方向性追随

仮説:
  「単発で大量(○○株以上)の約定が出たとき、その後の価格は同方向に動くか?」
  - 機関投資家の発注を捕捉して追随できるか
  - 大口プリント直後の最良値が買い・売りどちらか?

設計:
  - 個別ティックで TradingVolume が銘柄99%ile以上のものを「大口」と定義
  - 大口プリント直前数秒の価格と直後の価格を比較し、約定方向を推定
    (Lee-Ready 法に近い: 直前midpriceより上 → 買い主導、下 → 売り主導)
  - フォワードリターン: 大口プリント時点の価格 → 1min, 5min, 15min, 30min 後

データ:
  - 5銘柄 × 4か月 (A と同条件で比較可能に)
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
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

COST_BPS = 4
SYMBOLS = {
    '72030': 'トヨタ',
    '99840': 'ソフトバンクG',
    '83060': 'MUFG',
    '68570': 'アドバンテスト',
    '67580': 'ソニーG',
}
START = '2026-01-01'
END   = '2026-04-30'


def fetch_large_prints(code: str, start: str, end: str, pct=99.0) -> pd.DataFrame:
    """
    大口ティックを抽出 + 直前/直後の価格情報を付与
    pct: 出来高分布のN%以上 = 大口
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
        SELECT Date, Time, Price, TradingVolume,
               date_add(CAST(Date AS TIMESTAMP),
                        INTERVAL (
                            EXTRACT(HOUR FROM CAST(Time AS TIME)) * 3600
                          + EXTRACT(MINUTE FROM CAST(Time AS TIME)) * 60
                          + EXTRACT(SECOND FROM CAST(Time AS TIME))
                        ) SECOND) AS ts
        FROM {csv_part}
        WHERE Date BETWEEN ? AND ? AND Code IN (?, ?)
    ), thresh AS (
        SELECT QUANTILE_CONT(TradingVolume, {pct/100.0}) AS lim FROM t
    ), with_neighbors AS (
        SELECT
            t.ts, t.Price AS price, t.TradingVolume AS vol,
            -- 直前ティックの価格 (LAG)
            LAG(Price, 1) OVER (ORDER BY ts) AS prev_price,
            LAG(Price, 5) OVER (ORDER BY ts) AS prev5_price,
            -- 後続ティックの累積価格を1分・5分・15分・30分後で
            -- 単純化: 時間窓ごとに後続の代表価格 (median price)を使う
            t.Date AS d
        FROM t
    )
    SELECT * FROM with_neighbors
    WHERE vol >= (SELECT lim FROM thresh)
    ORDER BY ts
    """
    con = duckdb.connect()
    con.execute("PRAGMA threads=4")
    df = con.execute(sql, [s, e, _norm5(code), _norm4(code)]).df()
    con.close()
    return df


def fetch_full_ticks_with_forward(code: str, start: str, end: str) -> pd.DataFrame:
    """
    全ティックを取得し、各ティックに対して前後の価格情報を付与する。
    その後、Pythonで「大口」判定 + フォワードリターンを計算する。
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
        SELECT Date AS d, Time AS tm, Price AS price, TradingVolume AS vol,
               date_add(CAST(Date AS TIMESTAMP),
                        INTERVAL (
                            EXTRACT(HOUR FROM CAST(Time AS TIME)) * 3600
                          + EXTRACT(MINUTE FROM CAST(Time AS TIME)) * 60
                          + EXTRACT(SECOND FROM CAST(Time AS TIME))
                        ) SECOND) AS ts
        FROM {csv_part}
        WHERE Date BETWEEN ? AND ? AND Code IN (?, ?)
        ORDER BY ts
    """
    con = duckdb.connect()
    con.execute("PRAGMA threads=4")
    df = con.execute(sql, [s, e, _norm5(code), _norm4(code)]).df()
    con.close()
    return df


def compute_features(ticks: pd.DataFrame, vol_thresh: float) -> pd.DataFrame:
    """
    大口ティックに対してフォワードリターンを計算 (per-day)
    Lee-Ready 風の方向推定:
      tick uptick (price > prev) → 買い主導 (+1)
      tick downtick (price < prev) → 売り主導 (-1)
      zero-tick → 直前のtick方向 (簡略化)
    """
    if len(ticks) == 0:
        return pd.DataFrame()

    ticks = ticks.sort_values('ts').reset_index(drop=True)
    ticks['date_only'] = pd.to_datetime(ticks['d']).dt.date

    out = []
    for d, g in ticks.groupby('date_only'):
        g = g.sort_values('ts').reset_index(drop=True)
        n = len(g)
        if n < 100:
            continue
        prices = g['price'].values.astype(float)
        ts_arr = g['ts'].values  # numpy datetime64
        vols = g['vol'].values

        # 前ティック比方向
        direction = np.zeros(n)
        for i in range(1, n):
            diff = prices[i] - prices[i-1]
            if diff > 0:
                direction[i] = 1
            elif diff < 0:
                direction[i] = -1
            else:
                direction[i] = direction[i-1]

        # 取引時間内のみ
        ts_pd = pd.to_datetime(ts_arr)
        h = ts_pd.hour.values
        m = ts_pd.minute.values
        in_session = (
            ((h == 9) | (h == 10) | ((h == 11) & (m <= 25))) |
            (((h == 12) & (m >= 30)) | (h == 13) | (h == 14))
        )

        # 大口判定
        is_large = (vols >= vol_thresh) & in_session
        large_idx = np.where(is_large)[0]
        if len(large_idx) == 0:
            continue

        # ts_int: 同日内なので秒単位 (epoch秒)で十分
        # DuckDB returns datetime64[us] なので明示的に秒へ変換
        ts_int = ts_pd.astype('datetime64[s]').astype('int64').values  # 秒

        for idx in large_idx:
            t0 = ts_int[idx]
            p0 = prices[idx]
            dir0 = direction[idx]

            row = {
                'ts': ts_arr[idx],
                'date': d,
                'price': p0,
                'vol': vols[idx],
                'direction': int(dir0),
            }
            valid = True
            for k_min, label in [(1, 'fwd_1min'), (5, 'fwd_5min'),
                                 (15, 'fwd_15min'), (30, 'fwd_30min')]:
                t_target = t0 + k_min * 60
                j = np.searchsorted(ts_int, t_target)
                if j >= n:
                    row[label] = np.nan
                else:
                    # 同日であることを ts_int の差で確認 (1日=86400秒)
                    if (ts_int[j] - ts_int[idx]) > 86400:
                        row[label] = np.nan
                    else:
                        row[label] = (prices[j] / p0 - 1) * 10000  # bps
            out.append(row)

    return pd.DataFrame(out)


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


def main():
    import time
    t0 = time.time()

    cache = Path('large_prints_cache.parquet')
    if cache.exists():
        print(f"キャッシュからロード: {cache}")
        all_lp = pd.read_parquet(cache)
    else:
        all_dfs = []
        for code, name in SYMBOLS.items():
            print(f"  {code} {name} ティック取得中...")
            tk0 = time.time()
            ticks = fetch_full_ticks_with_forward(code, START, END)
            if len(ticks) == 0:
                continue
            # 各銘柄ごとに99%ile を計算
            vol_99 = np.percentile(ticks['vol'].values, 99)
            print(f"    ticks: {len(ticks):,}, 99%ile: {int(vol_99)}株 ({time.time()-tk0:.1f}s)")
            lp = compute_features(ticks, vol_99)
            lp['code'] = code
            lp['vol_99'] = vol_99
            all_dfs.append(lp)
            print(f"    大口プリント: {len(lp):,}")

        all_lp = pd.concat(all_dfs, ignore_index=True)
        all_lp.to_parquet(cache)
        print(f"\n  キャッシュ保存: {cache}")

    print(f"\n総大口プリント: {len(all_lp):,}, 銘柄: {all_lp['code'].nunique()}")
    print(f"  方向 +1 (uptick): {(all_lp['direction'] == 1).sum():,}")
    print(f"  方向 -1 (downtick): {(all_lp['direction'] == -1).sum():,}")
    print(f"  方向 0: {(all_lp['direction'] == 0).sum():,}")

    # ---- ルール検証 ----
    res = []
    def add(r):
        if r: res.append(r)

    # B1: 大口買い (+1) → 同方向継続
    for k in [1, 5, 15, 30]:
        cond = all_lp['direction'] == 1
        add(test_rule(all_lp, cond, f'fwd_{k}min', f'B1_BigBuy_Follow_Buy_{k}min'))
        cond = all_lp['direction'] == -1
        add(test_rule(all_lp, cond, f'fwd_{k}min', f'B2_BigSell_Follow_Sell_{k}min', direction=-1))

    # B3: 大口買い → 反転売り (リバーサル)
    for k in [1, 5, 15, 30]:
        cond = all_lp['direction'] == 1
        add(test_rule(all_lp, cond, f'fwd_{k}min', f'B3_BigBuy_Reversal_Sell_{k}min', direction=-1))
        cond = all_lp['direction'] == -1
        add(test_rule(all_lp, cond, f'fwd_{k}min', f'B4_BigSell_Reversal_Buy_{k}min'))

    # B5: 超大口 (99.9%ile相当 = vol が銘柄ごとに約2倍以上) → 継続
    # 各銘柄99%ile が vol_99 に入ってるので、その2倍を「超大口」とする
    super_thresh = all_lp.groupby('code')['vol'].transform(lambda x: x.quantile(0.95))
    is_super = all_lp['vol'] >= super_thresh
    for k in [5, 15, 30]:
        cond = is_super & (all_lp['direction'] == 1)
        add(test_rule(all_lp, cond, f'fwd_{k}min', f'B5_SuperBigBuy_Follow_{k}min'))
        cond = is_super & (all_lp['direction'] == -1)
        add(test_rule(all_lp, cond, f'fwd_{k}min', f'B6_SuperBigSell_Follow_{k}min', direction=-1))

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
    fig.suptitle('B. 大口約定の方向性追随 (Lee-Ready風方向推定)\n5銘柄×4か月、各銘柄99%ile以上の出来高ティック',
                 fontsize=13, fontweight='bold', y=0.99)

    # color
    def col(rule):
        if rule.startswith('B1') or rule.startswith('B2'): return '#2196F3'
        if rule.startswith('B3') or rule.startswith('B4'): return '#E53935'
        if rule.startswith('B5') or rule.startswith('B6'): return '#FF9800'
        return 'gray'

    # Left: t bar
    ax1 = fig.add_axes([0.05, 0.10, 0.42, 0.82])
    sorted_r = results.sort_values('t_stat')
    ys = range(len(sorted_r))
    ax1.barh(list(ys), sorted_r['t_stat'],
             color=[col(r) for r in sorted_r['rule']], alpha=0.8, height=0.7)
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
    ax2 = fig.add_axes([0.54, 0.55, 0.42, 0.36])
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

    # Bottom right: 有望表
    ax3 = fig.add_axes([0.54, 0.10, 0.42, 0.36])
    ax3.axis('off')
    if len(sig) > 0:
        tbl = sig.head(10)[['rule', 'N', 'mean_net', 't_stat', 'win_rate', 'sharpe']].copy()
        tbl.columns = ['ルール', 'N', 'net(bps)', 't値', '勝率%', 'Sharpe']
        tbl['ルール'] = tbl['ルール'].str[:24]
        table = ax3.table(cellText=tbl.values, colLabels=tbl.columns,
                          cellLoc='center', loc='upper center',
                          bbox=[0, 0.1, 1, 0.85])
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        for (r, c), cell in table.get_celld().items():
            if r == 0:
                cell.set_facecolor('#1565C0')
                cell.set_text_props(color='white', fontweight='bold')
            elif r % 2 == 0:
                cell.set_facecolor('#E3F2FD')
            cell.set_edgecolor('#BDBDBD')
        ax3.set_title(f'有望ルール: {len(sig)}件', fontsize=10, fontweight='bold', y=1.02)
    else:
        ax3.text(0.5, 0.5, '有望ルール: 0件', ha='center', va='center',
                 fontsize=14, fontweight='bold', color='red')

    fig.text(0.99, 0.005,
             'データ: 2026/1〜2026/4 / 主要5銘柄 / JQuantsティック | 大口=銘柄別99%ile以上 | コスト4bps往復',
             ha='right', va='bottom', fontsize=7, color='gray')
    plt.savefig('result.png', dpi=100, bbox_inches='tight', facecolor='white')
    print(f"\nresult.png 保存完了 ({time.time()-t0:.1f}秒)")


if __name__ == '__main__':
    main()
