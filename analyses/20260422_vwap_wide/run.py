"""
VWAP戦略 大規模スクリーニング (117銘柄 × 18パターン)
フォルダ: analyses/20260422_vwap_wide/
"""

import warnings
warnings.filterwarnings("ignore")

import psycopg2
import pandas as pd
import numpy as np
import time
import sys
from datetime import datetime

# ── 設定 ──────────────────────────────────────────────────────────────
PG_CONFIG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}

SIGNAL_TIMES = [("09:30", 9, 30), ("10:00", 10, 0), ("11:00", 11, 0)]
THRESHOLDS   = [20, 30, 50]            # bps
STRATEGIES   = ["Trend", "Reversion"]

COST_BPS = 4.0   # 往復コスト (bps)
MIN_N    = 30
MIN_SHARPE = 2.0
MIN_TSTAT  = 2.0

SECTORS = {
    '非鉄金属':     ['5706.T','5711.T','5713.T','5714.T','5801.T','5802.T','5803.T'],
    '鉄鋼':         ['5401.T','5411.T','5631.T'],
    'エネルギー':   ['1605.T','5016.T','5020.T'],
    '化学':         ['4043.T','4063.T','4183.T','4188.T','3407.T'],
    '半導体・電子部品': ['6146.T','6723.T','6857.T','6920.T','6963.T','6976.T','6981.T',
                       '285A.T','3436.T','6525.T','6526.T','6590.T'],
    '機械・重工':   ['6273.T','6301.T','6305.T','6323.T','6367.T','6594.T',
                     '7011.T','7012.T','7013.T'],
    '電機・精密':   ['6098.T','6501.T','6503.T','6506.T','6702.T','6752.T','6758.T',
                     '6762.T','6861.T','6902.T','6954.T','7741.T'],
    '自動車':       ['7201.T','7203.T','7261.T','7267.T','7269.T','7270.T','6103.T'],
    '商社':         ['8001.T','8002.T','8015.T','8031.T','8053.T','8058.T'],
    '銀行・金融':   ['8306.T','8308.T','8316.T','8354.T','8411.T','8604.T',
                     '8750.T','8766.T'],
    '海運':         ['9101.T','9104.T','9107.T'],
    '不動産':       ['8801.T','8802.T','8830.T'],
    '通信・サービス': ['9432.T','9433.T','9434.T','4324.T','4704.T','4751.T'],
    '電力・鉄道':   ['9020.T','9022.T','9023.T','9501.T','9502.T','9503.T'],
    '内需・消費':   ['1925.T','1928.T','2413.T','2502.T','2503.T','2801.T','2802.T',
                     '2914.T','3064.T','3382.T','4452.T','4502.T','4503.T','4523.T',
                     '4568.T','4578.T','4661.T','4901.T','4911.T','6758.T','7974.T',
                     '8113.T','8267.T','9983.T','9984.T'],
}

def get_sector(sym: str) -> str:
    for sec, lst in SECTORS.items():
        if sym in lst:
            return sec
    return 'その他'


# ── DB から銘柄リスト取得 ──────────────────────────────────────────────
def get_symbols(conn) -> list[str]:
    sql = """
        SELECT symbol
        FROM intraday_data
        WHERE symbol LIKE '%%.T'
          AND timestamp >= '2025-04-01'
          AND timestamp <  '2026-04-22'
        GROUP BY symbol
        HAVING COUNT(DISTINCT DATE(timestamp + INTERVAL '9 hours')) >= 150
        ORDER BY symbol
    """
    df = pd.read_sql(sql, conn)
    return df['symbol'].tolist()


# ── 1銘柄のデータをロード ──────────────────────────────────────────────
def load_symbol(conn, sym: str) -> pd.DataFrame:
    sql = f"""
        SELECT timestamp, open, high, low, close, volume
        FROM intraday_data
        WHERE symbol = '{sym}'
          AND timestamp >= '2025-04-01'
          AND timestamp <  '2026-04-22'
        ORDER BY timestamp
    """
    df = pd.read_sql(sql, conn)
    if df.empty:
        return df
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['jst'] = df['timestamp'] + pd.Timedelta(hours=9)
    df = df.dropna(subset=['close']).sort_values('jst').reset_index(drop=True)
    return df


# ── VWAP 計算（日次リセット） ──────────────────────────────────────────
def calc_daily_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """
    各日の9:00から累積VWAPを計算する。
    volume=0 or NaN の場合は close×1 として扱う。
    """
    df = df.copy()
    df['date'] = df['jst'].dt.date

    # volume が 0 または NaN のバーは close を重みとして使う
    df['vol_eff'] = df['volume'].fillna(0.0)
    df.loc[df['vol_eff'] <= 0, 'vol_eff'] = 1.0  # weight = close × 1

    # price × weight
    df['pv'] = df['close'] * df['vol_eff']

    # 日次グループで累積
    df['cum_pv']  = df.groupby('date')['pv'].cumsum()
    df['cum_vol'] = df.groupby('date')['vol_eff'].cumsum()
    df['vwap']    = df['cum_pv'] / df['cum_vol']

    # 乖離 bps
    df['dev'] = (df['close'] - df['vwap']) / df['vwap'] * 10000.0

    return df


# ── 1パターンのバックテスト ───────────────────────────────────────────
def backtest_pattern(df: pd.DataFrame, sig_label: str, sig_h: int, sig_m: int,
                     threshold: float, strategy: str) -> dict | None:
    """
    df: JSTインデックス済み1銘柄データ（calc_daily_vwap適用済み）
    戻り値: {N, mean_bps, WR, Sharpe, t_stat} or None
    """
    trades = []

    for date, day_df in df.groupby('date'):
        day_df = day_df.sort_values('jst').reset_index(drop=True)

        # シグナルバー（指定時刻のバー）を探す
        sig_mask = (day_df['jst'].dt.hour == sig_h) & (day_df['jst'].dt.minute == sig_m)
        sig_rows = day_df[sig_mask]
        if sig_rows.empty:
            continue
        sig_idx = sig_rows.index[0]
        sig_dev = sig_rows['dev'].iloc[0]

        # 方向判定
        if strategy == "Trend":
            if sig_dev >= threshold:
                direction = 1   # Long
            elif sig_dev <= -threshold:
                direction = -1  # Short
            else:
                continue
        else:  # Reversion
            if sig_dev >= threshold:
                direction = -1  # Short
            elif sig_dev <= -threshold:
                direction = 1   # Long
            else:
                continue

        # エントリー: シグナルの次バーの open
        if sig_idx + 1 >= len(day_df):
            continue
        entry_price = day_df['open'].iloc[sig_idx + 1]
        if pd.isna(entry_price) or entry_price <= 0:
            continue

        # エグジット: 15:20〜15:30 の最後の close
        exit_mask = (day_df['jst'].dt.hour == 15) & (day_df['jst'].dt.minute >= 20)
        exit_rows = day_df[exit_mask]
        if exit_rows.empty:
            continue
        exit_price = exit_rows['close'].iloc[-1]
        if pd.isna(exit_price) or exit_price <= 0:
            continue

        # P&L (bps)
        raw_bps = direction * (exit_price - entry_price) / entry_price * 10000.0
        net_bps = raw_bps - COST_BPS

        trades.append(net_bps)

    if len(trades) < MIN_N:
        return None

    arr = np.array(trades)
    n    = len(arr)
    mean = arr.mean()
    std  = arr.std(ddof=1)
    if std == 0:
        return None

    sharpe = mean / std * np.sqrt(252)
    t_stat = mean / (std / np.sqrt(n))
    wr     = (arr > 0).mean() * 100.0

    return {"N": n, "mean_bps": round(mean, 3), "WR": round(wr, 2),
            "Sharpe": round(sharpe, 3), "t_stat": round(t_stat, 3)}


# ── メイン ───────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    print(f"[{datetime.now():%H:%M:%S}] DB接続中...")
    conn = psycopg2.connect(**PG_CONFIG)

    symbols = get_symbols(conn)
    print(f"[{datetime.now():%H:%M:%S}] 対象銘柄数: {len(symbols)}")

    all_results = []
    total = len(symbols)

    for i, sym in enumerate(symbols, 1):
        t1 = time.time()
        df = load_symbol(conn, sym)
        if df.empty or len(df) < 200:
            print(f"  [{i:3d}/{total}] {sym}: データ不足 → スキップ")
            continue

        df = calc_daily_vwap(df)
        sym_results = []

        for sig_label, sig_h, sig_m in SIGNAL_TIMES:
            for thr in THRESHOLDS:
                for strat in STRATEGIES:
                    res = backtest_pattern(df, sig_label, sig_h, sig_m, thr, strat)
                    if res is not None:
                        row = {
                            "symbol":       sym,
                            "sector":       get_sector(sym),
                            "signal_time":  sig_label,
                            "threshold":    thr,
                            "strategy":     strat,
                            **res,
                        }
                        sym_results.append(row)

        elapsed = time.time() - t1
        qualified = sum(1 for r in sym_results
                        if r['Sharpe'] >= MIN_SHARPE and r['t_stat'] >= MIN_TSTAT)
        print(f"  [{i:3d}/{total}] {sym:10s}  patterns={len(sym_results):2d}  "
              f"qualified={qualified:2d}  ({elapsed:.1f}s)")

        all_results.extend(sym_results)

    conn.close()

    if not all_results:
        print("合格戦略なし")
        return

    df_all = pd.DataFrame(all_results)

    # 合格フィルタ
    df_q = df_all[(df_all['Sharpe']  >= MIN_SHARPE) &
                  (df_all['t_stat']  >= MIN_TSTAT)  &
                  (df_all['N']       >= MIN_N)].copy()
    df_q = df_q.sort_values('Sharpe', ascending=False).reset_index(drop=True)

    # CSV保存
    out_dir = "/Users/Yusuke/claude-code/japan-stocks/analyses/20260422_vwap_wide"
    df_all.to_csv(f"{out_dir}/results_all.csv", index=False)
    df_q.to_csv(f"{out_dir}/results_qualified.csv", index=False)
    print(f"\n全パターン数: {len(df_all):,}  合格数: {len(df_q):,}")

    # ── TOP20 ────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("合格戦略 TOP20 (Sharpe降順)")
    print("="*80)
    cols = ['symbol','sector','signal_time','threshold','strategy','N','mean_bps','WR','Sharpe','t_stat']
    print(df_q[cols].head(20).to_string(index=True))

    # ── 銘柄タイプ分類 ────────────────────────────────────────────────
    print("\n" + "="*80)
    print("銘柄タイプ分類 (合格戦略ベース)")
    print("="*80)
    sym_trend     = set(df_q[df_q['strategy'] == 'Trend']['symbol'])
    sym_reversion = set(df_q[df_q['strategy'] == 'Reversion']['symbol'])
    sym_both      = sym_trend & sym_reversion
    sym_trend_only = sym_trend - sym_both
    sym_rev_only   = sym_reversion - sym_both
    all_q_syms = sym_trend | sym_reversion

    print(f"Trend型のみ     : {len(sym_trend_only):3d}銘柄  {sorted(sym_trend_only)}")
    print(f"Reversion型のみ : {len(sym_rev_only):3d}銘柄  {sorted(sym_rev_only)}")
    print(f"両方            : {len(sym_both):3d}銘柄  {sorted(sym_both)}")
    print(f"合格なし        : {total - len(all_q_syms):3d}銘柄")

    # ── セクター別サマリー ────────────────────────────────────────────
    print("\n" + "="*80)
    print("セクター別サマリー")
    print("="*80)

    # 全銘柄のセクター分布
    all_syms_sector = {}
    for sym in symbols:
        all_syms_sector[sym] = get_sector(sym)
    sector_total = pd.Series(all_syms_sector).value_counts()

    sec_summary = (df_q.groupby('sector')
                   .agg(合格戦略数=('Sharpe','count'),
                        合格銘柄数=('symbol','nunique'),
                        平均Sharpe=('Sharpe','mean'),
                        最大Sharpe=('Sharpe','max'),
                        平均mean_bps=('mean_bps','mean'))
                   .round(3))
    sec_summary['全銘柄数'] = sec_summary.index.map(lambda s: sector_total.get(s, 0))
    sec_summary = sec_summary.sort_values('平均Sharpe', ascending=False)
    print(sec_summary.to_string())

    elapsed_total = time.time() - t0
    print(f"\n総実行時間: {elapsed_total/60:.1f}分")
    print(f"results_qualified.csv → {out_dir}/results_qualified.csv")
    print(f"results_all.csv       → {out_dir}/results_all.csv")


if __name__ == "__main__":
    main()
