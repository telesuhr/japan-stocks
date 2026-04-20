"""
半導体セクター ペア平均回帰LS戦略 バックテスト
- スプレッド = 寄付比変化率A - 寄付比変化率B
- Zスコアが±閾値を超えたらエントリー (高い方ショート、低い方ロング)
- |Z| < エグジット閾値で決済
- コスト: 片側2bps × 2銘柄 × 往復 = 8bps
"""
import psycopg2
import pandas as pd
import numpy as np
from itertools import combinations

PG_CONFIG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}

SYMBOLS = {
    "8035.T": "TEL",
    "6857.T": "Advantest",
    "6920.T": "Lasertec",
    "6146.T": "DISCO",
    "6723.T": "Renesas",
    "4063.T": "ShinEtsu",
}
START = "2025-04-21"
END = "2026-04-21"

COST_BPS_ROUND_TRIP_LS = 8  # 2bps × 2銘柄 × 往復


def load_panel():
    conn = psycopg2.connect(**PG_CONFIG)
    frames = {}
    for sym in SYMBOLS:
        q = f"""SELECT timestamp, open, close
                FROM intraday_data
                WHERE symbol='{sym}' AND timestamp >= '{START}' AND timestamp < '{END}'
                ORDER BY timestamp"""
        df = pd.read_sql(q, conn)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['jst'] = df['timestamp'] + pd.Timedelta(hours=9)
        df = df.dropna(subset=['open','close']).set_index('jst').sort_index()
        h, m = df.index.hour, df.index.minute
        morning = (h == 9) | (h == 10) | ((h == 11) & (m <= 30))
        afternoon = ((h == 12) & (m >= 30)) | (h == 13) | (h == 14) | ((h == 15) & (m <= 30))
        df = df[morning | afternoon]
        frames[sym] = df
    conn.close()
    return frames


def compute_ret_from_open(frames):
    """各日の寄付比変化率 (=当日の累積リターン)"""
    rets = {}
    for sym, df in frames.items():
        # 日ごとに寄付価格で割る
        day_open = df.groupby(df.index.date)['open'].transform('first')
        ret = df['close'] / day_open - 1
        rets[sym] = ret
    panel = pd.concat(rets, axis=1)
    panel.columns = list(SYMBOLS.keys())
    panel = panel.dropna()
    return panel


def backtest_pair(panel, a, b, z_entry=2.0, z_exit=0.5, window=20, stop_z=4.0, max_hold_min=60):
    """ペアLS平均回帰
    spread = ret_a - ret_b
    Z > +z_entry: A高すぎ → A売り / B買い
    Z < -z_entry: A安すぎ → A買い / B売り
    |Z| < z_exit で決済
    |Z| > stop_z で損切 or max_hold超過で強制決済
    """
    spread = panel[a] - panel[b]
    # 日を跨がない window
    mean = spread.rolling(window).mean()
    std = spread.rolling(window).std()
    z = (spread - mean) / std

    # 日ごとのインデックス
    dates = spread.index.date
    prev_date = np.r_[dates[:1], dates[:-1]]
    same_day = dates == prev_date  # 前バーと同日

    trades = []
    position = 0  # +1: A long/B short, -1: A short/B long
    entry_i = None
    entry_spread = None
    entry_time = None

    z_arr = z.values
    spread_arr = spread.values
    idx = spread.index
    n = len(spread)

    for i in range(window, n):
        # 日替わり: 強制クローズ
        if position != 0 and not same_day[i]:
            exit_spread = spread_arr[i - 1]
            # PnL: position=+1ならspreadが縮小(entry_spread高い→低いへ)で利益ではなく、
            # position +1 = A long (A上がる) + B short (B下がる) → spread増加で利益
            # ただしエントリーはZ<-entryのとき position=+1 (Aを買う=Aが割安)
            # つまり spread が entry_spread より上昇したら勝ち
            pnl_raw = (exit_spread - entry_spread) * position
            trades.append({
                'entry_time': entry_time, 'exit_time': idx[i-1],
                'entry_z': z_arr[entry_i], 'exit_z': z_arr[i-1],
                'position': position, 'pnl_raw': pnl_raw,
                'pnl_bps': pnl_raw * 10000 - COST_BPS_ROUND_TRIP_LS,
                'hold_min': i - 1 - entry_i, 'reason': 'day_end'
            })
            position = 0
            entry_i = None

        zi = z_arr[i]
        if np.isnan(zi):
            continue

        if position == 0:
            if zi >= z_entry:
                # Aが高すぎ → A売り/B買い → position = -1 (spread縮小で利益)
                position = -1
                entry_i = i
                entry_spread = spread_arr[i]
                entry_time = idx[i]
            elif zi <= -z_entry:
                position = +1
                entry_i = i
                entry_spread = spread_arr[i]
                entry_time = idx[i]
        else:
            hold = i - entry_i
            reason = None
            if abs(zi) <= z_exit:
                reason = 'mean_rev'
            elif abs(zi) >= stop_z:
                reason = 'stop'
            elif hold >= max_hold_min:
                reason = 'time'
            elif position == +1 and zi > 0:
                # 反対側到達でも利確
                reason = 'flip'
            elif position == -1 and zi < 0:
                reason = 'flip'

            if reason:
                exit_spread = spread_arr[i]
                pnl_raw = (exit_spread - entry_spread) * position
                trades.append({
                    'entry_time': entry_time, 'exit_time': idx[i],
                    'entry_z': z_arr[entry_i], 'exit_z': zi,
                    'position': position, 'pnl_raw': pnl_raw,
                    'pnl_bps': pnl_raw * 10000 - COST_BPS_ROUND_TRIP_LS,
                    'hold_min': hold, 'reason': reason
                })
                position = 0
                entry_i = None

    return pd.DataFrame(trades) if trades else pd.DataFrame()


def evaluate(tdf, label):
    if len(tdf) == 0:
        return None
    arr = tdf['pnl_bps'].values
    wr = (arr > 0).mean() * 100
    pos = arr[arr > 0].sum()
    neg = abs(arr[arr <= 0].sum())
    pf = pos / neg if neg > 0 else np.inf
    mean = arr.mean()
    total = arr.sum()
    sharpe = mean / arr.std() * np.sqrt(252) if arr.std() > 0 else 0
    mean_hold = tdf['hold_min'].mean()
    return {
        'label': label, 'n': len(tdf), 'wr': wr, 'pf': pf,
        'mean_bps': mean, 'total_bps': total, 'sharpe': sharpe, 'mean_hold': mean_hold
    }


def main():
    print("=" * 90)
    print("半導体セクター ペア平均回帰LS (直近1年, コスト往復8bps)")
    print("=" * 90)
    frames = load_panel()
    panel = compute_ret_from_open(frames)
    print(f"パネル: {len(panel):,} bars, {panel.index.min()} ~ {panel.index.max()}\n")

    # STEP 1: 各ペアのスプレッド統計 & 基本パラメータでバックテスト
    pairs = list(combinations(SYMBOLS.keys(), 2))
    print(f"=== STEP 1: 全{len(pairs)}ペア 基本パラメータ (window=20, entry=2.0, exit=0.5) ===\n")
    print(f"{'Pair':<25} {'N':>5} {'WR':>6} {'PF':>5} {'Mean':>8} {'Total':>9} {'Sharpe':>7} {'Hold':>6}")
    base_results = []
    for a, b in pairs:
        tdf = backtest_pair(panel, a, b, z_entry=2.0, z_exit=0.5, window=20)
        r = evaluate(tdf, f"{a}-{b}")
        if r:
            base_results.append(r)
            print(f"{SYMBOLS[a]:<10}-{SYMBOLS[b]:<12} {r['n']:>5} {r['wr']:>5.1f}% {r['pf']:>5.2f} "
                  f"{r['mean_bps']:>+7.2f} {r['total_bps']:>+8.0f} {r['sharpe']:>+6.2f} {r['mean_hold']:>5.1f}m")

    base_df = pd.DataFrame(base_results).sort_values('total_bps', ascending=False)
    print("\n=== Top 3 ペア (基本param, total_bps順) ===")
    print(base_df.head(3).to_string(index=False))

    # STEP 2: 上位ペアでパラメータ感応度
    top3 = base_df.head(3)['label'].tolist()
    print("\n" + "=" * 90)
    print("=== STEP 2: 上位3ペア パラメータ感応度 ===")
    print("=" * 90)
    param_grid = []
    for label in top3:
        a, b = label.split('-')
        print(f"\n[{SYMBOLS[a]} - {SYMBOLS[b]}]")
        print(f"{'window':>7} {'entry':>6} {'exit':>5} {'N':>5} {'WR':>6} {'PF':>5} {'Mean':>8} {'Total':>9} {'Sharpe':>7}")
        for window in [10, 20, 30, 60]:
            for entry in [1.5, 2.0, 2.5, 3.0]:
                for exit_z in [0.0, 0.3, 0.5]:
                    tdf = backtest_pair(panel, a, b, z_entry=entry, z_exit=exit_z, window=window)
                    r = evaluate(tdf, f"{a}-{b}")
                    if r and r['n'] >= 30:
                        r.update({'pair': f"{SYMBOLS[a]}-{SYMBOLS[b]}", 'window': window, 'entry': entry, 'exit': exit_z})
                        param_grid.append(r)
                        if r['pf'] >= 1.0:
                            print(f"{window:>7} {entry:>6.1f} {exit_z:>5.1f} {r['n']:>5} {r['wr']:>5.1f}% {r['pf']:>5.2f} "
                                  f"{r['mean_bps']:>+7.2f} {r['total_bps']:>+8.0f} {r['sharpe']:>+6.2f}")

    # STEP 3: ベスト総合ランキング
    if param_grid:
        grid_df = pd.DataFrame(param_grid)
        valid = grid_df[(grid_df.n >= 50) & (grid_df.pf >= 1.0)]
        if len(valid) > 0:
            print("\n" + "=" * 90)
            print("=== STEP 3: Top 15 Best Configs (N>=50, PF>=1.0) ===")
            print("=" * 90)
            best = valid.sort_values('sharpe', ascending=False).head(15)
            cols = ['pair', 'window', 'entry', 'exit', 'n', 'wr', 'pf', 'mean_bps', 'total_bps', 'sharpe', 'mean_hold']
            print(best[cols].to_string(index=False))


if __name__ == "__main__":
    main()
