#!/usr/bin/env python3
"""
非鉄ダイバージェンスLS バックテスト
5711.T (三菱マテリアル) vs 5713.T (住友金属鉱山)

戦略概要:
  - 両銘柄の寄付比リターンをトラッキング
  - 9:30時点でのスプレッド乖離 ≥ 閾値 → 遅行銘柄をBuy、先行銘柄をSell
  - 同日引成で決済（純粋イントラデイ）

シグナル:
  spread = ret_5711 - ret_5713  (9:30時点の寄付比変化率の差)
  spread ≥ +th → 5711 Short / 5713 Long  (5711が先行、5713が割安)
  spread ≤ -th → 5711 Long  / 5713 Short  (5713が先行、5711が割安)
"""
import sys
import psycopg2
import pandas as pd
import numpy as np
from itertools import product

PG_CONFIG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}

SYMBOL_A = "5711.T"  # 三菱マテリアル
SYMBOL_B = "5713.T"  # 住友金属鉱山
COST_BPS = 4          # 往復コスト (LS×2銘柄 = 4回取引)
SIGNAL_HOUR = 9
SIGNAL_MIN = 30


def load_intraday(symbol: str) -> pd.DataFrame:
    conn = psycopg2.connect(**PG_CONFIG)
    q = f"""
        SELECT timestamp, open, close, volume
        FROM intraday_data
        WHERE symbol = '{symbol}'
          AND close IS NOT NULL
        ORDER BY timestamp
    """
    df = pd.read_sql(q, conn)
    conn.close()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['jst'] = df['timestamp'] + pd.Timedelta(hours=9)
    return df.set_index('jst').sort_index()


def compute_daily_metrics(df: pd.DataFrame, signal_hour: int = 9, signal_min: int = 30):
    """日ごとの寄付き価格・9:30時点価格・引値を返す"""
    records = []
    for dt, g in df.groupby(df.index.date):
        # 前場のみ
        morning = g[(g.index.hour >= 9) & (g.index.hour < 15)]
        if morning.empty:
            continue

        open_bar = morning.iloc[0]
        open_price = float(open_bar['open']) if not pd.isna(open_bar['open']) else float(open_bar['close'])

        # 9:30時点
        at_signal = morning[
            (morning.index.hour == signal_hour) & (morning.index.minute <= signal_min)
        ]
        if at_signal.empty:
            continue
        signal_price = float(at_signal.iloc[-1]['close'])

        # 引値（最後のclose）
        close_price = float(morning.iloc[-1]['close'])

        ret_signal = (signal_price / open_price - 1) * 100  # %
        ret_close  = (close_price  / open_price - 1) * 100

        records.append({
            'date': dt,
            'open': open_price,
            'price_signal': signal_price,
            'close': close_price,
            'ret_signal': ret_signal,
            'ret_close': ret_close,
        })
    return pd.DataFrame(records).set_index('date')


def backtest(df_a: pd.DataFrame, df_b: pd.DataFrame, threshold: float, cost_bps: float = COST_BPS):
    """
    threshold: spread乖離閾値 (%)
    シグナル: spread = ret_a - ret_b at 9:30
      spread ≥ +th → Short A / Long B (Aが過剰先行 → 平均回帰)
      spread ≤ -th → Long A / Short B
    """
    common_dates = df_a.index.intersection(df_b.index)
    trades = []

    for dt in common_dates:
        ra_sig = df_a.loc[dt, 'ret_signal']
        rb_sig = df_b.loc[dt, 'ret_signal']
        spread = ra_sig - rb_sig

        if spread >= threshold:
            # Short A / Long B
            # A: エントリーはsignal_price, 決済はclose → short → gain if A↓
            ret_a = -(df_a.loc[dt, 'ret_close'] - df_a.loc[dt, 'ret_signal'])  # short
            ret_b =  (df_b.loc[dt, 'ret_close'] - df_b.loc[dt, 'ret_signal'])  # long
            side = "Short_A_Long_B"
        elif spread <= -threshold:
            # Long A / Short B
            ret_a =  (df_a.loc[dt, 'ret_close'] - df_a.loc[dt, 'ret_signal'])
            ret_b = -(df_b.loc[dt, 'ret_close'] - df_b.loc[dt, 'ret_signal'])
            side = "Long_A_Short_B"
        else:
            continue

        pnl_bps = (ret_a + ret_b) / 2 * 100 - cost_bps  # 平均リターン→bps, コスト差引
        trades.append({'date': dt, 'spread_entry': spread, 'side': side, 'pnl_bps': pnl_bps})

    return pd.DataFrame(trades)


def evaluate(trades: pd.DataFrame, label: str = ""):
    if trades.empty:
        print(f"{label}: トレードなし")
        return {}
    arr = trades['pnl_bps'].values
    n   = len(arr)
    mean = arr.mean()
    std  = arr.std(ddof=1) if n > 1 else 1
    wr  = (arr > 0).mean() * 100
    sharpe = mean / std * np.sqrt(252) if std > 0 else 0
    t_stat = mean / (std / np.sqrt(n)) if std > 0 else 0
    total  = arr.sum()
    print(f"{label}: N={n:3d}  Mean={mean:+.1f}bps  WR={wr:.1f}%  Sharpe={sharpe:+.2f}  t={t_stat:+.2f}  Total={total:+.0f}bps")
    return {'n': n, 'mean': mean, 'wr': wr, 'sharpe': sharpe, 't': t_stat}


def main():
    print("=" * 70)
    print("非鉄ダイバージェンスLS バックテスト")
    print(f"  {SYMBOL_A} (三菱マテリアル) vs {SYMBOL_B} (住友金属鉱山)")
    print(f"  シグナル: 9:30時点の寄付比リターン差 → 逆張りLS → 引成決済")
    print(f"  コスト: 往復 {COST_BPS}bps")
    print("=" * 70)

    print("\n[データロード中...]")
    df_a_raw = load_intraday(SYMBOL_A)
    df_b_raw = load_intraday(SYMBOL_B)

    print(f"  {SYMBOL_A}: {len(df_a_raw)}行  {df_a_raw.index[0].date()} 〜 {df_a_raw.index[-1].date()}")
    print(f"  {SYMBOL_B}: {len(df_b_raw)}行  {df_b_raw.index[0].date()} 〜 {df_b_raw.index[-1].date()}")

    print("\n[日次メトリクス計算...]")
    df_a = compute_daily_metrics(df_a_raw)
    df_b = compute_daily_metrics(df_b_raw)
    print(f"  {SYMBOL_A}: {len(df_a)}日  {SYMBOL_B}: {len(df_b)}日")

    # スプレッド統計
    common = df_a.index.intersection(df_b.index)
    spread_series = df_a.loc[common, 'ret_signal'] - df_b.loc[common, 'ret_signal']
    print(f"\n[スプレッド統計 (9:30 ret差, %)]")
    print(f"  共通日数: {len(common)}")
    print(f"  平均: {spread_series.mean():+.3f}%")
    print(f"  標準偏差: {spread_series.std():.3f}%")
    print(f"  |spread| ≥ 0.3%: {(spread_series.abs() >= 0.3).sum()}日 ({(spread_series.abs() >= 0.3).mean()*100:.1f}%)")
    print(f"  |spread| ≥ 0.5%: {(spread_series.abs() >= 0.5).sum()}日 ({(spread_series.abs() >= 0.5).mean()*100:.1f}%)")
    print(f"  |spread| ≥ 1.0%: {(spread_series.abs() >= 1.0).sum()}日 ({(spread_series.abs() >= 1.0).mean()*100:.1f}%)")

    # パラメータグリッド
    print("\n[パラメータグリッド]")
    thresholds = [0.2, 0.3, 0.5, 0.7, 1.0]
    results = []
    for th in thresholds:
        trades = backtest(df_a, df_b, threshold=th)
        label = f"spread≥{th}%"
        r = evaluate(trades, label)
        if r:
            r['threshold'] = th
            results.append(r)

    # ベスト設定での詳細分析
    print("\n[最良設定での月別P&L]")
    if results:
        best = max(results, key=lambda x: x['sharpe'])
        best_th = best['threshold']
        trades_best = backtest(df_a, df_b, threshold=best_th)
        print(f"  threshold={best_th}%")
        trades_best['month'] = pd.to_datetime(trades_best['date']).dt.to_period('M')
        monthly = trades_best.groupby('month')['pnl_bps'].agg(['count', 'mean', 'sum'])
        monthly.columns = ['N', 'Mean', 'Total']
        for m, row in monthly.iterrows():
            print(f"    {m}: N={int(row['N'])}  Mean={row['Mean']:+.1f}bps  Total={row['Total']:+.0f}bps")

    # 方向別分析（ベスト設定）
    if results:
        trades_best = backtest(df_a, df_b, threshold=best_th)
        print(f"\n[方向別パフォーマンス (threshold={best_th}%)]")
        for side, g in trades_best.groupby('side'):
            arr = g['pnl_bps'].values
            print(f"  {side}: N={len(arr)}  Mean={arr.mean():+.1f}bps  WR={(arr>0).mean()*100:.1f}%")

    # 結論
    print("\n" + "=" * 70)
    print("[結論]")
    if results:
        best = max(results, key=lambda x: x['sharpe'])
        if best['sharpe'] >= 5.0 and best['t'] >= 2.0:
            print(f"✅ threshold={best['threshold']}% → Sharpe={best['sharpe']:+.2f}, t={best['t']:+.2f}")
            print(f"   N={best['n']}, Mean={best['mean']:+.1f}bps, WR={best['wr']:.1f}%")
            print("   採用候補。OoS検証を推奨。")
        else:
            print(f"⚠️  最良 threshold={best['threshold']}% でも Sharpe={best['sharpe']:+.2f}, t={best['t']:+.2f}")
            print("   統計的有意性が不十分。追加検証またはユニバース拡大が必要。")
    print("=" * 70)


if __name__ == "__main__":
    main()
