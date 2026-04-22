#!/usr/bin/env python3
"""
lasertec_ma25_support — シグナル検知

使い方:
  # 本日の判定
  python3 signal_check.py

  # 過去日の判定
  python3 signal_check.py --date 2026-04-01

  # 過去の全シグナル履歴
  python3 signal_check.py --history
"""
import argparse
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2

PG_CONFIG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}
SYM = "6920.T"
MA_PERIOD = 25
DD_THRESH = 5.0
TOUCH_TOL_PCT = 1.0
SLOPE_LOOKBACK = 5
HOLD_DAYS = 10
STOP_PCT = 10.0


def load_daily():
    conn = psycopg2.connect(**PG_CONFIG)
    df = pd.read_sql(
        "SELECT trade_date, open, high, low, close FROM daily_stats "
        "WHERE symbol=%s ORDER BY trade_date", conn, params=(SYM,))
    conn.close()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.set_index("trade_date").sort_index()
    df = df.astype({c: float for c in ["open", "high", "low", "close"]})
    return df


def annotate(df):
    df = df.copy()
    df["ma25"] = df["close"].rolling(MA_PERIOD).mean()
    df["ma25_5d_ago"] = df["ma25"].shift(SLOPE_LOOKBACK)
    df["hh20"] = df["close"].rolling(20).max()
    df["dd20_pct"] = (df["close"] / df["hh20"] - 1) * 100
    lo, hi = df["low"], df["high"]
    ma = df["ma25"]
    df["touched"] = (lo <= ma * (1 + TOUCH_TOL_PCT / 100)) & (hi >= ma * (1 - TOUCH_TOL_PCT / 100))
    df["downtrend"] = df["dd20_pct"] <= -DD_THRESH
    df["slope_up"] = df["ma25"] > df["ma25_5d_ago"]
    df["signal"] = df["touched"] & df["downtrend"] & df["slope_up"] & df["ma25"].notna()
    return df


def check_date(target_date=None):
    df = load_daily()
    df = annotate(df)
    if target_date is None:
        target_date = df.index[-1].date()
    ts = pd.Timestamp(target_date)
    if ts not in df.index:
        print(f"Date {target_date} not in data. Latest: {df.index[-1].date()}")
        return
    row = df.loc[ts]
    print("=" * 70)
    print(f"lasertec_ma25_support シグナルチェック — {target_date}")
    print("=" * 70)
    print(f"\n[6920.T] 終値 {row['close']:.0f}  日中High {row['high']:.0f}  Low {row['low']:.0f}")
    print(f"  MA25:     {row['ma25']:.1f}" if not pd.isna(row['ma25']) else "  MA25: データ不足")
    if not pd.isna(row['ma25']):
        print(f"  MA25 ±1%: [{row['ma25']*0.99:.1f}, {row['ma25']*1.01:.1f}]")
    print(f"  20日高値: {row['hh20']:.0f}")
    print(f"  dd20:     {row['dd20_pct']:+.2f}%")
    print(f"  MA25 5日前: {row['ma25_5d_ago']:.1f}" if not pd.isna(row['ma25_5d_ago']) else "")
    if not pd.isna(row['ma25']) and not pd.isna(row['ma25_5d_ago']):
        slope_pct = (row['ma25'] / row['ma25_5d_ago'] - 1) * 100
        print(f"  MA25 傾き: {slope_pct:+.2f}% (5日変化)")

    print("\n[判定]")
    print(f"  ① 下落局面 (dd20 ≤ -{DD_THRESH}%):        {'✅ YES' if row['downtrend'] else '❌ NO'}")
    print(f"  ② MA25 接触 (±{TOUCH_TOL_PCT}%):             {'✅ YES' if row['touched'] else '❌ NO'}")
    print(f"  ③ MA25 上昇中 (slope>0):          {'✅ YES' if row['slope_up'] else '❌ NO'}")
    print()
    if row["signal"]:
        print("🔔 エントリーシグナル発生!")
        print(f"   → 翌営業日 寄成 Long (¥500-1,000万)")
        print(f"   → Stop: 約定価格 × 0.90 (逆指値成行)")
        print(f"   → Exit: 10営業日後 15:25 引成")
        # 過去日なら実際の結果を再生
        idx = df.index.get_loc(ts)
        if idx + 1 < len(df):
            entry_next_open = df.iloc[idx + 1]["open"]
            print(f"   (翌営業日 {df.index[idx+1].date()} 寄り想定エントリー: {entry_next_open:.0f})")
            stop_level = entry_next_open * 0.90
            # walk forward
            fut = df.iloc[idx + 1: idx + 1 + HOLD_DAYS + 1]
            stop_hit = False; exit_price = None; exit_date = None
            for d, f in fut.iloc[1:].iterrows():
                if f["low"] <= stop_level:
                    exit_price = stop_level; exit_date = d; stop_hit = True; break
            if exit_price is None and len(fut) > HOLD_DAYS:
                exit_date = fut.index[HOLD_DAYS]
                exit_price = fut.iloc[HOLD_DAYS]["close"]
            if exit_price is not None:
                ret_pct = (exit_price / entry_next_open - 1) * 100 - 0.04
                print(f"   [過去再生] Exit {exit_date.date()} @ {exit_price:.0f} "
                      f"→ {'STOP' if stop_hit else 'TIME'} {ret_pct:+.2f}%")
    else:
        print("シグナルなし (本日エントリーは見送り)")
    print("=" * 70)


def history():
    """過去の全シグナルとパフォーマンス"""
    df = load_daily()
    df = annotate(df)
    sig_dates = df[df["signal"]].index
    print("=" * 90)
    print(f"lasertec_ma25_support 過去シグナル一覧  ({len(sig_dates)} 件)")
    print("=" * 90)
    print(f"{'日付':<12} {'終値':>8} {'MA25':>8} {'dd20%':>7} {'翌寄':>8} {'Exit':>10} {'Ret%':>7} {'Stop':>5}")
    total = 0.0; wins = 0; losses = 0
    for ts in sig_dates:
        idx = df.index.get_loc(ts)
        if idx + 1 >= len(df):
            continue
        entry = df.iloc[idx + 1]["open"]
        stop_level = entry * 0.90
        fut = df.iloc[idx + 1: idx + 1 + HOLD_DAYS + 1]
        stop_hit = False; exit_price = None; exit_date = None
        for d, r in fut.iloc[1:].iterrows():
            if r["low"] <= stop_level:
                exit_price = stop_level; exit_date = d; stop_hit = True; break
        if exit_price is None and len(fut) > HOLD_DAYS:
            exit_date = fut.index[HOLD_DAYS]
            exit_price = fut.iloc[HOLD_DAYS]["close"]
        elif exit_price is None:
            exit_date = fut.index[-1]; exit_price = fut.iloc[-1]["close"]
        ret_pct = (exit_price / entry - 1) * 100 - 0.04
        total += ret_pct
        wins += 1 if ret_pct > 0 else 0
        losses += 1 if ret_pct <= 0 else 0
        print(f"{ts.date().isoformat():<12} {df.loc[ts,'close']:>8.0f} "
              f"{df.loc[ts,'ma25']:>8.1f} {df.loc[ts,'dd20_pct']:>+7.2f} "
              f"{entry:>8.0f} {exit_date.date().isoformat():>10} "
              f"{ret_pct:>+7.2f} {'STOP' if stop_hit else '':>5}")
    print("-" * 90)
    n = wins + losses
    print(f"N={n}  勝={wins}  敗={losses}  WR={wins/n*100:.1f}%  "
          f"累積(単純和)={total:+.1f}%  平均={total/n:+.2f}%" if n > 0 else "N=0")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", type=str, help="YYYY-MM-DD (default: today)")
    ap.add_argument("--history", action="store_true", help="過去の全シグナル履歴")
    args = ap.parse_args()
    if args.history:
        history()
    else:
        target = date.today() if not args.date else datetime.strptime(args.date, "%Y-%m-%d").date()
        check_date(target)


if __name__ == "__main__":
    main()
