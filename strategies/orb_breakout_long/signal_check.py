#!/usr/bin/env python3
"""
orb_breakout_long — シグナル検知

使い方:
  # 過去日チェック
  python3 signal_check.py --date 2026-04-15

  # 本日チェック (default)
  python3 signal_check.py

  # リアルタイム監視 (09:30-15:25)
  python3 signal_check.py --live
"""
import argparse
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2

PG_CONFIG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}

# (symbol, name, OR_minutes)
TARGETS = [
    ("5706.T", "三井金属", 30),
    ("6146.T", "ディスコ", 60),
]
EXIT_HM = (15, 25)


def load_today_intraday(sym, target_date):
    """JST の target_date 9:00-15:30 の1分足を取得"""
    conn = psycopg2.connect(**PG_CONFIG)
    start_utc = datetime.combine(target_date, datetime.min.time())  # JST 9:00 = UTC 00:00
    end_utc = start_utc + timedelta(hours=9)  # JST 9:00 〜 18:00
    df = pd.read_sql(
        "SELECT timestamp, open, high, low, close, volume FROM intraday_data "
        "WHERE symbol=%s AND timestamp >= %s AND timestamp < %s ORDER BY timestamp",
        conn, params=(sym, start_utc, end_utc),
    )
    conn.close()
    if df.empty:
        return None
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["jst"] = df["timestamp"] + pd.Timedelta(hours=9)
    df = df.set_index("jst").sort_index()
    h, m = df.index.hour, df.index.minute
    morning = (h == 9) | (h == 10) | ((h == 11) & (m <= 30))
    afternoon = ((h == 12) & (m >= 30)) | (h == 13) | (h == 14) | ((h == 15) & (m <= 30))
    return df[morning | afternoon].copy()


def detect_orb_signal(df, or_minutes):
    """Opening Range Breakout Long を検知 (今日に限らず過去日でも使える)"""
    h = df.index.hour; m = df.index.minute
    min_of_day = h * 60 + m
    or_start = 9 * 60
    or_end = or_start + or_minutes
    or_bars = df[(min_of_day >= or_start) & (min_of_day < or_end)]
    if len(or_bars) < max(3, or_minutes // 2):
        return None
    or_high = or_bars["high"].max()
    or_low = or_bars["low"].min()
    post = df[min_of_day >= or_end].copy()
    if post.empty:
        return {"or_high": or_high, "or_low": or_low, "triggered": False}

    # 最初に or_high を超えたバーを探す
    for ts, bar in post.iterrows():
        if bar["high"] > or_high:
            return {
                "or_high": or_high, "or_low": or_low,
                "triggered": True,
                "entry_time": ts, "entry_price": or_high,
                "exit_bar": None,
            }
    return {"or_high": or_high, "or_low": or_low, "triggered": False}


def evaluate_after_entry(df, sig):
    """エントリー後の経路を評価 (過去日再生用)"""
    if not sig["triggered"]:
        return sig
    entry_time = sig["entry_time"]; or_low = sig["or_low"]
    after = df[df.index >= entry_time]
    stop_hit = False; exit_price = None; exit_time = None
    for ts, bar in after.iterrows():
        if bar["low"] <= or_low:
            exit_price = or_low; exit_time = ts; stop_hit = True; break
    if exit_price is None:
        time_bars = after[(after.index.hour == EXIT_HM[0]) & (after.index.minute == EXIT_HM[1])]
        if not time_bars.empty:
            exit_price = time_bars.iloc[0]["close"]; exit_time = time_bars.index[0]
        else:
            exit_price = after.iloc[-1]["close"]; exit_time = after.index[-1]
    gross_bps = (exit_price / sig["entry_price"] - 1) * 10000
    net_bps = gross_bps - 4.0  # コスト控除
    sig.update({"exit_time": exit_time, "exit_price": exit_price,
                "stop_hit": stop_hit, "gross_bps": gross_bps, "net_bps": net_bps})
    return sig


def run_for_date(target_date):
    print("=" * 70)
    print(f"orb_breakout_long シグナルチェック — {target_date}")
    print("=" * 70)
    triggered = 0
    for sym, name, or_min in TARGETS:
        df = load_today_intraday(sym, target_date)
        if df is None or len(df) < or_min + 10:
            print(f"\n[{sym} {name}] データ不足")
            continue
        sig = detect_orb_signal(df, or_min)
        if sig is None:
            print(f"\n[{sym} {name}] OR計算不可")
            continue
        print(f"\n[{sym} {name}] (OR={or_min}分)")
        print(f"  OR High: {sig['or_high']:.1f}  OR Low: {sig['or_low']:.1f}  "
              f"Range: {(sig['or_high']-sig['or_low'])/sig['or_low']*10000:.0f}bps")
        if not sig["triggered"]:
            print("  シグナルなし (OR High ブレイクなし)")
            continue
        triggered += 1
        sig = evaluate_after_entry(df, sig)
        print(f"  🔔 ブレイク発生")
        print(f"     Entry: {sig['entry_time'].strftime('%H:%M')} @ {sig['entry_price']:.1f}")
        if sig.get("exit_time"):
            tag = "STOP" if sig["stop_hit"] else "TIME"
            print(f"     Exit ({tag}): {sig['exit_time'].strftime('%H:%M')} @ {sig['exit_price']:.1f}")
            print(f"     Gross: {sig['gross_bps']:+.0f}bps  Net: {sig['net_bps']:+.0f}bps")
    print("\n" + "=" * 70)
    print(f"トリガ銘柄数: {triggered}/{len(TARGETS)}")
    print("=" * 70)


def run_live():
    """リアルタイム監視"""
    print("orb_breakout_long Live 監視 (Ctrl+C で停止)")
    today = date.today()
    triggered = set()
    while True:
        now = datetime.now()
        if now.hour < 9 or now.hour >= 16:
            print(f"{now.strftime('%H:%M')} — 市場時間外")
            time.sleep(60); continue
        for sym, name, or_min in TARGETS:
            if sym in triggered:
                continue
            # OR 完了前はスキップ
            or_end_min = 9 * 60 + or_min
            if now.hour * 60 + now.minute < or_end_min:
                continue
            df = load_today_intraday(sym, today)
            if df is None:
                continue
            sig = detect_orb_signal(df, or_min)
            if sig and sig["triggered"]:
                print(f"\n🔔 [{now.strftime('%H:%M')}] {sym} {name} ORB Long!")
                print(f"   Entry ~{sig['entry_price']:.1f} / Stop {sig['or_low']:.1f}")
                triggered.add(sym)
        time.sleep(60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", type=str, help="YYYY-MM-DD (default: today)")
    ap.add_argument("--live", action="store_true", help="リアルタイム監視")
    args = ap.parse_args()

    if args.live:
        run_live()
    else:
        target = date.today() if not args.date else datetime.strptime(args.date, "%Y-%m-%d").date()
        run_for_date(target)


if __name__ == "__main__":
    main()
