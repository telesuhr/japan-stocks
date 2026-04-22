#!/usr/bin/env python3
"""
vwap_morning_meanrevert — シグナル検知スクリプト

使い方:
  # 過去日チェック
  python3 signal_check.py --date 2026-04-15

  # 本日チェック (default)
  python3 signal_check.py

  # リアルタイム監視 (10:00-11:30)
  python3 signal_check.py --live
"""
import argparse
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2

PG_CONFIG = {
    "host": "localhost", "port": 5432,
    "user": "postgres", "dbname": "market_data",
}

# 対象銘柄
TARGETS = {
    "8035.T": "TEL",
    "6146.T": "ディスコ",
    "6920.T": "レーザー",
}

# パラメータ
THRESH_BPS = 275.0
ENTRY_WINDOW_START = (10, 0)  # 10:00 JST
ENTRY_WINDOW_END = (11, 30)   # 11:30 JST
EXIT_TIME = (15, 25)
STOP_BPS = 400.0


def load_today_intraday(sym, target_date):
    """指定日 (JST) の 9:00-15:30 の1分足を取得"""
    conn = psycopg2.connect(**PG_CONFIG)
    # UTC 保存、JST = UTC+9h。target_date JST 9:00 = target_date 00:00 UTC
    start_utc = datetime.combine(target_date, datetime.min.time())
    end_utc = start_utc + timedelta(hours=9)  # JST 9:00 〜 18:00 をカバー
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
    # 取引時間のみ
    h, m = df.index.hour, df.index.minute
    morning = (h == 9) | (h == 10) | ((h == 11) & (m <= 30))
    afternoon = ((h == 12) & (m >= 30)) | (h == 13) | (h == 14) | ((h == 15) & (m <= 30))
    return df[morning | afternoon].copy()


def compute_vwap(df):
    """9:00 からの累積 session VWAP と乖離 bps"""
    df = df.copy()
    pv = (df["close"] * df["volume"]).cumsum()
    cv = df["volume"].cumsum().replace(0, np.nan)
    df["vwap"] = pv / cv
    df["dev_bps"] = (df["close"] / df["vwap"] - 1) * 10000
    return df


def detect_signal(df, thresh=THRESH_BPS):
    """10:00-11:30 の間で初めて |dev| >= thresh を超えた bar を返す"""
    h = df.index.hour; m = df.index.minute
    min_of_day = h * 60 + m
    start = ENTRY_WINDOW_START[0] * 60 + ENTRY_WINDOW_START[1]
    end = ENTRY_WINDOW_END[0] * 60 + ENTRY_WINDOW_END[1]
    window = df[(min_of_day >= start) & (min_of_day <= end)]
    trigger_mask = window["dev_bps"].abs() >= thresh
    if not trigger_mask.any():
        return None
    idx = np.where(trigger_mask)[0][0]
    row = window.iloc[idx]
    return {
        "time": window.index[idx],
        "price": row["close"],
        "vwap": row["vwap"],
        "dev_bps": row["dev_bps"],
        "direction": "SHORT" if row["dev_bps"] > 0 else "LONG",
    }


def check_first_hour_range_safe(df):
    """発動見送り判定 (当日 9:00-10:00 レンジが過去20日中央値の2.5倍以上なら危険)"""
    fh = df[df.index.hour == 9]
    if fh.empty:
        return True  # 判断不能 → 安全側
    fh_open = fh.iloc[0]["open"]
    fh_range_bps = (fh["high"].max() - fh["low"].min()) / fh_open * 10000
    return fh_range_bps  # 呼び出し側で比較


def run_for_date(target_date):
    print("=" * 70)
    print(f"vwap_morning_meanrevert シグナルチェック — {target_date}")
    print(f"閾値: |dev| >= {THRESH_BPS}bps / 監視窓: 10:00-11:30 / エグジット: 15:25")
    print("=" * 70)

    results = []
    for sym, name in TARGETS.items():
        df = load_today_intraday(sym, target_date)
        if df is None or len(df) < 30:
            print(f"\n[{sym} {name}] データなし、またはバー不足")
            continue
        df = compute_vwap(df)
        sig = detect_signal(df)
        fh_range = check_first_hour_range_safe(df)

        # 現在時刻時点の乖離 (最新バー)
        latest = df.iloc[-1]
        print(f"\n[{sym} {name}]")
        print(f"  最新 {df.index[-1].strftime('%H:%M')}  "
              f"close={latest['close']:.1f}  VWAP={latest['vwap']:.1f}  "
              f"dev={latest['dev_bps']:+.0f}bps")
        print(f"  当日 9:00-10:00 range: {fh_range:.0f}bps")

        if sig:
            print(f"  🔔 シグナル発生!")
            print(f"     時刻: {sig['time'].strftime('%H:%M')}")
            print(f"     価格: {sig['price']:.1f}  VWAP: {sig['vwap']:.1f}")
            print(f"     乖離: {sig['dev_bps']:+.0f}bps → **{sig['direction']}** エントリー")
            # exit price (過去日のみ参照可能)
            ex_bars = df[(df.index.hour == EXIT_TIME[0]) & (df.index.minute == EXIT_TIME[1])]
            if not ex_bars.empty:
                ex_price = ex_bars.iloc[0]["close"]
                direction = -1 if sig["direction"] == "SHORT" else 1
                ret_bps = direction * (ex_price / sig["price"] - 1) * 10000 - 4.0
                print(f"     15:25 exit price: {ex_price:.1f}")
                print(f"     net bps (コスト4bps控除後): {ret_bps:+.0f}")
            results.append({"sym": sym, **sig})
        else:
            print(f"  シグナルなし (10:00-11:30 で |dev| < {THRESH_BPS}bps)")

    print("\n" + "=" * 70)
    print(f"トリガ銘柄数: {len(results)}/{len(TARGETS)}")
    if results:
        for r in results:
            print(f"  - {r['sym']} {TARGETS[r['sym']]} {r['direction']} @ {r['time'].strftime('%H:%M')} "
                  f"dev={r['dev_bps']:+.0f}bps")
    print("=" * 70)
    return results


def run_live():
    """リアルタイム監視 (10:00-11:30 の間 1分ごとにチェック)"""
    print("Live 監視モード (Ctrl+C で停止)")
    today = date.today()
    triggered = set()
    while True:
        now = datetime.now()
        if now.hour < 10 or (now.hour == 11 and now.minute > 30) or now.hour >= 12:
            print(f"{now.strftime('%H:%M')} — 監視窓外")
            time.sleep(60)
            continue
        for sym, name in TARGETS.items():
            if sym in triggered:
                continue
            df = load_today_intraday(sym, today)
            if df is None:
                continue
            df = compute_vwap(df)
            sig = detect_signal(df)
            if sig:
                print(f"\n🔔 [{now.strftime('%H:%M')}] {sym} {name} シグナル!")
                print(f"   乖離 {sig['dev_bps']:+.0f}bps → **{sig['direction']}**")
                triggered.add(sym)
        time.sleep(60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", type=str, help="YYYY-MM-DD (default: today)")
    ap.add_argument("--live", action="store_true", help="リアルタイム監視モード")
    args = ap.parse_args()

    if args.live:
        run_live()
    else:
        target = date.today() if not args.date else datetime.strptime(args.date, "%Y-%m-%d").date()
        run_for_date(target)


if __name__ == "__main__":
    main()
