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
STOP_BPS = 400.0              # エントリー後 逆行 400bps で 強制手仕舞い
RANGE_MULT_SKIP = 2.5         # 9:00-10:00 レンジが過去20日中央値の X 倍超なら発動見送り
BASELINE_DAYS = 20            # ベースライン参照営業日数


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


def compute_first_hour_range_bps(df):
    """当日 9:00-10:00 のレンジを bps で返す (計算不能時は None)"""
    fh = df[df.index.hour == 9]
    if fh.empty:
        return None
    fh_open = fh.iloc[0]["open"]
    return (fh["high"].max() - fh["low"].min()) / fh_open * 10000


def load_baseline_first_hour_range(sym, ref_date, n_days=BASELINE_DAYS):
    """過去 n_days 営業日の 9:00-10:00 レンジ中央値 (bps) を返す"""
    conn = psycopg2.connect(**PG_CONFIG)
    start_utc = datetime.combine(ref_date - timedelta(days=int(n_days * 1.7) + 10),
                                 datetime.min.time())
    end_utc = datetime.combine(ref_date, datetime.min.time())
    df = pd.read_sql(
        "SELECT timestamp, open, high, low FROM intraday_data "
        "WHERE symbol=%s AND timestamp >= %s AND timestamp < %s ORDER BY timestamp",
        conn, params=(sym, start_utc, end_utc),
    )
    conn.close()
    if df.empty:
        return None
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["jst"] = df["timestamp"] + pd.Timedelta(hours=9)
    df = df.set_index("jst").sort_index()
    # 9:00-10:00 のみ
    fh = df[df.index.hour == 9]
    ranges = []
    for d, g in fh.groupby(fh.index.date):
        if len(g) < 10:
            continue
        g_valid = g.dropna(subset=["open", "high", "low"])
        if g_valid.empty:
            continue
        op = g_valid.iloc[0]["open"]
        hi = g_valid["high"].max(); lo = g_valid["low"].min()
        if op <= 0 or pd.isna(hi) or pd.isna(lo):
            continue
        ranges.append((hi - lo) / op * 10000)
    if len(ranges) < 5:
        return None
    # 直近 n_days のみ採用
    ranges = ranges[-n_days:]
    return float(np.median(ranges))


def check_range_skip(sym, target_date, df):
    """レンジ skip 判定。(skip?, fh_bps, baseline_bps, ratio)"""
    fh_bps = compute_first_hour_range_bps(df)
    if fh_bps is None:
        return False, None, None, None  # 判定不能 → skipしない (警告だけ)
    baseline = load_baseline_first_hour_range(sym, target_date)
    if baseline is None or baseline <= 0:
        return False, fh_bps, None, None
    ratio = fh_bps / baseline
    return (ratio >= RANGE_MULT_SKIP), fh_bps, baseline, ratio


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
        skip, fh_bps, baseline, ratio = check_range_skip(sym, target_date, df)

        # 現在時刻時点の乖離 (最新バー)
        latest = df.iloc[-1]
        print(f"\n[{sym} {name}]")
        print(f"  最新 {df.index[-1].strftime('%H:%M')}  "
              f"close={latest['close']:.1f}  VWAP={latest['vwap']:.1f}  "
              f"dev={latest['dev_bps']:+.0f}bps")
        if fh_bps is not None and baseline is not None:
            print(f"  当日 9:00-10:00 range: {fh_bps:.0f}bps / "
                  f"直近{BASELINE_DAYS}日中央値 {baseline:.0f}bps ({ratio:.2f}×)")
        elif fh_bps is not None:
            print(f"  当日 9:00-10:00 range: {fh_bps:.0f}bps (ベースライン取得失敗)")

        if skip:
            print(f"  ⚠️ レンジ過大 ({ratio:.2f}× ≥ {RANGE_MULT_SKIP}×) → **発動見送り**")
            continue
        if sig:
            print(f"  🔔 シグナル発生!")
            print(f"     時刻: {sig['time'].strftime('%H:%M')}")
            print(f"     価格: {sig['price']:.1f}  VWAP: {sig['vwap']:.1f}")
            print(f"     乖離: {sig['dev_bps']:+.0f}bps → **{sig['direction']}** エントリー")
            # Stop level (エントリー価格から 400bps 逆行)
            stop_mult = 1 + STOP_BPS/10000 if sig["direction"] == "SHORT" else 1 - STOP_BPS/10000
            stop_price = sig["price"] * stop_mult
            print(f"     Stop (±{STOP_BPS:.0f}bps): {stop_price:.1f}")
            # exit price (過去日のみ参照可能) + Stop ヒット判定
            after = df[df.index >= sig["time"]]
            stop_hit = False; exit_price = None; exit_time = None
            for ts, bar in after.iterrows():
                if sig["direction"] == "SHORT" and bar["high"] >= stop_price:
                    stop_hit = True; exit_price = stop_price; exit_time = ts; break
                if sig["direction"] == "LONG" and bar["low"] <= stop_price:
                    stop_hit = True; exit_price = stop_price; exit_time = ts; break
            if not stop_hit:
                ex_bars = df[(df.index.hour == EXIT_TIME[0]) & (df.index.minute == EXIT_TIME[1])]
                if not ex_bars.empty:
                    exit_price = ex_bars.iloc[0]["close"]; exit_time = ex_bars.index[0]
                else:
                    # 15:25 exact が無ければ 15:25 以前の最後の bar を採用
                    prior = df[df.index <= df.index.normalize()[0] + pd.Timedelta(hours=15, minutes=25)]
                    if not prior.empty:
                        exit_price = prior.iloc[-1]["close"]; exit_time = prior.index[-1]
            if exit_price is not None:
                direction = -1 if sig["direction"] == "SHORT" else 1
                ret_bps = direction * (exit_price / sig["price"] - 1) * 10000 - 4.0
                tag = "STOP" if stop_hit else "TIME"
                print(f"     Exit ({tag}): {exit_time.strftime('%H:%M')} @ {exit_price:.1f}")
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
    """リアルタイム監視 (10:00-11:30 でエントリー検知 + 15:25 まで Stop 400bps 監視)"""
    print("Live 監視モード (Ctrl+C で停止)")
    print(f"  エントリー窓: 10:00-11:30 / Stop ±{STOP_BPS:.0f}bps / Exit: 15:25")
    today = date.today()
    # sym -> {entry_price, direction, entry_time, stop_price, closed}
    positions = {}
    skipped = set()  # レンジ過大で当日見送り

    while True:
        now = datetime.now()
        hhmm = now.hour * 60 + now.minute
        # 市場外 → スリープ
        if now.hour < 9 or hhmm >= 15 * 60 + 30:
            print(f"{now.strftime('%H:%M')} — 市場時間外")
            time.sleep(60); continue

        for sym, name in TARGETS.items():
            if sym in skipped:
                continue
            df = load_today_intraday(sym, today)
            if df is None or len(df) < 5:
                continue
            df = compute_vwap(df)
            latest_bar = df.iloc[-1]

            # ── ポジション保有中: Stop / Exit 監視 ──
            if sym in positions and not positions[sym]["closed"]:
                p = positions[sym]
                direction = -1 if p["direction"] == "SHORT" else 1
                adverse_bps = -direction * (latest_bar["close"] / p["entry_price"] - 1) * 10000
                # Stop 判定 (1分足の high/low で厳密に)
                stop_hit = False
                if p["direction"] == "SHORT" and latest_bar["high"] >= p["stop_price"]:
                    stop_hit = True
                if p["direction"] == "LONG" and latest_bar["low"] <= p["stop_price"]:
                    stop_hit = True
                if stop_hit:
                    print(f"\n🛑 [{now.strftime('%H:%M')}] {sym} {name} STOP ヒット!")
                    print(f"   想定 Stop: {p['stop_price']:.1f}  — **即座に成行決済**")
                    p["closed"] = True
                    continue
                # 時間決済
                if hhmm >= EXIT_TIME[0] * 60 + EXIT_TIME[1]:
                    print(f"\n⏰ [{now.strftime('%H:%M')}] {sym} {name} 時間決済 (15:25)")
                    p["closed"] = True
                    continue
                # 定常ステータス表示
                print(f"  [{now.strftime('%H:%M')}] {sym} 保有中 {p['direction']} "
                      f"entry={p['entry_price']:.1f} now={latest_bar['close']:.1f} "
                      f"adverse={adverse_bps:+.0f}bps / stop={STOP_BPS:.0f}bps")
                continue

            # ── 未エントリー: エントリー窓 10:00-11:30 でのみ検知 ──
            if sym in positions:
                continue  # 当日既に建玉・決済済み
            if hhmm < ENTRY_WINDOW_START[0] * 60 + ENTRY_WINDOW_START[1]:
                continue
            if hhmm > ENTRY_WINDOW_END[0] * 60 + ENTRY_WINDOW_END[1]:
                continue
            # レンジ skip 判定 (10:00 時点で確定)
            skip, fh_bps, baseline, ratio = check_range_skip(sym, today, df)
            if skip:
                print(f"⚠️ [{now.strftime('%H:%M')}] {sym} {name} レンジ過大 "
                      f"({ratio:.2f}× ≥ {RANGE_MULT_SKIP}×) → 当日見送り")
                skipped.add(sym)
                continue
            sig = detect_signal(df)
            if sig:
                stop_mult = (1 + STOP_BPS/10000) if sig["direction"] == "SHORT" \
                            else (1 - STOP_BPS/10000)
                positions[sym] = {
                    "entry_price": sig["price"],
                    "direction": sig["direction"],
                    "entry_time": sig["time"],
                    "stop_price": sig["price"] * stop_mult,
                    "closed": False,
                }
                print(f"\n🔔 [{now.strftime('%H:%M')}] {sym} {name} シグナル!")
                print(f"   乖離 {sig['dev_bps']:+.0f}bps → **{sig['direction']}** "
                      f"@ {sig['price']:.1f}")
                print(f"   Stop: {positions[sym]['stop_price']:.1f} "
                      f"(±{STOP_BPS:.0f}bps)  Exit 予定: 15:25")
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
