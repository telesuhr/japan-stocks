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
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2

PG_CONFIG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}

# (code5, name, OR_minutes) — JQuants 5桁
TARGETS = [
    ("57060", "三井金属 (5706.T)", 30),
    ("61460", "ディスコ (6146.T)", 60),
]
EXIT_HM = (15, 25)

# 他戦略との衝突解決
#   vwap_morning_meanrevert: ディスコ (61460) 重複 → vwap 発動ならORB スキップ
#   sox_overnight_short は廃止 (LME/米マクロ依存戦略は _archive/ へ)
VWAP_THRESH_BPS = 275.0
VWAP_ENTRY_START = (10, 0)
VWAP_ENTRY_END = (11, 30)
VWAP_CONFLICT_SYMS = {"61460"}  # ディスコのみ重複


def load_today_intraday(sym, target_date):
    """JST の target_date 9:00-15:30 の1分足を取得 (新DB stocks_intraday)"""
    conn = psycopg2.connect(**PG_CONFIG)
    start = datetime.combine(target_date, datetime.min.time())
    end = start + timedelta(days=1)
    df = pd.read_sql(
        "SELECT ts, open, high, low, close, volume FROM stocks_intraday "
        "WHERE code=%s AND ts >= %s AND ts < %s ORDER BY ts",
        conn, params=(sym, start, end),
    )
    conn.close()
    if df.empty:
        return None
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.set_index("ts").sort_index()
    h, m = df.index.hour, df.index.minute
    morning = (h == 9) | (h == 10) | ((h == 11) & (m <= 30))
    afternoon = ((h == 12) & (m >= 30)) | (h == 13) | (h == 14) | ((h == 15) & (m <= 30))
    return df[morning | afternoon].copy()


def detect_vwap_conflict(df, up_to_time=None):
    """当日ディスコの vwap_morning_meanrevert シグナル発動有無を判定。

    True なら vwap 側が優先 → ORB をスキップ。
    up_to_time (datetime or None) が指定された場合、その時刻までに限定して判定。
    """
    if df is None or df.empty:
        return False, None
    sub = df if up_to_time is None else df[df.index <= up_to_time]
    if sub.empty:
        return False, None
    pv = (sub["close"] * sub["volume"]).cumsum()
    cv = sub["volume"].cumsum().replace(0, np.nan)
    vwap = pv / cv
    dev_bps = (sub["close"] / vwap - 1) * 10000
    h = sub.index.hour; m = sub.index.minute
    min_of_day = h * 60 + m
    start = VWAP_ENTRY_START[0] * 60 + VWAP_ENTRY_START[1]
    end = VWAP_ENTRY_END[0] * 60 + VWAP_ENTRY_END[1]
    window_mask = (min_of_day >= start) & (min_of_day <= end)
    trigger = (dev_bps.abs() >= VWAP_THRESH_BPS) & window_mask
    if not trigger.any():
        return False, None
    idx = np.where(trigger.values)[0][0]
    return True, {"time": sub.index[idx], "dev_bps": float(dev_bps.iloc[idx])}


def detect_sox_conflict(target_date):
    """旧: sox_overnight_short 衝突判定。LME/米マクロ依存戦略は廃止 → 常に non-active を返す。"""
    return False, {"date": target_date, "ret_pct": 0.0, "note": "sox 戦略廃止のため常に通過"}


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

    # ── 事前: 市場レベル衝突判定 (.SOX 前日急落 → 当日全 ORB スキップ) ──
    sox_active, sox_info = detect_sox_conflict(target_date)
    if sox_active is True:
        print(f"\n🛑 sox_overnight_short 発動日 "
              f"(.SOX {sox_info['date']} ret={sox_info['ret_pct']:+.2f}%)")
        print("   → 市場下落バイアス下での ORB Long は回避 → **当日全 ORB スキップ**")
        print("=" * 70)
        return
    else:
        print("\n✅ sox 判定: 廃止 (LME/米マクロ依存戦略は _archive/ へ) → ORB 続行")

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

        # ── 銘柄レベル衝突判定: ディスコは vwap 優先 ──
        # 過去日評価: 1日を通して vwap が発動するか確認 (後知恵OK)
        # ライブ運用: ORB エントリー時点までの vwap 発動のみ確認 (run_live 側で別処理)
        if sym in VWAP_CONFLICT_SYMS:
            vwap_fired, vwap_info = detect_vwap_conflict(df)  # 全日スキャン
            if vwap_fired:
                orb_entry = sig.get("entry_time")
                timing = ""
                if orb_entry is not None:
                    delta = (vwap_info["time"] - orb_entry).total_seconds() / 60
                    timing = f" (ORB入場 {orb_entry.strftime('%H:%M')} "
                    timing += f"→ vwap {'後' if delta>0 else '前'} {abs(delta):.0f}分)"
                print(f"  🛑 vwap_morning_meanrevert 発動日 "
                      f"({vwap_info['time'].strftime('%H:%M')} "
                      f"dev={vwap_info['dev_bps']:+.0f}bps){timing}")
                print("     → **vwap 優先** (Sharpe+6.11 > +2.15) → ORB スキップ")
                continue
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
    skipped = set()

    # 市場開始前に sox 判定 (毎ループでMariaDB叩くのは避ける)
    sox_active, sox_info = detect_sox_conflict(today)
    if sox_active is True:
        print(f"🛑 本日は sox_overnight_short 発動日 "
              f"(.SOX {sox_info['date']} {sox_info['ret_pct']:+.2f}%) → ORB 全銘柄スキップ")
        return
    elif sox_active is None:
        print(f"⚠️ sox 判定不能: {sox_info} (手動確認後に続行)")
    else:
        print(f"✅ sox チェック通過 (.SOX {sox_info['ret_pct']:+.2f}%)")

    while True:
        now = datetime.now()
        if now.hour < 9 or now.hour >= 16:
            print(f"{now.strftime('%H:%M')} — 市場時間外")
            time.sleep(60); continue
        for sym, name, or_min in TARGETS:
            if sym in triggered or sym in skipped:
                continue
            # OR 完了前はスキップ
            or_end_min = 9 * 60 + or_min
            if now.hour * 60 + now.minute < or_end_min:
                continue
            df = load_today_intraday(sym, today)
            if df is None:
                continue
            # ディスコは vwap 発動チェック (エントリー前)
            if sym in VWAP_CONFLICT_SYMS:
                vwap_fired, vwap_info = detect_vwap_conflict(df)
                if vwap_fired:
                    print(f"🛑 [{now.strftime('%H:%M')}] {sym} {name} "
                          f"vwap 発動済 ({vwap_info['time'].strftime('%H:%M')} "
                          f"dev={vwap_info['dev_bps']:+.0f}bps) → ORB 本日スキップ")
                    skipped.add(sym)
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
