#!/usr/bin/env python3
"""
ENEOS VWAP Trend シグナル自動判定スクリプト (v2.0 — JQuants/PG新DB対応)

9:30 頃に実行:
  python3 signal_check.py

シグナル = 5020.T (50200, ENEOS) の 9:30時点 VWAP乖離 ≥ ±50bps
発動 → 9:31〜 成行エントリー → 15:30 引成決済

DB: stocks_intraday (5桁code, ts は JST naive)
"""
import sys
from datetime import date, datetime, timedelta
import psycopg2
import pandas as pd

PG_CONFIG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}

CODE5 = "50200"
RIC   = "5020.T"
NAME  = "ENEOS"
THRESHOLD = 50  # bps
SIGNAL_HOUR = 9
SIGNAL_MIN = 30


def fetch_vwap_dev(target_date: date = None):
    if target_date is None:
        target_date = date.today()
    conn = psycopg2.connect(**PG_CONFIG)
    start = datetime.combine(target_date, datetime.min.time())
    end = start + timedelta(days=1)
    df = pd.read_sql(
        "SELECT ts, open, high, low, close, volume FROM stocks_intraday "
        "WHERE code = %s AND ts >= %s AND ts < %s ORDER BY ts",
        conn, params=(CODE5, start, end))
    conn.close()

    if df.empty:
        return None

    df["ts"] = pd.to_datetime(df["ts"])
    df = df.set_index("ts").sort_index()

    morning = df[df.index.hour >= 9]
    if morning.empty:
        return None

    vol = morning["volume"].fillna(0)
    vol = vol.where(vol > 0, 1.0)
    cum_pv = (morning["close"] * vol).cumsum()
    cum_vol = vol.cumsum()
    vwap_series = cum_pv / cum_vol

    bar_930 = morning[
        (morning.index.hour == SIGNAL_HOUR) & (morning.index.minute == SIGNAL_MIN)
    ]
    if bar_930.empty:
        return None

    latest = bar_930.iloc[-1]
    close_930 = float(latest["close"])
    vwap_930 = float(vwap_series.loc[bar_930.index[-1]])
    dev_bps = (close_930 / vwap_930 - 1) * 10000

    return {
        "time": bar_930.index[-1].strftime("%H:%M"),
        "close": close_930,
        "vwap": vwap_930,
        "dev_bps": dev_bps,
    }


def check_signal():
    print("=" * 65)
    print(f"ENEOS ({RIC}, code={CODE5}) VWAP Trend シグナル判定 (新DB v2.0)")
    print(f"実行時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S JST')}")
    print("=" * 65)

    today = date.today()
    weekday = today.strftime("%A")
    print(f"\n本日: {today} ({weekday})")

    if today.weekday() >= 5:
        print("\n🚫 土日のためスキップ")
        return 0

    try:
        g = fetch_vwap_dev()
    except Exception as e:
        print(f"\n❌ データ取得エラー: {e}")
        return 1

    if g is None:
        print("\n❌ データ不足 (9:30バーが取得できない)")
        return 1

    print(f"\n[VWAP乖離 @ {g['time']}]")
    print(f"  現値:  ¥{g['close']:,.1f}")
    print(f"  VWAP:  ¥{g['vwap']:,.1f}")
    print(f"  乖離:  {g['dev_bps']:+.1f} bps  (閾値 ±{THRESHOLD}bps)")

    if g["dev_bps"] >= THRESHOLD:
        direction = "Long"; signal = True
    elif g["dev_bps"] <= -THRESHOLD:
        direction = "Short"; signal = True
    else:
        direction = None; signal = False

    print(f"  判定:  {'✅ 通過 → ' + direction if signal else '❌ 不足 → スキップ'}")

    if not signal:
        print("\n🚫 本日はシグナル不発 → スキップ")
        return 0

    print("\n" + "=" * 65)
    print(f"🟢 シグナル発動 → {direction} エントリー")
    print("=" * 65)
    shares_1000 = int(10_000_000 / g["close"] / 100) * 100
    print(f"\n[発注目安 ¥1,000万の場合]  株数: {shares_1000:,}株 @ ¥{g['close']:,.1f}")
    print(f"\n[決済] 15:25-15:29 引成 (CLO)")
    return 0


if __name__ == "__main__":
    sys.exit(check_signal())
