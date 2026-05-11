#!/usr/bin/env python3
"""
TOPIX夜間ギャップ シグナル自動判定スクリプト (v2.0 — JQuants/PG新DB対応)

Day N 09:05 頃に実行:
  python3 signal_check.py

シグナル = TOPIX (index_daily code='0000') 前日終値 → Day N 寄付 変化率 ≥ +0.3%
発動 → Day N 15:30 引成 Long (CORE5 各¥1,000万)
決済 → Day N+1 09:00 寄成

DB: PostgreSQL market_data.public
  - index_daily (TOPIX 日足 OHLC)
  - stocks_daily (CORE5 日足、必要時)
"""
import sys
from datetime import date, datetime
import psycopg2

PG_CONFIG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}

THRESHOLD_PCT = 0.3
TOPIX_CODE = "0000"

# CORE5 — JQuants 5桁コード
CORE5 = [
    ("57110", "5711.T", "三菱マテリアル"),
    ("65010", "6501.T", "日立製作所"),
    ("70110", "7011.T", "三菱重工業"),
    ("50160", "5016.T", "ＪＸ金属"),
    ("45020", "4502.T", "武田薬品"),
]


def fetch_topix_gap(target_date: date):
    """
    TOPIX (index_daily) の前日終値 → target_date 当日寄付 の変化率を計算。
    index_daily は日足 OHLC のみ。当日 open を「寄付値 ≒ 9:00近辺の値」として使う。
    """
    conn = psycopg2.connect(**PG_CONFIG)
    cur = conn.cursor()
    cur.execute("""
        SELECT date, open, close FROM index_daily
        WHERE code = %s AND date <= %s
        ORDER BY date DESC LIMIT 5
    """, (TOPIX_CODE, target_date))
    rows = cur.fetchall()
    cur.close()
    conn.close()

    if len(rows) < 2:
        return None

    today_row = None
    prev_row = None
    for r in rows:
        d = r[0]
        if d == target_date:
            today_row = r
        elif today_row is not None and prev_row is None:
            prev_row = r
            break

    if today_row is None:
        today_row = rows[0]
        prev_row = rows[1]

    today_open = float(today_row[1])
    prev_close = float(prev_row[2])

    return {
        "prev_date": prev_row[0],
        "prev_close": prev_close,
        "today_date": today_row[0],
        "today_open": today_open,
        "gap_pct": (today_open / prev_close - 1) * 100,
    }


def check_signal():
    print("=" * 70)
    print("TOPIX夜間ギャップ シグナル判定 (新DB v2.0)")
    print(f"実行時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S JST')}")
    print("=" * 70)

    today = date.today()
    weekday = today.strftime("%a")
    print(f"\n本日 (エントリー候補日 Day N): {today} ({weekday})")

    try:
        g = fetch_topix_gap(today)
    except Exception as e:
        print(f"❌ TOPIXデータ取得エラー: {e}")
        return 1

    if g is None:
        print("❌ TOPIXデータ不足 (前日終値 or 当日寄付取得不可)")
        return 1

    print("\n[TOPIX夜間変化率]")
    print(f"  前日 ({g['prev_date']}) 終値: {g['prev_close']:.2f}")
    print(f"  当日 ({g['today_date']}) 寄付: {g['today_open']:.2f}")
    print(f"  ギャップ: {g['gap_pct']:+.3f}% (閾値 +{THRESHOLD_PCT:.1f}%)")

    passed = g["gap_pct"] >= THRESHOLD_PCT
    print(f"  判定: {'✅ 通過' if passed else '❌ 不足'}")

    if not passed:
        print("\n🚫 本日はシグナル不発 → スキップ")
        return 0

    if today.weekday() == 3:
        print("\n🚫 木曜エントリーのためスキップ")
        return 0

    print("\n" + "=" * 70)
    print("🟢 シグナル発動候補 — Day N 15:30 引成 Long (予定)")
    print("=" * 70)
    print("\n[対象銘柄 各¥1,000万]")
    for code5, ric, name in CORE5:
        print(f"  {ric} ({code5}) {name}")

    print("\n[発注タイミング]")
    print("  Day N   15:27-15:29: 5銘柄 引成 (CLO) 新規買い")
    print("  Day N+1 08:55-08:59: 5銘柄 寄成 (OPG) 決済売り")

    return 0


if __name__ == "__main__":
    sys.exit(check_signal())
