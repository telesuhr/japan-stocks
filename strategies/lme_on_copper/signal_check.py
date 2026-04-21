#!/usr/bin/env python3
"""
LME銅 東京時間シグナル 自動判定スクリプト (v1.0)

Day N 15:25 頃に実行:
  python3 signal_check.py

シグナル = LME銅(CMCU3) 東京時間変化率 (JST 9:00 → 15:25) ≥ +1.0%
発動 → Day N 15:30 引成 Long (CORE5 各¥1,000万)
決済 → Day N+1 09:00 寄成
"""
import sys
import os
from datetime import date, datetime, time
import psycopg2
import pandas as pd

PG_CONFIG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}

THRESHOLD_PCT = 1.0
SYMBOL_LME = "CMCU3"
CORE5 = [
    ("5711.T", "三菱マテリアル"),
    ("6501.T", "日立製作所"),
    ("7011.T", "三菱重工業"),
    ("5016.T", "出光興産"),
    ("4502.T", "武田薬品"),
]

# 夏時間 (BST): 3月最終日曜〜10月最終日曜 → 東京オープン JST 9:00 = UTC 0:00
# 冬時間 (GMT): それ以外              → 東京オープン JST 10:00 = UTC 1:00
BST_PERIODS = [
    (date(2024, 3, 31), date(2024, 10, 27)),
    (date(2025, 3, 30), date(2025, 10, 26)),
    (date(2026, 3, 29), date(2026, 10, 25)),
]


def is_bst(d: date) -> bool:
    for start, end in BST_PERIODS:
        if start <= d <= end:
            return True
    return False


def fetch_lme_tokyo_change(target_date: date) -> dict | None:
    """
    target_date の LME銅 東京時間変化率を計算する。
    東京オープン (夏:9:00 / 冬:10:00) → 15:25 の変化率。
    """
    conn = psycopg2.connect(**PG_CONFIG)
    q = f"""
        SELECT timestamp, open, high, low, close
        FROM intraday_data
        WHERE symbol = '{SYMBOL_LME}'
          AND DATE(timestamp + INTERVAL '9 hours') = '{target_date}'
          AND close IS NOT NULL
        ORDER BY timestamp
    """
    df = pd.read_sql(q, conn)
    conn.close()

    if df.empty:
        return None

    df["jst"] = pd.to_datetime(df["timestamp"]) + pd.Timedelta(hours=9)
    df = df.sort_values("jst").set_index("jst")

    bst = is_bst(target_date)
    open_hour = 9 if bst else 10

    # 東京オープン価格
    open_bar = df[df.index.hour == open_hour]
    if open_bar.empty:
        return None
    tokyo_open = float(open_bar["close"].iloc[0])
    open_time = open_bar.index[0]

    # 15:25 直前の最新値
    lte_1525 = df[df.index <= df.index[0].replace(hour=15, minute=25)]
    if lte_1525.empty:
        return None
    current = float(lte_1525["close"].iloc[-1])
    current_time = lte_1525.index[-1]

    change_pct = (current / tokyo_open - 1) * 100

    return {
        "tokyo_open": tokyo_open,
        "open_time": open_time,
        "current": current,
        "current_time": current_time,
        "change_pct": change_pct,
        "bst": bst,
    }


def check_signal():
    print("=" * 70)
    print("LME銅 東京時間シグナル判定")
    print(f"実行時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S JST')}")
    print("=" * 70)

    today = date.today()
    weekday = today.strftime("%A")
    print(f"\n本日 (エントリー候補日 Day N): {today} ({weekday})")

    # 木曜チェック (先にやる)
    is_thursday = today.weekday() == 3
    if is_thursday:
        print("\n🚫 木曜日のためスキップ確定")
        print("   skipped_reason=thursday")
        print("   (木曜発動日のSharpe = -1.31 — 除外ルール)")
        return 0

    # LMEデータ取得
    try:
        g = fetch_lme_tokyo_change(today)
    except Exception as e:
        print(f"\n❌ LMEデータ取得エラー: {e}")
        return 1

    if g is None:
        print("\n❌ LMEデータ不足 (東京オープンまたは15:25バーが取得できない)")
        return 1

    bst_label = "夏時間(BST)" if g["bst"] else "冬時間(GMT)"
    print(f"\n[LME銅 東京時間変化率] ({bst_label})")
    print(f"  東京オープン ({g['open_time'].strftime('%H:%M')}): {g['tokyo_open']:,.1f} USD/t")
    print(f"  現在値      ({g['current_time'].strftime('%H:%M')}): {g['current']:,.1f} USD/t")
    print(f"  変化率: {g['change_pct']:+.3f}% (閾値 +{THRESHOLD_PCT:.1f}%)")

    passed = g["change_pct"] >= THRESHOLD_PCT
    print(f"  判定: {'✅ 通過' if passed else '❌ 不足'}")

    if not passed:
        print("\n" + "=" * 70)
        print("🚫 本日はシグナル不発 → スキップ")
        print("   skipped_reason=lme_below_threshold")
        print("=" * 70)
        return 0

    print("\n" + "=" * 70)
    print("🟢 シグナル発動候補 — Day N 15:30 引成 Long (予定)")
    print("=" * 70)
    print("\n[対象銘柄 各¥1,000万]")
    for code, name in CORE5:
        print(f"  {code} {name}")

    print("\n[重要: 最終確認事項]")
    print("  □ Day N または Day N+1 に決算発表銘柄がないか")
    print("  □ FOMC/米雇用統計/日銀会合が夜間に控えていないか")
    print("     → 控えている場合はサイズ半減 (各¥500万) を検討")
    print("  □ 日経225先物が -2% 超の急落中でないか")
    print("     → 急落中はスキップ (skipped_reason=panic)")

    print("\n[発注タイミング]")
    print("  Day N   15:27-15:29: 5銘柄 引成 (CLO) 新規買い")
    print("  Day N+1 08:55-08:59: 5銘柄 寄成 (OPG) 決済売り")

    return 0


if __name__ == "__main__":
    sys.exit(check_signal())
