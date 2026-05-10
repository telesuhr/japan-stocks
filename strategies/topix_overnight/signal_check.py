#!/usr/bin/env python3
"""
TOPIX夜間ギャップ シグナル自動判定スクリプト (v1.0)

Day N 09:05 頃に実行:
  python3 signal_check.py

シグナル = TOPIX (.TOPX) 前日終値 → Day N 09:00 変化率 ≥ +0.3%
発動 → Day N 15:30 引成 Long (CORE5 各¥1,000万)
決済 → Day N+1 09:00 寄成
"""
import sys
import os
from datetime import date, datetime
import psycopg2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'analyses', '20260421_common')))
import mdutil as U

THRESHOLD_PCT = 0.3
SYMBOL_TOPIX = ".TOPX"
CORE5 = [
    ("5711.T", "三菱マテリアル"),
    ("6501.T", "日立製作所"),
    ("7011.T", "三菱重工業"),
    ("5016.T", "出光興産"),
    ("4502.T", "武田薬品"),
]


def fetch_topix_gap(target_date):
    """
    TOPIX の前日終値 → target_date 09:00近辺の変化率を計算。
    """
    import pandas as pd
    conn = psycopg2.connect(**U.PG_CONFIG)
    q = f"""
        SELECT timestamp, open, close,
               (timestamp + INTERVAL '9 hours') AS jst
        FROM intraday_data
        WHERE symbol = '{SYMBOL_TOPIX}'
          AND close IS NOT NULL
          AND DATE(timestamp + INTERVAL '9 hours') <= '{target_date}'
        ORDER BY timestamp DESC
        LIMIT 3000
    """
    df = pd.read_sql(q, conn)
    conn.close()
    if df.empty:
        return None
    df['jst'] = pd.to_datetime(df['jst'])
    df['jst_date'] = df['jst'].dt.date
    df = df.sort_values('jst')

    # 前日データ
    prev_dates = sorted([d for d in df['jst_date'].unique() if d < target_date])
    if not prev_dates:
        return None
    prev_date = prev_dates[-1]
    prev_day = df[df['jst_date'] == prev_date]
    if prev_day.empty:
        return None
    prev_close = float(prev_day['close'].iloc[-1])

    # 当日 09:00近辺 (最初のバー)
    today = df[df['jst_date'] == target_date]
    morning = today[today['jst'].dt.hour <= 9]
    if morning.empty:
        return None
    today_open = float(morning['close'].iloc[0])

    return {
        'prev_date': prev_date,
        'prev_close': prev_close,
        'today_open': today_open,
        'gap_pct': (today_open / prev_close - 1) * 100,
    }


def check_signal():
    print("=" * 70)
    print(f"TOPIX夜間ギャップ シグナル判定")
    print(f"実行時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S JST')}")
    print("=" * 70)

    today = date.today()
    weekday = today.strftime('%a')
    print(f"\n本日 (エントリー候補日 Day N): {today} ({weekday})")

    try:
        g = fetch_topix_gap(today)
    except Exception as e:
        print(f"❌ TOPIXデータ取得エラー: {e}")
        return 1

    if g is None:
        print("❌ TOPIXデータ不足 (前日終値 or 本日9:00バー取得不可)")
        return 1

    print(f"\n[TOPIX夜間変化率]")
    print(f"  前日 ({g['prev_date']}) 終値: {g['prev_close']:.2f}")
    print(f"  本日 09:00近辺:            {g['today_open']:.2f}")
    print(f"  変化率: {g['gap_pct']:+.3f}% (閾値 +{THRESHOLD_PCT:.1f}%)")

    passed = g['gap_pct'] >= THRESHOLD_PCT
    print(f"  判定: {'✅ 通過' if passed else '❌ 不足'}")

    if not passed:
        print("\n" + "=" * 70)
        print("🚫 本日はシグナル不発 → スキップ")
        print("   skipped_reason=topix_below_threshold")
        print("=" * 70)
        return 0

    # 曜日チェック
    is_thursday = today.weekday() == 3
    print(f"\n[曜日チェック] Day N: {weekday}")
    print(f"  判定: {'❌ 木曜除外' if is_thursday else '✅ OK'}")
    if is_thursday:
        print("\n🚫 木曜エントリーのためスキップ")
        print("   skipped_reason=thursday")
        return 0

    print("\n" + "=" * 70)
    print("🟢 シグナル発動候補 — Day N 15:30 引成 Long (予定)")
    print("=" * 70)
    print("\n[対象銘柄 各¥1,000万]")
    for code, name in CORE5:
        print(f"  {code} {name}")

    print("\n[重要: 最終確認事項]")
    print("  □ lme_on_copper シグナルが発動していないか")
    print("     → 発動していれば本戦略は取消 (重複回避)")
    print("     → 15:15にlme_on_copper signal_check.pyを実行して再確認")
    print("  □ Day N / Day N+1 の決算発表銘柄なし")
    print("  □ Day N+1 の配当落ち該当なし")
    print("  □ FOMC/米雇用統計/日銀会合前でない (該当ならサイズ半減)")
    print("  □ 日経225先物・CME急落中でない")

    print("\n[発注タイミング]")
    print("  Day N   15:27-15:29: 5銘柄 引成 (CLO) 新規買い")
    print("  Day N+1 08:55-08:59: 5銘柄 寄成 (OPG) 決済売り")

    return 0


if __name__ == "__main__":
    sys.exit(check_signal())
