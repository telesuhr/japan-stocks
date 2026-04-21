#!/usr/bin/env python3
"""
LME銅シグナル自動判定スクリプト

毎朝 06:00-08:30 に実行:
  python3 signal_check.py

出力: 本日エントリーすべきか、スキップすべきかを表示
"""
import sys
import os
from datetime import date, datetime, timedelta
import psycopg2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'analyses', '20260421_common')))
import mdutil as U

THRESHOLD_PCT = 1.0
CORE5 = [
    ("5711.T", "三菱マテリアル"),
    ("6501.T", "日立製作所"),
    ("7011.T", "三菱重工業"),
    ("5016.T", "出光興産"),
    ("4502.T", "武田薬品"),
]
SYMBOL_LME = "CMCU3"


def fetch_lme_last_two_closes():
    """LME銅の直近2営業日の終値を取得"""
    conn = psycopg2.connect(**U.PG_CONFIG)
    q = f"""
        SELECT DATE(timestamp + INTERVAL '9 hours') as jst_date,
               close, timestamp
        FROM intraday_data
        WHERE symbol = '{SYMBOL_LME}'
          AND close IS NOT NULL
        ORDER BY timestamp DESC
        LIMIT 600
    """
    import pandas as pd
    df = pd.read_sql(q, conn)
    conn.close()
    # 日付ごとに最後 (LME終値) を取得
    df = df.groupby('jst_date').agg({'close': 'last', 'timestamp': 'last'}).sort_index()
    return df.tail(5)  # 直近5営業日


def check_signal():
    print("=" * 65)
    print(f"LME銅シグナル判定  {datetime.now().strftime('%Y-%m-%d %H:%M:%S JST')}")
    print("=" * 65)

    today = date.today()
    weekday = today.strftime('%a')

    # ① LME 前日変化率
    try:
        lme = fetch_lme_last_two_closes()
    except Exception as e:
        print(f"❌ LMEデータ取得エラー: {e}")
        return 1

    if len(lme) < 2:
        print("❌ LMEデータ不足 (2営業日分以上が必要)")
        return 1

    latest = lme.iloc[-1]
    prev = lme.iloc[-2]
    change_pct = (latest['close'] / prev['close'] - 1) * 100

    print(f"\n[LME銅 終値]")
    print(f"  前々日 ({lme.index[-2]}): ${prev['close']:.2f}")
    print(f"  前日  ({lme.index[-1]}): ${latest['close']:.2f}")
    print(f"  変化率: {change_pct:+.2f}% (閾値 +{THRESHOLD_PCT:.1f}%)")

    lme_pass = change_pct >= THRESHOLD_PCT
    print(f"  判定: {'✅ 通過' if lme_pass else '❌ 不足'}")

    if not lme_pass:
        print("\n" + "=" * 65)
        print("🚫 本日はシグナル不発 → スキップ")
        print("   skipped_reason=lme_below_threshold")
        print("=" * 65)
        return 0

    # ② 曜日チェック
    print(f"\n[曜日チェック]")
    print(f"  本日: {today} ({weekday})")
    is_thursday = today.weekday() == 3
    print(f"  判定: {'❌ 木曜 (除外)' if is_thursday else '✅ OK'}")

    if is_thursday:
        print("\n" + "=" * 65)
        print("🚫 木曜日のためスキップ (曜日アノマリー Sharpe-1.31)")
        print("   skipped_reason=thursday")
        print("=" * 65)
        return 0

    # ③ シグナル発動
    print("\n" + "=" * 65)
    print("🟢 シグナル発動 — 本日エントリー")
    print("=" * 65)
    print("\n[対象銘柄 各¥1,000万]")
    for code, name in CORE5:
        print(f"  {code} {name}")

    print("\n[次のステップ]")
    print("  1. 各銘柄の前日終値を確認し概算株数を算出")
    print("  2. 08:45-09:00 に 5銘柄 寄成 (成行寄付) 買い発注")
    print("  3. 09:00 約定確認")
    print("  4. 15:25 に 引成 (成行引け) 売り発注")
    print("  5. 15:30 決済後 trade_log.csv に記録")

    print("\n[要確認]")
    print("  □ 各銘柄の本日±1営業日が決算発表日でないか")
    print("  □ 日経平均の寄付が -2%超のギャップダウンでないか")
    print("  □ 該当銘柄にストップ安・売買停止がないか")

    return 0


if __name__ == "__main__":
    sys.exit(check_signal())
