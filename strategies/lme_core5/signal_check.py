#!/usr/bin/env python3
"""
LME銅 東京セッション シグナル自動判定スクリプト (v1.1)

本戦略は Day N 大引け Long エントリー → Day N+1 寄付 決済 のONホールド戦略。
シグナル = LME銅 東京セッション中 (JST 9:00 → 15:25) の変化率。

Day N の 15:15-15:25 に実行:
  python3 signal_check.py

出力: 本日大引けでエントリーすべきか、スキップすべきかを表示
"""
import sys
import os
from datetime import date, datetime, timedelta
import psycopg2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'analyses', '20260421_common')))
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


def fetch_lme_tokyo_session(target_date):
    """
    LME銅の target_date 東京セッション (JST 9:00 開始 → 15:25 直近) を取得。

    Returns:
      dict { 'open_9': price, 'close_1525': price } or None
    """
    import pandas as pd
    conn = psycopg2.connect(**U.PG_CONFIG)
    # JST対応: timestamp + 9h でJST
    q = f"""
        SELECT timestamp, open, close,
               (timestamp + INTERVAL '9 hours') AS jst
        FROM intraday_data
        WHERE symbol = '{SYMBOL_LME}'
          AND close IS NOT NULL
          AND DATE(timestamp + INTERVAL '9 hours') = '{target_date}'
        ORDER BY timestamp
    """
    df = pd.read_sql(q, conn)
    conn.close()
    if df.empty:
        return None
    df['jst'] = pd.to_datetime(df['jst'])
    df['jst_hour'] = df['jst'].dt.hour
    df['jst_minute'] = df['jst'].dt.minute

    # JST 9:00 以降で最初のバー
    morning = df[df['jst_hour'] >= 9]
    if morning.empty:
        return None
    open_9 = morning.iloc[0]['open']

    # JST 15:25 以前で最後のバー (15:25 を超えない)
    mask_1525 = (df['jst_hour'] < 15) | ((df['jst_hour'] == 15) & (df['jst_minute'] <= 25))
    afternoon = df[(df['jst_hour'] >= 9) & mask_1525]
    if afternoon.empty:
        return None
    close_1525 = afternoon.iloc[-1]['close']

    return {
        'open_9': float(open_9),
        'close_1525': float(close_1525),
        'open_jst': morning.iloc[0]['jst'],
        'close_jst': afternoon.iloc[-1]['jst'],
    }


def check_signal():
    print("=" * 70)
    print(f"LME銅 東京セッション シグナル判定 (v1.1 ONホールド版)")
    print(f"実行時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S JST')}")
    print("=" * 70)

    today = date.today()
    weekday = today.strftime('%a')
    print(f"\n本日 (エントリー日 Day N): {today} ({weekday})")

    # ① LME 東京セッション変化率
    print(f"\n[LME銅 東京セッション変化率]")
    try:
        lme = fetch_lme_tokyo_session(today)
    except Exception as e:
        print(f"❌ LMEデータ取得エラー: {e}")
        return 1

    if lme is None:
        print("❌ 本日のLMEデータ不足 (JST 9:00-15:25 の1分足が必要)")
        print("   → シグナル判定不可。手動で Reuters/Bloomberg 確認して判断。")
        return 1

    change_pct = (lme['close_1525'] / lme['open_9'] - 1) * 100
    print(f"  JST 9:00  ({lme['open_jst']}): ${lme['open_9']:.2f}")
    print(f"  JST 15:25 ({lme['close_jst']}): ${lme['close_1525']:.2f}")
    print(f"  変化率: {change_pct:+.2f}% (閾値 +{THRESHOLD_PCT:.1f}%)")

    lme_pass = change_pct >= THRESHOLD_PCT
    print(f"  判定: {'✅ 通過' if lme_pass else '❌ 不足'}")

    if not lme_pass:
        print("\n" + "=" * 70)
        print("🚫 本日はシグナル不発 → 引成エントリー見送り")
        print("   skipped_reason=lme_below_threshold")
        print("=" * 70)
        return 0

    # ② 曜日チェック (エントリー日=Day N が木曜でないこと)
    print(f"\n[曜日チェック (エントリー日)]")
    is_thursday = today.weekday() == 3
    print(f"  本日 Day N: {weekday}")
    print(f"  判定: {'❌ 木曜 (除外: 翌金曜寄付決済でSharpe-1.31)' if is_thursday else '✅ OK'}")

    if is_thursday:
        print("\n" + "=" * 70)
        print("🚫 木曜エントリーのためスキップ")
        print("   skipped_reason=thursday")
        print("=" * 70)
        return 0

    # ③ シグナル発動
    print("\n" + "=" * 70)
    print("🟢 シグナル発動 — Day N 大引け引成Longエントリー")
    print("=" * 70)
    print("\n[対象銘柄 各¥1,000万]")
    for code, name in CORE5:
        print(f"  {code} {name}")

    print("\n[発注タイミング]")
    print(f"  Day N  15:27-15:29: 5銘柄 引成 (CLO) 新規買い発注")
    print(f"  Day N  15:30:       大引け約定")
    print(f"  Day N+1 08:55-08:59: 5銘柄 寄成 (OPG) 決済売り発注")
    print(f"  Day N+1 09:00:       寄付で決済 → P&L確定")

    print("\n[要手動確認 (自動チェック外)]")
    print("  □ Day N または Day N+1 が各銘柄の決算発表日でないか")
    print("  □ Day N+1 が配当落ち日でないか")
    print("  □ FOMC/米雇用統計/日銀会合が Day N+1 朝までに控えていないか")
    print("     → 控えている場合は各¥500万にサイズ半減")
    print("  □ 日経225先物・CMEが急落中 (-2%超) でないか")
    print("  □ 対象銘柄に特別気配・売買停止がないか")

    return 0


if __name__ == "__main__":
    sys.exit(check_signal())
