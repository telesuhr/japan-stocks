#!/usr/bin/env python3
"""
.SOX急落 → TOPIX日中Short シグナル判定スクリプト (v1.1)

Day N 朝 07:00-08:30 JST に実行:
  python3 signal_check.py               # シグナル判定
  python3 signal_check.py --verify-db   # MariaDB 接続 + daily_data テーブル存在確認のみ

主シグナル: 前日 .SOX 日次リターン ≤ -2.0%
推奨AND条件: 前日 ESc1 日次リターン ≤ -1.0%
除外: VIX < 15 (低ボラ) or VIX ≥ 35 (パニック) or 火曜日
発動 → Day N 09:00 寄成 Short (1306.T)
決済 → Day N 15:30 引成 買戻し

依存:
  - NAS MariaDB 100.92.181.92 (fallback: 192.168.0.250) database=refinitiv_news
  - テーブル: daily_data (symbol, trade_date, close)
  - 必要シンボル: .SOX, ESc1, NQc1, VXc1
"""
import argparse
import sys
from datetime import date, datetime, timedelta
import pandas as pd

try:
    import pymysql
except ImportError:
    print("❌ pymysql がインストールされていません: pip install pymysql")
    sys.exit(1)

MARIA = dict(host='100.92.181.92', port=3306, user='rfnews',
             password='Bleach@924', database='refinitiv_news')
LAN_FALLBACK = '192.168.0.250'

SOX_THRESHOLD = -2.0  # %
ES_THRESHOLD_PREFERRED = -1.0  # %
VIX_LOW = 15.0
VIX_PANIC = 35.0


def _connect_maria():
    """MariaDB 接続 (Tailscale → LAN fallback)"""
    try:
        return pymysql.connect(**MARIA, connect_timeout=5)
    except Exception:
        return pymysql.connect(**{**MARIA, 'host': LAN_FALLBACK}, connect_timeout=5)


def verify_db():
    """MariaDB 接続 + daily_data テーブル + 必要シンボルの存在確認"""
    print("=" * 70)
    print("sox_overnight_short — MariaDB 依存性検証")
    print("=" * 70)
    try:
        conn = _connect_maria()
    except Exception as e:
        print(f"❌ MariaDB 接続失敗: {e}")
        print(f"   接続先候補: {MARIA['host']} / {LAN_FALLBACK}")
        return 1
    cur = conn.cursor()
    print(f"✅ 接続成功 → {MARIA['database']}")
    cur.execute("SHOW TABLES LIKE 'daily_data'")
    if not cur.fetchone():
        print("❌ テーブル 'daily_data' が存在しません")
        cur.close(); conn.close(); return 1
    print("✅ テーブル 'daily_data' 存在")
    required = ['.SOX', 'ESc1', 'NQc1', 'VXc1']
    for sym in required:
        cur.execute("SELECT MAX(trade_date), COUNT(*) FROM daily_data WHERE symbol=%s", (sym,))
        latest, n = cur.fetchone()
        if not n:
            print(f"❌ {sym}: データなし")
        else:
            print(f"✅ {sym}: N={n}  latest={latest}")
    cur.close(); conn.close()
    print("=" * 70)
    return 0


def fetch_daily(symbol, days_back=10):
    """NAS MariaDB から symbol の直近N営業日分取得"""
    conn = _connect_maria()

    cutoff = (date.today() - timedelta(days=days_back * 2)).strftime('%Y-%m-%d')
    q = f"""
        SELECT trade_date, close
        FROM daily_data
        WHERE symbol = %s AND trade_date >= %s
        ORDER BY trade_date DESC LIMIT {days_back}
    """
    df = pd.read_sql(q, conn, params=[symbol, cutoff])
    conn.close()
    if df.empty:
        return None
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df = df.sort_values('trade_date').reset_index(drop=True)
    df['ret_pct'] = df['close'].pct_change() * 100
    return df


def latest_return(df, name):
    if df is None or len(df) < 2:
        return None, None
    last = df.iloc[-1]
    return float(last['ret_pct']), last['trade_date'].date()


def check_signal():
    print("=" * 70)
    print(".SOX急落 → TOPIX日中Short シグナル判定")
    print(f"実行時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S JST')}")
    print("=" * 70)

    today = date.today()
    weekday = today.strftime('%a')
    print(f"\n本日 (エントリー候補日 Day N): {today} ({weekday})")

    # 各指数取得
    try:
        sox = fetch_daily('.SOX')
        es = fetch_daily('ESc1')
        nq = fetch_daily('NQc1')
        vix = fetch_daily('VXc1')
    except Exception as e:
        print(f"❌ MariaDB接続エラー: {e}")
        return 1

    sox_ret, sox_date = latest_return(sox, '.SOX')
    es_ret, _ = latest_return(es, 'ESc1')
    nq_ret, _ = latest_return(nq, 'NQc1')
    vix_close = float(vix['close'].iloc[-1]) if vix is not None and len(vix) > 0 else None

    if sox_ret is None:
        print("❌ .SOX データ取得失敗")
        return 1

    print(f"\n[前日米国市場 ({sox_date})]")
    print(f"  .SOX  Ret = {sox_ret:+.2f}% (閾値 ≤ {SOX_THRESHOLD:.1f}%)")
    if es_ret is not None:
        print(f"  ESc1  Ret = {es_ret:+.2f}% (推奨 ≤ {ES_THRESHOLD_PREFERRED:.1f}%)")
    if nq_ret is not None:
        print(f"  NQc1  Ret = {nq_ret:+.2f}%")
    if vix_close is not None:
        print(f"  VIX   終値 = {vix_close:.2f}")

    # ① 主シグナル
    if sox_ret > SOX_THRESHOLD:
        print(f"\n❌ .SOX 閾値未達 ({sox_ret:+.2f}% > {SOX_THRESHOLD:.1f}%)")
        print("   skipped_reason=signal_not_fired")
        return 0
    print(f"\n✅ ① .SOX 主シグナル発動 ({sox_ret:+.2f}%)")

    # ② VIX レジーム
    if vix_close is not None:
        if vix_close < VIX_LOW:
            print(f"❌ VIX 低すぎ ({vix_close:.2f} < {VIX_LOW})  → 低ボラ時は不発")
            print("   skipped_reason=vix_too_low")
            return 0
        if vix_close >= VIX_PANIC:
            print(f"❌ VIX パニック圏 ({vix_close:.2f} ≥ {VIX_PANIC})  → 反発リスク大")
            print("   skipped_reason=vix_panic")
            return 0
        print(f"✅ ② VIX レジーム OK ({vix_close:.2f})")

    # ③ ESc1 AND条件 (推奨)
    es_ok = es_ret is not None and es_ret <= ES_THRESHOLD_PREFERRED
    if es_ok:
        print(f"✅ ③ ESc1 AND条件 成立 ({es_ret:+.2f}%)  → 強シグナル (Sharpe+2.83)")
        signal_strength = "STRONG"
    else:
        print(f"⚠️  ③ ESc1 AND条件 不成立 ({es_ret:+.2f}% > {ES_THRESHOLD_PREFERRED:.1f}%)")
        print("    → 標準シグナル (Sharpe+2.11)、サイズ半減推奨")
        signal_strength = "NORMAL"

    # ④ 曜日チェック
    if today.weekday() == 1:  # Tuesday
        print("❌ ④ 火曜エントリー → スキップ推奨 (バックテスト劣化)")
        print("   skipped_reason=tuesday_entry")
        return 0
    print(f"✅ ④ 曜日 OK ({weekday})")

    print("\n" + "=" * 70)
    print("🟢 シグナル発動候補 — Day N 09:00 寄成 Short 1306.T")
    print(f"   強度: {signal_strength}")
    print("=" * 70)

    print("\n[最終確認事項 Day N 08:55 まで]")
    print("  □ 日経225先物 (大阪夜間) が既に -3% 超の急落中でないか")
    print("  □ topix_overnight (Long) シグナル発動 → 本戦略を優先、Long側キャンセル")
    print("  □ FOMC/日銀会合/日本重要指標 当日発表なし")
    print("  □ 1306.T 配当落ち日でない (3月末・9月末注意)")
    print("  □ 日銀緊急声明・ETF買入再開観測なし")

    print("\n[発注タイミング]")
    print("  Day N 08:57-08:59: 1306.T 寄成 (OPG) Short 新規")
    print("  Day N 15:27-15:29: 1306.T 引成 (CLO) 買戻し 決済")

    if signal_strength == "STRONG":
        print("\n[推奨サイズ]")
        print("  通常サイズ ¥1,000-3,000万")
    else:
        print("\n[推奨サイズ]")
        print("  半減 ¥500-1,500万 (ESc1 AND条件不成立のため)")

    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-db", action="store_true",
                    help="MariaDB 接続 + daily_data テーブル存在確認のみ実行")
    args = ap.parse_args()
    if args.verify_db:
        sys.exit(verify_db())
    sys.exit(check_signal())
