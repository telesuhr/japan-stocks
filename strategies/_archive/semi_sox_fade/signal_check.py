#!/usr/bin/env python3
"""
semi_sox_fade — シグナルチェック

毎営業日 08:30 JST 以降に実行 (NY 引けは JST 06:00):
  python3 signal_check.py
"""
import argparse, sys
from datetime import datetime, date, timedelta
import pymysql, pandas as pd, numpy as np

MARIA = dict(host="100.92.181.92", port=3306, user="rfnews",
             password="Bleach@924", database="refinitiv_news")
LAN_FALLBACK = dict(host="192.168.0.250", port=3306, user="rfnews",
                    password="Bleach@924", database="refinitiv_news")

SYMBOLS = {
    "8035.T": "東京エレクトロン",
    "6857.T": "アドバンテスト",
    "6963.T": "ローム",
    "6526.T": "ソシオネクスト",
    "6525.T": "KOKUSAI ELECTRIC",
}
THRESH = 30  # bps


def _connect():
    try:
        return pymysql.connect(**MARIA, connect_timeout=5)
    except Exception:
        return pymysql.connect(**LAN_FALLBACK, connect_timeout=5)


def get_sox_change(asof: date):
    """asof 当日朝に効く SOX 変化 (asof より前の最新 NY セッション)"""
    conn = _connect()
    df = pd.read_sql(
        "SELECT trade_date, close FROM daily_data WHERE symbol='.SOX' "
        "ORDER BY trade_date DESC LIMIT 5", conn)
    conn.close()
    df['trade_date'] = pd.to_datetime(df['trade_date']).dt.date
    df = df.sort_values('trade_date').reset_index(drop=True)
    df = df[df['trade_date'] < asof]
    if len(df) < 2:
        return None
    df['chg_bps'] = df['close'].pct_change() * 10000
    last = df.iloc[-1]
    return {"date": last['trade_date'], "close": float(last['close']),
            "chg_bps": float(last['chg_bps'])}


def verify_db():
    print("DB接続テスト ...")
    try:
        conn = _connect()
        c = conn.cursor()
        c.execute("SELECT MAX(trade_date) FROM daily_data WHERE symbol='.SOX'")
        print(f"  ✓ SOX 最新: {c.fetchone()[0]}")
        conn.close()
        return True
    except Exception as e:
        print(f"  ✗ FAIL: {e}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", default=None)
    ap.add_argument("--verify-db", action="store_true")
    args = ap.parse_args()

    if args.verify_db:
        sys.exit(0 if verify_db() else 1)

    asof = datetime.strptime(args.asof, "%Y-%m-%d").date() if args.asof else date.today()
    print("=" * 80)
    print(f"semi_sox_fade シグナル  asof={asof}")
    print("=" * 80)

    info = get_sox_change(asof)
    if info is None:
        print("✗ SOX データ不足")
        sys.exit(1)

    chg = info['chg_bps']
    print(f"\n.SOX 直近セッション ({info['date']}):")
    print(f"  終値: {info['close']:.2f}")
    print(f"  前日比: {chg:+.1f} bps")

    print("\n" + "-" * 80)
    if abs(chg) >= THRESH:
        # フェード = SOX 強い → ショート / 弱い → ロング
        direction = "SHORT" if chg > 0 else "LONG"
        print(f"🔔 シグナル発動 |SOX|≥{THRESH} bps")
        print(f"   方向 (フェード): {direction}  (寄成エントリ → 大引け決済)")
        print(f"   実績: mean +28.5 bps, Sharpe 1.71, WR 52.7%, t 3.86")
        for s, name in SYMBOLS.items():
            print(f"     {s} ({name})")
    else:
        print(f"⚪ シグナルなし (|SOX|={abs(chg):.1f}bps < {THRESH})")

    print("-" * 80)
    print("\n■ ポジションサイジング")
    print("  各銘柄に総資金 1/5 を均等配分")
    print("  推奨総資金 ¥1,500-2,500 万 (各銘柄 ¥300-500 万)")
    print("\n■ 銘柄別期待 Sharpe (バックテスト)")
    print("  6526 ソシオネクスト  +2.16  ← 最強")
    print("  8035 TEL              +1.92")
    print("  6525 KOKUSAI          +1.74")
    print("  6963 ローム           +1.53")
    print("  6857 アドバンテスト   +1.44")


if __name__ == "__main__":
    main()
