#!/usr/bin/env python3
"""
nonferrous_lme_link — シグナルチェック

毎営業日 08:55 JST 以降に実行:
  python3 signal_check.py
オプション:
  --asof YYYY-MM-DD   特定日のシグナルを再現
  --verify-db         DB接続テストのみ
  --threshold N       LME ON 変化閾値 (bps, デフォルト: 80 と 150 の両方)
"""
import argparse
import sys
from datetime import datetime, date, timedelta
import psycopg2
import pandas as pd
import numpy as np

PG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}

# 対象 8 銘柄 (バックテストで全銘柄プラス寄与)
SYMBOLS = {
    "5711.T": "三菱マテリアル",
    "5706.T": "三井金属",
    "5713.T": "住友金属鉱山 ★最強連動 (Sharpe 11.2)",
    "5714.T": "DOWA",
    "5016.T": "JX 金属",
    "5801.T": "古河電工",
    "5802.T": "住友電工 (連動弱、最小ロット推奨)",
    "5803.T": "フジクラ",
}

LME_SYMBOL = "CMCU3"
TIER1 = 150  # bps; |ON|>150 → 引け まで保有 (Sharpe 6.5+)
TIER2 = 80   # bps; |ON|>80  → 11:00 決済    (Sharpe 2.6+)


def load_lme(asof_date: date):
    """asof_date 当日 08:55 までの直近 LME と、24h 前の直近 LME を取得"""
    conn = psycopg2.connect(**PG)
    df = pd.read_sql(
        f"SELECT timestamp, close FROM intraday_data WHERE symbol='{LME_SYMBOL}' "
        f"AND timestamp >= '{asof_date - timedelta(days=4)}' "
        f"ORDER BY timestamp", conn)
    conn.close()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['jst'] = df['timestamp'] + pd.Timedelta(hours=9)
    df = df.dropna(subset=['close']).set_index('jst').sort_index()

    cutoff_now = pd.Timestamp(asof_date) + pd.Timedelta(hours=8, minutes=55)
    cutoff_prev = cutoff_now - pd.Timedelta(days=1)
    cur = df[df.index <= cutoff_now]
    prev = df[df.index <= cutoff_prev]
    if not len(cur) or not len(prev):
        return None
    return {
        "now_ts": cur.index[-1],
        "now_px": float(cur['close'].iloc[-1]),
        "prev_ts": prev.index[-1],
        "prev_px": float(prev['close'].iloc[-1]),
        "on_bps": (cur['close'].iloc[-1]/prev['close'].iloc[-1] - 1) * 10000,
    }


def verify_db():
    print("DB接続テスト ...")
    try:
        conn = psycopg2.connect(**PG)
        c = conn.cursor()
        c.execute(f"SELECT MAX(timestamp) FROM intraday_data WHERE symbol='{LME_SYMBOL}'")
        r = c.fetchone()
        print(f"  ✓ PostgreSQL: LME 最新 = {r[0]}")
        for s in SYMBOLS:
            c.execute(f"SELECT MAX(timestamp) FROM intraday_data WHERE symbol='{s}'")
            r = c.fetchone()
            print(f"  ✓ {s}: {r[0]}")
        conn.close()
    except Exception as e:
        print(f"  ✗ FAIL: {e}")
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", default=None, help="YYYY-MM-DD (デフォルト=今日)")
    ap.add_argument("--verify-db", action="store_true")
    args = ap.parse_args()

    if args.verify_db:
        sys.exit(0 if verify_db() else 1)

    asof = datetime.strptime(args.asof, "%Y-%m-%d").date() if args.asof else date.today()
    print("=" * 80)
    print(f"nonferrous_lme_link シグナル  asof={asof}")
    print("=" * 80)

    info = load_lme(asof)
    if info is None:
        print("✗ LME データ取得失敗 (DB接続またはデータ未到着)")
        sys.exit(1)

    on = info["on_bps"]
    print(f"\nLME 銅 ({LME_SYMBOL}):")
    print(f"  現在 (JST {info['now_ts']}): {info['now_px']:.2f}")
    print(f"  24h前 (JST {info['prev_ts']}): {info['prev_px']:.2f}")
    print(f"  ON 変化: {on:+.1f} bps")

    direction = "LONG" if on > 0 else "SHORT"

    print("\n" + "-" * 80)
    if abs(on) >= TIER1:
        print(f"🔔 【TIER1 シグナル発動】 |ON| ≥ {TIER1} bps")
        print(f"   方向: {direction}  (寄成エントリ → 大引け決済)")
        print(f"   実績期待: mean +109 bps, Sharpe 6.5, WR 65%")
        for s, name in SYMBOLS.items():
            print(f"     {s} ({name})")
    elif abs(on) >= TIER2:
        print(f"🟡 【TIER2 シグナル発動】 {TIER2} ≤ |ON| < {TIER1} bps")
        print(f"   方向: {direction}  (寄成エントリ → 11:00 決済)")
        print(f"   実績期待: mean +38 bps, Sharpe 2.6, WR 59%")
        for s, name in SYMBOLS.items():
            print(f"     {s} ({name})")
    else:
        print(f"⚪ シグナルなし (|ON|={abs(on):.1f}bps < {TIER2})")
        print(f"   B戦略 (Disp 11:30) のみ午後に実施")

    print("-" * 80)
    print("\n■ ポジションサイジング")
    print("  各シグナル発動日: 8銘柄に資金均等配分 (1/8 ずつ)")
    print("  推奨総資金 ¥2,000-3,000 万 (各銘柄 ¥250-375 万)")
    print("  TIER1 と TIER2 が同日重なる場合は TIER1 ルール優先 (大引け決済)")
    print("\n■ 個別銘柄ウェイト調整 (任意)")
    print("  最強: 5713 住友金鉱 (Sharpe 11)、5803 フジクラ (Sh 9)")
    print("  弱め: 5802 住友電工 (Sh 1.9) — 半量推奨")


if __name__ == "__main__":
    main()
