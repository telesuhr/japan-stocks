#!/usr/bin/env python3
"""
引け板寄せ下落リバウンド戦略 — シグナル判定スクリプト (v0.1 候補/事後判定)

【重要】本スクリプトは引け後の事後判定 (日中足の 15:24 と 15:30 から close_jump を計算)。
ペーパートレード記録・検証用。
ライブ執行には プレクロージング(15:25-30)の気配監視 + 買いMOC が必須
(README.md「実約定の前提条件」参照)。

Day N 15:35 以降に実行:
    python3 signal_check.py [--date YYYY-MM-DD] [--top 200] [--threshold -50]

シグナル条件 (Day N 引け):
    close_jump = 引値(15:30板寄せ) / 15:24連続最終値 − 1 ≤ threshold(既定 −50bps)
発動 → Day N 引け MOC買い (実運用はプレクロージング気配で前倒し判断)
決済 → Day N+1 09:00〜09:15

バックテスト実績 (2024-11〜2026-05, 流動性上位200, N=1,821):
    gross +21.7bps/泊, 勝率56.5%, net Sharpe 2.00(往復10bps, IS1.82/OOS2.15)
"""
import sys
import csv
import argparse
from datetime import datetime
from pathlib import Path
import psycopg2
import pandas as pd

PG_CONFIG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}
HERE = Path(__file__).parent

THRESHOLD_BPS = -50.0   # close_jump 閾値
TOP_N = 200             # 流動性上位ユニバース
EXIT_NOTE = "翌09:00〜09:15 決済"


def latest_trading_day():
    conn = psycopg2.connect(**PG_CONFIG)
    cur = conn.cursor()
    cur.execute("SELECT MAX(ts::date) FROM stocks_intraday")
    d = cur.fetchone()[0]
    conn.close()
    return d


def liquid_universe(target_date, top_n):
    conn = psycopg2.connect(**PG_CONFIG)
    df = pd.read_sql(f"""
        SELECT code FROM stocks_daily
        WHERE date <= '{target_date}' AND date >= '{target_date}'::date - INTERVAL '400 days'
          AND turnover_value > 0
        GROUP BY code ORDER BY AVG(turnover_value) DESC LIMIT {top_n}
    """, conn)
    conn.close()
    return df['code'].tolist()


def fetch_jumps(target_date, codes):
    """target_date の 15:24 close と 15:30 close から close_jump を計算"""
    conn = psycopg2.connect(**PG_CONFIG)
    ph = ','.join("'" + c + "'" for c in codes)
    sql = f"""
        WITH bars AS (
            SELECT i.code, i.ts::time AS t, i.close
            FROM stocks_intraday i
            WHERE i.code IN ({ph})
              AND i.ts::date = '{target_date}'
              AND i.ts::time IN ('15:24:00','15:30:00')
        )
        SELECT b1.code,
               b24.close AS c24, b30.close AS c30,
               (b30.close / NULLIF(b24.close,0) - 1) * 10000 AS jump_bps
        FROM (SELECT DISTINCT code FROM bars) b1
        JOIN bars b24 ON b24.code = b1.code AND b24.t = '15:24:00'
        JOIN bars b30 ON b30.code = b1.code AND b30.t = '15:30:00'
        WHERE b24.close > 0 AND b30.close > 0
    """
    df = pd.read_sql(sql, conn)
    conn.close()
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', help='判定基準日 (YYYY-MM-DD)。未指定なら最新営業日')
    ap.add_argument('--top', type=int, default=TOP_N, help='流動性上位ユニバース数')
    ap.add_argument('--threshold', type=float, default=THRESHOLD_BPS, help='close_jump閾値(bps)')
    args = ap.parse_args()

    target = datetime.strptime(args.date, '%Y-%m-%d').date() if args.date else latest_trading_day()
    print(f"\n{'=' * 70}")
    print(f"  引け板寄せ下落リバウンド — シグナル判定 ({target})")
    print(f"{'=' * 70}")
    print(f"  ※事後判定 (ペーパー検証用)。ライブはプレクロージング気配監視が必須")

    codes = liquid_universe(target, args.top)
    print(f"\n  流動性上位ユニバース: {len(codes)}銘柄")

    jumps = fetch_jumps(target, codes)
    if jumps.empty:
        print(f"\n  ❌ 当日の 15:24/15:30 板寄せデータなし (非営業日 or 新制度前?)")
        print(f"\n  判定: SKIP")
        return

    sig = jumps[jumps['jump_bps'] <= args.threshold].sort_values('jump_bps')
    if sig.empty:
        print(f"\n  ❌ jump≤{args.threshold:.0f}bps の銘柄なし — ノートレード")
        print(f"\n  判定: SKIP")
        return

    print(f"\n  ✅ シグナル発生: {len(sig)}銘柄 (引けで{abs(args.threshold):.0f}bps以上下落)")
    print(f"\n  {'コード':<8} {'15:24値':>10} {'引値(15:30)':>12} {'jump(bps)':>10}")
    print(f"  " + "-" * 44)
    for _, r in sig.head(30).iterrows():
        print(f"  {r['code']:<8} {float(r['c24']):>10,.1f} {float(r['c30']):>12,.1f} {float(r['jump_bps']):>10.1f}")

    print(f"\n  ─ 執行 ─")
    print(f"  Day N 引け MOC買い (各銘柄均等)、{EXIT_NOTE}")
    print(f"  ※実運用: プレクロージング(15:25-30)気配で jump≤{args.threshold:.0f}bps を検出し買いMOC投入")
    print(f"\n  判定: GO ({len(sig)}銘柄 Long候補)")

    # ログ追記
    log_path = HERE / "signals_log.csv"
    log_exists = log_path.exists()
    with open(log_path, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if not log_exists:
            w.writerow(['signal_date', 'code', 'c24', 'c30', 'jump_bps', 'exit_target'])
        for _, r in sig.iterrows():
            w.writerow([target, r['code'], f"{float(r['c24']):.1f}",
                        f"{float(r['c30']):.1f}", f"{float(r['jump_bps']):.1f}",
                        'next_day_0900_0915'])
    print(f"\n  ログ追記: {log_path}")


if __name__ == '__main__':
    main()
