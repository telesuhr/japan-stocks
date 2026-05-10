#!/usr/bin/env python3
"""
銀行株 出来高吸収逆張り戦略 — シグナル自動判定スクリプト (v1.0)

Day N 15:35 引け後に実行:
    python3 signal_check.py [--date YYYY-MM-DD]

シグナル条件 (Day N 引け時点):
    1. ホワイトリスト22銘柄のうち、当日に
        - 出来高 ≥ 過去20日平均の1.5倍
        - 当日リターン (始値→終値) < 0%（陰線）
        - 売買代金 ≥ 10億円
        が成立した銘柄を抽出

発動 → Day N+1 09:00 寄成Long (各¥100万 / 最大3銘柄同時保有)
決済 → Day N+5 (5営業日後) 引成 — SL/TPなし

バックテスト実績:
    N=771, net mean +1.67%, 勝率62.3%, Sharpe 1.84, PF 2.13
    Walk-forward: train(2024) Sharpe 2.29 → test(2025-26) Sharpe 1.59
"""
import sys
import csv
import argparse
from datetime import date, datetime
from pathlib import Path
import psycopg2
import pandas as pd

PG_CONFIG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}
HERE = Path(__file__).parent

# パラメータ（バックテストでの最適値）
VOL_RATIO_MIN  = 1.5
DAY_RET_MAX    = 0.0       # 陰線条件
TURNOVER_MIN   = 1_000_000_000   # 10億円
HOLD_DAYS      = 5
MAX_POSITIONS  = 3         # 同時保有数上限
POSITION_SIZE  = 1_000_000 # 1ポジション100万円


def load_whitelist():
    """ホワイトリスト銘柄をロード"""
    wl_path = HERE / "whitelist.csv"
    df = pd.read_csv(wl_path, dtype={'code5': str})
    return df


def fetch_signals(target_date: date, codes: list[str]) -> pd.DataFrame:
    """target_date の引け時点でシグナルが成立した銘柄を返す"""
    conn = psycopg2.connect(**PG_CONFIG)
    placeholders = ','.join("'" + c + "'" for c in codes)
    sql = f"""
    WITH base AS (
        SELECT d.code, s.name_ja, d.date,
               d.adj_open, d.adj_close, d.adj_volume,
               d.turnover_value
        FROM stocks_daily d
        JOIN symbol_master s ON s.code5 = d.code
        WHERE d.code IN ({placeholders})
          AND d.date >= '{target_date}'::date - INTERVAL '40 days'
          AND d.date <= '{target_date}'
          AND d.adj_close > 0 AND d.adj_open > 0
    ),
    ind AS (
        SELECT *,
            AVG(adj_volume) OVER (
                PARTITION BY code ORDER BY date
                ROWS BETWEEN 21 PRECEDING AND 2 PRECEDING
            ) AS vol_ma20,
            (adj_close/adj_open - 1)*100 AS day_ret
        FROM base
    )
    SELECT code, name_ja, date, adj_open, adj_close, adj_volume,
           vol_ma20, day_ret, turnover_value,
           ROUND((adj_volume / NULLIF(vol_ma20, 0))::numeric, 2) AS vol_ratio
    FROM ind
    WHERE date = '{target_date}'
      AND vol_ma20 IS NOT NULL AND vol_ma20 > 0
      AND adj_volume / vol_ma20 >= {VOL_RATIO_MIN}
      AND day_ret < {DAY_RET_MAX}
      AND turnover_value >= {TURNOVER_MIN}
    ORDER BY vol_ratio DESC
    """
    df = pd.read_sql(sql, conn)
    conn.close()
    return df


def latest_trading_day():
    """DBに登録された最新の営業日を取得"""
    conn = psycopg2.connect(**PG_CONFIG)
    cur = conn.cursor()
    cur.execute("SELECT MAX(date) FROM stocks_daily")
    d = cur.fetchone()[0]
    conn.close()
    return d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', help='判定基準日 (YYYY-MM-DD)。未指定なら最新営業日')
    parser.add_argument('--max-show', type=int, default=10, help='表示するシグナル数の上限')
    args = parser.parse_args()

    target = datetime.strptime(args.date, '%Y-%m-%d').date() if args.date else latest_trading_day()
    print(f"\n{'=' * 70}")
    print(f"  銀行株 出来高吸収逆張り戦略 — シグナル判定 ({target})")
    print(f"{'=' * 70}")

    # ホワイトリスト
    wl = load_whitelist()
    print(f"\n  ホワイトリスト: {len(wl)}銘柄")

    # シグナル抽出
    sig = fetch_signals(target, wl['code5'].tolist())

    if sig.empty:
        print(f"\n  ❌ シグナルなし — Day N+1 はノートレード")
        print(f"\n  判定: SKIP")
        return

    # 結果表示
    print(f"\n  ✅ シグナル発生: {len(sig)}銘柄")
    print(f"\n  {'銘柄':<24}  {'出来高倍率':>10}  {'当日Ret':>8}  {'売買代金(億)':>12}  {'当日終値':>10}")
    print(f"  " + "-" * 75)
    for _, r in sig.iterrows():
        tv_oku = float(r['turnover_value']) / 1e8
        print(f"  {r['name_ja']:<24}  {float(r['vol_ratio']):>9.2f}x  "
              f"{float(r['day_ret']):>+7.2f}%  {tv_oku:>11.1f}億  "
              f"{float(r['adj_close']):>10,.1f}")

    # 同時保有制約の適用
    n_to_take = min(len(sig), MAX_POSITIONS)
    selected = sig.head(n_to_take)   # vol_ratio降順で上位N

    print(f"\n  ─ 採択 (最大{MAX_POSITIONS}銘柄、vol_ratio降順) ─")
    print(f"  Day N+1 09:00 寄成Long、Day N+{HOLD_DAYS} 引成Exit")
    total_size = 0
    for _, r in selected.iterrows():
        print(f"    [BUY] {r['name_ja']} ({r['code']}) — ¥{POSITION_SIZE:,}")
        total_size += POSITION_SIZE
    print(f"\n  必要資金: ¥{total_size:,}")
    print(f"\n  判定: GO ({n_to_take}銘柄 Long)")

    # ログ追記
    log_path = HERE / "signals_log.csv"
    log_exists = log_path.exists()
    with open(log_path, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if not log_exists:
            w.writerow(['signal_date', 'code', 'name', 'vol_ratio', 'day_ret',
                        'turnover_oku', 'sig_close', 'selected', 'entry_target_date'])
        for _, r in sig.iterrows():
            sel = 'Y' if r['code'] in selected['code'].values else 'N'
            w.writerow([
                target, r['code'], r['name_ja'],
                f"{float(r['vol_ratio']):.2f}",
                f"{float(r['day_ret']):.2f}",
                f"{float(r['turnover_value'])/1e8:.1f}",
                f"{float(r['adj_close']):.1f}",
                sel,
                'next_business_day_open',
            ])
    print(f"\n  ログ追記: {log_path}")


if __name__ == '__main__':
    main()
