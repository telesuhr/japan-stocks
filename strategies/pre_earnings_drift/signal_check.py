#!/usr/bin/env python3
"""
Pre-Earnings Drift (PED) Long戦略 — シグナル自動判定 (v1.0)

Day N 15:25 引け直前に実行:
    python3 signal_check.py [--date YYYY-MM-DD]

【戦略】
  決算発表"前"の期待感ドリフトを捕まえる:
  - 5営業日後 (T+5) に本決算予定 → 翌日 (T+1) 寄成 Long、決算前日 (T+4) 引成 Exit
  - 3営業日後 (T+3) に 2Q/3Q 決算予定 → 翌日 (T+1) 寄成 Long、決算前日 (T+2) 引成 Exit

【シグナル条件】
  - earnings_calendar に該当日の決算予定あり
  - fq ∈ {'本決算','第２四半期','第３四半期'} (1Q決算は除外)
  - 当日 (T) 売買代金 ≥ 5億円
  - 除外セクター: 医薬品 / 陸運業 (バックテストで負Sharpe)
  - プライム市場のみ

【発注】
  Day N+1 09:00 寄成 Long (1ポジション ¥100万)
  最大同時保有数: 15銘柄 (決算ピーク期は満杯になる)

【決済】
  - 決算発表前日 (T-1) 15:30 引成 Sell
  - SL: なし (バックテストで悪化を確認)

【バックテスト実績】(2024-01〜2026-05, N=3959)
  net mean +0.90%/トレード, 勝率 60.1%, Sharpe +2.07, t-stat +18.24
  Walk-forward: train 2024 +1.56 / test 2025-26 +2.33
"""
import sys
import csv
import argparse
from datetime import date, datetime, timedelta
from pathlib import Path
import psycopg2
import pandas as pd

PG_CONFIG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}
HERE = Path(__file__).parent

# パラメータ（バックテスト最適値）
TURNOVER_MIN     = 500_000_000
MAX_POSITIONS    = 15
POSITION_SIZE    = 1_000_000
EXCLUDE_SECTORS  = ['医薬品', '陸運業']

# 決算タイプ別エントリーオフセット (発表日からの営業日数)
# 例: 5日後発表 → 今日エントリー → 4日後に発表前日Exit (T-1)
DOC_TYPE_RULES = {
    '本決算':    {'lead_days': 5, 'label': 'FY (本決算)'},
    '第２四半期': {'lead_days': 3, 'label': '2Q決算'},
    '第３四半期': {'lead_days': 3, 'label': '3Q決算'},
    # '第１四半期' is excluded - backtest Sharpe < 0.6
    # '第４四半期' is rare (1 record)
}


def latest_trading_day():
    conn = psycopg2.connect(**PG_CONFIG)
    cur = conn.cursor()
    cur.execute("SELECT MAX(date) FROM stocks_daily")
    d = cur.fetchone()[0]
    conn.close()
    return d


def next_business_day(date_val, n=1):
    """次のN営業日 (日本: 月-金、簡易)"""
    d = date_val
    cnt = 0
    while cnt < n:
        d += timedelta(days=1)
        if d.weekday() < 5:  # 月-金
            cnt += 1
    return d


def fetch_signals(target_date: date) -> pd.DataFrame:
    """
    target_date 引け時点での判定:
      - 5営業日後に本決算予定の銘柄
      - 3営業日後に 2Q/3Q決算予定の銘柄
    """
    target_5d = next_business_day(target_date, 5)
    target_3d = next_business_day(target_date, 3)
    excludes = "','".join(EXCLUDE_SECTORS)

    sql = f"""
    WITH cal_5d AS (
        -- 5営業日後に本決算
        SELECT ec.code, ec.co_name, ec.fq, ec.date AS earn_date,
               '本決算' AS doc_class, 5 AS lead_days
        FROM earnings_calendar ec
        WHERE ec.date = '{target_5d}'
          AND ec.fq = '本決算'
          AND ec.section = 'プライム'
    ),
    cal_3d AS (
        -- 3営業日後に四半期決算 (2Q/3Q のみ)
        SELECT ec.code, ec.co_name, ec.fq, ec.date AS earn_date,
               ec.fq AS doc_class, 3 AS lead_days
        FROM earnings_calendar ec
        WHERE ec.date = '{target_3d}'
          AND ec.fq IN ('第２四半期', '第３四半期')
          AND ec.section = 'プライム'
    ),
    cal_union AS (
        SELECT * FROM cal_5d
        UNION ALL
        SELECT * FROM cal_3d
    ),
    today AS (
        -- 当日 (T) の終値・売買代金
        SELECT d.code, s.name_ja, s.sector33_nm AS sector,
               d.adj_close, d.turnover_value
        FROM stocks_daily d
        JOIN symbol_master s ON s.code5 = d.code
        WHERE d.date = '{target_date}' AND s.market = '0111'
          AND s.sector33_nm NOT IN ('{excludes}')
    )
    SELECT c.code, t.name_ja, t.sector, c.fq, c.earn_date, c.lead_days,
           t.adj_close AS t_close,
           t.turnover_value AS t_tv
    FROM cal_union c
    JOIN today t ON t.code = c.code
    WHERE t.turnover_value >= {TURNOVER_MIN}
    ORDER BY c.lead_days, t.turnover_value DESC
    """
    conn = psycopg2.connect(**PG_CONFIG)
    df = pd.read_sql(sql, conn)
    conn.close()
    return df


def load_open_positions():
    """signals_log.csv から進行中のポジションを取得"""
    log = HERE / 'signals_log.csv'
    if not log.exists():
        return []
    df = pd.read_csv(log)
    if df.empty:
        return []
    df = df[df['selected'] == 'Y'].copy()
    df['signal_date'] = pd.to_datetime(df['signal_date'])
    df['earn_date']   = pd.to_datetime(df['earn_date'])
    today = pd.Timestamp.today().normalize()
    # 決算日まで保有中
    open_pos = df[df['earn_date'] > today]
    return open_pos['code'].tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', help='判定基準日 (T = 今日, YYYY-MM-DD)')
    parser.add_argument('--no-log', action='store_true')
    args = parser.parse_args()

    target = datetime.strptime(args.date, '%Y-%m-%d').date() if args.date else latest_trading_day()

    print(f"\n{'=' * 75}")
    print(f"  Pre-Earnings Drift (PED) Long 戦略 — シグナル判定 ({target})")
    print(f"{'=' * 75}")
    print(f"\n  判定条件:")
    print(f"    本決算 → 5営業日後 ({next_business_day(target, 5)}) 予定銘柄")
    print(f"    2Q/3Q → 3営業日後 ({next_business_day(target, 3)}) 予定銘柄")
    print(f"    流動性: 売買代金 ≥ {TURNOVER_MIN/1e8:.0f}億")
    print(f"    除外セクター: {', '.join(EXCLUDE_SECTORS)}")

    sig = fetch_signals(target)

    if sig.empty:
        print(f"\n  ❌ シグナルなし — 翌日エントリーなし")
        print(f"\n  判定: SKIP")
        return

    print(f"\n  ✅ シグナル発生: {len(sig)}銘柄")
    print(f"\n  {'銘柄':<24}  {'セクター':<10}  {'fq':<10}  {'決算日':<12}  "
          f"{'保有日':>3}  {'当日終値':>10}  {'売買代金(億)':>11}")
    print(f"  " + "-" * 95)
    for _, r in sig.iterrows():
        cl = float(r['t_close'])
        tv_oku = float(r['t_tv']) / 1e8
        print(f"  {r['name_ja']:<24}  {r['sector']:<10}  {r['fq']:<10}  "
              f"{str(r['earn_date'])[:10]:<12}  {int(r['lead_days'])-1:>3}日  "
              f"{cl:>10,.1f}  {tv_oku:>10.1f}億")

    # 既保有除外
    open_codes = load_open_positions()
    n_open = len(open_codes)
    n_available = MAX_POSITIONS - n_open
    print(f"\n  現在の保有: {n_open} / 上限 {MAX_POSITIONS}, 追加可能枠: {n_available}")

    sig_new = sig[~sig['code'].isin(open_codes)]
    n_take = min(len(sig_new), n_available)
    if n_take == 0:
        print(f"\n  ⚠️ 枠なし or 全既保有 → SKIP")
        return

    # 売買代金降順で採択
    selected = sig_new.sort_values('t_tv', ascending=False).head(n_take)
    print(f"\n  ─ 採択: {n_take}銘柄 (売買代金降順) ─")
    print(f"  Day N+1 09:00 寄成 Long × ¥{POSITION_SIZE:,}/銘柄")
    print(f"  → 決算前日 (T-1) 15:30 引成 Exit")
    total = 0
    for _, r in selected.iterrows():
        hold_days = int(r['lead_days']) - 1
        exit_d = next_business_day(target, hold_days)
        print(f"    [BUY] {r['name_ja']:<22}  決算: {str(r['earn_date'])[:10]}  "
              f"Exit予定: {exit_d}")
        total += POSITION_SIZE
    print(f"\n  必要追加資金: ¥{total:,}")
    print(f"\n  判定: GO ({n_take}銘柄 Long)")

    # ログ
    if not args.no_log:
        log_path = HERE / 'signals_log.csv'
        log_exists = log_path.exists()
        with open(log_path, 'a', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            if not log_exists:
                w.writerow(['signal_date','code','name','sector','fq','earn_date',
                            'lead_days','t_close','turnover_oku','selected'])
            for _, r in sig.iterrows():
                sel = 'Y' if r['code'] in selected['code'].values else 'N'
                w.writerow([target, r['code'], r['name_ja'], r['sector'], r['fq'],
                            r['earn_date'], r['lead_days'],
                            f"{float(r['t_close']):.1f}",
                            f"{float(r['t_tv'])/1e8:.1f}",
                            sel])
        print(f"\n  ログ追記: {log_path}")


if __name__ == '__main__':
    main()
