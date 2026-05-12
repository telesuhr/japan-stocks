#!/usr/bin/env python3
"""
PEAD (Post-Earnings Announcement Drift) Long戦略 — シグナル自動判定 (v1.0)

Day N 15:25 引け直前に実行:
    python3 signal_check.py [--date YYYY-MM-DD]

【シグナル条件】
  前営業日 (Day N-1) 15:00 以降に決算発表があった銘柄のうち、
  Day N の寄付ギャップが +7% 以上だった銘柄を抽出。

【発注】
  Day N 15:30 引成 Long (1ポジション¥100万)
  最大同時保有数: 10銘柄 (古い順に決済)

【決済】
  - 損切: -5% (intraday low が約定価格 × 0.95 以下)
  - 時間切: Day N+5 (5営業日後) 15:30 引成
  - 利食: なし (上方向は無制限ホールド)

【除外セクター】(バックテストで負のSharpe)
  銀行業 / 食料品 / 金属製品 / サービス業

【バックテスト実績】(2024-01〜2026-05, N=1244)
  net mean +0.88%/トレード, 勝率 49.8%, Sharpe +2.19, t-stat +5.43
  Walk-forward: train 2024 +2.15 / test 2025-26 +2.24
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

# パラメータ（バックテストで決定）
GAP_MIN_PCT      = 7.0
TURNOVER_MIN     = 500_000_000          # 売買代金 5億
SL_PCT           = -5.0
HOLD_DAYS        = 5
MAX_POSITIONS    = 10
POSITION_SIZE    = 1_000_000
EXCLUDE_SECTORS  = ['銀行業', '食料品', '金属製品', 'サービス業']
AC_TIME_THRESHOLD = '15:00:00'   # AC = after close


def latest_trading_day():
    conn = psycopg2.connect(**PG_CONFIG)
    cur = conn.cursor()
    cur.execute("SELECT MAX(date) FROM stocks_daily")
    d = cur.fetchone()[0]
    conn.close()
    return d


def prev_trading_day(target_date: date) -> date:
    """target_date より前の最新営業日を返す"""
    conn = psycopg2.connect(**PG_CONFIG)
    cur = conn.cursor()
    cur.execute("""
        SELECT MAX(date) FROM stocks_daily
        WHERE date < %s
    """, (target_date,))
    d = cur.fetchone()[0]
    conn.close()
    return d


def fetch_signals(target_date: date) -> pd.DataFrame:
    """
    target_date (= 決算翌日 = T+1) の寄付ギャップ判定。
    判定基準:
      - 前営業日 (T) 15:00 以降に決算発表
      - target_date の寄付 / 前営業日終値 - 1 >= +7%
      - target_date の売買代金 >= 5億 (実取引可能ライン)
      - 除外セクターに該当しない
    """
    prev_d = prev_trading_day(target_date)
    if prev_d is None:
        return pd.DataFrame()

    excludes = "','".join(EXCLUDE_SECTORS)
    sql = f"""
    WITH ac_events AS (
        -- 前営業日(T)に引け後発表(15:00以降)した銘柄。同日複数開示は最初の本決算系を採用
        SELECT DISTINCT ON (f.code) f.code, f.disc_date, f.disc_time, f.doc_type
        FROM fin_summary f
        WHERE f.disc_date = '{prev_d}'
          AND f.disc_time >= '{AC_TIME_THRESHOLD}'
        ORDER BY f.code,
                 -- 本決算 > 四半期 > 修正 > その他 の優先順位（簡易）
                 CASE
                   WHEN f.doc_type LIKE 'FY%' THEN 1
                   WHEN f.doc_type LIKE '3Q%' THEN 2
                   WHEN f.doc_type LIKE '2Q%' THEN 3
                   WHEN f.doc_type LIKE '1Q%' THEN 4
                   WHEN f.doc_type LIKE 'Earn%' THEN 5
                   WHEN f.doc_type LIKE 'Div%' THEN 6
                   ELSE 9
                 END,
                 f.disc_time
    ),
    today AS (
        -- target_date (T+1) の OHLC
        SELECT d.code, s.name_ja, s.sector33_nm AS sector,
               d.adj_open, d.adj_close, d.adj_high, d.adj_low,
               d.turnover_value, d.adj_volume
        FROM stocks_daily d
        JOIN symbol_master s ON s.code5 = d.code
        WHERE d.date = '{target_date}' AND s.market = '0111'
          AND s.sector33_nm NOT IN ('{excludes}')
    ),
    prev_close AS (
        -- 前営業日(T) の終値
        SELECT code, adj_close AS t_close
        FROM stocks_daily WHERE date = '{prev_d}'
    )
    SELECT a.code, t.name_ja, t.sector, a.doc_type, a.disc_time,
           pc.t_close,
           t.adj_open  AS tp1_open,
           t.adj_close AS tp1_close,
           t.turnover_value AS tp1_tv,
           ROUND(((t.adj_open / NULLIF(pc.t_close, 0) - 1) * 100)::numeric, 2) AS gap_pct
    FROM ac_events a
    JOIN today t  ON t.code = a.code
    JOIN prev_close pc ON pc.code = a.code
    WHERE t.adj_open / pc.t_close >= {1 + GAP_MIN_PCT/100}
      AND t.turnover_value >= {TURNOVER_MIN}
    ORDER BY gap_pct DESC
    """
    conn = psycopg2.connect(**PG_CONFIG)
    df = pd.read_sql(sql, conn)
    conn.close()
    return df


def load_open_positions():
    """signals_log.csv から現在進行中のポジションを取得"""
    log = HERE / 'signals_log.csv'
    if not log.exists():
        return []
    df = pd.read_csv(log)
    if df.empty:
        return []
    df = df[df['selected'] == 'Y'].copy()
    # 5営業日以内のものだけ「進行中」と見なす（簡易判定）
    today = pd.Timestamp.today().normalize()
    df['signal_date'] = pd.to_datetime(df['signal_date'])
    df['days_since'] = (today - df['signal_date']).dt.days
    open_pos = df[df['days_since'] <= HOLD_DAYS + 2]
    return open_pos['code'].tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', help='判定基準日 (T+1日, YYYY-MM-DD)')
    parser.add_argument('--no-log', action='store_true', help='ログ追記しない')
    args = parser.parse_args()

    target = datetime.strptime(args.date, '%Y-%m-%d').date() if args.date else latest_trading_day()

    print(f"\n{'=' * 75}")
    print(f"  PEAD Long 戦略 — シグナル判定 ({target})")
    print(f"{'=' * 75}")
    print(f"\n  判定条件:")
    print(f"    - 前営業日 ({prev_trading_day(target)}) 15:00以降の決算発表")
    print(f"    - 当日 寄付ギャップ ≥ +{GAP_MIN_PCT}%")
    print(f"    - 当日 売買代金 ≥ {TURNOVER_MIN/1e8:.0f}億円")
    print(f"    - 除外セクター: {', '.join(EXCLUDE_SECTORS)}")

    sig = fetch_signals(target)

    if sig.empty:
        print(f"\n  ❌ シグナルなし — 本日はノートレード")
        print(f"\n  判定: SKIP")
        return

    print(f"\n  ✅ シグナル発生: {len(sig)}銘柄")
    print(f"\n  {'銘柄':<26}  {'セクター':<10}  {'発表時刻':<6}  "
          f"{'ギャップ':>7}  {'引け値':>10}  {'売買代金(億)':>11}")
    print(f"  " + "-" * 85)
    for _, r in sig.iterrows():
        gap = float(r['gap_pct'])
        tv_oku = float(r['tp1_tv']) / 1e8
        cl = float(r['tp1_close'])
        flag = " 🔥" if gap >= 15 else (" ★" if gap >= 10 else "")
        print(f"  {r['name_ja']:<26}  {r['sector']:<10}  {str(r['disc_time'])[:5]}  "
              f"{gap:>+6.2f}%  {cl:>10,.1f}  {tv_oku:>10.1f}億{flag}")

    # 現在の保有数チェック
    open_pos_codes = load_open_positions()
    n_open = len(open_pos_codes)
    n_available = MAX_POSITIONS - n_open
    print(f"\n  現在の保有銘柄数: {n_open} / 上限 {MAX_POSITIONS}")
    print(f"  追加可能枠: {n_available}")

    # 重複除外
    sig_new = sig[~sig['code'].isin(open_pos_codes)]
    if len(sig_new) < len(sig):
        print(f"  → 既保有銘柄を除外: {len(sig)-len(sig_new)}件")

    # 採択
    n_take = min(len(sig_new), n_available)
    if n_take == 0:
        print(f"\n  ⚠️ 保有枠なし or 全銘柄既保有 → SKIP")
        return

    selected = sig_new.head(n_take)
    print(f"\n  ─ 採択: {n_take}銘柄 (ギャップ大きい順) ─")
    print(f"  Day N 15:30 引成 Long × ¥{POSITION_SIZE:,}/銘柄")
    print(f"  → Day N+{HOLD_DAYS} 15:30 引成決済 (SL -{abs(SL_PCT)}%)")
    total = 0
    for _, r in selected.iterrows():
        sl_price = float(r['tp1_close']) * (1 + SL_PCT/100)
        print(f"    [BUY] {r['name_ja']:<24} 引成 ¥{POSITION_SIZE:,}  "
              f"(SL: ¥{sl_price:>.1f})")
        total += POSITION_SIZE
    print(f"\n  必要追加資金: ¥{total:,}")
    print(f"\n  判定: GO ({n_take}銘柄 Long)")

    # ログ追記
    if not args.no_log:
        log_path = HERE / 'signals_log.csv'
        log_exists = log_path.exists()
        with open(log_path, 'a', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            if not log_exists:
                w.writerow(['signal_date', 'code', 'name', 'sector', 'doc_type',
                            'disc_time', 'gap_pct', 'tp1_close', 'turnover_oku',
                            'selected', 'sl_price', 'exit_target_date'])
            target_exit = target + timedelta(days=HOLD_DAYS + 2)  # 営業日近似
            for _, r in sig.iterrows():
                sel = 'Y' if r['code'] in selected['code'].values else 'N'
                sl_price = float(r['tp1_close']) * (1 + SL_PCT/100)
                w.writerow([
                    target, r['code'], r['name_ja'], r['sector'], r['doc_type'],
                    r['disc_time'], f"{float(r['gap_pct']):.2f}",
                    f"{float(r['tp1_close']):.1f}",
                    f"{float(r['tp1_tv'])/1e8:.1f}",
                    sel, f"{sl_price:.1f}",
                    target_exit.strftime('%Y-%m-%d'),
                ])
        print(f"\n  ログ追記: {log_path}")


if __name__ == '__main__':
    main()
