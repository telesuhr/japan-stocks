#!/usr/bin/env python3
"""
過売り反発 (Oversold MA25 Reversal) Long戦略 — シグナル自動判定 (v1.0)

Day N 15:25 引け直前に実行:
    python3 signal_check.py [--date YYYY-MM-DD]

【戦略】
  MA25 から -20% 以上下に乖離した銘柄を翌寄りLong、5営業日後の引けで決済。
  過売り状態からの平均回帰反発を狙う。

【シグナル条件】
  - プライム市場銘柄
  - 当日終値が25日移動平均の80%以下 (dist_ma25 ≤ -20%)
  - 当日売買代金 ≥ 10億円
  - 推奨セクター: 銀行業 / 輸送用機器 / 機械 / 電気機器 / 化学
  - 非推奨セクター: サービス業 / 情報通信業 (Sharpe < 2.0)

【発注】
  Day N+1 09:00 寄成 Long (1ポジション ¥100万)
  最大同時保有数: 10銘柄 (vol_distance降順)

【決済】
  - Day N+5 (5営業日後) 15:30 引成 Sell
  - SL: -7% (オプション、Sharpe 2.67維持)
       SLなし設定がベスト (Sharpe 3.21) だがDDが大きい
  - TP: なし

【バックテスト実績】(2024-01〜2026-05, N=1,625)
  net mean +3.91%/トレード, 勝率 70.7%, Sharpe +3.21, t-stat +18.71

【⚠️ 注意】
  2026年に弱化傾向 (Sharpe -0.58, N=245)
  → 月次でnet<0が続く場合は要停止検討
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

# パラメータ (バックテスト最適値)
MA25_DIST_MAX    = -20.0          # MA25からの乖離 (これより下)
TURNOVER_MIN     = 1_000_000_000  # 10億
HOLD_DAYS        = 5
SL_PCT           = -7.0           # SLなしが最強だがDD抑制のため
MAX_POSITIONS    = 10
POSITION_SIZE    = 1_000_000

# 推奨セクター (Sharpe>2.0)
PREFERRED_SECTORS = ['銀行業','輸送用機器','機械','電気機器','化学','非鉄金属',
                     'ガラス･土石製品','建設業','精密機器']
EXCLUDE_SECTORS   = ['医薬品']     # 過売りでもリバーサル弱い


def latest_trading_day():
    conn = psycopg2.connect(**PG_CONFIG)
    cur = conn.cursor()
    cur.execute("SELECT MAX(date) FROM stocks_daily")
    d = cur.fetchone()[0]
    conn.close()
    return d


def fetch_signals(target_date: date) -> pd.DataFrame:
    """target_date 引け時点で MA25 から -20% 以上乖離した銘柄を抽出"""
    excludes = "','".join(EXCLUDE_SECTORS)
    sql = f"""
    WITH base AS (
        SELECT d.code, s.name_ja, s.sector33_nm AS sector,
               d.date, d.adj_close, d.turnover_value,
               d.adj_volume
        FROM stocks_daily d
        JOIN symbol_master s ON s.code5 = d.code
        WHERE d.code IN (
              SELECT code5 FROM symbol_master
              WHERE market = '0111' AND sector33_nm NOT IN ('{excludes}')
        )
          AND d.date >= '{target_date}'::date - INTERVAL '45 days'
          AND d.date <= '{target_date}'
          AND d.adj_close > 0
    ),
    ind AS (
        SELECT *,
            AVG(adj_close) OVER (PARTITION BY code ORDER BY date
                                 ROWS BETWEEN 25 PRECEDING AND 1 PRECEDING) AS ma25
        FROM base
    )
    SELECT code, name_ja, sector, date,
           adj_close, ma25, turnover_value,
           ROUND(((adj_close/NULLIF(ma25,0) - 1) * 100)::numeric, 2) AS dist_ma25
    FROM ind
    WHERE date = '{target_date}'
      AND ma25 IS NOT NULL AND ma25 > 0
      AND (adj_close / ma25 - 1) * 100 <= {MA25_DIST_MAX}
      AND turnover_value >= {TURNOVER_MIN}
    ORDER BY dist_ma25 ASC   -- 乖離が深い順
    """
    conn = psycopg2.connect(**PG_CONFIG)
    df = pd.read_sql(sql, conn)
    conn.close()
    return df


def load_open_positions():
    log = HERE / 'signals_log.csv'
    if not log.exists():
        return []
    df = pd.read_csv(log)
    if df.empty: return []
    df = df[df['selected'] == 'Y'].copy()
    today = pd.Timestamp.today().normalize()
    df['signal_date'] = pd.to_datetime(df['signal_date'])
    df['days_since'] = (today - df['signal_date']).dt.days
    open_pos = df[df['days_since'] <= HOLD_DAYS + 2]
    return open_pos['code'].tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', help='判定基準日 (T = 今日, YYYY-MM-DD)')
    parser.add_argument('--no-log', action='store_true')
    args = parser.parse_args()

    target = datetime.strptime(args.date, '%Y-%m-%d').date() if args.date else latest_trading_day()

    print(f"\n{'=' * 75}")
    print(f"  過売り反発 Long 戦略 — シグナル判定 ({target})")
    print(f"{'=' * 75}")
    print(f"\n  判定条件:")
    print(f"    - 25日移動平均からの乖離 ≤ {MA25_DIST_MAX}%")
    print(f"    - 当日売買代金 ≥ {TURNOVER_MIN/1e8:.0f}億円")
    print(f"    - 除外セクター: {', '.join(EXCLUDE_SECTORS)}")

    sig = fetch_signals(target)

    if sig.empty:
        print(f"\n  ❌ シグナルなし — 翌日エントリーなし")
        print(f"\n  判定: SKIP")
        return

    # 推奨/非推奨セクター区分
    sig['is_preferred'] = sig['sector'].isin(PREFERRED_SECTORS)

    print(f"\n  ✅ シグナル発生: {len(sig)}銘柄 (推奨セクター: {sig['is_preferred'].sum()}銘柄)")
    print(f"\n  {'銘柄':<24}  {'セクター':<10}  {'MA25乖離':>9}  "
          f"{'当日終値':>10}  {'売買代金(億)':>11}  推奨")
    print(f"  " + "-" * 85)
    for _, r in sig.iterrows():
        dist = float(r['dist_ma25'])
        tv_oku = float(r['turnover_value']) / 1e8
        cl = float(r['adj_close'])
        mark = "★" if r['is_preferred'] else ""
        flag = " 🔥" if dist <= -30 else (" ★" if dist <= -25 else "")
        print(f"  {r['name_ja']:<24}  {r['sector']:<10}  {dist:>+8.2f}%  "
              f"{cl:>10,.1f}  {tv_oku:>10.1f}億  {mark}{flag}")

    # 既保有除外
    open_codes = load_open_positions()
    n_open = len(open_codes)
    n_available = MAX_POSITIONS - n_open
    print(f"\n  現在保有: {n_open} / 上限 {MAX_POSITIONS}, 追加可能枠: {n_available}")

    sig_new = sig[~sig['code'].isin(open_codes)].copy()

    # 採択ロジック: 推奨セクター優先で、乖離が深い順
    preferred = sig_new[sig_new['is_preferred']].sort_values('dist_ma25')
    others    = sig_new[~sig_new['is_preferred']].sort_values('dist_ma25')
    selected = pd.concat([preferred, others]).head(n_available)

    n_take = len(selected)
    if n_take == 0:
        print(f"\n  ⚠️ 採択可能銘柄なし → SKIP")
        return

    print(f"\n  ─ 採択: {n_take}銘柄 (推奨セクター優先・乖離深い順) ─")
    print(f"  Day N+1 09:00 寄成 Long × ¥{POSITION_SIZE:,}/銘柄")
    print(f"  → Day N+{HOLD_DAYS} 15:30 引成 Exit, SL {SL_PCT}%")
    total = 0
    for _, r in selected.iterrows():
        sl_p = float(r['adj_close']) * (1 + SL_PCT/100)
        mark = "★" if r['is_preferred'] else " "
        print(f"    [BUY]{mark} {r['name_ja']:<22}  {r['sector']:<10}  "
              f"MA25乖離 {float(r['dist_ma25']):>+6.2f}%  (SL目安: ¥{sl_p:.1f})")
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
                w.writerow(['signal_date','code','name','sector','dist_ma25',
                            'adj_close','turnover_oku','is_preferred','selected'])
            for _, r in sig.iterrows():
                sel = 'Y' if r['code'] in selected['code'].values else 'N'
                w.writerow([target, r['code'], r['name_ja'], r['sector'],
                            f"{float(r['dist_ma25']):.2f}",
                            f"{float(r['adj_close']):.1f}",
                            f"{float(r['turnover_value'])/1e8:.1f}",
                            'Y' if r['is_preferred'] else 'N',
                            sel])
        print(f"\n  ログ追記: {log_path}")


if __name__ == '__main__':
    main()
