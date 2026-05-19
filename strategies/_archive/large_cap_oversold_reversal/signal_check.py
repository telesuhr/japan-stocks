#!/usr/bin/env python3
"""
大型株 過売り反発 (Large Cap Oversold Reversal) Long戦略 — シグナル自動判定 (v1.0)

Day N 15:25 引け直前に実行:
    python3 signal_check.py [--date YYYY-MM-DD]

【ユニバース】
  TOPIX Core30 (31銘柄) + Large70 (69銘柄) = 100銘柄
  小型株のリスク (流動性消失・上場廃止) を完全排除

【戦略】
  20日下落率 -20% 以上 の銘柄を翌寄りLong、5営業日後の引けで決済
  超大型・大型株の急落後の反発を狙う

【シグナル条件】
  - TOPIX scale_cat ∈ {Core30, Large70}
  - 20日下落率 < -20%  (主シグナル) または MA25乖離 < -15% (補助)
  - 当日売買代金 ≥ 30億円 (大型株なら通常クリア)

【発注】
  Day N+1 09:00 寄成 Long (1ポジション ¥100万)
  最大同時保有数: 8銘柄

【決済】
  - Day N+5 (5営業日後) 15:30 引成 Sell
  - SL: なし (バックテストで最強)
       オプションで -10% (Sharpe +2.62 → +2.92 だがDD抑制)

【バックテスト実績】(2024-01〜2026-05, N=642)
  net mean +3.39%/トレード, 勝率 68.8%, Sharpe +2.92, t-stat +10.73
  Walk-forward: train 2024 +4.76 / test 2025-26 +1.44

【Core30 vs Large70】
  Core30 2026: Sharpe +0.83 (プラス維持！)
  Large70 2026: Sharpe -0.69
  → 弱化局面では Core30 優先採択を推奨
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
RET_20D_MAX        = -20.0       # 20日下落率
MA25_DIST_AUX      = -15.0       # 補助シグナル (MA25乖離)
TURNOVER_MIN       = 3_000_000_000  # 30億 (大型株向けに高めに設定)
HOLD_DAYS          = 5
SL_PCT             = None        # SLなしが最強
MAX_POSITIONS      = 8
POSITION_SIZE      = 1_000_000

# 推奨セクター (Sharpe>2.0)
PREFERRED_SECTORS = ['卸売業','保険業','非鉄金属','銀行業','輸送用機器','電気機器','機械']
EXCLUDE_SECTORS   = ['精密機器']  # バックテストで負Sharpe


def latest_trading_day():
    conn = psycopg2.connect(**PG_CONFIG)
    cur = conn.cursor()
    cur.execute("SELECT MAX(date) FROM stocks_daily")
    d = cur.fetchone()[0]
    conn.close()
    return d


def fetch_signals(target_date: date) -> pd.DataFrame:
    """target_date 引け時点の大型株過売りシグナル"""
    excludes = "','".join(EXCLUDE_SECTORS)
    sql = f"""
    WITH base AS (
        SELECT d.code, s.name_ja, s.sector33_nm AS sector, s.scale_cat,
               d.date, d.adj_close, d.turnover_value
        FROM stocks_daily d
        JOIN symbol_master s ON s.code5 = d.code
        WHERE s.market = '0111'
          AND s.scale_cat IN ('TOPIX Core30', 'TOPIX Large70')
          AND s.sector33_nm NOT IN ('{excludes}')
          AND d.date >= '{target_date}'::date - INTERVAL '45 days'
          AND d.date <= '{target_date}'
          AND d.adj_close > 0
    ),
    ind AS (
        SELECT *,
            AVG(adj_close) OVER (PARTITION BY code ORDER BY date
                                 ROWS BETWEEN 25 PRECEDING AND 1 PRECEDING) AS ma25,
            LAG(adj_close, 20) OVER (PARTITION BY code ORDER BY date) AS close_20d_ago
        FROM base
    )
    SELECT code, name_ja, sector, scale_cat, date,
           adj_close, ma25, close_20d_ago, turnover_value,
           ROUND(((adj_close/NULLIF(ma25,0) - 1) * 100)::numeric, 2) AS dist_ma25,
           ROUND(((adj_close/NULLIF(close_20d_ago,0) - 1) * 100)::numeric, 2) AS ret_20d
    FROM ind
    WHERE date = '{target_date}'
      AND ma25 IS NOT NULL AND close_20d_ago IS NOT NULL
      AND (
           (adj_close/close_20d_ago - 1) * 100 <= {RET_20D_MAX}
        OR (adj_close/ma25 - 1) * 100 <= {MA25_DIST_AUX}
      )
      AND turnover_value >= {TURNOVER_MIN}
    ORDER BY ret_20d ASC
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
    parser.add_argument('--date', help='判定基準日 (YYYY-MM-DD)')
    parser.add_argument('--no-log', action='store_true')
    args = parser.parse_args()

    target = datetime.strptime(args.date, '%Y-%m-%d').date() if args.date else latest_trading_day()

    print(f"\n{'=' * 78}")
    print(f"  大型株 過売り反発 Long 戦略 — シグナル判定 ({target})")
    print(f"{'=' * 78}")
    print(f"\n  ユニバース: TOPIX Core30 + Large70 (100銘柄)")
    print(f"  シグナル条件:")
    print(f"    - 20日下落率 ≤ {RET_20D_MAX}% (主) または MA25乖離 ≤ {MA25_DIST_AUX}%")
    print(f"    - 当日売買代金 ≥ {TURNOVER_MIN/1e8:.0f}億円")
    print(f"    - 除外セクター: {', '.join(EXCLUDE_SECTORS)}")

    sig = fetch_signals(target)

    if sig.empty:
        print(f"\n  ❌ シグナルなし — 翌日エントリーなし")
        print(f"\n  判定: SKIP")
        return

    # フラグ追加
    sig['is_core30']   = (sig['scale_cat'] == 'TOPIX Core30')
    sig['is_preferred']= sig['sector'].isin(PREFERRED_SECTORS)

    print(f"\n  ✅ シグナル発生: {len(sig)}銘柄 (Core30: {sig['is_core30'].sum()}, 推奨セクター: {sig['is_preferred'].sum()})")
    print(f"\n  {'銘柄':<22}  {'規模':<12}  {'セクター':<10}  {'20d↓':>7}  "
          f"{'MA25↓':>7}  {'売買代金(億)':>11}  ⭐推奨")
    print(f"  " + "-" * 95)
    for _, r in sig.iterrows():
        ret20 = float(r['ret_20d'])
        ma25d = float(r['dist_ma25'])
        tv_oku = float(r['turnover_value']) / 1e8
        scale = r['scale_cat'].replace('TOPIX ', '')
        marks = []
        if r['is_core30']: marks.append("Core")
        if r['is_preferred']: marks.append("★")
        if ret20 <= -30: marks.append("🔥")
        mark_str = "/".join(marks)
        print(f"  {r['name_ja']:<22}  {scale:<12}  {r['sector']:<10}  "
              f"{ret20:>+6.2f}%  {ma25d:>+6.2f}%  {tv_oku:>10.1f}億  {mark_str}")

    # 既保有除外
    open_codes = load_open_positions()
    n_open = len(open_codes)
    n_available = MAX_POSITIONS - n_open
    print(f"\n  現在保有: {n_open} / 上限 {MAX_POSITIONS}, 追加可能枠: {n_available}")
    sig_new = sig[~sig['code'].isin(open_codes)].copy()

    # 採択ロジック: Core30 + 推奨セクター優先で、20日下落深い順
    sig_new['priority'] = sig_new['is_core30'].astype(int)*2 + sig_new['is_preferred'].astype(int)
    selected = sig_new.sort_values(['priority','ret_20d'], ascending=[False, True]).head(n_available)

    n_take = len(selected)
    if n_take == 0:
        print(f"\n  ⚠️ 採択可能銘柄なし → SKIP")
        return

    print(f"\n  ─ 採択: {n_take}銘柄 (Core30+推奨セクター優先、下落深い順) ─")
    print(f"  Day N+1 09:00 寄成 Long × ¥{POSITION_SIZE:,}/銘柄")
    print(f"  → Day N+{HOLD_DAYS} 15:30 引成 Exit (SL: なし)")
    total = 0
    for _, r in selected.iterrows():
        scale = r['scale_cat'].replace('TOPIX ', '')
        mark = "★" if r['is_preferred'] else " "
        print(f"    [BUY]{mark} {r['name_ja']:<20}  {scale:<10}  {r['sector']:<10}  "
              f"20d={float(r['ret_20d']):>+6.2f}%")
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
                w.writerow(['signal_date','code','name','sector','scale_cat',
                            'ret_20d','dist_ma25','adj_close','turnover_oku',
                            'is_core30','is_preferred','selected'])
            for _, r in sig.iterrows():
                sel = 'Y' if r['code'] in selected['code'].values else 'N'
                w.writerow([target, r['code'], r['name_ja'], r['sector'], r['scale_cat'],
                            f"{float(r['ret_20d']):.2f}",
                            f"{float(r['dist_ma25']):.2f}",
                            f"{float(r['adj_close']):.1f}",
                            f"{float(r['turnover_value'])/1e8:.1f}",
                            'Y' if r['is_core30'] else 'N',
                            'Y' if r['is_preferred'] else 'N',
                            sel])
        print(f"\n  ログ追記: {log_path}")


if __name__ == '__main__':
    main()
