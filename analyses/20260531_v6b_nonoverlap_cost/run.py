"""
V6b スコア 非重複20日 + コスト込み 実力検証

前研究 (20260531_v6_universe50_ls) で V6b = r20_adj + 0.5*d75 の
日次オーバーラップ L/S Sharpe=4.65 を得たが、日次エントリーは保有期間20日と
重複するため系列相関で Sharpe が過大評価される。

本スクリプトは:
  1. 非重複20日リバランス (重複なしの独立サンプル) で Sharpe を正しく計測
  2. 全20位相 (offset 0〜19) で平均し開始日依存を除去
  3. 売買コスト + 空売り調達料を控除した純損益で評価
  4. コスト感度分析 (片側 5/10/15bps)

ユニバース: 50銘柄 (auKabu PORTFOLIO_ALL)
期間: IS 2022-01-01〜2023-12-31 / OOS 2024-01-01〜2026-05-31
L/S: top8 Long / bottom8 Short, 保有20日
"""
from __future__ import annotations

import os
import sys
import psycopg2
import pandas as pd
import numpy as np
from scipy.stats import ttest_1samp

sys.stdout.reconfigure(line_buffering=True)
DB_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost/market_data")

CODES4 = [
    '5713','5711','5706','5714','5016','5801','5802','5803',
    '8035','6857','6920','6146','7735','4063','3436','7741','6963','6526','9984','4062','6723','285A','6525',
    '8306','8316','8411','7011','7013','7012','6503','6501','6758','7203','7267','8058','8031',
    '6981','6762','6971','6976','4004','8766','1605','6861','6954','9432','7974','9983','6098','9433',
]
CODES5 = [c + '0' for c in CODES4]
CODE_LIST = ','.join(f"'{c}'" for c in CODES5)

IS_START  = pd.Timestamp("2022-01-01")
IS_END    = pd.Timestamp("2023-12-31")
OOS_START = pd.Timestamp("2024-01-01")
HOLD = 20
N_SIDE = 8
PERIODS_PER_YEAR = 252 / HOLD  # ≈12.6 (非重複20日)
ANN = np.sqrt(PERIODS_PER_YEAR)


def fetch(sql):
    conn = psycopg2.connect(DB_URL)
    df = pd.read_sql(sql, conn)
    conn.close()
    return df


def sharpe_period(rets: np.ndarray) -> float:
    r = rets[~np.isnan(rets)]
    if len(r) < 5 or r.std(ddof=1) == 0:
        return float("nan")
    return float(r.mean() / r.std(ddof=1) * ANN)


print("=" * 76)
print("V6b 非重複20日 + コスト込み 実力検証")
print("=" * 76)
print("\n[データ取得中]")

prices = fetch(f"""
    SELECT LEFT(code,4) c, date, adj_close::float ac
    FROM stocks_daily WHERE code IN ({CODE_LIST})
      AND date >= '2020-07-01' AND adj_close > 0 ORDER BY code, date
""")
prices['date'] = pd.to_datetime(prices['date'])

# コード別 (date_int, ac) 配列を事前計算 (date は int64ナノ秒で統一)
by_code = {}
for code, g in prices.groupby('c'):
    g = g.sort_values('date')
    by_code[code] = (g['date'].values.astype('datetime64[ns]').astype(np.int64), g['ac'].values)

# 共通取引日カレンダー (全銘柄和集合)
all_dates = np.array(sorted(prices['date'].unique())).astype('datetime64[ns]')
all_dates_int = all_dates.astype(np.int64)
date_idx = {int(d): i for i, d in enumerate(all_dates_int)}
print(f"  銘柄: {len(by_code)}, 取引日: {len(all_dates)}")


def v6b_score(ac: np.ndarray, i: int):
    """インデックスiの時点でのV6bスコア。データ不足ならNone"""
    if i < 89:
        return None
    last = ac[i]
    r20 = last / ac[i - 20] - 1
    ma75 = ac[i - 74:i + 1].mean()
    d75 = last / ma75 - 1
    daily = ac[i - 19:i + 1] / ac[i - 20:i] - 1
    vol20 = float(np.std(daily, ddof=1) * np.sqrt(252))
    if vol20 <= 0:
        return None
    return r20 / vol20 + 0.5 * d75


# 各銘柄の (date_index -> score, fwd20) を事前計算
# fwd20 はその銘柄自身のadj_close系列で20本先
score_table = {}  # code -> dict(global_date_index -> (score, fwd))
for code, (dates, ac) in by_code.items():
    n = len(ac)
    tbl = {}
    # この銘柄のローカルindex i を global date index にマップ
    local_to_global = np.array([date_idx[int(d)] for d in dates])
    for i in range(n):
        if i + HOLD >= n:
            continue
        sc = v6b_score(ac, i)
        if sc is None:
            continue
        fwd = ac[i + HOLD] / ac[i] - 1
        tbl[local_to_global[i]] = (sc, fwd)
    score_table[code] = tbl


def run_ls(entry_global_indices, c_side_bps, borrow_ann_bps):
    """指定エントリー日リストで非重複L/Sを実行し、コスト後 period return 列を返す。"""
    # コスト: L/S 1サイクルで long/short 各々 entry+exit = 4*c_side, 空売り20日調達料
    trade_cost = 4 * c_side_bps / 1e4
    borrow_cost = borrow_ann_bps / 1e4 * (HOLD / 252)
    total_cost = trade_cost + borrow_cost

    out = []
    for gi in entry_global_indices:
        rows = []
        for code, tbl in score_table.items():
            if gi in tbl:
                rows.append((tbl[gi][0], tbl[gi][1]))
        if len(rows) < 2 * N_SIDE:
            continue
        rows.sort(key=lambda x: x[0], reverse=True)
        longs = np.mean([r[1] for r in rows[:N_SIDE]])
        shorts = np.mean([r[1] for r in rows[-N_SIDE:]])
        gross = longs - shorts
        net = gross - total_cost
        out.append((all_dates[gi], gross, net))
    return out


# ======================================================
# A. 非重複20日 L/S (全20位相平均) — コスト前/後
# ======================================================
print("\n" + "=" * 76)
print("A. 非重複20日 L/S Sharpe (全20位相平均, top8/bottom8)")
print("=" * 76)

eval_start_gi = int(np.searchsorted(all_dates, np.datetime64("2021-07-01")))
C_SIDE = 10      # 片側10bps (大型株の往復スプレッド+手数料の現実的中央値)
BORROW = 150     # 空売り調達料 年率1.5%

print(f"\n  コスト前提: 片側 {C_SIDE}bps (L/S 1周=4倍), 空売り調達 年率{BORROW/100:.1f}%")
print(f"  {'期間':<14} {'gross Sh':<11} {'net Sh':<11} {'net mean%/20d':<15} {'net t値':<9} {'n周期'}")
print("  " + "-" * 70)

phase_summary = {p: {'gross': [], 'net': []} for p in ['全期間', 'IS', 'OOS']}

for phase in range(HOLD):
    entries = list(range(eval_start_gi + phase, len(all_dates), HOLD))
    res = run_ls(entries, C_SIDE, BORROW)
    for dt, gross, net in res:
        ts = pd.Timestamp(dt)
        phase_summary['全期間']['gross'].append(gross)
        phase_summary['全期間']['net'].append(net)
        if ts <= IS_END and ts >= IS_START:
            phase_summary['IS']['gross'].append(gross)
            phase_summary['IS']['net'].append(net)
        elif ts >= OOS_START:
            phase_summary['OOS']['gross'].append(gross)
            phase_summary['OOS']['net'].append(net)

for label in ['全期間', 'IS', 'OOS']:
    g = np.array(phase_summary[label]['gross'])
    nt = np.array(phase_summary[label]['net'])
    sg = sharpe_period(g)
    sn = sharpe_period(nt)
    mn = np.nanmean(nt) * 100
    tt = ttest_1samp(nt[~np.isnan(nt)], 0)[0] if len(nt) > 5 else float('nan')
    print(f"  {label:<14} {sg:<11.2f} {sn:<11.2f} {mn:<15.3f} {tt:<9.2f} {len(nt)}")

# ======================================================
# B. コスト感度分析 (全期間 net Sharpe)
# ======================================================
print("\n" + "=" * 76)
print("B. コスト感度分析 (全期間 net Sharpe, 全20位相平均)")
print("=" * 76)
print(f"\n  {'片側コスト':<14} {'空売り調達':<14} {'gross Sh':<11} {'net Sh':<11} {'net mean%/20d'}")
print("  " + "-" * 64)

for c_side in [5, 10, 15]:
    for borrow in [0, 150, 300]:
        allnet, allgross = [], []
        for phase in range(HOLD):
            entries = list(range(eval_start_gi + phase, len(all_dates), HOLD))
            res = run_ls(entries, c_side, borrow)
            for _, gross, net in res:
                allgross.append(gross)
                allnet.append(net)
        g = np.array(allgross); nt = np.array(allnet)
        print(f"  片側{c_side:>2}bps      年率{borrow/100:>4.1f}%      "
              f"{sharpe_period(g):<11.2f} {sharpe_period(nt):<11.2f} {np.nanmean(nt)*100:.3f}")

# ======================================================
# C. 単一位相 (offset=0) の年別 net リターン
# ======================================================
print("\n" + "=" * 76)
print(f"C. 年別 net リターン (offset=0位相, 片側{C_SIDE}bps + 調達{BORROW/100:.1f}%)")
print("=" * 76)

entries0 = list(range(eval_start_gi, len(all_dates), HOLD))
res0 = run_ls(entries0, C_SIDE, BORROW)
ldf = pd.DataFrame(res0, columns=['date', 'gross', 'net'])
ldf['date'] = pd.to_datetime(ldf['date'])
ldf['year'] = ldf['date'].dt.year

print(f"\n  {'年':<8} {'n周期':<8} {'net合計%':<12} {'net Sharpe':<12} {'勝率%'}")
print("  " + "-" * 50)
for yr, g in ldf.groupby('year'):
    nt = g['net'].values
    cumret = (np.prod(1 + nt) - 1) * 100
    sh = sharpe_period(nt)
    wr = (nt > 0).mean() * 100
    print(f"  {yr:<8} {len(g):<8} {cumret:<12.1f} {sh:<12.2f} {wr:.0f}")

# 保存
out_dir = os.path.dirname(__file__)
ldf.to_csv(os.path.join(out_dir, "nonoverlap_offset0.csv"), index=False)
print(f"\n  保存: nonoverlap_offset0.csv")
print("\n完了")
