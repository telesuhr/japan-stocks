"""
MA25乖離 過売り反発 戦略 — 実SL/TP & Walk-forward 検証
======================================================
分析日: 2026-05-15

【戦略仮説】
  MA25から -15% 以上下に乖離した銘柄を翌寄りLong、5-10日保有
  → 平均回帰による反発を捕る

【検証】
  1. MA25乖離閾値 (-10/-12/-15/-20%) のグリッドサーチ
  2. 保有期間 (5/10/15/20日) の最適化
  3. SL効果 (-3/-5/-7%)
  4. セクター別 (シクリカルが強いか)
  5. 流動性フィルタ (5億/10億/30億)
  6. Walk-forward (2024訓練 → 2025-26テスト)
  7. ポートフォリオシム (最大同時保有数)
  8. Strategy昇格判定 (7基準)
"""
import psycopg2
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

PG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}
COST_PCT_RT = 0.10

def q(sql):
    conn = psycopg2.connect(**PG)
    df = pd.read_sql(sql, conn)
    conn.close()
    return df


print("=" * 78)
print("  MA25乖離 過売り反発 戦略 — 実SL/TP & Walk-forward 検証")
print("=" * 78)
print("\n  データ取得中...")

# 株価データ + 出口判定用OHLC
sql = """
WITH base AS (
    SELECT d.code, s.name_ja, s.sector33_nm AS sector,
           d.date, d.adj_open, d.adj_close, d.adj_high, d.adj_low,
           d.adj_volume, d.turnover_value
    FROM stocks_daily d
    JOIN symbol_master s ON s.code5 = d.code
    WHERE s.market = '0111' AND d.date BETWEEN '2023-10-01' AND '2026-05-14'
      AND d.adj_close > 0 AND d.adj_open > 0
),
ind AS (
    SELECT *,
        AVG(adj_close) OVER (PARTITION BY code ORDER BY date
                             ROWS BETWEEN 25 PRECEDING AND 1 PRECEDING) AS ma25,
        AVG(adj_volume) OVER (PARTITION BY code ORDER BY date
                              ROWS BETWEEN 21 PRECEDING AND 2 PRECEDING) AS vol_ma20,
        LEAD(adj_open, 1) OVER (PARTITION BY code ORDER BY date) AS next_open
    FROM base
)
SELECT * FROM ind
WHERE date >= '2024-01-04' AND ma25 IS NOT NULL
"""

df = q(sql)
for c in ['adj_open','adj_close','adj_high','adj_low','adj_volume','turnover_value',
          'ma25','vol_ma20','next_open']:
    df[c] = pd.to_numeric(df[c], errors='coerce')
df['date'] = pd.to_datetime(df['date'])

df['dist_ma25'] = (df['adj_close'] / df['ma25'] - 1) * 100
df = df.sort_values(['code','date']).reset_index(drop=True)
code_dates = {c: g['date'].tolist() for c, g in df.groupby('code')}
df_idx = df.set_index(['code','date'])
print(f"  ロード: {len(df):,}行")


def simulate_trade(code, sig_date, sl_pct, hold_days):
    """Entry = sig_date翌日寄り, Exit = hold_days後の引け or SL"""
    if code not in code_dates:
        return None
    dates = code_dates[code]
    try:
        idx = dates.index(sig_date)
    except ValueError:
        return None
    if idx + 1 >= len(dates) or idx + hold_days >= len(dates):
        return None
    d_entry = dates[idx + 1]
    try:
        entry = df_idx.loc[(code, d_entry), 'next_open']
        if pd.isna(entry):
            entry = df_idx.loc[(code, d_entry), 'adj_open']
    except KeyError:
        return None
    if pd.isna(entry):
        return None
    sl_price = entry * (1 + sl_pct / 100) if sl_pct else None

    for k in range(1, hold_days + 1):
        d_k = dates[idx + 1 + k - 1]   # 翌日からホールド
        if k == 1:
            d_k = dates[idx + 1]
        else:
            d_k = dates[min(idx + 1 + k - 1, len(dates) - 1)]
        if idx + k >= len(dates):
            break
        d_k = dates[idx + k]
        try:
            row = df_idx.loc[(code, d_k)]
        except KeyError:
            continue
        hi = row['adj_high']
        lo = row['adj_low']
        cl = row['adj_close']
        if pd.isna(hi) or pd.isna(lo) or pd.isna(cl):
            continue
        if sl_price and lo <= sl_price:
            return {'ret': (sl_price/entry - 1)*100, 'reason': 'SL', 'day': k}
        if k == hold_days:
            return {'ret': (cl/entry - 1)*100, 'reason': 'TIME', 'day': k}
    return None


def run_backtest(sig_df, sl_pct, hold_days):
    trades = []
    for _, r in sig_df.iterrows():
        t = simulate_trade(r['code'], r['date'], sl_pct, hold_days)
        if t is None:
            continue
        t.update({'code': r['code'], 'name': r['name_ja'],
                  'sector': r['sector'], 'date': r['date'],
                  'dist_ma25': r['dist_ma25']})
        trades.append(t)
    return pd.DataFrame(trades)


# ════════════════════════════════════════════════════════════════
# 1. MA25乖離閾値 × 保有期間 × SL グリッドサーチ
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("【1】MA25乖離閾値 × 保有期間 × SL グリッドサーチ")
print("=" * 78)

base = df[df['turnover_value'] >= 1_000_000_000].copy()
print(f"  ベースサンプル (売買代金10億+): {len(base):,}")

print(f"\n  {'乖離閾値':<14}  {'Hold':>4}  {'SL':>5}  {'N':>5}  "
      f"{'gross':>7}  {'net':>7}  {'勝率':>6}  {'PF':>5}  {'Sharpe':>7}")
print("  " + "-" * 90)

grid = []
for dist_lo, dist_hi in [(-99, -20), (-20, -15), (-99, -15), (-15, -12), (-12, -10)]:
    label_dist = f"{dist_lo}〜{dist_hi}%" if dist_lo > -99 else f"<{dist_hi}%"
    sub_sig = base[(base['dist_ma25'] >= dist_lo) & (base['dist_ma25'] < dist_hi)]
    if len(sub_sig) < 100:
        continue
    for hold in [5, 10, 15]:
        for sl in [None, -3, -5, -7]:
            trades = run_backtest(sub_sig.iloc[::3], sl, hold)  # サンプリングで高速化
            if len(trades) < 50:
                continue
            net = trades['ret'] - COST_PCT_RT
            sharpe = net.mean() / trades['ret'].std() * np.sqrt(252/hold) if trades['ret'].std() > 0 else 0
            pf = net[net>0].sum() / abs(net[net<=0].sum()) if (net<=0).any() else 99
            sl_str = f"{sl}%" if sl else "なし"
            flag = " ★★" if sharpe > 1.8 else (" ★" if sharpe > 1.3 else "")
            grid.append({
                'dist_label': label_dist, 'hold': hold, 'sl': sl,
                'N': len(trades), 'gross': trades['ret'].mean(),
                'net': net.mean(), 'wr': (net>0).mean()*100,
                'pf': pf, 'sharpe': sharpe,
            })
            print(f"  {label_dist:<14}  {hold:>3}d  {sl_str:>5}  {len(trades):>5}  "
                  f"{trades['ret'].mean():>+6.3f}%  {net.mean():>+6.3f}%  "
                  f"{(net>0).mean()*100:>5.1f}%  {pf:>4.2f}  {sharpe:>+6.2f}{flag}")

grid_df = pd.DataFrame(grid).sort_values('sharpe', ascending=False)


# ════════════════════════════════════════════════════════════════
# 2. ベスト設定の詳細
# ════════════════════════════════════════════════════════════════
best = grid_df.iloc[0]
print("\n" + "=" * 78)
print(f"【2】ベスト設定: 乖離 {best['dist_label']}, hold {int(best['hold'])}d, SL {best['sl']}")
print("=" * 78)

# フルランで再実行 (サンプリングなし)
parts = best['dist_label'].split('〜')
if '<' in best['dist_label']:
    dist_lo, dist_hi = -99, float(best['dist_label'].replace('<','').replace('%',''))
else:
    dist_lo = float(parts[0])
    dist_hi = float(parts[1].replace('%',''))

best_sig = base[(base['dist_ma25'] >= dist_lo) & (base['dist_ma25'] < dist_hi)]
print(f"\n  シグナル数 (全期間): {len(best_sig):,}")
print(f"  バックテスト実行中...")
trades_full = run_backtest(best_sig, best['sl'], int(best['hold']))
trades_full['year'] = trades_full['date'].dt.year
trades_full['net_ret'] = trades_full['ret'] - COST_PCT_RT

print(f"  実行トレード: {len(trades_full):,}")

# 年次パフォーマンス
print(f"\n  ─ 年次パフォーマンス ─")
print(f"  {'年':>4}  {'N':>5}  {'gross':>7}  {'net':>7}  {'勝率':>6}  {'Sharpe':>7}")
print("  " + "-" * 60)
for year, g in trades_full.groupby('year'):
    r = g['ret']
    net = g['net_ret']
    sharpe = net.mean()/r.std()*np.sqrt(252/int(best['hold'])) if r.std()>0 else 0
    print(f"  {year}  {len(r):>5}  {r.mean():>+6.3f}%  {net.mean():>+6.3f}%  "
          f"{(net>0).mean()*100:>5.1f}%  {sharpe:>+6.2f}")

# Walk-forward
train = trades_full[trades_full['year'] == 2024]
test  = trades_full[trades_full['year'] >= 2025]
print(f"\n  ─ Walk-forward ─")
for label, g in [("Train 2024", train), ("Test 2025-26", test)]:
    if len(g) == 0: continue
    net = g['net_ret']
    sharpe = net.mean()/g['ret'].std()*np.sqrt(252/int(best['hold'])) if g['ret'].std()>0 else 0
    print(f"  {label:<14}: N={len(net):>4}  net={net.mean():>+.3f}%  Sharpe={sharpe:+.2f}")

# Exit理由
print(f"\n  ─ Exit理由 ─")
for reason, g in trades_full.groupby('reason'):
    pct = len(g)/len(trades_full)*100
    print(f"    {reason}: N={len(g)} ({pct:.1f}%)  平均ret={g['net_ret'].mean():+.3f}%  保有={g['day'].mean():.1f}日")


# ════════════════════════════════════════════════════════════════
# 3. セクター別
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("【3】セクター別パフォーマンス (N≥50のみ)")
print("=" * 78)
print(f"\n  {'セクター':<14}  {'N':>5}  {'net':>7}  {'勝率':>6}  {'Sharpe':>7}")
print("  " + "-" * 55)
sect_r = []
for sect, g in trades_full.groupby('sector'):
    if len(g) < 50: continue
    net = g['net_ret']
    sharpe = net.mean()/g['ret'].std()*np.sqrt(252/int(best['hold'])) if g['ret'].std()>0 else 0
    sect_r.append({'sector': sect, 'N': len(g), 'net': net.mean(),
                   'wr': (net>0).mean()*100, 'sharpe': sharpe})
sect_df_r = pd.DataFrame(sect_r).sort_values('sharpe', ascending=False)
for _, r in sect_df_r.iterrows():
    flag = " ★★" if r['sharpe']>2.0 else (" ★" if r['sharpe']>1.5 else "")
    print(f"  {r['sector']:<14}  {int(r['N']):>5}  {r['net']:>+6.3f}%  "
          f"{r['wr']:>5.1f}%  {r['sharpe']:>+6.2f}{flag}")


# ════════════════════════════════════════════════════════════════
# 4. 流動性フィルタ別
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("【4】流動性フィルタ別パフォーマンス")
print("=" * 78)

# 流動性別の trades (mergeしてフィルタ)
df_sig = base[(base['dist_ma25'] >= dist_lo) & (base['dist_ma25'] < dist_hi)][['code','date','turnover_value']]
trades_with_tv = trades_full.merge(df_sig, on=['code','date'], how='left')
trades_with_tv['turnover_value'] = pd.to_numeric(trades_with_tv['turnover_value'], errors='coerce')

print(f"\n  {'流動性':<18}  {'N':>5}  {'net':>7}  {'勝率':>6}  {'Sharpe':>7}")
print("  " + "-" * 55)
for tv_min, label in [(0.5e9, '5億+'), (1e9, '10億+'), (3e9, '30億+'), (5e9, '50億+')]:
    sub = trades_with_tv[trades_with_tv['turnover_value'] >= tv_min]
    if len(sub) < 50: continue
    net = sub['net_ret']
    sharpe = net.mean()/sub['ret'].std()*np.sqrt(252/int(best['hold'])) if sub['ret'].std()>0 else 0
    print(f"  売買代金 {label:<10}  {len(sub):>5}  {net.mean():>+6.3f}%  "
          f"{(net>0).mean()*100:>5.1f}%  {sharpe:>+6.2f}")


# ════════════════════════════════════════════════════════════════
# 5. ポートフォリオシム
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("【5】ポートフォリオシム — 同時保有数別")
print("=" * 78)

trades_full['exit_date'] = trades_full.apply(
    lambda r: r['date'] + pd.tseries.offsets.BDay(int(r['day']) + 1), axis=1)
trades_sorted = trades_full.sort_values('date').reset_index(drop=True)

POSITION = 1_000_000
for max_pos in [5, 10, 15, 25]:
    open_pos = []
    pnls = []
    skipped = 0
    for _, t in trades_sorted.iterrows():
        open_pos = [d for d in open_pos if d > t['date']]
        if len(open_pos) >= max_pos:
            skipped += 1
            continue
        open_pos.append(t['exit_date'])
        pnls.append({'date': t['date'], 'pnl': POSITION * t['net_ret'] / 100})
    if not pnls: continue
    p = pd.DataFrame(pnls).set_index('date')
    p['cum'] = p['pnl'].cumsum()
    max_dd = (p['cum'] - p['cum'].cummax()).min()
    daily = p['pnl'].resample('B').sum().fillna(0)
    sharpe_d = daily.mean()/daily.std()*np.sqrt(252) if daily.std()>0 else 0
    print(f"\n  最大{max_pos}銘柄:")
    print(f"    執行: {len(p):,} / スキップ: {skipped:,}")
    print(f"    総P&L: {p['pnl'].sum():>+12,.0f}円")
    print(f"    1トレード平均: {p['pnl'].mean():>+10,.0f}円")
    print(f"    最大DD: {max_dd:>+12,.0f}円")
    print(f"    日次Sharpe: {sharpe_d:+.2f}")


# ════════════════════════════════════════════════════════════════
# 6. Strategy昇格判定
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("【6】Strategy昇格判定")
print("=" * 78)

r_all = trades_full['ret']
net_all = trades_full['net_ret']
sharpe_all = net_all.mean()/r_all.std()*np.sqrt(252/int(best['hold'])) if r_all.std()>0 else 0
t_stat = r_all.mean()/r_all.std()*np.sqrt(len(r_all))
pf = net_all[net_all>0].sum()/abs(net_all[net_all<=0].sum()) if (net_all<=0).any() else 99
yearly_pos = all(g['ret'].mean()>0 for _, g in trades_full.groupby('year'))
test_pos = test['net_ret'].mean() > 0 if len(test) > 10 else False

criteria = [
    ("N ≥ 500",                len(r_all) >= 500,        f"{len(r_all)}"),
    ("net mean > 0",           net_all.mean() > 0,       f"{net_all.mean():+.3f}%"),
    ("Sharpe > 2.0",           sharpe_all > 2.0,         f"{sharpe_all:+.2f}"),
    ("t-stat > 3.0",           t_stat > 3.0,             f"{t_stat:+.2f}"),
    ("PF > 1.3",               pf > 1.3,                 f"{pf:.2f}"),
    ("年次全部 net>0",          yearly_pos,                "yearly"),
    ("Walk-forward Test > 0",  test_pos,                  f"test {test['net_ret'].mean():+.3f}%"),
]
print(f"\n  {'基準':<24}  {'結果':<10}  値")
print("  " + "-" * 55)
for name, ok, val in criteria:
    mark = "✅ PASS" if ok else "❌ FAIL"
    print(f"  {name:<24}  {mark:<10}  {val}")
passed = sum(1 for _, ok, _ in criteria if ok)
print(f"\n  合格基準: {passed} / {len(criteria)}")
if passed >= 6:
    print("  判定: 🎯 Strategy昇格を強く推奨")
elif passed >= 5:
    print("  判定: ⚠️ 条件付き昇格")
else:
    print("  判定: ❌ research段階")

# CSV出力
out_dir = '/Users/Yusuke/claude-code/japan-stocks/analyses/20260515_oversold_ma25_validation'
trades_full.to_csv(f'{out_dir}/trades.csv', index=False)
grid_df.to_csv(f'{out_dir}/grid.csv', index=False)
sect_df_r.to_csv(f'{out_dir}/sector.csv', index=False)
print(f"\n  ✅ 完了")
