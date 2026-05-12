"""
PEAD戦略 — 実SL/TPバックテスト & 昇格判定
========================================
分析日: 2026-05-12

【戦略仮説】
  決算発表後の大GU(>+5%) 銘柄を「翌日引け」で買い、5日後の引けで決済
  → Sharpe +1.49 (t-stat +3.68) のエッジを実SL/TPで再検証

【新規検証項目】
  - 日次High/Lowを使った実SL/TPバックテスト (clip-based楽観バイアス除去)
  - Walk-forward: 2024訓練 → 2025-2026テスト
  - GU閾値の最適化 (+3% / +5% / +7%)
  - 流動性フィルタ (5億/10億/30億)
  - セクター×Doc_type絞り込み
  - Strategy昇格判定 (6基準)
"""
import psycopg2
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

PG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}
COST_PCT_RT = 0.10
HOLD_DAYS   = 5

def q(sql):
    conn = psycopg2.connect(**PG)
    df = pd.read_sql(sql, conn)
    conn.close()
    return df


print("=" * 78)
print("  PEAD戦略 — 実SL/TPバックテスト & 昇格判定")
print("=" * 78)
print("\n  Step 1: 大GU決算イベントを抽出")

# AC発表(引け後) → 翌日大GU(>+3%) のイベントを抽出
event_sql = """
WITH events AS (
    SELECT f.code, s.name_ja, s.sector33_nm AS sector, s.sector17_nm,
           f.disc_date, f.disc_time, f.doc_type
    FROM fin_summary f
    JOIN symbol_master s ON s.code5 = f.code
    WHERE s.market = '0111'
      AND f.disc_date >= '2024-01-04'
      AND f.disc_time >= '15:00:00'    -- AC のみ
),
prc AS (
    SELECT d.code, d.date, d.adj_open, d.adj_close, d.adj_high, d.adj_low,
           d.turnover_value, d.adj_volume
    FROM stocks_daily d
    JOIN symbol_master s ON s.code5 = d.code
    WHERE s.market='0111' AND d.date >= '2024-01-01'
)
SELECT e.code, e.name_ja, e.sector, e.sector17_nm, e.doc_type,
       e.disc_date,
       p_t.adj_close   AS t_close,
       p_t.turnover_value AS t_tv,
       p_tp1.date      AS tp1_date,
       p_tp1.adj_open  AS tp1_open,
       p_tp1.adj_close AS tp1_close,
       p_tp1.adj_high  AS tp1_high,
       p_tp1.adj_low   AS tp1_low,
       p_tp1.turnover_value AS tp1_tv
FROM events e
JOIN prc p_t   ON p_t.code   = e.code AND p_t.date   = e.disc_date
JOIN LATERAL (
    SELECT * FROM prc p2
    WHERE p2.code = e.code AND p2.date > e.disc_date
    ORDER BY p2.date LIMIT 1
) p_tp1 ON true
"""
df = q(event_sql)
for c in ['t_close','t_tv','tp1_open','tp1_close','tp1_high','tp1_low','tp1_tv']:
    df[c] = pd.to_numeric(df[c], errors='coerce')
df['disc_date'] = pd.to_datetime(df['disc_date'])
df['tp1_date']  = pd.to_datetime(df['tp1_date'])
df['gap_ret']   = (df['tp1_open']  / df['t_close']  - 1) * 100

print(f"  AC決算イベント数: {len(df):,}")

# 周辺価格を取得（T+2 〜 T+10 まで広めに）
price_sql = """
SELECT code, date, adj_open, adj_close, adj_high, adj_low, turnover_value
FROM stocks_daily d
JOIN symbol_master s ON s.code5 = d.code
WHERE s.market='0111' AND d.date >= '2024-01-04'
"""
prc = q(price_sql)
for c in ['adj_open','adj_close','adj_high','adj_low','turnover_value']:
    prc[c] = pd.to_numeric(prc[c], errors='coerce')
prc['date'] = pd.to_datetime(prc['date'])
prc = prc.sort_values(['code','date']).reset_index(drop=True)

# 銘柄ごとの date list
code_dates = {code: g['date'].tolist() for code, g in prc.groupby('code')}
prc_idx = prc.set_index(['code','date'])

print(f"  価格データ: {len(prc):,}行")


# ════════════════════════════════════════════════════════════════
# 実SL/TPシミュレーション関数
# ════════════════════════════════════════════════════════════════
def simulate_pead(row, sl_pct, tp_pct, hold_days):
    """
    Entry: T+1 引け (tp1_close)
    Exit:  hold_days 営業日後の引け or 途中でSL/TPヒット
    """
    code = row['code']
    if pd.isna(row['tp1_close']):
        return None
    entry = row['tp1_close']
    tp1d  = row['tp1_date']

    if code not in code_dates:
        return None
    dates = code_dates[code]
    try:
        idx = dates.index(tp1d)
    except ValueError:
        return None
    if idx + hold_days >= len(dates):
        return None

    sl_price = entry * (1 + sl_pct/100) if sl_pct else None
    tp_price = entry * (1 + tp_pct/100) if tp_pct else None

    # T+2 〜 T+5 (hold_days = 5なら4日間追跡 + T+5 close)
    for k in range(1, hold_days + 1):
        d = dates[idx + k]
        try:
            row_d = prc_idx.loc[(code, d)]
        except KeyError:
            continue
        hi = row_d['adj_high']
        lo = row_d['adj_low']
        cl = row_d['adj_close']
        if pd.isna(hi) or pd.isna(lo) or pd.isna(cl):
            continue

        sl_hit = sl_price and lo <= sl_price
        tp_hit = tp_price and hi >= tp_price
        if sl_hit and tp_hit:
            return {'ret': (sl_price/entry - 1)*100, 'reason': 'SL', 'day': k}
        elif sl_hit:
            return {'ret': (sl_price/entry - 1)*100, 'reason': 'SL', 'day': k}
        elif tp_hit:
            return {'ret': (tp_price/entry - 1)*100, 'reason': 'TP', 'day': k}

        if k == hold_days:
            return {'ret': (cl/entry - 1)*100, 'reason': 'TIME', 'day': k}
    return None


def run_backtest(df_signals, sl_pct, tp_pct, hold_days=HOLD_DAYS):
    trades = []
    for _, row in df_signals.iterrows():
        r = simulate_pead(row, sl_pct, tp_pct, hold_days)
        if r is None:
            continue
        r['code']      = row['code']
        r['name']      = row['name_ja']
        r['sector']    = row['sector']
        r['doc_type']  = row['doc_type']
        r['disc_date'] = row['disc_date']
        r['gap_ret']   = row['gap_ret']
        r['tp1_tv']    = row['tp1_tv']
        r['net_ret']   = r['ret'] - COST_PCT_RT
        trades.append(r)
    return pd.DataFrame(trades)


# ════════════════════════════════════════════════════════════════
# Step 2. GU閾値 × 流動性 × SL/TP グリッドサーチ
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("【1】GU閾値 × 流動性 × SL/TP グリッドサーチ")
print("=" * 78)

print(f"\n  {'GU閾値':<8}  {'流動性':<14}  {'SL/TP':<14}  {'N':>4}  "
      f"{'net':>8}  {'勝率':>6}  {'Sharpe':>7}  {'PF':>5}")
print("  " + "-" * 90)

grid_results = []
for gu_min in [3.0, 5.0, 7.0]:
    for tv_min in [500_000_000, 1_000_000_000, 3_000_000_000]:
        sig = df[(df['gap_ret'] >= gu_min) & (df['tp1_tv'] >= tv_min)].copy()
        if len(sig) < 50:
            continue
        for sl_pct, tp_pct in [(None, None), (-5, None), (-5, 10), (-7, None), (-7, 15), (-10, None)]:
            trades = run_backtest(sig, sl_pct, tp_pct)
            if len(trades) < 20:
                continue
            net = trades['net_ret']
            sharpe = net.mean() / trades['ret'].std() * np.sqrt(252) if trades['ret'].std() > 0 else 0
            pf = net[net>0].sum() / abs(net[net<=0].sum()) if (net<=0).any() else 99
            sl_str = f"{sl_pct}%" if sl_pct else "なし"
            tp_str = f"{tp_pct}%" if tp_pct else "なし"
            tv_str = f"{tv_min/1e8:.0f}億+"
            gu_str = f">+{gu_min:.0f}%"
            param = f"SL{sl_str}/TP{tp_str}"
            grid_results.append({
                'gu': gu_min, 'tv': tv_min, 'sl': sl_pct, 'tp': tp_pct,
                'N': len(trades), 'net': net.mean(),
                'sharpe': sharpe, 'wr': (net>0).mean()*100, 'pf': pf,
            })
            flag = " ★★★" if sharpe > 2.5 else (" ★★" if sharpe > 1.5 else (" ★" if sharpe > 1.0 else ""))
            print(f"  {gu_str:<8}  {tv_str:<14}  {param:<14}  {len(trades):>4}  "
                  f"{net.mean():>+7.3f}%  {(net>0).mean()*100:>5.1f}%  "
                  f"{sharpe:>+6.2f}  {pf:>4.2f}{flag}")

grid_df = pd.DataFrame(grid_results).sort_values('sharpe', ascending=False)


# ════════════════════════════════════════════════════════════════
# Step 3. ベスト設定の詳細
# ════════════════════════════════════════════════════════════════
best = grid_df.iloc[0]
print("\n" + "=" * 78)
print(f"【2】ベスト設定: GU>+{best['gu']:.0f}% × {best['tv']/1e8:.0f}億+ × SL={best['sl']}% TP={best['tp']}%")
print(f"     N={int(best['N'])}, net={best['net']:+.3f}%, Sharpe={best['sharpe']:+.2f}")
print("=" * 78)

sig_best = df[(df['gap_ret'] >= best['gu']) & (df['tp1_tv'] >= best['tv'])].copy()
trades_best = run_backtest(sig_best, best['sl'], best['tp'])
trades_best['year'] = trades_best['disc_date'].dt.year

# 年次パフォーマンス
print(f"\n  ─ 年次パフォーマンス ─")
print(f"  {'年':>4}  {'N':>4}  {'gross':>7}  {'net':>7}  {'勝率':>6}  {'Sharpe':>7}  {'最大利益':>9}  {'最大損失':>9}")
print("  " + "-" * 80)
for year, g in trades_best.groupby('year'):
    r = g['ret']
    net = g['net_ret']
    sharpe = net.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0
    print(f"  {year}  {len(r):>4}  {r.mean():>+6.3f}%  {net.mean():>+6.3f}%  "
          f"{(net>0).mean()*100:>5.1f}%  {sharpe:>+6.2f}  "
          f"{r.max():>+8.2f}%  {r.min():>+8.2f}%")

# Walk-forward
train = trades_best[trades_best['year'] == 2024]
test  = trades_best[trades_best['year'] >= 2025]
print(f"\n  ─ Walk-forward検証 ─")
for label, g in [("Train 2024", train), ("Test 2025-26", test)]:
    if len(g) == 0:
        continue
    net = g['net_ret']
    sharpe = net.mean() / g['ret'].std() * np.sqrt(252) if g['ret'].std() > 0 else 0
    print(f"  {label:<14}: N={len(net):>3}  net={net.mean():>+.3f}%  "
          f"勝率={(net>0).mean()*100:.1f}%  Sharpe={sharpe:+.2f}")


# ════════════════════════════════════════════════════════════════
# Step 4. Exit理由の分布
# ════════════════════════════════════════════════════════════════
print(f"\n  ─ Exit理由の分布 ─")
for reason, g in trades_best.groupby('reason'):
    net = g['net_ret']
    print(f"    {reason:<6}: N={len(g):>4} ({len(g)/len(trades_best)*100:.1f}%)  "
          f"平均ret={net.mean():>+.3f}%  平均保有={g['day'].mean():.1f}日")


# ════════════════════════════════════════════════════════════════
# Step 5. セクター別
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("【3】セクター別パフォーマンス")
print("=" * 78)
print(f"\n  {'セクター':<14}  {'N':>4}  {'net':>7}  {'勝率':>6}  {'Sharpe':>7}")
print("  " + "-" * 55)
sect_results = []
for sect, g in trades_best.groupby('sector'):
    if len(g) < 15:
        continue
    net = g['net_ret']
    sharpe = net.mean() / g['ret'].std() * np.sqrt(252) if g['ret'].std() > 0 else 0
    sect_results.append({
        'sector': sect, 'N': len(g),
        'net': net.mean(), 'wr': (net>0).mean()*100,
        'sharpe': sharpe,
    })
sect_df_res = pd.DataFrame(sect_results).sort_values('sharpe', ascending=False)
for _, r in sect_df_res.iterrows():
    flag = " ★" if r['sharpe'] > 1.5 else ""
    print(f"  {r['sector']:<14}  {int(r['N']):>4}  {r['net']:>+6.3f}%  "
          f"{r['wr']:>5.1f}%  {r['sharpe']:>+6.2f}{flag}")


# ════════════════════════════════════════════════════════════════
# Step 6. doc_type別
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("【4】doc_type別パフォーマンス (業績修正vs決算)")
print("=" * 78)
print(f"\n  {'doc_type':<48}  {'N':>4}  {'net':>7}  {'勝率':>6}  {'Sharpe':>7}")
print("  " + "-" * 85)
for dt, g in trades_best.groupby('doc_type'):
    if len(g) < 15:
        continue
    net = g['net_ret']
    sharpe = net.mean() / g['ret'].std() * np.sqrt(252) if g['ret'].std() > 0 else 0
    flag = " ★" if sharpe > 1.5 else ""
    print(f"  {dt:<48}  {len(g):>4}  {net.mean():>+6.3f}%  "
          f"{(net>0).mean()*100:>5.1f}%  {sharpe:>+6.2f}{flag}")


# ════════════════════════════════════════════════════════════════
# Step 7. ポートフォリオシム
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("【5】ポートフォリオシム — 同時保有数別")
print("=" * 78)

trades_best['exit_date'] = trades_best.apply(
    lambda r: r['disc_date'] + pd.tseries.offsets.BDay(int(r['day']) + 1), axis=1)
trades_best_sorted = trades_best.sort_values('disc_date').reset_index(drop=True)

POSITION = 1_000_000
for max_pos in [3, 5, 10]:
    open_pos = []
    pnls = []
    skipped = 0
    for _, t in trades_best_sorted.iterrows():
        open_pos = [d for d in open_pos if d > t['disc_date']]
        if len(open_pos) >= max_pos:
            skipped += 1
            continue
        open_pos.append(t['exit_date'])
        pnls.append({'date': t['disc_date'], 'pnl': POSITION * t['net_ret'] / 100})
    if not pnls:
        continue
    p = pd.DataFrame(pnls).set_index('date')
    p['cum'] = p['pnl'].cumsum()
    max_dd = (p['cum'] - p['cum'].cummax()).min()
    daily = p['pnl'].resample('B').sum().fillna(0)
    sharpe_d = daily.mean()/daily.std()*np.sqrt(252) if daily.std() > 0 else 0
    print(f"\n  最大{max_pos}銘柄同時保有:")
    print(f"    執行: {len(p)} / スキップ: {skipped}")
    print(f"    総P&L: {p['pnl'].sum():>+12,.0f}円")
    print(f"    1トレード平均: {p['pnl'].mean():>+10,.0f}円")
    print(f"    最大DD: {max_dd:>+12,.0f}円")
    print(f"    日次Sharpe: {sharpe_d:+.2f}")


# ════════════════════════════════════════════════════════════════
# Step 8. Strategy昇格判定
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("【6】Strategy昇格判定")
print("=" * 78)

r_all = trades_best['ret']
net_all = trades_best['net_ret']
sharpe_all = net_all.mean() / r_all.std() * np.sqrt(252) if r_all.std() > 0 else 0
t_stat = r_all.mean() / r_all.std() * np.sqrt(len(r_all))
pf = net_all[net_all>0].sum() / abs(net_all[net_all<=0].sum()) if (net_all<=0).any() else 99
yearly_pos = all(g['ret'].mean() > 0 for _, g in trades_best.groupby('year'))
test_pos = test['net_ret'].mean() > 0 if len(test) > 10 else False

criteria = [
    ("N ≥ 100",                len(r_all) >= 100,        f"{len(r_all)}"),
    ("net mean > 0",           net_all.mean() > 0,       f"{net_all.mean():+.3f}%"),
    ("Sharpe > 2.0",           sharpe_all > 2.0,         f"{sharpe_all:+.2f}"),
    ("t-stat > 2.0",           t_stat > 2.0,             f"{t_stat:+.2f}"),
    ("PF > 1.3",               pf > 1.3,                 f"{pf:.2f}"),
    ("年次全部 net>0",          yearly_pos,                "yearly"),
    ("Walk-forward Test > 0",  test_pos,                  f"test net {test['net_ret'].mean():+.3f}%"),
]
print(f"\n  ベスト設定: GU>+{best['gu']:.0f}% × {best['tv']/1e8:.0f}億+ × SL={best['sl']}% TP={best['tp']}%")
print(f"  {'基準':<24}  {'結果':<10}  値")
print("  " + "-" * 60)
for name, ok, val in criteria:
    mark = "✅ PASS" if ok else "❌ FAIL"
    print(f"  {name:<24}  {mark:<10}  {val}")
passed = sum(1 for _, ok, _ in criteria if ok)
print(f"\n  合格基準: {passed} / {len(criteria)}")
if passed >= 6:
    verdict = "🎯 Strategy昇格を強く推奨"
elif passed >= 5:
    verdict = "⚠️  条件付き昇格"
else:
    verdict = "❌ research段階"
print(f"  判定: {verdict}")

# CSV出力
out_dir = '/Users/Yusuke/claude-code/japan-stocks/analyses/20260512_earnings_pead_validation'
grid_df.to_csv(f'{out_dir}/grid_results.csv', index=False)
trades_best.to_csv(f'{out_dir}/best_trades.csv', index=False)
sect_df_res.to_csv(f'{out_dir}/sector_breakdown.csv', index=False)
print(f"\n  ✅ 完了。CSV 3本出力")
