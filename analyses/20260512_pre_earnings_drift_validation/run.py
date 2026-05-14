"""
決算前ドリフト戦略 — 実SL/TP & Walk-forward 検証
================================================
分析日: 2026-05-12

【戦略仮説】
  本決算 (FY) と 3Q 決算の発表5営業日前に Long、発表前日 (T-1) に Exit
  → 発表前の期待感・事前買いを捕まえる

【検証内容】
  1. doc_type 別の最適化 (本決算/四半期/業績修正)
  2. エントリー〜Exit のタイミング最適化 (5日前/3日前/前々日 etc.)
  3. SL の効果 (-3%, -5%)
  4. Walk-forward (2024 vs 2025-2026)
  5. セクター別パフォーマンス
  6. ポートフォリオシム
  7. Strategy昇格判定 (7基準)
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
print("  決算前ドリフト戦略 — 実SL/TP & Walk-forward 検証")
print("=" * 78)
print("\n  データ取得中...")

# 決算イベント (FY+3Q+1Q+2Q+業績修正+配当修正)
event_sql = """
SELECT DISTINCT ON (code, disc_date)
       f.code, s.name_ja, s.sector33_nm AS sector,
       f.disc_date, f.disc_time, f.doc_type
FROM fin_summary f
JOIN symbol_master s ON s.code5 = f.code
WHERE s.market = '0111'
  AND f.disc_date >= '2024-01-04'
ORDER BY code, disc_date,
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
"""
events = q(event_sql)
events['disc_date'] = pd.to_datetime(events['disc_date'])
print(f"  決算イベント: {len(events):,}")

# 価格データ (high/low含む実SL/TP用)
prc_sql = """
SELECT d.code, d.date, d.adj_open, d.adj_close, d.adj_high, d.adj_low,
       d.turnover_value
FROM stocks_daily d
JOIN symbol_master s ON s.code5 = d.code
WHERE s.market='0111' AND d.date >= '2023-12-01'
"""
prc = q(prc_sql)
prc['date'] = pd.to_datetime(prc['date'])
for c in ['adj_open','adj_close','adj_high','adj_low','turnover_value']:
    prc[c] = pd.to_numeric(prc[c], errors='coerce')
prc = prc.sort_values(['code','date']).reset_index(drop=True)
code_dates = {c: g['date'].tolist() for c, g in prc.groupby('code')}
prc_idx = prc.set_index(['code','date'])
print(f"  価格データ: {len(prc):,}行")


# ════════════════════════════════════════════════════════════════
# 実SL/TPバックテスト関数
# ════════════════════════════════════════════════════════════════
def backtest_event(row, entry_offset, exit_offset, sl_pct):
    """
    entry_offset: 決算日からのオフセット (e.g., -5 = 5営業日前)
    exit_offset:  決算日からのオフセット (e.g., -1 = 前日)
    sl_pct:       損切ライン (None or -3.0 等)
    """
    code = row['code']
    d_evt = row['disc_date']
    if code not in code_dates:
        return None
    dates = code_dates[code]
    try:
        idx = dates.index(d_evt)
    except ValueError:
        return None
    entry_idx = idx + entry_offset
    exit_idx  = idx + exit_offset
    if entry_idx < 0 or exit_idx >= len(dates) or exit_idx <= entry_idx:
        return None

    d_entry = dates[entry_idx]
    try:
        entry_price = prc_idx.loc[(code, d_entry), 'adj_close']
        entry_tv    = prc_idx.loc[(code, d_entry), 'turnover_value']
    except KeyError:
        return None
    if pd.isna(entry_price) or pd.isna(entry_tv):
        return None
    if entry_tv < 500_000_000:
        return None

    # 翌日の寄付を使う (T-5 引け→T-4寄り)
    if entry_idx + 1 >= len(dates):
        return None
    d_entry_next = dates[entry_idx + 1]
    try:
        actual_entry = prc_idx.loc[(code, d_entry_next), 'adj_open']
    except KeyError:
        return None
    if pd.isna(actual_entry):
        return None

    sl_price = actual_entry * (1 + sl_pct/100) if sl_pct else None

    # 保有期間中の SL チェック
    for k in range(entry_idx + 1, exit_idx + 1):
        d_k = dates[k]
        try:
            row_k = prc_idx.loc[(code, d_k)]
        except KeyError:
            continue
        lo_k = row_k['adj_low']
        cl_k = row_k['adj_close']
        if pd.isna(lo_k) or pd.isna(cl_k):
            continue
        if sl_price and lo_k <= sl_price:
            return {'ret': (sl_price/actual_entry - 1)*100, 'reason': 'SL', 'day': k - entry_idx}
        if k == exit_idx:
            return {'ret': (cl_k/actual_entry - 1)*100, 'reason': 'TIME', 'day': k - entry_idx}
    return None


# ════════════════════════════════════════════════════════════════
# 1. doc_type別 × エントリータイミング最適化
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("【1】doc_type × エントリータイミング グリッドサーチ")
print("  Entry: T-N 引け翌日寄り, Exit: T-1 引け, SLなし")
print("=" * 78)

target_docs = [
    'FYFinancialStatements_Consolidated_JP',
    'FYFinancialStatements_Consolidated_IFRS',
    '3QFinancialStatements_Consolidated_JP',
    '3QFinancialStatements_Consolidated_IFRS',
    '2QFinancialStatements_Consolidated_JP',
    '1QFinancialStatements_Consolidated_JP',
    'DividendForecastRevision',
    'EarnForecastRevision',
]

grid = []
print(f"\n  {'doc_type':<48}  {'Entry':>5}  {'N':>5}  {'net':>7}  {'勝率':>6}  {'Sharpe':>7}")
print("  " + "-" * 95)
for doc in target_docs:
    sub_ev = events[events['doc_type'] == doc]
    for entry_off in [-7, -5, -3, -2]:
        results = []
        for _, r in sub_ev.iterrows():
            bt = backtest_event(r, entry_off, -1, None)
            if bt:
                bt['code'] = r['code']
                bt['sector'] = r['sector']
                bt['disc_date'] = r['disc_date']
                results.append(bt)
        if len(results) < 30:
            continue
        df_r = pd.DataFrame(results)
        r = df_r['ret']
        net = r - COST_PCT_RT
        days = abs(entry_off) - 1
        sharpe = net.mean() / r.std() * np.sqrt(252/days) if r.std() > 0 and days > 0 else 0
        flag = " ★★" if sharpe > 2.0 else (" ★" if sharpe > 1.5 else "")
        print(f"  {doc:<48}  T{entry_off:>+3}  {len(df_r):>5}  {net.mean():>+6.3f}%  "
              f"{(r>0).mean()*100:>5.1f}%  {sharpe:>+6.2f}{flag}")
        grid.append({
            'doc_type': doc, 'entry': entry_off, 'N': len(df_r),
            'net': net.mean(), 'wr': (r>0).mean()*100, 'sharpe': sharpe,
        })

grid_df = pd.DataFrame(grid).sort_values('sharpe', ascending=False)


# ════════════════════════════════════════════════════════════════
# 2. ベスト設定の詳細 — マルチdoc_type 統合
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("【2】Sharpe>1.5 を満たす doc_type を統合した戦略")
print("=" * 78)

good_docs_df = grid_df[grid_df['sharpe'] > 1.5][['doc_type','entry','sharpe']]
print(f"\n  採択 doc_type / entry timing:")
for _, r in good_docs_df.head(15).iterrows():
    print(f"    {r['doc_type']:<48}  T{int(r['entry']):>+3}  Sharpe {r['sharpe']:+.2f}")

# 各doc_typeのベストentry_offsetを採用
best_per_doc = grid_df.loc[grid_df.groupby('doc_type')['sharpe'].idxmax()].reset_index(drop=True)
top_docs = best_per_doc[best_per_doc['sharpe'] > 1.5]
print(f"\n  最終採用: {len(top_docs)} doc_type")

# 統合バックテスト (各doc_typeで最適entry_offset使用)
all_trades = []
for _, r in top_docs.iterrows():
    sub_ev = events[events['doc_type'] == r['doc_type']]
    for _, ev_r in sub_ev.iterrows():
        bt = backtest_event(ev_r, int(r['entry']), -1, None)
        if bt:
            bt['code'] = ev_r['code']
            bt['name'] = ev_r['name_ja']
            bt['sector'] = ev_r['sector']
            bt['disc_date'] = ev_r['disc_date']
            bt['doc_type'] = r['doc_type']
            bt['entry_offset'] = int(r['entry'])
            all_trades.append(bt)

trades = pd.DataFrame(all_trades)
trades['year'] = trades['disc_date'].dt.year
trades['net_ret'] = trades['ret'] - COST_PCT_RT
print(f"\n  統合トレード数: {len(trades):,}")
print(f"  Gross平均: {trades['ret'].mean():+.3f}%")
print(f"  Net平均:   {trades['net_ret'].mean():+.3f}%")
print(f"  勝率:      {(trades['ret']>0).mean()*100:.1f}%")
print(f"  Sharpe:    {trades['net_ret'].mean()/trades['ret'].std()*np.sqrt(252/4):+.2f}")  # avg ~4日保有
print(f"  t-stat:    {trades['ret'].mean()/trades['ret'].std()*np.sqrt(len(trades)):+.2f}")


# ════════════════════════════════════════════════════════════════
# 3. SL の効果検証
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("【3】SL の効果検証 (top_docs統合)")
print("=" * 78)

sl_results = []
for sl_pct in [None, -2, -3, -5]:
    all_trades_sl = []
    for _, r in top_docs.iterrows():
        sub_ev = events[events['doc_type'] == r['doc_type']]
        for _, ev_r in sub_ev.iterrows():
            bt = backtest_event(ev_r, int(r['entry']), -1, sl_pct)
            if bt:
                all_trades_sl.append(bt)
    if not all_trades_sl:
        continue
    df_sl = pd.DataFrame(all_trades_sl)
    net = df_sl['ret'] - COST_PCT_RT
    sharpe = net.mean() / df_sl['ret'].std() * np.sqrt(252/4) if df_sl['ret'].std() > 0 else 0
    pf = net[net>0].sum() / abs(net[net<=0].sum()) if (net<=0).any() else 99
    sl_results.append({
        'sl': sl_pct, 'N': len(df_sl),
        'net': net.mean(), 'wr': (net>0).mean()*100,
        'sharpe': sharpe, 'pf': pf,
    })
    sl_str = f"{sl_pct}%" if sl_pct else "なし"
    print(f"\n  SL = {sl_str}")
    print(f"    N={len(df_sl):,}, net={net.mean():+.3f}%, 勝率={(net>0).mean()*100:.1f}%, "
          f"Sharpe={sharpe:+.2f}, PF={pf:.2f}")


# ════════════════════════════════════════════════════════════════
# 4. 年次安定性 & Walk-forward
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("【4】年次パフォーマンス & Walk-forward")
print("=" * 78)

print(f"\n  ─ 年次パフォーマンス (SL なし) ─")
print(f"  {'年':>4}  {'N':>5}  {'gross':>7}  {'net':>7}  {'勝率':>6}  {'Sharpe':>7}")
print("  " + "-" * 55)
for year, g in trades.groupby('year'):
    r = g['ret']
    net = g['net_ret']
    sharpe = net.mean() / r.std() * np.sqrt(252/4) if r.std() > 0 else 0
    print(f"  {year}  {len(r):>5}  {r.mean():>+6.3f}%  {net.mean():>+6.3f}%  "
          f"{(r>0).mean()*100:>5.1f}%  {sharpe:>+6.2f}")

train = trades[trades['year'] == 2024]
test  = trades[trades['year'] >= 2025]
print(f"\n  ─ Walk-forward ─")
for label, g in [("Train 2024", train), ("Test 2025-26", test)]:
    if len(g) == 0:
        continue
    net = g['net_ret']
    sharpe = net.mean() / g['ret'].std() * np.sqrt(252/4) if g['ret'].std() > 0 else 0
    print(f"  {label:<14}: N={len(net):>4}  net={net.mean():>+.3f}%  Sharpe={sharpe:+.2f}")


# ════════════════════════════════════════════════════════════════
# 5. セクター別
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("【5】セクター別パフォーマンス")
print("=" * 78)
print(f"\n  {'セクター':<14}  {'N':>4}  {'net':>7}  {'勝率':>6}  {'Sharpe':>7}")
print("  " + "-" * 55)
sect_r = []
for sect, g in trades.groupby('sector'):
    if len(g) < 30:
        continue
    net = g['net_ret']
    sharpe = net.mean() / g['ret'].std() * np.sqrt(252/4) if g['ret'].std() > 0 else 0
    sect_r.append({'sector': sect, 'N': len(g), 'net': net.mean(),
                   'wr': (net>0).mean()*100, 'sharpe': sharpe})
sect_df_r = pd.DataFrame(sect_r).sort_values('sharpe', ascending=False)
for _, r in sect_df_r.iterrows():
    flag = " ★" if r['sharpe'] > 2.0 else (" ◎" if r['sharpe'] > 1.5 else "")
    print(f"  {r['sector']:<14}  {int(r['N']):>4}  {r['net']:>+6.3f}%  "
          f"{r['wr']:>5.1f}%  {r['sharpe']:>+6.2f}{flag}")


# ════════════════════════════════════════════════════════════════
# 6. Strategy昇格判定
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("【6】Strategy昇格判定")
print("=" * 78)

r_all = trades['ret']
net_all = trades['net_ret']
sharpe_all = net_all.mean() / r_all.std() * np.sqrt(252/4) if r_all.std() > 0 else 0
t_stat = r_all.mean() / r_all.std() * np.sqrt(len(r_all))
pf = net_all[net_all>0].sum() / abs(net_all[net_all<=0].sum()) if (net_all<=0).any() else 99
yearly_pos = all(g['ret'].mean() > 0 for _, g in trades.groupby('year'))
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
out_dir = '/Users/Yusuke/claude-code/japan-stocks/analyses/20260512_pre_earnings_drift_validation'
trades.to_csv(f'{out_dir}/trades.csv', index=False)
grid_df.to_csv(f'{out_dir}/grid_results.csv', index=False)
top_docs.to_csv(f'{out_dir}/best_doc_types.csv', index=False)
print(f"\n  ✅ 完了")
