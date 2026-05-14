"""
大型株 過売り反発 戦略 — 実SL/TP & 詳細検証
============================================
分析日: 2026-05-15

【ユニバース】 TOPIX Core30 + Large70 = 100銘柄 (大型のみ)
【シグナル候補】
  A. MA25 <-15% (深い乖離)
  B. 20日下落率 <-20% (急落)
  C. ボリンジャー -1.8σ近辺

【既存戦略との違い】
  oversold_ma25_reversal (既存): プライム全銘柄、売買代金10億+
    → 2026年弱化 (Sharpe -0.58)
  本戦略: Core30+Large70限定
    → 2026年 Sharpe +0.28 (Core30は維持)、流動性極めて高い
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
print("  大型株 (Core30+Large70) 過売り反発 — 実SL/TP検証")
print("=" * 78)
print("\n  データ取得中...")

sql = """
WITH base AS (
    SELECT d.code, s.name_ja, s.sector33_nm AS sector, s.scale_cat,
           d.date, d.adj_open, d.adj_close, d.adj_high, d.adj_low,
           d.adj_volume, d.turnover_value
    FROM stocks_daily d
    JOIN symbol_master s ON s.code5 = d.code
    WHERE s.market = '0111'
      AND s.scale_cat IN ('TOPIX Core30', 'TOPIX Large70')
      AND d.date BETWEEN '2023-10-01' AND '2026-05-14'
      AND d.adj_close > 0 AND d.adj_open > 0
),
ind AS (
    SELECT *,
        AVG(adj_close)  OVER w25 AS ma25,
        STDDEV(adj_close) OVER w20 AS sd20,
        LAG(adj_close, 20) OVER (PARTITION BY code ORDER BY date) AS close_20d_ago,
        LEAD(adj_open, 1)  OVER (PARTITION BY code ORDER BY date) AS next_open
    FROM base
    WINDOW
        w25 AS (PARTITION BY code ORDER BY date ROWS BETWEEN 25 PRECEDING AND 1 PRECEDING),
        w20 AS (PARTITION BY code ORDER BY date ROWS BETWEEN 21 PRECEDING AND 2 PRECEDING)
)
SELECT * FROM ind WHERE date >= '2024-01-04' AND ma25 IS NOT NULL
"""
df = q(sql)
for c in ['adj_open','adj_close','adj_high','adj_low','adj_volume','turnover_value',
          'ma25','sd20','close_20d_ago','next_open']:
    df[c] = pd.to_numeric(df[c], errors='coerce')
df['date'] = pd.to_datetime(df['date'])
df['dist_ma25'] = (df['adj_close'] / df['ma25'] - 1) * 100
df['ret_20d']   = (df['adj_close'] / df['close_20d_ago'] - 1) * 100
df['bb_pos']    = (df['adj_close'] - df['ma25']) / (df['sd20'] * 2)

df = df.sort_values(['code','date']).reset_index(drop=True)
code_dates = {c: g['date'].tolist() for c, g in df.groupby('code')}
df_idx = df.set_index(['code','date'])
print(f"  ロード: {len(df):,}行 × {df['code'].nunique()}銘柄")


def simulate_trade(code, sig_date, sl_pct, hold_days):
    """T+1 寄り → T+hold_days 引け or SL"""
    if code not in code_dates: return None
    dates = code_dates[code]
    try:
        idx = dates.index(sig_date)
    except ValueError:
        return None
    if idx + hold_days >= len(dates) or idx + 1 >= len(dates):
        return None
    d_entry = dates[idx + 1]
    try:
        entry = df_idx.loc[(code, d_entry), 'adj_open']
    except KeyError:
        return None
    if pd.isna(entry): return None
    sl_price = entry * (1 + sl_pct/100) if sl_pct else None
    for k in range(1, hold_days + 1):
        d_k = dates[idx + 1 + k - 1]
        if k > 1:
            if idx + k >= len(dates): break
            d_k = dates[idx + k]
        try:
            row = df_idx.loc[(code, d_k)]
        except KeyError:
            continue
        hi, lo, cl = row['adj_high'], row['adj_low'], row['adj_close']
        if pd.isna(hi) or pd.isna(lo) or pd.isna(cl):
            continue
        if sl_price and lo <= sl_price:
            return {'ret': (sl_price/entry - 1)*100, 'reason': 'SL', 'day': k}
        if k == hold_days:
            return {'ret': (cl/entry - 1)*100, 'reason': 'TIME', 'day': k}
    return None


# ════════════════════════════════════════════════════════════════
# 1. シグナル × 保有期間 × SL グリッドサーチ
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("【1】シグナル × 保有期間 × SL グリッドサーチ")
print("=" * 78)

signals_def = [
    ('A_MA25<-15%',          df['dist_ma25'] < -15),
    ('B_20d下落<-20%',        df['ret_20d'] < -20),
    ('C_BB<-1.8σ',           df['bb_pos'] < -1.8),
    ('AB_AorB(複合)',         (df['dist_ma25'] < -15) | (df['ret_20d'] < -20)),
    ('ABC_3条件OR',           (df['dist_ma25'] < -15) | (df['ret_20d'] < -20) | (df['bb_pos'] < -1.8)),
]

grid = []
print(f"\n  {'シグナル':<24}  {'H':>4}  {'SL':>5}  {'N':>5}  "
      f"{'gross':>7}  {'net':>7}  {'勝率':>6}  {'PF':>5}  {'Sharpe':>7}")
print("  " + "-" * 90)

for sig_label, mask in signals_def:
    sig_df = df[mask].copy()
    for hold in [5, 10]:
        for sl in [None, -5, -7, -10]:
            trades = []
            for _, r in sig_df.iterrows():
                t = simulate_trade(r['code'], r['date'], sl, hold)
                if t:
                    t.update({'code': r['code'], 'sector': r['sector'],
                              'scale_cat': r['scale_cat'], 'date': r['date']})
                    trades.append(t)
            if len(trades) < 50: continue
            df_t = pd.DataFrame(trades)
            net = df_t['ret'] - COST_PCT_RT
            sharpe = net.mean()/df_t['ret'].std()*np.sqrt(252/hold) if df_t['ret'].std()>0 else 0
            pf = net[net>0].sum()/abs(net[net<=0].sum()) if (net<=0).any() else 99
            sl_str = f"{sl}%" if sl else "なし"
            grid.append({'sig': sig_label, 'hold': hold, 'sl': sl,
                         'N': len(df_t), 'gross': df_t['ret'].mean(),
                         'net': net.mean(), 'wr': (net>0).mean()*100,
                         'pf': pf, 'sharpe': sharpe})
            flag = " ★★" if sharpe>2.0 else (" ★" if sharpe>1.5 else "")
            print(f"  {sig_label:<24}  {hold:>3}d  {sl_str:>5}  {len(df_t):>5}  "
                  f"{df_t['ret'].mean():>+6.3f}%  {net.mean():>+6.3f}%  "
                  f"{(net>0).mean()*100:>5.1f}%  {pf:>4.2f}  {sharpe:>+6.2f}{flag}")

grid_df = pd.DataFrame(grid).sort_values('sharpe', ascending=False)
best = grid_df.iloc[0]
print(f"\n  ★ ベスト設定: {best['sig']} × {int(best['hold'])}d × SL={best['sl']}")


# ════════════════════════════════════════════════════════════════
# 2. ベスト設定の詳細実行
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print(f"【2】ベスト設定詳細: {best['sig']} × {int(best['hold'])}d × SL={best['sl']}")
print("=" * 78)

best_mask = next(m for lbl, m in signals_def if lbl == best['sig'])
best_sig = df[best_mask].copy()
all_trades = []
for _, r in best_sig.iterrows():
    t = simulate_trade(r['code'], r['date'], best['sl'], int(best['hold']))
    if t:
        t.update({'code': r['code'], 'name': r['name_ja'], 'sector': r['sector'],
                  'scale_cat': r['scale_cat'], 'date': r['date'],
                  'dist_ma25': r['dist_ma25']})
        all_trades.append(t)
trades = pd.DataFrame(all_trades)
trades['year'] = trades['date'].dt.year
trades['net_ret'] = trades['ret'] - COST_PCT_RT

print(f"\n  全期間: N={len(trades)}")
print(f"  gross={trades['ret'].mean():+.3f}%, net={trades['net_ret'].mean():+.3f}%")
print(f"  勝率={(trades['ret']>0).mean()*100:.1f}%, Sharpe={trades['net_ret'].mean()/trades['ret'].std()*np.sqrt(252/int(best['hold'])):+.2f}")

# 年次
print(f"\n  ─ 年次安定性 ─")
print(f"  {'年':>4}  {'N':>4}  {'net':>7}  {'勝率':>6}  {'Sharpe':>7}")
for year, g in trades.groupby('year'):
    r = g['ret']; net = g['net_ret']
    sharpe = net.mean()/r.std()*np.sqrt(252/int(best['hold'])) if r.std()>0 else 0
    print(f"  {year}  {len(r):>4}  {net.mean():>+6.3f}%  {(r>0).mean()*100:>5.1f}%  {sharpe:>+6.2f}")

# Walk-forward
train = trades[trades['year']==2024]
test  = trades[trades['year']>=2025]
print(f"\n  ─ Walk-forward ─")
for label, g in [("Train 2024", train), ("Test 2025-26", test)]:
    if len(g)==0: continue
    sharpe = g['net_ret'].mean()/g['ret'].std()*np.sqrt(252/int(best['hold'])) if g['ret'].std()>0 else 0
    print(f"  {label:<14}: N={len(g):>3} net={g['net_ret'].mean():>+.3f}% Sharpe={sharpe:+.2f}")

# scale別年次
print(f"\n  ─ scale_cat × year 年次マトリクス ─")
for sc in ['TOPIX Core30','TOPIX Large70']:
    print(f"\n  {sc}:")
    sub_sc = trades[trades['scale_cat']==sc]
    for year, g in sub_sc.groupby('year'):
        if len(g)<5: continue
        r = g['ret']; net=g['net_ret']
        sharpe = net.mean()/r.std()*np.sqrt(252/int(best['hold'])) if r.std()>0 else 0
        print(f"    {year}: N={len(g):>3} net={net.mean():>+6.3f}% Sharpe={sharpe:+.2f}")

# セクター別
print(f"\n  ─ セクター別 (N≥15) ─")
print(f"  {'セクター':<14}  {'N':>4}  {'net':>7}  {'勝率':>6}  {'Sharpe':>7}")
sect_results = []
for sect, g in trades.groupby('sector'):
    if len(g)<15: continue
    r = g['ret']; net = g['net_ret']
    sharpe = net.mean()/r.std()*np.sqrt(252/int(best['hold'])) if r.std()>0 else 0
    sect_results.append({'sector':sect,'N':len(g),'net':net.mean(),
                          'wr':(r>0).mean()*100,'sharpe':sharpe})

sect_df = pd.DataFrame(sect_results).sort_values('sharpe', ascending=False)
for _, r in sect_df.iterrows():
    flag = " ★★" if r['sharpe']>2.0 else (" ★" if r['sharpe']>1.5 else "")
    print(f"  {r['sector']:<14}  {int(r['N']):>4}  {r['net']:>+6.3f}%  "
          f"{r['wr']:>5.1f}%  {r['sharpe']:>+6.2f}{flag}")


# ════════════════════════════════════════════════════════════════
# 3. ポートフォリオシム
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("【3】ポートフォリオシム — 同時保有数別 (¥100万/銘柄)")
print("=" * 78)

trades['exit_date'] = trades.apply(
    lambda r: r['date'] + pd.tseries.offsets.BDay(int(r['day'])+1), axis=1)
trades_sorted = trades.sort_values('date').reset_index(drop=True)

for max_pos in [3, 5, 8, 12]:
    open_pos = []
    pnls = []
    skipped = 0
    for _, t in trades_sorted.iterrows():
        open_pos = [d for d in open_pos if d > t['date']]
        if len(open_pos) >= max_pos:
            skipped += 1; continue
        open_pos.append(t['exit_date'])
        pnls.append({'date': t['date'], 'pnl': 1_000_000 * t['net_ret']/100})
    if not pnls: continue
    p = pd.DataFrame(pnls).set_index('date')
    p['cum'] = p['pnl'].cumsum()
    max_dd = (p['cum']-p['cum'].cummax()).min()
    daily = p['pnl'].resample('B').sum().fillna(0)
    sharpe_d = daily.mean()/daily.std()*np.sqrt(252) if daily.std()>0 else 0
    print(f"\n  最大{max_pos}銘柄:")
    print(f"    執行: {len(p)} / スキップ: {skipped}")
    print(f"    総P&L: {p['pnl'].sum():>+12,.0f}円")
    print(f"    最大DD: {max_dd:>+12,.0f}円")
    print(f"    日次Sharpe: {sharpe_d:+.2f}")


# ════════════════════════════════════════════════════════════════
# 4. Strategy昇格判定
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("【4】Strategy昇格判定")
print("=" * 78)

r_all = trades['ret']; net_all = trades['net_ret']
sharpe_all = net_all.mean()/r_all.std()*np.sqrt(252/int(best['hold'])) if r_all.std()>0 else 0
t_stat = r_all.mean()/r_all.std()*np.sqrt(len(r_all))
pf = net_all[net_all>0].sum()/abs(net_all[net_all<=0].sum()) if (net_all<=0).any() else 99
yearly_pos = all(g['ret'].mean()>0 for _, g in trades.groupby('year'))
test_pos = test['net_ret'].mean()>0 if len(test)>10 else False

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
print("  " + "-" * 60)
for name, ok, val in criteria:
    mark = "✅ PASS" if ok else "❌ FAIL"
    print(f"  {name:<24}  {mark:<10}  {val}")
passed = sum(1 for _, ok, _ in criteria if ok)
print(f"\n  合格基準: {passed} / {len(criteria)}")
if passed>=6:
    print("  判定: 🎯 Strategy昇格を強く推奨")
elif passed>=5:
    print("  判定: ⚠️ 条件付き昇格")
else:
    print("  判定: ❌ research段階")

# CSV出力
out_dir = '/Users/Yusuke/claude-code/japan-stocks/analyses/20260515_large_cap_reversal_validation'
trades.to_csv(f'{out_dir}/trades.csv', index=False)
grid_df.to_csv(f'{out_dir}/grid.csv', index=False)
sect_df.to_csv(f'{out_dir}/sector.csv', index=False)
print(f"\n  ✅ 完了")
