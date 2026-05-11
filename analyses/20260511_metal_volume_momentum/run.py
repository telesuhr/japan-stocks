"""
金属製品セクター 出来高順張り戦略 — 検証
=========================================
分析日: 2026-05-11
対象:   プライム「金属製品」セクター + 周辺セクター比較 × 2024-2026

【仮説】
  先行分析で「金属製品 vol≥3x 翌日+1.28% / 低出来高比+1.26%」を発見。
  → 大商い × 翌日 LONG が機能するか？日中エントリーの再現性は？

【検証軸】
  1. ベースライン: 金属製品 vol_ratio別 翌日リターン
  2. 価格方向条件: 当日陽線/陰線で違いはあるか
  3. 出来高倍率閾値の最適化
  4. 実SL/TPバックテスト (日次high/lowで判定)
  5. 銘柄別の効き方
  6. 周辺セクター比較 (その他製品, ゴム製品)
  7. Walk-forward
  8. Strategy昇格判定
"""
import psycopg2
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

PG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}
COST_PCT_RT = 0.10
START = '2024-01-04'
END   = '2026-05-08'
TARGET_SECTORS = ['金属製品', 'その他製品', 'ゴム製品']

def q(sql):
    conn = psycopg2.connect(**PG)
    df = pd.read_sql(sql, conn)
    conn.close()
    return df

print("=" * 78)
print("  金属製品セクター 出来高順張り戦略 — 検証")
print("=" * 78)

# データ取得
sectors_in = "','".join(TARGET_SECTORS)
sql = f"""
WITH base AS (
    SELECT d.code, s.name_ja, s.sector33_nm AS sector,
           d.date, d.adj_open, d.adj_close, d.adj_high, d.adj_low,
           d.adj_volume, d.turnover_value
    FROM stocks_daily d
    JOIN symbol_master s ON s.code5 = d.code
    WHERE s.market = '0111' AND s.sector33_nm IN ('{sectors_in}')
      AND d.date BETWEEN '{START}' AND '{END}'
      AND d.adj_close > 0 AND d.adj_open > 0
),
ind AS (
    SELECT *,
        AVG(adj_volume) OVER (PARTITION BY code ORDER BY date
                              ROWS BETWEEN 21 PRECEDING AND 2 PRECEDING) AS vol_ma20,
        LEAD(adj_open, 1) OVER (PARTITION BY code ORDER BY date) AS next_open,
        LEAD(adj_close, 1) OVER (PARTITION BY code ORDER BY date) AS next_close,
        LEAD(adj_high, 1) OVER (PARTITION BY code ORDER BY date) AS next_high,
        LEAD(adj_low, 1) OVER (PARTITION BY code ORDER BY date) AS next_low,
        LEAD(adj_close, 3) OVER (PARTITION BY code ORDER BY date) AS close_3d,
        LEAD(adj_close, 5) OVER (PARTITION BY code ORDER BY date) AS close_5d
    FROM base
)
SELECT * FROM ind WHERE vol_ma20 IS NOT NULL AND vol_ma20 > 0 AND next_open IS NOT NULL
"""

df = q(sql)
for c in ['adj_open','adj_close','adj_high','adj_low','adj_volume','turnover_value','vol_ma20',
          'next_open','next_close','next_high','next_low','close_3d','close_5d']:
    df[c] = pd.to_numeric(df[c], errors='coerce')
df['date'] = pd.to_datetime(df['date'])

df['vol_ratio'] = df['adj_volume'] / df['vol_ma20']
df['day_ret']   = (df['adj_close'] / df['adj_open']  - 1) * 100
df['fwd_open_to_close'] = (df['next_close'] / df['next_open'] - 1) * 100  # 翌日日中
df['fwd_close_to_close']= (df['next_close'] / df['adj_close']  - 1) * 100  # 引→翌引
df['fwd_3d']    = (df['close_3d'] / df['next_open'] - 1) * 100  # 翌寄→3日後引
df['fwd_5d']    = (df['close_5d'] / df['next_open'] - 1) * 100  # 翌寄→5日後引

# 流動性フィルタ
df = df[df['turnover_value'] >= 300_000_000]  # 3億+（金属製品は中小型多いため緩めに）

print(f"\n  サンプル: {len(df):,}行 × {df['code'].nunique()}銘柄")
for s in TARGET_SECTORS:
    n = len(df[df['sector']==s])
    print(f"    {s}: {n:,}行 × {df[df['sector']==s]['code'].nunique()}銘柄")


# ════════════════════════════════════════════════════════════════
# 1. vol_ratio別 翌日リターン（金属製品のみ）
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("【1】金属製品セクター — vol_ratio別 翌日リターン")
print("=" * 78)

metal = df[df['sector']=='金属製品'].copy()
print(f"\n  対象: {len(metal):,}行 × {metal['code'].nunique()}銘柄")

bins = [(0.0, 0.5), (0.5, 0.8), (0.8, 1.2), (1.2, 1.5), (1.5, 2.0),
        (2.0, 3.0), (3.0, 5.0), (5.0, 99.0)]

print(f"\n  {'vol_ratio':<14}  {'N':>6}  {'翌日(寄→引)':>11}  {'勝率':>6}  "
      f"{'翌日(引→引)':>11}  {'3日後':>9}  {'5日後':>9}")
print("  " + "-" * 75)
for lo, hi in bins:
    sub = metal[(metal['vol_ratio'] >= lo) & (metal['vol_ratio'] < hi)]
    if len(sub) < 10:
        continue
    r_oc = sub['fwd_open_to_close'].dropna()
    r_cc = sub['fwd_close_to_close'].dropna()
    r_3d = sub['fwd_3d'].dropna()
    r_5d = sub['fwd_5d'].dropna()
    label = f"{lo:.1f}x〜{hi:.1f}x" if hi < 99 else f"{lo:.1f}x+"
    print(f"  {label:<14}  {len(sub):>6}  {r_oc.mean():>+10.3f}%  "
          f"{(r_oc>0).mean()*100:>5.1f}%  "
          f"{r_cc.mean():>+10.3f}%  "
          f"{r_3d.mean():>+8.3f}%  {r_5d.mean():>+8.3f}%")


# ════════════════════════════════════════════════════════════════
# 2. 当日陽線/陰線別
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("【2】金属製品 vol≥3x × 当日値動き別")
print("=" * 78)

big_vol = metal[metal['vol_ratio'] >= 3.0]
print(f"\n  N={len(big_vol)}")

direction_buckets = [
    ('陽線(+1%超)',    big_vol['day_ret'] >  1.0),
    ('陽線(0〜+1%)',   (big_vol['day_ret'] > 0) & (big_vol['day_ret'] <= 1.0)),
    ('陰線(-1%〜0)',  (big_vol['day_ret'] >= -1.0) & (big_vol['day_ret'] < 0)),
    ('陰線(-1%超下)',  big_vol['day_ret'] < -1.0),
]
print(f"\n  {'当日値動き':<16}  {'N':>4}  {'翌日(寄→引)':>11}  {'勝率':>6}  "
      f"{'翌日(引→引)':>11}  {'3日後':>9}  {'5日後':>9}")
print("  " + "-" * 75)
for label, mask in direction_buckets:
    sub = big_vol[mask]
    if len(sub) < 5:
        continue
    r_oc = sub['fwd_open_to_close'].dropna()
    r_cc = sub['fwd_close_to_close'].dropna()
    r_3d = sub['fwd_3d'].dropna()
    r_5d = sub['fwd_5d'].dropna()
    print(f"  {label:<16}  {len(sub):>4}  {r_oc.mean():>+10.3f}%  "
          f"{(r_oc>0).mean()*100:>5.1f}%  "
          f"{r_cc.mean():>+10.3f}%  "
          f"{r_3d.mean():>+8.3f}%  {r_5d.mean():>+8.3f}%")


# ════════════════════════════════════════════════════════════════
# 3. 実SL/TPバックテスト (翌日のhigh/lowベース)
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("【3】実SL/TPバックテスト (翌日high/lowで判定、コスト後)")
print("=" * 78)

# シグナル: vol≥3x × 当日陽線 → 翌寄り買い、翌日中にSL/TPなければ翌日引け決済
def simulate_1day(row, sl_pct, tp_pct):
    """1日保有のSL/TP判定"""
    if pd.isna(row['next_open']):
        return None
    entry = row['next_open']
    sl_price = entry * (1 + sl_pct/100) if sl_pct else None
    tp_price = entry * (1 + tp_pct/100) if tp_pct else None
    hi = row['next_high']
    lo = row['next_low']
    cl = row['next_close']
    if pd.isna(hi) or pd.isna(lo) or pd.isna(cl):
        return None
    sl_hit = sl_price and lo <= sl_price
    tp_hit = tp_price and hi >= tp_price
    if sl_hit and tp_hit:
        return (sl_price/entry - 1) * 100   # 保守的にSL優先
    if sl_hit:
        return (sl_price/entry - 1) * 100
    if tp_hit:
        return (tp_price/entry - 1) * 100
    return (cl/entry - 1) * 100  # 引成


# 各シグナル条件 × SL/TPグリッド
signal_specs = [
    ('vol≥3x × 陽線',     (metal['vol_ratio'] >= 3.0) & (metal['day_ret'] > 0)),
    ('vol≥3x × 大陽線',   (metal['vol_ratio'] >= 3.0) & (metal['day_ret'] > 1.0)),
    ('vol≥2x × 陽線',     (metal['vol_ratio'] >= 2.0) & (metal['day_ret'] > 0)),
    ('vol≥1.5x × 陽線',   (metal['vol_ratio'] >= 1.5) & (metal['day_ret'] > 0)),
    ('vol≥3x × 全方向',   (metal['vol_ratio'] >= 3.0)),
]

print(f"\n  {'シグナル':<22}  {'SL/TP':<14}  {'N':>4}  {'gross':>7}  {'net':>7}  "
      f"{'勝率':>6}  {'Sharpe':>7}")
print("  " + "-" * 80)

best_records = []
for sig_label, mask in signal_specs:
    sig_df = metal[mask].copy()
    for sl_pct, tp_pct in [(None, None), (-3, None), (-3, 5), (-5, 8), (-5, None), (-2, 4)]:
        rets = sig_df.apply(lambda r: simulate_1day(r, sl_pct, tp_pct), axis=1).dropna()
        if len(rets) < 20:
            continue
        net = rets - COST_PCT_RT
        sharpe = net.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
        sl_str = f"{sl_pct}%" if sl_pct else "なし"
        tp_str = f"{tp_pct}%" if tp_pct else "なし"
        param = f"SL{sl_str}/TP{tp_str}"
        flag = " ★★★" if sharpe > 2.5 else (" ★★" if sharpe > 1.5 else "")
        print(f"  {sig_label:<22}  {param:<14}  {len(rets):>4}  "
              f"{rets.mean():>+6.3f}%  {net.mean():>+6.3f}%  "
              f"{(net>0).mean()*100:>5.1f}%  {sharpe:>+6.2f}{flag}")
        best_records.append({
            'sig': sig_label, 'sl': sl_pct, 'tp': tp_pct,
            'N': len(rets), 'net': net.mean(), 'sharpe': sharpe,
            'wr': (net>0).mean()*100,
        })


# ════════════════════════════════════════════════════════════════
# 4. ベスト設定の詳細
# ════════════════════════════════════════════════════════════════
best_df = pd.DataFrame(best_records).sort_values('sharpe', ascending=False)
best = best_df.iloc[0]
print(f"\n" + "=" * 78)
print(f"【4】ベスト設定: {best['sig']} / SL={best['sl']}% TP={best['tp']}%")
print(f"     N={int(best['N'])}, net mean={best['net']:+.3f}%, Sharpe={best['sharpe']:+.2f}")
print("=" * 78)

# 年次安定性
best_mask_label = best['sig']
best_mask = next(m for lbl, m in signal_specs if lbl == best_mask_label)
best_sig_df = metal[best_mask].copy()
best_sig_df['ret'] = best_sig_df.apply(lambda r: simulate_1day(r, best['sl'], best['tp']), axis=1)
best_sig_df = best_sig_df.dropna(subset=['ret'])
best_sig_df['year'] = best_sig_df['date'].dt.year

print(f"\n  ─ 年次パフォーマンス ─")
print(f"  {'年':>4}  {'N':>4}  {'gross':>7}  {'net':>7}  {'勝率':>6}  {'Sharpe':>7}")
print("  " + "-" * 50)
for year, g in best_sig_df.groupby('year'):
    r = g['ret']
    net = r - COST_PCT_RT
    sharpe = net.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0
    print(f"  {year}  {len(r):>4}  {r.mean():>+6.3f}%  {net.mean():>+6.3f}%  "
          f"{(r>0).mean()*100:>5.1f}%  {sharpe:>+6.2f}")


# ════════════════════════════════════════════════════════════════
# 5. 銘柄別
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print(f"【5】銘柄別 ({best_mask_label}, SL={best['sl']}% TP={best['tp']}%)")
print("=" * 78)
print(f"\n  {'銘柄':<22}  {'N':>4}  {'net':>8}  {'勝率':>6}  {'Sharpe':>7}")
print("  " + "-" * 60)
sym_results = []
for code, g in best_sig_df.groupby('code'):
    if len(g) < 5:
        continue
    r = g['ret']
    net = r - COST_PCT_RT
    sharpe = net.mean()/r.std()*np.sqrt(252) if r.std() > 0 else 0
    sym_results.append({
        'code': code, 'name': g['name_ja'].iloc[0],
        'N': len(r), 'net_avg': net.mean(),
        'wr': (net>0).mean()*100, 'sharpe': sharpe,
    })

sym_df = pd.DataFrame(sym_results).sort_values('sharpe', ascending=False)
for _, r in sym_df.iterrows():
    flag = " ★" if r['sharpe'] > 1.5 else ""
    print(f"  {r['name']:<22}  {r['N']:>4}  {r['net_avg']:>+7.3f}%  "
          f"{r['wr']:>5.1f}%  {r['sharpe']:>+6.2f}{flag}")


# ════════════════════════════════════════════════════════════════
# 6. 周辺セクター比較 (その他製品 / ゴム製品)
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("【6】周辺セクター比較 (vol≥3x × 陽線、SL/TPなし)")
print("=" * 78)

print(f"\n  {'セクター':<14}  {'N':>4}  {'gross翌日':>10}  {'net':>8}  {'勝率':>6}  {'Sharpe':>7}")
print("  " + "-" * 65)
for sect in TARGET_SECTORS:
    sub = df[(df['sector']==sect) & (df['vol_ratio']>=3.0) & (df['day_ret']>0)].copy()
    if len(sub) < 10:
        continue
    sub['ret'] = sub.apply(lambda r: simulate_1day(r, None, None), axis=1)
    sub = sub.dropna(subset=['ret'])
    r = sub['ret']
    net = r - COST_PCT_RT
    sharpe = net.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0
    print(f"  {sect:<14}  {len(r):>4}  {r.mean():>+9.3f}%  {net.mean():>+7.3f}%  "
          f"{(r>0).mean()*100:>5.1f}%  {sharpe:>+6.2f}")


# ════════════════════════════════════════════════════════════════
# 7. Walk-forward (2024訓練 / 2025-26テスト)
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("【7】Walk-forward 検証")
print("=" * 78)

train = best_sig_df[best_sig_df['year'] == 2024]
test  = best_sig_df[best_sig_df['year'] >= 2025]
print(f"\n  訓練 (2024):       N={len(train):>3}, net={train['ret'].mean()-COST_PCT_RT:>+.3f}%, "
      f"Sharpe={(train['ret'].mean()-COST_PCT_RT)/train['ret'].std()*np.sqrt(252):+.2f}")
print(f"  テスト (2025-26):   N={len(test):>3}, net={test['ret'].mean()-COST_PCT_RT:>+.3f}%, "
      f"Sharpe={(test['ret'].mean()-COST_PCT_RT)/test['ret'].std()*np.sqrt(252):+.2f}")


# ════════════════════════════════════════════════════════════════
# 8. Strategy昇格判定
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("【8】Strategy昇格判定")
print("=" * 78)

r_all = best_sig_df['ret']
net_all = r_all - COST_PCT_RT
sharpe_all = net_all.mean() / r_all.std() * np.sqrt(252) if r_all.std() > 0 else 0
t_stat = r_all.mean() / r_all.std() * np.sqrt(len(r_all))
pf = net_all[net_all>0].sum() / abs(net_all[net_all<=0].sum()) if (net_all<=0).any() else 99
yearly_pos = all(g['ret'].mean() > 0 for _, g in best_sig_df.groupby('year'))

criteria = [
    ("N≥100",            len(r_all) >= 100,      f"{len(r_all)}"),
    ("net mean > 0",     net_all.mean() > 0,     f"{net_all.mean():+.3f}%"),
    ("Sharpe > 2.0",     sharpe_all > 2.0,       f"{sharpe_all:+.2f}"),
    ("t-stat > 2.0",     t_stat > 2.0,           f"{t_stat:+.2f}"),
    ("PF > 1.3",         pf > 1.3,               f"{pf:.2f}"),
    ("年次全部 net>0",    yearly_pos,             "yearly check"),
]
print(f"\n  ベスト設定: {best_mask_label} / SL={best['sl']}% TP={best['tp']}%")
print(f"  {'基準':<20}  {'結果':<10}  値")
print("  " + "-" * 50)
for name, ok, val in criteria:
    mark = "✅ PASS" if ok else "❌ FAIL"
    print(f"  {name:<20}  {mark:<10}  {val}")
passed = sum(1 for _, ok, _ in criteria if ok)
print(f"\n  合格基準: {passed} / {len(criteria)}")
if passed >= 5:
    print("  判定: 🎯 Strategy昇格を強く推奨")
elif passed >= 4:
    print("  判定: ⚠️  条件付き昇格")
else:
    print("  判定: ❌ research段階")

# CSV出力
out_dir = '/Users/Yusuke/claude-code/japan-stocks/analyses/20260511_metal_volume_momentum'
best_df.to_csv(f'{out_dir}/sltp_grid.csv', index=False)
sym_df.to_csv(f'{out_dir}/by_symbol.csv', index=False)
print(f"\n  ✅ 完了")
