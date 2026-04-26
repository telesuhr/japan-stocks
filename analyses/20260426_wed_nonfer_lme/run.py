"""
A. 水曜 非鉄Long × LME条件フィルタ
  包括分析では H1マイナス・H2集中で保留判定。
  LME前日条件を重ねると H1/H2 ともに安定するか？
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '20260421_common'))
import mdutil as U
import pandas as pd
import numpy as np
from scipy import stats
import pymysql

COST = U.COST_BPS

def load_basket(basket):
    frames = []
    for sym, _ in basket:
        df = U.load_jp_daily(sym)
        df['intra'] = (df['close'] / df['open'] - 1) * 10_000
        frames.append(df[['intra']].dropna())
    combined = pd.concat(frames).groupby(level=0).mean()
    combined.index = pd.to_datetime(combined.index)
    combined['dow'] = combined.index.dayofweek
    combined['year_half'] = ['H1' if d < pd.Timestamp('2025-10-01') else 'H2'
                              for d in combined.index]
    return combined

def load_lme():
    conn = pymysql.connect(**U.NAS_CONFIG)
    df = pd.read_sql(
        "SELECT trade_date, close FROM daily_data WHERE symbol='CMCU3' ORDER BY trade_date", conn)
    conn.close()
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df = df.set_index('trade_date').sort_index()
    df['lme_ret'] = (df['close'] / df['close'].shift(1) - 1) * 100
    return df

def st(arr):
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 3:
        return dict(n=len(arr), mean=np.nan, t=np.nan, sharpe=np.nan, maxdd=np.nan, wr=np.nan)
    t, _ = stats.ttest_1samp(arr, 0)
    cum = arr.cumsum()
    dd = (cum - np.maximum.accumulate(cum)).min()
    return dict(n=len(arr), mean=arr.mean(), t=t,
                sharpe=arr.mean()/arr.std()*np.sqrt(252) if arr.std() > 0 else 0,
                maxdd=dd, wr=(arr > 0).mean() * 100)

print("データロード中...")
nonfer = load_basket(U.NONFERROUS)
lme    = load_lme()

# 水曜フィルタ
wed = nonfer[nonfer['dow'] == 2].copy()
wed['net'] = wed['intra'] - COST

# 前日LME付与
for d in wed.index:
    prev = lme[lme.index < pd.Timestamp(d)]
    wed.loc[d, 'lme_prev'] = prev.iloc[-1]['lme_ret'] if len(prev) > 0 else np.nan

print(f"\n水曜 非鉄 全日: N={len(wed)}")

# ── LME条件別 全期間 ─────────────────────────────────────────
print("\n" + "="*65)
print("[全期間] LME条件別 成績")
print("="*65)

lme_buckets = [
    ('全水曜 (無条件)',        wed.index.map(lambda d: True)),
    ('LME前日 < -1%',         wed['lme_prev'] < -1),
    ('LME前日 -1~0%',         (wed['lme_prev'] >= -1) & (wed['lme_prev'] < 0)),
    ('LME前日 0~+1%',         (wed['lme_prev'] >= 0) & (wed['lme_prev'] < 1)),
    ('LME前日 >= +1%',        wed['lme_prev'] >= 1),
    ('LME前日 >= 0% (緩い条件)', wed['lme_prev'] >= 0),
]

print(f"\n  {'条件':22s}  {'N':>4}  {'mean':>8}  {'t':>7}  {'Sharpe':>8}  {'WR':>6}  {'MaxDD':>8}")
for label, mask in lme_buckets:
    arr = wed.loc[mask, 'net'].values
    s = st(arr)
    if s['n'] < 1:
        continue
    mean_str = f"{s['mean']:+8.1f}bps" if not np.isnan(s['mean']) else "        n/a"
    t_str    = f"{s['t']:+7.2f}"       if not np.isnan(s['t'])    else "    n/a"
    sh_str   = f"{s['sharpe']:+8.2f}"  if not np.isnan(s['sharpe']) else "     n/a"
    wr_str   = f"{s['wr']:5.0f}%"      if not np.isnan(s['wr'])   else "  n/a"
    dd_str   = f"{s['maxdd']:+8.0f}bps" if not np.isnan(s['maxdd']) else "     n/a"
    print(f"  {label:22s}  {s['n']:4.0f}  {mean_str}  {t_str}  {sh_str}  {wr_str}  {dd_str}")

# ── H1/H2 分割 × LME条件 ─────────────────────────────────────
print("\n" + "="*65)
print("[H1/H2分割] LME条件別")
print("="*65)

focus_buckets = [
    ('全水曜 (無条件)',    wed.index.map(lambda d: True)),
    ('LME >= +1%',        wed['lme_prev'] >= 1),
    ('LME >= 0%',         wed['lme_prev'] >= 0),
    ('LME >= -0.5%',      wed['lme_prev'] >= -0.5),
]

h1h2_results = []
for label, mask in focus_buckets:
    sub = wed[mask]
    sH1 = st(sub.loc[sub['year_half'] == 'H1', 'net'].values)
    sH2 = st(sub.loc[sub['year_half'] == 'H2', 'net'].values)
    sAll = st(sub['net'].values)
    h1h2_results.append({'label': label, 'mask': mask, 'H1': sH1, 'H2': sH2, 'All': sAll})
    print(f"\n  [{label}]  N={sAll['n']}")
    for tag, s in [('H1', sH1), ('H2', sH2), ('All', sAll)]:
        if s['n'] < 1: continue
        mean_str = f"{s['mean']:+.1f}bps" if not np.isnan(s['mean']) else "n/a"
        t_str    = f"t={s['t']:+.2f}"     if not np.isnan(s['t'])    else "t=n/a"
        both_pos = ""
        if tag == 'All' and not np.isnan(sH1['mean']) and not np.isnan(sH2['mean']):
            both_pos = " [H1/H2双方プラス]" if sH1['mean'] > 0 and sH2['mean'] > 0 else " [H1/H2どちらかマイナス]"
        print(f"    {tag}: N={s['n']:3.0f}  mean={mean_str}  {t_str}  Sharpe={s['sharpe']:+.2f}{both_pos}")

# ── 可視化 ─────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['Yu Gothic', 'Meiryo', 'MS Gothic', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

fig = plt.figure(figsize=(12, 6.75), facecolor='white')
fig.patch.set_facecolor('white')
fig.text(0.5, 0.985, '水曜 非鉄バスケット Long × LME条件フィルタ — H1/H2安定化検証',
         ha='center', va='top', fontsize=12, fontweight='bold')

# ── Panel 1: LME条件別 mean (全期間) ─────────────────────────
ax1 = fig.add_axes([0.05, 0.12, 0.27, 0.77])
bucket_labels = ['全水曜', 'LME<-1%', '-1~0%', '0~+1%', '>=+1%', '>=0%']
bucket_masks  = [
    wed.index.map(lambda d: True),
    wed['lme_prev'] < -1,
    (wed['lme_prev'] >= -1) & (wed['lme_prev'] < 0),
    (wed['lme_prev'] >= 0) & (wed['lme_prev'] < 1),
    wed['lme_prev'] >= 1,
    wed['lme_prev'] >= 0,
]
means_b = []
ns_b    = []
for mask in bucket_masks:
    s = st(wed.loc[mask, 'net'].values)
    means_b.append(s['mean'] if not np.isnan(s['mean']) else 0)
    ns_b.append(s['n'])

colors_b = ['#2c3e50' if v > 0 else '#e74c3c' for v in means_b]
bars = ax1.bar(range(len(bucket_labels)), means_b, color=colors_b, alpha=0.8)
for bar, n, m in zip(bars, ns_b, means_b):
    ax1.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + (3 if m >= 0 else -14),
             f'N={n}', ha='center', fontsize=7)
ax1.axhline(0, color='black', lw=0.8)
ax1.set_xticks(range(len(bucket_labels)))
ax1.set_xticklabels(bucket_labels, fontsize=8, rotation=20, ha='right')
ax1.set_ylabel('平均リターン コスト後 (bps)', fontsize=8)
ax1.set_title('LME条件別 平均\n(全期間)', fontsize=9)
ax1.grid(axis='y', alpha=0.3)

# ── Panel 2: H1/H2 比較 (無条件 vs LME>=+1%) ─────────────────
ax2 = fig.add_axes([0.37, 0.12, 0.27, 0.77])
conditions_h1h2 = ['無条件', 'LME>=+1%', 'LME>=0%', 'LME>=-0.5%']
h1_means = []
h2_means = []
ns_h1h2  = []
for res in h1h2_results:
    h1_means.append(res['H1']['mean'] if not np.isnan(res['H1']['mean']) else 0)
    h2_means.append(res['H2']['mean'] if not np.isnan(res['H2']['mean']) else 0)
    ns_h1h2.append(res['All']['n'])
x = np.arange(len(conditions_h1h2))
ax2.bar(x - 0.2, h1_means, 0.35, label='H1 (〜2025-09)', color='#3498db', alpha=0.8)
ax2.bar(x + 0.2, h2_means, 0.35, label='H2 (2025-10〜)', color='#e74c3c', alpha=0.8)
for xi, (h1, h2, n) in enumerate(zip(h1_means, h2_means, ns_h1h2)):
    ax2.text(xi, max(h1, h2) + 5, f'N={n}', ha='center', fontsize=7)
ax2.axhline(0, color='black', lw=0.8)
ax2.set_xticks(x)
ax2.set_xticklabels(conditions_h1h2, fontsize=8, rotation=15, ha='right')
ax2.set_ylabel('平均リターン コスト後 (bps)', fontsize=8)
ax2.set_title('H1/H2 安定性比較\n(条件別)', fontsize=9)
ax2.legend(fontsize=8)
ax2.grid(axis='y', alpha=0.3)

# ── Panel 3: エクイティカーブ (無条件 vs LME>=+1%) ───────────
ax3 = fig.add_axes([0.69, 0.12, 0.29, 0.77])
colors_ec = {'無条件': '#95a5a6', 'LME>=+1%': '#27ae60', 'LME>=0%': '#3498db'}
for (label, mask), color in zip(
        [('無条件', wed.index.map(lambda d: True)),
         ('LME>=0%', wed['lme_prev'] >= 0),
         ('LME>=+1%', wed['lme_prev'] >= 1)],
        ['#95a5a6', '#3498db', '#27ae60']):
    sub = wed.loc[mask, 'net'].sort_index()
    s = st(sub.values)
    cum = sub.cumsum()
    ax3.plot(range(len(cum)), cum.values, color=color, lw=1.6,
             label=f'{label} (N={s["n"]}, Sharpe={s["sharpe"]:+.2f})')
ax3.axhline(0, color='black', lw=0.8)
ax3.set_title('エクイティカーブ比較\n(フィルタ強度別)', fontsize=9)
ax3.set_ylabel('累積リターン (bps)', fontsize=8)
ax3.set_xlabel('水曜トレード数', fontsize=8)
ax3.legend(fontsize=7.5, loc='upper left')
ax3.grid(alpha=0.3)

fig.text(0.99, 0.005, f'データ: {U.START}~{U.END} / コスト={COST}bps往復',
         ha='right', va='bottom', fontsize=6.5, color='gray')

out = os.path.join(os.path.dirname(__file__), 'result.png')
plt.savefig(out, dpi=100, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\nresult.png 保存完了")

# CSV
wed[['intra', 'net', 'year_half', 'lme_prev']].to_csv(
    os.path.join(os.path.dirname(__file__), 'wed_nonfer_lme.csv'))
print("CSV保存完了")
