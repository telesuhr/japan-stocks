"""
戦略ポートフォリオ統合分析
  B. Post-War レジーム 全戦略再評価 (ST1/ST2/ST3)
  C. 戦略間相関・複合エクイティカーブ・月次P&L
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '20260421_common'))
import mdutil as U
import pandas as pd
import numpy as np
from scipy import stats
import pymysql

COST = U.COST_BPS  # 4.0 bps

# ── データロード ────────────────────────────────────────────
def load_basket(basket):
    frames = []
    for sym, _ in basket:
        df = U.load_jp_daily(sym)
        df['on']    = (df['open']  / df['close'].shift(1) - 1) * 10_000
        df['intra'] = (df['close'] / df['open']           - 1) * 10_000
        frames.append(df[['on', 'intra']].dropna())
    combined = pd.concat(frames).groupby(level=0).mean()
    combined.index = pd.to_datetime(combined.index)
    combined['dow'] = combined.index.dayofweek
    combined['regime'] = combined.index.map(
        lambda d: 'War'  if pd.Timestamp('2026-02-28') <= d <= pd.Timestamp('2026-03-25')
             else ('Post' if d > pd.Timestamp('2026-03-25') else 'Pre'))
    combined['ym'] = combined.index.to_period('M')
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

def lme_prev(dates, lme_df):
    result = {}
    for d in dates:
        prev = lme_df[lme_df.index < pd.Timestamp(d)]
        result[d] = prev.iloc[-1]['lme_ret'] if len(prev) > 0 else np.nan
    return result

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
core5    = load_basket(U.CORE5)
shipping = load_basket(U.SHIPPING)
energy   = load_basket(U.ENERGY)
lme      = load_lme()
print(f"  CORE5 N={len(core5)}, 海運 N={len(shipping)}, エネルギー N={len(energy)}, LME N={len(lme)}")

# ── 戦略シグナル生成 ────────────────────────────────────────
# ST1: LME >= +1% → CORE5 ON (当日ON, 前日引→寄)
lme_map_c5 = lme_prev(core5.index, lme)
core5['lme_prev'] = pd.Series(lme_map_c5)
st1 = core5[core5['lme_prev'] >= 1.0].copy()
st1['net'] = st1['on'] - COST

# ST2: 月曜 海運 Short (イントラデイ)
st2 = shipping[shipping['dow'] == 0].copy()
st2['net'] = -st2['intra'] - COST

# ST3: 木曜 エネルギー Intra × LME >= +1% (V2)
thu_en = energy[energy['dow'] == 3].copy()
lme_map_en = lme_prev(thu_en.index, lme)
thu_en['lme_prev'] = pd.Series(lme_map_en)
st3 = thu_en[thu_en['lme_prev'] >= 1.0].copy()
st3['net'] = st3['intra'] - COST

# ── B: レジーム別成績 ──────────────────────────────────────
print("\n" + "="*65)
print("[B] Post-War レジーム 全戦略再評価")
print("="*65)

strategies_meta = [
    ("ST1 LME+1%→CORE5 ON", st1),
    ("ST2 月曜海運Short",    st2),
    ("ST3 木曜ENIntra LME+1%", st3),
]

regime_stats = {}  # {strat_name: {regime: stats}}
for name, df in strategies_meta:
    print(f"\n  [{name}]  全体N={len(df)}")
    print(f"  {'レジーム':8s}  {'N':>4}  {'mean':>8}  {'t':>7}  {'Sharpe':>8}  {'WR':>6}")
    regime_stats[name] = {}
    for reg in ['Pre', 'War', 'Post']:
        arr = df.loc[df['regime'] == reg, 'net'].values
        s = st(arr)
        regime_stats[name][reg] = s
        mean_str = f"{s['mean']:+8.1f}bps" if not np.isnan(s['mean']) else "        n/a"
        t_str    = f"{s['t']:+7.2f}"       if not np.isnan(s['t'])    else "    n/a"
        sh_str   = f"{s['sharpe']:+8.2f}"  if not np.isnan(s['sharpe']) else "     n/a"
        wr_str   = f"{s['wr']:5.0f}%"      if not np.isnan(s['wr'])   else "  n/a"
        print(f"  {reg:8s}  {s['n']:4.0f}  {mean_str}  {t_str}  {sh_str}  {wr_str}")

# ── C: ポートフォリオ統合 ────────────────────────────────────
print("\n" + "="*65)
print("[C] 戦略ポートフォリオ統合")
print("="*65)

pnl = pd.DataFrame({
    'ST1': st1['net'],
    'ST2': st2['net'],
    'ST3': st3['net'],
})

print(f"\n  発動日数:  ST1={len(st1)}  ST2={len(st2)}  ST3={len(st3)}")
for a, b in [('ST1', 'ST2'), ('ST1', 'ST3'), ('ST2', 'ST3')]:
    n_ov = pnl[[a, b]].dropna().shape[0]
    print(f"  同日発動: {a}&{b} → {n_ov}日")

print("\n  相関 (同日発動ペア):")
for a, b in [('ST1', 'ST2'), ('ST1', 'ST3'), ('ST2', 'ST3')]:
    pair = pnl[[a, b]].dropna()
    if len(pair) >= 5:
        r = pair.corr().iloc[0, 1]
        print(f"    {a} vs {b}: r={r:+.3f}  N={len(pair)}")
    else:
        print(f"    {a} vs {b}: N={len(pair)} (少なすぎて算出不可)")

# 複合エクイティ: 各戦略は独立1単位、同日複数発動は加算
combined = pnl.sum(axis=1, min_count=1).dropna().sort_index()
s_comb = st(combined.values)
print(f"\n  複合ポートフォリオ (各戦略独立1単位):")
print(f"    発動日数 (any): {len(combined)}")
print(f"    mean/発動日:    {s_comb['mean']:+.1f}bps")
print(f"    Sharpe:         {s_comb['sharpe']:+.2f}")
print(f"    WR:             {s_comb['wr']:.0f}%")
print(f"    MaxDD:          {s_comb['maxdd']:+.0f}bps")
print(f"    期間累計:        {combined.sum():+.0f}bps")

# 月別
monthly_comb = combined.groupby(combined.index.to_period('M')).sum()
print(f"\n  月別 複合P&L:")
for ym, v in monthly_comb.items():
    flag = ' [WIN]' if v > 0 else ' [LOSS]'
    print(f"    {ym}: {v:+.0f}bps{flag}")

# ── 可視化 ───────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['Yu Gothic', 'Meiryo', 'MS Gothic', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

fig = plt.figure(figsize=(12, 6.75), facecolor='white')
fig.patch.set_facecolor('white')
fig.text(0.5, 0.985, '戦略ポートフォリオ統合 — レジーム再評価 / 相関 / 複合エクイティカーブ',
         ha='center', va='top', fontsize=12, fontweight='bold')

# ── Panel 1: レジーム × 戦略 grouped bar ─────────────────────
ax1 = fig.add_axes([0.05, 0.12, 0.28, 0.77])
reg_colors = {'Pre': '#3498db', 'War': '#e74c3c', 'Post': '#f39c12'}
st_labels  = ['ST1\nCORE5 ON', 'ST2\n海運Short', 'ST3\nENIntra']
x = np.arange(len(st_labels))
width = 0.22
for i, (reg, color) in enumerate(reg_colors.items()):
    means = []
    for name, _ in strategies_meta:
        s = regime_stats[name][reg]
        means.append(s['mean'] if not np.isnan(s['mean']) else 0)
    bars = ax1.bar(x + (i - 1) * width, means, width,
                   label=reg, color=color, alpha=0.85)
    for bar, m in zip(bars, means):
        if abs(m) > 5:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (3 if m >= 0 else -12),
                     f'{m:+.0f}', ha='center', va='bottom', fontsize=6.5)
ax1.axhline(0, color='black', lw=0.8)
ax1.set_xticks(x); ax1.set_xticklabels(st_labels, fontsize=8.5)
ax1.set_ylabel('平均リターン (bps)', fontsize=8)
ax1.set_title('B. レジーム別 成績\n(青=Pre/赤=War/橙=Post)', fontsize=9)
ax1.legend(fontsize=7.5, loc='upper right')
ax1.grid(axis='y', alpha=0.3)

# ── Panel 2: 複合エクイティカーブ ───────────────────────────
ax2 = fig.add_axes([0.38, 0.12, 0.35, 0.77])
cum = combined.cumsum()
ax2.plot(range(len(cum)), cum.values, color='#2c3e50', lw=1.8)
ax2.fill_between(range(len(cum)), cum.values, 0,
                 where=cum.values >= 0, alpha=0.12, color='#27ae60')
ax2.fill_between(range(len(cum)), cum.values, 0,
                 where=cum.values < 0, alpha=0.12, color='#e74c3c')
# レジームシェード
reg_map = {'Pre': '#95a5a6', 'War': '#e74c3c', 'Post': '#f39c12'}
for reg, color in reg_map.items():
    idx = [i for i, d in enumerate(combined.index) if
           (d < pd.Timestamp('2026-02-28') and reg == 'Pre') or
           (pd.Timestamp('2026-02-28') <= d <= pd.Timestamp('2026-03-25') and reg == 'War') or
           (d > pd.Timestamp('2026-03-25') and reg == 'Post')]
    if idx:
        ax2.axvspan(min(idx), max(idx), alpha=0.07, color=color, label=reg)
ax2.axhline(0, color='black', lw=0.8)
ax2.set_title('C. 複合エクイティカーブ\n(ST1+ST2+ST3 独立1単位)', fontsize=9)
ax2.set_ylabel('累積リターン (bps)', fontsize=8)
ax2.set_xlabel(f'N={len(combined)}発動日', fontsize=8)
ax2.legend(fontsize=7, loc='upper left')
ax2.grid(alpha=0.3)
stat_txt = (f'total: {combined.sum():+.0f}bps\n'
            f'mean/day: {s_comb["mean"]:+.1f}bps\n'
            f'Sharpe: {s_comb["sharpe"]:+.2f}\n'
            f'WR: {s_comb["wr"]:.0f}%\n'
            f'MaxDD: {s_comb["maxdd"]:+.0f}bps')
ax2.text(0.02, 0.98, stat_txt, transform=ax2.transAxes,
         fontsize=7.5, va='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='#f8f9fa', alpha=0.8))

# ── Panel 3: 月次P&L ─────────────────────────────────────────
ax3 = fig.add_axes([0.78, 0.12, 0.20, 0.77])
ym_labels = [str(y)[-5:] for y in monthly_comb.index]
colors_m = ['#27ae60' if v > 0 else '#e74c3c' for v in monthly_comb.values]
ax3.barh(range(len(monthly_comb)), monthly_comb.values, color=colors_m, alpha=0.85)
ax3.axvline(0, color='black', lw=0.8)
ax3.set_yticks(range(len(monthly_comb)))
ax3.set_yticklabels(ym_labels, fontsize=7.5)
ax3.set_xlabel('月次合計 (bps)', fontsize=7.5)
ax3.set_title('月次P&L\n複合', fontsize=9)
ax3.grid(axis='x', alpha=0.3)
win_m = (monthly_comb > 0).sum()
ax3.text(0.5, -0.07, f'勝ち月: {win_m}/{len(monthly_comb)}',
         transform=ax3.transAxes, ha='center', fontsize=8)

fig.text(0.99, 0.005, f'データ: {U.START}~{U.END} / コスト={COST}bps往復',
         ha='right', va='bottom', fontsize=6.5, color='gray')

out = os.path.join(os.path.dirname(__file__), 'result.png')
plt.savefig(out, dpi=100, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\nresult.png 保存完了")

# CSV
pnl.to_csv(os.path.join(os.path.dirname(__file__), 'strategy_pnl.csv'))
pd.DataFrame({'ym': monthly_comb.index.astype(str), 'combined_bps': monthly_comb.values}).to_csv(
    os.path.join(os.path.dirname(__file__), 'monthly_combined.csv'), index=False)
print("CSV保存完了")
