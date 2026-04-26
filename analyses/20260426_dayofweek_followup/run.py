"""
曜日効果 深掘り3本立て
  A. 水曜 非鉄ロング  (寄→引 コスト後バックテスト)
  B. 月曜 海運ショート (寄→引 コスト後バックテスト)
  C. 木曜ON × LMEシグナル条件付き分解
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '20260421_common'))
import mdutil as U
import pandas as pd
import numpy as np
from scipy import stats
import pymysql

# ── 設定 ────────────────────────────────────────────────
COST = U.COST_BPS          # 4.0 bps (片道2bps×往復)
COST_SHORT = U.COST_BPS    # イントラ当日ショートも同じコスト

# ── データロード ─────────────────────────────────────────
def load_basket_daily(basket):
    """バスケット等加重の日次 open/close → ON/Intra/Full bps"""
    frames = []
    for sym, _ in basket:
        df = U.load_jp_daily(sym)
        df['on']    = (df['open']  / df['close'].shift(1) - 1) * 10_000
        df['intra'] = (df['close'] / df['open']           - 1) * 10_000
        df['full']  = (df['close'] / df['close'].shift(1) - 1) * 10_000
        frames.append(df[['on','intra','full']].dropna())
    combined = pd.concat(frames).groupby(level=0).mean()
    combined['dow'] = pd.to_datetime(combined.index).dayofweek
    return combined

def load_lme_daily():
    """CMCU3 日次終値 → 前日比 bps"""
    conn = pymysql.connect(**U.NAS_CONFIG)
    df = pd.read_sql(
        "SELECT trade_date, close FROM daily_data WHERE symbol='CMCU3' ORDER BY trade_date",
        conn)
    conn.close()
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df = df.set_index('trade_date').sort_index()
    df['lme_ret'] = (df['close'] / df['close'].shift(1) - 1) * 100  # %
    return df[['lme_ret']].dropna()

def st(arr):
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 5:
        return {}
    t, _ = stats.ttest_1samp(arr, 0)
    cum = arr.cumsum()
    dd = (cum - np.maximum.accumulate(cum)).min()
    return dict(n=len(arr), mean=arr.mean(), t=t,
                sharpe=arr.mean()/arr.std()*np.sqrt(252) if arr.std()>0 else 0,
                maxdd=dd, wr=(arr>0).mean()*100, total=arr.sum())

# ── データ取得 ───────────────────────────────────────────
print("データロード中...")
nonfer   = load_basket_daily(U.NONFERROUS)
shipping = load_basket_daily(U.SHIPPING)
core5    = load_basket_daily(U.CORE5)
semicon  = load_basket_daily(U.SEMICON)
energy   = load_basket_daily(U.ENERGY)
lme      = load_lme_daily()
print(f"  非鉄 N={len(nonfer)}, 海運 N={len(shipping)}, LME N={len(lme)}")

# ── A. 水曜非鉄ロング ────────────────────────────────────
wed_nonfer = nonfer[nonfer['dow'] == 2].copy()
wed_nonfer['net'] = wed_nonfer['intra'] - COST
wed_nonfer['year_half'] = ['H1' if pd.Timestamp(d) < pd.Timestamp('2025-10-01') else 'H2'
                            for d in wed_nonfer.index]

stA = st(wed_nonfer['net'])
stA_H1 = st(wed_nonfer.loc[wed_nonfer['year_half']=='H1','net'])
stA_H2 = st(wed_nonfer.loc[wed_nonfer['year_half']=='H2','net'])

print(f"\n[A] 水曜 非鉄ロング (コスト後)")
print(f"  全期間 N={stA['n']} mean={stA['mean']:+.1f}bps t={stA['t']:+.2f} Sharpe={stA['sharpe']:+.2f} WR={stA['wr']:.1f}% MaxDD={stA['maxdd']:+.0f}")
print(f"  H1     N={stA_H1['n']} mean={stA_H1['mean']:+.1f}bps t={stA_H1['t']:+.2f}")
print(f"  H2     N={stA_H2['n']} mean={stA_H2['mean']:+.1f}bps t={stA_H2['t']:+.2f}")

# ── B. 月曜海運ショート ──────────────────────────────────
mon_ship = shipping[shipping['dow'] == 0].copy()
mon_ship['net'] = -mon_ship['intra'] - COST_SHORT   # ショート = 符号反転
mon_ship['year_half'] = ['H1' if pd.Timestamp(d) < pd.Timestamp('2025-10-01') else 'H2'
                          for d in mon_ship.index]

stB = st(mon_ship['net'])
stB_H1 = st(mon_ship.loc[mon_ship['year_half']=='H1','net'])
stB_H2 = st(mon_ship.loc[mon_ship['year_half']=='H2','net'])

print(f"\n[B] 月曜 海運ショート (コスト後)")
print(f"  全期間 N={stB['n']} mean={stB['mean']:+.1f}bps t={stB['t']:+.2f} Sharpe={stB['sharpe']:+.2f} WR={stB['wr']:.1f}% MaxDD={stB['maxdd']:+.0f}")
print(f"  H1     N={stB_H1['n']} mean={stB_H1['mean']:+.1f}bps t={stB_H1['t']:+.2f}")
print(f"  H2     N={stB_H2['n']} mean={stB_H2['mean']:+.1f}bps t={stB_H2['t']:+.2f}")

# ── C. 木曜ON × LME条件分解 ──────────────────────────────
# 木曜ONに対応するLMEは前営業日 (水曜) のLME終値変化率
# lme.index はLME営業日 → pd.Seriesとして前日比をシフト
lme_shifted = lme['lme_ret'].copy()

def get_lme_for_date(d):
    """日付dの前営業日LMEリターンを返す"""
    d_ts = pd.Timestamp(d)
    prev = lme[lme.index < d_ts]
    if len(prev) == 0:
        return np.nan
    return prev.iloc[-1]['lme_ret']

# 各セクターの木曜ON
sectors_thu = {
    'CORE5':    core5[core5['dow']==3]['on'],
    'NONFER':   nonfer[nonfer['dow']==3]['on'],
    'SEMICON':  semicon[semicon['dow']==3]['on'],
    'ENERGY':   energy[energy['dow']==3]['on'],
    'SHIPPING': shipping[shipping['dow']==3]['on'],
}

thu_records = []
for sec, ser in sectors_thu.items():
    for d, v in ser.items():
        lme_val = get_lme_for_date(d)
        if np.isnan(lme_val):
            continue
        if lme_val >= 1.0:
            cond = 'LME>=+1%'
        elif lme_val <= -1.0:
            cond = 'LME<=-1%'
        else:
            cond = '-1%<LME<+1%'
        thu_records.append({'sector': sec, 'cond': cond, 'on': v, 'lme': lme_val})

thu_df = pd.DataFrame(thu_records)

print(f"\n[C] 木曜ON × LME条件分解")
COND_ORDER = ['LME<=-1%', '-1%<LME<+1%', 'LME>=+1%']
for sec in sectors_thu:
    sub = thu_df[thu_df['sector']==sec]
    print(f"  {sec}:")
    for cond in COND_ORDER:
        arr = sub.loc[sub['cond']==cond,'on'].values
        if len(arr) < 3:
            continue
        t, _ = stats.ttest_1samp(arr, 0)
        print(f"    {cond:18s} N={len(arr):3d} mean={arr.mean():+7.1f}bps t={t:+.2f}")

# ── 可視化 ───────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['Yu Gothic', 'Meiryo', 'MS Gothic', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

fig = plt.figure(figsize=(12, 6.75), facecolor='white')
fig.patch.set_facecolor('white')
fig.text(0.5, 0.98, '曜日効果バックテスト — 水曜非鉄Long / 月曜海運Short / 木曜ON×LME条件分解',
         ha='center', va='top', fontsize=13, fontweight='bold')

# ── Panel A: 水曜非鉄ロング エクイティカーブ ────────────
ax1 = fig.add_axes([0.05, 0.14, 0.27, 0.72])
cum_A = wed_nonfer['net'].cumsum()
ax1.plot(range(len(cum_A)), cum_A.values, color='#27ae60', linewidth=1.8)
ax1.fill_between(range(len(cum_A)), cum_A.values, 0,
                 where=cum_A.values>=0, alpha=0.15, color='#27ae60')
ax1.fill_between(range(len(cum_A)), cum_A.values, 0,
                 where=cum_A.values<0, alpha=0.15, color='#e74c3c')
ax1.axhline(0, color='black', linewidth=0.8)
ax1.set_title('A. 水曜 非鉄バスケット Long\n(寄→引, コスト後)', fontsize=9)
ax1.set_ylabel('累積リターン (bps)', fontsize=8)
ax1.set_xlabel(f'N={stA["n"]}トレード', fontsize=8)
ax1.grid(alpha=0.3)
stat_txt = (f'mean: {stA["mean"]:+.1f}bps\n'
            f't統計: {stA["t"]:+.2f}\n'
            f'Sharpe: {stA["sharpe"]:+.2f}\n'
            f'WR: {stA["wr"]:.0f}%\n'
            f'MaxDD: {stA["maxdd"]:+.0f}bps\n'
            f'H1: {stA_H1["mean"]:+.1f}bps (t={stA_H1["t"]:+.2f})\n'
            f'H2: {stA_H2["mean"]:+.1f}bps (t={stA_H2["t"]:+.2f})')
ax1.text(0.03, 0.97, stat_txt, transform=ax1.transAxes,
         fontsize=7.5, va='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='#f0fff0', alpha=0.8))

# ── Panel B: 月曜海運ショート エクイティカーブ ──────────
ax2 = fig.add_axes([0.37, 0.14, 0.27, 0.72])
cum_B = mon_ship['net'].cumsum()
ax2.plot(range(len(cum_B)), cum_B.values, color='#e74c3c', linewidth=1.8)
ax2.fill_between(range(len(cum_B)), cum_B.values, 0,
                 where=cum_B.values>=0, alpha=0.15, color='#27ae60')
ax2.fill_between(range(len(cum_B)), cum_B.values, 0,
                 where=cum_B.values<0, alpha=0.15, color='#e74c3c')
ax2.axhline(0, color='black', linewidth=0.8)
ax2.set_title('B. 月曜 海運バスケット Short\n(寄→引, コスト後)', fontsize=9)
ax2.set_ylabel('累積リターン (bps)', fontsize=8)
ax2.set_xlabel(f'N={stB["n"]}トレード', fontsize=8)
ax2.grid(alpha=0.3)
stat_txt2 = (f'mean: {stB["mean"]:+.1f}bps\n'
             f't統計: {stB["t"]:+.2f}\n'
             f'Sharpe: {stB["sharpe"]:+.2f}\n'
             f'WR: {stB["wr"]:.0f}%\n'
             f'MaxDD: {stB["maxdd"]:+.0f}bps\n'
             f'H1: {stB_H1["mean"]:+.1f}bps (t={stB_H1["t"]:+.2f})\n'
             f'H2: {stB_H2["mean"]:+.1f}bps (t={stB_H2["t"]:+.2f})')
ax2.text(0.03, 0.97, stat_txt2, transform=ax2.transAxes,
         fontsize=7.5, va='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='#fff0f0', alpha=0.8))

# ── Panel C: 木曜ON × LME条件分解 棒グラフ ──────────────
ax3 = fig.add_axes([0.70, 0.14, 0.28, 0.72])

sec_order = ['CORE5', 'NONFER', 'SEMICON', 'ENERGY', 'SHIPPING']
cond_colors = {'LME<=-1%': '#3498db', '-1%<LME<+1%': '#95a5a6', 'LME>=+1%': '#e74c3c'}
cond_labels = {'LME<=-1%': 'LME≦-1%', '-1%<LME<+1%': '中立', 'LME>=+1%': 'LME≧+1%'}
x = np.arange(len(sec_order))
width = 0.25

for i, cond in enumerate(COND_ORDER):
    means_c = []
    for sec in sec_order:
        arr = thu_df.loc[(thu_df['sector']==sec)&(thu_df['cond']==cond),'on'].values
        means_c.append(arr.mean() if len(arr) >= 3 else 0)
    bars = ax3.bar(x + (i-1)*width, means_c, width,
                   color=cond_colors[cond], alpha=0.8,
                   label=cond_labels[cond])

ax3.axhline(0, color='black', linewidth=0.8)
ax3.set_xticks(x)
ax3.set_xticklabels(sec_order, fontsize=8)
ax3.set_ylabel('平均ONリターン (bps)', fontsize=8)
ax3.set_title('C. 木曜ON × LME前日条件\n(赤=LME大幅上昇日が悪い)', fontsize=9)
ax3.legend(fontsize=7.5, loc='upper right')
ax3.grid(axis='y', alpha=0.3)

# フッター
fig.text(0.99, 0.01, f'データ: {U.START}~{U.END} / 日本株1分足 (Refinitiv) / コスト={COST}bps往復',
         ha='right', va='bottom', fontsize=7, color='gray')

out = os.path.join(os.path.dirname(__file__), 'result.png')
plt.savefig(out, dpi=100, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\nresult.png 保存完了")

# CSV保存
wed_nonfer[['intra','net']].to_csv(
    os.path.join(os.path.dirname(__file__), 'wed_nonfer_trades.csv'))
mon_ship[['intra','net']].to_csv(
    os.path.join(os.path.dirname(__file__), 'mon_shipping_trades.csv'))
thu_df.to_csv(
    os.path.join(os.path.dirname(__file__), 'thu_on_lme_decomp.csv'), index=False)
print("CSV保存完了")
