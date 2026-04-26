"""
戦略設計 2本立て
  S1. 月曜 海運Short  — 実装詳細設計 (月別・レジーム・フィルタ・ポジションサイズ)
  S2. 木曜ON エネルギー — タイミング分解 (Gap vs Intraday, LME条件別)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '20260421_common'))
import mdutil as U
import pandas as pd
import numpy as np
from scipy import stats
import pymysql

COST = U.COST_BPS  # 4.0bps

# ── 共通: データロード ────────────────────────────────────
def load_basket(basket):
    frames = []
    for sym, name in basket:
        df = U.load_jp_daily(sym)
        df['on']    = (df['open']  / df['close'].shift(1) - 1) * 10_000
        df['intra'] = (df['close'] / df['open']           - 1) * 10_000
        df['full']  = (df['close'] / df['close'].shift(1) - 1) * 10_000
        df['sym'] = sym
        frames.append(df[['on','intra','full','sym']].dropna())
    combined = pd.concat(frames).groupby(level=0)[['on','intra','full']].mean()
    combined.index = pd.to_datetime(combined.index)
    combined['dow'] = combined.index.dayofweek
    combined['month'] = combined.index.month
    combined['ym'] = combined.index.to_period('M')
    combined['regime'] = combined.index.map(
        lambda d: 'War' if pd.Timestamp('2026-02-28') <= d <= pd.Timestamp('2026-03-25')
                  else ('Post' if d > pd.Timestamp('2026-03-25') else 'Pre'))
    return combined

def load_lme():
    conn = pymysql.connect(**U.NAS_CONFIG)
    df = pd.read_sql("SELECT trade_date, close FROM daily_data WHERE symbol='CMCU3' ORDER BY trade_date", conn)
    conn.close()
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df = df.set_index('trade_date').sort_index()
    df['lme_ret'] = (df['close'] / df['close'].shift(1) - 1) * 100
    return df

def st(arr):
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 3: return {'n':len(arr),'mean':np.nan,'t':np.nan,'sharpe':np.nan,'maxdd':np.nan,'wr':np.nan}
    t, _ = stats.ttest_1samp(arr, 0)
    cum = arr.cumsum()
    dd = (cum - np.maximum.accumulate(cum)).min()
    return dict(n=len(arr), mean=arr.mean(), t=t,
                sharpe=arr.mean()/arr.std()*np.sqrt(252) if arr.std()>0 else 0,
                maxdd=dd, wr=(arr>0).mean()*100)

print("データロード中...")
shipping = load_basket(U.SHIPPING)
energy   = load_basket(U.ENERGY)
lme      = load_lme()

# ── S1: 月曜海運Short 詳細設計 ────────────────────────────
print("\n" + "="*60)
print("[S1] 月曜 海運Short 詳細設計")
print("="*60)

mon = shipping[shipping['dow']==0].copy()
mon['net'] = -mon['intra'] - COST

# 月別集計
print("\n[月別 P&L]")
monthly_pnl = []
for ym, g in mon.groupby('ym'):
    s = st(g['net'])
    monthly_pnl.append({'ym': str(ym), 'n': s['n'], 'mean': s['mean'], 't': s['t'], 'total': g['net'].sum()})
    flag = ' [WIN]' if s['mean'] > 0 else ' [LOSS]'
    print(f"  {ym}: N={s['n']} mean={s['mean']:+.1f}bps total={g['net'].sum():+.0f}bps{flag}")

# レジーム別集計
print("\n[レジーム別]")
for reg in ['Pre','War','Post']:
    s = st(mon.loc[mon['regime']==reg,'net'])
    print(f"  {reg:5s}: N={s['n']:3.0f} mean={s['mean']:+7.1f}bps t={s['t']:+.2f} WR={s['wr']:.0f}%")

# 追加フィルタ検討: LME前日との組合せ
print("\n[フィルタ検討: 前週金曜LME条件]")
for d, row in mon.iterrows():
    prev_lme = lme[lme.index < pd.Timestamp(d)]
    mon.loc[d,'lme_prev'] = prev_lme.iloc[-1]['lme_ret'] if len(prev_lme) > 0 else np.nan

for label, mask in [
    ('LME前日下落 (<0%)',    mon['lme_prev'] < 0),
    ('LME前日フラット (0-1%)', (mon['lme_prev'] >= 0) & (mon['lme_prev'] < 1)),
    ('LME前日上昇 (>=1%)',   mon['lme_prev'] >= 1),
]:
    s = st(mon.loc[mask,'net'])
    print(f"  {label}: N={s['n']:.0f} mean={s['mean']:+.1f}bps t={s['t']:+.2f}")

# スリッページ感度
print("\n[スリッページ感度 (コスト変化)]")
for extra in [0, 2, 4, 6, 10]:
    arr = -mon['intra'].values - (COST + extra)
    s = st(arr)
    print(f"  コスト {COST+extra:.0f}bps: mean={s['mean']:+.1f}bps Sharpe={s['sharpe']:+.2f}")

stS1 = st(mon['net'])
cum_S1 = mon['net'].cumsum()

# ── S2: 木曜ON エネルギー 詳細設計 ───────────────────────
print("\n" + "="*60)
print("[S2] 木曜ON エネルギー戦略 詳細設計")
print("="*60)

thu = energy[energy['dow']==3].copy()

# 前日LME紐付け
for d, row in thu.iterrows():
    prev_lme = lme[lme.index < pd.Timestamp(d)]
    thu.loc[d,'lme_prev'] = prev_lme.iloc[-1]['lme_ret'] if len(prev_lme) > 0 else np.nan

# LME条件別 ON vs Intraday の分解
print("\n[LME条件別 ON / Intraday / Full 分解]")
print(f"{'条件':20s} {'N':>4} {'ON':>8} {'Intra':>8} {'Full':>8} {'t(ON)':>7}")
for label, mask in [
    ('LME < -1%',      thu['lme_prev'] < -1),
    ('-1% <= LME < 0%', (thu['lme_prev'] >= -1) & (thu['lme_prev'] < 0)),
    ('0% <= LME < 1%',  (thu['lme_prev'] >= 0) & (thu['lme_prev'] < 1)),
    ('LME >= +1%',      thu['lme_prev'] >= 1),
]:
    g = thu[mask]
    so = st(g['on']); si = st(g['intra']); sf = st(g['full'])
    print(f"  {label:20s} {so['n']:4.0f} {so['mean']:+8.1f} {si['mean']:+8.1f} {sf['mean']:+8.1f} {so['t']:+7.2f}")

# LME>=+1% の月別
lme_up_thu = thu[thu['lme_prev'] >= 1.0].copy()
print(f"\n[LME>=+1% 木曜のみ: 月別 ON リターン] N={len(lme_up_thu)}")
for ym, g in lme_up_thu.groupby('ym'):
    print(f"  {ym}: N={len(g)} ON={g['on'].mean():+.1f}bps Intra={g['intra'].mean():+.1f}bps")

# タイミング戦略の比較
print("\n[戦略バリアント比較 (LME>=+1% 木曜)]")
variants = {
    'V1: ON   (前日引→寄, コスト4bps)': lme_up_thu['on'] - COST,
    'V2: Intra(寄→引,   コスト4bps)':   lme_up_thu['intra'] - COST,
    'V3: Full (前日引→引,コスト4bps)':  lme_up_thu['full'] - COST,
}
for label, arr in variants.items():
    s = st(arr)
    print(f"  {label}: N={s['n']:.0f} mean={s['mean']:+.1f}bps Sharpe={s['sharpe']:+.2f} WR={s['wr']:.0f}%")

# レジーム別 (LME>=+1%)
print("\n[レジーム別 ON (LME>=+1%)]")
for reg in ['Pre','War','Post']:
    s = st(lme_up_thu.loc[lme_up_thu['regime']==reg,'on'])
    print(f"  {reg:5s}: N={s['n']:.0f} mean={s['mean']:+.1f}bps t={s['t']:+.2f}")

stS2_on    = st(lme_up_thu['on'] - COST)
stS2_intra = st(lme_up_thu['intra'] - COST)
cum_S2_on    = (lme_up_thu['on'] - COST).cumsum()
cum_S2_intra = (lme_up_thu['intra'] - COST).cumsum()

# ── 可視化 ───────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['Yu Gothic', 'Meiryo', 'MS Gothic', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

fig = plt.figure(figsize=(12, 6.75), facecolor='white')
fig.patch.set_facecolor('white')
fig.text(0.5, 0.98, '戦略設計: 月曜海運Short (S1) / 木曜ONエネルギー (S2)',
         ha='center', va='top', fontsize=13, fontweight='bold')

# ── S1: 月次棒グラフ + エクイティカーブ ──────────────────
ax1 = fig.add_axes([0.05, 0.55, 0.27, 0.35])  # 月次P&L
mp = pd.DataFrame(monthly_pnl).set_index('ym')
colors_bar = ['#27ae60' if v > 0 else '#e74c3c' for v in mp['total']]
ax1.bar(range(len(mp)), mp['total'], color=colors_bar, alpha=0.8, width=0.7)
ax1.axhline(0, color='black', lw=0.8)
ax1.set_title('S1: 月曜海運Short — 月次P&L', fontsize=9)
ax1.set_ylabel('月次累計 (bps)', fontsize=7)
ax1.set_xticks(range(len(mp)))
ax1.set_xticklabels([str(y)[-5:] for y in mp.index], fontsize=6, rotation=45)
ax1.grid(axis='y', alpha=0.3)
win_months = (mp['total'] > 0).sum()
ax1.text(0.02, 0.97, f'勝ち月: {win_months}/{len(mp)}', transform=ax1.transAxes,
         fontsize=8, va='top', bbox=dict(facecolor='white', alpha=0.7))

ax2 = fig.add_axes([0.05, 0.12, 0.27, 0.35])  # エクイティカーブ
ax2.plot(range(len(cum_S1)), cum_S1.values, color='#e74c3c', lw=1.8)
ax2.fill_between(range(len(cum_S1)), cum_S1.values, 0,
                 where=cum_S1.values>=0, alpha=0.15, color='#27ae60')
ax2.fill_between(range(len(cum_S1)), cum_S1.values, 0,
                 where=cum_S1.values<0, alpha=0.15, color='#e74c3c')
ax2.axhline(0, color='black', lw=0.8)
ax2.set_title('S1: 累積エクイティカーブ', fontsize=9)
ax2.set_ylabel('累積リターン (bps)', fontsize=7)
ax2.set_xlabel(f'N={stS1["n"]} / Sharpe={stS1["sharpe"]:+.2f} / MaxDD={stS1["maxdd"]:+.0f}bps', fontsize=7)
ax2.grid(alpha=0.3)
reg_colors = {'Pre':'#95a5a6','War':'#e74c3c','Post':'#f39c12'}
for reg, color in reg_colors.items():
    idx = [i for i,d in enumerate(mon.index) if mon.loc[d,'regime']==reg]
    if idx: ax2.axvspan(min(idx), max(idx), alpha=0.08, color=color, label=reg)
ax2.legend(fontsize=6, loc='upper left')

# ── S2: ON vs Intra 比較 + エクイティカーブ ──────────────
ax3 = fig.add_axes([0.38, 0.55, 0.28, 0.35])
conds = ['LME<-1%', '-1~0%', '0~+1%', '>=+1%']
masks = [
    thu['lme_prev'] < -1,
    (thu['lme_prev'] >= -1) & (thu['lme_prev'] < 0),
    (thu['lme_prev'] >= 0) & (thu['lme_prev'] < 1),
    thu['lme_prev'] >= 1,
]
on_means    = [thu.loc[m,'on'].mean() if m.sum()>2 else 0 for m in masks]
intra_means = [thu.loc[m,'intra'].mean() if m.sum()>2 else 0 for m in masks]
x = np.arange(len(conds))
ax3.bar(x - 0.2, on_means, 0.35, label='ON(前日引→寄)', color='#3498db', alpha=0.8)
ax3.bar(x + 0.2, intra_means, 0.35, label='Intra(寄→引)', color='#e67e22', alpha=0.8)
ax3.axhline(0, color='black', lw=0.8)
ax3.set_xticks(x); ax3.set_xticklabels(conds, fontsize=8)
ax3.set_title('S2: 木曜エネルギー — LME条件別\nON vs Intraday', fontsize=9)
ax3.set_ylabel('平均リターン (bps)', fontsize=7)
ax3.legend(fontsize=7); ax3.grid(axis='y', alpha=0.3)

ax4 = fig.add_axes([0.38, 0.12, 0.28, 0.35])
ax4.plot(range(len(cum_S2_on)), cum_S2_on.values, color='#3498db', lw=1.8, label=f'V1 ON (Sharpe={stS2_on["sharpe"]:+.2f})')
ax4.plot(range(len(cum_S2_intra)), cum_S2_intra.values, color='#e67e22', lw=1.8, ls='--', label=f'V2 Intra (Sharpe={stS2_intra["sharpe"]:+.2f})')
ax4.axhline(0, color='black', lw=0.8)
ax4.set_title(f'S2: LME>=+1% 木曜 N={len(lme_up_thu)}', fontsize=9)
ax4.set_ylabel('累積リターン (bps)', fontsize=7)
ax4.legend(fontsize=7); ax4.grid(alpha=0.3)

# ── 戦略カード (右パネル) ──────────────────────────────────
ax5 = fig.add_axes([0.70, 0.08, 0.28, 0.85])
ax5.axis('off')
card = (
    "■ S1: 月曜 海運Short\n"
    f"  対象: 9101/9104/9107 等加重\n"
    f"  エントリー: 月曜 9:00 寄付成行\n"
    f"  決済: 月曜 15:30 引成\n"
    f"  頻度: 週1回 (月曜毎)\n"
    f"  コスト想定: 4bps往復\n\n"
    f"  N={stS1['n']:.0f}  mean={stS1['mean']:+.1f}bps\n"
    f"  Sharpe={stS1['sharpe']:+.2f}  WR={stS1['wr']:.0f}%\n"
    f"  MaxDD={stS1['maxdd']:+.0f}bps\n"
    f"  H1/H2双方プラス → 安定\n\n"
    "─────────────────────────\n\n"
    "■ S2: 木曜 エネルギーON\n"
    f"  対象: 1605/5016/5020 等加重\n"
    f"  シグナル: LME>=+1% (JST3:00確認)\n"
    f"  エントリー: 木曜 9:00 寄付成行\n"
    f"     ※ ONリターンは既に織込済\n"
    f"  決済候補:\n"
    f"    V1(ON):前日引→寄 理論値\n"
    f"    V2(Intra): 寄→引 実現可能\n\n"
    f"  V2 Sharpe={stS2_intra['sharpe']:+.2f}\n"
    f"  V2 mean={stS2_intra['mean']:+.1f}bps\n"
    f"  発生頻度: 月1-2回\n\n"
    "─────────────────────────\n"
    "  ★ S1は即実装可能\n"
    "  ★ S2はIntraで近似実装"
)
ax5.text(0.02, 0.98, card, transform=ax5.transAxes, fontsize=8.5,
         va='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='#f8f9fa', alpha=0.9))

fig.text(0.99, 0.01, f'データ: {U.START}~{U.END} / コスト={COST}bps往復',
         ha='right', va='bottom', fontsize=7, color='gray')

out = os.path.join(os.path.dirname(__file__), 'result.png')
plt.savefig(out, dpi=100, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\nresult.png 保存完了")

# CSV
mon[['intra','net','regime','lme_prev']].to_csv(
    os.path.join(os.path.dirname(__file__), 'S1_mon_shipping.csv'))
lme_up_thu[['on','intra','full','regime','lme_prev']].to_csv(
    os.path.join(os.path.dirname(__file__), 'S2_thu_energy_lme_up.csv'))
print("CSV保存完了")
