"""
日経先物イントラデイ全体プロファイル
- 分別の平均価格経路（from open）
- 時間帯別ボラ・出来高・方向バイアス
- 後場寄付（12:30）・大引け前の挙動
- 時間帯×ムーブ幅の継続率ヒートマップ
- 曜日効果
"""
import psycopg2
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

PG_CONFIG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}

print("データロード中...")
conn = psycopg2.connect(**PG_CONFIG)
df = pd.read_sql(
    "SELECT timestamp, open, high, low, close, volume "
    "FROM intraday_data WHERE symbol='JNIc1' ORDER BY timestamp", conn)
conn.close()

df['timestamp'] = pd.to_datetime(df['timestamp'])
df['jst'] = df['timestamp'] + pd.Timedelta(hours=9)
df = df.dropna(subset=['close']).copy()
for col in ['open','high','low','close','volume']:
    df[col] = df[col].astype(float)
df = df.set_index('jst').sort_index()
df['hm'] = df.index.hour * 60 + df.index.minute
df['date'] = df.index.date

# 日中のみ
day_mask = ((df['hm'] >= 8*60+45) & (df['hm'] <= 11*60+30)) | \
           ((df['hm'] >= 12*60+30) & (df['hm'] <= 15*60+15))
day = df[day_mask].copy()

print(f"  日数: {day['date'].nunique()}, 総バー数: {len(day):,}")

# ─────────────────────────────────────
# 1. 分別 平均価格経路（寄付=0基準）
# ─────────────────────────────────────
print("\n[1] 平均イントラデイ価格経路を計算中...")

# 各日の前場寄付価格を取得
open_prices = {}
for dt, grp in day.groupby('date'):
    morn = grp[grp['hm'] == 8*60+45]
    if len(morn) == 0:
        morn = grp[grp['hm'] <= 9*60].head(1)
    if len(morn) > 0:
        open_prices[dt] = morn['close'].iloc[0]

# 各バーの寄付比リターン（%）
day['open_price'] = day['date'].map(open_prices)
day['ret_from_open'] = (day['close'] / day['open_price'] - 1) * 100

# 分別（hm）平均・中央値
intraday_path = day.groupby('hm')['ret_from_open'].agg(
    mean='mean', median='median', std='std',
    q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75)
).reset_index()

# ─────────────────────────────────────
# 2. 分別 ボラ・出来高・方向バイアス
# ─────────────────────────────────────
day['ret1'] = day.groupby('date')['close'].pct_change() * 100

hm_stats = day.groupby('hm').agg(
    vol=('ret1', lambda x: x[x.abs() < 2].std()),
    avg_vol=('volume', 'mean'),
    up_rate=('ret1', lambda x: (x > 0).mean() * 100),
    avg_ret=('ret1', 'mean'),
    n=('ret1', 'count')
).reset_index()
hm_stats = hm_stats[hm_stats['n'] >= 50]

# ─────────────────────────────────────
# 3. 後場寄付（12:30）の挙動
# ─────────────────────────────────────
print("[3] 後場寄付を分析中...")
pm_open_results = []

for dt, grp in day.groupby('date'):
    grp = grp.sort_index()
    pm = grp[(grp['hm'] >= 12*60+30) & (grp['hm'] < 12*60+35)]
    am_close_bar = grp[(grp['hm'] >= 11*60+25) & (grp['hm'] <= 11*60+30)]
    if len(pm) == 0 or len(am_close_bar) == 0:
        continue

    am_close = am_close_bar['close'].iloc[-1]
    pm_open  = pm['close'].iloc[0]
    pm_gap   = (pm_open / am_close - 1) * 100  # 前場引け→後場寄付ギャップ

    # 後場最初30分の動き
    pm_30 = grp[(grp['hm'] >= 12*60+30) & (grp['hm'] < 13*60)]
    pm_30_ret = (pm_30['close'].iloc[-1] / pm_open - 1) * 100 if len(pm_30) > 3 else np.nan

    # 当日前場の累積リターン
    am = grp[(grp['hm'] >= 9*60) & (grp['hm'] <= 11*60+30)]
    am_ret = (am['close'].iloc[-1] / am['close'].iloc[0] - 1) * 100 if len(am) > 5 else np.nan

    pm_open_results.append({'date': dt, 'pm_gap': pm_gap, 'pm_30_ret': pm_30_ret, 'am_ret': am_ret})

pm_df = pd.DataFrame(pm_open_results).dropna()
pm_df = pm_df[(pm_df['pm_gap'].abs() < 2) & (pm_df['pm_30_ret'].abs() < 3)]

r_pm, p_pm = stats.pearsonr(pm_df['pm_gap'], pm_df['pm_30_ret'])
print(f"  後場ギャップ→後場30分 相関: r={r_pm:+.3f}, p={p_pm:.3f}")
r_am, p_am = stats.pearsonr(pm_df['am_ret'], pm_df['pm_30_ret'])
print(f"  前場リターン→後場30分 相関: r={r_am:+.3f}, p={p_am:.3f}")

# ─────────────────────────────────────
# 4. 大引け前30分（14:45-15:15）
# ─────────────────────────────────────
print("[4] 大引け前を分析中...")
close_results = []

for dt, grp in day.groupby('date'):
    grp = grp.sort_index()
    pre_close = grp[(grp['hm'] >= 14*60+45) & (grp['hm'] <= 15*60+15)]
    mid_day   = grp[(grp['hm'] >= 13*60) & (grp['hm'] < 14*60+45)]
    open_p    = open_prices.get(dt)
    if len(pre_close) < 3 or len(mid_day) < 5 or open_p is None:
        continue

    day_ret_to_1445 = (mid_day['close'].iloc[-1] / open_p - 1) * 100
    close_30_ret    = (pre_close['close'].iloc[-1] / pre_close['close'].iloc[0] - 1) * 100

    close_results.append({'date': dt, 'day_ret_to_1445': day_ret_to_1445,
                           'close_30_ret': close_30_ret})

close_df = pd.DataFrame(close_results).dropna()
close_df = close_df[(close_df['day_ret_to_1445'].abs() < 5) & (close_df['close_30_ret'].abs() < 3)]

r_cl, p_cl = stats.pearsonr(close_df['day_ret_to_1445'], close_df['close_30_ret'])
print(f"  14:45時点の損益 → 大引け前30分 相関: r={r_cl:+.3f}, p={p_cl:.3f}")

cont_close = (np.sign(close_df['day_ret_to_1445']) == np.sign(close_df['close_30_ret'])).mean() * 100
print(f"  14:45時点の方向と大引け方向一致率: {cont_close:.0f}%")

# ─────────────────────────────────────
# 5. 継続率ヒートマップ（時間帯 × 方向強度）
# ─────────────────────────────────────
print("[5] 継続率ヒートマップを計算中...")
day['fwd5'] = day.groupby('date')['close'].transform(
    lambda x: x.shift(-5) / x - 1) * 100

time_slots = [
    ('8:45-9:00', 8*60+45, 9*60),
    ('9:00-9:30', 9*60, 9*60+30),
    ('9:30-10:00', 9*60+30, 10*60),
    ('10:00-10:30', 10*60, 10*60+30),
    ('10:30-11:00', 10*60+30, 11*60),
    ('11:00-11:30', 11*60, 11*60+30),
    ('12:30-13:00', 12*60+30, 13*60),
    ('13:00-14:00', 13*60, 14*60),
    ('14:00-14:45', 14*60, 14*60+45),
    ('14:45-15:15', 14*60+45, 15*60+15),
]
mag_bins = [0.0, 0.05, 0.15, 0.35, 99]
mag_labels = ['<0.05%', '0.05-0.15%', '0.15-0.35%', '>0.35%']

heatmap_up = pd.DataFrame(index=[s[0] for s in time_slots], columns=mag_labels, dtype=float)
heatmap_dn = pd.DataFrame(index=[s[0] for s in time_slots], columns=mag_labels, dtype=float)

day['roll5'] = day.groupby('date')['close'].transform(
    lambda x: x.pct_change(5)) * 100

for slot_label, lo, hi in time_slots:
    sub = day[(day['hm'] >= lo) & (day['hm'] < hi)].dropna(subset=['roll5', 'fwd5'])
    sub = sub[sub['fwd5'].abs() < 3]
    for i, (mag_lo, mag_hi) in enumerate(zip(mag_bins[:-1], mag_bins[1:])):
        mag_label = mag_labels[i]
        up_mask = (sub['roll5'] > mag_lo) & (sub['roll5'] <= mag_hi if mag_hi < 99 else sub['roll5'] > mag_lo)
        dn_mask = (sub['roll5'] < -mag_lo) & (sub['roll5'] >= -mag_hi if mag_hi < 99 else sub['roll5'] < -mag_lo)
        if up_mask.sum() >= 20:
            heatmap_up.loc[slot_label, mag_label] = (sub[up_mask]['fwd5'] > 0).mean() * 100
        if dn_mask.sum() >= 20:
            heatmap_dn.loc[slot_label, mag_label] = (sub[dn_mask]['fwd5'] < 0).mean() * 100

# ─────────────────────────────────────
# 6. 曜日効果
# ─────────────────────────────────────
print("[6] 曜日効果を計算中...")
day_of_week_results = []
for dt, grp in day.groupby('date'):
    grp = grp.sort_index()
    open_p = open_prices.get(dt)
    if open_p is None or len(grp) < 10:
        continue
    day_ret = (grp['close'].iloc[-1] / open_p - 1) * 100
    dow = pd.Timestamp(dt).dayofweek  # 0=Mon
    day_of_week_results.append({'dow': dow, 'day_ret': day_ret})

dow_df = pd.DataFrame(day_of_week_results)
dow_df = dow_df[dow_df['day_ret'].abs() < 5]
dow_stats = dow_df.groupby('dow')['day_ret'].agg(['mean','std','count'])
dow_stats.index = ['Mon','Tue','Wed','Thu','Fri']

print("  曜日別平均リターン（寄付比）:")
print(dow_stats.to_string())

# ─────────────────────────────────────
# 7. 可視化
# ─────────────────────────────────────
print("\nグラフ作成中...")
fig = plt.figure(figsize=(20, 22))
gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.55, wspace=0.38)
fig.suptitle('Nikkei Futures (JNIc1) - Full Intraday Profile', fontsize=15, fontweight='bold')

# ────────── Row 0 ──────────

# (A) 平均価格経路
ax1 = fig.add_subplot(gs[0, :2])
ip = intraday_path.copy()
# 昼休みにギャップを入れる
ip_am = ip[ip['hm'] <= 11*60+30]
ip_pm = ip[ip['hm'] >= 12*60+30]

def hm_to_str(hm):
    return f"{hm//60:02d}:{hm%60:02d}"

for part, color in [(ip_am, 'steelblue'), (ip_pm, 'darkorange')]:
    xs = [hm_to_str(h) for h in part['hm']]
    ax1.plot(xs, part['mean'], color=color, lw=2, zorder=3)
    ax1.fill_between(xs, part['q25'], part['q75'], color=color, alpha=0.15)
    ax1.fill_between(xs, part['mean']-part['std'], part['mean']+part['std'], color=color, alpha=0.08)

ax1.axhline(0, color='black', lw=0.8, ls='--')
ax1.set_title('(A) Avg Intraday Price Path (from open, %, IQR shaded)')
ax1.set_xlabel('Time (JST)')
ax1.set_ylabel('Return from Open (%)')
xticks_all = list(ip_am['hm'].values) + list(ip_pm['hm'].values)
xticklabels = [hm_to_str(h) for h in xticks_all]
step = max(1, len(xticks_all) // 20)
ax1.set_xticks(range(0, len(xticks_all), step))
ax1.set_xticklabels(xticklabels[::step], rotation=60, fontsize=8)
ax1.legend(['AM session','PM session'], loc='upper left')

# (B) 曜日効果
ax2 = fig.add_subplot(gs[0, 2])
colors_dow = ['steelblue' if v >= 0 else 'tomato' for v in dow_stats['mean']]
bars = ax2.bar(dow_stats.index, dow_stats['mean'], color=colors_dow, alpha=0.85)
ax2.errorbar(dow_stats.index, dow_stats['mean'],
             yerr=dow_stats['std']/np.sqrt(dow_stats['count']),
             fmt='none', color='black', capsize=4)
ax2.axhline(0, color='black', lw=0.8)
ax2.set_title('(B) Day-of-Week Effect (full day ret from open)')
ax2.set_ylabel('Avg Return (%)')
for bar, v in zip(bars, dow_stats['mean']):
    ax2.text(bar.get_x()+bar.get_width()/2, v+0.005*np.sign(v), f'{v:+.3f}%', ha='center', fontsize=9)

# ────────── Row 1 ──────────

# (C) ボラティリティ（分別）
ax3 = fig.add_subplot(gs[1, 0])
hs = hm_stats.copy()
hs_am = hs[hs['hm'] <= 11*60+30]
hs_pm = hs[hs['hm'] >= 12*60+30]
ax3.plot(range(len(hs_am)), hs_am['vol'], color='steelblue', lw=1.5, label='AM')
ax3.plot(range(len(hs_am), len(hs_am)+len(hs_pm)), hs_pm['vol'], color='darkorange', lw=1.5, label='PM')
ax3.axhline(hs['vol'].mean(), color='gray', lw=1, ls='--', label='avg')
xsep = len(hs_am)
ax3.axvline(xsep, color='green', lw=1, ls=':', label='12:30')
ax3.set_title('(C) Volatility by Minute')
ax3.set_ylabel('Std Dev of 1min ret (%)')
ax3.legend(fontsize=8)
ax3.set_xticks([0, len(hs_am)//4, len(hs_am)//2, len(hs_am)*3//4, xsep,
                xsep+len(hs_pm)//3, xsep+len(hs_pm)*2//3])
ax3.set_xticklabels(['8:45','9:28','10:12','10:57','12:30','13:25','14:20'], rotation=45, fontsize=8)

# (D) 出来高（分別）
ax4 = fig.add_subplot(gs[1, 1])
ax4.fill_between(range(len(hs_am)), hs_am['avg_vol'], alpha=0.6, color='steelblue', label='AM')
ax4.fill_between(range(len(hs_am), len(hs_am)+len(hs_pm)), hs_pm['avg_vol'], alpha=0.6, color='darkorange', label='PM')
ax4.axvline(xsep, color='green', lw=1, ls=':', label='12:30')
ax4.set_title('(D) Avg Volume by Minute')
ax4.set_ylabel('Avg Volume (contracts)')
ax4.legend(fontsize=8)
ax4.set_xticks([0, len(hs_am)//4, len(hs_am)//2, len(hs_am)*3//4, xsep,
                xsep+len(hs_pm)//3, xsep+len(hs_pm)*2//3])
ax4.set_xticklabels(['8:45','9:28','10:12','10:57','12:30','13:25','14:20'], rotation=45, fontsize=8)

# (E) 上昇バー率（方向バイアス）
ax5 = fig.add_subplot(gs[1, 2])
ax5.plot(range(len(hs_am)), hs_am['up_rate'], color='steelblue', lw=1.5, label='AM')
ax5.plot(range(len(hs_am), len(hs_am)+len(hs_pm)), hs_pm['up_rate'], color='darkorange', lw=1.5, label='PM')
ax5.axhline(50, color='black', lw=1, ls='--')
ax5.axvline(xsep, color='green', lw=1, ls=':', label='12:30')
ax5.set_title('(E) Up-bar Rate (directional bias %)')
ax5.set_ylabel('% of up bars')
ax5.legend(fontsize=8)
ax5.set_xticks([0, len(hs_am)//4, len(hs_am)//2, len(hs_am)*3//4, xsep,
                xsep+len(hs_pm)//3, xsep+len(hs_pm)*2//3])
ax5.set_xticklabels(['8:45','9:28','10:12','10:57','12:30','13:25','14:20'], rotation=45, fontsize=8)
ax5.set_ylim(40, 60)

# ────────── Row 2 ──────────

# (F) 継続率ヒートマップ（上昇）
ax6 = fig.add_subplot(gs[2, :])
hm_up_vals = heatmap_up.astype(float)
cmap = LinearSegmentedColormap.from_list('rg', ['tomato', 'white', 'steelblue'])
im = ax6.imshow(hm_up_vals.T, aspect='auto', cmap=cmap, vmin=35, vmax=65)
ax6.set_xticks(range(len(time_slots)))
ax6.set_xticklabels([s[0] for s in time_slots], rotation=30, ha='right', fontsize=9)
ax6.set_yticks(range(len(mag_labels)))
ax6.set_yticklabels(mag_labels)
ax6.set_title('(F) Continuation Rate Heatmap - UPWARD move → next 5min (blue=continuation, red=reversal)')
plt.colorbar(im, ax=ax6, label='Continuation Rate (%)')
for i in range(hm_up_vals.shape[0]):
    for j in range(hm_up_vals.shape[1]):
        v = hm_up_vals.iloc[i, j]
        if not np.isnan(v):
            ax6.text(i, j, f'{v:.0f}', ha='center', va='center',
                     fontsize=9, color='black' if 40 < v < 60 else 'white')

# ────────── Row 3 ──────────

# (G) 後場ギャップ分析
ax7 = fig.add_subplot(gs[3, 0])
ax7.scatter(pm_df['pm_gap'], pm_df['pm_30_ret'], alpha=0.4, s=15, color='darkorange')
ax7.axhline(0, color='k', lw=0.5); ax7.axvline(0, color='k', lw=0.5)
z = np.polyfit(pm_df['pm_gap'], pm_df['pm_30_ret'], 1)
xl = np.linspace(pm_df['pm_gap'].min(), pm_df['pm_gap'].max(), 100)
ax7.plot(xl, np.poly1d(z)(xl), 'r-', lw=1.5, label=f'r={r_pm:+.3f} p={p_pm:.3f}')
ax7.set_title('(G) PM Gap (11:30→12:30) vs PM 30min Ret')
ax7.set_xlabel('Gap (%)'); ax7.set_ylabel('PM 30min Ret (%)'); ax7.legend(fontsize=8)

# (H) 大引け前（前場+後場の累積方向 → 大引け）
ax8 = fig.add_subplot(gs[3, 1])
ax8.scatter(close_df['day_ret_to_1445'], close_df['close_30_ret'], alpha=0.4, s=15, color='purple')
ax8.axhline(0, color='k', lw=0.5); ax8.axvline(0, color='k', lw=0.5)
z2 = np.polyfit(close_df['day_ret_to_1445'], close_df['close_30_ret'], 1)
xl2 = np.linspace(close_df['day_ret_to_1445'].min(), close_df['day_ret_to_1445'].max(), 100)
ax8.plot(xl2, np.poly1d(z2)(xl2), 'r-', lw=1.5, label=f'r={r_cl:+.3f} p={p_cl:.3f}')
ax8.set_title('(H) Day Ret @14:45 vs Last-30min Ret')
ax8.set_xlabel('Ret from open @14:45 (%)'); ax8.set_ylabel('14:45-15:15 Ret (%)'); ax8.legend(fontsize=8)

# (I) 後場最初30分 × 前場リターン
ax9 = fig.add_subplot(gs[3, 2])
ax9.scatter(pm_df['am_ret'], pm_df['pm_30_ret'], alpha=0.4, s=15, color='teal')
ax9.axhline(0, color='k', lw=0.5); ax9.axvline(0, color='k', lw=0.5)
z3 = np.polyfit(pm_df['am_ret'], pm_df['pm_30_ret'], 1)
xl3 = np.linspace(pm_df['am_ret'].min(), pm_df['am_ret'].max(), 100)
ax9.plot(xl3, np.poly1d(z3)(xl3), 'r-', lw=1.5, label=f'r={r_am:+.3f} p={p_am:.3f}')
ax9.set_title('(I) AM Ret vs PM First 30min Ret')
ax9.set_xlabel('AM Session Ret (%)'); ax9.set_ylabel('PM 30min Ret (%)'); ax9.legend(fontsize=8)

plt.savefig('/Users/Yusuke/claude-code/japan-stocks/analyses/20260423_nikkei_momentum_reversal/result_profile.png',
            dpi=150, bbox_inches='tight')
print("グラフ保存: result_profile.png")

# ─────────────────────────────────────
# サマリー出力
# ─────────────────────────────────────
print("\n" + "="*65)
print("【イントラデイ特徴 サマリー】")
print("="*65)
print("\n■ 平均価格経路の特徴")
ip_subset = intraday_path.copy()
peak_hm = ip_subset.loc[ip_subset['mean'].idxmax(), 'hm']
trough_hm = ip_subset.loc[ip_subset['mean'].idxmin(), 'hm']
print(f"  当日平均ピーク: {peak_hm//60:02d}:{peak_hm%60:02d}  "
      f"({ip_subset['mean'].max():+.3f}%)")
print(f"  当日平均トラフ: {trough_hm//60:02d}:{trough_hm%60:02d}  "
      f"({ip_subset['mean'].min():+.3f}%)")

print("\n■ ボラティリティ最大時間帯 TOP5")
top_vol = hm_stats.nlargest(5, 'vol')[['hm','vol','avg_vol']]
for _, row in top_vol.iterrows():
    print(f"  {int(row['hm'])//60:02d}:{int(row['hm'])%60:02d}  vol={row['vol']:.4f}%  vol={row['avg_vol']:.0f}枚")

print("\n■ 出来高最大時間帯 TOP5")
top_vol = hm_stats.nlargest(5, 'avg_vol')[['hm','avg_vol','vol']]
for _, row in top_vol.iterrows():
    print(f"  {int(row['hm'])//60:02d}:{int(row['hm'])%60:02d}  {row['avg_vol']:.0f}枚  ボラ={row['vol']:.4f}%")

print("\n■ 後場（12:30）")
print(f"  ギャップ→後場30分 相関: r={r_pm:+.3f} ({'有意' if p_pm<0.05 else '非有意'})")
print(f"  前場リターン→後場30分 相関: r={r_am:+.3f} ({'有意' if p_am<0.05 else '非有意'})")

print("\n■ 大引け前30分")
print(f"  当日損益方向との一致率: {cont_close:.0f}%  "
      f"({'継続' if cont_close>50 else '反転'}傾向)")
print(f"  相関: r={r_cl:+.3f} ({'有意' if p_cl<0.05 else '非有意'})")

print("\n■ 曜日効果")
for dow_name, row in dow_stats.iterrows():
    print(f"  {dow_name}: {row['mean']:+.3f}%  (N={int(row['count'])})")

print("\n完了")
