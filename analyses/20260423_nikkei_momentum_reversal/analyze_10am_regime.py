"""
10時前後でレジームが変わるか検証
- 時間帯別ボラティリティ・出来高
- 時間帯別継続率（モメンタム vs 平均回帰）
- 9:00-10:00 の動きが 10:00 以降を予測するか
- 10時前後の自己相関変化
"""
import psycopg2
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

PG_CONFIG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}

print("データロード中...")
conn = psycopg2.connect(**PG_CONFIG)
df = pd.read_sql(
    "SELECT timestamp, open, high, low, close, volume "
    "FROM intraday_data WHERE symbol='JNIc1' ORDER BY timestamp",
    conn
)
conn.close()

df['timestamp'] = pd.to_datetime(df['timestamp'])
df['jst'] = df['timestamp'] + pd.Timedelta(hours=9)
df = df.dropna(subset=['close']).copy()
df['close'] = df['close'].astype(float)
df['open']  = df['open'].astype(float)
df['high']  = df['high'].astype(float)
df['low']   = df['low'].astype(float)
df['volume'] = df['volume'].astype(float)
df = df.set_index('jst').sort_index()

# 日中セッション（前場 8:45-11:30、後場 12:30-15:15）のみ
df['hm'] = df.index.hour * 60 + df.index.minute
day_mask = ((df['hm'] >= 8*60+45) & (df['hm'] <= 11*60+30)) | \
           ((df['hm'] >= 12*60+30) & (df['hm'] <= 15*60+15))
day = df[day_mask].copy()
day['date'] = day.index.date
day['ret1'] = day.groupby('date')['close'].pct_change() * 100

# ─────────────────────────────────────
# 1. 時間帯バンド定義（5分刻み）
# ─────────────────────────────────────
bands_5min = [
    ('8:45', 8*60+45, 8*60+50),
    ('8:50', 8*60+50, 8*60+55),
    ('8:55', 8*60+55, 9*60+0),
    ('9:00', 9*60+0,  9*60+5),
    ('9:05', 9*60+5,  9*60+10),
    ('9:10', 9*60+10, 9*60+15),
    ('9:15', 9*60+15, 9*60+20),
    ('9:20', 9*60+20, 9*60+30),
    ('9:30', 9*60+30, 9*60+45),
    ('9:45', 9*60+45, 10*60+0),
    ('10:00', 10*60+0, 10*60+15),
    ('10:15', 10*60+15, 10*60+30),
    ('10:30', 10*60+30, 10*60+45),
    ('10:45', 10*60+45, 11*60+0),
    ('11:00', 11*60+0, 11*60+15),
    ('11:15', 11*60+15, 11*60+30),
    ('12:30', 12*60+30, 12*60+45),
    ('12:45', 12*60+45, 13*60+0),
    ('13:00', 13*60+0, 13*60+30),
    ('13:30', 13*60+30, 14*60+0),
    ('14:00', 14*60+0, 14*60+30),
    ('14:30', 14*60+30, 15*60+0),
    ('15:00', 15*60+0, 15*60+16),
]

# ─────────────────────────────────────
# 2. 時間帯別 ボラティリティ・出来高
# ─────────────────────────────────────
print("\n=== 時間帯別 ボラティリティ・出来高 ===")
band_stats = []
for label, lo, hi in bands_5min:
    sub = day[(day['hm'] >= lo) & (day['hm'] < hi)]
    sub_ret = sub['ret1'].dropna()
    sub_ret = sub_ret[sub_ret.abs() < 2]  # スパイク除外
    if len(sub_ret) < 20:
        continue
    vol = sub_ret.std()
    avg_vol = sub['volume'].mean()
    n = len(sub_ret)
    band_stats.append({'band': label, 'vol': vol, 'avg_volume': avg_vol, 'n': n})

band_df = pd.DataFrame(band_stats)
print(band_df.to_string(index=False))

# ─────────────────────────────────────
# 3. 時間帯別 継続率（5分後の方向一致率）
# ─────────────────────────────────────
print("\n=== 時間帯別 5分後継続率 ===")
day['fwd5'] = day.groupby('date')['close'].transform(lambda x: x.shift(-5).pct_change(5)) * 100

cont_stats = []
for label, lo, hi in bands_5min:
    sub = day[(day['hm'] >= lo) & (day['hm'] < hi)].dropna(subset=['ret1', 'fwd5'])
    sub = sub[sub['ret1'].abs() > 0.05]  # 動きが小さすぎるバーは除外
    if len(sub) < 30:
        continue

    # 5分ローリングリターン（前5バー）
    day_roll5 = day.groupby('date')['close'].transform(lambda x: x.pct_change(5)) * 100
    sub_roll = day_roll5[(day['hm'] >= lo) & (day['hm'] < hi)]
    fwd_sub  = day['fwd5'][(day['hm'] >= lo) & (day['hm'] < hi)]

    valid = pd.DataFrame({'roll': sub_roll, 'fwd': fwd_sub}).dropna()
    valid = valid[(valid['roll'].abs() > 0.05) & (valid['fwd'].abs() < 5) & (valid['roll'].abs() < 5)]
    if len(valid) < 30:
        continue

    up_mask = valid['roll'] > 0
    dn_mask = valid['roll'] < 0

    up_cont = (valid[up_mask]['fwd'] > 0).mean() * 100 if up_mask.sum() > 10 else np.nan
    dn_cont = (valid[dn_mask]['fwd'] < 0).mean() * 100 if dn_mask.sum() > 10 else np.nan
    avg_cont = (up_cont + dn_cont) / 2

    cont_stats.append({'band': label, 'up_cont': up_cont, 'dn_cont': dn_cont,
                        'avg_cont': avg_cont, 'n': len(valid)})

cont_df = pd.DataFrame(cont_stats)
print(cont_df.to_string(index=False))

# ─────────────────────────────────────
# 4. 9:00-10:00 累積リターン → 10:00以降の予測力
# ─────────────────────────────────────
print("\n=== 9:00-10:00 累積リターン → 10:00以降の傾向 ===")
predict_results = []

for dt, grp in day.groupby('date'):
    grp = grp.sort_index()
    first_hour = grp[(grp['hm'] >= 9*60) & (grp['hm'] < 10*60)]
    after_10   = grp[(grp['hm'] >= 10*60) & (grp['hm'] <= 11*60+30)]

    if len(first_hour) < 5 or len(after_10) < 5:
        continue

    fh_ret = (first_hour['close'].iloc[-1] / first_hour['close'].iloc[0] - 1) * 100
    a10_ret = (after_10['close'].iloc[-1] / after_10['close'].iloc[0] - 1) * 100

    # 10:00時点の価格と9:00時点の価格
    p9  = first_hour['close'].iloc[0]
    p10 = first_hour['close'].iloc[-1]

    predict_results.append({
        'date': dt,
        'first_hour_ret': fh_ret,
        'after10_ret': a10_ret,
    })

pred_df = pd.DataFrame(predict_results)
pred_df = pred_df[pred_df['first_hour_ret'].abs() < 5]

r, p = stats.pearsonr(pred_df['first_hour_ret'], pred_df['after10_ret'])
print(f"  9:00-10:00 vs 10:00-11:30 相関: r={r:+.3f}  p={p:.4f}")
print(f"  方向一致率: {(np.sign(pred_df['first_hour_ret']) == np.sign(pred_df['after10_ret'])).mean()*100:.1f}%")

# 幅別集計
for label, lo, hi in [('小(<0.3%)', 0, 0.3), ('中(0.3-1%)', 0.3, 1.0), ('大(>1%)', 1.0, 99)]:
    for direction in ['上昇', '下落']:
        if direction == '上昇':
            mask = (pred_df['first_hour_ret'] > lo) & (pred_df['first_hour_ret'] <= hi if hi < 99 else pred_df['first_hour_ret'] > lo)
        else:
            mask = (pred_df['first_hour_ret'] < -lo) & (pred_df['first_hour_ret'] >= -hi if hi < 99 else pred_df['first_hour_ret'] < -lo)
        sub = pred_df[mask]
        if len(sub) < 5:
            continue
        cont = (np.sign(sub['first_hour_ret']) == np.sign(sub['after10_ret'])).mean() * 100
        avg = sub['after10_ret'].mean()
        print(f"  {direction}/{label:12s}: N={len(sub):3d}  10時以降継続率={cont:.0f}%  平均={avg:+.3f}%")

# ─────────────────────────────────────
# 5. 10時の方向転換パターン分析
#    9:30-10:00 vs 10:00-10:30 の方向
# ─────────────────────────────────────
print("\n=== 9:30-10:00 vs 10:00-10:30 方向比較 ===")
reversal_results = []

for dt, grp in day.groupby('date'):
    grp = grp.sort_index()
    pre10  = grp[(grp['hm'] >= 9*60+30) & (grp['hm'] < 10*60)]
    post10 = grp[(grp['hm'] >= 10*60) & (grp['hm'] < 10*60+30)]

    if len(pre10) < 3 or len(post10) < 3:
        continue

    pre_ret  = (pre10['close'].iloc[-1] / pre10['close'].iloc[0] - 1) * 100
    post_ret = (post10['close'].iloc[-1] / post10['close'].iloc[0] - 1) * 100

    reversal_results.append({'date': dt, 'pre_ret': pre_ret, 'post_ret': post_ret})

rev_df = pd.DataFrame(reversal_results)
rev_df = rev_df[(rev_df['pre_ret'].abs() < 3) & (rev_df['post_ret'].abs() < 3)]

r2, p2 = stats.pearsonr(rev_df['pre_ret'], rev_df['post_ret'])
cont_rate = (np.sign(rev_df['pre_ret']) == np.sign(rev_df['post_ret'])).mean() * 100
print(f"  相関: r={r2:+.3f}  p={p2:.4f}")
print(f"  方向一致（継続）率: {cont_rate:.1f}%  → 反転率: {100-cont_rate:.1f}%")

# 動き幅別
for lo, hi, label in [(0, 0.2, '小'), (0.2, 0.5, '中'), (0.5, 99, '大')]:
    for direction in ['上昇', '下落']:
        if direction == '上昇':
            mask = (rev_df['pre_ret'] > lo) & (rev_df['pre_ret'] <= hi if hi < 99 else rev_df['pre_ret'] > lo)
        else:
            mask = (rev_df['pre_ret'] < -lo) & (rev_df['pre_ret'] >= -hi if hi < 99 else rev_df['pre_ret'] < -lo)
        sub = rev_df[mask]
        if len(sub) < 5:
            continue
        cont = (np.sign(sub['pre_ret']) == np.sign(sub['post_ret'])).mean() * 100
        print(f"  {direction}/{label}: N={len(sub):3d}  継続={cont:.0f}%  反転={100-cont:.0f}%  post平均={sub['post_ret'].mean():+.3f}%")

# ─────────────────────────────────────
# 6. 可視化
# ─────────────────────────────────────
print("\nグラフ作成中...")
fig = plt.figure(figsize=(18, 16))
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.35)
fig.suptitle('Nikkei Futures (JNIc1) - 10:00 Regime Change Analysis', fontsize=14, fontweight='bold')

# (A) 時間帯別ボラティリティ
ax1 = fig.add_subplot(gs[0, 0])
colors_vol = ['tomato' if '10:' in b or b in ['9:45'] else 'steelblue' if b in ['8:45','8:50','8:55','9:00','9:05'] else 'cornflowerblue'
              for b in band_df['band']]
ax1.bar(band_df['band'], band_df['vol'], color=colors_vol, alpha=0.85)
ax1.axvline(band_df[band_df['band'] == '10:00'].index[0] if '10:00' in band_df['band'].values else 10,
            color='red', lw=2, ls='--', label='10:00')
ax1.set_title('(A) Volatility by Time Band (1min ret std)')
ax1.set_xlabel('Time (JST)')
ax1.set_ylabel('Std Dev (%)')
ax1.tick_params(axis='x', rotation=60)
ax1.legend()

# (B) 時間帯別出来高
ax2 = fig.add_subplot(gs[0, 1])
ax2.bar(band_df['band'], band_df['avg_volume'], color='steelblue', alpha=0.75)
idx_10 = band_df[band_df['band'] == '10:00'].index
if len(idx_10) > 0:
    ax2.axvline(idx_10[0], color='red', lw=2, ls='--', label='10:00')
ax2.set_title('(B) Avg Volume by Time Band')
ax2.set_xlabel('Time (JST)')
ax2.set_ylabel('Avg Volume (contracts)')
ax2.tick_params(axis='x', rotation=60)
ax2.legend()

# (C) 時間帯別継続率
ax3 = fig.add_subplot(gs[1, 0])
ax3.axhline(50, color='black', lw=1, ls='--', label='50% (random)')
ax3.plot(cont_df['band'], cont_df['up_cont'], 'o-', color='steelblue', label='Up continuation', alpha=0.8)
ax3.plot(cont_df['band'], cont_df['dn_cont'], 's-', color='tomato', label='Down continuation', alpha=0.8)
ax3.plot(cont_df['band'], cont_df['avg_cont'], '^-', color='purple', label='Average', alpha=0.9, lw=2)
idx_10_cont = cont_df[cont_df['band'] == '10:00'].index
if len(idx_10_cont) > 0:
    ax3.axvline(idx_10_cont[0], color='red', lw=2, ls=':', label='10:00')
ax3.set_title('(C) 5-min Continuation Rate by Time Band')
ax3.set_xlabel('Time (JST)')
ax3.set_ylabel('Continuation Rate (%)')
ax3.tick_params(axis='x', rotation=60)
ax3.legend(fontsize=8)
ax3.set_ylim(30, 70)

# (D) 9:00-10:00 vs 10:00-11:30 散布図
ax4 = fig.add_subplot(gs[1, 1])
ax4.scatter(pred_df['first_hour_ret'], pred_df['after10_ret'], alpha=0.4, s=15, color='steelblue')
ax4.axhline(0, color='black', lw=0.5)
ax4.axvline(0, color='black', lw=0.5)
z = np.polyfit(pred_df['first_hour_ret'], pred_df['after10_ret'], 1)
xline = np.linspace(pred_df['first_hour_ret'].min(), pred_df['first_hour_ret'].max(), 100)
ax4.plot(xline, np.poly1d(z)(xline), 'r-', lw=1.5, label=f'r={r:+.3f} p={p:.3f}')
ax4.set_title('(D) 9:00-10:00 Ret vs 10:00-11:30 Ret')
ax4.set_xlabel('9:00-10:00 Ret (%)')
ax4.set_ylabel('10:00-11:30 Ret (%)')
ax4.legend()

# (E) 9:30-10:00 vs 10:00-10:30
ax5 = fig.add_subplot(gs[2, 0])
ax5.scatter(rev_df['pre_ret'], rev_df['post_ret'], alpha=0.4, s=15, color='darkorange')
ax5.axhline(0, color='black', lw=0.5)
ax5.axvline(0, color='black', lw=0.5)
z2 = np.polyfit(rev_df['pre_ret'], rev_df['post_ret'], 1)
xline2 = np.linspace(rev_df['pre_ret'].min(), rev_df['pre_ret'].max(), 100)
ax5.plot(xline2, np.poly1d(z2)(xline2), 'r-', lw=1.5, label=f'r={r2:+.3f} p={p2:.3f}')
ax5.set_title('(E) 9:30-10:00 Ret vs 10:00-10:30 Ret')
ax5.set_xlabel('9:30-10:00 Ret (%)')
ax5.set_ylabel('10:00-10:30 Ret (%)')
ax5.legend()

# (F) 継続率サマリーバーチャート（時間帯ブロック別）
ax6 = fig.add_subplot(gs[2, 1])
time_blocks = [
    ('8:45-9:00', 8*60+45, 9*60),
    ('9:00-9:30', 9*60, 9*60+30),
    ('9:30-10:00', 9*60+30, 10*60),
    ('10:00-10:30', 10*60, 10*60+30),
    ('10:30-11:00', 10*60+30, 11*60),
    ('11:00-11:30', 11*60, 11*60+30),
    ('12:30-13:30', 12*60+30, 13*60+30),
    ('13:30-15:15', 13*60+30, 15*60+15),
]
block_conts = []
block_labels = []
for blabel, blo, bhi in time_blocks:
    sub = cont_df[(cont_df.apply(
        lambda r: any(r['band'] == b for b, lo, hi in bands_5min if lo >= blo and hi <= bhi), axis=1
    ))]
    # 直接フィルタリング
    sub2 = []
    for b, lo, hi in bands_5min:
        if lo >= blo and hi <= bhi:
            row = cont_df[cont_df['band'] == b]
            if len(row) > 0:
                sub2.append(row['avg_cont'].values[0])
    if sub2:
        block_conts.append(np.mean(sub2))
        block_labels.append(blabel)

colors_bar = ['tomato' if c < 50 else 'steelblue' for c in block_conts]
bars = ax6.bar(block_labels, block_conts, color=colors_bar, alpha=0.85)
ax6.axhline(50, color='black', lw=1, ls='--')
ax6.set_title('(F) Avg Continuation Rate by Time Block')
ax6.set_xlabel('Time Block')
ax6.set_ylabel('Avg Continuation Rate (%)')
ax6.tick_params(axis='x', rotation=45)
for bar, val in zip(bars, block_conts):
    ax6.text(bar.get_x() + bar.get_width()/2, val + 0.3, f'{val:.1f}%', ha='center', fontsize=8)
ax6.set_ylim(35, 65)

plt.savefig('/Users/Yusuke/claude-code/japan-stocks/analyses/20260423_nikkei_momentum_reversal/result_10am.png',
            dpi=150, bbox_inches='tight')
print("グラフ保存: result_10am.png")
print("\n完了")
