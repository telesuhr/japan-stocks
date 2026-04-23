"""
日経平均先物 (JNIc1) モメンタム vs 平均回帰分析
- 上昇/下落後に継続するか反転するかの傾向を検証
- 時間帯別（前場/後場/夜間）、ムーブ幅別で分析
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

# ─────────────────────────────────────
# 1. データロード
# ─────────────────────────────────────
print("データロード中...")
conn = psycopg2.connect(**PG_CONFIG)
df = pd.read_sql(
    "SELECT timestamp, open, high, low, close, volume "
    "FROM intraday_data WHERE symbol = 'JNIc1' ORDER BY timestamp",
    conn
)
conn.close()

df['timestamp'] = pd.to_datetime(df['timestamp'])
df['jst'] = df['timestamp'] + pd.Timedelta(hours=9)
df = df.dropna(subset=['close']).copy()
df = df.set_index('jst').sort_index()
df['close'] = df['close'].astype(float)
df['open']  = df['open'].astype(float)
df['volume'] = df['volume'].astype(float)

print(f"  総バー数: {len(df):,}  期間: {df.index[0].date()} ～ {df.index[-1].date()}")

# ─────────────────────────────────────
# 2. セッション分類
#   大阪先物（日経mini）
#   昼間: 8:45-15:15 JST
#   夜間: 16:30-翌5:30 JST
# ─────────────────────────────────────
def classify_session(ts):
    h, m = ts.hour, ts.minute
    t = h * 60 + m
    day_start  = 8  * 60 + 45
    day_end    = 15 * 60 + 15
    night_start= 16 * 60 + 30
    if day_start <= t <= day_end:
        return 'day'
    elif t >= night_start or t <= 5 * 60 + 30:
        return 'night'
    else:
        return 'break'  # 昼休み

df['session'] = df.index.map(classify_session)
df = df[df['session'] != 'break']

# 前場/後場サブ分類
def classify_sub(ts):
    h, m = ts.hour, ts.minute
    t = h * 60 + m
    if 8*60+45 <= t < 11*60+30:
        return 'morning'   # 前場
    elif 12*60+30 <= t <= 15*60+15:
        return 'afternoon' # 後場
    else:
        return 'night'

df['sub_session'] = df.index.map(classify_sub)

# ─────────────────────────────────────
# 3. 1分足リターン計算
# ─────────────────────────────────────
df['ret1'] = df['close'].pct_change() * 100  # %

# ─────────────────────────────────────
# 4. 自己相関分析（1分足）
# ─────────────────────────────────────
print("\n=== 1分足リターン自己相関 ===")
# 1分足ギャップ（セッション跨ぎを除外）
valid = df['ret1'].dropna()
for lag in [1, 2, 3, 5, 10, 15, 30]:
    r, p = stats.pearsonr(valid[:-lag], valid[lag:])
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
    print(f"  Lag {lag:2d}分: r={r:+.4f}  p={p:.4f} {sig}")

# ─────────────────────────────────────
# 5. N分後の継続 vs 反転率
#    ある時点でのN分ローリングリターンに対し
#    さらにN分後の累積リターンの方向を集計
# ─────────────────────────────────────
print("\n=== Nmin後の継続 vs 反転率 ===")
results = []

for lookback in [5, 10, 15, 30]:
    df[f'roll_{lookback}'] = df['close'].pct_change(lookback) * 100
    df[f'fwd_{lookback}']  = df['close'].shift(-lookback).pct_change(lookback) * 100

    sub = df.dropna(subset=[f'roll_{lookback}', f'fwd_{lookback}'])
    # セッション内のみ（大きなギャップを除外）
    sub = sub[sub[f'roll_{lookback}'].abs() < 5]  # 5%超のムーブは除外

    for mag_label, lo, hi in [('小(<0.1%)', 0, 0.1), ('中(0.1-0.3%)', 0.1, 0.3), ('大(>0.3%)', 0.3, 99)]:
        for direction, sign in [('上昇', 1), ('下落', -1)]:
            mask = (sub[f'roll_{lookback}'] * sign > lo * sign) & \
                   (sub[f'roll_{lookback}'] * sign <= hi * sign if hi < 99 else sub[f'roll_{lookback}'] * sign > lo * sign)
            # 上昇: roll > lo, 下落: roll < -lo
            if direction == '上昇':
                mask = (sub[f'roll_{lookback}'] > lo) & (sub[f'roll_{lookback}'] <= hi if hi < 99 else sub[f'roll_{lookback}'] > lo)
            else:
                mask = (sub[f'roll_{lookback}'] < -lo) & (sub[f'roll_{lookback}'] >= -hi if hi < 99 else sub[f'roll_{lookback}'] < -lo)

            grp = sub[mask]
            if len(grp) < 30:
                continue

            fwd = grp[f'fwd_{lookback}']
            if direction == '上昇':
                cont_rate = (fwd > 0).mean() * 100
            else:
                cont_rate = (fwd < 0).mean() * 100

            avg_fwd = fwd.mean()
            results.append({
                'lookback': lookback,
                'direction': direction,
                'magnitude': mag_label,
                'n': len(grp),
                'continuation_rate(%)': cont_rate,
                'avg_fwd_ret(%)': avg_fwd if direction == '上昇' else -avg_fwd,
            })

res_df = pd.DataFrame(results)
print(res_df.to_string(index=False))

# ─────────────────────────────────────
# 6. 時間帯別モメンタム特性
# ─────────────────────────────────────
print("\n=== セッション別 継続率（15分ローリング） ===")
lookback = 15
df[f'roll_{lookback}'] = df['close'].pct_change(lookback) * 100
df[f'fwd_{lookback}']  = df['close'].shift(-lookback).pct_change(lookback) * 100

for session_label in ['morning', 'afternoon', 'night']:
    sub = df[df['sub_session'] == session_label].dropna(subset=[f'roll_{lookback}', f'fwd_{lookback}'])
    sub = sub[sub[f'roll_{lookback}'].abs() < 3]
    if len(sub) < 100:
        continue

    up_mask = sub[f'roll_{lookback}'] > 0.1
    dn_mask = sub[f'roll_{lookback}'] < -0.1

    up_cont = (sub[up_mask][f'fwd_{lookback}'] > 0).mean() * 100
    dn_cont = (sub[dn_mask][f'fwd_{lookback}'] < 0).mean() * 100

    print(f"  {session_label:10s}: 上昇後継続={up_cont:.1f}%({up_mask.sum()}件)  下落後継続={dn_cont:.1f}%({dn_mask.sum()}件)")

# ─────────────────────────────────────
# 7. 寄付ギャップ分析（日中セッションのみ）
#    ギャップ後の当日リターンの方向
# ─────────────────────────────────────
print("\n=== 寄付ギャップ → 当日リターン ===")
day_df = df[df['session'] == 'day'].copy()
day_df['date'] = day_df.index.date

gap_results = []
for dt, grp in day_df.groupby('date'):
    grp = grp.sort_index()
    if len(grp) < 10:
        continue
    open_p  = grp['close'].iloc[0]
    close_p = grp['close'].iloc[-1]

    # 前日終値を使いたいが、代わりに前バーを使う
    # 昨日のdayセッション最終バー
    prev_days = [d for d in day_df['date'].unique() if d < dt]
    if not prev_days:
        continue
    prev_dt = max(prev_days)
    prev_grp = day_df[day_df['date'] == prev_dt]
    if len(prev_grp) == 0:
        continue
    prev_close = prev_grp['close'].iloc[-1]

    gap = (open_p / prev_close - 1) * 100
    day_ret = (close_p / open_p - 1) * 100  # 当日オープンからクローズ

    gap_results.append({'date': dt, 'gap': gap, 'day_ret': day_ret})

gap_df = pd.DataFrame(gap_results)
gap_df = gap_df[gap_df['gap'].abs() < 5]  # 5%超は除外

for label, lo, hi in [('小GU(0~0.3%)', 0, 0.3), ('中GU(0.3~1%)', 0.3, 1.0), ('大GU(>1%)', 1.0, 99),
                       ('小GD(0~-0.3%)', -0.3, 0), ('中GD(-0.3~-1%)', -1.0, -0.3), ('大GD(<-1%)', -99, -1.0)]:
    if lo >= 0:
        mask = (gap_df['gap'] > lo) & (gap_df['gap'] <= hi if hi < 99 else gap_df['gap'] > lo)
    else:
        mask = (gap_df['gap'] < hi) & (gap_df['gap'] >= lo if lo > -99 else gap_df['gap'] < hi)

    sub = gap_df[mask]
    if len(sub) < 5:
        continue
    cont = (np.sign(sub['gap']) == np.sign(sub['day_ret'])).mean() * 100
    avg  = sub['day_ret'].mean()
    print(f"  {label:20s}: N={len(sub):3d}  ギャップフィル率(反転)={100-cont:.0f}%  平均当日リターン={avg:+.3f}%")

# ─────────────────────────────────────
# 8. ORB（オープンレンジブレイクアウト）分析
# ─────────────────────────────────────
print("\n=== ORB分析（前場開始 8:45-9:15のレンジ） ===")
orb_results = []
ORB_MIN = 30  # 分

for dt, grp in day_df.groupby('date'):
    grp = grp.sort_index()
    # ORBウィンドウ: 8:45-9:15
    orb = grp.between_time('08:45', '09:15')
    rest = grp.between_time('09:16', '15:15')
    if len(orb) < 5 or len(rest) < 10:
        continue

    orb_high = orb['high'].astype(float).max()
    orb_low  = orb['low'].astype(float).min()
    orb_range = orb_high - orb_low

    if orb_range < 10:  # レンジが小さすぎる日は除外
        continue

    open_p = orb['close'].iloc[0]
    close_p = rest['close'].iloc[-1]
    day_ret = (close_p / open_p - 1) * 100

    # レンジ上抜け/下抜けした最初のバー
    broke_up   = (rest['close'].astype(float) > orb_high).any()
    broke_down = (rest['close'].astype(float) < orb_low).any()

    if broke_up and not broke_down:
        direction = 'breakup'
        ret_from_break = day_ret  # 上抜け継続
    elif broke_down and not broke_up:
        direction = 'breakdown'
        ret_from_break = -day_ret
    elif broke_up and broke_down:
        direction = 'both'
        ret_from_break = np.nan
    else:
        direction = 'range'
        ret_from_break = np.nan

    orb_results.append({'date': dt, 'direction': direction, 'day_ret': day_ret,
                         'orb_range': orb_range, 'ret_from_break': ret_from_break})

orb_df = pd.DataFrame(orb_results)
print(f"  分析日数: {len(orb_df)}")
for dir_label in ['breakup', 'breakdown', 'both', 'range']:
    sub = orb_df[orb_df['direction'] == dir_label]
    if len(sub) == 0:
        continue
    # breakup: day_ret > 0 が継続
    if dir_label == 'breakup':
        cont = (sub['day_ret'] > 0).mean() * 100
    elif dir_label == 'breakdown':
        cont = (sub['day_ret'] < 0).mean() * 100
    else:
        cont = np.nan
    avg = sub['day_ret'].mean()
    print(f"  {dir_label:10s}: N={len(sub):3d}  継続率={cont:.0f}%  平均当日リターン={avg:+.3f}%")

# ─────────────────────────────────────
# 9. 可視化
# ─────────────────────────────────────
print("\nグラフ作成中...")
fig = plt.figure(figsize=(18, 14))
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.4)
fig.suptitle('日経平均先物 (JNIc1) モメンタム vs 平均回帰 分析', fontsize=14, fontweight='bold')

# (A) 自己相関
ax1 = fig.add_subplot(gs[0, 0])
lags_plot = [1,2,3,5,10,15,30]
corrs = []
for lag in lags_plot:
    r, _ = stats.pearsonr(valid[:-lag], valid[lag:])
    corrs.append(r)
colors = ['red' if r < 0 else 'steelblue' for r in corrs]
ax1.bar([str(l) for l in lags_plot], corrs, color=colors, alpha=0.8)
ax1.axhline(0, color='black', lw=0.8)
ax1.set_title('(A) 1分足リターン自己相関')
ax1.set_xlabel('ラグ（分）')
ax1.set_ylabel('相関係数')
ax1.set_ylim(-0.3, 0.3)

# (B) 継続率 ヒートマップ的バーチャート（15分）
ax2 = fig.add_subplot(gs[0, 1])
sub15 = res_df[res_df['lookback'] == 15].copy()
labels = [f"{r['direction']}/{r['magnitude']}" for _, r in sub15.iterrows()]
conts = sub15['continuation_rate(%)'].values
colors2 = ['tomato' if c < 50 else 'steelblue' for c in conts]
bars = ax2.barh(labels, conts, color=colors2, alpha=0.8)
ax2.axvline(50, color='black', lw=1, ls='--')
ax2.set_title('(B) 継続率（15分ローリング）')
ax2.set_xlabel('継続率 (%)')
ax2.set_xlim(30, 70)
for bar, val in zip(bars, conts):
    ax2.text(val + 0.3, bar.get_y() + bar.get_height()/2, f'{val:.1f}%', va='center', fontsize=8)

# (C) ルックバック別継続率（上昇方向）
ax3 = fig.add_subplot(gs[0, 2])
for mag in ['小(<0.1%)', '中(0.1-0.3%)', '大(>0.3%)']:
    sub_m = res_df[(res_df['direction'] == '上昇') & (res_df['magnitude'] == mag)]
    if len(sub_m) > 0:
        ax3.plot(sub_m['lookback'], sub_m['continuation_rate(%)'], marker='o', label=mag)
ax3.axhline(50, color='gray', lw=1, ls='--')
ax3.set_title('(C) 上昇後継続率 × ルックバック')
ax3.set_xlabel('ルックバック（分）')
ax3.set_ylabel('継続率 (%)')
ax3.legend(fontsize=8)
ax3.set_ylim(35, 65)

# (D) ギャップ vs 当日リターン
ax4 = fig.add_subplot(gs[1, 0])
ax4.scatter(gap_df['gap'], gap_df['day_ret'], alpha=0.3, s=10, color='gray')
ax4.axhline(0, color='black', lw=0.5)
ax4.axvline(0, color='black', lw=0.5)
z = np.polyfit(gap_df['gap'], gap_df['day_ret'], 1)
xline = np.linspace(gap_df['gap'].min(), gap_df['gap'].max(), 100)
ax4.plot(xline, np.poly1d(z)(xline), 'r-', lw=1.5, label=f'slope={z[0]:.2f}')
ax4.set_title('(D) 寄付ギャップ vs 当日リターン')
ax4.set_xlabel('ギャップ (%)')
ax4.set_ylabel('当日リターン from open (%)')
ax4.legend()

# (E) ORB結果
ax5 = fig.add_subplot(gs[1, 1])
orb_dirs = ['breakup', 'breakdown', 'both', 'range']
orb_counts = [len(orb_df[orb_df['direction'] == d]) for d in orb_dirs]
ax5.pie(orb_counts, labels=orb_dirs, autopct='%1.0f%%', startangle=90)
ax5.set_title('(E) ORBパターン分布')

# (F) ORB後当日リターン
ax6 = fig.add_subplot(gs[1, 2])
orb_means = []
orb_labels = []
for d in ['breakup', 'breakdown']:
    sub = orb_df[orb_df['direction'] == d]['day_ret']
    if len(sub) > 0:
        orb_means.append(sub.mean())
        orb_labels.append(d)
colors_orb = ['steelblue' if m > 0 else 'tomato' for m in orb_means]
ax6.bar(orb_labels, orb_means, color=colors_orb, alpha=0.8)
ax6.axhline(0, color='black', lw=0.8)
ax6.set_title('(F) ORBブレイク後の平均当日リターン')
ax6.set_ylabel('平均リターン (%)')

# (G) 時間帯別平均リターン分布
ax7 = fig.add_subplot(gs[2, :2])
for session_label, color in [('morning', 'steelblue'), ('afternoon', 'orange'), ('night', 'gray')]:
    sub = df[df['sub_session'] == session_label]['ret1'].dropna()
    sub = sub[sub.abs() < 1]
    ax7.hist(sub, bins=80, alpha=0.5, label=session_label, color=color, density=True)
ax7.set_title('(G) セッション別 1分足リターン分布')
ax7.set_xlabel('1分足リターン (%)')
ax7.set_ylabel('密度')
ax7.legend()
ax7.set_xlim(-0.5, 0.5)

# (H) 時系列（2026年以降）
ax8 = fig.add_subplot(gs[2, 2])
recent = df[df.index >= '2026-01-01']['close'].resample('1D').last().dropna()
ax8.plot(recent.index, recent.values, lw=1, color='navy')
ax8.set_title('(H) 日経先物（2026年〜）')
ax8.set_xlabel('日付')
ax8.set_ylabel('価格')
ax8.tick_params(axis='x', rotation=45)

plt.savefig('/Users/Yusuke/claude-code/japan-stocks/analyses/20260423_nikkei_momentum_reversal/result.png',
            dpi=150, bbox_inches='tight')
print("グラフ保存: result.png")

# ─────────────────────────────────────
# 10. サマリー出力
# ─────────────────────────────────────
print("\n" + "="*60)
print("【総合サマリー】")
print("="*60)

# 自己相関1分
r1, p1 = stats.pearsonr(valid[:-1], valid[1:])
print(f"\n■ 1分足自己相関(Lag1): r={r1:+.4f}  → {'平均回帰傾向' if r1 < 0 else 'モメンタム傾向'}")

# 5分継続率
sub5up = res_df[(res_df['lookback']==5) & (res_df['direction']=='上昇') & (res_df['magnitude']=='中(0.1-0.3%)')]
sub5dn = res_df[(res_df['lookback']==5) & (res_df['direction']=='下落') & (res_df['magnitude']=='中(0.1-0.3%)')]
if len(sub5up) > 0:
    print(f"■ 5分上昇後継続率（中幅）: {sub5up['continuation_rate(%)'].values[0]:.1f}%")
if len(sub5dn) > 0:
    print(f"■ 5分下落後継続率（中幅）: {sub5dn['continuation_rate(%)'].values[0]:.1f}%")

# ORB
breakup_cont = (orb_df[orb_df['direction']=='breakup']['day_ret'] > 0).mean() * 100
breakdn_cont = (orb_df[orb_df['direction']=='breakdown']['day_ret'] < 0).mean() * 100
print(f"■ ORB上抜け後の継続率: {breakup_cont:.0f}%")
print(f"■ ORB下抜け後の継続率: {breakdn_cont:.0f}%")

# ギャップ
gap_slope = np.polyfit(gap_df['gap'], gap_df['day_ret'], 1)[0]
print(f"■ ギャップ→当日リターン傾き: {gap_slope:+.2f}  → {'ギャップフィル傾向' if gap_slope < 0 else 'ギャップ継続傾向'}")

print("\n完了")
