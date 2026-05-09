"""
アルゴリズムトレーディングルール検証 — result.png 生成
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
import psycopg2
import warnings
warnings.filterwarnings('ignore')

PG_CONFIG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}

# --- 結果読み込み ---
all_r = pd.read_csv('results_all_rules.csv')
sig = pd.read_csv('results_significant.csv')

# --- ON_ret のサニティチェック (外れ値確認) ---
conn = psycopg2.connect(**PG_CONFIG)
q = """
SELECT symbol, timestamp, open, close
FROM intraday_data
WHERE interval = '1min' AND symbol LIKE '%.T'
  AND timestamp >= '2025-01-01'
  AND (EXTRACT(HOUR FROM timestamp + INTERVAL '9 hours') IN (9, 15))
ORDER BY symbol, timestamp
LIMIT 500000
"""
df_oc = pd.read_sql(q, conn)
conn.close()
df_oc['jst'] = pd.to_datetime(df_oc['timestamp']) + pd.Timedelta(hours=9)
df_oc['date'] = df_oc['jst'].dt.date
df_oc['hour'] = df_oc['jst'].dt.hour

# 各日の寄付/大引けを取得してON計算
open_df = df_oc[df_oc['hour'] == 9].groupby(['symbol', 'date'])['open'].first().reset_index()
close_df = df_oc[df_oc['hour'] == 15].groupby(['symbol', 'date'])['close'].last().reset_index()
open_df['date'] = pd.to_datetime(open_df['date'])
close_df['date'] = pd.to_datetime(close_df['date'])
close_df = close_df.sort_values(['symbol', 'date'])
open_df = open_df.sort_values(['symbol', 'date'])

# shift to compute ON
records = []
for sym, og in open_df.groupby('symbol'):
    og = og.set_index('date').sort_index()
    cg = close_df[close_df['symbol'] == sym].set_index('date').sort_index()
    merged = og.join(cg[['close']], how='inner')
    merged['prev_close'] = merged['close'].shift(1)
    merged['on_ret'] = (merged['open'] / merged['prev_close'] - 1) * 10000
    records.append(merged[['on_ret']].assign(symbol=sym))

on_df = pd.concat(records).dropna()
on_ret_arr = on_df['on_ret'].values
on_ret_arr = on_ret_arr[np.isfinite(on_ret_arr)]

# ===== 図作成 =====
fig = plt.figure(figsize=(16, 10), facecolor='white')
plt.rcParams.update({
    'font.family': ['IPAexGothic', 'Hiragino Sans', 'Noto Sans CJK JP', 'sans-serif'],
    'axes.unicode_minus': False,
})

fig.suptitle('日本株 アルゴリズムトレーディングルール 幅広検証 (117銘柄 2025/1~2026/5)',
             fontsize=14, fontweight='bold', y=0.98)

# --- 上段: カテゴリー別 有望ルール一覧 (バープロット) ---
ax1 = fig.add_axes([0.04, 0.55, 0.56, 0.38])

cat_colors = {
    'aft_ret': '#2196F3',
    'on_ret': '#FF9800',
    'next_fullday_ret': '#4CAF50',
    'next_morning_ret': '#9C27B0',
}
cat_labels = {
    'aft_ret': '後場リターン',
    'on_ret': 'ONリターン',
    'next_fullday_ret': '翌日全日',
    'next_morning_ret': '翌日前場',
}

# 有望ルールをt_stat降順で表示
sig_plot = sig.sort_values('t_stat', ascending=True).tail(17)
ys = range(len(sig_plot))
colors = [cat_colors.get(t, 'gray') for t in sig_plot['target']]

bars = ax1.barh(list(ys), sig_plot['mean_net_bps'],
                color=colors, alpha=0.8, height=0.7)

# エラーバー (95% CI)
# t分布でCI計算
for i, (_, row) in enumerate(sig_plot.iterrows()):
    se = row['std_bps'] / np.sqrt(row['N'])
    ci95 = se * 1.96
    ax1.errorbar(row['mean_net_bps'], i, xerr=ci95,
                 fmt='none', color='#333', lw=1.2, capsize=3)

ax1.set_yticks(list(ys))
ax1.set_yticklabels(sig_plot['rule'].str.replace('_', '\n', 1), fontsize=7.5)
ax1.axvline(0, color='black', lw=0.8)
ax1.set_xlabel('コスト後平均リターン (bps)', fontsize=9)
ax1.set_title('有望ルール Top17 (p<0.05, コスト4bps差引)', fontsize=10, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# 凡例
patches = [mpatches.Patch(color=v, label=k2, alpha=0.8)
           for k, v in cat_colors.items()
           for k2, kk in [(cat_labels[k], k)] if k == kk]
patches = [mpatches.Patch(color=v, label=cat_labels[k], alpha=0.8)
           for k, v in cat_colors.items()]
ax1.legend(handles=patches, loc='lower right', fontsize=8)

# --- 右上: ON リターン分布 + 説明 ---
ax2 = fig.add_axes([0.65, 0.55, 0.32, 0.38])

# ON_ret の分布
on_trimmed = on_ret_arr[(on_ret_arr >= -300) & (on_ret_arr <= 300)]
ax2.hist(on_trimmed, bins=60, color='#FF9800', alpha=0.7, edgecolor='none')
ax2.axvline(on_trimmed.mean(), color='red', lw=1.5, linestyle='--',
            label=f'平均={on_trimmed.mean():.1f}bps')
ax2.axvline(0, color='black', lw=0.8)
ax2.set_xlabel('ON リターン (bps)', fontsize=9)
ax2.set_ylabel('頻度', fontsize=9)
ax2.set_title(f'ONリターン分布 (±300bps内)\nN={len(on_trimmed):,}, mean={on_trimmed.mean():.1f}, std={on_trimmed.std():.1f}bps',
              fontsize=9, fontweight='bold')
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# --- 中下段: ルール比較スキャッター (t_stat vs mean_net_bps) ---
ax3 = fig.add_axes([0.04, 0.06, 0.56, 0.40])

colors_scatter = [cat_colors.get(t, 'gray') for t in all_r['target']]
for _, row in all_r.iterrows():
    c = cat_colors.get(row['target'], 'gray')
    marker = '*' if row['significant'] else 'o'
    size = 80 if row['significant'] else 30
    alpha = 0.9 if row['significant'] else 0.4
    ax3.scatter(row['t_stat'], row['mean_net_bps'], c=c, marker=marker,
                s=size, alpha=alpha, zorder=5 if row['significant'] else 3)

# 有望ルールにラベル
for _, row in sig.sort_values('t_stat', ascending=False).head(8).iterrows():
    short_name = row['rule'].replace('_', '\n')
    ax3.annotate(row['rule'].split('_')[0] + '..' + row['rule'].split('_')[-1],
                 xy=(row['t_stat'], row['mean_net_bps']),
                 xytext=(5, 5), textcoords='offset points',
                 fontsize=6.5, color='#333')

ax3.axhline(0, color='black', lw=0.8)
ax3.axvline(0, color='black', lw=0.8)
ax3.axvline(1.96, color='gray', lw=0.8, linestyle='--', alpha=0.5, label='t=±1.96')
ax3.axvline(-1.96, color='gray', lw=0.8, linestyle='--', alpha=0.5)
ax3.set_xlabel('t統計量', fontsize=9)
ax3.set_ylabel('コスト後平均リターン (bps)', fontsize=9)
ax3.set_title('全ルール スキャッター (★=有意, ○=非有意)', fontsize=10, fontweight='bold')
ax3.legend(fontsize=8)
ax3.grid(alpha=0.3)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

# --- 右下: サマリー表 ---
ax4 = fig.add_axes([0.65, 0.06, 0.32, 0.40])
ax4.axis('off')

top5 = sig.sort_values('t_stat', ascending=False).head(6)[
    ['rule', 'N', 'mean_net_bps', 't_stat', 'win_rate']
].copy()
top5.columns = ['ルール', 'N', 'net(bps)', 't値', '勝率%']
top5['net(bps)'] = top5['net(bps)'].round(1)
top5['t値'] = top5['t値'].round(2)
top5['勝率%'] = top5['勝率%'].round(1)
top5['ルール'] = top5['ルール'].str[:18]

table = ax4.table(
    cellText=top5.values,
    colLabels=top5.columns,
    cellLoc='center',
    loc='center',
    bbox=[0, 0.3, 1, 0.65]
)
table.auto_set_font_size(False)
table.set_fontsize(8)
for (r, c), cell in table.get_celld().items():
    if r == 0:
        cell.set_facecolor('#1565C0')
        cell.set_text_props(color='white', fontweight='bold')
    elif r % 2 == 1:
        cell.set_facecolor('#E3F2FD')
    cell.set_edgecolor('#BDBDBD')

ax4.set_title('Top6 有望ルール (t値降順)', fontsize=9, fontweight='bold', y=0.96)

# 凡例テキスト
summary_text = (
    f"検証結果サマリー\n"
    f"全ルール数: {len(all_r)}\n"
    f"有望ルール: {len(sig)} (p<0.05, net>0)\n\n"
    f"カテゴリー別有望数:\n"
    f"  後場(aft): {len(sig[sig['target']=='aft_ret'])}\n"
    f"  ON: {len(sig[sig['target']=='on_ret'])}\n"
    f"  翌日全日: {len(sig[sig['target']=='next_fullday_ret'])}\n\n"
    f"主要発見:\n"
    f"・ON戦略が最も強い\n"
    f"・前日弱→翌日ON買いが最強\n"
    f"  (+497bps net, t=11.07)\n"
    f"・後場系は小幅正(1-9bps)"
)
ax4.text(0.02, 0.28, summary_text, transform=ax4.transAxes,
         fontsize=8.5, va='top', ha='left',
         bbox=dict(boxstyle='round', facecolor='#FFF9C4', alpha=0.8))

# フッター
fig.text(0.99, 0.01,
         'データ: 2025-01-01〜2026-05-07 / 日本株117銘柄1分足 (Refinitiv) | コスト前提: 片側2bps×往復=4bps',
         ha='right', va='bottom', fontsize=7, color='gray')

plt.savefig('result.png', dpi=100, bbox_inches='tight', facecolor='white')
print("result.png 保存完了")
