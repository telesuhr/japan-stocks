#!/usr/bin/env python3
"""全採用戦略 継続性検証 — result.png 生成"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams.update({
    'font.family': ['IPAexGothic', 'Noto Sans CJK JP', 'Hiragino Sans', 'sans-serif'],
    'axes.unicode_minus': False,
    'figure.facecolor': 'white',
    'axes.facecolor': '#f8f9fa',
    'grid.alpha': 0.3,
})

# READMEのSharpe vs 今回のSharpe
strategies = [
    'lme_on_copper',
    'topix_overnight',
    'eneos_vwap_trend',
    'vwap_morning_meanrevert',
    'orb_breakout_long',
    'lasertec_ma25_support',
    'sox_overnight_short\n(TOPIX代理)',
]
readme_sharpe = [12.34, 4.79, 5.54, 6.11, 2.15, 7.68, 2.11]
current_sharpe = [-3.14, 6.27, 3.81, 6.76, 2.31, 21.47, -49.47]
current_n = [11, 75, 80, 52, 344, 33, 142]

# 色分け: ✅緑 / ⚠️黄 / ❌赤
def verdict_color(cur, ref):
    if cur >= ref * 0.7:
        return '#2ecc71'
    elif cur >= 2.0:
        return '#f39c12'
    else:
        return '#e74c3c'

colors = [verdict_color(c, r) for c, r in zip(current_sharpe, readme_sharpe)]
labels_short = [
    'LME ON\n銅Long', 'TOPIX\nON Long', 'ENEOS\nVWAP', 'VWAP\n平均回帰',
    'ORB\nブレイク', 'Laser\nMA25', 'SOX\nShort'
]

fig = plt.figure(figsize=(12, 6.75), facecolor='white')
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

# ── 左: Sharpe 比較バーチャート ──
x = np.arange(len(strategies))
w = 0.35
bars1 = ax1.bar(x - w/2, readme_sharpe, w, label='README記載', color='#3498db', alpha=0.7)
bars2 = ax1.bar(x + w/2, [min(c, 25) for c in current_sharpe], w,
                label='今回検証', color=colors, alpha=0.9)

ax1.axhline(2.0, color='gray', linestyle='--', linewidth=0.8, label='最低基準 Sharpe=2.0')
ax1.set_xticks(x)
ax1.set_xticklabels(labels_short, fontsize=8)
ax1.set_ylabel('Sharpe比 (コスト控除後)')
ax1.set_title('Sharpe比 比較 (README vs 今回)', fontsize=11, fontweight='bold')
ax1.legend(fontsize=8)
ax1.set_ylim(-6, 26)
ax1.grid(axis='y')

# 値ラベル
for bar, val in zip(bars2, current_sharpe):
    disp = min(val, 25)
    ax1.text(bar.get_x() + bar.get_width()/2,
             disp + 0.3 if disp >= 0 else disp - 1.5,
             f'{val:.1f}', ha='center', va='bottom' if disp >= 0 else 'top',
             fontsize=7, fontweight='bold')

# ── 右: 判定サマリー表 ──
ax2.axis('off')
table_data = []
for i, (s, rd, cur, n) in enumerate(zip(labels_short, readme_sharpe, current_sharpe, current_n)):
    name = s.replace('\n', ' ')
    if cur >= rd * 0.7:
        v = '✅ 継続'
    elif cur >= 2.0:
        v = '⚠️ 低下'
    else:
        v = '❌ 劣化'
    table_data.append([name, f'{rd:.2f}', f'{cur:.2f}', str(n), v])

col_labels = ['戦略', 'README\nSharpe', '今回\nSharpe', 'N', '判定']
tbl = ax2.table(
    cellText=table_data,
    colLabels=col_labels,
    cellLoc='center', loc='center',
    bbox=[0.0, 0.05, 1.0, 0.90]
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)

# 行色付け
for i, row in enumerate(table_data):
    v = row[4]
    color = '#d5f5e3' if '継続' in v else ('#fef9e7' if '低下' in v else '#fadbd8')
    for j in range(5):
        tbl[(i+1, j)].set_facecolor(color)

ax2.set_title('戦略別 判定サマリー', fontsize=11, fontweight='bold')

# ヘッダー色
for j in range(5):
    tbl[(0, j)].set_facecolor('#2c3e50')
    tbl[(0, j)].set_text_props(color='white', fontweight='bold')

fig.suptitle('採用戦略 継続性バックテスト — 2026-05-09',
             fontsize=14, fontweight='bold', y=0.98)

p_green = mpatches.Patch(color='#d5f5e3', label='✅ 継続 (70%以上)')
p_yellow = mpatches.Patch(color='#fef9e7', label='⚠️ 低下 (Sharpe≥2.0)')
p_red = mpatches.Patch(color='#fadbd8', label='❌ 劣化 (Sharpe<2.0)')
fig.legend(handles=[p_green, p_yellow, p_red],
           loc='lower center', ncol=3, fontsize=9, framealpha=0.9,
           bbox_to_anchor=(0.5, 0.0))

fig.text(0.99, 0.01,
         'データ: 2024-11〜2026-05 / 日本株1分足+日足 (Refinitiv) | コスト2bps往復',
         ha='right', va='bottom', fontsize=7, color='gray')

plt.tight_layout(rect=[0, 0.06, 1, 0.97])
plt.savefig('result.png', dpi=100, bbox_inches='tight', facecolor='white')
print("result.png 保存完了")
