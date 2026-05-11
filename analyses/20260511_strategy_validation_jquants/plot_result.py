#!/usr/bin/env python3
"""result.png 生成 — JQuants継続性検証"""
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

labels = ['topix\novernight','eneos\nVWAP','VWAP\n平均回帰','ORB\nブレイク','laser\nMA25','bank\n吸収','pair\n18ペア']
ref_sh = [6.27, 3.81, 6.76, 2.31, 7.57, 1.84, 1.37]
new_sh = [0.58, 2.97, 4.81, 2.19, 2.95, 3.94, 0.65]
N      = [363,  97,   77,   469,  39,   908,  447]

def color(c, r):
    if c >= r * 0.7: return '#2ecc71'
    elif c >= 1.0:   return '#f39c12'
    else:            return '#e74c3c'
colors = [color(c, r) for c, r in zip(new_sh, ref_sh)]

fig = plt.figure(figsize=(12, 6.75), facecolor='white')
ax1 = fig.add_subplot(1, 2, 1); ax2 = fig.add_subplot(1, 2, 2)

x = np.arange(len(labels)); w = 0.35
ax1.bar(x - w/2, ref_sh, w, label='前回検証 (archive 1年)', color='#3498db', alpha=0.7)
ax1.bar(x + w/2, new_sh, w, label='今回検証 (JQuants 5年)', color=colors, alpha=0.9)
ax1.axhline(2.0, color='gray', linestyle='--', linewidth=0.8, label='採用基準 Sharpe=2.0')
ax1.axhline(0,   color='black', linewidth=0.5)
ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=8)
ax1.set_ylabel('Sharpe比 (年率)')
ax1.set_title('Sharpe比 比較 (前回 vs JQuants長期)', fontsize=11, fontweight='bold')
ax1.legend(fontsize=8, loc='upper right')
ax1.grid(axis='y')
for i, (rs, ns) in enumerate(zip(ref_sh, new_sh)):
    ax1.text(i + w/2, ns + 0.2 if ns >= 0 else ns - 0.5,
             f'{ns:.2f}', ha='center', va='bottom' if ns>=0 else 'top',
             fontsize=8, fontweight='bold')

ax2.axis('off')
table_data = []
for lbl, rs, ns, n in zip(labels, ref_sh, new_sh, N):
    if ns >= rs * 0.7: v = '✅ 継続'
    elif ns >= 1.0:    v = '⚠️ 低下'
    else:              v = '❌ 劣化'
    table_data.append([lbl.replace('\n',' '), f'{rs:.2f}', f'{ns:.2f}', str(n), v])

tbl = ax2.table(cellText=table_data,
                colLabels=['戦略','前回\nSharpe','今回\nSharpe','N','判定'],
                cellLoc='center', loc='center', bbox=[0,0.05,1,0.90])
tbl.auto_set_font_size(False); tbl.set_fontsize(9)
for i, row in enumerate(table_data):
    bg = '#d5f5e3' if '継続' in row[4] else ('#fef9e7' if '低下' in row[4] else '#fadbd8')
    for j in range(5): tbl[(i+1, j)].set_facecolor(bg)
for j in range(5):
    tbl[(0, j)].set_facecolor('#2c3e50'); tbl[(0, j)].set_text_props(color='white', fontweight='bold')
ax2.set_title('戦略別 判定サマリー (Nは検証トレード数)', fontsize=11, fontweight='bold')

fig.suptitle('採用7戦略 JQuantsベース継続性検証 — 2026-05-11',
             fontsize=14, fontweight='bold', y=0.98)
fig.legend(handles=[mpatches.Patch(color='#d5f5e3', label='✅継続'),
                    mpatches.Patch(color='#fef9e7', label='⚠️低下'),
                    mpatches.Patch(color='#fadbd8', label='❌劣化')],
           loc='lower center', ncol=3, fontsize=9, bbox_to_anchor=(0.5, 0.0))
fig.text(0.99, 0.01,
         'データ: イントラ2024-05〜2026-05 / 日足2021-05〜2026-05 (JQuants) | コスト2bps片道',
         ha='right', va='bottom', fontsize=7, color='gray')
plt.tight_layout(rect=[0, 0.06, 1, 0.97])
plt.savefig('result.png', dpi=100, bbox_inches='tight', facecolor='white')
print("result.png 保存完了")
