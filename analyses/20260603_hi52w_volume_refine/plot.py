"""result.png — #10精密化: ボラ調整で改善も2.0未達 を1枚で"""
import sys, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
sys.stdout.reconfigure(line_buffering=True)

plt.rcParams.update({
    'font.family': ['Noto Sans CJK JP', 'IPAexGothic', 'sans-serif'],
    'axes.unicode_minus': False, 'figure.facecolor': 'white',
    'axes.facecolor': '#f8f9fa', 'grid.alpha': 0.3,
})
try:
    from matplotlib import font_manager
    font_manager.fontManager.addfont('/root/.fonts/NotoSansJP.ttf')
    plt.rcParams['font.family'] = ['Noto Sans JP', 'sans-serif']
except Exception:
    pass

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6.75), facecolor='white')
fig.suptitle('#10「52週高値×出来高急増」精密化 — ボラ調整で改善も昇格基準(2.0)未達',
             fontsize=15, fontweight='bold', y=0.98)

# 左: 変種比較 (L/S全期間 と Long-only市場超過 Sharpe)
labels = ['元\nhi_prox', 'ボラ調整\nhi_voladj', '出来高を\n連続信号化', '複合\nz(hi)+z(vol)']
ls = [0.85, 1.22, -0.94, -0.12]
lo = [0.44, 0.84, -0.47, -0.22]
x = np.arange(len(labels)); w = 0.38
ax1.bar(x-w/2, ls, w, label='L/S (上位/下位5分位)', color='#1f77b4')
ax1.bar(x+w/2, lo, w, label='Long-only (市場超過)', color='#ff7f0e')
ax1.axhline(2.0, color='red', ls='--', lw=1.2, label='昇格基準 2.0')
ax1.axhline(0, color='gray', lw=0.8)
ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=9)
ax1.set_ylabel('Sharpe (月次非重複, コスト20bps, √(252/20)年率)')
ax1.set_title('シグナル設計比較: ボラ調整が効く・出来高は連続化すると逆効果', fontsize=10)
ax1.legend(fontsize=8, loc='upper right'); ax1.set_ylim(-1.3, 2.3)
for i, v in enumerate(ls): ax1.text(i-w/2, v+(0.05 if v>=0 else -0.18), f'{v:+.2f}', ha='center', fontsize=8)
for i, v in enumerate(lo): ax1.text(i+w/2, v+(0.05 if v>=0 else -0.18), f'{v:+.2f}', ha='center', fontsize=8)

# 右: 年別L/S Sharpe (元 vs ボラ調整) — レジーム依存の改善
yrs = ['2023', '2024', '2025']
base_y = [3.44, -0.04, 0.74]
va_y = [1.99, 0.21, 1.00]
x2 = np.arange(len(yrs))
ax2.bar(x2-w/2, base_y, w, label='元 hi_prox', color='#aec7e8')
ax2.bar(x2+w/2, va_y, w, label='ボラ調整 hi_voladj', color='#1f77b4')
ax2.axhline(0, color='gray', lw=0.8)
ax2.set_xticks(x2); ax2.set_xticklabels(yrs)
ax2.set_ylabel('L/S Sharpe (年別)')
ax2.set_title('年別: 元は2023集中→ボラ調整で各年プラスに均す(全期間1.22/OOS0.86)', fontsize=10)
ax2.legend(fontsize=8)
for i, v in enumerate(base_y): ax2.text(i-w/2, v+0.05, f'{v:+.2f}', ha='center', fontsize=8)
for i, v in enumerate(va_y): ax2.text(i+w/2, v+0.05, f'{v:+.2f}', ha='center', fontsize=8)

fig.text(0.99, 0.01, 'データ: 2021-10〜2026-06 / 流動性上位500・月次L/S・セクター中立 (JQuants日足)',
         ha='right', va='bottom', fontsize=8, color='gray')
plt.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.savefig('result.png', dpi=100, bbox_inches='tight', facecolor='white')
print("saved result.png")
