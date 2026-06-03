"""result.png — 17業種ローテーション: 資金循環は逆張りでなくモメンタム"""
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
fig.suptitle('日本株セクター「資金循環」の正体 — 過熱売り(逆張り)でなくモメンタム継続',
             fontsize=15, fontweight='bold', y=0.98)

# 左: ルックバック別 MOM vs REV Sharpe (K=3)
Ls = ['1ヶ月', '3ヶ月', '6ヶ月', '12ヶ月']
mom = [0.73, 1.49, 0.99, 1.40]
rev = [0.26, 0.19, 0.33, 0.19]
x = np.arange(len(Ls)); w = 0.38
ax1.bar(x - w/2, mom, w, label='MOM 上位3業種ロング(順位陥落で利確)', color='#1f77b4')
ax1.bar(x + w/2, rev, w, label='REV 下位3業種ロング(過熱売り=資金循環逆張り)', color='#d62728')
ax1.axhline(0.82, color='gray', ls=':', lw=1.2, label='TOPIX買い持ち 0.82')
ax1.axhline(0, color='gray', lw=0.8)
ax1.set_xticks(x); ax1.set_xticklabels(Ls)
ax1.set_xlabel('相対強さの計測期間 (trailing)')
ax1.set_ylabel('TOPIX超過 Sharpe (月次非重複・√12年率・コスト10bps)')
ax1.set_title('強い業種は翌月も強い(MOM圧勝)。出遅れ拾い(REV)は機能せず\n規範IC L=3 +0.094(t+2.8)/L=12 +0.118(t+3.3) 全て正', fontsize=10)
ax1.legend(fontsize=8, loc='upper right'); ax1.set_ylim(0, 1.9)
for i, v in enumerate(mom): ax1.text(i - w/2, v + 0.03, f'{v:+.2f}', ha='center', fontsize=8)
for i, v in enumerate(rev): ax1.text(i + w/2, v + 0.03, f'{v:+.2f}', ha='center', fontsize=8)

# 右: 年別 MOM L3K3 超過Sharpe (9年すべてプラス)
yrs = ['2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024', '2025']
ysh = [4.17, 0.61, 1.77, 0.21, 3.23, 2.53, 1.18, 0.59, 1.94]
cols = ['#2ca02c' if v > 0 else '#d62728' for v in ysh]
ax2.bar(yrs, ysh, color=cols)
ax2.axhline(0, color='gray', lw=0.8)
ax2.set_ylabel('年別 TOPIX超過 Sharpe')
ax2.set_title('MOM(3ヶ月)上位3業種は9年すべてプラス=レジーム非依存\n高流動性(ADV≥10億)で全Sharpe2.09(IS3.03/OOS1.53,t6.5)に上昇', fontsize=10)
for i, v in enumerate(ysh): ax2.text(i, v + 0.06, f'{v:+.2f}', ha='center', fontsize=8)
ax2.set_ylim(0, 4.6)

fig.text(0.99, 0.01, 'データ: 2016-06〜2026-06 / TSE 17業種・流動性≥1億円/日・等加重・月次リバランス (JQuants日足)',
         ha='right', va='bottom', fontsize=8, color='gray')
plt.tight_layout(rect=[0, 0.02, 1, 0.95])
plt.savefig('result.png', dpi=100, bbox_inches='tight', facecolor='white')
print("saved result.png")
