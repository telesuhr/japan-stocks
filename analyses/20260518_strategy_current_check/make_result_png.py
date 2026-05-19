"""9戦略 直近検証 → result.png"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from templates._style import (
    apply_base_style, add_footer, add_kpi_row, save,
    BG, PANEL, GRID, TEXT, TEXT_DIM,
    ACCENT, ACCENT2, ACCENT3, ACCENT4,
    LW, LW_THIN, LW_THICK,
)

df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'summary.csv'))

# 並び順をベースSharpe降順に
df = df.sort_values('baseline', ascending=False).reset_index(drop=True)

apply_base_style()
fig = plt.figure(figsize=(12, 6.75), facecolor=BG)

fig.text(0.5, 0.97, '採用9戦略 直近6ヶ月 継続性チェック (2025-11〜2026-05)',
         ha='center', va='top', fontsize=14, fontweight='bold', color=TEXT)
fig.text(0.5, 0.935, '5年検証Sharpe (Base) vs 直近6ヶ月Sharpe (Recent) / 昇格基準 Sharpe≥2.0',
         ha='center', va='top', fontsize=8.5, color=TEXT_DIM)

# KPI: 継続/低下/劣化のカウント
n_ok    = (df['judge'].str.contains('継続')).sum()
n_low   = (df['judge'].str.contains('低下')).sum()
n_bad   = (df['judge'].str.contains('劣化')).sum()
add_kpi_row(fig, [
    {"label": "戦略数", "value": str(len(df)), "color": ACCENT},
    {"label": "✅ 継続",     "value": str(n_ok),  "color": ACCENT2},
    {"label": "⚠ 低下",      "value": str(n_low), "color": ACCENT4},
    {"label": "❌ 劣化",     "value": str(n_bad), "color": ACCENT3},
    {"label": "判定継続率",  "value": f"{n_ok/len(df)*100:.0f}%", "color": ACCENT2},
], y=0.855)

gs = gridspec.GridSpec(2, 2, figure=fig,
                       left=0.07, right=0.97,
                       top=0.74, bottom=0.10,
                       hspace=0.55, wspace=0.30)

ax_sh   = fig.add_subplot(gs[0, :])
ax_diff = fig.add_subplot(gs[1, 0])
ax_perf = fig.add_subplot(gs[1, 1])

for ax in [ax_sh, ax_diff, ax_perf]:
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.tick_params(colors=TEXT_DIM, labelsize=8)
    ax.grid(axis='y', zorder=0)

# ─── ① 5年Base vs 直近Sharpe 比較 ───
names = df['name'].tolist()
short_names = {
    'vwap_morning_meanrevert':       'vwap_meanrevert',
    'bank_absorption':               'bank_absorption',
    'lasertec_ma25_support':         'lasertec_ma25',
    'eneos_vwap_trend':              'eneos_vwap',
    'orb_breakout_long':             'orb_breakout',
    'oversold_ma25_reversal':        'oversold_ma25',
    'large_cap_oversold_reversal':   'large_cap_oversold',
    'earnings_pead':                 'earnings_pead',
    'pre_earnings_drift':            'pre_earnings_drift',
}
labels = [short_names[n] for n in names]
base   = df['baseline'].values
recent = df['sharpe'].values

x = np.arange(len(labels))
w = 0.38
ax_sh.bar(x - w/2, base,   w, color=TEXT_DIM, alpha=0.7, label='5年Sharpe (Base)')
recent_colors = []
for j, v in zip(df['judge'].values, recent):
    if '継続' in j: recent_colors.append(ACCENT2)
    elif '低下' in j: recent_colors.append(ACCENT4)
    else: recent_colors.append(ACCENT3)
ax_sh.bar(x + w/2, recent, w, color=recent_colors, alpha=0.9, label='直近6M Sharpe')
ax_sh.axhline(2.0, color=ACCENT, linewidth=LW, linestyle=':', label='昇格基準')
ax_sh.axhline(0, color=GRID, linewidth=LW_THIN, linestyle='--')
ax_sh.set_xticks(x)
ax_sh.set_xticklabels(labels, fontsize=8, color=TEXT_DIM, rotation=20, ha='right')
ax_sh.set_ylabel('Sharpe', fontsize=8, color=TEXT_DIM)
ax_sh.set_title('Sharpe比較: 5年検証(Base) vs 直近6ヶ月(Recent)',
                 fontsize=9, color=TEXT, pad=4, loc='left', fontweight='bold')
ax_sh.legend(fontsize=8, framealpha=0.7, labelcolor=TEXT_DIM)
for xi, v in zip(x - w/2, base):
    ax_sh.text(xi, v + 0.15, f'{v:.1f}', ha='center', fontsize=6.5, color=TEXT_DIM)
for xi, v in zip(x + w/2, recent):
    ax_sh.text(xi, v + (0.15 if v >= 0 else -0.4), f'{v:+.2f}',
               ha='center', fontsize=7, color=TEXT, fontweight='bold',
               va='bottom' if v >= 0 else 'top')

# ─── ② 差分 (Recent - Base) ───
diff = recent - base
ax_diff.barh(range(len(labels)), diff,
             color=[ACCENT2 if d >= 0 else ACCENT3 for d in diff], alpha=0.85)
ax_diff.set_yticks(range(len(labels)))
ax_diff.set_yticklabels(labels, fontsize=7, color=TEXT_DIM)
ax_diff.invert_yaxis()
ax_diff.axvline(0, color=GRID, linewidth=LW_THIN, linestyle='--')
ax_diff.axvline(-1.5, color=ACCENT4, linewidth=LW, linestyle=':')
ax_diff.set_xlabel('Δ Sharpe (Recent - Base)', fontsize=8, color=TEXT_DIM)
ax_diff.set_title('5年基準からの乖離 (赤=劣化, 緑=改善)',
                   fontsize=9, color=TEXT, pad=4, loc='left', fontweight='bold')
for i, d in enumerate(diff):
    ax_diff.text(d + (0.05 if d >= 0 else -0.05), i,
                  f'{d:+.2f}', va='center',
                  ha='left' if d >= 0 else 'right',
                  fontsize=7, color=TEXT)

# ─── ③ WR vs PF 散布 ───
for j, n, wr, pf in zip(df['judge'], names, df['WR'], df['PF']):
    if '継続' in j:    c = ACCENT2
    elif '低下' in j: c = ACCENT4
    else: c = ACCENT3
    ax_perf.scatter(wr, pf, s=120, color=c, alpha=0.85, edgecolor='white', linewidth=0.5)
    ax_perf.annotate(short_names[n], (wr, pf),
                      fontsize=6.5, color=TEXT_DIM,
                      xytext=(5, 3), textcoords='offset points')
ax_perf.axhline(1.3, color=ACCENT, linewidth=LW, linestyle=':', alpha=0.7)
ax_perf.axvline(50,  color=GRID, linewidth=LW_THIN, linestyle='--')
ax_perf.text(50.5, 1.32, 'PF=1.3', fontsize=7, color=ACCENT)
ax_perf.set_xlabel('勝率 %', fontsize=8, color=TEXT_DIM)
ax_perf.set_ylabel('Profit Factor', fontsize=8, color=TEXT_DIM)
ax_perf.set_title('直近6ヶ月 勝率 × PF',
                   fontsize=9, color=TEXT, pad=4, loc='left', fontweight='bold')

add_footer(fig, source='JQuants 日足+1分足',
           period='直近6ヶ月: 2025-11-15 〜 2026-05-15')

out = os.path.join(os.path.dirname(__file__), 'result.png')
save(fig, out)
