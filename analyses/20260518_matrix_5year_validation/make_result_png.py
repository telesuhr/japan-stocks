"""5年検証 → result.png"""
import sys, os, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

from templates._style import (
    apply_base_style, add_footer, add_kpi_row, save,
    BG, PANEL, GRID, TEXT, TEXT_DIM,
    ACCENT, ACCENT2, ACCENT3, ACCENT4,
    LW, LW_THIN, LW_THICK,
)

with open(os.path.join(os.path.dirname(__file__), 'results.pkl'), 'rb') as f:
    R = pickle.load(f)
overall, yearly, rolling, strats, years = R['overall'], R['yearly'], R['rolling'], R['strats'], R['years']

strat_names = list(strats.keys())
short_names = {
    'Baseline (Buy&Hold)':            'Baseline',
    'Matrix (6コンポーネント)':           'Matrix',
    'Aggressive L/S (9コンポーネント)':   'Aggressive',
    'Monday ON-Down Buy (条件付き)':    'MonON-Down',
}
colors = {
    'Baseline (Buy&Hold)':            TEXT_DIM,
    'Matrix (6コンポーネント)':           ACCENT2,
    'Aggressive L/S (9コンポーネント)':   ACCENT4,
    'Monday ON-Down Buy (条件付き)':    ACCENT3,
}

apply_base_style()
fig = plt.figure(figsize=(12, 6.75), facecolor=BG)

fig.text(0.5, 0.97, '【検証】マトリクス戦略 5年バックテスト (2021-05〜2026-05)',
         ha='center', va='top', fontsize=14, fontweight='bold', color=TEXT)
fig.text(0.5, 0.935, '直近半年のSharpe 6.83 → 5年通算 1.25 / 昇格基準 (Sharpe≥2.0) クリアならず',
         ha='center', va='top', fontsize=8.5, color=ACCENT3)

add_kpi_row(fig, [
    {"label": "Aggressive 5年Sharpe", "value": f"{overall['Aggressive L/S (9コンポーネント)']['sharpe']:.2f}", "color": ACCENT3},
    {"label": "2026 Sharpe (直近)",   "value": f"+{yearly['Aggressive L/S (9コンポーネント)'].get(2026, 0):.2f}", "color": ACCENT2},
    {"label": "2023 Sharpe (悪年)",  "value": f"{yearly['Aggressive L/S (9コンポーネント)'].get(2023, 0):.2f}", "color": ACCENT3},
    {"label": "12ヶ月ローリング中央値", "value": f"+{np.median(rolling['Aggressive L/S (9コンポーネント)'].values):.2f}", "color": ACCENT4},
    {"label": "昇格基準", "value": "2.0", "color": ACCENT},
], y=0.855)

gs = gridspec.GridSpec(2, 2, figure=fig,
                       left=0.06, right=0.97,
                       top=0.74, bottom=0.10,
                       hspace=0.55, wspace=0.30)

ax_cum    = fig.add_subplot(gs[0, :])
ax_yearly = fig.add_subplot(gs[1, 0])
ax_roll   = fig.add_subplot(gs[1, 1])

for ax in [ax_cum, ax_yearly, ax_roll]:
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.tick_params(colors=TEXT_DIM, labelsize=8)
    ax.grid(axis='y', zorder=0)

# ─── ① 5年累積カーブ ─────────
for name in strat_names:
    cum = overall[name]['cum']
    color = colors[name]
    lw = LW_THICK + 0.5 if 'Aggressive' in name else LW
    ax_cum.plot(cum.index, cum.values, color=color, linewidth=lw,
                label=f"{short_names[name]}: {overall[name]['final']:+.0f}% (Sharpe {overall[name]['sharpe']:.2f})",
                alpha=1.0 if 'Aggressive' in name or 'Baseline' in name else 0.85)
ax_cum.axhline(0, color=GRID, linewidth=LW_THIN, linestyle='--')
# 直近半年を強調
recent_start = pd.Timestamp('2025-11-01')
ax_cum.axvspan(recent_start, cum.index[-1], color=ACCENT2, alpha=0.10)
ax_cum.text(recent_start, ax_cum.get_ylim()[1]*0.92, ' OOS発見期間\n (Sharpe 6.83だった範囲)',
            fontsize=7.5, color=ACCENT2)
ax_cum.set_ylabel('累積リターン (%)', fontsize=8, color=TEXT_DIM)
ax_cum.set_title('5年累積リターン (直近半年「だけ」強い)',
                  fontsize=9, color=TEXT, pad=4, loc='left', fontweight='bold')
ax_cum.legend(fontsize=8, loc='upper left', framealpha=0.7, labelcolor=TEXT_DIM)

# ─── ② 年次別 Sharpe ─────────
x = np.arange(len(years))
w = 0.2
for i, name in enumerate(strat_names):
    vals = [yearly[name].get(y, 0) for y in years]
    ax_yearly.bar(x + (i-1.5)*w, vals, w, color=colors[name], alpha=0.85,
                  label=short_names[name])
ax_yearly.axhline(2.0, color=ACCENT, linewidth=LW, linestyle=':', label='昇格基準 (2.0)')
ax_yearly.axhline(0, color=GRID, linewidth=LW_THIN, linestyle='--')
ax_yearly.set_xticks(x)
ax_yearly.set_xticklabels([str(y) for y in years], fontsize=8, color=TEXT_DIM)
ax_yearly.set_ylabel('Sharpe', fontsize=8, color=TEXT_DIM)
ax_yearly.set_title('年次別 Sharpe — 2025年以降だけ機能',
                     fontsize=9, color=TEXT, pad=4, loc='left', fontweight='bold')
ax_yearly.legend(fontsize=6.5, framealpha=0.7, labelcolor=TEXT_DIM, ncol=2, loc='upper left')

# ─── ③ ローリング12M Sharpe ─────────
for name in ['Matrix (6コンポーネント)', 'Aggressive L/S (9コンポーネント)']:
    s = rolling[name]
    ax_roll.plot(s.index, s.values, color=colors[name], linewidth=LW_THICK,
                 label=short_names[name], alpha=0.9)
ax_roll.axhline(2.0, color=ACCENT, linewidth=LW, linestyle=':', label='昇格基準')
ax_roll.axhline(0, color=GRID, linewidth=LW_THIN, linestyle='--')
ax_roll.set_ylabel('Sharpe', fontsize=8, color=TEXT_DIM)
ax_roll.set_title('12ヶ月ローリング Sharpe — 大半の期間で基準未達',
                   fontsize=9, color=TEXT, pad=4, loc='left', fontweight='bold')
ax_roll.legend(fontsize=8, framealpha=0.7, labelcolor=TEXT_DIM)

add_footer(fig, source='JQuants 日足 5年',
           period=f"{R['START']} 〜 {R['END']}")

out = os.path.join(os.path.dirname(__file__), 'result.png')
save(fig, out)
