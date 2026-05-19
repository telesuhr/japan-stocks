"""マトリクス戦略バックテスト → result.png"""
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
all_p, is_p, oos_p = R['all_perf'], R['is_perf'], R['oos_perf']
strats = R['strat_daily_returns']
IS_END, OOS_START = R['IS_END'], R['OOS_START']

strat_names = list(strats.keys())
short_names = {
    'Baseline (16銘柄Buy&Hold)': 'Baseline',
    'Conservative (火水木ON非鉄 + 金イントラ半導体)': 'Conservative',
    'Matrix (各曜日最強アクション)': 'Matrix',
    'Aggressive (L/S 完全版)': 'Aggressive',
}
colors = {
    'Baseline (16銘柄Buy&Hold)':                      TEXT_DIM,
    'Conservative (火水木ON非鉄 + 金イントラ半導体)':      ACCENT,
    'Matrix (各曜日最強アクション)':                     ACCENT2,
    'Aggressive (L/S 完全版)':                        ACCENT4,
}

apply_base_style()
fig = plt.figure(figsize=(12, 6.75), facecolor=BG)

fig.text(0.5, 0.97, 'マトリクス戦略バックテスト（曜日 × セクター × セッション）',
         ha='center', va='top', fontsize=14, fontweight='bold', color=TEXT)
fig.text(0.5, 0.935, '2024-05〜2026-05 / コスト4bps/取引 / IS-OOS分割検証',
         ha='center', va='top', fontsize=8.5, color=TEXT_DIM)

best_oos = max(oos_p.items(), key=lambda x: x[1]['sharpe'])
add_kpi_row(fig, [
    {"label": "OOS Sharpe (Aggressive)",  "value": f"{oos_p['Aggressive (L/S 完全版)']['sharpe']:.2f}", "color": ACCENT2},
    {"label": "OOS MDD",                  "value": f"{oos_p['Aggressive (L/S 完全版)']['mdd']:.1f}%",   "color": ACCENT2},
    {"label": "全期間 累積",              "value": f"{all_p['Aggressive (L/S 完全版)']['final']:+.0f}%", "color": ACCENT4},
    {"label": "Baseline MDD",             "value": f"{all_p['Baseline (16銘柄Buy&Hold)']['mdd']:.0f}%", "color": ACCENT3},
    {"label": "Sharpe改善倍率",            "value": f"×{oos_p['Aggressive (L/S 完全版)']['sharpe']/oos_p['Baseline (16銘柄Buy&Hold)']['sharpe']:.1f}", "color": ACCENT2},
], y=0.855)

gs = gridspec.GridSpec(2, 3, figure=fig,
                       left=0.06, right=0.97,
                       top=0.74, bottom=0.10,
                       hspace=0.60, wspace=0.42)

ax_cum    = fig.add_subplot(gs[0, :])     # 上全幅: 累積カーブ
ax_sh     = fig.add_subplot(gs[1, 0])     # 下左: Sharpe比較
ax_mdd    = fig.add_subplot(gs[1, 1])     # 下中: MDD比較
ax_ret    = fig.add_subplot(gs[1, 2])     # 下右: 累積リターン比較

for ax in [ax_cum, ax_sh, ax_mdd, ax_ret]:
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.tick_params(colors=TEXT_DIM, labelsize=8)
    ax.grid(axis='y', zorder=0)

# ─── ① 累積カーブ ────────────────────
for name in strat_names:
    cum = all_p[name]['cum_series']
    color = colors[name]
    lw = LW_THICK + 0.5 if 'Aggressive' in name else LW
    ax_cum.plot(cum.index, cum.values, color=color, linewidth=lw,
                label=f"{short_names[name]}: {all_p[name]['final']:+.0f}% (Sharpe {all_p[name]['sharpe']:.2f})",
                alpha=1.0 if 'Aggressive' in name or 'Baseline' in name else 0.85)
ax_cum.axhline(0, color=GRID, linewidth=LW_THIN, linestyle='--')
ax_cum.axvline(IS_END, color=GRID, linewidth=LW, linestyle=':', alpha=0.8)
ax_cum.text(IS_END, ax_cum.get_ylim()[1]*0.85, ' IS終 / OOS開始',
            fontsize=7, color=TEXT_DIM)
ax_cum.set_ylabel('累積リターン (%)', fontsize=8, color=TEXT_DIM)
ax_cum.set_title('全期間 累積リターン推移 (IS+OOS)',
                  fontsize=9, color=TEXT, pad=4, loc='left', fontweight='bold')
ax_cum.legend(fontsize=8, loc='upper left', framealpha=0.7, labelcolor=TEXT_DIM)

# ─── ② Sharpe比較 (IS vs OOS) ──────
x = np.arange(len(strat_names))
w = 0.4
is_sh  = [is_p[n]['sharpe'] for n in strat_names]
oos_sh = [oos_p[n]['sharpe'] for n in strat_names]
ax_sh.bar(x - w/2, is_sh,  w, color=TEXT_DIM, alpha=0.65, label='IS')
ax_sh.bar(x + w/2, oos_sh, w, color=[colors[n] for n in strat_names], alpha=0.9, label='OOS')
ax_sh.set_xticks(x)
ax_sh.set_xticklabels([short_names[n] for n in strat_names],
                       fontsize=7, color=TEXT_DIM, rotation=15, ha='right')
ax_sh.axhline(0, color=GRID, linewidth=LW_THIN, linestyle='--')
ax_sh.set_ylabel('Sharpe', fontsize=8, color=TEXT_DIM)
ax_sh.set_title('Sharpe (IS vs OOS)',
                 fontsize=9, color=TEXT, pad=4, loc='left', fontweight='bold')
ax_sh.legend(fontsize=7, framealpha=0.7, labelcolor=TEXT_DIM)
for xi, v in zip(x + w/2, oos_sh):
    ax_sh.text(xi, v + 0.2, f'{v:.1f}', ha='center', fontsize=7, color=TEXT, fontweight='bold')

# ─── ③ MDD比較 ──────────────────
all_mdd = [all_p[n]['mdd'] for n in strat_names]
oos_mdd = [oos_p[n]['mdd'] for n in strat_names]
ax_mdd.bar(x - w/2, all_mdd, w, color=TEXT_DIM, alpha=0.65, label='全期間')
ax_mdd.bar(x + w/2, oos_mdd, w, color=[colors[n] for n in strat_names], alpha=0.9, label='OOS')
ax_mdd.set_xticks(x)
ax_mdd.set_xticklabels([short_names[n] for n in strat_names],
                        fontsize=7, color=TEXT_DIM, rotation=15, ha='right')
ax_mdd.set_ylabel('%', fontsize=8, color=TEXT_DIM)
ax_mdd.set_title('最大ドローダウン',
                  fontsize=9, color=TEXT, pad=4, loc='left', fontweight='bold')
ax_mdd.legend(fontsize=7, framealpha=0.7, labelcolor=TEXT_DIM)
for xi, v in zip(x - w/2, all_mdd):
    ax_mdd.text(xi, v - 1.5, f'{v:.0f}', ha='center', fontsize=7, color=TEXT_DIM, va='top')
for xi, v in zip(x + w/2, oos_mdd):
    ax_mdd.text(xi, v - 0.5, f'{v:.1f}', ha='center', fontsize=7, color=TEXT, fontweight='bold', va='top')

# ─── ④ OOS累積リターン ──────────
oos_finals = [oos_p[n]['final'] for n in strat_names]
ax_ret.bar(x, oos_finals,
           color=[colors[n] for n in strat_names], alpha=0.9, width=0.6)
ax_ret.set_xticks(x)
ax_ret.set_xticklabels([short_names[n] for n in strat_names],
                        fontsize=7, color=TEXT_DIM, rotation=15, ha='right')
ax_ret.axhline(0, color=GRID, linewidth=LW_THIN, linestyle='--')
ax_ret.set_ylabel('%', fontsize=8, color=TEXT_DIM)
ax_ret.set_title('OOS 累積リターン',
                  fontsize=9, color=TEXT, pad=4, loc='left', fontweight='bold')
for xi, v in zip(x, oos_finals):
    ax_ret.text(xi, v + 1.5, f'{v:+.0f}%', ha='center',
                 fontsize=8, color=TEXT, fontweight='bold')

add_footer(fig, source='JQuants 日足',
           period=f"{R['START']} 〜 {R['END']}")

out = os.path.join(os.path.dirname(__file__), 'result.png')
save(fig, out)
