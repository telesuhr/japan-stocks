"""
最適ポートフォリオ構築バックテスト → result.png
"""
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

results     = R['results']
is_results  = R['is_results']
ALL         = R['ALL']
OOS_START, OOS_END = R['OOS_START'], R['OOS_END']
IS_START,  IS_END  = R['IS_START'],  R['IS_END']
TOP3, TOP5  = R['TOP3'], R['TOP5']

# 戦略コードと色
strat_keys = list(results.keys())
strat_colors = {
    strat_keys[0]: TEXT_DIM,     # A
    strat_keys[1]: '#ce93d8',    # B (紫)
    strat_keys[2]: ACCENT,       # C (青)
    strat_keys[3]: ACCENT2,      # D (緑)
    strat_keys[4]: ACCENT4,      # E (オレンジ)
}

apply_base_style()
fig = plt.figure(figsize=(12, 6.75), facecolor=BG)

# タイトル
fig.text(0.5, 0.97, '半導体AI×非鉄金属　最適ポートフォリオ構築',
         ha='center', va='top', fontsize=15, fontweight='bold', color=TEXT)
fig.text(0.5, 0.935, 'IS: 2024-05〜2025-10で銘柄選定 / OOS: 2025-11〜2026-05で戦略適用 / コスト4bps/トレード',
         ha='center', va='top', fontsize=8.5, color=TEXT_DIM)

# KPI（ベスト戦略の数値）
best = strat_keys[4]  # E
add_kpi_row(fig, [
    {"label": "ベスト戦略",         "value": "E", "color": ACCENT4},
    {"label": "OOS累積",             "value": f"{results[best]['final_ret']:+.0f}%", "color": ACCENT2},
    {"label": "Sharpe",              "value": f"{results[best]['sharpe']:.2f}", "color": ACCENT2},
    {"label": "MDD",                 "value": f"{results[best]['mdd']:.1f}%", "color": ACCENT3},
    {"label": "Buy&Hold比",         "value": f"+{results[best]['final_ret']-results[strat_keys[0]]['final_ret']:.0f}%", "color": ACCENT4},
], y=0.855)

gs = gridspec.GridSpec(2, 3, figure=fig,
                       left=0.06, right=0.97,
                       top=0.74, bottom=0.10,
                       hspace=0.60, wspace=0.42)

ax_cum    = fig.add_subplot(gs[0, :2])   # 上左: OOS累積
ax_kpi    = fig.add_subplot(gs[0, 2])    # 上右: Sharpe比較
ax_isos   = fig.add_subplot(gs[1, :2])   # 下左: IS vs OOS の累積
ax_mdd    = fig.add_subplot(gs[1, 2])    # 下右: MDD比較

for ax in [ax_cum, ax_kpi, ax_isos, ax_mdd]:
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.grid(axis='y', zorder=0)
    ax.tick_params(colors=TEXT_DIM, labelsize=8)

# ─── ① OOS 累積リターン推移 ──────────────
for name in strat_keys:
    cum = results[name]['cum']
    color = strat_colors[name]
    lw = LW_THICK + 0.3 if name == best else LW
    alpha = 1.0 if name == best else 0.7
    ax_cum.plot(cum.index, cum.values, color=color, linewidth=lw,
                label=f"{name.split(':')[0]}: {results[name]['final_ret']:+.0f}%",
                alpha=alpha, zorder=5 if name == best else 3)
ax_cum.axhline(0, color=GRID, linewidth=LW_THIN, linestyle='--')
ax_cum.set_ylabel('累積リターン (%)', fontsize=8, color=TEXT_DIM)
ax_cum.set_title('OOS期間 戦略別 累積リターン推移',
                  fontsize=9, color=TEXT, pad=4, loc='left', fontweight='bold')
ax_cum.legend(fontsize=7.5, loc='upper left', framealpha=0.7, labelcolor=TEXT_DIM)

# ─── ② Sharpe比較 ────────────────────────
labels_short = [k.split(':')[0] for k in strat_keys]
oos_sharpes  = [results[k]['sharpe'] for k in strat_keys]
is_sharpes   = [is_results[k]['sharpe'] for k in strat_keys]
x = np.arange(len(strat_keys))
w = 0.38
bars1 = ax_kpi.bar(x - w/2, is_sharpes, w, color=TEXT_DIM, alpha=0.7, label='IS')
bars2 = ax_kpi.bar(x + w/2, oos_sharpes, w,
                   color=[strat_colors[k] for k in strat_keys], alpha=0.85, label='OOS')
ax_kpi.set_xticks(x)
ax_kpi.set_xticklabels(labels_short, fontsize=8, color=TEXT_DIM)
ax_kpi.set_ylabel('Sharpe', fontsize=8, color=TEXT_DIM)
ax_kpi.set_title('Sharpe比較 (IS vs OOS)',
                  fontsize=9, color=TEXT, pad=4, loc='left', fontweight='bold')
ax_kpi.legend(fontsize=7, framealpha=0.7, labelcolor=TEXT_DIM)
ax_kpi.axhline(0, color=GRID, linewidth=LW_THIN, linestyle='--')
for b, v in zip(bars2, oos_sharpes):
    ax_kpi.text(b.get_x() + b.get_width()/2, v + 0.15,
                f'{v:.1f}', ha='center', fontsize=7, color=TEXT, fontweight='bold')

# ─── ③ IS / OOS の累積リターン比較 (棒) ──
is_finals  = [is_results[k]['final_ret'] for k in strat_keys]
oos_finals = [results[k]['final_ret'] for k in strat_keys]
bars1 = ax_isos.bar(x - w/2, is_finals, w, color=TEXT_DIM, alpha=0.7, label='IS (2024-05〜2025-10)')
bars2 = ax_isos.bar(x + w/2, oos_finals, w,
                    color=[strat_colors[k] for k in strat_keys], alpha=0.85,
                    label='OOS (2025-11〜2026-05)')
ax_isos.axhline(0, color=GRID, linewidth=LW_THIN, linestyle='--')
ax_isos.set_xticks(x)
ax_isos.set_xticklabels(labels_short, fontsize=8, color=TEXT_DIM)
ax_isos.set_ylabel('累積リターン (%)', fontsize=8, color=TEXT_DIM)
ax_isos.set_title('累積リターン (IS vs OOS, コスト4bps込み)',
                   fontsize=9, color=TEXT, pad=4, loc='left', fontweight='bold')
ax_isos.legend(fontsize=7, framealpha=0.7, labelcolor=TEXT_DIM, loc='upper left')
for b, v in zip(bars1, is_finals):
    ax_isos.text(b.get_x() + b.get_width()/2, v + (3 if v >= 0 else -8),
                 f'{v:+.0f}%', ha='center', fontsize=7, color=TEXT_DIM,
                 va='bottom' if v >= 0 else 'top')
for b, v in zip(bars2, oos_finals):
    ax_isos.text(b.get_x() + b.get_width()/2, v + (3 if v >= 0 else -8),
                 f'{v:+.0f}%', ha='center', fontsize=7.5, color=TEXT, fontweight='bold',
                 va='bottom' if v >= 0 else 'top')

# ─── ④ MDD比較 ─────────────────────────
oos_mdds = [results[k]['mdd'] for k in strat_keys]
bars = ax_mdd.bar(x, oos_mdds,
                  color=[strat_colors[k] for k in strat_keys],
                  alpha=0.85, width=0.6)
ax_mdd.set_xticks(x)
ax_mdd.set_xticklabels(labels_short, fontsize=8, color=TEXT_DIM)
ax_mdd.set_ylabel('%', fontsize=8, color=TEXT_DIM)
ax_mdd.set_title('OOS 最大ドローダウン',
                  fontsize=9, color=TEXT, pad=4, loc='left', fontweight='bold')
for b, v in zip(bars, oos_mdds):
    ax_mdd.text(b.get_x() + b.get_width()/2, v - 0.7,
                f'{v:.1f}%', ha='center', fontsize=7.5, color=TEXT,
                fontweight='bold', va='top')

# 凡例の補足（戦略略称）
legend_text = (
    "A: Buy&Hold(16銘柄)    B: 水曜のみ(16銘柄)    C: 水曜のみ(Top3)    "
    "D: 月曜回避(16銘柄)    E: 月曜回避(Top5)"
)
fig.text(0.5, 0.025, legend_text, ha='center', va='bottom',
         fontsize=7.5, color=TEXT_DIM)

# フッター
add_footer(fig,
           source='日本株日足 (JQuants)',
           period=f"IS {IS_START.date()}〜{IS_END.date()} / OOS {OOS_START.date()}〜{OOS_END.date()}")

out = os.path.join(os.path.dirname(__file__), 'result.png')
save(fig, out)
