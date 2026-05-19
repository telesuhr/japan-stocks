"""
曜日別ON × セッション分解 → result.png
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
result   = R['result']
rank_fri = R['rank_fri']
SEMI, NONFER, ALL = R['SEMI'], R['NONFER'], R['ALL']

DOW_LABELS = {0:'月', 1:'火', 2:'水', 3:'木', 4:'金'}

# 集計テーブル
def build_table():
    rows = []
    for dow in range(5):
        for sec_code in ['semi', 'nonfer']:
            sub = result[(result['dow'] == dow) & (result['sector'] == sec_code)]
            if len(sub) == 0: continue
            daily_pnl = sub.groupby('date')['on_gap'].mean() - 0.04
            cum = ((1 + daily_pnl/100).cumprod().iloc[-1] - 1) * 100
            rows.append(dict(
                dow=dow, sector=sec_code,
                on=sub['on_gap'].mean(),
                sess=sub['day_open_close'].mean(),
                full=sub['full_day'].mean(),
                wr=(sub['on_gap']>0).mean()*100,
                cum=cum,
            ))
    return pd.DataFrame(rows)

tbl = build_table()

apply_base_style()
fig = plt.figure(figsize=(12, 6.75), facecolor=BG)

fig.text(0.5, 0.97, '曜日別ON ＆ 日中セッション分解（半導体AI × 非鉄金属）',
         ha='center', va='top', fontsize=14, fontweight='bold', color=TEXT)
fig.text(0.5, 0.935, '定義: 月曜ON=金曜引け→月曜寄付 / 金曜ON=木曜引け→金曜寄付 / 直近半年 (2025-11〜2026-05)',
         ha='center', va='top', fontsize=8.5, color=TEXT_DIM)

# KPI
add_kpi_row(fig, [
    {"label": "ベスト: 水曜ON非鉄",  "value": "+28.9%", "color": ACCENT2},
    {"label": "木曜ON 半導体",        "value": "+22.1%", "color": ACCENT2},
    {"label": "ワースト: 月曜ON半導体","value": "-21.7%", "color": ACCENT3},
    {"label": "金曜ON 半導体",        "value": "-18.6%", "color": ACCENT3},
    {"label": "アドバンテスト金曜ON", "value": "-1.30%/日", "color": ACCENT3},
], y=0.855)

gs = gridspec.GridSpec(2, 3, figure=fig,
                       left=0.06, right=0.97,
                       top=0.74, bottom=0.10,
                       hspace=0.65, wspace=0.42)

ax_heat  = fig.add_subplot(gs[0, :2])
ax_cum   = fig.add_subplot(gs[0, 2])
ax_decomp= fig.add_subplot(gs[1, :2])
ax_fri   = fig.add_subplot(gs[1, 2])

for ax in [ax_heat, ax_cum, ax_decomp, ax_fri]:
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.tick_params(colors=TEXT_DIM, labelsize=8)
    ax.grid(axis='y', zorder=0)

# ─── ① 曜日 × セクター ON heatmap ──────────
sec_order = ['semi','nonfer']
sec_label = ['半導体AI','非鉄金属']
mat = np.zeros((2, 5))
for i, sec in enumerate(sec_order):
    for d in range(5):
        v = tbl[(tbl['dow']==d) & (tbl['sector']==sec)]['on'].values
        mat[i, d] = v[0] if len(v) else np.nan

vmax = np.nanmax(np.abs(mat))
im = ax_heat.imshow(mat, cmap='RdYlGn', vmin=-vmax, vmax=vmax, aspect='auto')
ax_heat.set_xticks(range(5))
ax_heat.set_xticklabels([f'{DOW_LABELS[d]}曜ON' for d in range(5)], fontsize=9, color=TEXT_DIM)
ax_heat.set_yticks(range(2))
ax_heat.set_yticklabels(sec_label, fontsize=9, color=TEXT_DIM)
ax_heat.set_title('曜日 × セクター ON ギャップ平均 (%)',
                   fontsize=9, color=TEXT, pad=4, loc='left', fontweight='bold')
for i in range(2):
    for j in range(5):
        ax_heat.text(j, i, f'{mat[i,j]:+.2f}%', ha='center', va='center',
                     fontsize=10, color='black' if abs(mat[i,j])<0.5 else 'white',
                     fontweight='bold')
ax_heat.grid(False)

# ─── ② 各曜日ON累積（コスト後） ──────────
x = np.arange(5)
w = 0.38
semi_cum  = [tbl[(tbl['dow']==d) & (tbl['sector']=='semi')]['cum'].values[0]   for d in range(5)]
nonf_cum  = [tbl[(tbl['dow']==d) & (tbl['sector']=='nonfer')]['cum'].values[0] for d in range(5)]
ax_cum.bar(x - w/2, semi_cum, w,
           color=[ACCENT2 if v >= 0 else ACCENT3 for v in semi_cum],
           alpha=0.85, label='半導体AI')
ax_cum.bar(x + w/2, nonf_cum, w,
           color=[ACCENT2 if v >= 0 else ACCENT3 for v in nonf_cum],
           alpha=0.55, label='非鉄金属')
ax_cum.axhline(0, color=GRID, linewidth=LW_THIN, linestyle='--')
ax_cum.set_xticks(x)
ax_cum.set_xticklabels([f'{DOW_LABELS[d]}ON' for d in range(5)], fontsize=8, color=TEXT_DIM)
ax_cum.set_ylabel('累積%', fontsize=8, color=TEXT_DIM)
ax_cum.set_title('各曜日ON のみ保有 累積 (コスト後)',
                  fontsize=9, color=TEXT, pad=4, loc='left', fontweight='bold')
ax_cum.legend(fontsize=7, framealpha=0.7, labelcolor=TEXT_DIM)
for xi, v in zip(x - w/2, semi_cum):
    ax_cum.text(xi, v + (1 if v >= 0 else -3),
                f'{v:+.0f}', ha='center',
                va='bottom' if v >= 0 else 'top',
                fontsize=7, color=TEXT, fontweight='bold')
for xi, v in zip(x + w/2, nonf_cum):
    ax_cum.text(xi, v + (1 if v >= 0 else -3),
                f'{v:+.0f}', ha='center',
                va='bottom' if v >= 0 else 'top',
                fontsize=7, color=TEXT_DIM, fontweight='bold')

# ─── ③ ON vs 寄→引セッション 分解バー ────
semi_on   = [tbl[(tbl['dow']==d) & (tbl['sector']=='semi')]['on'].values[0]   for d in range(5)]
nonf_on   = [tbl[(tbl['dow']==d) & (tbl['sector']=='nonfer')]['on'].values[0] for d in range(5)]
semi_sess = [tbl[(tbl['dow']==d) & (tbl['sector']=='semi')]['sess'].values[0]   for d in range(5)]
nonf_sess = [tbl[(tbl['dow']==d) & (tbl['sector']=='nonfer')]['sess'].values[0] for d in range(5)]

x = np.arange(5)
w = 0.18
ax_decomp.bar(x - 1.5*w, semi_on,   w, color=ACCENT,  alpha=0.85, label='ON (半導体)')
ax_decomp.bar(x - 0.5*w, semi_sess, w, color=ACCENT,  alpha=0.45, label='寄→引 (半導体)')
ax_decomp.bar(x + 0.5*w, nonf_on,   w, color=ACCENT4, alpha=0.85, label='ON (非鉄)')
ax_decomp.bar(x + 1.5*w, nonf_sess, w, color=ACCENT4, alpha=0.45, label='寄→引 (非鉄)')
ax_decomp.axhline(0, color=GRID, linewidth=LW_THIN, linestyle='--')
ax_decomp.set_xticks(x)
ax_decomp.set_xticklabels([f'{DOW_LABELS[d]}曜' for d in range(5)], fontsize=9, color=TEXT_DIM)
ax_decomp.set_ylabel('%', fontsize=8, color=TEXT_DIM)
ax_decomp.set_title('1日リターン分解: ON gap (前日引け→当日寄付) vs 寄→引',
                     fontsize=9, color=TEXT, pad=4, loc='left', fontweight='bold')
ax_decomp.legend(fontsize=7, framealpha=0.7, labelcolor=TEXT_DIM, ncol=2, loc='upper left')

# ─── ④ 金曜ON銘柄ランキング ──────────
codes = rank_fri.index.tolist()
labels = [ALL[c][:7] for c in codes]
vals = rank_fri['on_mean'].values
colors = [ACCENT2 if v >= 0 else ACCENT3 for v in vals]

ax_fri.barh(range(len(labels)), vals, color=colors, alpha=0.85)
ax_fri.set_yticks(range(len(labels)))
ax_fri.set_yticklabels(labels, fontsize=6.5, color=TEXT_DIM)
ax_fri.invert_yaxis()
ax_fri.axvline(0, color=GRID, linewidth=LW_THIN, linestyle='--')
ax_fri.set_xlabel('%', fontsize=8, color=TEXT_DIM)
ax_fri.set_title('金曜ON 銘柄別 (木曜引け→金曜寄付)',
                  fontsize=9, color=TEXT, pad=4, loc='left', fontweight='bold')
for i, v in enumerate(vals):
    ax_fri.text(v + (0.04 if v >= 0 else -0.04), i,
                f'{v:+.2f}', va='center',
                ha='left' if v >= 0 else 'right',
                fontsize=6.5, color=TEXT)

add_footer(fig, source='JQuants 日足',
           period=f"{R['START']} 〜 {R['END']}")

out = os.path.join(os.path.dirname(__file__), 'result.png')
save(fig, out)
