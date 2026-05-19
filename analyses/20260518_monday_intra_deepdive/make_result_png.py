"""月曜イントラ深掘り → result.png"""
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

ps     = R['pattern_semi']
pn     = R['pattern_nonfer']
mon_on = R['mon_on']
rank   = R['rank']
vs     = R['vol_semi']
vn     = R['vol_nonfer']
ALL, SEMI, NONFER = R['ALL'], R['SEMI'], R['NONFER']

apply_base_style()
fig = plt.figure(figsize=(12, 6.75), facecolor=BG)

fig.text(0.5, 0.97, '月曜イントラ深掘り：半導体AI × 非鉄金属',
         ha='center', va='top', fontsize=14, fontweight='bold', color=TEXT)
fig.text(0.5, 0.935, '前場で沈み後場で戻すV字 ／ ON下げ後の半導体イントラがゴールデン',
         ha='center', va='top', fontsize=8.5, color=TEXT_DIM)

# KPI
add_kpi_row(fig, [
    {"label": "半導体 月曜引け",  "value": "+0.28%", "color": ACCENT2},
    {"label": "非鉄 月曜引け",    "value": "+0.04%", "color": ACCENT4},
    {"label": "ON下げ後 半導体寄→引", "value": "+0.50%/勝率59%", "color": ACCENT2},
    {"label": "ON下げ後 非鉄寄→引",  "value": "+0.69%/勝率52%", "color": ACCENT2},
    {"label": "TOP銘柄: SUMCO",     "value": "+1.01%/70%", "color": ACCENT2},
], y=0.855)

gs = gridspec.GridSpec(2, 3, figure=fig,
                       left=0.06, right=0.97,
                       top=0.74, bottom=0.10,
                       hspace=0.65, wspace=0.42)

ax_intra  = fig.add_subplot(gs[0, :2])
ax_on     = fig.add_subplot(gs[0, 2])
ax_rank   = fig.add_subplot(gs[1, :2])
ax_vol    = fig.add_subplot(gs[1, 2])

for ax in [ax_intra, ax_on, ax_rank, ax_vol]:
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.tick_params(colors=TEXT_DIM, labelsize=8)
    ax.grid(axis='y', zorder=0)

# ─── ① イントラ波形 ────────────────────
key_times = [(9*60,'9:00'),(9*60+30,'9:30'),(10*60,'10:00'),(10*60+30,'10:30'),
             (11*60,'11:00'),(11*60+30,'11:30'),(12*60+30,'12:30'),
             (13*60,'13:00'),(14*60,'14:00'),(15*60,'15:00'),(15*60+30,'15:30')]
mins  = [m for m, _ in key_times]
labs  = [l for _, l in key_times]
vs_s  = [ps.get(m, np.nan) for m in mins]
vs_n  = [pn.get(m, np.nan) for m in mins]
xi = np.arange(len(mins))

ax_intra.plot(xi, vs_s, color=ACCENT,  linewidth=LW_THICK, label='半導体AI',
              zorder=5, marker='o', markersize=4)
ax_intra.plot(xi, vs_n, color=ACCENT4, linewidth=LW_THICK, label='非鉄金属',
              zorder=5, marker='s', markersize=4)
ax_intra.axhline(0, color=GRID, linewidth=LW_THIN, linestyle='--')
ax_intra.axvline(5.5, color=GRID, linewidth=LW_THIN, linestyle=':')
ax_intra.text(5.65, min(min(vs_s), min(vs_n))*0.7, '昼休み', fontsize=7, color=TEXT_DIM)
ax_intra.set_xticks(xi)
ax_intra.set_xticklabels(labs, fontsize=7, color=TEXT_DIM, rotation=30, ha='right')
ax_intra.set_ylabel('%', fontsize=8, color=TEXT_DIM)
ax_intra.set_title('月曜イントラ 寄付比累積リターン (セクター平均, 直近半年)',
                    fontsize=9, color=TEXT, pad=4, loc='left', fontweight='bold')
ax_intra.legend(fontsize=8, framealpha=0.7, labelcolor=TEXT_DIM, loc='lower left')

# V字ボトム注釈
idx_n_bot = int(np.nanargmin(vs_n))
ax_intra.annotate(f'非鉄底 {vs_n[idx_n_bot]:.2f}%',
                   xy=(idx_n_bot, vs_n[idx_n_bot]),
                   xytext=(idx_n_bot+1, vs_n[idx_n_bot]-0.06),
                   fontsize=7.5, color=ACCENT4,
                   arrowprops=dict(arrowstyle='->', color=ACCENT4, lw=1))

# ─── ② ON結果別の月曜寄→引 ────────
mon_on_clean = mon_on.dropna(subset=['on_gap','mon_full'])
mon_on_clean['sector_label'] = mon_on_clean['sector'].map({'semi':'半導体','nonfer':'非鉄'})

categories = ['ON下げ\n(<-0.5%)', 'ON±\nフラット', 'ON上げ\n(>+0.5%)']
data = {'半導体':[], '非鉄':[]}
for sec_code, sec_name in [('semi','半導体'),('nonfer','非鉄')]:
    s = mon_on_clean[mon_on_clean['sector']==sec_code]
    data[sec_name] = [
        s[s['on_gap']<-0.5]['mon_full'].mean(),
        s[s['on_gap'].abs()<=0.5]['mon_full'].mean(),
        s[s['on_gap']>0.5]['mon_full'].mean(),
    ]

x = np.arange(3)
w = 0.35
ax_on.bar(x - w/2, data['半導体'], w,
          color=[ACCENT2 if v >= 0 else ACCENT3 for v in data['半導体']],
          alpha=0.9, label='半導体')
ax_on.bar(x + w/2, data['非鉄'], w,
          color=[ACCENT2 if v >= 0 else ACCENT3 for v in data['非鉄']],
          alpha=0.55, label='非鉄')
ax_on.axhline(0, color=GRID, linewidth=LW_THIN, linestyle='--')
ax_on.set_xticks(x)
ax_on.set_xticklabels(categories, fontsize=8, color=TEXT_DIM)
ax_on.set_ylabel('%', fontsize=8, color=TEXT_DIM)
ax_on.set_title('月曜ON結果別 寄→引リターン',
                 fontsize=9, color=TEXT, pad=4, loc='left', fontweight='bold')
ax_on.legend(fontsize=7, framealpha=0.7, labelcolor=TEXT_DIM)
for xi, v in zip(x - w/2, data['半導体']):
    ax_on.text(xi, v + (0.04 if v >= 0 else -0.10),
                f'{v:+.2f}', ha='center', fontsize=7.5,
                va='bottom' if v >= 0 else 'top', color=TEXT, fontweight='bold')
for xi, v in zip(x + w/2, data['非鉄']):
    ax_on.text(xi, v + (0.04 if v >= 0 else -0.10),
                f'{v:+.2f}', ha='center', fontsize=7.5,
                va='bottom' if v >= 0 else 'top', color=TEXT_DIM, fontweight='bold')

# ─── ③ 銘柄別 ランキング ────────────
codes_ord = rank.index.tolist()
labels = [ALL[c][:7] for c in codes_ord]
vals = rank['avg'].values
sectors = ['semi' if c in SEMI else 'nonfer' for c in codes_ord]
colors_bar = [ACCENT if s=='semi' else ACCENT4 for s in sectors]
# プラスは濃く、マイナスはACCENT3に
colors_final = [(c if v >= 0 else ACCENT3) for c, v in zip(colors_bar, vals)]

x = np.arange(len(labels))
bars = ax_rank.bar(x, vals, color=colors_final, alpha=0.85, width=0.7)
ax_rank.axhline(0, color=GRID, linewidth=LW_THIN, linestyle='--')
ax_rank.set_xticks(x)
ax_rank.set_xticklabels(labels, fontsize=7, color=TEXT_DIM, rotation=45, ha='right')
ax_rank.set_ylabel('%', fontsize=8, color=TEXT_DIM)
ax_rank.set_title('銘柄別 月曜寄→引リターン平均 (青=半導体, 橙=非鉄, 赤=マイナス)',
                   fontsize=9, color=TEXT, pad=4, loc='left', fontweight='bold')
for xi, v in zip(x, vals):
    ax_rank.text(xi, v + (0.04 if v >= 0 else -0.06),
                  f'{v:+.2f}', ha='center', fontsize=6.5,
                  va='bottom' if v >= 0 else 'top', color=TEXT, fontweight='bold')

# ─── ④ 出来高パターン ─────────
buckets = [(9*60, 9*60+30, '9:00-9:30'),
           (9*60+30, 10*60+30, '9:30-10:30'),
           (10*60+30, 11*60+30, '10:30-11:30'),
           (12*60+30, 13*60+30, '12:30-13:30'),
           (13*60+30, 14*60+30, '13:30-14:30'),
           (14*60+30, 15*60+30, '14:30-15:30')]
bucket_labels = [b[2] for b in buckets]
semi_vols   = [vs.loc[(vs.index>=s) & (vs.index<e)].sum()*100 for s,e,_ in buckets]
nonfer_vols = [vn.loc[(vn.index>=s) & (vn.index<e)].sum()*100 for s,e,_ in buckets]

x = np.arange(len(bucket_labels))
w = 0.4
ax_vol.bar(x - w/2, semi_vols,   w, color=ACCENT,  alpha=0.85, label='半導体')
ax_vol.bar(x + w/2, nonfer_vols, w, color=ACCENT4, alpha=0.85, label='非鉄')
ax_vol.set_xticks(x)
ax_vol.set_xticklabels(bucket_labels, fontsize=7, color=TEXT_DIM, rotation=45, ha='right')
ax_vol.set_ylabel('出来高 %', fontsize=8, color=TEXT_DIM)
ax_vol.set_title('月曜 時間帯別 出来高シェア',
                  fontsize=9, color=TEXT, pad=4, loc='left', fontweight='bold')
ax_vol.legend(fontsize=7, framealpha=0.7, labelcolor=TEXT_DIM)
for xi, v in zip(x - w/2, semi_vols):
    ax_vol.text(xi, v + 0.5, f'{v:.0f}%', ha='center', fontsize=6.5, color=TEXT)
for xi, v in zip(x + w/2, nonfer_vols):
    ax_vol.text(xi, v + 0.5, f'{v:.0f}%', ha='center', fontsize=6.5, color=TEXT_DIM)

add_footer(fig, source='JQuants 1分足+日足',
           period=f"{R['START']} 〜 {R['END']}")

out = os.path.join(os.path.dirname(__file__), 'result.png')
save(fig, out)
