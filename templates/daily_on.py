"""
日次・ON傾向テンプレ
使い方:
    from templates.daily_on import plot_daily_on
    plot_daily_on(
        title="TOPIX ON ギャップ分布",
        subtitle="前日終値→翌寄付 / 2025-2026",
        gaps=series_of_gap_pct,        # pd.Series (%) 日次ギャップ
        by_dow=dict_of_series,         # {"月":series, "火":series, ...} (省略可)
        fill_rate=0.63,                # ギャップフィル率 (省略可)
        path="result.png",
    )
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from ._style import *


def plot_daily_on(
    title: str,
    subtitle: str,
    gaps: pd.Series,
    by_dow: dict | None = None,
    fill_rate: float | None = None,
    path: str = "result.png",
    period: str = "2025-01〜2026-05",
):
    apply_base_style()
    fig = plt.figure(figsize=(12, 6.75), facecolor=BG)

    fig.text(0.5, 0.97, title, ha="center", va="top",
             fontsize=15, fontweight="bold", color=TEXT)
    fig.text(0.5, 0.925, subtitle, ha="center", va="top",
             fontsize=8.5, color=TEXT_DIM)

    g = gaps.dropna().values
    pos_rate = (g > 0).mean() * 100
    mean_gap = g.mean()
    median_gap = np.median(g)

    kpis = [
        {"label": "GU率", "value": f"{pos_rate:.0f}%",
         "color": ACCENT2 if pos_rate > 50 else ACCENT3},
        {"label": "平均ギャップ", "value": f"{mean_gap:+.2f}%",
         "color": ACCENT2 if mean_gap > 0 else ACCENT3},
        {"label": "中央値", "value": f"{median_gap:+.2f}%", "color": ACCENT},
        {"label": "N", "value": str(len(g)), "color": TEXT_DIM},
    ]
    if fill_rate is not None:
        kpis.append({"label": "フィル率", "value": f"{fill_rate*100:.0f}%",
                     "color": ACCENT4})
    add_kpi_row(fig, kpis, y=0.875)

    has_dow = by_dow is not None and len(by_dow) > 0

    gs = gridspec.GridSpec(
        1, 2 if has_dow else 1, figure=fig,
        left=0.07, right=0.96, bottom=0.1, top=0.77,
        wspace=0.12,
        width_ratios=[2, 1] if has_dow else [1],
    )

    # --- ギャップ分布ヒストグラム ---
    ax1 = fig.add_subplot(gs[0])
    bins = np.linspace(np.percentile(g, 1), np.percentile(g, 99), 40)
    ax1.hist(g[g >= 0], bins=bins, color=ACCENT2, alpha=0.85, lw=0)
    ax1.hist(g[g < 0], bins=bins, color=ACCENT3, alpha=0.85, lw=0)
    ax1.axvline(0, color=GRID, lw=LW_THIN)
    ax1.axvline(mean_gap, color=ACCENT4, lw=LW, linestyle="--")
    ax1.text(mean_gap, ax1.get_ylim()[1] * 0.9,
             f" 平均{mean_gap:+.2f}%", ha="left" if mean_gap >= 0 else "right",
             fontsize=7.5, color=ACCENT4)
    ax1.set_xlabel("ギャップ (%)", fontsize=8, color=TEXT_DIM)
    ax1.set_ylabel("頻度", fontsize=8, color=TEXT_DIM)
    ax1.grid(axis="y", zorder=0)

    # --- 曜日別箱ひげ ---
    if has_dow:
        ax2 = fig.add_subplot(gs[1])
        labels = list(by_dow.keys())
        data = [by_dow[k].dropna().values for k in labels]
        bp = ax2.boxplot(
            data,
            labels=labels,
            patch_artist=True,
            medianprops={"color": ACCENT4, "linewidth": LW},
            whiskerprops={"color": TEXT_DIM, "linewidth": LW_THIN},
            capprops={"color": TEXT_DIM, "linewidth": LW_THIN},
            flierprops={"marker": ".", "markersize": 2,
                        "markerfacecolor": TEXT_DIM, "linestyle": "none"},
            boxprops={"linewidth": LW_THIN},
        )
        for patch, k in zip(bp["boxes"], labels):
            med = np.median(by_dow[k].dropna())
            patch.set_facecolor(ACCENT2 if med >= 0 else ACCENT3)
            patch.set_alpha(0.4)
        ax2.axhline(0, color=GRID, lw=LW_THIN)
        ax2.set_ylabel("ギャップ (%)", fontsize=8, color=TEXT_DIM)
        ax2.grid(axis="y", zorder=0)

    add_footer(fig, period=period)
    save(fig, path)
