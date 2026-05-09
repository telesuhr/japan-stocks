"""
イントラデイ強弱テンプレ
使い方:
    from templates.intraday import plot_intraday
    plot_intraday(
        title="半導体セクター 時間帯別強弱",
        subtitle="前場9:00-11:30 / 2025-2026",
        by_time=df_pivot,   # DataFrame: index=time(str), columns=銘柄/カテゴリ, values=平均リターン(%)
        vol_by_time=ser,    # pd.Series: index=time, values=出来高（省略可）
        path="result.png",
    )

    # または単一銘柄のイントラデイパターン
    plot_intraday(
        title="レーザーテック 時間帯別平均リターン",
        subtitle="9:00-15:30 1分足 / 2025-2026",
        by_time=df_single,  # DataFrame: index=time, columns=["mean", "std"]
        path="result.png",
    )
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from ._style import *


def plot_intraday(
    title: str,
    subtitle: str,
    by_time: pd.DataFrame,
    vol_by_time: pd.Series | None = None,
    path: str = "result.png",
    period: str = "2025-01〜2026-05",
):
    apply_base_style()
    fig = plt.figure(figsize=(12, 6.75), facecolor=BG)

    fig.text(0.5, 0.97, title, ha="center", va="top",
             fontsize=15, fontweight="bold", color=TEXT)
    fig.text(0.5, 0.925, subtitle, ha="center", va="top",
             fontsize=8.5, color=TEXT_DIM)

    has_vol = vol_by_time is not None
    n_rows = 2 if has_vol else 1
    height_ratios = [3, 1] if has_vol else [1]

    gs = gridspec.GridSpec(
        n_rows, 1, figure=fig,
        left=0.08, right=0.96, bottom=0.1, top=0.84,
        hspace=0.08,
        height_ratios=height_ratios,
    )

    ax1 = fig.add_subplot(gs[0])
    x = np.arange(len(by_time))
    cols = by_time.columns.tolist()

    # 単一銘柄 (mean/std列) vs 複数銘柄
    if set(cols) >= {"mean", "std"}:
        mean = by_time["mean"].values
        std = by_time["std"].values
        ax1.fill_between(x, mean - std, mean + std,
                         color=ACCENT, alpha=0.12)
        ax1.plot(x, mean, color=ACCENT, lw=LW, zorder=3)
        ax1.axhline(0, color=GRID, lw=LW_THIN)
        # 正/負ゾーン塗り
        ax1.fill_between(x, mean, 0,
                         where=(mean >= 0), color=ACCENT2, alpha=0.2)
        ax1.fill_between(x, mean, 0,
                         where=(mean < 0), color=ACCENT3, alpha=0.2)
    else:
        palette = [ACCENT, ACCENT2, ACCENT3, ACCENT4, "#ce93d8", "#80cbc4"]
        for i, col in enumerate(cols):
            vals = by_time[col].values
            c = palette[i % len(palette)]
            ax1.plot(x, vals, color=c, lw=LW, label=col, zorder=3)
        ax1.axhline(0, color=GRID, lw=LW_THIN)
        ax1.legend(loc="upper right", fontsize=7, framealpha=0.6)

    ax1.set_ylabel("平均リターン (%)", fontsize=8, color=TEXT_DIM)
    ax1.grid(axis="y", zorder=0)

    # x軸ラベル: 30分おきに表示
    labels = by_time.index.tolist()
    step = max(1, len(labels) // 12)
    ax1.set_xticks(x[::step])
    ax1.set_xticklabels(labels[::step], fontsize=7, rotation=0)
    if has_vol:
        ax1.set_xticklabels([])

    # 出来高
    if has_vol:
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        vol = vol_by_time.reindex(by_time.index).fillna(0).values
        ax2.bar(x, vol, color=ACCENT, alpha=0.5, width=0.8, lw=0)
        ax2.set_ylabel("出来高", fontsize=7, color=TEXT_DIM)
        ax2.set_xticks(x[::step])
        ax2.set_xticklabels(labels[::step], fontsize=7)
        ax2.grid(axis="y", zorder=0)
        ax2.spines["bottom"].set_visible(False)

    add_footer(fig, period=period)
    save(fig, path)
