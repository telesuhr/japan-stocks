"""
バックテスト結果テンプレ
使い方:
    from templates.backtest import plot_backtest
    plot_backtest(
        title="ORB ブレイクアウト",
        subtitle="日本株前場 / 9:00-9:30レンジ × 前場引け",
        equity=df["cumulative_pnl"],          # pd.Series, index=日付
        trades=df["pnl"],                     # pd.Series, トレードごとPnL(%)
        sharpe=3.2, pf=1.8, win_rate=61, n=344,
        cost_bps=2,
        path="result.png",
    )
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from ._style import *


def plot_backtest(
    title: str,
    subtitle: str,
    equity: pd.Series,
    trades: pd.Series,
    sharpe: float,
    pf: float,
    win_rate: float,
    n: int,
    cost_bps: float = 2,
    path: str = "result.png",
    period: str = "2025-01〜2026-05",
):
    apply_base_style()
    fig = plt.figure(figsize=(12, 6.75), facecolor=BG)

    # タイトル
    fig.text(0.5, 0.97, title, ha="center", va="top",
             fontsize=15, fontweight="bold", color=TEXT)
    fig.text(0.5, 0.925, subtitle, ha="center", va="top",
             fontsize=8.5, color=TEXT_DIM)

    # KPIバー
    kpis = [
        {"label": "Sharpe", "value": f"{sharpe:.2f}",
         "color": ACCENT2 if sharpe >= 2.0 else ACCENT3},
        {"label": "PF", "value": f"{pf:.2f}",
         "color": ACCENT2 if pf >= 1.3 else ACCENT4},
        {"label": "勝率", "value": f"{win_rate:.0f}%",
         "color": ACCENT},
        {"label": "N", "value": f"{n}",
         "color": TEXT_DIM},
        {"label": "コスト", "value": f"{cost_bps}bps",
         "color": TEXT_DIM},
    ]
    add_kpi_row(fig, kpis, y=0.875)

    # レイアウト: 左=エクイティカーブ, 右=リターン分布
    gs = gridspec.GridSpec(
        1, 2, figure=fig,
        left=0.07, right=0.96, bottom=0.1, top=0.77,
        wspace=0.12,
        width_ratios=[3, 1],
    )

    # --- エクイティカーブ ---
    ax1 = fig.add_subplot(gs[0])
    eq = equity.values
    dates = np.arange(len(eq))
    color_eq = ACCENT2 if eq[-1] >= 0 else ACCENT3
    ax1.plot(dates, eq, color=color_eq, lw=LW, zorder=3)
    ax1.fill_between(dates, eq, 0, alpha=0.15, color=color_eq, zorder=2)
    ax1.axhline(0, color=GRID, lw=LW_THIN, zorder=1)
    ax1.set_ylabel("累積損益 (%)", fontsize=8, color=TEXT_DIM)
    ax1.grid(axis="y", zorder=0)
    ax1.set_xlim(0, len(eq) - 1)

    # 最大ドローダウン帯をハイライト
    peak = np.maximum.accumulate(eq)
    dd = eq - peak
    worst = np.argmin(dd)
    peak_idx = np.argmax(eq[:worst + 1])
    ax1.axvspan(peak_idx, worst, alpha=0.12, color=ACCENT3, zorder=1)

    # xtickをインデックスではなく月表示（equityのindexが日付なら）
    try:
        idx = equity.index
        ticks = []
        labels = []
        prev_month = None
        for i, d in enumerate(idx):
            m = (d.year, d.month)
            if m != prev_month:
                ticks.append(i)
                labels.append(d.strftime("%y/%m"))
                prev_month = m
        ax1.set_xticks(ticks[::2])
        ax1.set_xticklabels(labels[::2], fontsize=7)
    except Exception:
        pass

    # --- リターン分布 ---
    ax2 = fig.add_subplot(gs[1])
    ret = trades.dropna().values
    bins = np.linspace(ret.min(), ret.max(), 30)
    ax2.hist(ret[ret >= 0], bins=bins, color=ACCENT2, alpha=0.85, lw=0, orientation="horizontal")
    ax2.hist(ret[ret < 0], bins=bins, color=ACCENT3, alpha=0.85, lw=0, orientation="horizontal")
    ax2.axhline(0, color=GRID, lw=LW_THIN)
    ax2.set_xlabel("頻度", fontsize=8, color=TEXT_DIM)
    ax2.set_ylabel("1トレード損益 (%)", fontsize=8, color=TEXT_DIM)
    ax2.grid(axis="x", zorder=0)
    mean_val = ret.mean()
    ax2.axhline(mean_val, color=ACCENT4, lw=LW, linestyle="--")
    ax2.text(ax2.get_xlim()[1] * 0.95, mean_val,
             f"平均{mean_val:+.3f}%", ha="right", va="bottom",
             fontsize=7, color=ACCENT4)

    add_footer(fig, period=period)
    save(fig, path)
