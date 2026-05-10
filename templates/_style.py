"""共通スタイル定義"""

BG = "#0a0a0f"
PANEL = "#12121a"
GRID = "#1e1e2e"
TEXT = "#e0e0e0"
TEXT_DIM = "#606070"
ACCENT = "#4fc3f7"       # ライトブルー（メイン）
ACCENT2 = "#81c784"      # グリーン（正）
ACCENT3 = "#e57373"      # レッド（負）
ACCENT4 = "#ffb74d"      # オレンジ（中立・注目）

LW = 0.8                 # 標準線幅
LW_THIN = 0.5
LW_THICK = 1.2

FONT_MAIN = ["Hiragino Sans", "BIZ UDGothic", "Apple SD Gothic Neo", "sans-serif"]

import matplotlib.pyplot as plt
import matplotlib as mpl

def apply_base_style():
    mpl.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor": PANEL,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_DIM,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": LW_THIN,
        "xtick.color": TEXT_DIM,
        "ytick.color": TEXT_DIM,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.major.width": LW_THIN,
        "ytick.major.width": LW_THIN,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "grid.color": GRID,
        "grid.linewidth": LW_THIN,
        "grid.linestyle": "-",
        "legend.facecolor": PANEL,
        "legend.edgecolor": GRID,
        "legend.labelcolor": TEXT_DIM,
        "legend.fontsize": 8,
        "font.family": FONT_MAIN,
        "axes.unicode_minus": False,
        "text.color": TEXT,
    })

def new_fig(title=None, subtitle=None):
    apply_base_style()
    fig = plt.figure(figsize=(12, 6.75), facecolor=BG)
    if title:
        fig.text(0.5, 0.95, title, ha="center", va="top",
                 fontsize=16, fontweight="bold", color=TEXT)
    if subtitle:
        fig.text(0.5, 0.91, subtitle, ha="center", va="top",
                 fontsize=9, color=TEXT_DIM)
    return fig

def add_footer(fig, source="日本株1分足 (Refinitiv)", period="2025-01〜2026-05"):
    fig.text(0.99, 0.012, f"{period}  |  {source}",
             ha="right", va="bottom", fontsize=7, color=TEXT_DIM)

def add_kpi_row(fig, kpis: list[dict], y=0.84):
    """kpis = [{"label": "Sharpe", "value": "3.2", "color": ACCENT2}, ...]"""
    n = len(kpis)
    xs = [0.1 + i * (0.8 / (n - 1)) for i in range(n)] if n > 1 else [0.5]
    for x, kpi in zip(xs, kpis):
        color = kpi.get("color", ACCENT)
        fig.text(x, y, kpi["value"], ha="center", va="bottom",
                 fontsize=22, fontweight="bold", color=color)
        fig.text(x, y - 0.04, kpi["label"], ha="center", va="bottom",
                 fontsize=8, color=TEXT_DIM)

def save(fig, path="result.png"):
    plt.savefig(path, dpi=100, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)
    print(f"saved: {path}")
