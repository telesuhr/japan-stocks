"""優良(高ROE)の検証結果の図。左=ROE分位が逆行 / 右=優良と高配当の年次ミラー。"""
import sys
sys.stdout.reconfigure(line_buffering=True)
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
try:
    import matplotlib.font_manager as fm
    fm.fontManager.addfont("/root/.fonts/NotoSansJP.ttf")
    plt.rcParams["font.family"] = fm.FontProperties(fname="/root/.fonts/NotoSansJP.ttf").get_name()
    plt.rcParams["axes.unicode_minus"] = False
except Exception:
    pass

Q = pd.read_csv(HERE / "roe_quintiles.csv")
M = pd.read_csv(HERE / "quality_vs_yield_monthly.csv", index_col=0, parse_dates=True)
yr = M.groupby(M.index.year).apply(lambda x: ((1 + x).prod() - 1) * 100)

fig, axes = plt.subplots(1, 2, figsize=(12, 6.75), facecolor="white")

ax = axes[0]
col = ["#0969da"] + ["#8c959f"] * 3 + ["#cf222e", "#bf8700"]
b = ax.bar(range(len(Q)), Q["年率"], color=col[:len(Q)])
ax.set_xticks(range(len(Q)))
ax.set_xticklabels(["Q1\n低ROE", "Q2", "Q3", "Q4", "Q5\n高ROE", "市場\nEW"], fontsize=9)
for i, v in enumerate(Q["年率"]):
    ax.text(i, v + 0.2, f"{v:.1f}%", ha="center", fontsize=9,
            fontweight="bold" if i in (0, 4) else "normal")
ax.set_ylabel("年率リターン %（配当込み・コスト込み）")
ax.set_ylim(0, 12)
ax.set_title("「優良＝高ROE」は効かない。むしろ逆行\n高ROE(7.1%)は低ROE(10.0%)にも市場(9.8%)にも負ける", fontsize=11)
ax.grid(alpha=0.3, axis="y")

ax = axes[1]
x = np.arange(len(yr))
ax.bar(x - 0.22, yr["優良30"], 0.44, label="優良30（高ROE・財務健全）", color="#8250df")
ax.bar(x + 0.22, yr["高配当30"], 0.44, label="高配当30", color="#0969da")
ax.axhline(0, color="black", lw=0.9)
for y in [2020, 2022]:
    if y in yr.index:
        ax.annotate("", xy=(list(yr.index).index(y), yr.loc[y].max() + 3), xytext=(list(yr.index).index(y), yr.loc[y].max() + 12),
                    arrowprops=dict(arrowstyle="->", color="#cf222e", lw=1.4))
ax.text(-0.4, 57, "2020: 高配当−9.4% / 優良+29.6%\n2022: 高配当+18.4% / 優良−18.7%",
        fontsize=8.5, color="#cf222e", va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cf222e", lw=0.8))
ax.set_xticks(x)
ax.set_xticklabels(yr.index, rotation=45, fontsize=8)
ax.set_ylabel("年次リターン %")
ax.set_ylim(-35, 66)
ax.set_title("両者は同じ年に沈まない（相関0.52・銘柄重複8%）\n優良: IS15.8%→OOS1.7% / 高配当: IS2.6%→OOS21.0% の鏡像", fontsize=11)
ax.legend(fontsize=9, loc="upper right")
ax.grid(alpha=0.3, axis="y")

fig.suptitle("「優良銘柄を買っておけば安心」は日本株では成立しない", fontsize=14)
fig.text(0.99, 0.005, "データ: JQuants stocks_daily + fin_summary 2016-2026 / PIT・生存者バイアス排除 / "
                      "配当込み・コスト込み / 年1回入替", ha="right", va="bottom", fontsize=8, color="gray")
fig.savefig(HERE / "quality.png", dpi=100, bbox_inches="tight", facecolor="white")
print("saved quality.png")
