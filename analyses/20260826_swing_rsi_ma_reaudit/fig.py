"""左=主張Sharpeの崩壊 / 右=戦略は同銘柄BHに毎年負ける。"""
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

Y = pd.read_csv(HERE / "yearly_vs_bh.csv")
fig, axes = plt.subplots(1, 2, figsize=(12, 6.75), facecolor="white")

# ---- 左: Sharpe の崩壊
ax = axes[0]
labels = ["RSI<30\n反発", "MA25/75\n順張り"]
claimed = [6.15, 2.87]
actual = [0.68, 1.43]
bh = [1.01, 1.31]
x = np.arange(2)
w = 0.26
ax.bar(x - w, claimed, w, label="当時の主張 (OOS Sharpe)", color="#cf222e")
ax.bar(x, actual, w, label="正しく測り直し", color="#0969da")
ax.bar(x + w, bh, w, label="同じ銘柄を持ち続けただけ", color="#8c959f")
for i in range(2):
    ax.text(i - w, claimed[i] + 0.12, f"{claimed[i]:.2f}", ha="center", fontsize=10, fontweight="bold", color="#cf222e")
    ax.text(i, actual[i] + 0.12, f"{actual[i]:.2f}", ha="center", fontsize=10, fontweight="bold", color="#0969da")
    ax.text(i + w, bh[i] + 0.12, f"{bh[i]:.2f}", ha="center", fontsize=9, color="#57606a")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("Sharpe (OOS 2021-01〜2026-08)")
ax.set_ylim(0, 7.1)
ax.axhline(2.0, color="#bf8700", ls="--", lw=1.2)
ax.text(1.42, 2.12, "昇格基準 2.0", fontsize=8.5, color="#bf8700", ha="right")
ax.set_title("「OOS Sharpe +6.15」は測定の錯覚だった\n決済日だけを並べて√245を掛けていた", fontsize=11)
ax.legend(fontsize=8.5, loc="upper right")
ax.grid(alpha=0.3, axis="y")

# ---- 右: 年次で戦略 vs BH
ax = axes[1]
d = Y[Y["構成"] == "MA25/75順張り"]
x = np.arange(len(d))
ax.bar(x - 0.21, d["戦略"], 0.42, label="MA25/75 でシグナル売買", color="#0969da")
ax.bar(x + 0.21, d["BH"], 0.42, label="同じ8銘柄を持ち続けただけ", color="#8c959f")
ax.axhline(0, color="black", lw=0.9)
for i, (s, b) in enumerate(zip(d["戦略"], d["BH"])):
    if b - s > 3:
        ax.annotate(f"−{b-s:.0f}pt", xy=(i + 0.21, b), xytext=(i, max(s, b) + 6),
                    fontsize=8, color="#cf222e", ha="center",
                    arrowprops=dict(arrowstyle="->", color="#cf222e", lw=1.0))
ax.set_xticks(x)
ax.set_xticklabels(d["date"], fontsize=9)
ax.set_ylabel("年次リターン %")
ax.set_ylim(-25, 108)
ax.set_title("売買した方が負ける — 6年中5年でBHに劣後\n年率 36.2% vs 44.8%（−8.6pt/年）", fontsize=11)
ax.legend(fontsize=9, loc="upper left")
ax.grid(alpha=0.3, axis="y")

fig.suptitle("放置されていたスイング候補2本は、どちらも「持っていただけ」に負ける", fontsize=14)
fig.text(0.99, 0.005, "データ: JQuants stocks_daily 2016-2026 / 21銘柄 / IS(2016-20)選別→OOS(2021-)評価 / "
                      "日次PF系列・√252 / 往復16bps / 分割はadj_factorから自前復元",
         ha="right", va="bottom", fontsize=7.5, color="gray")
fig.savefig(HERE / "result.png", dpi=100, bbox_inches="tight", facecolor="white")
print("saved result.png")
