"""result.png 生成: PEAD拡張検証の主要結果まとめ"""
import sys; sys.stdout.reconfigure(line_buffering=True)
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUT = Path(__file__).parent
import matplotlib.font_manager as fm
_font_path = "/root/.fonts/NotoSansJP.ttf"
if Path(_font_path).exists():
    fm.fontManager.addfont(_font_path)
    _fp = fm.FontProperties(fname=_font_path)
    plt.rcParams["font.family"] = _fp.get_name()
plt.rcParams["axes.unicode_minus"] = False

res = pd.read_csv(OUT / "results.csv")

# 20日保有・ALLのシャープのみ抜粋
SCENARIOS = {
    "A_baseline":      "A. ベースライン\n(全決算10分位L/S)",
    "B_car0_3pct":     "B. 極端フィルター\n(|car0|≥3%)",
    "C_sector_neutral":"C. セクター中立\n(業種内top/bottom20%)",
    "D_sector_extreme":"D. セクター中立\n+|car0|≥3%",
}
hold = 20
colors = {"IS": "#4C72B0", "OOS": "#DD8452", "ALL": "#55A868"}

fig, axes = plt.subplots(1, 2, figsize=(12, 6.75), facecolor="white")
fig.suptitle("PEAD拡張検証: セクター中立化・極端フィルターの効果\n（決算後価格反応モメンタム L/S, 20日保有, コスト20bps後）",
             fontsize=13, fontweight="bold", y=0.98)

# ─── 左パネル: 各シナリオのSharpe (IS/OOS/ALL) ───
ax = axes[0]
sub = res[res["hold"] == hold]
x = np.arange(len(SCENARIOS))
width = 0.25
for i, (split, color) in enumerate(colors.items()):
    sharpes = []
    for sc in SCENARIOS:
        row = sub[(sub["scenario"] == sc) & (sub["label"] == split)]
        sharpes.append(row["sharpe"].values[0] if len(row) > 0 else np.nan)
    bars = ax.bar(x + (i - 1) * width, sharpes, width, label=split, color=color, alpha=0.85)

ax.axhline(0, color="black", linewidth=0.8)
ax.axhline(2.0, color="red", linewidth=1.2, linestyle="--", alpha=0.7, label="昇格基準 Sharpe=2.0")
ax.set_xticks(x)
ax.set_xticklabels([SCENARIOS[sc] for sc in SCENARIOS], fontsize=8.5)
ax.set_ylabel("年率Sharpe（コスト後）")
ax.set_title("シナリオ別Sharpe比較（20日保有）", fontsize=11)
ax.legend(fontsize=9)
ax.set_ylim(-2.0, 2.5)
ax.grid(axis="y", alpha=0.3)

# ─── 右パネル: ベースライン vs セクター中立 の保有期間比較 ───
ax2 = axes[1]
holds = [5, 10, 20]
for sc, label, color, ls in [
    ("A_baseline", "A. ベースライン", "#4C72B0", "-"),
    ("C_sector_neutral", "C. セクター中立", "#C44E52", "--"),
]:
    for split, alpha in [("ALL", 1.0), ("IS", 0.5), ("OOS", 0.8)]:
        vals = []
        for h in holds:
            row = res[(res["scenario"] == sc) & (res["hold"] == h) & (res["label"] == split)]
            vals.append(row["sharpe"].values[0] if len(row) > 0 else np.nan)
        lw = 2.0 if split == "ALL" else 1.0
        ax2.plot(holds, vals, marker="o", linewidth=lw, linestyle=ls, color=color, alpha=alpha,
                 label=f"{label} ({split})")

ax2.axhline(0, color="black", linewidth=0.8)
ax2.axhline(2.0, color="red", linewidth=1.2, linestyle=":", alpha=0.7, label="昇格基準")
ax2.set_xlabel("保有期間（営業日）")
ax2.set_ylabel("年率Sharpe（コスト後）")
ax2.set_title("ベースライン vs セクター中立: 保有期間別Sharpe", fontsize=11)
ax2.set_xticks(holds)
ax2.legend(fontsize=7.5, loc="lower right")
ax2.set_ylim(-2.0, 2.0)
ax2.grid(alpha=0.3)

# 注釈
fig.text(0.01, 0.01,
    "データ: 2021-2026 / fin_summary決算イベント11,305件 / 流動性≥10億円 / TOPIX超過リターン",
    ha="left", va="bottom", fontsize=7.5, color="gray")

plt.tight_layout(rect=[0, 0.04, 1, 0.95])
plt.savefig(OUT / "result.png", dpi=100, bbox_inches="tight", facecolor="white")
print("result.png saved")
