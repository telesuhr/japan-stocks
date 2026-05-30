"""result.png 生成（X投稿用 1200x675）。edge_summary.csv を読んで否定的結果を可視化。"""
import sys
sys.stdout.reconfigure(line_buffering=True)
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd

OUTDIR = Path(__file__).parent
# 日本語フォント
fp = "/root/.fonts/NotoSansJP.ttf"
if Path(fp).exists():
    font_manager.fontManager.addfont(fp)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()
plt.rcParams["axes.unicode_minus"] = False

summ = pd.read_csv(OUTDIR / "edge_summary.csv")

# ALL の kind×direction×horizon を棒グラフ化
d = summ[summ["label"] == "ALL"].copy()
d["key"] = d["kind"].str.replace("volumeSpike", "出来高スパイク").str.replace("vwapDeviation", "VWAP乖離") \
    + "\n" + d["direction"].str.replace("momentum", "順張り").str.replace("reversion", "逆張り") \
    + " " + d["horizon_min"].astype(str) + "分"

fig = plt.figure(figsize=(12, 6.75), facecolor="white")
ax = fig.add_subplot(111)
colors = ["#c0392b"] * len(d)  # 全てマイナス=赤
bars = ax.bar(range(len(d)), d["mean_bps"], color=colors, alpha=0.85)
ax.set_xticks(range(len(d)))
ax.set_xticklabels(d["key"], fontsize=8, rotation=0)
ax.axhline(0, color="#333", linewidth=1)
ax.set_ylabel("1トレード平均損益 (bps, コスト8bps控除後)", fontsize=11)
ax.grid(axis="y", alpha=0.3)

# 各バーに t値 を注記
for i, (_, r) in enumerate(d.iterrows()):
    ax.text(i, r["mean_bps"] - 0.3, f"t={r['t_stat']:.0f}", ha="center", va="top",
            fontsize=7, color="#555")

fig.suptitle("板シグナルにイントラ・エッジは無かった（非鉄・半導体22銘柄／2年）",
             fontsize=15, fontweight="bold", y=0.98)
ax.set_title("出来高スパイク／VWAP乖離 シグナル発火後リターン — 全方向・全期間でコスト後マイナス (N≈11万)",
             fontsize=10, color="#444", pad=8)

fig.text(0.99, 0.01,
         "データ: 2024-05〜2026-05 / 日本株1分足 (JQuants, stocks_intraday) / IS・OOSとも同傾向",
         ha="right", va="bottom", fontsize=8, color="gray")
plt.tight_layout(rect=[0, 0.02, 1, 0.95])
plt.savefig(OUTDIR / "result.png", dpi=100, bbox_inches="tight", facecolor="white")
print("saved result.png")
