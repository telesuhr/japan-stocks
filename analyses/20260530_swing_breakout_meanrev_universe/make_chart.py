"""result.png 生成（X投稿用 1200x675）。平均回帰 vs ブレイクアウトのエッジ比較。"""
import sys
sys.stdout.reconfigure(line_buffering=True)
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd

OUTDIR = Path(__file__).parent
fp = "/root/.fonts/NotoSansJP.ttf"
if Path(fp).exists():
    font_manager.fontManager.addfont(fp)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()
plt.rcParams["axes.unicode_minus"] = False

s = pd.read_csv(OUTDIR / "edge_summary.csv").drop_duplicates()
a = s[s["label"] == "ALL"].copy()
a["lbl"] = (a["family"].str.replace("MR_zscore", "平均回帰z").str.replace("MR_rsi", "平均回帰RSI")
            .str.replace("BO_donchian", "ブレイク") + " " + a["param"] + " " + a["hold"].astype(str) + "日")
a = a.sort_values("mean_bps")

fig = plt.figure(figsize=(12, 6.75), facecolor="white")
ax = fig.add_subplot(111)
colors = ["#2980b9" if "平均回帰" in l else "#c0392b" for l in a["lbl"]]
ax.barh(range(len(a)), a["mean_bps"], color=colors, alpha=0.85)
ax.set_yticks(range(len(a)))
ax.set_yticklabels(a["lbl"], fontsize=8)
ax.axvline(0, color="#333", linewidth=1)
ax.set_xlabel("1トレード平均損益 (bps, コスト10bps控除後)", fontsize=11)
ax.grid(axis="x", alpha=0.3)
for i, (_, r) in enumerate(a.iterrows()):
    ax.text(r["mean_bps"] + (1 if r["mean_bps"] >= 0 else -1), i, f"t={r['t_stat']:.1f}",
            va="center", ha="left" if r["mean_bps"] >= 0 else "right", fontsize=7, color="#555")

fig.suptitle("平均回帰は効くが2022年以降に偏在、ブレイクアウトは全期間ダメ（全上場・流動性≥10億円・10年）",
             fontsize=13, fontweight="bold", y=0.98)
ax.set_title("青=平均回帰 / 赤=ブレイクアウト　t値は大きいがzスコア系の利益はOOSに集中＝レジーム依存の疑い",
             fontsize=9.5, color="#444", pad=8)
fig.text(0.99, 0.01,
         "データ: 2016-05〜2026-05 / 日本株日足 (JQuants, 分割調整) / 流動性: トレーリング60日平均売買代金≥10億円/日",
         ha="right", va="bottom", fontsize=7.5, color="gray")
plt.tight_layout(rect=[0, 0.02, 1, 0.94])
plt.savefig(OUTDIR / "result.png", dpi=100, bbox_inches="tight", facecolor="white")
print("saved result.png")
