import sys; sys.stdout.reconfigure(line_buffering=True)
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy import stats
import os

FONT_PATH = "/root/.fonts/NotoSansJP.ttf"
if os.path.exists(FONT_PATH):
    font_manager.fontManager.addfont(FONT_PATH)
    plt.rcParams["font.family"] = "Noto Sans JP"
plt.rcParams.update({"axes.facecolor":"#0d1117","figure.facecolor":"#0d1117",
                     "text.color":"#e6edf3","axes.labelcolor":"#e6edf3",
                     "xtick.color":"#8b949e","ytick.color":"#8b949e",
                     "axes.edgecolor":"#30363d","grid.color":"#21262d","grid.alpha":0.5})

df  = pd.read_csv("individual.csv", parse_dates=["dt"])
mkt = pd.read_csv("market_daily.csv", parse_dates=["dt"], index_col=0)

BINS   = [-0.999,-0.03,-0.02,-0.01,-0.005, 0.0, 0.005, 0.01, 0.02, 0.03, 0.999]
LABELS = ["≤-3%","-3〜-2%","-2〜-1%","-1〜-0.5%","-0.5〜0%",
          "0〜+0.5%","+0.5〜+1%","+1〜+2%","+2〜+3%","≥+3%"]
df["am_bucket"] = pd.cut(df.am_ret, bins=BINS, labels=LABELS)

# ── バケット統計 ──
rows = []
for lbl in LABELS:
    sub = df[df.am_bucket == lbl].pm_ret
    if len(sub) < 10: continue
    t, _ = stats.ttest_1samp(sub, 0)
    rows.append({"lbl":lbl, "n":len(sub), "mean":sub.mean()*100,
                 "win":(sub>0).mean()*100, "t":t})
bkt = pd.DataFrame(rows)

fig, axes = plt.subplots(1, 3, figsize=(15, 6), facecolor="#0d1117")
fig.suptitle("前場リターン → 後場リターン  (個別銘柄・流動性ADV≥100億・2024/11〜2026/06)",
             fontsize=13, color="#e6edf3", y=1.01)

# ── 左: バケット別 後場平均リターン ──
ax = axes[0]
ax.set_facecolor("#161b22")
colors = ["#2ea043" if v > 0 else "#f85149" for v in bkt["mean"]]
bars = ax.barh(bkt["lbl"], bkt["mean"], color=colors, alpha=0.85, height=0.7)
ax.axvline(0, color="#8b949e", lw=1)

# t値で有意なバーに星印
for i, row in bkt.iterrows():
    sig = "**" if abs(row.t) >= 3.0 else ("*" if abs(row.t) >= 2.0 else "")
    if sig:
        x_pos = row["mean"] + (0.015 if row["mean"] >= 0 else -0.015)
        ax.text(x_pos, i, sig, va="center", ha="center", fontsize=11,
                color="#d29922", fontweight="bold")

ax.set_xlabel("後場平均リターン (%)"); ax.set_title("後場平均リターン by 前場バケット", fontsize=11, color="#8b949e")
ax.grid(True, axis="x"); ax.set_xlim(-0.3, 0.35)
ax.text(0.5, -0.12, "** t≥3.0  * t≥2.0", transform=ax.transAxes,
        fontsize=8, color="#d29922", ha="center")

# ── 中: 後場勝率 ──
ax2 = axes[1]
ax2.set_facecolor("#161b22")
win_colors = ["#2ea043" if v >= 52 else ("#f85149" if v <= 48 else "#8b949e") for v in bkt["win"]]
ax2.barh(bkt["lbl"], bkt["win"], color=win_colors, alpha=0.85, height=0.7)
ax2.axvline(50, color="#8b949e", lw=1, linestyle="--")
ax2.set_xlabel("後場勝率 (%)"); ax2.set_title("後場勝率 by 前場バケット", fontsize=11, color="#8b949e")
ax2.set_xlim(44, 56); ax2.grid(True, axis="x")
for i, row in bkt.iterrows():
    ax2.text(row["win"]+0.1, i, f"{row['win']:.1f}%", va="center", fontsize=8, color="#e6edf3")

# ── 右: 散布図 + 回帰線 ──
ax3 = axes[2]
ax3.set_facecolor("#161b22")
sample = df.sample(min(5000, len(df)), random_state=42)
sc_color = ["#2ea043" if v > 0 else "#f85149" for v in sample.am_ret]
ax3.scatter(sample.am_ret*100, sample.pm_ret*100, c=sc_color, alpha=0.15, s=4)

# 回帰線
x = df.am_ret.values; y = df.pm_ret.values
sl, ic, r, p, _ = stats.linregress(x, y)
xr = np.linspace(-0.12, 0.12, 100)
ax3.plot(xr*100, (sl*xr+ic)*100, color="#d29922", lw=2, label=f"β={sl:.3f}, r={r:.3f}")
ax3.axhline(0, color="#8b949e", lw=0.7, alpha=0.6)
ax3.axvline(0, color="#8b949e", lw=0.7, alpha=0.6)
ax3.set_xlabel("前場リターン (%)"); ax3.set_ylabel("後場リターン (%)")
ax3.set_title("前場 vs 後場 散布図", fontsize=11, color="#8b949e")
ax3.legend(fontsize=9, framealpha=0, labelcolor="#e6edf3")
ax3.grid(True); ax3.set_xlim(-8, 8); ax3.set_ylim(-8, 8)
ax3.text(0.05, 0.95, f"β={sl:+.4f}\nR²={r**2:.4f}\np={p:.1e}",
         transform=ax3.transAxes, fontsize=9, color="#d29922", va="top",
         bbox=dict(facecolor="#1c2128", alpha=0.8, boxstyle="round,pad=0.4"))

# 前場-3%超での回復アノテート
ax3.annotate("前場≤-3%\n後場+0.22%**\n(t=3.98)",
             xy=(-5, 0.22*3), xytext=(-7.5, 3),
             fontsize=8, color="#2ea043",
             arrowprops=dict(arrowstyle="->", color="#2ea043", lw=1.2))

plt.tight_layout()
plt.savefig("result.png", dpi=130, bbox_inches="tight", facecolor="#0d1117")
print("result.png 保存完了")
