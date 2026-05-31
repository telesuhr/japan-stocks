"""result.png 生成: 信用残月次L/S の IS/OOS比較"""
import sys; sys.stdout.reconfigure(line_buffering=True)
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path

OUT = Path(__file__).parent
_font = "/root/.fonts/NotoSansJP.ttf"
if Path(_font).exists():
    fm.fontManager.addfont(_font)
    plt.rcParams["font.family"] = fm.FontProperties(fname=_font).get_name()
plt.rcParams["axes.unicode_minus"] = False

r = pd.read_csv(OUT / "results.csv")
r = r[r["mode"] == "sector_neutral"]  # セクター中立版が最もクリーン

fig, axes = plt.subplots(1, 2, figsize=(12, 6.75), facecolor="white")
fig.suptitle("信用残ファクター 月次L/S (セクター中立, コスト10bps後)\nIS:2021-2023 / OOS:2024-2026",
             fontsize=13, fontweight="bold")

# 左: 各ファクターのIS/OOS Sharpe
ax = axes[0]
factors = ["margin_ratio", "long_chg", "short_chg", "net_chg"]
fnames = ["信用倍率\n(margin_ratio)", "信用買増\n(long_chg)", "信用売増\n(short_chg)", "需給変化\n(net_chg)"]
x = np.arange(len(factors))
w = 0.25
for i, (split, color) in enumerate([("IS", "#4C72B0"), ("OOS", "#DD8452"), ("ALL", "#55A868")]):
    vals = [r[(r["factor"]==f) & (r["label"]==split)]["sharpe"].values[0]
            if len(r[(r["factor"]==f) & (r["label"]==split)]) > 0 else np.nan
            for f in factors]
    ax.bar(x + (i-1)*w, vals, w, label=split, color=color, alpha=0.85)
ax.axhline(0, color="black", lw=0.8)
ax.axhline(2.0, color="red", lw=1.2, ls="--", alpha=0.7, label="昇格基準 Sharpe=2.0")
ax.set_xticks(x)
ax.set_xticklabels(fnames, fontsize=9)
ax.set_ylabel("年率Sharpe（コスト後）")
ax.set_title("ファクター別Sharpe（月次L/S, セクター中立）", fontsize=11)
ax.legend(fontsize=9)
ax.set_ylim(-2.5, 2.5)
ax.grid(axis="y", alpha=0.3)

# 右: short_chg の月次累積L/S推移（IS/OOS）
ax2 = axes[1]
panel = pd.read_csv(OUT / "panel.csv")
panel["month_end"] = pd.to_datetime(panel["month_end"])

# short_chg月次L/Sを再計算して累積プロット
months = sorted(panel["month_end"].unique())
ls_series = []
for me in months:
    x2 = panel[panel["month_end"] == me].dropna(subset=["short_chg", "fwd20_xs", "sector33_nm"]).copy()
    if len(x2) < 20:
        ls_series.append({"month": me, "ls": np.nan})
        continue
    x2["fz"] = x2.groupby("sector33_nm")["short_chg"].transform(
        lambda s: (s - s.mean()) / s.std() if s.std() > 0 else 0.0)
    x2["q"] = pd.qcut(x2["fz"].rank(method="first"), 5, labels=False)
    lr = x2[x2["q"] == 4]["fwd20_xs"].mean()
    sr = x2[x2["q"] == 0]["fwd20_xs"].mean()
    ls_series.append({"month": me, "ls": (lr - sr) * 1e4})

ls_df = pd.DataFrame(ls_series).dropna()
ls_df["net"] = ls_df["ls"] - 10  # コスト10bps
ls_df["cumret"] = ls_df["net"].cumsum()
OOS_START = pd.Timestamp("2024-01-01")

is_part = ls_df[ls_df["month"] < OOS_START]
oos_part = ls_df[ls_df["month"] >= OOS_START]
ax2.plot(is_part["month"], is_part["cumret"], color="#4C72B0", lw=2, label=f"IS (Sharpe={r[(r['factor']=='short_chg')&(r['label']=='IS')]['sharpe'].values[0]:.2f})")
if len(oos_part) > 0:
    oos_cumret = oos_part["net"].cumsum() + is_part["cumret"].iloc[-1]
    ax2.plot(oos_part["month"], oos_cumret, color="#DD8452", lw=2, label=f"OOS (Sharpe={r[(r['factor']=='short_chg')&(r['label']=='OOS')]['sharpe'].values[0]:.2f})")
ax2.axvline(OOS_START, color="gray", ls="--", lw=1.2, label="IS/OOS境界")
ax2.axhline(0, color="black", lw=0.8)
ax2.set_xlabel("月")
ax2.set_ylabel("累積L/Sリターン（bps, コスト後）")
ax2.set_title("信用売増 (short_chg) L/S 累積推移\n→ IS好調・OOSで急失速", fontsize=11)
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

fig.text(0.01, 0.01, "データ: JQuants信用残2021-2026 / 月次月末サンプリング / 流動性≥10億円 / TOPIX超過",
         ha="left", va="bottom", fontsize=7.5, color="gray")
plt.tight_layout(rect=[0, 0.04, 1, 0.94])
plt.savefig(OUT / "result.png", dpi=100, bbox_inches="tight", facecolor="white")
print("result.png saved")
