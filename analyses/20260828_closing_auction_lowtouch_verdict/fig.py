"""左=低タッチで実行できる決済ほどエッジが薄い / 右=ペーパー期間で符号反転。"""
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

s = pd.read_csv(HERE / "signals.csv", parse_dates=["d"])
EX = [("翌09:00\n寄成(MOO)", "nx_open0900", True),
      ("翌09:05", "nx_c0905", False),
      ("翌09:15", "nx_c0915", False),
      ("翌引け\n引成(MOC)", "nx_close", True)]

IS, OOS = [], []
for lb, col, lt in EX:
    x = s.dropna(subset=["entry_adj", col])
    for bucket, sub in ((IS, x[x["d"] <= "2026-05-28"]), (OOS, x[x["d"] >= "2026-05-29"])):
        g = (sub[col] / sub["entry_adj"] - 1) * 1e4
        bucket.append({"lb": lb, "gross": g.mean(), "win": (g > 0).mean() * 100, "lt": lt})
IS, OOS = pd.DataFrame(IS), pd.DataFrame(OOS)

fig, axes = plt.subplots(1, 2, figsize=(12, 6.75), facecolor="white")
x = np.arange(len(EX))
COST = 10.0

# ---- 左: IS net Sharpe と昇格基準2.0（低タッチ可否で塗り分け）
ax = axes[0]
IS_SH = [0.79, 1.69, 2.03, 1.28]      # run.py STEP5 の IS net Sharpe（往復10bps）
cols = ["#0969da" if lt else "#c9d1d9" for lt in IS["lt"]]
ax.bar(x, IS_SH, 0.6, color=cols, edgecolor="#57606a", linewidth=0.6)
ax.axhline(2.0, color="#bf8700", ls="--", lw=1.3)
ax.text(-0.42, 2.08, "昇格基準 2.0", fontsize=8.5, color="#bf8700", ha="left")
for i, v in enumerate(IS_SH):
    ax.text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=10, fontweight="bold",
            color="#0969da" if IS["lt"][i] else "#57606a")
ax.set_xticks(x)
ax.set_xticklabels(IS["lb"], fontsize=9.5)
ax.set_ylabel("net Sharpe（IS 2024-05〜2026-05・往復10bps）")
ax.set_ylim(0, 2.45)
ax.set_title("基準2.0に届くのはザラ場執行が要る09:15だけ\n青=前夜に予約可 / 灰=ザラ場執行が必要", fontsize=11)
ax.grid(alpha=0.3, axis="y")

# ---- 右: IS vs ペーパー期間で符号反転
ax = axes[1]
w = 0.38
ax.bar(x - w / 2, IS["gross"], w, label="IS (2024-05〜2026-05)", color="#0969da")
ax.bar(x + w / 2, OOS["gross"], w, label="ペーパー期間 (2026-05-29〜)", color="#cf222e")
ax.axhline(0, color="black", lw=0.9)
for i in range(len(EX)):
    ax.text(i - w / 2, IS["gross"][i] + 1.3, f"{IS['gross'][i]:.0f}", ha="center", fontsize=9, color="#0969da")
    ax.text(i + w / 2, OOS["gross"][i] - 3.4, f"{OOS['gross'][i]:.0f}", ha="center", fontsize=9, color="#cf222e")
    ax.text(i, -19.5, f"勝率\n{IS['win'][i]:.0f}%→{OOS['win'][i]:.0f}%", ha="center",
            fontsize=8, color="#57606a")
ax.set_xticks(x)
ax.set_xticklabels(IS["lb"], fontsize=9.5)
ax.set_ylabel("gross bps / 取引")
ax.set_ylim(-24, 58)
ax.set_title("4つの決済すべてがペーパー期間で符号反転\n勝率も 55〜59% → 46〜47% に低下", fontsize=11)
ax.legend(fontsize=9, loc="upper left")
ax.grid(alpha=0.3, axis="y")

fig.suptitle("引け板寄せリバウンド候補は却下 — 低タッチで実行できる形にすると何も残らない", fontsize=14)
fig.text(0.99, 0.005, "データ: JQuants stocks_intraday 2024-05〜2026-08 / PIT流動性上位200 / "
                      "close_jump≤−50bps N=3,217 / 分割はadj_factorで調整 / 日次PF系列・√252",
         ha="right", va="bottom", fontsize=7.5, color="gray")
fig.savefig(HERE / "result.png", dpi=100, bbox_inches="tight", facecolor="white")
print("saved result.png")
