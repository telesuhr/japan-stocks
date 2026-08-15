"""結果図: 候補#12は執行タイミングで消え、候補#13はユニバースのバイアスで消える。"""
import sys
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import matplotlib.font_manager as fm
    fm.fontManager.addfont("/root/.fonts/NotoSansJP.ttf")
    plt.rcParams["font.family"] = fm.FontProperties(fname="/root/.fonts/NotoSansJP.ttf").get_name()
    plt.rcParams["axes.unicode_minus"] = False
except Exception:
    pass

H = pd.read_csv("h4_execution_timing.csv")
M = pd.read_csv("mom_decomp.csv")

fig, axes = plt.subplots(1, 2, figsize=(12, 6.75), facecolor="white")

# 左: GapRev 執行タイミング減衰
ax = axes[0]
x = np.arange(len(H))
lbl = ["9:00\n寄成(元)", "9:05", "9:10", "9:20", "9:30"]
ax.bar(x - 0.2, H["net"], 0.4, label="平均 net%", color="#0969da")
ax.bar(x + 0.2, H["med"], 0.4, label="中央値 net%", color="#bf8700")
ax.axhline(0, color="black", lw=0.9)
for i, v in enumerate(H["net"]):
    ax.text(i - 0.2, v + 0.03, f"{v:.2f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(lbl, fontsize=9)
ax.set_ylabel("シグナル日あたり L/Sスプレッド %（8bps後）")
ax.set_title("#12 GapRev: エッジは寄付き10分で消える\n"
             "9:10以降は中央値マイナス＝数日の外れ値だけ", fontsize=11)
ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")

# 右: 6Mモメンタム 分解
ax = axes[1]
names = ["(元)\n存続ユニバース\n×終値→終値", "(A)\nPITユニバース\n×終値→終値",
         "(B)\n存続ユニバース\n×寄成→引成", "(A+B)\nPIT\n×寄成→引成"]
x = np.arange(4)
ax.bar(x - 0.2, M["IS"], 0.4, label="IS Sharpe", color="#8250df")
ax.bar(x + 0.2, M["OOS"], 0.4, label="OOS Sharpe", color="#2da44e")
ax.axhline(0, color="black", lw=0.9)
ax.axhline(0.5, color="red", ls="--", lw=1, label="採用目安 0.5")
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8)
ax.set_ylabel("月次Sharpe（年率換算・8bps後）")
ax.set_title("#13 6Mモメンタム: エッジの正体は\n生存者バイアス（ユニバース修正で0.57→0.12）", fontsize=11)
ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")

fig.suptitle("市場中立L/S候補2本の再検証 — どちらも実運用では消える", fontsize=14)
fig.text(0.99, 0.01,
         "データ: stocks_daily 2016-2026 / stocks_intraday 2024-05〜2026-08 / L/S往復8bps",
         ha="right", va="bottom", fontsize=8, color="gray")
fig.savefig("result.png", dpi=100, bbox_inches="tight", facecolor="white")
print("saved result.png")
