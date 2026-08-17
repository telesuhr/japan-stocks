"""結果図: 左=バリュー優位は足元も継続(PIT形成→前向き) / 右=スクリーン通過14銘柄の位置。"""
import sys
sys.stdout.reconfigure(line_buffering=True)
import pandas as pd
import matplotlib.pyplot as plt

try:
    import matplotlib.font_manager as fm
    fm.fontManager.addfont("/root/.fonts/NotoSansJP.ttf")
    plt.rcParams["font.family"] = fm.FontProperties(fname="/root/.fonts/NotoSansJP.ttf").get_name()
    plt.rcParams["axes.unicode_minus"] = False
except Exception:
    pass

R = pd.read_csv("rotation_pit.csv")
C = pd.read_csv("candidates_nonfin.csv")
F = pd.read_csv("candidates_fin.csv")

fig, axes = plt.subplots(1, 2, figsize=(12, 6.75), facecolor="white")

# 左: PIT形成 → 前向きリターンの Q1-Q5 スプレッド
ax = axes[0]
x = range(len(R))
col = ["#0969da" if v >= 0 else "#cf222e" for v in R["spread"]]
ax.bar(x, R["spread"], color=col)
ax.plot(x, R["mkt"], color="#bf8700", lw=1.6, marker="o", ms=3.5, label="市場EW（同期間）")
ax.axhline(0, color="black", lw=0.9)
ax.set_xticks(list(x))
ax.set_xticklabels([s[2:] for s in R["form"]], rotation=60, fontsize=7.5)
ax.set_ylabel("形成日→2026-08-14 の累積リターン差 %")
ax.set_xlabel("PBR分位の形成日（PIT・非金融・ADV≥3億）")
ax.set_title("割安(Q1) − 割高(Q5) は 12回中11回プラス\n＝バリュー優位は足元も継続", fontsize=11)
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis="y")

# 右: 通過銘柄の PBR × 予想利回り（バブル=予想ROE）。純利/営利≥1.0は一過性利益の疑いで別色
A = pd.concat([C.assign(g="非金融"), F.assign(g="金融")], ignore_index=True)
A["warn"] = A["純利/営利"].fillna(0) >= 1.0
ax = axes[1]
for (grp, w), d in A.groupby(["g", "warn"]):
    ax.scatter(d["PBR"], d["予想利回%"], s=d["予想ROE%"] * 14, alpha=0.8,
               color="#cf222e" if w else ("#0969da" if grp == "非金融" else "#8250df"),
               marker="X" if w else "o", edgecolor="white")
for _, r in A.iterrows():
    ax.annotate(r["銘柄"][:7], (r["PBR"], r["予想利回%"]),
                textcoords="offset points", xytext=(0, 9), ha="center", fontsize=7.5)
ax.set_ylim(A["予想利回%"].min() - 0.35, A["予想利回%"].max() + 0.45)
for c, m, lb in [("#0969da", "o", "非金融"), ("#8250df", "o", "金融(リース等)"),
                 ("#cf222e", "X", "純利益>営業利益＝一過性の疑い")]:
    ax.scatter([], [], color=c, marker=m, s=60, label=lb)
ax.axhline(3.0, color="gray", ls="--", lw=0.9)
ax.axvline(1.0, color="gray", ls="--", lw=0.9)
ax.set_xlabel("PBR（自己資本=TA×自己資本比率ベース）")
ax.set_ylabel("会社予想 配当利回り %")
ax.set_title("通過17銘柄（バブル径＝予想ROE）\nPBR≤1・予想PER≤13・利回≥3%・ROE≥8%・増益・5期黒字", fontsize=11)
ax.legend(fontsize=8, loc="lower left", framealpha=0.9)
ax.grid(alpha=0.3)

fig.suptitle("中長期・現物向け 割安株スクリーン（2026-08-14基準）", fontsize=14)
fig.text(0.99, 0.005,
         "データ: JQuants stocks_daily / fin_summary（会社予想ベース）· 全上場3,927銘柄から抽出",
         ha="right", va="bottom", fontsize=8, color="gray")
fig.savefig("result.png", dpi=100, bbox_inches="tight", facecolor="white")
print("saved result.png")
