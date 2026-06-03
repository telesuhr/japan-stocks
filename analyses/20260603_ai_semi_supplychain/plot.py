"""
AI半導体サプライチェーン リードラグ分析 — 可視化
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import font_manager
import os

FONT_PATH = "/root/.fonts/NotoSansJP.ttf"
if os.path.exists(FONT_PATH):
    font_manager.fontManager.addfont(FONT_PATH)
    plt.rcParams["font.family"] = "Noto Sans JP"
plt.rcParams.update({"axes.facecolor":"#0d1117","figure.facecolor":"#0d1117",
                     "text.color":"#e6edf3","axes.labelcolor":"#e6edf3",
                     "xtick.color":"#8b949e","ytick.color":"#8b949e",
                     "axes.edgecolor":"#30363d","grid.color":"#21262d",
                     "grid.alpha":0.6,"axes.titlesize":11,"axes.labelsize":10})

LAYER_COLORS = {"上流":"#58a6ff","装置":"#f0883e","部品":"#3fb950","AI下流":"#d2a8ff"}
LAYER_ORDER  = ["上流","装置","部品","AI下流"]
LAG_RANGE    = list(range(-5, 6))

def main():
    df = pd.read_csv("crosscorr_all.csv")
    rolling = pd.read_csv("rolling_corr.csv", index_col=0, parse_dates=True)
    lr = pd.read_csv("layer_returns.csv", index_col=0, parse_dates=True)

    fig = plt.figure(figsize=(14, 9), dpi=130, facecolor="#0d1117")
    fig.text(0.012, 0.97, "AI半導体サプライチェーン リードラグ分析",
             fontsize=14, fontweight="bold", color="#e6edf3", va="top")
    fig.text(0.012, 0.94, "4層(上流材料・製造装置・デバイス部品・AI/DC下流)の日足リターン先行性検証 / 2018-2026",
             fontsize=9, color="#8b949e", va="top")

    # ── 左上: クロス相関 (装置が各層に対して先行 / 全期間) ──
    ax1 = fig.add_axes([0.06, 0.55, 0.40, 0.34])
    ax1.set_facecolor("#161b22")
    ax1.set_title("製造装置 → 各層のクロス相関 (全期間)", fontsize=10, color="#e6edf3", pad=6)

    all_df = df[df.period=="all"]
    for follow in LAYER_ORDER:
        if follow == "装置": continue
        sub = all_df[(all_df.lead=="装置") & (all_df.follow==follow)]
        sub = sub.sort_values("lag")
        color = LAYER_COLORS[follow]
        ax1.plot(sub.lag, sub.r, color=color, linewidth=2, marker="o", markersize=4, label=follow)
    ax1.axvline(0, color="#8b949e", linewidth=1, linestyle="--", alpha=0.6)
    ax1.axhline(0, color="#8b949e", linewidth=0.5, alpha=0.4)
    ax1.set_xlabel("Lag (日) — 正: 装置が先行"); ax1.set_ylabel("Pearson r")
    ax1.set_xticks(LAG_RANGE)
    ax1.legend(fontsize=9, framealpha=0.0, labelcolor="#e6edf3")
    ax1.grid(True); ax1.set_ylim(-0.1, 0.85)
    # 「ほぼ対称=共通因子」をアノテート
    ax1.text(0.5, 0.06, "lag+N ≈ lag-N → 共通因子で同時に動く", transform=ax1.transAxes,
             fontsize=8, color="#f0883e", ha="center",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#1c2128", alpha=0.8))

    # ── 右上: 翌日予測 IC バー ──
    ax2 = fig.add_axes([0.56, 0.55, 0.40, 0.34])
    ax2.set_facecolor("#161b22")
    ax2.set_title("翌日予測 Rank IC（今日の層リターン → 翌日の他層）", fontsize=10, color="#e6edf3", pad=6)

    pairs_ic = {
        "装置→部品":  +0.0139,
        "部品→AI下流": +0.0099,
        "上流→部品":   +0.0078,
        "部品→上流":   +0.0056,
        "装置→上流":   +0.0045,
        "AI下流→部品": +0.0030,
        "部品→装置":  -0.0007,
        "AI下流→装置":-0.0030,
        "装置→AI下流":-0.0074,
        "AI下流→上流":-0.0125,
        "上流→AI下流":-0.0191,
        "上流→装置":  -0.0258,
    }
    pair_labels = list(pairs_ic.keys())
    ics = list(pairs_ic.values())
    colors = ["#2ea043" if v > 0 else "#f85149" for v in ics]
    y_pos = range(len(pair_labels))
    ax2.barh(y_pos, ics, color=colors, alpha=0.85, height=0.7)
    ax2.set_yticks(y_pos); ax2.set_yticklabels(pair_labels, fontsize=9)
    ax2.axvline(0, color="#8b949e", linewidth=1)
    ax2.set_xlabel("Rank IC (t < 1.2 → 全て非有意)")
    # 有意ライン（仮にN=2049で |IC|=0.043がt=2.0相当）
    sig_level = 2.0 / np.sqrt(2049 - 2 + 4)  # ≈ 0.044
    ax2.axvline(sig_level, color="#d29922", linewidth=1, linestyle=":", alpha=0.7)
    ax2.axvline(-sig_level, color="#d29922", linewidth=1, linestyle=":", alpha=0.7)
    ax2.text(sig_level+0.001, len(pair_labels)-0.8, "t=2.0", color="#d29922", fontsize=7)
    ax2.grid(True, axis="x"); ax2.set_facecolor("#161b22")
    ax2.text(0.5, 0.04, "全12ペアが有意水準内 → 予測力ゼロ", transform=ax2.transAxes,
             fontsize=8, color="#f85149", ha="center",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#1c2128", alpha=0.8))

    # ── 左下: ローリング相関 (装置→AI下流 lag=0 vs lag+1) ──
    ax3 = fig.add_axes([0.06, 0.10, 0.40, 0.34])
    ax3.set_facecolor("#161b22")
    ax3.set_title("装置 × AI下流 — ローリング相関 120日 (lag=0同時 vs lag+1先行)", fontsize=10, color="#e6edf3", pad=6)

    # lag=0: 同時相関
    col_装置 = [c for c in lr.columns if "製造装置" in c or "装置" in c][0]
    col_ai   = [c for c in lr.columns if "AI" in c or "下流" in c][0]
    both = pd.DataFrame({"x":lr[col_装置], "y":lr[col_ai]}).dropna()
    r0_roll = both["x"].rolling(120).corr(both["y"])
    ax3.plot(r0_roll.index, r0_roll, color="#58a6ff", linewidth=1.5, label="lag=0 (同時)", alpha=0.9)
    ax3.plot(rolling.index, rolling.iloc[:,0], color="#d2a8ff", linewidth=1.5, linestyle="--", label="lag=+1 (装置先行)", alpha=0.9)
    ax3.axvline(pd.Timestamp("2023-01-01"), color="#d29922", linewidth=1, linestyle=":", alpha=0.8)
    ax3.text(pd.Timestamp("2023-01-15"), ax3.get_ylim()[0]+0.02 if ax3.get_ylim()[0] else 0.1,
             "AI相場\n開始", color="#d29922", fontsize=7)
    ax3.set_ylabel("Pearson r"); ax3.legend(fontsize=9, framealpha=0.0, labelcolor="#e6edf3")
    ax3.grid(True); ax3.set_ylim(-0.1, 1.0)

    # ── 右下: 結論サマリー ──
    ax4 = fig.add_axes([0.56, 0.10, 0.40, 0.34])
    ax4.set_facecolor("#161b22")
    ax4.axis("off")
    ax4.set_title("分析結論", fontsize=10, color="#e6edf3", pad=6)

    conclusions = [
        ("❌", "上流→下流のリードラグなし",
         "lag=+N と lag=-N が対称 → 共通因子(SOX/NASDAQ/円安)\nで全層が同時に動く。前回の1分足分析と同結論が\n日足でも再現。"),
        ("❌", "翌日予測力ゼロ",
         "全12ペアの翻日Rank IC は |IC|<0.03, |t|<1.2。\n「装置が強い日の翌日に部品を買う」は機能しない。"),
        ("❌", "SBGも例外でなし",
         "SBGのクロス相関はlag=±2がほぼ同値。\n他の半導体層と同じタイミングで動く。"),
        ("✅", "AI相場で相関は安定継続",
         "AI相場(2023-)でも装置↔AI下流の同時相関\n≈0.47は維持。セクター内の強連動は構造的。"),
        ("💡", "示唆: モメンタム戦略で使う",
         "リードラグ利用は不可だが、「全層が同時に\n動く」ならセクター全体の勢い判定に活用可。\n→ 電機・精密 #1 保有継続の追加根拠。"),
    ]

    y = 0.95
    for icon, title, detail in conclusions:
        color = "#2ea043" if icon=="✅" else ("#d29922" if icon=="💡" else "#f85149")
        ax4.text(0.02, y, f"{icon} {title}", transform=ax4.transAxes,
                 fontsize=9, fontweight="bold", color=color, va="top")
        ax4.text(0.04, y-0.05, detail, transform=ax4.transAxes,
                 fontsize=8, color="#8b949e", va="top", linespacing=1.5)
        y -= 0.20

    # 出典注記
    fig.text(0.012, 0.02,
             "DB: stocks_daily / 16銘柄+SBG / 日次等ウェイト層リターン / ADV≥1億円フィルタなし / 前回(20260421)は1分足で同結論",
             fontsize=7.5, color="#656d76")

    plt.savefig("result.png", bbox_inches="tight", dpi=130, facecolor="#0d1117")
    print("result.png 保存完了")


if __name__ == "__main__":
    main()
