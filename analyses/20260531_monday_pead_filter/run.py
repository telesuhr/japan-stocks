"""
月曜エントリーフィルター × PEAD gap≥7% 戦略 検証
================================================================
earnings_pead戦略 (gap≥7%, Long-only, Sharpe 2.19) に対して、
月曜エントリー（木曜発表→金曜ギャップ→月曜引けエントリー）限定で
パフォーマンスが改善するかを検証。

仮説:
  - 週末の情報消化不全により、月曜エントリーは他曜日より優位?
  - 週末をまたぐことで機関投資家の追随買いが月曜に集中?

データ: pead_obs.csv (2021-2026, N≈11,305件)
  - car0: 決算発表日翌日の寄付ギャップ (decimal, e.g. 0.07 = +7%)
  - entry_date: エントリー日 (= gap発生日 = 決算発表翌日)
  - d5/d10/d20: entry_dateから5/10/20日後の累積リターン (bps)

分析方法:
  A. gap≥7% Long-only の全サンプル vs 月曜エントリーのみ
  B. 曜日別パフォーマンス比較 (全5曜日)
  C. IS(2021-2023) / OOS(2024-2026) split
  D. threshold感度分析 (gap≥5/7/10/15%)
"""
import sys; sys.stdout.reconfigure(line_buffering=True)
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

OUT = Path(__file__).parent
OBS_CSV = Path(__file__).parent.parent / "20260530_pead_price_reaction" / "pead_obs.csv"

COST_BPS = 20.0      # 往復コスト (片道10bps想定)
GAP_THRESH = 0.07    # earnings_pead の閾値 (7%)
OOS_START = "2024-01-01"

DOW_NAMES = {0: "月曜", 1: "火曜", 2: "水曜", 3: "木曜", 4: "金曜"}


def load_data():
    df = pd.read_csv(OBS_CSV)
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["dow"] = df["entry_date"].dt.dayofweek  # 0=月, 4=金
    df["dow_name"] = df["dow"].map(DOW_NAMES)
    for col in ["car0", "d5", "d10", "d20"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def sharpe_longonly(series_bps, cost_bps=COST_BPS, hold_days=5):
    """Long-only の日次リターン (bps) からSharpeを計算。
    bps → decimal → net → 年率Sharpe"""
    net = (series_bps - cost_bps) / 1e4   # decimal
    if len(net) < 10 or net.std() == 0:
        return {"n": len(net), "gross_bps": np.nan, "net_bps": np.nan,
                "sharpe": np.nan, "t_stat": np.nan, "win_rate": np.nan}
    ann_factor = np.sqrt(245 / hold_days)
    sharpe = net.mean() / net.std() * ann_factor
    t_stat = net.mean() / (net.std() / np.sqrt(len(net)))
    return {
        "n": len(net),
        "gross_bps": round(net.mean() * 1e4 + cost_bps, 1),
        "net_bps": round(net.mean() * 1e4, 1),
        "sharpe": round(sharpe, 2),
        "t_stat": round(t_stat, 2),
        "win_rate": round((net > 0).mean(), 3),
    }


def section_a(df):
    """A. gap≥7% Long-only: 全サンプル vs 月曜エントリー比較"""
    print("===== A. gap≥7% Long-only: 全体 vs 月曜エントリー =====")
    g7 = df[df["car0"] >= GAP_THRESH].copy()
    mon = g7[g7["dow"] == 0]  # 月曜エントリー

    rows = []
    for label, subset in [("全体(gap≥7%)", g7), ("月曜エントリー(gap≥7%)", mon)]:
        for hold, col in [(5, "d5"), (10, "d10"), (20, "d20")]:
            s = subset.dropna(subset=[col])
            r = sharpe_longonly(s[col], hold_days=hold)
            r["label"] = label
            r["hold"] = hold
            rows.append(r)
            print(f"  {label} hold={hold}d: n={r['n']} gross={r['gross_bps']}bps "
                  f"net={r['net_bps']}bps Sharpe={r['sharpe']} t={r['t_stat']} "
                  f"win={r['win_rate']}")

    return pd.DataFrame(rows)


def section_b(df):
    """B. 曜日別パフォーマンス比較"""
    print("\n===== B. 曜日別 Long-only (gap≥7%, d5) =====")
    g7 = df[df["car0"] >= GAP_THRESH].dropna(subset=["d5"])
    rows = []
    for dow in range(5):
        sub = g7[g7["dow"] == dow]
        r = sharpe_longonly(sub["d5"], hold_days=5)
        r["dow"] = dow
        r["dow_name"] = DOW_NAMES[dow]
        rows.append(r)
        print(f"  {DOW_NAMES[dow]}: n={r['n']} gross={r['gross_bps']}bps "
              f"net={r['net_bps']}bps Sharpe={r['sharpe']} t={r['t_stat']}")
    return pd.DataFrame(rows)


def section_c(df):
    """C. IS/OOS split"""
    print("\n===== C. IS / OOS split (gap≥7%, d5) =====")
    g7 = df[df["car0"] >= GAP_THRESH].dropna(subset=["d5"])
    is_df = g7[g7["entry_date"] < OOS_START]
    oos_df = g7[g7["entry_date"] >= OOS_START]
    mon_is = is_df[is_df["dow"] == 0]
    mon_oos = oos_df[oos_df["dow"] == 0]

    rows = []
    for label, subset in [("IS全体", is_df), ("IS月曜", mon_is),
                           ("OOS全体", oos_df), ("OOS月曜", mon_oos)]:
        r = sharpe_longonly(subset["d5"], hold_days=5)
        r["label"] = label
        rows.append(r)
        print(f"  {label}: n={r['n']} gross={r['gross_bps']}bps "
              f"net={r['net_bps']}bps Sharpe={r['sharpe']} t={r['t_stat']}")
    return pd.DataFrame(rows)


def section_d(df):
    """D. gap閾値感度分析 × 月曜フィルター"""
    print("\n===== D. gap閾値感度 × 月曜フィルター (d5) =====")
    rows = []
    for thresh in [0.05, 0.07, 0.10, 0.15]:
        g = df[df["car0"] >= thresh].dropna(subset=["d5"])
        mon = g[g["dow"] == 0]
        r_all = sharpe_longonly(g["d5"], hold_days=5)
        r_mon = sharpe_longonly(mon["d5"], hold_days=5)
        rows.append({"thresh": thresh, "group": "全体",
                     "n": r_all["n"], "sharpe": r_all["sharpe"], "t": r_all["t_stat"]})
        rows.append({"thresh": thresh, "group": "月曜",
                     "n": r_mon["n"], "sharpe": r_mon["sharpe"], "t": r_mon["t_stat"]})
        print(f"  gap≥{thresh*100:.0f}% 全体: n={r_all['n']} Sharpe={r_all['sharpe']} t={r_all['t_stat']}")
        print(f"  gap≥{thresh*100:.0f}% 月曜: n={r_mon['n']} Sharpe={r_mon['sharpe']} t={r_mon['t_stat']}")
    return pd.DataFrame(rows)


def make_chart(df_b, df_c, df_d):
    """result.png: 3パネル可視化"""
    # フォント設定
    for fname in ["NotoSansCJK-Regular.ttc", "NotoSansJP-Regular.ttf",
                  "IPAexGothic.ttf", "IPAGothic.ttf"]:
        found = [f for f in fm.findSystemFonts() if fname.lower() in f.lower()]
        if found:
            plt.rcParams["font.family"] = fm.FontProperties(fname=found[0]).get_name()
            break

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.75))
    fig.patch.set_facecolor("white")
    fig.suptitle("月曜エントリーフィルター × PEAD gap≥7% 戦略", fontsize=16, fontweight="bold", y=1.01)

    # Panel 1: 曜日別Sharpe
    ax1 = axes[0]
    dows = df_b["dow_name"].tolist()
    sharpes = df_b["sharpe"].tolist()
    colors = ["#e74c3c" if d == "月曜" else "#3498db" for d in dows]
    bars = ax1.bar(dows, sharpes, color=colors, alpha=0.8, edgecolor="white")
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.axhline(1.0, color="green", linewidth=1.0, linestyle="--", alpha=0.6, label="Sharpe=1.0")
    for bar, v in zip(bars, sharpes):
        if pd.notna(v):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f"{v:.2f}", ha="center", va="bottom", fontsize=10)
    ax1.set_title("曜日別 Sharpe (gap≥7%, d5)", fontsize=12)
    ax1.set_ylabel("年率Sharpe")
    ax1.set_facecolor("#f8f9fa")
    ax1.grid(axis="y", alpha=0.3)
    ax1.legend(fontsize=9)

    # Panel 2: IS/OOS比較
    ax2 = axes[1]
    labels = df_c["label"].tolist()
    sharpes2 = df_c["sharpe"].tolist()
    ns = df_c["n"].tolist()
    colors2 = ["#3498db", "#e74c3c", "#2ecc71", "#e67e22"]
    bars2 = ax2.bar(labels, sharpes2, color=colors2, alpha=0.8, edgecolor="white")
    ax2.axhline(0, color="black", linewidth=0.8)
    for bar, v, n in zip(bars2, sharpes2, ns):
        if pd.notna(v):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f"{v:.2f}\n(n={n})", ha="center", va="bottom", fontsize=9)
    ax2.set_title("IS / OOS split (gap≥7%, d5)", fontsize=12)
    ax2.set_ylabel("年率Sharpe")
    ax2.set_facecolor("#f8f9fa")
    ax2.grid(axis="y", alpha=0.3)
    ax2.tick_params(axis="x", rotation=15)

    # Panel 3: gap閾値感度
    ax3 = axes[2]
    thresholds = sorted(df_d["thresh"].unique())
    x = np.arange(len(thresholds))
    w = 0.35
    s_all = df_d[df_d["group"] == "全体"]["sharpe"].tolist()
    s_mon = df_d[df_d["group"] == "月曜"]["sharpe"].tolist()
    ax3.bar(x - w/2, s_all, w, label="全体", color="#3498db", alpha=0.8, edgecolor="white")
    ax3.bar(x + w/2, s_mon, w, label="月曜のみ", color="#e74c3c", alpha=0.8, edgecolor="white")
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"≥{int(t*100)}%" for t in thresholds])
    ax3.axhline(0, color="black", linewidth=0.8)
    ax3.set_title("gap閾値感度 × 月曜フィルター", fontsize=12)
    ax3.set_ylabel("年率Sharpe")
    ax3.set_facecolor("#f8f9fa")
    ax3.grid(axis="y", alpha=0.3)
    ax3.legend()

    fig.text(0.5, -0.02,
             "データ: JQuants 2021-2026 / PEAD L/S pead_obs.csv (N=11,305) / コスト20bps",
             ha="center", fontsize=9, color="gray")

    plt.tight_layout()
    plt.savefig(OUT / "result.png", dpi=100, bbox_inches="tight", facecolor="white")
    print(f"\n  chart saved: {OUT / 'result.png'}")


def main():
    print("[RUN] 月曜エントリーフィルター × PEAD gap≥7%")
    df = load_data()
    g7 = df[df["car0"] >= GAP_THRESH]
    n_all = len(df)
    n_g7 = len(g7)
    n_mon = len(g7[g7["dow"] == 0])
    print(f"  全PEAD観測: {n_all:,}件")
    print(f"  gap≥7% フィルター後: {n_g7:,}件 ({n_g7/n_all:.1%})")
    print(f"  うち月曜エントリー: {n_mon:,}件 ({n_mon/n_g7:.1%})")
    print()

    df_a = section_a(df)
    df_b = section_b(df)
    df_c = section_c(df)
    df_d = section_d(df)

    df_a.to_csv(OUT / "section_a.csv", index=False)
    df_b.to_csv(OUT / "section_b_dow.csv", index=False)
    df_c.to_csv(OUT / "section_c_isoos.csv", index=False)
    df_d.to_csv(OUT / "section_d_threshold.csv", index=False)

    # 結論サマリー
    r_all = df_b[df_b["dow_name"] == "月曜"].iloc[0]  # actually section_a has 全体
    all_g7 = sharpe_longonly(g7.dropna(subset=["d5"])["d5"], hold_days=5)
    mon_g7 = sharpe_longonly(g7[g7["dow"] == 0].dropna(subset=["d5"])["d5"], hold_days=5)
    print("\n===== 結論サマリー =====")
    print(f"  earnings_pead(gap≥7%) 全体  : Sharpe={all_g7['sharpe']} t={all_g7['t_stat']} n={all_g7['n']}")
    print(f"  earnings_pead(gap≥7%) 月曜限定: Sharpe={mon_g7['sharpe']} t={mon_g7['t_stat']} n={mon_g7['n']}")
    if mon_g7["sharpe"] is not np.nan and all_g7["sharpe"] is not np.nan:
        diff = mon_g7["sharpe"] - all_g7["sharpe"]
        verdict = "改善" if diff > 0.1 else "悪化" if diff < -0.1 else "ほぼ同等"
        print(f"  月曜フィルターの効果: {verdict} (差={diff:+.2f})")
        if mon_g7["t_stat"] >= 2.0 and mon_g7["sharpe"] >= 1.5:
            print("  → 月曜フィルターは統計的に有意。signal_check.pyへの組み込みを検討。")
        else:
            print(f"  → t={mon_g7['t_stat']} < 2.0 or Sharpe={mon_g7['sharpe']} < 1.5: 閾値未達。フィルター追加は時期尚早。")

    make_chart(df_b, df_c, df_d)
    print("[DONE]")


if __name__ == "__main__":
    main()
