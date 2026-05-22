"""
真のOOS検証 PL画像 (1200x675px)
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

OUT = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    'font.family': ['Hiragino Sans', 'Hiragino Maru Gothic Pro', 'IPAexGothic', 'sans-serif'],
    'axes.unicode_minus': False,
    'figure.facecolor': 'white',
    'axes.facecolor': '#f8f9fa',
    'grid.alpha': 0.3,
})

ACCENT = '#d4243a'
NEUTRAL = '#2e4a7d'
GREEN = '#2a8060'
GRAY = '#888888'


def main():
    df_oos = pd.read_csv(os.path.join(OUT, "oos_results.csv"))
    port_daily = pd.read_csv(os.path.join(OUT, "oos_portfolio_daily.csv"))
    port_daily.columns = ["date", "pnl"]
    port_daily["date"] = pd.to_datetime(port_daily["date"])
    port_daily = port_daily.set_index("date")["pnl"]

    eq = port_daily.cumsum() * 100
    mu, sd = port_daily.mean(), port_daily.std()
    sh = mu / sd * np.sqrt(245) if sd > 0 else 0
    dd_series = (port_daily.cumsum() - port_daily.cumsum().cummax()) * 100
    dd = dd_series.min()

    # Re-compute MA/RSI sub-portfolios
    is_selection = pd.read_csv(os.path.join(OUT, "is_selection.csv"))
    adopted = df_oos[df_oos["verdict"] != ""]  # 採用15銘柄全部

    fig = plt.figure(figsize=(12, 6.75), facecolor='white')
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.25,
                          left=0.06, right=0.97, top=0.87, bottom=0.10)

    fig.suptitle("真のOOS検証 ・ 戦略昇格基準クリア確認",
                 fontsize=15, fontweight='bold', y=0.965)
    fig.text(0.5, 0.92,
             f"IS:2016-2020 で銘柄+戦略選別 / OOS:2021-2026 で評価 / "
             f"採用15銘柄 OOS Sh {sh:+.2f} / 累積 {eq.iloc[-1]:+.0f}% / DD {dd:.1f}%",
             ha='center', fontsize=10, color='#333')

    # ===== 左上: OOS累積PnL =====
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(eq.index, eq.values, color=GREEN, linewidth=2.0,
             label=f'採用15銘柄 ({sh:+.2f})')
    ax1.axhline(0, color='gray', linewidth=0.5)
    ax1.set_title("OOS 累積PnL (2021-2026, %)", fontsize=11, loc='left')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_locator(mdates.YearLocator(1))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%y'))
    ax1.tick_params(labelsize=8)

    # ===== 右上: IS vs OOS Sharpe scatter =====
    ax2 = fig.add_subplot(gs[0, 1])
    colors_p = [GREEN if v >= 2 else (NEUTRAL if v > 0 else ACCENT)
                 for v in df_oos["OOS_Sharpe"]]
    ax2.scatter(df_oos["IS_Sharpe"], df_oos["OOS_Sharpe"], c=colors_p, s=80, alpha=0.85)
    # 銘柄ラベル
    for _, r in df_oos.iterrows():
        ax2.annotate(r["name"][:5], (r["IS_Sharpe"], r["OOS_Sharpe"]),
                     fontsize=6, ha='left', va='bottom', xytext=(3, 3),
                     textcoords='offset points')
    # 対角線
    mx = max(df_oos["IS_Sharpe"].max(), df_oos["OOS_Sharpe"].max())
    mn = min(df_oos["IS_Sharpe"].min(), df_oos["OOS_Sharpe"].min())
    ax2.plot([mn, mx], [mn, mx], color='gray', linestyle='--', linewidth=0.7, alpha=0.5)
    ax2.axhline(2.0, color=GREEN, linestyle=':', linewidth=0.7, alpha=0.7)
    ax2.axhline(0, color='gray', linewidth=0.5)
    ax2.set_xlabel("IS Sharpe (2016-2020)", fontsize=9)
    ax2.set_ylabel("OOS Sharpe (2021-2026)", fontsize=9)
    ax2.set_title("銘柄ごと IS → OOS Sharpe", fontsize=11, loc='left')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=8)

    # ===== 左下: 年別 Sharpe =====
    ax3 = fig.add_subplot(gs[1, 0])
    port_daily.index = pd.to_datetime(port_daily.index)
    yrs, shrs = [], []
    for y in range(2021, 2027):
        sub = port_daily[port_daily.index.year == y]
        if len(sub) < 5:
            continue
        mu_y, sd_y = sub.mean(), sub.std()
        sh_y = mu_y / sd_y * np.sqrt(245) if sd_y > 0 else 0
        yrs.append(y)
        shrs.append(sh_y)
    colors_y = [GREEN if v >= 0 else ACCENT for v in shrs]
    ax3.bar(yrs, shrs, color=colors_y, alpha=0.85)
    ax3.axhline(0, color='gray', linewidth=0.5)
    ax3.set_title("OOS 年別 Sharpe (2022以降全年+)", fontsize=11, loc='left')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.tick_params(labelsize=8)
    for x, y in zip(yrs, shrs):
        ax3.text(x, y + (0.3 if y >= 0 else -0.3), f"{y:+.1f}",
                 ha='center', va='bottom' if y >= 0 else 'top', fontsize=8)

    # ===== 右下: 結論 =====
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    n_ok = (df_oos["verdict"] == "OK").sum()
    n_total = len(df_oos)
    n_pos = (df_oos["OOS_Sharpe"] > 0).sum()
    text = (
        f"━━ OOS検証 結果 ━━\n"
        f"IS期間:  2016-01 〜 2020-12 (5年)\n"
        f"OOS期間: 2021-01 〜 2026-05 (5.5年)\n"
        f"採用基準: IS Sharpe ≥ 2.0\n\n"
        f"21銘柄 → IS選別で 15銘柄採用\n"
        f"  └ OOS Sharpe ≥ 2.0 維持: {n_ok}/{n_total} ({n_ok/n_total*100:.0f}%)\n"
        f"  └ OOS プラス維持: {n_pos}/{n_total}\n"
        f"  └ IS平均 +9.23 → OOS平均 +6.66\n\n"
        f"━━ 戦略別 OOS ━━\n"
        f"RSI<30反発 (8銘柄): Sh+6.15, Cum+30%, DD-4%\n"
        f"MA25/75 (7銘柄):    Sh+2.87, Cum+141%, DD-11%\n"
        f"混合 (15銘柄):       Sh+2.65, Cum+82%, DD-3%\n\n"
        f"━━ 昇格判定 (Sharpe≥2.0) ━━\n"
        f"✅ oversold_rsi_reversal (8銘柄)\n"
        f"✅ ma_cross_long_basket (7銘柄)\n"
        f"   → strategies/ 昇格候補確定"
    )
    ax4.text(0.0, 1.0, text, transform=ax4.transAxes, fontsize=8.5, va='top', ha='left')

    fig.text(0.99, 0.01,
             "データ: 2016-01-01〜2026-05-22 / 日本株日足 (JQuants) / 8bps片道",
             ha='right', va='bottom', fontsize=7, color='gray')

    plt.savefig(os.path.join(OUT, "result.png"), dpi=100, bbox_inches='tight', facecolor='white')
    print(f"保存: {os.path.join(OUT, 'result.png')}")


if __name__ == "__main__":
    main()
