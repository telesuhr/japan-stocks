"""
10年バックテストの可視化 (1200x675px)

レイアウト:
  左上: 10年累積PnL (全7銘柄プール + 唯一プラスの住友電工)
  右上: 年別 Sharpe バー
  左下: 銘柄別 全期間 Sharpe
  右下: 統計サマリ + 教訓テキスト
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

ACCENT = '#d4243a'   # 赤 (損失)
NEUTRAL = '#2e4a7d'  # 青 (益)
GRAY = '#888888'


def main():
    # 全7銘柄プール 日次PnL
    daily = pd.read_csv(os.path.join(OUT, "all_daily_pnl.csv"))
    daily.columns = ["date", "pnl"]
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.set_index("date")["pnl"]
    eq_all = daily.cumsum() * 100

    # 住友電工単独
    trades = pd.read_csv(os.path.join(OUT, "all_trades.csv"))
    trades["date"] = pd.to_datetime(trades["date"])
    sumiden = trades[trades["code"] == 58020].copy()
    eq_sumi = sumiden.groupby("date")["pnl"].sum().sort_index().cumsum() * 100

    # 集計
    n = len(daily)
    mu = daily.mean()
    sd = daily.std()
    sharpe = mu / sd * np.sqrt(245) if sd > 0 else 0
    eq = daily.cumsum()
    dd = (eq - eq.cummax()).min()
    wr = (daily > 0).mean() * 100
    pf = (daily[daily > 0].sum() / -daily[daily < 0].sum()) if (daily < 0).any() else np.inf
    cum_total = eq.iloc[-1] * 100

    # 年別
    yearly = pd.read_csv(os.path.join(OUT, "yearly_results.csv"))
    by_symbol = pd.read_csv(os.path.join(OUT, "by_symbol_results.csv"))

    fig = plt.figure(figsize=(12, 6.75), facecolor='white')
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.25,
                          left=0.06, right=0.97, top=0.87, bottom=0.10)

    fig.suptitle("寄付ギャップフェード戦略 10年バックテスト (却下)",
                 fontsize=15, fontweight='bold', y=0.965)
    fig.text(0.5, 0.92,
             f"全7銘柄プール / 10年 / 8 bps片道 / Sharpe {sharpe:+.2f} / "
             f"累積 {cum_total:+.1f}% / DD {dd*100:.1f}%",
             ha='center', fontsize=10, color='#333')

    # ===== 左上: 累積PnL =====
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(eq_all.index, eq_all.values, color=ACCENT, linewidth=1.5,
             label=f'全7銘柄プール ({sharpe:+.2f})')
    ax1.plot(eq_sumi.index, eq_sumi.values, color=NEUTRAL, linewidth=1.5,
             label='住友電工単独 (+0.90)')
    ax1.axhline(0, color='gray', linewidth=0.5)
    ax1.set_title("累積PnL (%, コスト後, 10年)", fontsize=11, loc='left')
    ax1.legend(loc='lower left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_locator(mdates.YearLocator(2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.tick_params(labelsize=8)

    # ===== 右上: 年別 Sharpe =====
    ax2 = fig.add_subplot(gs[0, 1])
    colors_y = [NEUTRAL if v >= 0 else ACCENT for v in yearly["Sharpe"]]
    ax2.bar(yearly["year"], yearly["Sharpe"], color=colors_y, alpha=0.85)
    ax2.axhline(0, color='gray', linewidth=0.5)
    ax2.set_title("年別 Sharpe (10年, ほぼ全敗)", fontsize=11, loc='left')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(labelsize=8)
    for x, y in zip(yearly["year"], yearly["Sharpe"]):
        ax2.text(x, y + (0.1 if y >= 0 else -0.1), f"{y:+.1f}",
                 ha='center', va='bottom' if y >= 0 else 'top', fontsize=7)

    # ===== 左下: 銘柄別 Sharpe =====
    ax3 = fig.add_subplot(gs[1, 0])
    by_symbol = by_symbol.sort_values("Sharpe", ascending=False)
    colors_s = [NEUTRAL if v >= 0 else ACCENT for v in by_symbol["Sharpe"]]
    ax3.barh(range(len(by_symbol)), by_symbol["Sharpe"], color=colors_s, alpha=0.85)
    ax3.set_yticks(range(len(by_symbol)))
    ax3.set_yticklabels(by_symbol["name"], fontsize=9)
    ax3.invert_yaxis()
    ax3.axvline(0, color='gray', linewidth=0.5)
    ax3.set_title("銘柄別 Sharpe (10年全期間, 1勝6敗)", fontsize=11, loc='left')
    ax3.grid(True, alpha=0.3, axis='x')
    ax3.tick_params(labelsize=8)
    for i, s in enumerate(by_symbol["Sharpe"]):
        ax3.text(s + (0.05 if s >= 0 else -0.05), i, f"{s:+.2f}",
                 va='center', ha='left' if s >= 0 else 'right', fontsize=8)

    # ===== 右下: 統計+教訓 =====
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    text = (
        f"対象: 非鉄7銘柄 (JX金属除く=データ短い)\n"
        f"期間: 2016-05-10 〜 2026-05-22 (10年)\n"
        f"戦略: |寄付ギャップ|≥1.5% で逆張り\n"
        f"     寄付→引け1日完結, コスト8bps片道\n\n"
        f"取引数  : 3,976  (発火日 1,468日)\n"
        f"Sharpe : {sharpe:+.2f}\n"
        f"累積   : {cum_total:+.1f} %\n"
        f"最大DD : {dd*100:.1f} %\n"
        f"勝率   : {wr:.1f} %  /  PF: {pf:.2f}\n\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"教訓: 13ヶ月で見つけた Sharpe +2.49 は\n"
        f"     2025-2026 限定のレジーム現象。\n"
        f"     コスト前でも Sharpe ≒ 0 = エッジなし。\n"
        f"     直近データのみで戦略採用すべきでない。"
    )
    ax4.text(0.0, 1.0, text, transform=ax4.transAxes, fontsize=9, va='top', ha='left')

    fig.text(0.99, 0.01,
             "データ: 2016-05-10〜2026-05-22 / 日本株日足 (JQuants) / OMEN PostgreSQL",
             ha='right', va='bottom', fontsize=7, color='gray')

    out_path = os.path.join(OUT, "result.png")
    plt.savefig(out_path, dpi=100, bbox_inches='tight', facecolor='white')
    print(f"保存: {out_path}")


if __name__ == "__main__":
    main()
