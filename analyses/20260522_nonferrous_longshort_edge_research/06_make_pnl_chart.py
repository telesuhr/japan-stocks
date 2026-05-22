"""
ポートフォリオ PL を X 投稿用 1200×675px グラフに整形。

レイアウト (2x2):
  左上: 累積PnL曲線 (ポートフォリオ + 個別ペア)
  右上: ドローダウン
  左下: 月次リターン棒グラフ
  右下: 統計サマリ (テキスト)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    'font.family': ['Hiragino Sans', 'Hiragino Maru Gothic Pro', 'IPAexGothic',
                    'Noto Sans CJK JP', 'sans-serif'],
    'axes.unicode_minus': False,
    'figure.facecolor': 'white',
    'axes.facecolor': '#f8f9fa',
    'grid.alpha': 0.3,
})

ACCENT = '#d4243a'      # アクセント (ポートフォリオ)
NEUTRAL = '#2e4a7d'     # ニュートラル青
GRAY = '#888888'
SUBCOLORS = ['#a0a0a0', '#b8b8b8', '#888888', '#7a7a7a', '#5a5a5a']


def main():
    df = pd.read_csv(os.path.join(OUT_DIR, "portfolio_top5_daily_pnl.csv"),
                     index_col=0, parse_dates=True)
    print(f"日次PnLデータ: {df.shape}, 列: {df.columns.tolist()}")

    pair_cols = [c for c in df.columns if c != "portfolio"]
    port = df["portfolio"]
    eq_port = port.cumsum() * 100  # % 単位

    # 統計値
    n = len(port)
    mu = port.mean()
    sd = port.std()
    sharpe = mu / sd * np.sqrt(245) if sd > 0 else 0
    eq = port.cumsum()
    dd_series = eq - eq.cummax()
    maxdd = dd_series.min() * 100
    wr = (port > 0).mean() * 100
    pf = (port[port > 0].sum() / -port[port < 0].sum()) if (port < 0).any() else np.inf
    cum_total = eq.iloc[-1] * 100

    fig = plt.figure(figsize=(12, 6.75), facecolor='white')
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.25,
                          left=0.06, right=0.97, top=0.88, bottom=0.10)

    # ===== タイトル =====
    fig.suptitle(
        "非鉄金属8銘柄 ペアトレード L/S ポートフォリオ",
        fontsize=15, fontweight='bold', y=0.965
    )
    fig.text(0.5, 0.92,
             f"Top-5ペア並列 / 8 bps片道 / Sharpe {sharpe:+.2f} / 累積 {cum_total:+.1f}% / DD {maxdd:.1f}%",
             ha='center', fontsize=10, color='#333')

    # ===== 左上: 累積PnL =====
    ax1 = fig.add_subplot(gs[0, 0])
    for i, p in enumerate(pair_cols):
        eq_p = df[p].cumsum() * 100
        ax1.plot(eq_p.index, eq_p.values, color=SUBCOLORS[i % len(SUBCOLORS)],
                 linewidth=0.9, alpha=0.7, label=p)
    ax1.plot(eq_port.index, eq_port.values, color=ACCENT, linewidth=2.0,
             label='ポートフォリオ')
    ax1.set_title("累積PnL (%, コスト後)", fontsize=11, loc='left')
    ax1.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
    ax1.legend(loc='upper left', fontsize=7, ncol=2, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%y/%m'))
    ax1.tick_params(axis='both', labelsize=8)

    # ===== 右上: ドローダウン =====
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.fill_between(dd_series.index, dd_series.values * 100, 0,
                      color=ACCENT, alpha=0.4)
    ax2.plot(dd_series.index, dd_series.values * 100, color=ACCENT, linewidth=1)
    ax2.set_title("ドローダウン (%)", fontsize=11, loc='left')
    ax2.axhline(0, color='gray', linewidth=0.5)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%y/%m'))
    ax2.tick_params(axis='both', labelsize=8)

    # ===== 左下: 月次リターン =====
    ax3 = fig.add_subplot(gs[1, 0])
    monthly = port.resample('ME').sum() * 100
    colors_m = [ACCENT if v >= 0 else NEUTRAL for v in monthly.values]
    ax3.bar(monthly.index, monthly.values, color=colors_m, width=20, alpha=0.8)
    ax3.set_title("月次リターン (%)", fontsize=11, loc='left')
    ax3.axhline(0, color='gray', linewidth=0.5)
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%y/%m'))
    ax3.tick_params(axis='both', labelsize=8)

    # ===== 右下: 統計サマリ =====
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    stats_text = (
        f"対象: 非鉄金属8銘柄 (5706/5711/5713/5714/5016 + 5801/5802/5803)\n\n"
        f"戦略     : ペアトレード Zスコア平均回帰 (Top-5並列)\n"
        f"期間     : {port.index.min():%Y-%m-%d} 〜 {port.index.max():%Y-%m-%d} ({n}営業日)\n"
        f"コスト   : 片道 8 bps (往復16bps × 2銘柄)\n\n"
        f"日次平均  : {mu*100:+.3f} %\n"
        f"日次std   : {sd*100:.2f} %\n"
        f"Sharpe   : {sharpe:+.2f}\n"
        f"勝率     : {wr:.1f} %\n"
        f"PF       : {pf:.2f}\n"
        f"累積PnL  : {cum_total:+.1f} %\n"
        f"最大DD   : {maxdd:.1f} %\n"
    )
    ax4.text(0.0, 1.0, stats_text, transform=ax4.transAxes,
             fontsize=10, va='top', ha='left')

    # フッター
    fig.text(0.99, 0.01,
             f"データ: {port.index.min():%Y-%m-%d}〜{port.index.max():%Y-%m-%d} / "
             f"日本株1分足 (JQuants) / OMEN PostgreSQL",
             ha='right', va='bottom', fontsize=7, color='gray')

    out_path = os.path.join(OUT_DIR, "result.png")
    plt.savefig(out_path, dpi=100, bbox_inches='tight', facecolor='white')
    print(f"保存: {out_path}")


if __name__ == "__main__":
    main()
