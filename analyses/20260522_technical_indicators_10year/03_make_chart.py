"""
テクニカル戦略 10年バックテスト PL画像 (1200x675px)

レイアウト:
  左上: 累積PnL (Top3 vs Top5 vs Top7 vs Buy&Hold平均)
  右上: 戦略別 平均Sharpe (バー)
  左下: 年別 Sharpe (Top3) - 弱点も見える化
  右下: 統計サマリ
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

SYMBOLS = {
    "57060": "三井金属", "57110": "三菱マテリアル", "57130": "住友金属鉱山",
    "57140": "DOWA HD",
    "58010": "古河電工", "58020": "住友電工", "58030": "フジクラ",
}


def build_portfolio_pnl(codes, K, start_year=None):
    """各銘柄のtrades CSVから等ウェイトポートフォリオの日次PnLを構築"""
    daily_list = []
    for c in codes:
        path = os.path.join(OUT, f"trades_MA2575_{c}.csv")
        t = pd.read_csv(path)
        if len(t) == 0:
            continue
        t["exit_date"] = pd.to_datetime(t["exit_date"])
        if start_year:
            t = t[t["exit_date"].dt.year >= start_year]
        sub = t.groupby("exit_date")["pnl"].sum() / K
        daily_list.append(sub)
    port = pd.concat(daily_list).groupby(level=0).sum().sort_index()
    return port


def main():
    # 戦略別 平均Sharpe (スクリーニング結果)
    strat_summary = pd.read_csv(os.path.join(OUT, "strategy_summary.csv"))
    bh = pd.read_csv(os.path.join(OUT, "buyhold_benchmark.csv"))

    # Top3 ポートフォリオ (Sharpe順)
    top3_codes = ["57060", "57130", "58010"]  # 三井金属, 住友金属鉱山, 古河電工
    top5_codes = top3_codes + ["57110", "58020"]  # +三菱マテ, +住友電工
    top7_codes = list(SYMBOLS.keys())

    port3 = build_portfolio_pnl(top3_codes, K=3)
    port5 = build_portfolio_pnl(top5_codes, K=5)
    port7 = build_portfolio_pnl(top7_codes, K=7)

    # Buy&Hold 平均 (全7銘柄等ウェイト)
    # 各銘柄の日次logリターンを取得して7銘柄平均
    # 簡易: stocks_daily から再取得
    import psycopg2
    PG = {"host": os.environ.get("PGHOST", "omen"), "port": 5432,
          "user": "postgres", "dbname": "market_data"}
    bh_daily_list = []
    for c in top7_codes:
        conn = psycopg2.connect(**PG)
        sql = "SELECT date, close FROM stocks_daily WHERE code=%s AND date BETWEEN '2016-05-10' AND '2026-05-22' ORDER BY date"
        df = pd.read_sql(sql, conn, params=(c,))
        conn.close()
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        r = np.log(df["close"] / df["close"].shift(1)).dropna() / 7
        bh_daily_list.append(r)
    bh_port = pd.concat(bh_daily_list).groupby(level=0).sum()

    # 各ポートフォリオ集計
    def stats(port):
        mu, sd = port.mean(), port.std()
        sh = mu / sd * np.sqrt(245) if sd > 0 else 0
        eq = port.cumsum()
        dd = (eq - eq.cummax()).min()
        return sh, eq, dd

    sh3, eq3, dd3 = stats(port3)
    sh5, eq5, dd5 = stats(port5)
    sh7, eq7, dd7 = stats(port7)
    shB, eqB, ddB = stats(bh_port)

    fig = plt.figure(figsize=(12, 6.75), facecolor='white')
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.25,
                          left=0.06, right=0.97, top=0.87, bottom=0.10)

    fig.suptitle("テクニカル指標 10年バックテスト ・ MA25/75順張りが本命",
                 fontsize=15, fontweight='bold', y=0.965)
    fig.text(0.5, 0.92,
             f"Top3銘柄プール / MA25/75 GC / 8bps / "
             f"Sharpe {sh3:+.2f} / 累積 {eq3.iloc[-1]*100:+.1f}% / DD {dd3*100:.1f}%",
             ha='center', fontsize=10, color='#333')

    # ===== 左上: 累積PnL =====
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(eqB.index, eqB.values * 100, color=GRAY, linewidth=1.2, alpha=0.8,
             label=f'Buy&Hold 平均 ({shB:+.2f})')
    ax1.plot(eq7.index, eq7.values * 100, color=NEUTRAL, linewidth=1.2, alpha=0.7,
             label=f'全7銘柄 ({sh7:+.2f})')
    ax1.plot(eq5.index, eq5.values * 100, color=GREEN, linewidth=1.4,
             label=f'Top5 ({sh5:+.2f})')
    ax1.plot(eq3.index, eq3.values * 100, color=ACCENT, linewidth=2.0,
             label=f'Top3 ({sh3:+.2f})')
    ax1.axhline(0, color='gray', linewidth=0.5)
    ax1.set_title("累積PnL (%, コスト後, 10年)", fontsize=11, loc='left')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_locator(mdates.YearLocator(2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.tick_params(labelsize=8)

    # ===== 右上: 戦略別平均Sharpe =====
    ax2 = fig.add_subplot(gs[0, 1])
    strat_summary = strat_summary.sort_values("avg_Sharpe", ascending=True)
    name_map = {
        "S1_RSI_long": "RSI<30 反発ロング",
        "S2_RSI_short": "RSI>70 ショート",
        "S3_MACD_long": "MACD GCロング",
        "S4_MA25_75": "MA25/75 順張り",
        "S5_BB_long": "BB(20,2σ) 逆張り",
    }
    labels = [name_map.get(s, s) for s in strat_summary["strat"]]
    sharpes = strat_summary["avg_Sharpe"].values
    colors_b = [NEUTRAL if v >= 0 else ACCENT for v in sharpes]
    ax2.barh(range(len(labels)), sharpes, color=colors_b, alpha=0.85)
    ax2.set_yticks(range(len(labels)))
    ax2.set_yticklabels(labels, fontsize=9)
    ax2.axvline(0, color='gray', linewidth=0.5)
    ax2.axvline(0.51, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    ax2.text(0.51, len(labels)-0.5, ' BH=0.51', fontsize=7, color='gray', va='top')
    ax2.set_title("戦略別 平均Sharpe (全7銘柄, 10年)", fontsize=11, loc='left')
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.tick_params(labelsize=8)
    for i, s in enumerate(sharpes):
        ax2.text(s + (0.1 if s >= 0 else -0.1), i, f"{s:+.2f}",
                 va='center', ha='left' if s >= 0 else 'right', fontsize=8)

    # ===== 左下: 年別Sharpe (Top3) =====
    ax3 = fig.add_subplot(gs[1, 0])
    yearly_sh = []
    yearly_lab = []
    for y in range(2016, 2027):
        sub = port3[port3.index.year == y]
        if len(sub) < 2:
            continue
        mu, sd = sub.mean(), sub.std()
        sh = mu / sd * np.sqrt(245) if sd > 0 else 0
        yearly_sh.append(sh)
        yearly_lab.append(y)
    colors_y = [NEUTRAL if s >= 0 else ACCENT for s in yearly_sh]
    ax3.bar(yearly_lab, yearly_sh, color=colors_y, alpha=0.85)
    ax3.axhline(0, color='gray', linewidth=0.5)
    ax3.set_title("Top3 年別 Sharpe (レンジ相場が弱点)", fontsize=11, loc='left')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.tick_params(labelsize=8)
    for x, y in zip(yearly_lab, yearly_sh):
        ax3.text(x, y + (0.3 if y >= 0 else -0.3), f"{y:+.1f}",
                 ha='center', va='bottom' if y >= 0 else 'top', fontsize=7)

    # ===== 右下: 統計サマリ =====
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    text = (
        f"対象: 非鉄7銘柄 (10年, 2016-05〜2026-05)\n"
        f"戦略: MA25 > MA75 ゴールデンクロス順張り\n"
        f"     翌寄付エントリ・DC翌寄付クローズ\n"
        f"     コスト 8bps 片道\n\n"
        f"━━ Top3 ポートフォリオ (本命) ━━\n"
        f"銘柄  : 三井金属, 住友金属鉱山, 古河電工\n"
        f"N取引 : {sum(pd.read_csv(os.path.join(OUT,f'trades_MA2575_{c}.csv')).shape[0] for c in top3_codes)}\n"
        f"Sharpe: {sh3:+.2f}\n"
        f"累積  : {eq3.iloc[-1]*100:+.1f} %\n"
        f"最大DD: {dd3*100:.1f} %\n"
        f"BH対比: +{eq3.iloc[-1]*100 - eqB.iloc[-1]*100:.0f}% (BH={eqB.iloc[-1]*100:+.0f}%)\n\n"
        f"コスト耐性 (50bps片道でも Sharpe +2.44)\n"
        f"IS/OOS両方プラス: OOS Sharpe +4.09\n"
        f"弱点: 2019-2023 のレンジ相場で連敗"
    )
    ax4.text(0.0, 1.0, text, transform=ax4.transAxes, fontsize=9, va='top', ha='left')

    fig.text(0.99, 0.01,
             "データ: 2016-05-10〜2026-05-22 / 日本株日足 (JQuants) / OMEN PostgreSQL",
             ha='right', va='bottom', fontsize=7, color='gray')

    plt.savefig(os.path.join(OUT, "result.png"), dpi=100, bbox_inches='tight', facecolor='white')
    print(f"保存: {os.path.join(OUT, 'result.png')}")


if __name__ == "__main__":
    main()
