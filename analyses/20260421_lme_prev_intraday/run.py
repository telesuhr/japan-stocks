"""
LME前日終値変化率 → 翌日日本株イントラデイ戦略バックテスト

戦略:
- シグナル: Day N の LME銅(CMCU3) close_price の前日比変化率
- エントリー: Day N+1 の日本株 9:00 寄付（始値）
- 決済: 9:30, 10:00, 11:30, 13:00, 15:30 の各時点の終値
- 方向: Long（LME上昇→買い）/ Short（LME下落→売り）両方
- コスト: 往復4bps
"""

import sys
import warnings
import numpy as np
import pandas as pd
import psycopg2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from datetime import date, timedelta
from pathlib import Path

warnings.filterwarnings('ignore')

# ─── 設定 ────────────────────────────────────────────────────────────────────

PG_LME = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "lme_copper_analytics"}
PG_MARKET = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}

STOCKS = {
    "5711.T": "三菱マテリアル",
    "6501.T": "日立",
    "7011.T": "三菱重工",
    "5016.T": "出光",
    "4502.T": "武田",
    "5713.T": "住友金属鉱山",
    "5706.T": "三井金属",
    "6963.T": "ローム",
    "4063.T": "信越化学",
}

# コア非鉄金属5銘柄
CORE_SYMBOLS = ["5713.T", "5711.T", "5706.T", "6501.T", "7011.T"]

THRESHOLDS = [0.5, 1.0, 1.5, 2.0]  # LME変化率の絶対値閾値（%）
EXIT_TIMES = [
    (9, 30, "09:30"),
    (10, 0, "10:00"),
    (11, 30, "11:30"),
    (13, 0, "13:00"),
    (15, 30, "15:30"),
]
COST_BPS = 4  # 往復コスト（bps）

OUT_DIR = Path(__file__).parent

# ─── データロード ──────────────────────────────────────────────────────────────

def load_lme_data() -> pd.Series:
    """LME銅 (CMCU3) の日次終値をロード。インデックスはdate型"""
    conn = psycopg2.connect(**PG_LME)
    sql = """
        SELECT p.trade_date::date AS trade_date,
               COALESCE(p.settlement_price, p.close_price)::float AS price
        FROM market_data.price_data p
        JOIN reference.ric_codes r ON p.ric_id = r.ric_id
        WHERE r.ric_code = 'CMCU3'
        ORDER BY p.trade_date
    """
    df = pd.read_sql(sql, conn)
    conn.close()
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    df = df.dropna(subset=["price"]).set_index("trade_date")["price"]
    return df


def load_stock_data(symbol: str) -> pd.DataFrame:
    """1分足データをロード。インデックスはJST"""
    conn = psycopg2.connect(**PG_MARKET)
    sql = """
        SELECT timestamp, open::float, close::float
        FROM intraday_data
        WHERE symbol = %s
        ORDER BY timestamp
    """
    df = pd.read_sql(sql, conn, params=(symbol,))
    conn.close()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["jst"] = df["timestamp"] + pd.Timedelta(hours=9)
    df = df.dropna(subset=["open", "close"]).set_index("jst").sort_index()
    return df[["open", "close"]]


# ─── 日次サマリー作成 ─────────────────────────────────────────────────────────

def make_daily_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    各営業日について:
    - open_price: 9:00 始値（なければ9時台の最初のbar）
    - exit_{label}: 各決済時刻の終値（なければその時刻より前の最後のbar）
    """
    df["date"] = df.index.date

    records = []
    for dt, g in df.groupby("date"):
        g = g.sort_index()

        # エントリー: 9:00 の open
        g_9 = g[(g.index.hour == 9) & (g.index.minute == 0)]
        if not g_9.empty:
            entry_price = float(g_9["open"].iloc[0])
        else:
            g_9x = g[g.index.hour == 9]
            if g_9x.empty:
                continue
            entry_price = float(g_9x["open"].iloc[0])

        row = {"date": dt, "open_price": entry_price}

        # 各決済時刻の終値
        for h, m, label in EXIT_TIMES:
            # exit_time 以前の最後のbarのclose
            g_before = g[(g.index.hour < h) | ((g.index.hour == h) & (g.index.minute <= m))]
            if g_before.empty:
                row[f"exit_{label}"] = np.nan
            else:
                row[f"exit_{label}"] = float(g_before["close"].iloc[-1])

        records.append(row)

    summary = pd.DataFrame(records)
    if summary.empty:
        return summary
    summary["date"] = pd.to_datetime(summary["date"]).dt.date
    return summary.set_index("date")


# ─── LMEシグナル計算 ──────────────────────────────────────────────────────────

def compute_lme_signal(lme: pd.Series) -> pd.Series:
    """LME前日比変化率（%）を計算"""
    ret = lme.pct_change() * 100
    return ret


def get_prev_lme_ret(trade_date: date, lme_ret: pd.Series) -> float | None:
    """trade_date より前の最後のLME変化率を返す"""
    idx = lme_ret.index
    prev = idx[idx < trade_date]
    if len(prev) == 0:
        return None
    return float(lme_ret.loc[prev[-1]])


# ─── バックテスト ─────────────────────────────────────────────────────────────

def backtest_single(
    daily: pd.DataFrame,
    lme_ret: pd.Series,
    threshold: float,
    direction: str,
    exit_label: str,
) -> pd.Series:
    """
    1銘柄×1パラメータセットのトレードリストを返す。

    direction: 'long' (LME上昇 → Long) / 'short' (LME上昇 → Short)
    """
    exit_col = f"exit_{exit_label}"
    trades = []

    for trade_date, row in daily.iterrows():
        sig = get_prev_lme_ret(trade_date, lme_ret)
        if sig is None or np.isnan(sig):
            continue

        # シグナルが閾値以上か？
        if abs(sig) < threshold:
            continue

        # 方向を決定
        if direction == "long":
            pos = 1 if sig > 0 else -1
        else:  # short
            pos = -1 if sig > 0 else 1

        entry = row["open_price"]
        ex = row.get(exit_col, np.nan)
        if np.isnan(entry) or np.isnan(ex):
            continue

        gross_ret = (ex / entry - 1) * 100 * pos
        net_ret = gross_ret - COST_BPS * 0.01  # bps → %

        trades.append({"date": trade_date, "signal": sig, "pos": pos,
                        "entry": entry, "exit": ex, "gross_ret": gross_ret, "net_ret": net_ret})

    return pd.DataFrame(trades)


def evaluate(trades: pd.DataFrame) -> dict:
    """バックテスト評価指標を計算"""
    if trades.empty or len(trades) < 5:
        return {"n": 0, "mean": np.nan, "wr": np.nan, "pf": np.nan, "sharpe": np.nan}

    arr = trades["net_ret"].values
    n = len(arr)
    mean_ret = arr.mean()
    wr = (arr > 0).mean() * 100

    pos_sum = arr[arr > 0].sum()
    neg_sum = abs(arr[arr <= 0].sum())
    pf = pos_sum / neg_sum if neg_sum > 0 else np.nan

    std = arr.std()
    sharpe = mean_ret / std * np.sqrt(252) if std > 0 else np.nan

    return {"n": n, "mean": mean_ret, "wr": wr, "pf": pf, "sharpe": sharpe}


# ─── メイン ──────────────────────────────────────────────────────────────────

def main():
    print("=== LME前日終値変化率 → 翌日イントラデイ戦略バックテスト ===\n")

    # LMEデータロード
    print("LMEデータロード中...")
    lme = load_lme_data()
    lme_ret = compute_lme_signal(lme)
    print(f"  LMEデータ: {lme.index[0]} ～ {lme.index[-1]}  ({len(lme)} 日)\n")

    # 全銘柄の日次サマリーを作成
    print("株式1分足データロード＆日次サマリー作成中...")
    summaries = {}
    for sym, name in STOCKS.items():
        print(f"  {sym} {name} ...", end=" ")
        df = load_stock_data(sym)
        daily = make_daily_summary(df)
        summaries[sym] = daily
        print(f"{len(daily)} 日")

    print()

    # グリッドサーチ
    print("グリッドサーチ実行中...")
    results = []
    all_trades = {}  # (sym, threshold, direction, exit_label) -> trades DataFrame

    for sym, name in STOCKS.items():
        daily = summaries[sym]
        for thr in THRESHOLDS:
            for direction in ["long", "short"]:
                for _, _, exit_label in EXIT_TIMES:
                    trades = backtest_single(daily, lme_ret, thr, direction, exit_label)
                    metrics = evaluate(trades)
                    row = {
                        "symbol": sym, "name": name,
                        "threshold": thr, "direction": direction, "exit": exit_label,
                        **metrics,
                    }
                    results.append(row)
                    all_trades[(sym, thr, direction, exit_label)] = trades

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUT_DIR / "results_grid.csv", index=False)
    print(f"  results_grid.csv 保存 ({len(results_df)} 行)\n")

    # ─── バスケット結果 ──────────────────────────────────────────────────────────
    print("コア5銘柄バスケット計算中...")
    basket_records = []

    for thr in THRESHOLDS:
        for direction in ["long", "short"]:
            for _, _, exit_label in EXIT_TIMES:
                # 各日のバスケットリターン（等加重平均）
                daily_baskets = {}
                for sym in CORE_SYMBOLS:
                    trades = all_trades.get((sym, thr, direction, exit_label), pd.DataFrame())
                    if trades.empty:
                        continue
                    for _, tr in trades.iterrows():
                        d = tr["date"]
                        if d not in daily_baskets:
                            daily_baskets[d] = []
                        daily_baskets[d].append(tr["net_ret"])

                basket_rets = {d: np.mean(v) for d, v in daily_baskets.items() if v}
                if not basket_rets:
                    continue

                arr = np.array(list(basket_rets.values()))
                n = len(arr)
                mean_ret = arr.mean()
                wr = (arr > 0).mean() * 100
                pos_sum = arr[arr > 0].sum()
                neg_sum = abs(arr[arr <= 0].sum())
                pf = pos_sum / neg_sum if neg_sum > 0 else np.nan
                std = arr.std()
                sharpe = mean_ret / std * np.sqrt(252) if std > 0 else np.nan

                basket_records.append({
                    "threshold": thr, "direction": direction, "exit": exit_label,
                    "n": n, "mean": mean_ret, "wr": wr, "pf": pf, "sharpe": sharpe,
                })

    basket_df = pd.DataFrame(basket_records)
    basket_df.to_csv(OUT_DIR / "basket_results.csv", index=False)
    print(f"  basket_results.csv 保存 ({len(basket_df)} 行)\n")

    # ─── 結果サマリー表示 ────────────────────────────────────────────────────────
    print("=== トップ戦略（個別銘柄、Sharpe上位10） ===")
    top = results_df.dropna(subset=["sharpe"]).sort_values("sharpe", ascending=False).head(10)
    print(top[["symbol", "name", "threshold", "direction", "exit", "n", "mean", "wr", "pf", "sharpe"]].to_string(index=False))

    print("\n=== バスケット トップ戦略（Sharpe上位10） ===")
    top_basket = basket_df.dropna(subset=["sharpe"]).sort_values("sharpe", ascending=False).head(10)
    print(top_basket.to_string(index=False))

    # ─── 可視化 ─────────────────────────────────────────────────────────────────
    print("\n可視化中...")

    # fig1: Sharpe ヒートマップ（銘柄 × (exit × threshold)）
    # Long + Short 方向ごとに表示
    for direction in ["long", "short"]:
        df_dir = results_df[results_df["direction"] == direction].copy()
        df_dir["param"] = df_dir["threshold"].astype(str) + "% / " + df_dir["exit"]
        pivot = df_dir.pivot_table(index="symbol", columns="param", values="sharpe", aggfunc="first")

        fig, ax = plt.subplots(figsize=(16, 6))
        sns.heatmap(pivot, annot=True, fmt=".2f", center=0, cmap="RdYlGn",
                    linewidths=0.5, ax=ax, cbar_kws={"label": "Sharpe"})
        ax.set_title(f"Sharpe ヒートマップ ({direction.upper()})  ─  LME前日変化率 → 翌日イントラデイ",
                     fontsize=12, pad=10)
        ax.set_xlabel("threshold / exit time")
        ax.set_ylabel("symbol")
        plt.xticks(rotation=45, ha="right", fontsize=7)
        plt.tight_layout()
        fig.savefig(OUT_DIR / f"fig1_sharpe_heatmap_{direction}.png", dpi=120)
        plt.close(fig)
        print(f"  fig1_sharpe_heatmap_{direction}.png 保存")

    # fig2: 上位5戦略のエクイティカーブ
    top5 = results_df.dropna(subset=["sharpe"]).sort_values("sharpe", ascending=False).head(5)
    fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=False)
    for i, (_, row) in enumerate(top5.iterrows()):
        sym = row["symbol"]
        thr = row["threshold"]
        direction = row["direction"]
        exit_label = row["exit"]
        trades = all_trades.get((sym, thr, direction, exit_label), pd.DataFrame())
        if trades.empty:
            continue
        trades = trades.sort_values("date")
        cum = trades["net_ret"].cumsum()
        ax = axes[i]
        ax.plot(range(len(cum)), cum.values, linewidth=1.5)
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.fill_between(range(len(cum)), 0, cum.values, alpha=0.2)
        ax.set_title(
            f"{sym} ({row['name']})  th={thr}%  {direction}  exit={exit_label}  "
            f"Sharpe={row['sharpe']:.2f}  n={int(row['n'])}  mean={row['mean']:.3f}%",
            fontsize=9,
        )
        ax.set_ylabel("累積リターン (%)")
        ax.grid(alpha=0.3)
    fig.suptitle("上位5戦略 エクイティカーブ", fontsize=12)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig2_equity_curves.png", dpi=120)
    plt.close(fig)
    print("  fig2_equity_curves.png 保存")

    # fig3: LME前日変化率 vs 翌日リターン（コア5バスケット）散布図
    fig, axes = plt.subplots(1, len(EXIT_TIMES), figsize=(18, 5))
    for ax, (h, m, exit_label) in zip(axes, EXIT_TIMES):
        all_x = []
        all_y = []
        for sym in CORE_SYMBOLS:
            daily = summaries[sym]
            exit_col = f"exit_{exit_label}"
            for trade_date, row_d in daily.iterrows():
                sig = get_prev_lme_ret(trade_date, lme_ret)
                if sig is None or np.isnan(sig):
                    continue
                entry = row_d["open_price"]
                ex = row_d.get(exit_col, np.nan)
                if np.isnan(entry) or np.isnan(ex):
                    continue
                ret = (ex / entry - 1) * 100
                all_x.append(sig)
                all_y.append(ret)

        x = np.array(all_x)
        y = np.array(all_y)
        ax.scatter(x, y, alpha=0.2, s=8, color="steelblue")

        if len(x) > 10:
            # 回帰直線
            coef = np.polyfit(x, y, 1)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, np.polyval(coef, x_line), "r-", linewidth=1.5)
            corr = np.corrcoef(x, y)[0, 1]
            ax.set_title(f"exit={exit_label}\nCorr={corr:.3f}  slope={coef[0]:.4f}", fontsize=9)
        else:
            ax.set_title(f"exit={exit_label}", fontsize=9)

        ax.axhline(0, color="black", linewidth=0.5)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_xlabel("LME前日変化率 (%)")
        ax.set_ylabel("翌日リターン (%)")
        ax.grid(alpha=0.3)

    fig.suptitle("LME前日変化率 vs 翌日寄付～各時点リターン（コア5バスケット等加重）", fontsize=11)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig3_scatter.png", dpi=120)
    plt.close(fig)
    print("  fig3_scatter.png 保存")

    print("\n=== 完了 ===")

    # ─── README用サマリー生成 ────────────────────────────────────────────────────
    return results_df, basket_df, all_trades, summaries, lme_ret


if __name__ == "__main__":
    results_df, basket_df, all_trades, summaries, lme_ret = main()
