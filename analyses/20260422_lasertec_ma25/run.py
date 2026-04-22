"""
レーザーテック (6920.T) MA25 サポート狙撃 — 深掘り

検証項目:
1. OoS (H1/H2 分割) — エッジ維持?
2. パラメータ頑健性 (MA期間, 許容tol, DD閾値, 保有期間)
3. コスト込み実効 Sharpe
4. トレード詳細リスト (再現性確認)
5. vwap_morning_meanrevert との同日重複分析
"""
import psycopg2, pandas as pd, numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}
OUT = Path(__file__).parent
SYM = "6920.T"
COST_PCT = 0.04  # 往復 4bps = 0.04%


def load_daily(sym):
    conn = psycopg2.connect(**PG)
    df = pd.read_sql(
        "SELECT trade_date, open, high, low, close, volume FROM daily_stats "
        "WHERE symbol=%s ORDER BY trade_date", conn, params=(sym,))
    conn.close()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.set_index("trade_date").sort_index()
    df = df.astype({c: float for c in ["open", "high", "low", "close"]})
    return df


def annotate(df, mas, hhn=20):
    for n in mas:
        df[f"ma{n}"] = df["close"].rolling(n).mean()
        df[f"ma{n}_slope"] = df[f"ma{n}"].diff(5)
    df[f"hh{hhn}"] = df["close"].rolling(hhn).max()
    df[f"dd{hhn}"] = (df["close"] / df[f"hh{hhn}"] - 1) * 100
    return df


def find_events(df, ma_col, tol_pct, dd_thresh, dd_col, slope_col=None, slope_up_only=False):
    ma = df[ma_col]
    lo, hi = df["low"], df["high"]
    lower = ma * (1 - tol_pct / 100)
    upper = ma * (1 + tol_pct / 100)
    touched = (lo <= upper) & (hi >= lower)
    downtrend = df[dd_col] <= -dd_thresh
    event = touched & downtrend & df[ma_col].notna()
    if slope_up_only and slope_col is not None:
        event = event & (df[slope_col] > 0)
    return event


def forward_returns(df, event_mask, fwd_days):
    """エントリー = 当日close、Exit = N日後close。コスト 0.04% (往復) 控除"""
    rows = []
    for i, (dt, row) in enumerate(df[event_mask].iterrows()):
        entry = row["close"]
        dates = df.index
        idx = dates.get_loc(dt)
        out = {"entry_date": dt, "entry": entry, "ma_touch": row.name}
        for d in fwd_days:
            if idx + d >= len(dates):
                out[f"fwd{d}_pct"] = np.nan
                out[f"fwd{d}_exit"] = np.nan
                continue
            exit_p = df["close"].iloc[idx + d]
            ret_pct = (exit_p / entry - 1) * 100 - COST_PCT
            out[f"fwd{d}_pct"] = ret_pct
            out[f"fwd{d}_exit"] = exit_p
        rows.append(out)
    return pd.DataFrame(rows)


def stats(arr):
    arr = np.asarray(arr); arr = arr[~np.isnan(arr)]
    if len(arr) < 3:
        return dict(n=len(arr), mean=np.nan, std=np.nan, wr=np.nan, tstat=np.nan, sharpe=np.nan)
    m = arr.mean(); s = arr.std(ddof=1)
    # 日次換算は保有日数で変わるので fwd日数で割って per-day化
    return dict(n=len(arr), mean=m, std=s, wr=(arr > 0).mean() * 100,
                tstat=(m / (s / np.sqrt(len(arr)))) if s > 0 else np.nan,
                sharpe=(m / s * np.sqrt(252)) if s > 0 else np.nan)


def split_oos(trades, col):
    trades = trades.sort_values("entry_date").reset_index(drop=True)
    mid = len(trades) // 2
    return stats(trades[col].values), stats(trades.iloc[:mid][col].values), stats(trades.iloc[mid:][col].values)


def main():
    print("=" * 100)
    print(f"レーザーテック (6920.T) MA25 サポート狙撃 — 深掘り")
    print("=" * 100)
    df = load_daily(SYM)
    print(f"Daily: {df.index.min().date()} ~ {df.index.max().date()}  N={len(df)}")

    # ベース設定
    MAS = [10, 15, 20, 25, 30, 40, 50, 75, 100]
    df = annotate(df, MAS)

    # =========================================================================
    # Part 1: ベース設定 (MA25, tol=1.0, dd=-5%, slope=up/all) の OoS
    # =========================================================================
    print("\n[Part 1] ベース設定の OoS (MA25, tol=1.0%, dd20≤-5%)")
    print("-" * 100)
    for slope_filter in ["all", "up"]:
        evt = find_events(df, "ma25", 1.0, 5.0, "dd20",
                          slope_col="ma25_slope", slope_up_only=(slope_filter == "up"))
        trades = forward_returns(df, evt, [1, 3, 5, 10, 20])
        print(f"\nslope={slope_filter}  N={len(trades)}")
        for d in [1, 3, 5, 10, 20]:
            col = f"fwd{d}_pct"
            full, h1, h2 = split_oos(trades, col)
            print(f"  fwd{d:>2}: N={full['n']:3d}  "
                  f"Full mean={full['mean']:+.2f}% WR={full['wr']:.1f}% t={full['tstat']:+.2f}  "
                  f"H1 mean={h1['mean']:+.2f}% t={h1['tstat']:+.2f}  "
                  f"H2 mean={h2['mean']:+.2f}% t={h2['tstat']:+.2f}")
        if slope_filter == "up":
            trades.to_csv(OUT / "trades_ma25_up.csv", index=False)
        else:
            trades.to_csv(OUT / "trades_ma25_all.csv", index=False)

    # =========================================================================
    # Part 2: パラメータ頑健性
    # =========================================================================
    print("\n[Part 2] パラメータ頑健性  (slope=up fwd=5)")
    print("-" * 100)
    rows = []
    for ma in [15, 20, 25, 30, 40, 50]:
        for tol in [0.5, 1.0, 1.5, 2.0]:
            for dd in [3.0, 5.0, 7.0, 10.0]:
                evt = find_events(df, f"ma{ma}", tol, dd, "dd20",
                                  slope_col=f"ma{ma}_slope", slope_up_only=True)
                t = forward_returns(df, evt, [5])
                if len(t) < 10: continue
                full, h1, h2 = split_oos(t, "fwd5_pct")
                rows.append({
                    "ma": ma, "tol%": tol, "dd%": dd, "n": full["n"],
                    "mean%": full["mean"], "wr%": full["wr"], "t": full["tstat"],
                    "h1_mean%": h1["mean"], "h1_t": h1["tstat"],
                    "h2_mean%": h2["mean"], "h2_t": h2["tstat"],
                })
    grid = pd.DataFrame(rows).sort_values("t", ascending=False)
    print(grid.head(20).to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    grid.to_csv(OUT / "grid.csv", index=False)

    # =========================================================================
    # Part 3: 保有期間スイープ (ベース設定で)
    # =========================================================================
    print("\n[Part 3] 保有期間スイープ (MA25, tol=1.0, dd=-5%, slope=up)")
    print("-" * 100)
    evt = find_events(df, "ma25", 1.0, 5.0, "dd20",
                      slope_col="ma25_slope", slope_up_only=True)
    # 拡張して fwd1-20日
    trades_ext = forward_returns(df, evt, list(range(1, 21)))
    rows = []
    for d in range(1, 21):
        col = f"fwd{d}_pct"
        full, h1, h2 = split_oos(trades_ext, col)
        rows.append({"days": d, "n": full["n"], "mean%": full["mean"], "wr%": full["wr"],
                     "t": full["tstat"], "sharpe": full["sharpe"],
                     "h1_mean%": h1["mean"], "h2_mean%": h2["mean"],
                     "h1_t": h1["tstat"], "h2_t": h2["tstat"]})
    hold = pd.DataFrame(rows)
    print(hold.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    hold.to_csv(OUT / "holding_period.csv", index=False)

    # =========================================================================
    # Part 4: トレード詳細
    # =========================================================================
    print("\n[Part 4] 全トレード詳細 (slope=up, fwd5)")
    print("-" * 100)
    trades = pd.read_csv(OUT / "trades_ma25_up.csv")
    trades["entry_date"] = pd.to_datetime(trades["entry_date"])
    trades["month"] = trades["entry_date"].dt.to_period("M")
    print(trades[["entry_date", "entry", "fwd1_pct", "fwd5_pct", "fwd10_pct", "fwd20_pct"]]
          .to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    # 月別
    print("\n月別の集計 (fwd5):")
    monthly = trades.groupby("month")["fwd5_pct"].agg(["count", "mean", lambda s: (s > 0).mean() * 100])
    monthly.columns = ["n", "mean%", "wr%"]
    print(monthly.to_string(float_format=lambda x: f"{x:.2f}"))

    # =========================================================================
    # Part 5: Plot
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 価格+MA25+event
    ax = axes[0, 0]
    ax.plot(df.index, df["close"], "-", lw=1, label="Close")
    ax.plot(df.index, df["ma25"], "--", lw=1, label="MA25", color="orange")
    # 下降局面 shading
    dd_mask = df["dd20"] <= -5
    ax.fill_between(df.index, df["close"].min(), df["close"].max(),
                    where=dd_mask, alpha=0.08, color="red")
    # events
    ev_dates = trades["entry_date"]
    ev_prices = trades["entry"]
    ax.scatter(ev_dates, ev_prices, c="blue", marker="^", s=40, label=f"Entry (N={len(trades)})", zorder=5)
    ax.set_title("6920.T Close + MA25 + MA25 サポート接触 (下落局面)")
    ax.set_ylabel("Price"); ax.legend(); ax.grid(alpha=0.3)

    # fwd5 histogram
    ax = axes[0, 1]
    ax.hist(trades["fwd5_pct"].dropna(), bins=25, color="teal", alpha=0.7)
    ax.axvline(0, color="k"); ax.axvline(trades["fwd5_pct"].mean(), color="red", label=f"Mean {trades['fwd5_pct'].mean():.2f}%")
    ax.set_title("5日後リターン分布")
    ax.set_xlabel("fwd5 %"); ax.legend(); ax.grid(alpha=0.3)

    # Holding period
    ax = axes[1, 0]
    ax.plot(hold["days"], hold["mean%"], "-o", label="Full")
    ax.plot(hold["days"], hold["h1_mean%"], "--s", label="H1", alpha=0.6)
    ax.plot(hold["days"], hold["h2_mean%"], "--^", label="H2", alpha=0.6)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title("保有期間 vs 平均リターン")
    ax.set_xlabel("保有日数"); ax.set_ylabel("%"); ax.legend(); ax.grid(alpha=0.3)

    # t-stat vs holding
    ax = axes[1, 1]
    ax.plot(hold["days"], hold["t"], "-o", label="Full t-stat")
    ax.axhline(2, color="g", ls="--", label="t=2.0")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title("保有期間 vs t-stat")
    ax.set_xlabel("保有日数"); ax.set_ylabel("t"); ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT / "result.png", dpi=100)
    print(f"\nSaved plot: {OUT/'result.png'}")
    print("\nDONE")


if __name__ == "__main__":
    main()
