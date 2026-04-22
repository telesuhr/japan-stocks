"""
レーザーテック MA25 サポート — ストップロス設計検証

検証:
1. Stop level: -3/-5/-7/-10/-15%/None
2. Stop判定: 日中 Low ベース / 終値 close ベース
3. 保有期間: 10営業日固定 (ベストSharpe)
4. 出力: Sharpe, WR, max loss, avg loss, avg win, PF, stop hit率
5. H1/H2 のロバスト性
"""
import psycopg2, pandas as pd, numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}
OUT = Path(__file__).parent
SYM = "6920.T"
COST_PCT = 0.04
HOLD = 10  # ベスト設定


def load_daily(sym):
    conn = psycopg2.connect(**PG)
    df = pd.read_sql(
        "SELECT trade_date, open, high, low, close FROM daily_stats "
        "WHERE symbol=%s ORDER BY trade_date", conn, params=(sym,))
    conn.close()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.set_index("trade_date").sort_index()
    df = df.astype({c: float for c in ["open", "high", "low", "close"]})
    return df


def find_entries(df, ma=25, tol_pct=1.0, dd_thresh=5.0, slope_up=True):
    df = df.copy()
    df["ma"] = df["close"].rolling(ma).mean()
    df["slope"] = df["ma"].diff(5)
    df["hh20"] = df["close"].rolling(20).max()
    df["dd20"] = (df["close"] / df["hh20"] - 1) * 100
    lo, hi = df["low"], df["high"]
    ma_s = df["ma"]
    touched = (lo <= ma_s * (1 + tol_pct / 100)) & (hi >= ma_s * (1 - tol_pct / 100))
    ev = touched & (df["dd20"] <= -dd_thresh) & df["ma"].notna()
    if slope_up:
        ev = ev & (df["slope"] > 0)
    df["event"] = ev
    return df


def simulate(df, hold_days=HOLD, stop_pct=None, stop_mode="low"):
    """
    エントリー: event 日の close で買う
    Exit: stop_hit OR hold_days 後の close
    stop_mode: 'low' = 日中 Low が stop_level 到達 / 'close' = 日次 close が stop_level 到達
    """
    trades = []
    dates = df.index
    for i, dt in enumerate(dates):
        if not df.loc[dt, "event"]:
            continue
        entry = df.loc[dt, "close"]
        stop_level = entry * (1 + stop_pct / 100) if stop_pct is not None else None
        exit_price = None; exit_date = None; stop_hit = False
        # Walk forward
        for j in range(1, hold_days + 1):
            if i + j >= len(dates):
                break
            nd = dates[i + j]
            bar = df.loc[nd]
            if stop_level is not None:
                if stop_mode == "low" and bar["low"] <= stop_level:
                    exit_price = stop_level  # stop fills at stop level
                    exit_date = nd; stop_hit = True
                    break
                if stop_mode == "close" and bar["close"] <= stop_level:
                    exit_price = bar["close"]
                    exit_date = nd; stop_hit = True
                    break
        if exit_price is None:
            # 時間決済
            j_exit = min(hold_days, len(dates) - 1 - i)
            if j_exit == 0:
                continue
            exit_date = dates[i + j_exit]
            exit_price = df.loc[exit_date, "close"]
        ret_pct = (exit_price / entry - 1) * 100 - COST_PCT
        trades.append({"entry_date": dt, "entry": entry, "exit_date": exit_date,
                       "exit": exit_price, "ret_pct": ret_pct, "stop_hit": stop_hit,
                       "hold_days": (exit_date - dt).days})
    return pd.DataFrame(trades)


def metrics(tr):
    if len(tr) == 0:
        return {}
    r = tr["ret_pct"].values
    wins = r[r > 0]; losses = r[r <= 0]
    pf = wins.sum() / abs(losses.sum()) if losses.sum() < 0 else float('inf')
    return dict(
        n=len(r), mean=r.mean(), wr=(r > 0).mean() * 100,
        tstat=(r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))) if r.std(ddof=1) > 0 else np.nan,
        sharpe=(r.mean() / r.std(ddof=1) * np.sqrt(252 / HOLD)) if r.std(ddof=1) > 0 else np.nan,
        max_loss=r.min(), avg_win=wins.mean() if len(wins) else np.nan,
        avg_loss=losses.mean() if len(losses) else np.nan, pf=pf,
        stop_rate=tr["stop_hit"].mean() * 100,
    )


def split_h12_metrics(tr):
    tr = tr.sort_values("entry_date").reset_index(drop=True)
    mid = len(tr) // 2
    return metrics(tr), metrics(tr.iloc[:mid]), metrics(tr.iloc[mid:])


def main():
    print("=" * 110)
    print(f"レーザーテック MA25 サポート — ストップロス検証 (保有 {HOLD}日固定)")
    print("=" * 110)
    df = load_daily(SYM)
    df = find_entries(df, ma=25, tol_pct=1.0, dd_thresh=5.0, slope_up=True)
    print(f"Events: {df['event'].sum()}  Period: {df.index.min().date()} ~ {df.index.max().date()}")

    # =========================================================================
    # Part 1: Stop-level スイープ (stop_mode=low, 日中 Low 基準)
    # =========================================================================
    print("\n[Part 1] Stop スイープ  (mode=low: 日中Lowヒットで stop level 約定)")
    print("-" * 110)
    rows = []
    for stop in [None, -3, -5, -7, -10, -15]:
        tr = simulate(df, hold_days=HOLD, stop_pct=stop, stop_mode="low")
        if len(tr) == 0: continue
        full, h1, h2 = split_h12_metrics(tr)
        rows.append({
            "stop%": stop if stop is not None else "None",
            "n": full["n"], "mean%": full["mean"], "wr%": full["wr"],
            "t": full["tstat"], "sharpe": full["sharpe"],
            "max_loss%": full["max_loss"], "avg_loss%": full["avg_loss"],
            "avg_win%": full["avg_win"], "pf": full["pf"],
            "stop_rate%": full["stop_rate"],
            "h1_sharpe": h1["sharpe"], "h1_t": h1["tstat"],
            "h2_sharpe": h2["sharpe"], "h2_t": h2["tstat"],
        })
    grid = pd.DataFrame(rows)
    print(grid.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    grid.to_csv(OUT / "stop_sweep_low.csv", index=False)

    # =========================================================================
    # Part 2: Stop-level スイープ (stop_mode=close, 終値基準)
    # =========================================================================
    print("\n[Part 2] Stop スイープ  (mode=close: 終値ヒットで翌日成行相当)")
    print("-" * 110)
    rows = []
    for stop in [None, -3, -5, -7, -10, -15]:
        tr = simulate(df, hold_days=HOLD, stop_pct=stop, stop_mode="close")
        if len(tr) == 0: continue
        full, h1, h2 = split_h12_metrics(tr)
        rows.append({
            "stop%": stop if stop is not None else "None",
            "n": full["n"], "mean%": full["mean"], "wr%": full["wr"],
            "t": full["tstat"], "sharpe": full["sharpe"],
            "max_loss%": full["max_loss"], "avg_loss%": full["avg_loss"],
            "avg_win%": full["avg_win"], "pf": full["pf"],
            "stop_rate%": full["stop_rate"],
            "h1_sharpe": h1["sharpe"], "h1_t": h1["tstat"],
            "h2_sharpe": h2["sharpe"], "h2_t": h2["tstat"],
        })
    grid2 = pd.DataFrame(rows)
    print(grid2.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    grid2.to_csv(OUT / "stop_sweep_close.csv", index=False)

    # =========================================================================
    # Part 3: 保有期間 × Stop の 2D (stop_mode=low)
    # =========================================================================
    print("\n[Part 3] 保有期間 × Stop  (mode=low, slope=up)")
    print("-" * 110)
    rows = []
    for hold in [5, 7, 10, 14, 20]:
        for stop in [None, -3, -5, -7, -10]:
            tr = simulate(df, hold_days=hold, stop_pct=stop, stop_mode="low")
            if len(tr) == 0: continue
            m = metrics(tr)
            # 保有期間差で sharpe 再調整
            if tr["ret_pct"].std(ddof=1) > 0:
                sh = tr["ret_pct"].mean() / tr["ret_pct"].std(ddof=1) * np.sqrt(252 / hold)
            else: sh = np.nan
            rows.append({
                "hold": hold, "stop%": stop if stop is not None else "None",
                "n": m["n"], "mean%": m["mean"], "wr%": m["wr"], "t": m["tstat"],
                "sharpe": sh, "max_loss%": m["max_loss"], "pf": m["pf"],
                "stop_rate%": m["stop_rate"],
            })
    grid3 = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    print(grid3.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    grid3.to_csv(OUT / "hold_stop_2d.csv", index=False)

    # =========================================================================
    # Part 4: ベスト構成のトレードリスト
    # =========================================================================
    print("\n[Part 4] 推奨構成 (hold=10, stop=-5%, mode=low) トレードリスト")
    print("-" * 110)
    tr = simulate(df, hold_days=10, stop_pct=-5, stop_mode="low")
    tr = tr.sort_values("entry_date").reset_index(drop=True)
    print(tr.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    tr.to_csv(OUT / "best_trades.csv", index=False)

    # Equity curve
    tr["cum_pct"] = tr["ret_pct"].cumsum()
    print(f"\n累積リターン (単純足し算): {tr['cum_pct'].iloc[-1]:.1f}%")
    print(f"最大ドローダウン: {(tr['cum_pct'] - tr['cum_pct'].cummax()).min():.1f}%")

    # =========================================================================
    # Part 5: Plot
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax = axes[0, 0]
    x = [str(s) for s in grid["stop%"]]
    ax.plot(x, grid["sharpe"], "-o", label="Full")
    ax.plot(x, grid["h1_sharpe"], "--s", label="H1", alpha=0.7)
    ax.plot(x, grid["h2_sharpe"], "--^", label="H2", alpha=0.7)
    ax.set_title(f"Stop スイープ Sharpe (hold={HOLD}d, mode=low)")
    ax.set_xlabel("Stop level %"); ax.set_ylabel("Sharpe")
    ax.axhline(2, color="g", ls="--"); ax.axhline(0, color="k", lw=0.5)
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.bar(x, grid["stop_rate%"], color="orange", alpha=0.7)
    ax.set_title("Stop 発動率 (%)")
    ax.set_xlabel("Stop level %"); ax.set_ylabel("%")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(tr["entry_date"], tr["cum_pct"], "-o", color="teal")
    ax.set_title(f"推奨構成 (hold=10, stop=-5%) 累積リターン")
    ax.set_xlabel("Date"); ax.set_ylabel("Cumulative %")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.hist(tr["ret_pct"], bins=25, color="steelblue", alpha=0.7)
    ax.axvline(0, color="k"); ax.axvline(tr["ret_pct"].mean(), color="red",
                                          label=f"mean {tr['ret_pct'].mean():.2f}%")
    ax.set_title(f"推奨構成リターン分布  N={len(tr)}  WR={(tr['ret_pct']>0).mean()*100:.1f}%")
    ax.set_xlabel("ret %"); ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT / "stop_analysis.png", dpi=100)
    print(f"\nSaved: {OUT/'stop_analysis.png'}")
    print("\nDONE")


if __name__ == "__main__":
    main()
