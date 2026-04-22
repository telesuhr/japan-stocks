"""
VWAP mean-revert 戦略の精緻化・採用化検証

先行分析 (20260422_vwap_meanrevert):
  Best: 11:00 entry × 200bps thresh → Full Sharpe+3.80, t+2.50, N=109
  H1 +5.43 / H2 +2.20 (t+1.03) ← H2 弱化が唯一の懸念

本分析の目的:
  (1) H2 弱化の原因特定 (乖離頻度低下? 回帰弱化? 銘柄シフト?)
  (2) 閾値・時刻・銘柄選別で Sharpe をさらに押し上げ、H2 でも t>=2 を達成
  (3) 実運用可能な採用スペック確定
"""

import psycopg2, pandas as pd, numpy as np
from pathlib import Path
from itertools import combinations
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}
OUT = Path(__file__).parent
COST_BPS = 4.0

SYMBOLS = {
    "8035.T": "TEL",
    "6857.T": "アドバン",
    "6146.T": "ディスコ",
    "6920.T": "レーザー",
    "6503.T": "三菱電機",
    "5711.T": "三菱マテ",
    "5713.T": "住友鉱山",
    "5706.T": "三井金属",
}


def load_intraday(sym):
    conn = psycopg2.connect(**PG)
    df = pd.read_sql(
        f"SELECT timestamp, open, high, low, close, volume FROM intraday_data "
        f"WHERE symbol='{sym}' ORDER BY timestamp",
        conn,
    )
    conn.close()
    if df.empty:
        return None
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["jst"] = df["timestamp"] + pd.Timedelta(hours=9)
    df = df.dropna(subset=["open", "close", "volume"]).set_index("jst").sort_index()
    h, m = df.index.hour, df.index.minute
    morning = (h == 9) | (h == 10) | ((h == 11) & (m <= 30))
    afternoon = ((h == 12) & (m >= 30)) | (h == 13) | (h == 14) | ((h == 15) & (m <= 30))
    df = df[morning | afternoon].copy()
    return df


def build_trades(df, sym, entry_hm=(11, 0), exit_hm=(15, 25), thresh_bps=200, tol=2):
    """
    entry_hm: (hour, minute) でday-VWAP乖離を判定
    exit_hm: その時刻のclose決済 (tol分以内に最も近い bar)
    """
    trades = []
    for date, g in df.groupby(df.index.date):
        g = g.sort_index().copy()
        pv = (g["close"] * g["volume"]).cumsum()
        cv = g["volume"].cumsum().replace(0, np.nan)
        g["vwap"] = pv / cv
        g["dev_bps"] = (g["close"] / g["vwap"] - 1) * 10000

        bar_min = g.index.hour * 60 + g.index.minute

        def pick(hm, col, tol_):
            t = hm[0] * 60 + hm[1]
            diff = np.abs(bar_min - t)
            idx = diff.argmin()
            if diff[idx] > tol_:
                return None, None, None
            return g.iloc[idx][col], g.iloc[idx]["dev_bps"], g.index[idx]

        ent_price, dev, ent_ts = pick(entry_hm, "close", tol)
        if ent_price is None or np.isnan(dev):
            continue
        if abs(dev) < thresh_bps:
            continue
        exit_price, _, _ = pick(exit_hm, "close", 5)
        if exit_price is None:
            # fallback: use last bar
            exit_price = g.iloc[-1]["close"]

        direction = -np.sign(dev)
        ret_bps = direction * (exit_price / ent_price - 1) * 10000 - COST_BPS
        trades.append({
            "date": date, "sym": sym, "dev_bps": dev, "direction": int(direction),
            "ret_bps": ret_bps,
        })
    return pd.DataFrame(trades)


def stats(arr):
    arr = np.asarray(arr)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 3:
        return dict(n=len(arr), mean=np.nan, std=np.nan, sharpe=np.nan, tstat=np.nan, wr=np.nan)
    m = arr.mean(); s = arr.std(ddof=1)
    return dict(
        n=len(arr), mean=m, std=s,
        sharpe=(m / s * np.sqrt(252)) if s > 0 else np.nan,
        tstat=(m / (s / np.sqrt(len(arr)))) if s > 0 else np.nan,
        wr=(arr > 0).mean() * 100,
    )


def split_stats(df_trades, label="Full"):
    df_trades = df_trades.sort_values("date").reset_index(drop=True)
    mid = len(df_trades) // 2
    full = stats(df_trades["ret_bps"].values)
    h1 = stats(df_trades.iloc[:mid]["ret_bps"].values)
    h2 = stats(df_trades.iloc[mid:]["ret_bps"].values)
    return {"label": label, "full_n": full["n"], "full_sharpe": full["sharpe"], "full_t": full["tstat"],
            "h1_n": h1["n"], "h1_sharpe": h1["sharpe"], "h1_t": h1["tstat"],
            "h2_n": h2["n"], "h2_sharpe": h2["sharpe"], "h2_t": h2["tstat"]}


def main():
    print("=" * 90)
    print("VWAP mean-revert 戦略 精緻化")
    print("=" * 90)

    all_data = {}
    for sym in SYMBOLS:
        df = load_intraday(sym)
        if df is not None and len(df) > 100:
            all_data[sym] = df
    print(f"Loaded {len(all_data)} symbols")
    print()

    # --- Part 1: H2 弱化の診断 ---
    print("=" * 90)
    print("[Part 1] H2 弱化の診断 (entry=11:00 / thresh=200bps)")
    print("=" * 90)
    base_trades = []
    for sym, df in all_data.items():
        t = build_trades(df, sym, (11, 0), (15, 25), 200)
        if not t.empty:
            base_trades.append(t)
    pooled = pd.concat(base_trades, ignore_index=True).sort_values("date").reset_index(drop=True)

    # (1a) 月次パフォーマンス
    pooled["ym"] = pd.to_datetime(pooled["date"]).dt.to_period("M")
    print("\n月次 pooled (entry=11:00 / thresh=200bps):")
    mo = pooled.groupby("ym").agg(n=("ret_bps", "size"), mean=("ret_bps", "mean"),
                                    std=("ret_bps", "std")).round(1)
    mo["tstat"] = (mo["mean"] / (mo["std"] / np.sqrt(mo["n"]))).round(2)
    print(mo.to_string())
    mo.to_csv(OUT / "monthly_base.csv")

    # (1b) 乖離頻度の時系列 (|dev|>=200 だけ採用になるので、全日の乖離分布を見る)
    # 11:00 時点の |dev_bps| 分布を月次で取得
    print("\n11:00 時点の |乖離bps| 分布 (月次):")
    def all_devs(sym, df):
        rows = []
        for date, g in df.groupby(df.index.date):
            g = g.sort_index().copy()
            pv = (g["close"] * g["volume"]).cumsum()
            cv = g["volume"].cumsum().replace(0, np.nan)
            g["vwap"] = pv / cv
            g["dev_bps"] = (g["close"] / g["vwap"] - 1) * 10000
            bar_min = g.index.hour * 60 + g.index.minute
            t = 11 * 60
            idx = np.abs(bar_min - t).argmin()
            if abs(bar_min[idx] - t) > 2:
                continue
            rows.append({"date": date, "sym": sym, "abs_dev": abs(g.iloc[idx]["dev_bps"])})
        return pd.DataFrame(rows)
    all_dev = pd.concat([all_devs(s, d) for s, d in all_data.items()], ignore_index=True)
    all_dev["ym"] = pd.to_datetime(all_dev["date"]).dt.to_period("M")
    dev_stats = all_dev.groupby("ym")["abs_dev"].agg(
        mean="mean", median="median", p90=lambda x: x.quantile(0.9), trigger_rate=lambda x: (x >= 200).mean() * 100
    ).round(1)
    print(dev_stats.to_string())
    dev_stats.to_csv(OUT / "dev_distribution_monthly.csv")

    # (1c) 銘柄別 H1/H2
    print("\n銘柄別 H1/H2 パフォーマンス (entry=11:00 / thresh=200bps):")
    rows = []
    for sym in pooled["sym"].unique():
        sub = pooled[pooled["sym"] == sym].sort_values("date").reset_index(drop=True)
        if len(sub) < 6:
            continue
        mid = len(sub) // 2
        s1, s2 = stats(sub.iloc[:mid]["ret_bps"].values), stats(sub.iloc[mid:]["ret_bps"].values)
        rows.append({"sym": sym, "name": SYMBOLS[sym], "h1_n": s1["n"], "h1_sharpe": s1["sharpe"], "h1_t": s1["tstat"],
                     "h2_n": s2["n"], "h2_sharpe": s2["sharpe"], "h2_t": s2["tstat"]})
    by_stock_h12 = pd.DataFrame(rows)
    print(by_stock_h12.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    by_stock_h12.to_csv(OUT / "by_stock_h1h2.csv", index=False)
    print()

    # --- Part 2: グリッド探索 ---
    print("=" * 90)
    print("[Part 2] 閾値 × エントリー時刻 × 銘柄サブセット グリッド探索")
    print("=" * 90)

    # 対象銘柄 (先行分析で positive だった semi + 非鉄)
    candidate_syms = ["8035.T", "6857.T", "6146.T", "6920.T", "5711.T", "5706.T"]
    candidate_syms = [s for s in candidate_syms if s in all_data]

    thresh_list = [150, 200, 250, 300]
    entry_list = [(10, 30), (11, 0), (11, 15), (11, 25)]
    exit_list = [(11, 25), (15, 25)]

    grid_rows = []
    for th in thresh_list:
        for ent in entry_list:
            for exi in exit_list:
                if (ent[0] * 60 + ent[1]) >= (exi[0] * 60 + exi[1]):
                    continue
                trades_list = []
                for sym in candidate_syms:
                    t = build_trades(all_data[sym], sym, ent, exi, th)
                    if not t.empty:
                        trades_list.append(t)
                if not trades_list:
                    continue
                pooled_g = pd.concat(trades_list, ignore_index=True)
                s = split_stats(pooled_g)
                grid_rows.append({
                    "entry": f"{ent[0]:02d}:{ent[1]:02d}",
                    "exit": f"{exi[0]:02d}:{exi[1]:02d}",
                    "thresh": th,
                    **s,
                })
    grid = pd.DataFrame(grid_rows).sort_values("full_sharpe", ascending=False)
    print(grid.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    grid.to_csv(OUT / "grid_results.csv", index=False)
    print()

    # H2 で t>=2 & Full Sharpe>=2 を満たす config
    adopted = grid[(grid["h2_t"] >= 2.0) & (grid["full_sharpe"] >= 2.0) & (grid["full_n"] >= 30)]
    print("[採用基準] Full Sharpe≥2 & H2 t≥2 & N≥30 クリア config:")
    if adopted.empty:
        print("  → なし。H2 t≥2 を維持するのは困難。")
    else:
        print(adopted.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print()

    # --- Part 3: 銘柄サブセット最適化 (TEL+ディスコ+レーザー+アドバン) ---
    print("=" * 90)
    print("[Part 3] 銘柄サブセット最適化 (Best config = 11:00 entry × 200 thresh × 15:25 exit)")
    print("=" * 90)
    universe = ["8035.T", "6857.T", "6146.T", "6920.T", "5711.T", "5706.T"]
    universe = [s for s in universe if s in all_data]
    sub_rows = []
    for k in range(2, len(universe) + 1):
        for combo in combinations(universe, k):
            trades_list = []
            for sym in combo:
                t = build_trades(all_data[sym], sym, (11, 0), (15, 25), 200)
                if not t.empty:
                    trades_list.append(t)
            if not trades_list:
                continue
            pooled_s = pd.concat(trades_list, ignore_index=True)
            s = split_stats(pooled_s)
            sub_rows.append({"combo": "+".join(SYMBOLS[x] for x in combo), "k": k, **s})
    sub_df = pd.DataFrame(sub_rows).sort_values("full_sharpe", ascending=False).head(20)
    print("Top-20 サブセット:")
    print(sub_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    sub_df.to_csv(OUT / "subset_topN.csv", index=False)
    print()

    # --- Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    ax = axes[0, 0]
    mo_plot = mo.reset_index()
    ax.bar(range(len(mo_plot)), mo_plot["mean"],
           color=["g" if m > 0 else "r" for m in mo_plot["mean"]])
    ax.set_xticks(range(len(mo_plot)))
    ax.set_xticklabels([str(y) for y in mo_plot["ym"]], rotation=45, fontsize=8)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title("Monthly mean bp (entry=11:00 / thresh=200bps, all universe)")
    ax.set_ylabel("bps")

    ax = axes[0, 1]
    ds = dev_stats.reset_index()
    ax.plot(range(len(ds)), ds["median"], label="median |dev|", color="b")
    ax.plot(range(len(ds)), ds["p90"], label="p90 |dev|", color="orange")
    ax2 = ax.twinx()
    ax2.bar(range(len(ds)), ds["trigger_rate"], color="gray", alpha=0.3, label="trigger rate (|dev|>=200)")
    ax2.set_ylabel("trigger rate (%)")
    ax.set_xticks(range(len(ds)))
    ax.set_xticklabels([str(y) for y in ds["ym"]], rotation=45, fontsize=8)
    ax.set_title("11:00 VWAP |dev| distribution (monthly)")
    ax.set_ylabel("bps")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")

    ax = axes[1, 0]
    if not by_stock_h12.empty:
        x = np.arange(len(by_stock_h12))
        ax.bar(x - 0.2, by_stock_h12["h1_sharpe"], 0.4, label="H1")
        ax.bar(x + 0.2, by_stock_h12["h2_sharpe"], 0.4, label="H2")
        ax.set_xticks(x)
        ax.set_xticklabels(by_stock_h12["name"], rotation=30)
        ax.axhline(0, color="k", lw=0.5)
        ax.axhline(2, color="g", ls="--", lw=0.5)
        ax.set_title("銘柄別 H1/H2 Sharpe")
        ax.legend()

    ax = axes[1, 1]
    top10 = grid.head(10)
    ax.barh(range(len(top10)), top10["full_sharpe"])
    ax.set_yticks(range(len(top10)))
    ax.set_yticklabels([f"{r['entry']}→{r['exit']} th={r['thresh']}" for _, r in top10.iterrows()], fontsize=8)
    ax.axvline(2, color="g", ls="--")
    ax.set_title("Grid Top-10 Full Sharpe")
    plt.tight_layout()
    plt.savefig(OUT / "result.png", dpi=100)
    print(f"Saved: {OUT/'result.png'}")


if __name__ == "__main__":
    main()
