"""
Pattern 1: 前引→後寄 リバーサル (Lunch Gap Fade)
Pattern 2: 引け15分 ドリフト (Closing Drift)

対象: 12銘柄 (半導体5 + 非鉄3 + 自動車2 + 商社2)
期間: 2024-11 ~ 2026-04
コスト: 4bps

P1: 11:25 の 9:00からの累積リターンが ±X bps を超えていれば 逆方向 12:35決済
P2: 14:55 の 9:00からの累積リターンが ±X bps を超えていれば 同方向 15:25決済
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
    "8035.T": "TEL", "6857.T": "アドバン", "6146.T": "ディスコ",
    "6920.T": "レーザー", "6503.T": "三菱電機",
    "5711.T": "三菱マテ", "5713.T": "住友鉱山", "5706.T": "三井金属",
    "7203.T": "トヨタ", "7267.T": "ホンダ",
    "8058.T": "三菱商事", "8031.T": "三井物産",
}

def load_intraday(sym):
    conn = psycopg2.connect(**PG)
    df = pd.read_sql(
        f"SELECT timestamp, open, high, low, close, volume FROM intraday_data "
        f"WHERE symbol='{sym}' ORDER BY timestamp",
        conn,
    )
    conn.close()
    if df.empty: return None
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["jst"] = df["timestamp"] + pd.Timedelta(hours=9)
    df = df.dropna(subset=["open", "close"]).set_index("jst").sort_index()
    h, m = df.index.hour, df.index.minute
    morning = (h == 9) | (h == 10) | ((h == 11) & (m <= 30))
    afternoon = ((h == 12) & (m >= 30)) | (h == 13) | (h == 14) | ((h == 15) & (m <= 30))
    return df[morning | afternoon].copy()


def pick(g, h, m, col="close", tol=3):
    bar_min = g.index.hour * 60 + g.index.minute
    target = h * 60 + m
    diff = np.abs(bar_min - target)
    idx = diff.argmin()
    if diff[idx] > tol: return None
    return g.iloc[idx][col]


def build_p1_trades(df, sym, thresh_bps, exit_time=(12, 35)):
    """P1: 11:25 close vs 9:00 open で |ret| >= thresh → 逆方向 → 12:35決済"""
    trades = []
    for date, g in df.groupby(df.index.date):
        g = g.sort_index()
        p_open = pick(g, 9, 0, "open")
        p_1125 = pick(g, 11, 25, "close")
        p_exit = pick(g, exit_time[0], exit_time[1], "close", tol=5)
        if any(x is None for x in [p_open, p_1125, p_exit]):
            continue
        pre_ret_bps = (p_1125 / p_open - 1) * 10000
        if abs(pre_ret_bps) < thresh_bps:
            continue
        direction = -np.sign(pre_ret_bps)
        ret_bps = direction * (p_exit / p_1125 - 1) * 10000 - COST_BPS
        trades.append({"date": date, "sym": sym, "pre_ret": pre_ret_bps,
                       "direction": int(direction), "ret_bps": ret_bps})
    return pd.DataFrame(trades)


def build_p2_trades(df, sym, thresh_bps, entry_hm=(14, 55), exit_hm=(15, 25)):
    """P2: 14:55 close vs 9:00 open, |ret| >= thresh → 同方向 → 15:25決済"""
    trades = []
    for date, g in df.groupby(df.index.date):
        g = g.sort_index()
        p_open = pick(g, 9, 0, "open")
        p_ent = pick(g, entry_hm[0], entry_hm[1], "close")
        p_exit = pick(g, exit_hm[0], exit_hm[1], "close", tol=5)
        if any(x is None for x in [p_open, p_ent, p_exit]):
            continue
        day_ret_bps = (p_ent / p_open - 1) * 10000
        if abs(day_ret_bps) < thresh_bps:
            continue
        direction = np.sign(day_ret_bps)  # 順張り
        ret_bps = direction * (p_exit / p_ent - 1) * 10000 - COST_BPS
        trades.append({"date": date, "sym": sym, "day_ret": day_ret_bps,
                       "direction": int(direction), "ret_bps": ret_bps})
    return pd.DataFrame(trades)


def build_p2_no_thresh(df, sym, entry_hm=(14, 55), exit_hm=(15, 25)):
    """P2版B: 閾値なし、day-return 符号だけで順張り (N稼ぐ版)"""
    trades = []
    for date, g in df.groupby(df.index.date):
        g = g.sort_index()
        p_open = pick(g, 9, 0, "open")
        p_ent = pick(g, entry_hm[0], entry_hm[1], "close")
        p_exit = pick(g, exit_hm[0], exit_hm[1], "close", tol=5)
        if any(x is None for x in [p_open, p_ent, p_exit]):
            continue
        day_ret_bps = (p_ent / p_open - 1) * 10000
        direction = np.sign(day_ret_bps) if day_ret_bps != 0 else 0
        if direction == 0: continue
        ret_bps = direction * (p_exit / p_ent - 1) * 10000 - COST_BPS
        trades.append({"date": date, "sym": sym, "day_ret": day_ret_bps,
                       "direction": int(direction), "ret_bps": ret_bps})
    return pd.DataFrame(trades)


def stats(arr):
    arr = np.asarray(arr); arr = arr[~np.isnan(arr)]
    if len(arr) < 3:
        return dict(n=len(arr), mean=np.nan, sharpe=np.nan, tstat=np.nan, wr=np.nan)
    m = arr.mean(); s = arr.std(ddof=1)
    return dict(n=len(arr), mean=m,
                sharpe=(m / s * np.sqrt(252)) if s > 0 else np.nan,
                tstat=(m / (s / np.sqrt(len(arr)))) if s > 0 else np.nan,
                wr=(arr > 0).mean() * 100)


def split_h12(df_trades):
    df_trades = df_trades.sort_values("date").reset_index(drop=True)
    mid = len(df_trades) // 2
    return (stats(df_trades["ret_bps"].values),
            stats(df_trades.iloc[:mid]["ret_bps"].values),
            stats(df_trades.iloc[mid:]["ret_bps"].values))


def summarize_grid(builder, thresh_list, all_data):
    """閾値スイープの pooled 結果"""
    rows = []
    for th in thresh_list:
        trades_list = []
        for sym, df in all_data.items():
            t = builder(df, sym, th)
            if not t.empty:
                trades_list.append(t)
        if not trades_list:
            continue
        pooled = pd.concat(trades_list, ignore_index=True)
        full, h1, h2 = split_h12(pooled)
        rows.append({
            "thresh": th, "n": full["n"],
            "full_sharpe": full["sharpe"], "full_t": full["tstat"], "mean_bp": full["mean"], "wr": full["wr"],
            "h1_n": h1["n"], "h1_sharpe": h1["sharpe"], "h1_t": h1["tstat"],
            "h2_n": h2["n"], "h2_sharpe": h2["sharpe"], "h2_t": h2["tstat"],
        })
    return pd.DataFrame(rows)


def by_stock_best(builder, thresh, all_data):
    rows = []
    for sym, df in all_data.items():
        t = builder(df, sym, thresh)
        if t.empty or len(t) < 10:
            rows.append({"sym": sym, "name": SYMBOLS[sym], "n": len(t), "sharpe": np.nan, "tstat": np.nan})
            continue
        s = stats(t["ret_bps"].values)
        rows.append({"sym": sym, "name": SYMBOLS[sym], "n": s["n"],
                     "sharpe": s["sharpe"], "tstat": s["tstat"], "mean_bp": s["mean"], "wr": s["wr"]})
    return pd.DataFrame(rows).sort_values("sharpe", ascending=False, na_position="last")


def best_subset(builder, thresh, all_data, universe, k_range=(2, 6)):
    rows = []
    for k in range(k_range[0], min(k_range[1], len(universe)) + 1):
        for combo in combinations(universe, k):
            tl = [builder(all_data[s], s, thresh) for s in combo]
            tl = [t for t in tl if not t.empty]
            if not tl: continue
            pooled = pd.concat(tl, ignore_index=True)
            if len(pooled) < 10: continue
            full, h1, h2 = split_h12(pooled)
            rows.append({"combo": "+".join(SYMBOLS[x] for x in combo), "k": k,
                         "n": full["n"], "full_sharpe": full["sharpe"], "full_t": full["tstat"],
                         "h1_sharpe": h1["sharpe"], "h1_t": h1["tstat"],
                         "h2_sharpe": h2["sharpe"], "h2_t": h2["tstat"]})
    return pd.DataFrame(rows).sort_values("full_sharpe", ascending=False).head(15)


def main():
    print("=" * 90)
    print("Pattern 1 + 2: Lunch Gap Fade + Closing Drift")
    print("=" * 90)
    all_data = {}
    for sym in SYMBOLS:
        df = load_intraday(sym)
        if df is not None:
            all_data[sym] = df
    print(f"Loaded {len(all_data)} symbols\n")

    # =========================================================================
    # Pattern 1: 前引→後寄 Fade
    # =========================================================================
    print("=" * 90)
    print("[Pattern 1] 前引→後寄 リバーサル (11:25 entry → 12:35 exit)")
    print("=" * 90)
    p1_thresh = [50, 75, 100, 150, 200, 250, 300]
    p1_grid = summarize_grid(lambda df, s, th: build_p1_trades(df, s, th), p1_thresh, all_data)
    print("\n全銘柄 pooled (閾値スイープ):")
    print(p1_grid.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    p1_grid.to_csv(OUT / "p1_grid_pooled.csv", index=False)

    # 銘柄別 (100bps閾値)
    print("\n銘柄別 (閾値=100bps):")
    p1_by_stock = by_stock_best(lambda df, s, th: build_p1_trades(df, s, th), 100, all_data)
    print(p1_by_stock.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    p1_by_stock.to_csv(OUT / "p1_by_stock_th100.csv", index=False)

    # 銘柄別 (150bps)
    print("\n銘柄別 (閾値=150bps):")
    p1_by_stock150 = by_stock_best(lambda df, s, th: build_p1_trades(df, s, th), 150, all_data)
    print(p1_by_stock150.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    # 全銘柄で subset search (閾値=100)
    print("\nサブセット Top-15 (閾値=100bps):")
    p1_subset = best_subset(lambda df, s, th: build_p1_trades(df, s, th), 100,
                            all_data, list(all_data.keys()), k_range=(2, 6))
    print(p1_subset.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    p1_subset.to_csv(OUT / "p1_subset_th100.csv", index=False)

    # =========================================================================
    # Pattern 2: 引け15分ドリフト
    # =========================================================================
    print("\n" + "=" * 90)
    print("[Pattern 2] 引け15分ドリフト (14:55 entry → 15:25 exit)")
    print("=" * 90)

    # 2a: 閾値付き (順張り)
    p2_thresh = [0, 50, 100, 150, 200, 300, 500]
    p2_grid = summarize_grid(lambda df, s, th: build_p2_trades(df, s, th), p2_thresh, all_data)
    print("\n全銘柄 pooled (閾値スイープ, 順張り):")
    print(p2_grid.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    p2_grid.to_csv(OUT / "p2_grid_pooled.csv", index=False)

    # 銘柄別 (100bps)
    print("\n銘柄別 (閾値=100bps, 順張り):")
    p2_by_stock = by_stock_best(lambda df, s, th: build_p2_trades(df, s, th), 100, all_data)
    print(p2_by_stock.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    p2_by_stock.to_csv(OUT / "p2_by_stock_th100.csv", index=False)

    # サブセット
    print("\nサブセット Top-15 (閾値=100bps):")
    p2_subset = best_subset(lambda df, s, th: build_p2_trades(df, s, th), 100,
                            all_data, list(all_data.keys()), k_range=(2, 6))
    print(p2_subset.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    p2_subset.to_csv(OUT / "p2_subset_th100.csv", index=False)

    # 逆張り版検証 (P2 Fade)
    print("\n[P2b] 逆張り版 (14:55 time 累積リターンと逆方向, fade)")
    def build_p2b(df, sym, thresh_bps):
        t = build_p2_trades(df, sym, thresh_bps)
        if t.empty: return t
        t = t.copy()
        t["ret_bps"] = -t["ret_bps"] - 2 * COST_BPS  # 符号反転 (コスト再計上)
        return t
    p2b_grid = summarize_grid(build_p2b, p2_thresh, all_data)
    print(p2b_grid.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    p2b_grid.to_csv(OUT / "p2b_grid_fade.csv", index=False)

    # =========================================================================
    # Plot
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    ax = axes[0, 0]
    ax.plot(p1_grid["thresh"], p1_grid["full_sharpe"], "-o", label="Full")
    ax.plot(p1_grid["thresh"], p1_grid["h2_sharpe"], "-s", label="H2")
    ax.set_title("P1 Lunch Gap Fade: Sharpe vs thresh")
    ax.set_xlabel("|pre_ret| thresh bps"); ax.set_ylabel("Sharpe")
    ax.axhline(2, color="g", ls="--"); ax.axhline(0, color="k", lw=0.5)
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(p1_grid["thresh"], p1_grid["n"], "-o", color="purple")
    ax.set_title("P1: N vs thresh"); ax.set_xlabel("thresh"); ax.set_ylabel("N")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(p2_grid["thresh"], p2_grid["full_sharpe"], "-o", label="Full 順張り")
    ax.plot(p2_grid["thresh"], p2_grid["h2_sharpe"], "-s", label="H2 順張り")
    ax.plot(p2b_grid["thresh"], p2b_grid["full_sharpe"], "--o", label="Full 逆張り")
    ax.set_title("P2 Closing Drift: Sharpe vs thresh")
    ax.set_xlabel("|day_ret| thresh bps"); ax.set_ylabel("Sharpe")
    ax.axhline(2, color="g", ls="--"); ax.axhline(0, color="k", lw=0.5)
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    # 銘柄別バー
    p1_plot = p1_by_stock.dropna(subset=["sharpe"]).sort_values("sharpe")
    x = np.arange(len(p1_plot))
    ax.barh(x - 0.2, p1_plot["sharpe"], 0.4, label="P1 (th=100)")
    p2_merge = p2_by_stock.set_index("sym").reindex(p1_plot["sym"])
    ax.barh(x + 0.2, p2_merge["sharpe"].values, 0.4, label="P2 (th=100)")
    ax.set_yticks(x); ax.set_yticklabels(p1_plot["name"])
    ax.axvline(0, color="k", lw=0.5); ax.axvline(2, color="g", ls="--")
    ax.set_title("Sharpe by stock")
    ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT / "result.png", dpi=100)
    print(f"\nSaved: {OUT/'result.png'}")


if __name__ == "__main__":
    main()
