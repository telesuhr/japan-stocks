"""
VWAP mean-revert 戦略 — 追加磨き込み (b,c,d)

ベースライン (refined): TEL+ディスコ+レーザー, 11:00 entry, 200bps thresh, 15:25 exit
  → Full Sharpe+6.76, t+2.73, N=41; H2 Sharpe+9.94, t+2.87

(b) 閾値スイープ (150/175/200/225/250/275/300) × 採用サブセット
(c) 連続エントリー時刻拡張 (10:00-11:25 の任意タイミングで初回シグナル)
(d) ボラレジーム層別 (当日 9:00-10:00 レンジ / 前日 daily 変化率)
"""

import psycopg2, pandas as pd, numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}
OUT = Path(__file__).parent
COST_BPS = 4.0

CORE = {
    "8035.T": "TEL",
    "6146.T": "ディスコ",
    "6920.T": "レーザー",
    "6857.T": "アドバン",
    "5711.T": "三菱マテ",
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


def build_trades_with_regime(df, sym, entry_hm, exit_hm, thresh_bps, tol=2):
    """エントリー+レジーム特徴量 (first-hour range bps, day-open) を記録"""
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
                return None
            return g.iloc[idx][col]

        ent_price = pick(entry_hm, "close", tol)
        dev = pick(entry_hm, "dev_bps", tol)
        if ent_price is None or dev is None or np.isnan(dev) or abs(dev) < thresh_bps:
            continue

        # First-hour range (9:00-10:00) as vol proxy
        fh = g[(g.index.hour == 9)]
        if fh.empty:
            continue
        fh_open = fh.iloc[0]["open"]
        fh_range_bps = (fh["high"].max() - fh["low"].min()) / fh_open * 10000

        # Day open (for normalizing dev)
        day_open = g.iloc[0]["open"]

        exit_price = pick(exit_hm, "close", 5)
        if exit_price is None:
            exit_price = g.iloc[-1]["close"]

        direction = -np.sign(dev)
        ret_bps = direction * (exit_price / ent_price - 1) * 10000 - COST_BPS
        trades.append({
            "date": date, "sym": sym, "dev_bps": dev, "abs_dev": abs(dev),
            "direction": int(direction), "ret_bps": ret_bps,
            "fh_range_bps": fh_range_bps, "day_open": day_open,
        })
    return pd.DataFrame(trades)


def build_first_trigger(df, sym, scan_start=(10, 0), scan_end=(11, 30),
                         exit_hm=(15, 25), thresh_bps=200):
    """10:00〜11:30 の間で **最初に** 閾値を超えた時刻でエントリー (1日1回)"""
    trades = []
    start_min = scan_start[0] * 60 + scan_start[1]
    end_min = scan_end[0] * 60 + scan_end[1]
    for date, g in df.groupby(df.index.date):
        g = g.sort_index().copy()
        pv = (g["close"] * g["volume"]).cumsum()
        cv = g["volume"].cumsum().replace(0, np.nan)
        g["vwap"] = pv / cv
        g["dev_bps"] = (g["close"] / g["vwap"] - 1) * 10000
        bar_min = g.index.hour * 60 + g.index.minute
        mask = (bar_min >= start_min) & (bar_min <= end_min) & (g["dev_bps"].abs() >= thresh_bps)
        if not mask.any():
            continue
        first_idx = np.where(mask)[0][0]
        ent_price = g.iloc[first_idx]["close"]
        dev = g.iloc[first_idx]["dev_bps"]
        ent_ts = g.index[first_idx]

        # first-hour range
        fh = g[(g.index.hour == 9)]
        fh_range_bps = (fh["high"].max() - fh["low"].min()) / fh.iloc[0]["open"] * 10000

        # exit
        target = exit_hm[0] * 60 + exit_hm[1]
        diff = np.abs(bar_min - target)
        exi_idx = diff.argmin()
        if diff[exi_idx] > 5:
            exit_price = g.iloc[-1]["close"]
        else:
            exit_price = g.iloc[exi_idx]["close"]

        direction = -np.sign(dev)
        ret_bps = direction * (exit_price / ent_price - 1) * 10000 - COST_BPS
        trades.append({
            "date": date, "sym": sym, "dev_bps": dev, "abs_dev": abs(dev),
            "direction": int(direction), "ret_bps": ret_bps,
            "ent_hour": ent_ts.hour, "ent_min": ent_ts.minute,
            "fh_range_bps": fh_range_bps,
        })
    return pd.DataFrame(trades)


def stats(arr):
    arr = np.asarray(arr); arr = arr[~np.isnan(arr)]
    if len(arr) < 3:
        return dict(n=len(arr), mean=np.nan, sharpe=np.nan, tstat=np.nan, wr=np.nan)
    m = arr.mean(); s = arr.std(ddof=1)
    return dict(
        n=len(arr), mean=m,
        sharpe=(m / s * np.sqrt(252)) if s > 0 else np.nan,
        tstat=(m / (s / np.sqrt(len(arr)))) if s > 0 else np.nan,
        wr=(arr > 0).mean() * 100,
    )


def split_h12(df_trades):
    df_trades = df_trades.sort_values("date").reset_index(drop=True)
    mid = len(df_trades) // 2
    return (stats(df_trades["ret_bps"].values),
            stats(df_trades.iloc[:mid]["ret_bps"].values),
            stats(df_trades.iloc[mid:]["ret_bps"].values))


def main():
    print("=" * 90)
    print("VWAP mean-revert 追加磨き込み (b,c,d)")
    print("=" * 90)

    all_data = {s: load_intraday(s) for s in CORE}
    all_data = {s: d for s, d in all_data.items() if d is not None}
    print(f"Loaded {len(all_data)} symbols")
    print()

    SUBSETS = {
        "T+D+L": ["8035.T", "6146.T", "6920.T"],
        "T+D+L+MM": ["8035.T", "6146.T", "6920.T", "5711.T"],
        "T+D+L+MK": ["8035.T", "6146.T", "6920.T", "5706.T"],
    }

    # ==========================================================================
    # (b) 閾値スイープ × サブセット (11:00 entry, 15:25 exit)
    # ==========================================================================
    print("=" * 90)
    print("[b] 閾値スイープ (11:00 entry × 15:25 exit)")
    print("=" * 90)
    threshes = [150, 175, 200, 225, 250, 275, 300]
    rows = []
    for sname, syms in SUBSETS.items():
        for th in threshes:
            trades_list = [build_trades_with_regime(all_data[s], s, (11, 0), (15, 25), th)
                           for s in syms if s in all_data]
            trades_list = [t for t in trades_list if not t.empty]
            if not trades_list:
                continue
            pooled = pd.concat(trades_list, ignore_index=True)
            full, h1, h2 = split_h12(pooled)
            rows.append({
                "subset": sname, "thresh": th, "n": full["n"],
                "full_sharpe": full["sharpe"], "full_t": full["tstat"],
                "h1_n": h1["n"], "h1_sharpe": h1["sharpe"], "h1_t": h1["tstat"],
                "h2_n": h2["n"], "h2_sharpe": h2["sharpe"], "h2_t": h2["tstat"],
            })
    b_df = pd.DataFrame(rows).sort_values(["subset", "thresh"])
    print(b_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    b_df.to_csv(OUT / "b_thresh_sweep.csv", index=False)
    print()

    # ==========================================================================
    # (c) 連続エントリー (10:00-11:30 初回シグナル) × 閾値
    # ==========================================================================
    print("=" * 90)
    print("[c] 連続エントリー 10:00-11:30 初回発生時に1回 (1日1回)")
    print("=" * 90)
    rows = []
    for sname, syms in SUBSETS.items():
        for th in threshes:
            tl = [build_first_trigger(all_data[s], s, (10, 0), (11, 30), (15, 25), th)
                  for s in syms if s in all_data]
            tl = [t for t in tl if not t.empty]
            if not tl:
                continue
            pooled = pd.concat(tl, ignore_index=True)
            full, h1, h2 = split_h12(pooled)
            rows.append({
                "subset": sname, "thresh": th, "n": full["n"],
                "full_sharpe": full["sharpe"], "full_t": full["tstat"],
                "h1_n": h1["n"], "h1_sharpe": h1["sharpe"], "h1_t": h1["tstat"],
                "h2_n": h2["n"], "h2_sharpe": h2["sharpe"], "h2_t": h2["tstat"],
            })
    c_df = pd.DataFrame(rows).sort_values(["subset", "thresh"])
    print(c_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    c_df.to_csv(OUT / "c_continuous_entry.csv", index=False)
    print()

    # 採用候補抽出
    c_adopt = c_df[(c_df["full_sharpe"] >= 2.0) & (c_df["h2_t"] >= 2.0) & (c_df["n"] >= 30)]
    print("[c] Full Sharpe≥2 & H2 t≥2 & N≥30 クリア:")
    if c_adopt.empty:
        print("  → なし")
    else:
        print(c_adopt.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print()

    # ==========================================================================
    # (d) ボラレジーム層別 (first-hour range で三分割)
    # ==========================================================================
    print("=" * 90)
    print("[d] ボラレジーム層別 (当日 9:00-10:00 レンジを三分位で層別)")
    print("=" * 90)
    # ベースは T+D+L, 11:00 entry, 200 thresh, 15:25 exit
    core_trades = pd.concat(
        [build_trades_with_regime(all_data[s], s, (11, 0), (15, 25), 200)
         for s in SUBSETS["T+D+L"] if s in all_data],
        ignore_index=True,
    )
    # 銘柄別にレンジ分位を計算 (銘柄絶対比較ではなく銘柄内)
    def assign_regime(g):
        q = g["fh_range_bps"].quantile([1/3, 2/3]).values
        g["regime"] = np.where(g["fh_range_bps"] <= q[0], "low",
                        np.where(g["fh_range_bps"] <= q[1], "mid", "high"))
        return g
    core_trades = core_trades.groupby("sym", group_keys=False).apply(assign_regime)

    print("\nレジーム別 pooled (T+D+L, 11:00 entry, 200 thresh):")
    rows = []
    for reg, sub in core_trades.groupby("regime"):
        full, h1, h2 = split_h12(sub)
        rows.append({"regime": reg, "n": full["n"],
                     "full_sharpe": full["sharpe"], "full_t": full["tstat"],
                     "h1_sharpe": h1["sharpe"], "h1_t": h1["tstat"],
                     "h2_sharpe": h2["sharpe"], "h2_t": h2["tstat"]})
    d_df = pd.DataFrame(rows).set_index("regime").loc[["low", "mid", "high"]]
    print(d_df.to_string(float_format=lambda x: f"{x:.2f}"))
    d_df.to_csv(OUT / "d_regime_stratified.csv")
    print()

    # |dev| と first-hour range の相関 (高ボラ日ほど乖離大?)
    corr = core_trades[["abs_dev", "fh_range_bps"]].corr().iloc[0, 1]
    print(f"|dev| vs first-hour range 相関: {corr:.2f}")
    print()

    # レジームごとの mean |dev| も
    reg_devs = core_trades.groupby("regime")[["abs_dev", "fh_range_bps", "ret_bps"]].mean()
    print("\nレジーム別 平均値:")
    print(reg_devs.round(1).to_string())
    print()

    # 組み合わせ: (c) 連続エントリー × 高ボラレジーム除外
    print("=" * 90)
    print("[d2] (c) 連続エントリー × 高ボラ日除外 (low+mid のみ)")
    print("=" * 90)
    for sname, syms in SUBSETS.items():
        for th in [200, 225, 250]:
            tl = [build_first_trigger(all_data[s], s, (10, 0), (11, 30), (15, 25), th)
                  for s in syms if s in all_data]
            tl = [t for t in tl if not t.empty]
            if not tl:
                continue
            p = pd.concat(tl, ignore_index=True)
            # 銘柄内 first-hour range 三分位
            p = p.groupby("sym", group_keys=False).apply(
                lambda g: g.assign(
                    regime=np.where(g["fh_range_bps"] <= g["fh_range_bps"].quantile(1/3), "low",
                             np.where(g["fh_range_bps"] <= g["fh_range_bps"].quantile(2/3), "mid", "high"))
                )
            )
            lm = p[p["regime"].isin(["low", "mid"])]
            full, h1, h2 = split_h12(lm)
            print(f"  {sname:12s} th={th:3d} low+mid only: N={full['n']:3d} "
                  f"FullS={full['sharpe']:+5.2f} t={full['tstat']:+5.2f}  "
                  f"H2S={h2['sharpe']:+5.2f} t={h2['tstat']:+5.2f}  H2N={h2['n']}")
    print()

    # --- Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    ax = axes[0, 0]
    for sname in SUBSETS:
        sub = b_df[b_df["subset"] == sname]
        ax.plot(sub["thresh"], sub["full_sharpe"], "-o", label=sname)
    ax.set_title("(b) Thresh Sweep: Full Sharpe")
    ax.set_xlabel("thresh bps"); ax.set_ylabel("Full Sharpe")
    ax.axhline(2, color="g", ls="--", lw=0.5); ax.axhline(0, color="k", lw=0.5)
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    for sname in SUBSETS:
        sub = b_df[b_df["subset"] == sname]
        ax.plot(sub["thresh"], sub["h2_t"], "-o", label=sname)
    ax.set_title("(b) Thresh Sweep: H2 t-stat")
    ax.set_xlabel("thresh bps"); ax.set_ylabel("H2 t")
    ax.axhline(2, color="g", ls="--", lw=0.5); ax.axhline(0, color="k", lw=0.5)
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    for sname in SUBSETS:
        sub = c_df[c_df["subset"] == sname]
        ax.plot(sub["thresh"], sub["full_sharpe"], "-o", label=sname)
    ax.set_title("(c) Continuous entry: Full Sharpe")
    ax.set_xlabel("thresh bps")
    ax.axhline(2, color="g", ls="--", lw=0.5); ax.axhline(0, color="k", lw=0.5)
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    d_plot = d_df[["full_sharpe", "h2_sharpe"]].reset_index()
    x = np.arange(len(d_plot))
    ax.bar(x - 0.2, d_plot["full_sharpe"], 0.4, label="Full")
    ax.bar(x + 0.2, d_plot["h2_sharpe"], 0.4, label="H2")
    ax.set_xticks(x); ax.set_xticklabels(d_plot["regime"])
    ax.set_title("(d) Regime stratified Sharpe (T+D+L / 11:00 / 200)")
    ax.axhline(2, color="g", ls="--", lw=0.5); ax.axhline(0, color="k", lw=0.5)
    ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT / "result.png", dpi=100)
    print(f"Saved: {OUT/'result.png'}")


if __name__ == "__main__":
    main()
