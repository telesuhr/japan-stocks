"""
ORB (Opening Range Breakout) 戦略検証
- OR期間: 15min / 30min / 60min
- エントリー: ブレイク即時 (次バー寄)
- 方向: Long / Short / 両方
- エグジット: 引け15:25 (Stop=ORレンジの反対端)
- コスト: 4bps

銘柄ユニバース (16):
 半導体5: TEL, アドバン, ディスコ, レーザー, 三菱電機
 非鉄3: 三菱マテ, 住友鉱山, 三井金属
 自動車2: トヨタ, ホンダ
 商社2: 三菱商事, 三井物産
 エネルギー: ENEOS
 その他: 三菱重工(7011), 日本郵船(9101)
 ETF: 1306.T
"""
import psycopg2, pandas as pd, numpy as np
from pathlib import Path
from itertools import combinations

PG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}
OUT = Path(__file__).parent
COST_BPS = 4.0

SYMBOLS = {
    "8035.T": "TEL", "6857.T": "アドバン", "6146.T": "ディスコ",
    "6920.T": "レーザー", "6503.T": "三菱電機",
    "5711.T": "三菱マテ", "5713.T": "住友鉱山", "5706.T": "三井金属",
    "7203.T": "トヨタ", "7267.T": "ホンダ",
    "8058.T": "三菱商事", "8031.T": "三井物産",
    "5020.T": "ENEOS", "7011.T": "三菱重工", "9101.T": "日本郵船",
    "1306.T": "TOPIX_ETF",
}

def load_intraday(sym):
    conn = psycopg2.connect(**PG)
    df = pd.read_sql(
        f"SELECT timestamp, open, high, low, close, volume FROM intraday_data "
        f"WHERE symbol='{sym}' ORDER BY timestamp", conn)
    conn.close()
    if df.empty: return None
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["jst"] = df["timestamp"] + pd.Timedelta(hours=9)
    df = df.dropna(subset=["open", "close"]).set_index("jst").sort_index()
    h, m = df.index.hour, df.index.minute
    morning = (h == 9) | (h == 10) | ((h == 11) & (m <= 30))
    afternoon = ((h == 12) & (m >= 30)) | (h == 13) | (h == 14) | ((h == 15) & (m <= 30))
    return df[morning | afternoon].copy()


def build_orb_trades(df, sym, or_minutes, direction="both", stop_mode="orrange", exit_hm=(15, 25)):
    """
    ORB: 9:00 から or_minutes 間を Opening Range とし、
    その後 Hi/Lo をブレイクしたら 次バー寄でエントリー。
    Stop: ORの反対端 (stop_mode='orrange') or None
    Exit: exit_hm の close (時間決済)
    """
    trades = []
    for date, g in df.groupby(df.index.date):
        g = g.sort_index()
        # Opening Range
        h_ = g.index.hour; m_ = g.index.minute
        min_of_day = h_ * 60 + m_
        or_start = 9 * 60
        or_end = or_start + or_minutes  # exclusive
        or_bars = g[(min_of_day >= or_start) & (min_of_day < or_end)]
        if len(or_bars) < max(3, or_minutes // 2):
            continue
        or_high = or_bars["high"].max()
        or_low = or_bars["low"].min()
        or_open = or_bars.iloc[0]["open"]
        or_range_bps = (or_high - or_low) / or_open * 10000
        # Post-OR bars
        post = g[min_of_day >= or_end].copy()
        if post.empty: continue
        exit_min = exit_hm[0] * 60 + exit_hm[1]
        post = post[(post.index.hour * 60 + post.index.minute) <= exit_min]
        if post.empty: continue

        # Detect first breakout bar
        entry_price = None; entry_dir = 0; entry_time = None
        for ts, bar in post.iterrows():
            # Use high/low of bar to detect break
            if bar["high"] > or_high and direction in ("both", "long"):
                entry_dir = 1; entry_time = ts
                # fill at next bar open (approx bar close + small slippage). Use bar close as entry
                entry_price = or_high  # trigger level as fill (simple)
                break
            if bar["low"] < or_low and direction in ("both", "short"):
                entry_dir = -1; entry_time = ts
                entry_price = or_low
                break
        if entry_dir == 0 or entry_price is None:
            continue

        # Walk forward from entry to exit_hm, check stop
        after = post[post.index >= entry_time]
        exit_price = None; stop_hit = False
        stop_level = or_low if entry_dir == 1 else or_high
        for ts, bar in after.iterrows():
            if stop_mode == "orrange":
                if entry_dir == 1 and bar["low"] <= stop_level:
                    exit_price = stop_level; stop_hit = True; break
                if entry_dir == -1 and bar["high"] >= stop_level:
                    exit_price = stop_level; stop_hit = True; break
        if exit_price is None:
            # time exit
            exit_bars = after[(after.index.hour == exit_hm[0]) & (after.index.minute == exit_hm[1])]
            if exit_bars.empty:
                exit_price = after.iloc[-1]["close"]
            else:
                exit_price = exit_bars.iloc[0]["close"]

        ret_bps = entry_dir * (exit_price / entry_price - 1) * 10000 - COST_BPS
        trades.append({
            "date": date, "sym": sym, "or_minutes": or_minutes,
            "or_range_bps": or_range_bps, "direction": int(entry_dir),
            "entry_time": entry_time.strftime("%H:%M"),
            "entry": entry_price, "exit": exit_price,
            "stop_hit": stop_hit, "ret_bps": ret_bps,
        })
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


def main():
    print("=" * 100)
    print("ORB Opening Range Breakout — 検証")
    print("=" * 100)
    all_data = {}
    for sym in SYMBOLS:
        df = load_intraday(sym)
        if df is not None and len(df) > 1000:
            all_data[sym] = df
    print(f"Loaded {len(all_data)} symbols\n")

    # -------- Part 1: OR期間 × 方向 × Stop の Pooled Grid --------
    print("=" * 100)
    print("[Part 1] Pooled grid  (全銘柄 pool, Stop=ORレンジ)")
    print("=" * 100)
    rows = []
    for or_min in [15, 30, 60]:
        for dirn in ["both", "long", "short"]:
            for stop in ["orrange", "none"]:
                trades = []
                for sym, df in all_data.items():
                    t = build_orb_trades(df, sym, or_min, direction=dirn, stop_mode=stop)
                    if not t.empty: trades.append(t)
                if not trades: continue
                pooled = pd.concat(trades, ignore_index=True)
                full, h1, h2 = split_h12(pooled)
                rows.append({
                    "or_min": or_min, "dir": dirn, "stop": stop,
                    "n": full["n"], "full_sharpe": full["sharpe"], "full_t": full["tstat"],
                    "mean_bp": full["mean"], "wr": full["wr"],
                    "h1_sharpe": h1["sharpe"], "h1_t": h1["tstat"],
                    "h2_sharpe": h2["sharpe"], "h2_t": h2["tstat"],
                })
    grid = pd.DataFrame(rows).sort_values("full_sharpe", ascending=False)
    print(grid.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    grid.to_csv(OUT / "grid.csv", index=False)

    # -------- Part 2: 銘柄別 (ベストパラメータで) --------
    # 先頭のベスト構成を取り出す
    best = grid.iloc[0]
    print(f"\n[Best config] or_min={best['or_min']}  dir={best['dir']}  stop={best['stop']}  sharpe={best['full_sharpe']:.2f}")
    print("\n[Part 2] 銘柄別 (ベスト構成)")
    print("=" * 100)
    rows = []
    for sym, df in all_data.items():
        t = build_orb_trades(df, sym, int(best["or_min"]),
                             direction=best["dir"], stop_mode=best["stop"])
        if t.empty or len(t) < 10:
            rows.append({"sym": sym, "name": SYMBOLS[sym], "n": len(t), "sharpe": np.nan, "tstat": np.nan})
            continue
        s = stats(t["ret_bps"].values)
        rows.append({"sym": sym, "name": SYMBOLS[sym], "n": s["n"],
                     "sharpe": s["sharpe"], "tstat": s["tstat"], "mean_bp": s["mean"], "wr": s["wr"]})
    by_stock = pd.DataFrame(rows).sort_values("sharpe", ascending=False, na_position="last")
    print(by_stock.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    by_stock.to_csv(OUT / "by_stock_best.csv", index=False)

    # -------- Part 3: 銘柄別 (全パラメータ組み合わせ Top) --------
    print("\n[Part 3] 銘柄×パラメータ Top-20")
    print("=" * 100)
    rows = []
    for sym, df in all_data.items():
        for or_min in [15, 30, 60]:
            for dirn in ["both", "long", "short"]:
                for stop in ["orrange", "none"]:
                    t = build_orb_trades(df, sym, or_min, direction=dirn, stop_mode=stop)
                    if t.empty or len(t) < 15: continue
                    full, h1, h2 = split_h12(t)
                    rows.append({
                        "sym": sym, "name": SYMBOLS[sym],
                        "or_min": or_min, "dir": dirn, "stop": stop,
                        "n": full["n"], "sharpe": full["sharpe"], "t": full["tstat"],
                        "h1_sharpe": h1["sharpe"], "h2_sharpe": h2["sharpe"], "h2_t": h2["tstat"],
                    })
    per_stock = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    print(per_stock.head(20).to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    per_stock.to_csv(OUT / "per_stock_params.csv", index=False)

    # -------- Part 4: OR range フィルタ  --------
    # 大きすぎるOR, 小さすぎるORを除外するとどうか
    print("\n[Part 4] ORレンジフィルタ (or_min=30, both, stop=orrange)")
    print("=" * 100)
    # 再構築し or_range_bps でフィルタ
    all_trades = []
    for sym, df in all_data.items():
        t = build_orb_trades(df, sym, 30, direction="both", stop_mode="orrange")
        if not t.empty: all_trades.append(t)
    all_t = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    if not all_t.empty:
        filter_rows = []
        for lo, hi in [(0, 50), (50, 100), (100, 150), (150, 200), (200, 300), (300, 500), (500, 10000)]:
            sub = all_t[(all_t["or_range_bps"] >= lo) & (all_t["or_range_bps"] < hi)]
            if len(sub) < 5: continue
            s = stats(sub["ret_bps"].values)
            filter_rows.append({"range_lo": lo, "range_hi": hi, "n": s["n"],
                                "sharpe": s["sharpe"], "t": s["tstat"], "mean_bp": s["mean"], "wr": s["wr"]})
        fdf = pd.DataFrame(filter_rows)
        print(fdf.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
        fdf.to_csv(OUT / "orrange_filter.csv", index=False)

    # -------- Part 5: Top銘柄サブセット (params固定) --------
    print("\n[Part 5] Top銘柄サブセット最適化")
    print("=" * 100)
    # ベストパラメータで sharpe>0 の銘柄をユニバースとする
    top_candidates = by_stock[by_stock["sharpe"] > 0].head(10)["sym"].tolist()
    print(f"候補 (sharpe>0 上位10): {[SYMBOLS[s] for s in top_candidates]}")
    if len(top_candidates) >= 2:
        rows = []
        for k in range(2, min(6, len(top_candidates)) + 1):
            for combo in combinations(top_candidates, k):
                tl = []
                for s in combo:
                    t = build_orb_trades(all_data[s], s, int(best["or_min"]),
                                         direction=best["dir"], stop_mode=best["stop"])
                    if not t.empty: tl.append(t)
                if not tl: continue
                pooled = pd.concat(tl, ignore_index=True)
                if len(pooled) < 20: continue
                full, h1, h2 = split_h12(pooled)
                rows.append({
                    "combo": "+".join(SYMBOLS[x] for x in combo), "k": k, "n": full["n"],
                    "full_sharpe": full["sharpe"], "full_t": full["tstat"],
                    "h1_sharpe": h1["sharpe"], "h2_sharpe": h2["sharpe"], "h2_t": h2["tstat"],
                })
        subs = pd.DataFrame(rows).sort_values("full_sharpe", ascending=False).head(15)
        print(subs.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
        subs.to_csv(OUT / "subset.csv", index=False)

    print("\nDONE")


if __name__ == "__main__":
    main()
