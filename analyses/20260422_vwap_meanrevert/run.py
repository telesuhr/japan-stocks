"""
VWAP 乖離 mean-revert バッテリー

仮説: 前場終了時点 (11:00 or 11:25) で day-VWAP からの乖離が大きい銘柄は
       引け (15:25) までに VWAP 方向へ回帰する (機関のVWAP執行が引力)

エントリー: 11:00 (or 11:25) に day-VWAP からの乖離 bps が閾値超
            → 逆方向 (上乖離ならShort / 下乖離ならLong)
エグジット: 15:25 成行
コスト: 4bps/トレード (片道2bps × 往復)

対象: 半導体5 + 非鉄3 + 自動車2 + 商社2 (合計12銘柄)
期間: 2024-11 ~ 2026-04 (PostgreSQL intraday_data)
"""

import psycopg2, pandas as pd, numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}
OUT = Path(__file__).parent
COST_BPS = 4.0

SYMBOLS = {
    # 半導体
    "8035.T": "TEL",
    "6857.T": "アドバン",
    "6146.T": "ディスコ",
    "6920.T": "レーザー",
    "6503.T": "三菱電機",
    # 非鉄
    "5711.T": "三菱マテ",
    "5713.T": "住友鉱山",
    "5706.T": "三井金属",
    # 自動車
    "7203.T": "トヨタ",
    "7267.T": "ホンダ",
    # 商社
    "8058.T": "三菱商事",
    "8031.T": "三井物産",
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
    # 取引時間のみ (前場 9:00-11:30 / 後場 12:30-15:30)
    h, m = df.index.hour, df.index.minute
    morning = (h == 9) | (h == 10) | ((h == 11) & (m <= 30))
    afternoon = ((h == 12) & (m >= 30)) | (h == 13) | (h == 14) | ((h == 15) & (m <= 30))
    df = df[morning | afternoon].copy()
    return df


def build_trades(df, entry_time=(11, 0), exit_time=(15, 25), thresh_bps=100):
    """
    entry_time: (hour, minute) でその時刻までの day-VWAP を計算し、乖離を評価
    exit_time: その時刻の close で決済
    thresh_bps: |乖離| >= thresh_bps でエントリー
    """
    trades = []
    for date, g in df.groupby(df.index.date):
        # day-VWAP 累積
        g = g.copy()
        pv = (g["close"] * g["volume"]).cumsum()
        cv = g["volume"].cumsum().replace(0, np.nan)
        g["vwap"] = pv / cv
        g["dev_bps"] = (g["close"] / g["vwap"] - 1) * 10000

        # entry bar
        ent_bars = g[(g.index.hour == entry_time[0]) & (g.index.minute == entry_time[1])]
        if ent_bars.empty:
            continue
        ent = ent_bars.iloc[0]
        ent_price = ent["close"]
        dev = ent["dev_bps"]

        if abs(dev) < thresh_bps:
            continue

        # exit bar
        exi_bars = g[(g.index.hour == exit_time[0]) & (g.index.minute == exit_time[1])]
        if exi_bars.empty:
            # fallback: last bar of the day
            exi = g.iloc[-1]
        else:
            exi = exi_bars.iloc[0]
        exit_price = exi["close"]

        # 逆方向エントリー: dev>0 (上乖離) → Short, dev<0 (下乖離) → Long
        direction = -np.sign(dev)  # +1=Long, -1=Short
        ret_bps = direction * (exit_price / ent_price - 1) * 10000 - COST_BPS

        trades.append(
            {
                "date": date,
                "dev_bps": dev,
                "direction": int(direction),
                "ret_bps": ret_bps,
                "ent_price": ent_price,
                "exit_price": exit_price,
            }
        )
    return pd.DataFrame(trades)


def stats(trades):
    if trades.empty or len(trades) < 5:
        return dict(n=len(trades), mean=np.nan, std=np.nan, sharpe=np.nan, tstat=np.nan, wr=np.nan)
    r = trades["ret_bps"].values
    mean = r.mean()
    std = r.std(ddof=1)
    sharpe = mean / std * np.sqrt(252) if std > 0 else np.nan
    tstat = mean / (std / np.sqrt(len(r))) if std > 0 else np.nan
    wr = (r > 0).mean() * 100
    return dict(n=len(r), mean=mean, std=std, sharpe=sharpe, tstat=tstat, wr=wr)


def main():
    print("=" * 80)
    print("VWAP 乖離 mean-revert バッテリー")
    print("=" * 80)

    # グリッド: エントリー時刻 × 閾値
    entry_grids = [(11, 0), (11, 25), (13, 0), (13, 30)]
    thresh_grids = [50, 100, 150, 200]

    # まず各銘柄ロード
    all_data = {}
    for sym in SYMBOLS:
        df = load_intraday(sym)
        if df is None or len(df) < 100:
            print(f"  SKIP {sym}: no data")
            continue
        all_data[sym] = df
        print(f"  {sym} {SYMBOLS[sym]}: {len(df):,} bars, {df.index.date[0]} ~ {df.index.date[-1]}")
    print()

    # 全組合せ探索
    rows = []
    all_trades_by_config = {}
    for ent in entry_grids:
        for th in thresh_grids:
            pooled = []
            per_stock_stats = {}
            for sym, df in all_data.items():
                trades = build_trades(df, entry_time=ent, exit_time=(15, 25), thresh_bps=th)
                if not trades.empty:
                    trades["sym"] = sym
                    pooled.append(trades)
                    per_stock_stats[sym] = stats(trades)
            if not pooled:
                continue
            pooled_df = pd.concat(pooled, ignore_index=True)
            s = stats(pooled_df)
            rows.append(
                {
                    "entry": f"{ent[0]:02d}:{ent[1]:02d}",
                    "thresh_bps": th,
                    "n": s["n"],
                    "sharpe": s["sharpe"],
                    "tstat": s["tstat"],
                    "mean": s["mean"],
                    "wr": s["wr"],
                }
            )
            all_trades_by_config[(ent, th)] = (pooled_df, per_stock_stats)

    grid = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    grid.to_csv(OUT / "grid.csv", index=False)
    print("=" * 80)
    print("Pooled グリッド結果 (全銘柄合算)")
    print("=" * 80)
    print(grid.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print()

    # Best config を詳細出力
    if len(grid) == 0:
        print("No trades generated.")
        return
    best = grid.iloc[0]
    best_key = ((int(best["entry"].split(":")[0]), int(best["entry"].split(":")[1])), best["thresh_bps"])
    pooled_best, per_stock_best = all_trades_by_config[best_key]

    print("=" * 80)
    print(f"Best config: entry={best['entry']} thresh={best['thresh_bps']}bps")
    print(f"  N={best['n']} Sharpe={best['sharpe']:.2f} t={best['tstat']:.2f} "
          f"mean={best['mean']:.1f}bp WR={best['wr']:.1f}%")
    print("=" * 80)
    print("銘柄別 (Best config):")
    rows2 = []
    for sym, s in per_stock_best.items():
        rows2.append(
            {
                "sym": sym,
                "name": SYMBOLS[sym],
                "n": s["n"],
                "sharpe": s["sharpe"],
                "tstat": s["tstat"],
                "mean": s["mean"],
                "wr": s["wr"],
            }
        )
    by_stock = pd.DataFrame(rows2).sort_values("sharpe", ascending=False)
    by_stock.to_csv(OUT / "by_stock_best.csv", index=False)
    print(by_stock.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print()

    # H1/H2 OoS 分割 (best config)
    pooled_best = pooled_best.sort_values("date").reset_index(drop=True)
    mid = len(pooled_best) // 2
    h1 = pooled_best.iloc[:mid]
    h2 = pooled_best.iloc[mid:]
    print("=" * 80)
    print("H1/H2 OoS (Best config)")
    print("=" * 80)
    for label, sub in [("Full", pooled_best), ("H1", h1), ("H2", h2)]:
        s = stats(sub)
        date_range = f"{sub['date'].min()} ~ {sub['date'].max()}" if len(sub) else ""
        print(f"  {label:5s} N={s['n']:4d} Sharpe={s['sharpe']:+.2f} t={s['tstat']:+.2f} "
              f"mean={s['mean']:+.1f}bp WR={s['wr']:.1f}% [{date_range}]")
    print()

    # Plot: pooled 累積
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    cum = pooled_best.sort_values("date")["ret_bps"].cumsum().values
    ax.plot(cum)
    ax.set_title(f"Pooled cum ret (bps) | Best: entry={best['entry']} th={best['thresh_bps']}")
    ax.set_xlabel("trade #")
    ax.set_ylabel("cum bps")
    ax.grid(alpha=0.3)
    ax.axhline(0, color="k", lw=0.5)

    ax = axes[1]
    by_stock_sorted = by_stock.dropna(subset=["sharpe"]).sort_values("sharpe")
    ax.barh(by_stock_sorted["name"], by_stock_sorted["sharpe"])
    ax.set_title("Sharpe by stock (Best config)")
    ax.set_xlabel("Sharpe")
    ax.axvline(0, color="k", lw=0.5)
    ax.axvline(2.0, color="g", lw=0.5, ls="--", label="採用基準")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "result.png", dpi=100)
    print(f"Saved: {OUT/'result.png'}")


if __name__ == "__main__":
    main()
