"""
マルチペア並列ポートフォリオ + Z閾値最適化

ロジック:
  - 04 で確認した上位ペアを並列運用
  - 各ペアは entry_z / exit_z パラメータで再最適化
  - 各日のポートフォリオPnL = Σ(各ペアのPnL) / N_pairs (ドル等量配分)
  - 同一銘柄が複数ペアに登場する場合、ネットエクスポージャ計算は厳密にしない
    (実運用では銘柄ベースのネッティングで取引コスト削減可)

評価:
  - entry_z ∈ {2.0, 2.5, 3.0} × exit_z ∈ {0.0, 0.5} のグリッド
  - 各 (entry_z, exit_z) の組合せで Top-K (K=3, 5, 8) ペア並列のSharpe
  - コスト感度 (5, 8, 10, 12 bps 片道)
"""

import os
import psycopg2
import pandas as pd
import numpy as np
from itertools import combinations
from datetime import time as dtime

PG_CONFIG = {
    "host": os.environ.get("PGHOST", "omen"),
    "port": int(os.environ.get("PGPORT", 5432)),
    "user": os.environ.get("PGUSER", "postgres"),
    "dbname": os.environ.get("PGDATABASE", "market_data"),
}

SYMBOLS = {
    "57060": "三井金属", "57110": "三菱マテリアル", "57130": "住友金属鉱山",
    "57140": "DOWA HD", "50160": "JX金属",
    "58010": "古河電工", "58020": "住友電工", "58030": "フジクラ",
}
START = "2025-04-01"
END = "2026-05-21"


def fetch_minute_close(code: str) -> pd.Series:
    conn = psycopg2.connect(**PG_CONFIG)
    sql = "SELECT ts, close FROM stocks_intraday WHERE code=%s AND ts>=%s AND ts<=%s ORDER BY ts"
    df = pd.read_sql(sql, conn, params=(code, START, END + " 23:59:59"))
    conn.close()
    df["ts"] = pd.to_datetime(df["ts"])
    return df.set_index("ts")["close"].rename(code)


def fetch_daily_close(code: str) -> pd.Series:
    conn = psycopg2.connect(**PG_CONFIG)
    sql = "SELECT date, close FROM stocks_daily WHERE code=%s AND date BETWEEN %s AND %s ORDER BY date"
    df = pd.read_sql(sql, conn, params=(code, START, END))
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["close"].rename(code)


def pair_backtest(p_a, p_b, daily_a, daily_b,
                  beta_window=20, entry_z=2.5, exit_z=0.5, cost_oneway=0.0008):
    df = pd.concat([p_a, p_b], axis=1).dropna()
    df.columns = ["A", "B"]
    df["date"] = df.index.normalize()

    common_days = daily_a.index.intersection(daily_b.index)
    la = np.log(daily_a.loc[common_days])
    lb = np.log(daily_b.loc[common_days])
    beta_daily = la.rolling(beta_window).cov(lb) / lb.rolling(beta_window).var()
    beta_daily = beta_daily.shift(1)

    trades = []
    for d, g in df.groupby("date"):
        if d not in beta_daily.index or pd.isna(beta_daily.loc[d]):
            continue
        beta = beta_daily.loc[d]
        if beta <= 0 or beta > 5:
            continue
        spread = np.log(g["A"]) - beta * np.log(g["B"])
        morning = spread.between_time("09:00", "10:00")
        if len(morning) < 30:
            continue
        mu, sd = morning.mean(), morning.std()
        if sd == 0 or pd.isna(sd):
            continue
        z = (spread - mu) / sd
        signal_zone = z.between_time("10:00", "15:25")

        position = 0
        entry_pa = entry_pb = entry_time = None
        for ts, zv in signal_zone.items():
            if pd.isna(zv):
                continue
            pa_now = g.loc[ts, "A"]
            pb_now = g.loc[ts, "B"]
            if position == 0:
                if zv > entry_z:
                    position = -1
                    entry_pa, entry_pb, entry_time = pa_now, pb_now, ts
                elif zv < -entry_z:
                    position = +1
                    entry_pa, entry_pb, entry_time = pa_now, pb_now, ts
            else:
                should_exit = abs(zv) < exit_z or ts.time() >= dtime(15, 25)
                if should_exit:
                    r_a = np.log(pa_now / entry_pa)
                    r_b = np.log(pb_now / entry_pb)
                    pnl = (r_a - r_b) / 2 if position == +1 else (-r_a + r_b) / 2
                    pnl -= cost_oneway * 2
                    trades.append({"date": d, "pnl": pnl, "position": position,
                                   "entry_time": entry_time, "exit_time": ts})
                    position = 0
                    entry_pa = entry_pb = entry_time = None
        # forced close
        if position != 0 and entry_time is not None:
            last_ts = signal_zone.index[-1]
            pa_now, pb_now = g.loc[last_ts, "A"], g.loc[last_ts, "B"]
            r_a = np.log(pa_now / entry_pa)
            r_b = np.log(pb_now / entry_pb)
            pnl = (r_a - r_b) / 2 if position == +1 else (-r_a + r_b) / 2
            pnl -= cost_oneway * 2
            trades.append({"date": d, "pnl": pnl, "position": position,
                           "entry_time": entry_time, "exit_time": last_ts})

    return pd.DataFrame(trades)


def metrics(trades_df, ann=245):
    if len(trades_df) < 5:
        return {}
    n = len(trades_df)
    mu = trades_df["pnl"].mean()
    sd = trades_df["pnl"].std()
    eq = trades_df["pnl"].cumsum()
    dd = (eq - eq.cummax()).min()
    n_days = trades_df["date"].nunique()
    tpd = n / n_days
    sharpe = mu / sd * np.sqrt(ann * tpd) if sd > 0 else 0
    wr = (trades_df["pnl"] > 0).mean() * 100
    pf = (trades_df["pnl"][trades_df["pnl"] > 0].sum() /
          -trades_df["pnl"][trades_df["pnl"] < 0].sum()) if (trades_df["pnl"] < 0).any() else np.inf
    return {
        "N": n, "N_days": n_days, "trades/day": round(tpd, 2),
        "mean_%/trade": mu * 100, "Sharpe": sharpe,
        "Winrate_%": wr, "PF": pf,
        "Cum_%": eq.iloc[-1] * 100, "MaxDD_%": dd * 100,
    }


def main():
    print("=== マルチペア並列 + Zパラメータ最適化 ===\n")
    print("--- データ読み込み ---")
    minutes = {c: fetch_minute_close(c) for c in SYMBOLS}
    dailies = {c: fetch_daily_close(c) for c in SYMBOLS}

    # Step1: 各ペアを各 (entry_z, exit_z) 組合せで評価, コスト=8bps
    # 結果からトップペアを選び、ポートフォリオを構成
    cost = 0.0008
    z_grid = [(2.0, 0.5), (2.5, 0.5), (3.0, 0.5), (2.5, 0.0), (3.0, 0.0)]
    pairs = list(combinations(SYMBOLS.keys(), 2))

    print(f"\n--- 各ペア × (entry_z, exit_z) でバックテスト (コスト {cost*10000:.0f} bps片道) ---")
    cache = {}  # (pair, ez, xz) → trades_df
    rows = []
    for a, b in pairs:
        for ez, xz in z_grid:
            tdf = pair_backtest(minutes[a], minutes[b], dailies[a], dailies[b],
                                entry_z=ez, exit_z=xz, cost_oneway=cost)
            m = metrics(tdf)
            if not m:
                continue
            rows.append({"pair": f"{a}-{b}", "name": f"{SYMBOLS[a]}/{SYMBOLS[b]}",
                         "entry_z": ez, "exit_z": xz, **m})
            cache[(f"{a}-{b}", ez, xz)] = tdf

    df_all = pd.DataFrame(rows)
    df_all = df_all.sort_values("Sharpe", ascending=False)
    print("\n--- 全組合せ Top15 (Sharpe順) ---")
    print(df_all.head(15)[["pair", "name", "entry_z", "exit_z", "N", "Sharpe",
                            "PF", "Cum_%", "MaxDD_%"]].to_string(index=False))

    # Step2: ペアごと最良パラメータ
    best_per_pair = df_all.sort_values("Sharpe", ascending=False).groupby("pair").head(1)
    best_per_pair = best_per_pair.sort_values("Sharpe", ascending=False)
    print(f"\n--- 各ペアの最良パラメータ (Top12) ---")
    print(best_per_pair.head(12)[["pair", "name", "entry_z", "exit_z",
                                    "N", "Sharpe", "PF", "Cum_%", "MaxDD_%"]].to_string(index=False))

    # Step3: ポートフォリオ構成 (Top K ペア並列, 等ウェイト)
    print(f"\n--- ポートフォリオ (Top-K ペア並列, 等ウェイト, コスト 8 bps) ---")
    # 日次PnL DataFrame を構成
    daily_pnls = {}
    for _, row in best_per_pair.iterrows():
        key = (row["pair"], row["entry_z"], row["exit_z"])
        if key not in cache:
            continue
        tdf = cache[key]
        daily = tdf.groupby("date")["pnl"].sum()
        daily_pnls[row["pair"]] = daily

    df_daily = pd.DataFrame(daily_pnls).fillna(0)
    # トップKペアでポートフォリオ
    for K in [3, 5, 8, 12]:
        top_pairs = best_per_pair.head(K)["pair"].tolist()
        avail = [p for p in top_pairs if p in df_daily.columns]
        if not avail:
            continue
        port = df_daily[avail].mean(axis=1)  # 等ウェイト
        n = len(port)
        mu = port.mean()
        sd = port.std()
        sharpe = mu / sd * np.sqrt(245) if sd > 0 else 0
        eq = port.cumsum()
        dd = (eq - eq.cummax()).min()
        wr = (port > 0).mean() * 100
        print(f"  Top-{K:2d} pairs: N_days={n}, daily_mean={mu*100:+.3f}%, "
              f"std={sd*100:.2f}%, Sharpe={sharpe:.2f}, Cum={eq.iloc[-1]*100:+.1f}%, "
              f"DD={dd*100:.1f}%, Winrate={wr:.1f}%")

    # Step4: コスト感度 (best Top-K)
    K = 5
    top_pairs = best_per_pair.head(K)["pair"].tolist()
    print(f"\n--- Top-{K} ポートフォリオ × コスト感度 ---")
    for c_bps in [3, 5, 8, 10, 12, 15]:
        cost = c_bps / 10000
        daily_pnls_c = {}
        for _, row in best_per_pair.head(K).iterrows():
            tdf = pair_backtest(minutes[row["pair"].split("-")[0]],
                                minutes[row["pair"].split("-")[1]],
                                dailies[row["pair"].split("-")[0]],
                                dailies[row["pair"].split("-")[1]],
                                entry_z=row["entry_z"], exit_z=row["exit_z"],
                                cost_oneway=cost)
            if len(tdf) > 0:
                daily_pnls_c[row["pair"]] = tdf.groupby("date")["pnl"].sum()
        df_c = pd.DataFrame(daily_pnls_c).fillna(0)
        port = df_c.mean(axis=1)
        mu, sd = port.mean(), port.std()
        sharpe = mu / sd * np.sqrt(245) if sd > 0 else 0
        eq = port.cumsum()
        dd = (eq - eq.cummax()).min()
        print(f"  cost={c_bps:2d} bps: Sharpe={sharpe:+.2f}, Cum={eq.iloc[-1]*100:+.1f}%, "
              f"DD={dd*100:.1f}%, daily_mean={mu*100:+.3f}%")

    # 保存
    out_dir = os.path.dirname(os.path.abspath(__file__))
    df_all.to_csv(os.path.join(out_dir, "multipair_all_combos.csv"), index=False)
    best_per_pair.to_csv(os.path.join(out_dir, "multipair_best_per_pair.csv"), index=False)
    # 採用ポートフォリオ (Top-5 8bps) の日次PnL を保存
    port5_daily_pnls = {}
    for _, row in best_per_pair.head(5).iterrows():
        key = (row["pair"], row["entry_z"], row["exit_z"])
        if key in cache:
            port5_daily_pnls[row["pair"]] = cache[key].groupby("date")["pnl"].sum()
    df_port = pd.DataFrame(port5_daily_pnls).fillna(0)
    df_port["portfolio"] = df_port.mean(axis=1)
    df_port.to_csv(os.path.join(out_dir, "portfolio_top5_daily_pnl.csv"))
    print(f"\n保存: multipair_*.csv, portfolio_top5_daily_pnl.csv")


if __name__ == "__main__":
    main()
