#!/usr/bin/env python3
"""
D: 既存戦略 PnL × アルゴ活動度の日次相関
TOPIX 全体の「アルゴ支配度」を日次指標化し、8 戦略 PnL との相関を取る。

日次アルゴ活動度 (TOPIX ベースで):
 - topix_rv          : TOPIX 日内 volatility (bps)
 - topix_close_skew  : 引け前 30 分のリターン÷日中リターンの絶対値
 - topix_twap_pulse  : TOPIX volume lag-5 自己相関 (その日)
 - topix_trend       : TOPIX 日中 ret の AR(1)
 - topix_open_gap    : 前日 close → 寄り ギャップ %
 - semi_disp         : 8035/6857/5713/5802 の 5 分ret横断分散 (セクター分散)
"""
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import psycopg2

warnings.filterwarnings("ignore")
PG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}
ROOT = Path("/Users/Yusuke/claude-code/japan-stocks/analyses")
STRAT_CSV = ROOT / "20260422_pair_trading/all_strategies_daily_pnl.csv"


def load(sym):
    conn = psycopg2.connect(**PG)
    df = pd.read_sql(
        f"SELECT timestamp,open,high,low,close,volume FROM intraday_data "
        f"WHERE symbol='{sym}' ORDER BY timestamp", conn)
    conn.close()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["jst"] = df["timestamp"] + pd.Timedelta(hours=9)
    df = df.set_index("jst").sort_index().dropna(subset=["open"])
    h = df.index.hour; m = df.index.minute
    mask = ((h == 9) | (h == 10) |
            ((h == 11) & (m <= 30)) |
            ((h == 12) & (m >= 30)) |
            (h == 13) | (h == 14) |
            ((h == 15) & (m <= 30)))
    return df[mask]


def daily_topix_metrics(idx_df):
    rows = []
    for d, g in idx_df.groupby(idx_df.index.date):
        if len(g) < 30: continue
        r = g["close"].pct_change().dropna()
        day_ret = g["close"].iloc[-1] / g["open"].iloc[0] - 1
        last30 = g.between_time("15:00", "15:30")
        last30_r = last30["close"].iloc[-1] / last30["close"].iloc[0] - 1 if len(last30) > 5 else 0
        # open gap
        prev_close = g["close"].shift(1).dropna()
        v = g["volume"].values
        v0 = v[:-5]; v1 = v[5:]
        pulse = np.corrcoef(v0, v1)[0, 1] if len(v0) > 10 and np.std(v0) > 0 and np.std(v1) > 0 else np.nan
        rows.append({
            "date": pd.Timestamp(d),
            "topix_rv": r.std() * 1e4,
            "topix_day_ret": day_ret * 1e4,
            "topix_close_skew": abs(last30_r) / (abs(day_ret) + 1e-6),
            "topix_twap_pulse": pulse,
            "topix_trend": r.autocorr(lag=1) if len(r) > 20 else np.nan,
        })
    return pd.DataFrame(rows).set_index("date")


def daily_semi_disp():
    semis = ["8035.T", "6857.T", "5713.T", "5802.T"]
    by_day = {}
    for s in semis:
        d = load(s)
        for day, g in d.groupby(d.index.date):
            c = g["close"].resample("5min").last().dropna()
            r = c.pct_change().dropna()
            if len(r) < 10: continue
            key = pd.Timestamp(day)
            by_day.setdefault(key, {})[s] = r.std()
    rows = []
    for day, d in by_day.items():
        if len(d) < 3: continue
        rows.append({"date": day, "semi_disp": np.std(list(d.values())) * 1e4})
    return pd.DataFrame(rows).set_index("date")


def main():
    print("=" * 120)
    print("D. 既存 8 戦略 PnL × アルゴ活動度 相関")
    print("=" * 120)

    if not STRAT_CSV.exists():
        print(f"  ✗ {STRAT_CSV} not found"); return
    pnl = pd.read_csv(STRAT_CSV, index_col=0, parse_dates=True)
    print(f"  戦略 PnL: {pnl.shape}")

    idx = load(".TOPX")
    tm = daily_topix_metrics(idx)
    sd = daily_semi_disp()
    feats = tm.join(sd, how="outer")
    print(f"  アルゴ活動度: {feats.shape}")

    # 結合
    aligned = pnl.join(feats, how="inner").dropna(how="any")
    print(f"  共通期間: {aligned.shape[0]} days")

    strat_cols = pnl.columns.tolist()
    feat_cols = feats.columns.tolist()

    print("\n" + "=" * 120)
    print("【Pearson 相関: 戦略日次 PnL × アルゴ活動度】")
    print("=" * 120)
    corr_tbl = pd.DataFrame(index=strat_cols, columns=feat_cols, dtype=float)
    for s in strat_cols:
        for f in feat_cols:
            x = aligned[s]; y = aligned[f]
            mk = x.notna() & y.notna() & (x != 0)
            if mk.sum() < 20:
                corr_tbl.loc[s, f] = np.nan; continue
            corr_tbl.loc[s, f] = x[mk].corr(y[mk])
    print(corr_tbl.round(3).to_string())

    print("\n" + "=" * 120)
    print("【注目ペア (|ρ| > 0.15)】")
    print("=" * 120)
    hits = []
    for s in strat_cols:
        for f in feat_cols:
            c = corr_tbl.loc[s, f]
            if pd.notna(c) and abs(c) > 0.15:
                hits.append((s, f, c))
    for s, f, c in sorted(hits, key=lambda x: -abs(x[2])):
        print(f"  {s:<25} × {f:<20} ρ = {c:+.3f}")
    if not hits:
        print("  該当なし")

    # レジーム分析: 各指標の high/low 日で戦略 Sharpe 比較
    print("\n" + "=" * 120)
    print("【レジーム別 戦略 Sharpe (アルゴ活動度 上位 30% vs 下位 30%)】")
    print("=" * 120)
    for f in feat_cols:
        print(f"\n  ── {f} ──")
        q33, q67 = aligned[f].quantile([0.33, 0.67])
        lo_days = aligned[aligned[f] <= q33]
        hi_days = aligned[aligned[f] >= q67]
        header = f"    {'戦略':<25} {'LO Sharpe':>11} {'HI Sharpe':>11} {'Δ':>9}  n_lo n_hi"
        print(header)
        for s in strat_cols:
            lo = lo_days[s]; hi = hi_days[s]
            lo_a = lo[lo != 0]; hi_a = hi[hi != 0]
            if len(lo_a) < 10 or len(hi_a) < 10: continue
            sh_lo = lo_a.mean() / lo_a.std() * np.sqrt(252) if lo_a.std() > 0 else 0
            sh_hi = hi_a.mean() / hi_a.std() * np.sqrt(252) if hi_a.std() > 0 else 0
            print(f"    {s:<25} {sh_lo:>+11.2f} {sh_hi:>+11.2f} {sh_hi-sh_lo:>+9.2f}  "
                  f"{len(lo_a):>4} {len(hi_a):>4}")

    corr_tbl.to_csv("strategy_algo_corr.csv")
    aligned.to_csv("strategy_algo_merged.csv")
    print("\n→ strategy_algo_corr.csv / strategy_algo_merged.csv 保存")


if __name__ == "__main__":
    main()
