#!/usr/bin/env python3
"""
ペアトレード日次分析 — Phase 1+2

1. 各セクター内の全ペアで相関・コインテグレーション・スプレッドMR特性を計算
2. 上位ペアをバックテスト (Z-score MR: Entry|Z|>2 / Exit|Z|<0.5 / Stop|Z|>4)
3. Sharpe/PF/WR/コスト後を出力
"""
import warnings
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
import psycopg2
from statsmodels.tsa.stattools import adfuller, coint
import statsmodels.api as sm

warnings.filterwarnings("ignore")

PG = dict(host="localhost", port=5432, user="postgres", dbname="market_data")

SECTORS = {
    "半導体装置": ["8035.T", "6146.T", "6920.T", "6857.T", "6963.T", "6526.T", "3436.T", "6525.T", "6323.T"],
    "非鉄金属": ["5711.T", "5706.T", "5713.T"],
    "電線": ["5801.T", "5802.T", "5803.T"],
    "自動車": ["7201.T", "7203.T", "7267.T", "7270.T", "7269.T", "7261.T"],
    "商社": ["8001.T", "8002.T", "8015.T", "8031.T", "8053.T", "8058.T", "8267.T"],
    "銀行": ["8306.T", "8316.T", "8411.T", "8308.T", "8354.T"],
    "通信": ["9432.T", "9433.T", "9434.T"],
    "鉄道": ["9020.T", "9022.T"],
    "不動産": ["8801.T", "8802.T", "8830.T"],
    "海運": ["9101.T", "9104.T", "9107.T"],
    "ビール": ["2502.T", "2503.T"],
    "医薬": ["4502.T", "4503.T", "4523.T", "4568.T", "4578.T", "4151.T", "4188.T"],
    "重工": ["7011.T", "7012.T", "7013.T"],
    "電機": ["6501.T", "6503.T", "6594.T", "6752.T", "6758.T", "6702.T"],
    "化粧品・食品": ["2801.T", "2802.T", "2914.T", "4452.T"],
    "不動産投資": ["8801.T", "8802.T", "1925.T", "1928.T"],
    "小売": ["9983.T", "8267.T", "3382.T"],
    "保険": ["8766.T", "8750.T"],
    "鉱山": ["1605.T", "5020.T"],  # エネルギー
}

COST_BPS_PER_SIDE = 4.0  # 片側 4bps → ペア往復 = 16bps


def load_daily_close(symbols):
    """intraday_data から日次終値 (15:30 JST バー) を近似取得。
    実運用では daily_stats を使うが、揃って取れる方を使う。"""
    conn = psycopg2.connect(**PG)
    frames = {}
    for s in symbols:
        df = pd.read_sql(
            "SELECT timestamp, close FROM intraday_data WHERE symbol=%s ORDER BY timestamp",
            conn, params=(s,))
        if df.empty:
            continue
        df["jst"] = pd.to_datetime(df["timestamp"]) + pd.Timedelta(hours=9)
        df = df.set_index("jst").sort_index()
        # 日次終値 = 各日の最終バー close
        daily = df["close"].groupby(df.index.date).last()
        daily.index = pd.to_datetime(daily.index)
        frames[s] = daily
    conn.close()
    if not frames:
        return pd.DataFrame()
    px = pd.concat(frames, axis=1).dropna(how="any")  # 全銘柄揃う日のみ
    return px


def analyze_pair(p1, p2, px, lookback=60):
    """ペア統計を計算: 相関、ADF (スプレッド)、コインテグレーション、β"""
    lp = np.log(px[[p1, p2]].dropna())
    if len(lp) < 80:
        return None
    ret = lp.diff().dropna()
    corr = ret[p1].corr(ret[p2])

    # Hedge ratio β: p1 = α + β p2 + ε
    X = sm.add_constant(lp[p2])
    res = sm.OLS(lp[p1], X).fit()
    beta = res.params[p2]
    alpha = res.params["const"]
    spread = lp[p1] - beta * lp[p2] - alpha

    # ADF on spread
    try:
        adf_p = adfuller(spread, maxlag=10, autolag="AIC")[1]
    except Exception:
        adf_p = np.nan

    # Engle-Granger coint
    try:
        coint_p = coint(lp[p1], lp[p2])[1]
    except Exception:
        coint_p = np.nan

    # Hurst exponent (0<H<0.5: 平均回帰性, H=0.5: ランダム, H>0.5: 持続)
    def hurst(ts, max_lag=20):
        lags = range(2, max_lag)
        tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
        return np.polyfit(np.log(lags), np.log(tau), 1)[0]
    try:
        H = hurst(spread.values)
    except Exception:
        H = np.nan

    return dict(p1=p1, p2=p2, n=len(lp), corr=corr, beta=beta,
                adf_p=adf_p, coint_p=coint_p, hurst=H,
                spread_mean=spread.mean(), spread_std=spread.std())


def backtest_pair(p1, p2, px, window=40, entry_z=2.0, exit_z=0.5, stop_z=4.0,
                  max_hold=20, cost_bps=COST_BPS_PER_SIDE * 2):
    """ローリングZスコアのMRペアトレード。
    Long spread = Long p1 / Short (β × p2)。
    Trade unit: 1 σ スプレッドあたりの bps を p1 + p2 の平均ログリターンで表現。
    """
    lp = np.log(px[[p1, p2]].dropna())
    if len(lp) < window + 30:
        return None

    X = sm.add_constant(lp[p2])
    res = sm.OLS(lp[p1], X).fit()
    beta = res.params[p2]
    spread = lp[p1] - beta * lp[p2]
    mu = spread.rolling(window).mean()
    sd = spread.rolling(window).std()
    z = (spread - mu) / sd

    trades = []
    pos = 0; entry_idx = None; entry_spread = None; entry_z_val = None
    for i in range(window, len(spread)):
        zi = z.iloc[i]
        if pos == 0:
            if zi >= entry_z:
                pos = -1; entry_idx = i; entry_spread = spread.iloc[i]; entry_z_val = zi
            elif zi <= -entry_z:
                pos = 1; entry_idx = i; entry_spread = spread.iloc[i]; entry_z_val = zi
        else:
            hold = i - entry_idx
            # エグジット条件
            exit_reason = None
            if abs(zi) < exit_z:
                exit_reason = "MR"
            elif abs(zi) > stop_z:
                exit_reason = "STOP"
            elif hold >= max_hold:
                exit_reason = "TIME"
            if exit_reason:
                # P&L 定義:
                #   pos = +1 (Long spread)  → スプレッド上昇で収益
                #   pos = -1 (Short spread) → スプレッド下降で収益
                # Entry の z-score が正 (高すぎ) → pos=-1 (下落期待)
                #   → z_t が 0 に戻るとき spread 減少 → +収益
                pnl_log = pos * (spread.iloc[i] - entry_spread)
                # Long p1 (+1) / Short β p2 (-β): グロスノーショナル = 1 + |β|
                gross_notional = 1 + abs(beta)
                gross_bps = pnl_log * 10000 / gross_notional  # ノーショナル正規化
                net_bps = gross_bps - cost_bps
                trades.append(dict(
                    entry_date=spread.index[entry_idx], exit_date=spread.index[i],
                    entry_z=entry_z_val, exit_z=zi, hold=hold,
                    pos=pos, gross_bps=gross_bps, net_bps=net_bps,
                    reason=exit_reason))
                pos = 0; entry_idx = None

    if not trades:
        return dict(p1=p1, p2=p2, n_trades=0, beta=beta)
    td = pd.DataFrame(trades)
    arr = td["net_bps"].values
    sharpe = arr.mean() / arr.std() * np.sqrt(252 / td["hold"].mean()) if arr.std() > 0 else 0
    wr = (arr > 0).mean() * 100
    pf = arr[arr > 0].sum() / abs(arr[arr < 0].sum()) if (arr < 0).any() else np.inf
    t_stat = arr.mean() / arr.std() * np.sqrt(len(arr)) if arr.std() > 0 else 0
    return dict(p1=p1, p2=p2, beta=beta, n_trades=len(td),
                mean_bps=arr.mean(), median_bps=np.median(arr),
                sharpe=sharpe, wr=wr, pf=pf, t_stat=t_stat,
                avg_hold=td["hold"].mean(),
                gross_mean=td["gross_bps"].mean(),
                wins_mr=int((td["reason"] == "MR").sum()),
                stops=int((td["reason"] == "STOP").sum()),
                times=int((td["reason"] == "TIME").sum()))


def main():
    print("=" * 90)
    print("ペアトレード日次分析 — Phase 1 (スクリーニング)")
    print("=" * 90)

    all_stats = []
    for sector, syms in SECTORS.items():
        px = load_daily_close(syms)
        if px.empty or len(px.columns) < 2:
            continue
        print(f"\n[{sector}]  銘柄数: {len(px.columns)}  データ日数: {len(px)}")
        pair_rows = []
        for p1, p2 in combinations(px.columns, 2):
            r = analyze_pair(p1, p2, px)
            if r:
                r["sector"] = sector
                pair_rows.append(r)
        if not pair_rows:
            continue
        df = pd.DataFrame(pair_rows).sort_values("coint_p")
        print(df[["p1", "p2", "n", "corr", "beta", "adf_p", "coint_p", "hurst"]]
              .head(8).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        all_stats.append(df)

    if not all_stats:
        print("データ不足で終了")
        return
    all_df = pd.concat(all_stats, ignore_index=True)
    all_df.to_csv(Path(__file__).parent / "screening.csv", index=False)
    print("\n" + "=" * 90)
    print("スクリーニング結果 (coint_p < 0.10 & hurst < 0.45 & corr > 0.6)")
    print("=" * 90)
    good = all_df[(all_df["coint_p"] < 0.10) & (all_df["hurst"] < 0.45)
                  & (all_df["corr"] > 0.60)].sort_values("coint_p")
    print(good[["sector", "p1", "p2", "corr", "beta", "coint_p", "hurst"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # Phase 2: バックテスト
    print("\n" + "=" * 90)
    print("Phase 2 — 有望ペアのバックテスト (Entry|Z|>2 / Exit|Z|<0.5 / Stop|Z|>4 / MaxHold=20日)")
    print("=" * 90)
    candidates = good if len(good) > 0 else all_df.sort_values("coint_p").head(30)
    bt_rows = []
    for _, r in candidates.iterrows():
        sector = r["sector"]; p1 = r["p1"]; p2 = r["p2"]
        syms = SECTORS[sector]
        px = load_daily_close(syms)
        bt = backtest_pair(p1, p2, px)
        if bt and bt.get("n_trades", 0) > 0:
            bt["sector"] = sector
            bt_rows.append(bt)
    if not bt_rows:
        print("バックテスト対象なし")
        return
    bt_df = pd.DataFrame(bt_rows).sort_values("sharpe", ascending=False)
    bt_df.to_csv(Path(__file__).parent / "backtest.csv", index=False)
    print(bt_df[["sector", "p1", "p2", "n_trades", "mean_bps",
                 "sharpe", "t_stat", "wr", "pf", "avg_hold",
                 "wins_mr", "stops", "times"]]
          .to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print("\n" + "=" * 90)
    print("採用候補 (Sharpe≥2 & N≥30 & t≥2)")
    print("=" * 90)
    adopt = bt_df[(bt_df["sharpe"] >= 2.0) & (bt_df["n_trades"] >= 30) & (bt_df["t_stat"] >= 2.0)]
    if len(adopt):
        print(adopt.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    else:
        print("なし — 基準未達。次段階としてイントラデイ検証 or パラメータ調整を検討")


if __name__ == "__main__":
    main()
