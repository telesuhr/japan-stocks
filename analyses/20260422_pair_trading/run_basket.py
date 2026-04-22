#!/usr/bin/env python3
"""
セクター内クロスセクショナル Long-Short バスケット

各セクター内で「過去 N 日リターンの Z-score 偏差」で銘柄をランキング。
  - 下位 k 銘柄を Long (過剰下落 → 反発期待)
  - 上位 k 銘柄を Short (過剰上昇 → 調整期待)
  - 保有 h 日、毎日 1/N ずつ組み替え (重複保有で平滑化)

仮説: ペアトレードより銘柄数が多い分、統計的に安定するはず。
"""
import warnings
import numpy as np
import pandas as pd
import pymysql
from pathlib import Path

warnings.filterwarnings("ignore")

MARIA = dict(host="100.92.181.92", port=3306, user="rfnews",
             password="Bleach@924", database="refinitiv_news")

SECTORS = {
    "半導体": ["8035.T", "6146.T", "6920.T", "6857.T", "6963.T", "6526.T",
             "3436.T", "6525.T", "6323.T"],
    "自動車": ["7201.T", "7203.T", "7267.T", "7270.T", "7269.T", "7261.T"],
    "商社":   ["8001.T", "8002.T", "8015.T", "8031.T", "8053.T", "8058.T"],
    "銀行":   ["8306.T", "8316.T", "8411.T"],
    "医薬":   ["4502.T", "4503.T", "4523.T", "4568.T", "4578.T"],
    "海運":   ["9101.T", "9104.T", "9107.T"],
    "電機":   ["6501.T", "6503.T", "6752.T", "6758.T", "6702.T"],
    "通信":   ["9432.T", "9433.T", "9434.T"],
    "非鉄":   ["5711.T", "5706.T", "5713.T"],
    "不動産": ["8801.T", "8802.T"],
}

COST_BPS_PER_SIDE = 2.0  # バスケットなので低コスト (ワンサイド1000万想定)


def load_daily_maria(symbols):
    conn = pymysql.connect(**MARIA)
    placeholders = ",".join(["%s"] * len(symbols))
    q = f"""SELECT symbol, trade_date, close FROM daily_data
            WHERE symbol IN ({placeholders}) ORDER BY symbol, trade_date"""
    df = pd.read_sql(q, conn, params=tuple(symbols))
    conn.close()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    px = df.pivot(index="trade_date", columns="symbol", values="close").astype(float)
    return px


def sector_basket_backtest(sector_name, symbols, px, lookback=20, hold=5, k_frac=0.33,
                           cost_bps=COST_BPS_PER_SIDE):
    """
    各日 t:
      - 過去 lookback 日のリターン Z-score をセクター内で算出
      - Z 下位 k 銘柄 Long / Z 上位 k 銘柄 Short (等ウェイト)
      - hold 日保有 → 均等にずらしてエントリー (毎日 1/hold のターンオーバー)
    """
    avail = [s for s in symbols if s in px.columns]
    if len(avail) < 3:
        return None
    sub = px[avail].dropna(how="all")
    lp = np.log(sub).dropna(how="any")
    if len(lp) < lookback + 30:
        return None

    # 過去 lookback 日リターン
    lookback_ret = lp - lp.shift(lookback)
    # 横断面 Z-score (各日、セクター内)
    zscore = lookback_ret.sub(lookback_ret.mean(axis=1), axis=0)\
                         .div(lookback_ret.std(axis=1).replace(0, np.nan), axis=0)

    k = max(1, int(round(len(avail) * k_frac)))
    daily_ret = lp.diff()  # 1日ログリターン

    # 各日のシグナル → hold 日間の等価ウェイト
    pnl = pd.Series(0.0, index=lp.index)
    rebal_cost = pd.Series(0.0, index=lp.index)
    positions_history = []  # list of (entry_date, long_syms, short_syms, exit_date)

    for i in range(lookback, len(lp) - 1):
        z = zscore.iloc[i]
        ranked = z.dropna().sort_values()
        if len(ranked) < 2 * k:
            continue
        longs = ranked.index[:k].tolist()
        shorts = ranked.index[-k:].tolist()
        entry_i = i
        exit_i = min(i + hold, len(lp) - 1)
        # 保有期間中の PnL (等ウェイト L/S, dollar-neutral)
        for j in range(entry_i + 1, exit_i + 1):
            long_r = daily_ret.iloc[j][longs].mean()
            short_r = daily_ret.iloc[j][shorts].mean()
            # 1 枚あたり PnL = (long_r - short_r) / hold → hold 個のサブポジションで按分
            pnl.iloc[j] += (long_r - short_r) / hold
        # リバランスコスト: エントリー日と exit 日に 2k 銘柄ずつ売買
        rebal_cost.iloc[entry_i] += (cost_bps * 2 / 10000) / hold  # 1/hold サイズ

    net_pnl_bps = pnl * 10000 - rebal_cost * 10000
    return net_pnl_bps


def stats(pnl_bps, label=""):
    x = pnl_bps.dropna()
    arr = x.values
    sd = arr.std() if arr.std() > 0 else 1e-9
    sharpe = arr.mean() / sd * np.sqrt(252)
    t = arr.mean() / sd * np.sqrt(len(arr))
    cum = x.cumsum()
    mdd = (cum.cummax() - cum).max()
    return dict(label=label, days=len(x),
                daily_mean=arr.mean(), sharpe=sharpe, t=t,
                cum_bps=cum.iloc[-1] if len(cum) else 0,
                mdd_bps=mdd)


def main():
    print("=" * 120)
    print("セクター内クロスセクショナル L/S バスケット")
    print("  シグナル: 過去 20日リターンの横断面 Z-score")
    print("  ポジション: 下位33% Long / 上位33% Short (dollar-neutral)")
    print("  保有 5日 / rolling 1/5 turnover / Cost 片側2bps")
    print("=" * 120)

    all_syms = list({s for syms in SECTORS.values() for s in syms})
    px = load_daily_maria(all_syms)
    print(f"データ: {len(px)}日 ({px.index.min().date()} 〜 {px.index.max().date()})\n")

    sector_pnls = {}
    print(f"{'セクター':<12} {'銘柄':>4} {'日数':>5} {'daily(bps)':>11} {'Sharpe':>7} {'t':>6} "
          f"{'Cum':>8} {'MDD':>6}")
    print("-" * 80)
    for sec, syms in SECTORS.items():
        pnl = sector_basket_backtest(sec, syms, px)
        if pnl is None:
            continue
        s = stats(pnl, sec)
        sector_pnls[sec] = pnl
        avail = sum(1 for x in syms if x in px.columns)
        print(f"{sec:<12} {avail:>4d} {s['days']:>5d} {s['daily_mean']:>+11.3f} "
              f"{s['sharpe']:>+7.2f} {s['t']:>+6.2f} "
              f"{s['cum_bps']:>+8.1f} {s['mdd_bps']:>6.1f}")

    # --- 全セクター統合 (Equal-Weight) ---
    if not sector_pnls:
        print("データ不足")
        return
    pnl_mat = pd.DataFrame(sector_pnls).fillna(0.0)
    combined = pnl_mat.mean(axis=1)

    print("\n" + "=" * 120)
    print("全セクター統合 (Equal-Weight Portfolio)")
    print("=" * 120)
    s = stats(combined, "Combined")
    print(f"日次平均: {s['daily_mean']*252:+.1f} bps/年")
    print(f"Sharpe: {s['sharpe']:+.2f}")
    print(f"t-stat: {s['t']:+.2f}")
    print(f"累積: {s['cum_bps']:+.1f} bps / MDD: {s['mdd_bps']:.1f} bps")

    # H1/H2
    mid = combined.index[len(combined) // 2]
    h1 = combined[combined.index < mid]; h2 = combined[combined.index >= mid]
    print(f"\nH1: Sharpe={stats(h1)['sharpe']:+.2f}  Cum={stats(h1)['cum_bps']:+.1f}bps")
    print(f"H2: Sharpe={stats(h2)['sharpe']:+.2f}  Cum={stats(h2)['cum_bps']:+.1f}bps")

    # セクター間相関
    print("\n" + "=" * 120)
    print("セクター間 PnL 相関")
    print("=" * 120)
    corr = pnl_mat.corr()
    m = corr.values.copy(); np.fill_diagonal(m, np.nan)
    print(f"平均 |ρ| = {np.nanmean(np.abs(m)):.3f}")

    # 年次
    print("\n年次:")
    yr = combined.groupby(combined.index.year).agg(
        cum=lambda x: x.sum(),
        sharpe=lambda x: (x.mean()/(x.std() if x.std()>0 else 1e-9))*np.sqrt(252))
    print(yr.to_string(float_format=lambda x: f"{x:+.2f}"))

    combined.to_csv(Path(__file__).parent / "basket_pnl.csv", header=["bps"])

    # 採用判定
    print("\n" + "=" * 120)
    print("採用判定")
    print("=" * 120)
    checks = {
        "Full Sharpe ≥ 1.0": s["sharpe"] >= 1.0,
        "Full t-stat ≥ 2.0": s["t"] >= 2.0,
        "H1 Sharpe ≥ 0.5": stats(h1)["sharpe"] >= 0.5,
        "H2 Sharpe ≥ 0.5": stats(h2)["sharpe"] >= 0.5,
    }
    for k, v in checks.items():
        print(f"  {'✓' if v else '✗'} {k}")


if __name__ == "__main__":
    main()
