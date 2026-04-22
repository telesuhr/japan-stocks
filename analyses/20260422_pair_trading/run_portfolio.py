#!/usr/bin/env python3
"""
統合ペアトレード・ポートフォリオ

run_wf_long.py の best-Sharpe パラメータのうち、H1/H2 とも Sharpe>0 の
ペアを選抜し、Equal-Weight でポートフォリオ化。
  - 相関行列で独立性をチェック
  - 統合 Sharpe / t-stat / MDD
  - H1/H2 OoS での安定性検証
"""
import warnings
from itertools import product
import numpy as np
import pandas as pd
import pymysql
import statsmodels.api as sm
from pathlib import Path

warnings.filterwarnings("ignore")

MARIA = dict(host="100.92.181.92", port=3306, user="rfnews",
             password="Bleach@924", database="refinitiv_news")

# run_wf_long.py 結果から H1>0 & H2>0 を満たしたペア + best params
BOTH_POSITIVE_PAIRS = [
    # (p1, p2, 名称, Entry_Z, Z-win, MaxHold)
    ("7011.T", "7013.T", "重工: MHI/IHI",           2.0, 20, 30),
    ("8306.T", "8411.T", "銀行: MUFG/みずほ",          2.0, 60, 30),
    ("8306.T", "8316.T", "銀行: MUFG/SMFG",         2.0, 60, 20),
    ("6146.T", "6323.T", "半導体: ディスコ/ローツェ",        2.0, 40, 10),
    ("9432.T", "9433.T", "通信: NTT/KDDI",          2.5, 60, 10),
    ("8002.T", "8031.T", "商社: 丸紅/三井物産",           2.5, 60, 10),
    ("5711.T", "5713.T", "非鉄: 三菱マテ/住友金鉱",         2.5, 20, 30),
    ("5711.T", "5706.T", "非鉄: 三菱マテ/三井金",          2.5, 20, 30),
    ("7203.T", "7267.T", "自動車: トヨタ/ホンダ",          1.5, 20, 30),
    ("6501.T", "6503.T", "電機: 日立/三菱電",            2.5, 20, 10),
    ("7270.T", "7269.T", "自動車: スバル/スズキ",          2.5, 40, 10),
    ("9020.T", "9022.T", "鉄道: JR東/JR東海",          2.5, 40, 30),
    ("4503.T", "4578.T", "医薬: アステラス/大塚",          2.5, 40, 10),
    ("6758.T", "6702.T", "電機: ソニー/富士通",           2.0, 40, 10),
    ("5802.T", "5801.T", "電線: 住電/古河",             1.5, 40, 20),
    ("6920.T", "6857.T", "半導体: レーザー/アドバン",         2.5, 40, 10),
    ("8035.T", "6920.T", "半導体: TEL/レーザー",         2.5, 40, 10),
    ("1605.T", "5020.T", "エネルギー: INPEX/ENEOS",    2.5, 20, 30),
]

COST_BPS = 8.0


def load_daily_maria(symbols):
    conn = pymysql.connect(**MARIA)
    placeholders = ",".join(["%s"] * len(symbols))
    q = f"""SELECT symbol, trade_date, close FROM daily_data
            WHERE symbol IN ({placeholders}) ORDER BY symbol, trade_date"""
    df = pd.read_sql(q, conn, params=tuple(symbols))
    conn.close()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.pivot(index="trade_date", columns="symbol", values="close").astype(float)


def rolling_beta(y, x, w):
    betas = pd.Series(index=y.index, dtype=float)
    for i in range(w, len(y)):
        Y = y.iloc[i-w:i]; X = sm.add_constant(x.iloc[i-w:i])
        try: betas.iloc[i] = sm.OLS(Y, X).fit().params.iloc[1]
        except Exception: pass
    return betas


def backtest_with_daily_pnl(p1, p2, px, bw=60, zw=40, entry_z=2.0, exit_z=0.5,
                             stop_z=4.0, max_hold=20, cost=COST_BPS):
    """バックテストに加え、保有期間中を按分した日次 PnL 系列も返す"""
    lp = np.log(px[[p1, p2]].dropna())
    if len(lp) < bw + zw + 30:
        return None, None

    betas = rolling_beta(lp[p1], lp[p2], bw)
    spread = pd.Series(index=lp.index, dtype=float)
    for i in range(len(lp)):
        b = betas.iloc[i]
        if pd.notna(b):
            spread.iloc[i] = lp[p1].iloc[i] - b * lp[p2].iloc[i]
    mu = spread.rolling(zw).mean()
    sd = spread.rolling(zw).std()
    z = (spread - mu) / sd

    daily_pnl = pd.Series(0.0, index=lp.index)
    trades = []
    pos = 0; ei = None; es = None; ez = None; eb = None
    for i in range(bw + zw, len(spread)):
        zi = z.iloc[i]
        if pd.isna(zi):
            continue
        if pos == 0:
            if zi >= entry_z:
                pos = -1; ei = i; es = spread.iloc[i]; ez = zi; eb = betas.iloc[i]
            elif zi <= -entry_z:
                pos = 1; ei = i; es = spread.iloc[i]; ez = zi; eb = betas.iloc[i]
        else:
            hold = i - ei; r = None
            if abs(zi) < exit_z: r = "MR"
            elif abs(zi) > stop_z: r = "STOP"
            elif hold >= max_hold: r = "TIME"
            if r:
                sn = lp[p1].iloc[i] - eb * lp[p2].iloc[i]
                pnl = pos * (sn - es)
                gross = pnl * 10000 / (1 + abs(eb))
                net = gross - cost
                trades.append(dict(
                    entry=spread.index[ei], exit=spread.index[i],
                    hold=hold, net=net, reason=r))
                # 日次 PnL に按分: 出口日に net を計上 (trade-level simplicity)
                daily_pnl.iloc[i] += net
                pos = 0
    return pd.DataFrame(trades), daily_pnl


def portfolio_stats(daily_pnl, label=""):
    """日次 PnL series からポートフォリオ統計"""
    x = daily_pnl.dropna()
    active = x[x != 0]
    arr = x.values
    trading_days = len(x[x.index >= x[x != 0].index.min()]) if (x != 0).any() else len(x)
    mean = arr.mean()
    sd = arr.std() if arr.std() > 0 else 1e-9
    sharpe = mean / sd * np.sqrt(252)
    t = mean / sd * np.sqrt(len(arr))
    cum = x.cumsum()
    mdd = (cum.cummax() - cum).max()
    wr = (active > 0).mean() * 100 if len(active) else 0
    return dict(
        label=label, days=len(x), trades=len(active),
        daily_mean=mean, daily_sd=sd,
        sharpe=sharpe, t=t, wr=wr,
        cum_bps=cum.iloc[-1] if len(cum) else 0,
        mdd_bps=mdd)


def main():
    print("=" * 120)
    print("統合ペアトレード・ポートフォリオ (両期プラスのペアを Equal-Weight)")
    print("  Cost 8bps/往復  /  β trailing 60日  /  Entry・ZW・HD はペアごと最適値")
    print("=" * 120)

    syms = list({s for p1, p2, *_ in BOTH_POSITIVE_PAIRS for s in (p1, p2)})
    px = load_daily_maria(syms)
    print(f"データ: {len(px)}日 ({px.index.min().date()} 〜 {px.index.max().date()})\n")

    # 各ペアで daily_pnl を計算
    pair_pnls = {}   # label -> daily_pnl series
    pair_trades = {}
    print(f"{'ペア':<26} {'N':>4} {'Sharpe':>7} {'t':>6} {'WR':>5} {'MDD':>7} {'Cum':>8}")
    print("-" * 80)
    for p1, p2, lbl, ez, zw, hd in BOTH_POSITIVE_PAIRS:
        td, dp = backtest_with_daily_pnl(
            p1, p2, px, zw=zw, entry_z=ez, max_hold=hd,
            exit_z=max(0.3, ez*0.25))
        if td is None or len(td) < 20:
            continue
        s = portfolio_stats(dp, lbl)
        pair_pnls[lbl] = dp
        pair_trades[lbl] = td
        print(f"{lbl:<26} {s['trades']:>4d} {s['sharpe']:>+7.2f} {s['t']:>+6.2f} "
              f"{s['wr']:>5.1f} {s['mdd_bps']:>7.1f} {s['cum_bps']:>+8.1f}")

    # 全ペアを共通インデックスに揃える
    pnl_mat = pd.DataFrame(pair_pnls).fillna(0.0)

    # --- 相関行列 (trade日のみ) ---
    print("\n" + "=" * 120)
    print("ペア間 PnL 相関 (平均絶対相関)")
    print("=" * 120)
    corr = pnl_mat.corr()
    # 対角除外の平均絶対相関
    m = corr.values.copy()
    np.fill_diagonal(m, np.nan)
    avg_abs_corr = np.nanmean(np.abs(m))
    print(f"平均 |ρ| = {avg_abs_corr:.3f}  (低いほど分散効果大)")
    # Top 5 相関ペア
    pairs_corr = []
    names = corr.columns.tolist()
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            pairs_corr.append((names[i], names[j], corr.iloc[i, j]))
    pairs_corr.sort(key=lambda x: abs(x[2]), reverse=True)
    print("\nTop-10 高相関ペア:")
    for a, b, c in pairs_corr[:10]:
        print(f"  {c:+.3f}  {a}  ×  {b}")

    # --- Equal-Weight ポートフォリオ ---
    print("\n" + "=" * 120)
    print("Equal-Weight ポートフォリオ統計")
    print("=" * 120)
    n_pairs = len(pnl_mat.columns)
    port_pnl = pnl_mat.sum(axis=1) / n_pairs  # 等配分で1/N
    s = portfolio_stats(port_pnl, "Portfolio")
    print(f"ペア数: {n_pairs}")
    print(f"取引日数: {s['days']}  /  PnL発生日: {s['trades']}")
    print(f"日次平均: {s['daily_mean']*252:+.1f} bps/年 (= {s['daily_mean']:+.3f} bps/日)")
    print(f"年率 Sharpe: {s['sharpe']:+.2f}")
    print(f"t-stat: {s['t']:+.2f}")
    print(f"累積 PnL: {s['cum_bps']:+.1f} bps")
    print(f"Max Drawdown: {s['mdd_bps']:.1f} bps")

    # H1/H2 分割
    print("\n" + "-" * 80)
    print("H1/H2 時間分割")
    print("-" * 80)
    mid = port_pnl.index[len(port_pnl) // 2]
    h1 = port_pnl[port_pnl.index < mid]
    h2 = port_pnl[port_pnl.index >= mid]
    for name, s_pnl in [("H1 (~前半5.5年)", h1), ("H2 (後半5.5年)", h2)]:
        s = portfolio_stats(s_pnl, name)
        print(f"{name:<20} Sharpe={s['sharpe']:+.2f}  t={s['t']:+.2f}  "
              f"Cum={s['cum_bps']:+.1f}bps  MDD={s['mdd_bps']:.1f}bps")

    # --- 年次パフォーマンス ---
    print("\n" + "=" * 120)
    print("年次パフォーマンス")
    print("=" * 120)
    yr = port_pnl.groupby(port_pnl.index.year).agg(
        cum_bps=lambda x: x.sum(),
        sharpe=lambda x: (x.mean() / (x.std() if x.std()>0 else 1e-9)) * np.sqrt(252),
        trades=lambda x: (x != 0).sum())
    print(yr.to_string(float_format=lambda x: f"{x:+.2f}"))

    # --- 保存 ---
    port_pnl.to_csv(Path(__file__).parent / "portfolio_pnl.csv", header=["bps"])
    pnl_mat.to_csv(Path(__file__).parent / "pair_pnl_matrix.csv")
    print("\nSaved: portfolio_pnl.csv / pair_pnl_matrix.csv")

    # --- 採用判定 ---
    print("\n" + "=" * 120)
    print("採用判定")
    print("=" * 120)
    port_sh = portfolio_stats(port_pnl)["sharpe"]
    port_t = portfolio_stats(port_pnl)["t"]
    h1_sh = portfolio_stats(h1)["sharpe"]
    h2_sh = portfolio_stats(h2)["sharpe"]
    checks = {
        "Full Sharpe ≥ 1.0": port_sh >= 1.0,
        "Full t-stat ≥ 2.0": port_t >= 2.0,
        "H1 Sharpe ≥ 0.5": h1_sh >= 0.5,
        "H2 Sharpe ≥ 0.5": h2_sh >= 0.5,
    }
    for k, v in checks.items():
        print(f"  {'✓' if v else '✗'} {k}")
    if all(checks.values()):
        print("\n→ 採用候補！ 次段階: パラメータ安定性 / トランザクションコスト感応度")
    else:
        print("\n→ 未達。スパース化 (上位N選抜) や 重み最適化を検討")


if __name__ == "__main__":
    main()
