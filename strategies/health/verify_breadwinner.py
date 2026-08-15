"""
稼ぎ頭 pre_earnings_drift / earnings_pead の「本当に生きているか」honest検証。

health_check.py の summary_stats は per-trade列に mean/std*sqrt(252) を掛けており、
SUMMARY.md が警告する「per-trade×√252 = 約5倍の過大評価」そのもの。
ここでは以下の honest 指標で再評価する:
  1. 日次EWポートフォリオ収益系列の真のSharpe（重複保有を正しく平均）
  2. per-trade を √(252/平均保有日) で正しく年率化した値との対比
  3. 集中度（mean vs median・上位トレードの寄与）
  4. 期間安定性（前半/後半・月次）
コストは往復4bps（entry/exit各2bps）を日次パスに織り込む。
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
from pathlib import Path
import pandas as pd
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _lib import get_conn  # noqa: E402
from _lib import pre_earnings_drift, earnings_pead  # noqa: E402

ONE_WAY = 2.0 / 10000.0  # 片道2bps


def daily_portfolio_series(trades: pd.DataFrame) -> pd.Series:
    """各トレードの保有期間の日次リターンを再構成し、日次EWポートフォリオ系列を返す。
       entry日: open→close、以降: prev_close→close。entry日/exit日に片道コスト控除。"""
    if trades.empty:
        return pd.Series(dtype=float)
    codes = trades["symbol"].unique().tolist()
    dmin = pd.to_datetime(trades["entry_date"]).min() - pd.Timedelta(days=7)
    dmax = pd.to_datetime(trades["exit_date"]).max()
    conn = get_conn()
    bars = pd.read_sql("""
        SELECT code, date, adj_open, adj_close
        FROM stocks_daily
        WHERE code = ANY(%(codes)s) AND date BETWEEN %(dmin)s AND %(dmax)s
          AND adj_close > 0
        ORDER BY code, date
    """, conn, params={"codes": codes, "dmin": str(dmin.date()), "dmax": str(dmax.date())})
    conn.close()
    bars["date"] = pd.to_datetime(bars["date"])
    bars = bars.sort_values(["code", "date"])
    bars["prev_close"] = bars.groupby("code")["adj_close"].shift(1)

    # 各トレードの日次リターン片を集める: {date: [ret, ...]}
    day_rets = {}
    bar_idx = {c: g.set_index("date") for c, g in bars.groupby("code")}
    for _, t in trades.iterrows():
        c = t["symbol"]
        e = pd.Timestamp(t["entry_date"]); x = pd.Timestamp(t["exit_date"])
        g = bar_idx.get(c)
        if g is None:
            continue
        hold = g.loc[(g.index >= e) & (g.index <= x)]
        if hold.empty:
            continue
        for i, (d, row) in enumerate(hold.iterrows()):
            if i == 0:
                r = row["adj_close"] / row["adj_open"] - 1.0 - ONE_WAY  # entry日: 寄→引, 建てコスト
            else:
                r = row["adj_close"] / row["prev_close"] - 1.0
            if d == x:
                r -= ONE_WAY  # exit日: 決済コスト
            day_rets.setdefault(d, []).append(r)
    if not day_rets:
        return pd.Series(dtype=float)
    ser = pd.Series({d: np.mean(v) for d, v in day_rets.items()}).sort_index()
    return ser


def report(name, mod, start, end):
    trades = mod.compute_trades(start, end)
    n = len(trades)
    print(f"\n{'='*66}\n{name}  期間 {start}〜{end}  トレード数 N={n}")
    if n == 0:
        print("  トレードなし"); return
    net = trades["net_ret"]
    hold_days = (pd.to_datetime(trades["exit_date"]) - pd.to_datetime(trades["entry_date"])).dt.days
    avg_hold = hold_days.mean()

    # per-trade（health_checkの過大評価版）
    sh_inflated = net.mean() / net.std() * np.sqrt(252)
    # per-trade を正しく年率化（保有日換算・重複無視の理論上限）
    sh_proper_pertrade = net.mean() / net.std() * np.sqrt(252 / max(avg_hold, 1))
    t_stat = net.mean() / net.std() * np.sqrt(n)

    print(f"  平均/trade  {net.mean()*100:+.3f}%   中央値 {net.median()*100:+.3f}%   勝率 {(net>0).mean()*100:.1f}%")
    print(f"  平均保有 {avg_hold:.1f}日")
    print(f"  [誇張] per-trade×√252 Sharpe = {sh_inflated:.2f}   ← health_checkの数字")
    print(f"  [補正] per-trade×√(252/hold) Sharpe = {sh_proper_pertrade:.2f}   (t/trade={t_stat:.2f})")

    # 集中度: 上位5トレードを除くと?
    top5 = net.sort_values(ascending=False).head(5).sum()
    total = net.sum()
    print(f"  集中度: 合計net {total*100:+.1f}% のうち上位5トレードが {top5*100:+.1f}%（{top5/total*100:.0f}%）")

    # 日次ポートフォリオ系列（honestなSharpe）
    ser = daily_portfolio_series(trades)
    if len(ser) > 1:
        sh_daily = ser.mean() / ser.std() * np.sqrt(252)
        t_daily = ser.mean() / ser.std() * np.sqrt(len(ser))
        cum = (1 + ser).prod() - 1
        mdd = (( (1+ser).cumprod() / (1+ser).cumprod().cummax() ) - 1).min()
        print(f"  ★[honest] 日次EWポートフォリオ Sharpe = {sh_daily:.2f}  (t={t_daily:.2f}, 稼働{len(ser)}日)")
        print(f"     累積 {cum*100:+.2f}%   最大DD {mdd*100:.2f}%   平均 {ser.mean()*1e4:+.1f}bp/稼働日")

    # 期間安定性（前半/後半）
    trades2 = trades.sort_values("entry_date")
    half = n // 2
    for lbl, sub in [("前半", trades2.iloc[:half]), ("後半", trades2.iloc[half:])]:
        s = sub["net_ret"]
        print(f"     {lbl}: N={len(s)} 平均{s.mean()*100:+.3f}%/trade 勝率{(s>0).mean()*100:.0f}%")


if __name__ == "__main__":
    # 直近60営業日相当 と 1年 の両方で見る
    for start, end, lbl in [("2026-05-21", "2026-08-14", "直近60営業日"),
                            ("2025-08-01", "2026-08-14", "直近1年")]:
        print(f"\n\n############ {lbl} ############")
        report("pre_earnings_drift", pre_earnings_drift, start, end)
        report("earnings_pead", earnings_pead, start, end)
