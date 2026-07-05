"""closing_auction_rebound の健全性再構成。

定義 (strategies/closing_auction_rebound/ 準拠):
- ユニバース: 直近400日平均売買代金 上位200 (期間頭 as-of で固定。日次更新との差は僅少)
- シグナル: close_jump = 15:30 close / 15:24 close - 1 ≤ -50bps
- entry = 当日15:30 引けMOC / exit = 翌営業日 09:00 バー close
- クレンジング: |overnight| > 10% は分割未調整等の異常値として除外 (refined版準拠)
- コスト: 往復 2×COST_ONE_WAY_BPS
"""
import pandas as pd

from . import get_conn, COST_ONE_WAY_BPS

THRESHOLD_BPS = -50.0
TOP_N = 200
NAME = "closing_auction_rebound"


def compute_trades(start_date: str, end_date: str) -> pd.DataFrame:
    conn = get_conn()
    uni = pd.read_sql(f"""
        SELECT code FROM stocks_daily
        WHERE date < %s AND date >= %s::date - INTERVAL '400 days'
          AND turnover_value > 0
        GROUP BY code ORDER BY AVG(turnover_value) DESC LIMIT {TOP_N}
    """, conn, params=(start_date, start_date))
    codes = tuple(uni["code"].tolist())
    if not codes:
        conn.close()
        return pd.DataFrame(columns=["entry_date", "exit_date", "symbol",
                                     "gross_ret", "net_ret"])
    bars = pd.read_sql("""
        SELECT code, ts::date AS d, ts::time AS t, close
        FROM stocks_intraday
        WHERE code IN %s
          AND ts >= %s::date AND ts < %s::date + INTERVAL '7 days'
          AND ts::time IN ('09:00:00','15:24:00','15:30:00')
    """, conn, params=(codes, start_date, end_date))
    conn.close()

    piv = bars.pivot_table(index=["code", "d"], columns="t", values="close")
    piv.columns = [str(c)[:5] for c in piv.columns]
    piv = piv.reset_index()
    days = sorted(piv["d"].unique())
    next_day = {d: days[i + 1] for i, d in enumerate(days[:-1])}

    o900 = piv.set_index(["code", "d"])["09:00"] if "09:00" in piv.columns else None
    rows = []
    for _, r in piv.iterrows():
        d = r["d"]
        if str(d) > end_date or d not in next_day:
            continue
        c24, c30 = r.get("15:24"), r.get("15:30")
        if not (pd.notna(c24) and pd.notna(c30) and c24 > 0 and c30 > 0):
            continue
        jump_bps = (c30 / c24 - 1) * 1e4
        if jump_bps > THRESHOLD_BPS:
            continue
        try:
            ex = o900.loc[(r["code"], next_day[d])]
        except KeyError:
            continue
        if not (pd.notna(ex) and ex > 0):
            continue
        gross = float(ex / c30 - 1)
        if abs(gross) > 0.10:  # 分割未調整等の異常値
            continue
        rows.append({"entry_date": d, "exit_date": next_day[d], "symbol": r["code"],
                     "gross_ret": gross,
                     "net_ret": gross - 2 * COST_ONE_WAY_BPS / 1e4})
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["entry_date", "exit_date", "symbol", "gross_ret", "net_ret"])


def health(start_date: str, end_date: str) -> dict:
    """IS基準(net Sharpe 2.0-2.8)と同じ意味論=日次等加重バスケット・非重複1泊で評価。
    per-trade×√252だと同日複数トレードで水増しされるため日次集計してから√252。"""
    from . import summary_stats
    trades = compute_trades(start_date, end_date)
    if len(trades) == 0:
        return {"strategy": NAME, "n": 0, "sharpe": None, "t_stat": None,
                "win_rate": None, "mean_pct": None, "signal_days": 0}
    daily = trades.groupby("entry_date")["net_ret"].mean()
    stats = summary_stats(daily, NAME)
    stats["strategy"] = NAME
    stats["n"] = len(trades)           # トレード数は表示用に個別件数
    stats["signal_days"] = len(daily)  # Sharpe計算は日次バスケット
    return stats
