"""
lasertec_ma25_support 健全性チェックモジュール

銘柄: 6920 (69200)
シグナル: dd20 ≤ -5%, MA25接触±1%, MA25上昇中, 10日クールダウンあり
Entry: 翌営業日open
Exit : signal + 10営業日 close
コスト: 往復4bps
"""
import pandas as pd
import numpy as np
from . import get_conn, net_ret, summary_stats

SYM = "69200"
MA_PERIOD = 25
DD_THRESH = -5.0
TOUCH_TOL = 0.01  # ±1%
SLOPE_LB = 5
HOLD_DAYS = 10
COOLDOWN = 10


def compute_trades(start_date: str, end_date: str) -> pd.DataFrame:
    conn = get_conn()
    # シグナル判定に必要な過去データも含めて取得
    sd_ext = (pd.Timestamp(start_date) - pd.Timedelta(days=90)).strftime("%Y-%m-%d")
    ed_ext = (pd.Timestamp(end_date) + pd.Timedelta(days=20)).strftime("%Y-%m-%d")
    df = pd.read_sql(
        "SELECT date, open, high, low, close, adj_open, adj_close "
        "FROM stocks_daily WHERE code=%s AND date >= %s AND date <= %s ORDER BY date",
        conn, params=(SYM, sd_ext, ed_ext)
    )
    conn.close()

    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df = df.astype({c: float for c in ["open", "high", "low", "close", "adj_open", "adj_close"]})

    df["ma25"] = df["adj_close"].rolling(MA_PERIOD).mean()
    df["ma25_5d"] = df["ma25"].shift(SLOPE_LB)
    df["hh20"] = df["adj_close"].rolling(20).max()
    df["dd20"] = (df["adj_close"] / df["hh20"] - 1) * 100

    df["touched"] = (
        (df["low"] <= df["ma25"] * (1 + TOUCH_TOL)) &
        (df["high"] >= df["ma25"] * (1 - TOUCH_TOL))
    )
    df["downtrend"] = df["dd20"] <= DD_THRESH
    df["slope_up"] = df["ma25"] > df["ma25_5d"]
    df["signal"] = df["touched"] & df["downtrend"] & df["slope_up"] & df["ma25"].notna()

    bdays = df.index.tolist()

    rows = []
    last_entry_idx = -999
    for i, (ts, row) in enumerate(df.iterrows()):
        d = ts.date()
        if str(d) < start_date or str(d) > end_date:
            continue
        if not row["signal"]:
            continue
        # クールダウン
        if i - last_entry_idx < COOLDOWN:
            continue

        # entry: 翌営業日open
        if i + 1 >= len(bdays):
            continue
        entry_ts = bdays[i + 1]
        entry_open = float(df.loc[entry_ts, "adj_open"])
        if entry_open <= 0:
            continue

        # exit: signal + 10営業日 close
        if i + 1 + HOLD_DAYS >= len(bdays):
            continue
        exit_ts = bdays[i + 1 + HOLD_DAYS]
        exit_close = float(df.loc[exit_ts, "adj_close"])
        if exit_close <= 0:
            continue

        gross = exit_close / entry_open - 1
        rows.append({
            "entry_date": entry_ts.date(),
            "exit_date": exit_ts.date(),
            "symbol": SYM,
            "dd20": round(float(row["dd20"]), 2),
            "ma25": round(float(row["ma25"]), 1),
            "gross_ret": gross,
            "net_ret": net_ret(gross),
        })
        last_entry_idx = i

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["entry_date", "exit_date", "symbol", "dd20", "gross_ret", "net_ret"])


def health(start_date: str, end_date: str) -> dict:
    trades = compute_trades(start_date, end_date)
    n = len(trades)
    if n == 0:
        return {"strategy": "lasertec_ma25_support", "n": 0, "sharpe": None,
                "t_stat": None, "win_rate": None, "mean_pct": None,
                "signal_days": 0}
    stats = summary_stats(trades["net_ret"], "lasertec_ma25_support")
    stats["strategy"] = "lasertec_ma25_support"
    stats["signal_days"] = n
    return stats
