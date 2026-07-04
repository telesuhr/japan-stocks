"""
eneos_vwap_trend 健全性チェックモジュール

銘柄: 5020 (50200, ENEOS)
シグナル: 9:30 VWAP乖離 ≥ 50bps → トレンドフォロー (乖離方向と同方向)
Entry: 9:31 バーopen
Exit : 15:30 close (当日引け)
コスト: 往復4bps
"""
import pandas as pd
import numpy as np
from datetime import timedelta
from . import get_conn, net_ret, summary_stats

CODE5 = "50200"
THRESHOLD = 50  # bps
ENTRY_H, ENTRY_M = 9, 31
EXIT_H,  EXIT_M  = 15, 30


def _get_intraday(start_date: str, end_date: str) -> pd.DataFrame:
    conn = get_conn()
    sd = pd.Timestamp(start_date)
    ed = pd.Timestamp(end_date) + timedelta(days=1)
    df = pd.read_sql(
        "SELECT ts, open, close, volume FROM stocks_intraday "
        "WHERE code=%s AND ts >= %s AND ts < %s ORDER BY ts",
        conn, params=(CODE5, sd, ed)
    )
    conn.close()
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.set_index("ts")
    return df


def compute_trades(start_date: str, end_date: str) -> pd.DataFrame:
    df = _get_intraday(start_date, end_date)
    if df.empty:
        return pd.DataFrame()

    bdays = sorted(set(df.index.date))
    rows = []

    for d in bdays:
        if str(d) < start_date or str(d) > end_date:
            continue

        day_df = df[df.index.date == d]
        if day_df.empty:
            continue

        morning = day_df[day_df.index.hour >= 9]
        if morning.empty:
            continue

        # 9:00からの累積VWAP
        vol = morning["volume"].fillna(0).clip(lower=1)
        cum_pv  = (morning["close"] * vol).cumsum()
        cum_vol = vol.cumsum()
        vwap_series = cum_pv / cum_vol

        # 9:30 バーのVWAP乖離
        bar_930 = morning[(morning.index.hour == 9) & (morning.index.minute == 30)]
        if bar_930.empty:
            continue

        vwap_930 = float(vwap_series.loc[bar_930.index[-1]])
        close_930 = float(bar_930.iloc[-1]["close"])
        dev_bps = (close_930 / vwap_930 - 1) * 10000

        if abs(dev_bps) < THRESHOLD:
            continue

        direction = 1 if dev_bps > 0 else -1  # トレンドフォロー

        # entry: 9:31 open
        bar_entry = day_df[(day_df.index.hour == ENTRY_H) & (day_df.index.minute == ENTRY_M)]
        if bar_entry.empty:
            continue
        entry_price = float(bar_entry.iloc[0]["open"])

        # exit: 15:30 close
        bar_exit = day_df[(day_df.index.hour == EXIT_H) & (day_df.index.minute == EXIT_M)]
        if bar_exit.empty:
            # 15:29以降の最後バー
            late = day_df[day_df.index.hour >= 15]
            if late.empty:
                continue
            exit_price = float(late.iloc[-1]["close"])
        else:
            exit_price = float(bar_exit.iloc[-1]["close"])

        if entry_price <= 0:
            continue

        gross = direction * (exit_price / entry_price - 1)
        rows.append({
            "entry_date": d,
            "symbol": CODE5,
            "dev_bps": round(dev_bps, 1),
            "direction": "Long" if direction == 1 else "Short",
            "gross_ret": gross,
            "net_ret": net_ret(gross),
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["entry_date", "symbol", "dev_bps", "gross_ret", "net_ret"])


def health(start_date: str, end_date: str) -> dict:
    trades = compute_trades(start_date, end_date)
    n = len(trades)
    if n == 0:
        return {"strategy": "eneos_vwap_trend", "n": 0, "sharpe": None,
                "t_stat": None, "win_rate": None, "mean_pct": None, "signal_days": 0}
    stats = summary_stats(trades["net_ret"], "eneos_vwap_trend")
    stats["strategy"] = "eneos_vwap_trend"
    stats["signal_days"] = n
    return stats
