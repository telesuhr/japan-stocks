"""
vwap_morning_meanrevert 健全性チェックモジュール

対象: 8035(80350), 6146(61460), 6920(69200)
シグナル: 10:00-11:30 にVWAP乖離 ≥ 275bps (最初の発生)
Entry: 次の1分足open
Exit : 15:25 close (引け直前) = 当日 15:25 バーのclose
方向: 乖離>0 → Short, 乖離<0 → Long (平均回帰)
コスト: 往復4bps
"""
import pandas as pd
import numpy as np
from datetime import timedelta
from . import get_conn, net_ret, summary_stats

TARGETS = {"80350": "TEL(8035)", "61460": "ディスコ(6146)", "69200": "レーザー(6920)"}
THRESH_BPS = 275.0
EXIT_HOUR, EXIT_MIN = 15, 25


def _get_intraday(code: str, start_date: str, end_date: str) -> pd.DataFrame:
    conn = get_conn()
    sd = pd.Timestamp(start_date)
    ed = pd.Timestamp(end_date) + timedelta(days=1)
    df = pd.read_sql(
        "SELECT ts, open, high, low, close, volume FROM stocks_intraday "
        "WHERE code=%s AND ts >= %s AND ts < %s ORDER BY ts",
        conn, params=(code, sd, ed)
    )
    conn.close()
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.set_index("ts")
    return df


def _day_signals(df_day: pd.DataFrame, date_val) -> list:
    """1日分の分足から first-touch シグナルを返す"""
    rows = []
    if df_day.empty:
        return rows

    # 累積VWAP計算 (9:00から)
    morning = df_day[(df_day.index.hour >= 9)]
    vol = morning["volume"].fillna(0).clip(lower=1)
    cum_pv = (morning["close"] * vol).cumsum()
    cum_vol = vol.cumsum()
    vwap = (cum_pv / cum_vol).rename("vwap")
    dev_bps = ((morning["close"] / vwap) - 1) * 10000

    # 監視窓 10:00-11:30
    window = dev_bps[
        ((dev_bps.index.hour == 10) |
         ((dev_bps.index.hour == 11) & (dev_bps.index.minute <= 30)))
    ]

    signaled = False
    for ts, d in window.items():
        if abs(d) >= THRESH_BPS and not signaled:
            signaled = True
            direction = -1 if d > 0 else 1  # Long if oversold, Short if overbought

            # next bar open as entry price
            entry_ts_candidates = df_day.index[df_day.index > ts]
            if len(entry_ts_candidates) == 0:
                continue
            entry_ts = entry_ts_candidates[0]
            entry_price = float(df_day.loc[entry_ts, "open"])

            # exit: 15:25 bar close
            exit_mask = (df_day.index.hour == EXIT_HOUR) & (df_day.index.minute == EXIT_MIN)
            exit_bars = df_day[exit_mask]
            if exit_bars.empty:
                # 最後の利用可能なバーを使用
                exit_bars = df_day[df_day.index.hour >= 15]
                if exit_bars.empty:
                    continue
            exit_price = float(exit_bars.iloc[-1]["close"])

            if entry_price <= 0:
                continue

            # gross return (direction考慮)
            gross = direction * (exit_price / entry_price - 1)
            rows.append({
                "entry_date": date_val,
                "signal_ts": ts,
                "direction": "Long" if direction == 1 else "Short",
                "dev_bps": round(d, 1),
                "gross_ret": gross,
                "net_ret": net_ret(gross),
            })

    return rows


def compute_trades(start_date: str, end_date: str) -> pd.DataFrame:
    all_rows = []
    bdays = pd.bdate_range(start_date, end_date)

    for code, name in TARGETS.items():
        df = _get_intraday(code, start_date, end_date)
        if df.empty:
            continue

        for day in bdays:
            d = day.date()
            day_df = df[(df.index.date == d)]
            if day_df.empty:
                continue
            rows = _day_signals(day_df, d)
            for r in rows:
                r["symbol"] = code
                r["symbol_name"] = name
                all_rows.append(r)

    return pd.DataFrame(all_rows) if all_rows else pd.DataFrame(
        columns=["entry_date", "symbol", "dev_bps", "gross_ret", "net_ret"])


def health(start_date: str, end_date: str) -> dict:
    trades = compute_trades(start_date, end_date)
    n = len(trades)
    if n == 0:
        return {"strategy": "vwap_morning_meanrevert", "n": 0, "sharpe": None,
                "t_stat": None, "win_rate": None, "mean_pct": None}
    stats = summary_stats(trades["net_ret"], "vwap_morning_meanrevert")
    stats["strategy"] = "vwap_morning_meanrevert"
    stats["signal_days"] = trades["entry_date"].nunique()
    return stats
