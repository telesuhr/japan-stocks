"""
bank_absorption 健全性チェックモジュール

シグナル: ホワイトリスト銀行株 vol_ratio ≥ 1.5, 陰線, 売買代金 ≥ 10億円
Entry: 翌営業日open, 最大3銘柄
Exit : signal + 5営業日 close
コスト: 往復4bps
"""
import pandas as pd
import numpy as np
from pathlib import Path
from . import get_conn, net_ret, summary_stats, STRATEGIES_ROOT

VOL_RATIO_MIN = 1.5
DAY_RET_MAX = 0.0
TURNOVER_MIN = 1_000_000_000
HOLD_DAYS = 5
MAX_POS = 3

WL_PATH = STRATEGIES_ROOT / "bank_absorption" / "whitelist.csv"


def _load_whitelist():
    return pd.read_csv(WL_PATH, dtype={"code5": str})["code5"].tolist()


def compute_trades(start_date: str, end_date: str) -> pd.DataFrame:
    codes = _load_whitelist()
    if not codes:
        return pd.DataFrame()

    conn = get_conn()
    sd_ext = (pd.Timestamp(start_date) - pd.Timedelta(days=50)).strftime("%Y-%m-%d")
    ed_ext = (pd.Timestamp(end_date) + pd.Timedelta(days=10)).strftime("%Y-%m-%d")

    placeholders = ",".join(f"'{c}'" for c in codes)
    daily = pd.read_sql(f"""
        SELECT code, date, adj_open, adj_close, adj_volume, turnover_value
        FROM stocks_daily
        WHERE code IN ({placeholders})
          AND date >= %(sd)s AND date <= %(ed)s
          AND adj_close > 0 AND adj_open > 0
        ORDER BY code, date
    """, conn, params={"sd": sd_ext, "ed": ed_ext})
    conn.close()

    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values(["code", "date"])

    # vol_ma20 (過去21日, 当日除く直近20日)
    daily["vol_ma20"] = daily.groupby("code")["adj_volume"].transform(
        lambda x: x.shift(1).rolling(20).mean()
    )
    daily["vol_ratio"] = daily["adj_volume"] / daily["vol_ma20"].replace(0, np.nan)
    daily["day_ret"] = daily["adj_close"] / daily["adj_open"] - 1

    bdays = sorted(daily["date"].dt.date.unique().tolist())
    daily_idx = daily.set_index(["code", "date"])

    rows = []
    for d in bdays:
        if str(d) < start_date or str(d) > end_date:
            continue

        d_ts = pd.Timestamp(d)
        try:
            d_idx = bdays.index(d)
        except ValueError:
            continue
        if d_idx + HOLD_DAYS >= len(bdays):
            continue

        entry_date = bdays[d_idx + 1]
        exit_date  = bdays[d_idx + HOLD_DAYS]

        # 当日のシグナル銘柄抽出
        signals = []
        for code in codes:
            try:
                row = daily_idx.loc[(code, d_ts)]
            except KeyError:
                continue
            if pd.isna(row["vol_ratio"]) or row["vol_ratio"] < VOL_RATIO_MIN:
                continue
            if row["day_ret"] >= DAY_RET_MAX:
                continue
            if pd.isna(row["turnover_value"]) or row["turnover_value"] < TURNOVER_MIN:
                continue
            signals.append((code, float(row["vol_ratio"])))

        # vol_ratio降順で上位3銘柄
        signals.sort(key=lambda x: -x[1])
        signals = signals[:MAX_POS]

        for code, vr in signals:
            entry_ts = pd.Timestamp(entry_date)
            exit_ts  = pd.Timestamp(exit_date)
            try:
                entry_open = float(daily_idx.loc[(code, entry_ts), "adj_open"])
                exit_close = float(daily_idx.loc[(code, exit_ts),  "adj_close"])
            except KeyError:
                continue
            if entry_open <= 0 or exit_close <= 0:
                continue
            gross = exit_close / entry_open - 1
            rows.append({
                "entry_date": entry_date,
                "exit_date": exit_date,
                "symbol": code,
                "vol_ratio": round(vr, 2),
                "gross_ret": gross,
                "net_ret": net_ret(gross),
            })

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["entry_date", "exit_date", "symbol", "vol_ratio", "gross_ret", "net_ret"])


def health(start_date: str, end_date: str) -> dict:
    trades = compute_trades(start_date, end_date)
    n = len(trades)
    if n == 0:
        return {"strategy": "bank_absorption", "n": 0, "sharpe": None,
                "t_stat": None, "win_rate": None, "mean_pct": None, "signal_days": 0}
    stats = summary_stats(trades["net_ret"], "bank_absorption")
    stats["strategy"] = "bank_absorption"
    stats["signal_days"] = trades["entry_date"].nunique()
    return stats
