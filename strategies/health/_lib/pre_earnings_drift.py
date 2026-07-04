"""
pre_earnings_drift 健全性チェックモジュール

シグナル:
  - earnings_calendar から3/5営業日後に決算予定 (本決算/2Q/3Q) の銘柄
  - 当日売買代金 ≥ 5億円、プライム、除外セクター以外
Entry: T+1 open (翌寄り)
Exit : 決算前日 close (T+2 or T+4)
"""
import pandas as pd
import numpy as np
from datetime import timedelta
from . import get_conn, net_ret, summary_stats, STRATEGIES_ROOT

TURNOVER_MIN = 500_000_000
EXCLUDE_SECTORS = ["医薬品", "陸運業"]
DOC_TYPE_RULES = {
    "本決算":    {"lead_days": 5},
    "第２四半期": {"lead_days": 3},
    "第３四半期": {"lead_days": 3},
}


def compute_trades(start_date: str, end_date: str) -> pd.DataFrame:
    conn = get_conn()

    # 判定期間より少し前のデータから取得 (earnings window考慮)
    sd = (pd.Timestamp(start_date) - timedelta(days=7)).strftime("%Y-%m-%d")

    # earnings_calendar の対象期間 (発表日ベース)
    ec = pd.read_sql("""
        SELECT date, code, fq, sector_nm, section
        FROM earnings_calendar
        WHERE date >= %(sd)s AND date <= %(ed)s
          AND fq IN ('本決算', '第２四半期', '第３四半期')
          AND section = 'プライム'
    """, conn, params={"sd": sd, "ed": end_date})
    ec["date"] = pd.to_datetime(ec["date"])

    # stocks_daily の期間データ
    daily = pd.read_sql("""
        SELECT d.code, d.date, d.adj_open, d.adj_close, d.turnover_value,
               s.sector33_nm
        FROM stocks_daily d
        JOIN symbol_master s ON s.code5 = d.code
        WHERE d.date >= %(sd)s AND d.date <= %(ed)s
          AND d.adj_close > 0
        ORDER BY d.code, d.date
    """, conn, params={"sd": sd, "ed": end_date})
    conn.close()

    daily["date"] = pd.to_datetime(daily["date"])
    daily_idx = daily.set_index(["code", "date"])

    bdays = sorted(daily["date"].dt.date.unique().tolist())

    rows = []
    for _, ec_row in ec.iterrows():
        earnings_date = ec_row["date"].date()
        lead = DOC_TYPE_RULES[ec_row["fq"]]["lead_days"]

        # エントリー判定日 = earnings_date - lead_days 営業日
        try:
            ed_idx = bdays.index(earnings_date)
            if ed_idx < lead:
                continue
            signal_date = bdays[ed_idx - lead]
        except ValueError:
            continue

        if str(signal_date) < start_date or str(signal_date) > end_date:
            continue

        code = ec_row["code"]

        # 除外セクターチェック
        if ec_row["sector_nm"] in EXCLUDE_SECTORS:
            continue

        signal_ts = pd.Timestamp(signal_date)
        try:
            sd_row = daily_idx.loc[(code, signal_ts)]
        except KeyError:
            continue

        # 売買代金フィルター
        if pd.isna(sd_row["turnover_value"]) or sd_row["turnover_value"] < TURNOVER_MIN:
            continue

        # entry_date = signal + 1営業日
        try:
            s_idx = bdays.index(signal_date)
            if s_idx + 1 >= len(bdays):
                continue
            entry_date = bdays[s_idx + 1]
            # exit_date = earnings_date - 1 営業日
            if ed_idx - 1 < 0:
                continue
            exit_date = bdays[ed_idx - 1]
        except (ValueError, IndexError):
            continue

        if entry_date >= exit_date:
            continue

        # リターン計算
        entry_ts = pd.Timestamp(entry_date)
        exit_ts  = pd.Timestamp(exit_date)
        try:
            entry_open = daily_idx.loc[(code, entry_ts), "adj_open"]
            exit_close = daily_idx.loc[(code, exit_ts), "adj_close"]
        except KeyError:
            continue

        if pd.isna(entry_open) or pd.isna(exit_close) or entry_open <= 0:
            continue

        gross = exit_close / entry_open - 1
        rows.append({
            "entry_date": entry_date,
            "exit_date": exit_date,
            "symbol": code,
            "fq": ec_row["fq"],
            "gross_ret": gross,
            "net_ret": net_ret(gross),
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["entry_date", "exit_date", "symbol", "fq", "gross_ret", "net_ret"])


def health(start_date: str, end_date: str) -> dict:
    trades = compute_trades(start_date, end_date)
    n = len(trades)
    if n == 0:
        return {"strategy": "pre_earnings_drift", "n": 0, "sharpe": None,
                "t_stat": None, "win_rate": None, "mean_pct": None}
    stats = summary_stats(trades["net_ret"], "pre_earnings_drift")
    stats["strategy"] = "pre_earnings_drift"
    stats["signal_days"] = trades["entry_date"].nunique()
    return stats
