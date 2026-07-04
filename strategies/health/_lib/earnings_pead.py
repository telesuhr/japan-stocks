"""
earnings_pead 健全性チェックモジュール

シグナル:
  - 前営業日 15:00 以降の決算発表 (tdnet_disclosures + earnings_calendar 突合)
  - 翌日ギャップ ≥ +7%
  - 売買代金 ≥ 5億円, 除外セクター以外
Entry: signal_date close (引成)
Exit : signal_date + 5営業日 close
"""
import pandas as pd
import numpy as np
from datetime import timedelta
from . import get_conn, net_ret, summary_stats

GAP_MIN_PCT = 7.0
TURNOVER_MIN = 500_000_000
HOLD_DAYS = 5
EXCLUDE_SECTORS = ["銀行業", "食料品", "金属製品", "サービス業"]
AC_HOUR = 15  # 15:00以降


def compute_trades(start_date: str, end_date: str) -> pd.DataFrame:
    conn = get_conn()

    sd = (pd.Timestamp(start_date) - timedelta(days=14)).strftime("%Y-%m-%d")
    ed = (pd.Timestamp(end_date) + timedelta(days=10)).strftime("%Y-%m-%d")

    # tdnet_disclosures: codeはすでに5桁 (例: 72030)
    disc = pd.read_sql("""
        SELECT DISTINCT
            DATE(disclosed_at AT TIME ZONE 'Asia/Tokyo') AS disc_date,
            code AS code5,
            disclosed_at,
            title
        FROM tdnet_disclosures
        WHERE disclosed_at >= %(sd)s AND disclosed_at <= %(ed)s
          AND (title LIKE '%%決算%%' OR title LIKE '%%業績%%' OR title LIKE '%%四半期%%')
    """, conn, params={"sd": sd, "ed": ed})

    # stocks_daily
    daily = pd.read_sql("""
        SELECT d.code, d.date, d.adj_open, d.adj_close, d.turnover_value,
               s.sector33_nm
        FROM stocks_daily d
        JOIN symbol_master s ON s.code5 = d.code
        WHERE d.date >= %(sd)s AND d.date <= %(ed)s
          AND d.adj_close > 0 AND d.adj_open > 0
        ORDER BY d.code, d.date
    """, conn, params={"sd": sd, "ed": ed})
    conn.close()

    daily["date"] = pd.to_datetime(daily["date"])
    daily_idx = daily.set_index(["code", "date"])
    bdays = sorted(daily["date"].dt.date.unique().tolist())

    disc["disclosed_at"] = pd.to_datetime(disc["disclosed_at"])
    disc["disc_date"] = pd.to_datetime(disc["disc_date"]).dt.date

    rows = []
    # signal_date = ギャップ当日 (前日15:00以降に発表があった翌営業日)
    for signal_date in bdays:
        if str(signal_date) < start_date or str(signal_date) > end_date:
            continue

        try:
            s_idx = bdays.index(signal_date)
            prev_date = bdays[s_idx - 1] if s_idx > 0 else None
        except ValueError:
            continue
        if prev_date is None:
            continue

        # 前営業日 15:00 以降の決算発表
        cutoff = pd.Timestamp(prev_date).replace(hour=AC_HOUR, minute=0, second=0)
        day_end = pd.Timestamp(signal_date).replace(hour=9, minute=0, second=0)
        today_disc = disc[
            (disc["disclosed_at"] >= cutoff) & (disc["disclosed_at"] < day_end)
        ].copy()
        if today_disc.empty:
            continue

        announced_codes = set(today_disc["code5"].dropna().unique())

        for code in announced_codes:
            if len(code) != 5:
                continue
            sig_ts = pd.Timestamp(signal_date)
            prev_ts = pd.Timestamp(prev_date)

            try:
                sig_row = daily_idx.loc[(code, sig_ts)]
                prev_row = daily_idx.loc[(code, prev_ts)]
            except KeyError:
                continue

            # 除外セクター
            sector = sig_row.get("sector33_nm", "")
            if sector in EXCLUDE_SECTORS:
                continue

            # ギャップ計算 (翌日open / 前日close - 1)
            gap = sig_row["adj_open"] / prev_row["adj_close"] - 1
            if gap < GAP_MIN_PCT / 100:
                continue

            # 売買代金
            if pd.isna(sig_row["turnover_value"]) or sig_row["turnover_value"] < TURNOVER_MIN:
                continue

            # exit: signal + 5営業日 close
            if s_idx + HOLD_DAYS >= len(bdays):
                continue
            exit_date = bdays[s_idx + HOLD_DAYS]
            exit_ts = pd.Timestamp(exit_date)

            try:
                exit_close = daily_idx.loc[(code, exit_ts), "adj_close"]
                entry_close = sig_row["adj_close"]
            except KeyError:
                continue

            if pd.isna(exit_close) or entry_close <= 0:
                continue

            gross = exit_close / entry_close - 1
            rows.append({
                "entry_date": signal_date,
                "exit_date": exit_date,
                "symbol": code,
                "gap_pct": round(gap * 100, 2),
                "gross_ret": gross,
                "net_ret": net_ret(gross),
            })

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["entry_date", "exit_date", "symbol", "gap_pct", "gross_ret", "net_ret"])


def health(start_date: str, end_date: str) -> dict:
    trades = compute_trades(start_date, end_date)
    n = len(trades)
    if n == 0:
        return {"strategy": "earnings_pead", "n": 0, "sharpe": None,
                "t_stat": None, "win_rate": None, "mean_pct": None}
    stats = summary_stats(trades["net_ret"], "earnings_pead")
    stats["strategy"] = "earnings_pead"
    stats["signal_days"] = trades["entry_date"].nunique()
    return stats
