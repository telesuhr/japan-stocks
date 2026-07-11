"""営業日カレンダー・SQ日。実データ (stocks_daily) の日付を正とする。"""
from __future__ import annotations

import datetime as dt

from . import db


def trading_days(start: str | None = None, end: str | None = None, conn=None) -> list[dt.date]:
    """データが存在する営業日リスト（昇順）。"""
    where, params = ["1=1"], []
    if start:
        where.append("date >= %s")
        params.append(start)
    if end:
        where.append("date <= %s")
        params.append(end)
    sql = f"SELECT DISTINCT date FROM stocks_daily WHERE {' AND '.join(where)} ORDER BY date"
    return db.read_sql(sql, params, conn)["date"].tolist()


def latest_trading_day(conn=None) -> dt.date:
    return db.read_sql("SELECT max(date) AS d FROM stocks_daily", conn=conn)["d"].iloc[0]


def sq_date(year: int, month: int) -> dt.date:
    """メジャー/マイナーSQ = 第2金曜（祝日調整なしの理論日。厳密には trading_days で前営業日に丸める）。"""
    d = dt.date(year, month, 1)
    first_friday = d + dt.timedelta(days=(4 - d.weekday()) % 7)
    return first_friday + dt.timedelta(days=7)
