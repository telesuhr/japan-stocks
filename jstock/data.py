"""market_data からの標準ローダ。code は JQuants 5桁が canonical。"""
from __future__ import annotations

import pandas as pd

from . import db


def to_code5(code: str) -> str:
    """'7203' → '72030'（普通株）。5桁はそのまま返す。"""
    code = str(code)
    return code if len(code) == 5 else code + "0"


def load_daily(
    codes: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    adjusted: bool = True,
    conn=None,
) -> pd.DataFrame:
    """stocks_daily を取得。adjusted=True で調整後OHLCV（分割・併合対応）を o/h/l/c/v 名で返す。"""
    if adjusted:
        cols = ("date, code, adj_open AS open, adj_high AS high, adj_low AS low, "
                "adj_close AS close, adj_volume AS volume, turnover_value")
    else:
        cols = "date, code, open, high, low, close, volume, turnover_value"
    where, params = ["1=1"], []
    if codes:
        where.append("code = ANY(%s)")
        params.append([to_code5(c) for c in codes])
    if start:
        where.append("date >= %s")
        params.append(start)
    if end:
        where.append("date <= %s")
        params.append(end)
    sql = f"SELECT {cols} FROM stocks_daily WHERE {' AND '.join(where)} ORDER BY code, date"
    return db.read_sql(sql, params, conn)


def load_intraday(
    codes: list[str],
    start: str,
    end: str,
    conn=None,
) -> pd.DataFrame:
    """stocks_intraday (1分足) を取得。ts は JST naive（変換不要）。end は排他 (<)。

    JQuants 確定版（平日17:00バッチ反映）。当日リアルタイムは aukabu.bars_1min（監視専用）。
    """
    sql = """
        SELECT ts, code, open, high, low, close, volume, turnover_value
        FROM stocks_intraday
        WHERE code = ANY(%s) AND ts >= %s AND ts < %s
        ORDER BY code, ts
    """
    return db.read_sql(sql, [[to_code5(c) for c in codes], start, end], conn)
