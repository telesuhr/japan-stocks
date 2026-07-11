"""PostgreSQL (market_data) 接続。環境変数は libpq 標準名に統一。"""
from __future__ import annotations

import os
import warnings

import pandas as pd
import psycopg2


def pg_config() -> dict:
    return {
        "host": os.environ.get("PGHOST", "localhost"),  # 外部からは PGHOST=omen
        "port": int(os.environ.get("PGPORT", 5432)),
        "user": os.environ.get("PGUSER", "postgres"),
        "dbname": os.environ.get("PGDATABASE", "market_data"),
        # password は ~/.pgpass / PGPASSWORD から libpq が拾う
    }


def connect():
    return psycopg2.connect(**pg_config())


def read_sql(sql: str, params=None, conn=None) -> pd.DataFrame:
    """pd.read_sql のラッパ。教訓6: psycopg2 直接続の UserWarning は抑制してよい。

    conn を渡さなければ都度接続して閉じる。ループ内で呼ぶなら conn を使い回すこと。
    """
    own = conn is None
    if own:
        conn = connect()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return pd.read_sql(sql, conn, params=params)
    finally:
        if own:
            conn.close()
