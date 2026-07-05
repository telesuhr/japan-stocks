"""
戦略健全性チェック共通ライブラリ

各モジュールは compute_trades(start_date, end_date) -> pd.DataFrame を実装する。
返却カラム: entry_date, exit_date, symbol, gross_ret, net_ret
"""
import sys
from pathlib import Path
import pandas as pd
import psycopg2
import numpy as np

PG_CONFIG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}
COST_ONE_WAY_BPS = 2.0  # 片道2bps (往復4bps)

STRATEGIES_ROOT = Path(__file__).resolve().parent.parent.parent


def get_conn():
    return psycopg2.connect(**PG_CONFIG)


def trading_days_between(start: str, end: str) -> list:
    """営業日リストを trading_calendar テーブルから取得"""
    conn = get_conn()
    try:
        df = pd.read_sql(
            # hol_div=1 が営業日 (フィルタ無しだと土日祝も返る)
            "SELECT date FROM trading_calendar WHERE date >= %s AND date <= %s "
            "AND hol_div::text = '1' ORDER BY date",
            conn, params=(start, end)
        )
        return df["date"].tolist()
    except Exception:
        # trading_calendar がない場合は pandas で代替
        dates = pd.bdate_range(start, end)
        return [d.date() for d in dates]
    finally:
        conn.close()


def nth_bday_after(d, n: int, bdays: list):
    """bdays リストからd以降n番目の営業日を返す"""
    try:
        idx = bdays.index(d)
        if idx + n < len(bdays):
            return bdays[idx + n]
    except ValueError:
        pass
    # fallback: 単純営業日計算
    from datetime import timedelta
    cur = d
    cnt = 0
    while cnt < n:
        cur += timedelta(days=1)
        if cur.weekday() < 5:
            cnt += 1
    return cur


def net_ret(gross: float) -> float:
    """往復コスト(4bps)を控除したネットリターン"""
    return gross - COST_ONE_WAY_BPS * 2 / 10000


def summary_stats(returns: pd.Series, label: str = "") -> dict:
    """シャープ・t値・勝率・MDD を返す"""
    r = returns.dropna()
    n = len(r)
    if n == 0:
        return {"n": 0, "mean_pct": 0, "sharpe": 0, "t_stat": 0, "win_rate": 0, "mdd_pct": 0}
    mean = r.mean()
    std = r.std(ddof=1)
    sharpe = mean / std * np.sqrt(252) if std > 0 else 0
    t_stat = mean / std * np.sqrt(n) if std > 0 else 0
    wins = (r > 0).sum()
    win_rate = wins / n
    cum = (1 + r).cumprod()
    roll_max = cum.cummax()
    mdd = ((cum - roll_max) / roll_max).min() * 100
    return {
        "n": n,
        "mean_pct": round(mean * 100, 3),
        "sharpe": round(sharpe, 2),
        "t_stat": round(t_stat, 2),
        "win_rate": round(win_rate * 100, 1),
        "mdd_pct": round(mdd, 2),
    }
