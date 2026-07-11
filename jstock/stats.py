"""検証用の標準統計。昇格基準 (SUMMARY.md): Sharpe >= 2.0 / N >= 30 / IS・OOS一貫。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sharpe(daily_returns, ann: int = 252) -> float:
    r = pd.Series(daily_returns).dropna()
    if len(r) < 2 or r.std() == 0:
        return float("nan")
    return float(r.mean() / r.std() * np.sqrt(ann))


def t_stat(returns) -> float:
    r = pd.Series(returns).dropna()
    if len(r) < 2 or r.std() == 0:
        return float("nan")
    return float(r.mean() / (r.std() / np.sqrt(len(r))))


def max_drawdown(daily_returns) -> float:
    """日次リターン系列から最大ドローダウン（負の小数）を返す。"""
    cum = (1 + pd.Series(daily_returns).fillna(0)).cumprod()
    return float((cum / cum.cummax() - 1).min())


def is_oos_split(df: pd.DataFrame, split_date: str, date_col: str = "date"):
    """IS/OOS 分割（教訓: 仮説はISで固定し、同じルールをOOSに適用する）。"""
    d = pd.to_datetime(df[date_col])
    return df[d < split_date].copy(), df[d >= split_date].copy()


def summary(daily_returns, label: str = "") -> dict:
    r = pd.Series(daily_returns).dropna()
    return {
        "label": label,
        "n": int(len(r)),
        "ann_return": float(r.mean() * 252),
        "sharpe": sharpe(r),
        "t": t_stat(r),
        "max_dd": max_drawdown(r),
        "win_rate": float((r > 0).mean()) if len(r) else float("nan"),
    }
