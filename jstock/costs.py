"""取引コスト前提（教訓2: 必ずコスト込みで評価）。analyses/README.md の前提と一致させること。"""
from __future__ import annotations

ONE_WAY_BPS = 2.0        # 片側 2bps
ROUND_TRIP_BPS = 4.0     # 往復 4bps
LS_ROUND_TRIP_BPS = 8.0  # ロングショートは×2銘柄で 8bps


def net_returns(gross, round_trips: float = 1.0, ls: bool = False):
    """グロスリターン（小数）から往復コストを控除。round_trips=1トレードあたりの往復回数。"""
    bps = LS_ROUND_TRIP_BPS if ls else ROUND_TRIP_BPS
    return gross - round_trips * bps / 10_000.0
