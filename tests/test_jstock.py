"""jstock 回帰テスト（DB接続不要・純粋関数のみ）。

実行: cd japan-stocks && python3 -m pytest tests/ -q
壊れたら実害が出る順の C-2（スクリーナー・分析の共通基盤）。
"""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from jstock import costs, stats
from jstock.data import to_code5
from jstock.jcal import sq_date


# ---- data.to_code5: 5桁変換（間違うとクエリが黙って0行になる） ----

def test_to_code5_from_4digit():
    assert to_code5("7203") == "72030"


def test_to_code5_passthrough_5digit():
    assert to_code5("72030") == "72030"


def test_to_code5_accepts_int():
    assert to_code5(7203) == "72030"


# ---- costs: コスト前提（教訓2）。数値を変えたら analyses/README.md も更新すること ----

def test_cost_constants_contract():
    assert costs.ONE_WAY_BPS == 2.0
    assert costs.ROUND_TRIP_BPS == 4.0
    assert costs.LS_ROUND_TRIP_BPS == 8.0


def test_net_returns_single():
    assert costs.net_returns(0.0100) == pytest.approx(0.0096)


def test_net_returns_ls_double_cost():
    assert costs.net_returns(0.0100, ls=True) == pytest.approx(0.0092)


def test_net_returns_multiple_round_trips():
    assert costs.net_returns(0.0100, round_trips=2.0) == pytest.approx(0.0092)


def test_net_returns_vectorized():
    gross = pd.Series([0.01, -0.01])
    net = costs.net_returns(gross)
    assert net.iloc[0] == pytest.approx(0.0096)
    assert net.iloc[1] == pytest.approx(-0.0104)


# ---- stats: 検証統計（/backtest-validation の基準値計算に使われる） ----

def test_sharpe_known_value():
    r = pd.Series([0.01, -0.005, 0.008, 0.002] * 30)
    expected = r.mean() / r.std() * np.sqrt(252)
    assert stats.sharpe(r) == pytest.approx(float(expected))


def test_sharpe_zero_std_is_nan():
    # 1.0 は二進で正確に表現でき std が厳密に 0 になる。
    # 0.01 のような値だと float 誤差で std≈1e-18 となりガードをすり抜ける
    # （jstock.stats.sharpe の既知エッジケース。定数系列は実務では発生しない）
    assert np.isnan(stats.sharpe([1.0] * 10))


def test_sharpe_too_short_is_nan():
    assert np.isnan(stats.sharpe([0.01]))


def test_t_stat_known_value():
    r = pd.Series([0.01, 0.02, 0.015, 0.005])
    expected = r.mean() / (r.std() / np.sqrt(4))
    assert stats.t_stat(r) == pytest.approx(float(expected))


def test_max_drawdown():
    # 100 -> 110 -> 99 -> 108.9: 最大DDは 110->99 の -10%
    r = [0.10, -0.10, 0.10]
    assert stats.max_drawdown(r) == pytest.approx(-0.10)


def test_max_drawdown_monotonic_up_is_zero():
    assert stats.max_drawdown([0.01, 0.01, 0.01]) == pytest.approx(0.0)


def test_is_oos_split_no_overlap_no_loss():
    df = pd.DataFrame({"date": pd.date_range("2025-01-01", periods=100), "x": range(100)})
    is_, oos = stats.is_oos_split(df, "2025-03-01")
    assert len(is_) + len(oos) == 100
    assert pd.to_datetime(is_["date"]).max() < pd.Timestamp("2025-03-01")
    assert pd.to_datetime(oos["date"]).min() >= pd.Timestamp("2025-03-01")


def test_summary_keys_contract():
    """/backtest-validation・SUMMARY.md が前提とする出力キー。"""
    rep = stats.summary([0.01, -0.005, 0.002], "test")
    assert set(rep.keys()) == {"label", "n", "ann_return", "sharpe", "t", "max_dd", "win_rate"}
    assert rep["n"] == 3


# ---- jcal.sq_date: 第2金曜（祝日調整なしの理論日） ----

def test_sq_date_known_months():
    assert sq_date(2026, 7) == dt.date(2026, 7, 10)   # 2026-07: 第2金曜
    assert sq_date(2026, 1) == dt.date(2026, 1, 9)
    assert sq_date(2025, 12) == dt.date(2025, 12, 12)


def test_sq_date_always_friday():
    for y in (2025, 2026):
        for m in range(1, 13):
            assert sq_date(y, m).weekday() == 4
