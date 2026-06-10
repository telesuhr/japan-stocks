"""採用6戦略の足元エッジ確認 (90/180/365日窓 × コスト感応度)

Dashboard_CC/strategy_oos_monitor.py の oos_* 関数を再利用し、
sharpe_from_trades をフックして生トレード系列を捕捉 → 複数コストで再評価する。
20260531_strategy_recency_check の再実行版 (データ +10日)。
"""
import os, sys
sys.stdout.reconfigure(line_buffering=True)
from datetime import timedelta
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent.parent / "Dashboard_CC"))
import strategy_oos_monitor as mon

# ---- sharpe_from_trades をフックして生トレードを捕捉 ----
captured = {}
_orig = mon.sharpe_from_trades

def hook(returns, hold_days=5):
    hook.last = (pd.Series(returns).dropna(), hold_days)
    return _orig(returns, hold_days)

mon.sharpe_from_trades = hook

def metrics(trades: pd.Series, hold_days: int, cost_bps: float) -> dict:
    if len(trades) < 5:
        return {"n": len(trades), "mean_bps": np.nan, "sharpe": np.nan,
                "t_stat": np.nan, "win_rate": np.nan}
    net = trades - cost_bps / 1e4
    sd = net.std()
    return {
        "n": len(net),
        "mean_bps": round(net.mean() * 1e4, 1),
        "sharpe": round(net.mean() / sd * np.sqrt(245 / hold_days), 2) if sd > 0 else np.nan,
        "t_stat": round(net.mean() / (sd / np.sqrt(len(net))), 2) if sd > 0 else np.nan,
        "win_rate": round((net > 0).mean(), 3),
    }

RUNNERS = [
    ("pre_earnings_drift",      mon.oos_pre_earnings_drift),
    ("earnings_pead",           mon.oos_earnings_pead),
    ("bank_absorption",         mon.oos_bank_absorption),
    ("lasertec_ma25_support",   mon.oos_lasertec_ma25),
    ("vwap_morning_meanrevert", mon.oos_vwap_morning_mr),
    ("eneos_vwap_trend",        mon.oos_eneos_vwap_trend),
]
# 戦略タイプ別の現実的コスト (往復bps)。スイング=20、イントラ=4(設計値)/10(保守)
COSTS = {
    "pre_earnings_drift":      [0, 20],
    "earnings_pead":           [0, 20],
    "bank_absorption":         [0, 20],
    "lasertec_ma25_support":   [0, 20],
    "vwap_morning_meanrevert": [0, 4, 10, 20],
    "eneos_vwap_trend":        [0, 4, 10, 20],
}

_, latest = mon.get_dates()
print(f"最新営業日: {latest}")

rows = []
for win in [90, 180, 365]:
    start = latest - timedelta(days=win)
    print(f"\n===== 窓 {win}日 ({start} 〜 {latest}) =====")
    for name, fn in RUNNERS:
        hook.last = None
        try:
            fn(start, latest)
        except Exception as e:
            print(f"  {name}: ERROR {e}")
            continue
        if hook.last is None:
            print(f"  {name}: トレードなし")
            continue
        trades, hold = hook.last
        for cost in COSTS[name]:
            m = metrics(trades, hold, cost)
            m.update({"strategy": name, "window": win, "cost_bps": cost})
            rows.append(m)
        m20 = [r for r in rows if r["strategy"] == name and r["window"] == win]
        g = m20[0]; n_ = g["n"]
        nets = " / ".join(f"{r['cost_bps']}bps:Sh{r['sharpe']}" for r in m20[1:])
        print(f"  {name:24s} n={n_:4d} gross={g['mean_bps']:+7.1f}bps Sh{g['sharpe']} | {nets}")

df = pd.DataFrame(rows)
df.to_csv(HERE / "results.csv", index=False)
print(f"\nsaved {HERE / 'results.csv'}")
