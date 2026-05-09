"""テンプレのプレビュー生成 (ダミーデータ)"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from templates import plot_backtest, plot_daily_on, plot_intraday

rng = np.random.default_rng(42)

# --- backtest preview ---
n = 344
pnl = rng.normal(0.04, 0.18, n)
equity = pd.Series(np.cumsum(pnl),
                   index=pd.bdate_range("2025-01-06", periods=n, freq="B"))
plot_backtest(
    title="ORB ブレイクアウト",
    subtitle="日本株前場 / 9:00-9:30レンジ × 前場引け",
    equity=equity,
    trades=pd.Series(pnl),
    sharpe=3.2, pf=1.8, win_rate=61, n=n,
    cost_bps=2,
    path="/tmp/preview_backtest.png",
)

# --- daily_on preview ---
gaps = pd.Series(rng.normal(0.08, 0.45, 300))
dow_labels = ["月", "火", "水", "木", "金"]
by_dow = {d: gaps.iloc[i::5] for i, d in enumerate(dow_labels)}
plot_daily_on(
    title="TOPIX ON ギャップ分布",
    subtitle="前日終値→翌寄付 / 2025-2026",
    gaps=gaps,
    by_dow=by_dow,
    fill_rate=0.63,
    path="/tmp/preview_daily_on.png",
)

# --- intraday preview ---
times = [f"{h:02d}:{m:02d}" for h in range(9, 16)
         for m in range(0, 60, 5)
         if not (h == 11 and m > 30) and not (h == 12 and m < 30)
         and not (h == 15 and m > 30)]
n_t = len(times)
mean_r = np.sin(np.linspace(0, np.pi, n_t)) * 0.03 + rng.normal(0, 0.005, n_t)
std_r  = np.abs(rng.normal(0.02, 0.005, n_t))
df_single = pd.DataFrame({"mean": mean_r, "std": std_r}, index=times)
vol = pd.Series(np.abs(rng.normal(1e6, 3e5, n_t)) * (1 + np.sin(np.linspace(0, np.pi*2, n_t))*0.5),
                index=times)
plot_intraday(
    title="レーザーテック 時間帯別平均リターン",
    subtitle="9:00-15:30 1分足 / 2025-2026",
    by_time=df_single,
    vol_by_time=vol,
    path="/tmp/preview_intraday.png",
)

print("done: /tmp/preview_*.png")
