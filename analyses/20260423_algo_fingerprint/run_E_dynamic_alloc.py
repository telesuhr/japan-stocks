#!/usr/bin/env python3
"""
E: 動的アロケーション検証
semi_disp (半導体セクター分散) を日次で計算し、
分散 HI 日は 「分散で儲かる戦略」に寄せ、LO 日は「共通因子で儲かる戦略」に寄せる。

- Base (static):     推奨配分 (前セッションで構築済)
- Dynamic disp:      semi_disp の当日予測 (前日値 proxy) を元に重みを動かす
"""
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path("/Users/Yusuke/claude-code/japan-stocks/analyses")

pnl = pd.read_csv(ROOT / "20260422_pair_trading/all_strategies_daily_pnl.csv",
                  index_col=0, parse_dates=True)
feats = pd.read_csv("strategy_algo_merged.csv", index_col=0, parse_dates=True)

# 共通期間で整形
df = pnl.join(feats[["semi_disp", "topix_rv", "topix_twap_pulse"]], how="inner").dropna(
    subset=["semi_disp"])
strats = pnl.columns.tolist()

# 静的 base 配分 (前セッション推奨)
BASE_W = {
    "lme_on_copper": 0.20, "nonferrous_D_strong": 0.20,
    "pair_portfolio": 0.15, "eneos_vwap_trend": 0.10,
    "nonferrous_D_freq": 0.10, "nonferrous_B_disp": 0.08,
    "semi_sox_fade": 0.10, "lasertec_ma25": 0.07,
}

# 動的: semi_disp の 20 日移動平均と比較して、HI 日は dispersion 型、LO 日は共通因子型へ寄せる
# HI で儲かる: nonferrous_D_freq, eneos_vwap_trend, semi_sox_fade
# LO で儲かる: pair_portfolio
HI_W = {
    "lme_on_copper": 0.15, "nonferrous_D_strong": 0.15,
    "pair_portfolio": 0.05, "eneos_vwap_trend": 0.15,
    "nonferrous_D_freq": 0.20, "nonferrous_B_disp": 0.10,
    "semi_sox_fade": 0.15, "lasertec_ma25": 0.05,
}
LO_W = {
    "lme_on_copper": 0.20, "nonferrous_D_strong": 0.20,
    "pair_portfolio": 0.30, "eneos_vwap_trend": 0.05,
    "nonferrous_D_freq": 0.03, "nonferrous_B_disp": 0.07,
    "semi_sox_fade": 0.05, "lasertec_ma25": 0.10,
}


def mix(row, w):
    return sum(row.get(k, 0) * w.get(k, 0) for k in w)


# semi_disp 20 日移動中央値を signal として使う
df["disp_ma20"] = df["semi_disp"].rolling(20, min_periods=10).median()
df["disp_signal"] = np.sign(df["semi_disp"] - df["disp_ma20"])  # +1=HI, -1=LO

# 前日シグナルを翌日に適用 (look-ahead 防止)
df["disp_sig_prev"] = df["disp_signal"].shift(1)

base_pnl, hi_pnl, lo_pnl, dyn_pnl = [], [], [], []
for _, row in df.iterrows():
    base_pnl.append(mix(row, BASE_W))
    hi_pnl.append(mix(row, HI_W))
    lo_pnl.append(mix(row, LO_W))
    sig = row["disp_sig_prev"]
    if sig == 1:
        w = HI_W
    elif sig == -1:
        w = LO_W
    else:
        w = BASE_W
    dyn_pnl.append(mix(row, w))

df["base"] = base_pnl
df["hi_only"] = hi_pnl
df["lo_only"] = lo_pnl
df["dynamic"] = dyn_pnl


def perf(s, label):
    a = s[s.notna()]
    sh = a.mean() / a.std() * np.sqrt(252) if a.std() > 0 else 0
    cum = a.cumsum()
    mdd = (cum.cummax() - cum).max()
    t = a.mean() / (a.std() / np.sqrt(len(a))) if a.std() > 0 else 0
    print(f"  {label:<25} mean {a.mean():+6.2f}  Sh {sh:+5.2f}  t {t:+5.2f}  "
          f"sum {a.sum():+7.0f}  MDD {mdd:>5.0f}  N={len(a)}")


print("=" * 100)
print("E. 動的アロケーション検証 (semi_disp 信号)")
print("=" * 100)
perf(df["base"], "Static BASE (既定推奨)")
perf(df["hi_only"], "HI_W 固定 (分散型寄せ)")
perf(df["lo_only"], "LO_W 固定 (共通因子型)")
perf(df["dynamic"], "DYNAMIC (disp_signal)")

# レジーム別検証
print("\n【DYNAMIC 配分のレジーム別内訳】")
hi_days = df[df["disp_sig_prev"] == 1]
lo_days = df[df["disp_sig_prev"] == -1]
for label, subset in [("HI 日 (dispersion>MA)", hi_days),
                      ("LO 日 (dispersion<MA)", lo_days)]:
    a = subset["dynamic"]
    sh = a.mean() / a.std() * np.sqrt(252) if a.std() > 0 else 0
    print(f"  {label:<30} N={len(a):>4}  mean {a.mean():+6.2f}  Sh {sh:+5.2f}")

# 各戦略の HI/LO Sharpe 再確認 (共通期間)
print("\n【最終: 戦略別 HI/LO Sharpe (検証)】")
for s in strats:
    hi = hi_days[s]; lo = lo_days[s]
    hi = hi[hi != 0]; lo = lo[lo != 0]
    if len(hi) < 5 or len(lo) < 5: continue
    sh_hi = hi.mean() / hi.std() * np.sqrt(252) if hi.std() > 0 else 0
    sh_lo = lo.mean() / lo.std() * np.sqrt(252) if lo.std() > 0 else 0
    print(f"  {s:<25} LO Sh {sh_lo:+5.2f} (N={len(lo)})  "
          f"HI Sh {sh_hi:+5.2f} (N={len(hi)})  Δ {sh_hi-sh_lo:+5.2f}")

df[["base", "hi_only", "lo_only", "dynamic", "semi_disp", "disp_sig_prev"]].to_csv(
    "dynamic_allocation.csv")
print("\n→ dynamic_allocation.csv 保存")
