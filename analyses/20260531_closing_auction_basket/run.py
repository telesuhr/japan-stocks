"""
クロージングオークション下落側Long を既存戦略バスケットに加えた分散効果検証

目的: 昇格判断の核心 = この戦略が既存4戦略バスケットへ低相関で分散寄与するか。
overnight戦略 (引け→翌朝) は日中/スイング戦略と時間軸が異なり低相関の見込み。

データ:
  - 既存4戦略 sleeve: 20260531_portfolio_daily_sharpe/sleeve_daily_returns.csv
  - 新戦略 sleeve: 20260531_closing_auction_exec/exec_obs.csv の ret_open(bps) を
    auction日ごとに平均 → 日次sleeve (比率)
"""
from __future__ import annotations
import os, sys
import pandas as pd
import numpy as np

sys.stdout.reconfigure(line_buffering=True)
HERE = os.path.dirname(__file__)
SLEEVES = os.path.join(HERE, '..', '20260531_portfolio_daily_sharpe', 'sleeve_daily_returns.csv')
EXEC = os.path.join(HERE, '..', '20260531_closing_auction_exec', 'exec_obs.csv')


def sharpe(s, ann=np.sqrt(252)):
    s = pd.Series(s).dropna()
    return float(s.mean()/s.std()*ann) if len(s) >= 10 and s.std() > 0 else float('nan')


print("="*76); print("クロージングオークション下落側Long × 既存4戦略 分散効果"); print("="*76)

# 既存4戦略 sleeve (日次比率)
sl = pd.read_csv(SLEEVES, index_col=0, parse_dates=True)
print(f"\n既存sleeve: {sl.shape[1]}戦略, {sl.index.min().date()}〜{sl.index.max().date()}")

# 新戦略 sleeve: auction日ごとに ret_open(bps) 平均 → 比率
ex = pd.read_csv(EXEC, parse_dates=['date'])
ca = ex.groupby('date')['ret_open'].mean() / 1e4   # bps→比率
ca.name = 'closing_auction'
print(f"新戦略sleeve: {len(ca)}日, {ca.index.min().date()}〜{ca.index.max().date()}")

# 結合 (新戦略のデータ存在日に既存をreindex)
df = sl.join(ca, how='outer').fillna(0.0)
# 共通期間 = 新戦略開始以降
start = ca.index.min(); end = min(sl.index.max(), ca.index.max())
df = df[(df.index >= start) & (df.index <= end)]
print(f"共通期間: {start.date()}〜{end.date()} ({len(df)}日)")

# ============================================
print("\n"+"="*76); print("A. 各戦略の日次Sharpe (共通期間)"); print("="*76)
print(f"  {'戦略':<26} {'日次Sharpe':>10} {'年率収益%':>10}")
for c in df.columns:
    s = df[c][df[c] != 0]
    print(f"  {c:<26} {sharpe(df[c]):>10.2f} {df[c].mean()*252*100:>10.1f}")

# ============================================
print("\n"+"="*76); print("B. 相関行列 (closing_auction vs 既存)"); print("="*76)
corr = df.corr()
print(corr.round(2).to_string())
print(f"\n  closing_auction の対既存 平均相関: {corr['closing_auction'].drop('closing_auction').mean():+.3f}")

# ============================================
print("\n"+"="*76); print("C. バスケット合成Sharpe (等加重)"); print("="*76)
old = df[['eneos_vwap_trend','vwap_morning_meanrevert','lasertec_ma25_support','bank_absorption']].mean(axis=1)
new = df.mean(axis=1)  # 5戦略
print(f"  {'構成':<28} {'日次Sharpe':>10} {'年率収益%':>10} {'年率vol%':>9}")
print(f"  {'既存4戦略 等加重':<28} {sharpe(old):>10.2f} {old.mean()*252*100:>10.1f} {old.std()*np.sqrt(252)*100:>9.1f}")
print(f"  {'+closing_auction (5戦略)':<28} {sharpe(new):>10.2f} {new.mean()*252*100:>10.1f} {new.std()*np.sqrt(252)*100:>9.1f}")
print(f"\n  分散効果: {sharpe(new)-sharpe(old):+.2f} (4戦略→5戦略)")

# closing_auction単独 Sharpe (発火日のみ)
ca_active = ca[ca != 0]
print(f"\n  参考: closing_auction単独(発火日のみ)Sharpe={sharpe(ca):.2f}, 発火{len(ca)}日/{len(df)}日")

df.to_csv(os.path.join(HERE,'basket5_daily.csv'))
print(f"\n  保存: basket5_daily.csv\n完了")
