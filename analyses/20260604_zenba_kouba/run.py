"""
前場暴落 → 後場リターン分析
前場が大きく下落した日に、後場で回復するのか継続下落するのか。
個別銘柄 + 市場集計(等ウェイト)の両面で検証。
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
from scipy import stats

# ── データ読み込み ─────────────────────────────────────────────────
df = pd.read_csv("raw.csv", parse_dates=["dt"])

# 外れ値除去（前場/後場 |ret|>15% はデータ異常）
df = df[(df.am_ret.abs() <= 0.15) & (df.pm_ret.abs() <= 0.15)]
print(f"全行: {len(df):,}行 / {df.dt.nunique()}営業日 / {df.code.nunique()}銘柄")

# ── 市場集計: 日次等ウェイト平均 ─────────────────────────────────
mkt = df.groupby("dt")[["am_ret","pm_ret"]].mean().rename(
    columns={"am_ret":"mkt_am","pm_ret":"mkt_pm"})
mkt = mkt.sort_index()
print(f"\n市場平均 前場: mean={mkt.mkt_am.mean()*100:.3f}%, std={mkt.mkt_am.std()*100:.2f}%")
print(f"市場平均 後場: mean={mkt.mkt_pm.mean()*100:.3f}%, std={mkt.mkt_pm.std()*100:.2f}%")

# ── バケット分析 ─────────────────────────────────────────────────
BINS = [-0.999, -0.03, -0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02, 0.03, 0.999]
LABELS = ["≤-3%", "-3〜-2%", "-2〜-1%", "-1〜-0.5%",
          "-0.5〜0%", "0〜+0.5%", "+0.5〜+1%", "+1〜+2%", "+2〜+3%", "≥+3%"]

# 市場レベルのバケット
mkt["am_bucket"] = pd.cut(mkt.mkt_am, bins=BINS, labels=LABELS)

print("\n" + "="*72)
print("【市場集計】前場リターン → 後場リターン (等ウェイト平均)")
print("="*72)
print(f"{'前場リターン':>12} {'N':>4} {'後場平均':>9} {'後場中央':>9} {'後場勝率':>9} {'t値':>7}")
print("-"*72)
for lbl in LABELS:
    sub = mkt[mkt.am_bucket == lbl].mkt_pm
    if len(sub) < 3: continue
    mean_ = sub.mean()*100
    med_  = sub.median()*100
    win_  = (sub > 0).mean()*100
    t_, _ = stats.ttest_1samp(sub, 0)
    print(f"  {lbl:>10}  {len(sub):>4}  {mean_:>+8.3f}%  {med_:>+8.3f}%  {win_:>8.1f}%  {t_:>7.2f}")

# ── 個別銘柄レベルのバケット ────────────────────────────────────
df["am_bucket"] = pd.cut(df.am_ret, bins=BINS, labels=LABELS)

print("\n" + "="*72)
print("【個別銘柄】前場リターン → 後場リターン (全銘柄プール)")
print("="*72)
print(f"{'前場リターン':>12} {'N':>6} {'後場平均':>9} {'後場中央':>9} {'後場勝率':>9} {'t値':>7}")
print("-"*72)
for lbl in LABELS:
    sub = df[df.am_bucket == lbl].pm_ret
    if len(sub) < 10: continue
    mean_ = sub.mean()*100
    med_  = sub.median()*100
    win_  = (sub > 0).mean()*100
    t_, _ = stats.ttest_1samp(sub, 0)
    print(f"  {lbl:>10}  {len(sub):>6}  {mean_:>+8.3f}%  {med_:>+8.3f}%  {win_:>8.1f}%  {t_:>7.2f}")

# ── 相関・線形回帰 ────────────────────────────────────────────────
print("\n" + "="*72)
print("【個別銘柄】前場リターン vs 後場リターン 線形回帰")
print("="*72)
x = df.am_ret.values
y = df.pm_ret.values
slope, intercept, r, p, se = stats.linregress(x, y)
print(f"  回帰係数 β = {slope:+.4f}  (前場-1%あたり後場 {slope*-0.01*100:+.3f}%)")
print(f"  切片 α    = {intercept*100:+.4f}%")
print(f"  R²        = {r**2:.4f}")
print(f"  p値       = {p:.4e}")
print(f"  相関係数  = {r:+.4f}")

# ── 市場暴落日の特定 ─────────────────────────────────────────────
print("\n" + "="*72)
print("【市場集計】前場-2%以下の日の詳細")
print("="*72)
crash_days = mkt[mkt.mkt_am <= -0.02].sort_values("mkt_am")
print(f"前場-2%以下: {len(crash_days)}日")
print(f"  → 後場プラス(回復): {(crash_days.mkt_pm > 0).sum()}日 ({(crash_days.mkt_pm > 0).mean()*100:.1f}%)")
print(f"  → 後場マイナス(継続): {(crash_days.mkt_pm < 0).sum()}日 ({(crash_days.mkt_pm < 0).mean()*100:.1f}%)")
print(f"  → 後場平均リターン: {crash_days.mkt_pm.mean()*100:+.3f}%")
print()
print(crash_days[["mkt_am","mkt_pm"]].assign(
    mkt_am=lambda d: d.mkt_am*100, mkt_pm=lambda d: d.mkt_pm*100
).rename(columns={"mkt_am":"前場%","mkt_pm":"後場%"}).to_string(float_format=lambda x: f"{x:+.2f}"))

# ── 結果保存 ──────────────────────────────────────────────────────
mkt.to_csv("market_daily.csv")
df[["code","dt","am_ret","pm_ret","am_bucket"]].to_csv("individual.csv", index=False)
print("\n保存: market_daily.csv, individual.csv")
