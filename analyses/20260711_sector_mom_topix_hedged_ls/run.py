"""
セクターMOM等加重 + TOPIX ヘッジ L/S

仮説: 17業種等加重セクターMOM (L3K1, ADV>=10億) を
      TOPIX等額ショートでヘッジすると MDD が激減してリスク調整後αが改善する

前回分析 (20260706_sector_rotation_promotion) の月次リターン系列を再利用して
L/S 構造の絶対Sharpe・MDD・既存バスケットとの相関を検証する

教訓適用:
1. 超過リターン計算済みの系列を使う (同時反応の罠なし)
2. ヘッジコスト込み (TOPIX ETF借入 1%/年 + 既適用のLong側コスト)
3. IS/OOS分割で一般化を確認
4. バスケット相関で低相関寄与を実測
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

try:
    import matplotlib.font_manager as fm
    fp = fm.FontProperties(fname="/root/.fonts/NotoSansJP.ttf")
    plt.rcParams["font.family"] = fp.get_name()
except Exception:
    pass

HERE = Path(__file__).resolve().parent
PREV = HERE.parent / "20260706_sector_rotation_promotion"
BASKET = HERE.parent / "20260531_portfolio_daily_sharpe"

# -------------------------------------------------------------------
# 1. データ読み込み
# -------------------------------------------------------------------
# 前回の月次リターン系列 (ADV>=10億 等加重, コスト込み)
excess_df = pd.read_csv(PREV / "strategy_monthly_excess.csv", index_col=0)
abs_df    = pd.read_csv(PREV / "strategy_monthly_abs.csv",    index_col=0)

# 列名を統一
excess_m = excess_df.iloc[:, 0].rename("excess")
abs_m    = abs_df.iloc[:, 0].rename("abs_long")

df_m = pd.DataFrame({"excess": excess_m, "abs_long": abs_m})
df_m.index = pd.to_datetime(df_m.index + "-01")
df_m = df_m.sort_index()

# TOPIX 月次リターン (abs_long - excess で逆算)
df_m["topix_m"] = df_m["abs_long"] - df_m["excess"]

print(f"月次データ: {df_m.index[0].strftime('%Y-%m')} 〜 {df_m.index[-1].strftime('%Y-%m')} ({len(df_m)} ヶ月)")

# -------------------------------------------------------------------
# 2. L/S ポートフォリオ構築
# -------------------------------------------------------------------
# コスト
# Long側: 前回コスト (片側10bps × 入替率) が abs_long に既に込み
# Short側: TOPIX ETF 借入コスト ≈ 1%/年 = 8.3bps/月 + フリクション 2bps/月
HEDGE_COST_PER_MONTH = 0.0010  # 10bps/月 (1.2%/年相当, やや保守的)

# L/S の月次リターン = excess - hedge_cost
# ※ excess = abs_long - topix_m すでに「Long益 - TOPIX損益」
df_m["ls_gross"] = df_m["excess"]
df_m["ls_net"]   = df_m["excess"] - HEDGE_COST_PER_MONTH

# -------------------------------------------------------------------
# 3. IS / OOS 分割
# -------------------------------------------------------------------
IS_END  = "2021-06-01"
OOS_STA = "2021-07-01"

is_  = df_m[df_m.index <= IS_END]
oos_ = df_m[df_m.index >= OOS_STA]

def sharpe_annual(s, periods_per_year=12):
    s = s.dropna()
    if len(s) < 3 or s.std() == 0:
        return float("nan")
    return float(s.mean() / s.std() * np.sqrt(periods_per_year))

def max_dd(s):
    cum = (1 + s).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return float(dd.min())

def tstat(s):
    s = s.dropna()
    if len(s) < 3:
        return float("nan")
    return float(s.mean() / (s.std() / np.sqrt(len(s))))

def win_rate(s):
    s = s.dropna()
    return float((s > 0).mean())

def stats_block(label, s):
    return {
        "label": label,
        "n": len(s.dropna()),
        "ann_ret": float(s.dropna().mean() * 12),
        "sharpe": sharpe_annual(s),
        "t": tstat(s),
        "mdd": max_dd(s),
        "wr": win_rate(s),
    }

rows = []
for label, series in [
    ("Long-only (前回再現)",  df_m["abs_long"]),
    ("L/S gross",            df_m["ls_gross"]),
    ("L/S net (-10bps/月)",  df_m["ls_net"]),
]:
    rows.append(stats_block(f"{label} 全期間",    series))
    rows.append(stats_block(f"{label} IS",         series[series.index <= IS_END]))
    rows.append(stats_block(f"{label} OOS",        series[series.index >= OOS_STA]))

result = pd.DataFrame(rows).set_index("label")
pd.set_option("display.float_format", "{:.3f}".format)
print("\n=== 全体統計 ===")
print(result[["n","ann_ret","sharpe","t","mdd","wr"]].to_string())

# -------------------------------------------------------------------
# 4. 下落耐性: TOPIX下落月の条件付きリターン
# -------------------------------------------------------------------
down_months  = df_m[df_m["topix_m"] < 0]
up_months    = df_m[df_m["topix_m"] >= 0]
print(f"\n=== 下落耐性 (TOPIX下落月 n={len(down_months)}) ===")
for col, label in [("abs_long","Long-only"), ("ls_net","L/S net")]:
    d = down_months[col]
    u = up_months[col]
    print(f"  {label}:  下落月 avg={d.mean()*100:+.2f}%  Sh={sharpe_annual(d):.2f} | "
          f"上昇月 avg={u.mean()*100:+.2f}%  Sh={sharpe_annual(u):.2f}")

# 弱気レジーム (TOPIX trailing 12M negative)
topix_cum = (1 + df_m["topix_m"]).cumprod()
topix_trail12 = topix_cum / topix_cum.shift(12) - 1
bear_regime = topix_trail12 < 0
bear_df = df_m[bear_regime]
print(f"\n=== 弱気レジーム (TOPIX 12M-) n={len(bear_df)} ===")
for col, label in [("abs_long","Long-only"), ("ls_net","L/S net")]:
    b = bear_df[col]
    print(f"  {label}: avg={b.mean()*100:+.2f}%/月  Sh={sharpe_annual(b):.2f}  MDD={max_dd(b)*100:.1f}%")

# -------------------------------------------------------------------
# 5. 既存バスケットとの相関
# -------------------------------------------------------------------
sleeve = pd.read_csv(BASKET / "sleeve_daily_returns.csv", index_col=0, parse_dates=True)
# 月次集計
sleeve_m = sleeve.resample("MS").sum()

# L/S net を月次で合わせる
common_idx = df_m.index.intersection(sleeve_m.index)
if len(common_idx) > 5:
    merged = pd.DataFrame({"ls_net": df_m.loc[common_idx, "ls_net"]})
    for col in sleeve_m.columns:
        merged[col] = sleeve_m.loc[common_idx, col]
    corr = merged.corr()["ls_net"].drop("ls_net")
    print(f"\n=== 既存バスケットとの相関 (共通期間 n={len(common_idx)}) ===")
    for k, v in corr.items():
        print(f"  {k}: {v:+.3f}")
    print(f"  等加重平均相関: {corr.mean():+.3f}")

# -------------------------------------------------------------------
# 6. 累積曲線の可視化
# -------------------------------------------------------------------
fig = plt.figure(figsize=(14, 8))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

ax1 = fig.add_subplot(gs[0, :])
cum_long  = (1 + df_m["abs_long"]).cumprod()
cum_ls    = (1 + df_m["ls_net"]).cumprod()
cum_topix = (1 + df_m["topix_m"]).cumprod()
ax1.plot(df_m.index, cum_long,  label="Long-only (等加重L3K1)", lw=1.8)
ax1.plot(df_m.index, cum_ls,    label=f"L/S net (hedge -10bps/月)", lw=1.8, color="tab:green")
ax1.plot(df_m.index, cum_topix, label="TOPIX", lw=1.2, color="grey", alpha=0.7)
ax1.axvline(pd.Timestamp(IS_END), color="red", lw=0.8, linestyle="--", alpha=0.7, label="IS/OOS分割")
ax1.set_title("セクターMOM 等加重 L3K1: Long-only vs L/Sヘッジ")
ax1.set_ylabel("累積倍率")
ax1.legend(fontsize=9)
ax1.set_yscale("log")

# 月次ヒストグラム
ax2 = fig.add_subplot(gs[1, 0])
ax2.hist(df_m["abs_long"] * 100, bins=25, alpha=0.6, label="Long-only", color="tab:blue")
ax2.hist(df_m["ls_net"] * 100,   bins=25, alpha=0.6, label="L/S net",   color="tab:green")
ax2.set_title("月次リターン分布")
ax2.set_xlabel("月次リターン (%)")
ax2.legend(fontsize=9)

# DD比較
ax3 = fig.add_subplot(gs[1, 1])
dd_long = (cum_long / cum_long.cummax() - 1) * 100
dd_ls   = (cum_ls   / cum_ls.cummax()   - 1) * 100
ax3.fill_between(df_m.index, dd_long, 0, alpha=0.4, label="Long-only DD", color="tab:blue")
ax3.fill_between(df_m.index, dd_ls,   0, alpha=0.4, label="L/S net DD",   color="tab:green")
ax3.set_title("最大ドローダウン比較")
ax3.set_ylabel("DD (%)")
ax3.legend(fontsize=9)

fig.suptitle("セクターMOM TOPIX-ヘッジ L/S 検証 (ADV≥10億 等加重, 月次, コスト込み)", fontsize=12)
fig.savefig(HERE / "result.png", dpi=100, bbox_inches="tight")
print("\nsaved result.png")

# -------------------------------------------------------------------
# 7. IS/OOS サマリー表示
# -------------------------------------------------------------------
print("\n=== IS/OOS 比較サマリー ===")
print(f"{'':35s} {'全期間':>10s} {'IS':>10s} {'OOS':>10s}")
for col, label in [
    ("abs_long", "Long-only"),
    ("ls_gross", "L/S gross"),
    ("ls_net",   "L/S net (-10bps/月)"),
]:
    sh_all = sharpe_annual(df_m[col])
    sh_is  = sharpe_annual(df_m.loc[df_m.index <= IS_END, col])
    sh_oos = sharpe_annual(df_m.loc[df_m.index >= OOS_STA, col])
    mdd_all = max_dd(df_m[col])
    mdd_oos = max_dd(df_m.loc[df_m.index >= OOS_STA, col])
    print(f"  Sharpe {label:28s}  {sh_all:>+8.2f}  {sh_is:>+8.2f}  {sh_oos:>+8.2f}")
    print(f"  MDD    {label:28s}  {mdd_all*100:>+8.1f}%  {'':>9s}  {mdd_oos*100:>+8.1f}%")
    print()
