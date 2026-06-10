"""
引けロングポジション + 日経ETFショートヘッジ 検証

戦略ベース: jump≤-75bps (auKabu50銘柄) 引値買い→翌9:00売り
ヘッジ追加: 日経225ETF (13210) を15:30に空売り→翌9:00返済

比較3案:
  A) ノーヘッジ   (従来)
  B) ドル中立    (ロング総額と同額のNikkeiショート)
  C) ベータ中立  (IS期間のベータ推定→OOSに適用)
"""
from __future__ import annotations
import sys
import pandas as pd
import numpy as np
import psycopg2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from numpy.polynomial import polynomial as P

sys.stdout.reconfigure(line_buffering=True)

IS_START  = "2024-11-05"
OOS_START = pd.Timestamp("2025-08-05")
OOS_END   = "2026-06-05"
INIT_CAP  = 10_000_000
COST_BPS  = 10
THR_BPS   = -75
NIKKEI_CODE = "13210"   # NEXT FUNDS 日経225 ETF

PF_ALL = {
    "57130","57110","57060","57140","50160","58010","58020","58030",
    "80350","68570","69200","61460","77350","40630","34360","77410",
    "69630","65260","99840","40620","67230","285A0","65250",
    "83060","83160","84110","70110","70130","70120","65030",
    "65010","67580","72030","72670","80580","80310",
    "69810","67620","69710","69760","40040","87660","16050",
    "68610","69540","94320","79740","99830",
}

conn = psycopg2.connect(host="localhost", dbname="market_data", user="postgres")

# ── 個別株データ取得 ──────────────────────────────────────────────────
print("[データ取得中: 個別株]")
codes_ph = ",".join([f"'{c}'" for c in PF_ALL])
df = pd.read_sql(f"""
    SELECT code, DATE(ts) AS date,
        MAX(CASE WHEN ts::time='15:24:00' THEN close END) AS c1524,
        MAX(CASE WHEN ts::time='15:30:00' THEN close END) AS c1530,
        MAX(CASE WHEN ts::time='09:00:00' THEN open  END) AS o0900
    FROM stocks_intraday
    WHERE code IN ({codes_ph})
      AND ts >= '{IS_START}' AND ts <= '{OOS_END} 23:59:59'
      AND ts::time IN ('15:24:00','15:30:00','09:00:00')
    GROUP BY code, DATE(ts) ORDER BY code, date
""", conn)

# ── 日経ETFデータ取得 ─────────────────────────────────────────────────
print("[データ取得中: 日経ETF 13210]")
nk = pd.read_sql(f"""
    SELECT DATE(ts) AS date,
        MAX(CASE WHEN ts::time='15:30:00' THEN close END) AS nk1530,
        MAX(CASE WHEN ts::time='09:00:00' THEN open  END) AS nk0900
    FROM stocks_intraday
    WHERE code = '{NIKKEI_CODE}'
      AND ts >= '{IS_START}' AND ts <= '{OOS_END} 23:59:59'
      AND ts::time IN ('15:30:00','09:00:00')
    GROUP BY DATE(ts) ORDER BY date
""", conn)
conn.close()

# ── 前処理 ────────────────────────────────────────────────────────────
df["date"] = pd.to_datetime(df["date"])
for col in ["c1524","c1530","o0900"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.sort_values(["code","date"]).reset_index(drop=True)
df["close_jump"] = df["c1530"] / df["c1524"] - 1
df["next_open"]  = df.groupby("code")["o0900"].shift(-1)
df["overnight"]  = df["next_open"] / df["c1530"] - 1
df = df[df["c1524"].notna() & df["c1530"].notna() & df["next_open"].notna()]
df = df[df["overnight"].abs() <= 0.10]
df = df[df["close_jump"].abs() <= 0.05]
df["jump_bps"]   = df["close_jump"] * 1e4
df["on_bps"]     = df["overnight"] * 1e4
df["period"]     = np.where(df["date"] < OOS_START, "IS", "OOS")

# 日経ETF: 翌日の09:00が当日のヘッジアンワインド
nk["date"] = pd.to_datetime(nk["date"])
nk = nk.sort_values("date").reset_index(drop=True)
nk["nk_next_open"] = nk["nk0900"].shift(-1)
nk["nk_on"] = nk["nk_next_open"] / nk["nk1530"] - 1   # 日経の夜間リターン
nk = nk[nk["nk1530"].notna() & nk["nk_next_open"].notna()]
nk_on = nk.set_index("date")["nk_on"]

# ── シグナル絞り込み ─────────────────────────────────────────────────
sig = df[df["jump_bps"] <= THR_BPS].copy()

# 日次集計: ロングサイドの等加重リターン
daily_long = sig.groupby("date").apply(
    lambda g: (g["on_bps"] - COST_BPS).mean() / 1e4,
    include_groups=False
).rename("long_ret")

daily_nk = nk_on.rename("nk_ret")

combined = pd.concat([daily_long, daily_nk], axis=1).dropna()
combined["period"] = np.where(combined.index < OOS_START, "IS", "OOS")

print(f"\nシグナル発火日: {len(combined)}日  (IS:{(combined['period']=='IS').sum()} OOS:{(combined['period']=='OOS').sum()})")

# ── ベータ推定 (IS期間のみ) ───────────────────────────────────────────
is_data = combined[combined["period"] == "IS"]
x = is_data["nk_ret"].values
y = is_data["long_ret"].values
beta_is, intercept = np.polyfit(x, y, 1)
ss_res = np.sum((y - (beta_is * x + intercept))**2)
ss_tot = np.sum((y - y.mean())**2)
r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
print(f"\n[ベータ推定 (IS期間)]")
print(f"  ロングポートフォリオ β vs 日経: {beta_is:.3f}")
print(f"  R²={r2:.3f}  N={len(is_data)}")

# ── 3戦略のリターン計算 ───────────────────────────────────────────────
# A: ノーヘッジ
combined["ret_no_hedge"] = combined["long_ret"]

# B: ドル中立 (1:1, ショートにもCOSTかかる)
combined["ret_dollar_neutral"] = combined["long_ret"] - combined["nk_ret"] - COST_BPS / 1e4

# C: ベータ中立 (IS推定βでOOSに適用)
combined["ret_beta_neutral"]   = combined["long_ret"] - beta_is * combined["nk_ret"] - (beta_is * COST_BPS) / 1e4

# ── 資金曲線 ─────────────────────────────────────────────────────────
all_dates = sorted(df["date"].unique())
date_idx = pd.DatetimeIndex(all_dates)

def capital_curve(ret_series):
    r = ret_series.reindex(date_idx).fillna(0)
    return INIT_CAP * (1 + r).cumprod()

cap_A = capital_curve(combined["ret_no_hedge"])
cap_B = capital_curve(combined["ret_dollar_neutral"])
cap_C = capital_curve(combined["ret_beta_neutral"])

# ── 統計サマリー ─────────────────────────────────────────────────────
def stats(curve, ret_series, label):
    final   = curve.iloc[-1]
    total_r = final / INIT_CAP - 1
    n_years = len(curve) / 252
    cagr    = (final / INIT_CAP) ** (1 / n_years) - 1
    dd      = curve / curve.cummax() - 1
    max_dd  = dd.min()
    fire    = ret_series.reindex(date_idx).dropna()
    fire    = fire[fire != 0]
    sharpe  = fire.mean() / fire.std() * np.sqrt(252) if len(fire) > 1 else float("nan")

    print(f"\n  ── {label} ──")
    print(f"  最終資産:  {final:>12,.0f}円  ({total_r*100:+.1f}%)")
    print(f"  CAGR:      {cagr*100:>+.1f}%/年")
    print(f"  最大DD:    {max_dd*100:.1f}%")
    print(f"  Sharpe:    {sharpe:+.2f}  (発火日={len(fire)}日)")

    for period, mask in [("IS", date_idx < OOS_START), ("OOS", date_idx >= OOS_START)]:
        c_p = curve[mask]
        r_p = ret_series.reindex(date_idx[mask]).dropna()
        r_p = r_p[r_p != 0]
        if len(c_p) < 2: continue
        ret_total = c_p.iloc[-1] / c_p.iloc[0] - 1
        n_y = len(c_p) / 252
        cagr_p = (1 + ret_total) ** (1 / n_y) - 1 if n_y > 0 else float("nan")
        sh_p = r_p.mean() / r_p.std() * np.sqrt(252) if len(r_p) > 1 else float("nan")
        dd_p = (c_p / c_p.cummax() - 1).min()
        print(f"    {period}: CAGR {cagr_p*100:+.1f}%  Sharpe {sh_p:+.2f}  MDD {dd_p*100:.1f}%  ({len(r_p)}日)")

print("\n" + "="*72)
print("ヘッジ戦略比較 (初期資金 1,000万円・再投資)")
print("="*72)
stats(cap_A, combined["ret_no_hedge"],       "A) ノーヘッジ  (ロングのみ)")
stats(cap_B, combined["ret_dollar_neutral"], "B) ドル中立   (日経ショート 1:1)")
stats(cap_C, combined["ret_beta_neutral"],   f"C) ベータ中立 (β={beta_is:.2f}×日経ショート)")

# ── 日経との相関分析 ─────────────────────────────────────────────────
print("\n" + "="*72)
print("相関・ヘッジ効果分析")
print("="*72)
for period in ["IS", "OOS", "全期間"]:
    if period == "IS":
        d = combined[combined["period"] == "IS"]
    elif period == "OOS":
        d = combined[combined["period"] == "OOS"]
    else:
        d = combined
    if len(d) < 5: continue
    corr_a = d["long_ret"].corr(d["nk_ret"])
    corr_b = d["ret_dollar_neutral"].corr(d["nk_ret"])
    corr_c = d["ret_beta_neutral"].corr(d["nk_ret"])
    std_a  = d["long_ret"].std() * 1e4
    std_b  = d["ret_dollar_neutral"].std() * 1e4
    std_c  = d["ret_beta_neutral"].std() * 1e4
    print(f"\n  [{period}] N={len(d)}日")
    print(f"    戦略vs日経 相関:  A={corr_a:+.3f}  B={corr_b:+.3f}  C={corr_c:+.3f}")
    print(f"    日次ボラ(bps):   A={std_a:.1f}  B={std_b:.1f}  C={std_c:.1f}")

# ── GD日の比較 ───────────────────────────────────────────────────────
print("\n" + "="*72)
print("日経大幅GD日 (-2%以下) のパフォーマンス")
print("="*72)
gd_days = combined[combined["nk_ret"] <= -0.02].copy()
print(f"\n  日経GD≤-2%の翌朝 N={len(gd_days)}日")
print(f"  {'日付':<12} {'日経ON':>8} {'A:無ヘッジ':>10} {'B:ドル中立':>10} {'C:β中立':>10}")
print("  " + "-"*55)
for dt, row in gd_days.iterrows():
    print(f"  {str(dt.date()):<12} {row['nk_ret']*100:>+7.1f}%  "
          f"{row['ret_no_hedge']*100:>+9.1f}%  "
          f"{row['ret_dollar_neutral']*100:>+9.1f}%  "
          f"{row['ret_beta_neutral']*100:>+9.1f}%")
if len(gd_days) > 0:
    print(f"\n  平均:")
    print(f"    A={gd_days['ret_no_hedge'].mean()*100:+.2f}%  "
          f"B={gd_days['ret_dollar_neutral'].mean()*100:+.2f}%  "
          f"C={gd_days['ret_beta_neutral'].mean()*100:+.2f}%")

# ── グラフ ───────────────────────────────────────────────────────────
print("\n[グラフ作成中...]")
fig, axes = plt.subplots(3, 1, figsize=(12, 14))
fig.suptitle("引けロング + 日経ショートヘッジ 比較 (1,000万円・再投資)", fontsize=14)

try:
    import matplotlib.font_manager as fm
    fp = fm.FontProperties(fname="/root/.fonts/NotoSansJP.ttf")
    plt.rcParams["font.family"] = fp.get_name()
except:
    pass

ax1, ax2, ax3 = axes

# 上段: 資金曲線
ax1.plot(cap_A.index, cap_A.values/1e4, label="A) ノーヘッジ", color="#e74c3c", lw=2)
ax1.plot(cap_B.index, cap_B.values/1e4, label="B) ドル中立(1:1)", color="#3498db", lw=2)
ax1.plot(cap_C.index, cap_C.values/1e4, label=f"C) β中立(β={beta_is:.2f})", color="#2ecc71", lw=2)
ax1.axhline(1000, color="black", lw=0.8, ls=":")
ax1.axvline(OOS_START, color="orange", lw=1.5, ls="--", alpha=0.7, label="IS/OOS境界")
ax1.set_ylabel("資産残高 (万円)")
ax1.legend(fontsize=9)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:,.0f}万"))
ax1.grid(True, alpha=0.3)
ax1.set_title("資金曲線", fontsize=11)

# 中段: ドローダウン
for cap, label, color in [
    (cap_A, "A) ノーヘッジ",       "#e74c3c"),
    (cap_B, "B) ドル中立",        "#3498db"),
    (cap_C, f"C) β中立",          "#2ecc71"),
]:
    dd = (cap / cap.cummax() - 1) * 100
    ax2.fill_between(dd.index, dd.values, 0, alpha=0.4, color=color, label=label)
ax2.axvline(OOS_START, color="orange", lw=1.5, ls="--", alpha=0.7)
ax2.set_ylabel("ドローダウン (%)")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_title("ドローダウン", fontsize=11)

# 下段: ヘッジ効果の散布図 (日経ON vs 各戦略リターン)
ax3.scatter(combined["nk_ret"]*100, combined["ret_no_hedge"]*100,
            alpha=0.5, color="#e74c3c", s=20, label="A) ノーヘッジ")
ax3.scatter(combined["nk_ret"]*100, combined["ret_beta_neutral"]*100,
            alpha=0.5, color="#2ecc71", s=20, label=f"C) β中立")
x_line = np.linspace(combined["nk_ret"].min()*100, combined["nk_ret"].max()*100, 100)
# ベータフィット線
ax3.plot(x_line, intercept*100 + beta_is*x_line,
         color="#e74c3c", lw=1.5, ls="--", label=f"フィット線 (β={beta_is:.2f})")
ax3.axhline(0, color="gray", lw=0.8)
ax3.axvline(0, color="gray", lw=0.8)
ax3.set_xlabel("日経ETF オーバーナイトリターン (%)")
ax3.set_ylabel("戦略リターン (%)")
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.set_title("日経との相関 (ヘッジ前後)", fontsize=11)

plt.tight_layout()
out = Path(__file__).parent / "nikkei_hedge.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"  保存: {out}")
print("\n[DONE]")
