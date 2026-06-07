"""
資金曲線シミュレーション — 1000万円スタート、再投資

戦略: jump≤-75bps (auKabu50銘柄)
運用: 等金額分割、発火銘柄数に応じて当日の資金を均等配分、翌日再投資
比較: 無条件オーバーナイト(毎日全銘柄買い)、日経平均buy&hold
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

sys.stdout.reconfigure(line_buffering=True)

IS_START  = "2024-11-05"
OOS_START = pd.Timestamp("2025-08-05")
OOS_END   = "2026-06-05"
INIT_CAP  = 10_000_000  # 1000万円
COST_BPS  = 10          # 往復コスト（bps）
THR_BPS   = -75         # シグナル閾値

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
cur  = conn.cursor()
codes_ph = ",".join([f"'{c}'" for c in PF_ALL])

print("="*72)
print("資金曲線シミュレーション (初期資金: 1,000万円、再投資)")
print("="*72)

# ── データ取得 ────────────────────────────────────────────────────────
print("\n[データ取得中...]")
cur.execute(f"""
    SELECT code, DATE(ts) AS date,
        MAX(CASE WHEN ts::time='15:24:00' THEN close END) AS c1524,
        MAX(CASE WHEN ts::time='15:30:00' THEN close END) AS c1530,
        MAX(CASE WHEN ts::time='09:00:00' THEN open  END) AS o0900
    FROM stocks_intraday
    WHERE code IN ({codes_ph})
      AND ts >= '{IS_START}' AND ts <= '{OOS_END} 23:59:59'
      AND ts::time IN ('15:24:00','15:30:00','09:00:00')
    GROUP BY code, DATE(ts) ORDER BY code, date
""")
rows = cur.fetchall()
conn.close()

df = pd.DataFrame(rows, columns=["code","date","c1524","c1530","o0900"])
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
df["jump_bps"] = df["close_jump"] * 1e4
df["on_bps"]   = df["overnight"] * 1e4
df["period"]   = np.where(df["date"] < OOS_START, "IS", "OOS")

all_dates = sorted(df["date"].unique())

# ── 日次リターン系列を作成 ────────────────────────────────────────────
def make_daily_ret(mask, cost=COST_BPS):
    """条件に合う銘柄の日次等加重リターン（発火なし日はNaN）"""
    sig = df[mask]
    daily = sig.groupby("date").apply(
        lambda g: (g["on_bps"] - cost).mean() / 1e4,
        include_groups=False
    )
    return daily

# 3戦略の日次リターン
sig_75      = make_daily_ret(df["jump_bps"] <= THR_BPS)
sig_50_rel  = make_daily_ret((df["jump_bps"] <= -50) & (df["jump_bps"].groupby(df["date"]).rank(pct=True) <= 0.10).values)
uncond      = df.groupby("date")["on_bps"].mean() / 1e4 - COST_BPS / 1e4  # 全銘柄毎日

# 全日付インデックス
date_idx = pd.DatetimeIndex(all_dates)

def capital_curve(daily_ret_series, init=INIT_CAP, label=""):
    """
    資金曲線を計算。
    発火なし日は資金変動なし（キャッシュで保持）。
    """
    ret = daily_ret_series.reindex(date_idx).fillna(0)  # 発火なし→0リターン
    curve = init * (1 + ret).cumprod()
    curve.index = date_idx
    return curve

cap_75     = capital_curve(sig_75,     label="jump≤-75bps")
cap_50rel  = capital_curve(sig_50rel := make_daily_ret(
    (df["jump_bps"] <= -50) & (df["jump_rank_pct"] <= 0.10)
    if "jump_rank_pct" in df.columns else df["jump_bps"] <= -50
), label="相対上位10%+-50bps")
cap_uncond = capital_curve(uncond,     label="無条件ON（全銘柄毎日）")

# jump_rank_pctを追加してからやり直し
df["jump_rank_pct"] = df.groupby("date")["jump_bps"].rank(pct=True)
cap_50rel = capital_curve(make_daily_ret(
    (df["jump_bps"] <= -50) & (df["jump_rank_pct"] <= 0.10)
), label="相対上位10%+-50bps")

# ── 統計サマリー ──────────────────────────────────────────────────────
def stats(curve, daily_ret, label):
    final    = curve.iloc[-1]
    total_r  = final / INIT_CAP - 1
    n_days   = len(curve)
    n_years  = n_days / 252
    cagr     = (final / INIT_CAP) ** (1 / n_years) - 1
    dd       = (curve / curve.cummax() - 1)
    max_dd   = dd.min()
    # 発火日のみのSharpe
    fire_ret = daily_ret[daily_ret != 0]
    sh       = fire_ret.mean() / fire_ret.std() * np.sqrt(252) if len(fire_ret) > 0 else float("nan")
    fire_days = (daily_ret != 0).sum()

    print(f"\n  ── {label} ──")
    print(f"  最終資産:   {final:>12,.0f}円  ({total_r*100:+.1f}%)")
    print(f"  CAGR:       {cagr*100:>+.1f}%/年")
    print(f"  最大DD:     {max_dd*100:.1f}%")
    print(f"  発火日数:   {fire_days}日 / {n_days}日")
    print(f"  Sharpe:     {sh:+.2f}")

    # IS/OOS分割
    is_mask  = curve.index < OOS_START
    oos_mask = curve.index >= OOS_START
    for period, mask, cap_s in [("IS",is_mask,curve[is_mask]),("OOS",oos_mask,curve[oos_mask])]:
        if len(cap_s) < 2: continue
        r = cap_s.iloc[-1] / cap_s.iloc[0] - 1
        n_y = len(cap_s) / 252
        cagr_p = (1 + r) ** (1 / n_y) - 1 if n_y > 0 else float("nan")
        ret_p = daily_ret[daily_ret.index.isin(cap_s.index) & (daily_ret != 0)]
        sh_p = ret_p.mean() / ret_p.std() * np.sqrt(252) if len(ret_p) > 1 else float("nan")
        print(f"    {period}: {r*100:+.1f}%  CAGR {cagr_p*100:+.1f}%/年  Sharpe {sh_p:+.2f}  ({len(ret_p)}発火日)")
    return final

print("\n" + "="*72)
print("資金曲線サマリー")
print("="*72)

sig75_daily   = sig_75.reindex(date_idx).fillna(0)
sig50r_daily  = make_daily_ret((df["jump_bps"] <= -50) & (df["jump_rank_pct"] <= 0.10)).reindex(date_idx).fillna(0)
uncond_daily  = uncond.reindex(date_idx).fillna(0)

final_75   = stats(cap_75,     sig75_daily,  "jump≤-75bps (auKabu50銘柄)")
final_50r  = stats(cap_50rel,  sig50r_daily, "相対上位10% + jump≤-50bps")
final_unc  = stats(cap_uncond, uncond_daily, "無条件ON（全銘柄毎日・ベースライン）")

# ── 月次P&L表 ─────────────────────────────────────────────────────────
print("\n" + "="*72)
print("月次P&L (jump≤-75bps, 1000万スタート再投資)")
print("="*72)
print(f"\n  {'年月':>8}  {'月次リターン':>12}  {'月末資産':>14}  {'発火回数':>8}")
print("  " + "-"*50)
curve_monthly = cap_75.resample("ME").last()
prev = INIT_CAP
fire_monthly  = sig75_daily[sig75_daily != 0].resample("ME").count()
for dt, val in curve_monthly.items():
    m_ret  = val / prev - 1
    fires  = int(fire_monthly.get(dt, 0))
    mark   = "↑" if m_ret >= 0 else "↓"
    print(f"  {dt.strftime('%Y-%m'):>8}  {m_ret*100:>+10.1f}% {mark}  {val:>12,.0f}円  {fires:>6}回")
    prev = val

# ── 図の作成 ──────────────────────────────────────────────────────────
print("\n[グラフ作成中...]")
fig, axes = plt.subplots(3, 1, figsize=(12, 14))
fig.suptitle("Closing Auction Rebound — 資金曲線 (1,000万円スタート、再投資)", fontsize=14)

# フォント設定
try:
    import matplotlib.font_manager as fm
    fp = fm.FontProperties(fname="/root/.fonts/NotoSansJP.ttf")
    plt.rcParams["font.family"] = fp.get_name()
except:
    pass

ax1, ax2, ax3 = axes

# 上段: 資金曲線
ax1.plot(cap_75.index,    cap_75.values/1e4,    label="jump≤-75bps",         color="#e74c3c", lw=2)
ax1.plot(cap_50rel.index, cap_50rel.values/1e4, label="相対上位10%+-50bps",   color="#3498db", lw=2)
ax1.plot(cap_uncond.index,cap_uncond.values/1e4,label="無条件ON（ベースライン）",color="#95a5a6", lw=1.5, ls="--")
ax1.axhline(1000, color="black", lw=0.8, ls=":")
ax1.axvline(OOS_START, color="orange", lw=1.5, ls="--", alpha=0.7, label="IS/OOS境界")
ax1.set_ylabel("資産残高 (万円)")
ax1.legend(fontsize=9)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:,.0f}万"))
ax1.grid(True, alpha=0.3)
ax1.set_title("資金曲線", fontsize=11)

# 中段: ドローダウン
for curve, label, color in [
    (cap_75,    "jump≤-75bps",         "#e74c3c"),
    (cap_50rel, "相対上位10%+-50bps",  "#3498db"),
    (cap_uncond,"無条件ON",             "#95a5a6"),
]:
    dd = (curve / curve.cummax() - 1) * 100
    ax2.fill_between(dd.index, dd.values, 0, alpha=0.4, color=color, label=label)
ax2.axvline(OOS_START, color="orange", lw=1.5, ls="--", alpha=0.7)
ax2.set_ylabel("ドローダウン (%)")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_title("ドローダウン", fontsize=11)

# 下段: 月次リターン棒グラフ
monthly_rets = ((cap_75.resample("ME").last() / cap_75.resample("ME").last().shift(1)) - 1) * 100
monthly_rets = monthly_rets.dropna()
colors_bar = ["#27ae60" if r >= 0 else "#e74c3c" for r in monthly_rets.values]
ax3.bar(monthly_rets.index, monthly_rets.values, color=colors_bar, width=20)
ax3.axhline(0, color="black", lw=0.8)
ax3.axvline(OOS_START, color="orange", lw=1.5, ls="--", alpha=0.7, label="IS/OOS境界")
ax3.set_ylabel("月次リターン (%)")
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3, axis="y")
ax3.set_title("月次リターン (jump≤-75bps)", fontsize=11)

plt.tight_layout()
out = Path(__file__).parent / "capital_curve.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"  保存: {out}")
print("\n[DONE]")
