"""
Closing Auction Rebound — 全上場銘柄版

流動性フィルター別に検証し、auKabu銘柄との比較も行う。
その後、閾値 -75bps・銘柄選定へと進む。
"""
from __future__ import annotations
import sys, os
import pandas as pd
import numpy as np
import psycopg2

sys.stdout.reconfigure(line_buffering=True)

START = "2024-11-05"
END   = "2026-06-05"
OOS_START = pd.Timestamp("2025-08-05")

# auKabu PORTFOLIO_15 (非鉄8+半導体14)
PF15 = {
    "57130","57110","57060","57140","50160","58010","58020","58030",
    "80350","68570","69200","61460","77350","40630","34360","77410",
    "69630","65260","99840","40620","67230","285A0",
}

conn = psycopg2.connect(host="localhost", dbname="market_data", user="postgres")
cur = conn.cursor()

print("="*72)
print("Closing Auction Rebound — 全上場銘柄検証")
print("="*72)

# ── Step1: 流動性フィルター用 ADV を stocks_daily から取得 ────────────
print("\n[1] 銘柄別 ADV 取得 (stocks_daily, 2024-11-05〜) ...")
cur.execute(f"""
    SELECT code, AVG(turnover_value) AS adv
    FROM stocks_daily
    WHERE date >= '{START}' AND date <= '{END}'
      AND turnover_value > 0
    GROUP BY code
""")
adv_rows = cur.fetchall()
adv = {r[0]: float(r[1]) for r in adv_rows}
print(f"  銘柄数: {len(adv):,}")

# ── Step2: 15:24/15:30/翌09:00 データ取得 ─────────────────────────────
print("\n[2] 15:24 / 15:30 / 翌09:00 データ取得中 (全銘柄)...")
print("    (時間がかかります...)")

cur.execute(f"""
    SELECT
        code,
        DATE(ts) AS date,
        MAX(CASE WHEN ts::time = '15:24:00' THEN close END) AS c1524,
        MAX(CASE WHEN ts::time = '15:30:00' THEN close END) AS c1530,
        MAX(CASE WHEN ts::time = '09:00:00' THEN open  END) AS o0900
    FROM stocks_intraday
    WHERE ts >= '{START}'
      AND ts <= '{END} 23:59:59'
      AND ts::time IN ('15:24:00','15:30:00','09:00:00')
    GROUP BY code, DATE(ts)
    ORDER BY code, date
""")
rows = cur.fetchall()
conn.close()

df = pd.DataFrame(rows, columns=["code","date","c1524","c1530","o0900"])
df["date"] = pd.to_datetime(df["date"])
for col in ["c1524","c1530","o0900"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
print(f"  取得: {len(df):,} 行 ({df['code'].nunique():,} 銘柄, {df['date'].nunique()} 日)")

# ── Step3: close_jump / overnight 計算 ────────────────────────────────
print("\n[3] jump / overnight 計算・クレンジング ...")
df = df.sort_values(["code","date"]).reset_index(drop=True)
df["close_jump"] = df["c1530"] / df["c1524"] - 1
df["next_open"]  = df.groupby("code")["o0900"].shift(-1)
df["overnight"]  = df["next_open"] / df["c1530"] - 1

n0 = len(df)
df = df[df["c1524"].notna() & df["c1530"].notna() & df["next_open"].notna()]
df = df[df["overnight"].abs() <= 0.10]
df = df[df["close_jump"].abs() <= 0.05]
print(f"  クレンジング: {n0:,} → {len(df):,}")

df["jump_bps"] = df["close_jump"] * 1e4
df["on_bps"]   = df["overnight"] * 1e4
df["adv"]      = df["code"].map(adv).fillna(0)
df["period"]   = np.where(df["date"] >= OOS_START, "OOS", "IS")
df["in_pf15"]  = df["code"].isin(PF15)

def sharpe(x, ann=np.sqrt(252)):
    x = pd.Series(x).dropna()
    if len(x) < 10 or x.std() == 0: return float("nan")
    return float(x.mean() / x.std() * ann)

def row_str(label, ser, cost=10):
    ret = ser - cost
    n = len(ret)
    if n < 10: return
    sh = sharpe(ret); wr = (ret>0).mean()*100
    t  = ret.mean()/(ret.std()/np.sqrt(n))
    print(f"  {label:<30s} n={n:6,}  mean={ret.mean():+6.1f}bps  Sharpe={sh:+5.2f}  勝率={wr:4.1f}%  t={t:+5.2f}")

# ── A. 流動性フィルター × 閾値 -50bps ────────────────────────────────
print("\n" + "="*72)
print("A. 流動性フィルター別 (jump≤-50bps, cost=10bps)")
print("="*72)
print(f"  {'流動性フィルター':<30s} {'n':>7}  {'mean bps':>9}  {'Sharpe':>7}  {'勝率%':>6}  {'t':>6}")
print("  " + "-"*68)
for adv_thr, label in [(0,"全銘柄"),(1e8,"ADV≥1億"),(5e8,"ADV≥5億"),(1e9,"ADV≥10億"),(5e9,"ADV≥50億"),(1e10,"ADV≥100億")]:
    sig = df[(df["adv"]>=adv_thr) & (df["jump_bps"]<=-50)]["on_bps"]
    row_str(label, sig, cost=10)

# ── B. 閾値感度 (ADV≥10億) ────────────────────────────────────────────
print("\n" + "="*72)
print("B. 閾値感度 (ADV≥10億, cost=10bps)")
print("="*72)
print(f"  {'閾値':<30s} {'n':>7}  {'mean bps':>9}  {'Sharpe':>7}  {'勝率%':>6}  {'t':>6}")
print("  " + "-"*68)
liq = df[df["adv"]>=1e9]
for thr, label in [(-25,"jump≤-25bps"),(-50,"jump≤-50bps"),(-75,"jump≤-75bps"),(-100,"jump≤-100bps"),(-150,"jump≤-150bps")]:
    sig = liq[liq["jump_bps"]<=thr]["on_bps"]
    row_str(label, sig, cost=10)

# ── C. IS / OOS (ADV≥10億, jump≤-50bps) ──────────────────────────────
print("\n" + "="*72)
print("C. IS / OOS 比較 (ADV≥10億, jump≤-50bps, cost=10bps)")
print("="*72)
base = liq[liq["jump_bps"]<=-50]
for period, label in [("IS","IS  (2024-11〜2025-08)"),("OOS","OOS (2025-08〜2026-06)"),("全期間","全期間")]:
    if period == "全期間":
        sub = base["on_bps"]
    else:
        sub = base[base["period"]==period]["on_bps"]
    row_str(label, sub, cost=10)

# ── D. 閾値-75bps IS/OOS (ADV≥10億) ─────────────────────────────────
print("\n" + "="*72)
print("D. 閾値 -75bps IS/OOS (ADV≥10億, cost=10bps)")
print("="*72)
base75 = liq[liq["jump_bps"]<=-75]
for period, label in [("IS","IS"),("OOS","OOS"),("全期間","全期間")]:
    sub = base75["on_bps"] if period=="全期間" else base75[base75["period"]==period]["on_bps"]
    row_str(label, sub, cost=10)

# ── E. auKabu PF15 vs 全銘柄(ADV≥10億) 比較 ─────────────────────────
print("\n" + "="*72)
print("E. auKabu PF15 vs 全銘柄(ADV≥10億) 比較 (jump≤-50bps, cost=10bps)")
print("="*72)
row_str("PF15 (非鉄+半導体22)", df[df["in_pf15"] & (df["jump_bps"]<=-50)]["on_bps"], 10)
row_str("全銘柄 ADV≥10億",       liq[liq["jump_bps"]<=-50]["on_bps"], 10)
row_str("全銘柄 ADV≥10億 (PF15除く)", liq[~liq["in_pf15"] & (liq["jump_bps"]<=-50)]["on_bps"], 10)

# ── F. 銘柄別 成績 Top20 (ADV≥10億, jump≤-50bps, cost=10bps, n≥10) ──
print("\n" + "="*72)
print("F. 銘柄別 成績 Top20 (ADV≥10億, jump≤-50bps, cost=10bps, n≥10)")
print("="*72)
stock_res = []
for code, grp in liq[liq["jump_bps"]<=-50].groupby("code"):
    ret = grp["on_bps"] - 10
    if len(ret) < 10: continue
    stock_res.append({
        "code": code,
        "n": len(ret),
        "mean": ret.mean(),
        "sharpe": sharpe(ret),
        "winrate": (ret > 0).mean() * 100,
        "adv_B": adv.get(code, 0)/1e8,
        "pf15": "★" if code in PF15 else "",
    })
sr = pd.DataFrame(stock_res).sort_values("sharpe", ascending=False)
print(f"\n  {'code':>6}  {'n':>4}  {'mean':>9}  {'Sharpe':>7}  {'勝率%':>6}  {'ADV(億)':>8}  PF15")
print("  " + "-"*58)
for _, r in sr.head(20).iterrows():
    print(f"  {r['code']:>6}  {int(r['n']):>4}  {r['mean']:+9.1f}bps  {r['sharpe']:+7.2f}  "
          f"{r['winrate']:6.1f}%  {r['adv_B']:>8.0f}  {r['pf15']}")

# ── G. 発火頻度 ──────────────────────────────────────────────────────
print("\n" + "="*72)
print("G. 発火頻度 (ADV≥10億, jump≤-50bps)")
print("="*72)
freq = liq[liq["jump_bps"]<=-50].groupby("date").size()
print(f"  発火日数: {len(freq)} 日 / 全 {df['date'].nunique()} 日")
print(f"  1日あたり平均: {freq.mean():.1f} 銘柄  中央値: {freq.median():.0f}  最大: {freq.max()}")
print(f"  分布: 0銘柄={df['date'].nunique()-len(freq)}日  1銘柄={(freq==1).sum()}日  "
      f"2〜5銘柄={(freq.between(2,5)).sum()}日  6+銘柄={(freq>=6).sum()}日")

print("\n[DONE]")
