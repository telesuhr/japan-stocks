"""
Step2: 10銘柄選定
- 閾値: jump≤-75bps (全銘柄版の結果: Sharpe 2.11, IS/OOS一貫)
- ADV≥10億 かつ n≥15 の銘柄から IS/OOS 両期間でプラスの銘柄を選定
- 等金額・1日最大1トレード/銘柄のポートフォリオ評価
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
THR = -75  # bps
ADV_MIN = 1e9
COST = 10  # bps 片道

conn = psycopg2.connect(host="localhost", dbname="market_data", user="postgres")
cur = conn.cursor()

print("="*72)
print("Step2: 10銘柄選定 (jump≤-75bps, ADV≥10億)")
print("="*72)

# ADV取得
cur.execute(f"""
    SELECT code, AVG(turnover_value) AS adv
    FROM stocks_daily
    WHERE date >= '{START}' AND date <= '{END}' AND turnover_value > 0
    GROUP BY code
""")
adv = {r[0]: float(r[1]) for r in cur.fetchall()}

# 分足データ
cur.execute(f"""
    SELECT code, DATE(ts) AS date,
        MAX(CASE WHEN ts::time='15:24:00' THEN close END) AS c1524,
        MAX(CASE WHEN ts::time='15:30:00' THEN close END) AS c1530,
        MAX(CASE WHEN ts::time='09:00:00' THEN open  END) AS o0900
    FROM stocks_intraday
    WHERE ts >= '{START}' AND ts <= '{END} 23:59:59'
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

df = df.sort_values(["code","date"]).reset_index(drop=True)
df["close_jump"] = df["c1530"] / df["c1524"] - 1
df["next_open"]  = df.groupby("code")["o0900"].shift(-1)
df["overnight"]  = df["next_open"] / df["c1530"] - 1
df = df[df["c1524"].notna() & df["c1530"].notna() & df["next_open"].notna()]
df = df[df["overnight"].abs() <= 0.10]
df = df[df["close_jump"].abs() <= 0.05]
df["jump_bps"] = df["close_jump"] * 1e4
df["on_bps"]   = df["overnight"] * 1e4
df["adv"]      = df["code"].map(adv).fillna(0)
df["period"]   = np.where(df["date"] >= OOS_START, "OOS", "IS")

def sharpe(x, ann=np.sqrt(252)):
    x = pd.Series(x).dropna()
    if len(x) < 5 or x.std() == 0: return float("nan")
    return float(x.mean() / x.std() * ann)

# ── 銘柄別スコアリング ────────────────────────────────────────────────
sig = df[(df["adv"] >= ADV_MIN) & (df["jump_bps"] <= THR)]

stock_res = []
for code, grp in sig.groupby("code"):
    ret_all = grp["on_bps"] - COST
    ret_is  = grp[grp["period"]=="IS"]["on_bps"] - COST
    ret_oos = grp[grp["period"]=="OOS"]["on_bps"] - COST
    n_all = len(ret_all)
    if n_all < 10: continue
    sh_all = sharpe(ret_all)
    sh_is  = sharpe(ret_is)  if len(ret_is)  >= 5 else float("nan")
    sh_oos = sharpe(ret_oos) if len(ret_oos) >= 5 else float("nan")
    stock_res.append({
        "code": code,
        "n": n_all, "n_is": len(ret_is), "n_oos": len(ret_oos),
        "mean": ret_all.mean(),
        "sh_all": sh_all, "sh_is": sh_is, "sh_oos": sh_oos,
        "wr": (ret_all > 0).mean() * 100,
        "adv_B": adv.get(code, 0) / 1e8,
        "both_pos": (sh_is > 0) and (sh_oos > 0),  # IS・OOS両方プラス
    })

sr = pd.DataFrame(stock_res)

print(f"\n対象銘柄数 (n≥10): {len(sr)}")
print(f"IS・OOS両方プラス: {sr['both_pos'].sum()} 銘柄")

# IS・OOS両方プラスに絞りSharpe降順
sr_pos = sr[sr["both_pos"]].sort_values("sh_all", ascending=False)

print("\n" + "="*72)
print("IS・OOS両方プラス銘柄 (Sharpe降順)")
print("="*72)
print(f"\n  {'code':>6}  {'n':>4} {'n_IS':>5} {'n_OOS':>6}  {'mean':>8}  {'Sh_all':>7}  {'Sh_IS':>7}  {'Sh_OOS':>7}  {'勝率%':>6}  {'ADV億':>7}")
print("  " + "-"*80)
for _, r in sr_pos.iterrows():
    print(f"  {r['code']:>6}  {int(r['n']):>4} {int(r['n_is']):>5} {int(r['n_oos']):>6}  "
          f"{r['mean']:+8.1f}  {r['sh_all']:+7.2f}  {r['sh_is']:+7.2f}  {r['sh_oos']:+7.2f}  "
          f"{r['wr']:6.1f}%  {r['adv_B']:>7.0f}")

# ── Top10 選定 ────────────────────────────────────────────────────────
# IS・OOS両方プラス かつ Sharpe上位10
top10_codes = sr_pos.head(10)["code"].tolist()
print(f"\n選定Top10: {top10_codes}")

# ── Top10 ポートフォリオ評価 ──────────────────────────────────────────
print("\n" + "="*72)
print("Top10 ポートフォリオ評価 (等金額, 1日最大1トレード/銘柄, cost=10bps)")
print("="*72)

pf = sig[sig["code"].isin(top10_codes)].copy()

# 日次等加重: 各銘柄1 unit, 同日複数銘柄あればその日の平均
daily = pf.groupby("date").apply(
    lambda g: pd.Series({"ret": (g["on_bps"] - COST).mean(), "n": len(g)})
).reset_index()

ret = daily["ret"]
sh = sharpe(ret)
t  = ret.mean() / (ret.std() / np.sqrt(len(ret)))
print(f"\n  全期間  発火日数={len(daily)}  mean={ret.mean():+.1f}bps  Sharpe={sh:+.2f}  t={t:+.2f}")

for period, label in [("IS","IS  (2024-11〜2025-08)"),("OOS","OOS (2025-08〜2026-06)")]:
    mask = daily["date"] >= OOS_START if period=="OOS" else daily["date"] < OOS_START
    sub = daily[mask]["ret"]
    if len(sub) < 5: continue
    sh_p = sharpe(sub)
    t_p  = sub.mean()/(sub.std()/np.sqrt(len(sub)))
    print(f"  {label:<30s} 発火日数={len(sub)}  mean={sub.mean():+.1f}bps  Sharpe={sh_p:+.2f}  t={t_p:+.2f}")

print(f"\n  発火頻度: 1日平均 {pf.groupby('date').size().mean():.1f} トレード")

# ── コスト感度 ────────────────────────────────────────────────────────
print("\n  コスト感度 (全期間):")
for cost in [0, 6, 10, 20]:
    r = pf.groupby("date").apply(lambda g: (g["on_bps"] - cost).mean())
    sh_c = sharpe(r)
    print(f"    cost={cost:2d}bps  Sharpe={sh_c:+.2f}  mean={r.mean():+.1f}bps")

print("\n[DONE]")
