"""
Closing Auction Rebound — auKabu ポートフォリオ銘柄に絞った検証

戦略:
  - 15:30引け板寄せで close_jump = (15:30終値)/(15:24終値) - 1 ≤ -50bps の銘柄を買い
  - 翌営業日 09:00 寄りで売却
  - 期間: 2024-11-05〜 (クロージングオークション新制度以降)
  - 対象: auKabu PORTFOLIO_ALL (50銘柄超) と PORTFOLIO_15 (非鉄8+半導体14=22銘柄)

先行研究 (20260531_closing_auction_refined/exec):
  - 流動性上位200銘柄: net Sharpe 2.77 (往復6bps), 2.00 (10bps), 勝率57%
  - 本検証では auKabu 対象銘柄に絞った場合の再現性を確認する
"""
from __future__ import annotations
import sys, os
import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor

sys.stdout.reconfigure(line_buffering=True)

# ── auKabu ポートフォリオ銘柄 (4桁→5桁変換) ──────────────────────────
PORTFOLIO_15_4D = [
    "5713","5711","5706","5714","5016","5801","5802","5803",  # 非鉄8
    "8035","6857","6920","6146","7735","4063","3436","7741",  # 半導体8
    "6963","6526","9984","4062","6723","285A",               # 半導体6 (計14)
]
PORTFOLIO_ALL_4D = PORTFOLIO_15_4D + [
    "8306","8316","8411",           # 銀行3
    "7011","7013","7012","6503",    # 機械/防衛4
    "6501","6758",                  # 総合電機2
    "7203","7267",                  # 自動車2
    "8058","8031",                  # 商社2
    "6981","6762","6971","6976",    # 電子部品4
    "4004",                         # 素材1
    "8766",                         # 保険1
    "1605",                         # エネルギー1
    "6861","6954","9432","7974","9983",  # その他5
]

def to_code5(c: str) -> str:
    """4桁→5桁 (末尾0付加)。285A → 285A0"""
    return c + "0"

PF15_5D  = [to_code5(c) for c in PORTFOLIO_15_4D]
PF_ALL_5D = [to_code5(c) for c in PORTFOLIO_ALL_4D]

# ── DB接続・データ取得 ────────────────────────────────────────────────
print("="*72)
print("Closing Auction Rebound — auKabu ポートフォリオ検証")
print("="*72)

conn = psycopg2.connect(host="localhost", dbname="market_data", user="postgres")
cur = conn.cursor()

# 対象期間: 2024-11-05〜 (新制度)
START = "2024-11-05"
END   = "2026-06-05"

# 全対象銘柄の 15:24 / 15:30 足を取得
# stocks_intraday はパーティション (stocks_intraday_YYYYMM) に分かれているが
# 通常クエリで跨いで取れる
print(f"\n[1] DB から 15:24/15:30 データ取得中 ({START}〜{END}) ...")

codes_ph = ",".join([f"'{c}'" for c in PF_ALL_5D])
sql = f"""
    SELECT
        code,
        DATE(ts) AS date,
        MAX(CASE WHEN ts::time = '15:24:00' THEN close END) AS close_1524,
        MAX(CASE WHEN ts::time = '15:30:00' THEN close END) AS close_1530,
        MAX(CASE WHEN ts::time = '09:00:00' THEN open  END) AS open_0900
    FROM stocks_intraday
    WHERE code IN ({codes_ph})
      AND ts >= '{START}'
      AND ts <= '{END} 23:59:59'
      AND ts::time IN ('15:24:00','15:30:00','09:00:00')
    GROUP BY code, DATE(ts)
    ORDER BY code, date
"""
cur.execute(sql)
rows = cur.fetchall()
conn.close()

df = pd.DataFrame(rows, columns=["code","date","close_1524","close_1530","open_0900"])
df["date"] = pd.to_datetime(df["date"])
print(f"  取得: {len(df):,} 行 ({df['code'].nunique()} 銘柄, {df['date'].nunique()} 日)")

# ── close_jump 計算 ────────────────────────────────────────────────────
print("\n[2] close_jump / overnight 計算 ...")

df = df.sort_values(["code","date"]).reset_index(drop=True)
# Decimal型をfloatに変換
for col in ["close_1524","close_1530","open_0900"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["close_jump"] = df["close_1530"] / df["close_1524"] - 1

# 翌営業日の09:00 open を結合
df["next_open"] = df.groupby("code")["open_0900"].shift(-1)
df["overnight"] = df["next_open"] / df["close_1530"] - 1

# クレンジング
n0 = len(df)
df = df[df["close_1524"].notna() & df["close_1530"].notna() & df["next_open"].notna()]
df = df[df["overnight"].abs() <= 0.10]   # ±10%以内 (分割等の異常値除去)
df = df[df["close_jump"].abs() <= 0.05]  # ±5%以内
print(f"  クレンジング: {n0:,} → {len(df):,}")

df["jump_bps"] = df["close_jump"] * 1e4
df["on_bps"]   = df["overnight"] * 1e4

# ── IS / OOS 分割 ─────────────────────────────────────────────────────
OOS_START = pd.Timestamp("2025-08-05")
df["period"] = np.where(df["date"] >= OOS_START, "OOS", "IS")

# ── 分析関数 ──────────────────────────────────────────────────────────
def sharpe(x, ann=np.sqrt(252)):
    x = pd.Series(x).dropna()
    if len(x) < 10 or x.std() == 0: return float("nan")
    return float(x.mean() / x.std() * ann)

def summary(ser, cost_bps=0, label=""):
    ret = ser - cost_bps
    n   = len(ret)
    mn  = ret.mean()
    sh  = sharpe(ret)
    wr  = (ret > 0).mean() * 100
    t   = mn / (ret.std() / np.sqrt(n)) if ret.std() > 0 else float("nan")
    tag = f"  [{label}]" if label else " "
    print(f"{tag:30s} n={n:5d}  mean={mn:+6.1f}bps  Sharpe={sh:+5.2f}  勝率={wr:4.1f}%  t={t:+5.2f}")
    return sh

# ── A. 閾値別 Long 結果 ────────────────────────────────────────────────
print("\n" + "="*72)
print("A. close_jump 閾値別 Long-only リターン (全銘柄プール, コスト0)")
print("="*72)
print(f"  {'閾値':>10}  {'n':>6}  {'mean bps':>9}  {'Sharpe':>7}  {'勝率%':>6}  {'t':>6}")
print("  " + "-"*56)
for thr in [-25, -50, -75, -100, -150]:
    sig = df[df["jump_bps"] <= thr]["on_bps"]
    if len(sig) < 10: continue
    sh = sharpe(sig)
    wr = (sig > 0).mean() * 100
    t  = sig.mean() / (sig.std() / np.sqrt(len(sig)))
    print(f"  jump≤{thr:4d}bps  {len(sig):6d}  {sig.mean():+9.1f}  {sh:+7.2f}  {wr:6.1f}  {t:+6.2f}")

# ── B. ポートフォリオ別 × コスト感度 ─────────────────────────────────
print("\n" + "="*72)
print("B. ポートフォリオ別 × コスト感度 (閾値 -50bps)")
print("="*72)
THR = -50

for pf_name, pf_codes in [("PORTFOLIO_15 (非鉄+半導体22)", PF15_5D),
                            ("PORTFOLIO_ALL (全銘柄)", PF_ALL_5D)]:
    print(f"\n  ── {pf_name} ──")
    sub = df[df["code"].isin(pf_codes) & (df["jump_bps"] <= THR)]["on_bps"]
    if len(sub) < 5:
        print("    サンプル不足")
        continue
    for cost in [0, 6, 10, 20]:
        ret = sub - cost
        n = len(ret); sh = sharpe(ret); wr = (ret>0).mean()*100
        t = ret.mean()/(ret.std()/np.sqrt(n))
        print(f"    cost={cost:2d}bps  n={n:4d}  mean={ret.mean():+6.1f}bps  "
              f"Sharpe={sh:+5.2f}  勝率={wr:4.1f}%  t={t:+5.2f}")

# ── C. IS / OOS 比較 ─────────────────────────────────────────────────
print("\n" + "="*72)
print("C. IS / OOS 比較 (PORTFOLIO_ALL, jump≤-50bps, cost=10bps)")
print("="*72)
pf_sig = df[df["code"].isin(PF_ALL_5D) & (df["jump_bps"] <= THR)].copy()
for period in ["IS","OOS","全期間"]:
    if period == "全期間":
        sub = pf_sig["on_bps"]
    else:
        sub = pf_sig[pf_sig["period"]==period]["on_bps"]
    if len(sub) < 5: continue
    ret = sub - 10
    sh = sharpe(ret); wr = (ret>0).mean()*100
    t = ret.mean()/(ret.std()/np.sqrt(len(ret)))
    print(f"  {period:6s}  n={len(ret):4d}  mean={ret.mean():+6.1f}bps  "
          f"Sharpe={sh:+5.2f}  勝率={wr:4.1f}%  t={t:+5.2f}")

# ── D. 銘柄別 成績上位 ────────────────────────────────────────────────
print("\n" + "="*72)
print("D. 銘柄別 成績 (jump≤-50bps, cost=10bps, n≥5)")
print("="*72)
stock_res = []
for code, grp in pf_sig.groupby("code"):
    ret = grp["on_bps"] - 10
    if len(ret) < 5: continue
    sh = sharpe(ret)
    stock_res.append({
        "code": code,
        "n": len(ret),
        "mean": ret.mean(),
        "sharpe": sh,
        "winrate": (ret > 0).mean() * 100,
    })
sr = pd.DataFrame(stock_res).sort_values("sharpe", ascending=False)
print(f"\n  {'code':>6}  {'n':>4}  {'mean bps':>9}  {'Sharpe':>7}  {'勝率%':>6}")
print("  " + "-"*42)
for _, r in sr.head(15).iterrows():
    print(f"  {r['code']:>6}  {int(r['n']):>4}  {r['mean']:+9.1f}  {r['sharpe']:+7.2f}  {r['winrate']:6.1f}%")

# ── E. 発火頻度 (1日あたり何銘柄) ───────────────────────────────────
print("\n" + "="*72)
print("E. 発火頻度 (PORTFOLIO_ALL, jump≤-50bps)")
print("="*72)
freq = pf_sig.groupby("date").size()
print(f"  発火日数: {len(freq)} 日 / 全 {df['date'].nunique()} 日")
print(f"  1日あたり平均: {freq.mean():.2f} 銘柄")
print(f"  分布: 0銘柄={df['date'].nunique()-len(freq)}日  "
      f"1銘柄={( freq==1).sum()}日  2銘柄={(freq==2).sum()}日  "
      f"3+銘柄={(freq>=3).sum()}日")

print("\n[DONE]")
