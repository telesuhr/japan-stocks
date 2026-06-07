"""
ベータ確認: 「相場が上がっただけ」でないかを検証

1. 無条件オーバーナイト（シグナルなし）と比較
2. jump方向別リターン（上昇側・中立・下落側）
3. TOPIX日次リターンとの相関
4. 相場上昇期 vs 下落期 でのサブグループ検証
"""
from __future__ import annotations
import sys
import pandas as pd
import numpy as np
import psycopg2

sys.stdout.reconfigure(line_buffering=True)

IS_START = "2024-11-05"
OOS_END  = "2026-06-05"
OOS_START = pd.Timestamp("2025-08-05")
COST = 10

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

print("="*72)
print("ベータ確認: 「相場上昇バイアス」の有無を検証")
print("="*72)

# ── データ取得（前回と同じ） ──────────────────────────────────────────
codes_ph = ",".join([f"'{c}'" for c in PF_ALL])
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

# 市場リターン: auKabu銘柄の等加重日次リターン
cur.execute(f"""
    WITH daily_ret AS (
        SELECT code, date,
               close / LAG(close) OVER (PARTITION BY code ORDER BY date) - 1 AS ret
        FROM stocks_daily
        WHERE code IN ({codes_ph})
          AND date >= '{IS_START}' AND date <= '{OOS_END}'
    )
    SELECT date, AVG(ret) AS mkt_ret
    FROM daily_ret
    WHERE ret IS NOT NULL
    GROUP BY date ORDER BY date
""")
mkt_rows = cur.fetchall()
conn.close()

mkt = pd.DataFrame(mkt_rows, columns=["date","mkt_ret"])
mkt["date"] = pd.to_datetime(mkt["date"])
mkt = mkt.dropna()

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

def sharpe(x, ann=np.sqrt(252)):
    x = pd.Series(x).dropna()
    if len(x) < 5 or x.std() == 0: return float("nan")
    return float(x.mean() / x.std() * ann)

# ── 1. 無条件オーバーナイト vs シグナル付き ─────────────────────────
print("\n" + "="*72)
print("1. 無条件オーバーナイト vs シグナル付き (全期間, cost=10bps)")
print("="*72)
unconditional = df["on_bps"]
signal_50 = df[df["jump_bps"] <= -50]["on_bps"]
signal_75 = df[df["jump_bps"] <= -75]["on_bps"]

print(f"\n  {'ケース':<30}  {'n':>6}  {'mean bps':>9}  {'Sharpe':>7}")
print("  " + "-"*58)
for label, ser in [
    ("無条件 (毎日全銘柄買い)", unconditional),
    ("jump≤-50bps (コスト前)", signal_50),
    ("jump≤-50bps (コスト10bps後)", signal_50 - COST),
    ("jump≤-75bps (コスト前)", signal_75),
    ("jump≤-75bps (コスト10bps後)", signal_75 - COST),
]:
    print(f"  {label:<30}  {len(ser):>6}  {ser.mean():>+9.1f}  {sharpe(ser):>+7.2f}")

excess_50 = (signal_50 - COST).mean() - unconditional.mean()
excess_75 = (signal_75 - COST).mean() - unconditional.mean()
print(f"\n  無条件比の超過リターン:")
print(f"    jump≤-50bps (cost後): {excess_50:+.1f}bps")
print(f"    jump≤-75bps (cost後): {excess_75:+.1f}bps")

# ── 2. jump方向別リターン ─────────────────────────────────────────────
print("\n" + "="*72)
print("2. close_jump方向別 翌朝リターン (シグナルの方向依存性確認)")
print("="*72)
print(f"\n  {'jumpバケット':<22}  {'n':>6}  {'mean bps':>9}  {'Sharpe':>7}")
print("  " + "-"*50)
buckets = [
    ("大幅下落 jump≤-75bps",   df["jump_bps"] <= -75),
    ("下落    -75<jump≤-50bps", (df["jump_bps"] > -75) & (df["jump_bps"] <= -50)),
    ("小幅下落 -50<jump≤-25bps",(df["jump_bps"] > -50) & (df["jump_bps"] <= -25)),
    ("中立    -25<jump≤+25bps", (df["jump_bps"] > -25) & (df["jump_bps"] <= 25)),
    ("小幅上昇 +25<jump≤+50bps",(df["jump_bps"] > 25)  & (df["jump_bps"] <= 50)),
    ("上昇    jump>+50bps",      df["jump_bps"] > 50),
]
for label, mask in buckets:
    ser = df[mask]["on_bps"]
    if len(ser) < 5: continue
    print(f"  {label:<22}  {len(ser):>6}  {ser.mean():>+9.1f}  {sharpe(ser):>+7.2f}")

print("\n  ※ 下落側が高く上昇側が低ければ、単なる市場β上昇ではなく")
print("     「引け大幅売り→翌朝反発」という構造的パターンが存在する")

# ── 3. 相場局面別サブグループ ─────────────────────────────────────────
print("\n" + "="*72)
print("3. 相場局面別検証 (市場の方向性とシグナルの独立性)")
print("="*72)

# 市場リターンをdfに結合
df2 = df.merge(mkt.rename(columns={"mkt_ret":"mkt"}), on="date", how="left")

# 翌日の市場リターンで「上昇日」「下落日」を分類
# (翌日市場が上がる日に多く発火していれば、市場β依存)
df2["next_mkt"] = df2.groupby("date")["mkt"].transform("mean")  # 当日市場リターン

# 当日市場が下落した日 vs 上昇した日にシグナルが発火するか
print("\n  シグナル発火日の市場環境:")
sig = df2[df2["jump_bps"] <= -50]
up_days   = (sig["next_mkt"] > 0).mean() * 100
down_days = (sig["next_mkt"] <= 0).mean() * 100
print(f"    発火日のうち当日市場上昇: {up_days:.1f}%  下落: {down_days:.1f}%")

all_up   = (df2["next_mkt"] > 0).mean() * 100
print(f"    全日のうち市場上昇: {all_up:.1f}%  (発火日と分布が近ければβ依存でない)")

# 当日市場方向別にシグナルリターンを分解
print(f"\n  当日市場方向 × シグナルリターン (jump≤-50bps, cost=10bps):")
print(f"  {'当日市場':<16}  {'n':>5}  {'mean bps':>9}  {'Sharpe':>7}")
print("  " + "-"*44)
for label, msk in [("市場上昇日", sig["next_mkt"] > 0),
                    ("市場下落日", sig["next_mkt"] <= 0)]:
    sub = sig[msk]["on_bps"] - COST
    if len(sub) < 5: continue
    print(f"  {label:<16}  {len(sub):>5}  {sub.mean():>+9.1f}  {sharpe(sub):>+7.2f}")

# ── 4. OOS期間の相場環境確認 ─────────────────────────────────────────
print("\n" + "="*72)
print("4. 期間別の市場リターン確認 (OOSが強いのは単純に相場が上がったから?)")
print("="*72)

# 各期間の市場平均リターン
for period, start, end in [
    ("IS  (2024-11〜2025-08)", IS_START, "2025-08-04"),
    ("OOS (2025-08〜2026-06)", "2025-08-05", OOS_END),
]:
    sub = mkt[(mkt["date"] >= start) & (mkt["date"] <= end)]["mkt_ret"].dropna()
    cum = (1 + sub).prod() - 1
    avg = sub.mean() * 1e4
    print(f"  {period}:  日次平均 {avg:+.1f}bps  累積 {cum*100:+.1f}%  ({len(sub)}日)")

# 期間別の無条件オーバーナイトも確認
print()
for period, mask in [("IS  無条件ON", df["period"]=="IS"),
                      ("OOS 無条件ON", df["period"]=="OOS")]:
    sub = df[mask]["on_bps"]
    print(f"  {period}:  mean {sub.mean():+.1f}bps  Sharpe {sharpe(sub):+.2f}")

print()
for period, mask in [("IS  jump≤-50bps (cost後)", df["period"]=="IS"),
                      ("OOS jump≤-50bps (cost後)", df["period"]=="OOS")]:
    sub = df[mask & (df["jump_bps"] <= -50)]["on_bps"] - COST
    if len(sub) < 5: continue
    print(f"  {period}:  mean {sub.mean():+.1f}bps  Sharpe {sharpe(sub):+.2f}  n={len(sub)}")

print("\n[DONE]")
