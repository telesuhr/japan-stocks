"""
Closing Auction Rebound — 追加バックテスト

前回の残課題:
  - -50bpsはコスト後で無条件ONに負けていた
  - 市場下落日に+19.3bps、上昇日に+1.2bps と局面依存が強い
  - OOS上位銘柄を固定した場合どうなるか

検証項目:
  A. 真の超過リターン（日次で無条件ON差引）の再評価
  B. 市場方向フィルター（15:24時点の市場状況を使う）
  C. OOS上位固定銘柄で長期ウォークフォワード
  D. 閾値を動的に設定（上位N%に絞る）
  E. 保有時間の感度（09:00/09:05/09:15/10:00決済）
"""
from __future__ import annotations
import sys
import pandas as pd
import numpy as np
import psycopg2

sys.stdout.reconfigure(line_buffering=True)

IS_START  = "2024-11-05"
OOS_START = pd.Timestamp("2025-08-05")
OOS_END   = "2026-06-05"
COST      = 10

PF_ALL = {
    "57130","57110","57060","57140","50160","58010","58020","58030",
    "80350","68570","69200","61460","77350","40630","34360","77410",
    "69630","65260","99840","40620","67230","285A0","65250",
    "83060","83160","84110","70110","70130","70120","65030",
    "65010","67580","72030","72670","80580","80310",
    "69810","67620","69710","69760","40040","87660","16050",
    "68610","69540","94320","79740","99830",
}

# OOS個別成績上位（前回D節から: OOS Sharpe≥4 かつ n≥8）
TOP_OOS = {"285A0","69710","65260","72030","80580","67580","58010","79740","50160","72670"}

conn = psycopg2.connect(host="localhost", dbname="market_data", user="postgres")
cur  = conn.cursor()

print("="*72)
print("Closing Auction Rebound — 追加バックテスト")
print("="*72)

# ── データ取得（15:24 / 15:30 / 翌09:00 + 複数exit時刻）────────────
codes_ph = ",".join([f"'{c}'" for c in PF_ALL])
print("\n[データ取得中...]")
cur.execute(f"""
    SELECT code, DATE(ts) AS date,
        MAX(CASE WHEN ts::time='15:24:00' THEN close END) AS c1524,
        MAX(CASE WHEN ts::time='15:30:00' THEN close END) AS c1530,
        MAX(CASE WHEN ts::time='09:00:00' THEN open  END) AS o0900,
        MAX(CASE WHEN ts::time='09:05:00' THEN open  END) AS o0905,
        MAX(CASE WHEN ts::time='09:15:00' THEN open  END) AS o0915,
        MAX(CASE WHEN ts::time='10:00:00' THEN open  END) AS o1000
    FROM stocks_intraday
    WHERE code IN ({codes_ph})
      AND ts >= '{IS_START}' AND ts <= '{OOS_END} 23:59:59'
      AND ts::time IN ('15:24:00','15:30:00','09:00:00','09:05:00','09:15:00','10:00:00')
    GROUP BY code, DATE(ts)
    ORDER BY code, date
""")
rows = cur.fetchall()
conn.close()

df = pd.DataFrame(rows, columns=["code","date","c1524","c1530","o0900","o0905","o0915","o1000"])
df["date"] = pd.to_datetime(df["date"])
for col in ["c1524","c1530","o0900","o0905","o0915","o1000"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.sort_values(["code","date"]).reset_index(drop=True)
df["close_jump"] = df["c1530"] / df["c1524"] - 1

# 各exit時刻のオーバーナイトリターン
for col, exit_col in [("o0900","on_0900"),("o0905","on_0905"),("o0915","on_0915"),("o1000","on_1000")]:
    nxt = df.groupby("code")[col].shift(-1)
    df[exit_col] = (nxt / df["c1530"] - 1) * 1e4

df["on_bps"] = df["on_0900"]
df = df[df["c1524"].notna() & df["c1530"].notna() & df["on_bps"].notna()]
df = df[df["on_bps"].abs() <= 1000]  # ±10%
df = df[df["close_jump"].abs() <= 0.05]
df["jump_bps"] = df["close_jump"] * 1e4
df["period"]   = np.where(df["date"] < OOS_START, "IS", "OOS")
df["in_top"]   = df["code"].isin(TOP_OOS)
print(f"  {len(df):,}行 ({df['code'].nunique()}銘柄, {df['date'].nunique()}日)")

# 日次市場リターン（全銘柄等加重）
daily_mkt = df.groupby("date")["close_jump"].mean().rename("mkt_jump")
df = df.join(daily_mkt, on="date")
# 市場が下落した日（15:24時点までの動きを15:30でも代替）
df["mkt_down"] = df["mkt_jump"] < 0

def sharpe(x, ann=np.sqrt(252)):
    x = pd.Series(x).dropna()
    if len(x) < 5 or x.std() == 0: return float("nan")
    return float(x.mean() / x.std() * ann)

def tstat(x):
    x = pd.Series(x).dropna()
    if len(x) < 5 or x.std() == 0: return float("nan")
    return float(x.mean() / (x.std() / np.sqrt(len(x))))

def pf_daily_sharpe(sig_df, cost=COST):
    """日次等加重ポートフォリオのSharpe"""
    daily = sig_df.groupby("date").apply(
        lambda g: (g["on_bps"] - cost).mean(), include_groups=False
    )
    return sharpe(daily), tstat(daily), len(daily), daily.mean()

# 無条件ON（ベースライン）
uncond_daily = df.groupby("date")["on_bps"].mean()
uncond_mean = uncond_daily.mean()
uncond_sh   = sharpe(uncond_daily)

# ══════════════════════════════════════════════════════════════════════
# A. 真の超過リターン（日次で無条件ON差引）
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*72)
print("A. 真の超過リターン (日次シグナルret - その日の無条件ON平均)")
print(f"   無条件ON: mean={uncond_mean:+.1f}bps  Sharpe={uncond_sh:+.2f}")
print("="*72)
print(f"\n  {'条件':<32}  {'n日':>5}  {'excess mean':>12}  {'excess Sh':>10}  {'raw mean':>9}")
print("  " + "-"*68)

for thr, label in [(-25,"jump≤-25bps"), (-50,"jump≤-50bps"), (-75,"jump≤-75bps"), (-100,"jump≤-100bps")]:
    sig = df[df["jump_bps"] <= thr]
    sig_daily = sig.groupby("date").apply(
        lambda g: (g["on_bps"] - COST).mean(), include_groups=False
    )
    # その日の無条件ONを引く
    excess = sig_daily - uncond_daily.reindex(sig_daily.index)
    sh_ex = sharpe(excess)
    t_ex  = tstat(excess)
    print(f"  {label:<32}  {len(excess):>5}  {excess.mean():>+12.1f}bps  "
          f"{sh_ex:>+10.2f} (t={t_ex:+.2f})  {sig_daily.mean():>+9.1f}bps")

# IS/OOS別
print(f"\n  IS/OOS別 (jump≤-75bps):")
for period, mask in [("IS", df["period"]=="IS"), ("OOS", df["period"]=="OOS")]:
    sig = df[mask & (df["jump_bps"] <= -75)]
    sig_daily = sig.groupby("date").apply(
        lambda g: (g["on_bps"] - COST).mean(), include_groups=False
    )
    unc_p = uncond_daily[uncond_daily.index.isin(sig_daily.index)]
    excess = sig_daily - unc_p
    print(f"    {period}  n={len(excess)}  excess={excess.mean():+.1f}bps  Sh={sharpe(excess):+.2f}  t={tstat(excess):+.2f}")

# ══════════════════════════════════════════════════════════════════════
# B. 市場方向フィルター（15:30時点の市場jumpで判断）
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*72)
print("B. 市場方向 × シグナル (jump≤-50bps, cost=10bps)")
print("   市場方向 = 当日全銘柄の close_jump の平均符号")
print("="*72)
print(f"\n  {'条件':<36}  {'n':>5}  {'mean':>8}  {'Sharpe':>7}  {'t':>6}")
print("  " + "-"*62)

for mkt_cond, mkt_label in [(True,"市場下落日"), (False,"市場上昇日"), (None,"全日")]:
    for thr, thr_label in [(-50,"jump≤-50"), (-75,"jump≤-75")]:
        mask = df["jump_bps"] <= thr
        if mkt_cond is not None:
            mask = mask & (df["mkt_down"] == mkt_cond)
        sig = df[mask]["on_bps"] - COST
        if len(sig) < 5: continue
        label = f"{mkt_label} + {thr_label}bps"
        print(f"  {label:<36}  {len(sig):>5}  {sig.mean():>+8.1f}  {sharpe(sig):>+7.2f}  {tstat(sig):>+6.2f}")

# ══════════════════════════════════════════════════════════════════════
# C. OOS上位固定銘柄（前回の上位10銘柄を固定）
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*72)
print(f"C. OOS上位固定10銘柄 (jump≤-50bps, cost=10bps)")
print(f"   銘柄: {', '.join(sorted(TOP_OOS))}")
print("="*72)
print("   ※ これはOOSデータで選んでいるため真のWFでない — 上限の参考値")

for period, mask in [("IS", df["period"]=="IS"), ("OOS", df["period"]=="OOS"), ("全期間", pd.Series(True, index=df.index))]:
    sig = df[mask & df["in_top"] & (df["jump_bps"] <= -50)]
    if len(sig) < 5: continue
    sh, t, n_d, mn = pf_daily_sharpe(sig)
    print(f"  {period:<8}  n={len(sig):3d}  発火日={n_d:3d}  mean={mn:+.1f}bps  Sharpe={sh:+.2f}  t={t:+.2f}")

# ══════════════════════════════════════════════════════════════════════
# D. 上位N%フィルター（閾値固定でなく相対的な大きさで選ぶ）
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*72)
print("D. 当日下落ワースト上位N% フィルター (相対閾値, cost=10bps)")
print("   毎日、全銘柄中jumpが最も低い上位N%の銘柄のみエントリー")
print("="*72)
print(f"\n  {'上位N%':>8}  {'平均n/日':>8}  {'mean':>8}  {'Sharpe':>7}  {'IS Sh':>7}  {'OOS Sh':>7}")
print("  " + "-"*56)

# 各日のjumpパーセンタイル閾値を計算
df["jump_rank_pct"] = df.groupby("date")["jump_bps"].rank(pct=True)

for pct in [0.05, 0.10, 0.15, 0.20]:
    sig = df[df["jump_rank_pct"] <= pct]
    is_  = sig[sig["period"]=="IS"]["on_bps"] - COST
    oos_ = sig[sig["period"]=="OOS"]["on_bps"] - COST
    all_ = sig["on_bps"] - COST
    avg_n = sig.groupby("date").size().mean()
    print(f"  上位{int(pct*100):2d}%      {avg_n:>8.1f}  {all_.mean():>+8.1f}  "
          f"{sharpe(all_):>+7.2f}  {sharpe(is_):>+7.2f}  {sharpe(oos_):>+7.2f}")

# 相対+絶対の組み合わせ
print(f"\n  相対(上位10%) + 絶対(-50bps以上) の組み合わせ:")
sig_combo = df[(df["jump_rank_pct"] <= 0.10) & (df["jump_bps"] <= -50)]
is_c  = sig_combo[sig_combo["period"]=="IS"]["on_bps"] - COST
oos_c = sig_combo[sig_combo["period"]=="OOS"]["on_bps"] - COST
all_c = sig_combo["on_bps"] - COST
print(f"  n={len(sig_combo)}  mean={all_c.mean():+.1f}bps  Sh={sharpe(all_c):+.2f}  IS={sharpe(is_c):+.2f}  OOS={sharpe(oos_c):+.2f}")

# ══════════════════════════════════════════════════════════════════════
# E. 保有時間の感度（exit時刻別）
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*72)
print("E. 保有時間の感度 (jump≤-75bps, cost=10bps)")
print("="*72)
print(f"\n  {'exit時刻':>10}  {'n':>5}  {'mean':>8}  {'Sharpe':>7}  {'IS Sh':>7}  {'OOS Sh':>7}")
print("  " + "-"*54)

sig75 = df[df["jump_bps"] <= -75].copy()
for exit_col, label in [("on_0900","09:00寄"), ("on_0905","09:05"), ("on_0915","09:15"), ("on_1000","10:00")]:
    valid = sig75[sig75[exit_col].notna()]
    all_ = valid[exit_col] - COST
    is_  = valid[valid["period"]=="IS"][exit_col] - COST
    oos_ = valid[valid["period"]=="OOS"][exit_col] - COST
    print(f"  {label:>10}  {len(valid):>5}  {all_.mean():>+8.1f}  "
          f"{sharpe(all_):>+7.2f}  {sharpe(is_):>+7.2f}  {sharpe(oos_):>+7.2f}")

# ══════════════════════════════════════════════════════════════════════
# F. 総合まとめ：最良の組み合わせを探す
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*72)
print("F. 組み合わせ探索 (IS→OOS両方Sharpe>0が条件, cost=10bps)")
print("="*72)
print(f"\n  {'条件':<45}  {'n':>4}  {'IS Sh':>7}  {'OOS Sh':>7}  {'全 Sh':>7}  {'全 t':>6}")
print("  " + "-"*72)

combos = [
    ("jump≤-75 全銘柄",
        df["jump_bps"] <= -75),
    ("jump≤-75 × 市場下落日",
        (df["jump_bps"] <= -75) & df["mkt_down"]),
    ("jump≤-75 × 市場上昇日",
        (df["jump_bps"] <= -75) & ~df["mkt_down"]),
    ("jump≤-50 × 市場下落日",
        (df["jump_bps"] <= -50) & df["mkt_down"]),
    ("jump≤-50 × 相対上位10%",
        (df["jump_bps"] <= -50) & (df["jump_rank_pct"] <= 0.10)),
    ("jump≤-75 × 上位OOS銘柄",
        (df["jump_bps"] <= -75) & df["in_top"]),
    ("jump≤-50 × 上位OOS銘柄 × 市場下落",
        (df["jump_bps"] <= -50) & df["in_top"] & df["mkt_down"]),
]

for label, mask in combos:
    sig = df[mask]
    is_  = sig[sig["period"]=="IS"]["on_bps"] - COST
    oos_ = sig[sig["period"]=="OOS"]["on_bps"] - COST
    all_ = sig["on_bps"] - COST
    if len(all_) < 10: continue
    sh_is  = sharpe(is_) if len(is_) >= 5 else float("nan")
    sh_oos = sharpe(oos_) if len(oos_) >= 5 else float("nan")
    both_pos = (sh_is > 0) and (sh_oos > 0) if not (np.isnan(sh_is) or np.isnan(sh_oos)) else False
    mark = "★" if both_pos else " "
    print(f"  {mark}{label:<44}  {len(all_):>4}  {sh_is:>+7.2f}  {sh_oos:>+7.2f}  "
          f"{sharpe(all_):>+7.2f}  {tstat(all_):>+6.2f}")

print("\n  ★ = IS・OOS両方プラス")
print("\n[DONE]")
