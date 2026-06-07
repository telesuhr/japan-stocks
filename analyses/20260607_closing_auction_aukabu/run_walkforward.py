"""
Closing Auction Rebound — 正しいウォークフォワード

手順:
  1. ISデータ (2024-11-05〜2025-08-04) のみで銘柄選定
     - IS ADVで流動性フィルター
     - IS Sharpe上位N銘柄を選定 (OOSデータは一切見ない)
  2. 選定した銘柄をOOS (2025-08-05〜2026-06-05) で初めて評価
  3. 選定銘柄数Nのロバスト性確認 (N=5/10/15/20)
  4. 閾値のロバスト性確認 (-50/-75/-100bps)
"""
from __future__ import annotations
import sys
import pandas as pd
import numpy as np
import psycopg2

sys.stdout.reconfigure(line_buffering=True)

IS_START  = "2024-11-05"
IS_END    = "2025-08-04"
OOS_START = "2025-08-05"
OOS_END   = "2026-06-05"
COST = 10  # bps

conn = psycopg2.connect(host="localhost", dbname="market_data", user="postgres")
cur  = conn.cursor()

print("="*72)
print("Closing Auction Rebound — 正しいウォークフォワード")
print(f"  IS : {IS_START} 〜 {IS_END}")
print(f"  OOS: {OOS_START} 〜 {OOS_END}")
print("="*72)

# ── Step1: IS ADV (IS期間のみで計算) ─────────────────────────────────
print("\n[1] IS ADV 取得 (IS期間のみ) ...")
cur.execute(f"""
    SELECT code, AVG(turnover_value) AS adv
    FROM stocks_daily
    WHERE date >= '{IS_START}' AND date <= '{IS_END}' AND turnover_value > 0
    GROUP BY code
""")
is_adv = {r[0]: float(r[1]) for r in cur.fetchall()}

# ── Step2: IS + OOS 両期間のデータ取得 ───────────────────────────────
print("[2] 分足データ取得 (IS+OOS) ...")
cur.execute(f"""
    SELECT code, DATE(ts) AS date,
        MAX(CASE WHEN ts::time='15:24:00' THEN close END) AS c1524,
        MAX(CASE WHEN ts::time='15:30:00' THEN close END) AS c1530,
        MAX(CASE WHEN ts::time='09:00:00' THEN open  END) AS o0900
    FROM stocks_intraday
    WHERE ts >= '{IS_START}' AND ts <= '{OOS_END} 23:59:59'
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

n0 = len(df)
df = df[df["c1524"].notna() & df["c1530"].notna() & df["next_open"].notna()]
df = df[df["overnight"].abs() <= 0.10]
df = df[df["close_jump"].abs() <= 0.05]
df["jump_bps"] = df["close_jump"] * 1e4
df["on_bps"]   = df["overnight"] * 1e4
df["is_adv"]   = df["code"].map(is_adv).fillna(0)
df["period"]   = np.where(df["date"] < pd.Timestamp(OOS_START), "IS", "OOS")

print(f"  取得: {n0:,} → クレンジング後 {len(df):,} ({df['code'].nunique():,}銘柄)")
print(f"  IS: {(df['period']=='IS').sum():,}行  OOS: {(df['period']=='OOS').sum():,}行")

def sharpe(x, ann=np.sqrt(252)):
    x = pd.Series(x).dropna()
    if len(x) < 5 or x.std() == 0: return float("nan")
    return float(x.mean() / x.std() * ann)

def tstat(x):
    x = pd.Series(x).dropna()
    if len(x) < 5 or x.std() == 0: return float("nan")
    return float(x.mean() / (x.std() / np.sqrt(len(x))))

# ── Step3: IS期間のみで銘柄スコアリング ──────────────────────────────
print("\n[3] IS期間のみで銘柄スコアリング ...")

def score_stocks(thr_bps, adv_min):
    """ISデータのみで閾値・流動性フィルターを適用し、銘柄をSharpe降順で返す"""
    is_sig = df[
        (df["period"] == "IS") &
        (df["is_adv"] >= adv_min) &
        (df["jump_bps"] <= thr_bps)
    ]
    res = []
    for code, grp in is_sig.groupby("code"):
        ret = grp["on_bps"] - COST
        if len(ret) < 8: continue   # IS期間でのトレード数最低8
        res.append({
            "code": code,
            "n_is": len(ret),
            "sh_is": sharpe(ret),
            "mean_is": ret.mean(),
            "t_is": tstat(ret),
        })
    return pd.DataFrame(res).sort_values("sh_is", ascending=False)

# ── Step4: OOS評価 ────────────────────────────────────────────────────
def eval_oos(selected_codes, thr_bps, adv_min):
    """選定銘柄をOOSで評価 (ADVフィルターはIS ADVで行う)"""
    oos_sig = df[
        (df["period"] == "OOS") &
        (df["is_adv"] >= adv_min) &
        (df["jump_bps"] <= thr_bps) &
        (df["code"].isin(selected_codes))
    ]
    if len(oos_sig) < 5:
        return dict(n=0, mean=float("nan"), sh=float("nan"), t=float("nan"),
                    wr=float("nan"), fire_days=0, daily_trades=float("nan"))
    # 日次等加重ポートフォリオ
    daily = oos_sig.groupby("date").apply(
        lambda g: (g["on_bps"] - COST).mean(), include_groups=False
    )
    ret = daily
    return dict(
        n=len(oos_sig),
        mean=ret.mean(),
        sh=sharpe(ret),
        t=tstat(ret),
        wr=(oos_sig["on_bps"] - COST > 0).mean() * 100,
        fire_days=len(ret),
        daily_trades=oos_sig.groupby("date").size().mean(),
    )

# ── メイン検証: 閾値×N銘柄 ───────────────────────────────────────────
print("\n" + "="*72)
print("4. 閾値 × 上位N銘柄 OOS Sharpe (IS Sharpe上位で選定, ADV≥10億)")
print("   ※ 銘柄選定はISデータのみ使用")
print("="*72)
ADV_MIN = 1e9

header = f"  {'閾値':>10}  {'N':>4}  {'IS選定候補':>8}  {'OOS Sh':>8}  {'OOS mean':>9}  {'OOS t':>7}  {'勝率%':>6}  {'1日avg':>7}"
print(header)
print("  " + "-"*75)

best_result = {}
for thr in [-50, -75, -100]:
    cands = score_stocks(thr, ADV_MIN)
    print(f"\n  ── jump≤{thr}bps  IS候補={len(cands)}銘柄 ──")
    for n_sel in [5, 10, 15, 20]:
        if len(cands) < n_sel:
            print(f"  jump≤{thr:4d}bps  N={n_sel:2d}  候補不足({len(cands)}銘柄)")
            continue
        sel = cands.head(n_sel)["code"].tolist()
        r = eval_oos(sel, thr, ADV_MIN)
        if r["n"] == 0:
            print(f"  jump≤{thr:4d}bps  N={n_sel:2d}  OOSシグナルなし")
            continue
        print(f"  jump≤{thr:4d}bps  N={n_sel:2d}  候補{len(cands):3d}銘柄  "
              f"OOS Sh={r['sh']:+6.2f}  mean={r['mean']:+7.1f}bps  "
              f"t={r['t']:+5.2f}  勝率={r['wr']:4.1f}%  1日{r['daily_trades']:.1f}本")
        best_result[(thr, n_sel)] = r

# ── 最良条件の詳細 ────────────────────────────────────────────────────
# -75bps N=10 を詳細表示
print("\n" + "="*72)
print("5. 詳細: jump≤-75bps, IS上位10銘柄の選定→OOS評価")
print("="*72)

cands75 = score_stocks(-75, ADV_MIN)
top10 = cands75.head(10)

print("\n  ── IS選定銘柄 (ISデータのみで評価) ──")
print(f"  {'code':>6}  {'n_IS':>5}  {'IS mean':>8}  {'IS Sh':>7}  {'IS t':>6}")
print("  " + "-"*45)
for _, r in top10.iterrows():
    print(f"  {r['code']:>6}  {int(r['n_is']):>5}  {r['mean_is']:+8.1f}bps  "
          f"{r['sh_is']:+7.2f}  {r['t_is']:+6.2f}")

sel10 = top10["code"].tolist()
oos10 = eval_oos(sel10, -75, ADV_MIN)

print(f"\n  ── OOS評価 (選定後、初めて見るデータ) ──")
print(f"  n={oos10['n']}  mean={oos10['mean']:+.1f}bps  Sharpe={oos10['sh']:+.2f}  "
      f"t={oos10['t']:+.2f}  勝率={oos10['wr']:.1f}%")
print(f"  発火日数={oos10['fire_days']}  1日平均={oos10['daily_trades']:.1f}トレード")

# コスト感度 (OOS)
print("\n  コスト感度 (OOS, 上位10銘柄):")
oos_sig10 = df[
    (df["period"] == "OOS") & (df["is_adv"] >= ADV_MIN) &
    (df["jump_bps"] <= -75) & (df["code"].isin(sel10))
]
for cost in [0, 6, 10, 20]:
    daily = oos_sig10.groupby("date").apply(
        lambda g: (g["on_bps"] - cost).mean(), include_groups=False
    )
    sh_c = sharpe(daily)
    print(f"    cost={cost:2d}bps  Sharpe={sh_c:+.2f}  mean={daily.mean():+.1f}bps")

# ── IS選定銘柄のOOS個別成績 ──────────────────────────────────────────
print("\n  ── OOS個別成績 (選定10銘柄, cost=10bps) ──")
print(f"  {'code':>6}  {'n_OOS':>6}  {'OOS mean':>9}  {'OOS Sh':>8}  {'勝率%':>6}")
print("  " + "-"*48)
for code in sel10:
    grp = oos_sig10[oos_sig10["code"] == code]["on_bps"] - COST
    if len(grp) < 3:
        print(f"  {code:>6}  OOSシグナル少 (n={len(grp)})")
        continue
    print(f"  {code:>6}  {len(grp):>6}  {grp.mean():+9.1f}bps  "
          f"{sharpe(grp):+8.2f}  {(grp>0).mean()*100:6.1f}%")

# ── 銘柄名を symbol_master から取得 ──────────────────────────────────
print("\n  ── 選定銘柄名 ──")
conn2 = psycopg2.connect(host="localhost", dbname="market_data", user="postgres")
cur2  = conn2.cursor()
codes_ph = ",".join([f"'{c}'" for c in sel10])
cur2.execute(f"""
    SELECT code5, name FROM symbol_master
    WHERE code5 IN ({codes_ph})
""")
names = {r[0]: r[1] for r in cur2.fetchall()}
conn2.close()
for code in sel10:
    print(f"  {code}  {names.get(code, '?')}")

print("\n[DONE]")
