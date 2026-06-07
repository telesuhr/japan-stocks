"""
Closing Auction Rebound — auKabu50銘柄 正しいウォークフォワード

ISデータのみで銘柄選定 → OOSで初評価
"""
from __future__ import annotations
import sys
import pandas as pd
import numpy as np
import psycopg2

sys.stdout.reconfigure(line_buffering=True)

IS_START  = "2024-11-05"
IS_END    = "2025-08-04"
OOS_START = pd.Timestamp("2025-08-05")
OOS_END   = "2026-06-05"
COST      = 10  # bps

# auKabu PORTFOLIO_ALL (5桁コード)
PF_ALL = {
    # 非鉄8
    "57130","57110","57060","57140","50160","58010","58020","58030",
    # 半導体15
    "80350","68570","69200","61460","77350","40630","34360","77410",
    "69630","65260","99840","40620","67230","285A0","65250",
    # 銀行3
    "83060","83160","84110",
    # 機械/防衛4
    "70110","70130","70120","65030",
    # 総合電機2
    "65010","67580",
    # 自動車2
    "72030","72670",
    # 商社2
    "80580","80310",
    # 電子部品4
    "69810","67620","69710","69760",
    # 素材1
    "40040",
    # 保険1
    "87660",
    # エネルギー1
    "16050",
    # その他5
    "68610","69540","94320","79740","99830",
}

print("="*72)
print("Closing Auction Rebound — auKabu50銘柄 ウォークフォワード")
print(f"  IS : {IS_START} 〜 {IS_END}")
print(f"  OOS: {OOS_START.date()} 〜 {OOS_END}")
print(f"  対象: {len(PF_ALL)} 銘柄")
print("="*72)

# ── データ取得 ────────────────────────────────────────────────────────
conn = psycopg2.connect(host="localhost", dbname="market_data", user="postgres")
cur  = conn.cursor()

print("\n[1] 分足データ取得 ...")
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
    GROUP BY code, DATE(ts)
    ORDER BY code, date
""")
rows = cur.fetchall()

# 銘柄名取得
cur.execute(f"SELECT code5, name_ja, sector17_nm FROM symbol_master WHERE code5 IN ({codes_ph})")
names = {r[0]: (r[1], r[2]) for r in cur.fetchall()}
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
df["period"]   = np.where(df["date"] < OOS_START, "IS", "OOS")
print(f"  {n0:,} → {len(df):,}行  IS:{(df['period']=='IS').sum():,}  OOS:{(df['period']=='OOS').sum():,}")

def sharpe(x, ann=np.sqrt(252)):
    x = pd.Series(x).dropna()
    if len(x) < 5 or x.std() == 0: return float("nan")
    return float(x.mean() / x.std() * ann)

def tstat(x):
    x = pd.Series(x).dropna()
    if len(x) < 5 or x.std() == 0: return float("nan")
    return float(x.mean() / (x.std() / np.sqrt(len(x))))

# ── A. 全体サマリー（参考） ───────────────────────────────────────────
print("\n" + "="*72)
print("A. 全体サマリー (閾値別, コスト10bps)")
print("="*72)
print(f"  {'閾値':>10}  {'n':>5}  {'mean':>8}  {'Sharpe':>7}  {'IS Sh':>7}  {'OOS Sh':>7}  {'t':>6}")
print("  " + "-"*62)
for thr in [-25, -50, -75, -100]:
    sig = df[df["jump_bps"] <= thr]
    if len(sig) < 10: continue
    is_  = (sig[sig["period"]=="IS"]["on_bps"] - COST)
    oos_ = (sig[sig["period"]=="OOS"]["on_bps"] - COST)
    all_ = sig["on_bps"] - COST
    print(f"  jump≤{thr:4d}bps  {len(sig):5d}  {all_.mean():+8.1f}  "
          f"{sharpe(all_):+7.2f}  {sharpe(is_):+7.2f}  {sharpe(oos_):+7.2f}  {tstat(all_):+6.2f}")

# ── B. IS選定 → OOS評価（ウォークフォワード本体） ────────────────────
print("\n" + "="*72)
print("B. ウォークフォワード: ISで選定 → OOSで初評価 (コスト10bps)")
print("="*72)

for thr in [-50, -75, -100]:
    # IS期間で銘柄スコアリング
    is_sig = df[(df["period"]=="IS") & (df["jump_bps"] <= thr)]
    stock_is = []
    for code, grp in is_sig.groupby("code"):
        ret = grp["on_bps"] - COST
        if len(ret) < 5: continue
        stock_is.append({
            "code": code,
            "n_is": len(ret),
            "sh_is": sharpe(ret),
            "mean_is": ret.mean(),
            "t_is": tstat(ret),
        })
    if not stock_is:
        print(f"\n  ── jump≤{thr}bps  IS候補なし (n≥5を満たす銘柄なし) ──")
        continue
    cands = pd.DataFrame(stock_is).sort_values("sh_is", ascending=False)
    n_cand = len(cands)

    print(f"\n  ── jump≤{thr}bps  IS候補={n_cand}銘柄 ──")
    print(f"  {'N選定':>6}  {'OOS n':>6}  {'OOS mean':>9}  {'OOS Sh':>8}  {'OOS t':>7}  {'勝率%':>6}  {'1日avg':>7}")
    print("  " + "-"*60)

    for n_sel in [3, 5, 8, 10, 15, n_cand]:
        if n_sel > n_cand: continue
        sel = cands.head(n_sel)["code"].tolist()
        oos = df[(df["period"]=="OOS") & (df["jump_bps"] <= thr) & (df["code"].isin(sel))]
        if len(oos) < 5:
            print(f"  N={n_sel:2d}  OOSシグナルなし")
            continue
        # 日次等加重
        daily = oos.groupby("date").apply(
            lambda g: (g["on_bps"] - COST).mean(), include_groups=False
        )
        ret_i = oos["on_bps"] - COST
        print(f"  N={n_sel:2d}      {len(oos):>5}  {daily.mean():+9.1f}  "
              f"{sharpe(daily):+8.2f}  {tstat(daily):+7.2f}  "
              f"{(ret_i>0).mean()*100:6.1f}%  {oos.groupby('date').size().mean():7.1f}本")

# ── C. ベスト条件の詳細 (-50bps, 全銘柄) ─────────────────────────────
print("\n" + "="*72)
print("C. 詳細: jump≤-50bps, IS上位10銘柄 → OOS")
print("="*72)

THR = -50
is_sig50 = df[(df["period"]=="IS") & (df["jump_bps"] <= THR)]
stock_is50 = []
for code, grp in is_sig50.groupby("code"):
    ret = grp["on_bps"] - COST
    if len(ret) < 5: continue
    stock_is50.append({
        "code": code, "n_is": len(ret),
        "sh_is": sharpe(ret), "mean_is": ret.mean(), "t_is": tstat(ret),
    })
cands50 = pd.DataFrame(stock_is50).sort_values("sh_is", ascending=False)

print(f"\n  IS選定Top10 (jump≤-50bps):")
print(f"  {'code':>6}  {'銘柄名':<18}  {'セクター':<16}  {'n_IS':>5}  {'IS mean':>8}  {'IS Sh':>7}  {'IS t':>6}")
print("  " + "-"*78)
for _, r in cands50.head(10).iterrows():
    nm, sec = names.get(r["code"], ("?","?"))
    print(f"  {r['code']:>6}  {nm[:18]:<18}  {sec[:16]:<16}  {int(r['n_is']):>5}  "
          f"{r['mean_is']:+8.1f}  {r['sh_is']:+7.2f}  {r['t_is']:+6.2f}")

sel10_50 = cands50.head(10)["code"].tolist()
oos50 = df[(df["period"]=="OOS") & (df["jump_bps"] <= THR) & (df["code"].isin(sel10_50))]
daily50 = oos50.groupby("date").apply(lambda g: (g["on_bps"]-COST).mean(), include_groups=False)

print(f"\n  OOS結果 (初めて見るデータ, cost=10bps):")
print(f"  Sharpe={sharpe(daily50):+.2f}  mean={daily50.mean():+.1f}bps  t={tstat(daily50):+.2f}  "
      f"勝率={((oos50['on_bps']-COST)>0).mean()*100:.1f}%  "
      f"発火日数={len(daily50)}  1日avg={oos50.groupby('date').size().mean():.1f}本")

print(f"\n  コスト感度 (OOS):")
for cost in [0, 6, 10, 20]:
    d = oos50.groupby("date").apply(lambda g: (g["on_bps"]-cost).mean(), include_groups=False)
    print(f"    cost={cost:2d}bps  Sharpe={sharpe(d):+.2f}  mean={d.mean():+.1f}bps")

print(f"\n  OOS銘柄別成績:")
print(f"  {'code':>6}  {'銘柄名':<18}  {'n_OOS':>6}  {'mean':>8}  {'Sh':>7}  {'勝率%':>6}")
print("  " + "-"*62)
for code in sel10_50:
    grp = oos50[oos50["code"]==code]["on_bps"] - COST
    nm  = names.get(code, ("?","?"))[0]
    if len(grp) < 3:
        print(f"  {code:>6}  {nm[:18]:<18}  OOS発火なし")
        continue
    print(f"  {code:>6}  {nm[:18]:<18}  {len(grp):>6}  {grp.mean():+8.1f}  "
          f"{sharpe(grp):+7.2f}  {(grp>0).mean()*100:6.1f}%")

# ── D. 全銘柄 OOS成績一覧 ────────────────────────────────────────────
print("\n" + "="*72)
print("D. 全銘柄 OOS成績一覧 (jump≤-50bps, cost=10bps, n≥5)")
print("="*72)
oos_all = df[(df["period"]=="OOS") & (df["jump_bps"] <= -50)]
all_res = []
for code, grp in oos_all.groupby("code"):
    ret = grp["on_bps"] - COST
    if len(ret) < 5: continue
    nm, sec = names.get(code, ("?","?"))
    all_res.append({"code":code,"銘柄":nm[:14],"セクター":sec[:12],
                    "n":len(ret),"mean":ret.mean(),"sh":sharpe(ret),"wr":(ret>0).mean()*100})
ar = pd.DataFrame(all_res).sort_values("sh", ascending=False)
print(f"\n  {'code':>6}  {'銘柄':<14}  {'セクター':<12}  {'n':>4}  {'mean':>8}  {'Sh':>7}  {'勝率%':>6}")
print("  " + "-"*68)
for _, r in ar.iterrows():
    print(f"  {r['code']:>6}  {r['銘柄']:<14}  {r['セクター']:<12}  {int(r['n']):>4}  "
          f"{r['mean']:+8.1f}  {r['sh']:+7.2f}  {r['wr']:6.1f}%")

print("\n[DONE]")
