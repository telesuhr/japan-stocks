"""
ORB包括検証: 銘柄スクリーニング × パラメータグリッド × IS/OOS

流動性上位30銘柄 × 162パラメータ組合でIS/OOS検証
IS: 2024-06-01 〜 2025-06-30 (約13ヶ月)
OOS: 2025-07-01 〜 2026-07-03 (約12ヶ月)
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
import warnings; warnings.filterwarnings("ignore")

import os, psycopg2, pandas as pd, numpy as np
from datetime import time
from itertools import product
from pathlib import Path

HERE = Path(__file__).parent
PG = dict(host=os.environ.get("PGHOST","localhost"),
          port=int(os.environ.get("PGPORT",5432)),
          user=os.environ.get("PGUSER","postgres"),
          dbname=os.environ.get("PGDATABASE","market_data"))

IS_START  = "2024-06-01"
IS_END    = "2025-06-30"
OOS_START = "2025-07-01"
OOS_END   = "2026-07-03"
COST_BPS  = 4
N_STOCKS  = 30

# 162組合グリッド
GRID = {
    "range_end_m": [5, 10, 20],              # 3
    "entry_end_h": [10, 11],                 # 2
    "profit_pct":  [0.010, 0.015, 0.020, 0.030],  # 4
    "stop_pct":    [0.005, 0.010, 0.015],    # 3
    "direction":   ["both", "long", "short"],# 3
}  # 3×2×4×3×3 = 216組合

BASE = dict(range_end_m=10, entry_end_h=10,
            profit_pct=0.02, stop_pct=0.01, direction="both")


def get_conn():
    return psycopg2.connect(**PG)


def top_stocks():
    conn = get_conn(); cur = conn.cursor()
    cur.execute("""
        SELECT d.code, s.name_ja
        FROM stocks_daily d
        JOIN symbol_master s ON s.code5 = d.code
        WHERE d.date >= %s AND d.date <= %s AND s.market='0111'
        GROUP BY d.code, s.name_ja
        HAVING AVG(d.turnover_value) >= 1e9 AND COUNT(*) >= 200
        ORDER BY AVG(d.turnover_value) DESC
        LIMIT %s
    """, (IS_START, OOS_END, N_STOCKS))
    res = cur.fetchall(); conn.close()
    return [(r[0], r[1]) for r in res]


def load_stock(code, start, end):
    conn = get_conn(); cur = conn.cursor()
    cur.execute("""
        SELECT ts, open::float, high::float, low::float, close::float
        FROM stocks_intraday
        WHERE code=%s AND ts>=%s AND ts<%s
        ORDER BY ts
    """, (code, start, end+" 23:59:59"))
    rows = cur.fetchall(); conn.close()
    if not rows: return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["ts","open","high","low","close"])
    df["ts"] = pd.to_datetime(df["ts"])
    return df.set_index("ts")


def simulate(df, range_end_m, entry_end_h, profit_pct, stop_pct, direction):
    if df is None or df.empty: return []
    r_end = time(9, range_end_m); e_end = time(entry_end_h, 0); f_ext = time(15, 0)
    cost = COST_BPS / 10000; trades = []

    for d, day in df.groupby(df.index.date):
        rng = day[day.index.time <= r_end]
        if len(rng) < 2: continue
        rh, rl = rng["high"].max(), rng["low"].min()
        if rh <= rl: continue

        win  = day[(day.index.time > r_end) & (day.index.time <= e_end)]
        rest = day[day.index.time > r_end]
        if win.empty: continue

        ep = side = ets = None
        for ts, bar in win.iterrows():
            if direction in ("both","long") and bar["high"] > rh:
                ep=rh; side=1; ets=ts; break
            if direction in ("both","short") and bar["low"] < rl:
                ep=rl; side=-1; ets=ts; break
        if ep is None: continue

        tgt = ep*(1+side*profit_pct); stp = ep*(1-side*stop_pct)
        xp = xt = None
        for ts, bar in rest[rest.index > ets].iterrows():
            if side==1:
                if bar["low"]<=stp:  xp=stp; xt="stop"; break
                if bar["high"]>=tgt: xp=tgt; xt="tp";   break
            else:
                if bar["high"]>=stp: xp=stp; xt="stop"; break
                if bar["low"]<=tgt:  xp=tgt; xt="tp";   break
            if bar.name.time() >= f_ext:
                xp=bar["close"]; xt="force"; break
        if xp is None:
            af = rest[rest.index>ets]
            if af.empty: continue
            xp=af.iloc[-1]["close"]; xt="force"

        gross = side*(xp/ep-1)
        trades.append({"date":d,"side":"L"if side==1 else"S","xt":xt,"net":gross-cost})
    return trades


def stats(trades):
    if len(trades) < 5: return None
    r = np.array([t["net"] for t in trades])
    n,mu,sd = len(r),r.mean(),r.std()
    if sd==0: return None
    wins=r[r>0]; loses=abs(r[r<0])
    pf = wins.sum()/loses.sum() if loses.sum()>0 else 0
    return dict(n=n, sh=round(mu/sd*np.sqrt(252),2),
                t=round(mu/sd*np.sqrt(n),2),
                wr=round((r>0).mean()*100,1),
                mu=round(mu*100,3), pf=round(pf,2),
                tp_r=round(sum(1 for t in trades if t["xt"]=="tp")/n*100,1))


# ===== メイン =====
stocks = top_stocks()
print(f"\n対象 {len(stocks)} 銘柄: {[c[:4] for c,n in stocks[:8]]}...")

combos = list(product(*GRID.values()))
print(f"グリッド: {len(combos)} 組合 × {len(stocks)} 銘柄 = {len(combos)*len(stocks):,} バックテスト\n")

# データロード + Phase1 + Phase2 を銘柄ループで一緒に実行
phase1 = []  # ベースパラメータ結果
grid_by_combo = {combo: {"is":[], "oos":[]} for combo in combos}

for ci, (code, name) in enumerate(stocks):
    print(f"  [{ci+1:2}/{len(stocks)}] {code[:4]} {name[:12]}...", end=" ", flush=True)

    df_is  = load_stock(code, IS_START, IS_END)
    df_oos = load_stock(code, OOS_START, OOS_END)

    # Phase 1: ベースパラメータ
    tr_is  = simulate(df_is,  **BASE)
    tr_oos = simulate(df_oos, **BASE)
    si = stats(tr_is); so = stats(tr_oos)
    if si and si["n"] >= 20:
        phase1.append({"code":code,"name":name[:10],
                        "is_sh":si["sh"],"is_n":si["n"],"is_wr":si["wr"],
                        "is_pf":si["pf"],"is_mu":si["mu"],"is_tp":si["tp_r"],
                        "oos_sh":so["sh"] if so else None,
                        "oos_n":so["n"] if so else 0,
                        "oos_wr":so["wr"] if so else None})

    # Phase 2: 全グリッド
    for combo in combos:
        p = dict(zip(GRID.keys(), combo))
        ti = simulate(df_is,  **p); to = simulate(df_oos, **p)
        si2 = stats(ti); so2 = stats(to)
        if si2 and si2["n"]>=15: grid_by_combo[combo]["is"].append(si2["sh"])
        if so2 and so2["n"]>=8:  grid_by_combo[combo]["oos"].append(so2["sh"])

    print(f"IS N={len(tr_is)}, OOS N={len(tr_oos)}", flush=True)

# ===== Phase 1 結果 =====
df1 = pd.DataFrame(phase1).sort_values("is_sh", ascending=False)
df1.to_csv(HERE/"phase1_scan.csv", index=False)

print(f"\n{'='*70}")
print("Phase 1: ベースパラメータ結果 (range=10m, entry≤10h, TP=2%, SL=1%, Both)")
print(f"{'='*70}")
print(f"  有効: {len(df1)} 銘柄")
print(f"  IS Sh>0: {(df1['is_sh']>0).sum()}  IS Sh>1: {(df1['is_sh']>1).sum()}  IS Sh>2: {(df1['is_sh']>2).sum()}")
print(f"  IS&OOS両Sh>0: {((df1['is_sh']>0)&(df1['oos_sh'].fillna(-9)>0)).sum()}")
print()
print(f"  {'code':<7}{'name':<12}{'IS Sh':>7}{'IS N':>6}{'IS TP%':>7}{'IS PF':>6}{'OOS Sh':>8}{'OOS WR':>8}")
for _, r in df1.head(30).iterrows():
    oos_sh = f"{r['oos_sh']:.2f}" if r['oos_sh'] is not None else "  N/A"
    oos_wr = f"{r['oos_wr']:.1f}" if r['oos_wr'] is not None else "  N/A"
    print(f"  {r['code'][:4]:<7}{r['name']:<12}{r['is_sh']:>7.2f}{r['is_n']:>6}"
          f"{r['is_tp']:>7.1f}{r['is_pf']:>6.2f}{oos_sh:>8}{oos_wr:>8}")

# ===== Phase 2 結果 =====
grid_res = []
for combo, d in grid_by_combo.items():
    if len(d["is"]) < 3: continue
    p = dict(zip(GRID.keys(), combo))
    grid_res.append({**p,
        "is_m": round(np.mean(d["is"]),3),
        "is_md":round(np.median(d["is"]),3),
        "oos_m": round(np.mean(d["oos"]),3) if d["oos"] else None,
        "oos_md":round(np.median(d["oos"]),3) if d["oos"] else None,
        "oos_pos":sum(1 for x in d["oos"] if x>0),
        "oos_ns": len(d["oos"]),
    })

df2 = pd.DataFrame(grid_res)
df2.to_csv(HERE/"phase2_grid.csv", index=False)
dg = df2[df2["oos_m"].notna()].sort_values("oos_m", ascending=False)

print(f"\n{'='*70}")
print(f"Phase 2: グリッド結果 ({len(dg)} 有効組合 / OOS平均Sh順)")
print(f"{'='*70}")
print(f"  {'Rng':>4} {'Ent':>4} {'TP%':>5} {'SL%':>6} {'Dir':<6} {'IS_m':>6} {'OOS_m':>6} {'OOS+':>6}")
for _, r in dg.head(25).iterrows():
    print(f"  {r['range_end_m']:>4}m {r['entry_end_h']:>4}h "
          f"{r['profit_pct']*100:>4.1f}% {r['stop_pct']*100:>5.2f}% "
          f"{r['direction']:<6} {r['is_m']:>6.2f} {r['oos_m']:>6.2f} {r['oos_pos']:>6}")

# ===== Phase 3: 感度 =====
print(f"\n{'='*70}")
print("Phase 3: パラメータ感度")
print(f"{'='*70}")

for dim, vals in [("direction",["both","long","short"]),
                  ("range_end_m",[5,10,20]),
                  ("entry_end_h",[10,11]),
                  ("profit_pct",[0.010,0.015,0.020,0.030]),
                  ("stop_pct",[0.005,0.010,0.015])]:
    print(f"\n  [{dim}]")
    for v in vals:
        sub = dg[dg[dim]==v]
        if sub.empty: continue
        label = f"{v*100:.1f}%" if "pct" in dim else str(v)
        top5  = sub.nlargest(5,"oos_m")
        print(f"    {label:>8}: OOS mean={sub['oos_m'].mean():>+.3f}"
              f"  best={sub['oos_m'].max():>+.3f}"
              f"  top5avg={top5['oos_m'].mean():>+.3f}"
              f"  Sh>0率={(sub['oos_m']>0).mean()*100:.0f}%")

print(f"\n\n✅ 完了 → phase1_scan.csv, phase2_grid.csv")
