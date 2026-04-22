"""Quick subset optimization: 各銘柄の個別ベスト構成を pool して basket を作る"""
import psycopg2, pandas as pd, numpy as np
from pathlib import Path
from itertools import combinations
import sys
sys.path.insert(0, str(Path(__file__).parent))
from run import load_intraday, build_orb_trades, stats, split_h12, SYMBOLS

# 個別ベスト (per_stock_params.csv の sharpe>1.0 上位)
# (sym, or_min, dir, stop) を構成ごとに固定
TOP = pd.read_csv(Path(__file__).parent / "per_stock_params.csv")
# sym ごとに最もSharpeが高い構成を1つに絞る
idx = TOP.groupby("sym")["sharpe"].idxmax()
best_per_sym = TOP.loc[idx].sort_values("sharpe", ascending=False)
print("各銘柄ベスト構成:")
print(best_per_sym[["sym","name","or_min","dir","stop","n","sharpe","h2_sharpe"]].to_string(index=False))

# sharpe > 0.8 の銘柄を候補に
cand = best_per_sym[best_per_sym["sharpe"] > 0.8].head(10)
print(f"\n候補: {cand['name'].tolist()}")

# データ読み込み
all_data = {}
for sym in cand["sym"]:
    df = load_intraday(sym)
    if df is not None: all_data[sym] = df

# 各銘柄を個別ベスト構成で trades を作成
trades_by_sym = {}
for _, row in cand.iterrows():
    sym = row["sym"]
    if sym not in all_data: continue
    t = build_orb_trades(all_data[sym], sym, int(row["or_min"]),
                         direction=row["dir"], stop_mode=row["stop"])
    trades_by_sym[sym] = t

# サブセット探索 (各銘柄は自分のベスト構成)
rows = []
syms = list(trades_by_sym.keys())
for k in range(2, min(6, len(syms)) + 1):
    for combo in combinations(syms, k):
        pooled = pd.concat([trades_by_sym[s] for s in combo], ignore_index=True)
        if len(pooled) < 30: continue
        full, h1, h2 = split_h12(pooled)
        rows.append({
            "combo": "+".join(SYMBOLS[x] for x in combo), "k": k, "n": full["n"],
            "full_sharpe": full["sharpe"], "full_t": full["tstat"], "mean_bp": full["mean"],
            "h1_sharpe": h1["sharpe"], "h2_sharpe": h2["sharpe"], "h2_t": h2["tstat"],
        })
subs = pd.DataFrame(rows).sort_values("full_sharpe", ascending=False)
print("\nTop-15 Subset (各銘柄個別ベスト構成で pool):")
print(subs.head(15).to_string(index=False, float_format=lambda x: f"{x:.2f}"))
subs.to_csv(Path(__file__).parent / "subset_quick.csv", index=False)
print("\n採用基準クリア (Sharpe>=2.0 & N>=30 & t>=2.0):")
ok = subs[(subs["full_sharpe"] >= 2.0) & (subs["n"] >= 30) & (subs["full_t"] >= 2.0)]
print(ok.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
