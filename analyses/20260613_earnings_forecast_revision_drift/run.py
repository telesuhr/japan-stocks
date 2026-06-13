"""業績予想の修正後ドリフト検証。
仮説: 営業利益の上方修正→開示後の正の超過ドリフト、下方→負。PEADのガイダンス改定版。
先読み排除(教訓1): 開示の次の寄りでエントリ。ギャップ(取引不可)とドリフト(取引可)を分解。コスト往復20bp(教訓2)。
"""
import os, sys, datetime
sys.stdout.reconfigure(line_buffering=True)
import numpy as np, pandas as pd, psycopg2
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

PG = {"host": os.environ.get("PGHOST", "localhost"), "port": int(os.environ.get("PGPORT", 5432)),
      "user": os.environ.get("PGUSER", "postgres"), "password": os.environ.get("PGPASSWORD", "postgres"),
      "dbname": os.environ.get("PGDATABASE", "market_data")}
HORIZONS = [1, 3, 5, 10]
COST_RT = 0.0020              # 往復20bp(中小型混在で高め)
OOS_START = pd.Timestamp("2023-01-01")
HOLD = 5                      # 戦略の保有日数

conn = psycopg2.connect(**PG)
rev = pd.read_sql("SELECT code, disc_date, disc_time, direction, rev_op_pct FROM public.earnings_forecast_revisions WHERE rev_op_pct IS NOT NULL", conn)
sd = pd.read_sql("SELECT code,date,adj_open,adj_close FROM stocks_daily WHERE code=ANY(%s) AND date>='2016-01-01'",
                 conn, params=[list(rev.code.unique())])
top = pd.read_sql("SELECT date,open,close FROM index_daily WHERE code='0000' AND date>='2016-01-01'", conn)
conn.close()

# --- TOPIX 準備 ---
top["date"] = pd.to_datetime(top["date"]); top = top.sort_values("date").reset_index(drop=True)
top_dates = top["date"].values.astype("datetime64[D]")
top_open = top["open"].astype(float).values; top_close = top["close"].astype(float).values
top_idx = {d: i for i, d in enumerate(top_dates)}

# --- 銘柄別 価格配列 ---
sd["date"] = pd.to_datetime(sd["date"]); sd = sd.sort_values(["code", "date"])
carr = {}
for code, g in sd.groupby("code"):
    carr[code] = (g["date"].values.astype("datetime64[D]"),
                  g["adj_open"].astype(float).values, g["adj_close"].astype(float).values)

# --- イベントごとに ギャップ / ドリフト 超過を算出 ---
rev["disc_date"] = pd.to_datetime(rev["disc_date"])
rows = []
for r in rev.itertuples():
    a = carr.get(r.code)
    if a is None:
        continue
    cd, co, cc = a
    disc = np.datetime64(r.disc_date.date(), "D")
    preopen = (r.disc_time is not None) and (r.disc_time < datetime.time(9, 0))
    ei = np.searchsorted(cd, disc, side=("left" if preopen else "right"))
    if ei < 1 or ei >= len(cd):
        continue
    entry_date = cd[ei]
    ti = top_idx.get(entry_date)
    if ti is None or ti < 1:
        continue
    entry_open = co[ei]
    if entry_open <= 0:
        continue
    rec = {"code": r.code, "entry_date": pd.Timestamp(entry_date), "op": float(r.rev_op_pct),
           "direction": r.direction,
           "exc_gap": (entry_open / cc[ei - 1] - 1) - (top_open[ti] / top_close[ti - 1] - 1)}
    for h in HORIZONS:
        xi = ei + h
        if xi >= len(cd):
            rec[f"d{h}"] = np.nan; continue
        txi = top_idx.get(cd[xi])
        if txi is None:
            rec[f"d{h}"] = np.nan; continue
        rec[f"d{h}"] = (cc[xi] / entry_open - 1) - (top_close[txi] / top_open[ti] - 1)
    rows.append(rec)

df = pd.DataFrame(rows).dropna(subset=["d5"])
print(f"events n={len(df)}  {df.entry_date.min().date()} ~ {df.entry_date.max().date()}")

# --- 1) 営業利益改定率 五分位 → ギャップ vs ドリフト ---
df["q"] = pd.qcut(df["op"], 5, labels=[1, 2, 3, 4, 5])
print("\n=== rev_op_pct 五分位 → 超過リターン(bp) [gap=取引不可 / d1..d10=寄り後ドリフト=取引可] ===")
agg = df.groupby("q").agg(n=("op", "size"), gap=("exc_gap", "mean"),
                          d1=("d1", "mean"), d3=("d3", "mean"), d5=("d5", "mean"), d10=("d10", "mean"))
for c in ["gap", "d1", "d3", "d5", "d10"]:
    agg[c] = (agg[c] * 1e4).round(1)
print(agg.to_string())

# --- 2) 戦略: Q5中立ロング + Q1中立ショート, 翌寄り, HOLD日, コスト込 ---
def stats(sub, sign):
    pnl = sign * sub[f"d{HOLD}"] - COST_RT
    if len(pnl) == 0:
        return None
    return dict(n=len(pnl), net_bp=pnl.mean() * 1e4, hit=(pnl > 0).mean() * 100,
                sharpe=(pnl.mean() / pnl.std() * np.sqrt(252 / HOLD)) if pnl.std() > 0 else np.nan)

print(f"\n=== 戦略: Q5上方=中立ロング / Q1下方=中立ショート, 翌寄り{HOLD}日保有, 往復{COST_RT*1e4:.0f}bp ===")
for label, d in [("全期間", df), ("IS(〜2022)", df[df.entry_date < OOS_START]), ("OOS(2023〜)", df[df.entry_date >= OOS_START])]:
    longs = stats(d[d.q == 5], +1); shorts = stats(d[d.q == 1], -1)
    both = pd.concat([(+1) * d[d.q == 5][f"d{HOLD}"] - COST_RT, (-1) * d[d.q == 1][f"d{HOLD}"] - COST_RT])
    print(f"  {label:11} ロングQ5: n={longs['n']:4d} net={longs['net_bp']:6.1f}bp 勝率{longs['hit']:.0f}% | "
          f"ショートQ1: n={shorts['n']:4d} net={shorts['net_bp']:6.1f}bp 勝率{shorts['hit']:.0f}% | "
          f"LS合算 net={both.mean()*1e4:6.1f}bp Sharpe={both.mean()/both.std()*np.sqrt(252/HOLD):.2f}")

# --- 3) ギャップ vs ドリフトの含意を一行で ---
g5, d5_5 = agg.loc[5, "gap"], agg.loc[5, "d5"]
g1, d5_1 = agg.loc[1, "gap"], agg.loc[1, "d5"]
print(f"\n[含意] Q5(大幅上方): ギャップ{g5:.0f}bp(取れない) vs 寄り後5日ドリフト{d5_5:.0f}bp(取れる)")
print(f"        Q1(大幅下方): ギャップ{g1:.0f}bp vs 寄り後5日ドリフト{d5_1:.0f}bp")

# --- 可視化 ---
try:
    import matplotlib.font_manager as fm
    fm.fontManager.addfont("/root/.fonts/NotoSansJP.ttf")
    plt.rcParams["font.family"] = "Noto Sans JP"
    plt.rcParams["axes.unicode_minus"] = False
except Exception:
    pass
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2))

# 左: 累積超過(ギャップ→ドリフト) Q5/Q3/Q1
xs = ["gap", "d1", "d3", "d5", "d10"]
xpos = [0, 1, 3, 5, 10]
for q, col, lab in [(5, "#27ae60", "Q5 大幅上方"), (3, "#95a5a6", "Q3 中位"), (1, "#c0392b", "Q1 大幅下方")]:
    ys = [agg.loc[q, c] for c in xs]
    ax1.plot(xpos, ys, "o-", color=col, label=lab, lw=1.8)
ax1.axvline(0.5, color="gray", ls=":", lw=1)
ax1.text(0.5, ax1.get_ylim()[1], " ←取引不可|取引可→", fontsize=8, va="top", color="gray")
ax1.axhline(0, color="k", lw=0.8)
ax1.set_xlabel("開示後の経過(0=ギャップ, 以降 寄り後の営業日)")
ax1.set_ylabel("TOPIX超過リターン (bp)")
ax1.set_title(f"業績予想修正後のドリフト (n={len(df)}, 2016-2026)\nギャップに集約か、寄り後も続くか")
ax1.legend(fontsize=9)

# 右: 戦略 累積ネットP&L (Q5ロング+Q1ショート)
ls = pd.concat([
    df[df.q == 5].assign(pnl=lambda x: x[f"d{HOLD}"] - COST_RT)[["entry_date", "pnl"]],
    df[df.q == 1].assign(pnl=lambda x: -x[f"d{HOLD}"] - COST_RT)[["entry_date", "pnl"]],
]).sort_values("entry_date")
ax2.plot(ls["entry_date"], ls["pnl"].cumsum() * 100, color="#2980b9", lw=1.4)
ax2.axvline(OOS_START, color="red", ls="--", lw=1, alpha=0.7)
ax2.text(OOS_START, ax2.get_ylim()[1], " OOS", color="red", va="top", fontsize=9)
ax2.axhline(0, color="k", lw=0.8)
ax2.set_ylabel("累積ネット超過 (%, コスト込)")
ax2.set_title(f"戦略 累積P&L: Q5中立ロング+Q1中立ショート\n翌寄り{HOLD}日保有・往復{COST_RT*1e4:.0f}bp")
fig.tight_layout()
fig.savefig(os.path.join(os.path.dirname(__file__), "result.png"), dpi=100, bbox_inches="tight")
print("\nsaved result.png")
