"""ギャップダウン日の日中回復を寄り前変数で分類できるかの検証。

仮説 H1(ADR乖離)/H2(ギャップ深さ)/H4(25日線乖離)/H5(円方向) は README.md で事前固定。
"""
import os
import sys
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
import psycopg2
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PG_CONFIG = {
    "host": os.environ.get("PGHOST", "localhost"),
    "port": int(os.environ.get("PGPORT", 5432)),
    "user": os.environ.get("PGUSER", "postgres"),
    "dbname": os.environ.get("PGDATABASE", "market_data"),
}

START = "2020-01-01"
GAP_TH = -1.0          # ギャップダウン閾値(%)
SPLIT_DATE = "2023-01-01"

conn = psycopg2.connect(**PG_CONFIG)

# ── 日経225 日足: 連動ETF 1321 を現物プロキシに使う ──────────────────────────
# index_daily の NK225D/F は夜間込み先物系列で寄りギャップが夜間に吸収される
# (gap std 0.26%)。ETF 1321 は 9:00 現物寄り板を持つ (gap std 1.12%)。
nk = pd.read_sql(
    "SELECT date, open, close FROM stocks_daily WHERE code='13210' AND date>=%s ORDER BY date",
    conn, params=[START])
nk["date"] = pd.to_datetime(nk["date"])
nk = nk.set_index("date").astype(float)
nk["prev_close"] = nk["close"].shift(1)
nk["gap_pct"] = (nk["open"] / nk["prev_close"] - 1) * 100
nk["oc_ret"] = (nk["close"] / nk["open"] - 1) * 100
nk["ma25_dev"] = (nk["prev_close"] / nk["close"].shift(1).rolling(25).mean() - 1) * 100

# ── 米国側（ADR・SOX・ドル円）: 日本の日付dに対し直近の米営業日を対応付け ──────
mac = pd.read_sql(
    """SELECT symbol, trade_date, close FROM macro.daily_ohlcv
       WHERE (symbol LIKE 'ADR_%%' OR symbol IN ('.SOX','JPY=')) AND trade_date>=%s
       ORDER BY trade_date""",
    conn, params=["2019-12-01"])
conn.close()
mac["trade_date"] = pd.to_datetime(mac["trade_date"])
piv = mac.pivot_table(index="trade_date", columns="symbol", values="close")
rets = piv.pct_change() * 100

adr_cols = [c for c in piv.columns if c.startswith("ADR_")]
us = pd.DataFrame({
    "adr_avg": rets[adr_cols].mean(axis=1),
    "sox": rets[".SOX"],
    "jpy_chg": rets["JPY="],
})
us["adr_excess"] = us["adr_avg"] - us["sox"]
us = us.dropna(subset=["sox"])

# 日本の各営業日に「その寄り前までに確定している直近の米営業日」を対応付け
us_dates = us.index
def last_us(d):
    i = us_dates.searchsorted(d) - 1   # d当日の米国セッションは日本の寄り後なので除外
    return us_dates[i] if i >= 0 else pd.NaT

nk["us_date"] = [last_us(d) for d in nk.index]
nk = nk.join(us, on="us_date")

# ── イベント抽出 ──────────────────────────────────────────────────────────────
ev = nk[(nk["gap_pct"] <= GAP_TH)].dropna(
    subset=["oc_ret", "adr_excess", "ma25_dev", "jpy_chg"]).copy()
ev["recovered"] = ev["oc_ret"] > 0
print(f"イベント数(gap<= {GAP_TH}%): {len(ev)}  期間 {ev.index.min().date()}〜{ev.index.max().date()}")
print(f"無条件: 回復率 {ev['recovered'].mean()*100:.1f}% / 平均oc_ret {ev['oc_ret'].mean():+.2f}%")

# ── 仮説別の検定 ──────────────────────────────────────────────────────────────
# (特徴量, 有利とする側, 仮説ラベル)
FEATURES = [
    ("adr_excess", "high", "H1 ADR乖離(底堅い)"),
    ("gap_pct",    "low",  "H2 ギャップ深い"),
    ("ma25_dev",   "low",  "H4 過熱解消済み"),
    ("jpy_chg",    "high", "H5 円安方向"),
]

def eval_split(df, col, side):
    med = df[col].median()
    fav = df[df[col] >= med] if side == "high" else df[df[col] < med]
    unf = df[df[col] < med] if side == "high" else df[df[col] >= med]
    t, p = stats.ttest_ind(fav["oc_ret"], unf["oc_ret"], equal_var=False)
    rho, rho_p = stats.spearmanr(df[col], df["oc_ret"])
    if side == "low":
        rho, = (-rho,)
    return {"n_fav": len(fav), "fav_oc": fav["oc_ret"].mean(), "unf_oc": unf["oc_ret"].mean(),
            "fav_rec": fav["recovered"].mean() * 100, "unf_rec": unf["recovered"].mean() * 100,
            "t": t if side == "high" else -t, "p": p, "rho_fav": rho}

rows = []
for period, df in [("full", ev),
                   ("2020-2022", ev[ev.index < SPLIT_DATE]),
                   ("2023-2026", ev[ev.index >= SPLIT_DATE])]:
    for col, side, label in FEATURES:
        r = eval_split(df, col, side)
        rows.append({"period": period, "hypothesis": label,
                     "n_fav": r["n_fav"],
                     "fav_oc_ret": round(r["fav_oc"], 2), "unf_oc_ret": round(r["unf_oc"], 2),
                     "diff": round(r["fav_oc"] - r["unf_oc"], 2),
                     "fav_recov_%": round(r["fav_rec"], 1), "unf_recov_%": round(r["unf_rec"], 1),
                     "t": round(r["t"], 2), "rho(+=支持)": round(r["rho_fav"], 2)})
res = pd.DataFrame(rows)
print("\n=== 仮説別: 有利側 vs 不利側の oc_ret(寄り→引け%) ===")
print(res.to_string(index=False))

# ── 複合スコア ────────────────────────────────────────────────────────────────
ev["score"] = 0
for col, side, _ in FEATURES:
    med = ev[col].median()
    ev["score"] += ((ev[col] >= med) if side == "high" else (ev[col] < med)).astype(int)
sc = ev.groupby("score").agg(n=("oc_ret", "size"), oc_ret=("oc_ret", "mean"),
                             recov=("recovered", "mean"))
sc["recov"] *= 100
print("\n=== 複合スコア(0-4)別 ===")
print(sc.round(2).to_string())

lo = ev[ev["score"] <= 1]["oc_ret"]
hi = ev[ev["score"] >= 3]["oc_ret"]
t_sc, p_sc = stats.ttest_ind(hi, lo, equal_var=False)
print(f"\nscore>=3 vs <=1: {hi.mean():+.2f}% vs {lo.mean():+.2f}% (t={t_sc:.2f}, p={p_sc:.3f}, n={len(hi)}/{len(lo)})")

res.to_csv("results.csv", index=False)
ev[["gap_pct", "oc_ret", "adr_excess", "ma25_dev", "jpy_chg", "score", "recovered"]].to_csv("events.csv")

# ── 可視化 ────────────────────────────────────────────────────────────────────
import matplotlib.font_manager as fm
for _f in ["/mnt/c/Windows/Fonts/YuGothM.ttc", "/mnt/c/Windows/Fonts/meiryo.ttc"]:
    if os.path.exists(_f):
        fm.fontManager.addfont(_f)
        plt.rcParams["font.family"] = fm.FontProperties(fname=_f).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(1, 2, figsize=(12, 6.75), facecolor="white")
full = res[res["period"] == "full"]
x = np.arange(len(full))
axes[0].bar(x - 0.2, full["fav_oc_ret"], 0.4, label="条件あり(有利側)", color="#2b8cbe")
axes[0].bar(x + 0.2, full["unf_oc_ret"], 0.4, label="条件なし(不利側)", color="#cccccc")
axes[0].set_xticks(x, [h.split(" ")[0] for h in full["hypothesis"]])
axes[0].axhline(0, color="gray", lw=0.8)
axes[0].set_ylabel("寄り→引け 平均リターン (%)")
axes[0].set_title("仮説別: 中央値分割の oc_ret 差")
axes[0].legend(fontsize=9)

axes[1].bar(sc.index.astype(str), sc["oc_ret"], color="#2b8cbe")
for i, (idx, row) in enumerate(sc.iterrows()):
    axes[1].text(i, row["oc_ret"], f"n={int(row['n'])}\n{row['recov']:.0f}%",
                 ha="center", va="bottom", fontsize=8)
axes[1].axhline(0, color="gray", lw=0.8)
axes[1].set_xlabel("複合スコア(有利条件の本数)")
axes[1].set_ylabel("寄り→引け 平均リターン (%)")
axes[1].set_title("複合スコア別 oc_ret（注記=件数/回復率）")

fig.suptitle("日経ギャップダウン(≤-1%)日の日中回復は寄り前変数で分けられるか", fontsize=13)
fig.text(0.99, 0.01, "データ: 2020-01〜2026-06 / index_daily NK225D・ADR17・.SOX・JPY= (yfinance/Refinitiv)",
         ha="right", va="bottom", fontsize=8, color="gray")
plt.tight_layout(rect=[0, 0.02, 1, 0.95])
fig.savefig("result.png", dpi=100, bbox_inches="tight", facecolor="white")
print("saved result.png / results.csv / events.csv")
