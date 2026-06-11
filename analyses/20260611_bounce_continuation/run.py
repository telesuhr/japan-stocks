"""前場の「戻りの質」で後場継続/だましを識別できるかの検証。

仮説 H1(VWAP奪回)/H2(出来高の伴い方)/H3(安値の時刻)/H4(戻し率) は README.md で事前固定。
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

CODE = "13210"
DROP_TH = -1.0      # 前場安値の前日終値比(%)
RETRACE_MIN = 1 / 3  # 前場引けまでの最低戻し率（=戻り兆候の定義）
SPLIT_DATE = "2025-06-01"

conn = psycopg2.connect(**PG_CONFIG)
bars = pd.read_sql(
    """SELECT ts, open, high, low, close, volume FROM stocks_intraday
       WHERE code=%s ORDER BY ts""",
    conn, params=[CODE])
daily = pd.read_sql(
    "SELECT date, close FROM stocks_daily WHERE code=%s ORDER BY date",
    conn, params=[CODE])
conn.close()

bars["ts"] = pd.to_datetime(bars["ts"])
bars["date"] = bars["ts"].dt.date
daily["date"] = pd.to_datetime(daily["date"]).dt.date
prev_close = daily.set_index("date")["close"].astype(float).shift(1)

events = []
for d, g in bars.groupby("date"):
    pc = prev_close.get(d)
    if pc is None or np.isnan(pc):
        continue
    g = g.drop(columns=["date"]).set_index("ts").astype(float)
    am = g.between_time("09:00", "11:30")
    pm = g.between_time("12:30", "15:30")
    if len(am) < 60 or len(pm) < 60:
        continue

    am_low = am["low"].min()
    drop_pct = (am_low / pc - 1) * 100
    if drop_pct > DROP_TH:
        continue
    p_close_am = am["close"].iloc[-1]
    retrace = (p_close_am - am_low) / (pc - am_low) if pc > am_low else np.nan
    if not (retrace >= RETRACE_MIN):
        continue  # 戻り兆候のない日は対象外

    # 前場引け時点で確定する特徴量
    vwap = (am["close"] * am["volume"]).sum() / am["volume"].sum()
    ret1m = am["close"].pct_change()
    upvol = am["volume"][ret1m > 0].sum() / am["volume"].sum()
    low_minute = (am["low"].idxmin() - am.index[0]).total_seconds() / 60

    pm_ret = (pm["close"].iloc[-1] / p_close_am - 1) * 100
    events.append({"date": d, "drop_pct": drop_pct, "retrace": retrace,
                   "vwap_above": p_close_am > vwap,
                   "vwap_dev": (p_close_am / vwap - 1) * 100,
                   "upvol": upvol, "low_minute": low_minute,
                   "pm_ret": pm_ret, "continued": pm_ret > 0})

ev = pd.DataFrame(events).set_index("date")
ev.index = pd.to_datetime(ev.index)
print(f"イベント数(前場安値<={DROP_TH}% かつ 1/3以上戻し): {len(ev)}")
print(f"無条件: 後場継続率 {ev['continued'].mean()*100:.1f}% / 平均後場リターン {ev['pm_ret'].mean():+.2f}%")

FEATURES = [
    ("vwap_dev",   "high", "H1 VWAP奪回"),
    ("upvol",      "high", "H2 出来高伴う"),
    ("low_minute", "low",  "H3 安値が早い"),
    ("retrace",    "high", "H4 半値戻し級"),
]

def eval_split(df, col, side):
    med = df[col].median()
    fav = df[df[col] >= med] if side == "high" else df[df[col] < med]
    unf = df[df[col] < med] if side == "high" else df[df[col] >= med]
    t, p = stats.ttest_ind(fav["pm_ret"], unf["pm_ret"], equal_var=False)
    rho, _ = stats.spearmanr(df[col], df["pm_ret"])
    if side == "low":
        rho = -rho
    return fav, unf, t if side == "high" else -t, p, rho

rows = []
for period, df in [("full", ev),
                   ("front", ev[ev.index < SPLIT_DATE]),
                   ("back", ev[ev.index >= SPLIT_DATE])]:
    for col, side, label in FEATURES:
        if len(df) < 10:
            continue
        fav, unf, t, p, rho = eval_split(df, col, side)
        rows.append({"period": period, "hypothesis": label, "n_fav": len(fav),
                     "fav_pm": round(fav["pm_ret"].mean(), 2),
                     "unf_pm": round(unf["pm_ret"].mean(), 2),
                     "diff": round(fav["pm_ret"].mean() - unf["pm_ret"].mean(), 2),
                     "fav_cont%": round(fav["continued"].mean() * 100, 1),
                     "unf_cont%": round(unf["continued"].mean() * 100, 1),
                     "t": round(t, 2), "p": round(p, 3), "rho": round(rho, 2)})
res = pd.DataFrame(rows)
print("\n=== 仮説別: 有利側 vs 不利側の後場リターン(%) ===")
print(res.to_string(index=False))

# VWAP奪回は2値なのでそのまま比較も出す
va, vb = ev[ev["vwap_above"]], ev[~ev["vwap_above"]]
if len(vb) > 3:
    t, p = stats.ttest_ind(va["pm_ret"], vb["pm_ret"], equal_var=False)
    print(f"\nVWAP上 vs 下(2値): {va['pm_ret'].mean():+.2f}% (n={len(va)}, 継続{va['continued'].mean()*100:.0f}%)"
          f" vs {vb['pm_ret'].mean():+.2f}% (n={len(vb)}, 継続{vb['continued'].mean()*100:.0f}%) t={t:.2f} p={p:.3f}")

res.to_csv("results.csv", index=False)
ev.to_csv("events.csv")

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
axes[0].bar(x - 0.2, full["fav_pm"], 0.4, label="条件あり(有利側)", color="#2b8cbe")
axes[0].bar(x + 0.2, full["unf_pm"], 0.4, label="条件なし(不利側)", color="#cccccc")
for i, r in full.reset_index().iterrows():
    axes[0].text(i, max(r["fav_pm"], 0) + 0.01, f"t={r['t']}", ha="center", fontsize=8)
axes[0].set_xticks(x, [h.split(" ")[0] for h in full["hypothesis"]])
axes[0].axhline(0, color="gray", lw=0.8)
axes[0].set_ylabel("後場リターン (%)")
axes[0].set_title("仮説別: 前場の戻りの質 → 後場リターン")
axes[0].legend(fontsize=9)

axes[1].scatter(ev["vwap_dev"], ev["pm_ret"], s=18, alpha=0.6, color="#2b8cbe")
axes[1].axhline(0, color="gray", lw=0.8)
axes[1].axvline(0, color="gray", lw=0.8)
axes[1].set_xlabel("前場引けのVWAP乖離 (%)")
axes[1].set_ylabel("後場リターン (%)")
axes[1].set_title("H1: VWAP乖離 vs 後場リターン（散布）")

fig.suptitle("朝安→前場戻しの「質」で後場継続を識別できるか（ETF1321）", fontsize=13)
fig.text(0.99, 0.01, "データ: 2024-05〜2026-06 / stocks_intraday 1321 1分足 (JQuants)",
         ha="right", va="bottom", fontsize=8, color="gray")
plt.tight_layout(rect=[0, 0.02, 1, 0.95])
fig.savefig("result.png", dpi=100, bbox_inches="tight", facecolor="white")
print("saved result.png / results.csv / events.csv")
