"""SOX大幅高の翌日本営業日、寄ってからロングを建てて取れるか。

仮説 H1(ギャップで取られる)/H2(日中はエッジなし)/H3(極端日は巻き戻し) は README.md で事前固定。
寄りギャップ・日中(寄→引)・フルを分解し、コスト込みで「寄り後ロング」の期待値を測る。
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

PG = {"host": os.environ.get("PGHOST", "localhost"), "port": int(os.environ.get("PGPORT", 5432)),
      "user": os.environ.get("PGUSER", "postgres"), "dbname": os.environ.get("PGDATABASE", "market_data")}

START = "2020-01-01"
SEMIS = ["80350", "68570", "61460", "69200", "67620", "58030", "285A0"]
COST_BP = 4.0          # 寄り成行+引け成行 往復
SPLIT = "2023-01-01"

conn = psycopg2.connect(**PG)

sox = pd.read_sql("SELECT trade_date, close FROM macro.daily_ohlcv WHERE symbol='.SOX' AND trade_date>=%s ORDER BY trade_date",
                  conn, params=[START])
sox["trade_date"] = pd.to_datetime(sox["trade_date"])
sox["ret"] = sox["close"].pct_change() * 100

# JP銘柄 daily（1321 + 半導体）
jp = pd.read_sql("SELECT code, date, open, close FROM stocks_daily WHERE code = ANY(%s) AND date>=%s ORDER BY date",
                 conn, params=[["13210"] + SEMIS, START])
conn.close()
jp["date"] = pd.to_datetime(jp["date"])
jp[["open", "close"]] = jp[["open", "close"]].astype(float)


def panel(code):
    d = jp[jp["code"] == code].set_index("date").sort_index()
    d["prev_close"] = d["close"].shift(1)
    d["gap"] = (d["open"] / d["prev_close"] - 1) * 1e4
    d["intraday"] = (d["close"] / d["open"] - 1) * 1e4
    d["full"] = (d["close"] / d["prev_close"] - 1) * 1e4
    return d[["gap", "intraday", "full"]]


nikkei = panel("13210")
# 半導体バスケット（等加重）
semi_list = [panel(c) for c in SEMIS]
semi = pd.concat(semi_list).groupby(level=0).mean()

jp_dates = nikkei.index  # 1321の営業日を日本カレンダーに使う


def next_jp_date(d):
    i = jp_dates.searchsorted(d, side="right")
    return jp_dates[i] if i < len(jp_dates) else pd.NaT


def evaluate(name, jp_panel, threshold):
    ev_sox = sox[sox["ret"] >= threshold].copy()
    ev_sox["jp_next"] = ev_sox["trade_date"].map(next_jp_date)
    rows = jp_panel.reindex(ev_sox["jp_next"].dropna().unique()).dropna()
    base = jp_panel.dropna()
    out = {"target": name, "thr": threshold, "n": len(rows)}
    for col in ["gap", "intraday", "full"]:
        ev_mean = rows[col].mean()
        base_mean = base[col].mean()
        t, p = stats.ttest_1samp(rows[col], 0)
        out[f"{col}_bp"] = round(ev_mean, 1)
        out[f"{col}_excess"] = round(ev_mean - base_mean, 1)  # 無条件超過
        out[f"{col}_t"] = round(t, 2)
    out["intraday_net_bp"] = round(rows["intraday"].mean() - COST_BP, 1)
    out["intraday_win%"] = round((rows["intraday"] > 0).mean() * 100, 1)
    return out, rows


print("=== ベースライン(無条件 全営業日) ===")
for name, p in [("日経1321", nikkei), ("半導体バスケット", semi)]:
    b = p.dropna()
    print(f"{name}: gap {b['gap'].mean():+.1f}bp / 日中 {b['intraday'].mean():+.1f}bp / フル {b['full'].mean():+.1f}bp (n={len(b)})")

results = []
detail = {}
for name, p in [("日経1321", nikkei), ("半導体バスケット", semi)]:
    for thr in [3.0, 5.0]:
        r, rows = evaluate(name, p, thr)
        results.append(r)
        detail[(name, thr)] = rows
res = pd.DataFrame(results)
print("\n=== SOX大幅高の翌日本営業日（bp、excess=無条件超過、net=コスト4bps控除後）===")
show = res[["target", "thr", "n", "gap_bp", "gap_excess", "intraday_bp", "intraday_excess",
            "intraday_t", "intraday_net_bp", "intraday_win%", "full_bp"]]
print(show.to_string(index=False))

# 頑健性: 半導体+3%を前後で
semi_rows = detail[("半導体バスケット", 3.0)]
for lab, lo, hi in [("2020-2022", "2020-01-01", "2022-12-31"), ("2023-2026", "2023-01-01", "2026-12-31")]:
    sub = semi_rows[(semi_rows.index >= lo) & (semi_rows.index <= hi)]
    if len(sub):
        print(f"  [{lab}] 半導体+3% n={len(sub)}: ギャップ{sub['gap'].mean():+.1f} / 日中{sub['intraday'].mean():+.1f}bp")

res.to_csv("results.csv", index=False)

# ── 可視化 ──
import matplotlib.font_manager as fm
for _f in ["/mnt/c/Windows/Fonts/YuGothM.ttc", "/mnt/c/Windows/Fonts/meiryo.ttc"]:
    if os.path.exists(_f):
        fm.fontManager.addfont(_f); plt.rcParams["font.family"] = fm.FontProperties(fname=_f).get_name(); break
plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(1, 2, figsize=(12, 6.75), facecolor="white")
sub = res[res["thr"] == 3.0]
x = np.arange(len(sub)); w = 0.27
axes[0].bar(x - w, sub["gap_bp"], w, label="ギャップ(前夜→寄り)", color="#2b8cbe")
axes[0].bar(x, sub["intraday_bp"], w, label="日中(寄→引)", color="#e34a33")
axes[0].bar(x + w, sub["intraday_net_bp"], w, label="日中 コスト後", color="#fdbb84")
axes[0].set_xticks(x, sub["target"]); axes[0].axhline(0, color="gray", lw=0.8)
axes[0].set_ylabel("リターン (bp)"); axes[0].legend(fontsize=8)
axes[0].set_title("SOX+3%翌日: 上げはギャップに集約、寄→引は薄い/マイナス")

semi3 = detail[("半導体バスケット", 3.0)]
axes[1].scatter(semi3["gap"], semi3["intraday"], s=20, alpha=0.6, color="#2b8cbe")
axes[1].axhline(0, color="gray", lw=0.8); axes[1].axvline(0, color="gray", lw=0.8)
axes[1].set_xlabel("寄りギャップ (bp)"); axes[1].set_ylabel("日中 寄→引 (bp)")
axes[1].set_title("半導体: ギャップ大ほど日中は伸びない（巻き戻し）")

fig.suptitle("SOX大幅高の翌日本営業日に『寄ってからロング』は間に合うか", fontsize=13)
fig.text(0.99, 0.01, "データ: 2020-2026 / .SOX × stocks_daily(1321+半導体7) / 寄→引 往復コスト4bps控除",
         ha="right", va="bottom", fontsize=8, color="gray")
plt.tight_layout(rect=[0, 0.02, 1, 0.95])
fig.savefig("result.png", dpi=100, bbox_inches="tight", facecolor="white")
print("saved result.png / results.csv")
