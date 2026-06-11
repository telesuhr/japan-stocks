"""aukabu板シグナルのイベントスタディ。

方向つき板シグナル（imbalanceShift/vwapDeviation/marketPressure）の発火後
1/5/15分の前方リターンを、スプレッドコスト控除後で評価する。
仮説・設計は README.md（検証前に固定済み）を参照。
"""
import os
import sys
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
import psycopg2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PG_CONFIG = {
    "host": os.environ.get("PGHOST", "localhost"),
    "port": int(os.environ.get("PGPORT", 5432)),
    "user": os.environ.get("PGUSER", "postgres"),
    "dbname": os.environ.get("PGDATABASE", "market_data"),
}

START, END = "2026-05-22", "2026-06-10"
SPLIT = "2026-06-04"           # 前半/後半の頑健性チェック境界
COOLDOWN_MIN = 5               # 同一銘柄×種別×方向の間引き
HORIZONS = [1, 5, 15]
DIRECTIONAL = ("imbalanceShift", "vwapDeviation", "marketPressure")

conn = psycopg2.connect(**PG_CONFIG)

# ── 1. シグナル取得（方向つき + 合流再構成用にvolumeSpikeも） ────────────────
sig = pd.read_sql(
    """SELECT event_time, symbol, signal_type, value
       FROM aukabu.signals
       WHERE event_time::date BETWEEN %s AND %s
       ORDER BY symbol, event_time""",
    conn, params=[START, END])
# event_time は timestamptz(UTC) で返る → JSTへ変換して naive 化（stocks_intraday と揃える）
sig["event_time"] = (pd.to_datetime(sig["event_time"], utc=True)
                     .dt.tz_convert("Asia/Tokyo").dt.tz_localize(None))
sig["minute"] = sig["event_time"].dt.floor("min")
sig["direction"] = np.sign(sig["value"]).astype(int)
print(f"raw signals: {len(sig):,}")

# ── 2. 1分足（JQuants確定値）: 監視銘柄×期間分 ──────────────────────────────
codes5 = sorted({s + "0" for s in sig["symbol"].unique() if len(s) == 4})
px = pd.read_sql(
    """SELECT code, ts, close FROM stocks_intraday
       WHERE code = ANY(%s) AND ts::date BETWEEN %s AND %s""",
    conn, params=[codes5, START, END])
px["ts"] = pd.to_datetime(px["ts"])
px["symbol"] = px["code"].str[:4]
price = px.set_index(["symbol", "ts"])["close"].sort_index()
print(f"intraday bars: {len(px):,} ({px['symbol'].nunique()} symbols)")

# ── 3. 銘柄別スプレッド（bp） ────────────────────────────────────────────────
# kabu APIは Bid=売気配/Ask=買気配 の逆名称のため abs() で実スプレッドを取る
spread = pd.read_sql(
    """SELECT symbol, avg(abs(bid_price-ask_price)/nullif(price,0))*1e4 AS spread_bp
       FROM aukabu.snapshots_5sec
       WHERE bucket_ts::date BETWEEN %s AND %s
         AND price > 0 AND bid_price > 0 AND ask_price > 0
       GROUP BY symbol""",
    conn, params=[START, END]).set_index("symbol")["spread_bp"]
conn.close()
print(f"spread: median {spread.median():.1f}bp / range {spread.min():.1f}-{spread.max():.1f}bp")


def forward_returns(events: pd.DataFrame) -> pd.DataFrame:
    """各イベントにエントリー価格と前方リターン(bp, 方向符号調整済み)を付与。"""
    rows = []
    for sym, grp in events.groupby("symbol"):
        if sym not in price.index.get_level_values(0):
            continue
        p = price.loc[sym]
        idx = p.index
        for _, ev in grp.iterrows():
            t0 = ev["minute"]
            i = idx.searchsorted(t0)
            if i >= len(idx) or idx[i] != t0:
                continue  # 発火分の足がない（昼休み跨ぎ等）
            entry = p.iloc[i]
            row = {"symbol": sym, "minute": t0, "signal_type": ev["signal_type"],
                   "direction": ev["direction"], "entry": entry}
            ok = True
            for h in HORIZONS:
                j = idx.searchsorted(t0 + pd.Timedelta(minutes=h))
                # 同一日内に決済足があること（引け跨ぎは除外）
                if j >= len(idx) or idx[j].date() != t0.date():
                    ok = False
                    break
                row[f"ret{h}"] = (p.iloc[j] / entry - 1) * 1e4 * ev["direction"]
            if ok:
                rows.append(row)
    return pd.DataFrame(rows)


def dedup(events: pd.DataFrame) -> pd.DataFrame:
    """同一銘柄×種別×方向で5分以内の再発火を捨てる。"""
    out = []
    for _, grp in events.groupby(["symbol", "signal_type", "direction"]):
        last = None
        for _, ev in grp.sort_values("event_time").iterrows():
            if last is None or (ev["event_time"] - last).total_seconds() >= COOLDOWN_MIN * 60:
                out.append(ev)
                last = ev["event_time"]
    return pd.DataFrame(out)


# ── 4. 単発イベント（方向つきのみ・間引き後） ────────────────────────────────
single = dedup(sig[sig["signal_type"].isin(DIRECTIONAL) & (sig["direction"] != 0)])
print(f"directional events after cooldown: {len(single):,}")
ev_single = forward_returns(single)

# ── 5. 合流イベント（60秒以内に2種類以上が同方向、5分クールダウン） ──────────
comp_rows = []
for (sym, d), grp in sig[sig["direction"] != 0].groupby(["symbol", "direction"]):
    grp = grp.sort_values("event_time")
    times = grp["event_time"].values
    types = grp["signal_type"].values
    last_fire = None
    for i in range(len(grp)):
        window = (times >= times[i] - np.timedelta64(60, "s")) & (times <= times[i])
        n_types = len(set(types[window]))
        if n_types >= 2:
            t = grp["event_time"].iloc[i]
            if last_fire is None or (t - last_fire).total_seconds() >= COOLDOWN_MIN * 60:
                comp_rows.append({"symbol": sym, "event_time": t, "minute": t.floor("min"),
                                  "signal_type": "composite", "direction": d, "value": d})
                last_fire = t
composite = pd.DataFrame(comp_rows)
print(f"composite events: {len(composite):,}")
ev_comp = forward_returns(composite)

# ── 6. 集計 ──────────────────────────────────────────────────────────────────
def summarize(ev: pd.DataFrame, label: str) -> pd.DataFrame:
    res = []
    for st, grp in ev.groupby("signal_type"):
        cost = grp["symbol"].map(spread).fillna(spread.median())
        for h in HORIZONS:
            gross = grp[f"ret{h}"]
            net = gross - cost
            t = gross.mean() / (gross.std() / np.sqrt(len(gross))) if len(gross) > 2 else np.nan
            res.append({"period": label, "signal": st, "horizon_min": h, "n": len(grp),
                        "gross_bp": round(gross.mean(), 2), "t_stat": round(t, 1),
                        "net_bp": round(net.mean(), 2),
                        "win_rate": round((gross > 0).mean() * 100, 1)})
    return pd.DataFrame(res)


all_ev = pd.concat([ev_single, ev_comp], ignore_index=True)
results = [summarize(all_ev, "full")]
for label, lo, hi in [("first_half", START, "2026-06-03"), ("second_half", SPLIT, END)]:
    mask = (all_ev["minute"] >= lo) & (all_ev["minute"] <= hi + " 23:59")
    results.append(summarize(all_ev[mask], label))
summary = pd.concat(results, ignore_index=True)

print("\n=== イベントスタディ結果（bp・方向符号調整済み・netはスプレッド往復控除後） ===")
print(summary.to_string(index=False))
summary.to_csv("results.csv", index=False)

# ── 7. 可視化 ────────────────────────────────────────────────────────────────
import matplotlib.font_manager as fm
for _f in ["/root/.fonts/NotoSansJP.ttf", "/mnt/c/Windows/Fonts/YuGothM.ttc",
           "/mnt/c/Windows/Fonts/meiryo.ttc"]:
    if os.path.exists(_f):
        fm.fontManager.addfont(_f)
        plt.rcParams["font.family"] = fm.FontProperties(fname=_f).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(1, 2, figsize=(12, 6.75))
full = summary[summary["period"] == "full"]
for st, grp in full.groupby("signal"):
    axes[0].plot(grp["horizon_min"], grp["gross_bp"], marker="o", label=f"{st} (gross)")
    axes[0].plot(grp["horizon_min"], grp["net_bp"], marker="x", ls="--", label=f"{st} (net)")
axes[0].axhline(0, color="gray", lw=0.8)
axes[0].set_xlabel("分")
axes[0].set_ylabel("平均リターン (bp)")
axes[0].set_title("シグナル別 前方リターンの減衰（full）")
axes[0].legend(fontsize=8)

halves = summary[summary["period"] != "full"]
piv = halves.pivot_table(index=["signal", "horizon_min"], columns="period", values="net_bp")
piv.plot(kind="bar", ax=axes[1])
axes[1].axhline(0, color="gray", lw=0.8)
axes[1].set_title("net bp の前半/後半 頑健性")
axes[1].set_ylabel("net bp")
fig.suptitle("板シグナルのイベントスタディ: コスト控除後は全構成でマイナス", fontsize=13)
fig.text(0.99, 0.01, "データ: 2026-05-22〜06-10 / aukabu.signals × JQuants 1分足 / 50銘柄",
         ha="right", va="bottom", fontsize=8, color="gray")
plt.tight_layout(rect=[0, 0.02, 1, 0.96])
fig.savefig("result.png", dpi=100, bbox_inches="tight")
print("saved result.png / results.csv")
