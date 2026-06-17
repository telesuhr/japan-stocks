import os, sys
sys.stdout.reconfigure(line_buffering=True)
import psycopg2, pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

PG_CONFIG = {
    "host": os.environ.get("PGHOST", "omen"),
    "port": int(os.environ.get("PGPORT", 5432)),
    "user": os.environ.get("PGUSER", "postgres"),
    "dbname": os.environ.get("PGDATABASE", "market_data"),
}

# ── データ取得 ──────────────────────────────────────────────────────────────
print("DB接続中...")
conn = psycopg2.connect(**PG_CONFIG)

# 米国: ESc1 (E-mini S&P500先物)
us_sql = """
    SELECT trade_date, close
    FROM macro.daily_ohlcv
    WHERE symbol = 'ESc1'
      AND trade_date >= '2016-01-01'
    ORDER BY trade_date
"""
us = pd.read_sql(us_sql, conn, parse_dates=["trade_date"]).set_index("trade_date")
us["ret_us"] = us["close"].pct_change()

# 日本: TOPIX (0000) + 日経225 (N225)
jp_sql = """
    SELECT date, code, close
    FROM index_daily
    WHERE code IN ('0000', 'N225')
      AND date >= '2016-01-01'
    ORDER BY date
"""
jp_raw = pd.read_sql(jp_sql, conn, parse_dates=["date"])
conn.close()

jp = jp_raw.pivot(index="date", columns="code", values="close")
jp.columns = ["topix", "nk225"]
jp["ret_topix"] = jp["topix"].pct_change()
jp["ret_nk225"] = jp["nk225"].pct_change()

print(f"ESc1: {us.index.min().date()} ~ {us.index.max().date()} ({len(us)}営業日)")
print(f"TOPIX: {jp.index.min().date()} ~ {jp.index.max().date()} ({len(jp)}営業日)")

# ── US→JP アラインメント ────────────────────────────────────────────────────
# 各日本営業日 D_jp に対し、直前の米国営業日 D_us を特定
us_dates = us.index.sort_values()
jp_dates = jp.index.sort_values()

rows = []
for i, d_jp1 in enumerate(jp_dates[1:-1]):  # 最初と最後を除く
    # D_jp1 より前の最後の米国営業日
    prev_us = us_dates[us_dates < d_jp1]
    if len(prev_us) == 0:
        continue
    d_us = prev_us[-1]

    # D_jp2 = D_jp1 の翌日本営業日
    next_jp = jp_dates[jp_dates > d_jp1]
    if len(next_jp) == 0:
        continue
    d_jp2 = next_jp[0]

    ret_us   = us.loc[d_us, "ret_us"]
    ret_jp1  = jp.loc[d_jp1, "ret_topix"]
    ret_jp2  = jp.loc[d_jp2, "ret_topix"]
    ret_nk1  = jp.loc[d_jp1, "ret_nk225"]
    ret_nk2  = jp.loc[d_jp2, "ret_nk225"]

    if pd.isna(ret_us) or pd.isna(ret_jp1) or pd.isna(ret_jp2):
        continue

    rows.append({
        "d_us": d_us, "d_jp1": d_jp1, "d_jp2": d_jp2,
        "ret_us": ret_us,
        "ret_jp1": ret_jp1, "ret_jp2": ret_jp2,
        "ret_nk1": ret_nk1, "ret_nk2": ret_nk2,
    })

df = pd.DataFrame(rows)
print(f"\nアライン済みサンプル: {len(df)} 件")

# ── シグナル定義 ────────────────────────────────────────────────────────────
# 各閾値でのパターン: 米国下落 < thresh かつ 日本翌日上昇 > 0
thresholds = [0.0, -0.005, -0.010, -0.020]
thresh_labels = ["米国↓(any)", "米国↓>0.5%", "米国↓>1%", "米国↓>2%"]

results = []
for thresh, label in zip(thresholds, thresh_labels):
    mask = (df["ret_us"] < thresh) & (df["ret_jp1"] > 0)
    sub = df[mask]
    n = len(sub)
    if n == 0:
        continue
    ret2 = sub["ret_jp2"]
    mean2 = ret2.mean() * 100
    std2  = ret2.std() * 100
    pos_rate = (ret2 > 0).mean() * 100
    # ベンチマーク: 同期間の全 D_jp2 リターン
    bench = df.loc[~mask, "ret_jp2"]
    t, p = stats.ttest_ind(ret2, bench, equal_var=False)
    results.append(dict(
        label=label, n=n, mean2=mean2, std2=std2, pos_rate=pos_rate,
        bench_mean=bench.mean()*100, t=t, p=p
    ))
    print(f"\n[{label}] n={n}")
    print(f"  翌日TOPIX: mean={mean2:+.3f}% std={std2:.2f}% 勝率={pos_rate:.0f}%")
    print(f"  ベンチ:     mean={bench.mean()*100:+.3f}%  t={t:.2f} p={p:.3f}")

res_df = pd.DataFrame(results)

# ── 追加分析: 米国下落幅別の翌々日リターン分布 ─────────────────────────────
mask_signal = (df["ret_us"] < 0) & (df["ret_jp1"] > 0)
sig = df[mask_signal].copy()
non_sig = df[~mask_signal].copy()

# 米国下落幅でビン分け
sig["us_bin"] = pd.cut(sig["ret_us"]*100,
                       bins=[-np.inf, -2, -1, -0.5, 0],
                       labels=["<-2%", "-2~-1%", "-1~-0.5%", "-0.5~0%"])

bin_stats = sig.groupby("us_bin", observed=True)["ret_jp2"].agg(
    n="count", mean=lambda x: x.mean()*100, std=lambda x: x.std()*100,
    pos_rate=lambda x: (x>0).mean()*100
)
print("\n=== 米国下落幅別 翌々日TOPIX ===")
print(bin_stats.round(3))

# ── 可視化 ─────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(12, 6.75), facecolor="white")
fig.suptitle("US Down -> JP Bounce -> JP Next Day (TOPIX, 2016-2026)",
             fontsize=14, fontweight="bold", y=0.98)

axes = fig.subplots(1, 3)

# 1. 閾値別 翌々日平均リターン
ax = axes[0]
colors = ["#4C9BE8" if v >= 0 else "#E87D4C" for v in res_df["mean2"]]
bars = ax.bar(range(len(res_df)), res_df["mean2"], color=colors)
ax.axhline(0, color="gray", lw=0.8, ls="--")
ax.set_xticks(range(len(res_df)))
ax.set_xticklabels(res_df["label"], rotation=15, ha="right", fontsize=8)
ax.set_ylabel("Next-day TOPIX return (%)")
ax.set_title("Threshold vs D+2 return", fontsize=11)
for b, row in zip(bars, res_df.itertuples()):
    sig_mark = "*" if row.p < 0.05 else ("~" if row.p < 0.10 else "")
    ax.text(b.get_x()+b.get_width()/2,
            row.mean2 + (0.02 if row.mean2 >= 0 else -0.04),
            f"{row.mean2:+.3f}%{sig_mark}\nn={row.n}",
            ha="center", va="bottom" if row.mean2 >= 0 else "top", fontsize=8)

# 2. 米国下落幅別 箱ひげ
ax = axes[1]
bin_data = [sig[sig["us_bin"]==b]["ret_jp2"].dropna()*100
            for b in ["<-2%", "-2~-1%", "-1~-0.5%", "-0.5~0%"]]
bp = ax.boxplot(bin_data, labels=["<-2%","-2~-1%","-1~-0.5%","-0.5~0%"],
                patch_artist=True, medianprops=dict(color="red", lw=2))
for box in bp["boxes"]:
    box.set_facecolor("#A8D5E2")
ax.axhline(0, color="gray", lw=0.8, ls="--")
ax.set_title("US drop magnitude vs D+2 TOPIX", fontsize=11)
ax.set_ylabel("D+2 TOPIX return (%)")
ax.set_xlabel("US drop (%)")
for i, (b, data) in enumerate(zip(["<-2%","-2~-1%","-1~-0.5%","-0.5~0%"], bin_data)):
    n = len(data)
    ax.text(i+1, ax.get_ylim()[0]*0.9, f"n={n}", ha="center", fontsize=8, color="gray")

# 3. 累積 (シグナル日のD+2 vs 通常日のD+1)
ax = axes[2]
mask_base = (df["ret_us"] < -0.005) & (df["ret_jp1"] > 0)
sig05 = df[mask_base].sort_values("d_jp2")
non05 = df[~mask_base].sort_values("d_jp2")

cum_sig  = (1 + sig05["ret_jp2"]).cumprod() - 1
cum_non  = (1 + non05["ret_jp2"].iloc[:len(sig05)]).cumprod() - 1

ax.plot(range(len(cum_sig)), cum_sig.values*100, color="#4C9BE8", lw=1.5,
        label=f"Signal(US<-0.5%+JP+): n={len(sig05)}")
ax.plot(range(len(cum_non)), cum_non.values*100, color="#A0A0A0", lw=1, ls="--",
        label="Non-signal (D+1)")
ax.axhline(0, color="gray", lw=0.8, ls="--")
ax.set_title("Cumulative D+2 return (TOPIX, US<-0.5%)", fontsize=10)
ax.set_xlabel("Trade #")
ax.set_ylabel("Cumulative return (%)")
ax.legend(fontsize=8)

fig.text(0.99, 0.01, "Data: macro.daily_ohlcv ESc1 + index_daily TOPIX (2016-2026)",
         ha="right", va="bottom", fontsize=8, color="gray")
plt.tight_layout(rect=[0, 0.03, 1, 0.96])

out = Path(__file__).parent / "result.png"
fig.savefig(out, dpi=100, bbox_inches="tight", facecolor="white")
print(f"\nsaved {out}")

# ── サマリ ──────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
for row in res_df.itertuples():
    sig_mark = "★有意" if row.p < 0.05 else ("△弱い傾向" if row.p < 0.10 else "—非有意")
    print(f"{row.label}: D+2={row.mean2:+.3f}% vs ベンチ{row.bench_mean:+.3f}%  "
          f"t={row.t:.2f} p={row.p:.3f}  勝率={row.pos_rate:.0f}%  → {sig_mark}")
