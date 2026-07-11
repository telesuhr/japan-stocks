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

COST_BPS = 10  # 往復コスト
CODES = {"80350": "TEL(8035)", "40630": "信越化学(4063)"}
ADR_MAP = {"80350": "ADR_8035", "40630": "ADR_4063"}

# ─── データ取得 ──────────────────────────────────────────────────────────────
print("DB接続中...")
conn = psycopg2.connect(**PG_CONFIG)

# ADR / SMH
adr_sql = """
    SELECT symbol, trade_date, close
    FROM macro.daily_ohlcv
    WHERE symbol IN ('ADR_8035','ADR_4063','SMH','.SOX')
      AND trade_date >= '2020-01-01'
    ORDER BY trade_date
"""
adr_raw = pd.read_sql(adr_sql, conn, parse_dates=["trade_date"])
adr = adr_raw.pivot(index="trade_date", columns="symbol", values="close").sort_index()
adr_ret = adr.pct_change()

# 日足（open / close / prev_close）
daily_sql = """
    SELECT code, date, open, close, adj_close
    FROM stocks_daily
    WHERE code IN ('80350','40630') AND date >= '2020-01-01'
    ORDER BY date
"""
daily_raw = pd.read_sql(daily_sql, conn, parse_dates=["date"])

# イントラ（分足）
intra_sql = """
    SELECT code, ts, close
    FROM stocks_intraday
    WHERE code IN ('80350','40630') AND ts >= '2024-05-10'
    ORDER BY ts
"""
intra_raw = pd.read_sql(intra_sql, conn, parse_dates=["ts"])
conn.close()
print("データ取得完了")

# ─── アライン: US date → JP次営業日 ────────────────────────────────────────
us_dates = pd.DatetimeIndex(adr.index)
jp_all_dates = pd.DatetimeIndex(sorted(daily_raw["date"].unique()))

def next_jp_day(d_us, jp_dates):
    after = jp_dates[jp_dates > d_us]
    return after[0] if len(after) else pd.NaT

# ─── Part1: 日足分析（ギャップ + 日中方向）─────────────────────────────────
print("\n=== Part1: ギャップ予測 & 日中方向（日足 2020-2026） ===")

results_daily = {}
for code, name in CODES.items():
    df = daily_raw[daily_raw.code == code].set_index("date").sort_index()
    df["prev_close"] = df["close"].shift(1)
    df["gap"]        = (df["open"] - df["prev_close"]) / df["prev_close"]
    df["intra_ret"]  = (df["close"] - df["open"]) / df["open"]   # 日中: open→close
    df["daily_ret"]  = (df["close"] - df["prev_close"]) / df["prev_close"]

    adr_col = ADR_MAP[code]
    rows = []
    for d_us in us_dates:
        d_jp = next_jp_day(d_us, jp_all_dates)
        if pd.isna(d_jp) or d_jp not in df.index:
            continue
        adr_r = adr_ret.loc[d_us, adr_col] if d_us in adr_ret.index else np.nan
        smh_r = adr_ret.loc[d_us, "SMH"]   if d_us in adr_ret.index else np.nan
        if pd.isna(adr_r):
            continue
        rows.append(dict(
            d_us=d_us, d_jp=d_jp,
            adr_ret=adr_r, smh_ret=smh_r,
            gap=df.loc[d_jp, "gap"],
            intra_ret=df.loc[d_jp, "intra_ret"],
            daily_ret=df.loc[d_jp, "daily_ret"],
        ))
    df2 = pd.DataFrame(rows).dropna()
    results_daily[code] = df2

    # ADR → gap の相関
    r_ag, p_ag = stats.pearsonr(df2["adr_ret"], df2["gap"])
    # gap → 日中方向の相関
    r_gi, p_gi = stats.pearsonr(df2["gap"], df2["intra_ret"])
    # ADR → 日中方向
    r_ai, p_ai = stats.pearsonr(df2["adr_ret"], df2["intra_ret"])

    print(f"\n[{name}] n={len(df2)}")
    print(f"  ADR→ギャップ:     r={r_ag:.3f} p={p_ag:.4f} (ADRでギャップを予測)")
    print(f"  ギャップ→日中方向: r={r_gi:.3f} p={p_gi:.4f} (ギャップが継続するか)")
    print(f"  ADR→日中方向:     r={r_ai:.3f} p={p_ai:.4f} (ADRで日中方向を予測)")

    # ビン別: ADR幅 × 日中リターン平均
    df2["adr_bin"] = pd.cut(df2["adr_ret"]*100,
                            bins=[-np.inf, -3, -1.5, -0.5, 0.5, 1.5, 3, np.inf],
                            labels=["<-3%","-3~-1.5%","-1.5~-0.5%","±0.5%","+0.5~+1.5%","+1.5~+3%",">+3%"])
    bin_stats = df2.groupby("adr_bin", observed=True).agg(
        n=("intra_ret","count"),
        gap_mean=("gap", lambda x: x.mean()*100),
        intra_mean=("intra_ret", lambda x: x.mean()*100),
        intra_std=("intra_ret", lambda x: x.std()*100),
    )
    print(f"\n  ADRビン別 (gap%, 日中%):"); print(bin_stats.round(3))

# ─── Part2: イントラ保有期間分析 ─────────────────────────────────────────
print("\n\n=== Part2: 保有期間別リターン（イントラ 2024-2026） ===")

EXIT_TIMES = ["09:30", "10:00", "11:00", "11:30", "13:30", "15:00"]

# 各コード×各時刻の終値を一括ピボット（ベクトル化で高速化）
intra_raw["date_col"] = intra_raw["ts"].dt.date
intra_raw["time_col"] = intra_raw["ts"].dt.time

def build_exit_pivot(intra_sub, exit_times):
    """各日×各終了時刻の価格を事前計算してピボットテーブルに"""
    pivots = {}
    for t in exit_times:
        t_obj = pd.Timestamp(f"2000-01-01 {t}").time()
        # その日の t 以降の最初のバーの終値
        after = intra_sub[intra_sub["time_col"] >= t_obj].copy()
        # 日付ごとに最初の行を取得
        first = after.groupby("date_col")["close"].first()
        pivots[t] = first
    return pd.DataFrame(pivots)  # index=date

intra_dates = pd.DatetimeIndex(sorted(intra_raw["ts"].dt.date.unique())).map(pd.Timestamp)

results_intra = {}
for code, name in CODES.items():
    df_d = daily_raw[daily_raw.code == code].set_index("date").sort_index()
    intra_sub = intra_raw[intra_raw.code == code]
    exit_pivot = build_exit_pivot(intra_sub, EXIT_TIMES)  # index=date(date型)

    rows = []
    for d_us in us_dates:
        d_jp = next_jp_day(d_us, intra_dates)
        if pd.isna(d_jp) or d_jp not in df_d.index:
            continue
        adr_r = adr_ret.loc[d_us, ADR_MAP[code]] if d_us in adr_ret.index else np.nan
        if pd.isna(adr_r):
            continue
        entry = df_d.loc[d_jp, "open"]
        if pd.isna(entry) or entry == 0:
            continue
        d_jp_date = d_jp.date()
        if d_jp_date not in exit_pivot.index:
            continue
        row = {"d_jp": d_jp, "adr_ret": adr_r, "entry": entry}
        for t in EXIT_TIMES:
            exit_p = exit_pivot.loc[d_jp_date, t]
            row[f"ret_{t.replace(':','')}"] = (exit_p / entry - 1) if not pd.isna(exit_p) else np.nan
        rows.append(row)

    df3 = pd.DataFrame(rows).dropna(subset=["ret_0930"])
    results_intra[code] = df3
    print(f"\n[{name}] n={len(df3)}")

    # ADRビン × 保有期間別平均（コスト控除後）
    cost = COST_BPS / 10000
    df3["adr_bin"] = pd.cut(df3["adr_ret"]*100,
                            bins=[-np.inf, -1.5, -0.5, 0.5, 1.5, np.inf],
                            labels=["<-1.5%","-1.5~-0.5%","±0.5%","+0.5~+1.5%",">+1.5%"])
    for t in EXIT_TIMES:
        col = f"ret_{t.replace(':','')}"
        # ADRと同方向でロング（ADR>0→ロング, ADR<0→ショート）
        df3[f"dir_{col}"] = df3[col] * np.sign(df3["adr_ret"]) - cost
    dir_cols = [f"dir_ret_{t.replace(':','')}" for t in EXIT_TIMES]
    bin_mean = df3.groupby("adr_bin", observed=True)[dir_cols].mean() * 100
    bin_mean.columns = EXIT_TIMES
    bin_n = df3.groupby("adr_bin", observed=True).size()
    print(f"  ADRビン×保有期間 コスト後方向性リターン(%):")
    print(pd.concat([bin_n.rename("n"), bin_mean], axis=1).round(3))

    # 全体サマリ: ADR方向ロング/ショート のコスト後リターン
    print(f"\n  ADR方向ロング/ショート 平均 (コスト後):")
    for t in EXIT_TIMES:
        col = f"dir_ret_{t.replace(':','')}"
        vals = df3[col].dropna()
        mean_v = vals.mean() * 100
        tval, pval = stats.ttest_1samp(vals, 0)
        pos_rate = (vals > 0).mean() * 100
        print(f"    ~{t}: {mean_v:+.3f}%  t={tval:.2f} p={pval:.3f}  勝率={pos_rate:.0f}%  n={len(vals)}")

# ─── 可視化 ──────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(12, 6.75), facecolor="white")
fig.suptitle("Semi ADR Lead -> Japan Intraday: Gap & Direction (TEL 8035)",
             fontsize=13, fontweight="bold", y=0.99)
axes = fig.subplots(1, 3)

code = "80350"
df2 = results_daily[code]
df3 = results_intra[code]

# 1. ADR vs Gap 散布図
ax = axes[0]
ax.scatter(df2["adr_ret"]*100, df2["gap"]*100, alpha=0.3, s=10, color="#4C9BE8")
m, b = np.polyfit(df2["adr_ret"]*100, df2["gap"]*100, 1)
xs = np.linspace(df2["adr_ret"].min()*100, df2["adr_ret"].max()*100, 100)
ax.plot(xs, m*xs+b, color="red", lw=1.5)
r_ag, _ = stats.pearsonr(df2["adr_ret"], df2["gap"])
ax.set_xlabel("ADR_8035 return (%)")
ax.set_ylabel("Next-day Japan gap (%)")
ax.set_title(f"ADR -> Gap  r={r_ag:.3f}", fontsize=11)
ax.axhline(0, color="gray", lw=0.5, ls="--")
ax.axvline(0, color="gray", lw=0.5, ls="--")

# 2. Gap vs Intraday direction
ax = axes[1]
ax.scatter(df2["gap"]*100, df2["intra_ret"]*100, alpha=0.3, s=10, color="#E87D4C")
m2, b2 = np.polyfit(df2["gap"]*100, df2["intra_ret"]*100, 1)
xs2 = np.linspace(df2["gap"].min()*100, df2["gap"].max()*100, 100)
ax.plot(xs2, m2*xs2+b2, color="red", lw=1.5)
r_gi, _ = stats.pearsonr(df2["gap"], df2["intra_ret"])
ax.set_xlabel("Open gap (%)")
ax.set_ylabel("Intraday return open->close (%)")
ax.set_title(f"Gap -> Intraday  r={r_gi:.3f}", fontsize=11)
ax.axhline(0, color="gray", lw=0.5, ls="--")
ax.axvline(0, color="gray", lw=0.5, ls="--")

# 3. 保有期間別コスト後リターン（ADR方向ロング）
ax = axes[2]
cost = COST_BPS / 10000
hold_means = []
hold_ts = []
hold_ps = []
for t in EXIT_TIMES:
    col = f"dir_ret_{t.replace(':','')}"
    if col in df3.columns:
        vals = df3[col].dropna()
        hold_means.append(vals.mean()*100)
        tval, pval = stats.ttest_1samp(vals, 0)
        hold_ts.append(tval)
        hold_ps.append(pval)
        hold_ts.append(tval)

colors_h = ["#4C9BE8" if v >= 0 else "#E87D4C" for v in hold_means]
bars = ax.bar(EXIT_TIMES, hold_means, color=colors_h)
ax.axhline(0, color="gray", lw=0.8, ls="--")
ax.set_title("ADR-dir Long/Short by hold time\n(cost -10bp, TEL, 2024-)", fontsize=10)
ax.set_ylabel("Avg net return (%)")
ax.set_xlabel("Exit time")
for b, v, p in zip(bars, hold_means, hold_ps):
    mark = "*" if p < 0.05 else ("~" if p < 0.10 else "")
    ax.text(b.get_x()+b.get_width()/2,
            v + (0.01 if v >= 0 else -0.03),
            f"{v:+.3f}%{mark}",
            ha="center", va="bottom" if v >= 0 else "top", fontsize=8)

fig.text(0.99, 0.01, "Data: macro.daily_ohlcv ADR_8035 + stocks_daily/intraday 80350 (2020/2024-2026)",
         ha="right", va="bottom", fontsize=7, color="gray")
plt.tight_layout(rect=[0, 0.02, 1, 0.97])

out = Path(__file__).parent / "result.png"
fig.savefig(out, dpi=100, bbox_inches="tight", facecolor="white")
print(f"\nsaved {out}")
