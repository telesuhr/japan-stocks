"""
前場→後場 連続性分析（ランチブレークパターン）
仮説: 前場の方向性が後場に継続するか、リバーサルするか
- 前場リターン(9:00-11:30)の強弱で5分位に分けて後場(12:30-15:30)リターンを測定
- LME上昇日/下落日/全日 の条件別
- 非鉄8銘柄 + 半導体14銘柄
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import os
import psycopg2
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

PG_CONFIG = {
    "host": os.environ.get("PGHOST", "omen"),
    "port": int(os.environ.get("PGPORT", 5432)),
    "user": os.environ.get("PGUSER", "postgres"),
    "dbname": os.environ.get("PGDATABASE", "market_data"),
}

NONFER = ["57060","57110","57130","57140","50160","58010","58020","58030"]
SEMI   = ["69201","69541","68572","68472","30346","30350","76510",
          "79560","60458","28572","80358","68450","64521","285A0"]
ALL_CODES = NONFER + SEMI
SECTOR = {c:"非鉄" for c in NONFER}
SECTOR.update({c:"半導体" for c in SEMI})

START = "2024-05-01"
END   = "2026-05-30"

def get_conn():
    return psycopg2.connect(**PG_CONFIG)

def load_intraday(codes, start, end):
    conn = get_conn()
    sql = """
        SELECT code, ts, open, high, low, close, volume
        FROM stocks_intraday
        WHERE code = ANY(%s) AND ts >= %s AND ts < %s
        ORDER BY code, ts
    """
    df = pd.read_sql(sql, conn, params=(codes, start, end))
    conn.close()
    df["ts"] = pd.to_datetime(df["ts"])
    return df

def load_lme(start, end):
    conn = get_conn()
    sql = """
        SELECT trade_date AS date, close
        FROM macro.daily_ohlcv
        WHERE symbol = 'Cc1' AND trade_date >= %s AND trade_date < %s
        ORDER BY trade_date
    """
    df = pd.read_sql(sql, conn, params=(start, end))
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    df["lme_ret"] = df["close"].pct_change() * 100
    return df.set_index("date")

def period_return(grp, t_from_str, t_to_str):
    """grp: 当日の1分足DF、t_from_str/t_to_str: "HH:MM" JST"""
    import datetime
    t_from = datetime.time(*map(int, t_from_str.split(":")))
    t_to   = datetime.time(*map(int, t_to_str.split(":")))
    sub = grp[(grp["ts"].dt.time >= t_from) & (grp["ts"].dt.time < t_to)]
    if len(sub) < 2:
        return np.nan
    entry = sub.iloc[0]["open"]
    ex    = sub.iloc[-1]["close"]
    if entry <= 0:
        return np.nan
    return (ex / entry - 1) * 10000  # bps

def summarize(series):
    n  = len(series.dropna())
    mu = series.mean()
    se = series.std() / np.sqrt(n) if n > 1 else np.nan
    t  = mu / se if se and se > 0 else np.nan
    wr = (series > 0).mean() * 100
    return n, mu, t, wr

def main():
    print("データ読み込み...")
    df = load_intraday(ALL_CODES, START, END)
    print(f"  {len(df):,} rows ({df['code'].nunique()} codes)")
    df["ts"] = pd.to_datetime(df["ts"])

    lme = load_lme(START, END)

    # 日単位で前場・後場リターン計算
    records = []
    for (code, date), grp in df.groupby([df["code"], df["ts"].dt.date]):
        grp = grp.sort_values("ts")
        am  = period_return(grp, "09:00", "11:30")
        pm  = period_return(grp, "12:30", "15:30")
        pm1 = period_return(grp, "12:30", "13:30")  # 後場前半
        pm2 = period_return(grp, "13:30", "15:30")  # 後場後半
        if pd.isna(am) or pd.isna(pm):
            continue
        date_ts = pd.Timestamp(date)
        lme_ret = lme["lme_ret"].get(date_ts, np.nan)
        records.append({
            "code": code, "date": date, "date_ts": date_ts,
            "am_ret": am, "pm_ret": pm, "pm1_ret": pm1, "pm2_ret": pm2,
            "lme_ret": lme_ret,
            "sector": SECTOR.get(code, "OTHER"),
            "dow": date_ts.dayofweek,
        })

    panel = pd.DataFrame(records)
    print(f"  panel: {len(panel):,} code-days")

    # 前場リターンを各日×セクター内で5分位に分割（銘柄横断でクロスセクション）
    # ここでは銘柄ごとに分位（時系列の分位）
    panel["am_q5"] = panel.groupby("code")["am_ret"].transform(
        lambda x: pd.qcut(x.rank(method="first"), 5, labels=False))

    panel.to_csv("panel.csv", index=False)

    # ---- 集計 ----
    results = []

    def run_analysis(sub, label, target="pm_ret"):
        for q in range(5):
            grp = sub[sub["am_q5"] == q][target].dropna()
            n, mu, t, wr = summarize(grp)
            results.append({
                "label": label, "am_q5": q, "target": target,
                "n": n, "mean_bps": mu, "t": t, "wr": wr
            })

    for sector in ["非鉄", "半導体", "ALL"]:
        sub = panel if sector == "ALL" else panel[panel["sector"] == sector]

        # 全日
        run_analysis(sub, f"{sector}_全日", "pm_ret")
        run_analysis(sub, f"{sector}_全日_pm1", "pm1_ret")

        # LME条件
        run_analysis(sub[sub["lme_ret"] >= 1.0], f"{sector}_LME+1%", "pm_ret")
        run_analysis(sub[sub["lme_ret"] <= -1.0], f"{sector}_LME-1%", "pm_ret")

        # 曜日
        for dow, name in [(0,"月"),(1,"火"),(2,"水"),(3,"木"),(4,"金")]:
            run_analysis(sub[sub["dow"] == dow], f"{sector}_曜日_{name}", "pm_ret")

    res_df = pd.DataFrame(results)
    res_df.to_csv("lunch_summary.csv", index=False)
    print(f"  集計完了 -> lunch_summary.csv ({len(res_df)} rows)")

    # ---- グラフ ----
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), facecolor="white")
    labels_order = ["非鉄_全日", "半導体_全日", "非鉄_LME+1%", "非鉄_LME-1%", "非鉄_曜日_火", "非鉄_曜日_木"]
    titles       = ["非鉄 全日", "半導体 全日", "非鉄 LME+1%日", "非鉄 LME-1%日", "非鉄 火曜日", "非鉄 木曜日"]

    q_labels = ["Q1\n(前場弱)", "Q2", "Q3", "Q4", "Q5\n(前場強)"]
    colors = ["#e53935","#ef9a9a","#90a4ae","#80cbc4","#00897b"]

    for ax, lbl, title in zip(axes.flatten(), labels_order, titles):
        sub = res_df[(res_df["label"] == lbl) & (res_df["target"] == "pm_ret")]
        if sub.empty:
            ax.set_visible(False)
            continue
        sub = sub.sort_values("am_q5")
        bars = ax.bar(sub["am_q5"], sub["mean_bps"], color=colors, alpha=0.85)
        for bar, t_val, mu in zip(bars, sub["t"], sub["mean_bps"]):
            if abs(t_val) >= 2.0:
                ax.text(bar.get_x() + bar.get_width()/2,
                        mu + (0.5 if mu >= 0 else -1.5),
                        "*", ha="center", fontsize=12, color="black")
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(range(5))
        ax.set_xticklabels(q_labels, fontsize=8)
        ax.set_ylabel("後場リターン (bps)", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        # N を legend代わりに
        ns = sub["n"].tolist()
        ax.text(0.98, 0.02, f"N={min(ns)}〜{max(ns)}/Q", transform=ax.transAxes,
                ha="right", fontsize=7, color="gray")

    fig.suptitle("前場分位 → 後場リターン（ランチブレーク連続性, 2024/05〜2026/05, * = |t|≥2）",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig("result.png", dpi=100, bbox_inches="tight", facecolor="white")
    print("result.png 保存完了")

    # 注目エッジ
    print("\n=== Q1(前場弱)・Q5(前場強) の後場リターン比較 ===")
    for lbl in labels_order:
        sub = res_df[(res_df["label"] == lbl) & (res_df["target"] == "pm_ret")]
        if sub.empty:
            continue
        q1 = sub[sub["am_q5"] == 0].iloc[0]
        q5 = sub[sub["am_q5"] == 4].iloc[0]
        spread = q5["mean_bps"] - q1["mean_bps"]
        print(f"  {lbl:25s}  Q1={q1['mean_bps']:+.1f}bps(t={q1['t']:.2f})  Q5={q5['mean_bps']:+.1f}bps(t={q5['t']:.2f})  spread={spread:+.1f}bps")

if __name__ == "__main__":
    main()
