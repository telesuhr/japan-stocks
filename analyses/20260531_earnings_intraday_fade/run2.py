"""
決算翌日ザラ場フェード - SQL完結版（軽量）
仮説: 決算引け後発表→翌日寄り付きの大ギャップはザラ場でフェードされる
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

START = "2021-01-01"
END   = "2026-05-30"

def get_conn():
    return psycopg2.connect(**PG_CONFIG)

def load_obs():
    """SQL完結: 決算翌営業日の寄→引リターン + ギャップを一括取得"""
    conn = get_conn()
    sql = """
        WITH earn AS (
            -- 引け後発表の決算（最新発表のみ）
            SELECT DISTINCT ON (code, disc_date)
                code, disc_date
            FROM fin_summary
            WHERE disc_date >= %s AND disc_date < %s
              AND disc_time >= '15:00:00'
            ORDER BY code, disc_date, disc_time
        ),
        liquid AS (
            -- 流動性≥10億円の銘柄
            SELECT code FROM stocks_daily
            WHERE date >= %s AND date < %s
            GROUP BY code HAVING avg(turnover_value) >= 1e9
        ),
        next_trade AS (
            -- 各決算日に対して翌営業日の日足を結合
            SELECT
                e.code,
                e.disc_date,
                d.date AS trade_date,
                d.open,
                d.close,
                d.morning_open,
                LAG(d.close) OVER (PARTITION BY d.code ORDER BY d.date) AS prev_close
            FROM earn e
            JOIN liquid l ON l.code = e.code
            JOIN stocks_daily d ON d.code = e.code AND d.date > e.disc_date
            -- 翌営業日のみ: disc_dateの翌日〜disc_date+7日内で最小の日付
            WHERE d.date = (
                SELECT MIN(d2.date)
                FROM stocks_daily d2
                WHERE d2.code = e.code AND d2.date > e.disc_date AND d2.date <= e.disc_date + interval '10 days'
            )
        )
        SELECT
            code,
            disc_date,
            trade_date,
            open,
            close,
            prev_close,
            -- ギャップ = 寄り付き vs 前日終値 (bps)
            CASE WHEN prev_close > 0 THEN (open - prev_close) / prev_close * 10000 END AS gap_bps,
            -- 日中リターン = 寄→引 (bps)
            CASE WHEN open > 0 THEN (close - open) / open * 10000 END AS day_ret_bps
        FROM next_trade
        WHERE prev_close IS NOT NULL AND prev_close > 0 AND open > 0
        ORDER BY code, disc_date
    """
    print("SQLクエリ実行中...")
    df = pd.read_sql(sql, conn, params=(START, END, START, END))
    conn.close()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["disc_date"]  = pd.to_datetime(df["disc_date"])
    print(f"  {len(df):,} observations ({df['code'].nunique()} 銘柄)")
    return df

def analyze_fade(obs):
    """ギャップ分位×日中リターン（フェード検証）"""
    results = []

    def run_q(df, label, col="day_ret_bps", n_q=5):
        df = df.dropna(subset=["gap_bps", col])
        if len(df) < 20:
            return
        df = df.copy()
        df["gap_q"] = pd.qcut(df["gap_bps"].rank(method="first"), n_q, labels=False)
        for q in range(n_q):
            sub = df[df["gap_q"] == q][col]
            n = len(sub)
            mu = sub.mean()
            se = sub.std() / np.sqrt(n) if n > 1 else np.nan
            t  = mu / se if se and se > 0 else np.nan
            wr = (sub > 0).mean() * 100
            results.append({"label": label, "gap_q": q, "n": n,
                            "mean_bps": mu, "t": t, "wr": wr})

    # 全期間
    run_q(obs, "全期間")
    # IS/OOS
    run_q(obs[obs["trade_date"] < "2024-01-01"], "IS(<2024)")
    run_q(obs[obs["trade_date"] >= "2024-01-01"], "OOS(2024+)")
    # 大ギャップ
    run_q(obs[obs["gap_bps"] >= 150],  "大ギャップ上昇(≥150bps)")
    run_q(obs[obs["gap_bps"] <= -150], "大ギャップ下落(≤-150bps)")
    # gap vs morning (前場のみ)
    if "morning_ret" in obs.columns:
        run_q(obs, "前場リターン", "morning_ret")

    return pd.DataFrame(results)

def main():
    obs = load_obs()
    obs.to_csv("earnings_obs.csv", index=False)

    # 前場リターン（morning_open使用）
    if "morning_open" in obs.columns:
        obs["morning_ret"] = np.where(obs["morning_open"] > 0,
            (obs["close"] - obs["morning_open"]) / obs["morning_open"] * 10000, np.nan)

    res = analyze_fade(obs)
    res.to_csv("fade_summary.csv", index=False)
    print(f"集計完了: {len(res)} rows")

    # ---- グラフ ----
    labels_to_plot = ["全期間", "IS(<2024)", "OOS(2024+)", "大ギャップ上昇(≥150bps)", "大ギャップ下落(≤-150bps)"]
    fig, axes = plt.subplots(1, len(labels_to_plot), figsize=(18, 5), facecolor="white")

    q_labels = ["Q1\n(gap最小)", "Q2", "Q3", "Q4", "Q5\n(gap最大)"]
    colors   = ["#b71c1c","#ef5350","#90a4ae","#66bb6a","#1b5e20"]

    for ax, lbl in zip(axes, labels_to_plot):
        sub = res[res["label"] == lbl].sort_values("gap_q")
        if sub.empty:
            ax.text(0.5, 0.5, "データ不足", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(lbl, fontsize=9)
            continue
        bars = ax.bar(sub["gap_q"], sub["mean_bps"], color=colors[:len(sub)], alpha=0.85)
        for bar, t_val, mu in zip(bars, sub["t"], sub["mean_bps"]):
            if abs(t_val) >= 2.0:
                ax.text(bar.get_x()+bar.get_width()/2,
                        mu+(0.5 if mu>=0 else -2), "*", ha="center", fontsize=14)
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_title(lbl, fontsize=9)
        ax.text(0.98, 0.98, f"N={int(sub['n'].sum())}", transform=ax.transAxes,
                ha="right", va="top", fontsize=8, color="gray")
        ax.set_xticks(range(5))
        ax.set_xticklabels(q_labels, fontsize=7)
        ax.set_ylabel("日中リターン bps (寄→引)", fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("決算翌日: 寄付きギャップ分位 × ザラ場リターン（*=|t|≥2, 往復コスト未控除）",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig("result.png", dpi=100, bbox_inches="tight", facecolor="white")
    print("result.png 保存完了")

    print("\n=== Q1(gap最小) vs Q5(gap最大) ===")
    for lbl in labels_to_plot:
        sub = res[res["label"] == lbl].sort_values("gap_q")
        if len(sub) < 5:
            continue
        q1 = sub.iloc[0]
        q5 = sub.iloc[4]
        ls = q5["mean_bps"] - q1["mean_bps"]   # momentum spread (gap追随)
        fade = q1["mean_bps"] - q5["mean_bps"]  # fade spread (逆張り)
        print(f"  {lbl:30s}: Q1={q1['mean_bps']:+.1f}bps(t={q1['t']:.2f})  "
              f"Q5={q5['mean_bps']:+.1f}bps(t={q5['t']:.2f})  "
              f"momentum={ls:+.1f}  fade={fade:+.1f}")

if __name__ == "__main__":
    main()
