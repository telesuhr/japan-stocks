"""
前日大幅変動銘柄 → 翌日ギャップ & 日中リターン（全ユニバース・日足10年）
仮説: 前日±X%以上動いた銘柄は翌日に特定のパターン（フェードorモメンタム）がある
外部シグナル不要・純粋価格パターン
期間: 2016-05〜2026-05（IS: 2016-2020, OOS: 2021-2026）
コスト: 往復10bps（寄成/引成前提）
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

START    = "2016-05-01"
END      = "2026-05-31"
IS_END   = "2021-01-01"
OOS_ST   = "2021-01-01"
COST     = 10.0  # 往復 bps
LIQ_MIN  = 1e9   # 10億円/日

def get_conn():
    return psycopg2.connect(**PG_CONFIG)

def load_universe():
    """前日変動×翌日リターンのパネルをSQL完結で生成"""
    conn = get_conn()
    print("  SQLクエリ実行中（数分）...")
    sql = """
        WITH liq AS (
            SELECT code
            FROM stocks_daily
            WHERE date >= %s AND date <= %s
            GROUP BY code
            HAVING avg(turnover_value) >= %s
        ),
        -- Step1: 各日の終値リターンを計算
        day_ret_calc AS (
            SELECT
                d.code, d.date,
                d.adj_open, d.adj_close, d.turnover_value,
                LAG(d.adj_close) OVER (PARTITION BY d.code ORDER BY d.date) AS prev_close,
                -- 当日の full return (close vs prev_close)
                (d.adj_close - LAG(d.adj_close) OVER (PARTITION BY d.code ORDER BY d.date))
                  / NULLIF(LAG(d.adj_close) OVER (PARTITION BY d.code ORDER BY d.date), 0) * 100
                    AS full_ret_pct
            FROM stocks_daily d
            JOIN liq USING(code)
            WHERE d.date >= %s AND d.date <= %s
        ),
        -- Step2: 前日のリターンをシグナルとして付与
        with_signal AS (
            SELECT *,
                -- シグナル = 前日の full_ret_pct
                LAG(full_ret_pct) OVER (PARTITION BY code ORDER BY date) AS prev_ret_pct,
                -- 今日の寄→引
                (adj_close - adj_open) / NULLIF(adj_open, 0) * 10000 AS day_ret_bps,
                -- 今日のgap = 寄 vs 前日引
                (adj_open - prev_close) / NULLIF(prev_close, 0) * 10000 AS gap_bps
            FROM day_ret_calc
        ),
        with_mkt AS (
            SELECT *,
                AVG(full_ret_pct) OVER (PARTITION BY date) AS mkt_ret
            FROM with_signal
        )
        SELECT code, date, prev_ret_pct, day_ret_bps, gap_bps, mkt_ret, turnover_value
        FROM with_mkt
        WHERE prev_ret_pct IS NOT NULL
          AND prev_close IS NOT NULL AND prev_close > 0
          AND adj_open > 0
    """
    df = pd.read_sql(sql, conn, params=(START, END, LIQ_MIN, START, END))
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    print(f"  {len(df):,} rows, {df['code'].nunique()} 銘柄")
    return df

def summarize(series, cost=COST):
    s = series.dropna()
    n   = len(s)
    mu  = s.mean() - cost
    se  = s.std() / np.sqrt(n) if n > 1 else np.nan
    t   = mu / se if se and se > 0 else np.nan
    sh  = mu / s.std() * np.sqrt(250) if s.std() > 0 else np.nan
    return {"n": n, "mean_net": mu, "t": t, "sharpe": sh}

def main():
    print("ユニバース読み込み...")
    df = load_universe()

    # ---- 分析ループ ----
    thresholds = [3.0, 5.0, 7.0, 10.0]
    directions = [(+1, "上昇後"), (-1, "下落後")]
    targets    = [("day_ret_bps", "翌日day(寄→引)"), ("gap_bps", "翌日gap(前日引→寄)")]
    results = []

    for thresh in thresholds:
        for direc, direc_name in directions:
            # シグナル: 前日が±thresh%以上
            mask = df["prev_ret_pct"] * direc >= thresh
            fired = df[mask].copy()
            if len(fired) < 50:
                continue

            for ret_col, ret_name in targets:
                # 全期間・IS・OOSの3分割
                for period in ["ALL", "IS", "OOS"]:
                    if period == "IS":
                        sub = fired[fired["date"] < IS_END]
                    elif period == "OOS":
                        sub = fired[fired["date"] >= OOS_ST]
                    else:
                        sub = fired

                    # momentum方向
                    s = summarize(sub[ret_col] * direc)
                    s.update({
                        "thresh": thresh, "direction": direc_name,
                        "target": ret_name, "mode": "momentum", "period": period
                    })
                    results.append(s)

                    # fade方向
                    s2 = summarize(-sub[ret_col] * direc)
                    s2.update({
                        "thresh": thresh, "direction": direc_name,
                        "target": ret_name, "mode": "fade", "period": period
                    })
                    results.append(s2)

    res_df = pd.DataFrame(results)
    res_df.to_csv("surge_summary.csv", index=False)
    print(f"集計完了: {len(res_df)} rows")

    # ---- グラフ: 閾値 × IS/OOS Sharpe ヒートマップ ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), facecolor="white")

    combos = [
        ("上昇後", "翌日day(寄→引)", "momentum", "上昇後モメンタム(翌日寄→引)"),
        ("上昇後", "翌日day(寄→引)", "fade",      "上昇後フェード(翌日寄→引)"),
        ("下落後", "翌日day(寄→引)", "momentum", "下落後モメンタム(翌日寄→引)"),
        ("下落後", "翌日day(寄→引)", "fade",      "下落後フェード(翌日寄→引)"),
    ]

    for ax, (direc, target, mode, title) in zip(axes.flatten(), combos):
        sub = res_df[(res_df["direction"]==direc) & (res_df["target"]==target) & (res_df["mode"]==mode)]
        pivot_data = {}
        for period in ["IS", "OOS", "ALL"]:
            ps = sub[sub["period"]==period].set_index("thresh")["sharpe"]
            pivot_data[period] = ps
        pv = pd.DataFrame(pivot_data, index=thresholds)

        x = np.arange(len(thresholds))
        width = 0.25
        colors = {"IS":"#1565C0","OOS":"#F57F17","ALL":"#555555"}
        for i, (period, clr) in enumerate(colors.items()):
            vals = pv[period].values if period in pv.columns else np.zeros(len(thresholds))
            bars = ax.bar(x + i*width, vals, width=width, label=period, color=clr, alpha=0.8)
            for bar, v in zip(bars, vals):
                if abs(v) >= 0.5:
                    ax.text(bar.get_x()+bar.get_width()/2, v+(0.02 if v>=0 else -0.04),
                            f"{v:.1f}", ha="center", fontsize=7)
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_xticks(x + width)
        ax.set_xticklabels([f"±{t}%" for t in thresholds], fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("Sharpe", fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("前日大幅変動 → 翌日パターン（全ユニバース流動性≥10億, 往復10bps控除後）",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig("result.png", dpi=100, bbox_inches="tight", facecolor="white")
    print("result.png 保存完了")

    # === サマリー ===
    print("\n=== IS/OOS両方プラスのエッジ（Sharpe > 0.2）===")
    for _, (direc, target, mode, title) in enumerate(combos):
        sub = res_df[(res_df["direction"]==direc) & (res_df["target"]==target) & (res_df["mode"]==mode)]
        for thresh in thresholds:
            row = sub[sub["thresh"]==thresh]
            is_r  = row[row["period"]=="IS"]
            oos_r = row[row["period"]=="OOS"]
            if is_r.empty or oos_r.empty:
                continue
            is_sh  = is_r["sharpe"].values[0]
            oos_sh = oos_r["sharpe"].values[0]
            is_t   = is_r["t"].values[0]
            oos_t  = oos_r["t"].values[0]
            is_n   = is_r["n"].values[0]
            oos_n  = oos_r["n"].values[0]
            if is_sh > 0.2 and oos_sh > 0.2:
                print(f"  ★ {title} thresh={thresh}%: IS Sh={is_sh:.2f}(t={is_t:.2f},N={is_n:,})  OOS Sh={oos_sh:.2f}(t={oos_t:.2f},N={oos_n:,})")

    # ギャップも表示
    sub2 = res_df[res_df["target"]=="翌日gap(前日引→寄)"]
    print("\n=== ギャップ予測（IS/OOSともSharpe>0.3）===")
    for thresh in thresholds:
        for direc, mode in [("上昇後","momentum"),("上昇後","fade"),("下落後","momentum"),("下落後","fade")]:
            row = sub2[(sub2["thresh"]==thresh) & (sub2["direction"]==direc) & (sub2["mode"]==mode)]
            is_r  = row[row["period"]=="IS"]
            oos_r = row[row["period"]=="OOS"]
            if is_r.empty or oos_r.empty:
                continue
            is_sh  = is_r["sharpe"].values[0]
            oos_sh = oos_r["sharpe"].values[0]
            is_t   = is_r["t"].values[0]
            oos_t  = oos_r["t"].values[0]
            if is_sh > 0.3 and oos_sh > 0.3:
                print(f"  ★ gap: {direc}/{mode} thresh={thresh}%: IS={is_sh:.2f}(t={is_t:.2f})  OOS={oos_sh:.2f}(t={oos_t:.2f})")

if __name__ == "__main__":
    main()
