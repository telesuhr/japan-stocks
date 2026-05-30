"""
決算翌日ザラ場フェード分析
仮説: 決算引け後発表 → 翌日寄り付きの大ギャップは、ザラ場で部分的にフェードされる
- fin_summary から引け後発表(disc_time >= 15:00)を抽出
- 翌日の寄り付きギャップ(gap_bps = (open - prev_close) / prev_close * 10000)を計算
- ギャップを5分位に分けて、ザラ場リターン(open→close, open→11:30, open→13:30等)を測定
- フェード(逆張り)のエッジを検証
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
MIN_LIQUIDITY = 1e9  # 10億円/日

def get_conn():
    return psycopg2.connect(**PG_CONFIG)

def load_earnings():
    """引け後発表の決算イベントを取得"""
    conn = get_conn()
    sql = """
        SELECT DISTINCT ON (f.code, f.disc_date)
            f.code,
            f.disc_date,
            f.disc_time
        FROM fin_summary f
        JOIN symbol_master sm ON sm.code5 = f.code
        WHERE f.disc_date >= %s AND f.disc_date < %s
          AND f.disc_time >= '15:00:00'   -- 引け後発表のみ
          AND sm.market IS NOT NULL
        ORDER BY f.code, f.disc_date, f.disc_time
    """
    df = pd.read_sql(sql, conn, params=(START, END))
    conn.close()
    df["disc_date"] = pd.to_datetime(df["disc_date"])
    print(f"  決算件数: {len(df):,} (引け後発表)")
    return df

def load_daily_for_earnings(codes_dates):
    """決算翌営業日の日足データ (open, prev_close) を取得"""
    conn = get_conn()
    # 大量のcode+dateペアを扱うためにIN句で取得
    all_codes = list(set([c for c,d in codes_dates]))
    sql = """
        SELECT code, date, open, close, volume, turnover_value,
               LAG(close) OVER (PARTITION BY code ORDER BY date) as prev_close,
               LAG(date) OVER (PARTITION BY code ORDER BY date) as prev_date
        FROM stocks_daily
        WHERE code = ANY(%s) AND date >= %s AND date < %s
        ORDER BY code, date
    """
    df = pd.read_sql(sql, conn, params=(all_codes, START, END))
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    return df

def load_intraday_for_dates(code_date_pairs, horizon_mins=None):
    """決算翌日の1分足を取得（流動性ゲート済みcodeのみ）"""
    if not code_date_pairs:
        return pd.DataFrame()
    conn = get_conn()
    # 日付リストと銘柄リストでまとめて取得
    all_codes = list(set([c for c,d in code_date_pairs]))
    all_dates = list(set([str(d.date()) for c,d in code_date_pairs]))
    # 最小/最大日付で絞ってから後でフィルタ
    min_date = min(all_dates)
    max_date = max(all_dates)
    sql = """
        SELECT code, ts, open, close
        FROM stocks_intraday
        WHERE code = ANY(%s) AND ts::date >= %s AND ts::date <= %s
        ORDER BY code, ts
    """
    df = pd.read_sql(sql, conn, params=(all_codes, min_date, max_date))
    conn.close()
    df["ts"] = pd.to_datetime(df["ts"])
    df["date"] = df["ts"].dt.date
    return df

def main():
    print("決算データ読み込み...")
    earn = load_earnings()

    print("日足データ読み込み...")
    all_codes = earn["code"].unique().tolist()
    daily = load_daily_for_earnings(
        list(zip(all_codes, [None]*len(all_codes)))
    )

    # 流動性フィルタ: 60日平均売買代金 >= 10億円
    daily["avg_to60"] = daily.groupby("code")["turnover_value"].transform(
        lambda x: x.rolling(60, min_periods=20).mean()
    )
    liquid = daily[daily["avg_to60"] >= MIN_LIQUIDITY][["code","date"]].copy()
    liquid["is_liquid"] = True

    # 決算翌営業日を特定: disc_date → 翌営業日 (next trading day in daily)
    # daily のdate一覧から各銘柄の営業日を使って翌日を特定
    daily_sorted = daily.sort_values(["code","date"])

    # 決算に翌営業日の open/prev_close を付与
    records = []
    for _, row in earn.iterrows():
        code = row["code"]
        disc_dt = row["disc_date"]
        # その銘柄の disc_date より後の最初の営業日
        sub = daily_sorted[(daily_sorted["code"] == code) & (daily_sorted["date"] > disc_dt)]
        if sub.empty:
            continue
        next_row = sub.iloc[0]
        if pd.isna(next_row["prev_close"]) or next_row["prev_close"] <= 0:
            continue
        gap_bps = (next_row["open"] - next_row["prev_close"]) / next_row["prev_close"] * 10000
        records.append({
            "code": code,
            "disc_date": disc_dt,
            "trade_date": next_row["date"],
            "open": next_row["open"],
            "close": next_row["close"],
            "prev_close": next_row["prev_close"],
            "gap_bps": gap_bps,
            "day_ret_bps": (next_row["close"] / next_row["open"] - 1) * 10000,  # open→close
            "turnover": next_row.get("turnover_value", np.nan),
        })

    obs = pd.DataFrame(records)
    print(f"  翌営業日マッチ: {len(obs):,} 件")

    # 流動性ゲート
    obs = obs.merge(liquid.rename(columns={"date":"trade_date"}),
                    on=["code","trade_date"], how="inner")
    print(f"  流動性ゲート後: {len(obs):,} 件 ({obs['code'].nunique()} 銘柄)")

    obs.to_csv("earnings_obs.csv", index=False)

    # 1分足取得（2024/05以降のみ）
    print("イントラデイデータ読み込み (2024/05以降)...")
    obs_intra = obs[obs["trade_date"] >= pd.Timestamp("2024-05-01")].copy()
    if len(obs_intra) > 0:
        code_date_pairs = list(zip(obs_intra["code"], obs_intra["trade_date"]))
        intra = load_intraday_for_dates(code_date_pairs)
        print(f"  1分足: {len(intra):,} rows")
    else:
        intra = pd.DataFrame()

    # イントラ時間帯リターン計算
    import datetime
    def intra_ret(row, t_from_str, t_to_str):
        if intra.empty:
            return np.nan
        date_val = row["trade_date"].date() if hasattr(row["trade_date"], "date") else row["trade_date"]
        sub = intra[(intra["code"] == row["code"]) & (intra["date"] == date_val)]
        t_from = datetime.time(*map(int, t_from_str.split(":")))
        t_to   = datetime.time(*map(int, t_to_str.split(":")))
        sub2 = sub[(sub["ts"].dt.time >= t_from) & (sub["ts"].dt.time < t_to)]
        if len(sub2) < 2:
            return np.nan
        entry = row["open"]  # 寄り付きからの逆張り
        ex    = sub2.iloc[-1]["close"]
        if entry <= 0:
            return np.nan
        return (ex / entry - 1) * 10000  # bps (open=0基準)

    if len(obs_intra) > 0:
        print("  時間帯リターン計算中...")
        obs_intra = obs_intra.copy()
        obs_intra["ret_am"]  = obs_intra.apply(lambda r: intra_ret(r, "09:00","11:30"), axis=1)
        obs_intra["ret_pm"]  = obs_intra.apply(lambda r: intra_ret(r, "12:30","15:30"), axis=1)
        obs_intra["ret_30m"] = obs_intra.apply(lambda r: intra_ret(r, "09:00","09:30"), axis=1)
        obs_intra["ret_1h"]  = obs_intra.apply(lambda r: intra_ret(r, "09:00","10:00"), axis=1)
        obs_intra.to_csv("earnings_intra.csv", index=False)
        print(f"  earnings_intra.csv ({len(obs_intra)} rows)")

    # ---- 分析 ----
    def analyze(df, label, col="day_ret_bps", n_q=5):
        df = df.copy().dropna(subset=["gap_bps", col])
        if len(df) < 20:
            return []
        df["gap_q"] = pd.qcut(df["gap_bps"].rank(method="first"), n_q, labels=False)
        rows = []
        for q in range(n_q):
            sub = df[df["gap_q"] == q][col]
            n   = len(sub)
            mu  = sub.mean()
            se  = sub.std() / np.sqrt(n) if n > 1 else np.nan
            t   = mu / se if se and se > 0 else np.nan
            wr  = (sub > 0).mean() * 100
            rows.append({"label": label, "target": col, "gap_q": q,
                         "n": n, "mean_bps": mu, "t": t, "wr": wr})
        return rows

    all_results = []
    # 日足ベース: 全期間
    all_results += analyze(obs, "全期間_day", "day_ret_bps")
    # IS/OOS split
    obs_is  = obs[obs["trade_date"] < pd.Timestamp("2024-01-01")]
    obs_oos = obs[obs["trade_date"] >= pd.Timestamp("2024-01-01")]
    all_results += analyze(obs_is,  "IS(<2024)_day",  "day_ret_bps")
    all_results += analyze(obs_oos, "OOS(2024+)_day", "day_ret_bps")
    # 大ギャップ(>=100bps or <=-100bps)
    obs_big_up = obs[obs["gap_bps"] >= 100]
    obs_big_dn = obs[obs["gap_bps"] <= -100]
    all_results += analyze(obs_big_up, "大ギャップ上昇_day", "day_ret_bps")
    all_results += analyze(obs_big_dn, "大ギャップ下落_day", "day_ret_bps")

    # イントラベース
    if len(obs_intra) > 0:
        for col in ["ret_30m","ret_1h","ret_am","ret_pm","day_ret_bps"]:
            all_results += analyze(obs_intra, f"intra_{col}", col)

    res_df = pd.DataFrame(all_results)
    res_df.to_csv("fade_summary.csv", index=False)
    print(f"\n集計完了: {len(res_df)} rows -> fade_summary.csv")

    # ---- グラフ ----
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), facecolor="white")
    plot_configs = [
        ("全期間_day",     "day_ret_bps", "全期間 日次(寄→引)"),
        ("IS(<2024)_day",  "day_ret_bps", "IS(<2024) 日次"),
        ("OOS(2024+)_day", "day_ret_bps", "OOS(2024+) 日次"),
        ("大ギャップ上昇_day", "day_ret_bps", "大ギャップ上昇(≥100bps) 日次"),
        ("大ギャップ下落_day", "day_ret_bps", "大ギャップ下落(≤-100bps) 日次"),
        ("intra_ret_am",   "ret_am", "2024/05〜 前場(9-11:30)"),
    ]
    q_labels = ["Q1\n(gap最小)", "Q2", "Q3", "Q4", "Q5\n(gap最大)"]
    colors   = ["#b71c1c","#ef5350","#90a4ae","#66bb6a","#1b5e20"]

    for ax, (lbl, col, title) in zip(axes.flatten(), plot_configs):
        sub = res_df[(res_df["label"] == lbl) & (res_df["target"] == col)].sort_values("gap_q")
        if sub.empty:
            ax.text(0.5, 0.5, "データ不足", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title, fontsize=10)
            continue
        bars = ax.bar(sub["gap_q"], sub["mean_bps"], color=colors[:len(sub)], alpha=0.85)
        for bar, t_val, mu in zip(bars, sub["t"], sub["mean_bps"]):
            if abs(t_val) >= 2.0:
                ax.text(bar.get_x()+bar.get_width()/2,
                        mu+(0.5 if mu>=0 else -1.5), "*", ha="center", fontsize=12)
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_title(title, fontsize=10)
        ns = sub["n"].tolist()
        ax.text(0.98, 0.98, f"N={sum(ns)}", transform=ax.transAxes,
                ha="right", va="top", fontsize=8, color="gray")
        ax.set_xticks(range(len(q_labels)))
        ax.set_xticklabels(q_labels, fontsize=7)
        ax.set_ylabel("リターン bps (フェード方向)", fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("決算翌日 寄付きギャップ × ザラ場リターン（フェード検証, * = |t|≥2）",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig("result.png", dpi=100, bbox_inches="tight", facecolor="white")
    print("result.png 保存完了")

    # 主要な発見を出力
    print("\n=== 主要発見: Q1(大ギャップ下) vs Q5(大ギャップ上) のリターン ===")
    for lbl, col, title in plot_configs:
        sub = res_df[(res_df["label"] == lbl) & (res_df["target"] == col)]
        if sub.empty or len(sub) < 5:
            continue
        q1 = sub[sub["gap_q"] == 0].iloc[0] if len(sub[sub["gap_q"]==0]) > 0 else None
        q5 = sub[sub["gap_q"] == 4].iloc[0] if len(sub[sub["gap_q"]==4]) > 0 else None
        if q1 is None or q5 is None:
            continue
        ls = q5["mean_bps"] - q1["mean_bps"]
        print(f"  {title:30s}: Q1={q1['mean_bps']:+.1f}bps(t={q1['t']:.2f})  "
              f"Q5={q5['mean_bps']:+.1f}bps(t={q5['t']:.2f})  "
              f"L/S spread={ls:+.1f}bps")

if __name__ == "__main__":
    main()
