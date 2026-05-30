"""
時間帯別リターンプロファイル分析
- 非鉄8銘柄 + 半導体14銘柄
- 各時間帯のリターン (bps) を全日 / LME上昇日 / 曜日別 で集計
- LMEはmacro.daily_ohlcvから取得
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
import matplotlib.font_manager as fm
import warnings
warnings.filterwarnings("ignore")

PG_CONFIG = {
    "host": os.environ.get("PGHOST", "omen"),
    "port": int(os.environ.get("PGPORT", 5432)),
    "user": os.environ.get("PGUSER", "postgres"),
    "dbname": os.environ.get("PGDATABASE", "market_data"),
}

NONFER = ["57060", "57110", "57130", "57140", "50160", "58010", "58020", "58030"]
SEMI   = ["69201", "69541", "68572", "68472", "30346", "30350", "76510",
          "79560", "60458", "28572", "80358", "68450", "64521", "285A0"]
ALL_CODES = NONFER + SEMI
SECTOR = {c: "非鉄" for c in NONFER}
SECTOR.update({c: "半導体" for c in SEMI})

START = "2024-05-01"
END   = "2026-05-30"

# 時間帯定義 (JST)
SLOTS = [
    ("寄り30分",   "09:00", "09:30"),
    ("前場前半",   "09:30", "10:30"),
    ("前場後半",   "10:30", "11:30"),
    ("後場前半",   "12:30", "13:30"),
    ("後場後半",   "13:30", "14:30"),
    ("引け前",     "14:30", "15:30"),
]

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

def compute_slot_return(df_stock, slot_open, slot_close):
    """各営業日の時間帯リターン (bps)"""
    t_open  = pd.to_datetime(slot_open).time()
    t_close = pd.to_datetime(slot_close).time()
    results = []
    for (code, date), grp in df_stock.groupby(["code", df_stock["ts"].dt.date]):
        sub = grp[(grp["ts"].dt.time >= t_open) & (grp["ts"].dt.time < t_close)]
        if len(sub) < 2:
            continue
        entry = sub.iloc[0]["open"]
        ex    = sub.iloc[-1]["close"]
        if entry <= 0:
            continue
        ret_bps = (ex / entry - 1) * 10000
        results.append({"code": code, "date": date, "ret_bps": ret_bps})
    return pd.DataFrame(results)

def summarize(series, name=""):
    n = len(series)
    mu = series.mean()
    se = series.std() / np.sqrt(n) if n > 0 else np.nan
    t  = mu / se if se > 0 else np.nan
    wr = (series > 0).mean() * 100
    return {"label": name, "n": n, "mean_bps": mu, "se": se, "t": t, "wr": wr}

def main():
    print("データ読み込み...")
    df = load_intraday(ALL_CODES, START, END)
    print(f"  {len(df):,} rows loaded ({df['code'].nunique()} codes)")
    df["date"] = df["ts"].dt.date
    df["dow"]  = df["ts"].dt.dayofweek   # 0=Mon ... 4=Fri

    lme = load_lme(START, END)
    print(f"  LME: {len(lme)} days")

    # LMEシグナルを日付に付与
    df["date_ts"] = pd.to_datetime(df["date"])
    df = df.merge(lme[["lme_ret"]].rename(columns={"lme_ret": "lme_prev"}),
                  left_on="date_ts", right_index=True, how="left")

    records = []
    for slot_name, s_open, s_close in SLOTS:
        print(f"  スロット計算: {slot_name} ({s_open}-{s_close})...")
        slot_df = compute_slot_return(df, s_open, s_close)
        if slot_df.empty:
            continue
        # LME情報をmerge
        slot_df["date_ts"] = pd.to_datetime(slot_df["date"])
        slot_df = slot_df.merge(lme[["lme_ret"]].rename(columns={"lme_ret":"lme_prev"}),
                                left_on="date_ts", right_index=True, how="left")
        slot_df["sector"] = slot_df["code"].map(SECTOR)
        slot_df["dow"]    = slot_df["date_ts"].dt.dayofweek
        DOW_MAP = {0:"月",1:"火",2:"水",3:"木",4:"金"}

        for sector in ["非鉄", "半導体", "ALL"]:
            sub = slot_df if sector == "ALL" else slot_df[slot_df["sector"] == sector]

            # 全日
            r = summarize(sub["ret_bps"], f"{slot_name}|{sector}|全日")
            r.update({"slot": slot_name, "sector": sector, "cond": "全日"})
            records.append(r)

            # LME上昇日 (前日LME +1%以上)
            sub_lme_up = sub[sub["lme_prev"] >= 1.0]
            r = summarize(sub_lme_up["ret_bps"], f"{slot_name}|{sector}|LME+1%")
            r.update({"slot": slot_name, "sector": sector, "cond": "LME+1%"})
            records.append(r)

            # LME下落日 (前日LME -1%以下)
            sub_lme_dn = sub[sub["lme_prev"] <= -1.0]
            r = summarize(sub_lme_dn["ret_bps"], f"{slot_name}|{sector}|LME-1%"  )
            r.update({"slot": slot_name, "sector": sector, "cond": "LME-1%"})
            records.append(r)

            # 曜日別
            for dow_i, dow_name in DOW_MAP.items():
                sub_dow = sub[sub["dow"] == dow_i]
                r = summarize(sub_dow["ret_bps"], f"{slot_name}|{sector}|{dow_name}")
                r.update({"slot": slot_name, "sector": sector, "cond": f"曜日_{dow_name}"})
                records.append(r)

    result = pd.DataFrame(records)
    result.to_csv("timeofday_summary.csv", index=False)
    print(f"  集計完了: {len(result)} rows -> timeofday_summary.csv")

    # --- グラフ: 全日・非鉄 の時間帯別 mean_bps + t値 ---
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), facecolor="white")
    plt.rcParams.update({"font.family": ["IPAexGothic", "Noto Sans CJK JP", "sans-serif"],
                          "axes.unicode_minus": False})
    conds = ["全日", "LME+1%", "LME-1%", "曜日_月", "曜日_火", "曜日_木"]
    titles = ["全日", "前日LME +1%以上", "前日LME -1%以下", "月曜日", "火曜日", "木曜日"]
    slot_labels = [s[0] for s in SLOTS]
    colors = {"非鉄": "#2196F3", "半導体": "#FF5722"}

    for ax, cond, title in zip(axes.flatten(), conds, titles):
        for sector in ["非鉄", "半導体"]:
            sub = result[(result["cond"] == cond) & (result["sector"] == sector)]
            sub = sub.set_index("slot").reindex(slot_labels)
            mu = sub["mean_bps"].fillna(0)
            t_ = sub["t"].fillna(0)
            x  = np.arange(len(slot_labels))
            ax.bar(x + (0 if sector == "非鉄" else 0.35), mu,
                   width=0.35, label=sector, color=colors[sector], alpha=0.8)
            # t値>2の場合にアスタリスク
            for xi, (m, tv) in enumerate(zip(mu, t_)):
                if abs(tv) >= 2.0:
                    ax.text(xi + (0 if sector == "非鉄" else 0.35) + 0.175,
                            m + (0.5 if m >= 0 else -1.5), "*", ha="center",
                            fontsize=10, color="black")
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(np.arange(len(slot_labels)) + 0.175)
        ax.set_xticklabels(slot_labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("平均リターン (bps)", fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("時間帯別リターンプロファイル（非鉄・半導体, 2024/05〜2026/05, * = |t|≥2）",
                 fontsize=13, y=1.01)
    fig.text(0.99, 0.01, "データ: JQuants 1分足 / macro.daily_ohlcv",
             ha="right", fontsize=7, color="gray")
    plt.tight_layout()
    plt.savefig("result.png", dpi=100, bbox_inches="tight", facecolor="white")
    print("result.png 保存完了")

    # 上位のt値エッジを表示
    high_t = result[(result["n"] >= 30) & (result["t"].abs() >= 2.0)].copy()
    high_t = high_t.sort_values("t", key=abs, ascending=False)
    print("\n=== |t|≥2.0 かつ N≥30 のエッジ候補 ===")
    print(high_t[["slot","sector","cond","n","mean_bps","t","wr"]].to_string(index=False))

if __name__ == "__main__":
    main()
