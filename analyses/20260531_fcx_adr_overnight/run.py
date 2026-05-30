"""
FCX(銅株) / ADR / SOX シグナル → 翌日イントラエッジ検証
LMEの代替として、日々更新されるシグナルで非鉄・半導体の翌日パターンを測定

シグナル:
  FCX  (Freeport-McMoRan, 銅株)        → 非鉄8銘柄
  .SOX (半導体指数)                     → 半導体14銘柄
  NQc1 (NASDAQ先物)                    → 全体
  JPY= (ドル円, 下落=円高)             → 輸出株
  ADR_8035 (TEL-ADR)                   → 8035(東京エレクトロン)
  ADR_6920 (Lasertec ADR... 存在確認)  → 6920

測定: 翌日の寄→引(全日), 前場, 後場, 引け前(各bps, 往復8bps控除後)
IS: 2022-01〜2024-06 / OOS: 2024-07〜2026-05 (日足10年使用)
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

IS_END = "2024-07-01"
OOS_ST = "2024-07-01"
INTRA_ST = "2024-05-01"  # 1分足はここから

COST = 8.0  # 往復bps

def get_conn():
    return psycopg2.connect(**PG_CONFIG)

def load_macro(symbols, start="2020-01-01", end="2026-06-01"):
    conn = get_conn()
    sql = """
        SELECT symbol, trade_date, close
        FROM macro.daily_ohlcv
        WHERE symbol = ANY(%s) AND trade_date >= %s AND trade_date <= %s
        ORDER BY symbol, trade_date
    """
    df = pd.read_sql(sql, conn, params=(symbols, start, end))
    conn.close()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    # 各シンボルのリターンを計算
    df["ret"] = df.groupby("symbol")["close"].pct_change() * 100
    return df.pivot(index="trade_date", columns="symbol", values="ret")

def load_daily(codes, start="2022-01-01", end="2026-06-01"):
    conn = get_conn()
    sql = """
        SELECT code, date,
               adj_open, adj_close,
               LAG(adj_close) OVER (PARTITION BY code ORDER BY date) AS prev_close,
               turnover_value
        FROM stocks_daily
        WHERE code = ANY(%s) AND date >= %s AND date <= %s
        ORDER BY code, date
    """
    df = pd.read_sql(sql, conn, params=(codes, start, end))
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    # 翌日リターン: open→close (bps)
    df["day_ret"] = (df["adj_close"] / df["adj_open"] - 1) * 10000
    # ギャップ: open vs prev_close
    df["gap"] = np.where(df["prev_close"] > 0,
                         (df["adj_open"] / df["prev_close"] - 1) * 10000, np.nan)
    return df

def load_intraday_slots(codes, start=INTRA_ST, end="2026-06-01"):
    """寄り30分・前場・後場・引け前の日次リターンを1分足から計算"""
    import datetime
    conn = get_conn()
    sql = """
        SELECT code, ts, open, close
        FROM stocks_intraday
        WHERE code = ANY(%s) AND ts >= %s AND ts < %s
        ORDER BY code, ts
    """
    df = pd.read_sql(sql, conn, params=(codes, start, end))
    conn.close()
    df["ts"] = pd.to_datetime(df["ts"])
    df["date"] = df["ts"].dt.date

    slots = {
        "am1": (datetime.time(9,0),  datetime.time(9,30)),
        "am":  (datetime.time(9,0),  datetime.time(11,30)),
        "pm":  (datetime.time(12,30),datetime.time(15,30)),
        "cls": (datetime.time(14,30),datetime.time(15,30)),
    }

    rows = []
    for (code, date), grp in df.groupby(["code", "date"]):
        grp = grp.sort_values("ts")
        r = {"code": code, "date": pd.Timestamp(date)}
        for slot, (tf, tt) in slots.items():
            sub = grp[(grp["ts"].dt.time >= tf) & (grp["ts"].dt.time < tt)]
            if len(sub) < 2:
                r[slot] = np.nan
                continue
            entry = sub.iloc[0]["open"]
            ex    = sub.iloc[-1]["close"]
            r[slot] = (ex / entry - 1) * 10000 if entry > 0 else np.nan
        rows.append(r)
    return pd.DataFrame(rows)

def summarize(series, cost=COST):
    s = series.dropna()
    n  = len(s)
    mu = s.mean() - cost
    se = s.std() / np.sqrt(n) if n > 1 else np.nan
    t  = mu / se if se and se > 0 else np.nan
    sharpe = mu / s.std() * np.sqrt(250) if s.std() > 0 else np.nan
    return {"n": n, "mean_net": mu, "t": t, "sharpe": sharpe, "cum": mu * n}

def backtest(panel, sig_col, sig_thresh, direction, ret_col, label):
    """panel: date-indexed, sig_col: signal col, ret_col: return col"""
    fired = panel[panel[sig_col] * direction >= sig_thresh].copy()
    results = []
    for period, (st, en) in [("ALL",""), ("IS", IS_END), ("OOS", "")]:
        if period == "IS":
            sub = fired[fired.index < IS_END]
        elif period == "OOS":
            sub = fired[fired.index >= OOS_ST]
        else:
            sub = fired
        s = summarize(sub[ret_col])
        s.update({"label": label, "period": period,
                  "sig": sig_col, "thresh": sig_thresh, "direction": direction, "ret": ret_col})
        results.append(s)
    return results

def main():
    print("マクロデータ読み込み...")
    macro_syms = ["FCX", "HGc1", ".SOX", "NQc1", "JPY=", "AUD=",
                  "ADR_8035", "ADR_6920", "ADR_6758", "ADR_7203",
                  "ADR_9984", "ADR_6501"]
    macro = load_macro(macro_syms)
    print(f"  {macro.shape[0]} days, {macro.shape[1]} symbols")
    print(f"  最新: {macro.index.max().date()}")

    print("日足データ読み込み...")
    all_codes = NONFER + SEMI
    daily = load_daily(all_codes)
    print(f"  {len(daily):,} rows")

    # シグナルを翌営業日にシフト（前日シグナル→当日リターン）
    macro_shifted = macro.shift(1)  # 当日の前日シグナル

    # 日足にマクロをjoin
    # code別にpivotしてから各コードでマージ
    daily["date"] = pd.to_datetime(daily["date"])
    daily_idx = daily.set_index("date")

    # panelを作成
    results_all = []

    signal_configs = [
        # (シグナル列, 閾値, 方向, ターゲット銘柄リスト, リターン列, ラベル)
        ("FCX",    2.0, +1, NONFER, "day_ret", "FCX+2%→非鉄翌日Long"),
        ("FCX",    2.0, +1, NONFER, "gap",     "FCX+2%→非鉄翌日ギャップ"),
        ("FCX",   -2.0, -1, NONFER, "day_ret", "FCX-2%→非鉄翌日Short"),
        (".SOX",   2.0, +1, SEMI,   "day_ret", "SOX+2%→半導体翌日Long"),
        (".SOX",   2.0, +1, SEMI,   "gap",     "SOX+2%→半導体翌日ギャップ"),
        (".SOX",  -2.0, -1, SEMI,   "day_ret", "SOX-2%→半導体翌日Short"),
        ("NQc1",   1.5, +1, all_codes, "day_ret", "NQc1+1.5%→全体翌日Long"),
        ("JPY=",  -1.0, -1, ["69201","69541","68572","68472","30346"], "day_ret", "円高-1%→半導体5翌日Short"),
        ("AUD=",   1.0, +1, NONFER, "day_ret", "AUD+1%→非鉄翌日Long"),
        ("ADR_8035", 3.0, +1, ["80350"], "day_ret", "TEL-ADR+3%→TEL翌日Long"),
        ("ADR_6920", 3.0, +1, ["69200"], "day_ret", "LT-ADR+3%→LT翌日Long"),
        ("ADR_9984", 3.0, +1, ["99840"], "day_ret", "SB-ADR+3%→9984翌日Long"),
    ]

    print("\nバックテスト中...")
    for sig_col, thresh, direc, codes, ret_col, lbl in signal_configs:
        if sig_col not in macro_shifted.columns:
            print(f"  SKIP: {sig_col} (シンボルなし)")
            continue

        # 対象銘柄の等加重日次リターンを計算
        sub_daily = daily[daily["code"].isin(codes)].copy()
        sub_daily_grp = sub_daily.groupby("date")[ret_col].mean()

        # シグナルと結合
        panel = pd.DataFrame({
            sig_col: macro_shifted[sig_col],
            ret_col: sub_daily_grp
        }).dropna()

        fired = panel[panel[sig_col] * direc >= abs(thresh)]
        n_total = len(fired)
        if n_total < 10:
            print(f"  {lbl}: N={n_total} (少なすぎ, skip)")
            continue

        for period in ["ALL", "IS", "OOS"]:
            if period == "IS":
                sub = fired[fired.index < IS_END]
            elif period == "OOS":
                sub = fired[fired.index >= OOS_ST]
            else:
                sub = fired
            s = summarize(sub[ret_col])
            s.update({"label": lbl, "period": period})
            results_all.append(s)

        print(f"  {lbl}: N={n_total}")

    res_df = pd.DataFrame(results_all)
    res_df.to_csv("signal_summary.csv", index=False)
    print(f"\n集計完了 -> signal_summary.csv")

    # ---- グラフ ----
    all_labels = res_df["label"].unique().tolist()
    n_lbl = len(all_labels)
    cols = 4
    rows_g = (n_lbl + cols - 1) // cols
    fig, axes = plt.subplots(rows_g, cols, figsize=(18, rows_g*3.5), facecolor="white")
    axes = axes.flatten()

    for i, lbl in enumerate(all_labels):
        ax = axes[i]
        sub = res_df[res_df["label"] == lbl]
        periods = ["IS", "OOS", "ALL"]
        x = np.arange(len(periods))
        mu_vals = []
        for p in periods:
            row = sub[sub["period"] == p]
            mu_vals.append(row["mean_net"].values[0] if len(row) > 0 else np.nan)
        colors = ["#1565C0","#F57F17","#555555"]
        bars = ax.bar(x, mu_vals, color=colors, alpha=0.8)
        for bar, t_row, p in zip(bars, [sub[sub["period"]==pp]["t"].values for pp in periods], periods):
            tv = t_row[0] if len(t_row) > 0 else 0
            mu = bar.get_height()
            if abs(tv) >= 2.0:
                ax.text(bar.get_x()+bar.get_width()/2,
                        mu+(0.3 if mu>=0 else -0.8), "*", ha="center", fontsize=12)
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(periods, fontsize=8)
        ax.set_title(lbl, fontsize=8)
        ax.set_ylabel("net bps/day", fontsize=7)
        ax.grid(axis="y", alpha=0.3)
        # N
        n_all = sub[sub["period"]=="ALL"]["n"].values
        if len(n_all) > 0:
            ax.text(0.98, 0.98, f"N={n_all[0]}", transform=ax.transAxes,
                    ha="right", va="top", fontsize=7, color="gray")

    for i in range(len(all_labels), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("FCX/ADR/SOX/NQc1/JPY シグナル → 翌日イントラエッジ（往復8bps控除後, IS<2024-07 / OOS>=2024-07, *=|t|≥2）",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig("result.png", dpi=100, bbox_inches="tight", facecolor="white")
    print("result.png 保存完了")

    # === 結果サマリー ===
    print("\n=== IS/OOS両方プラスのエッジ候補 ===")
    for lbl in all_labels:
        sub = res_df[res_df["label"] == lbl]
        is_row  = sub[sub["period"]=="IS"]
        oos_row = sub[sub["period"]=="OOS"]
        if is_row.empty or oos_row.empty:
            continue
        is_mu  = is_row["mean_net"].values[0]
        oos_mu = oos_row["mean_net"].values[0]
        is_t   = is_row["t"].values[0]
        oos_t  = oos_row["t"].values[0]
        is_sh  = is_row["sharpe"].values[0]
        oos_sh = oos_row["sharpe"].values[0]
        mark = "★" if is_mu > 0 and oos_mu > 0 else " "
        print(f"  {mark} {lbl:40s}: IS={is_mu:+.1f}bps(t={is_t:.2f}, Sh={is_sh:.2f})  OOS={oos_mu:+.1f}bps(t={oos_t:.2f}, Sh={oos_sh:.2f})")

if __name__ == "__main__":
    main()
