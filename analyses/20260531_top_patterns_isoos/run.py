"""
IS/OOS検証: TOD・ランチリバーサルの上位パターン
検証対象:
  A) 火曜 前場後半(10:30-11:30) 非鉄 Short  (t=-5.42)
  B) 水曜 引け前(14:30-15:30) 全体 Long     (t=+5.35)
  C) LME下落日(-1%) 引け前 非鉄 Long        (t=+4.63)
  D) 毎日 寄り30分 非鉄 Long               (t=+4.40)
  E) 前場弱い非鉄の後場リバーサル Long      (t=+3.16, LME下落日はt=+3.58)
  F) 火曜 前場強い非鉄の後場 Short          (t=-2.85)

IS: 2024-05-01 〜 2025-06-30
OOS: 2025-07-01 〜 2026-05-30
コスト: 片道4bps (往復8bps)
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
import datetime
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

IS_END  = pd.Timestamp("2025-07-01")
OOS_ST  = pd.Timestamp("2025-07-01")
START   = pd.Timestamp("2024-05-01")
END     = pd.Timestamp("2026-05-31")
COST_BP = 8.0  # 往復

def get_conn():
    return psycopg2.connect(**PG_CONFIG)

def load_intraday():
    conn = get_conn()
    sql = """
        SELECT code, ts, open, close, volume
        FROM stocks_intraday
        WHERE code = ANY(%s) AND ts >= %s AND ts < %s
        ORDER BY code, ts
    """
    df = pd.read_sql(sql, conn, params=(ALL_CODES, START.strftime("%Y-%m-%d"), END.strftime("%Y-%m-%d")))
    conn.close()
    df["ts"] = pd.to_datetime(df["ts"])
    return df

def load_lme():
    conn = get_conn()
    sql = """
        SELECT trade_date AS date, close
        FROM macro.daily_ohlcv
        WHERE symbol = 'Cc1' AND trade_date >= %s AND trade_date < %s
        ORDER BY trade_date
    """
    df = pd.read_sql(sql, conn, params=(START.strftime("%Y-%m-%d"), END.strftime("%Y-%m-%d")))
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    df["lme_ret"] = df["close"].pct_change() * 100
    return df.set_index("date")

def period_ret(grp, t_from, t_to):
    """grp: 当日の code-date 1分足、返り値 bps (entry=open of first bar, exit=close of last bar)"""
    sub = grp[(grp["ts"].dt.time >= t_from) & (grp["ts"].dt.time < t_to)]
    if len(sub) < 2:
        return np.nan
    entry = sub.iloc[0]["open"]
    ex    = sub.iloc[-1]["close"]
    if entry <= 0:
        return np.nan
    return (ex / entry - 1) * 10000

T = {
    "am1": (datetime.time(9,0),  datetime.time(9,30)),
    "am2": (datetime.time(9,30), datetime.time(10,30)),
    "am3": (datetime.time(10,30),datetime.time(11,30)),
    "pm1": (datetime.time(12,30),datetime.time(13,30)),
    "pm2": (datetime.time(13,30),datetime.time(14,30)),
    "cls": (datetime.time(14,30),datetime.time(15,30)),
    "am":  (datetime.time(9,0),  datetime.time(11,30)),
    "pm":  (datetime.time(12,30),datetime.time(15,30)),
}

def build_panel(df, lme):
    rows = []
    for (code, date), grp in df.groupby(["code", df["ts"].dt.date]):
        grp = grp.sort_values("ts")
        date_ts = pd.Timestamp(date)
        lme_ret = lme["lme_ret"].get(date_ts, np.nan)
        sector  = SECTOR.get(code, "OTHER")
        dow     = date_ts.dayofweek
        r = {"code": code, "date": date_ts, "sector": sector,
             "dow": dow, "lme_ret": lme_ret}
        for slot, (tf, tt) in T.items():
            r[slot] = period_ret(grp, tf, tt)
        rows.append(r)
    return pd.DataFrame(rows)

def backtest(panel, signal_mask, direction, entry_slot, exit_slot,
             cost=COST_BP, label=""):
    """
    signal_mask: boolean mask on panel rows to trade
    direction: +1 (long) or -1 (short)
    entry/exit_slot: column name in panel
    Returns: daily PnL series (entry_slot の日付単位で集計)
    """
    sub = panel[signal_mask].copy()
    # per-position net return (after cost) in signal direction
    sub["ret"] = direction * (sub[exit_slot] - sub[entry_slot].fillna(0)) - cost
    # exit_slotのreturnをそのまま使う（entry_slotは0基準の場合もある）
    # シンプルに exit_slot だけ使う
    sub["ret"] = direction * sub[exit_slot] - cost
    # 日次: 全シグナル銘柄の等加重平均
    daily = sub.groupby("date")["ret"].mean().rename(label)
    return daily

def summarize(daily, name):
    n  = len(daily.dropna())
    mu = daily.mean()
    se = daily.std() / np.sqrt(n) if n > 1 else np.nan
    t  = mu / se if se and se > 0 else np.nan
    sharpe = mu / daily.std() * np.sqrt(250) if daily.std() > 0 else np.nan
    return {"name": name, "n_days": n, "mean_bps": mu, "t": t, "sharpe": sharpe,
            "cum_bps": daily.sum()}

def main():
    print("データ読み込み...")
    raw = load_intraday()
    print(f"  {len(raw):,} rows")
    lme = load_lme()
    print(f"  LME: {len(lme)} days")

    print("パネル構築中 (数分かかります)...")
    panel = build_panel(raw, lme)
    print(f"  panel: {len(panel):,} code-days")

    panel.to_csv("panel.csv", index=False)

    # ランチリバーサル用: 前場リターンの分位
    panel["am_q5"] = panel.groupby(["code", panel["date"].dt.year.astype(str) + panel["date"].dt.month.astype(str)])["am"].transform(
        lambda x: pd.qcut(x.rank(method="first"), 5, labels=False) if len(x.dropna()) >= 5 else np.nan
    )
    # 全期間分位のほうが安定
    panel["am_q5"] = panel.groupby("code")["am"].transform(
        lambda x: pd.qcut(x.rank(method="first"), 5, labels=False))

    IS  = panel[panel["date"] < IS_END]
    OOS = panel[panel["date"] >= OOS_ST]
    ALL = panel

    results = []
    pnl_dict = {}

    def run_bt(p, mask_fn, direction, slot, label):
        for name, sub in [("ALL", ALL), ("IS", IS), ("OOS", OOS)]:
            sp = sub[mask_fn(sub)].copy()
            sp["ret"] = direction * sp[slot] - COST_BP
            daily = sp.groupby("date")["ret"].mean()
            s = summarize(daily, f"{label}_{name}")
            s["period"] = name
            s["strategy"] = label
            results.append(s)
            if name == "ALL":
                pnl_dict[label] = daily

    # --- A: 火曜 前場後半 非鉄 Short ---
    run_bt(panel,
           lambda p: (p["dow"]==1) & (p["sector"]=="非鉄"),
           direction=-1, slot="am3",
           label="A_火曜_前場後半_非鉄Short")

    # --- B: 水曜 引け前 全体 Long ---
    run_bt(panel,
           lambda p: (p["dow"]==2),
           direction=+1, slot="cls",
           label="B_水曜_引け前_全体Long")

    # --- C: LME下落日 引け前 非鉄 Long ---
    run_bt(panel,
           lambda p: (p["lme_ret"] <= -1.0) & (p["sector"]=="非鉄"),
           direction=+1, slot="cls",
           label="C_LME下落_引け前_非鉄Long")

    # --- D: 毎日 寄り30分 非鉄 Long ---
    run_bt(panel,
           lambda p: p["sector"]=="非鉄",
           direction=+1, slot="am1",
           label="D_毎日_寄30分_非鉄Long")

    # --- E: 前場弱い非鉄 後場ロング (全日) ---
    run_bt(panel,
           lambda p: (p["sector"]=="非鉄") & (p["am_q5"]==0),
           direction=+1, slot="pm",
           label="E_前場弱_後場_非鉄Long")

    # --- E2: LME下落日 前場弱い非鉄 後場ロング ---
    run_bt(panel,
           lambda p: (p["sector"]=="非鉄") & (p["am_q5"]==0) & (p["lme_ret"]<=-1.0),
           direction=+1, slot="pm",
           label="E2_LME下落_前場弱_後場_非鉄Long")

    # --- F: 火曜 前場強い非鉄 後場 Short ---
    run_bt(panel,
           lambda p: (p["dow"]==1) & (p["sector"]=="非鉄") & (p["am_q5"]==4),
           direction=-1, slot="pm",
           label="F_火曜_前場強_後場_非鉄Short")

    # --- G: 水曜 前場弱い非鉄 後場ロング ---
    run_bt(panel,
           lambda p: (p["dow"]==2) & (p["sector"]=="非鉄") & (p["am_q5"]==0),
           direction=+1, slot="pm",
           label="G_水曜_前場弱_後場_非鉄Long")

    res_df = pd.DataFrame(results)
    res_df.to_csv("isoos_summary.csv", index=False)

    print("\n=== IS/OOS検証結果 ===")
    for _, row in res_df.iterrows():
        print(f"  {row['strategy']:40s} [{row['period']:3s}] "
              f"N={row['n_days']:4d}  mean={row['mean_bps']:+7.2f}bps  "
              f"t={row['t']:+6.2f}  Sharpe={row['sharpe']:+5.2f}  cum={row['cum_bps']:+.0f}bps")

    # ---- グラフ ----
    strategies = ["A_火曜_前場後半_非鉄Short","B_水曜_引け前_全体Long",
                  "C_LME下落_引け前_非鉄Long","D_毎日_寄30分_非鉄Long",
                  "E_前場弱_後場_非鉄Long","E2_LME下落_前場弱_後場_非鉄Long",
                  "F_火曜_前場強_後場_非鉄Short","G_水曜_前場弱_後場_非鉄Long"]

    n_strat = len([s for s in strategies if s in pnl_dict])
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), facecolor="white")
    axes = axes.flatten()

    colors_is  = "#1565C0"
    colors_oos = "#F57F17"

    ax_idx = 0
    for strat in strategies:
        if strat not in pnl_dict or ax_idx >= len(axes):
            continue
        ax = axes[ax_idx]; ax_idx += 1
        pnl = pnl_dict[strat].sort_index().cumsum()
        is_part  = pnl[pnl.index < IS_END]
        oos_part = pnl[pnl.index >= OOS_ST]

        ax.plot(is_part.index, is_part.values, color=colors_is, linewidth=1.5, label="IS")
        ax.plot(oos_part.index, oos_part.values, color=colors_oos, linewidth=1.5, label="OOS")
        ax.axvline(IS_END, color="gray", linestyle="--", linewidth=0.8)
        ax.axhline(0, color="gray", linewidth=0.6)

        # IS/OOS Sharpe を annotation
        sub_all = res_df[res_df["strategy"] == strat]
        is_s  = sub_all[sub_all["period"]=="IS"]["sharpe"].values
        oos_s = sub_all[sub_all["period"]=="OOS"]["sharpe"].values
        is_s  = is_s[0] if len(is_s) > 0 else np.nan
        oos_s = oos_s[0] if len(oos_s) > 0 else np.nan

        short_name = strat.replace("_", " ").replace("Long","L").replace("Short","S")
        ax.set_title(f"{short_name}\nIS Sharpe={is_s:+.2f}  OOS={oos_s:+.2f}", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        ax.set_ylabel("累積 bps", fontsize=8)
        ax.tick_params(axis="x", labelsize=7, rotation=20)

    for i in range(ax_idx, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("IS/OOS検証: イントラデイ時間帯・ランチリバーサル戦略 (往復8bps控除後)",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig("result.png", dpi=100, bbox_inches="tight", facecolor="white")
    print("\nresult.png 保存完了")

    # 昇格候補
    print("\n=== 昇格候補 (IS/OOSともSharpe>0.0) ===")
    for strat in strategies:
        sub = res_df[res_df["strategy"]==strat]
        if sub.empty:
            continue
        is_s  = sub[sub["period"]=="IS"]["sharpe"].values
        oos_s = sub[sub["period"]=="OOS"]["sharpe"].values
        if len(is_s)==0 or len(oos_s)==0:
            continue
        if is_s[0] > 0 and oos_s[0] > 0:
            print(f"  ★ {strat}: IS Sharpe={is_s[0]:+.2f}, OOS={oos_s[0]:+.2f}")
        else:
            print(f"    {strat}: IS={is_s[0]:+.2f}, OOS={oos_s[0]:+.2f}")

if __name__ == "__main__":
    main()
