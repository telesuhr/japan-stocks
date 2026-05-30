"""
JPY急変 × セクター別翌日イントラパターン
仮説: 前日の円安急進 / 円高急進は翌日の特定セクターの日中リターンに非対称なパターンを作る
シグナル: JPY= (USD/JPY, 上昇=円安), EURJPY=
対象セクター:
  - 輸出株（自動車: 72030/72670, 電機系 ADR）
  - 内需株（小売・不動産）
  - 非鉄（LME代替としてFCX連動）
期間: stocks_intraday あり (2024-05〜), 日足は長期
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

# セクター定義（5桁コード）
SECTORS = {
    "自動車": ["72030", "72670", "72610"],          # トヨタ・本田・スズキ
    "精密": ["74510", "77410"],                     # キヤノン・シチズン
    "非鉄": ["57060","57110","57130","57140","50160","58010","58020","58030"],
    "半導体": ["69201","69541","68572","68472","30346","80350"],
    "内需小売": ["97830","29130","83010"],           # ヨドバシ、ファミマ、三越
}

ALL_CODES = list(set(c for codes in SECTORS.values() for c in codes))

INTRA_ST = "2024-05-01"
INTRA_END= "2026-05-31"
DAILY_ST = "2022-01-01"
IS_END   = "2025-01-01"
OOS_ST   = "2025-01-01"
COST     = 8.0

def get_conn():
    return psycopg2.connect(**PG_CONFIG)

def load_macro():
    conn = get_conn()
    sql = """
        SELECT symbol, trade_date, close
        FROM macro.daily_ohlcv
        WHERE symbol IN ('JPY=', 'EURJPY=', 'AUD=', 'FCX', '.SOX', 'NQc1', 'VXc1', 'HGc1')
          AND trade_date >= %s AND trade_date <= %s
        ORDER BY symbol, trade_date
    """
    df = pd.read_sql(sql, conn, params=(DAILY_ST, INTRA_END))
    conn.close()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["ret"] = df.groupby("symbol")["close"].pct_change() * 100
    pivot = df.pivot(index="trade_date", columns="symbol", values="ret")
    return pivot

def load_intraday_slots(codes):
    """日次の各時間帯リターンを1分足から計算"""
    conn = get_conn()
    sql = """
        SELECT code, ts, open, close
        FROM stocks_intraday
        WHERE code = ANY(%s) AND ts >= %s AND ts < %s
        ORDER BY code, ts
    """
    df = pd.read_sql(sql, conn, params=(codes, INTRA_ST, INTRA_END))
    conn.close()
    df["ts"] = pd.to_datetime(df["ts"])
    df["date"] = df["ts"].dt.date

    slots = {
        "full":  (datetime.time(9,0),  datetime.time(15,30)),
        "am1":   (datetime.time(9,0),  datetime.time(9,30)),
        "am":    (datetime.time(9,0),  datetime.time(11,30)),
        "pm":    (datetime.time(12,30),datetime.time(15,30)),
        "cls":   (datetime.time(14,30),datetime.time(15,30)),
    }

    rows = []
    for (code, date), grp in df.groupby(["code","date"]):
        grp = grp.sort_values("ts")
        r = {"code": code, "date": pd.Timestamp(date)}
        for slot, (tf, tt) in slots.items():
            sub = grp[(grp["ts"].dt.time >= tf) & (grp["ts"].dt.time < tt)]
            if len(sub) < 2:
                r[slot] = np.nan
                continue
            entry = sub.iloc[0]["open"]
            ex    = sub.iloc[-1]["close"]
            r[slot] = (ex/entry - 1)*10000 if entry > 0 else np.nan
        rows.append(r)
    return pd.DataFrame(rows)

def summarize(series, cost=COST):
    s = series.dropna()
    n  = len(s)
    mu = s.mean() - cost
    se = s.std() / np.sqrt(n) if n > 1 else np.nan
    t  = mu / se if se and se > 0 else np.nan
    sh = mu / s.std() * np.sqrt(250) if s.std() > 0 else np.nan
    return {"n": n, "mean_net": mu, "t": t, "sharpe": sh}

def main():
    print("マクロデータ読み込み...")
    macro = load_macro()
    macro_shifted = macro.shift(1)  # 前日シグナル
    print(f"  {macro.shape}")

    print("1分足データ読み込み...")
    intra = load_intraday_slots(ALL_CODES)
    print(f"  intra: {len(intra):,} code-days")
    intra["sector"] = intra["code"].apply(
        lambda c: next((s for s, codes in SECTORS.items() if c in codes), "OTHER"))

    results = []

    # シグナル設定 (シグナル列, 閾値, 方向, 対象セクター, ターゲット時間帯)
    configs = [
        # JPY上昇=円安 → 輸出株Long
        ("JPY=",    1.0, +1, ["自動車","精密"],      ["full","am","pm","cls"], "円安1%→輸出Long"),
        ("JPY=",    1.5, +1, ["自動車","精密"],      ["full","am","cls"],       "円安1.5%→輸出Long"),
        # JPY下落=円高 → 輸出株Short
        ("JPY=",   -1.0, -1, ["自動車","精密"],      ["full","am","pm"],        "円高1%→輸出Short"),
        # 円安 → 非鉄逆(輸入コスト上昇でマイナス)
        ("JPY=",    1.0, +1, ["非鉄"],              ["full","am","pm"],        "円安1%→非鉄(コスト↑)"),
        # 円高 → 内需Good
        ("JPY=",   -1.0, -1, ["内需小売"],           ["full"],                  "円高1%→内需Long"),
        # SOX → 半導体
        (".SOX",    2.0, +1, ["半導体"],             ["full","am","pm","cls"], "SOX+2%→半導体Long"),
        (".SOX",   -2.0, -1, ["半導体"],             ["full","am","pm"],        "SOX-2%→半導体Short"),
        # FCX → 非鉄
        ("FCX",     2.0, +1, ["非鉄"],              ["full","am","pm","cls"], "FCX+2%→非鉄Long"),
        ("FCX",    -2.0, -1, ["非鉄"],              ["full","am","pm"],        "FCX-2%→非鉄Short"),
        # HGc1 (COMEX銅) → 非鉄
        ("HGc1",    1.5, +1, ["非鉄"],              ["full","am","pm","cls"], "COMEX銅+1.5%→非鉄Long"),
        # NQc1 + VIX低下 → 半導体
        ("NQc1",    1.5, +1, ["半導体"],             ["full","am","cls"],       "NQ+1.5%→半導体Long"),
        # AUD (commodity currency) → 非鉄
        ("AUD=",    1.0, +1, ["非鉄"],              ["full","am","cls"],       "AUD+1%→非鉄Long"),
    ]

    for sig, thresh, direc, sectors, slots, lbl in configs:
        if sig not in macro_shifted.columns:
            print(f"  SKIP: {sig} なし")
            continue

        # 対象銘柄のintraを等加重
        sector_codes = list(set(c for s in sectors for c in SECTORS.get(s,[])))
        sub_intra = intra[intra["code"].isin(sector_codes)].copy()
        sub_intra_grp = sub_intra.groupby("date")[slots].mean()

        # シグナルと結合
        sig_series = macro_shifted[sig]

        for slot in slots:
            panel = pd.DataFrame({
                "sig": sig_series,
                "ret": sub_intra_grp[slot] if slot in sub_intra_grp.columns else np.nan
            }).dropna()

            fired = panel[panel["sig"] * direc >= abs(thresh)]
            if len(fired) < 10:
                continue

            for period in ["ALL","IS","OOS"]:
                if period == "IS":
                    sub = fired[fired.index < IS_END]
                elif period == "OOS":
                    sub = fired[fired.index >= OOS_ST]
                else:
                    sub = fired
                s = summarize(sub["ret"] * direc)
                s.update({"label": lbl, "slot": slot, "period": period})
                results.append(s)

    res_df = pd.DataFrame(results)
    res_df.to_csv("jpy_summary.csv", index=False)
    print(f"集計完了: {len(res_df)} rows")

    # ---- グラフ ----
    labels = res_df["label"].unique().tolist()
    fig, axes = plt.subplots(3, 4, figsize=(18, 12), facecolor="white")
    axes = axes.flatten()

    for i, lbl in enumerate(labels[:len(axes)]):
        ax = axes[i]
        sub = res_df[(res_df["label"]==lbl) & (res_df["slot"]=="full")]
        if sub.empty:
            ax.text(0.5,0.5,"データなし",ha="center",va="center",transform=ax.transAxes)
            ax.set_title(lbl, fontsize=8)
            continue
        periods = ["IS","OOS","ALL"]
        x = np.arange(len(periods))
        mu_vals = [sub[sub["period"]==p]["mean_net"].values[0] if len(sub[sub["period"]==p])>0 else 0 for p in periods]
        t_vals  = [sub[sub["period"]==p]["t"].values[0] if len(sub[sub["period"]==p])>0 else 0 for p in periods]
        colors  = ["#1565C0","#F57F17","#555555"]
        bars = ax.bar(x, mu_vals, color=colors, alpha=0.8)
        for bar, tv, mu in zip(bars, t_vals, mu_vals):
            if abs(tv) >= 1.8:
                ax.text(bar.get_x()+bar.get_width()/2, mu+(0.3 if mu>=0 else -1), "*", ha="center", fontsize=12)
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels(periods, fontsize=8)
        ax.set_title(lbl, fontsize=8)
        ax.set_ylabel("net bps", fontsize=7)
        ax.grid(axis="y", alpha=0.3)
        n_all = sub[sub["period"]=="ALL"]["n"].values
        if len(n_all)>0:
            ax.text(0.98,0.98,f"N={n_all[0]}",transform=ax.transAxes,ha="right",va="top",fontsize=7,color="gray")

    for i in range(len(labels), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("JPY/FCX/SOX/HGc1 × セクター別翌日イントラ（往復8bps控除後, IS<2025 / OOS>=2025, *=|t|≥1.8）",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig("result.png", dpi=100, bbox_inches="tight", facecolor="white")
    print("result.png 保存完了")

    print("\n=== IS/OOS両方プラス（全日 full slot）===")
    for lbl in labels:
        sub = res_df[(res_df["label"]==lbl) & (res_df["slot"]=="full")]
        is_r  = sub[sub["period"]=="IS"]
        oos_r = sub[sub["period"]=="OOS"]
        if is_r.empty or oos_r.empty:
            continue
        is_sh  = is_r["sharpe"].values[0]; oos_sh = oos_r["sharpe"].values[0]
        is_t   = is_r["t"].values[0];      oos_t  = oos_r["t"].values[0]
        is_n   = is_r["n"].values[0];      oos_n  = oos_r["n"].values[0]
        mark = "★" if is_sh > 0 and oos_sh > 0 else " "
        print(f"  {mark} {lbl:40s}: IS Sh={is_sh:+.2f}(t={is_t:.2f},N={is_n})  OOS Sh={oos_sh:+.2f}(t={oos_t:.2f},N={oos_n})")

if __name__ == "__main__":
    main()
