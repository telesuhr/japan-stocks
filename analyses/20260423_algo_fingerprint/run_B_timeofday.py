#!/usr/bin/env python3
"""
B: 時間帯別アルゴ・レジーム検出
取引時間を 4 セッションに区切り、各セッションで指標を再計算する。
   S1: 09:00-09:30  (寄付後の機関執行)
   S2: 09:30-11:30  (前場通常)
   S3: 12:30-14:30  (後場通常)
   S4: 14:30-15:30  (引け前 + MOC)
"""
import warnings
import numpy as np
import pandas as pd
import psycopg2

warnings.filterwarnings("ignore")
PG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}

SYMS = ["8035.T", "6857.T", "6963.T", "6526.T",
        "5713.T", "5802.T", "5711.T", "5803.T"]
INDEX = ".TOPX"

SESSIONS = {
    "S1_寄付": ("09:00", "09:30"),
    "S2_前場": ("09:30", "11:30"),
    "S3_後場": ("12:30", "14:30"),
    "S4_引け": ("14:30", "15:30"),
}


def load(sym):
    conn = psycopg2.connect(**PG)
    df = pd.read_sql(
        f"SELECT timestamp,open,high,low,close,volume FROM intraday_data "
        f"WHERE symbol='{sym}' ORDER BY timestamp", conn)
    conn.close()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["jst"] = df["timestamp"] + pd.Timedelta(hours=9)
    return df.set_index("jst").sort_index().dropna(subset=["open"])


def slice_session(df, start, end):
    return df.between_time(start, end)


def hft_mr(df):
    r = df["close"].pct_change()
    r = r[r.abs() < 0.05].dropna()
    if len(r) < 100: return np.nan
    return r.autocorr(lag=1)


def kyle_lam(df):
    r = df["close"].pct_change().abs()
    v = df["volume"].replace(0, np.nan)
    lam = (r / np.sqrt(v)).dropna()
    lam = lam[lam < lam.quantile(0.99)]
    return lam.median() * 1e6


def false_break(df, lookback=20, forward=5):
    roll = df["high"].rolling(lookback).max().shift(1)
    broke = df["close"] > roll
    idx = np.where(broke.values)[0]
    fl = []
    for i in idx:
        if i + forward >= len(df) or i < 1: continue
        lv = roll.iloc[i]
        if pd.isna(lv): continue
        if df["low"].iloc[i + 1:i + 1 + forward].min() < lv:
            fl.append(1)
        else:
            fl.append(0)
    return np.mean(fl) if fl else np.nan


def large_persist(df, fwd=10):
    q = df["volume"].quantile(0.95)
    r = df["close"].pct_change()
    m = []
    big_idx = np.where(df["volume"].values >= q)[0]
    for i in big_idx:
        if i + fwd >= len(df) or i < 1: continue
        r0 = r.iloc[i]
        if pd.isna(r0) or r0 == 0: continue
        fwd_r = df["close"].iloc[i + fwd] / df["close"].iloc[i] - 1
        m.append(int(np.sign(r0) == np.sign(fwd_r)))
    return np.mean(m) if m else np.nan


def realized_vol(df):
    r = df["close"].pct_change().dropna()
    return r.std() * 1e4  # bps


def vol_share(df_sess, df_full):
    total_per_day = df_full.groupby(df_full.index.date)["volume"].sum()
    sess_per_day = df_sess.groupby(df_sess.index.date)["volume"].sum()
    share = (sess_per_day / total_per_day).dropna()
    return share.median()


def index_corr(df, idx_df):
    def r5(d): return d["close"].resample("5min").last().dropna().pct_change().dropna()
    rs = []
    for d, g in df.groupby(df.index.date):
        ig = idx_df[idx_df.index.date == d]
        if len(g) < 10 or len(ig) < 10: continue
        ra = r5(g); rb = r5(ig)
        a = pd.concat([ra, rb], axis=1, join="inner").dropna()
        if len(a) < 4: continue
        rs.append(a.iloc[:, 0].corr(a.iloc[:, 1]))
    return np.nanmean(rs) if rs else np.nan


def main():
    print("=" * 140)
    print("B. 時間帯別アルゴ・レジーム")
    print("=" * 140)
    idx = load(INDEX)

    metrics_labels = ["出来高シェア%", "HFT-MR(ρ1)", "Kyle λ", "False BO率",
                      "大口後持続", "TOPIX連動", "RealVol(bps)"]

    all_results = {}

    for sym in SYMS:
        df = load(sym)
        print(f"\n── {sym} ({len(df):,} bars) ──")
        print(f"  {'Session':<12}" + "".join(f"{c:>13}" for c in metrics_labels))
        result_rows = []
        for label, (s, e) in SESSIONS.items():
            sub = slice_session(df, s, e)
            if len(sub) == 0:
                result_rows.append([np.nan] * len(metrics_labels)); continue
            idx_sub = slice_session(idx, s, e)
            row = [
                vol_share(sub, df) * 100,
                hft_mr(sub),
                kyle_lam(sub),
                false_break(sub),
                large_persist(sub, fwd=min(10, max(3, len(sub) // 100))),
                index_corr(sub, idx_sub),
                realized_vol(sub),
            ]
            result_rows.append(row)
            print(f"  {label:<12}" +
                  "".join(f"{v:>13.3f}" if not pd.isna(v) else f"{'NA':>13}" for v in row))
        all_results[sym] = result_rows

    # 平均プロファイル
    print("\n" + "=" * 140)
    print("【時間帯別 8 銘柄平均】")
    print("=" * 140)
    print(f"  {'Session':<12}" + "".join(f"{c:>13}" for c in metrics_labels))
    avgs = {}
    for i, label in enumerate(SESSIONS.keys()):
        stack = np.array([all_results[s][i] for s in SYMS], dtype=float)
        av = np.nanmean(stack, axis=0)
        avgs[label] = av
        print(f"  {label:<12}" + "".join(f"{v:>13.3f}" for v in av))

    # レジーム自動判定
    print("\n" + "=" * 140)
    print("【時間帯別 レジーム診断 (平均値ベース)】")
    print("=" * 140)
    for label in SESSIONS.keys():
        a = avgs[label]
        vol_sh, hft, kyle, fb, lp, ic, rv = a
        diag = []
        if vol_sh > 20: diag.append(f"出来高集中({vol_sh:.1f}%)")
        if ic > 0.5: diag.append(f"指数連動強({ic:.2f})")
        elif ic < 0.35: diag.append(f"指数連動弱({ic:.2f})")
        if hft < -0.02: diag.append(f"MR/HFT痕跡({hft:+.3f})")
        elif hft > 0.02: diag.append(f"モメンタム({hft:+.3f})")
        if fb > 0.65: diag.append(f"だまし多({fb*100:.0f}%)")
        elif fb < 0.55: diag.append(f"だまし少({fb*100:.0f}%)")
        if lp > 0.53: diag.append(f"大口トレンド({lp:.2f})")
        elif lp < 0.47: diag.append(f"大口リバーサル({lp:.2f})")
        if rv > 15: diag.append(f"高ボラ({rv:.1f}bps)")
        elif rv < 8: diag.append(f"低ボラ({rv:.1f}bps)")
        print(f"  {label:<12} → {' / '.join(diag) if diag else '中性'}")

    # セッション間比較 CSV
    dfs = {}
    for i, label in enumerate(SESSIONS.keys()):
        dfs[label] = pd.DataFrame(
            {sym: all_results[sym][i] for sym in SYMS}, index=metrics_labels).T
    with pd.ExcelWriter("tod_regime.xlsx") as writer:
        for k, v in dfs.items():
            v.to_csv(f"tod_{k}.csv")
    print("\n→ tod_S*.csv 保存")


if __name__ == "__main__":
    main()
