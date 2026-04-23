#!/usr/bin/env python3
"""
F: 古河電工 (5803) の TWAP 方向推定
TWAP 執行痕跡が強い日に「買い or 売り」どちらだったかを peer 比較で推定する。

手法:
1. 日次の TWAP pulse = volume の lag-5 自己相関
2. 日次の peer-adjusted return = 5803 day_ret - peers 平均 day_ret
3. 日次の VWAP deviation = (close - VWAP) / VWAP  (執行者がいれば systematic bias)
4. pulse 強度別に上記 2 指標の符号分布を集計

peer = 同セクター (5713 住金/5711 三菱マテ/5802 住電/5801 古河+他)
"""
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import psycopg2

warnings.filterwarnings("ignore")
PG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}

TARGET = "5803.T"
PEERS = ["5713.T", "5711.T", "5802.T", "5801.T"]  # 5803 自身除く


def load(sym):
    conn = psycopg2.connect(**PG)
    df = pd.read_sql(
        f"SELECT timestamp,open,high,low,close,volume FROM intraday_data "
        f"WHERE symbol='{sym}' ORDER BY timestamp", conn)
    conn.close()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["jst"] = df["timestamp"] + pd.Timedelta(hours=9)
    df = df.set_index("jst").sort_index().dropna(subset=["open"])
    h = df.index.hour; m = df.index.minute
    mask = ((h == 9) | (h == 10) |
            ((h == 11) & (m <= 30)) |
            ((h == 12) & (m >= 30)) |
            (h == 13) | (h == 14) |
            ((h == 15) & (m <= 30)))
    return df[mask]


def daily_metrics(df):
    """1 銘柄の日次指標を出す"""
    rows = []
    for d, g in df.groupby(df.index.date):
        if len(g) < 60: continue
        day_ret = (g["close"].iloc[-1] / g["open"].iloc[0] - 1) * 1e4  # bps
        # TWAP pulse
        v = g["volume"].values
        v0 = v[:-5]; v1 = v[5:]
        pulse = np.corrcoef(v0, v1)[0, 1] if len(v0) > 10 and np.std(v0) > 0 and np.std(v1) > 0 else np.nan
        # VWAP & deviation at close
        tp = (g["high"] + g["low"] + g["close"]) / 3
        vwap = (tp * g["volume"]).sum() / g["volume"].sum()
        close_vwap_dev = (g["close"].iloc[-1] / vwap - 1) * 1e4  # bps
        # 午後の drift (12:30 以降の動き — TWAP 執行は午後にも続く)
        pm = g.between_time("12:30", "15:30")
        pm_ret = (pm["close"].iloc[-1] / pm["close"].iloc[0] - 1) * 1e4 if len(pm) > 5 else np.nan
        rows.append({
            "date": pd.Timestamp(d),
            "day_ret": day_ret,
            "pulse": pulse,
            "close_vwap_dev": close_vwap_dev,
            "pm_ret": pm_ret,
            "volume": g["volume"].sum(),
        })
    return pd.DataFrame(rows).set_index("date")


def main():
    print("=" * 110)
    print("F. 古河電工 (5803) TWAP 執行方向推定")
    print("=" * 110)

    m_target = daily_metrics(load(TARGET))
    m_target.columns = [f"5803_{c}" for c in m_target.columns]
    print(f"  5803: {len(m_target)} 日")

    peers_df = {}
    for p in PEERS:
        m = daily_metrics(load(p))
        peers_df[p] = m["day_ret"]
        print(f"  {p}: {len(m)} 日")

    peer_ret = pd.DataFrame(peers_df).mean(axis=1).rename("peer_ret")

    # 結合
    df = m_target.join(peer_ret, how="inner").dropna()
    df["peer_adj_ret"] = df["5803_day_ret"] - df["peer_ret"]
    print(f"  共通: {len(df)} 日")

    # ─────────────────────────────────────────────
    print("\n" + "=" * 110)
    print("【1. TWAP pulse 強度別 5803 peer-adjusted return 分布】")
    print("=" * 110)
    df["pulse_bucket"] = pd.qcut(df["5803_pulse"], 4, labels=["Q1_弱", "Q2", "Q3", "Q4_強"])
    print(f"  {'bucket':<12} {'N':>5} {'mean':>8} {'median':>8} {'sd':>8} "
          f"{'>0 %':>7} {'close-VWAP mean':>18}")
    for b in ["Q1_弱", "Q2", "Q3", "Q4_強"]:
        sub = df[df["pulse_bucket"] == b]
        if len(sub) < 5: continue
        pa = sub["peer_adj_ret"]
        cv = sub["5803_close_vwap_dev"]
        print(f"  {b:<12} {len(sub):>5} {pa.mean():>+8.1f} {pa.median():>+8.1f} {pa.std():>8.1f} "
              f"{(pa > 0).mean()*100:>6.1f}% {cv.mean():>+18.1f}")

    # ─────────────────────────────────────────────
    print("\n" + "=" * 110)
    print("【2. 高 pulse 日 (top 20%) の詳細】")
    print("=" * 110)
    thr = df["5803_pulse"].quantile(0.80)
    hi = df[df["5803_pulse"] >= thr].copy()
    print(f"  閾値: pulse >= {thr:.3f}  該当 {len(hi)} 日")
    print(f"  peer_adj_ret 統計:")
    pa = hi["peer_adj_ret"]
    print(f"    mean      {pa.mean():+8.2f} bps")
    print(f"    median    {pa.median():+8.2f} bps")
    print(f"    t-stat    {pa.mean()/(pa.std()/np.sqrt(len(pa))):+8.2f}")
    print(f"    >0 比率   {(pa > 0).mean()*100:5.1f}%")
    print(f"  close-VWAP deviation 統計:")
    cv = hi["5803_close_vwap_dev"]
    print(f"    mean      {cv.mean():+8.2f} bps")
    print(f"    >0 比率   {(cv > 0).mean()*100:5.1f}%")

    # 方向判定
    print("\n" + "=" * 110)
    print("【結論】")
    print("=" * 110)
    pa_mean = hi["peer_adj_ret"].mean()
    cv_mean = hi["close_vwap_dev" if "close_vwap_dev" in hi.columns else "5803_close_vwap_dev"].mean()
    if pa_mean > 10 and cv_mean > 0:
        verdict = "🟢 買い方 TWAP 支配 (peer 比上昇 + close > VWAP)"
    elif pa_mean < -10 and cv_mean < 0:
        verdict = "🔴 売り方 TWAP 支配 (peer 比下落 + close < VWAP)"
    elif abs(pa_mean) < 5 and abs(cv_mean) < 5:
        verdict = "⚪ 両方向混在 (単一大口ではなく、複数参加者の TWAP が交差)"
    else:
        verdict = f"🟡 弱いシグナル (peer_adj={pa_mean:+.1f}, cv={cv_mean:+.1f})"
    print(f"  {verdict}")

    # ─────────────────────────────────────────────
    print("\n" + "=" * 110)
    print("【3. 月別 pulse 方向 — 買い方/売り方の時期シフトを確認】")
    print("=" * 110)
    df["ym"] = df.index.to_period("M")
    monthly = df.groupby("ym").agg(
        n=("5803_pulse", "count"),
        pulse_mean=("5803_pulse", "mean"),
        peer_adj=("peer_adj_ret", "mean"),
        vwap_dev=("5803_close_vwap_dev", "mean"),
    )
    print(monthly.round(2).to_string())

    # 保存
    df.to_csv("twap_direction_5803.csv")
    monthly.to_csv("twap_direction_5803_monthly.csv")
    print("\n→ twap_direction_5803.csv / twap_direction_5803_monthly.csv 保存")


if __name__ == "__main__":
    main()
