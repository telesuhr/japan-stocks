#!/usr/bin/env python3
"""
G: 日中リアルタイム TWAP 方向判定 (5803)
前場 (9:00-11:30) データのみで午後 (12:30-15:30) の peer 対比方向を予測できるか検証。

アイデア:
  TWAP 執行は日中を通して継続する → 午前の footprint が午後を予測できるはず

前場特徴量:
  am_pulse         : 午前 volume の lag-5 自己相関
  am_peer_adj      : 午前の peer 対比リターン (5803 am_ret - peers am_ret)
  am_close_vs_vwap : 午前終了時点の close と午前 VWAP の差
  am_volume_rank   : 午前出来高の銘柄内 z-score (出来高異常日検出)
  am_drift         : 午前 5 分足 return の累積 / sqrt(時間)

ターゲット:
  pm_peer_adj      : 午後 (12:30 開始値 → 15:30 close) の peer 対比リターン
"""
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import psycopg2

warnings.filterwarnings("ignore")
PG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}

TARGET = "5803.T"
PEERS = ["5713.T", "5711.T", "5802.T", "5801.T"]


def load(sym):
    conn = psycopg2.connect(**PG)
    df = pd.read_sql(
        f"SELECT timestamp,open,high,low,close,volume FROM intraday_data "
        f"WHERE symbol='{sym}' ORDER BY timestamp", conn)
    conn.close()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["jst"] = df["timestamp"] + pd.Timedelta(hours=9)
    return df.set_index("jst").sort_index().dropna(subset=["open"])


def features(df):
    """1 銘柄の日次特徴量 (午前) とターゲット (午後)"""
    rows = []
    for d, g in df.groupby(df.index.date):
        am = g.between_time("09:00", "11:30")
        pm = g.between_time("12:30", "15:30")
        if len(am) < 60 or len(pm) < 60: continue
        # 午前特徴量
        vam = am["volume"].values
        v0 = vam[:-5]; v1 = vam[5:]
        am_pulse = (np.corrcoef(v0, v1)[0, 1]
                    if len(v0) > 10 and np.std(v0) > 0 and np.std(v1) > 0 else np.nan)
        am_ret = (am["close"].iloc[-1] / am["open"].iloc[0] - 1) * 1e4  # bps
        tp = (am["high"] + am["low"] + am["close"]) / 3
        am_vwap = (tp * am["volume"]).sum() / am["volume"].sum()
        am_close_vs_vwap = (am["close"].iloc[-1] / am_vwap - 1) * 1e4
        # 午後ターゲット
        pm_ret = (pm["close"].iloc[-1] / pm["open"].iloc[0] - 1) * 1e4
        day_ret = (g["close"].iloc[-1] / g["open"].iloc[0] - 1) * 1e4
        rows.append({
            "date": pd.Timestamp(d),
            "am_pulse": am_pulse,
            "am_ret": am_ret,
            "am_close_vs_vwap": am_close_vs_vwap,
            "am_volume": am["volume"].sum(),
            "pm_ret": pm_ret,
            "day_ret": day_ret,
        })
    return pd.DataFrame(rows).set_index("date")


def main():
    print("=" * 110)
    print("G. 前場特徴量 → 午後方向予測")
    print("=" * 110)

    # 5803
    f_target = features(load(TARGET))
    # 出来高 z-score (過去 20 日 rolling)
    f_target["am_vol_z"] = ((f_target["am_volume"] - f_target["am_volume"].rolling(20).mean())
                            / f_target["am_volume"].rolling(20).std())
    f_target.columns = [f"t_{c}" if c != "am_vol_z" else "am_vol_z" for c in f_target.columns]

    # peers 午前/午後リターン
    peer_am, peer_pm = {}, {}
    for p in PEERS:
        fp = features(load(p))
        peer_am[p] = fp["am_ret"]
        peer_pm[p] = fp["pm_ret"]

    p_am = pd.DataFrame(peer_am).mean(axis=1)
    p_pm = pd.DataFrame(peer_pm).mean(axis=1)

    df = f_target.join(p_am.rename("peer_am_ret")).join(p_pm.rename("peer_pm_ret")).dropna()
    df["am_peer_adj"] = df["t_am_ret"] - df["peer_am_ret"]
    df["pm_peer_adj"] = df["t_pm_ret"] - df["peer_pm_ret"]

    # bad tick 除外 (|pm_peer_adj| > 2000 bps は明らかな外れ値)
    n_before = len(df)
    df = df[df["pm_peer_adj"].abs() < 2000]
    print(f"  有効日: {len(df)} (bad tick {n_before-len(df)} 日除外)")

    # ─────────────────────────────────────────────
    print("\n" + "=" * 110)
    print("【1. 前場特徴量 vs 午後 peer_adj_ret の Pearson 相関】")
    print("=" * 110)
    feats = ["t_am_pulse", "am_peer_adj", "t_am_close_vs_vwap", "am_vol_z"]
    for f in feats:
        x = df[f]; y = df["pm_peer_adj"]
        mk = x.notna() & y.notna()
        c = x[mk].corr(y[mk])
        # t-stat
        n = mk.sum()
        t = c * np.sqrt(n - 2) / np.sqrt(1 - c * c) if abs(c) < 1 else np.inf
        print(f"  {f:<25} ρ = {c:+.4f}  (N={n}, t={t:+.2f})")

    # ─────────────────────────────────────────────
    print("\n" + "=" * 110)
    print("【2. am_peer_adj 符号別 午後 peer_adj_ret】")
    print("=" * 110)
    print(f"  {'ケース':<25} {'N':>4} {'mean':>9} {'median':>9} {'>0 比率':>9} {'Sh(bps)':>9}")
    for label, mask in [
        ("am_peer_adj > +30 bps", df["am_peer_adj"] > 30),
        ("am_peer_adj > 0", df["am_peer_adj"] > 0),
        ("am_peer_adj < 0", df["am_peer_adj"] < 0),
        ("am_peer_adj < -30 bps", df["am_peer_adj"] < -30),
    ]:
        sub = df[mask]
        if len(sub) < 5: continue
        pm = sub["pm_peer_adj"]
        sh = pm.mean() / pm.std() * np.sqrt(252) if pm.std() > 0 else 0
        print(f"  {label:<25} {len(sub):>4} {pm.mean():>+9.2f} {pm.median():>+9.2f} "
              f"{(pm>0).mean()*100:>8.1f}% {sh:>+9.2f}")

    # ─────────────────────────────────────────────
    print("\n" + "=" * 110)
    print("【3. 複合シグナル: am_pulse 強 かつ am_peer_adj 大】")
    print("=" * 110)
    print(f"  {'ケース':<45} {'N':>4} {'mean':>9} {'t-stat':>8} {'>0 %':>8}")
    pulse_hi = df["t_am_pulse"] > df["t_am_pulse"].median()
    for label, mask in [
        ("pulse 強 × am_peer_adj > +30", pulse_hi & (df["am_peer_adj"] > 30)),
        ("pulse 強 × am_peer_adj < -30", pulse_hi & (df["am_peer_adj"] < -30)),
        ("pulse 強 × am_close > am_VWAP", pulse_hi & (df["t_am_close_vs_vwap"] > 0)),
        ("pulse 強 × am_close < am_VWAP", pulse_hi & (df["t_am_close_vs_vwap"] < 0)),
    ]:
        sub = df[mask]
        if len(sub) < 5: continue
        pm = sub["pm_peer_adj"]
        t = pm.mean() / (pm.std() / np.sqrt(len(pm))) if pm.std() > 0 else 0
        print(f"  {label:<45} {len(sub):>4} {pm.mean():>+9.2f} {t:>+8.2f} "
              f"{(pm>0).mean()*100:>7.1f}%")

    # ─────────────────────────────────────────────
    print("\n" + "=" * 110)
    print("【4. 簡易戦略バックテスト: 午前シグナルで午後方向 bet】")
    print("=" * 110)
    # ルール: |am_peer_adj| > 30 かつ am_pulse > median  → 同方向で午後エントリー
    sig_mask = (df["am_peer_adj"].abs() > 30) & pulse_hi
    signals = df[sig_mask].copy()
    signals["direction"] = np.sign(signals["am_peer_adj"])
    signals["pnl_peer_adj"] = signals["direction"] * signals["pm_peer_adj"]
    signals["pnl_raw"] = signals["direction"] * signals["t_pm_ret"]
    print(f"  シグナル日: {len(signals)} / {len(df)} days")
    print(f"\n  peer-adjusted PnL (5803 - peers):")
    a = signals["pnl_peer_adj"]
    sh = a.mean()/a.std()*np.sqrt(252) if a.std() > 0 else 0
    print(f"    mean {a.mean():+6.2f} bps  std {a.std():5.1f}  "
          f"Sh {sh:+5.2f}  WR {(a>0).mean()*100:.1f}%  sum {a.sum():+6.0f}")
    print(f"\n  raw PnL (5803 午後リターン × direction):")
    a = signals["pnl_raw"]
    sh = a.mean()/a.std()*np.sqrt(252) if a.std() > 0 else 0
    print(f"    mean {a.mean():+6.2f} bps  std {a.std():5.1f}  "
          f"Sh {sh:+5.2f}  WR {(a>0).mean()*100:.1f}%  sum {a.sum():+6.0f}")

    # ─────────────────────────────────────────────
    # 追加: より細かい閾値スキャン
    print("\n" + "=" * 110)
    print("【5. 閾値スキャン: am_peer_adj 閾値を変えて Sharpe がどう変わるか】")
    print("=" * 110)
    print(f"  {'threshold(bps)':<15} {'N_sig':>6} {'pnl_mean':>10} {'Sh':>7} {'WR%':>7}")
    for th in [10, 20, 30, 50, 80, 100, 150]:
        m = (df["am_peer_adj"].abs() > th) & pulse_hi
        s = df[m].copy()
        if len(s) < 20: continue
        s["dir"] = np.sign(s["am_peer_adj"])
        pnl = s["dir"] * s["pm_peer_adj"]
        sh = pnl.mean()/pnl.std()*np.sqrt(252) if pnl.std() > 0 else 0
        print(f"  {th:<15} {len(s):>6} {pnl.mean():>+10.2f} {sh:>+7.2f} "
              f"{(pnl>0).mean()*100:>6.1f}")

    df.to_csv("intraday_direction_5803.csv")
    print("\n→ intraday_direction_5803.csv 保存")


if __name__ == "__main__":
    main()
