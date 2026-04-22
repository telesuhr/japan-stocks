#!/usr/bin/env python3
"""
A: アルゴ指紋プロファイリング
1 分 OHLCV + TOPIX から、銘柄別・全期間での「アルゴ痕跡」指標を 10 種計算。
"""
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import psycopg2

warnings.filterwarnings("ignore")
PG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}

SYMS = ["8035.T", "6857.T", "6963.T", "6526.T",
        "5713.T", "5802.T", "5711.T", "5803.T"]
INDEX = ".TOPX"


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


# =========================================================
# 指標 1: VWAP 引き寄せ度
#  日中 VWAP からの残差 z = (close - VWAP) / σ の 1 分 AR(1) 係数
#  負が強いほど → MR (VWAP 回帰アルゴ支配)
# =========================================================
def vwap_pull(df):
    coefs = []
    for d, g in df.groupby(df.index.date):
        if len(g) < 60: continue
        tp = (g["high"] + g["low"] + g["close"]) / 3
        cum_tpv = (tp * g["volume"]).cumsum()
        cum_v = g["volume"].cumsum().replace(0, np.nan)
        vwap = cum_tpv / cum_v
        resid = (g["close"] - vwap) / g["close"].std()
        r = resid.dropna()
        if len(r) < 30: continue
        # AR(1)
        r1 = r.iloc[1:].values
        r0 = r.iloc[:-1].values
        if np.std(r0) == 0: continue
        rho = np.corrcoef(r0, r1)[0, 1]
        coefs.append(rho)
    return np.nanmean(coefs) if coefs else np.nan


# =========================================================
# 指標 2: TWAP 出来高パルス
#  volume の lag-5 自己相関 (分単位の周期性)
# =========================================================
def twap_pulse(df, lag=5):
    vols = []
    for d, g in df.groupby(df.index.date):
        v = g["volume"].values
        if len(v) < 60: continue
        v0 = v[:-lag]; v1 = v[lag:]
        if np.std(v0) == 0 or np.std(v1) == 0: continue
        vols.append(np.corrcoef(v0, v1)[0, 1])
    return np.nanmean(vols) if vols else np.nan


# =========================================================
# 指標 3: 引け集中度
#   14:55-15:00 の 5 分 vol / 日中 5 分平均 vol
# =========================================================
def close_concentration(df):
    rs = []
    for d, g in df.groupby(df.index.date):
        if len(g) < 60: continue
        last = g.between_time("14:55", "15:30")["volume"].sum()
        avg5 = g["volume"].sum() / (len(g) / 5)
        if avg5 > 0:
            rs.append(last / avg5)
    return np.nanmedian(rs) if rs else np.nan


# =========================================================
# 指標 4: 寄付集中度
#   9:00-9:15 の vol / 日中 15 分平均
# =========================================================
def open_concentration(df):
    rs = []
    for d, g in df.groupby(df.index.date):
        if len(g) < 60: continue
        first = g.between_time("09:00", "09:15")["volume"].sum()
        avg15 = g["volume"].sum() / (len(g) / 15)
        if avg15 > 0:
            rs.append(first / avg15)
    return np.nanmedian(rs) if rs else np.nan


# =========================================================
# 指標 5: インデックス連動度
#   5 分 return と TOPIX 5 分 return の rolling 30 分 相関の平均
# =========================================================
def index_link(df, idx_df):
    def r5(d):
        c = d["close"].resample("5min").last().dropna()
        return c.pct_change().dropna()
    rs = []
    for d, g in df.groupby(df.index.date):
        ig = idx_df[idx_df.index.date == d]
        if len(g) < 30 or len(ig) < 30: continue
        ra = r5(g); rb = r5(ig)
        aligned = pd.concat([ra, rb], axis=1, join="inner").dropna()
        if len(aligned) < 10: continue
        rs.append(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
    return np.nanmean(rs) if rs else np.nan


# =========================================================
# 指標 6: HFT-MM 痕跡
#   1 分リターンの AR(1). 負が強い = micro-MR (マーケットメイクが強い)
# =========================================================
def hft_mr(df):
    rets = df["close"].pct_change()
    rets = rets[rets.abs() < 0.05].dropna()  # 異常値除外
    if len(rets) < 100: return np.nan
    return rets.autocorr(lag=1)


# =========================================================
# 指標 7: Kyle's λ (価格インパクト)
#   |ret| / sqrt(volume) の中央値 (日内時間加重で評価)
#   小さいほど深い市場 (大口が捌ける)
# =========================================================
def kyle_lambda(df):
    ret = df["close"].pct_change().abs()
    vol = df["volume"].replace(0, np.nan)
    lam = (ret / np.sqrt(vol)).dropna()
    lam = lam[lam < lam.quantile(0.99)]
    return lam.median() * 1e6  # 倍率調整


# =========================================================
# 指標 8: false break 率 (ストップ狩り痕跡)
#   20 本ブレイク後 5 本以内に元のレンジに戻る割合
# =========================================================
def false_break_rate(df, lookback=20, forward=5):
    rolls = df["high"].rolling(lookback).max().shift(1)
    broke = df["close"] > rolls
    idx = np.where(broke.values)[0]
    fl = []
    for i in idx:
        if i + forward >= len(df) or i < 1: continue
        lv = rolls.iloc[i]
        if pd.isna(lv): continue
        fwd_low = df["low"].iloc[i + 1:i + 1 + forward].min()
        fl.append(fwd_low < lv)
    return np.mean(fl) if fl else np.nan


# =========================================================
# 指標 9: 大出来高後の trend persistence
#   volume top 5% のバーの ret と次 10 分 ret の符号一致率
# =========================================================
def large_trade_persistence(df, forward=10):
    q = df["volume"].quantile(0.95)
    big = df[df["volume"] >= q]
    ret = df["close"].pct_change()
    matches = []
    for ts in big.index:
        try:
            pos = df.index.get_loc(ts)
        except KeyError: continue
        if isinstance(pos, slice): pos = pos.start
        if pos + forward >= len(df) or pos < 1: continue
        r0 = ret.iloc[pos]
        fwd = (df["close"].iloc[pos + forward] / df["close"].iloc[pos] - 1)
        if pd.isna(r0) or pd.isna(fwd) or r0 == 0: continue
        matches.append(np.sign(r0) == np.sign(fwd))
    return np.mean(matches) if matches else np.nan


# =========================================================
# 指標 10: MOC マーク度
#   last 1 min ret と 日中 ret の符号一致率
# =========================================================
def moc_mark(df):
    matches = []
    for d, g in df.groupby(df.index.date):
        if len(g) < 10: continue
        day_ret = g["close"].iloc[-1] / g["open"].iloc[0] - 1
        last_ret = g["close"].iloc[-1] / g["close"].iloc[-2] - 1 if len(g) > 1 else 0
        if day_ret == 0 or last_ret == 0: continue
        matches.append(np.sign(day_ret) == np.sign(last_ret))
    return np.mean(matches) if matches else np.nan


# =========================================================
def main():
    print("=" * 120)
    print("A. アルゴ指紋プロファイリング")
    print("=" * 120)
    idx_df = load(INDEX)
    print(f"  {INDEX}: {len(idx_df):,} bars loaded")

    metrics = ["VWAP引寄(ρ)", "TWAP出来高(ρ)", "引け集中", "寄付集中",
               "TOPIX連動(ρ)", "HFT-MR(ρ1)", "Kyle λ×1e6",
               "False BO率", "大口後持続", "MOCマーク率"]
    rows = {}

    for sym in SYMS:
        df = load(sym)
        if len(df) == 0:
            print(f"  ✗ {sym}: no data"); continue
        r = [
            vwap_pull(df),
            twap_pulse(df),
            close_concentration(df),
            open_concentration(df),
            index_link(df, idx_df),
            hft_mr(df),
            kyle_lambda(df),
            false_break_rate(df),
            large_trade_persistence(df),
            moc_mark(df),
        ]
        rows[sym] = r
        print(f"  ✓ {sym}: {len(df):,} bars")

    print("\n" + "=" * 120)
    df_out = pd.DataFrame(rows, index=metrics).T
    print(df_out.round(4).to_string())

    # z-score 正規化 (銘柄横断)
    print("\n" + "=" * 120)
    print("z-score 正規化 (各指標について銘柄横断で)  — 正 = その銘柄でその痕跡が強い")
    print("=" * 120)
    zdf = (df_out - df_out.mean()) / df_out.std()
    print(zdf.round(2).to_string())

    # 解釈支援: 各銘柄でトップ 3 の痕跡
    print("\n" + "=" * 120)
    print("各銘柄の支配アルゴ TOP3 (z-score 絶対値)")
    print("=" * 120)
    for sym in zdf.index:
        row = zdf.loc[sym]
        top = row.abs().sort_values(ascending=False).head(3)
        parts = []
        for m in top.index:
            sign = "+" if row[m] > 0 else "-"
            parts.append(f"{sign}{m}={row[m]:+.2f}")
        print(f"  {sym}: {'  |  '.join(parts)}")

    df_out.to_csv("fingerprint_raw.csv")
    zdf.to_csv("fingerprint_zscore.csv")
    print("\n→ fingerprint_raw.csv / fingerprint_zscore.csv 保存")


if __name__ == "__main__":
    main()
