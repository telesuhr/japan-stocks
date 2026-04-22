#!/usr/bin/env python3
"""
C: 20 銘柄アルゴ・クラスタリング
多様セクターの銘柄でアルゴ指紋を計算し、k-means で自動クラスタリング。
"""
import warnings
import numpy as np
import pandas as pd
import psycopg2
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")
PG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}

# 多様セクターから 20 銘柄
SYMS = [
    # 半導体・電機 (6)
    "8035.T", "6857.T", "6963.T", "6526.T", "6525.T", "6146.T",
    # 非鉄 (6)
    "5713.T", "5802.T", "5711.T", "5803.T", "5801.T", "5016.T",
    # 通信 (2)
    "9434.T", "9984.T",
    # 自動車 (2)
    "7203.T", "7267.T",
    # 商社 (2)
    "8058.T", "8031.T",
    # 銀行 (2)
    "8306.T", "8316.T",
]
INDEX = ".TOPX"


def load(sym):
    conn = psycopg2.connect(**PG)
    df = pd.read_sql(
        f"SELECT timestamp,open,high,low,close,volume FROM intraday_data "
        f"WHERE symbol='{sym}' ORDER BY timestamp", conn)
    conn.close()
    if len(df) == 0: return None
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


def compute_profile(df, idx_df):
    """10 指標の profile を 1 行で返す"""
    out = {}
    # 1. HFT-MR
    r = df["close"].pct_change(); r = r[r.abs() < 0.05].dropna()
    out["HFT_MR"] = r.autocorr(lag=1) if len(r) > 100 else np.nan
    # 2. Kyle λ
    rv = df["close"].pct_change().abs()
    v = df["volume"].replace(0, np.nan)
    lam = (rv / np.sqrt(v)).dropna()
    lam = lam[lam < lam.quantile(0.99)]
    out["Kyle_lam"] = lam.median() * 1e6
    # 3. 寄付集中
    rs = []
    for d, g in df.groupby(df.index.date):
        if len(g) < 60: continue
        first = g.between_time("09:00", "09:15")["volume"].sum()
        avg = g["volume"].sum() / (len(g) / 15)
        if avg > 0: rs.append(first / avg)
    out["Open_conc"] = np.nanmedian(rs) if rs else np.nan
    # 4. 引け集中
    rs = []
    for d, g in df.groupby(df.index.date):
        if len(g) < 60: continue
        last = g.between_time("14:55", "15:30")["volume"].sum()
        avg = g["volume"].sum() / (len(g) / 5)
        if avg > 0: rs.append(last / avg)
    out["Close_conc"] = np.nanmedian(rs) if rs else np.nan
    # 5. TWAP pulse
    vols = []
    for d, g in df.groupby(df.index.date):
        v = g["volume"].values
        if len(v) < 30: continue
        v0 = v[:-5]; v1 = v[5:]
        if np.std(v0) == 0 or np.std(v1) == 0: continue
        vols.append(np.corrcoef(v0, v1)[0, 1])
    out["TWAP_pulse"] = np.nanmean(vols) if vols else np.nan
    # 6. TOPIX link
    rs = []
    for d, g in df.groupby(df.index.date):
        ig = idx_df[idx_df.index.date == d]
        if len(g) < 30 or len(ig) < 30: continue
        ra = g["close"].resample("5min").last().dropna().pct_change().dropna()
        rb = ig["close"].resample("5min").last().dropna().pct_change().dropna()
        a = pd.concat([ra, rb], axis=1, join="inner").dropna()
        if len(a) < 10: continue
        rs.append(a.iloc[:, 0].corr(a.iloc[:, 1]))
    out["TOPIX_link"] = np.nanmean(rs) if rs else np.nan
    # 7. False BO 率
    roll = df["high"].rolling(20).max().shift(1)
    broke = df["close"] > roll
    idx_b = np.where(broke.values)[0]
    fl = []
    for i in idx_b:
        if i + 5 >= len(df) or i < 1: continue
        lv = roll.iloc[i]
        if pd.isna(lv): continue
        fl.append(int(df["low"].iloc[i + 1:i + 6].min() < lv))
    out["FalseBO"] = np.mean(fl) if fl else np.nan
    # 8. Large persist
    q = df["volume"].quantile(0.95)
    rr = df["close"].pct_change()
    m = []
    for i in np.where(df["volume"].values >= q)[0]:
        if i + 10 >= len(df) or i < 1: continue
        r0 = rr.iloc[i]
        if pd.isna(r0) or r0 == 0: continue
        fr = df["close"].iloc[i + 10] / df["close"].iloc[i] - 1
        m.append(int(np.sign(r0) == np.sign(fr)))
    out["LargePersist"] = np.mean(m) if m else np.nan
    # 9. MOC mark
    m2 = []
    for d, g in df.groupby(df.index.date):
        if len(g) < 10: continue
        day_r = g["close"].iloc[-1] / g["open"].iloc[0] - 1
        last_r = g["close"].iloc[-1] / g["close"].iloc[-2] - 1 if len(g) > 1 else 0
        if day_r == 0 or last_r == 0: continue
        m2.append(int(np.sign(day_r) == np.sign(last_r)))
    out["MOC_mark"] = np.mean(m2) if m2 else np.nan
    # 10. Daily RV (bps)
    out["Daily_RV"] = df["close"].pct_change().std() * 1e4
    return out


def main():
    print("=" * 130)
    print("C. 20 銘柄 アルゴ・クラスタリング")
    print("=" * 130)
    idx = load(INDEX)
    profiles = {}
    for sym in SYMS:
        df = load(sym)
        if df is None or len(df) < 5000:
            print(f"  ✗ {sym}: insufficient data"); continue
        profiles[sym] = compute_profile(df, idx)
        print(f"  ✓ {sym}: {len(df):,} bars")

    prof_df = pd.DataFrame(profiles).T
    print("\n指紋プロファイル (raw):")
    print(prof_df.round(3).to_string())

    # 欠損補完
    prof_df = prof_df.fillna(prof_df.median())

    # 標準化
    scaler = StandardScaler()
    Z = scaler.fit_transform(prof_df.values)

    print("\n" + "=" * 130)
    for k in [3, 4, 5]:
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(Z)
        labels = km.labels_
        print(f"\n▼ k={k} クラスタ結果")
        clusters = pd.DataFrame({"sym": prof_df.index, "cluster": labels})
        for c in sorted(set(labels)):
            members = clusters[clusters.cluster == c]["sym"].tolist()
            # クラスタの平均プロファイル
            mean_prof = prof_df.loc[members].mean()
            # 全体との差が大きいトップ指標
            diff = (mean_prof - prof_df.mean()) / prof_df.std()
            top_up = diff.abs().sort_values(ascending=False).head(3)
            charac = ", ".join(
                f"{('+' if diff[m] > 0 else '-')}{m}({diff[m]:+.2f})"
                for m in top_up.index)
            print(f"  Cluster {c}: {members}")
            print(f"              特徴: {charac}")

    # k=4 を CSV に
    km = KMeans(n_clusters=4, random_state=42, n_init=10).fit(Z)
    prof_df["cluster_k4"] = km.labels_
    prof_df.to_csv("cluster_profile.csv")
    print("\n→ cluster_profile.csv 保存")


if __name__ == "__main__":
    main()
