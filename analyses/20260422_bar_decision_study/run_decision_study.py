#!/usr/bin/env python3
"""
バー長 意思決定有用性スタディ (コスト非考慮)
人間トレーダーがエントリー/損切りを決める際、何分足が最も有用かを
5 つのメトリクスで評価する。

1. Body/Range 比         — 方向の明瞭さ
2. ATR 安定性 (CV)        — 損切幅の決めやすさ
3. スイング高値の信頼性    — N 本高値ブレイク後の延伸率 / false break 率
4. ストップ最適幅 (MAE)   — 次 M 本で高値超え確率 × 必要 ATR 倍率
5. 転換点ラグ             — 日中最高値→最安値反転までに何本で気付けるか
"""
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import psycopg2

warnings.filterwarnings("ignore")
PG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}
BARS = [1, 3, 5, 10, 15, 30]
SYMS = ["8035.T", "6857.T", "5713.T", "5802.T"]


def load(sym):
    conn = psycopg2.connect(**PG)
    df = pd.read_sql(
        f"SELECT timestamp,open,high,low,close,volume FROM intraday_data "
        f"WHERE symbol='{sym}' ORDER BY timestamp", conn)
    conn.close()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["jst"] = df["timestamp"] + pd.Timedelta(hours=9)
    df = df.set_index("jst").sort_index().dropna(subset=["open"])
    # 取引時間のみ
    h = df.index.hour; m = df.index.minute
    mask = ((h == 9) | (h == 10) |
            ((h == 11) & (m <= 30)) |
            (h == 12) & (m >= 30) |
            (h == 13) | (h == 14) |
            ((h == 15) & (m <= 30)))
    return df[mask]


def resample(df, n):
    if n == 1:
        return df[["open", "high", "low", "close", "volume"]].copy()
    # 日ごとにまとめてリサンプル (日をまたがないように)
    out = []
    for d, g in df.groupby(df.index.date):
        r = g.resample(f"{n}min", label="left", closed="left").agg(
            {"open": "first", "high": "max", "low": "min",
             "close": "last", "volume": "sum"}).dropna()
        out.append(r)
    return pd.concat(out) if out else pd.DataFrame()


def atr(b, n=14):
    tr = pd.concat([
        b["high"] - b["low"],
        (b["high"] - b["close"].shift()).abs(),
        (b["low"] - b["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()


# =========================================================
# 1. Body/Range 比
# =========================================================
def body_ratio(b):
    rng = (b["high"] - b["low"]).replace(0, np.nan)
    body = (b["close"] - b["open"]).abs()
    r = (body / rng).dropna()
    return r.mean(), r.median()


# =========================================================
# 2. ATR 安定性 (CV = std/mean)
# =========================================================
def atr_stability(b):
    a = atr(b, 14).dropna()
    if len(a) < 50: return np.nan
    # 20 本移動の CV
    cv = (a.rolling(20).std() / a.rolling(20).mean()).dropna()
    return cv.median()


# =========================================================
# 3. スイング高値の信頼性
#   N 本前までの高値を超えた次の足において、
#   その後 M 本以内に「そのブレイク高値 - 0.5 ATR」より上に居続けた割合
#   = ブレイク成功率 (follow-through)
# =========================================================
def breakout_followthrough(b, lookback=20, forward=5):
    if len(b) < lookback + forward + 20: return np.nan
    a = atr(b, 14)
    roll_hi = b["high"].rolling(lookback).max().shift(1)
    broke = b["close"] > roll_hi
    idx = np.where(broke.values)[0]
    results = []
    for i in idx:
        if i + forward >= len(b): continue
        brk_level = roll_hi.iloc[i]
        stop = brk_level - 0.5 * a.iloc[i] if not np.isnan(a.iloc[i]) else brk_level
        # 次 M 本で stop を下回ったか
        forward_low = b["low"].iloc[i + 1:i + 1 + forward].min()
        results.append(forward_low >= stop)
    if not results: return np.nan
    return np.mean(results)


# =========================================================
# 4. ストップ最適幅 — MAE 分布
#   各エントリー (単純に各バーの close で next-bar long) 後、
#   次 M 本の最大逆行 (= close - min low) を ATR 倍で表現
#   80% タイル点 = その倍率を損切にすればストップアウト率 20%
# =========================================================
def mae_quantile(b, forward=10, q=0.8):
    if len(b) < forward + 30: return np.nan
    a = atr(b, 14)
    entry = b["close"]
    maes = []
    for i in range(30, len(b) - forward):
        if np.isnan(a.iloc[i]) or a.iloc[i] == 0: continue
        fwd_low = b["low"].iloc[i + 1:i + 1 + forward].min()
        mae = (entry.iloc[i] - fwd_low) / a.iloc[i]  # ATR 倍
        maes.append(mae)
    if not maes: return np.nan
    return np.quantile(maes, q)


# =========================================================
# 5. 転換点ラグ
#   その日の最高値の直後 N 本以内に「下方向 1 ATR 動いた」ことが
#   何本後で検知できるか (= 反転認識の最短ラグ)
# =========================================================
def reversal_lag(b):
    lags = []
    a_all = atr(b, 14)
    for d, g in b.groupby(b.index.date):
        if len(g) < 10: continue
        ai = g.index.intersection(a_all.index)
        if len(ai) < 10: continue
        # 当日 ATR 平均
        adv = a_all.loc[ai].mean()
        if np.isnan(adv) or adv == 0: continue
        hi_idx = g["high"].idxmax()
        hi_pos = g.index.get_loc(hi_idx)
        hi_val = g["high"].iloc[hi_pos]
        # 以降のバーで low が hi - 1*ATR を下回るまでの本数
        for k in range(1, len(g) - hi_pos):
            if g["low"].iloc[hi_pos + k] <= hi_val - adv:
                lags.append(k)
                break
    if not lags: return np.nan
    return np.median(lags)


# =========================================================
def main():
    print("=" * 110)
    print("バー長 意思決定有用性スタディ (コスト無関係)")
    print("=" * 110)

    raw = {s: load(s) for s in SYMS}
    for s, d in raw.items():
        print(f"  {s}: {len(d):,} bars  {d.index.min().date()}〜{d.index.max().date()}")

    cols = ["body/range 平均", "body/range 中央", "ATR CV 中央",
            "BO 追従率(%)", "MAE 80%q (ATR)", "反転ラグ(本)"]
    summary = {}

    for n in BARS:
        rows = []
        for s in SYMS:
            b = resample(raw[s], n)
            br_m, br_md = body_ratio(b)
            cv = atr_stability(b)
            ft = breakout_followthrough(b)
            mae = mae_quantile(b)
            rl = reversal_lag(b)
            rows.append([br_m, br_md, cv, ft * 100 if ft else np.nan, mae, rl])
        arr = np.array(rows, dtype=float)
        avg = np.nanmean(arr, axis=0)
        summary[n] = avg
        print(f"\n── {n} 分足 ──────────────────────────────────────────")
        print(f"  {'銘柄':<10}" + "".join(f"{c:>16}" for c in cols))
        for s, r in zip(SYMS, rows):
            print(f"  {s:<10}" +
                  f"{r[0]:>16.3f}" f"{r[1]:>16.3f}" f"{r[2]:>16.3f}"
                  f"{r[3]:>16.1f}" f"{r[4]:>16.2f}" f"{r[5]:>16.1f}")
        print(f"  {'AVG':<10}" +
              f"{avg[0]:>16.3f}" f"{avg[1]:>16.3f}" f"{avg[2]:>16.3f}"
              f"{avg[3]:>16.1f}" f"{avg[4]:>16.2f}" f"{avg[5]:>16.1f}")

    # 総合表
    print("\n" + "=" * 110)
    print("【バー長別 4 銘柄平均サマリ】")
    print("=" * 110)
    print(f"  {'bar':>5}  {'body/rng':>10}  {'body/rng med':>14}  {'ATR CV':>8}  "
          f"{'BO追従%':>8}  {'MAE 80%q':>10}  {'反転ラグ本':>12}  {'反転ラグ分':>12}")
    for n, a in summary.items():
        print(f"  {n:>5}  {a[0]:>10.3f}  {a[1]:>14.3f}  {a[2]:>8.3f}  "
              f"{a[3]:>8.1f}  {a[4]:>10.2f}  {a[5]:>12.1f}  {a[5]*n:>12.1f}")

    print("\n解釈ガイド:")
    print("  body/rng: 高=足の方向が明瞭 (ヒゲ少)")
    print("  ATR CV  : 低=ATR が安定 → 損切幅を固定しやすい")
    print("  BO 追従%: 高=ブレイクが騙しにくい → エントリー判断に有用")
    print("  MAE 80%q: 低=1 ATR に近い損切で 80% のトレードが生存")
    print("  反転ラグ: 少本/短分=転換点を早く察知できる")

    pd.DataFrame(summary, index=cols).T.to_csv("decision_metrics.csv")
    print("\n→ decision_metrics.csv 保存")


if __name__ == "__main__":
    main()
