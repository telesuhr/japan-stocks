#!/usr/bin/env python3
"""
何分足が最も有効か — 定量検証

対象指標
 (a) リターン自己相関 ρ(1)〜ρ(5)     → MR or TF バイアスの存在と強さ
 (b) 分散比検定 VR(k)                  → k=2..10 でのランダムウォーク逸脱
 (c) N-bar ブレイクアウト戦略          → 手触り検証: どのbarで最もエッジ?
 (d) 単純 MR 戦略 (下落→買い/上昇→売り) → 同
 (e) シグナル/ノイズ (|ret|分布)        → コスト対比のエッジ余力
 (f) cost vs edge ブレークイーブン      → 往復 bps 何まで耐えるか
"""
import warnings
import numpy as np
import pandas as pd
import psycopg2
warnings.filterwarnings("ignore")

PG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}

# 流動性・セクター異なる 4 銘柄
SYMBOLS = {
    "8035.T": "TEL (半導体大型)",
    "6857.T": "Advantest (半導体)",
    "5713.T": "住友金鉱 (非鉄)",
    "5802.T": "住友電工 (電線)",
}
BAR_LENS = [1, 3, 5, 10, 15, 30]


def load_1min(symbol):
    conn = psycopg2.connect(**PG)
    df = pd.read_sql(
        f"SELECT timestamp,open,high,low,close,volume FROM intraday_data "
        f"WHERE symbol='{symbol}' ORDER BY timestamp", conn)
    conn.close()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['jst'] = df['timestamp'] + pd.Timedelta(hours=9)
    df = df.dropna(subset=['open']).set_index('jst').sort_index()
    h, m = df.index.hour, df.index.minute
    mask = (((h == 9)) | (h == 10) | ((h == 11) & (m <= 30)) |
            ((h == 12) & (m >= 30)) | (h == 13) | (h == 14) |
            ((h == 15) & (m <= 30)))
    return df[mask]


def resample_bars(df1, mins):
    if mins == 1: return df1
    rule = f"{mins}T"
    o = df1['open'].resample(rule).first()
    h = df1['high'].resample(rule).max()
    l = df1['low'].resample(rule).min()
    c = df1['close'].resample(rule).last()
    v = df1['volume'].resample(rule).sum()
    out = pd.DataFrame({"open":o,"high":h,"low":l,"close":c,"volume":v}).dropna()
    # 株式時間帯のみ
    hh, mm = out.index.hour, out.index.minute
    mask = (((hh == 9)) | (hh == 10) | ((hh == 11) & (mm <= 30)) |
            ((hh == 12) & (mm >= 30)) | (hh == 13) | (hh == 14) |
            ((hh == 15) & (mm <= 30)))
    return out[mask]


def daily_returns_by_session(df):
    """各取引日内の close→close bar リターン (連続bar内のみ)"""
    df = df.copy()
    df['ret'] = df['close'].pct_change() * 10000  # bps
    df['date'] = df.index.date
    # 日跨ぎのリターンを除外 (各日の1本目はNaN)
    df.loc[df.groupby('date').cumcount() == 0, 'ret'] = np.nan
    return df


def autocorr(series, lags=5):
    """NaN 耐性ある自己相関"""
    s = series.dropna()
    out = []
    for k in range(1, lags+1):
        if len(s) <= k: out.append(np.nan); continue
        c = np.corrcoef(s.iloc[:-k], s.iloc[k:])[0, 1]
        out.append(c)
    return out


def variance_ratio(series, k):
    """VR(k) = var(r_k) / (k * var(r_1)). <1: MR, >1: TF"""
    s = series.dropna().values
    if len(s) < 2*k: return np.nan
    v1 = np.var(s, ddof=1)
    # k-sum ret
    rk = np.convolve(s, np.ones(k), 'valid')
    vk = np.var(rk, ddof=1)
    return vk / (k * v1)


def test_breakout(df_bar, lookback=20, hold=5):
    """直近 lookback bar の高値突破 → long, hold bar 後 close で決済
    short: 安値割れ → short.
    各日内に限定 (日跨ぎなし)"""
    pnls = []
    for dt, g in df_bar.groupby(df_bar.index.date):
        g = g.reset_index(drop=True)
        if len(g) < lookback + hold + 2: continue
        for i in range(lookback, len(g) - hold):
            hi = g['high'].iloc[i-lookback:i].max()
            lo = g['low'].iloc[i-lookback:i].min()
            cur_hi = g['high'].iloc[i]
            cur_lo = g['low'].iloc[i]
            entry = g['close'].iloc[i]
            exit_ = g['close'].iloc[i+hold]
            if cur_hi > hi:
                pnls.append((exit_/entry - 1)*10000)
            elif cur_lo < lo:
                pnls.append((entry/exit_ - 1)*10000)
    return np.array(pnls)


def test_simple_mr(df_bar, lookback=3, hold=3):
    """直近 lookback bar 累積ダウン → 買い, 上げ → 売り. hold 後 close"""
    pnls = []
    for dt, g in df_bar.groupby(df_bar.index.date):
        g = g.reset_index(drop=True)
        if len(g) < lookback + hold + 2: continue
        for i in range(lookback, len(g) - hold):
            past = g['close'].iloc[i] / g['close'].iloc[i-lookback] - 1
            entry = g['close'].iloc[i]
            exit_ = g['close'].iloc[i+hold]
            if past < -0.001:  # 10bps以上下がった
                pnls.append((exit_/entry - 1)*10000)
            elif past > 0.001:
                pnls.append((entry/exit_ - 1)*10000)
    return np.array(pnls)


def strat_stats(arr, cost_bps=0):
    a = np.asarray(arr, float)
    a = a[~np.isnan(a)]
    a = a - cost_bps
    n = len(a)
    if n < 5: return {"N":n, "Sharpe":np.nan, "mean":np.nan, "t":np.nan}
    sd = a.std(ddof=1)
    sh = a.mean()/sd*np.sqrt(252) if sd > 0 else 0
    t = a.mean()/(sd/np.sqrt(n)) if sd > 0 else 0
    return {"N":n, "mean":a.mean(), "Sharpe":sh, "t":t}


def main():
    print("=" * 130)
    print("何分足が最も有効か — 定量検証")
    print("=" * 130)

    print("\nデータロード (1分足) ...")
    data = {s: load_1min(s) for s in SYMBOLS}
    for s, df in data.items():
        print(f"  {s} ({SYMBOLS[s]}): {len(df)} 分足")

    # ---- 各銘柄×barlen ----
    print("\n" + "=" * 130)
    print("【(a) リターン自己相関 ρ(1), ρ(2), ρ(3) — 負=MRバイアス, 正=TFバイアス】")
    print("=" * 130)
    print(f"{'銘柄':<25} {'bar':>5} {'ρ(1)':>8} {'ρ(2)':>8} {'ρ(3)':>8} {'ρ(4)':>8} {'ρ(5)':>8}")
    for s, df1 in data.items():
        for bl in BAR_LENS:
            bar = resample_bars(df1, bl)
            dr = daily_returns_by_session(bar)
            ac = autocorr(dr['ret'], 5)
            print(f"  {s:<23} {bl:>5} "
                  f"{ac[0]:>+8.3f} {ac[1]:>+8.3f} {ac[2]:>+8.3f} "
                  f"{ac[3]:>+8.3f} {ac[4]:>+8.3f}")

    # ---- 分散比 ----
    print("\n" + "=" * 130)
    print("【(b) 分散比 VR(k) (k=2,5,10) — <1:MR, ~1:ランダム, >1:トレンド】")
    print("=" * 130)
    print(f"{'銘柄':<25} {'bar':>5} {'VR(2)':>8} {'VR(5)':>8} {'VR(10)':>8} {'σ (bps)':>10}")
    for s, df1 in data.items():
        for bl in BAR_LENS:
            bar = resample_bars(df1, bl)
            dr = daily_returns_by_session(bar)
            vr2 = variance_ratio(dr['ret'], 2)
            vr5 = variance_ratio(dr['ret'], 5)
            vr10 = variance_ratio(dr['ret'], 10)
            sig = dr['ret'].std()
            print(f"  {s:<23} {bl:>5} "
                  f"{vr2:>8.3f} {vr5:>8.3f} {vr10:>8.3f} {sig:>10.1f}")

    # ---- ブレイクアウト ----
    print("\n" + "=" * 130)
    print("【(c) N-bar ブレイクアウト戦略 (lookback=20bar, hold=5bar), コスト0】")
    print("=" * 130)
    print(f"{'銘柄':<25} {'bar':>5} {'N':>7} {'mean':>8} {'Sharpe':>8} {'t':>6}")
    agg_bo = {bl: [] for bl in BAR_LENS}
    for s, df1 in data.items():
        for bl in BAR_LENS:
            bar = resample_bars(df1, bl)
            p = test_breakout(bar, lookback=20, hold=5)
            st = strat_stats(p)
            agg_bo[bl].extend(p.tolist())
            print(f"  {s:<23} {bl:>5} {st['N']:>7} "
                  f"{st['mean']:>+8.1f} {st['Sharpe']:>+8.2f} {st['t']:>+6.2f}")
    print(f"\n  --- 全銘柄プール ---")
    for bl in BAR_LENS:
        st = strat_stats(np.array(agg_bo[bl]))
        print(f"   bar={bl:>3}: N={st['N']:>7}  mean={st['mean']:>+6.1f}  "
              f"Sharpe={st['Sharpe']:>+5.2f}  t={st['t']:>+5.2f}")

    # ---- 単純 MR ----
    print("\n" + "=" * 130)
    print("【(d) 単純 MR 戦略 (過去3bar > ±10bps動き → 逆方向, 3bar保有), コスト0】")
    print("=" * 130)
    print(f"{'銘柄':<25} {'bar':>5} {'N':>7} {'mean':>8} {'Sharpe':>8} {'t':>6}")
    agg_mr = {bl: [] for bl in BAR_LENS}
    for s, df1 in data.items():
        for bl in BAR_LENS:
            bar = resample_bars(df1, bl)
            p = test_simple_mr(bar, lookback=3, hold=3)
            st = strat_stats(p)
            agg_mr[bl].extend(p.tolist())
            print(f"  {s:<23} {bl:>5} {st['N']:>7} "
                  f"{st['mean']:>+8.1f} {st['Sharpe']:>+8.2f} {st['t']:>+6.2f}")
    print(f"\n  --- 全銘柄プール ---")
    for bl in BAR_LENS:
        st = strat_stats(np.array(agg_mr[bl]))
        print(f"   bar={bl:>3}: N={st['N']:>7}  mean={st['mean']:>+6.1f}  "
              f"Sharpe={st['Sharpe']:>+5.2f}  t={st['t']:>+5.2f}")

    # ---- シグナル/ノイズ ----
    print("\n" + "=" * 130)
    print("【(e) bar リターンの絶対値統計 (シグナル強度)】")
    print("=" * 130)
    print(f"{'銘柄':<25} {'bar':>5} {'|ret| mean':>12} {'|ret| P50':>10} "
          f"{'|ret| P90':>10} {'|ret|>10 pct':>12}")
    for s, df1 in data.items():
        for bl in BAR_LENS:
            bar = resample_bars(df1, bl)
            dr = daily_returns_by_session(bar)
            abs_r = dr['ret'].abs().dropna()
            p50 = abs_r.median()
            p90 = abs_r.quantile(0.9)
            mean = abs_r.mean()
            over10 = (abs_r > 10).mean() * 100
            print(f"  {s:<23} {bl:>5} {mean:>12.1f} {p50:>10.1f} {p90:>10.1f} {over10:>11.1f}%")

    # ---- コスト耐性 (MR 戦略、往復 cost 変動) ----
    print("\n" + "=" * 130)
    print("【(f) 単純 MR 戦略: 往復コスト感応度 — 各 bar 長の Sharpe】")
    print("=" * 130)
    print(f"{'bar':>5} {'cost=0':>10} {'2bps':>10} {'4bps':>10} {'6bps':>10} {'8bps':>10} {'10bps':>10} {'16bps':>10}")
    for bl in BAR_LENS:
        arr = np.array(agg_mr[bl])
        row = []
        for c in [0, 2, 4, 6, 8, 10, 16]:
            st = strat_stats(arr, cost_bps=c)
            row.append(f"{st['Sharpe']:+.2f}")
        print(f"  {bl:>3} " + " ".join(f"{v:>10}" for v in row))

    print("\n" + "=" * 130)
    print("【(g) ブレイクアウト戦略: 往復コスト感応度】")
    print("=" * 130)
    print(f"{'bar':>5} {'cost=0':>10} {'2bps':>10} {'4bps':>10} {'6bps':>10} {'8bps':>10} {'10bps':>10} {'16bps':>10}")
    for bl in BAR_LENS:
        arr = np.array(agg_bo[bl])
        row = []
        for c in [0, 2, 4, 6, 8, 10, 16]:
            st = strat_stats(arr, cost_bps=c)
            row.append(f"{st['Sharpe']:+.2f}")
        print(f"  {bl:>3} " + " ".join(f"{v:>10}" for v in row))


if __name__ == "__main__":
    main()
