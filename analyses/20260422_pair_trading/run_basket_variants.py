#!/usr/bin/env python3
"""#2 バスケット: ルックバック / 方向 / 保有のグリッド探索"""
import warnings
import numpy as np
import pandas as pd
import pymysql

warnings.filterwarnings("ignore")
MARIA = dict(host="100.92.181.92", port=3306, user="rfnews",
             password="Bleach@924", database="refinitiv_news")

SECTORS = {
    "半導体": ["8035.T", "6146.T", "6920.T", "6857.T", "6963.T", "6526.T",
             "3436.T", "6525.T", "6323.T"],
    "自動車": ["7201.T", "7203.T", "7267.T", "7270.T", "7269.T", "7261.T"],
    "商社":   ["8001.T", "8002.T", "8015.T", "8031.T", "8053.T", "8058.T"],
    "銀行":   ["8306.T", "8316.T", "8411.T"],
    "医薬":   ["4502.T", "4503.T", "4523.T", "4568.T", "4578.T"],
    "海運":   ["9101.T", "9104.T", "9107.T"],
    "電機":   ["6501.T", "6503.T", "6752.T", "6758.T", "6702.T"],
    "通信":   ["9432.T", "9433.T", "9434.T"],
    "非鉄":   ["5711.T", "5706.T", "5713.T"],
}


def load(syms):
    conn = pymysql.connect(**MARIA)
    q = f"SELECT symbol,trade_date,close FROM daily_data WHERE symbol IN ({','.join(['%s']*len(syms))}) ORDER BY symbol,trade_date"
    df = pd.read_sql(q, conn, params=tuple(syms)); conn.close()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.pivot(index="trade_date", columns="symbol", values="close").astype(float)


def bt(syms, px, lookback, hold, sign, k_frac=0.33, cost=2.0):
    """sign=+1: 逆張り(losers long/winners short)  /  sign=-1: モメンタム"""
    avail = [s for s in syms if s in px.columns]
    if len(avail) < 3: return None
    lp = np.log(px[avail]).dropna(how="any")
    if len(lp) < lookback + 30: return None
    lret = lp - lp.shift(lookback)
    z = lret.sub(lret.mean(axis=1), axis=0).div(lret.std(axis=1).replace(0, np.nan), axis=0)
    k = max(1, int(round(len(avail)*k_frac)))
    dr = lp.diff()
    pnl = pd.Series(0.0, index=lp.index); rcost = pd.Series(0.0, index=lp.index)
    for i in range(lookback, len(lp)-1):
        zi = z.iloc[i].dropna().sort_values()
        if len(zi) < 2*k: continue
        longs = zi.index[:k].tolist() if sign > 0 else zi.index[-k:].tolist()
        shorts = zi.index[-k:].tolist() if sign > 0 else zi.index[:k].tolist()
        ex = min(i+hold, len(lp)-1)
        for j in range(i+1, ex+1):
            pnl.iloc[j] += (dr.iloc[j][longs].mean() - dr.iloc[j][shorts].mean())/hold
        rcost.iloc[i] += (cost*2/10000)/hold
    return pnl*10000 - rcost*10000


def stat(pnl):
    arr = pnl.dropna().values
    sd = arr.std() if arr.std()>0 else 1e-9
    return arr.mean(), arr.mean()/sd*np.sqrt(252), arr.mean()/sd*np.sqrt(len(arr)), pnl.cumsum().iloc[-1]


def main():
    all_syms = list({s for v in SECTORS.values() for s in v})
    px = load(all_syms)
    print("Variant grid: lookback × hold × sign (+=reversal, -=momentum)")
    print(f"{'Sector':<10} {'LB':>3} {'HD':>3} {'Sign':>5} {'daily':>7} {'Sharpe':>7} {'t':>6} {'Cum':>8}")
    print("-"*70)
    results = []
    for sec, syms in SECTORS.items():
        for lb in [5, 10, 20, 60]:
            for hd in [3, 5, 10]:
                for sg in [+1, -1]:
                    p = bt(syms, px, lb, hd, sg)
                    if p is None: continue
                    m, sh, t, cum = stat(p)
                    if abs(sh) > 0.5 and abs(t) > 2:
                        print(f"{sec:<10} {lb:>3d} {hd:>3d} {sg:>+5d} {m:>+7.2f} {sh:>+7.2f} {t:>+6.2f} {cum:>+8.0f}")
                    results.append((sec, lb, hd, sg, m, sh, t, cum, p))

    # 全セクターでベストな統合 (sign/lookback/hold を共通化)
    print("\n=== 統合 (各 lb,hd,sign 組み合わせで全セクター Equal-Weight) ===")
    print(f"{'LB':>3} {'HD':>3} {'Sign':>5} {'Sharpe':>7} {'t':>6} {'Cum':>8}  H1Sh/H2Sh")
    for lb in [5, 10, 20, 60]:
        for hd in [3, 5, 10]:
            for sg in [+1, -1]:
                pnls = [r[-1] for r in results if r[1]==lb and r[2]==hd and r[3]==sg and r[-1] is not None]
                if not pnls: continue
                combined = pd.concat(pnls, axis=1).fillna(0).mean(axis=1)
                m, sh, t, cum = stat(combined)
                mid = combined.index[len(combined)//2]
                h1 = combined[combined.index<mid]; h2 = combined[combined.index>=mid]
                _,s1,_,_ = stat(h1); _,s2,_,_ = stat(h2)
                flag = " ★" if sh > 0.8 and t > 2 and s1 > 0.3 and s2 > 0.3 else ""
                print(f"{lb:>3d} {hd:>3d} {sg:>+5d} {sh:>+7.2f} {t:>+6.2f} {cum:>+8.0f}  {s1:+.2f}/{s2:+.2f}{flag}")


if __name__ == "__main__":
    main()
