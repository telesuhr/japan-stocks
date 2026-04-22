#!/usr/bin/env python3
"""
Walk-Forward — 閾値緩和 / パラメータスイープで N を確保する。
"""
import warnings
from itertools import product
import numpy as np
import pandas as pd
import psycopg2
import statsmodels.api as sm

warnings.filterwarnings("ignore")
PG = dict(host="localhost", port=5432, user="postgres", dbname="market_data")

CANDIDATE_PAIRS = [
    ("7011.T", "7013.T", "重工: MHI/IHI"),
    ("8306.T", "8411.T", "銀行: MUFG/みずほ"),
    ("8306.T", "8316.T", "銀行: MUFG/SMFG"),
    ("8316.T", "8411.T", "銀行: SMFG/みずほ"),
    ("6146.T", "6323.T", "半導体: ディスコ/ローツェ"),
    ("9432.T", "9434.T", "通信: NTT/SB"),
    ("9432.T", "9433.T", "通信: NTT/KDDI"),
    ("8002.T", "8031.T", "商社: 丸紅/三井物産"),
    ("4502.T", "4568.T", "医薬: 武田/第一三共"),
    ("9101.T", "9107.T", "海運: 郵船/川崎汽船"),
    ("5711.T", "5713.T", "非鉄: 三菱マテ/住友金鉱"),
    ("7203.T", "7261.T", "自動車: トヨタ/マツダ"),
    ("8801.T", "8802.T", "不動産: 三井不/三菱地"),
    ("6501.T", "6503.T", "電機: 日立/三菱電"),
    ("8053.T", "8058.T", "商社: 住友商/三菱商"),
    ("7270.T", "7269.T", "自動車: スバル/スズキ"),
    ("9020.T", "9022.T", "鉄道: JR東/JR東海"),
    ("4503.T", "4578.T", "医薬: アステラス/大塚"),
    ("6758.T", "6702.T", "電機: ソニー/富士通"),
    ("5802.T", "5801.T", "電線: 住電/古河"),
]
COST_BPS = 8.0


def load_daily(symbols):
    conn = psycopg2.connect(**PG)
    frames = {}
    for s in symbols:
        df = pd.read_sql(
            "SELECT timestamp, close FROM intraday_data WHERE symbol=%s ORDER BY timestamp",
            conn, params=(s,))
        if df.empty: continue
        df["jst"] = pd.to_datetime(df["timestamp"]) + pd.Timedelta(hours=9)
        df = df.set_index("jst").sort_index()
        frames[s] = df["close"].groupby(df.index.date).last()
    conn.close()
    if not frames: return pd.DataFrame()
    px = pd.concat(frames, axis=1)
    px.index = pd.to_datetime(px.index)
    return px.dropna(how="any")


def rolling_beta(y, x, w):
    betas = pd.Series(index=y.index, dtype=float)
    for i in range(w, len(y)):
        Y = y.iloc[i-w:i]; X = sm.add_constant(x.iloc[i-w:i])
        try: betas.iloc[i] = sm.OLS(Y, X).fit().params.iloc[1]
        except: pass
    return betas


def backtest(p1, p2, px, bw=60, zw=40, entry_z=2.0, exit_z=0.5, stop_z=4.0,
             max_hold=20, cost=COST_BPS):
    lp = np.log(px[[p1, p2]].dropna())
    if len(lp) < bw + zw + 20:
        return None
    betas = rolling_beta(lp[p1], lp[p2], bw)
    spread = lp[p1] - betas * lp[p2]
    mu = spread.rolling(zw).mean()
    sd = spread.rolling(zw).std()
    z = (spread - mu) / sd

    trades = []
    pos = 0; ei = None; es = None; ez = None; eb = None
    for i in range(bw + zw, len(spread)):
        zi = z.iloc[i]
        if pd.isna(zi): continue
        if pos == 0:
            if zi >= entry_z:
                pos = -1; ei = i; es = spread.iloc[i]; ez = zi; eb = betas.iloc[i]
            elif zi <= -entry_z:
                pos = 1; ei = i; es = spread.iloc[i]; ez = zi; eb = betas.iloc[i]
        else:
            hold = i - ei; r = None
            if abs(zi) < exit_z: r = "MR"
            elif abs(zi) > stop_z: r = "STOP"
            elif hold >= max_hold: r = "TIME"
            if r:
                sn = lp[p1].iloc[i] - eb * lp[p2].iloc[i]
                pnl = pos * (sn - es)
                gross = pnl * 10000 / (1 + abs(eb))
                net = gross - cost
                trades.append(dict(entry=spread.index[ei], exit=spread.index[i],
                                   ez=ez, hold=hold, net=net, reason=r))
                pos = 0
    return pd.DataFrame(trades)


def summary(td):
    if td is None or len(td) == 0:
        return dict(n=0)
    arr = td["net"].values
    sd = arr.std()
    if sd == 0: sd = 1e-9
    return dict(
        n=len(td), mean=arr.mean(), sharpe=arr.mean()/sd*np.sqrt(252/td["hold"].mean()),
        t=arr.mean()/sd*np.sqrt(len(arr)), wr=(arr>0).mean()*100,
        avg_hold=td["hold"].mean())


def main():
    syms = list({s for p1,p2,_ in CANDIDATE_PAIRS for s in (p1,p2)})
    px = load_daily(syms)
    print(f"データ: {len(px)}日  銘柄数 {len(px.columns)}\n")

    # パラメータスイープ
    ENTRY = [1.5, 2.0, 2.5]
    Z_WIN = [20, 40]
    HOLD = [10, 20]

    best_configs = []
    print("="*110)
    print(f"{'ペア':<24} {'E':>4} {'ZW':>3} {'HD':>3}  {'N':>3} {'mean':>7} {'Sharpe':>7} {'t':>6} {'WR':>5} {'hold':>5}")
    print("="*110)

    for p1, p2, lbl in CANDIDATE_PAIRS:
        if p1 not in px.columns or p2 not in px.columns: continue
        pair_best = None
        for ez, zw, hd in product(ENTRY, Z_WIN, HOLD):
            td = backtest(p1, p2, px, zw=zw, entry_z=ez, max_hold=hd,
                          exit_z=max(0.3, ez*0.25))
            s = summary(td)
            if s["n"] >= 5:
                # 最良Sharpeを選択
                if pair_best is None or s.get("sharpe",0) > pair_best[0].get("sharpe",0):
                    pair_best = (s, ez, zw, hd, td)
        if pair_best is None:
            print(f"{lbl:<24}  (N<5)")
            continue
        s, ez, zw, hd, td = pair_best
        print(f"{lbl:<24} {ez:>4.1f} {zw:>3d} {hd:>3d}  {s['n']:>3d} "
              f"{s['mean']:>+7.1f} {s['sharpe']:>+7.2f} {s['t']:>+6.2f} "
              f"{s['wr']:>5.1f} {s['avg_hold']:>5.1f}")
        best_configs.append(dict(pair=lbl, p1=p1, p2=p2, ez=ez, zw=zw, hd=hd,
                                  **s, td=td))

    print()
    # H1/H2 OoS 検証
    print("="*110)
    print("H1/H2 時間分割 OoS 検証 (Sharpe≥1.5 かつ N≥10 のペアのみ)")
    print("="*110)
    print(f"{'ペア':<24} {'N':>3} {'Sh':>6} {'t':>5} |  {'H1N':>3} {'H1Sh':>6} {'H1M':>7} | {'H2N':>3} {'H2Sh':>6} {'H2M':>7}")
    print("-"*110)
    adopt = []
    for c in best_configs:
        if c["n"] < 10 or c["sharpe"] < 1.5:
            continue
        td = c["td"].sort_values("entry").reset_index(drop=True)
        mid = len(td) // 2
        h1 = td.iloc[:mid]; h2 = td.iloc[mid:]
        s1 = summary(h1); s2 = summary(h2)
        print(f"{c['pair']:<24} {c['n']:>3d} {c['sharpe']:>+6.2f} {c['t']:>+5.2f} | "
              f"{s1.get('n',0):>3d} {s1.get('sharpe',0):>+6.2f} {s1.get('mean',0):>+7.1f} | "
              f"{s2.get('n',0):>3d} {s2.get('sharpe',0):>+6.2f} {s2.get('mean',0):>+7.1f}")
        if s1.get("sharpe",0) >= 0.5 and s2.get("sharpe",0) >= 0.5:
            adopt.append(c)

    print()
    print("="*110)
    print(f"両期 Sharpe ≥ 0.5 を満たすペア: {len(adopt)} 件")
    for c in adopt:
        print(f"  - {c['pair']}  (Full N={c['n']}, Sharpe={c['sharpe']:.2f}, t={c['t']:.2f})")


if __name__ == "__main__":
    main()
