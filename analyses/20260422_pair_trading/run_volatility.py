#!/usr/bin/env python3
"""
#4 固定日数ホールド (ボラ売り的アプローチ)

通常のペア MR は Z 回帰で exit するが、これは「タイミングゲーム」。
ここでは:
  - Z が閾値を超えたら Short/Long スプレッド
  - 固定日数 (H日) 後に強制クローズ
  - 収益分布 ≒ 「H日後にどれだけ Z が縮んでいるか」の期待値を直接計測

これにより方向予測ではなく「統計的収束性」そのものを評価できる。
Kelly サイジング相当を目指すなら σ で割った t-score が本質。
"""
import warnings
import numpy as np
import pandas as pd
import pymysql
import statsmodels.api as sm
from pathlib import Path

warnings.filterwarnings("ignore")
MARIA = dict(host="100.92.181.92", port=3306, user="rfnews",
             password="Bleach@924", database="refinitiv_news")

PAIRS = [
    ("7011.T", "7013.T", "重工: MHI/IHI"),
    ("8306.T", "8411.T", "銀行: MUFG/みずほ"),
    ("8306.T", "8316.T", "銀行: MUFG/SMFG"),
    ("6146.T", "6323.T", "半導体: ディスコ/ローツェ"),
    ("9432.T", "9433.T", "通信: NTT/KDDI"),
    ("8002.T", "8031.T", "商社: 丸紅/三井物産"),
    ("5711.T", "5713.T", "非鉄: 三菱マテ/住友金鉱"),
    ("5711.T", "5706.T", "非鉄: 三菱マテ/三井金"),
    ("7203.T", "7267.T", "自動車: トヨタ/ホンダ"),
    ("6501.T", "6503.T", "電機: 日立/三菱電"),
    ("7270.T", "7269.T", "自動車: スバル/スズキ"),
    ("9020.T", "9022.T", "鉄道: JR東/JR東海"),
    ("4503.T", "4578.T", "医薬: アステラス/大塚"),
    ("6758.T", "6702.T", "電機: ソニー/富士通"),
    ("5802.T", "5801.T", "電線: 住電/古河"),
    ("6920.T", "6857.T", "半導体: レーザー/アドバン"),
    ("8035.T", "6920.T", "半導体: TEL/レーザー"),
    ("1605.T", "5020.T", "エネルギー: INPEX/ENEOS"),
]
COST = 8.0


def load(syms):
    conn = pymysql.connect(**MARIA)
    q = f"SELECT symbol,trade_date,close FROM daily_data WHERE symbol IN ({','.join(['%s']*len(syms))}) ORDER BY symbol,trade_date"
    df = pd.read_sql(q, conn, params=tuple(syms)); conn.close()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.pivot(index="trade_date", columns="symbol", values="close").astype(float)


def rolling_beta(y, x, w):
    b = pd.Series(index=y.index, dtype=float)
    for i in range(w, len(y)):
        Y = y.iloc[i-w:i]; X = sm.add_constant(x.iloc[i-w:i])
        try: b.iloc[i] = sm.OLS(Y, X).fit().params.iloc[1]
        except Exception: pass
    return b


def fixed_horizon_bt(p1, p2, px, bw=60, zw=40, entry_z=2.0, horizon=10, cost=COST):
    lp = np.log(px[[p1,p2]].dropna())
    if len(lp) < bw+zw+horizon+20: return None
    betas = rolling_beta(lp[p1], lp[p2], bw)
    spread = pd.Series(index=lp.index, dtype=float)
    for i in range(len(lp)):
        b = betas.iloc[i]
        if pd.notna(b): spread.iloc[i] = lp[p1].iloc[i] - b*lp[p2].iloc[i]
    mu = spread.rolling(zw).mean(); sd = spread.rolling(zw).std()
    z = (spread - mu) / sd

    trades = []
    i = bw + zw
    while i < len(spread) - horizon:
        zi = z.iloc[i]
        if pd.isna(zi) or pd.isna(betas.iloc[i]):
            i += 1; continue
        if abs(zi) >= entry_z:
            pos = -1 if zi > 0 else 1
            eb = betas.iloc[i]; es = spread.iloc[i]
            j = i + horizon
            sn = lp[p1].iloc[j] - eb*lp[p2].iloc[j]
            pnl = pos * (sn - es)
            gross = pnl * 10000 / (1 + abs(eb))
            net = gross - cost
            z_exit = z.iloc[j]
            trades.append(dict(entry=spread.index[i], exit=spread.index[j],
                               entry_z=zi, exit_z=z_exit, net=net))
            i = j + 1  # 重複禁止
        else:
            i += 1
    return pd.DataFrame(trades)


def stat(td):
    if td is None or len(td) == 0: return dict(n=0)
    arr = td["net"].values
    sd = arr.std() if arr.std()>0 else 1e-9
    return dict(n=len(td), mean=arr.mean(),
                sharpe=arr.mean()/sd*np.sqrt(252/10),  # horizon assumed 10
                t=arr.mean()/sd*np.sqrt(len(arr)),
                wr=(arr>0).mean()*100,
                z_contract_rate=(td["exit_z"].abs() < td["entry_z"].abs()).mean()*100)


def main():
    print("=" * 120)
    print("#4 固定日数ホールド戦略 (ボラ売り的)")
    print("  Entry |Z|≥2 でスプレッド方向張り、H日後に強制クローズ")
    print("  「Z が縮む確率 = 収束確率」を直接計測")
    print("=" * 120)

    syms = list({s for p1,p2,_ in PAIRS for s in (p1,p2)})
    px = load(syms)
    print(f"データ: {len(px)}日\n")

    for H in [5, 10, 20, 40]:
        print(f"\n【Horizon = {H}日】")
        print(f"  {'ペア':<26} {'N':>4} {'mean':>7} {'Sharpe':>7} {'t':>6} {'WR':>5} {'Z縮小率':>7}")
        all_nets = []
        for p1, p2, lbl in PAIRS:
            td = fixed_horizon_bt(p1, p2, px, horizon=H)
            s = stat(td)
            # Sharpe 年率は H で割る
            if s["n"] > 0:
                sd = td["net"].std() if td["net"].std()>0 else 1e-9
                sh = td["net"].mean()/sd*np.sqrt(252/H)
            else: sh = 0
            if s["n"] >= 5:
                print(f"  {lbl:<26} {s['n']:>4d} {s['mean']:>+7.1f} {sh:>+7.2f} {s['t']:>+6.2f} "
                      f"{s['wr']:>5.1f} {s['z_contract_rate']:>6.1f}%")
                all_nets.append(td["net"].values)
        if all_nets:
            cat = np.concatenate(all_nets)
            sd = cat.std() if cat.std()>0 else 1e-9
            sh = cat.mean()/sd*np.sqrt(252/H)
            print(f"  {'合算':<26} {len(cat):>4d} {cat.mean():>+7.1f} {sh:>+7.2f} "
                  f"{cat.mean()/sd*np.sqrt(len(cat)):>+6.2f} {(cat>0).mean()*100:>5.1f}")


if __name__ == "__main__":
    main()
