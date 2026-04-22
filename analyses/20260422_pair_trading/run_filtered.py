#!/usr/bin/env python3
"""
#3 条件付きペアMR: フィルタ付きエントリー

#1 で採用した 18 ペアに対し、以下のフィルタを通過したシグナルのみ取引:
  (a) Market regime:  ペア両銘柄の 200日 MA 乖離が ±15% 以内 (大相場でスキップ)
  (b) Volatility:     過去 60日実現ボラが 自身の 252日中央値以下
  (c) Half-life:      スプレッドの OU 半減期が 3-30 日の範囲内

これによりトレード数は減るが品質向上 (エッジ増加) を狙う。
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
    ("7011.T", "7013.T", "重工: MHI/IHI",           2.0, 20, 30),
    ("8306.T", "8411.T", "銀行: MUFG/みずほ",          2.0, 60, 30),
    ("8306.T", "8316.T", "銀行: MUFG/SMFG",         2.0, 60, 20),
    ("6146.T", "6323.T", "半導体: ディスコ/ローツェ",        2.0, 40, 10),
    ("9432.T", "9433.T", "通信: NTT/KDDI",          2.5, 60, 10),
    ("8002.T", "8031.T", "商社: 丸紅/三井物産",           2.5, 60, 10),
    ("5711.T", "5713.T", "非鉄: 三菱マテ/住友金鉱",         2.5, 20, 30),
    ("5711.T", "5706.T", "非鉄: 三菱マテ/三井金",          2.5, 20, 30),
    ("7203.T", "7267.T", "自動車: トヨタ/ホンダ",          1.5, 20, 30),
    ("6501.T", "6503.T", "電機: 日立/三菱電",            2.5, 20, 10),
    ("7270.T", "7269.T", "自動車: スバル/スズキ",          2.5, 40, 10),
    ("9020.T", "9022.T", "鉄道: JR東/JR東海",          2.5, 40, 30),
    ("4503.T", "4578.T", "医薬: アステラス/大塚",          2.5, 40, 10),
    ("6758.T", "6702.T", "電機: ソニー/富士通",           2.0, 40, 10),
    ("5802.T", "5801.T", "電線: 住電/古河",             1.5, 40, 20),
    ("6920.T", "6857.T", "半導体: レーザー/アドバン",         2.5, 40, 10),
    ("8035.T", "6920.T", "半導体: TEL/レーザー",         2.5, 40, 10),
    ("1605.T", "5020.T", "エネルギー: INPEX/ENEOS",    2.5, 20, 30),
]
COST_BPS = 8.0


def load_maria(syms):
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


def compute_half_life(spread_window):
    """OU 半減期: d(spread) = -λ (spread - μ) dt + σ dW
    → AR(1) の係数 φ に対し λ = -ln(φ), half_life = ln(2)/λ"""
    s = spread_window.dropna()
    if len(s) < 20: return np.nan
    ds = s.diff().dropna()
    lag = s.shift(1).dropna()
    # 共通 index
    common = ds.index.intersection(lag.index)
    ds = ds.loc[common]; lag = lag.loc[common]
    try:
        X = sm.add_constant(lag.values)
        res = sm.OLS(ds.values, X).fit()
        lam = -res.params[1]
        if lam <= 0: return np.nan
        return np.log(2) / lam
    except Exception:
        return np.nan


def backtest_filtered(p1, p2, px, bw=60, zw=40, entry_z=2.0, exit_z=0.5,
                     stop_z=4.0, max_hold=20, cost=COST_BPS,
                     use_regime=True, use_vol=True, use_hl=True):
    lp = np.log(px[[p1, p2]].dropna())
    if len(lp) < bw + zw + 260: return None, None

    betas = rolling_beta(lp[p1], lp[p2], bw)
    spread = pd.Series(index=lp.index, dtype=float)
    for i in range(len(lp)):
        b = betas.iloc[i]
        if pd.notna(b):
            spread.iloc[i] = lp[p1].iloc[i] - b * lp[p2].iloc[i]

    mu = spread.rolling(zw).mean(); sd = spread.rolling(zw).std()
    z = (spread - mu) / sd

    # ---- フィルタ系列 ----
    # (a) 200日MA 乖離
    ma200_1 = lp[p1].rolling(200).mean()
    ma200_2 = lp[p2].rolling(200).mean()
    dev1 = (lp[p1] - ma200_1).abs()
    dev2 = (lp[p2] - ma200_2).abs()

    # (b) 60日実現ボラ と 252日中央値
    ret1 = lp[p1].diff()
    vol1 = ret1.rolling(60).std()
    vol1_med = vol1.rolling(252).median()

    # (c) 60日半減期 (rolling)
    hl = pd.Series(index=lp.index, dtype=float)
    for i in range(260, len(lp)):
        hl.iloc[i] = compute_half_life(spread.iloc[max(0,i-60):i])

    trades = []
    pos = 0; ei = None; es = None; ez = None; eb = None
    filter_stats = dict(total_signals=0, pass_regime=0, pass_vol=0, pass_hl=0, pass_all=0)

    for i in range(bw + zw + 260, len(spread)):
        zi = z.iloc[i]
        if pd.isna(zi): continue
        if pos == 0 and abs(zi) >= entry_z:
            filter_stats["total_signals"] += 1
            ok_regime = True; ok_vol = True; ok_hl = True
            if use_regime:
                ok_regime = (dev1.iloc[i] < 0.15) and (dev2.iloc[i] < 0.15)
                if ok_regime: filter_stats["pass_regime"] += 1
            if use_vol:
                ok_vol = pd.notna(vol1.iloc[i]) and pd.notna(vol1_med.iloc[i]) and \
                         vol1.iloc[i] <= vol1_med.iloc[i]
                if ok_vol: filter_stats["pass_vol"] += 1
            if use_hl:
                ok_hl = pd.notna(hl.iloc[i]) and 3 <= hl.iloc[i] <= 30
                if ok_hl: filter_stats["pass_hl"] += 1
            if ok_regime and ok_vol and ok_hl:
                filter_stats["pass_all"] += 1
                pos = -1 if zi >= entry_z else 1
                ei = i; es = spread.iloc[i]; ez = zi; eb = betas.iloc[i]
        elif pos != 0:
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
                                   hold=hold, net=net, reason=r))
                pos = 0
    return pd.DataFrame(trades), filter_stats


def summary(td):
    if td is None or len(td) == 0: return dict(n=0)
    arr = td["net"].values
    sd = arr.std() if arr.std() > 0 else 1e-9
    return dict(n=len(td), mean=arr.mean(),
                sharpe=arr.mean()/sd*np.sqrt(252/td["hold"].mean()),
                t=arr.mean()/sd*np.sqrt(len(arr)),
                wr=(arr>0).mean()*100)


def main():
    print("=" * 120)
    print("#3 条件付きペアMR: Regime / Vol / HalfLife フィルタ")
    print("=" * 120)
    syms = list({s for p1,p2,*_ in PAIRS for s in (p1,p2)})
    px = load_maria(syms)
    print(f"データ: {len(px)}日\n")

    # フィルタなし vs あり
    for flt_label, flags in [("なし (ベースライン)", dict(use_regime=False, use_vol=False, use_hl=False)),
                              ("Regimeのみ",          dict(use_regime=True,  use_vol=False, use_hl=False)),
                              ("Volのみ",             dict(use_regime=False, use_vol=True,  use_hl=False)),
                              ("HLのみ",              dict(use_regime=False, use_vol=False, use_hl=True)),
                              ("Regime+Vol",          dict(use_regime=True,  use_vol=True,  use_hl=False)),
                              ("全フィルタ",            dict(use_regime=True,  use_vol=True,  use_hl=True))]:
        print(f"\n【フィルタ: {flt_label}】")
        print(f"  {'ペア':<26} {'N':>4} {'mean':>7} {'Sharpe':>7} {'t':>6} {'WR':>5}")
        all_nets = []
        tot_n = 0; sum_mean = 0
        for p1, p2, lbl, ez, zw, hd in PAIRS:
            td, fs = backtest_filtered(p1, p2, px, zw=zw, entry_z=ez, max_hold=hd,
                                       exit_z=max(0.3, ez*0.25), **flags)
            s = summary(td)
            if s["n"] >= 5:
                print(f"  {lbl:<26} {s['n']:>4d} {s['mean']:>+7.1f} {s['sharpe']:>+7.2f} "
                      f"{s['t']:>+6.2f} {s['wr']:>5.1f}")
                all_nets.append(td["net"].values)
                tot_n += s["n"]; sum_mean += s["n"] * s["mean"]
        # 全ペア合算 (等配分 trade 積み上げ)
        if all_nets:
            cat = np.concatenate(all_nets)
            sd_all = cat.std() if cat.std() > 0 else 1e-9
            print(f"  {'合算':<26} {len(cat):>4d} {cat.mean():>+7.1f} "
                  f"{cat.mean()/sd_all*np.sqrt(252/20):>+7.2f} "
                  f"{cat.mean()/sd_all*np.sqrt(len(cat)):>+6.2f} "
                  f"{(cat>0).mean()*100:>5.1f}")


if __name__ == "__main__":
    main()
