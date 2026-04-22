#!/usr/bin/env python3
"""
イントラデイ ペアトレード分析 (5分足)

5分リサンプル → 同一セッション内でZ-score MR。
翌日持ち越しなし (15:30 強制フラット)。Walk-forward でβ推定。

データ: 日本株分足、JST 9:00-15:30。
閾値: |Z|>2 Entry, |Z|<0.5 Exit, |Z|>4 Stop, 最大保有 60分。
コスト: 片側 4bps × 2銘柄 × 往復 = 16bps。
"""
import warnings
import numpy as np
import pandas as pd
import psycopg2
import statsmodels.api as sm
from pathlib import Path

warnings.filterwarnings("ignore")
PG = dict(host="localhost", port=5432, user="postgres", dbname="market_data")

# 相関・ビジネス類似性が高いペアを厳選
PAIRS = [
    ("8306.T", "8316.T", "銀行: MUFG/SMFG"),
    ("8306.T", "8411.T", "銀行: MUFG/みずほ"),
    ("8316.T", "8411.T", "銀行: SMFG/みずほ"),
    ("6146.T", "6323.T", "半導体: ディスコ/ローツェ"),
    ("8002.T", "8031.T", "商社: 丸紅/三井物産"),
    ("8053.T", "8058.T", "商社: 住友商/三菱商"),
    ("9101.T", "9104.T", "海運: 郵船/商船三井"),
    ("9101.T", "9107.T", "海運: 郵船/川崎汽船"),
    ("9104.T", "9107.T", "海運: 商船三井/川崎汽船"),
    ("7203.T", "7267.T", "自動車: トヨタ/ホンダ"),
    ("5801.T", "5802.T", "電線: 古河/住電"),
    ("6501.T", "6503.T", "電機: 日立/三菱電"),
    ("5711.T", "5713.T", "非鉄: 三菱マテ/住友金鉱"),
    ("5711.T", "5706.T", "非鉄: 三菱マテ/三井金"),
    ("8801.T", "8802.T", "不動産: 三井不/三菱地"),
    ("9432.T", "9433.T", "通信: NTT/KDDI"),
    ("7011.T", "7013.T", "重工: MHI/IHI"),
    ("4502.T", "4568.T", "医薬: 武田/第一三共"),
    ("2502.T", "2503.T", "ビール: アサヒ/キリン"),
    ("8035.T", "6146.T", "半導体: TEL/ディスコ"),
    ("8035.T", "6920.T", "半導体: TEL/レーザー"),
    ("6146.T", "6920.T", "半導体: ディスコ/レーザー"),
    ("6857.T", "6920.T", "半導体: アドバン/レーザー"),
]

COST_BPS = 16.0  # 往復 2銘柄


def load_5min(sym, start=None):
    conn = psycopg2.connect(**PG)
    q = "SELECT timestamp, close FROM intraday_data WHERE symbol=%s"
    params = [sym]
    if start:
        q += " AND timestamp >= %s"; params.append(start)
    q += " ORDER BY timestamp"
    df = pd.read_sql(q, conn, params=tuple(params))
    conn.close()
    if df.empty: return None
    df["jst"] = pd.to_datetime(df["timestamp"]) + pd.Timedelta(hours=9)
    df = df.set_index("jst")["close"].astype(float)
    # 取引時間のみ
    h = df.index.hour; m = df.index.minute
    mask = ((h == 9)) | ((h >= 10) & (h < 11)) | ((h == 11) & (m <= 30)) | \
           ((h == 12) & (m >= 30)) | ((h == 13)) | ((h == 14)) | ((h == 15) & (m <= 30))
    df = df[mask]
    # 5分リサンプル (session内)
    r5 = df.resample("5min", label="right", closed="right").last().dropna()
    # 取引時間フィルタ再適用
    h = r5.index.hour; m = r5.index.minute
    mask2 = ((h == 9) & (m >= 5)) | ((h >= 10) & (h < 11)) | ((h == 11) & (m <= 30)) | \
           ((h == 12) & (m >= 35)) | ((h == 13)) | ((h == 14)) | ((h == 15) & (m <= 30))
    r5 = r5[mask2]
    return r5


def session_index(idx):
    """各 bar がどの取引日の何分目かを返す"""
    dates = pd.Series(idx.date, index=idx)
    return dates


def backtest_intraday(p1, p2, px1, px2, beta_days=20, z_bars=30,
                      entry_z=2.0, exit_z=0.5, stop_z=4.0, max_hold_bars=12,
                      cost_bps=COST_BPS):
    """
    イントラデイ pair MR:
      - β は過去 beta_days の日足終値 (resample) から OLS 推定 (walk-forward)
      - z-score は同日中の直近 z_bars 分足スプレッドから
      - エントリー後 max_hold_bars (5分×12=60分) 経過で time exit
      - 15:25 に強制フラット
    """
    # 共通 index に揃える
    px = pd.concat({p1: px1, p2: px2}, axis=1).dropna()
    lp = np.log(px)
    # 日次 β (trailing beta_days days of *session close*)
    daily_close = lp.groupby(lp.index.date).last()
    betas_daily = pd.Series(index=daily_close.index, dtype=float)
    for i in range(beta_days, len(daily_close)):
        y = daily_close[p1].iloc[i-beta_days:i]
        x = daily_close[p2].iloc[i-beta_days:i]
        X = sm.add_constant(x)
        try: betas_daily.iloc[i] = sm.OLS(y, X).fit().params.iloc[1]
        except: pass

    trades = []
    # 日ごとに処理
    for date, grp in lp.groupby(lp.index.date):
        if date not in betas_daily.index: continue
        beta = betas_daily.loc[date]
        if pd.isna(beta): continue
        spread = grp[p1] - beta * grp[p2]
        if len(spread) < z_bars + 2: continue
        mu = spread.rolling(z_bars).mean()
        sd = spread.rolling(z_bars).std()
        z = (spread - mu) / sd

        pos = 0; ei = None; es = None; ez = None
        idxs = spread.index.tolist()
        for j, t in enumerate(idxs):
            zi = z.iloc[j]
            if pd.isna(zi): continue
            # 15:25 までに必ずフラット
            is_last = (t.hour == 15 and t.minute >= 25)
            if pos == 0 and not is_last:
                if zi >= entry_z:
                    pos = -1; ei = j; es = spread.iloc[j]; ez = zi
                elif zi <= -entry_z:
                    pos = 1; ei = j; es = spread.iloc[j]; ez = zi
            elif pos != 0:
                hold_bars = j - ei
                r = None
                if abs(zi) < exit_z: r = "MR"
                elif abs(zi) > stop_z: r = "STOP"
                elif hold_bars >= max_hold_bars: r = "TIME"
                elif is_last: r = "EOD"
                if r:
                    sn = spread.iloc[j]
                    pnl = pos * (sn - es)
                    gross = pnl * 10000 / (1 + abs(beta))
                    net = gross - cost_bps
                    trades.append(dict(
                        entry_time=idxs[ei], exit_time=t,
                        hold_bars=hold_bars, ez=ez, pos=pos, beta=beta,
                        gross_bps=gross, net_bps=net, reason=r))
                    pos = 0
    return pd.DataFrame(trades)


def summary(td, label=""):
    if td is None or len(td) == 0:
        return dict(label=label, n=0)
    arr = td["net_bps"].values
    sd = arr.std() if arr.std() > 0 else 1e-9
    # Intraday: 日次サンプル (trade ~ 同日中) → 年率化は少なくとも N/年 で
    # ペアTradeは独立と見なし N trades の t-stat
    wr = (arr > 0).mean() * 100
    return dict(
        label=label, n=len(td),
        mean=arr.mean(), median=np.median(arr),
        sharpe_trade=arr.mean()/sd*np.sqrt(252),  # 1日1回換算の年率Sharpe
        t=arr.mean()/sd*np.sqrt(len(arr)),
        wr=wr,
        mr=int((td["reason"]=="MR").sum()),
        stop=int((td["reason"]=="STOP").sum()),
        time=int((td["reason"]=="TIME").sum()),
        eod=int((td["reason"]=="EOD").sum()),
        avg_hold_bars=td["hold_bars"].mean())


def main():
    print("=" * 120)
    print("イントラデイ ペアトレード (5分足) Walk-Forward バックテスト")
    print("  β: trailing 20日 日次終値OLS / Z-score: 同日中の直近30バー")
    print("  Entry|Z|≥2 / Exit|Z|<0.5 / Stop|Z|>4 / MaxHold 60分 / 15:25 EODフラット")
    print("  Cost: 16bps/往復 (片側4bps×2銘柄×往復)")
    print("=" * 120)

    rows = []
    for p1, p2, lbl in PAIRS:
        px1 = load_5min(p1); px2 = load_5min(p2)
        if px1 is None or px2 is None:
            continue
        td = backtest_intraday(p1, p2, px1, px2)
        s = summary(td, lbl)
        # H1/H2 split
        if s["n"] >= 10:
            td2 = td.sort_values("entry_time").reset_index(drop=True)
            mid = len(td2) // 2
            h1 = summary(td2.iloc[:mid], "H1")
            h2 = summary(td2.iloc[mid:], "H2")
        else:
            h1 = {"n": 0}; h2 = {"n": 0}
        rows.append({**s, "h1_n": h1.get("n",0), "h1_sharpe": h1.get("sharpe_trade",0),
                     "h1_mean": h1.get("mean",0),
                     "h2_n": h2.get("n",0), "h2_sharpe": h2.get("sharpe_trade",0),
                     "h2_mean": h2.get("mean",0)})

    df = pd.DataFrame(rows).sort_values("sharpe_trade", ascending=False)
    df.to_csv(Path(__file__).parent / "intraday_results.csv", index=False)

    print(f"\n{'ペア':<24} {'N':>4} {'mean':>7} {'median':>7} {'Sharpe':>7} {'t':>6} {'WR':>5} {'MR':>4} {'STP':>4} {'TIM':>4} {'EOD':>4} {'bar':>4}")
    print("-" * 120)
    for _, r in df.iterrows():
        if r["n"] == 0:
            print(f"{r['label']:<24}   (データ不足)")
            continue
        print(f"{r['label']:<24} {r['n']:>4d} {r['mean']:>+7.1f} {r['median']:>+7.1f} "
              f"{r['sharpe_trade']:>+7.2f} {r['t']:>+6.2f} {r['wr']:>5.1f} "
              f"{r['mr']:>4d} {r['stop']:>4d} {r['time']:>4d} {r['eod']:>4d} "
              f"{r['avg_hold_bars']:>4.1f}")

    print("\n" + "=" * 120)
    print("H1/H2 OoS")
    print("=" * 120)
    print(f"{'ペア':<24} {'Full N':>6} {'Sh':>6} {'t':>5} |  {'H1N':>4} {'H1Sh':>6} {'H1M':>7} | {'H2N':>4} {'H2Sh':>6} {'H2M':>7}")
    print("-" * 120)
    for _, r in df.iterrows():
        if r["n"] < 10: continue
        print(f"{r['label']:<24} {r['n']:>6d} {r['sharpe_trade']:>+6.2f} {r['t']:>+5.2f} | "
              f"{r['h1_n']:>4d} {r['h1_sharpe']:>+6.2f} {r['h1_mean']:>+7.1f} | "
              f"{r['h2_n']:>4d} {r['h2_sharpe']:>+6.2f} {r['h2_mean']:>+7.1f}")

    print("\n" + "=" * 120)
    print("採用候補 (Sharpe≥2 & N≥30 & t≥2 & H2 Sharpe≥1)")
    print("=" * 120)
    adopt = df[(df["sharpe_trade"] >= 2.0) & (df["n"] >= 30) & (df["t"] >= 2.0) &
               (df["h2_sharpe"] >= 1.0)]
    if len(adopt) > 0:
        print(adopt.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    else:
        print("なし")


if __name__ == "__main__":
    main()
