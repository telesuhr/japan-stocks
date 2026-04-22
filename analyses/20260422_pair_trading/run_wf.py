#!/usr/bin/env python3
"""
ペアトレード Walk-Forward 検証 (look-ahead バイアス除去)

各日 t でβを trailing window で推定 → スプレッドZスコアでエントリー。
H1/H2 時間分割で OoS 検証。
"""
import warnings
from itertools import combinations
import numpy as np
import pandas as pd
import psycopg2
import statsmodels.api as sm

warnings.filterwarnings("ignore")

PG = dict(host="localhost", port=5432, user="postgres", dbname="market_data")

# Phase 1 で良好だった上位候補 + sector内コインテグレーション上位
CANDIDATE_PAIRS = [
    # (p1, p2, ラベル)
    ("7011.T", "7013.T", "重工: MHI/IHI"),
    ("8306.T", "8411.T", "銀行: MUFG/みずほ"),
    ("8306.T", "8316.T", "銀行: MUFG/SMFG"),
    ("8316.T", "8411.T", "銀行: SMFG/みずほ"),
    ("6146.T", "6323.T", "半導体: ディスコ/ローツェ"),
    ("9432.T", "9434.T", "通信: NTT/SB"),
    ("8002.T", "8031.T", "商社: 丸紅/三井物産"),
    ("4502.T", "4568.T", "医薬: 武田/第一三共"),
    ("9101.T", "9107.T", "海運: 郵船/川崎汽船"),
    ("5711.T", "5713.T", "非鉄: 三菱マテ/住友金鉱"),
    ("7203.T", "7261.T", "自動車: トヨタ/マツダ"),
    ("8801.T", "8802.T", "不動産: 三井不/三菱地"),
    ("6501.T", "6503.T", "電機: 日立/三菱電"),
    ("8053.T", "8058.T", "商社: 住友商/三菱商"),
    ("8015.T", "8053.T", "商社: 豊通/住友商"),
    ("7270.T", "7269.T", "自動車: スバル/スズキ"),
    ("2502.T", "2503.T", "ビール: アサヒ/キリン"),
    ("9020.T", "9022.T", "鉄道: JR東/JR東海"),
    ("4503.T", "4578.T", "医薬: アステラス/大塚"),
    ("1605.T", "5020.T", "エネルギー: INPEX/ENEOS"),
]

COST_BPS_PER_SIDE = 4.0


def load_daily_close(symbols):
    conn = psycopg2.connect(**PG)
    frames = {}
    for s in symbols:
        df = pd.read_sql(
            "SELECT timestamp, close FROM intraday_data WHERE symbol=%s ORDER BY timestamp",
            conn, params=(s,))
        if df.empty:
            continue
        df["jst"] = pd.to_datetime(df["timestamp"]) + pd.Timedelta(hours=9)
        df = df.set_index("jst").sort_index()
        daily = df["close"].groupby(df.index.date).last()
        daily.index = pd.to_datetime(daily.index)
        frames[s] = daily
    conn.close()
    if not frames:
        return pd.DataFrame()
    px = pd.concat(frames, axis=1).dropna(how="any")
    return px


def rolling_beta(lp1, lp2, beta_window=60):
    """trailing beta_window 日で β をローリング推定"""
    betas = pd.Series(index=lp1.index, dtype=float)
    for i in range(beta_window, len(lp1)):
        y = lp1.iloc[i-beta_window:i]
        x = lp2.iloc[i-beta_window:i]
        X = sm.add_constant(x)
        try:
            b = sm.OLS(y, X).fit().params.iloc[1]
            betas.iloc[i] = b
        except Exception:
            pass
    return betas


def walk_forward_backtest(p1, p2, px, beta_window=60, z_window=40,
                          entry_z=2.0, exit_z=0.5, stop_z=4.0, max_hold=20,
                          cost_bps=COST_BPS_PER_SIDE * 2):
    """Walk-forward: β は trailing 推定。z-score も trailing。"""
    lp = np.log(px[[p1, p2]].dropna())
    if len(lp) < beta_window + z_window + 30:
        return None, None

    betas = rolling_beta(lp[p1], lp[p2], beta_window)
    # スプレッドも各日でそのときのβを使って計算
    spread = pd.Series(index=lp.index, dtype=float)
    for i in range(len(lp)):
        b = betas.iloc[i]
        if pd.notna(b):
            spread.iloc[i] = lp[p1].iloc[i] - b * lp[p2].iloc[i]

    # Z-score: trailing z_window
    mu = spread.rolling(z_window).mean()
    sd = spread.rolling(z_window).std()
    z = (spread - mu) / sd

    start_idx = beta_window + z_window
    trades = []
    pos = 0; entry_idx = None; entry_spread = None; entry_z_val = None; entry_beta = None
    for i in range(start_idx, len(spread)):
        zi = z.iloc[i]
        if pd.isna(zi):
            continue
        if pos == 0:
            if zi >= entry_z:
                pos = -1; entry_idx = i; entry_spread = spread.iloc[i]
                entry_z_val = zi; entry_beta = betas.iloc[i]
            elif zi <= -entry_z:
                pos = 1; entry_idx = i; entry_spread = spread.iloc[i]
                entry_z_val = zi; entry_beta = betas.iloc[i]
        else:
            hold = i - entry_idx
            reason = None
            if abs(zi) < exit_z:
                reason = "MR"
            elif abs(zi) > stop_z:
                reason = "STOP"
            elif hold >= max_hold:
                reason = "TIME"
            if reason:
                # エントリー時の β でロックして計算 (実運用と同じ)
                spread_now = lp[p1].iloc[i] - entry_beta * lp[p2].iloc[i]
                pnl_log = pos * (spread_now - entry_spread)
                gross_notional = 1 + abs(entry_beta)
                gross_bps = pnl_log * 10000 / gross_notional
                net_bps = gross_bps - cost_bps
                trades.append(dict(
                    entry_date=spread.index[entry_idx], exit_date=spread.index[i],
                    entry_z=entry_z_val, exit_z=zi, hold=hold, pos=pos,
                    beta=entry_beta, gross_bps=gross_bps, net_bps=net_bps,
                    reason=reason))
                pos = 0

    td = pd.DataFrame(trades)
    return td, z


def summarize(td, label=""):
    if td is None or len(td) == 0:
        return dict(label=label, n=0)
    arr = td["net_bps"].values
    wr = (arr > 0).mean() * 100
    pf = arr[arr > 0].sum() / abs(arr[arr < 0].sum()) if (arr < 0).any() else np.inf
    sharpe = arr.mean() / arr.std() * np.sqrt(252 / td["hold"].mean()) if arr.std() > 0 else 0
    t = arr.mean() / arr.std() * np.sqrt(len(arr)) if arr.std() > 0 else 0
    return dict(label=label, n=len(td),
                mean_bps=arr.mean(), median=np.median(arr),
                sharpe=sharpe, t=t, wr=wr, pf=pf,
                avg_hold=td["hold"].mean(),
                mr=int((td["reason"]=="MR").sum()),
                stop=int((td["reason"]=="STOP").sum()),
                time=int((td["reason"]=="TIME").sum()))


def split_h1_h2(td):
    if len(td) == 0:
        return None, None
    td = td.sort_values("entry_date").reset_index(drop=True)
    mid = td["entry_date"].iloc[len(td)//2]
    h1 = td[td["entry_date"] < mid]
    h2 = td[td["entry_date"] >= mid]
    return h1, h2


def main():
    print("=" * 100)
    print("ペアトレード Walk-Forward バックテスト (β を trailing 60日 推定 / z-score 40日)")
    print("パラメータ: Entry|Z|≥2.0 / Exit|Z|≤0.5 / Stop|Z|≥4.0 / MaxHold=20日 / Cost=8bps/往復")
    print("=" * 100)

    syms_all = list({s for p1, p2, _ in CANDIDATE_PAIRS for s in (p1, p2)})
    px_all = load_daily_close(syms_all)
    print(f"\nデータ: {len(px_all)}日 ({px_all.index.min().date()} 〜 {px_all.index.max().date()})  銘柄数 {len(px_all.columns)}")

    rows = []
    for p1, p2, label in CANDIDATE_PAIRS:
        if p1 not in px_all.columns or p2 not in px_all.columns:
            continue
        td, z = walk_forward_backtest(p1, p2, px_all)
        if td is None:
            continue
        full = summarize(td, label)
        h1, h2 = split_h1_h2(td)
        h1s = summarize(h1, "H1") if h1 is not None else {}
        h2s = summarize(h2, "H2") if h2 is not None else {}
        rows.append(dict(
            pair=label, p1=p1, p2=p2,
            n=full["n"], mean_bps=full.get("mean_bps", 0),
            sharpe=full.get("sharpe", 0), t=full.get("t", 0),
            wr=full.get("wr", 0), pf=full.get("pf", 0),
            avg_hold=full.get("avg_hold", 0),
            mr=full.get("mr", 0), stop=full.get("stop", 0), time=full.get("time", 0),
            h1_n=h1s.get("n", 0), h1_sharpe=h1s.get("sharpe", 0), h1_mean=h1s.get("mean_bps", 0),
            h2_n=h2s.get("n", 0), h2_sharpe=h2s.get("sharpe", 0), h2_mean=h2s.get("mean_bps", 0),
        ))

    df = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    df.to_csv("walkforward.csv", index=False)

    print("\n【Full期間】")
    cols_full = ["pair", "n", "mean_bps", "sharpe", "t", "wr", "pf", "avg_hold", "mr", "stop", "time"]
    print(df[cols_full].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print("\n【H1/H2 OoS 検証】")
    cols_oos = ["pair", "n", "sharpe", "h1_n", "h1_sharpe", "h1_mean", "h2_n", "h2_sharpe", "h2_mean"]
    print(df[cols_oos].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print("\n" + "=" * 100)
    print("採用候補 (Full Sharpe≥2 & N≥20 & t≥2 & H1/H2ともに Sharpe≥1)")
    print("=" * 100)
    adopt = df[(df["sharpe"] >= 2.0) & (df["n"] >= 20) & (df["t"] >= 2.0) &
               (df["h1_sharpe"] >= 1.0) & (df["h2_sharpe"] >= 1.0)]
    if len(adopt):
        print(adopt[cols_full + ["h1_sharpe", "h2_sharpe"]]
              .to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    else:
        print("なし")
        print("\n参考: Full Sharpe≥1.5 の緩和基準")
        loose = df[(df["sharpe"] >= 1.5) & (df["n"] >= 15) & (df["h2_sharpe"] >= 0.5)]
        if len(loose):
            print(loose[cols_full + ["h1_sharpe", "h2_sharpe"]]
                  .to_string(index=False, float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__":
    main()
