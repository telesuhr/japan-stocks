#!/usr/bin/env python3
"""
ペアトレード Walk-Forward 検証 (11年履歴 / MariaDB daily_data)

NAS MariaDB の daily_data (2015-01-05 〜 2026-04-22, 約2,760日) を使い、
各ペアで N≥50 を確保。H1/H2 で各5年超の OoS 検証を行う。

パラメータスイープ:
  Entry Z: 1.5 / 2.0 / 2.5
  Z-window: 20 / 40 / 60
  MaxHold: 10 / 20 / 30
  β-window: 60 (trailing)
"""
import warnings
from itertools import product
import numpy as np
import pandas as pd
import pymysql
import statsmodels.api as sm
from pathlib import Path

warnings.filterwarnings("ignore")

MARIA = dict(host="100.92.181.92", port=3306, user="rfnews",
             password="Bleach@924", database="refinitiv_news")

CANDIDATE_PAIRS = [
    ("7011.T", "7013.T", "重工: MHI/IHI"),
    ("8306.T", "8411.T", "銀行: MUFG/みずほ"),
    ("8306.T", "8316.T", "銀行: MUFG/SMFG"),
    ("8316.T", "8411.T", "銀行: SMFG/みずほ"),
    ("6146.T", "6323.T", "半導体: ディスコ/ローツェ"),
    ("9432.T", "9433.T", "通信: NTT/KDDI"),
    ("8002.T", "8031.T", "商社: 丸紅/三井物産"),
    ("4502.T", "4568.T", "医薬: 武田/第一三共"),
    ("9101.T", "9107.T", "海運: 郵船/川崎汽船"),
    ("9101.T", "9104.T", "海運: 郵船/商船三井"),
    ("9104.T", "9107.T", "海運: 商船三井/川崎汽船"),
    ("5711.T", "5713.T", "非鉄: 三菱マテ/住友金鉱"),
    ("5711.T", "5706.T", "非鉄: 三菱マテ/三井金"),
    ("7203.T", "7261.T", "自動車: トヨタ/マツダ"),
    ("7203.T", "7267.T", "自動車: トヨタ/ホンダ"),
    ("8801.T", "8802.T", "不動産: 三井不/三菱地"),
    ("6501.T", "6503.T", "電機: 日立/三菱電"),
    ("8053.T", "8058.T", "商社: 住友商/三菱商"),
    ("7270.T", "7269.T", "自動車: スバル/スズキ"),
    ("9020.T", "9022.T", "鉄道: JR東/JR東海"),
    ("4503.T", "4578.T", "医薬: アステラス/大塚"),
    ("6758.T", "6702.T", "電機: ソニー/富士通"),
    ("5802.T", "5801.T", "電線: 住電/古河"),
    ("2502.T", "2503.T", "ビール: アサヒ/キリン"),
    ("8035.T", "6146.T", "半導体: TEL/ディスコ"),
    ("6920.T", "6857.T", "半導体: レーザー/アドバン"),
    ("8035.T", "6920.T", "半導体: TEL/レーザー"),
    ("1605.T", "5020.T", "エネルギー: INPEX/ENEOS"),
]

COST_BPS = 8.0  # 片側4bps × 2銘柄 (= 16bps) の半分はスプレッド規模化で処理
                # → ここではエッジ/コスト比較用に 8bps/往復 (ペア) で統一


def load_daily_maria(symbols):
    """MariaDB daily_data から日次終値を取得。全銘柄揃う日のみ残す。"""
    conn = pymysql.connect(**MARIA)
    placeholders = ",".join(["%s"] * len(symbols))
    q = f"""SELECT symbol, trade_date, close FROM daily_data
            WHERE symbol IN ({placeholders}) ORDER BY symbol, trade_date"""
    df = pd.read_sql(q, conn, params=tuple(symbols))
    conn.close()
    if df.empty:
        return pd.DataFrame()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    px = df.pivot(index="trade_date", columns="symbol", values="close").astype(float)
    return px.dropna(how="any")


def rolling_beta(y, x, w):
    betas = pd.Series(index=y.index, dtype=float)
    for i in range(w, len(y)):
        Y = y.iloc[i-w:i]; X = sm.add_constant(x.iloc[i-w:i])
        try:
            betas.iloc[i] = sm.OLS(Y, X).fit().params.iloc[1]
        except Exception:
            pass
    return betas


def backtest(p1, p2, px, bw=60, zw=40, entry_z=2.0, exit_z=0.5, stop_z=4.0,
             max_hold=20, cost=COST_BPS):
    lp = np.log(px[[p1, p2]].dropna())
    if len(lp) < bw + zw + 30:
        return None
    betas = rolling_beta(lp[p1], lp[p2], bw)
    # spread (walk-forward β)
    spread = pd.Series(index=lp.index, dtype=float)
    for i in range(len(lp)):
        b = betas.iloc[i]
        if pd.notna(b):
            spread.iloc[i] = lp[p1].iloc[i] - b * lp[p2].iloc[i]
    mu = spread.rolling(zw).mean()
    sd = spread.rolling(zw).std()
    z = (spread - mu) / sd

    trades = []
    pos = 0; ei = None; es = None; ez = None; eb = None
    for i in range(bw + zw, len(spread)):
        zi = z.iloc[i]
        if pd.isna(zi):
            continue
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
                # エントリー時 β を固定して実運用ベースで PnL 計算
                sn = lp[p1].iloc[i] - eb * lp[p2].iloc[i]
                pnl = pos * (sn - es)
                gross = pnl * 10000 / (1 + abs(eb))
                net = gross - cost
                trades.append(dict(
                    entry=spread.index[ei], exit=spread.index[i],
                    ez=ez, hold=hold, beta=eb,
                    gross=gross, net=net, reason=r))
                pos = 0
    return pd.DataFrame(trades)


def summary(td):
    if td is None or len(td) == 0:
        return dict(n=0)
    arr = td["net"].values
    sd = arr.std() if arr.std() > 0 else 1e-9
    return dict(
        n=len(td), mean=arr.mean(),
        sharpe=arr.mean()/sd*np.sqrt(252/td["hold"].mean()),
        t=arr.mean()/sd*np.sqrt(len(arr)),
        wr=(arr > 0).mean()*100,
        pf=(arr[arr>0].sum()/abs(arr[arr<0].sum())) if (arr<0).any() else np.inf,
        avg_hold=td["hold"].mean(),
        mr=int((td["reason"]=="MR").sum()),
        stop=int((td["reason"]=="STOP").sum()),
        time=int((td["reason"]=="TIME").sum()))


def split_halves(td):
    if td is None or len(td) == 0:
        return None, None
    td2 = td.sort_values("entry").reset_index(drop=True)
    mid = len(td2) // 2
    return td2.iloc[:mid], td2.iloc[mid:]


def main():
    print("=" * 120)
    print("ペアトレード Walk-Forward 検証 (11年履歴 / MariaDB daily_data)")
    print("  β: trailing 60日 OLS  /  Z: 各パラメータで trailing")
    print("  Entry |Z|≥entry / Exit |Z|<exit / Stop |Z|>4 / MaxHold で time exit")
    print("  Cost: 8bps/往復 (ペア, スプレッド bps 単位)")
    print("=" * 120)

    syms = list({s for p1, p2, _ in CANDIDATE_PAIRS for s in (p1, p2)})
    px = load_daily_maria(syms)
    print(f"\nデータ: {len(px)}日 ({px.index.min().date()} 〜 {px.index.max().date()})  "
          f"銘柄 {len(px.columns)}")

    # --- 各ペアで固定パラメータ (E=2, ZW=40, HD=20) の Full run ---
    print("\n" + "=" * 120)
    print("【固定パラメータ】 E=2.0 / ZW=40 / HD=20")
    print("=" * 120)
    print(f"{'ペア':<26} {'N':>4} {'mean':>7} {'Sharpe':>7} {'t':>6} {'WR':>5} {'PF':>5} {'hold':>5} "
          f"{'MR':>4} {'STP':>4} {'TIM':>4}")
    print("-" * 120)

    fixed_rows = []
    for p1, p2, lbl in CANDIDATE_PAIRS:
        if p1 not in px.columns or p2 not in px.columns:
            print(f"{lbl:<26}  (データなし)")
            continue
        td = backtest(p1, p2, px)
        s = summary(td)
        if s["n"] == 0:
            print(f"{lbl:<26}  (トレードなし)")
            continue
        h1, h2 = split_halves(td)
        s1 = summary(h1); s2 = summary(h2)
        print(f"{lbl:<26} {s['n']:>4d} {s['mean']:>+7.1f} {s['sharpe']:>+7.2f} {s['t']:>+6.2f} "
              f"{s['wr']:>5.1f} {s['pf']:>5.2f} {s['avg_hold']:>5.1f} "
              f"{s['mr']:>4d} {s['stop']:>4d} {s['time']:>4d}")
        fixed_rows.append(dict(
            pair=lbl, p1=p1, p2=p2, **s,
            h1_n=s1.get("n",0), h1_sh=s1.get("sharpe",0), h1_mean=s1.get("mean",0),
            h2_n=s2.get("n",0), h2_sh=s2.get("sharpe",0), h2_mean=s2.get("mean",0),
            td=td))

    # --- H1/H2 OoS 表示 ---
    print("\n" + "=" * 120)
    print("【H1/H2 時間分割 OoS】固定パラメータ")
    print("=" * 120)
    print(f"{'ペア':<26} {'N':>4} {'Sh':>6} {'t':>5} |  {'H1N':>4} {'H1Sh':>6} {'H1M':>7} | "
          f"{'H2N':>4} {'H2Sh':>6} {'H2M':>7}")
    print("-" * 120)
    for r in fixed_rows:
        print(f"{r['pair']:<26} {r['n']:>4d} {r['sharpe']:>+6.2f} {r['t']:>+5.2f} | "
              f"{r['h1_n']:>4d} {r['h1_sh']:>+6.2f} {r['h1_mean']:>+7.1f} | "
              f"{r['h2_n']:>4d} {r['h2_sh']:>+6.2f} {r['h2_mean']:>+7.1f}")

    # --- パラメータスイープ ---
    print("\n" + "=" * 120)
    print("【パラメータスイープ】 各ペアの Best Sharpe (N≥30)")
    print("=" * 120)
    print(f"{'ペア':<26} {'E':>4} {'ZW':>3} {'HD':>3} {'N':>4} {'mean':>7} {'Sharpe':>7} "
          f"{'t':>6} {'WR':>5} | {'H1Sh':>6} {'H2Sh':>6}")
    print("-" * 120)

    ENTRY = [1.5, 2.0, 2.5]
    ZWIN  = [20, 40, 60]
    HOLD  = [10, 20, 30]

    sweep_rows = []
    for p1, p2, lbl in CANDIDATE_PAIRS:
        if p1 not in px.columns or p2 not in px.columns:
            continue
        best = None
        for ez, zw, hd in product(ENTRY, ZWIN, HOLD):
            td = backtest(p1, p2, px, zw=zw, entry_z=ez, max_hold=hd,
                          exit_z=max(0.3, ez*0.25))
            s = summary(td)
            if s["n"] < 30:
                continue
            if best is None or s["sharpe"] > best[0]["sharpe"]:
                best = (s, ez, zw, hd, td)
        if best is None:
            continue
        s, ez, zw, hd, td = best
        h1, h2 = split_halves(td)
        s1 = summary(h1); s2 = summary(h2)
        print(f"{lbl:<26} {ez:>4.1f} {zw:>3d} {hd:>3d} {s['n']:>4d} "
              f"{s['mean']:>+7.1f} {s['sharpe']:>+7.2f} {s['t']:>+6.2f} {s['wr']:>5.1f} | "
              f"{s1.get('sharpe',0):>+6.2f} {s2.get('sharpe',0):>+6.2f}")
        sweep_rows.append(dict(
            pair=lbl, p1=p1, p2=p2, ez=ez, zw=zw, hd=hd, **s,
            h1_sh=s1.get("sharpe",0), h2_sh=s2.get("sharpe",0),
            h1_mean=s1.get("mean",0), h2_mean=s2.get("mean",0),
            h1_n=s1.get("n",0), h2_n=s2.get("n",0)))

    # --- 採用判定 ---
    print("\n" + "=" * 120)
    print("【採用候補】 Full Sharpe≥2 & N≥50 & t≥2 & H1/H2 Sharpe≥1")
    print("=" * 120)
    if sweep_rows:
        df = pd.DataFrame(sweep_rows).sort_values("sharpe", ascending=False)
        df.to_csv(Path(__file__).parent / "long_history_results.csv", index=False)
        adopt = df[(df["sharpe"] >= 2.0) & (df["n"] >= 50) & (df["t"] >= 2.0) &
                   (df["h1_sh"] >= 1.0) & (df["h2_sh"] >= 1.0)]
        if len(adopt) > 0:
            print(adopt[["pair", "ez", "zw", "hd", "n", "mean", "sharpe", "t",
                        "wr", "h1_sh", "h2_sh"]]
                  .to_string(index=False, float_format=lambda x: f"{x:.2f}"))
        else:
            print("厳格基準では該当なし。")
            print("\n緩和基準 (Sharpe≥1.5 & N≥50 & H1/H2 Sharpe≥0.5):")
            loose = df[(df["sharpe"] >= 1.5) & (df["n"] >= 50) &
                       (df["h1_sh"] >= 0.5) & (df["h2_sh"] >= 0.5)]
            if len(loose) > 0:
                print(loose[["pair", "ez", "zw", "hd", "n", "mean", "sharpe", "t",
                            "wr", "h1_sh", "h2_sh"]]
                      .to_string(index=False, float_format=lambda x: f"{x:.2f}"))
            else:
                print("  → 該当なし")


if __name__ == "__main__":
    main()
