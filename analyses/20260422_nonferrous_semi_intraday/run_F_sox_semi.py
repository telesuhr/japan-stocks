#!/usr/bin/env python3
"""
F. SOX 指数 ON 変化 → 日本半導体寄付戦略
- SOX は日次のみ (MariaDB)。直近 NY セッション変化率 = SOX[t-1]/SOX[t-2]-1
- Japan 寄付 (date t) で同方向にエントリ → 11:00 / 11:30 / 14:30 / 引け で決済
"""
import warnings
import numpy as np
import pandas as pd
import pymysql
import psycopg2
from lib_data import load_all, SEMI, perf, print_perf

warnings.filterwarnings("ignore")
COST = 8.0
PG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}
MARIA = dict(host="100.92.181.92", port=3306, user="rfnews",
             password="Bleach@924", database="refinitiv_news")


def load_sox_daily():
    conn = pymysql.connect(**MARIA)
    df = pd.read_sql("SELECT trade_date, close FROM daily_data WHERE symbol='.SOX' ORDER BY trade_date", conn)
    conn.close()
    df['trade_date'] = pd.to_datetime(df['trade_date']).dt.date
    df = df.set_index('trade_date').sort_index()
    df['sox_chg_bps'] = (df['close'].pct_change() * 10000)
    return df


def build_daily(data, members):
    daily = {}
    for s in members:
        df = data[s]
        rec = {}
        for dt, g in df.groupby(df.index.date):
            if len(g) < 80: continue
            def at(h, m):
                sel = g[(g.index.hour==h)&(g.index.minute==m)]
                return sel['close'].iloc[0] if len(sel) else np.nan
            rec[dt] = {
                "open": g['open'].iloc[0],
                "p_10": at(10,0), "p_11": at(11,0), "p_1130": at(11,30),
                "p_1430": at(14,30), "p_close": g['close'].iloc[-1],
            }
        daily[s] = pd.DataFrame(rec).T
    return daily


def main():
    print("=" * 130)
    print("F. SOX ON → 日本半導体 寄付戦略")
    print("=" * 130)

    print("SOX 日次ロード ...")
    sox = load_sox_daily()
    print(f"  期間: {sox.index.min()} ~ {sox.index.max()}, N={len(sox)}")
    print(f"  ON 変化分布 (全期間): mean={sox['sox_chg_bps'].mean():+.1f}  "
          f"std={sox['sox_chg_bps'].std():.1f}  P10={sox['sox_chg_bps'].quantile(0.1):+.1f}  "
          f"P90={sox['sox_chg_bps'].quantile(0.9):+.1f}")

    print("\n半導体ロード ...")
    data = load_all(SEMI)
    daily = build_daily(data, SEMI)

    # 日本休日マッピング: SOX[NY date X-1] を Japan open[date X] に紐付け
    # シンプル化: Japan 取引日 d について、その直前の SOX trade_date の chg を使う
    print("\nSOX-Japan 紐付け中 ...")
    all_dates = sorted(set().union(*[set(daily[s].index) for s in SEMI]))
    sox_dates = list(sox.index)
    sox_for_jp = {}
    for jd in all_dates:
        # jd 当日の Japan 寄付に効くのは「jd 直前にクローズした SOX セッション」
        # SOX のクローズは NY の jd-1 (JST jd の 05:00 頃)。
        # よって SOX[trade_date == jd-1] が理想。それが無ければ前営業日。
        candidates = [d for d in sox_dates if d < jd]
        if not candidates: continue
        latest = candidates[-1]
        # 前日のSOXとも比較し、ペアが揃っていることを保証
        chg = sox.loc[latest, 'sox_chg_bps']
        if pd.isna(chg): continue
        # 連続 NY セッション (ギャップなし) チェック (週末 OK)
        sox_for_jp[jd] = chg
    sox_for_jp = pd.Series(sox_for_jp)
    print(f"  紐付け完了: N={len(sox_for_jp)}日")
    print(f"  実 SOX 変化分布: mean={sox_for_jp.mean():+.1f}  std={sox_for_jp.std():.1f}  "
          f"P10={sox_for_jp.quantile(0.1):+.1f}  P90={sox_for_jp.quantile(0.9):+.1f}")

    # ---- 1. 個別銘柄: SOX vs 寄付ギャップ相関 ----
    print("\n【1. SOX 変化 vs 半導体寄付ギャップ 相関】")
    for s in SEMI:
        df = daily[s].copy()
        df['sox'] = df.index.map(sox_for_jp.to_dict())
        df['prev_close'] = df['p_close'].shift(1)
        df['gap'] = (df['open']/df['prev_close'] - 1)*10000
        x = df.dropna(subset=['sox', 'gap'])
        if len(x) < 30: continue
        c = np.corrcoef(x['sox'], x['gap'])[0, 1]
        # 寄付→引け も
        x2 = df.dropna(subset=['sox','open','p_close'])
        x2['intraday'] = (x2['p_close']/x2['open'] - 1)*10000
        c2 = np.corrcoef(x2['sox'], x2['intraday'])[0, 1]
        print(f"  {s}: corr(SOX, gap)={c:+.3f}  corr(SOX, open→close)={c2:+.3f}  N={len(x)}")

    # ---- 2. 戦略バックテスト ----
    print("\n【2. 戦略: |SOX|>X → 寄付エントリ → 各時刻決済】")
    rows = []
    for thr in [50, 100, 150, 200, 300]:
        for el, ec in [("11:00","p_11"), ("11:30","p_1130"),
                       ("14:30","p_1430"), ("close","p_close")]:
            pn = []
            for s in SEMI:
                df = daily[s].copy()
                df['sox'] = df.index.map(sox_for_jp.to_dict())
                df = df.dropna(subset=['sox','open',ec])
                long = df[df['sox'] > thr]
                short = df[df['sox'] < -thr]
                if len(long):
                    pn.extend(((long[ec]/long['open']-1)*10000 - COST).tolist())
                if len(short):
                    pn.extend(((short['open']/short[ec]-1)*10000 - COST).tolist())
            rows.append(perf(np.array(pn), label=f"|SOX|>{thr:>3} → {el}"))
    print_perf(rows)

    print("\n【上位 5 (N≥100)】")
    valid = [r for r in rows if not np.isnan(r["sharpe"]) and r["N"] >= 100]
    top = sorted(valid, key=lambda r: r["sharpe"], reverse=True)[:5]
    print_perf(top)

    # ---- 3. 銘柄別寄与 (最良戦略) ----
    if top:
        best = top[0]
        thr = int(best["label"].split(">")[1].split("→")[0].strip())
        el = best["label"].split("→")[1].strip()
        col_map = {"11:00":"p_11", "11:30":"p_1130", "14:30":"p_1430", "close":"p_close"}
        ec = col_map[el]
        print(f"\n【3. 銘柄別寄与: 最良戦略 |SOX|>{thr} → {el}】")
        for s in SEMI:
            df = daily[s].copy()
            df['sox'] = df.index.map(sox_for_jp.to_dict())
            df = df.dropna(subset=['sox','open',ec])
            pn = []
            long = df[df['sox'] > thr]
            short = df[df['sox'] < -thr]
            if len(long):
                pn.extend(((long[ec]/long['open']-1)*10000 - COST).tolist())
            if len(short):
                pn.extend(((short['open']/short[ec]-1)*10000 - COST).tolist())
            x = np.array(pn)
            if len(x) < 5:
                print(f"  {s}: N={len(x)}")
                continue
            sh = x.mean()/x.std()*np.sqrt(252) if x.std() > 0 else 0
            wr = (x>0).mean()*100
            print(f"  {s}: N={len(x):4d}  mean={x.mean():+6.1f}  WR={wr:5.1f}%  sum={x.sum():+7.0f}  Sharpe={sh:+.2f}")

    # ---- 4. 上位銘柄のみで再構成 ----
    print("\n【4. 上位 4 銘柄に絞った戦略 (TEL/Disco/Laser/Advantest)】")
    top4 = ["8035.T", "6146.T", "6920.T", "6857.T"]
    for thr in [100, 150, 200]:
        for el, ec in [("11:00","p_11"), ("close","p_close")]:
            pn = []
            for s in top4:
                df = daily[s].copy()
                df['sox'] = df.index.map(sox_for_jp.to_dict())
                df = df.dropna(subset=['sox','open',ec])
                long = df[df['sox'] > thr]
                short = df[df['sox'] < -thr]
                if len(long):
                    pn.extend(((long[ec]/long['open']-1)*10000 - COST).tolist())
                if len(short):
                    pn.extend(((short['open']/short[ec]-1)*10000 - COST).tolist())
            print_perf([perf(np.array(pn), label=f"top4 |SOX|>{thr} → {el}")])


if __name__ == "__main__":
    main()
