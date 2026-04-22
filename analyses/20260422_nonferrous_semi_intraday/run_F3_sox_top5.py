#!/usr/bin/env python3
"""F3. SOX フェード — 上位5銘柄 (TEL/Advantest/Rohm/Sociony/KOKUSAI) で精緻化"""
import warnings, numpy as np, pandas as pd, pymysql
from lib_data import load_all, perf, print_perf
warnings.filterwarnings("ignore")

TOP5 = ["8035.T", "6857.T", "6963.T", "6526.T", "6525.T"]
NAMES = {"8035.T":"TEL", "6857.T":"アドバンテスト", "6963.T":"ローム",
         "6526.T":"ソシオネクスト", "6525.T":"KOKUSAI"}


def main():
    conn = pymysql.connect(host="100.92.181.92", port=3306, user="rfnews",
                           password="Bleach@924", database="refinitiv_news")
    sox = pd.read_sql("SELECT trade_date, close FROM daily_data WHERE symbol='.SOX' ORDER BY trade_date", conn)
    conn.close()
    sox['trade_date'] = pd.to_datetime(sox['trade_date']).dt.date
    sox = sox.set_index('trade_date').sort_index()
    sox['chg'] = sox['close'].pct_change() * 10000

    data = load_all(TOP5)
    daily = {}
    for s in TOP5:
        rec = {}
        for dt, g in data[s].groupby(data[s].index.date):
            if len(g) < 80: continue
            def at(h, m):
                sel = g[(g.index.hour==h)&(g.index.minute==m)]
                return sel['close'].iloc[0] if len(sel) else np.nan
            rec[dt] = {"open": g['open'].iloc[0],
                       "p_10":at(10,0), "p_11":at(11,0), "p_1130":at(11,30),
                       "p_13":at(13,0), "p_1430":at(14,30),
                       "p_close": g['close'].iloc[-1]}
        daily[s] = pd.DataFrame(rec).T

    all_dates = sorted(set().union(*[set(daily[s].index) for s in TOP5]))
    sox_for_jp = {}
    sd = list(sox.index)
    for jd in all_dates:
        cands = [d for d in sd if d < jd]
        if not cands: continue
        chg = sox.loc[cands[-1], 'chg']
        if not pd.isna(chg): sox_for_jp[jd] = chg
    sox_for_jp = pd.Series(sox_for_jp)

    print("=" * 130)
    print(f"F3. SOX フェード — 上位5銘柄: {', '.join(NAMES.values())}")
    print("=" * 130)

    # スキャン (コスト=8bps)
    rows = []
    for cs in [4]:  # 片側コスト
        cost = cs * 2
        for thr in [30, 50, 80, 100, 150, 200]:
            for el, ec in [("10:00","p_10"),("11:00","p_11"),("11:30","p_1130"),
                           ("13:00","p_13"),("14:30","p_1430"),("close","p_close")]:
                pn = []
                for s in TOP5:
                    df = daily[s].copy()
                    df['sox'] = df.index.map(sox_for_jp.to_dict())
                    df = df.dropna(subset=['sox','open',ec])
                    fs = df[df['sox'] > thr]
                    fl = df[df['sox'] < -thr]
                    if len(fs): pn.extend(((fs['open']/fs[ec]-1)*10000 - cost).tolist())
                    if len(fl): pn.extend(((fl[ec]/fl['open']-1)*10000 - cost).tolist())
                rows.append(perf(np.array(pn), label=f"|SOX|>{thr:>3} → {el:<6}"))
    print_perf(rows)

    print("\n上位 10 (N≥150):")
    valid = [r for r in rows if not np.isnan(r["sharpe"]) and r["N"] >= 150]
    top = sorted(valid, key=lambda r: r["sharpe"], reverse=True)[:10]
    print_perf(top)

    # 最良戦略のコスト感応度
    if top:
        best = top[0]
        thr = int(best["label"].split(">")[1].split("→")[0].strip())
        el = best["label"].split("→")[1].strip()
        col_map = {"10:00":"p_10","11:00":"p_11","11:30":"p_1130",
                   "13:00":"p_13","14:30":"p_1430","close":"p_close"}
        ec = col_map[el]
        print(f"\n最良戦略コスト感応度: FADE |SOX|>{thr} → {el}")
        rows2 = []
        for cs in [2, 4, 6, 8, 10, 15]:
            cost = cs * 2
            pn = []
            for s in TOP5:
                df = daily[s].copy()
                df['sox'] = df.index.map(sox_for_jp.to_dict())
                df = df.dropna(subset=['sox','open',ec])
                fs = df[df['sox'] > thr]
                fl = df[df['sox'] < -thr]
                if len(fs): pn.extend(((fs['open']/fs[ec]-1)*10000 - cost).tolist())
                if len(fl): pn.extend(((fl[ec]/fl['open']-1)*10000 - cost).tolist())
            rows2.append(perf(np.array(pn), label=f"片側 {cs:>2} bps"))
        print_perf(rows2)

        # 銘柄別寄与
        print(f"\n銘柄別寄与 (片側 4bps):")
        cost = 8
        for s in TOP5:
            df = daily[s].copy()
            df['sox'] = df.index.map(sox_for_jp.to_dict())
            df = df.dropna(subset=['sox','open',ec])
            pn = []
            fs = df[df['sox'] > thr]
            fl = df[df['sox'] < -thr]
            if len(fs): pn.extend(((fs['open']/fs[ec]-1)*10000 - cost).tolist())
            if len(fl): pn.extend(((fl[ec]/fl['open']-1)*10000 - cost).tolist())
            x = np.array(pn)
            sh = x.mean()/x.std()*np.sqrt(252) if x.std()>0 else 0
            wr = (x>0).mean()*100
            print(f"  {s} ({NAMES[s]}): N={len(x):4d}  mean={x.mean():+6.1f}  WR={wr:5.1f}%  Sharpe={sh:+.2f}")

        # 日次PnL系列を保存 (相関分析用)
        cost = 8
        pnl_by_date = {}
        for s in TOP5:
            df = daily[s].copy()
            df['sox'] = df.index.map(sox_for_jp.to_dict())
            df = df.dropna(subset=['sox','open',ec])
            fs = df[df['sox'] > thr]
            fl = df[df['sox'] < -thr]
            for d, row in fs.iterrows():
                pnl_by_date.setdefault(d, []).append((row['open']/row[ec]-1)*10000 - cost)
            for d, row in fl.iterrows():
                pnl_by_date.setdefault(d, []).append((row[ec]/row['open']-1)*10000 - cost)
        daily_pnl = pd.Series({d: np.mean(v) for d, v in pnl_by_date.items()}).sort_index()
        daily_pnl.to_csv("semi_sox_fade_daily_pnl.csv", header=['pnl_bps'])
        print(f"\n→ semi_sox_fade_daily_pnl.csv 保存 ({len(daily_pnl)}日)")
        print(f"  日次 Sharpe (取引日のみ): {daily_pnl.mean()/daily_pnl.std()*np.sqrt(252):+.2f}")
        print(f"  最良戦略パラメータ: |SOX|>{thr}, exit={el}")


if __name__ == "__main__":
    main()
