#!/usr/bin/env python3
"""F2. SOX → 半導体 寄付フェード戦略 (順張り反転)"""
import warnings, numpy as np, pandas as pd, pymysql
from lib_data import load_all, SEMI, perf, print_perf
warnings.filterwarnings("ignore")
COST = 8.0

def main():
    conn = pymysql.connect(host="100.92.181.92", port=3306, user="rfnews",
                           password="Bleach@924", database="refinitiv_news")
    sox = pd.read_sql("SELECT trade_date, close FROM daily_data WHERE symbol='.SOX' ORDER BY trade_date", conn)
    conn.close()
    sox['trade_date'] = pd.to_datetime(sox['trade_date']).dt.date
    sox = sox.set_index('trade_date').sort_index()
    sox['chg'] = sox['close'].pct_change() * 10000

    data = load_all(SEMI)
    daily = {}
    for s in SEMI:
        rec = {}
        for dt, g in data[s].groupby(data[s].index.date):
            if len(g) < 80: continue
            def at(h, m):
                sel = g[(g.index.hour==h)&(g.index.minute==m)]
                return sel['close'].iloc[0] if len(sel) else np.nan
            rec[dt] = {"open": g['open'].iloc[0],
                       "p_11": at(11,0), "p_1130": at(11,30),
                       "p_1430": at(14,30), "p_close": g['close'].iloc[-1]}
        daily[s] = pd.DataFrame(rec).T

    all_dates = sorted(set().union(*[set(daily[s].index) for s in SEMI]))
    sox_for_jp = {}
    sd = list(sox.index)
    for jd in all_dates:
        cands = [d for d in sd if d < jd]
        if not cands: continue
        chg = sox.loc[cands[-1], 'chg']
        if not pd.isna(chg): sox_for_jp[jd] = chg
    sox_for_jp = pd.Series(sox_for_jp)

    print("=" * 130)
    print("F2. SOX → 半導体 寄付フェード戦略 (SOX 強い → 寄付ショート)")
    print("=" * 130)
    rows = []
    for thr in [50, 100, 150, 200, 300]:
        for el, ec in [("11:00","p_11"), ("11:30","p_1130"),
                       ("14:30","p_1430"), ("close","p_close")]:
            pn = []
            for s in SEMI:
                df = daily[s].copy()
                df['sox'] = df.index.map(sox_for_jp.to_dict())
                df = df.dropna(subset=['sox','open',ec])
                # フェード: SOX > thr → Short (open→ec で逆方向)
                fade_short = df[df['sox'] > thr]
                fade_long = df[df['sox'] < -thr]
                if len(fade_short):
                    pn.extend(((fade_short['open']/fade_short[ec]-1)*10000 - COST).tolist())
                if len(fade_long):
                    pn.extend(((fade_long[ec]/fade_long['open']-1)*10000 - COST).tolist())
            rows.append(perf(np.array(pn), label=f"FADE |SOX|>{thr:>3} → {el}"))
    print_perf(rows)
    print("\n上位 5 (N≥100):")
    valid = [r for r in rows if not np.isnan(r["sharpe"]) and r["N"] >= 100]
    print_perf(sorted(valid, key=lambda r: r["sharpe"], reverse=True)[:5])

    # 上位戦略の銘柄別寄与
    if valid:
        best = sorted(valid, key=lambda r: r["sharpe"], reverse=True)[0]
        thr = int(best["label"].split(">")[1].split("→")[0].strip())
        el = best["label"].split("→")[1].strip()
        col_map = {"11:00":"p_11", "11:30":"p_1130", "14:30":"p_1430", "close":"p_close"}
        ec = col_map[el]
        print(f"\n銘柄別寄与: 最良戦略 FADE |SOX|>{thr} → {el}")
        for s in SEMI:
            df = daily[s].copy()
            df['sox'] = df.index.map(sox_for_jp.to_dict())
            df = df.dropna(subset=['sox','open',ec])
            pn = []
            fs = df[df['sox'] > thr]
            fl = df[df['sox'] < -thr]
            if len(fs): pn.extend(((fs['open']/fs[ec]-1)*10000 - COST).tolist())
            if len(fl): pn.extend(((fl[ec]/fl['open']-1)*10000 - COST).tolist())
            x = np.array(pn)
            if len(x) < 5: print(f"  {s}: N={len(x)}"); continue
            sh = x.mean()/x.std()*np.sqrt(252) if x.std()>0 else 0
            wr = (x>0).mean()*100
            print(f"  {s}: N={len(x):4d}  mean={x.mean():+6.1f}  WR={wr:5.1f}%  Sharpe={sh:+.2f}")

if __name__ == "__main__":
    main()
