#!/usr/bin/env python3
"""
D. LME銅オーバーナイト変化 → 非鉄寄付戦略
- 当日朝(JST 09:00)の直近 LME 値 vs 24h前 LME 値で ON 変化率を算出
- ON > +X bps: 非鉄を寄付買い → 引け / 10:00 / 11:00 などで決済
- ON < -X bps: 寄付売り → 同
"""
import warnings
import numpy as np
import pandas as pd
import psycopg2
from lib_data import load_all, NONFERROUS, perf, print_perf

warnings.filterwarnings("ignore")
PG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}
COST = 8.0


def load_lme():
    conn = psycopg2.connect(**PG)
    df = pd.read_sql(
        "SELECT timestamp, close FROM intraday_data WHERE symbol='CMCU3' ORDER BY timestamp", conn)
    conn.close()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['jst'] = df['timestamp'] + pd.Timedelta(hours=9)
    df = df.dropna(subset=['close']).set_index('jst').sort_index()
    return df


def compute_lme_on_change(lme_df, dates):
    """各日付について、当日 08:55 JST 直近 LME と 前日 08:55 直近 LME の差 (bps)"""
    out = {}
    for dt in dates:
        cutoff = pd.Timestamp(dt) + pd.Timedelta(hours=8, minutes=55)
        prev_cutoff = cutoff - pd.Timedelta(days=1)
        cur = lme_df[lme_df.index <= cutoff]
        prev = lme_df[lme_df.index <= prev_cutoff]
        if not len(cur) or not len(prev): continue
        # 24h以内に値があることを要求
        if (cutoff - cur.index[-1]).total_seconds() > 12 * 3600: continue
        if (prev_cutoff - prev.index[-1]).total_seconds() > 12 * 3600: continue
        chg = (cur['close'].iloc[-1] / prev['close'].iloc[-1] - 1) * 10000
        out[dt] = chg
    return pd.Series(out)


def main():
    print("=" * 120)
    print("D. LME銅 ON → 非鉄寄付戦略")
    print("=" * 120)

    print("LMEロード ...")
    lme = load_lme()
    print(f"  CMCU3: {len(lme)} bars  {lme.index.min()} ~ {lme.index.max()}")

    print("非鉄ロード ...")
    data = load_all(NONFERROUS)

    # 各銘柄の日次 OHLC + 中間価格
    print("\n日次グリッド構築中 ...")
    all_dates = set()
    for s in NONFERROUS:
        all_dates.update(data[s].index.date)
    all_dates = sorted(all_dates)
    lme_on = compute_lme_on_change(lme, all_dates)
    print(f"  LME ON 変化計算: N={len(lme_on)}日 (有効データ)")
    print(f"  ON 変化分布: mean={lme_on.mean():+.1f}bps  std={lme_on.std():.1f}bps  "
          f"P10={lme_on.quantile(0.1):+.1f}  P90={lme_on.quantile(0.9):+.1f}")

    # 各銘柄 day-level
    daily = {}
    for s in NONFERROUS:
        df = data[s]
        rec = {}
        for dt, g in df.groupby(df.index.date):
            if len(g) < 80: continue
            rec[dt] = {
                "open": g['open'].iloc[0],
                "p_10": g[(g.index.hour==10)&(g.index.minute==0)]['close'].iloc[0]
                        if len(g[(g.index.hour==10)&(g.index.minute==0)]) else np.nan,
                "p_11": g[(g.index.hour==11)&(g.index.minute==0)]['close'].iloc[0]
                        if len(g[(g.index.hour==11)&(g.index.minute==0)]) else np.nan,
                "p_1130": g[(g.index.hour==11)&(g.index.minute==30)]['close'].iloc[0]
                        if len(g[(g.index.hour==11)&(g.index.minute==30)]) else np.nan,
                "p_close": g['close'].iloc[-1],
            }
        daily[s] = pd.DataFrame(rec).T

    # ---- 1. 寄付ギャップとの相関 ----
    print("\n【LME ON vs 非鉄寄付ギャップ 相関】")
    for s in NONFERROUS:
        df = daily[s].copy()
        df['lme_on'] = pd.Series({pd.Timestamp(d).date() if not isinstance(d,(np.datetime64,pd.Timestamp)) else d: v
                                  for d, v in lme_on.items()})
        df['lme_on'] = df.index.map(lme_on.to_dict())
        df['prev_close'] = df['p_close'].shift(1)
        df['gap_bps'] = (df['open']/df['prev_close'] - 1) * 10000
        x = df.dropna(subset=['lme_on', 'gap_bps'])
        if len(x) < 30: continue
        c = np.corrcoef(x['lme_on'], x['gap_bps'])[0, 1]
        print(f"  {s}: corr(LME_ON, OpenGap) = {c:+.3f}  N={len(x)}")

    # ---- 2. 戦略バックテスト ----
    print("\n【戦略: LME_ON > +X → Long Open / 決済時刻別】")
    rows = []
    for thresh in [30, 50, 80, 100, 150]:
        for exit_label, exit_col in [("10:00", "p_10"), ("11:00", "p_11"),
                                      ("11:30", "p_1130"), ("close", "p_close")]:
            pnl_all = []
            for s in NONFERROUS:
                df = daily[s].copy()
                df['lme_on'] = df.index.map(lme_on.to_dict())
                df = df.dropna(subset=['lme_on', 'open', exit_col])
                # Long signal
                long_sig = df[df['lme_on'] > thresh]
                if len(long_sig):
                    pnl = (long_sig[exit_col]/long_sig['open'] - 1)*10000 - COST
                    pnl_all.extend(pnl.tolist())
                # Short signal
                short_sig = df[df['lme_on'] < -thresh]
                if len(short_sig):
                    pnl = (short_sig['open']/short_sig[exit_col] - 1)*10000 - COST
                    pnl_all.extend(pnl.tolist())
            rows.append(perf(np.array(pnl_all), label=f"|ON|>{thresh:>3} → {exit_label}"))
    print_perf(rows)

    print("\n【上位 5】")
    valid = [r for r in rows if not np.isnan(r["sharpe"]) and r["N"] >= 100]
    top = sorted(valid, key=lambda r: r["sharpe"], reverse=True)[:5]
    print_perf(top)

    # ---- 3. 銘柄別寄与 (最良条件で再走) ----
    if top:
        best = top[0]
        thr = int(best["label"].split(">")[1].split("→")[0].strip())
        exit_label = best["label"].split("→")[1].strip()
        col_map = {"10:00": "p_10", "11:00": "p_11", "11:30": "p_1130", "close": "p_close"}
        exit_col = col_map[exit_label]
        print(f"\n【銘柄別寄与: 最良戦略 |ON|>{thr} → {exit_label}】")
        for s in NONFERROUS:
            df = daily[s].copy()
            df['lme_on'] = df.index.map(lme_on.to_dict())
            df = df.dropna(subset=['lme_on', 'open', exit_col])
            pn = []
            long_sig = df[df['lme_on'] > thr]
            short_sig = df[df['lme_on'] < -thr]
            if len(long_sig):
                pn.extend(((long_sig[exit_col]/long_sig['open'] - 1)*10000 - COST).tolist())
            if len(short_sig):
                pn.extend(((short_sig['open']/short_sig[exit_col] - 1)*10000 - COST).tolist())
            x = np.array(pn)
            if len(x) < 5:
                print(f"  {s}: N={len(x)}")
                continue
            sh = x.mean()/x.std()*np.sqrt(252) if x.std() > 0 else 0
            wr = (x > 0).mean()*100
            print(f"  {s}: N={len(x):4d}  mean={x.mean():+6.1f}  WR={wr:5.1f}%  sum={x.sum():+7.0f}  Sharpe={sh:+.2f}")

    # ---- 4. B戦略との合成 ----
    # B戦略 (11:30/200bps/14:30 決済) と D の相関とアンサンブル可能性
    print("\n[note] B (11:30 disp 14:30 exit) と D (LME ON 寄付) は時刻が重ならず併用可能")


if __name__ == "__main__":
    main()
