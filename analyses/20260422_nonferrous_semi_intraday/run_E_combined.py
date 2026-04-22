#!/usr/bin/env python3
"""
E. 統合検証
 1. D 戦略のコスト感応度 (片側 2/4/6/8/10/15 bps)
 2. B + D ポートフォリオ合成 (日次 PnL ベース)
 3. 5713 住友金鉱の LME 個別連動深掘り
"""
import warnings
import numpy as np
import pandas as pd
import psycopg2
from lib_data import load_all, NONFERROUS, perf, print_perf

warnings.filterwarnings("ignore")
PG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}


def load_lme():
    conn = psycopg2.connect(**PG)
    df = pd.read_sql(
        "SELECT timestamp, close FROM intraday_data WHERE symbol='CMCU3' ORDER BY timestamp", conn)
    conn.close()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['jst'] = df['timestamp'] + pd.Timedelta(hours=9)
    return df.dropna(subset=['close']).set_index('jst').sort_index()


def lme_on(lme_df, dates):
    out = {}
    for dt in dates:
        cutoff = pd.Timestamp(dt) + pd.Timedelta(hours=8, minutes=55)
        prev = cutoff - pd.Timedelta(days=1)
        c = lme_df[lme_df.index <= cutoff]
        p = lme_df[lme_df.index <= prev]
        if not len(c) or not len(p): continue
        if (cutoff - c.index[-1]).total_seconds() > 12*3600: continue
        if (prev - p.index[-1]).total_seconds() > 12*3600: continue
        out[dt] = (c['close'].iloc[-1]/p['close'].iloc[-1] - 1)*10000
    return pd.Series(out)


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
                "p_1130": at(11,30), "p_1230": at(12,30),
                "p_1430": at(14,30), "p_close": g['close'].iloc[-1],
                "p_11": at(11,0),
            }
        daily[s] = pd.DataFrame(rec).T
    return daily


def d_signal_pnl(daily, lme_on_ser, members, thresh, exit_col, cost_side):
    """LME ON > thresh の日に各銘柄でエントリ → exit_col 決済。
    銘柄毎日次 PnL の dict[date]→平均PnL を返す"""
    cost = cost_side * 2
    by_date = {}
    for s in members:
        df = daily[s].copy()
        df['lme_on'] = df.index.map(lme_on_ser.to_dict())
        df = df.dropna(subset=['lme_on', 'open', exit_col])
        long = df[df['lme_on'] > thresh]
        short = df[df['lme_on'] < -thresh]
        for d, row in long.iterrows():
            by_date.setdefault(d, []).append((row[exit_col]/row['open']-1)*10000 - cost)
        for d, row in short.iterrows():
            by_date.setdefault(d, []).append((row['open']/row[exit_col]-1)*10000 - cost)
    return pd.Series({d: np.mean(v) for d, v in by_date.items()}).sort_index()


def b_signal_pnl(daily, members, check=(11,30), thresh=200, exit_col="p_1430", cost_side=4):
    """B 戦略の日次平均 PnL"""
    cost = cost_side * 2
    opens = pd.concat([daily[s]['open'] for s in members], axis=1); opens.columns=members
    chk_col = "p_1130"
    checks = pd.concat([daily[s][chk_col] for s in members], axis=1); checks.columns=members
    exits = pd.concat([daily[s][exit_col] for s in members], axis=1); exits.columns=members
    rets = (checks/opens - 1)*10000
    mean_ret = rets.mean(axis=1, skipna=True)
    dev = rets.sub(mean_ret, axis=0)
    pnl_short = (checks/exits - 1)*10000
    pnl_long = (exits/checks - 1)*10000
    pnl = pnl_short.where(dev > 0, pnl_long)
    pnl = pnl.where(dev.abs() > thresh) - cost
    return pnl.mean(axis=1, skipna=True).dropna()


def main():
    print("=" * 130)
    print("E. 統合検証")
    print("=" * 130)

    print("ロード ...")
    lme = load_lme()
    data = load_all(NONFERROUS)
    all_dates = sorted(set().union(*[set(data[s].index.date) for s in NONFERROUS]))
    lme_chg = lme_on(lme, all_dates)
    daily = build_daily(data, NONFERROUS)
    print(f"  LME ON N={len(lme_chg)}, 日次グリッド構築 OK")

    # ============================================================
    # 1. D コスト感応度
    # ============================================================
    print("\n" + "=" * 130)
    print("【1. D 戦略 コスト感応度 (|LME_ON|>150 → 引け)】")
    print("=" * 130)
    rows = []
    for cs in [2, 4, 6, 8, 10, 15]:
        s = d_signal_pnl(daily, lme_chg, NONFERROUS, 150, "p_close", cs)
        rows.append(perf(s.values, label=f"片側 {cs:>2} bps"))
    print_perf(rows)

    print("\n【|LME_ON|>80 → 11:00 のコスト感応度】")
    rows2 = []
    for cs in [2, 4, 6, 8, 10, 15]:
        s = d_signal_pnl(daily, lme_chg, NONFERROUS, 80, "p_11", cs)
        rows2.append(perf(s.values, label=f"片側 {cs:>2} bps"))
    print_perf(rows2)

    # ============================================================
    # 2. B + D 合成
    # ============================================================
    print("\n" + "=" * 130)
    print("【2. B + D ポートフォリオ合成 (日次 PnL ベース)】")
    print("=" * 130)
    d_strong = d_signal_pnl(daily, lme_chg, NONFERROUS, 150, "p_close", 4)
    d_freq = d_signal_pnl(daily, lme_chg, NONFERROUS, 80, "p_11", 4)
    b = b_signal_pnl(daily, NONFERROUS, exit_col="p_1430")

    # 日次系列を共通インデックスで揃え (取引日のみ非ゼロ、それ以外 0)
    all_idx = sorted(set(d_strong.index) | set(d_freq.index) | set(b.index))
    d_strong = d_strong.reindex(all_idx).fillna(0)
    d_freq = d_freq.reindex(all_idx).fillna(0)
    b = b.reindex(all_idx).fillna(0)
    print(f"  共通日数: {len(all_idx)}")
    print(f"  D_strong 取引日: {(d_strong != 0).sum()}, D_freq: {(d_freq != 0).sum()}, B: {(b != 0).sum()}")

    print("\n相関行列 (取引日 union, 非取引日=0):")
    corr_df = pd.DataFrame({"D_strong": d_strong, "D_freq": d_freq, "B": b}).corr()
    print(corr_df.round(3).to_string())

    # 両戦略が取引した日のみ
    print("\n両戦略取引日のみの相関:")
    series_map = {"D_strong": d_strong, "D_freq": d_freq, "B": b}
    pairs = [("D_strong", "B"), ("D_freq", "B"), ("D_strong", "D_freq")]
    for a, c in pairs:
        sa, sb = series_map[a], series_map[c]
        m = (sa != 0) & (sb != 0)
        if m.sum() < 5: continue
        cc = np.corrcoef(sa[m], sb[m])[0, 1]
        print(f"  {a:<10} vs {c:<10}: ρ={cc:+.3f}  N={m.sum()}")

    print("\n各戦略単独 (日次 PnL ベース):")
    print_perf([
        perf(d_strong.values, "D_strong (全日 0埋)"),
        perf(d_freq.values, "D_freq (全日 0埋)"),
        perf(b.values, "B (全日 0埋)"),
    ])

    print("\n統合戦略 (Equal-Weight):")
    combos = [
        ("D_strong + B (½ ½)", 0.5*d_strong + 0.5*b),
        ("D_freq + B (½ ½)", 0.5*d_freq + 0.5*b),
        ("D_strong + D_freq + B (⅓⅓⅓)", (d_strong + d_freq + b)/3),
        ("D_strong (50%) + D_freq (25%) + B (25%)",
            0.5*d_strong + 0.25*d_freq + 0.25*b),
    ]
    print_perf([perf(s.values, label=name) for name, s in combos])

    # ============================================================
    # 3. 5713 住友金鉱 LME 個別連動深掘り
    # ============================================================
    print("\n" + "=" * 130)
    print("【3. 5713 住友金鉱 単独 LME 連動】")
    print("=" * 130)
    s = "5713.T"
    df = daily[s].copy()
    df['lme_on'] = df.index.map(lme_chg.to_dict())
    df = df.dropna(subset=['lme_on', 'open', 'p_close'])
    print(f"  サンプル: {len(df)} 日")

    rows3 = []
    for thr in [30, 50, 80, 100, 150]:
        for ec in ['p_11', 'p_1130', 'p_1430', 'p_close']:
            sig_l = df[df['lme_on'] > thr]
            sig_s = df[df['lme_on'] < -thr]
            pn = []
            if len(sig_l):
                v = (sig_l[ec]/sig_l['open'] - 1)*10000 - 8
                pn.extend(v.dropna().tolist())
            if len(sig_s):
                v = (sig_s['open']/sig_s[ec] - 1)*10000 - 8
                pn.extend(v.dropna().tolist())
            rows3.append(perf(np.array(pn),
                              label=f"5713 |ON|>{thr:>3} → {ec.replace('p_','')}"))
    print_perf(rows3)

    print("\n【5713 上位 5 (N≥30)】")
    valid3 = [r for r in rows3 if not np.isnan(r["sharpe"]) and r["N"] >= 30]
    top3 = sorted(valid3, key=lambda r: r["sharpe"], reverse=True)[:5]
    print_perf(top3)

    # ============================================================
    # 4. 最終推奨設計
    # ============================================================
    print("\n" + "=" * 130)
    print("【最終推奨運用設計】")
    print("=" * 130)
    final = 0.5*d_strong + 0.25*d_freq + 0.25*b
    p = perf(final.values, "最終ポートフォリオ")
    n_active = (final != 0).sum()
    print(f"  日次 PnL N={p['N']}日中, 取引日={n_active}")
    print(f"  日次平均 {p['mean']:+.2f} bps  → 年率 {p['mean']*252:+.0f} bps")
    print(f"  Sharpe  {p['sharpe']:+.2f}")
    print(f"  t-stat  {p['t']:+.2f}")
    print(f"  累計    {p['sum']:+.0f} bps")
    print(f"  MDD     {p['mdd']:.0f} bps")
    print(f"  H1/H2   {p['h1']:+.2f} / {p['h2']:+.2f}")

    # CSV 保存
    out = pd.DataFrame({
        "D_strong": d_strong, "D_freq": d_freq, "B": b, "Combined": final,
    })
    out.to_csv("nonferrous_combined_pnl.csv")
    print("\n→ nonferrous_combined_pnl.csv 保存")


if __name__ == "__main__":
    main()
