"""
検証2: トレンドフィルタ追加
TOPIX/N225の長期トレンドに沿った時のみエントリ → レンジ相場回避効果を測る

仮説:
  MA25/75 戦略の弱点は 2019-2023 のレンジ相場での連敗。
  TOPIX(or N225) の MA(50) > MA(200) の時のみ個別MA エントリを許可すれば、
  下降トレンド or レンジで建玉せず DD を削減できる可能性。

対象: MA25/75 戦略で機能している銘柄 (非鉄7 + 銀行3 + 半導体2 + 商社1 + 小売1 = 14銘柄)
比較: フィルタなし vs フィルタあり (Sharpe / Cum / DD / 取引数)
"""

import os
import psycopg2
import pandas as pd
import numpy as np

PG_CONFIG = {
    "host": os.environ.get("PGHOST", "omen"),
    "port": int(os.environ.get("PGPORT", 5432)),
    "user": os.environ.get("PGUSER", "postgres"),
    "dbname": os.environ.get("PGDATABASE", "market_data"),
}

# MA戦略でプラスSharpeだった銘柄 (前検証から)
MA_WINNERS = {
    "57060": "三井金属", "57110": "三菱マテリアル", "57130": "住友金属鉱山",
    "57140": "DOWA HD", "58010": "古河電工", "58020": "住友電工", "58030": "フジクラ",
    "83060": "三菱UFJ", "83160": "三井住友", "84110": "みずほ",
    "68570": "アドバンテスト", "69200": "レーザーテック",
    "80310": "三井物産", "80580": "三菱商事",
    "99830": "ファストリテ", "72030": "トヨタ自動車",
}

START = "2016-05-10"
END = "2026-05-22"
COST = 0.0008


def fetch_daily(code, table="stocks_daily"):
    conn = psycopg2.connect(**PG_CONFIG)
    sql = f"SELECT date, open, high, low, close FROM {table} WHERE code=%s AND date BETWEEN %s AND %s ORDER BY date"
    df = pd.read_sql(sql, conn, params=(code, START, END))
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")


def fetch_index(code="0000"):
    conn = psycopg2.connect(**PG_CONFIG)
    sql = "SELECT date, close FROM index_daily WHERE code=%s AND date BETWEEN %s AND %s ORDER BY date"
    df = pd.read_sql(sql, conn, params=(code, START, END))
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["close"].rename("idx")


def ma_cross_filtered(daily, code, index_trend: pd.Series,
                       fast=25, slow=75, cost=COST,
                       use_filter=True):
    """
    MA25/75 GC + (オプション) TOPIX MA50/200 GC フィルタ
    """
    ma_f = daily["close"].rolling(fast).mean()
    ma_s = daily["close"].rolling(slow).mean()
    sig = (ma_f > ma_s)

    # トレンドフィルタ: index_trend は True/False
    if use_filter:
        # 同日のフィルタ状態と AND
        idx_aligned = index_trend.reindex(daily.index, method='ffill')
        sig = sig & idx_aligned

    pos = sig.astype(int)
    op = daily["open"]
    diff = pos.diff().fillna(0).astype(int)
    entries = list(diff[diff == +1].index)
    exits = list(diff[diff == -1].index)

    trades = []
    pending = None
    for d in daily.index:
        if d in entries and pending is None:
            pending = d
        elif d in exits and pending is not None:
            ei = daily.index.get_loc(pending) + 1
            xi = daily.index.get_loc(d) + 1
            if ei >= len(daily) or xi >= len(daily):
                pending = None
                continue
            ed = daily.index[ei]
            xd = daily.index[xi]
            ep = op.iloc[ei]
            xp = op.iloc[xi]
            r = np.log(xp / ep)
            pnl = r - cost * 2
            trades.append({"code": code, "entry_date": ed, "exit_date": xd,
                           "entry": ep, "exit": xp, "ret_pct": r*100, "pnl": pnl,
                           "hold_days": (xd-ed).days})
            pending = None
    if pending is not None:
        ei = daily.index.get_loc(pending) + 1
        if ei < len(daily):
            ed = daily.index[ei]
            ep = op.iloc[ei]
            xd = daily.index[-1]
            xp = daily["close"].iloc[-1]
            r = np.log(xp / ep)
            pnl = r - cost * 2
            trades.append({"code": code, "entry_date": ed, "exit_date": xd,
                           "entry": ep, "exit": xp, "ret_pct": r*100, "pnl": pnl,
                           "hold_days": (xd-ed).days})
    return pd.DataFrame(trades)


def metrics(t, ann=245):
    if len(t) < 5:
        return {}
    t = t.copy()
    t["exit_date"] = pd.to_datetime(t["exit_date"])
    daily = t.groupby("exit_date")["pnl"].sum()
    mu, sd = daily.mean(), daily.std()
    sh = mu / sd * np.sqrt(ann) if sd > 0 else 0
    eq = daily.cumsum()
    dd = (eq - eq.cummax()).min()
    wr = (t["pnl"] > 0).mean() * 100
    pf = (t["pnl"][t["pnl"]>0].sum() / -t["pnl"][t["pnl"]<0].sum()) if (t["pnl"]<0).any() else np.inf
    return {"N": len(t), "Sharpe": round(sh,2), "WR%": round(wr,1),
            "PF": round(pf,2), "Cum%": round(eq.iloc[-1]*100,1),
            "DD%": round(dd*100,1)}


def main():
    print("=== トレンドフィルタ検証 (TOPIX MA50/200) ===\n")
    # TOPIX 取得
    topix = fetch_index("0000")
    n225 = fetch_index("N225")
    print(f"TOPIX: {len(topix)}日, N225: {len(n225)}日")

    # TOPIX MA50/200 トレンド
    topix_ma50 = topix.rolling(50).mean()
    topix_ma200 = topix.rolling(200).mean()
    topix_trend = (topix_ma50 > topix_ma200)
    print(f"TOPIX MA50>MA200 (上昇トレンド) 日数: {topix_trend.sum()} / {len(topix_trend)} "
          f"({topix_trend.sum()/len(topix_trend)*100:.1f}%)")

    # N225 MA50/200
    n225_ma50 = n225.rolling(50).mean()
    n225_ma200 = n225.rolling(200).mean()
    n225_trend = (n225_ma50 > n225_ma200)
    print(f"N225 MA50>MA200 上昇トレンド: {n225_trend.sum()/len(n225_trend)*100:.1f}%")

    # 銘柄別比較
    print(f"\n--- フィルタなし vs TOPIXトレンドフィルタ ---")
    print(f"{'銘柄':<14} {'なしSh':>7} {'なしCum':>8} {'なしDD':>8} | "
          f"{'有りSh':>7} {'有りCum':>8} {'有りDD':>8} {'取引減':>6}")
    rows = []
    all_trades_filt = {}
    all_trades_nofilt = {}
    for code, name in MA_WINNERS.items():
        d = fetch_daily(code)
        t_no = ma_cross_filtered(d, code, topix_trend, use_filter=False)
        t_yes = ma_cross_filtered(d, code, topix_trend, use_filter=True)
        all_trades_nofilt[code] = t_no
        all_trades_filt[code] = t_yes
        m_no = metrics(t_no)
        m_yes = metrics(t_yes)
        rows.append({
            "code": code, "name": name,
            "no_N": m_no.get("N", 0), "no_Sh": m_no.get("Sharpe", 0),
            "no_Cum": m_no.get("Cum%", 0), "no_DD": m_no.get("DD%", 0),
            "yes_N": m_yes.get("N", 0), "yes_Sh": m_yes.get("Sharpe", 0),
            "yes_Cum": m_yes.get("Cum%", 0), "yes_DD": m_yes.get("DD%", 0),
            "trade_reduction%": (1 - m_yes.get("N", 0) / max(m_no.get("N", 1), 1)) * 100,
        })
        n_red = (1 - m_yes.get("N", 0) / max(m_no.get("N", 1), 1)) * 100
        print(f"{name:<14} {m_no.get('Sharpe',0):>+7.2f} {m_no.get('Cum%',0):>+8.1f} {m_no.get('DD%',0):>+8.1f} | "
              f"{m_yes.get('Sharpe',0):>+7.2f} {m_yes.get('Cum%',0):>+8.1f} {m_yes.get('DD%',0):>+8.1f} {n_red:>5.0f}%")

    df = pd.DataFrame(rows)

    # 集計
    print(f"\n--- フィルタ効果集計 ---")
    print(f"  平均Sharpe: なし {df['no_Sh'].mean():+.2f} → 有り {df['yes_Sh'].mean():+.2f} "
          f"(差 {df['yes_Sh'].mean() - df['no_Sh'].mean():+.2f})")
    print(f"  平均Cum%  : なし {df['no_Cum'].mean():+.1f}% → 有り {df['yes_Cum'].mean():+.1f}%")
    print(f"  平均DD%   : なし {df['no_DD'].mean():+.1f}% → 有り {df['yes_DD'].mean():+.1f}%")
    print(f"  平均取引数: なし {df['no_N'].mean():.0f} → 有り {df['yes_N'].mean():.0f} "
          f"({df['trade_reduction%'].mean():.0f}% 削減)")
    n_sharpe_up = (df["yes_Sh"] > df["no_Sh"]).sum()
    n_dd_better = (df["yes_DD"] > df["no_DD"]).sum()  # DDは負の値、大きい方が浅い
    print(f"  Sharpe改善銘柄: {n_sharpe_up}/{len(df)}")
    print(f"  DD改善 (浅化) 銘柄: {n_dd_better}/{len(df)}")

    # 年別比較 (Top3で)
    print(f"\n--- Top3銘柄プール 年別比較 ---")
    top3 = ["57060", "57130", "58010"]  # 前回検証の本命

    def daily_pnl(trades_dict, K=3):
        dlist = []
        for c in top3:
            t = trades_dict[c].copy()
            t["exit_date"] = pd.to_datetime(t["exit_date"])
            sub = t.groupby("exit_date")["pnl"].sum() / K
            dlist.append(sub)
        return pd.concat(dlist).groupby(level=0).sum().sort_index()

    pn_no = daily_pnl(all_trades_nofilt)
    pn_yes = daily_pnl(all_trades_filt)

    print(f"{'年':<6} {'なしSh':>7} {'なしCum':>8} | {'有りSh':>7} {'有りCum':>8}")
    for y in range(2016, 2027):
        sn = pn_no[pn_no.index.year == y]
        sy = pn_yes[pn_yes.index.year == y]
        if len(sn) < 2 and len(sy) < 2:
            continue
        sh_n = (sn.mean()/sn.std()*np.sqrt(245)) if len(sn) > 1 and sn.std()>0 else 0
        sh_y = (sy.mean()/sy.std()*np.sqrt(245)) if len(sy) > 1 and sy.std()>0 else 0
        cum_n = sn.sum()*100
        cum_y = sy.sum()*100
        print(f"{y:<6} {sh_n:>+7.2f} {cum_n:>+8.1f} | {sh_y:>+7.2f} {cum_y:>+8.1f}")

    # 全期間 Top3
    print(f"\n--- Top3 全期間 比較 ---")
    def stats(s):
        mu, sd = s.mean(), s.std()
        sh = mu / sd * np.sqrt(245) if sd > 0 else 0
        eq = s.cumsum()
        dd = (eq - eq.cummax()).min()
        return sh, eq.iloc[-1]*100, dd*100
    sh_n, cum_n, dd_n = stats(pn_no)
    sh_y, cum_y, dd_y = stats(pn_yes)
    print(f"  なし: Sharpe={sh_n:+.2f}, Cum={cum_n:+.1f}%, DD={dd_n:.1f}%")
    print(f"  有り: Sharpe={sh_y:+.2f}, Cum={cum_y:+.1f}%, DD={dd_y:.1f}%")

    # 保存
    out_dir = os.path.dirname(os.path.abspath(__file__))
    df.to_csv(os.path.join(out_dir, "trend_filter_results.csv"), index=False)
    pn_no.to_csv(os.path.join(out_dir, "top3_daily_no_filter.csv"))
    pn_yes.to_csv(os.path.join(out_dir, "top3_daily_with_filter.csv"))
    print(f"\n保存: trend_filter_results.csv, top3_daily_*.csv")


if __name__ == "__main__":
    main()
