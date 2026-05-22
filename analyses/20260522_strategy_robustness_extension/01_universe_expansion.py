"""
検証1: ユニバース拡張
MA25/75 順張り + RSI<30 反発ロングが非鉄以外のセクターでも機能するか

対象:
  非鉄 (基準):  5706, 5711, 5713, 5714, 5801, 5802, 5803  (前回検証済)
  銀行:        8306, 8316, 8411
  半導体:      6857, 6920, 8035
  商社:        8001, 8031, 8058
  小売:        3382, 9983
  自動車:      7203, 7267
  その他製品:   7974 (任天堂)

期間: 2016-05-10 〜 2026-05-22 (10年)
コスト: 8 bps片道

評価:
  - 各銘柄でMA25/75とRSI<30 をバックテスト
  - セクター別Sharpe集計
  - Buy&Hold ベンチマーク比較
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

UNIVERSE = {
    # 非鉄 (基準)
    "57060": ("三井金属", "非鉄金属"),
    "57110": ("三菱マテリアル", "非鉄金属"),
    "57130": ("住友金属鉱山", "非鉄金属"),
    "57140": ("DOWA HD", "非鉄金属"),
    "58010": ("古河電工", "非鉄金属"),
    "58020": ("住友電工", "非鉄金属"),
    "58030": ("フジクラ", "非鉄金属"),
    # 銀行
    "83060": ("三菱UFJ", "銀行業"),
    "83160": ("三井住友", "銀行業"),
    "84110": ("みずほ", "銀行業"),
    # 半導体
    "68570": ("アドバンテスト", "半導体"),
    "69200": ("レーザーテック", "半導体"),
    "80350": ("東京エレクトロン", "半導体"),
    # 商社
    "80010": ("伊藤忠商事", "商社"),
    "80310": ("三井物産", "商社"),
    "80580": ("三菱商事", "商社"),
    # 小売
    "33820": ("セブン&アイ", "小売業"),
    "99830": ("ファストリテ", "小売業"),
    # 自動車
    "72030": ("トヨタ自動車", "自動車"),
    "72670": ("ホンダ", "自動車"),
    # その他
    "79740": ("任天堂", "その他製品"),
}

START = "2016-05-10"
END = "2026-05-22"
COST_ONEWAY = 0.0008


def fetch_daily(code):
    conn = psycopg2.connect(**PG_CONFIG)
    sql = "SELECT date, open, high, low, close FROM stocks_daily WHERE code=%s AND date BETWEEN %s AND %s ORDER BY date"
    df = pd.read_sql(sql, conn, params=(code, START, END))
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")


def rsi(close, n=14):
    delta = close.diff()
    up = delta.clip(lower=0)
    dn = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/n, adjust=False).mean()
    ma_dn = dn.ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100 / (1 + ma_up/ma_dn)


def simulate_signal(daily, code, position_series, direction=+1, cost=COST_ONEWAY):
    """position_series (0/1) → entry/exit trades"""
    op = daily["open"]
    pos = position_series.fillna(0).astype(int)
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
            r = np.log(xp / ep) * direction
            pnl = r - cost * 2
            trades.append({
                "code": code, "entry_date": ed, "exit_date": xd,
                "entry": ep, "exit": xp,
                "hold_days": (xd - ed).days,
                "ret_pct": r * 100, "pnl": pnl,
            })
            pending = None
    if pending is not None:
        ei = daily.index.get_loc(pending) + 1
        if ei < len(daily):
            ed = daily.index[ei]
            ep = op.iloc[ei]
            xd = daily.index[-1]
            xp = daily["close"].iloc[-1]
            r = np.log(xp / ep) * direction
            pnl = r - cost * 2
            trades.append({
                "code": code, "entry_date": ed, "exit_date": xd,
                "entry": ep, "exit": xp,
                "hold_days": (xd - ed).days,
                "ret_pct": r * 100, "pnl": pnl,
            })
    return pd.DataFrame(trades)


def strat_ma_cross(daily, code, fast=25, slow=75, cost=COST_ONEWAY):
    ma_f = daily["close"].rolling(fast).mean()
    ma_s = daily["close"].rolling(slow).mean()
    pos = (ma_f > ma_s).astype(int)
    return simulate_signal(daily, code, pos, direction=+1, cost=cost)


def strat_rsi_long(daily, code, rsi_lo=30, rsi_hi=50, cost=COST_ONEWAY):
    r = rsi(daily["close"])
    pos = pd.Series(0, index=daily.index)
    state = 0
    for d in daily.index:
        if pd.isna(r.loc[d]):
            continue
        rv = r.loc[d]
        if state == 0 and rv < rsi_lo:
            state = 1
        elif state == 1 and rv > rsi_hi:
            state = 0
        pos.loc[d] = state
    return simulate_signal(daily, code, pos, direction=+1, cost=cost)


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
    return {"N": len(t), "Sharpe": round(sh, 2), "WR%": round(wr, 1),
            "PF": round(pf, 2), "Cum%": round(eq.iloc[-1]*100, 1),
            "DD%": round(dd*100, 1)}


def buyhold(daily, ann=245):
    r = np.log(daily["close"] / daily["close"].shift(1)).dropna()
    mu, sd = r.mean(), r.std()
    sh = mu / sd * np.sqrt(ann) if sd > 0 else 0
    eq = r.cumsum()
    dd = (eq - eq.cummax()).min()
    return {"Sharpe_BH": round(sh, 2), "Cum_BH%": round(eq.iloc[-1]*100, 1),
            "DD_BH%": round(dd*100, 1)}


def main():
    print("=== ユニバース拡張: MA25/75 + RSI<30 を 7セクター21銘柄で検証 ===\n")
    rows = []
    all_trades = {"MA": {}, "RSI": {}}
    for code, (name, sector) in UNIVERSE.items():
        d = fetch_daily(code)
        if len(d) < 200:
            continue
        m_bh = buyhold(d)
        t_ma = strat_ma_cross(d, code)
        t_rsi = strat_rsi_long(d, code)
        all_trades["MA"][code] = t_ma
        all_trades["RSI"][code] = t_rsi
        m_ma = metrics(t_ma)
        m_rsi = metrics(t_rsi)
        rows.append({
            "sector": sector, "code": code, "name": name,
            "BH_Sharpe": m_bh["Sharpe_BH"], "BH_Cum%": m_bh["Cum_BH%"],
            "MA_Sharpe": m_ma.get("Sharpe", 0), "MA_Cum%": m_ma.get("Cum%", 0),
            "MA_DD%": m_ma.get("DD%", 0), "MA_N": m_ma.get("N", 0),
            "RSI_Sharpe": m_rsi.get("Sharpe", 0), "RSI_Cum%": m_rsi.get("Cum%", 0),
            "RSI_DD%": m_rsi.get("DD%", 0), "RSI_N": m_rsi.get("N", 0),
        })

    df = pd.DataFrame(rows)

    # 全銘柄表
    print(f"{'銘柄':<14} {'sector':<10} {'BH_Sh':>6} {'MA_Sh':>6} {'RSI_Sh':>7} "
          f"{'MA-BH':>6} {'RSI-BH':>7}")
    for _, r in df.iterrows():
        print(f"{r['name']:<14} {r['sector']:<10} "
              f"{r['BH_Sharpe']:>+6.2f} {r['MA_Sharpe']:>+6.2f} {r['RSI_Sharpe']:>+7.2f} "
              f"{r['MA_Sharpe']-r['BH_Sharpe']:>+6.2f} {r['RSI_Sharpe']-r['BH_Sharpe']:>+7.2f}")

    # セクター別集計
    print(f"\n--- セクター別 平均Sharpe ---")
    sect_summary = df.groupby("sector").agg(
        n=("name", "count"),
        BH=("BH_Sharpe", "mean"),
        MA=("MA_Sharpe", "mean"),
        RSI=("RSI_Sharpe", "mean"),
        MA_excess=("MA_Sharpe", lambda s: s.mean() - df.loc[s.index, "BH_Sharpe"].mean()),
        RSI_excess=("RSI_Sharpe", lambda s: s.mean() - df.loc[s.index, "BH_Sharpe"].mean()),
    ).round(2)
    print(sect_summary.to_string())

    # MA 勝率・敗率
    print(f"\n--- MA25/75 セクター別 勝銘柄数 ---")
    for sect, g in df.groupby("sector"):
        w = (g["MA_Sharpe"] > 0).sum()
        wbh = (g["MA_Sharpe"] > g["BH_Sharpe"]).sum()
        print(f"  {sect:<10}: MA勝ち {w}/{len(g)}, BH超え {wbh}/{len(g)}")

    print(f"\n--- RSI<30 セクター別 勝銘柄数 ---")
    for sect, g in df.groupby("sector"):
        w = (g["RSI_Sharpe"] > 0).sum()
        print(f"  {sect:<10}: RSI勝ち {w}/{len(g)}")

    # 上位ランキング (MA)
    print(f"\n--- MA25/75 Sharpe ランキング (Top10) ---")
    top10 = df.sort_values("MA_Sharpe", ascending=False).head(10)
    print(top10[["sector", "name", "MA_Sharpe", "MA_Cum%", "MA_DD%", "BH_Sharpe"]].to_string(index=False))

    # 下位ランキング
    print(f"\n--- MA25/75 Sharpe ランキング (Bottom5) ---")
    bot5 = df.sort_values("MA_Sharpe", ascending=True).head(5)
    print(bot5[["sector", "name", "MA_Sharpe", "MA_Cum%", "MA_DD%", "BH_Sharpe"]].to_string(index=False))

    # 保存
    out_dir = os.path.dirname(os.path.abspath(__file__))
    df.to_csv(os.path.join(out_dir, "universe_results.csv"), index=False)
    sect_summary.to_csv(os.path.join(out_dir, "sector_summary.csv"))

    # 銘柄別trades保存
    for strat, td in all_trades.items():
        for c, t in td.items():
            if len(t) > 0:
                t.to_csv(os.path.join(out_dir, f"trades_{strat}_{c}.csv"), index=False)
    print(f"\n保存: universe_results.csv, sector_summary.csv, trades_*.csv")


if __name__ == "__main__":
    main()
