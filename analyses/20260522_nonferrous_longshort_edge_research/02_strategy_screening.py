"""
非鉄8銘柄ロングショート戦略 — 4候補スクリーニング

仮説:
  H1: グループ内クロスセクション平均回帰 (寄→引)
      前日寄→引で「グループ平均からの乖離」が大きい銘柄を翌日寄からフェード
  H2: 寄付ギャップフェード
      前日終値→寄付ギャップが大きい銘柄を寄付からフェード、引けでクローズ
  H3: 前場急進フェード
      前場(open→11:30)で「グループ平均からの乖離」が大きい銘柄を後場(12:30→15:30)でフェード
  H4: 高相関ペアスプレッド平均回帰 (57110-57130)
      Zスコア > +1.5 でロング/ショート、|Z| < 0.3 でクローズ

評価:
  N, 取引数, 平均リターン/取引, 標準偏差, Sharpe (年率), 勝率, PF, MaxDD
  コスト: 0.10% 片道 (=往復0.20%) で評価
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

SYMBOLS = {
    "57060": ("三井金属", "miner"),
    "57110": ("三菱マテリアル", "miner"),
    "57130": ("住友金属鉱山", "miner"),
    "57140": ("DOWA HD", "miner"),
    "50160": ("JX金属", "miner"),
    "58010": ("古河電工", "wire"),
    "58020": ("住友電工", "wire"),
    "58030": ("フジクラ", "wire"),
}

START = "2025-04-01"
END = "2026-05-21"
COST_ONEWAY = 0.001  # 0.10% 片道


def fetch_minute(code: str) -> pd.DataFrame:
    conn = psycopg2.connect(**PG_CONFIG)
    sql = """
        SELECT ts, open, high, low, close, volume, turnover_value
        FROM stocks_intraday
        WHERE code=%s AND ts>=%s AND ts<=%s
        ORDER BY ts
    """
    df = pd.read_sql(sql, conn, params=(code, START, END + " 23:59:59"))
    conn.close()
    df["ts"] = pd.to_datetime(df["ts"])
    return df.set_index("ts")


def daily_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = df.index.normalize()
    rows = []
    for d, g in df.groupby("date"):
        am = g.between_time("09:00", "11:30")
        pm = g.between_time("12:30", "15:30")
        if len(am) < 10 or len(pm) < 10:
            continue
        rows.append({
            "date": d.normalize(),
            "open": am.iloc[0]["open"],
            "am_close": am.iloc[-1]["close"],
            "pm_open": pm.iloc[0]["open"],
            "close": pm.iloc[-1]["close"],
        })
    out = pd.DataFrame(rows).set_index("date")
    return out


def fetch_prev_close(code: str) -> pd.DataFrame:
    """前日終値（日足から）を取得"""
    conn = psycopg2.connect(**PG_CONFIG)
    sql = """
        SELECT date, close FROM stocks_daily
        WHERE code=%s AND date>=%s AND date<=%s ORDER BY date
    """
    df = pd.read_sql(sql, conn, params=(code, START, END))
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df["prev_close"] = df["close"].shift(1)
    return df[["prev_close"]]


def summarize(name: str, pnl_per_trade: pd.Series, trades_per_day: float = 1.0,
              ann_factor: float = 245.0) -> dict:
    """PnL (per trade, fractional return on notional) を集計"""
    pnl_per_trade = pnl_per_trade.dropna()
    n = len(pnl_per_trade)
    if n < 5:
        return {"name": name, "n": n, "note": "サンプル不足"}
    mu = pnl_per_trade.mean()
    sig = pnl_per_trade.std()
    sharpe = mu / sig * np.sqrt(ann_factor * trades_per_day) if sig > 0 else 0
    winrate = (pnl_per_trade > 0).mean()
    gross_win = pnl_per_trade[pnl_per_trade > 0].sum()
    gross_loss = -pnl_per_trade[pnl_per_trade < 0].sum()
    pf = gross_win / gross_loss if gross_loss > 0 else np.inf
    # equity curve & DD (cumulative returns assuming 1 unit per trade)
    eq = pnl_per_trade.cumsum()
    dd = (eq - eq.cummax()).min()
    return {
        "name": name,
        "n_trades": n,
        "avg_ret_per_trade_%": mu * 100,
        "std_per_trade_%": sig * 100,
        "Sharpe_ann": sharpe,
        "winrate_%": winrate * 100,
        "PF": pf,
        "cum_PnL_%": eq.iloc[-1] * 100,
        "maxDD_%": dd * 100,
    }


# ===================== Strategy Implementations =====================

def strategy_H1_xs_meanrev_dayreturn(day_open: pd.DataFrame,
                                     day_close: pd.DataFrame,
                                     groups: dict) -> dict:
    """H1: 寄→引で「グループ平均との乖離」が大きい銘柄を、翌日寄→引でフェード。
    エントリ: t-1 の (寄→引リターン - グループ平均) → 上位/下位を逆方向に持つ
    ホールド: t の寄→引 (open→close)
    """
    # 寄→引リターン
    r_day = np.log(day_close / day_open)  # (date, code)
    # グループ平均
    miner_codes = [c for c, g in groups.items() if g == "miner" and c in r_day.columns]
    wire_codes = [c for c, g in groups.items() if g == "wire" and c in r_day.columns]

    # 各銘柄の「自グループ平均との差」を yesterday に算出
    deviation = pd.DataFrame(index=r_day.index, columns=r_day.columns, dtype=float)
    for c in r_day.columns:
        g = groups.get(c, ("", ""))[1] if isinstance(groups.get(c), tuple) else groups[c]
        peers = miner_codes if g == "miner" else wire_codes
        peers_excl = [p for p in peers if p != c]
        deviation[c] = r_day[c] - r_day[peers_excl].mean(axis=1)

    # signal: 翌日に「乖離を逆張り」 → 翌日リターン × (-sign(乖離_前日))
    signal = -np.sign(deviation.shift(1))  # 前日の乖離をフェード
    # ポジション: 各日、|乖離| ランクで top1 ロング/ short1 のドルニュートラル
    # 簡単化: 各日 wire群とminer群の各グループ内で乖離 max を short, min を long にする
    trades = []
    for d in r_day.index[1:]:
        for grp_codes in [miner_codes, wire_codes]:
            dev_y = deviation.shift(1).loc[d, grp_codes].dropna()
            if len(dev_y) < 3:
                continue
            short_code = dev_y.idxmax()  # 一番上ぶれた → ショート
            long_code = dev_y.idxmin()   # 一番下ぶれた → ロング
            r_long = r_day.loc[d, long_code]
            r_short = r_day.loc[d, short_code]
            pnl = (r_long - r_short) / 2 - COST_ONEWAY * 2  # 2銘柄 × 往復
            trades.append({"date": d, "pnl": pnl, "grp": "miner" if grp_codes == miner_codes else "wire"})
    df_trades = pd.DataFrame(trades)
    return {"trades": df_trades, "summary": summarize("H1 XS-MR (day)", df_trades.set_index("date")["pnl"])}


def strategy_H2_gap_fade(day_open: pd.DataFrame, day_close: pd.DataFrame,
                         prev_close_dict: dict, groups: dict) -> dict:
    """H2: 前日終値→当日寄付ギャップが「グループ平均ギャップとの乖離」最大の銘柄をフェード。
    エントリ: 寄付  エグジット: 引け
    各日 グループ内 top1 ショート / bottom1 ロング
    """
    prev_close = pd.DataFrame({c: prev_close_dict[c]["prev_close"] for c in prev_close_dict})
    # 日付揃え
    common = day_open.index.intersection(prev_close.index)
    op = day_open.loc[common]
    cl = day_close.loc[common]
    pc = prev_close.loc[common]
    gap = np.log(op / pc)  # 前日終値→当日寄付ギャップ
    r_intra = np.log(cl / op)  # 当日 寄→引

    miner_codes = [c for c, g in groups.items() if g == "miner" and c in gap.columns]
    wire_codes = [c for c, g in groups.items() if g == "wire" and c in gap.columns]

    trades = []
    for d in gap.index:
        for grp_codes in [miner_codes, wire_codes]:
            g_today = gap.loc[d, grp_codes].dropna()
            if len(g_today) < 3:
                continue
            grp_mean = g_today.mean()
            dev = g_today - grp_mean
            short_code = dev.idxmax()
            long_code = dev.idxmin()
            r_long = r_intra.loc[d, long_code]
            r_short = r_intra.loc[d, short_code]
            pnl = (r_long - r_short) / 2 - COST_ONEWAY * 2
            trades.append({"date": d, "pnl": pnl,
                           "grp": "miner" if grp_codes == miner_codes else "wire"})
    df_trades = pd.DataFrame(trades)
    return {"trades": df_trades, "summary": summarize("H2 Gap Fade", df_trades.set_index("date")["pnl"])}


def strategy_H3_am_fade(am_open: pd.DataFrame, am_close: pd.DataFrame,
                        pm_open: pd.DataFrame, pm_close: pd.DataFrame,
                        groups: dict) -> dict:
    """H3: 前場で「グループ平均との乖離」最大の銘柄を後場でフェード。
    エントリ: 12:30 寄付  エグジット: 15:30 引け
    """
    r_am = np.log(am_close / am_open)
    r_pm = np.log(pm_close / pm_open)

    miner_codes = [c for c, g in groups.items() if g == "miner" and c in r_am.columns]
    wire_codes = [c for c, g in groups.items() if g == "wire" and c in r_am.columns]

    trades = []
    for d in r_am.index:
        for grp_codes in [miner_codes, wire_codes]:
            ram_today = r_am.loc[d, grp_codes].dropna()
            if len(ram_today) < 3:
                continue
            dev = ram_today - ram_today.mean()
            short_code = dev.idxmax()  # 前場で上ぶれた → 後場ショート
            long_code = dev.idxmin()   # 前場で下ぶれた → 後場ロング
            r_long = r_pm.loc[d, long_code]
            r_short = r_pm.loc[d, short_code]
            pnl = (r_long - r_short) / 2 - COST_ONEWAY * 2
            trades.append({"date": d, "pnl": pnl,
                           "grp": "miner" if grp_codes == miner_codes else "wire"})
    df_trades = pd.DataFrame(trades)
    return {"trades": df_trades, "summary": summarize("H3 AM-Fade", df_trades.set_index("date")["pnl"])}


def strategy_H4_pair_zscore(day_open: pd.DataFrame, day_close: pd.DataFrame,
                            pair=("57110", "57130"),
                            window=30, entry_z=1.5, exit_z=0.3) -> dict:
    """H4: 高相関ペア (57110 三菱マテ / 57130 住友鉱山) のスプレッド Zスコア平均回帰
    スプレッド = log(P_A) - β * log(P_B)  (β は rolling window で OLS)
    寄付エントリ・引けエグジット (1日完結に簡略化)
    Z > +entry_z: A ショート B ロング
    Z < -entry_z: A ロング B ショート
    |Z| < exit_z: クローズ
    """
    a, b = pair
    # 引け価格でZスコア計算
    pa = day_close[a]
    pb = day_close[b]
    common = pa.dropna().index.intersection(pb.dropna().index)
    pa, pb = pa.loc[common], pb.loc[common]
    lpa = np.log(pa)
    lpb = np.log(pb)
    # ローリング β (cov/var)
    beta = lpa.rolling(window).cov(lpb) / lpb.rolling(window).var()
    spread = lpa - beta * lpb
    mu = spread.rolling(window).mean()
    sd = spread.rolling(window).std()
    z = (spread - mu) / sd

    # 当日引け Z で次日寄付エントリ・次日引けエグジット (1日保有)
    z_signal = z.shift(1)  # 前日引け時点の Z
    # 寄付価格 / 引け価格 (open→close)
    op_a, cl_a = day_open[a], day_close[a]
    op_b, cl_b = day_open[b], day_close[b]
    r_a = np.log(cl_a / op_a)
    r_b = np.log(cl_b / op_b)

    trades = []
    for d in z_signal.index:
        zv = z_signal.loc[d]
        if pd.isna(zv):
            continue
        if zv > entry_z:
            # A overpriced → short A, long B (β調整は1日なので簡易にA1:B1)
            pnl = (r_b.loc[d] - r_a.loc[d]) / 2 - COST_ONEWAY * 2
            trades.append({"date": d, "pnl": pnl, "z": zv, "side": "shortA"})
        elif zv < -entry_z:
            pnl = (r_a.loc[d] - r_b.loc[d]) / 2 - COST_ONEWAY * 2
            trades.append({"date": d, "pnl": pnl, "z": zv, "side": "longA"})
    df_trades = pd.DataFrame(trades)
    return {"trades": df_trades,
            "summary": summarize(f"H4 PairZ ({a}-{b}) entry={entry_z}",
                                 df_trades.set_index("date")["pnl"] if len(df_trades) else pd.Series(dtype=float),
                                 trades_per_day=0.3)}


# ===================== Main =====================

def main():
    print("--- データ読み込み ---")
    feats = {}
    prev_closes = {}
    for code in SYMBOLS:
        m = fetch_minute(code)
        feats[code] = daily_features(m)
        prev_closes[code] = fetch_prev_close(code)
        print(f"  {code}: {len(feats[code])}日")

    # ピボット
    day_open = pd.DataFrame({c: f["open"] for c, f in feats.items()})
    day_close = pd.DataFrame({c: f["close"] for c, f in feats.items()})
    am_open = day_open.copy()  # 寄付 = 前場寄付
    am_close = pd.DataFrame({c: f["am_close"] for c, f in feats.items()})
    pm_open = pd.DataFrame({c: f["pm_open"] for c, f in feats.items()})
    pm_close = day_close.copy()  # 引け = 後場引け

    groups = {c: g for c, (_, g) in SYMBOLS.items()}

    print("\n--- H1: グループ内クロスセクション平均回帰 (日次) ---")
    r1 = strategy_H1_xs_meanrev_dayreturn(day_open, day_close, groups)
    print(r1["summary"])

    print("\n--- H2: 寄付ギャップフェード ---")
    r2 = strategy_H2_gap_fade(day_open, day_close, prev_closes, groups)
    print(r2["summary"])

    print("\n--- H3: 前場急進フェード (後場で取る) ---")
    r3 = strategy_H3_am_fade(am_open, am_close, pm_open, pm_close, groups)
    print(r3["summary"])

    print("\n--- H4: 高相関ペア Zスコア (57110-57130, window=30, entry_z=1.5) ---")
    r4 = strategy_H4_pair_zscore(day_open, day_close, pair=("57110", "57130"))
    print(r4["summary"])

    # 他ペアも試す
    print("\n--- H4b: ペア Zスコア 他の高相関ペア ---")
    for pair in [("57110", "57140"), ("57130", "57140"), ("58010", "58020"), ("58020", "58030"), ("58010", "58030")]:
        rr = strategy_H4_pair_zscore(day_open, day_close, pair=pair)
        print(f"  pair={pair}: {rr['summary']}")

    # 結果保存
    out_dir = os.path.dirname(os.path.abspath(__file__))
    summary_rows = [r1["summary"], r2["summary"], r3["summary"], r4["summary"]]
    pd.DataFrame(summary_rows).to_csv(os.path.join(out_dir, "screening_summary.csv"), index=False)
    print(f"\n保存: screening_summary.csv")

    # 各戦略の trades 保存
    r1["trades"].to_csv(os.path.join(out_dir, "trades_H1.csv"), index=False)
    r2["trades"].to_csv(os.path.join(out_dir, "trades_H2.csv"), index=False)
    r3["trades"].to_csv(os.path.join(out_dir, "trades_H3.csv"), index=False)
    r4["trades"].to_csv(os.path.join(out_dir, "trades_H4.csv"), index=False)


if __name__ == "__main__":
    main()
