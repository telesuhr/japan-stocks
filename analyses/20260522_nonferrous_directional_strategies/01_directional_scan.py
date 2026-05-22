"""
非鉄8銘柄 シングルサイド戦略 スクリーニング

3戦略 × 8銘柄個別 + プール集計 を一気に評価。

戦略:
  A. ORB (Opening Range Breakout)
     寄付15分 (9:00-9:15) の高安を ORレンジ
     9:15以降に OR上抜け → ロング, OR下抜け → ショート
     15:25 強制クローズ
     1日1回エントリ (最初に発生したシグナルのみ)

  B. 寄付ギャップフェード
     前日終値→当日寄付ギャップ |G| > GAP_THR
     ギャップ上 → 寄付でショート, ギャップ下 → 寄付でロング
     15:25 引けクローズ

  C. VWAP当日平均回帰
     当日累積VWAP からのZスコア (rolling std)
     |Z| > 1.5 で逆張りエントリ
     |Z| < 0.3 でクローズ, 15:25 強制クローズ
     1日複数回エントリ可

コスト感度: 5 / 8 / 10 / 12 bps 片道
評価: 銘柄別 + プール (= 全銘柄シグナルを単純集計)
"""

import os
import psycopg2
import pandas as pd
import numpy as np
from datetime import time as dtime

PG_CONFIG = {
    "host": os.environ.get("PGHOST", "omen"),
    "port": int(os.environ.get("PGPORT", 5432)),
    "user": os.environ.get("PGUSER", "postgres"),
    "dbname": os.environ.get("PGDATABASE", "market_data"),
}

SYMBOLS = {
    "57060": "三井金属", "57110": "三菱マテリアル", "57130": "住友金属鉱山",
    "57140": "DOWA HD", "50160": "JX金属",
    "58010": "古河電工", "58020": "住友電工", "58030": "フジクラ",
}
START = "2025-04-01"
END = "2026-05-21"


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


def fetch_daily_close(code: str) -> pd.Series:
    conn = psycopg2.connect(**PG_CONFIG)
    sql = "SELECT date, close FROM stocks_daily WHERE code=%s AND date BETWEEN %s AND %s ORDER BY date"
    df = pd.read_sql(sql, conn, params=(code, START, END))
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["close"].rename(code)


# ===================== Strategy A: ORB =====================

def strat_ORB(minute_df: pd.DataFrame, code: str, or_min=15, cost_oneway=0.0008) -> pd.DataFrame:
    """寄付 or_min 分の高安をブレイクしたらフォロー、引けクローズ。
    1日1回 (先に発生したシグナルのみ)
    """
    df = minute_df.copy()
    df["date"] = df.index.normalize()
    trades = []
    for d, g in df.groupby("date"):
        or_zone = g.between_time("09:00", f"09:{or_min:02d}")
        if len(or_zone) < or_min - 2:
            continue
        or_high = or_zone["high"].max()
        or_low = or_zone["low"].min()
        # 9:16 以降でブレイク判定
        after = g.between_time(f"09:{or_min+1:02d}", "15:25")
        if len(after) == 0:
            continue
        entered = None  # +1 long, -1 short
        entry_price = None
        entry_ts = None
        for ts, row in after.iterrows():
            hi, lo, cl = row["high"], row["low"], row["close"]
            if entered is None:
                if hi > or_high:
                    entered = +1
                    entry_price = or_high  # ブレイク水準で約定仮定
                    entry_ts = ts
                    break
                elif lo < or_low:
                    entered = -1
                    entry_price = or_low
                    entry_ts = ts
                    break
        if entered is None:
            continue
        # 引けクローズ
        last = g.between_time("15:25", "15:30")
        if len(last) == 0:
            continue
        exit_price = last.iloc[-1]["close"]
        r = np.log(exit_price / entry_price)
        pnl = r * entered - cost_oneway * 2  # 往復2回
        trades.append({"date": d, "code": code, "side": entered,
                       "entry": entry_price, "exit": exit_price,
                       "or_high": or_high, "or_low": or_low,
                       "pnl": pnl, "entry_ts": entry_ts})
    return pd.DataFrame(trades)


# ===================== Strategy B: Gap Fade =====================

def strat_gap_fade(minute_df: pd.DataFrame, prev_close: pd.Series, code: str,
                   gap_thr=0.015, cost_oneway=0.0008) -> pd.DataFrame:
    """寄付ギャップ |G| > gap_thr で寄付逆張り、引けクローズ"""
    df = minute_df.copy()
    df["date"] = df.index.normalize()
    trades = []
    for d, g in df.groupby("date"):
        if d not in prev_close.index or pd.isna(prev_close.loc[d]):
            continue
        pc = prev_close.loc[d]
        am = g.between_time("09:00", "09:01")
        if len(am) == 0:
            continue
        open_px = am.iloc[0]["open"]
        gap = np.log(open_px / pc)
        if abs(gap) < gap_thr:
            continue
        side = -np.sign(gap)  # 上ギャップならショート(-1)、下ギャップならロング(+1)
        # 引けクローズ
        last = g.between_time("15:25", "15:30")
        if len(last) == 0:
            continue
        close_px = last.iloc[-1]["close"]
        r = np.log(close_px / open_px)
        pnl = r * side - cost_oneway * 2
        trades.append({"date": d, "code": code, "side": side, "gap": gap,
                       "entry": open_px, "exit": close_px, "pnl": pnl})
    return pd.DataFrame(trades)


# ===================== Strategy C: VWAP Reversion =====================

def strat_vwap_revert(minute_df: pd.DataFrame, code: str,
                      z_entry=1.5, z_exit=0.3, cost_oneway=0.0008) -> pd.DataFrame:
    """当日累積VWAPからの Zスコアで逆張り。
    Z>+z_entry → 反落ショート, Z<-z_entry → 反発ロング
    |Z|<z_exit でクローズ。15:25 強制クローズ
    """
    df = minute_df.copy()
    df["date"] = df.index.normalize()
    trades = []
    for d, g in df.groupby("date"):
        if len(g) < 30:
            continue
        # 当日累積VWAP
        tp = (g["high"] + g["low"] + g["close"]) / 3  # typical price
        cum_pv = (tp * g["volume"]).cumsum()
        cum_v = g["volume"].cumsum().replace(0, np.nan)
        vwap = cum_pv / cum_v
        # 価格-VWAP の rolling std (30分窓)
        dev = g["close"] - vwap
        sd = dev.rolling(30, min_periods=10).std()
        z = dev / sd

        # 10:00以降でエントリ判定
        signal_zone = z.between_time("10:00", "15:25")
        position = 0
        entry_price = None
        entry_ts = None
        for ts, zv in signal_zone.items():
            if pd.isna(zv):
                continue
            px = g.loc[ts, "close"]
            if position == 0:
                if zv > z_entry:
                    position = -1
                    entry_price = px
                    entry_ts = ts
                elif zv < -z_entry:
                    position = +1
                    entry_price = px
                    entry_ts = ts
            else:
                should_exit = abs(zv) < z_exit or ts.time() >= dtime(15, 25)
                if should_exit:
                    r = np.log(px / entry_price) * position
                    pnl = r - cost_oneway * 2
                    trades.append({"date": d, "code": code, "side": position,
                                   "entry": entry_price, "exit": px,
                                   "entry_ts": entry_ts, "exit_ts": ts,
                                   "entry_z": z.loc[entry_ts], "exit_z": zv,
                                   "pnl": pnl})
                    position = 0
                    entry_price = None
                    entry_ts = None
        # 強制クローズ
        if position != 0 and entry_price is not None:
            last_ts = signal_zone.index[-1]
            px = g.loc[last_ts, "close"]
            r = np.log(px / entry_price) * position
            pnl = r - cost_oneway * 2
            trades.append({"date": d, "code": code, "side": position,
                           "entry": entry_price, "exit": px,
                           "entry_ts": entry_ts, "exit_ts": last_ts,
                           "pnl": pnl})
    return pd.DataFrame(trades)


# ===================== Metrics =====================

def summarize(trades_df: pd.DataFrame, ann=245) -> dict:
    if len(trades_df) < 5:
        return {}
    n = len(trades_df)
    mu = trades_df["pnl"].mean()
    sd = trades_df["pnl"].std()
    eq = trades_df["pnl"].cumsum()
    dd = (eq - eq.cummax()).min()
    n_days = trades_df["date"].nunique()
    tpd = n / n_days
    sharpe = mu / sd * np.sqrt(ann * tpd) if sd > 0 else 0
    wr = (trades_df["pnl"] > 0).mean() * 100
    pf = (trades_df["pnl"][trades_df["pnl"] > 0].sum() /
          -trades_df["pnl"][trades_df["pnl"] < 0].sum()) if (trades_df["pnl"] < 0).any() else np.inf
    return {
        "N": n, "days": n_days, "tpd": round(tpd, 2),
        "mu%": mu * 100, "sd%": sd * 100,
        "Sharpe": round(sharpe, 2), "WR%": round(wr, 1),
        "PF": round(pf, 2), "Cum%": round(eq.iloc[-1] * 100, 1),
        "DD%": round(dd * 100, 1),
    }


def main():
    print("=== 非鉄8銘柄 シングルサイド戦略スクリーニング ===\n")
    print("--- データ読み込み ---")
    minutes = {}
    prev_closes = {}
    for c in SYMBOLS:
        m = fetch_minute(c)
        minutes[c] = m
        daily = fetch_daily_close(c)
        prev_closes[c] = daily.shift(1)
        print(f"  {c} {SYMBOLS[c]}: {len(m)}行")

    # === 銘柄別 × 戦略 ===
    cost = 0.0008
    print(f"\n--- 銘柄別 × 戦略評価 (コスト {cost*10000:.0f} bps片道) ---")
    rows = []
    all_trades = {"ORB": {}, "GapFade": {}, "VWAP_MR": {}}

    for c, name in SYMBOLS.items():
        # ORB
        t_orb = strat_ORB(minutes[c], c, or_min=15, cost_oneway=cost)
        all_trades["ORB"][c] = t_orb
        m = summarize(t_orb)
        if m:
            rows.append({"strat": "ORB(15min)", "code": c, "name": name, **m})

        # GapFade
        t_gf = strat_gap_fade(minutes[c], prev_closes[c], c, gap_thr=0.015, cost_oneway=cost)
        all_trades["GapFade"][c] = t_gf
        m = summarize(t_gf)
        if m:
            rows.append({"strat": "GapFade(1.5%)", "code": c, "name": name, **m})

        # VWAP MR
        t_vw = strat_vwap_revert(minutes[c], c, z_entry=1.5, z_exit=0.3, cost_oneway=cost)
        all_trades["VWAP_MR"][c] = t_vw
        m = summarize(t_vw)
        if m:
            rows.append({"strat": "VWAP_MR(1.5σ)", "code": c, "name": name, **m})

    df_all = pd.DataFrame(rows)
    print("\n[銘柄別 全戦略 (N≥10 only, Sharpe順)]")
    print(df_all[df_all["N"] >= 10].sort_values("Sharpe", ascending=False).to_string(index=False))

    # === 戦略ごとプール集計 ===
    print(f"\n--- プール集計 (全8銘柄まとめてシグナル) ---")
    for strat, trades_dict in all_trades.items():
        pool = pd.concat(trades_dict.values(), ignore_index=True)
        if len(pool) < 5:
            continue
        # 日次ポートフォリオ: 同日複数銘柄シグナルがあれば均等分散
        pool["pnl_scaled"] = pool.groupby("date")["pnl"].transform(
            lambda s: s / max(len(s), 1)
        )
        daily = pool.groupby("date")["pnl_scaled"].sum()
        n_days = len(daily)
        mu = daily.mean()
        sd = daily.std()
        sharpe = mu / sd * np.sqrt(245) if sd > 0 else 0
        eq = daily.cumsum()
        dd = (eq - eq.cummax()).min()
        wr = (daily > 0).mean() * 100
        pf = (daily[daily > 0].sum() / -daily[daily < 0].sum()) if (daily < 0).any() else np.inf
        print(f"  {strat:12s}: N_trades={len(pool):4d}, N_days={n_days:3d}, "
              f"daily_mu={mu*100:+.3f}%, Sharpe={sharpe:+.2f}, "
              f"PF={pf:.2f}, Cum={eq.iloc[-1]*100:+.1f}%, DD={dd*100:.1f}%, WR={wr:.1f}%")

    # === コスト感度 (戦略ごとプール) ===
    print(f"\n--- コスト感度 (戦略ごとプール) ---")
    for c_bps in [3, 5, 8, 10, 12, 15]:
        cost_c = c_bps / 10000
        cost_results = {}
        for strat_name, fn, kwargs in [
            ("ORB", strat_ORB, {"or_min": 15}),
            ("GapFade", strat_gap_fade, {"gap_thr": 0.015, "prev_close_each": True}),
            ("VWAP_MR", strat_vwap_revert, {"z_entry": 1.5, "z_exit": 0.3}),
        ]:
            pool_list = []
            for code in SYMBOLS:
                if strat_name == "GapFade":
                    t = strat_gap_fade(minutes[code], prev_closes[code], code,
                                       gap_thr=0.015, cost_oneway=cost_c)
                elif strat_name == "ORB":
                    t = strat_ORB(minutes[code], code, or_min=15, cost_oneway=cost_c)
                elif strat_name == "VWAP_MR":
                    t = strat_vwap_revert(minutes[code], code, z_entry=1.5, z_exit=0.3,
                                          cost_oneway=cost_c)
                pool_list.append(t)
            pool = pd.concat(pool_list, ignore_index=True)
            if len(pool) < 5:
                continue
            pool["pnl_scaled"] = pool.groupby("date")["pnl"].transform(
                lambda s: s / max(len(s), 1)
            )
            daily = pool.groupby("date")["pnl_scaled"].sum()
            mu, sd = daily.mean(), daily.std()
            sharpe = mu / sd * np.sqrt(245) if sd > 0 else 0
            cost_results[strat_name] = sharpe
        line = f"  cost={c_bps:2d} bps: "
        line += ", ".join([f"{k}={v:+.2f}" for k, v in cost_results.items()])
        print(line)

    # === 保存 ===
    out_dir = os.path.dirname(os.path.abspath(__file__))
    df_all.to_csv(os.path.join(out_dir, "by_symbol_results.csv"), index=False)
    for strat, td in all_trades.items():
        for c, t in td.items():
            if len(t) > 0:
                t.to_csv(os.path.join(out_dir, f"trades_{strat}_{c}.csv"), index=False)
    print(f"\n保存: by_symbol_results.csv, trades_*.csv")


if __name__ == "__main__":
    main()
