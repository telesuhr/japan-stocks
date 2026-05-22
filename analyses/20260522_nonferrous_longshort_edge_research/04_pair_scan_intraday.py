"""
全28ペア × 1分足ベース Zスコア平均回帰 スキャン

ロジック:
  各日内で各ペアの logスプレッド = log(P_A) - β * log(P_B) を 1分足で計算
  β はそのペアの過去20営業日の日次回帰係数 (毎日更新)
  当日内のスプレッドの当日VWAP/開場時平均からの Zスコア で:
    Z > +entry → A ショート B ロング
    Z < -entry → A ロング B ショート
    |Z| < exit_z → クローズ
    15:25 強制クローズ (オーバーナイトしない)

評価指標:
  total PnL, Sharpe (年率), 取引数, 勝率, 平均保有時間, MaxDD
  取引コスト: 片道 8bps (合計 16bps/往復ペア = 32bps/ラウンドL/S)

注: 全ペア × 全日を 1分足で計算すると重い。1分足にダウンサンプル + 軽量化。
"""

import os
import psycopg2
import pandas as pd
import numpy as np
from itertools import combinations
from datetime import time as dtime

PG_CONFIG = {
    "host": os.environ.get("PGHOST", "omen"),
    "port": int(os.environ.get("PGPORT", 5432)),
    "user": os.environ.get("PGUSER", "postgres"),
    "dbname": os.environ.get("PGDATABASE", "market_data"),
}

SYMBOLS = {
    "57060": "三井金属",
    "57110": "三菱マテリアル",
    "57130": "住友金属鉱山",
    "57140": "DOWA HD",
    "50160": "JX金属",
    "58010": "古河電工",
    "58020": "住友電工",
    "58030": "フジクラ",
}

START = "2025-04-01"
END = "2026-05-21"
COST_ONEWAY = 0.0008  # 8 bps 片道


def fetch_minute_close(code: str) -> pd.Series:
    conn = psycopg2.connect(**PG_CONFIG)
    sql = """
        SELECT ts, close
        FROM stocks_intraday
        WHERE code=%s AND ts>=%s AND ts<=%s
        ORDER BY ts
    """
    df = pd.read_sql(sql, conn, params=(code, START, END + " 23:59:59"))
    conn.close()
    df["ts"] = pd.to_datetime(df["ts"])
    return df.set_index("ts")["close"].rename(code)


def fetch_daily_close(code: str) -> pd.Series:
    conn = psycopg2.connect(**PG_CONFIG)
    sql = "SELECT date, close FROM stocks_daily WHERE code=%s AND date BETWEEN %s AND %s ORDER BY date"
    df = pd.read_sql(sql, conn, params=(code, START, END))
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["close"].rename(code)


def intraday_pair_backtest(p_a: pd.Series, p_b: pd.Series,
                           daily_a: pd.Series, daily_b: pd.Series,
                           beta_window=20, entry_z=2.0, exit_z=0.5,
                           cost_oneway=COST_ONEWAY) -> dict:
    """
    1分足でペアスプレッド戦略を当日完結バックテスト。
    各日:
      β = 過去20日終値の OLS slope (log(A) on log(B))
      スプレッド = log(A) - β log(B)
      その日のスプレッド平均・std を「9:00-10:00 の値」で算出 (lookahead 防止)
      Z = (spread - mean60min) / std60min
      Z > +entry → ショートA・ロングB
      Z < -entry → ロングA・ショートB
      |Z| < exit → クローズ
      15:25 強制クローズ
    """
    # 1分足を共通インデックスに揃え
    df = pd.concat([p_a, p_b], axis=1).dropna()
    df.columns = ["A", "B"]
    df["date"] = df.index.normalize()

    # β を日次更新 (前日終値までの 20日 OLS)
    common_days = daily_a.index.intersection(daily_b.index)
    la = np.log(daily_a.loc[common_days])
    lb = np.log(daily_b.loc[common_days])
    # rolling β
    beta_daily = la.rolling(beta_window).cov(lb) / lb.rolling(beta_window).var()
    beta_daily = beta_daily.shift(1)  # 当日には前日終値時点の β を使う

    trades = []
    minute_pnl = []  # 日次PnL

    for d, g in df.groupby("date"):
        if d not in beta_daily.index or pd.isna(beta_daily.loc[d]):
            continue
        beta = beta_daily.loc[d]
        if beta <= 0 or beta > 5:  # 異常値除外
            continue

        spread = np.log(g["A"]) - beta * np.log(g["B"])

        # 朝9:00-10:00で mean/std 推定 (これより前の lookahead は無し)
        morning = spread.between_time("09:00", "10:00")
        if len(morning) < 30:
            continue
        mu, sd = morning.mean(), morning.std()
        if sd == 0 or pd.isna(sd):
            continue
        z = (spread - mu) / sd

        # 10:00以降でエントリ判定
        signal_zone = z.between_time("10:00", "15:25")
        position = 0  # 0=no, +1=longA shortB, -1=shortA longB
        entry_price_a = None
        entry_price_b = None
        entry_time = None

        for ts, zv in signal_zone.items():
            if pd.isna(zv):
                continue
            pa_now = g.loc[ts, "A"]
            pb_now = g.loc[ts, "B"]

            if position == 0:
                if zv > entry_z:
                    position = -1
                    entry_price_a = pa_now
                    entry_price_b = pb_now
                    entry_time = ts
                elif zv < -entry_z:
                    position = +1
                    entry_price_a = pa_now
                    entry_price_b = pb_now
                    entry_time = ts
            else:
                # exit conditions
                should_exit = abs(zv) < exit_z
                # 15:25強制クローズ
                if ts.time() >= dtime(15, 25):
                    should_exit = True
                if should_exit:
                    # PnL: ロング側のリターン - ショート側のリターン
                    if position == +1:
                        # long A, short B
                        r_a = np.log(pa_now / entry_price_a)
                        r_b = np.log(pb_now / entry_price_b)
                        pnl = (r_a - r_b) / 2 - cost_oneway * 2
                    else:
                        # short A, long B
                        r_a = np.log(pa_now / entry_price_a)
                        r_b = np.log(pb_now / entry_price_b)
                        pnl = (-r_a + r_b) / 2 - cost_oneway * 2
                    trades.append({
                        "date": d,
                        "entry_time": entry_time,
                        "exit_time": ts,
                        "hold_min": int((ts - entry_time).total_seconds() / 60),
                        "position": position,
                        "entry_z": z.loc[entry_time] if entry_time in z.index else np.nan,
                        "exit_z": zv,
                        "pnl": pnl,
                    })
                    position = 0
                    entry_price_a = None
                    entry_price_b = None
                    entry_time = None

        # 強制クローズ: ループ抜けてもまだ position 残っていればクローズ
        if position != 0 and entry_time is not None:
            last_ts = signal_zone.index[-1]
            pa_now = g.loc[last_ts, "A"]
            pb_now = g.loc[last_ts, "B"]
            if position == +1:
                r_a = np.log(pa_now / entry_price_a)
                r_b = np.log(pb_now / entry_price_b)
                pnl = (r_a - r_b) / 2 - cost_oneway * 2
            else:
                r_a = np.log(pa_now / entry_price_a)
                r_b = np.log(pb_now / entry_price_b)
                pnl = (-r_a + r_b) / 2 - cost_oneway * 2
            trades.append({
                "date": d,
                "entry_time": entry_time,
                "exit_time": last_ts,
                "hold_min": int((last_ts - entry_time).total_seconds() / 60),
                "position": position,
                "entry_z": z.loc[entry_time] if entry_time in z.index else np.nan,
                "exit_z": z.loc[last_ts] if last_ts in z.index else np.nan,
                "pnl": pnl,
            })

    df_trades = pd.DataFrame(trades)
    if len(df_trades) < 5:
        return {"trades": df_trades, "metrics": {}}

    eq = df_trades["pnl"].cumsum()
    dd = (eq - eq.cummax()).min()
    n = len(df_trades)
    mu = df_trades["pnl"].mean()
    sd = df_trades["pnl"].std()
    # Sharpe 年率化: 取引/日 ベース
    trades_per_day = n / df_trades["date"].nunique()
    sharpe = mu / sd * np.sqrt(245 * trades_per_day) if sd > 0 else 0
    wr = (df_trades["pnl"] > 0).mean() * 100
    pf = (df_trades["pnl"][df_trades["pnl"] > 0].sum() /
          -df_trades["pnl"][df_trades["pnl"] < 0].sum()) if (df_trades["pnl"] < 0).any() else np.inf

    return {
        "trades": df_trades,
        "metrics": {
            "N_trades": n,
            "N_days_active": df_trades["date"].nunique(),
            "trades_per_active_day": round(trades_per_day, 2),
            "mean_per_trade_%": mu * 100,
            "std_per_trade_%": sd * 100,
            "Sharpe_ann": sharpe,
            "winrate_%": wr,
            "PF": pf,
            "avg_hold_min": df_trades["hold_min"].mean(),
            "cum_PnL_%": eq.iloc[-1] * 100,
            "maxDD_%": dd * 100,
        }
    }


def main():
    print("=== 全28ペア × 1分足 Zスコア平均回帰 スキャン ===")
    print(f"コスト: {COST_ONEWAY*10000:.0f} bps 片道, entry_z=2.0, exit_z=0.5")
    print(f"期間: {START} 〜 {END}\n")

    print("--- 1分足読み込み ---")
    minutes = {}
    dailies = {}
    for c in SYMBOLS:
        minutes[c] = fetch_minute_close(c)
        dailies[c] = fetch_daily_close(c)
        print(f"  {c} {SYMBOLS[c]}: {len(minutes[c])} 分足, {len(dailies[c])} 日足")

    pairs = list(combinations(SYMBOLS.keys(), 2))
    print(f"\n--- {len(pairs)} ペアをバックテスト ---")

    summary = []
    all_trades = {}
    for a, b in pairs:
        res = intraday_pair_backtest(
            minutes[a], minutes[b], dailies[a], dailies[b],
            entry_z=2.0, exit_z=0.5
        )
        m = res["metrics"]
        if not m:
            print(f"  {a}-{b}: trades<5, skip")
            continue
        row = {"pair": f"{a}-{b}", "name": f"{SYMBOLS[a]}/{SYMBOLS[b]}", **m}
        summary.append(row)
        all_trades[f"{a}-{b}"] = res["trades"]
        print(f"  {a}-{b} ({SYMBOLS[a]}/{SYMBOLS[b]}): "
              f"N={m['N_trades']}, Sharpe={m['Sharpe_ann']:.2f}, "
              f"PF={m['PF']:.2f}, Cum={m['cum_PnL_%']:.1f}%, "
              f"DD={m['maxDD_%']:.1f}%, hold={m['avg_hold_min']:.0f}m")

    df_sum = pd.DataFrame(summary).sort_values("Sharpe_ann", ascending=False)
    print(f"\n--- Sharpe ランキング (Top10) ---")
    print(df_sum.head(10)[["pair", "name", "N_trades", "Sharpe_ann",
                            "PF", "cum_PnL_%", "maxDD_%", "avg_hold_min"]].to_string(index=False))

    print(f"\n--- 取引コスト前評価 (Pure α 確認) ---")
    summary_nocost = []
    for a, b in pairs:
        res = intraday_pair_backtest(
            minutes[a], minutes[b], dailies[a], dailies[b],
            entry_z=2.0, exit_z=0.5, cost_oneway=0
        )
        m = res["metrics"]
        if not m:
            continue
        row = {"pair": f"{a}-{b}", "name": f"{SYMBOLS[a]}/{SYMBOLS[b]}",
               "Sharpe_nocost": m["Sharpe_ann"], "Cum_nocost_%": m["cum_PnL_%"]}
        summary_nocost.append(row)
    df_nc = pd.DataFrame(summary_nocost).sort_values("Sharpe_nocost", ascending=False)
    print(df_nc.head(10).to_string(index=False))

    # 保存
    out_dir = os.path.dirname(os.path.abspath(__file__))
    df_sum.to_csv(os.path.join(out_dir, "pair_scan_summary.csv"), index=False)
    df_nc.to_csv(os.path.join(out_dir, "pair_scan_summary_nocost.csv"), index=False)
    # トップ3ペアのトレード詳細だけ保存
    for pair_name, _ in df_sum.head(3).iterrows():
        pkey = df_sum.loc[pair_name, "pair"]
        if pkey in all_trades:
            all_trades[pkey].to_csv(os.path.join(out_dir, f"trades_pair_{pkey}.csv"), index=False)
    print(f"\n保存: pair_scan_summary.csv, top3 trades_pair_*.csv")


if __name__ == "__main__":
    main()
