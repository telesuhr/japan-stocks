"""
本命戦略: クロスセクション・モメンタム L/S (前場 → 後場)

ロジック:
  各銘柄の前場リターン (09:00 寄 → 11:30 終値) を、所属グループ内でランク化
  各日 12:30 後場寄付で:
    - 銅鉱山グループ (5銘柄): 前場リターン上位1ロング, 下位1ショート
    - 電線グループ   (3銘柄): 前場リターン上位1ロング, 下位1ショート
  15:30 引けで全クローズ → 1日完結
  各ポジションのドルウェイトを揃える (= L/S ドルニュートラル)

コスト: 0.10% 片道 (=往復0.20% per leg) ※マーケットインパクト考慮
評価:
  - 取引コスト 0bps / 5bps / 10bps / 15bps 片道 で感度分析
  - 日次 PL / 年率 Sharpe / MaxDD / 勝率 / PF
  - 半期ごとの安定性 (in-sample/out-of-sample 的視点)
  - 個別グループの寄与
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
            "turnover": g["turnover_value"].sum(),
        })
    out = pd.DataFrame(rows).set_index("date")
    return out


def backtest(am_open, am_close, pm_open, pm_close, groups, cost_oneway=0.001):
    """各日 12:30 後場寄付エントリ・15:30 引けクローズ。
    各グループ内 top1 long / bottom1 short, ドル等量。
    Returns: daily_pnl (Series, fractional return on gross notional)
    """
    r_am = np.log(am_close / am_open)
    r_pm = np.log(pm_close / pm_open)

    miner = [c for c, g in groups.items() if g == "miner"]
    wire = [c for c, g in groups.items() if g == "wire"]

    daily_pnl_total = []
    daily_pnl_miner = []
    daily_pnl_wire = []
    daily_positions = []

    for d in r_am.index:
        pnl_m, pnl_w = 0.0, 0.0
        pos_today = {"date": d}

        ram_m = r_am.loc[d, miner].dropna()
        if len(ram_m) >= 3:
            long_m = ram_m.idxmax()
            short_m = ram_m.idxmin()
            r_long = r_pm.loc[d, long_m]
            r_short = r_pm.loc[d, short_m]
            # ペアあたり 2 legs, 各 leg は notional 1 → グロス4倍にしないよう、
            # gross notional=2 (long 1 + short 1) でリターンは (r_long - r_short)/2
            pnl_m = (r_long - r_short) / 2 - cost_oneway * 2
            pos_today["miner_long"] = long_m
            pos_today["miner_short"] = short_m
            pos_today["pnl_miner"] = pnl_m

        ram_w = r_am.loc[d, wire].dropna()
        if len(ram_w) >= 3:
            long_w = ram_w.idxmax()
            short_w = ram_w.idxmin()
            r_long = r_pm.loc[d, long_w]
            r_short = r_pm.loc[d, short_w]
            pnl_w = (r_long - r_short) / 2 - cost_oneway * 2
            pos_today["wire_long"] = long_w
            pos_today["wire_short"] = short_w
            pos_today["pnl_wire"] = pnl_w

        # 2グループを同等ウェイトで合算 (各グループに資金半分)
        pnl_total = (pnl_m + pnl_w) / 2
        pos_today["pnl_total"] = pnl_total

        daily_pnl_total.append({"date": d, "pnl": pnl_total})
        daily_pnl_miner.append({"date": d, "pnl": pnl_m})
        daily_pnl_wire.append({"date": d, "pnl": pnl_w})
        daily_positions.append(pos_today)

    return {
        "total": pd.DataFrame(daily_pnl_total).set_index("date")["pnl"],
        "miner": pd.DataFrame(daily_pnl_miner).set_index("date")["pnl"],
        "wire": pd.DataFrame(daily_pnl_wire).set_index("date")["pnl"],
        "positions": pd.DataFrame(daily_positions),
    }


def metrics(pnl: pd.Series, ann=245):
    pnl = pnl.dropna()
    if len(pnl) < 5:
        return {}
    mu = pnl.mean()
    sd = pnl.std()
    sharpe = mu / sd * np.sqrt(ann) if sd > 0 else 0
    eq = pnl.cumsum()
    dd = (eq - eq.cummax()).min()
    wins = (pnl > 0).sum()
    losses = (pnl < 0).sum()
    pf = (pnl[pnl > 0].sum() / -pnl[pnl < 0].sum()) if losses > 0 else np.inf
    return {
        "N": len(pnl),
        "mean_%/d": mu * 100,
        "std_%/d": sd * 100,
        "Sharpe": sharpe,
        "Winrate_%": wins / len(pnl) * 100,
        "PF": pf,
        "Cum_%": eq.iloc[-1] * 100,
        "MaxDD_%": dd * 100,
    }


def main():
    print("=== クロスセクション・モメンタム L/S バックテスト ===\n")
    print("--- データ読み込み ---")
    feats = {}
    for code in SYMBOLS:
        m = fetch_minute(code)
        feats[code] = daily_features(m)
        print(f"  {code} {SYMBOLS[code][0]}: {len(feats[code])}日")

    am_open = pd.DataFrame({c: f["open"] for c, f in feats.items()})
    am_close = pd.DataFrame({c: f["am_close"] for c, f in feats.items()})
    pm_open = pd.DataFrame({c: f["pm_open"] for c, f in feats.items()})
    pm_close = pd.DataFrame({c: f["close"] for c, f in feats.items()})
    groups = {c: g for c, (_, g) in SYMBOLS.items()}

    # ---- コスト感度 ----
    print("\n--- コスト感度分析 ---")
    cost_sensitivity = []
    for c_bps in [0, 5, 10, 15, 20]:
        cost = c_bps / 10000
        res = backtest(am_open, am_close, pm_open, pm_close, groups, cost_oneway=cost)
        m = metrics(res["total"])
        m["cost_bps"] = c_bps
        cost_sensitivity.append(m)
    df_cost = pd.DataFrame(cost_sensitivity)
    print(df_cost.round(2).to_string(index=False))

    # ---- 基準コスト 10bps でフル結果 ----
    print("\n--- 詳細バックテスト (コスト 10bps 片道) ---")
    res = backtest(am_open, am_close, pm_open, pm_close, groups, cost_oneway=0.001)

    m_total = metrics(res["total"])
    m_miner = metrics(res["miner"])
    m_wire = metrics(res["wire"])

    print(f"\n[Total]    : {m_total}")
    print(f"[Miner群]  : {m_miner}")
    print(f"[Wire群]   : {m_wire}")

    # ---- 半期ごとの安定性 ----
    print("\n--- 半期ごとの Sharpe (安定性チェック) ---")
    pnl = res["total"]
    pnl.index = pd.to_datetime(pnl.index)
    halves = []
    for label, mask in [
        ("2025H1 (4-9)", (pnl.index >= "2025-04-01") & (pnl.index < "2025-10-01")),
        ("2025H2 (10-3)", (pnl.index >= "2025-10-01") & (pnl.index < "2026-04-01")),
        ("2026H1 (4-5)", (pnl.index >= "2026-04-01") & (pnl.index <= "2026-05-21")),
    ]:
        sub = pnl[mask]
        m = metrics(sub)
        m["period"] = label
        halves.append(m)
        print(f"  {label}: N={m.get('N')}, Sharpe={m.get('Sharpe', 0):.2f}, "
              f"Cum={m.get('Cum_%', 0):.1f}%, DD={m.get('MaxDD_%', 0):.1f}%")

    # ---- 保存 ----
    out_dir = os.path.dirname(os.path.abspath(__file__))
    res["positions"].to_csv(os.path.join(out_dir, "momentum_positions.csv"), index=False)
    res["total"].to_csv(os.path.join(out_dir, "momentum_pnl.csv"))
    df_cost.to_csv(os.path.join(out_dir, "cost_sensitivity.csv"), index=False)
    pd.DataFrame([m_total, m_miner, m_wire],
                 index=["Total", "Miner", "Wire"]).to_csv(
        os.path.join(out_dir, "momentum_metrics.csv"))
    print(f"\n保存: momentum_pnl.csv, momentum_positions.csv, cost_sensitivity.csv, momentum_metrics.csv")


if __name__ == "__main__":
    main()
