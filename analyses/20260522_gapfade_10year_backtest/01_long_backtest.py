"""
寄付ギャップフェード戦略 — 10年バックテスト (日足ベース)

戦略 (確定版):
  ・前日終値→当日寄付ギャップ = log(open / prev_close)
  ・|gap| ≥ 1.5% なら寄付Open価格でエントリ
  ・ギャップと逆方向 (gap>0 → ショート, gap<0 → ロング)
  ・引けClose価格でエグジット
  ・コスト 8 bps 片道 (= 16 bps 往復)
  ・1銘柄1日1ポジ

期間: 2016-05-10 〜 2026-05-22 (約10年, 2,452営業日)
対象7銘柄: 5706/5711/5713/5714/5801/5802/5803 (JX金属5016除く=データ短い)

検証項目:
  (1) 銘柄別 全期間 結果
  (2) 年別 Sharpe (安定性チェック)
  (3) IS (2016-2022) / OOS (2023-2026) 分割
  (4) ギャップ閾値スイープ (1.0 / 1.5 / 2.0 / 2.5 / 3.0%)
  (5) ポートフォリオ (Top-N選択)
  (6) コスト感度
  (7) ストップロス効果
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
    "57060": "三井金属", "57110": "三菱マテリアル", "57130": "住友金属鉱山",
    "57140": "DOWA HD",
    "58010": "古河電工", "58020": "住友電工", "58030": "フジクラ",
}
START = "2016-05-10"
END = "2026-05-22"
COST_ONEWAY = 0.0008


def fetch_daily(code: str) -> pd.DataFrame:
    conn = psycopg2.connect(**PG_CONFIG)
    sql = """
        SELECT date, open, high, low, close, volume,
               adj_open, adj_high, adj_low, adj_close, adj_volume
        FROM stocks_daily
        WHERE code=%s AND date BETWEEN %s AND %s
        ORDER BY date
    """
    df = pd.read_sql(sql, conn, params=(code, START, END))
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")


def gapfade_trades(daily: pd.DataFrame, code: str,
                   gap_thr: float = 0.015,
                   cost_oneway: float = COST_ONEWAY,
                   stop_loss_pct: float | None = None) -> pd.DataFrame:
    """
    寄付ギャップフェードを日足で評価。
    stop_loss_pct: 含み損率 (絶対値) でクローズ。例 0.03 → -3%でストップ。
                    日足だと正確な発火タイミングが分からないので、
                    日中安値(ロング時) / 高値(ショート時) で判定。
    """
    df = daily.copy()
    df["prev_close"] = df["close"].shift(1)
    df = df.dropna(subset=["prev_close", "open"])
    df["gap"] = np.log(df["open"] / df["prev_close"])
    sig = df[df["gap"].abs() >= gap_thr].copy()
    sig["side"] = -np.sign(sig["gap"])  # ギャップ逆方向

    rows = []
    for d, row in sig.iterrows():
        side = int(row["side"])
        open_px = row["open"]
        close_px = row["close"]
        high = row["high"]
        low = row["low"]

        # ストップロス処理
        stopped = False
        if stop_loss_pct is not None:
            if side == +1:  # long
                stop_px = open_px * (1 - stop_loss_pct)
                if low <= stop_px:
                    exit_px = stop_px
                    stopped = True
            else:  # short
                stop_px = open_px * (1 + stop_loss_pct)
                if high >= stop_px:
                    exit_px = stop_px
                    stopped = True
        if not stopped:
            exit_px = close_px

        r = np.log(exit_px / open_px) * side
        pnl = r - cost_oneway * 2
        rows.append({
            "date": d, "code": code, "side": side,
            "gap_pct": row["gap"] * 100,
            "entry": open_px, "exit": exit_px, "close": close_px,
            "high": high, "low": low,
            "stopped": stopped,
            "ret_pct": r * 100,
            "pnl": pnl,
        })
    return pd.DataFrame(rows)


def metrics(daily_pnl: pd.Series, ann=245) -> dict:
    s = daily_pnl.dropna()
    if len(s) < 5:
        return {}
    mu, sd = s.mean(), s.std()
    sharpe = mu / sd * np.sqrt(ann) if sd > 0 else 0
    eq = s.cumsum()
    dd = (eq - eq.cummax()).min()
    wr = (s > 0).mean() * 100
    pf = (s[s > 0].sum() / -s[s < 0].sum()) if (s < 0).any() else np.inf
    return {
        "N": len(s),
        "mu%": round(mu*100, 4),
        "Sharpe": round(sharpe, 2),
        "WR%": round(wr, 1),
        "PF": round(pf, 2),
        "Cum%": round(eq.iloc[-1]*100, 1),
        "DD%": round(dd*100, 1),
    }


def trades_to_daily(trades_df: pd.DataFrame) -> pd.Series:
    """同日複数銘柄ある場合は均等分散"""
    if len(trades_df) == 0:
        return pd.Series(dtype=float)
    tmp = trades_df.copy()
    tmp["pnl_scaled"] = tmp.groupby("date")["pnl"].transform(lambda s: s / max(len(s), 1))
    return tmp.groupby("date")["pnl_scaled"].sum().sort_index()


def main():
    print("=== 寄付ギャップフェード戦略 10年バックテスト ===\n")
    print(f"期間: {START} 〜 {END}")
    print(f"対象: {len(SYMBOLS)}銘柄, コスト: {COST_ONEWAY*10000:.0f}bps片道, 閾値: 1.5%\n")

    # --- データ読み込み ---
    dailies = {}
    for c in SYMBOLS:
        d = fetch_daily(c)
        dailies[c] = d
        print(f"  {c} {SYMBOLS[c]:10s}: {len(d)}日 ({d.index.min().date()} 〜 {d.index.max().date()})")

    # ===== (1) 銘柄別 全期間結果 =====
    print(f"\n--- (1) 銘柄別 全期間結果 (thr=1.5%, 8bps) ---")
    by_symbol_trades = {}
    rows = []
    for c, name in SYMBOLS.items():
        t = gapfade_trades(dailies[c], c, gap_thr=0.015, cost_oneway=COST_ONEWAY)
        by_symbol_trades[c] = t
        m = metrics(t.set_index("date")["pnl"]) if len(t) > 0 else {}
        rows.append({"code": c, "name": name, **m})
        print(f"  {c} {name:10s}: N={m.get('N',0):4d}, Sharpe={m.get('Sharpe',0):+.2f}, "
              f"WR={m.get('WR%',0):.1f}%, PF={m.get('PF',0):.2f}, "
              f"Cum={m.get('Cum%',0):+.1f}%, DD={m.get('DD%',0):.1f}%")
    df_by_symbol = pd.DataFrame(rows)

    # ===== (2) 年別 Sharpe (Top5 = 全銘柄/Sharpeソート) =====
    print(f"\n--- (2) 年別 Sharpe (全7銘柄プール) ---")
    all_trades = pd.concat(by_symbol_trades.values(), ignore_index=True)
    all_daily = trades_to_daily(all_trades)
    all_daily.index = pd.to_datetime(all_daily.index)
    print(f"  全期間: N={len(all_daily)}, total trades={len(all_trades)}")

    yearly_rows = []
    for year in range(2016, 2027):
        sub = all_daily[all_daily.index.year == year]
        n_trades = (all_trades["date"].dt.year == year).sum() if hasattr(all_trades["date"].iloc[0], 'year') else len(sub)
        m = metrics(sub)
        yearly_rows.append({"year": year, "N_days": m.get("N", 0), **m})
        print(f"  {year}: N_days={m.get('N',0):3d}, Sharpe={m.get('Sharpe',0):+.2f}, "
              f"Cum={m.get('Cum%',0):+.1f}%, DD={m.get('DD%',0):.1f}%, WR={m.get('WR%',0):.1f}%")
    df_yearly = pd.DataFrame(yearly_rows)

    # ===== (3) IS / OOS 分割 =====
    print(f"\n--- (3) IS (2016-2022) / OOS (2023-2026) 分割 ---")
    for label, start_y, end_y in [("IS (2016-2022)", 2016, 2022),
                                    ("OOS (2023-2026)", 2023, 2026)]:
        sub = all_daily[(all_daily.index.year >= start_y) & (all_daily.index.year <= end_y)]
        m = metrics(sub)
        print(f"  {label}: N_days={m.get('N',0):4d}, Sharpe={m.get('Sharpe',0):+.2f}, "
              f"Cum={m.get('Cum%',0):+.1f}%, DD={m.get('DD%',0):.1f}%, "
              f"WR={m.get('WR%',0):.1f}%, PF={m.get('PF',0):.2f}")

    # 銘柄別 IS/OOS
    print(f"\n--- 銘柄別 IS/OOS Sharpe ---")
    isoos_rows = []
    for c, name in SYMBOLS.items():
        t = by_symbol_trades[c]
        if len(t) == 0:
            continue
        t["date"] = pd.to_datetime(t["date"])
        is_trades = t[t["date"].dt.year <= 2022]
        oos_trades = t[t["date"].dt.year >= 2023]
        m_is = metrics(is_trades.set_index("date")["pnl"]) if len(is_trades) >= 5 else {}
        m_oos = metrics(oos_trades.set_index("date")["pnl"]) if len(oos_trades) >= 5 else {}
        isoos_rows.append({
            "code": c, "name": name,
            "IS_N": m_is.get("N", 0), "IS_Sharpe": m_is.get("Sharpe", 0),
            "IS_Cum%": m_is.get("Cum%", 0),
            "OOS_N": m_oos.get("N", 0), "OOS_Sharpe": m_oos.get("Sharpe", 0),
            "OOS_Cum%": m_oos.get("Cum%", 0),
        })
        print(f"  {c} {name:10s}: IS Sharpe={m_is.get('Sharpe',0):+.2f} (N={m_is.get('N',0):3d}) | "
              f"OOS Sharpe={m_oos.get('Sharpe',0):+.2f} (N={m_oos.get('N',0):3d})")
    df_isoos = pd.DataFrame(isoos_rows)

    # ===== (4) 閾値スイープ =====
    print(f"\n--- (4) 閾値スイープ (全7銘柄プール, 8bps) ---")
    thr_rows = []
    for thr in [0.010, 0.015, 0.020, 0.025, 0.030, 0.040]:
        td = {c: gapfade_trades(dailies[c], c, gap_thr=thr, cost_oneway=COST_ONEWAY) for c in SYMBOLS}
        pool = pd.concat(td.values(), ignore_index=True)
        daily = trades_to_daily(pool)
        m = metrics(daily)
        thr_rows.append({"thr%": thr*100, "N_trades": len(pool), **m})
        print(f"  thr={thr*100:.1f}%: N_trades={len(pool):4d}, N_days={m.get('N',0):4d}, "
              f"Sharpe={m.get('Sharpe',0):+.2f}, Cum={m.get('Cum%',0):+.1f}%, "
              f"DD={m.get('DD%',0):.1f}%, WR={m.get('WR%',0):.1f}%")
    df_thr = pd.DataFrame(thr_rows)

    # ===== (5) Top-N 銘柄選択 (IS Sharpe基準で選び、OOSで評価) =====
    print(f"\n--- (5) IS Sharpe基準でTop-N選択 → OOS評価 ---")
    is_sharpe_sorted = df_isoos.sort_values("IS_Sharpe", ascending=False)
    for K in [3, 4, 5, 6, 7]:
        top_codes = is_sharpe_sorted.head(K)["code"].tolist()
        oos_trades = []
        for c in top_codes:
            t = by_symbol_trades[c]
            t["date"] = pd.to_datetime(t["date"])
            oos_trades.append(t[t["date"].dt.year >= 2023])
        pool = pd.concat(oos_trades, ignore_index=True)
        daily = trades_to_daily(pool)
        m = metrics(daily)
        names_short = [SYMBOLS[c][:6] for c in top_codes]
        print(f"  Top{K}: [{', '.join(names_short)}]")
        print(f"        OOS: N_days={m.get('N',0)}, Sharpe={m.get('Sharpe',0):+.2f}, "
              f"Cum={m.get('Cum%',0):+.1f}%, DD={m.get('DD%',0):.1f}%, WR={m.get('WR%',0):.1f}%")

    # ===== (6) コスト感度 (全7銘柄プール, thr=1.5%) =====
    print(f"\n--- (6) コスト感度 (全7銘柄プール, thr=1.5%, 全期間) ---")
    for c_bps in [0, 3, 5, 8, 10, 15, 20, 30]:
        cost = c_bps / 10000
        td = {c: gapfade_trades(dailies[c], c, gap_thr=0.015, cost_oneway=cost) for c in SYMBOLS}
        pool = pd.concat(td.values(), ignore_index=True)
        daily = trades_to_daily(pool)
        m = metrics(daily)
        print(f"  cost={c_bps:2d} bps: Sharpe={m.get('Sharpe',0):+.2f}, "
              f"Cum={m.get('Cum%',0):+.1f}%, DD={m.get('DD%',0):.1f}%")

    # ===== (7) ストップロス効果 =====
    print(f"\n--- (7) ストップロス効果 (全7銘柄プール, thr=1.5%, 8bps) ---")
    for sl in [None, 0.05, 0.03, 0.02, 0.015]:
        td = {c: gapfade_trades(dailies[c], c, gap_thr=0.015, cost_oneway=COST_ONEWAY,
                                  stop_loss_pct=sl) for c in SYMBOLS}
        pool = pd.concat(td.values(), ignore_index=True)
        daily = trades_to_daily(pool)
        m = metrics(daily)
        n_stop = pool["stopped"].sum() if "stopped" in pool.columns else 0
        label = "なし" if sl is None else f"{sl*100:.1f}%"
        print(f"  SL={label:5s}: Sharpe={m.get('Sharpe',0):+.2f}, "
              f"Cum={m.get('Cum%',0):+.1f}%, DD={m.get('DD%',0):.1f}%, "
              f"発火{n_stop}回")

    # ===== 保存 =====
    out_dir = os.path.dirname(os.path.abspath(__file__))
    df_by_symbol.to_csv(os.path.join(out_dir, "by_symbol_results.csv"), index=False)
    df_yearly.to_csv(os.path.join(out_dir, "yearly_results.csv"), index=False)
    df_isoos.to_csv(os.path.join(out_dir, "isoos_results.csv"), index=False)
    df_thr.to_csv(os.path.join(out_dir, "threshold_sweep.csv"), index=False)
    all_trades.to_csv(os.path.join(out_dir, "all_trades.csv"), index=False)
    all_daily.to_csv(os.path.join(out_dir, "all_daily_pnl.csv"), header=["pnl"])
    print(f"\n保存: by_symbol_results.csv, yearly_results.csv, isoos_results.csv, "
          f"threshold_sweep.csv, all_trades.csv, all_daily_pnl.csv")


if __name__ == "__main__":
    main()
