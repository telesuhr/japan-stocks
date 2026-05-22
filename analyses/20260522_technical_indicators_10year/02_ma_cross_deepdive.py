"""
MA25/75 ゴールデンクロス戦略 深堀り

検証:
  (1) MAパラメータ感度 (10/30, 20/60, 25/75, 50/150, 50/200)
  (2) ストップロス・利確効果
  (3) 銘柄ポートフォリオ vs 単独
  (4) 年別 Sharpe (安定性)
  (5) コスト感度
  (6) MA cross + RSI フィルタの相乗効果
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


def ma_cross_trades(daily, code, fast=25, slow=75, cost=0.0008,
                     stop_loss=None, take_profit=None, rsi_filter=None):
    """
    MA fast/slow GC でロングエントリ・DCでクローズ。
    stop_loss: 例 0.05 → -5%でストップ (intraday low ベース)
    take_profit: 例 0.10 → +10%で利確
    rsi_filter: 例 (50, 70) → RSI < 50 のときのみエントリ (過熱フィルタ)
    """
    ma_f = daily["close"].rolling(fast).mean()
    ma_s = daily["close"].rolling(slow).mean()
    signal_on = ma_f > ma_s
    rsi_v = rsi(daily["close"]) if rsi_filter else None

    trades = []
    in_pos = False
    entry_date = None
    entry_px = None
    for i in range(1, len(daily)):
        d = daily.index[i]
        prev_d = daily.index[i-1]
        if pd.isna(ma_s.iloc[i-1]):
            continue
        # シグナル遷移
        was_on = signal_on.iloc[i-1]
        is_on = signal_on.iloc[i]

        if not in_pos and not was_on and is_on:
            # 翌寄付エントリ
            if i + 1 >= len(daily):
                continue
            # RSIフィルタ
            if rsi_filter and rsi_v is not None:
                lo, hi = rsi_filter
                if rsi_v.iloc[i] >= hi or rsi_v.iloc[i] < lo:
                    continue  # 過熱 or 過売りすぎはスキップ
            entry_date = daily.index[i+1]
            entry_px = daily["open"].iloc[i+1]
            in_pos = True
            high_water = entry_px
            continue

        if in_pos:
            cl = daily["close"].iloc[i]
            hi = daily["high"].iloc[i]
            lo = daily["low"].iloc[i]
            # SL/TP は当日中で判定
            exit_reason = None
            exit_px = None
            if stop_loss is not None:
                stop_px = entry_px * (1 - stop_loss)
                if lo <= stop_px:
                    exit_px = stop_px
                    exit_reason = "SL"
            if exit_reason is None and take_profit is not None:
                tp_px = entry_px * (1 + take_profit)
                if hi >= tp_px:
                    exit_px = tp_px
                    exit_reason = "TP"
            if exit_reason is None and was_on and not is_on:
                # 翌寄付クローズ
                if i + 1 < len(daily):
                    exit_date = daily.index[i+1]
                    exit_px = daily["open"].iloc[i+1]
                    exit_reason = "DC"
            if exit_px is not None:
                if exit_reason in ("SL", "TP"):
                    exit_date = d
                r = np.log(exit_px / entry_px)
                pnl = r - cost * 2
                trades.append({
                    "code": code,
                    "entry_date": entry_date, "exit_date": exit_date,
                    "entry": entry_px, "exit": exit_px,
                    "hold_days": (exit_date - entry_date).days,
                    "reason": exit_reason,
                    "ret_pct": r * 100, "pnl": pnl,
                })
                in_pos = False
                entry_date = entry_px = None

    # 残ポジ
    if in_pos and entry_px is not None:
        exit_date = daily.index[-1]
        exit_px = daily["close"].iloc[-1]
        r = np.log(exit_px / entry_px)
        pnl = r - cost * 2
        trades.append({
            "code": code, "entry_date": entry_date, "exit_date": exit_date,
            "entry": entry_px, "exit": exit_px,
            "hold_days": (exit_date - entry_date).days,
            "reason": "EOD",
            "ret_pct": r * 100, "pnl": pnl,
        })

    return pd.DataFrame(trades)


def metrics(t, ann=245):
    if len(t) < 5:
        return {}
    t = t.copy()
    t["exit_date"] = pd.to_datetime(t["exit_date"])
    daily = t.groupby("exit_date")["pnl"].sum()
    n = len(daily)
    mu, sd = daily.mean(), daily.std()
    sharpe = mu / sd * np.sqrt(ann) if sd > 0 else 0
    eq = daily.cumsum()
    dd = (eq - eq.cummax()).min()
    wr = (t["pnl"] > 0).mean() * 100
    pf = (t["pnl"][t["pnl"]>0].sum() / -t["pnl"][t["pnl"]<0].sum()) if (t["pnl"]<0).any() else np.inf
    avg_hold = t["hold_days"].mean() if "hold_days" in t.columns else 0
    return {"N": len(t), "Sharpe": round(sharpe,2), "WR%": round(wr,1),
            "PF": round(pf,2), "Cum%": round(eq.iloc[-1]*100,1),
            "DD%": round(dd*100,1), "avg_hold_d": round(avg_hold,1)}


def main():
    print("=== MA cross 戦略 深堀り ===\n")
    dailies = {c: fetch_daily(c) for c in SYMBOLS}

    # ============ (1) MAパラメータ感度 ============
    print("--- (1) MAパラメータ感度 (8bps, 全7銘柄プール) ---")
    print(f"{'fast/slow':<12} {'avg_Sharpe':>11} {'avg_Cum%':>10} {'n_winners':>10}")
    param_rows = []
    for fast, slow in [(5,20), (10,30), (10,50), (20,60), (25,75), (50,150), (50,200), (75,200)]:
        sharpes = []
        cums = []
        for c in SYMBOLS:
            t = ma_cross_trades(dailies[c], c, fast=fast, slow=slow)
            m = metrics(t)
            if m:
                sharpes.append(m["Sharpe"])
                cums.append(m["Cum%"])
        avg_sh = np.mean(sharpes) if sharpes else 0
        avg_cum = np.mean(cums) if cums else 0
        n_win = sum(1 for s in sharpes if s > 0)
        param_rows.append({"fast": fast, "slow": slow, "avg_Sharpe": round(avg_sh,2),
                           "avg_Cum%": round(avg_cum,1), "n_winners": f"{n_win}/{len(sharpes)}"})
        print(f"{fast:>3}/{slow:<7} {avg_sh:>+11.2f} {avg_cum:>+10.1f} {n_win:>5}/{len(sharpes)}")

    # ============ (2) ストップロス・利確 ============
    print("\n--- (2) SL/TP 効果 (fast=25, slow=75, 全7銘柄平均) ---")
    print(f"{'SL':<7} {'TP':<7} {'avg_Sharpe':>11} {'avg_Cum%':>10} {'avg_hold_d':>10}")
    for sl, tp in [(None, None), (0.05, None), (0.10, None), (None, 0.20),
                    (0.05, 0.20), (0.10, 0.30), (0.05, 0.10)]:
        sharpes, cums, holds = [], [], []
        for c in SYMBOLS:
            t = ma_cross_trades(dailies[c], c, fast=25, slow=75,
                                 stop_loss=sl, take_profit=tp)
            m = metrics(t)
            if m:
                sharpes.append(m["Sharpe"])
                cums.append(m["Cum%"])
                holds.append(m["avg_hold_d"])
        sl_lbl = "なし" if sl is None else f"{sl*100:.0f}%"
        tp_lbl = "なし" if tp is None else f"{tp*100:.0f}%"
        print(f"{sl_lbl:<7} {tp_lbl:<7} {np.mean(sharpes):>+11.2f} "
              f"{np.mean(cums):>+10.1f} {np.mean(holds):>10.0f}")

    # ============ (3) ポートフォリオ運用 (等ウェイト) ============
    print("\n--- (3) ポートフォリオ (Top-K 銘柄, fast=25, slow=75, 8bps) ---")
    # 全銘柄の trades 計算
    all_trades = {}
    for c in SYMBOLS:
        all_trades[c] = ma_cross_trades(dailies[c], c, fast=25, slow=75)

    # 銘柄別Sharpeで並べ
    ranked = sorted([(c, metrics(all_trades[c])["Sharpe"]) for c in SYMBOLS],
                     key=lambda x: -x[1])
    print(f"  銘柄ランキング: {[(SYMBOLS[c], round(s,2)) for c, s in ranked]}")

    # 等ウェイトポートフォリオ: 各日 各銘柄の保有/非保有を表すPnL系列を作り、平均
    # 簡易: trades の exit_date でPnLを置き、銘柄数で割って合算
    print(f"\n  {'K':<3} {'銘柄':<30} {'Sharpe':>7} {'Cum%':>8} {'DD%':>7}")
    for K in [3, 5, 7]:
        top_codes = [c for c, _ in ranked[:K]]
        # 日次PnL構築: 全銘柄のtrade.pnlをexit_dateに置く
        all_daily_list = []
        for c in top_codes:
            t = all_trades[c].copy()
            if len(t) == 0:
                continue
            t["exit_date"] = pd.to_datetime(t["exit_date"])
            sub = t.groupby("exit_date")["pnl"].sum() / K  # 等ウェイト = 各銘柄に 1/K
            all_daily_list.append(sub)
        port_daily = pd.concat(all_daily_list).groupby(level=0).sum()
        mu, sd = port_daily.mean(), port_daily.std()
        sharpe = mu / sd * np.sqrt(245) if sd > 0 else 0
        eq = port_daily.cumsum()
        dd = (eq - eq.cummax()).min()
        names_str = ", ".join([SYMBOLS[c] for c in top_codes])[:30]
        print(f"  {K:<3} {names_str:<30} {sharpe:>+7.2f} {eq.iloc[-1]*100:>+8.1f} {dd*100:>+7.1f}")

    # ============ (4) 年別 Sharpe (Top5 ポートフォリオ) ============
    print(f"\n--- (4) Top5 ポートフォリオ 年別 Sharpe ---")
    top5_codes = [c for c, _ in ranked[:5]]
    daily_list = []
    for c in top5_codes:
        t = all_trades[c].copy()
        if len(t) == 0:
            continue
        t["exit_date"] = pd.to_datetime(t["exit_date"])
        sub = t.groupby("exit_date")["pnl"].sum() / 5
        daily_list.append(sub)
    port_daily = pd.concat(daily_list).groupby(level=0).sum().sort_index()
    for year in range(2016, 2027):
        sub = port_daily[port_daily.index.year == year]
        if len(sub) < 2:
            continue
        mu, sd = sub.mean(), sub.std()
        sharpe = mu / sd * np.sqrt(245) if sd > 0 else 0
        n = len(sub)
        cum = sub.sum() * 100
        print(f"  {year}: N={n:3d}, Sharpe={sharpe:+.2f}, Cum={cum:+.1f}%")

    # ============ (5) コスト感度 (Top5) ============
    print(f"\n--- (5) コスト感度 (Top5 ポートフォリオ) ---")
    for c_bps in [0, 5, 8, 10, 15, 20, 30, 50]:
        cost = c_bps / 10000
        d_list = []
        for c in top5_codes:
            t = ma_cross_trades(dailies[c], c, fast=25, slow=75, cost=cost)
            if len(t) == 0:
                continue
            t["exit_date"] = pd.to_datetime(t["exit_date"])
            sub = t.groupby("exit_date")["pnl"].sum() / 5
            d_list.append(sub)
        port = pd.concat(d_list).groupby(level=0).sum() if d_list else pd.Series(dtype=float)
        if len(port) > 0:
            mu, sd = port.mean(), port.std()
            sharpe = mu / sd * np.sqrt(245) if sd > 0 else 0
            cum = port.cumsum().iloc[-1] * 100
            print(f"  cost={c_bps:3d} bps: Sharpe={sharpe:+.2f}, Cum={cum:+.1f}%")

    # ============ (6) IS Sharpe基準で銘柄選択 → OOSで評価 ============
    print(f"\n--- (6) IS Sharpe基準で選択 → OOS評価 ---")
    is_sharpes = {}
    for c in SYMBOLS:
        t = all_trades[c].copy()
        t["exit_date"] = pd.to_datetime(t["exit_date"])
        t_is = t[t["exit_date"].dt.year <= 2022]
        m = metrics(t_is) if len(t_is) >= 5 else {}
        is_sharpes[c] = m.get("Sharpe", -99)
    is_ranked = sorted(SYMBOLS.keys(), key=lambda c: -is_sharpes[c])

    for K in [3, 5, 7]:
        sel = is_ranked[:K]
        oos_list = []
        for c in sel:
            t = all_trades[c].copy()
            t["exit_date"] = pd.to_datetime(t["exit_date"])
            t_oos = t[t["exit_date"].dt.year >= 2023]
            sub = t_oos.groupby("exit_date")["pnl"].sum() / K
            oos_list.append(sub)
        port = pd.concat(oos_list).groupby(level=0).sum()
        mu, sd = port.mean(), port.std()
        sharpe = mu / sd * np.sqrt(245) if sd > 0 else 0
        cum = port.cumsum().iloc[-1] * 100
        names_str = ", ".join([SYMBOLS[c] for c in sel])
        print(f"  IS-Top{K}: [{names_str}]")
        print(f"           OOS Sharpe={sharpe:+.2f}, Cum={cum:+.1f}%")

    # 保存
    out_dir = os.path.dirname(os.path.abspath(__file__))
    pd.DataFrame(param_rows).to_csv(os.path.join(out_dir, "ma_param_sweep.csv"), index=False)
    port_daily.to_csv(os.path.join(out_dir, "ma_top5_daily_pnl.csv"))
    for c in SYMBOLS:
        all_trades[c].to_csv(os.path.join(out_dir, f"trades_MA2575_{c}.csv"), index=False)
    print(f"\n保存: ma_param_sweep.csv, ma_top5_daily_pnl.csv, trades_MA2575_*.csv")


if __name__ == "__main__":
    main()
