"""
テクニカル指標戦略 10年スクリーニング (日足ベース)

検証する戦略:
  S1: RSI(14) 過売り反発ロング     - RSI<30で買い, RSI>50でクローズ
  S2: RSI(14) 過買いショート       - RSI>70で売り, RSI<50でクローズ
  S3: MACD(12,26,9) ゴールデン順張り - GC翌日寄付ロング, DC翌日寄付クローズ
  S4: MA25/75 ゴールデン順張り      - 25>75 でロング, 25<75 でクローズ
  S5: ボリンジャーバンド(20,2σ)逆張り - close<lowerで買い, close>middleでクローズ
  S6: モメンタム(ROC20) 順張りLS    - 20日リターン上位ロング/下位ショート (XS)

執行モデル: シグナル発生 → 翌日寄付エントリ → エグジット条件達成翌日寄付クローズ
コスト: 8 bps 片道
対象: 非鉄7銘柄 2016-05-10 〜 2026-05-22

評価:
  - 銘柄別 Sharpe / 累積 / 勝率 / PF
  - 全期間 vs IS(2016-2022) vs OOS(2023-2026)
  - Buy&Hold ベンチマーク超過リターン
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


def fetch_daily(code):
    conn = psycopg2.connect(**PG_CONFIG)
    sql = """
        SELECT date, open, high, low, close, adj_open, adj_close
        FROM stocks_daily WHERE code=%s AND date BETWEEN %s AND %s ORDER BY date
    """
    df = pd.read_sql(sql, conn, params=(code, START, END))
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")


# ===================== Indicators =====================

def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    dn = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/n, adjust=False).mean()
    ma_dn = dn.ewm(alpha=1/n, adjust=False).mean()
    rs = ma_up / ma_dn
    return 100 - 100 / (1 + rs)


def macd(close: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    sig_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, sig_line


def bollinger(close: pd.Series, n=20, k=2):
    ma = close.rolling(n).mean()
    sd = close.rolling(n).std()
    return ma - k*sd, ma, ma + k*sd


# ===================== Trade Simulation =====================

def simulate_signal_to_trades(daily: pd.DataFrame, code: str,
                                signal: pd.Series, direction: int = +1,
                                cost_oneway: float = COST_ONEWAY) -> pd.DataFrame:
    """
    signal: 1 = position on, 0 = position off
    direction: +1 ロング戦略, -1 ショート戦略
    エントリ: signal 0→1 翌日寄付
    エグジット: signal 1→0 翌日寄付

    結果: trade DataFrame (entry_date, exit_date, entry, exit, pnl, ret_pct)
    """
    op = daily["open"]
    pos = signal.fillna(0).astype(int)
    # 0→1 のtransition で entry, 1→0 で exit
    diff = pos.diff().fillna(0).astype(int)
    # entry: prev=0, now=1 → diff=+1
    entries = diff[diff == +1].index
    # exit: prev=1, now=0 → diff=-1
    exits = diff[diff == -1].index

    # entry の翌営業日寄付で建玉
    trades = []
    pending_entry = None
    for d in daily.index:
        if d in entries and pending_entry is None:
            pending_entry = d
            continue
        if d in exits and pending_entry is not None:
            # 翌寄付エントリ - 翌寄付エグジット
            # entryシグナル発生日の「翌営業日 open」でエントリ
            # exitシグナル発生日の「翌営業日 open」でクローズ
            # 簡略化: 当日エントリ・当日エグジット (シグナル生成は前日終値ベース、執行は当日寄付) で十分
            entry_idx = daily.index.get_loc(pending_entry) + 1
            exit_idx = daily.index.get_loc(d) + 1
            if entry_idx >= len(daily) or exit_idx >= len(daily):
                pending_entry = None
                continue
            entry_date = daily.index[entry_idx]
            exit_date = daily.index[exit_idx]
            entry_px = op.iloc[entry_idx]
            exit_px = op.iloc[exit_idx]
            r = np.log(exit_px / entry_px) * direction
            pnl = r - cost_oneway * 2
            trades.append({
                "code": code,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "entry": entry_px,
                "exit": exit_px,
                "hold_days": (exit_date - entry_date).days,
                "ret_pct": r * 100,
                "pnl": pnl,
            })
            pending_entry = None
    # 残ったポジを最終日でクローズ
    if pending_entry is not None:
        entry_idx = daily.index.get_loc(pending_entry) + 1
        if entry_idx < len(daily):
            entry_date = daily.index[entry_idx]
            entry_px = op.iloc[entry_idx]
            exit_date = daily.index[-1]
            exit_px = daily["close"].iloc[-1]
            r = np.log(exit_px / entry_px) * direction
            pnl = r - cost_oneway * 2
            trades.append({
                "code": code, "entry_date": entry_date, "exit_date": exit_date,
                "entry": entry_px, "exit": exit_px,
                "hold_days": (exit_date - entry_date).days,
                "ret_pct": r * 100, "pnl": pnl,
            })
    return pd.DataFrame(trades)


# ===================== Strategy Signal Generators =====================

def strat_S1_rsi_long(daily, code, rsi_lo=30, rsi_hi=50):
    r = rsi(daily["close"], 14)
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
    return simulate_signal_to_trades(daily, code, pos, direction=+1)


def strat_S2_rsi_short(daily, code, rsi_hi=70, rsi_lo=50):
    r = rsi(daily["close"], 14)
    pos = pd.Series(0, index=daily.index)
    state = 0
    for d in daily.index:
        if pd.isna(r.loc[d]):
            continue
        rv = r.loc[d]
        if state == 0 and rv > rsi_hi:
            state = 1
        elif state == 1 and rv < rsi_lo:
            state = 0
        pos.loc[d] = state
    return simulate_signal_to_trades(daily, code, pos, direction=-1)


def strat_S3_macd_long(daily, code):
    m, s = macd(daily["close"])
    pos = (m > s).astype(int)
    return simulate_signal_to_trades(daily, code, pos, direction=+1)


def strat_S4_ma_cross_long(daily, code, fast=25, slow=75):
    ma_f = daily["close"].rolling(fast).mean()
    ma_s = daily["close"].rolling(slow).mean()
    pos = (ma_f > ma_s).astype(int)
    return simulate_signal_to_trades(daily, code, pos, direction=+1)


def strat_S5_bb_long(daily, code, n=20, k=2):
    lo, mid, hi = bollinger(daily["close"], n, k)
    pos = pd.Series(0, index=daily.index)
    state = 0
    for d in daily.index:
        cl = daily.loc[d, "close"]
        if pd.isna(lo.loc[d]):
            continue
        if state == 0 and cl < lo.loc[d]:
            state = 1
        elif state == 1 and cl > mid.loc[d]:
            state = 0
        pos.loc[d] = state
    return simulate_signal_to_trades(daily, code, pos, direction=+1)


def strat_S6_xs_momentum(dailies: dict, lookback=20, hold=20, cost_oneway=COST_ONEWAY):
    """クロスセクション・モメンタムLS
    各日 過去 lookback 日リターン → top1 long, bottom1 short
    hold 日後リバランス (隣接ポジション)
    """
    # 全銘柄日次収益
    closes = pd.DataFrame({c: d["close"] for c, d in dailies.items()})
    opens = pd.DataFrame({c: d["open"] for c, d in dailies.items()})
    rets = np.log(closes / closes.shift(lookback))
    common = closes.dropna().index
    rets = rets.loc[common]

    trades = []
    i = lookback
    while i < len(common) - hold - 1:
        d_sig = common[i]
        r_sig = rets.loc[d_sig].dropna()
        if len(r_sig) < 3:
            i += hold
            continue
        long_code = r_sig.idxmax()
        short_code = r_sig.idxmin()
        # entry: 翌寄付
        entry_d = common[i + 1]
        exit_d = common[min(i + 1 + hold, len(common)-1)]
        op_long_in = opens.loc[entry_d, long_code]
        op_long_out = opens.loc[exit_d, long_code]
        op_short_in = opens.loc[entry_d, short_code]
        op_short_out = opens.loc[exit_d, short_code]
        r_long = np.log(op_long_out / op_long_in)
        r_short = np.log(op_short_out / op_short_in)
        pnl = (r_long - r_short) / 2 - cost_oneway * 2  # 2銘柄分往復
        trades.append({
            "entry_date": entry_d, "exit_date": exit_d,
            "long_code": long_code, "short_code": short_code,
            "ret_long_pct": r_long * 100, "ret_short_pct": r_short * 100,
            "pnl": pnl,
        })
        i += hold
    return pd.DataFrame(trades)


# ===================== Metrics =====================

def metrics_from_trades(trades_df: pd.DataFrame, date_col="exit_date", ann=245) -> dict:
    if len(trades_df) < 5:
        return {}
    t = trades_df.copy()
    t["d"] = pd.to_datetime(t[date_col])
    # 日次PnL: 同日複数件は合算 (1ポジ前提なら最大1件/日)
    daily = t.groupby("d")["pnl"].sum()
    n = len(daily)
    mu, sd = daily.mean(), daily.std()
    sharpe = mu / sd * np.sqrt(ann) if sd > 0 else 0
    eq = daily.cumsum()
    dd = (eq - eq.cummax()).min()
    wr = (t["pnl"] > 0).mean() * 100
    pf = (t["pnl"][t["pnl"] > 0].sum() /
          -t["pnl"][t["pnl"] < 0].sum()) if (t["pnl"] < 0).any() else np.inf
    avg_hold = t["hold_days"].mean() if "hold_days" in t.columns else (
        (t["exit_date"] - t["entry_date"]).dt.days.mean() if "entry_date" in t.columns else 0)
    return {
        "N_trades": len(t), "N_days_active": n,
        "Sharpe": round(sharpe, 2), "WR%": round(wr, 1),
        "PF": round(pf, 2),
        "Cum%": round(eq.iloc[-1] * 100, 1),
        "DD%": round(dd * 100, 1),
        "avg_hold_d": round(avg_hold, 1),
    }


# ===================== Buy&Hold Benchmark =====================

def buyhold_metrics(daily: pd.DataFrame, ann=245) -> dict:
    r = np.log(daily["close"] / daily["close"].shift(1)).dropna()
    n = len(r)
    mu, sd = r.mean(), r.std()
    sharpe = mu / sd * np.sqrt(ann) if sd > 0 else 0
    eq = r.cumsum()
    dd = (eq - eq.cummax()).min()
    return {
        "N_days": n,
        "Sharpe_BH": round(sharpe, 2),
        "Cum_BH%": round(eq.iloc[-1] * 100, 1),
        "DD_BH%": round(dd * 100, 1),
    }


# ===================== Main =====================

def main():
    print("=== テクニカル指標 10年スクリーニング ===\n")
    print(f"対象: {len(SYMBOLS)}銘柄, 期間: {START} 〜 {END}, コスト: {COST_ONEWAY*10000:.0f}bps片道\n")

    print("--- データ読み込み ---")
    dailies = {}
    for c in SYMBOLS:
        d = fetch_daily(c)
        dailies[c] = d
        print(f"  {c} {SYMBOLS[c]:10s}: {len(d)}日")

    # ============ Buy&Hold ベンチマーク ============
    print(f"\n--- Buy&Hold ベンチマーク (10年) ---")
    bh_rows = []
    for c, name in SYMBOLS.items():
        m = buyhold_metrics(dailies[c])
        bh_rows.append({"code": c, "name": name, **m})
        print(f"  {c} {name:10s}: BH Sharpe={m['Sharpe_BH']:+.2f}, "
              f"Cum={m['Cum_BH%']:+.1f}%, DD={m['DD_BH%']:.1f}%")
    df_bh = pd.DataFrame(bh_rows)
    bh_avg = df_bh["Sharpe_BH"].mean()
    print(f"  → BH 平均 Sharpe: {bh_avg:+.2f}")

    # ============ 戦略実行 ============
    strategies = [
        ("S1_RSI_long",  lambda d, c: strat_S1_rsi_long(d, c)),
        ("S2_RSI_short", lambda d, c: strat_S2_rsi_short(d, c)),
        ("S3_MACD_long", lambda d, c: strat_S3_macd_long(d, c)),
        ("S4_MA25_75",   lambda d, c: strat_S4_ma_cross_long(d, c)),
        ("S5_BB_long",   lambda d, c: strat_S5_bb_long(d, c)),
    ]

    print(f"\n--- 戦略 × 銘柄 マトリクス (全期間, 8bps) ---")
    print(f"{'戦略':<14} {'銘柄':<10} {'N':>5} {'Sh':>6} {'WR%':>5} {'PF':>5} {'Cum%':>8} {'DD%':>7} {'hold':>5}")
    rows = []
    trades_all = {}
    for strat_name, fn in strategies:
        for c, name in SYMBOLS.items():
            t = fn(dailies[c], c)
            trades_all[(strat_name, c)] = t
            m = metrics_from_trades(t)
            if m:
                rows.append({"strat": strat_name, "code": c, "name": name, **m})
                print(f"{strat_name:<14} {name:<10} {m['N_trades']:>5} "
                      f"{m['Sharpe']:>+6.2f} {m['WR%']:>5.1f} {m['PF']:>5.2f} "
                      f"{m['Cum%']:>+8.1f} {m['DD%']:>+7.1f} {m['avg_hold_d']:>5.0f}")

    df_all = pd.DataFrame(rows)

    # ============ 戦略別 集計 ============
    print(f"\n--- 戦略別 集計 (全銘柄平均 / Sharpe順) ---")
    strat_summary = df_all.groupby("strat").agg(
        avg_Sharpe=("Sharpe", "mean"),
        median_Sharpe=("Sharpe", "median"),
        n_winners=("Sharpe", lambda s: (s > 0).sum()),
        n_total=("Sharpe", "count"),
        avg_Cum=("Cum%", "mean"),
        avg_PF=("PF", "mean"),
    ).sort_values("avg_Sharpe", ascending=False)
    print(strat_summary.round(2).to_string())

    # ============ S6: XS Momentum LS ============
    print(f"\n--- S6: クロスセクション・モメンタムLS (lookback=20, hold=20) ---")
    t6 = strat_S6_xs_momentum(dailies, lookback=20, hold=20)
    if len(t6) > 0:
        t6_metrics = {}
        t6["d"] = t6["exit_date"]
        daily_pnl = t6.groupby("d")["pnl"].sum()
        n = len(daily_pnl)
        # Sharpe注: trade間隔20日なので daily ベースのann化は要調整
        # tradesの実行頻度を考慮
        trades_per_year = len(t6) / ((END_DATE := pd.to_datetime(END)) - pd.to_datetime(START)).days * 365
        mu_t = t6["pnl"].mean()
        sd_t = t6["pnl"].std()
        sharpe_t = mu_t / sd_t * np.sqrt(trades_per_year) if sd_t > 0 else 0
        eq = t6["pnl"].cumsum()
        dd = (eq - eq.cummax()).min()
        wr = (t6["pnl"] > 0).mean() * 100
        pf = (t6["pnl"][t6["pnl"]>0].sum() / -t6["pnl"][t6["pnl"]<0].sum()) if (t6["pnl"]<0).any() else np.inf
        print(f"  N_trades={len(t6)}, trades/年={trades_per_year:.1f}, "
              f"Sharpe={sharpe_t:+.2f}, WR={wr:.1f}%, PF={pf:.2f}, "
              f"Cum={eq.iloc[-1]*100:+.1f}%, DD={dd*100:.1f}%")

    # ============ IS / OOS ============
    print(f"\n--- IS (2016-2022) / OOS (2023-2026) 分割 ---")
    print(f"{'戦略':<14} {'銘柄':<10} {'IS Sharpe':>10} {'OOS Sharpe':>11} {'判定':>4}")
    isoos_rows = []
    for (strat_name, c), t in trades_all.items():
        if len(t) < 10:
            continue
        t["exit_date"] = pd.to_datetime(t["exit_date"])
        t_is = t[t["exit_date"].dt.year <= 2022]
        t_oos = t[t["exit_date"].dt.year >= 2023]
        m_is = metrics_from_trades(t_is) if len(t_is) >= 5 else {}
        m_oos = metrics_from_trades(t_oos) if len(t_oos) >= 5 else {}
        sh_is = m_is.get("Sharpe", 0)
        sh_oos = m_oos.get("Sharpe", 0)
        # 両方プラスのみ残す
        verdict = "✓" if sh_is > 0 and sh_oos > 0 else ""
        isoos_rows.append({"strat": strat_name, "code": c, "name": SYMBOLS[c],
                           "IS_Sharpe": sh_is, "OOS_Sharpe": sh_oos,
                           "IS_Cum%": m_is.get("Cum%", 0),
                           "OOS_Cum%": m_oos.get("Cum%", 0),
                           "verdict": verdict})
        if sh_is > 0 and sh_oos > 0:
            print(f"{strat_name:<14} {SYMBOLS[c]:<10} {sh_is:>+10.2f} {sh_oos:>+11.2f} {verdict:>4}")
    df_isoos = pd.DataFrame(isoos_rows)
    n_winners = (df_isoos["verdict"] == "✓").sum()
    print(f"\n  IS/OOS両方プラス: {n_winners}/{len(df_isoos)} 組合せ")

    # ============ BH対比 (戦略がBHに勝ったか) ============
    print(f"\n--- 各戦略 vs Buy&Hold (10年, Sharpe差) ---")
    bh_dict = {r["code"]: r["Sharpe_BH"] for _, r in df_bh.iterrows()}
    df_all["BH_Sharpe"] = df_all["code"].map(bh_dict)
    df_all["Excess_Sharpe"] = df_all["Sharpe"] - df_all["BH_Sharpe"]
    excess_by_strat = df_all.groupby("strat")["Excess_Sharpe"].mean().sort_values(ascending=False)
    print(excess_by_strat.round(2).to_string())

    # ============ 保存 ============
    out_dir = os.path.dirname(os.path.abspath(__file__))
    df_all.to_csv(os.path.join(out_dir, "results_matrix.csv"), index=False)
    df_bh.to_csv(os.path.join(out_dir, "buyhold_benchmark.csv"), index=False)
    df_isoos.to_csv(os.path.join(out_dir, "isoos_results.csv"), index=False)
    strat_summary.to_csv(os.path.join(out_dir, "strategy_summary.csv"))
    for (strat, c), t in trades_all.items():
        if len(t) > 0:
            t.to_csv(os.path.join(out_dir, f"trades_{strat}_{c}.csv"), index=False)
    print(f"\n保存: results_matrix.csv, buyhold_benchmark.csv, isoos_results.csv, ...")


if __name__ == "__main__":
    main()
