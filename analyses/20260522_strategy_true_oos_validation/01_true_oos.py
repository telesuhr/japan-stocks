"""
真のOOS検証 — 選択バイアスを排除した戦略昇格判断

設計:
  IS期間  (2016-01-01 〜 2020-12-31, 約5年):
    - 21銘柄 × 2戦略 (MA25/75, RSI<30) を全部計算
    - 各銘柄ごとに「IS で良いほうの戦略」を選択
    - Sharpe ≥ 2.0 の組合せを「採用」と決定
    - パラメータは固定 (MA 25/75, RSI 30/50) — 過剰最適化回避

  OOS期間 (2021-01-01 〜 2026-05-22, 約5.5年):
    - IS で決定した銘柄+戦略をそのまま適用
    - 何も再選別しない
    - 評価指標を測定

  判定:
    - OOS Sharpe ≥ 2.0 を維持しているか (strategies/ 昇格基準)
    - IS と OOS の Sharpe 差分が許容範囲か
    - DD・取引数・勝率が劣化していないか
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
    "57060": ("三井金属", "非鉄金属"), "57110": ("三菱マテリアル", "非鉄金属"),
    "57130": ("住友金属鉱山", "非鉄金属"), "57140": ("DOWA HD", "非鉄金属"),
    "58010": ("古河電工", "非鉄金属"), "58020": ("住友電工", "非鉄金属"),
    "58030": ("フジクラ", "非鉄金属"),
    "83060": ("三菱UFJ", "銀行業"), "83160": ("三井住友", "銀行業"),
    "84110": ("みずほ", "銀行業"),
    "68570": ("アドバンテスト", "半導体"), "69200": ("レーザーテック", "半導体"),
    "80350": ("東京エレクトロン", "半導体"),
    "80010": ("伊藤忠商事", "商社"), "80310": ("三井物産", "商社"),
    "80580": ("三菱商事", "商社"),
    "33820": ("セブン&アイ", "小売業"), "99830": ("ファストリテ", "小売業"),
    "72030": ("トヨタ自動車", "自動車"), "72670": ("ホンダ", "自動車"),
    "79740": ("任天堂", "その他製品"),
}

IS_START = "2016-01-01"
IS_END = "2020-12-31"
OOS_START = "2021-01-01"
OOS_END = "2026-05-22"
COST = 0.0008
IS_SHARPE_THR = 2.0  # 採用基準


def fetch_daily(code, start, end):
    conn = psycopg2.connect(**PG_CONFIG)
    sql = "SELECT date, open, high, low, close FROM stocks_daily WHERE code=%s AND date BETWEEN %s AND %s ORDER BY date"
    df = pd.read_sql(sql, conn, params=(code, start, end))
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


def simulate_pos(daily, code, position_series, direction=+1, cost=COST):
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
            r = np.log(xp / ep) * direction
            pnl = r - cost * 2
            trades.append({"code": code, "entry_date": ed, "exit_date": xd,
                           "entry": ep, "exit": xp, "ret_pct": r*100, "pnl": pnl,
                           "hold_days": (xd-ed).days})
    return pd.DataFrame(trades)


def strat_ma(daily, code, fast=25, slow=75):
    ma_f = daily["close"].rolling(fast).mean()
    ma_s = daily["close"].rolling(slow).mean()
    pos = (ma_f > ma_s).astype(int)
    return simulate_pos(daily, code, pos, direction=+1)


def strat_rsi(daily, code, lo=30, hi=50):
    r = rsi(daily["close"])
    pos = pd.Series(0, index=daily.index)
    state = 0
    for d in daily.index:
        if pd.isna(r.loc[d]):
            continue
        rv = r.loc[d]
        if state == 0 and rv < lo:
            state = 1
        elif state == 1 and rv > hi:
            state = 0
        pos.loc[d] = state
    return simulate_pos(daily, code, pos, direction=+1)


def metrics(t, ann=245):
    if len(t) < 5:
        return {"N": len(t), "Sharpe": 0, "Cum%": 0, "DD%": 0, "WR%": 0, "PF": 0}
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


def filter_by_dates(trades_df, start, end):
    """trades の exit_date が [start, end] にあるものだけ抽出"""
    if len(trades_df) == 0:
        return trades_df
    t = trades_df.copy()
    t["exit_date"] = pd.to_datetime(t["exit_date"])
    return t[(t["exit_date"] >= start) & (t["exit_date"] <= end)]


def main():
    print("=" * 70)
    print("真のOOS検証 - 選択バイアス排除版")
    print(f"  IS期間:  {IS_START} 〜 {IS_END}")
    print(f"  OOS期間: {OOS_START} 〜 {OOS_END}")
    print(f"  採用基準: IS Sharpe ≥ {IS_SHARPE_THR}")
    print("=" * 70)
    print()

    # ------ 全期間で trades を1回計算しておく (シグナルロジック自体は時系列リーク無し) ------
    all_trades = {"MA": {}, "RSI": {}}
    for code, (name, sect) in UNIVERSE.items():
        d = fetch_daily(code, IS_START, OOS_END)
        all_trades["MA"][code] = strat_ma(d, code)
        all_trades["RSI"][code] = strat_rsi(d, code)

    # ============ STEP 1: IS期間で銘柄+戦略を選別 ============
    print("--- STEP 1: IS期間 (2016-2020) で銘柄×戦略を選別 ---")
    print(f"{'銘柄':<14} {'sector':<10} {'MA_IS_Sh':>9} {'RSI_IS_Sh':>10} {'最良戦略':>8} {'採用':>5}")
    selection = []
    for code, (name, sect) in UNIVERSE.items():
        t_ma_is = filter_by_dates(all_trades["MA"][code], IS_START, IS_END)
        t_rsi_is = filter_by_dates(all_trades["RSI"][code], IS_START, IS_END)
        m_ma = metrics(t_ma_is)
        m_rsi = metrics(t_rsi_is)
        ma_sh = m_ma["Sharpe"]
        rsi_sh = m_rsi["Sharpe"]
        best_strat = "MA" if ma_sh > rsi_sh else "RSI"
        best_sh = max(ma_sh, rsi_sh)
        adopted = "✓" if best_sh >= IS_SHARPE_THR else "✗"
        selection.append({
            "code": code, "name": name, "sector": sect,
            "MA_IS_Sharpe": ma_sh, "MA_IS_N": m_ma["N"],
            "RSI_IS_Sharpe": rsi_sh, "RSI_IS_N": m_rsi["N"],
            "best_strat": best_strat, "best_IS_Sharpe": best_sh,
            "adopted": adopted == "✓",
        })
        print(f"{name:<14} {sect:<10} {ma_sh:>+9.2f} {rsi_sh:>+10.2f} {best_strat:>8} {adopted:>5}")

    df_sel = pd.DataFrame(selection)
    n_adopted = df_sel["adopted"].sum()
    print(f"\n  採用銘柄: {n_adopted} / {len(df_sel)} (IS Sharpe ≥ {IS_SHARPE_THR})")

    adopted_df = df_sel[df_sel["adopted"]].copy()
    print(f"\n  採用一覧 (IS Sharpe降順):")
    for _, r in adopted_df.sort_values("best_IS_Sharpe", ascending=False).iterrows():
        print(f"    {r['name']:<14} ({r['sector']:<8}) → {r['best_strat']}, IS Sharpe {r['best_IS_Sharpe']:+.2f}")

    # ============ STEP 2: OOS期間で評価 ============
    print(f"\n--- STEP 2: OOS期間 (2021-2026) でそのまま評価 ---")
    print(f"{'銘柄':<14} {'IS_Sh':>7} {'OOS_Sh':>8} {'OOS_N':>6} {'OOS_Cum%':>9} {'OOS_DD%':>8} {'判定':>4}")
    oos_results = []
    for _, r in adopted_df.iterrows():
        code = r["code"]
        strat = r["best_strat"]
        t_oos = filter_by_dates(all_trades[strat][code], OOS_START, OOS_END)
        m_oos = metrics(t_oos)
        verdict = "OK" if m_oos["Sharpe"] >= IS_SHARPE_THR else (
            "△" if m_oos["Sharpe"] > 0 else "✗")
        oos_results.append({
            "code": code, "name": r["name"], "sector": r["sector"],
            "strat": strat,
            "IS_Sharpe": r["best_IS_Sharpe"],
            "OOS_N": m_oos["N"],
            "OOS_Sharpe": m_oos["Sharpe"],
            "OOS_Cum%": m_oos["Cum%"],
            "OOS_DD%": m_oos["DD%"],
            "OOS_WR%": m_oos["WR%"],
            "verdict": verdict,
        })
        print(f"{r['name']:<14} {r['best_IS_Sharpe']:>+7.2f} {m_oos['Sharpe']:>+8.2f} "
              f"{m_oos['N']:>6} {m_oos['Cum%']:>+9.1f} {m_oos['DD%']:>+8.1f} {verdict:>4}")

    df_oos = pd.DataFrame(oos_results)

    # 集計
    print(f"\n--- OOS結果 集計 ---")
    n_ok = (df_oos["verdict"] == "OK").sum()
    n_pos = (df_oos["OOS_Sharpe"] > 0).sum()
    n_neg = (df_oos["OOS_Sharpe"] < 0).sum()
    print(f"  OOS Sharpe ≥ {IS_SHARPE_THR} (昇格基準クリア): {n_ok} / {len(df_oos)}")
    print(f"  OOS Sharpe > 0  (ともかくプラス):              {n_pos} / {len(df_oos)}")
    print(f"  OOS Sharpe < 0 (機能停止):                    {n_neg} / {len(df_oos)}")
    print(f"  IS Sharpe平均 {df_oos['IS_Sharpe'].mean():+.2f} → OOS Sharpe平均 {df_oos['OOS_Sharpe'].mean():+.2f}")
    print(f"  劣化銘柄 (OOS<IS): {(df_oos['OOS_Sharpe']<df_oos['IS_Sharpe']).sum()} / {len(df_oos)}")

    # ============ STEP 3: ポートフォリオ運用 (OOS) ============
    print(f"\n--- STEP 3: 採用銘柄 等ウェイト ポートフォリオ OOS実績 ---")
    daily_list = []
    for _, r in df_oos.iterrows():
        code = r["code"]
        strat = r["strat"]
        t_oos = filter_by_dates(all_trades[strat][code], OOS_START, OOS_END)
        if len(t_oos) == 0:
            continue
        t_oos = t_oos.copy()
        t_oos["exit_date"] = pd.to_datetime(t_oos["exit_date"])
        sub = t_oos.groupby("exit_date")["pnl"].sum() / len(df_oos)
        daily_list.append(sub)
    port = pd.concat(daily_list).groupby(level=0).sum().sort_index()
    mu, sd = port.mean(), port.std()
    sharpe = mu / sd * np.sqrt(245) if sd > 0 else 0
    eq = port.cumsum()
    dd = (eq - eq.cummax()).min()
    wr = (port > 0).mean() * 100
    print(f"  N_days={len(port)}, Sharpe={sharpe:+.2f}, Cum={eq.iloc[-1]*100:+.1f}%, "
          f"DD={dd*100:.1f}%, WR={wr:.1f}%")

    # OOS年別
    print(f"\n--- OOS ポートフォリオ 年別 ---")
    for y in range(2021, 2027):
        sub = port[port.index.year == y]
        if len(sub) < 5:
            continue
        mu_y, sd_y = sub.mean(), sub.std()
        sh_y = mu_y / sd_y * np.sqrt(245) if sd_y > 0 else 0
        cum_y = sub.sum() * 100
        print(f"  {y}: N={len(sub)}, Sharpe={sh_y:+.2f}, Cum={cum_y:+.1f}%")

    # ============ STEP 4: 各戦略カテゴリの集計 ============
    print(f"\n--- STEP 4: 戦略カテゴリ別 OOS実績 ---")
    for strat in ["MA", "RSI"]:
        sub = df_oos[df_oos["strat"] == strat]
        if len(sub) == 0:
            continue
        print(f"\n  {strat} 採用銘柄 ({len(sub)}): "
              f"IS平均 {sub['IS_Sharpe'].mean():+.2f}, OOS平均 {sub['OOS_Sharpe'].mean():+.2f}")
        for _, r in sub.sort_values("OOS_Sharpe", ascending=False).iterrows():
            print(f"    {r['name']:<14} ({r['sector']:<8}): IS {r['IS_Sharpe']:+.2f} → OOS {r['OOS_Sharpe']:+.2f}")

        # 各カテゴリのポートフォリオSharpe
        cat_daily = []
        for _, r in sub.iterrows():
            t_oos = filter_by_dates(all_trades[r["strat"]][r["code"]], OOS_START, OOS_END)
            if len(t_oos) == 0:
                continue
            t_oos = t_oos.copy()
            t_oos["exit_date"] = pd.to_datetime(t_oos["exit_date"])
            s = t_oos.groupby("exit_date")["pnl"].sum() / len(sub)
            cat_daily.append(s)
        if cat_daily:
            cat_port = pd.concat(cat_daily).groupby(level=0).sum()
            mu_c, sd_c = cat_port.mean(), cat_port.std()
            sh_c = mu_c / sd_c * np.sqrt(245) if sd_c > 0 else 0
            cum_c = cat_port.cumsum().iloc[-1] * 100
            dd_c = (cat_port.cumsum() - cat_port.cumsum().cummax()).min() * 100
            print(f"    {strat}-only ポートフォリオ: Sharpe={sh_c:+.2f}, Cum={cum_c:+.1f}%, DD={dd_c:.1f}%")

    # 保存
    out_dir = os.path.dirname(os.path.abspath(__file__))
    df_sel.to_csv(os.path.join(out_dir, "is_selection.csv"), index=False)
    df_oos.to_csv(os.path.join(out_dir, "oos_results.csv"), index=False)
    port.to_csv(os.path.join(out_dir, "oos_portfolio_daily.csv"))
    print(f"\n保存: is_selection.csv, oos_results.csv, oos_portfolio_daily.csv")


if __name__ == "__main__":
    main()
