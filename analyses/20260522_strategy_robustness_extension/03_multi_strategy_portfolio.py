"""
検証3: マルチ戦略ポートフォリオ
MA25/75 (順張り) + RSI<30反発 (逆張り) を組み合わせた多様化ポートフォリオ

設計:
  (A) 各銘柄に「両戦略を並行運用」 → 同銘柄ダブルロング許容
  (B) 各銘柄に「最良戦略を選択」 (sector特性で順張り/逆張り)
  (C) 全銘柄 × 全戦略を一括ポートフォリオ (相関分散)

期待: 順張り (トレンド相場で勝つ) + 逆張り (レンジ相場で勝つ) = 相補的 → DD減・Sharpe維持
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

# 全21銘柄
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

START = "2016-05-10"
END = "2026-05-22"
COST = 0.0008


def load_trades(code, strat, search_dirs):
    """過去のスクリプトが保存した trades CSVを探す"""
    for d in search_dirs:
        path = os.path.join(d, f"trades_{strat}_{code}.csv")
        if os.path.exists(path):
            return pd.read_csv(path)
    return None


def metrics_from_daily(daily, ann=245):
    s = daily.dropna()
    if len(s) < 5:
        return {}
    mu, sd = s.mean(), s.std()
    sh = mu / sd * np.sqrt(ann) if sd > 0 else 0
    eq = s.cumsum()
    dd = (eq - eq.cummax()).min()
    return {"N_days": len(s), "Sharpe": round(sh,2),
            "Cum%": round(eq.iloc[-1]*100,1),
            "DD%": round(dd*100,1),
            "WR%": round((s>0).mean()*100, 1)}


def trades_to_daily(trades):
    if trades is None or len(trades) == 0:
        return pd.Series(dtype=float)
    t = trades.copy()
    t["exit_date"] = pd.to_datetime(t["exit_date"])
    return t.groupby("exit_date")["pnl"].sum()


def main():
    print("=== マルチ戦略ポートフォリオ検証 ===\n")
    # trades CSV を読む対象ディレクトリ
    expansion_dir = os.path.dirname(os.path.abspath(__file__))
    ma_winners_dir = expansion_dir  # 同じ場所

    # MA / RSI の trades を読む
    rows = []
    daily_pnl_dict = {"MA": {}, "RSI": {}}
    for code, (name, sector) in UNIVERSE.items():
        for strat in ["MA", "RSI"]:
            t = load_trades(code, strat, [expansion_dir])
            if t is None or len(t) == 0:
                continue
            daily_pnl_dict[strat][code] = trades_to_daily(t)
            m = metrics_from_daily(trades_to_daily(t))
            rows.append({"code": code, "name": name, "sector": sector,
                         "strat": strat, **m})
    df = pd.DataFrame(rows)

    # ============ (A) 全銘柄×両戦略 ポートフォリオ (等ウェイト) ============
    print("--- (A) 全21銘柄 × 両戦略 (42系列) 等ウェイト ポートフォリオ ---")
    series_list = []
    for strat in ["MA", "RSI"]:
        for c, s in daily_pnl_dict[strat].items():
            series_list.append(s / 42)  # 42系列等ウェイト
    port_all = pd.concat(series_list).groupby(level=0).sum().sort_index()
    m_all = metrics_from_daily(port_all)
    print(f"  N_days={m_all['N_days']}, Sharpe={m_all['Sharpe']:+.2f}, "
          f"Cum={m_all['Cum%']:+.1f}%, DD={m_all['DD%']:.1f}%, WR={m_all['WR%']:.1f}%")

    # ============ (B) 銘柄ごと最良戦略 + 等ウェイト ============
    print(f"\n--- (B) 銘柄ごと最良戦略を選択 → 等ウェイト ---")
    df_best = df.sort_values(["code", "Sharpe"], ascending=[True, False]).groupby("code").first()
    df_best = df_best.reset_index()
    df_best["strat_chosen"] = df_best["strat"]
    print(f"  戦略選択結果 (Sharpe降順):")
    for _, r in df_best.sort_values("Sharpe", ascending=False).iterrows():
        print(f"    {r['name']:<14} ({r['sector']:<8}): {r['strat_chosen']} Sh={r['Sharpe']:+.2f}")

    # 等ウェイトポートフォリオ (21系列)
    series_b = []
    K = len(df_best)
    for _, r in df_best.iterrows():
        s = daily_pnl_dict[r["strat_chosen"]].get(r["code"])
        if s is not None and len(s) > 0:
            series_b.append(s / K)
    port_best = pd.concat(series_b).groupby(level=0).sum().sort_index()
    m_b = metrics_from_daily(port_best)
    print(f"\n  ポートフォリオ: N_days={m_b['N_days']}, Sharpe={m_b['Sharpe']:+.2f}, "
          f"Cum={m_b['Cum%']:+.1f}%, DD={m_b['DD%']:.1f}%, WR={m_b['WR%']:.1f}%")

    # ============ (C) 上位選別: 各銘柄の最良戦略のうち、Sharpe > 2 だけ採用 ============
    print(f"\n--- (C) 最良戦略のうち Sharpe>2 だけ採用 ---")
    df_high = df_best[df_best["Sharpe"] > 2].copy()
    print(f"  採用銘柄数: {len(df_high)} ({len(df_best)}中)")
    series_c = []
    Kc = len(df_high)
    for _, r in df_high.iterrows():
        s = daily_pnl_dict[r["strat_chosen"]].get(r["code"])
        if s is not None and len(s) > 0:
            series_c.append(s / Kc)
    port_high = pd.concat(series_c).groupby(level=0).sum().sort_index()
    m_c = metrics_from_daily(port_high)
    print(f"  ポートフォリオ: N_days={m_c['N_days']}, Sharpe={m_c['Sharpe']:+.2f}, "
          f"Cum={m_c['Cum%']:+.1f}%, DD={m_c['DD%']:.1f}%, WR={m_c['WR%']:.1f}%")

    # ============ (D) MA-only / RSI-only との比較 ============
    print(f"\n--- (D) 単一戦略ポートフォリオとの比較 ---")
    for strat in ["MA", "RSI"]:
        series_d = [s / len(daily_pnl_dict[strat]) for s in daily_pnl_dict[strat].values()]
        port_d = pd.concat(series_d).groupby(level=0).sum().sort_index()
        m_d = metrics_from_daily(port_d)
        print(f"  {strat}-only (全21銘柄): Sharpe={m_d['Sharpe']:+.2f}, "
              f"Cum={m_d['Cum%']:+.1f}%, DD={m_d['DD%']:.1f}%")

    # ============ (E) MA-RSI 相関 ============
    print(f"\n--- (E) MA戦略 vs RSI戦略 日次PnL 相関 ---")
    ma_series = [s for s in daily_pnl_dict["MA"].values()]
    rsi_series = [s for s in daily_pnl_dict["RSI"].values()]
    if ma_series and rsi_series:
        ma_port = pd.concat([s/len(ma_series) for s in ma_series]).groupby(level=0).sum()
        rsi_port = pd.concat([s/len(rsi_series) for s in rsi_series]).groupby(level=0).sum()
        common = ma_port.index.intersection(rsi_port.index)
        if len(common) > 30:
            corr = ma_port.loc[common].corr(rsi_port.loc[common])
            print(f"  相関係数: {corr:+.3f}")
            print(f"  → {'低相関なので分散効果大' if abs(corr) < 0.3 else '相関あるので分散効果限定的'}")

    # ============ (F) 年別 (C: 上位選別) ============
    print(f"\n--- (F) 年別 (C) Sharpe>2選別ポートフォリオ ---")
    port_high.index = pd.to_datetime(port_high.index)
    for y in range(2016, 2027):
        sub = port_high[port_high.index.year == y]
        if len(sub) < 5:
            continue
        m = metrics_from_daily(sub)
        print(f"  {y}: N={m['N_days']}, Sharpe={m['Sharpe']:+.2f}, "
              f"Cum={m['Cum%']:+.1f}%, DD={m['DD%']:.1f}%")

    # 保存
    out_dir = os.path.dirname(os.path.abspath(__file__))
    df.to_csv(os.path.join(out_dir, "multi_strat_all.csv"), index=False)
    df_best.to_csv(os.path.join(out_dir, "multi_strat_best_per_symbol.csv"), index=False)
    port_all.to_csv(os.path.join(out_dir, "port_A_all_combined.csv"))
    port_best.to_csv(os.path.join(out_dir, "port_B_best_per_symbol.csv"))
    port_high.to_csv(os.path.join(out_dir, "port_C_high_quality.csv"))
    print(f"\n保存: multi_strat_*.csv, port_*.csv")


if __name__ == "__main__":
    main()
