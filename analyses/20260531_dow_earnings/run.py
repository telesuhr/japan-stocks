"""
曜日×決算タイミング交差検証
================================================================
仮説: 決算発表の曜日によってPEADの強さが異なるか。
  - 金曜引け後発表→月曜反応→火曜エントリー: 週末の不確実性・過大反応→大きいPEAD?
  - 月曜引け後発表→火曜反応→水曜エントリー: 週末ポジション調整なし→小さい反応?

評価:
  1. エントリー日の曜日別 PEAD Sharpe（既存pead_obs.csvを流用）
  2. 決算発表曜日（disc_date）の曜日別 car0（反応の大きさ）
  3. 最も有効な曜日フィルターがSharpe2.0に近づくか

注意: entry_date = react + 1 = disc_date + ~2営業日。
      entry_dateの曜日を使うのが最も実行可能性に近い。
"""
import sys; sys.stdout.reconfigure(line_buffering=True)
import os
from pathlib import Path
import numpy as np, pandas as pd

BASE_OBS = Path(__file__).parent.parent / "20260530_pead_price_reaction" / "pead_obs.csv"
OUT = Path(__file__).parent

COST_BPS = 20.0
N_Q = 10
OOS_START = "2024-01-01"
DOW_NAMES = {0: "月", 1: "火", 2: "水", 3: "木", 4: "金"}


def assign_q(df):
    df = df.dropna(subset=["car0"]).copy()
    parts = []
    for d, x in df.groupby("entry_date"):
        x = x.copy()
        if len(x) < N_Q * 2:
            x["q"] = np.nan
        else:
            x["q"] = pd.qcut(x["car0"].rank(method="first"), N_Q, labels=False).astype(float)
        parts.append(x)
    return pd.concat(parts).dropna(subset=["q"])


def ls_eval(df, label, hold=20):
    col = f"d{hold}"
    df = assign_q(df).dropna(subset=[col])
    if len(df) < 100:
        return None
    wk = df.groupby(["entry_date", "q"])[col].mean().unstack("q")
    wk.columns = [int(c) for c in wk.columns]
    if 0 not in wk.columns or N_Q - 1 not in wk.columns:
        return None
    ls = (wk[N_Q - 1] - wk[0]).dropna() / 1e4
    if len(ls) < 10:
        return None
    net = ls - COST_BPS / 1e4
    ann = net.mean() / net.std() * np.sqrt(245 / hold) if net.std() > 0 else np.nan
    t_stat = net.mean() / (net.std() / np.sqrt(len(net))) if net.std() > 0 else np.nan
    return {
        "label": label,
        "hold": hold,
        "n_obs": len(df),
        "n_days": len(ls),
        "gross_bps": round(ls.mean() * 1e4, 1),
        "net_bps": round(net.mean() * 1e4, 1),
        "win_rate": round((net > 0).mean(), 3),
        "sharpe": round(ann, 2),
        "t_stat": round(t_stat, 2),
    }


def main():
    print("[RUN] 曜日×決算エントリー交差検証")
    d = pd.read_csv(BASE_OBS)
    d["entry_date"] = pd.to_datetime(d["entry_date"])
    d["dow"] = d["entry_date"].dt.dayofweek
    d["dow_name"] = d["dow"].map(DOW_NAMES)
    print(f"  total obs={len(d):,}")

    rows = []

    # ─── 1. ベースライン（全曜日）
    for H in [5, 10, 20]:
        for split, sub in [("ALL", d), ("IS", d[d.entry_date < OOS_START]), ("OOS", d[d.entry_date >= OOS_START])]:
            r = ls_eval(sub, f"全曜日_{split}", H)
            if r:
                r["dow_filter"] = "全曜日"
                rows.append(r)

    # ─── 2. 曜日別
    for dow in range(5):
        name = DOW_NAMES[dow]
        sub_dow = d[d["dow"] == dow]
        for H in [5, 10, 20]:
            for split, sub in [("ALL", sub_dow), ("IS", sub_dow[sub_dow.entry_date < OOS_START]),
                                ("OOS", sub_dow[sub_dow.entry_date >= OOS_START])]:
                r = ls_eval(sub, f"{name}_{split}", H)
                if r:
                    r["dow_filter"] = name
                    rows.append(r)

    # ─── 3. 月曜除外 vs 月曜のみ
    for filter_name, mask in [
        ("月曜除外", d["dow"] != 0),
        ("月曜のみ", d["dow"] == 0),
        ("火水木", d["dow"].isin([1, 2, 3])),
        ("木金", d["dow"].isin([3, 4])),
    ]:
        sub_f = d[mask]
        for H in [5, 10, 20]:
            for split, sub in [("ALL", sub_f), ("IS", sub_f[sub_f.entry_date < OOS_START]),
                                ("OOS", sub_f[sub_f.entry_date >= OOS_START])]:
                r = ls_eval(sub, f"{filter_name}_{split}", H)
                if r:
                    r["dow_filter"] = filter_name
                    rows.append(r)

    results = pd.DataFrame(rows)
    results.to_csv(OUT / "results.csv", index=False)

    # ─── 表示: 20日保有のサマリー
    print("\n===== 曜日別 PEAD L/S Sharpe（20日保有, コスト20bps後） =====")
    summary = results[(results["hold"] == 20) & (results["label"].str.endswith("_ALL"))].copy()
    summary = summary[["dow_filter", "n_obs", "n_days", "gross_bps", "net_bps", "sharpe", "t_stat"]]
    print(summary.sort_values("sharpe", ascending=False).to_string(index=False))

    # ─── 曜日別 car0 の統計（反応の大きさ）
    print("\n===== エントリー曜日別 car0 統計（反応日のTOPIX超過） =====")
    car0_stats = d.groupby("dow_name")["car0"].agg(["count", "mean", "std"])
    car0_stats["mean%"] = (car0_stats["mean"] * 100).round(2)
    car0_stats["abs_mean%"] = (d.groupby("dow_name")["car0"].apply(lambda s: s.abs().mean()) * 100).round(2)
    print(car0_stats[["count", "mean%", "abs_mean%"]].to_string())

    # ─── IS/OOS比較（最良のフィルター）
    best_dow = summary.sort_values("sharpe", ascending=False).iloc[0]["dow_filter"]
    print(f"\n===== 最良フィルター「{best_dow}」の IS/OOS比較 =====")
    best = results[(results["dow_filter"] == best_dow) & (results["hold"] == 20)]
    print(best[["label", "n_obs", "n_days", "gross_bps", "net_bps", "sharpe", "t_stat"]].to_string(index=False))

    print("[DONE]")


if __name__ == "__main__":
    main()
