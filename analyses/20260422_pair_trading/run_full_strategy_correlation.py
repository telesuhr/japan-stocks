#!/usr/bin/env python3
"""
全戦略統合相関分析 (拡張版)
新規戦略 nonferrous_lme_link (D_strong/D_freq) + B_dispersion + semi_sox_fade を追加
"""
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
NF = ROOT / "20260422_nonferrous_semi_intraday"

SOURCES = {
    # 既存4戦略
    "pair_portfolio": {
        "path": Path(__file__).parent / "portfolio_pnl.csv",
        "kind": "daily_bps"},
    "eneos_vwap_trend": {
        "path": ROOT / "20260422_eneos_vwap_oos/trades_full.csv",
        "kind": "trade_date_bps", "date_col": "date", "pnl_col": "pnl_bps"},
    "lme_on_copper": {
        "path": ROOT / "20260421_lme_copper_overnight/backtest_trades.csv",
        "kind": "trade_date_bps", "date_col": "date", "pnl_col": "pnl_bps"},
    "lasertec_ma25": {
        "path": ROOT / "20260422_lasertec_ma25/best_trades.csv",
        "kind": "entry_exit_pct", "date_col": "exit_date", "pnl_col": "ret_pct"},
    # 新規戦略 (nonferrous_combined_pnl.csv に D_strong, D_freq, B, Combined カラム)
    "nonferrous_D_strong": {
        "path": NF / "nonferrous_combined_pnl.csv",
        "kind": "daily_csv_col", "col": "D_strong"},
    "nonferrous_D_freq": {
        "path": NF / "nonferrous_combined_pnl.csv",
        "kind": "daily_csv_col", "col": "D_freq"},
    "nonferrous_B_disp": {
        "path": NF / "nonferrous_combined_pnl.csv",
        "kind": "daily_csv_col", "col": "B"},
    "semi_sox_fade": {
        "path": NF / "semi_sox_fade_daily_pnl.csv",
        "kind": "daily_csv_col", "col": "pnl_bps"},
}


def load_series(name, cfg):
    p = cfg["path"]
    if not p.exists():
        print(f"  ✗ {name}: {p} not found")
        return None
    kind = cfg["kind"]
    if kind == "daily_bps":
        s = pd.read_csv(p, index_col=0, parse_dates=True)
        return s.iloc[:, 0].rename(name)
    if kind == "trade_date_bps":
        df = pd.read_csv(p, parse_dates=[cfg["date_col"]])
        s = df.groupby(cfg["date_col"])[cfg["pnl_col"]].sum()
        s.name = name
        return s
    if kind == "entry_exit_pct":
        df = pd.read_csv(p, parse_dates=[cfg["date_col"]])
        s = df.groupby(cfg["date_col"])[cfg["pnl_col"]].sum() * 100
        s.name = name
        return s
    if kind == "daily_csv_col":
        df = pd.read_csv(p, index_col=0, parse_dates=True)
        return df[cfg["col"]].rename(name)


def main():
    print("=" * 110)
    print("全戦略統合相関分析  (既存 4 + 新規 4 = 8 戦略)")
    print("=" * 110)
    series = {}
    for name, cfg in SOURCES.items():
        s = load_series(name, cfg)
        if s is None: continue
        s = s[s.notna()]
        active = s[s != 0]
        if not len(active): continue
        sh = s.mean()/s.std()*np.sqrt(252) if s.std() > 0 else 0
        print(f"  ✓ {name:<22} N={len(s):>4}  active={len(active):>4}  "
              f"{s.index.min().date()}〜{s.index.max().date()}  "
              f"mean {s.mean():+6.2f} bps  Sh {sh:+.2f}")
        series[name] = s

    df = pd.concat(series, axis=1).fillna(0.0)
    starts, ends = [], []
    for s in series.values():
        a = s[s != 0]
        if len(a):
            starts.append(a.index.min()); ends.append(a.index.max())
    cs = max(starts); ce = min(ends)
    print(f"\n共通期間: {cs.date()} 〜 {ce.date()}")
    df_c = df[(df.index >= cs) & (df.index <= ce)]

    print("\n" + "=" * 110)
    print("【1. 日次 PnL 相関行列 (共通期間、非取引日=0)】")
    print("=" * 110)
    corr = df_c.corr()
    print(corr.round(3).to_string())
    m = corr.values.copy(); np.fill_diagonal(m, np.nan)
    print(f"\n平均 |ρ| = {np.nanmean(np.abs(m)):.3f}")

    print("\n" + "=" * 110)
    print("【2. 両戦略が取引した日のみの相関】")
    print("=" * 110)
    names = list(series.keys())
    overlap = pd.DataFrame(np.nan, index=names, columns=names)
    for i, a in enumerate(names):
        for b in names[i:]:
            sa = df_c[a]; sb = df_c[b]
            mk = (sa != 0) & (sb != 0)
            if mk.sum() < 5: continue
            c = np.corrcoef(sa[mk], sb[mk])[0, 1]
            overlap.loc[a, b] = c; overlap.loc[b, a] = c
    print(overlap.round(3).to_string())

    print("\n" + "=" * 110)
    print("【3. 個別戦略パフォーマンス (共通期間)】")
    print("=" * 110)
    print(f"  {'戦略':<22} {'mean(bps)':>10} {'sd':>7} {'Sharpe':>8} {'sum':>9} {'MDD':>7}")
    for n in names:
        s = df_c[n]
        if s.std() == 0: continue
        sh = s.mean()/s.std()*np.sqrt(252)
        cum = s.cumsum()
        mdd = (cum.cummax() - cum).max()
        print(f"  {n:<22} {s.mean():>+10.2f} {s.std():>7.2f} {sh:>+8.2f} "
              f"{s.sum():>+9.0f} {mdd:>7.0f}")

    print("\n" + "=" * 110)
    print("【4. 統合ポートフォリオ各種】")
    print("=" * 110)

    def show(name, w):
        c = (df_c * pd.Series(w)).sum(axis=1)
        sh = c.mean()/c.std()*np.sqrt(252) if c.std() > 0 else 0
        cum = c.cumsum()
        mdd = (cum.cummax() - cum).max()
        t = c.mean()/(c.std()/np.sqrt(len(c))) if c.std() > 0 else 0
        print(f"  {name:<55} mean {c.mean():+6.2f}  Sh {sh:+5.2f}  t {t:+5.2f}  "
              f"sum {c.sum():+7.0f}  MDD {mdd:>5.0f}")

    # Equal weight 全戦略
    ew = {n: 1/len(names) for n in names}
    show("Equal-Weight 全 8 戦略", ew)

    # Sharpe 按分 (個別 Sharpe を重みに使用)
    sharpes = {}
    for n in names:
        s = df_c[n]
        sharpes[n] = max(s.mean()/s.std()*np.sqrt(252), 0) if s.std() > 0 else 0
    tot = sum(sharpes.values())
    if tot > 0:
        sw = {n: v/tot for n, v in sharpes.items()}
        show("Sharpe 按分 (全 8 戦略)", sw)

    # コア 4: pair / eneos / lme_on / nonferrous_D_strong
    core4 = ["pair_portfolio", "eneos_vwap_trend", "lme_on_copper", "nonferrous_D_strong"]
    if all(n in df_c.columns for n in core4):
        w = {n: 1/4 if n in core4 else 0 for n in names}
        show("コア 4 戦略 EW (pair/eneos/lme_on/D_strong)", w)

    # 推奨ポートフォリオ
    rec = {
        "lme_on_copper": 0.20,
        "nonferrous_D_strong": 0.20,
        "pair_portfolio": 0.15,
        "eneos_vwap_trend": 0.10,
        "nonferrous_D_freq": 0.10,
        "nonferrous_B_disp": 0.08,
        "semi_sox_fade": 0.10,
        "lasertec_ma25": 0.07,
    }
    rec = {n: rec.get(n, 0) for n in names}
    sum_r = sum(rec.values())
    rec = {n: v/sum_r for n, v in rec.items()}
    show("推奨配分 (リスク按分 × カテゴリ分散)", rec)

    print("\n" + "=" * 110)
    print("【5. クラスター解析 — 似た戦略をグルーピング】")
    print("=" * 110)
    # 簡易: 相関 > 0.3 のペアを表示
    print("相関 |ρ| > 0.20 のペア:")
    pairs_h = []
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            c = corr.iloc[i, j]
            if abs(c) > 0.20:
                pairs_h.append((names[i], names[j], c))
    if not pairs_h:
        print("  該当なし — 全戦略が低相関で独立性が高い")
    for a, b, c in sorted(pairs_h, key=lambda x: -abs(x[2])):
        print(f"  {a:<25} ↔ {b:<25} ρ = {c:+.3f}")

    # CSV保存
    df_c.to_csv("all_strategies_daily_pnl.csv")
    corr.to_csv("all_strategies_correlation.csv")
    print("\n→ all_strategies_daily_pnl.csv / all_strategies_correlation.csv 保存")


if __name__ == "__main__":
    main()
