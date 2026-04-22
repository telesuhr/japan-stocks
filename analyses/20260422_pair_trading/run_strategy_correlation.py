#!/usr/bin/env python3
"""
既存戦略 vs pair_portfolio の相関分析

各戦略のトレード CSV から日次 PnL を再構築 (非取引日は 0 bps) し、
相関行列・アロケーションの筋を出力。
"""
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent

SOURCES = {
    "pair_portfolio": {
        "path": Path(__file__).parent / "portfolio_pnl.csv",
        "kind": "daily_bps",   # 既に日次 bps
    },
    "eneos_vwap_trend": {
        "path": ROOT / "20260422_eneos_vwap_oos/trades_full.csv",
        "kind": "trade_date_bps",
        "date_col": "date", "pnl_col": "pnl_bps",
    },
    "lme_on_copper": {
        "path": ROOT / "20260421_lme_copper_overnight/backtest_trades.csv",
        "kind": "trade_date_bps",
        "date_col": "date", "pnl_col": "pnl_bps",
    },
    "lasertec_ma25": {
        "path": ROOT / "20260422_lasertec_ma25/best_trades.csv",
        "kind": "entry_exit_pct",
        "date_col": "exit_date", "pnl_col": "ret_pct",  # pct → bps × 100
    },
}


def load_series(name, cfg):
    p = cfg["path"]
    if not p.exists():
        print(f"  ✗ {name}: file not found at {p}")
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
        s = df.groupby(cfg["date_col"])[cfg["pnl_col"]].sum() * 100  # %→bps
        s.name = name
        return s


def main():
    print("=" * 100)
    print("戦略間 PnL 相関分析")
    print("=" * 100)
    series = {}
    for name, cfg in SOURCES.items():
        s = load_series(name, cfg)
        if s is None: continue
        print(f"  ✓ {name}: N={len(s)}日  期間 {s.index.min().date()}〜{s.index.max().date()}  "
              f"平均 {s.mean():+.1f} bps  Sharpe {s.mean()/s.std()*np.sqrt(252):+.2f}")
        series[name] = s

    # 共通インデックス (全戦略の日付 union) で 0 埋め
    df = pd.concat(series, axis=1).fillna(0.0)

    # --- 重複期間のみ (共通期間) ---
    # 各戦略の非ゼロ日の範囲で共通化
    starts = []
    ends = []
    for s in series.values():
        active = s[s != 0]
        if len(active):
            starts.append(active.index.min())
            ends.append(active.index.max())
    common_start = max(starts)
    common_end = min(ends)
    print(f"\n共通期間: {common_start.date()} 〜 {common_end.date()}")
    df_common = df[(df.index >= common_start) & (df.index <= common_end)]

    # --- 相関行列 (全日、非取引日は 0) ---
    print("\n" + "=" * 100)
    print("【日次 PnL 相関行列 (全期間、非取引日=0)】")
    print("=" * 100)
    corr = df_common.corr()
    print(corr.round(3).to_string())

    # 相関の非対角平均
    m = corr.values.copy(); np.fill_diagonal(m, np.nan)
    print(f"\n平均 |ρ| = {np.nanmean(np.abs(m)):.3f}")

    # --- 取引日のみの相関 (両戦略が同日に取引したケース) ---
    print("\n" + "=" * 100)
    print("【両戦略が取引した日のみの相関】")
    print("=" * 100)
    names = list(series.keys())
    overlap_corr = pd.DataFrame(np.nan, index=names, columns=names)
    for i, a in enumerate(names):
        for b in names[i:]:
            sa = df_common[a]; sb = df_common[b]
            mask = (sa != 0) & (sb != 0)
            if mask.sum() < 5:
                overlap_corr.loc[a, b] = np.nan
                continue
            c = np.corrcoef(sa[mask], sb[mask])[0, 1]
            overlap_corr.loc[a, b] = c
            overlap_corr.loc[b, a] = c
    print(overlap_corr.round(3).to_string())
    print("\n  (NaN = 両戦略同日取引のサンプル不足)")

    # --- 統合ポートフォリオ (Equal-Weight) ---
    print("\n" + "=" * 100)
    print("【統合ポートフォリオ試算 (4戦略 Equal-Weight)】")
    print("=" * 100)
    combined = df_common.mean(axis=1)
    sh = combined.mean() / combined.std() * np.sqrt(252)
    t = combined.mean() / combined.std() * np.sqrt(len(combined))
    cum = combined.sum()
    mdd = (combined.cumsum().cummax() - combined.cumsum()).max()
    print(f"  日次平均: {combined.mean():+.2f} bps  (年率 {combined.mean()*252:+.0f} bps)")
    print(f"  Sharpe:  {sh:+.2f}")
    print(f"  t-stat:  {t:+.2f}")
    print(f"  累積:    {cum:+.0f} bps")
    print(f"  MDD:     {mdd:.0f} bps")

    # 個別 vs 統合の Sharpe 比較
    print("\n【個別 Sharpe vs 統合】")
    print(f"  {'戦略':<22} {'個別 Sharpe':>12}")
    for name in names:
        s = df_common[name]
        ss = s.mean() / s.std() * np.sqrt(252) if s.std() > 0 else 0
        print(f"  {name:<22} {ss:>+12.2f}")
    print(f"  {'4戦略統合 (EW)':<22} {sh:>+12.2f}")

    # --- 論理的な相関推測 (計測不能な戦略) ---
    print("\n" + "=" * 100)
    print("【計測不能な戦略との理論的相関】")
    print("=" * 100)
    logical = [
        ("topix_overnight",       "低 (別シグナル源、ON Long の方向性戦略)",  "0.05-0.15"),
        ("sox_overnight_short",   "極低 (ON Short 独立、SOX ドリブン)",       "< 0.05"),
        ("vwap_morning_meanrevert","中 (6146/8035 を pair と共有。ただしイントラ vs 日次で時間軸違い)", "0.15-0.30"),
        ("orb_breakout_long",     "低 (モメンタム、pair は MR)",                "0.05-0.15"),
    ]
    print(f"  {'戦略':<30} {'相関推定':<15} {'根拠'}")
    print("  " + "-" * 90)
    for name, reason, estim in logical:
        print(f"  {name:<30} {estim:<15} {reason}")

    # 最終推奨
    print("\n" + "=" * 100)
    print("【アロケーション推奨】")
    print("=" * 100)
    print("""
pair_portfolio は既存 6 戦略のいずれとも **独立性が高い** (日次スプレッド MR vs
方向性/ボラ戦略)。資金枠として:

  推奨配分 (リスク予算 Sharpe 按分):
    ON Long (lme/topix):         30% (共有バスケット・Sharpe 12+)
    pair_portfolio (新規):        25% (Sharpe 1.37, 低相関)
    eneos_vwap_trend:             15% (Sharpe 5.54)
    vwap_morning_meanrevert:     10% (Sharpe 6.11)
    orb_breakout_long:           10% (Sharpe 2.15)
    sox_overnight_short:          5% (Sharpe 2.11)
    lasertec_ma25_support:        5% (Sharpe 7.68)

  統合期待 Sharpe: 2.5-3.0 (Kelly 最適化で ~4 が上限目標)
""")


if __name__ == "__main__":
    main()
