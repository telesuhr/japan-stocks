"""
戦略健全性チェック — メイン実行スクリプト

使い方:
  python3 health_check.py [--days N] [--end YYYY-MM-DD]

出力:
  - コンソール: テーブル形式サマリー
  - strategies/health/YYYYMMDD.md: レポートファイル
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import argparse
from datetime import date, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

# パス設定
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent.parent))  # japan-stocks root

from _lib import get_conn, summary_stats

# 各戦略モジュール
from _lib import pre_earnings_drift
from _lib import earnings_pead
from _lib import vwap_morning_meanrevert
from _lib import lasertec_ma25_support
from _lib import bank_absorption
from _lib import eneos_vwap_trend

# IS基準値 (SUMMARY.md より)
IS_BASELINE = {
    "pre_earnings_drift":      {"sharpe": 2.07, "freq_label": "週1000+件/年"},
    "earnings_pead":           {"sharpe": 2.19, "freq_label": "月複数回"},
    "vwap_morning_meanrevert": {"sharpe": 6.76, "freq_label": "週1-2回"},
    "lasertec_ma25_support":   {"sharpe": 7.57, "freq_label": "月1-2回"},
    "bank_absorption":         {"sharpe": 3.94, "freq_label": "週1-2回"},
    "eneos_vwap_trend":        {"sharpe": 3.81, "freq_label": "週1-2回"},
}

MODULES = [
    ("pre_earnings_drift",      pre_earnings_drift),
    ("earnings_pead",           earnings_pead),
    ("vwap_morning_meanrevert", vwap_morning_meanrevert),
    ("lasertec_ma25_support",   lasertec_ma25_support),
    ("bank_absorption",         bank_absorption),
    ("eneos_vwap_trend",        eneos_vwap_trend),
]


def get_bdays(start_str: str, end_str: str) -> list:
    """trading_calendarから営業日リスト取得"""
    try:
        conn = get_conn()
        df = pd.read_sql(
            "SELECT date FROM trading_calendar WHERE date >= %s AND date <= %s ORDER BY date",
            conn, params=(start_str, end_str)
        )
        conn.close()
        return df["date"].tolist()
    except Exception:
        return pd.bdate_range(start_str, end_str).date.tolist()


def verdict(stats: dict, is_sharpe: float) -> str:
    n = stats.get("n", 0)
    sh = stats.get("sharpe")
    if n == 0 or sh is None:
        return "🚨"  # シグナルゼロ
    if sh < 0:
        return "🚨"
    if sh >= is_sharpe * 0.6:
        return "✅"
    return "⚠️"


def run(n_days: int = 60, end_date_str: str = None):
    if end_date_str is None:
        # 最新営業日を取得
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("SELECT MAX(date) FROM stocks_daily")
        end_date = cur.fetchone()[0]
        conn.close()
    else:
        end_date = date.fromisoformat(end_date_str)

    bdays = get_bdays("2020-01-01", str(end_date))
    if len(bdays) < n_days:
        start_date = bdays[0] if bdays else end_date - timedelta(days=90)
    else:
        start_date = bdays[-n_days]

    start_str = str(start_date)
    end_str   = str(end_date)

    print(f"\n{'='*70}")
    print(f"戦略健全性チェック  期間: {start_str} 〜 {end_str} ({n_days}営業日)")
    print(f"{'='*70}\n")

    results = []
    for name, mod in MODULES:
        print(f"  [{name}] 計算中...", end=" ", flush=True)
        try:
            stats = mod.health(start_str, end_str)
            stats["name"] = name
            results.append(stats)
            n = stats.get("n", 0)
            sh = stats.get("sharpe", "N/A")
            print(f"N={n}, Sharpe={sh}")
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({"name": name, "strategy": name, "n": 0, "sharpe": None,
                             "t_stat": None, "win_rate": None, "mean_pct": None,
                             "signal_days": 0, "error": str(e)})

    print()
    _print_table(results)
    _save_report(results, start_str, end_str, n_days)
    return results


def _print_table(results: list):
    header = f"{'戦略':<30} {'IS Sh':>7} {'足元Sh':>7} {'t値':>6} {'N':>5} {'勝率%':>6} {'mean%':>7} {'判定'}"
    print(header)
    print("-" * 80)
    for r in results:
        name = r["name"]
        is_sh = IS_BASELINE.get(name, {}).get("sharpe", "-")
        sh    = r.get("sharpe")
        t     = r.get("t_stat")
        n     = r.get("n", 0)
        wr    = r.get("win_rate")
        m     = r.get("mean_pct")
        v     = verdict(r, is_sh if isinstance(is_sh, float) else 0)

        is_sh_s = f"{is_sh:.2f}" if isinstance(is_sh, float) else "-"
        sh_s  = f"{sh:.2f}"  if sh  is not None else "N/A"
        t_s   = f"{t:.2f}"   if t   is not None else "N/A"
        wr_s  = f"{wr:.1f}"  if wr  is not None else "N/A"
        m_s   = f"{m:.3f}"   if m   is not None else "N/A"

        print(f"{name:<30} {is_sh_s:>7} {sh_s:>7} {t_s:>6} {n:>5} {wr_s:>6} {m_s:>7} {v}")

    print()
    alarms = [r for r in results if verdict(r, IS_BASELINE.get(r["name"], {}).get("sharpe", 0) or 0) == "🚨"]
    warnings = [r for r in results if verdict(r, IS_BASELINE.get(r["name"], {}).get("sharpe", 0) or 0) == "⚠️"]
    if alarms:
        print(f"🚨 要対応: {', '.join(r['name'] for r in alarms)}")
    if warnings:
        print(f"⚠️ 劣化気味: {', '.join(r['name'] for r in warnings)}")
    if not alarms and not warnings:
        print("✅ 全戦略: 基準を満たしています")


def _save_report(results: list, start_str: str, end_str: str, n_days: int):
    today = date.today().strftime("%Y%m%d")
    out_path = HERE / f"{today}.md"

    lines = [
        f"# 戦略健全性チェック {date.today().strftime('%Y-%m-%d')}",
        f"",
        f"**期間**: {start_str} 〜 {end_str} ({n_days}営業日)  ",
        f"**基準**: IS Sharpe の 60%以上 → ✅ / 正だが60%未満 → ⚠️ / マイナスまたはシグナルゼロ → 🚨",
        f"",
        f"## サマリーテーブル",
        f"",
        f"| 戦略 | IS Sh | 足元 Sh | t値 | N | 勝率% | mean% | 頻度目安 | 判定 |",
        f"|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]

    for r in results:
        name  = r["name"]
        is_sh = IS_BASELINE.get(name, {}).get("sharpe", "-")
        freq  = IS_BASELINE.get(name, {}).get("freq_label", "-")
        sh    = r.get("sharpe")
        t     = r.get("t_stat")
        n     = r.get("n", 0)
        wr    = r.get("win_rate")
        m     = r.get("mean_pct")
        v     = verdict(r, is_sh if isinstance(is_sh, float) else 0)

        is_sh_s = f"{is_sh:.2f}" if isinstance(is_sh, float) else "-"
        sh_s  = f"{sh:.2f}"  if sh  is not None else "N/A"
        t_s   = f"{t:.2f}"   if t   is not None else "N/A"
        wr_s  = f"{wr:.1f}"  if wr  is not None else "N/A"
        m_s   = f"{m:.3f}"   if m   is not None else "N/A"

        lines.append(f"| {name} | {is_sh_s} | {sh_s} | {t_s} | {n} | {wr_s} | {m_s} | {freq} | {v} |")

    lines += [
        f"",
        f"## 戦略別所見",
        f"",
    ]

    for r in results:
        name = r["name"]
        is_sh = IS_BASELINE.get(name, {}).get("sharpe", 0) or 0
        v = verdict(r, is_sh)
        sh = r.get("sharpe")
        n = r.get("n", 0)
        err = r.get("error")

        lines.append(f"### {v} {name}")
        if err:
            lines.append(f"- **エラー**: {err}")
        elif n == 0:
            lines.append(f"- **シグナルなし**: 期間中に条件を満たすシグナルが発生していない")
            lines.append(f"- 原因候補: 市場レジーム変化、銘柄特性変化、データ欠損")
        else:
            sh_ratio = sh / is_sh if is_sh > 0 and sh is not None else 0
            lines.append(f"- 足元Sharpe: {sh:.2f} (IS比 {sh_ratio:.0%})")
            lines.append(f"- N={n}, 勝率={r.get('win_rate','N/A')}%, mean={r.get('mean_pct','N/A')}%")
            if v == "✅":
                lines.append(f"- IS基準の{sh_ratio:.0%}を維持。引き続き正常稼働")
            elif v == "⚠️":
                lines.append(f"- IS比{sh_ratio:.0%}に低下。劣化監視が必要")
                lines.append(f"- 原因候補: レジーム変化・銘柄構成変化・条件パラメータの陳腐化")
            elif v == "🚨":
                lines.append(f"- **要対応**: 足元マイナス。一時停止を検討")
                lines.append(f"- 原因切り分け: (1) 特定銘柄の異常? (2) マクロレジーム変化? (3) データ欠損?")
        lines.append(f"")

    lines += [
        f"---",
        f"*生成: {date.today()} / strategies/health/health_check.py*",
    ]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nレポート保存: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--end", type=str, default=None)
    args = parser.parse_args()
    run(n_days=args.days, end_date_str=args.end)
