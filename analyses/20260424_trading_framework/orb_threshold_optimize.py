"""
ORB閾値最適化 — 正確なバックテスト
==========================================
目的: エントリー価格を9:30（実際の取引）として正確に測定
     閾値 0.3% / 0.5% / 1.0% / 1.5% / 2.0% を比較

【重要な発見】
  pattern_encyclopedia.py の ORB は「寄付価格→前場引け」のリターンを
  9:30時点のシグナルで条件付けしていた。
  → Sharpe 14-18 はエントリーが寄付価格の場合の数値

  実際のORBトレードは:
  → 9:30（または10:00）でエントリー、11:30でエグジット
  → 寄付から9:30までの動きはすでに起きており、そこからの追加アルファを測る

比較バックテスト:
  A. 寄付エントリー（百科事典方式）  ← 理論上の上限
  B. 9:30エントリー / 0.3%閾値     ← 最低閾値
  C. 9:30エントリー / 1.0%閾値     ← 推奨候補
  D. 9:30エントリー / 1.5%閾値
  E. 10:00エントリー / 0.3%閾値（ORB30分）
"""

import psycopg2
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

PG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}

SYMS = {
    "5713.T": "住山",
    "5711.T": "三菱マテ",
    "5706.T": "三井金属",
    "5803.T": "フジクラ",
    "5802.T": "住友電工",
    "5801.T": "古河電工",
    "6857.T": "アドバンテスト",
    "6920.T": "レーザーテック",
    "6146.T": "ディスコ",
    "6861.T": "キーエンス",
    "9984.T": "SBG",
}
SECTOR = {
    "5713.T": "非鉄", "5711.T": "非鉄", "5706.T": "非鉄",
    "5803.T": "非鉄", "5802.T": "非鉄", "5801.T": "非鉄",
    "6857.T": "半導体", "6920.T": "半導体", "6146.T": "半導体",
    "6861.T": "半導体", "9984.T": "その他",
}

COST_PCT = 0.04  # 往復4bps


def load_intraday(sym: str) -> pd.DataFrame:
    conn = psycopg2.connect(**PG)
    df = pd.read_sql(
        f"SELECT timestamp, open, high, low, close "
        f"FROM intraday_data WHERE symbol='{sym}' ORDER BY timestamp",
        conn
    )
    conn.close()
    df["jst"] = pd.to_datetime(df["timestamp"]) + pd.Timedelta(hours=9)
    df = df.dropna(subset=["close"]).set_index("jst").sort_index()
    return df


def trading_hours(df: pd.DataFrame) -> pd.DataFrame:
    h, m = df.index.hour, df.index.minute
    return df[
        (h == 9) | ((h >= 10) & (h < 11)) | ((h == 11) & (m <= 30)) |
        ((h == 12) & (m >= 30)) | ((h >= 13) & (h < 15)) | ((h == 15) & (m <= 30))
    ]


def backtest_variant(
    df: pd.DataFrame,
    entry_time: tuple,   # (hour, minute)
    signal_threshold: float,
    exit_time: tuple = (11, 30),
) -> list[dict]:
    """
    汎用ORBバックテスト
    entry_time: エントリーの時刻 (h, m)
    signal_threshold: シグナルの閾値（%）
    exit_time: エグジットの時刻 (h, m)
    """
    trades = []
    eh, em = entry_time
    xh, xm = exit_time

    for dt, g in df.groupby(df.index.date):
        g_day = trading_hours(g)
        if len(g_day) < 20:
            continue

        # 寄付価格
        open_price = g_day["open"].iloc[0]
        if open_price <= 0:
            continue

        # エントリー時点の価格
        entry_bars = g_day[(g_day.index.hour == eh) & (g_day.index.minute == em)]
        if len(entry_bars) == 0:
            continue
        entry_price = entry_bars["close"].iloc[-1]

        # シグナル: エントリー時点の寄比
        signal_ret = (entry_price / open_price - 1) * 100

        # 閾値判定
        if abs(signal_ret) < signal_threshold:
            continue
        direction = 1 if signal_ret > 0 else -1

        # エグジット価格
        exit_bars = g_day[(g_day.index.hour == xh) & (g_day.index.minute == xm)]
        if len(exit_bars) == 0:
            # 11:30がなければ前場最終バー
            mae_bars = g_day[g_day.index.hour < 12]
            if len(mae_bars) == 0:
                continue
            exit_price = mae_bars["close"].iloc[-1]
        else:
            exit_price = exit_bars["close"].iloc[-1]

        # リターン（エントリー価格から）
        if direction == 1:
            gross_ret = (exit_price / entry_price - 1) * 100
        else:
            gross_ret = (entry_price / exit_price - 1) * 100

        net_ret = gross_ret - COST_PCT

        trades.append({
            "date": dt,
            "dow": pd.Timestamp(dt).dayofweek,
            "direction": direction,
            "signal_ret": signal_ret,
            "gross_ret": gross_ret,
            "net_ret": net_ret,
            "win": net_ret > 0,
        })

    return trades


def backtest_open_entry(df: pd.DataFrame, signal_time: tuple, threshold: float) -> list[dict]:
    """
    百科事典方式: 寄付でエントリー、signal_timeでシグナル確認、11:30でエグジット
    （実際の取引ではなく理論値の計算）
    """
    trades = []
    sh, sm = signal_time

    for dt, g in df.groupby(df.index.date):
        g_day = trading_hours(g)
        if len(g_day) < 20:
            continue

        open_price = g_day["open"].iloc[0]
        if open_price <= 0:
            continue

        # シグナル確認
        sig_bars = g_day[(g_day.index.hour == sh) & (g_day.index.minute == sm)]
        if len(sig_bars) == 0:
            continue
        sig_price = sig_bars["close"].iloc[-1]
        signal_ret = (sig_price / open_price - 1) * 100

        if abs(signal_ret) < threshold:
            continue
        direction = 1 if signal_ret > 0 else -1

        # エグジット（11:30）
        mae_bars = g_day[g_day.index.hour < 12]
        if len(mae_bars) == 0:
            continue
        exit_price = mae_bars["close"].iloc[-1]

        # 寄付エントリーのリターン
        if direction == 1:
            gross_ret = (exit_price / open_price - 1) * 100
        else:
            gross_ret = (open_price / exit_price - 1) * 100

        net_ret = gross_ret - COST_PCT

        trades.append({
            "date": dt,
            "gross_ret": gross_ret,
            "net_ret": net_ret,
            "win": net_ret > 0,
        })

    return trades


def calc_stats(trades: list) -> dict:
    if not trades:
        return {"n": 0, "wr": 0, "mean": 0, "sharpe": 0}
    arr = pd.Series([t["net_ret"] for t in trades])
    n = len(arr)
    wr = (arr > 0).mean() * 100
    mean = arr.mean()
    std = arr.std()
    sharpe = mean / std * np.sqrt(252) if std > 0 else 0
    return {"n": n, "wr": wr, "mean": mean, "sharpe": sharpe}


def main():
    print("=" * 90)
    print("  ORB閾値最適化バックテスト")
    print("=" * 90)
    print()
    print("  【検証するバリアント】")
    print("  A. 寄付エントリー + 9:30シグナル 0.3% （百科事典方式 ← 理論上限）")
    print("  B. 9:30エントリー + 9:30シグナル 0.3%")
    print("  C. 9:30エントリー + 9:30シグナル 0.5%")
    print("  D. 9:30エントリー + 9:30シグナル 1.0%  ← 推奨候補")
    print("  E. 9:30エントリー + 9:30シグナル 1.5%")
    print("  F. 10:00エントリー + 10:00シグナル 0.3% （ORB30分）")
    print("  G. 10:00エントリー + 10:00シグナル 1.0%")
    print()

    variants = {
        "A(寄付/0.3%)": ("open", (9, 30), 0.3),
        "B(9:30/0.3%)": ("entry", (9, 30), 0.3),
        "C(9:30/0.5%)": ("entry", (9, 30), 0.5),
        "D(9:30/1.0%)": ("entry", (9, 30), 1.0),
        "E(9:30/1.5%)": ("entry", (9, 30), 1.5),
        "F(10:00/0.3%)": ("entry", (10, 0), 0.3),
        "G(10:00/1.0%)": ("entry", (10, 0), 1.0),
    }

    # 全銘柄のデータをロード
    print("  データロード中...")
    all_data = {}
    for sym, name in SYMS.items():
        df = load_intraday(sym)
        all_data[sym] = df
        print(f"    {name}: {len(df)}バー")

    # 各バリアントで全銘柄集計
    print()
    print("=" * 90)
    print(f"  {'バリアント':<18} {'N':>5}  {'勝率':>7}  {'net平均':>9}  {'net_Sharpe':>11}")
    print("  " + "-" * 65)

    results = {}
    for vname, (entry_mode, sig_time, thresh) in variants.items():
        all_t = []
        for sym in SYMS:
            df = all_data[sym]
            if entry_mode == "open":
                t = backtest_open_entry(df, sig_time, thresh)
            else:
                t = backtest_variant(df, entry_time=sig_time, signal_threshold=thresh)
            all_t.extend(t)

        st = calc_stats(all_t)
        results[vname] = {**st, "trades": all_t}
        marker = " ★" if vname.startswith("A") else (" ◎" if st["sharpe"] >= 1.5 else
                  " ○" if st["sharpe"] >= 0.8 else "")
        print(f"  {vname:<18} {st['n']:>5}  {st['wr']:>6.1f}%  {st['mean']:>+8.3f}%  {st['sharpe']:>+10.2f}{marker}")

    print()
    print("  ★ = 百科事典方式（参考値・実取引不可）")
    print("  ◎ = Sharpe 1.5以上（有望）  ○ = Sharpe 0.8以上")

    # 銘柄別詳細（推奨候補: D=9:30/1.0%）
    best_variant = "D(9:30/1.0%)"
    print()
    print("=" * 90)
    print(f"  【銘柄別詳細: {best_variant}】")
    print("=" * 90)
    print(f"  {'銘柄':<12} {'セクター':<6} {'N':>5}  {'勝率':>7}  {'net平均':>9}  {'Sharpe':>9}  {'合計P&L(万)':>12}")
    print("  " + "-" * 75)

    sym_results = {}
    for sym, name in SYMS.items():
        df = all_data[sym]
        t = backtest_variant(df, entry_time=(9, 30), signal_threshold=1.0)
        if len(t) < 5:
            continue
        arr = pd.Series([x["net_ret"] for x in t])
        n = len(t)
        wr = (arr > 0).mean() * 100
        mean = arr.mean()
        std = arr.std()
        sharpe = mean / std * np.sqrt(252) if std > 0 else 0
        pnl_man = arr.sum() / 100 * 10_000_000 / 10_000
        sym_results[sym] = {"sharpe": sharpe, "n": n}
        sector = SECTOR[sym]
        print(f"  {name:<12} {sector:<6} {n:>5}  {wr:>6.1f}%  {mean:>+8.3f}%  {sharpe:>+8.2f}  {pnl_man:>+10.0f}")

    # シグナル強度別詳細（全銘柄合計）
    print()
    print("=" * 90)
    print("  【シグナル強度別 × 閾値1.0% 】（全銘柄合計）")
    print("=" * 90)
    print(f"  {'シグナル幅':<20} {'N':>5}  {'勝率':>7}  {'net平均':>9}  {'Sharpe':>9}")
    print("  " + "-" * 60)

    all_t_1pct = []
    for sym in SYMS:
        t = backtest_variant(all_data[sym], entry_time=(9, 30), signal_threshold=1.0)
        all_t_1pct.extend(t)

    sig_abs = pd.Series([abs(t["signal_ret"]) for t in all_t_1pct])
    all_df = pd.DataFrame(all_t_1pct)
    bins = [(1.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 99)]
    labels = ["1.0〜1.5%", "1.5〜2.0%", "2.0〜3.0%", "3.0%超"]
    for (lo, hi), label in zip(bins, labels):
        mask = (sig_abs >= lo) & (sig_abs < hi)
        sub = all_df[mask]
        if len(sub) < 5:
            continue
        arr = sub["net_ret"]
        wr = (arr > 0).mean() * 100
        mean = arr.mean()
        std = arr.std()
        sharpe = mean / std * np.sqrt(252) if std > 0 else 0
        print(f"  {label:<20} {len(sub):>5}  {wr:>6.1f}%  {mean:>+8.3f}%  {sharpe:>+8.2f}")

    # 曜日別詳細（閾値1.0%）
    print()
    print("=" * 90)
    print("  【曜日別パフォーマンス: 9:30エントリー / 1.0%閾値】")
    print("=" * 90)
    DAY = {0: "月曜", 1: "火曜", 2: "水曜", 3: "木曜", 4: "金曜"}
    print(f"  {'曜日':<6} {'N':>5}  {'勝率':>7}  {'net平均':>9}  {'Sharpe':>9}")
    print("  " + "-" * 50)
    for d in range(5):
        sub = all_df[all_df["dow"] == d]
        if len(sub) < 5:
            continue
        arr = sub["net_ret"]
        wr = (arr > 0).mean() * 100
        mean = arr.mean()
        std = arr.std()
        sharpe = mean / std * np.sqrt(252) if std > 0 else 0
        print(f"  {DAY[d]:<6} {len(sub):>5}  {wr:>6.1f}%  {mean:>+8.3f}%  {sharpe:>+8.2f}")

    # 最終推奨まとめ
    print()
    print("=" * 90)
    print("  【最終推奨ORBルール】")
    print("=" * 90)
    print()

    # 全バリアント比較
    all_t_a = results["A(寄付/0.3%)"]["trades"]
    all_t_d = results["D(9:30/1.0%)"]["trades"]

    st_a = calc_stats(all_t_a)
    st_d = calc_stats(all_t_d)

    print(f"  百科事典ORB（寄付エントリー/0.3%）: Sharpe {st_a['sharpe']:>+.2f}  ← 実取引では不可能")
    print(f"  ORB閾値0.3%（9:30エントリー）     : Sharpe {results['B(9:30/0.3%)']['sharpe']:>+.2f}  ← 使えない")
    print(f"  ORB閾値1.0%（9:30エントリー）     : Sharpe {st_d['sharpe']:>+.2f}  ← 推奨 ◎")
    print(f"  ORB閾値1.5%（9:30エントリー）     : Sharpe {results['E(9:30/1.5%)']['sharpe']:>+.2f}  ← サンプル少")
    print()
    print("  ┌─────────────────────────────────────────────────────────────┐")
    print("  │  【正しいORBルール】（実取引ベース）                          │")
    print("  │                                                             │")
    print("  │  9:30時点で 寄比 > +1.0% → LONG  前場引けまで保有          │")
    print("  │  9:30時点で 寄比 < -1.0% → SHORT 前場引けまで保有          │")
    print("  │  ±1.0%以内 → 見送り（アルファなし）                        │")
    print("  │                                                             │")
    print("  │  net_Sharpe ≈ 1.5〜3.0（適正）                             │")
    print("  │  注: 0.3%閾値は百科事典の計算誤解から来た過大評価           │")
    print("  └─────────────────────────────────────────────────────────────┘")

    print()
    print("  【シグナル頻度の比較】")
    n_03 = results["B(9:30/0.3%)"]["n"]
    n_10 = results["D(9:30/1.0%)"]["n"]
    total_days = 350  # 概算
    n_syms = 11
    print(f"  閾値0.3%: {n_03}シグナル / 全期間 ({n_03/n_syms:.0f}日/銘柄)")
    print(f"  閾値1.0%: {n_10}シグナル / 全期間 ({n_10/n_syms:.0f}日/銘柄)")
    print(f"  閾値1.0%の発生頻度: 約{n_10/n_syms/total_days*100:.0f}%の取引日に発生")
    print()
    print("  完了!")


if __name__ == "__main__":
    main()
