"""
ORBバックテスト — トレード明細付き完全バックテスト
=======================================================
戦略: 9:30時点の寄比±0.3%超でエントリー → 前場引け（11:30）でクローズ

出力:
  1. トレード明細CSV (orb_trades.csv)
  2. 月次/年次P&L集計
  3. ドローダウン分析
  4. 実行サイジング推奨
"""

import psycopg2
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime

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

# コスト設定
COST_ONE_WAY_BPS = 2  # 2bps/片道 = 4bps往復
POSITION_JPY = 10_000_000  # 1ポジション1000万円


def load_intraday(sym: str) -> pd.DataFrame:
    conn = psycopg2.connect(**PG)
    df = pd.read_sql(
        f"SELECT timestamp, open, high, low, close, volume "
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


def backtest_orb(sym: str, df: pd.DataFrame,
                 threshold: float = 0.3,
                 use_1000: bool = True) -> list[dict]:
    """
    ORB30分モメンタム戦略のバックテスト
    エントリー: 9:30時点で寄比 > +threshold% → LONG
                9:30時点で寄比 < -threshold% → SHORT
    エグジット: 11:30 前場引け（またはデータが切れる最後のバー）
    """
    trades = []

    for dt, g in df.groupby(df.index.date):
        g_day = trading_hours(g)
        if len(g_day) < 20:
            continue

        # 寄付価格
        open_price = g_day["open"].iloc[0]
        if open_price <= 0:
            continue

        # 9:30 チェックポイント
        cp930 = g_day[(g_day.index.hour == 9) & (g_day.index.minute == 30)]
        if len(cp930) == 0:
            continue
        price_930 = cp930["close"].iloc[-1]
        ret_930 = (price_930 / open_price - 1) * 100

        # 閾値判定
        if abs(ret_930) < threshold:
            continue
        direction = 1 if ret_930 > 0 else -1  # 1=LONG, -1=SHORT

        # エグジット: 前場引け（11:30）
        mae = g_day[g_day.index.hour < 12]
        if len(mae) == 0:
            continue
        exit_price = mae["close"].iloc[-1]

        # リターン計算（エントリーは9:30の終値）
        if direction == 1:
            gross_ret = (exit_price / price_930 - 1) * 100
        else:
            gross_ret = (price_930 / exit_price - 1) * 100

        # コスト（往復 2bps×2）
        cost_pct = COST_ONE_WAY_BPS * 2 / 100  # 0.04%
        net_ret = gross_ret - cost_pct

        # P&L（円換算）
        pnl_gross = POSITION_JPY * gross_ret / 100
        pnl_net = POSITION_JPY * net_ret / 100

        # 最大逆行（MAE）
        if direction == 1:
            mae_ret = ((mae[mae.index >= cp930.index[0]]["low"].min() / price_930) - 1) * 100
        else:
            mae_ret = ((price_930 / mae[mae.index >= cp930.index[0]]["high"].max()) - 1) * 100

        trades.append({
            "date": dt,
            "year": pd.Timestamp(dt).year,
            "month": pd.Timestamp(dt).month,
            "dow": pd.Timestamp(dt).dayofweek,
            "sym": sym,
            "name": SYMS[sym],
            "sector": SECTOR[sym],
            "direction": "LONG" if direction == 1 else "SHORT",
            "open_price": round(open_price, 1),
            "entry_price": round(price_930, 1),
            "exit_price": round(exit_price, 1),
            "signal_ret": round(ret_930, 3),    # 9:30時点の寄比（シグナル）
            "gross_ret": round(gross_ret, 3),
            "net_ret": round(net_ret, 3),
            "pnl_gross": round(pnl_gross, 0),
            "pnl_net": round(pnl_net, 0),
            "mae_ret": round(mae_ret, 3),       # 最大逆行
            "win": net_ret > 0,
        })

    return trades


def print_summary(all_trades: pd.DataFrame):
    """全体サマリー出力"""
    print("\n" + "=" * 80)
    print("【ORBバックテスト結果サマリー】")
    print("=" * 80)
    print(f"  期間: {all_trades['date'].min()} 〜 {all_trades['date'].max()}")
    print(f"  総トレード数: {len(all_trades):,} 件")
    print(f"  ポジションサイズ: {POSITION_JPY/1_000_000:.0f}百万円/銘柄")
    print()

    # 銘柄別サマリー
    print(f"{'銘柄':<12} {'N':>5} {'勝率':>7} {'net平均':>9} {'net合計':>12} {'Sharpe':>8} {'最大DD':>8}")
    print("-" * 75)

    sym_stats = []
    for sym, name in SYMS.items():
        t = all_trades[all_trades["sym"] == sym]
        if len(t) < 10:
            continue
        n = len(t)
        wr = t["win"].mean() * 100
        net_mean = t["net_ret"].mean()
        net_sum_pnl = t["pnl_net"].sum()
        net_std = t["net_ret"].std()
        sharpe = net_mean / net_std * np.sqrt(252) if net_std > 0 else 0
        # 最大ドローダウン（累積P&L）
        cum_pnl = t.sort_values("date")["pnl_net"].cumsum()
        roll_max = cum_pnl.cummax()
        dd = (cum_pnl - roll_max)
        max_dd = dd.min()

        sym_stats.append({
            "name": name, "n": n, "wr": wr, "net_mean": net_mean,
            "net_sum_pnl": net_sum_pnl, "sharpe": sharpe, "max_dd": max_dd
        })
        print(f"{name:<12} {n:>5} {wr:>6.1f}% {net_mean:>+8.3f}% {net_sum_pnl/1_000_000:>+10.2f}M {sharpe:>+7.2f} {max_dd/1_000_000:>+7.2f}M")

    total_pnl = all_trades["pnl_net"].sum()
    print("-" * 75)
    print(f"{'合計（全銘柄）':<12} {len(all_trades):>5} "
          f"{all_trades['win'].mean()*100:>6.1f}% "
          f"{all_trades['net_ret'].mean():>+8.3f}% "
          f"{total_pnl/1_000_000:>+10.2f}M")


def print_monthly(all_trades: pd.DataFrame):
    """月次P&L集計"""
    print("\n" + "=" * 80)
    print("【月次P&L（全銘柄合計）】  単位: 万円")
    print("=" * 80)

    all_trades["ym"] = all_trades["date"].astype(str).str[:7]
    monthly = all_trades.groupby("ym").agg(
        n=("pnl_net", "count"),
        pnl=("pnl_net", "sum"),
        wr=("win", "mean"),
        mean_ret=("net_ret", "mean")
    ).reset_index()
    monthly["pnl_man"] = monthly["pnl"] / 10_000

    print(f"  {'年月':<8} {'N':>4} {'勝率':>7} {'平均ret':>8} {'月P&L(万)':>12}")
    print("  " + "-" * 50)
    for _, row in monthly.iterrows():
        bar = "█" * int(max(0, row["pnl_man"]) / 50) if row["pnl_man"] > 0 else "░" * int(max(0, -row["pnl_man"]) / 50)
        marker = "▲" if row["pnl_man"] > 0 else "▼"
        print(f"  {row['ym']:<8} {row['n']:>4} {row['wr']*100:>6.1f}% "
              f"{row['mean_ret']:>+7.3f}% {row['pnl_man']:>+10.0f}  {marker} {bar[:30]}")

    print(f"\n  合計P&L: {monthly['pnl_man'].sum():>+.0f}万円")
    print(f"  月次勝率: {(monthly['pnl']>0).mean()*100:.1f}%")


def print_yearly(all_trades: pd.DataFrame):
    """年次P&L集計"""
    print("\n" + "=" * 80)
    print("【年次P&L】  単位: 万円")
    print("=" * 80)

    yearly = all_trades.groupby("year").agg(
        n=("pnl_net", "count"),
        pnl=("pnl_net", "sum"),
        wr=("win", "mean"),
    ).reset_index()
    yearly["pnl_man"] = yearly["pnl"] / 10_000

    print(f"  {'年':>5} {'N':>5} {'勝率':>7} {'年P&L(万)':>12}")
    print("  " + "-" * 35)
    for _, row in yearly.iterrows():
        marker = "▲" if row["pnl_man"] > 0 else "▼"
        print(f"  {row['year']:>5} {row['n']:>5} {row['wr']*100:>6.1f}% {row['pnl_man']:>+10.0f}  {marker}")


def print_dow_breakdown(all_trades: pd.DataFrame):
    """曜日別ORBパフォーマンス"""
    DAY = {0: "月曜", 1: "火曜", 2: "水曜", 3: "木曜", 4: "金曜"}
    print("\n" + "=" * 80)
    print("【曜日別ORBパフォーマンス】")
    print("=" * 80)
    print(f"  {'曜日':<6} {'N':>4} {'勝率':>7} {'net平均':>9} {'Sharpe':>8}")
    print("  " + "-" * 45)
    for d in range(5):
        t = all_trades[all_trades["dow"] == d]
        if len(t) < 5:
            continue
        wr = t["win"].mean() * 100
        net_mean = t["net_ret"].mean()
        net_std = t["net_ret"].std()
        sharpe = net_mean / net_std * np.sqrt(252) if net_std > 0 else 0
        print(f"  {DAY[d]:<6} {len(t):>4} {wr:>6.1f}% {net_mean:>+8.3f}% {sharpe:>+7.2f}")


def print_direction_breakdown(all_trades: pd.DataFrame):
    """LONG/SHORT別パフォーマンス"""
    print("\n" + "=" * 80)
    print("【LONG / SHORT 別パフォーマンス】")
    print("=" * 80)
    print(f"  {'方向':<8} {'N':>5} {'勝率':>7} {'net平均':>9} {'Sharpe':>8}")
    print("  " + "-" * 45)
    for direction in ["LONG", "SHORT"]:
        t = all_trades[all_trades["direction"] == direction]
        wr = t["win"].mean() * 100
        net_mean = t["net_ret"].mean()
        net_std = t["net_ret"].std()
        sharpe = net_mean / net_std * np.sqrt(252) if net_std > 0 else 0
        print(f"  {direction:<8} {len(t):>5} {wr:>6.1f}% {net_mean:>+8.3f}% {sharpe:>+7.2f}")


def print_sector_breakdown(all_trades: pd.DataFrame):
    """セクター別パフォーマンス"""
    print("\n" + "=" * 80)
    print("【セクター別パフォーマンス】")
    print("=" * 80)
    print(f"  {'セクター':<8} {'N':>5} {'勝率':>7} {'net平均':>9} {'Sharpe':>8} {'合計P&L(万)':>12}")
    print("  " + "-" * 60)
    for sector in ["非鉄", "半導体", "その他"]:
        t = all_trades[all_trades["sector"] == sector]
        if len(t) < 5:
            continue
        wr = t["win"].mean() * 100
        net_mean = t["net_ret"].mean()
        net_std = t["net_ret"].std()
        sharpe = net_mean / net_std * np.sqrt(252) if net_std > 0 else 0
        total_pnl = t["pnl_net"].sum() / 10_000
        print(f"  {sector:<8} {len(t):>5} {wr:>6.1f}% {net_mean:>+8.3f}% {sharpe:>+7.2f} {total_pnl:>+10.0f}")


def print_signal_strength(all_trades: pd.DataFrame):
    """シグナル強度別パフォーマンス（9:30の寄比幅別）"""
    print("\n" + "=" * 80)
    print("【シグナル強度別パフォーマンス（9:30寄比の絶対値）】")
    print("=" * 80)
    print(f"  {'シグナル幅':<16} {'N':>5} {'勝率':>7} {'net平均':>9} {'Sharpe':>8}")
    print("  " + "-" * 55)

    bins = [(0.3, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 99)]
    labels = ["0.3〜0.5%", "0.5〜1.0%", "1.0〜2.0%", "2.0%超"]
    sig = all_trades["signal_ret"].abs()
    for (lo, hi), label in zip(bins, labels):
        t = all_trades[(sig >= lo) & (sig < hi)]
        if len(t) < 5:
            continue
        wr = t["win"].mean() * 100
        net_mean = t["net_ret"].mean()
        net_std = t["net_ret"].std()
        sharpe = net_mean / net_std * np.sqrt(252) if net_std > 0 else 0
        print(f"  {label:<16} {len(t):>5} {wr:>6.1f}% {net_mean:>+8.3f}% {sharpe:>+7.2f}")


def print_drawdown(all_trades: pd.DataFrame):
    """ドローダウン分析（全銘柄ポートフォリオ）"""
    print("\n" + "=" * 80)
    print("【ドローダウン分析（全銘柄合計）】")
    print("=" * 80)

    daily_pnl = all_trades.groupby("date")["pnl_net"].sum().sort_index()
    cum_pnl = daily_pnl.cumsum()
    roll_max = cum_pnl.cummax()
    dd = cum_pnl - roll_max
    max_dd = dd.min()
    max_dd_date = dd.idxmin()

    # 回復期間
    if max_dd_date is not None:
        post_dd = cum_pnl[cum_pnl.index > max_dd_date]
        prev_peak = roll_max.loc[max_dd_date]
        recovery_dates = post_dd[post_dd >= prev_peak]
        recovery_days = (recovery_dates.index[0] - max_dd_date).days if len(recovery_dates) > 0 else None

    print(f"  最大ドローダウン: {max_dd/10_000:>+.0f}万円  ({max_dd_date})")
    if recovery_days is not None:
        print(f"  回復期間:         {recovery_days}日")
    else:
        print(f"  回復期間:         未回復（期間末まで）")

    # 月次最大損失
    monthly_pnl = all_trades.groupby(all_trades["date"].astype(str).str[:7])["pnl_net"].sum()
    worst_month = monthly_pnl.idxmin()
    print(f"  最悪月:           {worst_month}  ({monthly_pnl.min()/10_000:>+.0f}万円)")
    print(f"  最良月:           {monthly_pnl.idxmax()}  ({monthly_pnl.max()/10_000:>+.0f}万円)")

    # 連敗記録
    wins = all_trades.sort_values("date")["win"]
    max_losing = 0
    cur_losing = 0
    for w in wins:
        if not w:
            cur_losing += 1
            max_losing = max(max_losing, cur_losing)
        else:
            cur_losing = 0
    print(f"  最大連敗:         {max_losing}回")


def print_sizing_guide(all_trades: pd.DataFrame):
    """実運用のサイジング推奨"""
    print("\n" + "=" * 80)
    print("【実運用サイジング推奨】")
    print("=" * 80)

    # 銘柄別Sharpe
    sym_sharpe = {}
    for sym, name in SYMS.items():
        t = all_trades[all_trades["sym"] == sym]
        if len(t) < 10:
            continue
        net_mean = t["net_ret"].mean()
        net_std = t["net_ret"].std()
        sharpe = net_mean / net_std * np.sqrt(252) if net_std > 0 else 0
        sym_sharpe[sym] = (name, sharpe, net_mean, t["win"].mean() * 100)

    # Sharpe順に表示
    sorted_syms = sorted(sym_sharpe.items(), key=lambda x: x[1][1], reverse=True)

    print("  【Sharpeランキングによるサイズ配分案】")
    print(f"  {'順':>2} {'銘柄':<12} {'Sharpe':>8} {'net平均':>9} {'勝率':>7}  {'推奨サイズ':>10}")
    print("  " + "-" * 65)

    # Sharpe > 15 → フルサイズ, 12-15 → 75%, < 12 → 50%
    for i, (sym, (name, sharpe, net_mean, wr)) in enumerate(sorted_syms, 1):
        if sharpe >= 15:
            size = "1000万 (100%)"
            mark = "◎"
        elif sharpe >= 12:
            size = " 750万 (75%)"
            mark = "○"
        else:
            size = " 500万 (50%)"
            mark = "△"
        print(f"  {i:>2} {name:<12} {sharpe:>+7.2f} {net_mean:>+8.3f}% {wr:>6.1f}%  {mark} {size}")

    print()
    print("  【注意事項】")
    print("  ・同一日に全銘柄でシグナルが出た場合の総エクスポージャー: 約1億円")
    print("  ・相関が高い銘柄（同一セクター）は同時エントリーで倍リスク")
    print("  ・非鉄6銘柄は高相関 → 実質的に1銘柄の集中投資と同等")
    print("  ・推奨: 非鉄は最大3銘柄まで / 半導体は最大2銘柄まで")
    print()
    print("  【ストップロス設定】")
    print("  ・MAEの分布から: シグナル方向と逆に-1.0%で損切り推奨")
    print("    （MAE -1%超の日の最終損失 vs 全体PFの比較）")

    # MAE分布
    big_mae = all_trades[all_trades["mae_ret"] < -1.0]
    print(f"  ・MAE -1%超のトレード: {len(big_mae)}/{len(all_trades)} "
          f"({len(big_mae)/len(all_trades)*100:.1f}%) "
          f"その日の平均net_ret: {big_mae['net_ret'].mean():>+.3f}%")


def main():
    print("=" * 80)
    print("  ORBバックテスト実行中...")
    print("=" * 80)

    all_trades_list = []
    for sym, name in SYMS.items():
        print(f"  {name} ({sym}) ロード中...", end=" ")
        try:
            df = load_intraday(sym)
            trades = backtest_orb(sym, df, threshold=0.3)
            all_trades_list.extend(trades)
            print(f"{len(trades)}トレード")
        except Exception as e:
            print(f"エラー: {e}")

    if not all_trades_list:
        print("トレードなし。終了。")
        return

    all_trades = pd.DataFrame(all_trades_list)
    all_trades["date"] = pd.to_datetime(all_trades["date"])

    # CSV保存
    out_path = "orb_trades.csv"
    all_trades.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n  → トレード明細: {out_path} ({len(all_trades)}件)")

    # 各種分析出力
    print_summary(all_trades)
    print_sector_breakdown(all_trades)
    print_direction_breakdown(all_trades)
    print_dow_breakdown(all_trades)
    print_signal_strength(all_trades)
    print_monthly(all_trades)
    print_yearly(all_trades)
    print_drawdown(all_trades)
    print_sizing_guide(all_trades)

    print("\n" + "=" * 80)
    print(f"  完了: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
