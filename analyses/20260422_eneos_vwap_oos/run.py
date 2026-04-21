"""
ENEOS (5020.T) VWAP Trend 戦略 Out-of-Sample 検証

In-sample: 2025-04-01 〜 2025-10-10 (H1, 約6ヶ月)
OoS:       2025-10-11 〜 2026-04-21 (H2, 約6ヶ月)

検証項目:
  1. H1/H2 半期分割テスト
  2. パラメータ安定性 (signal_time × threshold の全組合せ)
  3. エネルギーセクター横断 (出光/INPEX との比較)
  4. ローリングSharpe (3ヶ月窓) — 性能が安定しているか
  5. 月別パフォーマンス
"""
import psycopg2
import pandas as pd
import numpy as np
from datetime import date
import warnings
warnings.filterwarnings("ignore")

PG_CONFIG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}

FULL_START = "2025-04-01"
FULL_END   = "2026-04-21"
H1_END     = "2025-10-10"  # in-sample
H2_START   = "2025-10-11"  # out-of-sample

COST_BPS = 4

# 対象銘柄
TARGET = {
    "5020.T": "ENEOS",
    "5016.T": "出光",
    "1605.T": "INPEX",
    "5019.T": "出光HD",
}

# (hour, minute) 形式で定義
SIGNAL_TIMES = [(9, 30), (10, 0), (11, 0)]
THRESHOLDS   = [20, 30, 50]  # bps


def load_stock(sym, start, end):
    conn = psycopg2.connect(**PG_CONFIG)
    q = f"""SELECT timestamp, open, high, low, close, volume FROM intraday_data
            WHERE symbol='{sym}' AND timestamp >= '{start}' AND timestamp < '{end}'
            ORDER BY timestamp"""
    df = pd.read_sql(q, conn); conn.close()
    df['jst'] = pd.to_datetime(df['timestamp']) + pd.Timedelta(hours=9)
    return df.dropna(subset=['close']).set_index('jst').sort_index()


def compute_vwap(day_df):
    """日次VWAP (9:00から累積)"""
    vol = day_df['volume'].fillna(0)
    vol = vol.where(vol > 0, 1.0)
    cum_vol = vol.cumsum()
    cum_pv  = (day_df['close'] * vol).cumsum()
    return cum_pv / cum_vol


def backtest_vwap_trend(df, signal_hour, signal_minute, threshold_bps):
    """
    VWAP Trend戦略: (signal_hour:signal_minute)時点のdev >= +th → Long / <= -th → Short
    エントリー: シグナル時刻より後の最初のバーopen / エグジット: 15:20-15:30の最後のclose
    """
    trades = []
    for d in sorted(set(df.index.date)):
        day = df[df.index.date == d]
        if len(day) < 10: continue

        # VWAP計算 (9:00以降の累積)
        morning = day[day.index.hour >= 9]
        if morning.empty: continue
        vwap = compute_vwap(morning)

        # signal_hour:signal_minute 以前のバーのみでシグナル判定（将来データ禁止）
        sig_bar = morning[
            (morning.index.hour < signal_hour) |
            ((morning.index.hour == signal_hour) & (morning.index.minute <= signal_minute))
        ]
        if sig_bar.empty: continue
        sig_price = float(sig_bar['close'].iloc[-1])
        sig_vwap  = float(vwap.loc[sig_bar.index[-1]])
        dev = (sig_price / sig_vwap - 1) * 10000  # bps

        if abs(dev) < threshold_bps: continue
        direction = 1 if dev > 0 else -1

        # エントリー: シグナル時刻より後の最初のバー
        entry_bar = day[
            (day.index.hour > signal_hour) |
            ((day.index.hour == signal_hour) & (day.index.minute > signal_minute))
        ]
        if entry_bar.empty: continue
        entry_price = float(entry_bar['open'].iloc[0])
        entry_time  = entry_bar.index[0]

        # エグジット: 15:20-15:30の最後のclose
        exit_bar = day[(day.index.hour == 15) & (day.index.minute >= 20)]
        if exit_bar.empty: continue
        exit_price = float(exit_bar['close'].iloc[-1])

        ret_bps = (exit_price / entry_price - 1) * 10000 * direction
        trades.append({
            'date': d, 'dev': dev, 'direction': direction,
            'entry': entry_price, 'exit': exit_price,
            'pnl_bps': ret_bps - COST_BPS,
        })
    return pd.DataFrame(trades)


def evaluate(tdf):
    if len(tdf) == 0: return None
    arr = tdf['pnl_bps'].values
    std = arr.std()
    mean = arr.mean()
    return {
        'n': len(arr),
        'mean': mean,
        'wr': (arr > 0).mean() * 100,
        'pf': arr[arr>0].sum() / abs(arr[arr<=0].sum()) if arr[arr<=0].sum() != 0 else np.inf,
        'sharpe': mean / std * np.sqrt(252) if std > 0 else 0,
        't_stat': mean / (std / np.sqrt(len(arr))) if std > 0 else 0,
        'total': arr.sum(),
    }


def fmt(r, show_total=False):
    if r is None: return "  N/A"
    s = (f"  N={r['n']:>3}, Mean={r['mean']:>+6.1f}bps, WR={r['wr']:>5.1f}%, "
         f"Sharpe={r['sharpe']:>+6.2f}, t={r['t_stat']:>+5.2f}")
    if show_total:
        s += f", Total={r['total']:>+6.0f}bps"
    return s


def rolling_sharpe(tdf, window=15):
    """ローリングSharpe (window=トレード数)"""
    if len(tdf) < window: return pd.Series(dtype=float)
    arr = tdf.set_index('date')['pnl_bps']
    rs = []
    for i in range(window, len(arr)+1):
        sub = arr.iloc[i-window:i]
        std = sub.std()
        rs.append({'date': arr.index[i-1],
                   'sharpe': sub.mean() / std * np.sqrt(252) if std > 0 else 0})
    return pd.DataFrame(rs).set_index('date')['sharpe']


def main():
    print("=" * 75)
    print("ENEOS (5020.T) VWAP Trend — Out-of-Sample 検証")
    print(f"H1(IS): {FULL_START}〜{H1_END}  /  H2(OoS): {H2_START}〜{FULL_END}")
    print("=" * 75)

    # ── 1. H1/H2 分割テスト (ENEOS, signal=9:30, th=50bps) ─────────────
    print(f"\n{'=' * 75}")
    print("1. H1 / H2 分割テスト  [ENEOS, signal=09:30, threshold=50bps]")
    print(f"{'=' * 75}")

    df_h1 = load_stock("5020.T", FULL_START, H1_END)
    df_h2 = load_stock("5020.T", H2_START, FULL_END)
    df_full = load_stock("5020.T", FULL_START, FULL_END)

    for label, df in [("Full  ", df_full), ("H1(IS)", df_h1), ("H2(OoS)", df_h2)]:
        tdf = backtest_vwap_trend(df, 9, 30, 50)
        r = evaluate(tdf)
        tag = " ← 採用根拠" if "Full" in label else (" ← OoS" if "H2" in label else "")
        print(f"  {label}:{fmt(r, show_total=True)}{tag}")

    # ── 2. パラメータ安定性 (全組合せ × Full期間) ────────────────────────
    print(f"\n{'=' * 75}")
    print("2. パラメータ安定性  [ENEOS, Full期間]")
    print(f"{'=' * 75}")
    print(f"  {'time':>6} {'th':>5}  {'N':>4} {'Mean':>7} {'WR':>6} {'Shp':>6} {'t':>5}")
    print(f"  {'-'*6} {'-'*5}  {'-'*4} {'-'*7} {'-'*6} {'-'*6} {'-'*5}")

    param_results = []
    for (hr, mn) in SIGNAL_TIMES:
        for th in THRESHOLDS:
            tdf = backtest_vwap_trend(df_full, hr, mn, th)
            r = evaluate(tdf)
            tag = " ◀ 採用値" if hr == 9 and mn == 30 and th == 50 else ""
            if r:
                param_results.append({'signal_time': f"{hr:02d}:{mn:02d}", 'threshold': th, **r})
                print(f"  {hr:02d}:{mn:02d} {th:>5}  {r['n']:>4} {r['mean']:>+6.1f} "
                      f"{r['wr']:>5.1f}% {r['sharpe']:>+6.2f} {r['t_stat']:>+5.2f}{tag}")
            else:
                print(f"  {hr:02d}:{mn:02d} {th:>5}  N/A{tag}")

    # ── 3. OoS パラメータ安定性 (H2のみ) ────────────────────────────────
    print(f"\n{'=' * 75}")
    print("3. パラメータ安定性  [ENEOS, H2(OoS)のみ]")
    print(f"{'=' * 75}")
    print(f"  {'time':>6} {'th':>5}  {'N':>4} {'Mean':>7} {'WR':>6} {'Shp':>6} {'t':>5}")
    print(f"  {'-'*6} {'-'*5}  {'-'*4} {'-'*7} {'-'*6} {'-'*6} {'-'*5}")

    for (hr, mn) in SIGNAL_TIMES:
        for th in THRESHOLDS:
            tdf = backtest_vwap_trend(df_h2, hr, mn, th)
            r = evaluate(tdf)
            tag = " ◀ 採用値" if hr == 9 and mn == 30 and th == 50 else ""
            if r:
                print(f"  {hr:02d}:{mn:02d} {th:>5}  {r['n']:>4} {r['mean']:>+6.1f} "
                      f"{r['wr']:>5.1f}% {r['sharpe']:>+6.2f} {r['t_stat']:>+5.2f}{tag}")
            else:
                print(f"  {hr:02d}:{mn:02d} {th:>5}   N<5{tag}")

    # ── 4. エネルギーセクター横断 ─────────────────────────────────────────
    print(f"\n{'=' * 75}")
    print("4. エネルギーセクター横断  [signal=09:30, threshold=50bps, Full]")
    print(f"{'=' * 75}")
    for sym, name in TARGET.items():
        df = load_stock(sym, FULL_START, FULL_END)
        if df.empty:
            print(f"  {sym} {name}: データなし")
            continue
        tdf = backtest_vwap_trend(df, 9, 30, 50)
        r = evaluate(tdf)
        tag = " ◀ 本命" if sym == "5020.T" else ""
        if r:
            print(f"  {sym} {name}:{fmt(r)}{tag}")
        else:
            print(f"  {sym} {name}: N/A")

    # ── 5. ENEOS 月別パフォーマンス ───────────────────────────────────────
    print(f"\n{'=' * 75}")
    print("5. 月別パフォーマンス  [ENEOS, signal=09:30, th=50bps]")
    print(f"{'=' * 75}")
    tdf_full = backtest_vwap_trend(df_full, 9, 30, 50)
    if not tdf_full.empty:
        tdf_full['ym'] = pd.to_datetime(tdf_full['date']).dt.to_period('M')
        print(f"  {'月':>8}  {'N':>4} {'Mean':>7} {'WR':>6} {'Total':>8}")
        for ym, grp in tdf_full.groupby('ym'):
            arr = grp['pnl_bps'].values
            wr  = (arr > 0).mean() * 100
            print(f"  {str(ym):>8}  {len(arr):>4} {arr.mean():>+6.1f} {wr:>5.1f}% {arr.sum():>+7.0f}bps")

    # ── 6. ローリングSharpe ────────────────────────────────────────────────
    print(f"\n{'=' * 75}")
    print("6. ローリングSharpe (直近15トレード窓)  [ENEOS, signal=09:30, th=50bps]")
    print(f"{'=' * 75}")
    rs = rolling_sharpe(tdf_full, window=15)
    if not rs.empty:
        print(f"  最小: {rs.min():+.2f}  最大: {rs.max():+.2f}  "
              f"平均: {rs.mean():+.2f}  負の期間: {(rs < 0).sum()}回/{len(rs)}回")
        for d, v in rs.items():
            bar = "█" * int(max(0, v) * 3) if v > 0 else "▒" * int(abs(min(0, v)) * 3)
            sign = "+" if v >= 0 else ""
            print(f"  {d}  {sign}{v:.2f}  {bar}")

    # CSV出力
    pd.DataFrame(param_results).to_csv("param_grid.csv", index=False)
    if not tdf_full.empty:
        tdf_full.to_csv("trades_full.csv", index=False)
    print(f"\n[出力] param_grid.csv / trades_full.csv")


if __name__ == "__main__":
    main()
