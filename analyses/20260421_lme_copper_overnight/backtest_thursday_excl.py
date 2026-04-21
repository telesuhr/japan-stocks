"""
木曜除外フィルタ追加バックテスト

木曜(weekday==3)を除外した場合の Core5 バスケット結果を計算し、
除外前後の数値を比較する。
"""
import psycopg2
import pandas as pd
import numpy as np
from datetime import date, time as dtime

PG_CONFIG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}

START = "2025-04-01"
END   = "2026-04-21"
COST_BPS = 4
OUTLIER_THRESHOLD_PCT = 15.0

BST_PERIODS = [
    (date(2024, 3, 31), date(2024, 10, 27)),
    (date(2025, 3, 30), date(2025, 10, 26)),
    (date(2026, 3, 29), date(2026, 10, 25)),
]

CORE5 = ["5711.T", "6501.T", "7011.T", "5016.T", "4502.T"]
CORE5_NAMES = {"5711.T": "三菱マテリアル", "6501.T": "日立",
               "7011.T": "三菱重工", "5016.T": "出光", "4502.T": "武田"}
THRESHOLD = 1.0


def is_bst(d):
    for s, e in BST_PERIODS:
        if s <= d < e:
            return True
    return False


def load_lme_signals(exclude_thursday=False):
    conn = psycopg2.connect(**PG_CONFIG)
    q = f"""SELECT timestamp, open, close FROM intraday_data
            WHERE symbol='CMCU3' AND timestamp >= '{START}' AND timestamp < '{END}'
            ORDER BY timestamp"""
    df = pd.read_sql(q, conn)
    conn.close()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['jst'] = df['timestamp'] + pd.Timedelta(hours=9)
    df = df.dropna(subset=['close']).set_index('jst').sort_index()
    signals = []
    for d in sorted(set(df.index.date)):
        if d.weekday() >= 5:
            continue
        if exclude_thursday and d.weekday() == 3:
            continue
        open_hour = 9 if is_bst(d) else 10
        open_target = pd.Timestamp.combine(d, dtime(open_hour, 0))
        close_target = pd.Timestamp.combine(d, dtime(15, 25))
        day = df[df.index.date == d]
        if len(day) == 0:
            continue
        after = day[day.index >= open_target]
        if len(after) == 0:
            continue
        ob = after.iloc[0]
        if (ob.name - open_target).total_seconds() > 1800:
            continue
        before = day[day.index <= close_target]
        if len(before) == 0:
            continue
        cb = before.iloc[-1]
        if (close_target - cb.name).total_seconds() > 1800:
            continue
        signals.append({
            'date': d,
            'move_pct': (cb['close'] / ob['open'] - 1) * 100,
            'dow': d.weekday(),
        })
    return pd.DataFrame(signals).set_index('date')


def load_jp_stock(sym):
    conn = psycopg2.connect(**PG_CONFIG)
    q = f"""SELECT timestamp, open, close FROM intraday_data
            WHERE symbol='{sym}' AND timestamp >= '{START}' AND timestamp < '{END}'
            ORDER BY timestamp"""
    df = pd.read_sql(q, conn)
    conn.close()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['jst'] = df['timestamp'] + pd.Timedelta(hours=9)
    df = df.dropna(subset=['open', 'close']).set_index('jst').sort_index()
    h, m = df.index.hour, df.index.minute
    df = df[((h == 9) & (m <= 5)) | ((h == 15) & (m >= 20) & (m <= 30))]
    daily = []
    for d in sorted(set(df.index.date)):
        gd = df[df.index.date == d]
        h2, m2 = gd.index.hour, gd.index.minute
        closes = gd[(h2 == 15) & (m2 >= 20)]
        opens  = gd[(h2 == 9) & (m2 <= 5)]
        if len(closes) == 0 or len(opens) == 0:
            continue
        daily.append({'date': d, 'jp_close': closes['close'].iloc[-1],
                      'jp_open': opens['open'].iloc[0]})
    return pd.DataFrame(daily).set_index('date')


def backtest_single(signals, jp_daily):
    trades = []
    jp_dates = sorted(jp_daily.index)
    for i, d in enumerate(jp_dates[:-1]):
        if d not in signals.index:
            continue
        m = signals.loc[d, 'move_pct']
        if m < THRESHOLD:
            continue
        entry = jp_daily.loc[d, 'jp_close']
        next_d = jp_dates[i + 1]
        exit_p = jp_daily.loc[next_d, 'jp_open']
        ret_pct = (exit_p / entry - 1) * 100
        if abs(ret_pct) > OUTLIER_THRESHOLD_PCT:
            continue
        gross = ret_pct * 100  # bps (Long only)
        trades.append({
            'entry_date': d,
            'exit_date': next_d,
            'lme_move_pct': m,
            'pnl_bps': gross - COST_BPS,
        })
    return pd.DataFrame(trades)


def evaluate(tdf):
    if len(tdf) == 0:
        return None
    arr = tdf['pnl_bps'].values
    wr = (arr > 0).mean() * 100
    pos = arr[arr > 0].sum()
    neg = abs(arr[arr <= 0].sum())
    pf = pos / neg if neg > 0 else np.inf
    mean = arr.mean()
    std  = arr.std()
    sharpe = mean / std * np.sqrt(252) if std > 0 else 0
    t_stat = mean / (std / np.sqrt(len(arr))) if std > 0 else 0
    return {'n': len(tdf), 'wr': wr, 'pf': pf, 'mean_bps': mean,
            'total_bps': arr.sum(), 'sharpe': sharpe, 't_stat': t_stat}


def basket_eval(signals, stock_data):
    """全トレード日でCore5の等加重平均リターンを計算"""
    all_dates = None
    stock_returns = {}
    jp_dates_all = {}

    for sym in CORE5:
        if sym not in stock_data:
            continue
        jp_daily = stock_data[sym]
        jp_dates = sorted(jp_daily.index)
        jp_dates_all[sym] = jp_dates
        trades = []
        for i, d in enumerate(jp_dates[:-1]):
            if d not in signals.index:
                continue
            m = signals.loc[d, 'move_pct']
            if m < THRESHOLD:
                continue
            entry = jp_daily.loc[d, 'jp_close']
            next_d = jp_dates[i + 1]
            exit_p = jp_daily.loc[next_d, 'jp_open']
            ret_pct = (exit_p / entry - 1) * 100
            if abs(ret_pct) > OUTLIER_THRESHOLD_PCT:
                continue
            gross = ret_pct * 100
            trades.append({'entry_date': d, 'pnl_bps': gross - COST_BPS})
        if trades:
            stock_returns[sym] = pd.DataFrame(trades).set_index('entry_date')

    # 全銘柄が揃った日のみ等加重平均
    common_dates = None
    for sym, df in stock_returns.items():
        s = set(df.index)
        common_dates = s if common_dates is None else common_dates & s
    if not common_dates:
        return None

    basket_pnl = []
    for d in sorted(common_dates):
        vals = [stock_returns[sym].loc[d, 'pnl_bps'] for sym in stock_returns if d in stock_returns[sym].index]
        basket_pnl.append({'entry_date': d, 'pnl_bps': np.mean(vals)})

    return evaluate(pd.DataFrame(basket_pnl))


def main():
    print("=" * 70)
    print("LME銅 lme_on_copper 戦略 — 木曜除外前後 比較")
    print(f"閾値: +{THRESHOLD}% / コスト: {COST_BPS}bps / Long Only")
    print("=" * 70)

    # シグナル日の木曜分析
    sig_all = load_lme_signals(exclude_thursday=False)
    sig_excl = load_lme_signals(exclude_thursday=True)
    fired_all  = sig_all[sig_all['move_pct'] >= THRESHOLD]
    fired_excl = sig_excl[sig_excl['move_pct'] >= THRESHOLD]

    thu_fired = fired_all[fired_all['dow'] == 3]
    print(f"\n[シグナル発動日の曜日分析 (閾値{THRESHOLD}%)]")
    print(f"  全発動日: {len(fired_all)} 日")
    print(f"  うち木曜: {len(thu_fired)} 日  {list(thu_fired.index)}")
    print(f"  木曜除外後: {len(fired_excl)} 日")

    # 木曜発動日のトレード個別確認
    print(f"\n[木曜発動日のLME変化率]")
    for d, row in thu_fired.iterrows():
        print(f"  {d} ({['Mon','Tue','Wed','Thu','Fri'][int(row['dow'])]}): LME {row['move_pct']:+.2f}%")

    # 各銘柄のデータロード
    print(f"\n[株式データロード中...]")
    stock_data = {}
    for sym in CORE5:
        stock_data[sym] = load_jp_stock(sym)
        print(f"  {sym} ({CORE5_NAMES[sym]}): {len(stock_data[sym])} 日")

    # 個別銘柄: 木曜除外前後比較
    print(f"\n{'=' * 70}")
    print(f"個別銘柄 比較 (threshold={THRESHOLD}%, Long Only)")
    print(f"{'=' * 70}")
    print(f"  {'Sym':<8} {'Name':<12} {'':>3}  {'N':>3} {'Mean':>7} {'WR':>6} {'PF':>5} {'Shp':>6} {'t':>5}")
    print(f"  {'-'*8} {'-'*12} {'-'*3}  {'-'*3} {'-'*7} {'-'*6} {'-'*5} {'-'*6} {'-'*5}")

    for sym in CORE5:
        name = CORE5_NAMES[sym]
        jp = stock_data[sym]
        for label, sig in [("込み", sig_all), ("除外", sig_excl)]:
            tdf = backtest_single(sig, jp)
            r = evaluate(tdf)
            if r:
                print(f"  {sym:<8} {name:<12} {label:>3}  {r['n']:>3} "
                      f"{r['mean_bps']:>+6.1f} {r['wr']:>5.1f}% {r['pf']:>5.2f} "
                      f"{r['sharpe']:>+6.2f} {r['t_stat']:>+5.2f}")
            else:
                print(f"  {sym:<8} {name:<12} {label:>3}  N/A")

    # バスケット: 木曜除外前後比較
    print(f"\n{'=' * 70}")
    print(f"Core5 バスケット等加重 比較")
    print(f"{'=' * 70}")
    for label, sig in [("木曜込み  ", sig_all), ("木曜除外後", sig_excl)]:
        r = basket_eval(sig, stock_data)
        if r:
            print(f"  {label}: N={r['n']:>3}, Mean={r['mean_bps']:>+7.1f}bps, "
                  f"WR={r['wr']:>5.1f}%, PF={r['pf']:>5.2f}, "
                  f"Sharpe={r['sharpe']:>+6.2f}, t={r['t_stat']:>+5.2f}")
        else:
            print(f"  {label}: データ不足")

    # 木曜のみのパフォーマンス
    print(f"\n{'=' * 70}")
    print(f"木曜発動日のみ パフォーマンス (参考)")
    print(f"{'=' * 70}")
    sig_thu_only = sig_all[(sig_all.index.isin(thu_fired.index))]
    for sym in CORE5:
        name = CORE5_NAMES[sym]
        jp = stock_data[sym]
        tdf = backtest_single(sig_thu_only, jp)
        r = evaluate(tdf)
        if r:
            print(f"  {sym} {name}: N={r['n']}, Mean={r['mean_bps']:>+.1f}bps, "
                  f"Sharpe={r['sharpe']:>+.2f}")
        else:
            print(f"  {sym} {name}: N/A (木曜発動なし)")


if __name__ == "__main__":
    main()
