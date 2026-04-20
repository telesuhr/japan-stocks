"""
LME銅 → 日本銅関連株 オーバーナイト戦略 バックテスト

戦略ロジック:
  1. LME銅(CMCU3)のオープン〜東京引け直前(JST 15:25)の値動きを計測
     - 夏時間(UK BST): LMEオープン = JST 9:00
     - 冬時間(UK GMT): LMEオープン = JST 10:00
  2. |値動き| > 閾値 なら引け(15:30)で日本銅関連株を同方向エントリー
  3. 翌営業日の寄付(9:00)でクローズ

対象日本株:
  5706.T 三井金属 / 5711.T 三菱マテリアル / 5713.T 住友金属鉱山
  5801.T 古河電工 / 5802.T 住友電工 / 5803.T フジクラ

コスト: 片側2bps × 往復 = 4bps
"""
import psycopg2
import pandas as pd
import numpy as np
from datetime import date, time as dtime

PG_CONFIG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}

JP_STOCKS = {
    "5706.T": "Mitsui Kinzoku",
    "5711.T": "Mitsubishi Materials",
    "5713.T": "Sumitomo Metal Mining",
    "5801.T": "Furukawa Electric",
    "5802.T": "Sumitomo Electric",
    "5803.T": "Fujikura",
}

START = "2025-04-01"
END = "2026-04-21"

COST_BPS = 4  # 往復2bps×2


# 英国夏時間 (BST) 期間: 3月最終日曜 〜 10月最終日曜
BST_PERIODS = [
    (date(2024, 3, 31), date(2024, 10, 27)),
    (date(2025, 3, 30), date(2025, 10, 26)),
    (date(2026, 3, 29), date(2026, 10, 25)),
]


def is_bst(d):
    for start, end in BST_PERIODS:
        if start <= d < end:
            return True
    return False


def load_lme_copper():
    conn = psycopg2.connect(**PG_CONFIG)
    q = f"""SELECT timestamp, open, close
            FROM intraday_data
            WHERE symbol='CMCU3' AND timestamp >= '{START}' AND timestamp < '{END}'
            ORDER BY timestamp"""
    df = pd.read_sql(q, conn)
    conn.close()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['jst'] = df['timestamp'] + pd.Timedelta(hours=9)
    df = df.dropna(subset=['close']).set_index('jst').sort_index()
    return df


def load_jp_stock(sym):
    conn = psycopg2.connect(**PG_CONFIG)
    q = f"""SELECT timestamp, open, close
            FROM intraday_data
            WHERE symbol='{sym}' AND timestamp >= '{START}' AND timestamp < '{END}'
            ORDER BY timestamp"""
    df = pd.read_sql(q, conn)
    conn.close()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['jst'] = df['timestamp'] + pd.Timedelta(hours=9)
    df = df.dropna(subset=['open', 'close']).set_index('jst').sort_index()
    # 日本株の取引時間のみ
    h, m = df.index.hour, df.index.minute
    mask = ((h == 9) | (h == 10) | ((h == 11) & (m <= 30)) |
            ((h == 12) & (m >= 30)) | (h == 13) | (h == 14) | ((h == 15) & (m <= 30)))
    return df[mask]


def build_lme_daily_signal(lme):
    """日ごとにLMEのオープン(9:00 or 10:00 JST)と15:25の価格から値動きを計算"""
    signals = []
    # 営業日（月〜金）でループ
    dates = sorted(set(lme.index.date))
    for d in dates:
        if d.weekday() >= 5:
            continue
        open_hour = 9 if is_bst(d) else 10
        open_target = pd.Timestamp.combine(d, dtime(open_hour, 0))
        close_target = pd.Timestamp.combine(d, dtime(15, 25))

        # オープン価格: target以降で最初のバー (±30分以内)
        day_lme = lme[lme.index.date == d]
        if len(day_lme) == 0:
            continue

        # オープン価格取得
        after_open = day_lme[day_lme.index >= open_target]
        if len(after_open) == 0:
            continue
        open_bar = after_open.iloc[0]
        if (open_bar.name - open_target).total_seconds() > 1800:  # 30分以上空いてる
            continue
        open_price = open_bar['open']

        # 15:25価格
        before_close = day_lme[day_lme.index <= close_target]
        if len(before_close) == 0:
            continue
        close_bar = before_close.iloc[-1]
        if (close_target - close_bar.name).total_seconds() > 1800:
            continue
        close_price = close_bar['close']

        move_pct = (close_price / open_price - 1) * 100
        signals.append({
            'date': d, 'bst': is_bst(d),
            'lme_open_time': open_bar.name,
            'lme_close_time': close_bar.name,
            'lme_open': open_price,
            'lme_close': close_price,
            'move_pct': move_pct,
        })
    return pd.DataFrame(signals).set_index('date')


def get_jp_close_open(jp_df):
    """日本株: 各日の引け(15:30 or 15:29の最終バー close)と翌営業日寄付(9:00 open)"""
    daily = []
    for d, g in jp_df.groupby(jp_df.index.date):
        if len(g) == 0:
            continue
        # 引け: 15:30以前の最後のバー
        close_bars = g[(g.index.hour == 15) & (g.index.minute >= 20) & (g.index.minute <= 30)]
        if len(close_bars) == 0:
            continue
        close_price = close_bars['close'].iloc[-1]
        close_time = close_bars.index[-1]

        # 寄付: 9:00〜9:05の最初のバー
        open_bars = g[(g.index.hour == 9) & (g.index.minute <= 5)]
        if len(open_bars) == 0:
            continue
        open_price = open_bars['open'].iloc[0]
        open_time = open_bars.index[0]

        daily.append({
            'date': d, 'jp_open_time': open_time, 'jp_open': open_price,
            'jp_close_time': close_time, 'jp_close': close_price,
        })
    return pd.DataFrame(daily).set_index('date')


def backtest(signals, jp_daily, threshold_pct):
    """
    signals: 日付indexでmove_pct
    jp_daily: 日付indexでjp_close, 翌日jp_open
    threshold: |move_pct|がこれを超えたらエントリー
    """
    trades = []
    jp_dates = sorted(jp_daily.index)
    for i, d in enumerate(jp_dates[:-1]):
        if d not in signals.index:
            continue
        sig = signals.loc[d]
        if abs(sig['move_pct']) < threshold_pct:
            continue
        direction = np.sign(sig['move_pct'])
        # エントリー: 当日引け
        entry_price = jp_daily.loc[d, 'jp_close']
        # 決済: 次の営業日寄付
        next_d = jp_dates[i + 1]
        exit_price = jp_daily.loc[next_d, 'jp_open']

        ret = (exit_price / entry_price - 1) * direction * 10000  # bps
        pnl_bps = ret - COST_BPS
        trades.append({
            'entry_date': d, 'exit_date': next_d,
            'lme_move_pct': sig['move_pct'],
            'direction': int(direction),
            'entry_price': entry_price, 'exit_price': exit_price,
            'gross_bps': ret, 'pnl_bps': pnl_bps,
        })
    return pd.DataFrame(trades)


def evaluate(tdf, label):
    if len(tdf) == 0:
        return None
    arr = tdf['pnl_bps'].values
    wr = (arr > 0).mean() * 100
    pos = arr[arr > 0].sum()
    neg = abs(arr[arr <= 0].sum())
    pf = pos / neg if neg > 0 else np.inf
    mean = arr.mean()
    total = arr.sum()
    sharpe = mean / arr.std() * np.sqrt(252) if arr.std() > 0 else 0
    return {
        'label': label, 'n': len(tdf), 'wr': wr, 'pf': pf,
        'mean_bps': mean, 'total_bps': total, 'sharpe': sharpe,
    }


def main():
    print("=" * 90)
    print("LME銅 → 日本銅関連株 オーバーナイト戦略")
    print(f"期間: {START} 〜 {END}")
    print("=" * 90)

    print("\n[1] LME銅データロード")
    lme = load_lme_copper()
    print(f"  CMCU3: {len(lme):,} bars, {lme.index.min()} ~ {lme.index.max()}")

    print("\n[2] LMEオープン→15:25 値動き計算")
    signals = build_lme_daily_signal(lme)
    print(f"  シグナル日数: {len(signals)}")
    print(f"  BST(夏時間, LME open 9:00): {signals['bst'].sum()}日")
    print(f"  GMT(冬時間, LME open 10:00): {(~signals['bst']).sum()}日")
    print(f"\n  値動き分布 (%):")
    print(signals['move_pct'].describe().round(3).to_string())
    print(f"\n  |move| 分布:")
    for th in [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]:
        n = (signals['move_pct'].abs() >= th).sum()
        print(f"    |move| >= {th}%: {n}日 ({n/len(signals)*100:.1f}%)")

    # 各銘柄ごとにバックテスト
    print("\n[3] 日本株別バックテスト")
    print("=" * 90)
    all_results = []
    for sym, name in JP_STOCKS.items():
        print(f"\n--- {sym} ({name}) ---")
        jp_df = load_jp_stock(sym)
        jp_daily = get_jp_close_open(jp_df)
        print(f"  日本株デイリー: {len(jp_daily)}日")
        print(f"  {'Threshold':>10} {'N':>5} {'WR':>6} {'PF':>5} {'Mean':>8} {'Total':>9} {'Sharpe':>7}")
        for th in [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]:
            tdf = backtest(signals, jp_daily, th)
            r = evaluate(tdf, f"{sym}_th{th}")
            if r:
                r.update({'symbol': sym, 'name': name, 'threshold': th})
                all_results.append(r)
                print(f"  {th:>9.2f}% {r['n']:>5} {r['wr']:>5.1f}% {r['pf']:>5.2f} "
                      f"{r['mean_bps']:>+7.2f} {r['total_bps']:>+8.0f} {r['sharpe']:>+6.2f}")

    # STEP 4: ロングのみ / ショートのみ分離
    print("\n\n[4] 方向別分析 (th=1.0% での検証)")
    print("=" * 90)
    print(f"  {'Symbol':<10} {'Dir':>5} {'N':>5} {'WR':>6} {'PF':>5} {'Mean':>8} {'Total':>9}")
    for sym, name in JP_STOCKS.items():
        jp_df = load_jp_stock(sym)
        jp_daily = get_jp_close_open(jp_df)
        tdf = backtest(signals, jp_daily, 1.0)
        if len(tdf) == 0:
            continue
        for dir_val, dir_lbl in [(1, 'Long'), (-1, 'Short')]:
            sub = tdf[tdf.direction == dir_val]
            r = evaluate(sub, f"{sym}_{dir_lbl}")
            if r:
                print(f"  {sym:<10} {dir_lbl:>5} {r['n']:>5} {r['wr']:>5.1f}% {r['pf']:>5.2f} "
                      f"{r['mean_bps']:>+7.2f} {r['total_bps']:>+8.0f}")

    # STEP 5: 等加重ポートフォリオ (全6銘柄を同時エントリー)
    print("\n\n[5] 等加重ポートフォリオ (全6銘柄同時エントリー)")
    print("=" * 90)
    print(f"  {'Threshold':>10} {'N_days':>7} {'Mean':>8} {'Total':>9} {'Sharpe':>7} {'WR':>6}")
    # 日次PnL (全銘柄平均)
    jp_dailies = {sym: get_jp_close_open(load_jp_stock(sym)) for sym in JP_STOCKS}
    for th in [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]:
        per_day = {}  # date -> list of pnl
        for sym, jp_daily in jp_dailies.items():
            tdf = backtest(signals, jp_daily, th)
            for _, row in tdf.iterrows():
                per_day.setdefault(row['entry_date'], []).append(row['pnl_bps'])
        if not per_day:
            continue
        daily_pnl = pd.Series({d: np.mean(v) for d, v in per_day.items()})
        arr = daily_pnl.values
        mean = arr.mean()
        total = arr.sum()
        sharpe = mean / arr.std() * np.sqrt(252) if arr.std() > 0 else 0
        wr = (arr > 0).mean() * 100
        print(f"  {th:>9.2f}% {len(arr):>7} {mean:>+7.2f} {total:>+8.0f} {sharpe:>+6.2f} {wr:>5.1f}%")


if __name__ == "__main__":
    main()
