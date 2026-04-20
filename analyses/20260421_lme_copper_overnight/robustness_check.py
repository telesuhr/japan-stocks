"""
ロバストネス検証: 期間分割 前半 vs 後半
オーバーフィットの可能性を排除するため、期間を分けて両方でエッジが出るかチェック
"""
import psycopg2
import pandas as pd
import numpy as np
from datetime import date, time as dtime

PG_CONFIG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}
START = "2025-04-01"
END = "2026-04-21"
OUTLIER_PCT = 15.0
COST_BPS = 4

# 期間分割: 半々に
SPLIT_DATE = date(2025, 10, 20)

BST_PERIODS = [
    (date(2024, 3, 31), date(2024, 10, 27)),
    (date(2025, 3, 30), date(2025, 10, 26)),
    (date(2026, 3, 29), date(2026, 10, 25)),
]


def is_bst(d):
    return any(s <= d < e for s, e in BST_PERIODS)


def load_lme_signals():
    conn = psycopg2.connect(**PG_CONFIG)
    q = f"SELECT timestamp, open, close FROM intraday_data WHERE symbol='CMCU3' AND timestamp >= '{START}' AND timestamp < '{END}' ORDER BY timestamp"
    df = pd.read_sql(q, conn); conn.close()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['jst'] = df['timestamp'] + pd.Timedelta(hours=9)
    df = df.dropna(subset=['close']).set_index('jst').sort_index()
    signals = []
    for d in sorted(set(df.index.date)):
        if d.weekday() >= 5: continue
        oh = 9 if is_bst(d) else 10
        ot = pd.Timestamp.combine(d, dtime(oh, 0))
        ct = pd.Timestamp.combine(d, dtime(15, 25))
        day = df[df.index.date == d]
        if len(day) == 0: continue
        after = day[day.index >= ot]
        if len(after) == 0: continue
        ob = after.iloc[0]
        if (ob.name - ot).total_seconds() > 1800: continue
        before = day[day.index <= ct]
        if len(before) == 0: continue
        cb = before.iloc[-1]
        if (ct - cb.name).total_seconds() > 1800: continue
        signals.append({'date': d, 'move_pct': (cb['close']/ob['open']-1)*100})
    return pd.DataFrame(signals).set_index('date')


def load_jp_daily(sym):
    conn = psycopg2.connect(**PG_CONFIG)
    q = f"SELECT timestamp, open, close FROM intraday_data WHERE symbol='{sym}' AND timestamp >= '{START}' AND timestamp < '{END}' ORDER BY timestamp"
    df = pd.read_sql(q, conn); conn.close()
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
        opens = gd[(h2 == 9) & (m2 <= 5)]
        if len(closes) == 0 or len(opens) == 0: continue
        daily.append({'date': d, 'jp_close': closes['close'].iloc[-1], 'jp_open': opens['open'].iloc[0]})
    return pd.DataFrame(daily).set_index('date')


def backtest_period(signals, jp, threshold, period_start, period_end):
    """periodを限定してバックテスト"""
    trades = []
    jp_dates = sorted(jp.index)
    for i, d in enumerate(jp_dates[:-1]):
        if d < period_start or d >= period_end: continue
        if d not in signals.index: continue
        m = signals.loc[d, 'move_pct']
        if abs(m) < threshold: continue
        direction = np.sign(m)
        entry = jp.loc[d, 'jp_close']
        next_d = jp_dates[i+1]
        exit_p = jp.loc[next_d, 'jp_open']
        on = (exit_p/entry - 1) * 100
        if abs(on) > OUTLIER_PCT: continue
        gross_bps = on * direction * 100
        trades.append({'entry': d, 'exit': next_d, 'move': m, 'dir': int(direction),
                       'pnl_bps': gross_bps - COST_BPS})
    return pd.DataFrame(trades)


def evaluate(tdf):
    if len(tdf) == 0: return None
    arr = tdf['pnl_bps'].values
    pos = arr[arr > 0].sum(); neg = abs(arr[arr <= 0].sum())
    return {
        'n': len(tdf), 'wr': (arr > 0).mean() * 100,
        'pf': pos/neg if neg > 0 else np.inf,
        'mean': arr.mean(), 'total': arr.sum(),
        'sharpe': arr.mean()/arr.std()*np.sqrt(252) if arr.std() > 0 else 0,
    }


SYMBOL_LABELS = {
    "5706.T": "三井金属", "5711.T": "三菱マテリアル", "5713.T": "住友金属鉱山",
    "5016.T": "出光", "1605.T": "INPEX",
    "4502.T": "武田", "4503.T": "アステラス", "4578.T": "大塚HD",
    "6146.T": "ディスコ", "6857.T": "アドバンテスト", "8035.T": "TEL", "4063.T": "信越化学",
    "6501.T": "日立", "7011.T": "三菱重工",
    "9101.T": "日本郵船", "9104.T": "商船三井",
    "6963.T": "ローム", "6098.T": "リクルート", "6702.T": "富士通",
    "8306.T": "三菱UFJ", "6305.T": "日立建機", "5332.T": "TOTO",
    "285A.T": "キオクシア",
}

# 全期間でSharpe上位だったコア銘柄
CORE_STOCKS = [
    "285A.T", "4502.T", "6501.T", "5016.T", "5711.T", "7011.T",
    "6963.T", "4063.T", "5706.T", "1605.T", "6146.T", "6857.T",
    "8306.T", "4503.T", "5713.T", "8035.T", "6305.T", "5332.T",
    "9101.T", "6098.T", "4578.T", "6702.T",
]


def main():
    print("=" * 110)
    print(f"ロバストネス検証: 前半 ({START} ~ {SPLIT_DATE}) vs 後半 ({SPLIT_DATE} ~ {END})")
    print("=" * 110)

    sig = load_lme_signals()
    first_sig = sig[sig.index < SPLIT_DATE]
    second_sig = sig[sig.index >= SPLIT_DATE]
    print(f"\n[シグナル分布]")
    print(f"  前半: N={len(first_sig)}, |move|>=1%: {(first_sig['move_pct'].abs()>=1).sum()}日")
    print(f"  後半: N={len(second_sig)}, |move|>=1%: {(second_sig['move_pct'].abs()>=1).sum()}日")
    print(f"  前半LME累積: {first_sig['move_pct'].sum():+.1f}%")
    print(f"  後半LME累積: {second_sig['move_pct'].sum():+.1f}%")

    # 各銘柄で前半・後半個別にバックテスト
    print("\n" + "=" * 110)
    print("個別銘柄 前半 vs 後半 (th=1.0%)")
    print("=" * 110)
    print(f"{'Symbol':<10} {'Name':<13} | "
          f"{'H1 N':>5} {'H1 WR':>6} {'H1 PF':>5} {'H1 Mean':>8} {'H1 Shp':>7} | "
          f"{'H2 N':>5} {'H2 WR':>6} {'H2 PF':>5} {'H2 Mean':>8} {'H2 Shp':>7} | "
          f"{'Stable':>7}")
    print("-" * 130)

    results = []
    for sym in CORE_STOCKS:
        jp = load_jp_daily(sym)
        if len(jp) == 0: continue
        t1 = backtest_period(sig, jp, 1.0, date(2025, 4, 1), SPLIT_DATE)
        t2 = backtest_period(sig, jp, 1.0, SPLIT_DATE, date(2026, 4, 21))
        r1 = evaluate(t1); r2 = evaluate(t2)
        if r1 and r2:
            # 両期間ともmean>0 かつ両期間でSharpe>=1.0 なら Stable
            stable = "✓" if (r1['mean'] > 0 and r2['mean'] > 0 and r1['sharpe'] >= 1.0 and r2['sharpe'] >= 1.0) else \
                     "△" if (r1['mean'] > 0 and r2['mean'] > 0) else "✗"
            results.append({'symbol': sym, 'name': SYMBOL_LABELS.get(sym, ''),
                            'h1_n': r1['n'], 'h1_mean': r1['mean'], 'h1_sharpe': r1['sharpe'],
                            'h2_n': r2['n'], 'h2_mean': r2['mean'], 'h2_sharpe': r2['sharpe'],
                            'stable': stable})
            print(f"{sym:<10} {SYMBOL_LABELS.get(sym,''):<13} | "
                  f"{r1['n']:>5} {r1['wr']:>5.1f}% {r1['pf']:>5.2f} {r1['mean']:>+7.1f} {r1['sharpe']:>+6.2f} | "
                  f"{r2['n']:>5} {r2['wr']:>5.1f}% {r2['pf']:>5.2f} {r2['mean']:>+7.1f} {r2['sharpe']:>+6.2f} | "
                  f"{stable:>7}")

    # ポートフォリオ検証: コア5銘柄等加重
    print("\n" + "=" * 110)
    print("コア5銘柄等加重ポートフォリオ検証 (Long only)")
    print("=" * 110)
    CORE_BASKET = ["5711.T", "6501.T", "7011.T", "5016.T", "4502.T"]
    print("バスケット: " + ", ".join([f"{s}({SYMBOL_LABELS.get(s, '')})" for s in CORE_BASKET]))

    # Long only (LMEアップ日のみ)でバスケット運用
    def basket_long(signals, threshold, period_start, period_end):
        """日付ごとにバスケット平均PnL"""
        jp_all = {s: load_jp_daily(s) for s in CORE_BASKET}
        per_date = {}
        for s in CORE_BASKET:
            tdf = backtest_period(signals, jp_all[s], threshold, period_start, period_end)
            for _, r in tdf.iterrows():
                if r['dir'] == 1:  # Long only
                    per_date.setdefault(r['entry'], []).append(r['pnl_bps'])
        if not per_date:
            return None
        daily = pd.Series({d: np.mean(v) for d, v in per_date.items()})
        return daily

    for th in [0.5, 0.8, 1.0, 1.5]:
        print(f"\n--- threshold = {th}% (Long only) ---")
        for lbl, s, e in [("全期間", date(2025,4,1), date(2026,4,21)),
                          ("前半H1", date(2025,4,1), SPLIT_DATE),
                          ("後半H2", SPLIT_DATE, date(2026,4,21))]:
            d = basket_long(sig, th, s, e)
            if d is not None and len(d) > 0:
                arr = d.values
                sharpe = arr.mean()/arr.std()*np.sqrt(252) if arr.std() > 0 else 0
                wr = (arr > 0).mean()*100
                print(f"  {lbl}: N_days={len(d):>3}, Mean={arr.mean():>+6.1f}bps, "
                      f"Total={arr.sum():>+6.0f}bps, WR={wr:>5.1f}%, Sharpe={sharpe:+.2f}")
            else:
                print(f"  {lbl}: データなし")

    # 月次リターン分析 (全期間通じてコンスタントにエッジが出ているか)
    print("\n" + "=" * 110)
    print("コア5銘柄Long only (th=1.0%) 月次PnL推移")
    print("=" * 110)
    d_full = basket_long(sig, 1.0, date(2025,4,1), date(2026,4,21))
    if d_full is not None:
        df_m = pd.DataFrame({'pnl': d_full})
        df_m.index = pd.to_datetime(df_m.index)
        monthly = df_m.resample('M').agg(pnl=('pnl','sum'), n=('pnl','count'), mean=('pnl','mean'))
        print(f"{'Month':<10} {'N':>4} {'Total(bps)':>11} {'Mean(bps)':>10}")
        for idx, row in monthly.iterrows():
            mstr = idx.strftime('%Y-%m')
            print(f"{mstr:<10} {int(row['n']):>4} {row['pnl']:>+10.0f} {row['mean']:>+9.1f}")
        pos_months = (monthly['pnl'] > 0).sum()
        tot_months = (monthly['n'] > 0).sum()
        print(f"\n勝ち月/取引月: {pos_months}/{tot_months}")

    # 最大ドローダウン (連敗時のリスク)
    print("\n" + "=" * 110)
    print("コア5銘柄Long (th=1.0%) リスク指標")
    print("=" * 110)
    if d_full is not None:
        cum = d_full.cumsum()
        running_max = cum.cummax()
        dd = cum - running_max
        max_dd = dd.min()
        max_dd_date = dd.idxmin()
        print(f"最大ドローダウン: {max_dd:.0f} bps")
        print(f"最大DD発生日: {max_dd_date}")
        # 連敗
        arr = d_full.values
        max_consec_loss = 0; cur = 0
        for x in arr:
            if x < 0:
                cur += 1
                max_consec_loss = max(max_consec_loss, cur)
            else:
                cur = 0
        print(f"最大連敗日数: {max_consec_loss}")
        print(f"累積PnL: {cum.iloc[-1]:+.0f} bps")
        print(f"年率換算(252営業日): {d_full.mean() * 252:+.0f} bps")


if __name__ == "__main__":
    main()
