"""
LME累積モメンタム × lme_on_copper 発動日重複分析

目的: lme_momentum (LB=10, Th=+3%) と lme_on_copper (+1.0%) の発動日が
      どれだけ重複するかを計測し、Overlay戦略として採用可能かを判断する。

判定基準:
  重複率 < 30%  → 独立したアルファ源 → Overlay採用推奨
  重複率 30-60% → 部分的に重複 → 条件付き採用検討
  重複率 > 60%  → 実質的に同じシグナル → 採用不要
"""
import psycopg2
import pandas as pd
import numpy as np
from datetime import date, time as dtime
import warnings
warnings.filterwarnings("ignore")

PG_CONFIG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}

START = "2025-04-01"
END   = "2026-04-21"
COST_BPS = 4
OUTLIER_PCT = 15.0

BST_PERIODS = [
    (date(2024, 3, 31), date(2024, 10, 27)),
    (date(2025, 3, 30), date(2025, 10, 26)),
    (date(2026, 3, 29), date(2026, 10, 25)),
]

CORE5 = ["5711.T", "6501.T", "7011.T", "5016.T", "4502.T"]

# lme_on_copper パラメータ
LME_ON_THRESHOLD = 1.0   # 当日東京時間変化率 ≥ +1.0%

# lme_momentum パラメータ (複数試す)
MOMENTUM_PARAMS = [
    {"lb": 3,  "th": 2.0},
    {"lb": 3,  "th": 3.0},
    {"lb": 5,  "th": 2.0},
    {"lb": 5,  "th": 3.0},
    {"lb": 10, "th": 2.0},
    {"lb": 10, "th": 3.0},  # ← 候補戦略 (Sharpe+7.55)
    {"lb": 10, "th": 5.0},
]


def is_bst(d):
    for s, e in BST_PERIODS:
        if s <= d < e:
            return True
    return False


def load_lme_intraday():
    """LME銅1分足"""
    conn = psycopg2.connect(**PG_CONFIG)
    q = f"""SELECT timestamp, open, close FROM intraday_data
            WHERE symbol='CMCU3' AND timestamp >= '{START}' AND timestamp < '{END}'
            ORDER BY timestamp"""
    df = pd.read_sql(q, conn); conn.close()
    df['jst'] = pd.to_datetime(df['timestamp']) + pd.Timedelta(hours=9)
    return df.dropna(subset=['close']).set_index('jst').sort_index()


def load_lme_daily():
    """LME銅日足 (intraday_dataから日次終値を生成)"""
    conn = psycopg2.connect(**PG_CONFIG)
    q = f"""SELECT DATE(timestamp + INTERVAL '9 hours') as date,
                   (array_agg(close ORDER BY timestamp DESC))[1] as close_price
            FROM intraday_data
            WHERE symbol='CMCU3' AND timestamp >= '{START}' AND timestamp < '{END}'
            GROUP BY 1 ORDER BY 1"""
    df = pd.read_sql(q, conn); conn.close()
    df['date'] = pd.to_datetime(df['date']).dt.date
    return df.set_index('date').sort_index()


def build_lme_on_signals(lme_intra, exclude_thursday=True):
    """lme_on_copper シグナル日を生成 (東京時間変化率 ≥ threshold%)"""
    signals = {}
    for d in sorted(set(lme_intra.index.date)):
        if date(d.year, d.month, d.day).weekday() >= 5: continue
        if exclude_thursday and date(d.year, d.month, d.day).weekday() == 3: continue
        open_hour = 9 if is_bst(date(d.year, d.month, d.day)) else 10
        ot = pd.Timestamp.combine(d, dtime(open_hour, 0))
        ct = pd.Timestamp.combine(d, dtime(15, 25))
        day = lme_intra[lme_intra.index.date == d]
        after = day[day.index >= ot]
        if after.empty or (after.index[0] - ot).total_seconds() > 1800: continue
        before = day[day.index <= ct]
        if before.empty or (ct - before.index[-1]).total_seconds() > 1800: continue
        move = (before['close'].iloc[-1] / after['open'].iloc[0] - 1) * 100
        signals[d] = move
    return pd.Series(signals)


def build_momentum_signals(lme_daily, lb, th):
    """lme_momentum シグナル日を生成 (過去lb日累積変化率 ≥ th%)"""
    pct = lme_daily['close_price'].pct_change() * 100
    cumulative = pct.rolling(lb).sum()
    fired = {}
    for d, val in cumulative.items():
        if pd.isna(val): continue
        if date(d.year, d.month, d.day).weekday() >= 5: continue
        if date(d.year, d.month, d.day).weekday() == 3: continue  # 木曜除外
        fired[d] = val
    return pd.Series(fired)


def load_stock(sym):
    conn = psycopg2.connect(**PG_CONFIG)
    q = f"""SELECT timestamp, open, close FROM intraday_data
            WHERE symbol='{sym}' AND timestamp >= '{START}' AND timestamp < '{END}'
            ORDER BY timestamp"""
    df = pd.read_sql(q, conn); conn.close()
    df['jst'] = pd.to_datetime(df['timestamp']) + pd.Timedelta(hours=9)
    df = df.dropna(subset=['open','close']).set_index('jst').sort_index()
    h, m = df.index.hour, df.index.minute
    df = df[((h == 9) & (m <= 5)) | ((h == 15) & (m >= 20) & (m <= 30))]
    daily = []
    for d in sorted(set(df.index.date)):
        gd = df[df.index.date == d]
        h2, m2 = gd.index.hour, gd.index.minute
        closes = gd[(h2 == 15) & (m2 >= 20)]
        opens  = gd[(h2 == 9)  & (m2 <= 5)]
        if closes.empty or opens.empty: continue
        daily.append({'date': d, 'close15': closes['close'].iloc[-1],
                      'open9': opens['open'].iloc[0]})
    return pd.DataFrame(daily).set_index('date')


def backtest_basket(signal_dates, stock_data):
    """指定シグナル日でCore5バスケットをONホールド"""
    per_stock = {}
    for sym in CORE5:
        if sym not in stock_data: continue
        jp = stock_data[sym]
        dates = sorted(jp.index)
        trades = []
        for i, d in enumerate(dates[:-1]):
            if d not in signal_dates: continue
            entry = jp.loc[d, 'close15']
            next_d = dates[i + 1]
            exit_p = jp.loc[next_d, 'open9']
            ret_pct = (exit_p / entry - 1) * 100
            if abs(ret_pct) > OUTLIER_PCT: continue
            trades.append({'date': d, 'pnl_bps': ret_pct * 100 - COST_BPS})
        if trades:
            per_stock[sym] = pd.DataFrame(trades).set_index('date')['pnl_bps']

    common = None
    for s in per_stock.values():
        common = set(s.index) if common is None else common & set(s.index)
    if not common: return None

    basket = [{'date': d,
               'pnl_bps': np.mean([per_stock[s][d] for s in per_stock if d in per_stock[s].index])}
              for d in sorted(common)]
    tdf = pd.DataFrame(basket)
    arr = tdf['pnl_bps'].values
    if len(arr) == 0: return None
    mean, std = arr.mean(), arr.std()
    return {
        'n': len(arr),
        'mean': mean,
        'wr': (arr > 0).mean() * 100,
        'pf': arr[arr>0].sum() / abs(arr[arr<=0].sum()) if arr[arr<=0].sum() != 0 else np.inf,
        'sharpe': mean / std * np.sqrt(252) if std > 0 else 0,
        't_stat': mean / (std / np.sqrt(len(arr))) if std > 0 else 0,
        'total': arr.sum(),
    }


def fmt(r):
    if r is None: return "N/A"
    return (f"N={r['n']:>3}, Mean={r['mean']:>+6.1f}bps, WR={r['wr']:>5.1f}%, "
            f"Sharpe={r['sharpe']:>+6.2f}, t={r['t_stat']:>+5.2f}")


def main():
    print("=" * 75)
    print("LME累積モメンタム × lme_on_copper 発動日重複分析")
    print(f"期間: {START} 〜 {END}  コスト: {COST_BPS}bps  木曜除外")
    print("=" * 75)

    # データロード
    print("\n[データロード中...]")
    lme_intra = load_lme_intraday()
    lme_daily = load_lme_daily()
    print(f"  LME1分足: {len(lme_intra)}行 / LME日足: {len(lme_daily)}行")

    stock_data = {}
    for sym in CORE5:
        df = load_stock(sym)
        if not df.empty:
            stock_data[sym] = df
    print(f"  株式データ: {len(stock_data)}銘柄ロード完了")

    # lme_on_copper シグナル
    lme_on_all = build_lme_on_signals(lme_intra)
    lme_on_fired = set(lme_on_all[lme_on_all >= LME_ON_THRESHOLD].index)
    print(f"\n[lme_on_copper] 発動日: {len(lme_on_fired)}日 (threshold≥+{LME_ON_THRESHOLD}%, 木曜除外)")

    # ── 重複分析 ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 75}")
    print("重複率分析 (lme_momentum各パラメータ vs lme_on_copper)")
    print(f"{'=' * 75}")
    print(f"  {'LB':>4} {'Th':>5}  {'N_mom':>6} {'N_both':>7} {'重複率':>8} {'mom_only':>9} {'on_only':>8}")
    print(f"  {'-'*4} {'-'*5}  {'-'*6} {'-'*7} {'-'*8} {'-'*9} {'-'*8}")

    results = []
    for p in MOMENTUM_PARAMS:
        mom_all = build_momentum_signals(lme_daily, p['lb'], p['th'])
        mom_fired = set(mom_all[mom_all >= p['th']].index)

        both     = mom_fired & lme_on_fired
        mom_only = mom_fired - lme_on_fired
        on_only  = lme_on_fired - mom_fired
        overlap_rate = len(both) / len(mom_fired) * 100 if mom_fired else 0

        tag = ""
        if p['lb'] == 10 and p['th'] == 3.0:
            tag = " ← 候補戦略"

        print(f"  {p['lb']:>4} {p['th']:>5.1f}%  {len(mom_fired):>6} {len(both):>7} "
              f"{overlap_rate:>7.1f}%  {len(mom_only):>9} {len(on_only):>8}{tag}")

        results.append({**p, 'n_mom': len(mom_fired), 'n_both': len(both),
                        'overlap_pct': overlap_rate, 'n_mom_only': len(mom_only),
                        'n_on_only': len(on_only), 'mom_fired': mom_fired,
                        'mom_only': mom_only})

    # ── 候補パラメータ (LB=10, Th=+3%) の詳細分析 ──────────────────────
    print(f"\n{'=' * 75}")
    print("詳細分析: LB=10, Th=+3% (候補戦略) × lme_on_copper")
    print(f"{'=' * 75}")

    best = next(r for r in results if r['lb'] == 10 and r['th'] == 3.0)
    mom_only_dates = best['mom_only']
    both_dates     = best['n_both']

    print(f"\n  lme_momentum発動日:    {best['n_mom']}日")
    print(f"  うち lme_on_copper も同日発動: {best['n_both']}日 ({best['overlap_pct']:.1f}%)")
    print(f"  momentum のみ発動 (純増分):   {len(mom_only_dates)}日")
    print(f"  lme_on_copper のみ発動:       {best['n_on_only']}日")

    # 重複日の一覧
    mom_all_best = build_momentum_signals(lme_daily, 10, 3.0)
    mom_fired_best = set(mom_all_best[mom_all_best >= 3.0].index)
    overlap_dates = sorted(mom_fired_best & lme_on_fired)
    print(f"\n  [重複日一覧]")
    for d in overlap_dates:
        on_val  = lme_on_all.get(d, float('nan'))
        mom_val = mom_all_best.get(d, float('nan'))
        print(f"    {d}: LME当日={on_val:+.2f}%, LME累積10日={mom_val:+.2f}%")

    # ── バックテスト比較 ─────────────────────────────────────────────────
    print(f"\n{'=' * 75}")
    print("Core5バスケット ONホールド バックテスト比較")
    print(f"{'=' * 75}")

    # lme_on_copper のみ
    r_on = backtest_basket(lme_on_fired, stock_data)
    print(f"\n  [A] lme_on_copper のみ   : {fmt(r_on)}")

    # lme_momentum (LB=10,Th=3%) のみ
    r_mom = backtest_basket(mom_fired_best, stock_data)
    print(f"  [B] lme_momentum(全発動) : {fmt(r_mom)}")

    # momentum_only (重複除外)
    r_mom_only = backtest_basket(mom_only_dates, stock_data)
    print(f"  [C] momentum_only (純増) : {fmt(r_mom_only)}")

    # OR条件 (どちらか発動)
    or_dates = mom_fired_best | lme_on_fired
    r_or = backtest_basket(or_dates, stock_data)
    print(f"  [D] OR (A∪B, 重複なし)  : {fmt(r_or)}")

    # ── Overlay戦略の試算 ────────────────────────────────────────────────
    print(f"\n{'=' * 75}")
    print("Overlay戦略 採否判定")
    print(f"{'=' * 75}")

    if r_mom_only and r_mom_only['n'] >= 10:
        sharpe_c = r_mom_only['sharpe']
        t_c      = r_mom_only['t_stat']
        if sharpe_c >= 2.0 and t_c >= 2.0:
            verdict = "✅ 採用推奨 — momentum_onlyで独立したエッジあり"
        elif sharpe_c >= 1.5 or t_c >= 1.5:
            verdict = "🔶 要観察 — エッジはあるがt-stat不足"
        else:
            verdict = "❌ 採用不要 — momentum_onlyのエッジ不十分"
    else:
        verdict = "⚠️  サンプル不足 — momentum_only N<10で統計的判断困難"

    print(f"\n  判定: {verdict}")
    if r_mom_only:
        print(f"  根拠: momentum_only Sharpe={r_mom_only['sharpe']:+.2f}, t={r_mom_only['t_stat']:+.2f}, N={r_mom_only['n']}")

    # CSV出力
    overlap_df = pd.DataFrame([{
        'lb': r['lb'], 'th': r['th'], 'n_mom': r['n_mom'],
        'n_both': r['n_both'], 'overlap_pct': r['overlap_pct'],
        'n_mom_only': r['n_mom_only'],
    } for r in results])
    overlap_df.to_csv("overlap_summary.csv", index=False)
    print("\n[出力] overlap_summary.csv")


if __name__ == "__main__":
    main()
