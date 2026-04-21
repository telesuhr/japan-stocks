#!/usr/bin/env python3
"""
非鉄・半導体 イントラデイ戦略 並列バックテスト

戦略:
  ① SOX寄付ギャップ・フェード (半導体)
     前日 .SOX ≤ -2% かつ 寄付 gap ≤ -1.5% → 9:05 Long, 9:45 決済
  ② LMEアジア時間 → 非鉄連動スキャルプ
     CMCU3 直近15分変化率 ≥ ±0.4% → 5711等追随, 15分保有
  ③ ORB (Opening Range Breakout) — 半導体
     9:30までのHigh/Lowをブレイク → 引けまでホールド

データ: PostgreSQL market_data.intraday_data (JST = UTC+9)
       .SOX daily: NAS MariaDB (歴史長期)
採用基準: Sharpe≥2.0 & N≥30 & t-stat≥2.0
コスト: 4bps (片道2 × 往復)
"""
import sys, os
from datetime import date, time as dtime
import numpy as np
import pandas as pd
import psycopg2, pymysql
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['Hiragino Sans','Arial Unicode MS','sans-serif']
plt.rcParams['axes.unicode_minus'] = False

PG = dict(host='localhost', port=5432, user='postgres', dbname='market_data')
MARIA = dict(host='100.92.181.92', port=3306, user='rfnews', password='Bleach@924',
             database='refinitiv_news', connect_timeout=10)

SEMI = ['6857.T','6920.T','6146.T','6503.T','8035.T']  # 半導体関連
NONF = ['5711.T','5713.T','5706.T','5801.T','5802.T']  # 非鉄・電線
COST_BPS = 4.0
OUT = os.path.dirname(os.path.abspath(__file__))


def fetch_intraday(sym):
    conn = psycopg2.connect(**PG)
    q = "SELECT timestamp, open, high, low, close, volume FROM intraday_data WHERE symbol=%s ORDER BY timestamp"
    df = pd.read_sql(q, conn, params=[sym])
    conn.close()
    if df.empty: return None
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['jst'] = df['timestamp'] + pd.Timedelta(hours=9)
    df = df.set_index('jst').sort_index()
    df['date'] = df.index.date
    return df


def fetch_sox_daily():
    conn = pymysql.connect(**MARIA)
    df = pd.read_sql("SELECT trade_date, close FROM daily_data WHERE symbol='.SOX' ORDER BY trade_date", conn)
    conn.close()
    df['trade_date'] = pd.to_datetime(df['trade_date']).dt.date
    df = df.sort_values('trade_date')
    df['ret_pct'] = df['close'].pct_change() * 100
    return df.set_index('trade_date')


def stats(r_bps):
    r = np.asarray(r_bps)
    r = r[~np.isnan(r)]
    n = len(r)
    if n < 2: return dict(n=n, mean=np.nan, sharpe=np.nan, tstat=np.nan, wr=np.nan, total=np.nan)
    m, s = r.mean(), r.std()
    return dict(n=n, mean=m, std=s, wr=(r>0).mean()*100,
                sharpe=(m/s)*np.sqrt(252) if s>0 else 0,
                tstat=(m/(s/np.sqrt(n))) if s>0 else 0,
                total=r.sum())


def pretty(name, s):
    return (f"  {name:30} N={s['n']:>4}  mean={s['mean']:+7.2f}bp  WR={s['wr']:>5.1f}%  "
            f"Sharpe={s['sharpe']:+5.2f}  t={s['tstat']:+5.2f}  total={s['total']:+8.0f}bp")


def session_bars(df, d):
    """Day d の前場+後場 (9:00-15:30) バー抽出"""
    day = df[df['date'] == d]
    return day[((day.index.hour == 9) |
                (day.index.hour == 10) |
                ((day.index.hour == 11) & (day.index.minute <= 30)) |
                ((day.index.hour == 12) & (day.index.minute >= 30)) |
                (day.index.hour == 13) |
                (day.index.hour == 14) |
                ((day.index.hour == 15) & (day.index.minute <= 30)))]


# ================================================================
# ① SOX寄付ギャップ・フェード
# ================================================================
def strat_sox_fade(semi_data, sox_daily):
    print("\n" + "="*90)
    print("① SOX寄付ギャップ・フェード  (.SOX<=-2% かつ 寄gap<=-1.5% → 9:05 Long, 9:45 決済)")
    print("="*90)

    # Build prev SOX ret per JP date
    sox = sox_daily.copy()
    sox_dates = np.array(sox.index)
    sox_rets = sox['ret_pct'].values

    def prev_sox(d):
        # d is a date
        mask = sox_dates < d
        if not mask.any(): return np.nan
        return sox_rets[mask][-1]

    results = {}
    for sym in semi_data:
        df = semi_data[sym]
        if df is None: continue
        dates = sorted(df['date'].unique())
        trades = []
        for d in dates:
            day = session_bars(df, d)
            if len(day) < 60: continue
            # Prev day close
            prev_day = df[df['date'] < d]
            if prev_day.empty: continue
            prev_close = float(prev_day['close'].iloc[-1])
            # Open at 9:00 = first bar's open
            open_900 = float(day['open'].iloc[0])
            gap_pct = (open_900 / prev_close - 1) * 100
            # .SOX prev
            s_ret = prev_sox(d)
            if np.isnan(s_ret): continue
            # Signal
            if s_ret > -2.0 or gap_pct > -1.5: continue
            # Entry 9:05 close, Exit 9:45 close
            e = day[(day.index.hour == 9) & (day.index.minute == 5)]
            x = day[(day.index.hour == 9) & (day.index.minute == 45)]
            if e.empty or x.empty: continue
            entry = float(e['close'].iloc[0]); exit_p = float(x['close'].iloc[0])
            ret_bps = (exit_p/entry - 1) * 10000 - COST_BPS  # Long
            trades.append((d, s_ret, gap_pct, entry, exit_p, ret_bps))
        if trades:
            arr = np.array([t[5] for t in trades])
            s = stats(arr); results[sym] = (s, trades)
            print(pretty(sym, s))

    # Aggregate (平均ポートフォリオ等)
    if results:
        all_ret = np.concatenate([np.array([t[5] for t in v[1]]) for v in results.values()])
        s_all = stats(all_ret)
        print(pretty("ALL (pooled)", s_all))
    return results


# ================================================================
# ② LMEアジア時間 → 非鉄連動スキャルプ (lag 2-3分)
# ================================================================
def strat_lme_scalp(nonf_data, lme_data):
    print("\n" + "="*90)
    print("② LMEアジア → 非鉄連動スキャルプ  (CMCU3 直近15分 ≥ ±0.4% → 追随 15分保有)")
    print("="*90)

    if lme_data is None or lme_data.empty:
        print("  CMCU3 データなし → スキップ")
        return {}
    # Resample LME to 1min close, compute 15-min rolling return
    lme_min = lme_data[['close']].copy()
    lme_min['ret15'] = lme_min['close'].pct_change(15) * 100  # 15分変化率

    results = {}
    for sym in nonf_data:
        df = nonf_data[sym]
        if df is None: continue
        dates = sorted(df['date'].unique())
        trades = []
        # Align LME to JP per minute
        for d in dates:
            day = session_bars(df, d)
            if len(day) < 60: continue
            lme_day = lme_min[lme_min.index.date == d]
            if len(lme_day) < 30: continue
            # 当日中で LME |ret15| >= 0.4% が発生した最初の時点 (9:15以降、15:15まで)
            trigger_times = lme_day[(lme_day.index.hour >= 9) & (lme_day.index.hour <= 14) &
                                    (lme_day['ret15'].abs() >= 0.4)]
            if trigger_times.empty: continue
            # 最初のトリガーのみ使用 (日1トレード)
            trig = trigger_times.iloc[0]
            trig_time = trig.name
            direction = 1 if trig['ret15'] > 0 else -1  # 追随
            # エントリー: トリガー+1分後の JP 銘柄 close
            entry_time = trig_time + pd.Timedelta(minutes=1)
            exit_time  = trig_time + pd.Timedelta(minutes=16)
            e = day[day.index >= entry_time]
            x = day[day.index >= exit_time]
            if e.empty or x.empty: continue
            entry = float(e['close'].iloc[0]); exit_p = float(x['close'].iloc[0])
            ret_bps = direction * (exit_p/entry - 1) * 10000 - COST_BPS
            trades.append((d, trig_time.time(), direction, entry, exit_p, ret_bps))
        if trades:
            arr = np.array([t[5] for t in trades])
            s = stats(arr); results[sym] = (s, trades)
            print(pretty(sym, s))

    if results:
        all_ret = np.concatenate([np.array([t[5] for t in v[1]]) for v in results.values()])
        s_all = stats(all_ret)
        print(pretty("ALL (pooled)", s_all))
    return results


# ================================================================
# ③ ORB (Opening Range Breakout)
# ================================================================
def strat_orb(semi_data, window_min=30):
    print("\n" + "="*90)
    print(f"③ ORB (9:00-9:{window_min:02d} レンジブレイク → 引け決済)")
    print("="*90)

    results = {}
    for sym in semi_data:
        df = semi_data[sym]
        if df is None: continue
        dates = sorted(df['date'].unique())
        trades = []
        # 20日平均出来高参照用
        daily_vol = {}
        for d in dates:
            day = session_bars(df, d)
            if len(day) < 60: continue
            daily_vol[d] = day['volume'].sum()
        vol_series = pd.Series(daily_vol).sort_index()
        vol20 = vol_series.rolling(20).mean()

        for d in dates:
            day = session_bars(df, d)
            if len(day) < 60: continue
            open_range = day[day.index < (pd.Timestamp(d) + pd.Timedelta(hours=9, minutes=window_min))]
            if len(open_range) < 5: continue
            or_high = open_range['high'].max()
            or_low = open_range['low'].min()
            rest = day[day.index >= (pd.Timestamp(d) + pd.Timedelta(hours=9, minutes=window_min))]
            if len(rest) < 20: continue
            # ブレイク検出
            vol_ok = vol20.get(d, 0) > 0  # 20日あるか
            entry_bar = None
            direction = 0
            cum_vol = 0
            for ts, row in rest.iterrows():
                cum_vol += row['volume']
                if row['high'] > or_high:
                    entry_bar = ts; direction = 1; break
                if row['low'] < or_low:
                    entry_bar = ts; direction = -1; break
            if entry_bar is None: continue
            entry_price = or_high if direction == 1 else or_low  # ブレイク境界で約定と仮定
            # 決済: 引成
            exit_bar = day.iloc[-1]
            exit_p = float(exit_bar['close'])
            # 損切り: レンジ反対側
            stop = or_low if direction == 1 else or_high
            # 途中でストップ到達?
            after = rest[rest.index >= entry_bar]
            stopped = False
            for ts, row in after.iterrows():
                if direction == 1 and row['low'] <= stop:
                    exit_p = stop; stopped = True; break
                if direction == -1 and row['high'] >= stop:
                    exit_p = stop; stopped = True; break
            ret_bps = direction * (exit_p/entry_price - 1) * 10000 - COST_BPS
            trades.append((d, direction, entry_price, exit_p, ret_bps, stopped))
        if trades:
            arr = np.array([t[4] for t in trades])
            s = stats(arr); results[sym] = (s, trades)
            stop_rate = np.mean([t[5] for t in trades]) * 100
            print(f"{pretty(sym, s)}  [StopHit {stop_rate:.0f}%]")

    if results:
        all_ret = np.concatenate([np.array([t[4] for t in v[1]]) for v in results.values()])
        s_all = stats(all_ret)
        print(pretty("ALL (pooled)", s_all))
    return results


# ================================================================
# Verdict
# ================================================================
def _extract_ret(tr):
    # ret_bps は数値、tupleの要素で float/int かつ bool でないもの最後
    for v in reversed(tr):
        if isinstance(v, bool): continue
        if isinstance(v, (int, float, np.floating)): return float(v)
    return np.nan

def verdict(label, res):
    if not res: return None
    all_ret = []
    for v in res.values():
        for t in v[1]:
            all_ret.append(_extract_ret(t))
    all_ret = np.array(all_ret)
    s = stats(all_ret)
    passed = (s['sharpe'] >= 2 and s['n'] >= 30 and s['tstat'] >= 2)
    print(f"\n  [{label}] Pooled: N={s['n']} Sharpe={s['sharpe']:+.2f} t={s['tstat']:+.2f}  → {'✅ PASS' if passed else '❌ FAIL'}")
    # 銘柄別採用候補
    for sym, (ss, tr) in res.items():
        ok = (ss['sharpe'] >= 2 and ss['n'] >= 30 and ss['tstat'] >= 2)
        if ok:
            print(f"      ✅ {sym}: Sharpe={ss['sharpe']:+.2f} t={ss['tstat']:+.2f} N={ss['n']}")
    return s


def plot_results(r1, r2, r3):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, res, title in [(axes[0], r1, '① SOX寄付フェード Long'),
                           (axes[1], r2, '② LMEアジア→非鉄 追随'),
                           (axes[2], r3, '③ ORB 半導体')]:
        if not res:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title); continue
        for sym, (ss, tr) in res.items():
            rets = np.array([_extract_ret(t) for t in tr])
            cum = np.cumsum(rets)
            ax.plot(cum, label=f"{sym} (N={ss['n']}, Sh={ss['sharpe']:+.1f})", alpha=0.7)
        ax.axhline(0, color='black', lw=0.5)
        ax.set_title(title); ax.set_xlabel('Trade #'); ax.set_ylabel('Cumulative bps')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUT, 'battery_result.png')
    plt.savefig(path, dpi=120, bbox_inches='tight')
    print(f"\nSaved: {path}")


def main():
    print("="*90)
    print("非鉄・半導体 イントラデイ戦略 並列バックテスト")
    print("="*90)

    print("\n[Loading data]")
    sox_daily = fetch_sox_daily()
    print(f"  .SOX daily: N={len(sox_daily)}  {sox_daily.index.min()}〜{sox_daily.index.max()}")

    semi_data = {s: fetch_intraday(s) for s in SEMI}
    nonf_data = {s: fetch_intraday(s) for s in NONF}
    lme_data = fetch_intraday('CMCU3')
    for s, d in {**semi_data, **nonf_data}.items():
        if d is not None:
            print(f"  {s}: {len(d):,} bars  {d.index.min().date()}〜{d.index.max().date()}")
    if lme_data is not None:
        print(f"  CMCU3: {len(lme_data):,} bars  {lme_data.index.min().date()}〜{lme_data.index.max().date()}")

    # Run 3 strategies
    r1 = strat_sox_fade(semi_data, sox_daily)
    r2 = strat_lme_scalp(nonf_data, lme_data)
    r3 = strat_orb(semi_data, window_min=30)

    # ③b ORB Fade (direction反転)
    print("\n" + "="*90)
    print("③b ORB Fade (9:30 ブレイク → 逆張り引けまで)  ※③が負 tで強いため Fade を検証")
    print("="*90)
    r3b = {}
    for sym, (ss, tr) in r3.items():
        # 符号反転 (cost差し引き済なので 2×cost 戻して再計算)
        new_tr = []
        for t in tr:
            ret = _extract_ret(t)
            # 原ret_bps = (dir × move) - cost  → fade = -(dir × move) - cost = -ret_orig - 2*cost
            fade_ret = -ret - 2*COST_BPS
            new_tr.append(tuple(list(t[:-2]) + [fade_ret, t[-1] if isinstance(t[-1], bool) else False]))
        arr = np.array([_extract_ret(t) for t in new_tr])
        s = stats(arr); r3b[sym] = (s, new_tr)
        print(pretty(sym, s))
    if r3b:
        all_ret = np.concatenate([np.array([_extract_ret(t) for t in v[1]]) for v in r3b.values()])
        s_all = stats(all_ret)
        print(pretty("ALL (pooled)", s_all))

    print("\n" + "="*90)
    print("採用判定 (Sharpe≥2 & N≥30 & t-stat≥2)")
    print("="*90)
    verdict('① SOX寄付フェード', r1)
    verdict('② LMEアジア→非鉄', r2)
    verdict('③ ORB半導体 (順張り)', r3)
    verdict('③b ORB Fade (逆張り)', r3b)

    plot_results(r1, r2, r3)


if __name__ == '__main__':
    main()
