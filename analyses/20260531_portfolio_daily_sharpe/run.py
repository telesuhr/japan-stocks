"""
戦略バスケットの日次ポートフォリオSharpe — 価格ベース4戦略の合成

Sharpe監査 (20260531_strategy_sharpe_audit) の結論:
「per-trade×√252 はstandalone年率を過大評価。正しくは日次ポートフォリオ
収益系列のSharpe、または6戦略バスケット全体の資金曲線で評価すべき」

本スクリプトは価格ベース4戦略 (eneos / vwap_morning / lasertec / bank_absorption)
を忠実に再構築し、(1)各戦略の日次ポートフォリオSharpe、(2)等加重バスケットの
合成Sharpe を測る。決算系2戦略 (earnings_pead 日次Sharpe≈1.1-1.3 既出 /
pre_earnings_drift) は fin_summary 依存で別途。

各戦略ロジックは 20260511_strategy_validation_jquants/run.py に準拠。
日次sleeve方式: 各戦略=等資金。日々sleeve収益=その日にアクティブな
トレードの per-day収益 (総リターン÷保有日) の平均。バスケット=sleeve等加重平均。
Sharpe = 日次収益 mean/std × √252。
"""
from __future__ import annotations

import os
import sys
import csv
import numpy as np
import pandas as pd
import psycopg2

sys.stdout.reconfigure(line_buffering=True)
PG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}
COST_BPS = 2.0
INTRADAY_START = "2024-05-09"
DAILY_START = "2021-05-09"
HERE = os.path.dirname(__file__)


def load_intraday(codes, start, end='2027-01-01'):
    conn = psycopg2.connect(**PG)
    ph = ','.join(['%s'] * len(codes))
    df = pd.read_sql(
        f"SELECT code, ts, open, high, low, close, volume FROM stocks_intraday "
        f"WHERE code IN ({ph}) AND ts >= %s AND ts < %s ORDER BY code, ts",
        conn, params=tuple(codes) + (start, end))
    conn.close()
    df['ts'] = pd.to_datetime(df['ts'])
    df['dt'] = df['ts'].dt.normalize()
    return df


def load_daily(codes, start=DAILY_START):
    conn = psycopg2.connect(**PG)
    ph = ','.join(['%s'] * len(codes))
    df = pd.read_sql(
        f"SELECT code, date, open, high, low, close, volume, turnover_value, "
        f"adj_open, adj_close FROM stocks_daily WHERE code IN ({ph}) AND date >= %s "
        f"ORDER BY code, date", conn, params=tuple(codes) + (start,))
    conn.close()
    df['date'] = pd.to_datetime(df['date'])
    return df


# 各戦略は (entry_date, hold_days, ret_bps) のリストを返す
def s_eneos():
    df = load_intraday(['50200'], INTRADAY_START)
    out = []
    for dt, g in df.groupby('dt'):
        if dt.weekday() >= 5: continue
        g = g.sort_values('ts')
        morning = g[g['ts'].dt.hour >= 9]
        if len(morning) < 5: continue
        vol = morning['volume'].fillna(0).clip(lower=1)
        vwap_s = (morning['close'] * vol).cumsum() / vol.cumsum()
        bar = morning[(morning['ts'].dt.hour == 9) & (morning['ts'].dt.minute == 30)]
        if bar.empty: continue
        c = float(bar['close'].iloc[-1]); v = float(vwap_s.loc[bar.index[-1]])
        if v <= 0: continue
        dev = (c / v - 1) * 10000
        if abs(dev) < 50: continue
        d = 1 if dev > 0 else -1
        hhmm = g['ts'].dt.hour * 60 + g['ts'].dt.minute
        after = g[hhmm > 9 * 60 + 30]; ex = g[hhmm >= 15 * 60 + 20]
        if after.empty or ex.empty: continue
        ep = float(after['close'].iloc[0]); xp = float(ex['close'].iloc[-1])
        if ep <= 0 or xp <= 0: continue
        ret = d * (xp / ep - 1) * 10000 - COST_BPS * 2
        out.append((dt, 1, ret))
    return out


def s_vwap_morning():
    TARGETS = ['80350', '61460', '69200']
    df_all = load_intraday(TARGETS, INTRADAY_START)
    out = []
    for sym in TARGETS:
        df = df_all[df_all['code'] == sym]
        for dt, g in df.groupby('dt'):
            if dt.weekday() >= 5: continue
            g = g.sort_values('ts').copy()
            h, m = g['ts'].dt.hour, g['ts'].dt.minute
            session = g[((h == 9) | (h == 10) | ((h == 11) & (m <= 30)) |
                         ((h == 12) & (m >= 30)) | (h == 13) | (h == 14) | ((h == 15) & (m <= 30)))]
            if len(session) < 10: continue
            vol = session['volume'].fillna(0).clip(lower=1)
            session = session.copy()
            session['vwap'] = ((session['close'] * vol).cumsum() / vol.cumsum()).values
            session['dev'] = (session['close'] / session['vwap'] - 1) * 10000
            session['hhmm'] = session['ts'].dt.hour * 60 + session['ts'].dt.minute
            window = session[(session['hhmm'] >= 600) & (session['hhmm'] <= 690)]
            trig = window[window['dev'].abs() >= 275]
            if trig.empty: continue
            row = trig.iloc[0]; dev = float(row['dev'])
            d = -1 if dev > 0 else 1
            ep = float(row['close'])
            if ep <= 0: continue
            stop = ep * (1 + 400 / 10000) if d == -1 else ep * (1 - 400 / 10000)
            after = session[session['ts'] >= row['ts']]
            xp = None
            for _, b in after.iterrows():
                if d == -1 and b['high'] >= stop: xp = stop; break
                if d == 1 and b['low'] <= stop: xp = stop; break
            if xp is None:
                ex = session[session['hhmm'] >= 920]
                xp = float(ex['close'].iloc[-1]) if not ex.empty else float(session['close'].iloc[-1])
            ret = d * (xp / ep - 1) * 10000 - COST_BPS * 2
            out.append((dt, 1, ret))
    return out


def s_lasertec():
    df = load_daily(['69200']).set_index('date').sort_index()
    df = df.astype({c: float for c in ['open', 'high', 'low', 'close']})
    df['ma25'] = df['close'].rolling(25).mean()
    df['ma25_5d_ago'] = df['ma25'].shift(5)
    df['hh20'] = df['close'].rolling(20).max()
    df['dd20'] = (df['close'] / df['hh20'] - 1) * 100
    lo, hi, ma = df['low'], df['high'], df['ma25']
    df['touched'] = (lo <= ma * 1.01) & (hi >= ma * 0.99)
    df['signal'] = df['touched'] & (df['dd20'] <= -5.0) & (df['ma25'] > df['ma25_5d_ago']) & df['ma25'].notna()
    HOLD = 10; CD = 10
    out = []; last_entry = None
    for idx in range(len(df)):
        if not df.iloc[idx]['signal']: continue
        sig_dt = df.index[idx]
        if last_entry is not None and (sig_dt - last_entry).days < CD: continue
        if idx + 1 >= len(df): continue
        entry = float(df.iloc[idx + 1]['open'])
        if entry <= 0 or np.isnan(entry): continue
        stop_level = entry * 0.90
        entry_dt = df.index[idx + 1]; last_entry = entry_dt
        fut = df.iloc[idx + 1:idx + 2 + HOLD]
        xp = None
        for _, r in fut.iloc[1:].iterrows():
            if r['low'] <= stop_level: xp = stop_level; break
        if xp is None:
            xp = float(fut.iloc[HOLD]['close']) if len(fut) > HOLD else float(fut.iloc[-1]['close'])
        ret = (xp / entry - 1) * 10000 - COST_BPS * 2
        out.append((entry_dt, HOLD, ret))
    return out


def s_bank():
    codes = []
    with open(os.path.join(HERE, '..', '..', 'strategies', 'bank_absorption', 'whitelist.csv')) as f:
        for r in csv.DictReader(f):
            c = str(r.get('code5') or '').strip()
            if c.isdigit():
                if len(c) == 4: c += '0'
                codes.append(c)
    df = load_daily(codes).sort_values(['code', 'date']).reset_index(drop=True)
    out = []
    for sym in codes:
        sd = df[df['code'] == sym].copy().reset_index(drop=True)
        if len(sd) < 30: continue
        sd['vol_ma20'] = sd['volume'].rolling(20).mean()
        sd['adj_ret'] = (sd['adj_close'] / sd['adj_open'] - 1)
        for i in range(20, len(sd) - 6):
            r = sd.iloc[i]
            if pd.isna(r['vol_ma20']) or r['vol_ma20'] <= 0: continue
            if r['volume'] < 1.5 * r['vol_ma20']: continue
            if r['adj_ret'] >= 0: continue
            if r['turnover_value'] < 10e8: continue
            entry_row = sd.iloc[i + 1]
            exit_row = sd.iloc[i + 5] if i + 5 < len(sd) else sd.iloc[-1]
            ep = float(entry_row['adj_open']); xp = float(exit_row['adj_close'])
            if ep <= 0 or xp <= 0: continue
            ret = (xp / ep - 1) * 10000 - COST_BPS * 2
            out.append((pd.Timestamp(entry_row['date']), 5, ret))
    return out


print("=" * 78)
print("戦略バスケット 日次ポートフォリオSharpe — 価格ベース4戦略")
print("=" * 78)
print("\n[各戦略のトレード再構築中]")

STRATS = {}
for name, fn in [('eneos_vwap_trend', s_eneos), ('vwap_morning_meanrevert', s_vwap_morning),
                 ('lasertec_ma25_support', s_lasertec), ('bank_absorption', s_bank)]:
    trades = fn()
    STRATS[name] = trades
    print(f"  {name}: {len(trades)} トレード")

# 共通営業日カレンダー
conn = psycopg2.connect(**PG)
cal = pd.read_sql("SELECT DISTINCT date FROM stocks_daily WHERE date >= %s ORDER BY date",
                  conn, params=(DAILY_START,))
conn.close()
cal['date'] = pd.to_datetime(cal['date'])
all_days = cal['date'].values


def sleeve_daily_returns(trades, days):
    """各営業日のsleeve収益 (アクティブトレードの per-day収益 平均, bps)。"""
    contrib = {pd.Timestamp(d): [] for d in days}
    day_index = pd.DatetimeIndex(days)
    for entry_dt, hold, ret in trades:
        entry_dt = pd.Timestamp(entry_dt).normalize()
        per_day = ret / max(hold, 1)
        pos = day_index.searchsorted(entry_dt)
        for k in range(max(hold, 1)):
            j = pos + k
            if j < len(day_index):
                contrib[day_index[j]].append(per_day)
    ser = pd.Series({d: (np.mean(v) if v else 0.0) for d, v in contrib.items()}).sort_index()
    return ser / 10000.0  # bps → 比率


def sharpe(s):
    s = s.dropna()
    if len(s) < 10 or s.std() == 0: return float('nan')
    return float(s.mean() / s.std() * np.sqrt(252))


print("\n" + "=" * 78)
print("A. 各戦略の日次ポートフォリオSharpe (sleeve単独)")
print("=" * 78)
print(f"\n  {'戦略':<26} {'活動日数':>8} {'日次Sh':>9} {'年率収益%':>10} {'日次平均bps':>12}")
print("  " + "-" * 68)

sleeves = {}
for name, trades in STRATS.items():
    ser = sleeve_daily_returns(trades, all_days)
    # 戦略のデータ存在期間に限定 (intradayは2024-05〜)
    active = ser[ser != 0]
    if len(active) == 0:
        continue
    start = active.index.min(); end = active.index.max()
    ser_p = ser[(ser.index >= start) & (ser.index <= end)]
    sleeves[name] = ser_p
    sh = sharpe(ser_p)
    ann_ret = ser_p.mean() * 252 * 100
    n_active = (ser_p != 0).sum()
    print(f"  {name:<26} {n_active:>8} {sh:>9.2f} {ann_ret:>10.1f} {ser_p.mean()*10000:>12.2f}")

print("\n" + "=" * 78)
print("B. 等加重バスケット 合成Sharpe")
print("=" * 78)

# 共通期間 (全sleeveがデータを持つ期間 = intraday開始以降)
common_start = max(s.index.min() for s in sleeves.values())
common_end = min(s.index.max() for s in sleeves.values())
print(f"\n  共通期間: {common_start.date()} 〜 {common_end.date()}")

aligned = pd.DataFrame({name: s.reindex(all_days).fillna(0.0) for name, s in sleeves.items()})
aligned = aligned[(aligned.index >= common_start) & (aligned.index <= common_end)]

# 等加重 (1/N)
basket = aligned.mean(axis=1)
print(f"\n  {'構成':<40} {'日次Sh':>9} {'年率収益%':>10} {'年率vol%':>10}")
print("  " + "-" * 70)
sh_b = sharpe(basket)
print(f"  {'4戦略 等加重':<40} {sh_b:>9.2f} {basket.mean()*252*100:>10.1f} {basket.std()*np.sqrt(252)*100:>10.1f}")

# 相関
print("\n  sleeve間 相関行列:")
corr = aligned.corr()
print(corr.round(2).to_string())

# 個別Sharpeの単純平均 vs バスケット (分散効果)
indiv_sh = [sharpe(aligned[c]) for c in aligned.columns]
print(f"\n  個別Sharpe平均: {np.nanmean(indiv_sh):.2f}  →  バスケットSharpe: {sh_b:.2f}")
print(f"  分散効果による改善: {sh_b - np.nanmean(indiv_sh):+.2f}")

print("\n" + "=" * 78)
print("C. 決算系2戦略を含めた展望 (参考)")
print("=" * 78)
print(f"""
  earnings_pead 日次ポートフォリオSharpe = 1.10〜1.32 (20260512検証 line340, 同時保有3-10銘柄)
  pre_earnings_drift = √(252/4)で報告2.07 (誠実な年率化、日次未測定)

  上記4戦略バスケット (Sharpe {sh_b:.2f}) に低相関の決算系2戦略を加えれば、
  6戦略バスケットの合成Sharpeはさらに向上する見込み。
  → これが「恒久エンジン」の真の評価軸。headline個別√252値ではなくバスケット資金曲線で見る。
""")

# 保存
aligned.to_csv(os.path.join(HERE, "sleeve_daily_returns.csv"))
pd.DataFrame({'basket': basket}).to_csv(os.path.join(HERE, "basket_daily.csv"))
print(f"  保存: sleeve_daily_returns.csv, basket_daily.csv")
print("\n完了")
