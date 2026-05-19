"""
マトリクス戦略 5年検証 (2021-05〜2026-05)
- 過去5年で戦略が機能するか
- 年次別パフォーマンス分解
- Walk-forward的に12ヶ月rollingで Sharpe安定性確認
"""
import sys, os, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import psycopg2
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

PG_CONFIG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}

SEMI = {
    '69200': 'レーザーテック', '80350': '東京エレクトロン', '68570': 'アドバンテスト',
    '61460': 'ディスコ', '40630': '信越化学', '69630': 'ローム',
    '77350': 'SCREEN', '34360': 'SUMCO', '65260': 'ソシオネクスト', '99840': 'SBG',
}
NONFER = {
    '57130': '住友金属鉱山', '57110': '三菱マテリアル', '57060': '三井金属',
    '57140': 'DOWA', '57150': '古河機械金属', '57270': '東邦チタニウム',
}
ALL = {**SEMI, **NONFER}
START = '2021-05-13'
END   = '2026-05-17'
COST_BPS = 4.0

# ── 5年データ取得 ──────────────────────
conn = psycopg2.connect(**PG_CONFIG)
placeholders = ','.join(f"'{c}'" for c in ALL)
daily = pd.read_sql(f"""
    SELECT code, date, open, close
    FROM stocks_daily
    WHERE code IN ({placeholders})
      AND date >= '{START}' AND date <= '{END}'
    ORDER BY code, date
""", conn)
conn.close()
daily['date'] = pd.to_datetime(daily['date']).dt.tz_localize(None)
print(f"取得: {len(daily):,}行 / {daily['date'].min().date()}〜{daily['date'].max().date()}")

def compute(g):
    g = g.sort_values('date').reset_index(drop=True)
    g['prev_close'] = g['close'].shift(1)
    g['on_gap']         = (g['open']  / g['prev_close'] - 1) * 100
    g['day_open_close'] = (g['close'] / g['open']       - 1) * 100
    g['full_day']       = (g['close'] / g['prev_close'] - 1) * 100
    g['dow']            = g['date'].dt.dayofweek
    g['sector']         = 'semi' if g['code'].iloc[0] in SEMI else 'nonfer'
    return g

result = daily.groupby('code', group_keys=False).apply(compute).dropna(subset=['on_gap'])
result = result[result['dow'] <= 4]

# ── 戦略リターン抽出 ────────────────
def strat_daily(dow, sector, session, sign=1):
    sub = result[(result['dow']==dow) & (result['sector']==sector)]
    return sub.groupby('date')[session].mean() * sign - COST_BPS/100

# ── 戦略定義 ───────────────────────
all_dates = sorted(result['date'].unique())
date_index = pd.DatetimeIndex(all_dates)

# Buy&Hold
def buyhold():
    return result.groupby('date')['full_day'].mean().reindex(date_index).fillna(0) - COST_BPS/100/len(date_index)*2

# Matrix
def matrix():
    comps = [
        (1, 'nonfer', 'on_gap',         1, 1/6),
        (2, 'nonfer', 'full_day',       1, 1/6),
        (3, 'nonfer', 'on_gap',         1, 1/6),
        (4, 'semi',   'day_open_close', 1, 1/6),
        (4, 'nonfer', 'day_open_close', 1, 1/6),
        (0, 'semi',   'on_gap',        -1, 1/6),
    ]
    series = []
    for dow, sec, sess, sign, w in comps:
        series.append(strat_daily(dow, sec, sess, sign).reindex(date_index).fillna(0) * w)
    return sum(series)

# Aggressive L/S
def aggressive():
    comps = [
        (1, 'nonfer', 'on_gap',         1, 1/9),
        (2, 'nonfer', 'full_day',       1, 1/9),
        (3, 'nonfer', 'on_gap',         1, 1/9),
        (4, 'semi',   'day_open_close', 1, 1/9),
        (4, 'nonfer', 'day_open_close', 1, 1/9),
        (1, 'semi',   'on_gap',         1, 1/9),
        (3, 'semi',   'on_gap',         1, 1/9),
        (0, 'semi',   'on_gap',        -1, 1/9),
        (4, 'semi',   'on_gap',        -1, 1/9),
    ]
    series = []
    for dow, sec, sess, sign, w in comps:
        series.append(strat_daily(dow, sec, sess, sign).reindex(date_index).fillna(0) * w)
    return sum(series)

# Monday ON-Down Buy
def monday_ondown():
    """月曜ONが-0.5%以下のときだけ寄付買い→引け売り"""
    daily_pnl = pd.Series(0.0, index=date_index)
    for sec_code in ['semi', 'nonfer']:
        sub = result[(result['dow']==0) & (result['sector']==sec_code)]
        on_gap_avg = sub.groupby('date')['on_gap'].mean()
        sess_avg   = sub.groupby('date')['day_open_close'].mean()
        signal = (on_gap_avg < -0.5)
        pnl = sess_avg.where(signal, 0) - COST_BPS/100 * signal.astype(int)
        daily_pnl = daily_pnl.add(pnl * 0.5, fill_value=0)  # 等加重で半分ずつ
    return daily_pnl

strats = {
    'Baseline (Buy&Hold)':  buyhold(),
    'Matrix (6コンポーネント)': matrix(),
    'Aggressive L/S (9コンポーネント)': aggressive(),
    'Monday ON-Down Buy (条件付き)': monday_ondown(),
}

# ── パフォーマンス計算 ──────────────
def perf(daily_ret):
    if len(daily_ret) == 0: return None
    cum = ((1 + daily_ret/100).cumprod() - 1) * 100
    final = cum.iloc[-1]
    n_days = len(daily_ret)
    ann = ((1 + final/100) ** (252/n_days) - 1) * 100 if (1+final/100)>0 else -100
    vol = daily_ret.std() * np.sqrt(252)
    active = daily_ret[daily_ret != 0]
    sh = active.mean()/active.std()*np.sqrt(252) if len(active)>0 and active.std()>0 else 0
    cum_ser = (1+daily_ret/100).cumprod()
    dd = (cum_ser-cum_ser.cummax())/cum_ser.cummax()*100
    mdd = dd.min()
    return dict(final=final, ann=ann, vol=vol, sharpe=sh, mdd=mdd, cum=cum)

print("\n" + "="*80)
print("【5年通算 (2021-05〜2026-05)】")
print("="*80)
print(f"\n{'戦略':40} {'累積':>10} {'年率':>8} {'σ':>7} {'Sharpe':>8} {'MDD':>8}")
print("-"*80)
overall = {}
for name, daily_ret in strats.items():
    p = perf(daily_ret)
    overall[name] = p
    print(f"{name:40} {p['final']:>+9.1f}% {p['ann']:>+7.1f}% "
          f"{p['vol']:>6.1f}% {p['sharpe']:>+7.2f} {p['mdd']:>+7.1f}%")

# ── 年次別パフォーマンス ──────────
print("\n" + "="*80)
print("【年次別 Sharpe】")
print("="*80)
yearly = {}
years = sorted(set(date_index.year))
print(f"\n{'戦略':40}", end='')
for y in years:
    print(f"{y:>7}", end='')
print()
print("-"*80)
for name, daily_ret in strats.items():
    yearly[name] = {}
    print(f"{name:40}", end='')
    for y in years:
        ys = daily_ret[daily_ret.index.year == y]
        if len(ys) < 20:
            print(f"{'--':>7}", end='')
            continue
        active = ys[ys != 0]
        if len(active) > 0 and active.std() > 0:
            sh = active.mean()/active.std()*np.sqrt(252)
            yearly[name][y] = sh
        else:
            sh = 0
            yearly[name][y] = 0
        print(f"{sh:>+6.2f}", end=' ')
    print()

print(f"\n{'戦略':40}", end='')
for y in years:
    print(f"{y:>7}", end='')
print()
print("-"*80)
print("年次累積リターン:")
for name, daily_ret in strats.items():
    print(f"{name:40}", end='')
    for y in years:
        ys = daily_ret[daily_ret.index.year == y]
        if len(ys) < 20:
            print(f"{'--':>7}", end='')
            continue
        cum = ((1+ys/100).cumprod().iloc[-1] - 1) * 100
        print(f"{cum:>+5.1f}% ", end='')
    print()

# ── ローリング12ヶ月Sharpe ──────────
print("\n" + "="*80)
print("【12ヶ月ローリングSharpe (Aggressive)】")
print("="*80)
rolling_data = {}
for name, daily_ret in strats.items():
    if 'Aggressive' not in name and 'Matrix' not in name: continue
    rolling_sh = []
    rolling_dates = []
    for i in range(252, len(daily_ret), 21):
        window = daily_ret.iloc[i-252:i]
        active = window[window != 0]
        if len(active) > 30 and active.std() > 0:
            sh = active.mean()/active.std()*np.sqrt(252)
            rolling_sh.append(sh)
            rolling_dates.append(daily_ret.index[i])
    rolling_data[name] = pd.Series(rolling_sh, index=rolling_dates)
    if len(rolling_sh) > 0:
        print(f"{name}: min {min(rolling_sh):+.2f} / max {max(rolling_sh):+.2f} / "
              f"median {np.median(rolling_sh):+.2f}")

with open(os.path.join(os.path.dirname(__file__), 'results.pkl'), 'wb') as f:
    pickle.dump(dict(
        overall=overall, yearly=yearly, rolling=rolling_data,
        strats=strats, years=years,
        START=START, END=END,
    ), f)
print("\n→ results.pkl 保存完了")
