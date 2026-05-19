"""
マトリクス戦略バックテスト
- 曜日 × セクター × セッション の組み合わせで「いいとこ取り」
- 比較対象: Baseline (Buy&Hold), Conservative, Aggressive
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
START = '2024-05-13'
END   = '2026-05-12'
IS_END   = pd.Timestamp('2025-10-31')
OOS_START = pd.Timestamp('2025-11-01')

COST_BPS = 4.0  # 1往復4bps

# ── データ取得 (全期間: ISとOOS両方含む) ──────
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

# ── ヘルパー：戦略の日次リターン抽出 ──────
def strat_daily(result, dow, sector, session, sign=1):
    """ session: 'on_gap', 'day_open_close', 'full_day' """
    sub = result[(result['dow']==dow) & (result['sector']==sector)]
    daily_ret = sub.groupby('date')[session].mean() * sign - COST_BPS/100
    return daily_ret

# ── 戦略構築 ────────────────────────
def build_strategy(label, components, date_index):
    """ components: [(dow, sector, session, sign, weight), ...]
    各コンポーネントは独立資金100%として扱い、最後に等ウェイト合算 """
    series_list = []
    for dow, sector, session, sign, weight in components:
        s = strat_daily(result, dow, sector, session, sign).reindex(date_index).fillna(0)
        series_list.append(s * weight)
    combined = sum(series_list)
    return combined

# 全営業日インデックス
all_dates = sorted(result['date'].unique())
date_index = pd.DatetimeIndex(all_dates)

# 戦略定義
# Conservative: 火・水・木のONを非鉄ロング + 金曜イントラ半導体ロング
strategies = {
    'Baseline (16銘柄Buy&Hold)': None,  # 別計算

    'Conservative (火水木ON非鉄 + 金イントラ半導体)': [
        (1, 'nonfer', 'on_gap',         1, 0.25),  # 火曜ON 非鉄
        (2, 'nonfer', 'on_gap',         1, 0.25),  # 水曜ON 非鉄
        (3, 'nonfer', 'on_gap',         1, 0.25),  # 木曜ON 非鉄
        (4, 'semi',   'day_open_close', 1, 0.25),  # 金曜イントラ 半導体
    ],

    'Matrix (各曜日最強アクション)': [
        (1, 'nonfer', 'on_gap',         1, 1/6),  # 火曜ON 非鉄
        (2, 'nonfer', 'full_day',       1, 1/6),  # 水曜全日 非鉄
        (3, 'nonfer', 'on_gap',         1, 1/6),  # 木曜ON 非鉄 (寄付クローズ)
        (4, 'semi',   'day_open_close', 1, 1/6),  # 金曜イントラ 半導体
        (4, 'nonfer', 'day_open_close', 1, 1/6),  # 金曜イントラ 非鉄
        (0, 'semi',   'on_gap',        -1, 1/6),  # 月曜ON ショート (半導体)
    ],

    'Aggressive (L/S 完全版)': [
        # ロング側
        (1, 'nonfer', 'on_gap',         1, 1/9),
        (2, 'nonfer', 'full_day',       1, 1/9),
        (3, 'nonfer', 'on_gap',         1, 1/9),
        (4, 'semi',   'day_open_close', 1, 1/9),
        (4, 'nonfer', 'day_open_close', 1, 1/9),
        (1, 'semi',   'on_gap',         1, 1/9),
        (3, 'semi',   'on_gap',         1, 1/9),
        # ショート側
        (0, 'semi',   'on_gap',        -1, 1/9),  # 月曜ON ショート 半導体
        (4, 'semi',   'on_gap',        -1, 1/9),  # 金曜ON ショート 半導体
    ],
}

# Buy&Hold
def buyhold_daily(result, date_index):
    daily_ret = result.groupby('date')['full_day'].mean()  # 全銘柄平均
    return daily_ret.reindex(date_index).fillna(0) - COST_BPS/100/len(date_index)*2

bh_daily = buyhold_daily(result, date_index)

# 各戦略の日次リターン
strat_daily_returns = {'Baseline (16銘柄Buy&Hold)': bh_daily}
for label, comps in strategies.items():
    if comps is None: continue
    strat_daily_returns[label] = build_strategy(label, comps, date_index)

# ── パフォーマンス計算 ──────────────
def perf(daily_ret, label_suffix):
    active = daily_ret[daily_ret != 0]
    cum = ((1 + daily_ret/100).cumprod() - 1) * 100
    final = cum.iloc[-1]
    n_days = len(daily_ret)
    ann = ((1 + final/100) ** (252/n_days) - 1) * 100 if (1+final/100)>0 else -100
    vol = daily_ret.std() * np.sqrt(252)
    sh = active.mean() / active.std() * np.sqrt(252) if len(active)>0 and active.std()>0 else 0
    cum_ser = (1 + daily_ret/100).cumprod()
    dd = (cum_ser - cum_ser.cummax()) / cum_ser.cummax() * 100
    mdd = dd.min()
    return dict(final=final, ann=ann, vol=vol, sharpe=sh, mdd=mdd,
                cum_series=cum)

print("="*92)
print("【全期間 (IS+OOS, 2024-05〜2026-05) のバックテスト結果】")
print("="*92)
print(f"\n{'戦略':50} {'累積':>9} {'年率':>8} {'σ':>7} {'Sharpe':>8} {'MDD':>8}")
print("-"*92)

all_perf = {}
for label, daily_ret in strat_daily_returns.items():
    p = perf(daily_ret, 'ALL')
    all_perf[label] = p
    print(f"{label:50} {p['final']:>+8.1f}% {p['ann']:>+7.1f}% "
          f"{p['vol']:>6.1f}% {p['sharpe']:>+7.2f} {p['mdd']:>+7.1f}%")

# IS/OOS分割
print(f"\n{'='*92}")
print("【IS期間 (2024-05〜2025-10)】")
print("="*92)
print(f"\n{'戦略':50} {'累積':>9} {'年率':>8} {'σ':>7} {'Sharpe':>8} {'MDD':>8}")
print("-"*92)
is_perf = {}
for label, daily_ret in strat_daily_returns.items():
    is_ret = daily_ret[(daily_ret.index <= IS_END)]
    p = perf(is_ret, 'IS')
    is_perf[label] = p
    print(f"{label:50} {p['final']:>+8.1f}% {p['ann']:>+7.1f}% "
          f"{p['vol']:>6.1f}% {p['sharpe']:>+7.2f} {p['mdd']:>+7.1f}%")

print(f"\n{'='*92}")
print("【OOS期間 (2025-11〜2026-05)】")
print("="*92)
print(f"\n{'戦略':50} {'累積':>9} {'年率':>8} {'σ':>7} {'Sharpe':>8} {'MDD':>8}")
print("-"*92)
oos_perf = {}
for label, daily_ret in strat_daily_returns.items():
    oos_ret = daily_ret[(daily_ret.index >= OOS_START)]
    p = perf(oos_ret, 'OOS')
    oos_perf[label] = p
    print(f"{label:50} {p['final']:>+8.1f}% {p['ann']:>+7.1f}% "
          f"{p['vol']:>6.1f}% {p['sharpe']:>+7.2f} {p['mdd']:>+7.1f}%")

# 保存
with open(os.path.join(os.path.dirname(__file__), 'results.pkl'), 'wb') as f:
    pickle.dump(dict(
        all_perf=all_perf, is_perf=is_perf, oos_perf=oos_perf,
        strat_daily_returns=strat_daily_returns,
        IS_END=IS_END, OOS_START=OOS_START,
        START=START, END=END,
    ), f)
print("\n→ results.pkl 保存完了")
