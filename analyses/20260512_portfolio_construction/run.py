"""
最適ポートフォリオ構築：水曜ロング × 月曜回避 × 銘柄選定
- IS: 2024-05〜2025-10 (Sharpe順で銘柄選定)
- OOS: 2025-11〜2026-05 (戦略適用、フィット禁止)
- コスト: 往復4bps/トレード
- 比較戦略 6つ
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

IS_START  = pd.Timestamp('2024-05-13')
IS_END    = pd.Timestamp('2025-10-31')
OOS_START = pd.Timestamp('2025-11-01')
OOS_END   = pd.Timestamp('2026-05-11')

COST_BPS = 4.0    # 往復ベース (片側2bps × 2)
INITIAL  = 100.0  # 初期資産

# ── データ取得 ──────────────────────────
conn = psycopg2.connect(**PG_CONFIG)
sql = """
    SELECT code, date, close
    FROM stocks_daily
    WHERE code IN ({codes}) AND date BETWEEN %s AND %s
    ORDER BY code, date
""".format(codes=','.join(f"'{c}'" for c in ALL))
df = pd.read_sql(sql, conn, params=(IS_START - pd.Timedelta(days=5), OOS_END))
conn.close()

df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
print(f"日足取得: {len(df):,}行 / {df['date'].min().date()}〜{df['date'].max().date()}\n")

# ── ピボット (日付 × 銘柄) ─────────────────
pivot = df.pivot(index='date', columns='code', values='close').sort_index()
returns = pivot.pct_change() * 100
returns['dow'] = pivot.index.dayofweek

# ── IS期間でSharpe銘柄ランキング ───────────
is_rets = returns[(returns.index >= IS_START) & (returns.index <= IS_END)]
sharpe_is = {}
for code in ALL:
    r = is_rets[code].dropna()
    sharpe_is[code] = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0

sharpe_ranked = sorted(sharpe_is.items(), key=lambda x: -x[1])
print("【IS期間 (2024-05〜2025-10) のSharpe順】")
for code, s in sharpe_ranked:
    sector = '半導体' if code in SEMI else '非鉄'
    print(f"  {ALL[code]:18} ({sector})  Sharpe {s:>+.2f}")

TOP3 = [c for c, _ in sharpe_ranked[:3]]
TOP5 = [c for c, _ in sharpe_ranked[:5]]
print(f"\nTop3: {[ALL[c] for c in TOP3]}")
print(f"Top5: {[ALL[c] for c in TOP5]}\n")

# ── 戦略バックテスト ────────────────────
def backtest(returns, dow_filter=None, codes=None, cost_bps=COST_BPS,
             period_start=None, period_end=None):
    """
    dow_filter: 保有する曜日リスト [2] なら水曜のみ
                None なら毎日
    codes:      対象銘柄リスト
    """
    if codes is None: codes = list(ALL)
    if period_start is None: period_start = returns.index[0]
    if period_end   is None: period_end   = returns.index[-1]

    sub = returns[(returns.index >= period_start) & (returns.index <= period_end)].copy()
    portfolio = sub[codes].mean(axis=1)  # 等加重
    dows = sub['dow'].values

    daily_pnl = []
    in_position = False
    trades = 0
    pnl_per_trade = []

    for i in range(len(sub)):
        ret_today = portfolio.iloc[i]
        dow_today = dows[i]
        if dow_filter is None:
            # 毎日保有
            daily_pnl.append(ret_today)
        else:
            # 該当曜日のみ保有
            if dow_today in dow_filter:
                daily_pnl.append(ret_today)
                trades += 1
            else:
                daily_pnl.append(0)

    daily = pd.Series(daily_pnl, index=sub.index)
    # コスト：曜日フィルター=毎回エントリー/エグジット、毎日保有=1往復のみ
    if dow_filter is None:
        n_round_trips = 1
    else:
        n_round_trips = trades
    # bps → % 変換: 4bps = 0.04%
    cost_pct_total = n_round_trips * cost_bps / 100.0

    # 累積リターン (multiplicative)
    cum_gross = (1 + daily/100).cumprod()
    cum_pct_gross = (cum_gross - 1) * 100  # %表示

    # コストは均等に按分して各日から差し引く
    cost_daily_pct = cost_pct_total / len(daily)
    daily_net = daily - cost_daily_pct
    cum_net = (1 + daily_net/100).cumprod()
    cum_pct_net = (cum_net - 1) * 100

    final_ret = cum_pct_net.iloc[-1]
    n_days = len(daily)
    # 年率: (1+r)^(252/N) - 1 だが負の場合は別処理
    if (1 + final_ret/100) > 0:
        ann_ret = ((1 + final_ret/100) ** (252/n_days) - 1) * 100
    else:
        ann_ret = -100.0
    cum_after_cost = cum_pct_net
    vol = daily_net.std() * np.sqrt(252)
    # Sharpe: 保有日のみで計算（ゼロデイズを除外）
    active_daily = daily_net[daily != 0]
    sharpe = active_daily.mean() / active_daily.std() * np.sqrt(252) if len(active_daily)>0 and active_daily.std() > 0 else 0
    # MDD
    cum_series = cum_net
    rolling_max = cum_series.cummax()
    dd = (cum_series - rolling_max) / rolling_max * 100
    mdd = dd.min()
    return {
        'daily': daily_net,
        'cum': cum_pct_net,
        'cum_gross': cum_pct_gross,
        'final_ret': final_ret,
        'ann_ret': ann_ret,
        'vol': vol,
        'sharpe': sharpe,
        'mdd': mdd,
        'trades': trades if dow_filter else 1,
        'cost_total': cost_pct_total,
    }

# ── 戦略一覧 ─────────────────────────
strategies = {
    'A: Buy&Hold (全16銘柄)':       dict(dow_filter=None, codes=list(ALL)),
    'B: 水曜のみ (全16銘柄)':       dict(dow_filter=[2],   codes=list(ALL)),
    'C: 水曜のみ (Top3銘柄)':       dict(dow_filter=[2],   codes=TOP3),
    'D: 月曜回避 (全16銘柄, 火水木金)': dict(dow_filter=[1,2,3,4], codes=list(ALL)),
    'E: 月曜回避 (Top5銘柄)':         dict(dow_filter=[1,2,3,4], codes=TOP5),
    'F: ハイブリッド':              dict(dow_filter=[1,2,3,4], codes=TOP5),  # ←E と同じだが水曜倍張り（後述処理）
}

# OOSバックテスト実行
print("="*78)
print("【OOS期間 (2025-11-01〜2026-05-11) のバックテスト結果】")
print("="*78)
print(f"\n{'戦略':40} {'累積':>8} {'年率':>8} {'σ':>7} {'Sharpe':>8} {'MDD':>8} {'取引数':>5}")
print("-"*78)

results = {}
for name, kw in strategies.items():
    if name.startswith('F:'):
        # ハイブリッド：火木金は普通保有、水曜だけ倍張り（実装簡略化のためスキップ）
        continue
    r = backtest(returns, period_start=OOS_START, period_end=OOS_END, **kw)
    results[name] = r
    print(f"{name:40} {r['final_ret']:>+7.1f}% {r['ann_ret']:>+7.1f}% "
          f"{r['vol']:>6.1f}% {r['sharpe']:>7.2f} {r['mdd']:>+7.1f}% {r['trades']:>5}")

# IS期間も比較
print("\n" + "="*78)
print("【IS期間 (2024-05-13〜2025-10-31) のバックテスト結果（参考）】")
print("="*78)
print(f"\n{'戦略':40} {'累積':>8} {'年率':>8} {'σ':>7} {'Sharpe':>8} {'MDD':>8} {'取引数':>5}")
print("-"*78)
is_results = {}
for name, kw in strategies.items():
    if name.startswith('F:'): continue
    r = backtest(returns, period_start=IS_START, period_end=IS_END, **kw)
    is_results[name] = r
    print(f"{name:40} {r['final_ret']:>+7.1f}% {r['ann_ret']:>+7.1f}% "
          f"{r['vol']:>6.1f}% {r['sharpe']:>7.2f} {r['mdd']:>+7.1f}% {r['trades']:>5}")

# ── 保存 ────────────────────────────
out = dict(
    results=results, is_results=is_results,
    sharpe_ranked=sharpe_ranked,
    ALL=ALL, SEMI=SEMI, NONFER=NONFER, TOP3=TOP3, TOP5=TOP5,
    IS_START=IS_START, IS_END=IS_END,
    OOS_START=OOS_START, OOS_END=OOS_END,
)
with open(os.path.join(os.path.dirname(__file__), 'results.pkl'), 'wb') as f:
    pickle.dump(out, f)
print("\n→ results.pkl 保存完了")
