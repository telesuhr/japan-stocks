"""
オーバーラップ過大評価は V4/V5 でも同じか — Sharpe年率換算の検証

第十六弾 (v6b_nonoverlap_cost) で V6b の日次オーバーラップ L/S Sharpe4.65 が
非重複だと 0.84 に激減した。これが V6b 固有か、それとも全スコア共通の
「年率換算アーティファクト」かを確認する。

仮説: 保有H日のリターンを日次サンプルとして √252 で年率化すると、
正しい √(252/H) に対し √H 倍 (H=20 → √20≈4.47倍) 過大評価される。
→ V4/V5/V6 すべて同じ倍率で盛れているはず。

検証スコア (すべて Long-only, score≥閾値):
  V4:  M + T + 0.5*S   閾値≥2.0
  V5b: M + 0.5*S       閾値≥2.0
  V6b: r20_adj + 0.5*d75  (連続値, 上位30%をLong)

比較: 同じ銘柄×日付サンプルに対し
  (a) 日次オーバーラップ ×√252
  (b) 非重複20日 (offset=0) ×√(252/20)
"""
from __future__ import annotations

import os
import sys
import psycopg2
import pandas as pd
import numpy as np

sys.stdout.reconfigure(line_buffering=True)
DB_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost/market_data")

CODES4 = [
    '5713','5711','5706','5714','5016','5801','5802','5803',
    '8035','6857','6920','6146','7735','4063','3436','7741','6963','6526','9984','4062','6723','285A','6525',
    '8306','8316','8411','7011','7013','7012','6503','6501','6758','7203','7267','8058','8031',
    '6981','6762','6971','6976','4004','8766','1605','6861','6954','9432','7974','9983','6098','9433',
]
CODES5 = [c + '0' for c in CODES4]
CODE_LIST = ','.join(f"'{c}'" for c in CODES5)
HOLD = 20


def fetch(sql):
    conn = psycopg2.connect(DB_URL)
    df = pd.read_sql(sql, conn)
    conn.close()
    return df


print("=" * 76)
print("オーバーラップ過大評価の検証 — V4/V5/V6 共通か")
print("=" * 76)
print("\n[データ取得中]")

prices = fetch(f"""
    SELECT LEFT(code,4) c, date, adj_close::float ac
    FROM stocks_daily WHERE code IN ({CODE_LIST})
      AND date >= '2020-07-01' AND adj_close > 0 ORDER BY code, date
""")
margin = fetch(f"""
    SELECT LEFT(code,4) c, date,
           CASE WHEN shrt_vol>0 THEN long_vol::float/shrt_vol ELSE NULL END ratio
    FROM jquants_margin_interest WHERE code IN ({CODE_LIST}) AND date >= '2020-07-01' ORDER BY code, date
""")
short_sale = fetch(f"""
    SELECT LEFT(code,4) c, calc_date date, SUM(shrt_pos_to_so)::float ratio
    FROM jquants_short_sale_report WHERE code IN ({CODE_LIST}) AND calc_date >= '2020-07-01'
    GROUP BY code, calc_date ORDER BY code, calc_date
""")
for d in [prices, margin, short_sale]:
    d['date'] = pd.to_datetime(d['date'])

by_code = {}
for code, g in prices.groupby('c'):
    g = g.sort_values('date')
    by_code[code] = (g['date'].values.astype('datetime64[ns]'), g['ac'].values)
all_dates = np.array(sorted(prices['date'].unique())).astype('datetime64[ns]')
print(f"  銘柄: {len(by_code)}, 取引日: {len(all_dates)}")


def supply_score(code, asof):
    ms = margin[(margin['c'] == code) & (margin['date'] <= asof) &
                (margin['date'] >= asof - pd.Timedelta(days=90))].sort_values('date')
    ss = short_sale[(short_sale['c'] == code) & (short_sale['date'] <= asof) &
                    (short_sale['date'] >= asof - pd.Timedelta(days=90))].sort_values('date')
    S = 0
    if len(ms) >= 2:
        mr, mo = ms.iloc[-1]['ratio'], ms.iloc[0]['ratio']
        if mo and mr and mo > 0 and mr > 0:
            chg = mr / mo - 1
            if chg < -0.2: S += 1
            elif chg > 0.3: S -= 1
    if len(ss) >= 2:
        sr, so = ss.iloc[-1]['ratio'], ss.iloc[0]['ratio']
        if sr is not None and so is not None:
            if sr - so < -0.005: S += 1
            elif sr - so > 0.005: S -= 1
    return max(-2, min(2, S))


# 全銘柄×全日のスコアと fwd を計算
rows = []
start = np.datetime64('2021-07-01')
for code5 in CODES5:
    code = code5[:4]
    if code not in by_code:
        continue
    dates, ac = by_code[code]
    n = len(ac)
    for i in range(n):
        if dates[i] < start or i < 89 or i + HOLD >= n:
            continue
        last = ac[i]
        r5 = last / ac[i-5] - 1
        r20 = last / ac[i-20] - 1
        r60 = last / ac[i-60] - 1
        ma25 = ac[i-24:i+1].mean(); ma75 = ac[i-74:i+1].mean()
        d25, d75 = last/ma25 - 1, last/ma75 - 1
        daily = ac[i-19:i+1] / ac[i-20:i] - 1
        vol20 = float(np.std(daily, ddof=1) * np.sqrt(252))
        if vol20 <= 0:
            continue
        mAvg = 0.4*r5 + 0.4*r20 + 0.2*r60
        M = 2 if mAvg>=0.05 else 1 if mAvg>=0.01 else -2 if mAvg<=-0.05 else -1 if mAvg<=-0.01 else 0
        T = (2 if d25>=0.05 and d75>=0.05 else 1 if d25>0.01 and d75>0.01 else
             -2 if d25<=-0.05 and d75<=-0.05 else -1 if d25<-0.01 and d75<-0.01 else 0)
        S = supply_score(code, pd.Timestamp(dates[i]))
        r20_adj = r20 / vol20
        fwd = ac[i+HOLD] / last - 1
        rows.append({'date': dates[i], 'code': code,
                     'V4': M+T+0.5*S, 'V5b': M+0.5*S, 'V6b': r20_adj+0.5*d75,
                     'fwd': fwd})

df = pd.DataFrame(rows)
df['date'] = pd.to_datetime(df['date'])
df['idx'] = df['date'].map({d: i for i, d in enumerate(pd.to_datetime(all_dates))})
print(f"  サンプル: {len(df):,}")

# 非重複: offset=0 から 20日刻みの date index のみ
nonoverlap_idx = set(range(int(df['idx'].min()), int(df['idx'].max())+1, HOLD))


def sharpe(rets, ann_factor):
    r = rets[~np.isnan(rets)]
    if len(r) < 5 or r.std(ddof=1) == 0:
        return float('nan')
    return float(r.mean() / r.std(ddof=1) * ann_factor)


print("\n" + "=" * 76)
print(f"Long-only Sharpe: (a)日次オーバーラップ×√252  vs  (b)非重複20日×√(252/20)")
print("=" * 76)
print(f"\n  {'スコア':<10} {'閾値':<18} {'(a)日次OL':<12} {'(b)非重複':<12} {'比率(a/b)':<10} {'√20理論'}")
print("  " + "-" * 70)

CONFIGS = [
    ('V4', 'score>=2.0', lambda d: d['V4'] >= 2.0),
    ('V5b', 'score>=2.0', lambda d: d['V5b'] >= 2.0),
    ('V6b', '上位30%', None),  # V6bは連続値→日毎に上位30%
]

ann_ol = np.sqrt(252)
ann_no = np.sqrt(252 / HOLD)

results = []
for name, thr_label, cond in CONFIGS:
    if cond is not None:
        sel = df[cond(df)]
    else:
        # 日毎に上位30%
        sel_parts = []
        for _, g in df.groupby('date'):
            k = max(1, int(len(g) * 0.3))
            sel_parts.append(g.nlargest(k, 'V6b'))
        sel = pd.concat(sel_parts)

    # (a) 日次オーバーラップ
    sh_a = sharpe(sel['fwd'].values, ann_ol)
    # (b) 非重複20日
    sel_no = sel[sel['idx'].isin(nonoverlap_idx)]
    sh_b = sharpe(sel_no['fwd'].values, ann_no)
    ratio = sh_a / sh_b if sh_b and not np.isnan(sh_b) else float('nan')
    print(f"  {name:<10} {thr_label:<18} {sh_a:<12.2f} {sh_b:<12.2f} {ratio:<10.2f} {np.sqrt(HOLD):.2f}")
    results.append({'score': name, 'sharpe_overlap': round(sh_a,3),
                    'sharpe_nonoverlap': round(sh_b,3), 'ratio': round(ratio,3)})

print(f"\n  理論倍率 √{HOLD} = {np.sqrt(HOLD):.2f}")
print("  → 比率が √20 に一致すれば、過大評価は年率換算アーティファクトで全スコア共通")

pd.DataFrame(results).to_csv(os.path.join(os.path.dirname(__file__), "results.csv"), index=False)
print("\n  保存: results.csv")
print("\n完了")
