"""
V4/V5 モメンタムスコア 日次エントリー戦略検証

既存の backtest_v5.py は週次(金曜)エントリー。
本スクリプトは「毎営業日スコアを計算してエントリー」した場合に
パフォーマンスが改善するか検証する。

対象: 22銘柄 (backtest_v5.py と同じユニバース)
期間: IS 2021-01-01 〜 2023-12-31 / OOS 2024-01-01 〜 2026-05-31
保有: 20日 / 30日

スコアバリアント:
  V4: M + T + 0.5*S   BUY≥2
  V5a: M + 0.5*S      BUY≥1.5
  V5b: M + 0.5*S      BUY≥2

セクション:
  A: 日次 vs 週次 Sharpe 比較 (全期間)
  B: IS / OOS スプリット
  C: 曜日別エントリー効果 (月曜〜金曜)
  D: スコア別 forward return 分布
"""
from __future__ import annotations

import os
import sys
import psycopg2
import pandas as pd
import numpy as np
from scipy.stats import ttest_1samp

sys.stdout.reconfigure(line_buffering=True)

DB_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost/market_data")

# --- ユニバース (backtest_v5.py と同一) ---
UNI = [
    ('80350','semi'),('68570','semi'),('69200','semi'),('61460','semi'),('77350','semi'),
    ('67230','semi'),('69630','semi'),('65260','semi'),('40620','semi'),('34360','semi'),
    ('40630','semi'),('77410','semi'),('99840','semi'),('285A0','semi'),
    ('58030','base'),('50160','base'),('58010','base'),('58020','base'),
    ('57130','base'),('57060','base'),('57110','base'),('57140','base'),
]
CODES = [u[0] for u in UNI]
CODE_LIST = ','.join(f"'{c}'" for c in CODES)

IS_END   = pd.Timestamp("2023-12-31")
OOS_START = pd.Timestamp("2024-01-01")


def fetch(sql: str) -> pd.DataFrame:
    conn = psycopg2.connect(DB_URL)
    df = pd.read_sql(sql, conn)
    conn.close()
    return df


def sharpe(rets: pd.Series, ann: int = 252) -> float:
    if len(rets) < 5 or rets.std() == 0:
        return float("nan")
    return float(rets.mean() / rets.std() * np.sqrt(ann))


def tstat(rets: pd.Series) -> float:
    if len(rets) < 5:
        return float("nan")
    stat, _ = ttest_1samp(rets.dropna(), 0)
    return float(stat)


print("=" * 72)
print("V4/V5 スコア 日次エントリー戦略検証")
print("=" * 72)

# ========================
# データ取得
# ========================
print("\n[データ取得中]")

prices = fetch(f"""
    SELECT LEFT(code,4) c, date, adj_close::float ac
    FROM stocks_daily
    WHERE code IN ({CODE_LIST})
      AND date >= '2020-07-01'
      AND adj_close > 0
    ORDER BY code, date
""")
margin = fetch(f"""
    SELECT LEFT(code,4) c, date,
           CASE WHEN shrt_vol > 0 THEN long_vol::float/shrt_vol ELSE NULL END ratio
    FROM jquants_margin_interest
    WHERE code IN ({CODE_LIST})
      AND date >= '2020-07-01'
    ORDER BY code, date
""")
short_sale = fetch(f"""
    SELECT LEFT(code,4) c, calc_date date,
           SUM(shrt_pos_to_so)::float ratio
    FROM jquants_short_sale_report
    WHERE code IN ({CODE_LIST})
      AND calc_date >= '2020-07-01'
    GROUP BY code, calc_date
    ORDER BY code, calc_date
""")
n225 = fetch("""
    SELECT date, close::float c
    FROM index_daily
    WHERE code = 'N225' AND date >= '2020-07-01'
    ORDER BY date
""")

for df_ in [prices, margin, short_sale, n225]:
    df_['date'] = pd.to_datetime(df_['date'])

n225 = n225.set_index('date').sort_index()
prices['date'] = pd.to_datetime(prices['date'])
all_dates = sorted(prices['date'].unique())

# 評価対象: 2021年以降 (スコア計算に75日必要)
eval_dates = [d for d in all_dates if d >= pd.Timestamp("2021-01-01")]
weekly_dates = [d for d in eval_dates if pd.Timestamp(d).weekday() == 4]

print(f"  全評価日(日次): {len(eval_dates)}, 週次(金曜): {len(weekly_dates)}, 銘柄: {len(CODES)}")


# ========================
# スコア計算関数
# ========================
def compute_components(code: str, asof: pd.Timestamp) -> dict | None:
    ps = prices[(prices['c'] == code) & (prices['date'] <= asof)].sort_values('date')
    if len(ps) < 75:
        return None
    last = ps.iloc[-1]['ac']

    def back(k):
        return last / ps.iloc[-1 - k]['ac'] - 1 if len(ps) > k else None

    r5, r20, r60 = back(5), back(20), back(60)
    ma25 = ps.iloc[-25:]['ac'].mean()
    ma75 = ps.iloc[-75:]['ac'].mean()
    d25, d75 = last / ma25 - 1, last / ma75 - 1

    mAvg = 0.4 * (r5 or 0) + 0.4 * (r20 or 0) + 0.2 * (r60 or 0)
    M = (2 if mAvg >= 0.05 else
         1 if mAvg >= 0.01 else
         -2 if mAvg <= -0.05 else
         -1 if mAvg <= -0.01 else 0)
    T = (2 if d25 >= 0.05 and d75 >= 0.05 else
         1 if d25 > 0.01 and d75 > 0.01 else
         -2 if d25 <= -0.05 and d75 <= -0.05 else
         -1 if d25 < -0.01 and d75 < -0.01 else 0)

    ms = margin[(margin['c'] == code) &
                (margin['date'] <= asof) &
                (margin['date'] >= asof - pd.Timedelta(days=90))].sort_values('date')
    ss = short_sale[(short_sale['c'] == code) &
                    (short_sale['date'] <= asof) &
                    (short_sale['date'] >= asof - pd.Timedelta(days=90))].sort_values('date')
    S = 0
    if len(ms) >= 2:
        mr, mo = ms.iloc[-1]['ratio'], ms.iloc[0]['ratio']
        if mo and mr and mo > 0 and mr > 0:
            chg = mr / mo - 1
            if chg < -0.2:
                S += 1
            elif chg > 0.3:
                S -= 1
    if len(ss) >= 2:
        sr, so = ss.iloc[-1]['ratio'], ss.iloc[0]['ratio']
        if sr is not None and so is not None:
            if sr - so < -0.005:
                S += 1
            elif sr - so > 0.005:
                S -= 1
    S = max(-2, min(2, S))
    return {'M': M, 'T': T, 'S': S}


# ========================
# 全データ計算
# ========================
print("\n[スコア計算中] 全日次×22銘柄 ...")
all_rows: list[dict] = []

for asof in eval_dates:
    asof = pd.Timestamp(asof)
    n225_past = n225[n225.index <= asof]
    n225_r60 = (n225_past['c'].iloc[-1] / n225_past['c'].iloc[-60] - 1
                if len(n225_past) >= 60 else None)

    for code5, grp in UNI:
        code = code5[:4]
        sc = compute_components(code, asof)
        if sc is None:
            continue
        price_now = prices[(prices['c'] == code) & (prices['date'] == asof)]
        if len(price_now) == 0:
            continue
        pn = price_now.iloc[0]['ac']

        fut = [d for d in all_dates if d > asof]
        fwds = {}
        for k in [5, 10, 20, 30]:
            if len(fut) > k:
                fd = fut[min(k - 1, len(fut) - 1)]
                pf = prices[(prices['c'] == code) & (prices['date'] == fd)]
                if len(pf):
                    fwds[k] = pf.iloc[0]['ac'] / pn - 1

        if not fwds:
            continue

        all_rows.append({
            **sc,
            'n225_r60': n225_r60,
            'date': asof,
            'code': code,
            'dow': asof.weekday(),  # 0=月, 4=金
            'is_friday': asof.weekday() == 4,
            **{f'fwd{k}': v for k, v in fwds.items()},
        })

df = pd.DataFrame(all_rows)
print(f"  サンプル: {len(df):,} (銘柄×評価日)")


# ========================
# スコア計算 (全バリアント)
# ========================
STRATEGIES = {
    'V4 (M+T+0.5S, gate-5%)': {
        'formula': lambda M, T, S: M + T + 0.5 * S,
        'hold': 20, 'buy_th': 2.0, 'gate': -0.05,
    },
    'V5a (M+0.5S,  gate-3%)': {
        'formula': lambda M, T, S: M + 0.5 * S,
        'hold': 30, 'buy_th': 1.5, 'gate': -0.03,
    },
    'V5b (M+0.5S≥2, gate-3%)': {
        'formula': lambda M, T, S: M + 0.5 * S,
        'hold': 30, 'buy_th': 2.0, 'gate': -0.03,
    },
}

for name, cfg in STRATEGIES.items():
    formula = cfg['formula']
    df[f'score_{name}'] = df.apply(
        lambda r: formula(r['M'], r['T'], r['S']), axis=1
    )

# ======================================================
# セクション A: 日次 vs 週次 Sharpe 比較
# ======================================================
print("\n" + "=" * 72)
print("A. 日次エントリー vs 週次(金曜)エントリー 全期間比較")
print("=" * 72)
print(f"\n{'戦略':<30} {'頻度':<8} {'n(BUY)':<10} {'mean(%)':<10} {'Sharpe':<10} {'t値'}")
print("-" * 72)

RESULT_ROWS: list[dict] = []

for name, cfg in STRATEGIES.items():
    hold = cfg['hold']
    buy_th = cfg['buy_th']
    gate = cfg['gate']
    fwd_col = f'fwd{hold}'
    score_col = f'score_{name}'
    if fwd_col not in df.columns:
        continue

    for freq_label, mask_freq in [("日次", df['date'] >= df['date'].min()),
                                   ("週次(金)", df['is_friday'])]:
        sub = df[mask_freq].dropna(subset=[fwd_col]).copy()
        sub['score'] = sub[score_col]
        if gate:
            sub.loc[
                sub['n225_r60'].notna() &
                (sub['n225_r60'] < gate) &
                (sub['score'] > 1), 'score'
            ] = 1

        buy = sub[sub['score'] >= buy_th][fwd_col]
        n = len(buy)
        mean_ = buy.mean() * 100 if n > 0 else float('nan')
        sh = sharpe(buy)
        t = tstat(buy)
        print(f"  {name:<28} {freq_label:<8} {n:<10} {mean_:<10.2f} {sh:<10.2f} {t:.2f}")

        RESULT_ROWS.append({
            'strategy': name, 'freq': freq_label,
            'n_buy': n, 'mean_pct': round(mean_, 3),
            'sharpe': round(sh, 3), 'tstat': round(t, 3),
        })

    print()

# ======================================================
# セクション B: IS / OOS スプリット
# ======================================================
print("=" * 72)
print("B. IS (2021-2023) / OOS (2024-2026) 日次エントリー")
print("=" * 72)
print(f"\n{'戦略':<30} {'期間':<8} {'n(BUY)':<10} {'mean(%)':<10} {'Sharpe':<10} {'t値'}")
print("-" * 72)

for name, cfg in STRATEGIES.items():
    hold = cfg['hold']
    buy_th = cfg['buy_th']
    gate = cfg['gate']
    fwd_col = f'fwd{hold}'
    score_col = f'score_{name}'
    if fwd_col not in df.columns:
        continue

    for period_label, mask_period in [
        ("IS(21-23)", df['date'] <= IS_END),
        ("OOS(24-26)", df['date'] >= OOS_START),
    ]:
        sub = df[mask_period].dropna(subset=[fwd_col]).copy()
        sub['score'] = sub[score_col]
        if gate:
            sub.loc[
                sub['n225_r60'].notna() &
                (sub['n225_r60'] < gate) &
                (sub['score'] > 1), 'score'
            ] = 1

        buy = sub[sub['score'] >= buy_th][fwd_col]
        n = len(buy)
        mean_ = buy.mean() * 100 if n > 0 else float('nan')
        sh = sharpe(buy)
        t = tstat(buy)
        print(f"  {name:<28} {period_label:<8} {n:<10} {mean_:<10.2f} {sh:<10.2f} {t:.2f}")

        RESULT_ROWS.append({
            'strategy': name, 'freq': f'daily_{period_label}',
            'n_buy': n, 'mean_pct': round(mean_, 3),
            'sharpe': round(sh, 3), 'tstat': round(t, 3),
        })

    print()

# ======================================================
# セクション C: 曜日別エントリー効果
# ======================================================
print("=" * 72)
print("C. 曜日別エントリー効果 (V4スコア, 保有20日)")
print("=" * 72)

DOW_NAMES = {0: '月曜', 1: '火曜', 2: '水曜', 3: '木曜', 4: '金曜'}
name_v4 = 'V4 (M+T+0.5S, gate-5%)'
cfg_v4 = STRATEGIES[name_v4]
hold = cfg_v4['hold']
buy_th = cfg_v4['buy_th']
gate = cfg_v4['gate']
fwd_col = f'fwd{hold}'

print(f"\n  スコア: {name_v4}  BUY≥{buy_th}")
print(f"  {'曜日':<8} {'n(BUY)':<10} {'mean(%)':<10} {'Sharpe':<10} {'t値'}")
print("  " + "-" * 48)

for dow, dow_name in DOW_NAMES.items():
    sub = df[df['dow'] == dow].dropna(subset=[fwd_col]).copy()
    sub['score'] = sub[f'score_{name_v4}']
    if gate:
        sub.loc[
            sub['n225_r60'].notna() &
            (sub['n225_r60'] < gate) &
            (sub['score'] > 1), 'score'
        ] = 1
    buy = sub[sub['score'] >= buy_th][fwd_col]
    n = len(buy)
    mean_ = buy.mean() * 100 if n > 0 else float('nan')
    sh = sharpe(buy)
    t = tstat(buy)
    print(f"  {dow_name:<8} {n:<10} {mean_:<10.2f} {sh:<10.2f} {t:.2f}")

# ======================================================
# セクション D: スコア分布 vs forward return
# ======================================================
print("\n" + "=" * 72)
print("D. スコア分位別 forward return (V4, 保有20日, 日次エントリー全期間)")
print("=" * 72)

fwd20 = 'fwd20'
sub_all = df.dropna(subset=[fwd20]).copy()
sub_all['score_v4'] = sub_all[f'score_{name_v4}']

print(f"\n  {'スコア':<10} {'n':<8} {'mean(%)':<10} {'Sharpe':<10}")
print("  " + "-" * 40)

for score_val in sorted(sub_all['score_v4'].unique()):
    grp = sub_all[sub_all['score_v4'] == score_val][fwd20]
    n = len(grp)
    mean_ = grp.mean() * 100
    sh = sharpe(grp)
    print(f"  {score_val:<10.1f} {n:<8} {mean_:<10.2f} {sh:.2f}")

# ======================================================
# セクション E: スコア変化(モメンタム)効果
# ======================================================
print("\n" + "=" * 72)
print("E. スコア変化効果 (前回比 +2以上上昇した銘柄の forward return)")
print("=" * 72)

# 各銘柄の日次スコア系列を作成
score_series = df[['date', 'code', f'score_{name_v4}', fwd20]].copy()
score_series = score_series.sort_values(['code', 'date'])
score_series['prev_score'] = score_series.groupby('code')[f'score_{name_v4}'].shift(5)
score_series['score_chg'] = score_series[f'score_{name_v4}'] - score_series['prev_score']

print(f"\n  スコア変化 (5営業日前比) 別 forward 20d return (V4スコア)")
print(f"  {'変化量':<12} {'n':<8} {'mean(%)':<10} {'Sharpe':<10}")
print("  " + "-" * 42)

for chg_label, chg_mask in [
    ("≥+2", score_series['score_chg'] >= 2),
    ("+1", score_series['score_chg'] == 1),
    ("0", score_series['score_chg'] == 0),
    ("-1", score_series['score_chg'] == -1),
    ("≤-2", score_series['score_chg'] <= -2),
]:
    grp = score_series[chg_mask].dropna(subset=[fwd20])[fwd20]
    n = len(grp)
    mean_ = grp.mean() * 100 if n > 0 else float('nan')
    sh = sharpe(grp)
    print(f"  {chg_label:<12} {n:<8} {mean_:<10.2f} {sh:.2f}")

# ======================================================
# サマリー出力
# ======================================================
print("\n" + "=" * 72)
print("まとめ")
print("=" * 72)

result_df = pd.DataFrame(RESULT_ROWS)
best_row = result_df.loc[result_df['sharpe'].idxmax()]
print(f"\n  最高 Sharpe: {best_row['strategy']} / {best_row['freq']} = {best_row['sharpe']:.2f}")

daily_v4 = result_df[(result_df['strategy'] == name_v4) & (result_df['freq'] == '日次')]
weekly_v4 = result_df[(result_df['strategy'] == name_v4) & (result_df['freq'] == '週次(金)')]
if len(daily_v4) and len(weekly_v4):
    d_sh = daily_v4.iloc[0]['sharpe']
    w_sh = weekly_v4.iloc[0]['sharpe']
    diff = d_sh - w_sh
    sign = "改善" if diff > 0.1 else ("悪化" if diff < -0.1 else "同等")
    print(f"\n  V4 日次 Sharpe: {d_sh:.2f}  週次(金) Sharpe: {w_sh:.2f}  → {sign} ({diff:+.2f})")

# CSV保存
out_path = os.path.join(os.path.dirname(__file__), "results.csv")
result_df.to_csv(out_path, index=False)
print(f"\n  結果保存: {out_path}")
print("\n完了")
