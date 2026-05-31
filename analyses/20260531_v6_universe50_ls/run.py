"""
V6スコア 50銘柄ユニバース クロスセクショナル L/S 検証

前研究 (20260531_v6score_comprehensive) で:
  - r20_adj (ボラ調整モメンタム) が最強因子 (OOS ICIR=7.30)
  - L/S は22銘柄では各サイド3銘柄のみ → 統計パワー不足
が課題だった。

本スクリプトは auKabu PORTFOLIO_ALL の50銘柄に拡張し、
各サイド top/bottom 8銘柄でL/Sを組成。V6スコアの堅牢性を検証する。

ユニバース: 50銘柄 (非鉄8/半導体15/銀行3/機械4/電機2/自動車2/商社2/電子部品4/化学1/保険1/エネ1/その他7)
期間: IS 2022-01-01〜2023-12-31 / OOS 2024-01-01〜2026-05-31
保有: 20日 / 日次リバランス

検証スコア:
  V4:  M + T + 0.5*S          (現行ダッシュボード)
  V6a: r20_adj (連続値)        (ボラ調整モメンタム単独)
  V6b: r20_adj + 0.5*d75      (ボラ調整 + 長期トレンド)
  V6c: 1.5*M + 0.5*T          (グリッドサーチ近似・離散)
"""
from __future__ import annotations

import os
import sys
import psycopg2
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, ttest_1samp

sys.stdout.reconfigure(line_buffering=True)

DB_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost/market_data")

# --- PORTFOLIO_ALL 50銘柄 (4桁) ---
CODES4 = [
    # 非鉄金属8
    '5713','5711','5706','5714','5016','5801','5802','5803',
    # 半導体15
    '8035','6857','6920','6146','7735','4063','3436','7741','6963','6526','9984','4062','6723','285A','6525',
    # 銀行3
    '8306','8316','8411',
    # 機械/防衛4
    '7011','7013','7012','6503',
    # 総合電機2
    '6501','6758',
    # 自動車2
    '7203','7267',
    # 商社2
    '8058','8031',
    # 電子部品4
    '6981','6762','6971','6976',
    # 化学/素材1
    '4004',
    # 保険1
    '8766',
    # エネルギー1
    '1605',
    # その他7
    '6861','6954','9432','7974','9983','6098','9433',
]
CODES5 = [c + '0' for c in CODES4]
CODE_LIST = ','.join(f"'{c}'" for c in CODES5)

IS_START  = pd.Timestamp("2022-01-01")
IS_END    = pd.Timestamp("2023-12-31")
OOS_START = pd.Timestamp("2024-01-01")
N_SIDE = 8   # L/S 各サイドの銘柄数
HOLD = 20


def fetch(sql: str) -> pd.DataFrame:
    conn = psycopg2.connect(DB_URL)
    df = pd.read_sql(sql, conn)
    conn.close()
    return df


def sharpe(rets: pd.Series, ann: int = 252) -> float:
    r = rets.dropna()
    if len(r) < 10 or r.std() == 0:
        return float("nan")
    return float(r.mean() / r.std() * np.sqrt(ann))


def tstat(rets: pd.Series) -> float:
    r = rets.dropna()
    if len(r) < 10:
        return float("nan")
    s, _ = ttest_1samp(r, 0)
    return float(s)


def maxdd(cum: pd.Series) -> float:
    """累積リターン系列の最大ドローダウン (%)"""
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return float(dd.min() * 100)


print("=" * 76)
print("V6スコア 50銘柄 クロスセクショナル L/S 検証")
print("=" * 76)
print("\n[データ取得中]")

prices = fetch(f"""
    SELECT LEFT(code,4) c, date, adj_close::float ac, volume::float vol
    FROM stocks_daily
    WHERE code IN ({CODE_LIST})
      AND date >= '2020-07-01' AND adj_close > 0
    ORDER BY code, date
""")
margin = fetch(f"""
    SELECT LEFT(code,4) c, date,
           CASE WHEN shrt_vol > 0 THEN long_vol::float/shrt_vol ELSE NULL END ratio
    FROM jquants_margin_interest
    WHERE code IN ({CODE_LIST}) AND date >= '2020-07-01'
    ORDER BY code, date
""")
short_sale = fetch(f"""
    SELECT LEFT(code,4) c, calc_date date, SUM(shrt_pos_to_so)::float ratio
    FROM jquants_short_sale_report
    WHERE code IN ({CODE_LIST}) AND calc_date >= '2020-07-01'
    GROUP BY code, calc_date ORDER BY code, calc_date
""")
n225 = fetch("""
    SELECT date, close::float c FROM index_daily
    WHERE code = 'N225' AND date >= '2020-07-01' ORDER BY date
""")

for d in [prices, margin, short_sale, n225]:
    d['date'] = pd.to_datetime(d['date'])
n225 = n225.set_index('date').sort_index()
all_dates = sorted(prices['date'].unique())
eval_dates = [d for d in all_dates if d >= pd.Timestamp("2021-07-01")]
print(f"  銘柄: {len(CODES4)}, 評価日: {len(eval_dates)}")


def compute_factors(code: str, asof: pd.Timestamp) -> dict | None:
    ps = prices[(prices['c'] == code) & (prices['date'] <= asof)].sort_values('date')
    if len(ps) < 90:
        return None
    ac = ps['ac'].values
    last = ac[-1]

    def back(k): return last / ac[-1-k] - 1 if len(ac) > k else None
    r5, r20, r60 = back(5), back(20), back(60)

    ma25 = ac[-25:].mean()
    ma75 = ac[-75:].mean()
    d25, d75 = last / ma25 - 1, last / ma75 - 1

    if len(ac) >= 21:
        daily_rets = ac[-21:-1] / ac[-22:-2] - 1
        vol20 = float(np.std(daily_rets, ddof=1) * np.sqrt(252))
    else:
        vol20 = None
    r20_adj = r20 / vol20 if (r20 is not None and vol20 and vol20 > 0) else None

    mAvg = 0.4 * (r5 or 0) + 0.4 * (r20 or 0) + 0.2 * (r60 or 0)
    M = (2 if mAvg >= 0.05 else 1 if mAvg >= 0.01 else
         -2 if mAvg <= -0.05 else -1 if mAvg <= -0.01 else 0)
    T = (2 if d25 >= 0.05 and d75 >= 0.05 else
         1 if d25 > 0.01 and d75 > 0.01 else
         -2 if d25 <= -0.05 and d75 <= -0.05 else
         -1 if d25 < -0.01 and d75 < -0.01 else 0)

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
    S = max(-2, min(2, S))

    return {'M': M, 'T': T, 'S': S, 'r20_adj': r20_adj, 'd75': d75, 'r20': r20}


print("\n[因子計算中] 全日次×50銘柄 ...")
rows = []
for asof in eval_dates:
    asof = pd.Timestamp(asof)
    for code5 in CODES5:
        code = code5[:4]
        fc = compute_factors(code, asof)
        if fc is None:
            continue
        pnow = prices[(prices['c'] == code) & (prices['date'] == asof)]
        if len(pnow) == 0:
            continue
        pn = pnow.iloc[0]['ac']
        fut = [d for d in all_dates if d > asof]
        if len(fut) < HOLD:
            continue
        fd = fut[HOLD - 1]
        pf = prices[(prices['c'] == code) & (prices['date'] == fd)]
        if len(pf) == 0:
            continue
        fwd = pf.iloc[0]['ac'] / pn - 1
        rows.append({**fc, 'date': asof, 'code': code, 'fwd': fwd})

df = pd.DataFrame(rows)
df['date'] = pd.to_datetime(df['date'])
print(f"  サンプル: {len(df):,}")

# スコア定義
SCORES = {
    'V4 (M+T+0.5S)':     lambda r: r['M'] + r['T'] + 0.5 * r['S'],
    'V6a (r20_adj)':     lambda r: r['r20_adj'] if pd.notna(r['r20_adj']) else np.nan,
    'V6b (r20_adj+0.5d75)': lambda r: (r['r20_adj'] + 0.5 * r['d75'])
                                       if pd.notna(r['r20_adj']) and pd.notna(r['d75']) else np.nan,
    'V6c (1.5M+0.5T)':   lambda r: 1.5 * r['M'] + 0.5 * r['T'],
}
for name, f in SCORES.items():
    df[name] = df.apply(f, axis=1)

# ======================================================
# A. クロスセクショナル L/S (top8 Long / bottom8 Short)
# ======================================================
print("\n" + "=" * 76)
print(f"A. クロスセクショナル L/S  (top{N_SIDE} Long / bottom{N_SIDE} Short, 保有{HOLD}日)")
print("=" * 76)

ls_summary = []
for name in SCORES:
    daily_ls = []
    for dt, g in df.dropna(subset=['fwd', name]).groupby('date'):
        if len(g) < 2 * N_SIDE:
            continue
        ranked = g.sort_values(name, ascending=False)
        longs = ranked.head(N_SIDE)['fwd'].mean()
        shorts = ranked.tail(N_SIDE)['fwd'].mean()
        daily_ls.append({'date': dt, 'ls': longs - shorts, 'long': longs, 'short': shorts})
    ls_df = pd.DataFrame(daily_ls).set_index('date').sort_index()

    print(f"\n  ── {name} ──")
    print(f"  {'期間':<14} {'n':<6} {'L/S Sh':<9} {'L/S t':<8} {'Long Sh':<9} {'Short Sh':<9} {'L/S mean%'}")
    for plabel, mask in [
        ("全期間", ls_df.index >= ls_df.index.min()),
        ("IS(22-23)", (ls_df.index >= IS_START) & (ls_df.index <= IS_END)),
        ("OOS(24-26)", ls_df.index >= OOS_START),
    ]:
        sub = ls_df[mask]
        sh_ls = sharpe(sub['ls']); t_ls = tstat(sub['ls'])
        sh_l = sharpe(sub['long']); sh_s = sharpe(sub['short'])
        m_ls = sub['ls'].mean() * 100
        print(f"  {plabel:<14} {len(sub):<6} {sh_ls:<9.2f} {t_ls:<8.2f} {sh_l:<9.2f} {sh_s:<9.2f} {m_ls:.2f}")
        ls_summary.append({'score': name, 'period': plabel, 'n': len(sub),
                           'ls_sharpe': round(sh_ls, 3), 'ls_t': round(t_ls, 3),
                           'long_sharpe': round(sh_l, 3), 'short_sharpe': round(sh_s, 3),
                           'ls_mean_pct': round(m_ls, 3)})

# ======================================================
# B. IC比較 (50銘柄、市場超過リターン)
# ======================================================
print("\n" + "=" * 76)
print("B. IC比較 (50銘柄 クロスセクショナル, 市場超過リターン)")
print("=" * 76)

mkt = df.groupby('date')['fwd'].transform('mean')
df['xs_fwd'] = df['fwd'] - mkt

def ic_icir(sub, fac):
    ics = []
    for _, g in sub.groupby('date'):
        s = g[[fac, 'xs_fwd']].dropna()
        if len(s) >= 8:
            ic, _ = spearmanr(s[fac], s['xs_fwd'])
            ics.append(ic)
    ic_s = pd.Series(ics).dropna()
    if len(ic_s) < 10 or ic_s.std() == 0:
        return float('nan'), float('nan')
    return float(ic_s.mean()), float(ic_s.mean() / ic_s.std() * np.sqrt(252))

print(f"\n  {'因子':<22} {'全期間IC':<12} {'全ICIR':<10} {'IS ICIR':<10} {'OOS ICIR'}")
print("  " + "-" * 62)
for fac in ['M', 'T', 'S', 'r20_adj', 'd75', 'r20']:
    sub_all = df[['date', fac, 'xs_fwd']].dropna()
    ic_a, icir_a = ic_icir(sub_all, fac)
    ic_is, icir_is = ic_icir(df[(df['date'] >= IS_START) & (df['date'] <= IS_END)], fac)
    ic_oos, icir_oos = ic_icir(df[df['date'] >= OOS_START], fac)
    print(f"  {fac:<22} {ic_a:<12.4f} {icir_a:<10.2f} {icir_is:<10.2f} {icir_oos:.2f}")

# ======================================================
# C. ベストL/Sの詳細 (累積リターン・最大DD)
# ======================================================
print("\n" + "=" * 76)
print("C. ベストスコアのL/S 累積パフォーマンス")
print("=" * 76)

ls_sum_df = pd.DataFrame(ls_summary)
best = ls_sum_df[ls_sum_df['period'] == '全期間'].sort_values('ls_sharpe', ascending=False).iloc[0]
best_name = best['score']
print(f"\n  最良スコア (全期間L/S Sharpe): {best_name} = {best['ls_sharpe']:.2f}")

daily_ls = []
for dt, g in df.dropna(subset=['fwd', best_name]).groupby('date'):
    if len(g) < 2 * N_SIDE:
        continue
    ranked = g.sort_values(best_name, ascending=False)
    daily_ls.append({'date': dt,
                     'ls': ranked.head(N_SIDE)['fwd'].mean() - ranked.tail(N_SIDE)['fwd'].mean()})
best_ls = pd.DataFrame(daily_ls).set_index('date').sort_index()

# 保有20日なので20日ごとの非重複でも近似値を出す（日次オーバーラップ前提のSharpeは上記、ここでは累積）
# 日次エントリーを 1/HOLD ずつ均等配分した擬似累積
best_ls['daily_contrib'] = best_ls['ls'] / HOLD
cum = (1 + best_ls['daily_contrib']).cumprod()
print(f"  擬似累積リターン (日次1/{HOLD}配分): {(cum.iloc[-1]-1)*100:.1f}%")
print(f"  最大ドローダウン: {maxdd(cum):.1f}%")

for plabel, mask in [
    ("IS(22-23)", (best_ls.index >= IS_START) & (best_ls.index <= IS_END)),
    ("OOS(24-26)", best_ls.index >= OOS_START),
]:
    sub = best_ls[mask]['daily_contrib']
    c = (1 + sub).cumprod()
    print(f"  {plabel}: 累積={(c.iloc[-1]-1)*100:.1f}%, MaxDD={maxdd(c):.1f}%, Sharpe={sharpe(sub):.2f}")

# 保存
out = os.path.dirname(__file__)
ls_sum_df.to_csv(os.path.join(out, "ls_summary.csv"), index=False)
print(f"\n  保存: ls_summary.csv")
print("\n完了")
