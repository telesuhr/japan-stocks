"""
SUE(業績サプライズ) × 初動反応(car0)の乖離 → PEADドリフト検証

レポート柱2の簡易版 (LLM不要)。レポートの主張:
「単体のSUE/LESはもはや効かないが、"高SUE × 低LES" の乖離局面でドリフトが出る」。
LES(テキストセンチメント)が無いので、初動の株価反応 car0 を LES の代理とする:

  仮説: 業績は良い(高SUE)のに初動の値動きが鈍い/負(低car0) = 市場の過小反応
        → その後 d10/d20 で SUE方向(上)へドリフトする。

データ:
  - 価格側: 20260530_pead_price_reaction/pead_obs.csv
      (code, entry_date=反応日, car0=反応日TOPIX超過, d5/d10/d20=先行TOPIX超過bps)
  - 業績側: fin_summary payload の OP(営業利益, 金融はNP代替) を
      前年同期(同cur_per_type)比YoYで標準化 → SUE
  - (code, disc_date) の SUE を entry_date に as-of 結合

検証:
  A. 単体SUE / 単体car0 の d20ドリフト (単体は効かないはず=レポート主張の追試)
  B. 2x2 квадрант (SUE高低 × car0高低) の d10/d20
  C. "高SUE×低car0" Long / "低SUE×高car0" Short の L/S, IS/OOS, コスト
"""
from __future__ import annotations

import os, sys, json
import psycopg2
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

sys.stdout.reconfigure(line_buffering=True)
PG = dict(host='localhost', port=5432, user='postgres', dbname='market_data')
HERE = os.path.dirname(__file__)
PEAD_OBS = os.path.join(HERE, '..', '20260530_pead_price_reaction', 'pead_obs.csv')
OOS_START = pd.Timestamp('2024-07-01')
COST_BPS = 20.0  # 往復 (Long/Shortスイング, 5-20日保有)


def fetch(sql, params=None):
    conn = psycopg2.connect(**PG)
    df = pd.read_sql(sql, conn, params=params)
    conn.close()
    return df


def num(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


print("=" * 76)
print("SUE × 初動反応(car0) 乖離 → PEADドリフト検証")
print("=" * 76)

# ---- 価格側 ----
px = pd.read_csv(PEAD_OBS)
px['entry_date'] = pd.to_datetime(px['entry_date'])
print(f"\n[価格側] pead_obs: {len(px):,} events, {px['code'].nunique()} codes")

# ---- 業績側: SUE 構築 ----
print("[業績側] fin_summary から SUE 構築中 ...")
ev = fetch("""
    SELECT DISTINCT ON (code, disc_date) code, disc_date, cur_per_type, payload
    FROM fin_summary
    WHERE disc_date >= '2020-01-01' AND cur_per_type IN ('1Q','2Q','3Q','FY')
    ORDER BY code, disc_date, disc_time
""")
ev['disc_date'] = pd.to_datetime(ev['disc_date'])

def get_val(p):
    p = p if isinstance(p, dict) else (json.loads(p) if p else {})
    op = num(p.get('OP'))
    if not np.isfinite(op):
        op = num(p.get('OrdP'))  # 経常利益 (金融)
    if not np.isfinite(op):
        op = num(p.get('NP'))    # 純利益 fallback
    return op

ev['val'] = ev['payload'].apply(get_val)
ev = ev.dropna(subset=['val']).drop(columns=['payload'])
ev = ev.sort_values(['code', 'cur_per_type', 'disc_date'])

# 前年同期(同cur_per_type)比 = 同codeの同typeで1つ前の開示
ev['prev_val'] = ev.groupby(['code', 'cur_per_type'])['val'].shift(1)
ev['prev_date'] = ev.groupby(['code', 'cur_per_type'])['disc_date'].shift(1)
# YoY (前回同type開示が ~1年前であることを確認: 250-480日)
gap_days = (ev['disc_date'] - ev['prev_date']).dt.days
ev = ev[(gap_days >= 250) & (gap_days <= 480)].copy()
# YoY成長率 (符号変化に頑健な定義)
denom = ev['prev_val'].abs().clip(lower=1.0)
ev['yoy'] = (ev['val'] - ev['prev_val']) / denom
ev = ev[np.isfinite(ev['yoy'])].copy()

# 外れ値クリップ後、開示日ごとにクロスセクション標準化 → SUE
ev['yoy_c'] = ev['yoy'].clip(-3, 3)
ev['SUE'] = ev.groupby('disc_date')['yoy_c'].transform(
    lambda s: (s - s.mean()) / s.std() if s.std() > 0 else 0.0)
print(f"  SUEイベント: {len(ev):,}, {ev['code'].nunique()} codes "
      f"({ev['disc_date'].min().date()}〜{ev['disc_date'].max().date()})")

# ---- as-of 結合: 各 pead_obs(entry_date) に直近の disc_date(<=entry, 4日以内) のSUEを割当 ----
ev_m = ev[['code', 'disc_date', 'SUE', 'yoy']].sort_values('disc_date')
px_m = px.sort_values('entry_date')
merged = pd.merge_asof(
    px_m, ev_m, by='code', left_on='entry_date', right_on='disc_date',
    direction='backward', tolerance=pd.Timedelta(days=4))
merged = merged.dropna(subset=['SUE', 'car0', 'd20']).copy()
print(f"  結合後サンプル: {len(merged):,} (SUE+price 両方あり)")

# car0 も日次クロスセクションで標準化 (LES代理)
merged['car0_z'] = merged.groupby('entry_date')['car0'].transform(
    lambda s: (s - s.mean()) / s.std() if s.std() > 0 else 0.0)
merged['period'] = np.where(merged['entry_date'] >= OOS_START, 'OOS', 'IS')


def sharpe_bps(x, ann=np.sqrt(252/15)):  # 平均保有~15営業日 → 非重複近似で年率化は控えめに
    x = pd.Series(x).dropna()
    if len(x) < 10 or x.std() == 0:
        return float('nan')
    return float(x.mean() / x.std() * ann)


# ============================================
# A. 単体SUE / 単体car0 の d20 ドリフト (単体は弱いはず)
# ============================================
print("\n" + "=" * 76)
print("A. 単体ファクターの d20ドリフト (単体は効かない=レポート主張の追試)")
print("=" * 76)
for col in ['SUE', 'car0_z']:
    rho, p = spearmanr(merged[col], merged['d20'])
    # 5分位
    q = pd.qcut(merged[col].rank(method='first'), 5, labels=False)
    top = merged[q == 4]['d20'].mean(); bot = merged[q == 0]['d20'].mean()
    print(f"  {col:<8}: d20とのρ={rho:+.4f}(p={p:.1e})  Q5={top:+.0f}bps Q1={bot:+.0f}bps  Q5-Q1={top-bot:+.0f}bps")

# ============================================
# B. 2x2 квадрант (SUE高低 × car0高低)
# ============================================
print("\n" + "=" * 76)
print("B. 2x2 квадрант別 forward drift (全期間)")
print("=" * 76)
merged['hiSUE'] = merged['SUE'] > merged['SUE'].median()
merged['hiCAR'] = merged['car0_z'] > merged['car0_z'].median()
print(f"\n  {'квадрант':<22} {'n':>6} {'d5':>9} {'d10':>9} {'d20':>9}")
print("  " + "-" * 58)
labels = {(True, False): '高SUE×低car0 ★', (True, True): '高SUE×高car0',
          (False, False): '低SUE×低car0', (False, True): '低SUE×高car0'}
for (hs, hc), lab in labels.items():
    g = merged[(merged['hiSUE'] == hs) & (merged['hiCAR'] == hc)]
    print(f"  {lab:<22} {len(g):>6} {g['d5'].mean():>9.0f} {g['d10'].mean():>9.0f} {g['d20'].mean():>9.0f}")

# ============================================
# C. L/S: 高SUE×低car0 Long / 低SUE×高car0 Short
# ============================================
print("\n" + "=" * 76)
print("C. 乖離 L/S: 高SUE×低car0 Long / 低SUE×高car0 Short")
print("=" * 76)
# ダイバージェンス・スコア = SUE - car0_z (高SUEかつ低car0で大)
merged['div'] = merged['SUE'] - merged['car0_z']

for H in ['d10', 'd20']:
    print(f"\n  --- 保有 {H} ---")
    print(f"  {'期間':<8} {'n(L)':>6} {'Long':>9} {'n(S)':>7} {'Short':>9} {'L-S(gross)':>11} {'net(20bps)':>11} {'Sharpe(net)':>11}")
    for label, sub in [('全期間', merged), ('IS', merged[merged.period=='IS']), ('OOS', merged[merged.period=='OOS'])]:
        thr_hi = sub['div'].quantile(0.8); thr_lo = sub['div'].quantile(0.2)
        longs = sub[sub['div'] >= thr_hi][H]
        shorts = sub[sub['div'] <= thr_lo][H]
        ls_gross = longs.mean() - shorts.mean()
        ls_net = ls_gross - COST_BPS
        # 日次L/S系列でSharpe
        daily = []
        for dt, g in sub.groupby('entry_date'):
            hi = g['div'].quantile(0.8); lo = g['div'].quantile(0.2)
            L = g[g['div']>=hi][H].mean(); S = g[g['div']<=lo][H].mean()
            if np.isfinite(L) and np.isfinite(S):
                daily.append(L - S)
        sh = sharpe_bps(pd.Series(daily) - COST_BPS) if len(daily)>=10 else float('nan')
        print(f"  {label:<8} {len(longs):>6} {longs.mean():>9.0f} {len(shorts):>7} {shorts.mean():>9.0f} "
              f"{ls_gross:>11.0f} {ls_net:>11.0f} {sh:>11.2f}")

# 高SUE×低car0 の Long-only も
print("\n  --- 高SUE×低car0 Long-only (d20, bps) ---")
star = merged[(merged['SUE'] > merged['SUE'].quantile(0.7)) &
              (merged['car0_z'] < merged['car0_z'].quantile(0.3))]
for label, sub in [('全期間', star), ('IS', star[star.period=='IS']), ('OOS', star[star.period=='OOS'])]:
    d = sub['d20']
    print(f"  {label:<8} n={len(sub):>5} d20平均={d.mean():+.0f}bps net={d.mean()-COST_BPS:+.0f}bps "
          f"勝率={ (d>0).mean()*100:.0f}% t={d.mean()/d.std()*np.sqrt(len(d)) if d.std()>0 else 0:+.1f}")

merged[['code','entry_date','SUE','car0','car0_z','div','d5','d10','d20','period']].to_csv(
    os.path.join(HERE,'sue_obs.csv'), index=False)
print(f"\n  保存: sue_obs.csv")
print("\n完了")
