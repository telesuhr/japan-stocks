"""
クロージング・オークション オーバーシュート → 翌朝反転 検証

レポート (日本株トレーディング戦略の徹底検討) 柱1b の検証。
2024-11-05 導入のクロージング・オークション (15:25-30 プレクロージング → 15:30 板寄せ) で、
パッシブファンドのMOC注文等により引け値が人為的にオーバーシュートする。
仮説: 引けの board-jump が大きいほど、翌朝の寄りで反転 (ミーン・リバージョン) する。

データ (確認済み):
  stocks_intraday は 15:24 (連続取引最終足) と 15:30 (板寄せ結果) を持つ。
  close_jump  = close[15:30] / close[15:24] - 1   (オークションのオーバーシュート)
  overnight   = open[翌09:00] / close[15:30] - 1   (翌朝の反転を測る)

ユニバース: 流動性上位200銘柄
期間: 2024-11-05 〜 (新制度のみ)。IS前半 / OOS後半 で分割。

検証:
  A. close_jump と overnight の相関 (反転なら負)
  B. close_jump 分位別 overnight リターン (単調反転か)
  C. クロスセクション L/S (top short / bottom long, overnight保有) Sharpe・コスト感度
  D. 月末効果 (インデックスリバランスでインバランス増幅?)
"""
from __future__ import annotations

import os, sys
import psycopg2
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

sys.stdout.reconfigure(line_buffering=True)
PG = dict(host='localhost', port=5432, user='postgres', dbname='market_data')
HERE = os.path.dirname(__file__)
START = '2024-11-05'
N_UNI = 200


def fetch(sql, params=None):
    conn = psycopg2.connect(**PG)
    df = pd.read_sql(sql, conn, params=params)
    conn.close()
    return df


print("=" * 76)
print("クロージング・オークション オーバーシュート → 翌朝反転 検証")
print("=" * 76)
print("\n[データ取得中]")

uni = fetch("""
    SELECT code FROM stocks_daily
    WHERE date >= '2025-05-01' AND turnover_value > 0
    GROUP BY code ORDER BY AVG(turnover_value) DESC LIMIT %s
""", (N_UNI,))
codes = uni['code'].tolist()
ph = ','.join(['%s'] * len(codes))

bars = fetch(f"""
    SELECT code, ts, open, close
    FROM stocks_intraday
    WHERE code IN ({ph})
      AND ts >= %s
      AND ts::time IN ('09:00:00','15:24:00','15:30:00')
    ORDER BY code, ts
""", tuple(codes) + (START,))
bars['ts'] = pd.to_datetime(bars['ts'])
bars['date'] = bars['ts'].dt.normalize()
bars['t'] = bars['ts'].dt.strftime('%H:%M')
print(f"  ユニバース: {len(codes)}, バー: {len(bars):,}")

# ピボット: 各 (code,date) の 15:24 close, 15:30 close, 09:00 open
piv = bars.pivot_table(index=['code', 'date'], columns='t',
                       values=['open', 'close'], aggfunc='first')
piv.columns = [f'{a}_{b}' for a, b in piv.columns]
piv = piv.reset_index()
# 必要列: close_15:24, close_15:30, open_09:00
piv = piv.rename(columns={'close_15:24': 'c24', 'close_15:30': 'c30', 'open_09:00': 'o9'})
piv = piv.sort_values(['code', 'date'])

# 翌営業日の 09:00 open を引当て
piv['next_o9'] = piv.groupby('code')['o9'].shift(-1)
piv['next_date'] = piv.groupby('code')['date'].shift(-1)

# シグナルとリターン
piv = piv.dropna(subset=['c24', 'c30', 'next_o9'])
piv = piv[(piv['c24'] > 0) & (piv['c30'] > 0) & (piv['next_o9'] > 0)]
piv['close_jump'] = piv['c30'] / piv['c24'] - 1
piv['overnight'] = piv['next_o9'] / piv['c30'] - 1
# 翌日寄り→引け(日中)も見る用に翌日の c30 (引け) を結合
piv['next_c30'] = piv.groupby('code')['c30'].shift(-1)
piv['next_intraday'] = np.where(piv['next_c30'].notna() & (piv['next_o9'] > 0),
                                piv['next_c30'] / piv['next_o9'] - 1, np.nan)

print(f"  有効サンプル: {len(piv):,} (code×day)")
print(f"  close_jump 分布: mean={piv['close_jump'].mean()*1e4:.1f}bps "
      f"std={piv['close_jump'].std()*1e4:.1f}bps "
      f"|jump|>30bps の割合={ (piv['close_jump'].abs()>0.003).mean()*100:.1f}%")

# 期間分割
piv['date'] = pd.to_datetime(piv['date'])
mid = piv['date'].quantile(0.5)
piv['period'] = np.where(piv['date'] <= mid, 'IS', 'OOS')
print(f"  IS/OOS 分割日: {pd.Timestamp(mid).date()}")


def sharpe(x, ann=np.sqrt(252)):
    x = pd.Series(x).dropna()
    if len(x) < 10 or x.std() == 0:
        return float('nan')
    return float(x.mean() / x.std() * ann)


# ============================================
# A. close_jump と overnight の相関
# ============================================
print("\n" + "=" * 76)
print("A. close_jump と 翌朝overnight リターンの関係 (反転なら負相関)")
print("=" * 76)

for label, sub in [('全期間', piv), ('IS', piv[piv.period == 'IS']), ('OOS', piv[piv.period == 'OOS'])]:
    s = sub[['close_jump', 'overnight']].dropna()
    rho, p = spearmanr(s['close_jump'], s['overnight'])
    # 日次クロスセクションIC平均
    ics = []
    for _, g in sub.groupby('date'):
        gg = g[['close_jump', 'overnight']].dropna()
        if len(gg) >= 10:
            ic, _ = spearmanr(gg['close_jump'], gg['overnight'])
            ics.append(ic)
    ic_mean = np.nanmean(ics) if ics else float('nan')
    print(f"  {label:<8} n={len(s):>6}  全体ρ={rho:+.4f} (p={p:.1e})  日次平均IC={ic_mean:+.4f}")

# ============================================
# B. close_jump 分位別 overnight
# ============================================
print("\n" + "=" * 76)
print("B. close_jump 十分位別 翌朝overnight (bps) — 全期間")
print("=" * 76)
piv_v = piv.dropna(subset=['close_jump', 'overnight']).copy()
piv_v['decile'] = pd.qcut(piv_v['close_jump'], 10, labels=False, duplicates='drop')
print(f"\n  {'分位':<6} {'jump平均bps':>12} {'overnight平均bps':>16} {'n':>7} {'勝率%':>7}")
print("  " + "-" * 52)
for d, g in piv_v.groupby('decile'):
    print(f"  D{int(d):<5} {g['close_jump'].mean()*1e4:>12.1f} {g['overnight'].mean()*1e4:>16.2f} "
          f"{len(g):>7} {(g['overnight']>0).mean()*100:>7.1f}")

# ============================================
# C. クロスセクション L/S (top jump=Short / bottom jump=Long, overnight保有)
# ============================================
print("\n" + "=" * 76)
print("C. クロスセクション L/S: 引けで top10%をShort / bottom10%をLong → 翌朝決済")
print("=" * 76)

def ls_daily(sub, q=0.1):
    out = []
    for dt, g in sub.groupby('date'):
        g = g.dropna(subset=['close_jump', 'overnight'])
        if len(g) < 20:
            continue
        k = max(1, int(len(g) * q))
        ranked = g.sort_values('close_jump')
        longs = ranked.head(k)['overnight'].mean()    # jump最小(下落) → 反発Long
        shorts = ranked.tail(k)['overnight'].mean()    # jump最大(上昇) → 反転Short
        out.append({'date': dt, 'ls': longs - shorts, 'long': longs, 'short_': shorts})
    return pd.DataFrame(out)

COST_RT = 0.0010  # 往復10bps (引け板寄せentry + 翌寄りexit, L/Sで両サイド)
for label, sub in [('全期間', piv_v), ('IS', piv_v[piv_v.period == 'IS']), ('OOS', piv_v[piv_v.period == 'OOS'])]:
    ld = ls_daily(sub)
    if len(ld) < 10:
        print(f"  {label}: サンプル不足"); continue
    gross = ld['ls']
    net = gross - COST_RT
    print(f"  {label:<8} n日={len(ld):>4}  gross Sh={sharpe(gross):>5.2f}  net Sh={sharpe(net):>5.2f}  "
          f"gross平均={gross.mean()*1e4:>6.1f}bps  net平均={net.mean()*1e4:>6.1f}bps  勝率={ (net>0).mean()*100:.0f}%")

# コスト感度 (全期間)
print("\n  コスト感度 (全期間 net Sharpe):")
ld_all = ls_daily(piv_v)
for c in [0, 5, 10, 20]:
    net = ld_all['ls'] - c/1e4
    print(f"    往復{c:>2}bps: net Sh={sharpe(net):+.2f}  net平均={net.mean()*1e4:+.1f}bps/日")

# ============================================
# D. 月末効果
# ============================================
print("\n" + "=" * 76)
print("D. 月末 (リバランス日) のインバランス増幅効果")
print("=" * 76)
piv_v['dom'] = piv_v['date'].dt.day
piv_v['is_monthend'] = piv_v['date'].dt.is_month_end
# 月末近傍5営業日
piv_v['month'] = piv_v['date'].dt.to_period('M')
piv_v['rank_from_end'] = piv_v.groupby(['code','month'])['date'].rank(ascending=False)
for label, mask in [('月末5営業日', piv_v['rank_from_end'] <= 5), ('それ以外', piv_v['rank_from_end'] > 5)]:
    sub = piv_v[mask]
    s = sub[['close_jump','overnight']].dropna()
    rho,_ = spearmanr(s['close_jump'], s['overnight']) if len(s)>10 else (np.nan,0)
    ld = ls_daily(sub)
    sh = sharpe(ld['ls']-COST_RT) if len(ld)>=10 else float('nan')
    print(f"  {label:<12} n={len(sub):>6}  jump×overnight ρ={rho:+.4f}  L/S net Sh={sh:+.2f}")

# 保存
piv_v[['code','date','close_jump','overnight','next_intraday','period']].to_csv(
    os.path.join(HERE, 'observations.csv'), index=False)
print(f"\n  保存: observations.csv")
print("\n完了")
