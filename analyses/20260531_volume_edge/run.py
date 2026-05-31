"""
出来高(volume)エッジの探索検証

レポート柱の周辺。出来高は単体アルファより条件変数になりがちだが、未検証の角度を探る。
クロスセクション (流動性上位500) で日足ベースに以下を検証:

  A. 出来高ショック (volume / 20日平均) 分位 → 翌日リターン (反転 or 継続)
  B. 出来高×価格方向: 大商い陽線/陰線 の翌日方向 (出来高を伴う動きは継続? 反転?)
  C. Amihud非流動性 (|ret|/turnover) → 翌日/5日リターン (非流動性プレミアム)
  D. 出来高急減 (枯れ) → リターン
  すべて TOPIX超過(市場中立)・コスト前後・IS/OOS

データ: stocks_daily (volume, turnover_value, adj_close), index_daily TOPIX(0000)
期間: 2021-01〜 / IS-OOS 2024-01
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
OOS = pd.Timestamp('2024-01-01')
COST = 20.0  # 往復bps (L/S翌日決済)


def fetch(sql, params=None):
    conn = psycopg2.connect(**PG); df = pd.read_sql(sql, conn, params=params); conn.close(); return df


print("="*76); print("出来高(volume)エッジ探索"); print("="*76)

uni = fetch("""
    SELECT code FROM stocks_daily WHERE date>='2024-05-01' AND turnover_value>0
    GROUP BY code ORDER BY AVG(turnover_value) DESC LIMIT 500
""")['code'].tolist()
ph = ','.join(['%s']*len(uni))
px = fetch(f"""
    SELECT code, date, adj_open::float ao, adj_close::float ac,
           volume::float vol, turnover_value::float tv
    FROM stocks_daily WHERE code IN ({ph}) AND date>='2020-09-01' AND adj_close>0 AND volume>0
    ORDER BY code, date
""", tuple(uni))
px['date'] = pd.to_datetime(px['date'])
idx = fetch("SELECT date, close::float c FROM index_daily WHERE code='0000' ORDER BY date")
idx['date'] = pd.to_datetime(idx['date']); idx = idx.set_index('date')['c'].sort_index()
print(f"  ユニバース {len(uni)}, 行 {len(px):,}")

# 特徴量
px = px.sort_values(['code', 'date'])
g = px.groupby('code')
px['ret1'] = g['ac'].pct_change()
px['vol_ma20'] = g['vol'].transform(lambda s: s.rolling(20).mean())
px['vol_shock'] = px['vol'] / px['vol_ma20']                    # 出来高ショック
px['amihud'] = px['ret1'].abs() / (px['tv'] / 1e8)             # |ret|/売買代金(億)
px['amihud_ma'] = g['amihud'].transform(lambda s: s.rolling(20).mean())
px['day_ret'] = px['ac'] / px['ao'] - 1                        # 当日 open→close
# 翌日リターン (翌日 open→close), 5日先
px['fwd1'] = g['ac'].shift(-1) / g['ao'].shift(-1) - 1
px['fwd5'] = g['ac'].shift(-5) / g['ao'].shift(-1) - 1
# 市場中立 (TOPIX当日 open→close を日次で引く)
idx_oc = (idx / idx.shift(0))  # placeholder
# TOPIX日次 open→close は無いので close→close で近似
tret = idx.pct_change()
px['mkt_fwd1'] = px['date'].map(tret.shift(-1))
px = px.dropna(subset=['vol_shock', 'fwd1', 'mkt_fwd1']).copy()
px['xs_fwd1'] = (px['fwd1'] - px['mkt_fwd1']) * 1e4   # bps, 市場超過
px['period'] = np.where(px['date'] >= OOS, 'OOS', 'IS')
print(f"  有効サンプル {len(px):,}")


def ic_ls(sub, fac, q=0.1):
    """日次クロスセクション: 上位q Long / 下位q Short の xs_fwd1 spread と日次Sharpe"""
    daily = []
    ics = []
    for dt, gg in sub.groupby('date'):
        s = gg[[fac, 'xs_fwd1']].dropna()
        if len(s) < 30: continue
        ic, _ = spearmanr(s[fac], s['xs_fwd1']); ics.append(ic)
        k = max(1, int(len(s)*q))
        r = s.sort_values(fac)
        daily.append(r.tail(k)['xs_fwd1'].mean() - r.head(k)['xs_fwd1'].mean())  # 高fac Long
    d = pd.Series(daily)
    sh = d.mean()/d.std()*np.sqrt(252) if len(d) >= 10 and d.std() > 0 else np.nan
    return np.nanmean(ics), d.mean(), sh, len(d)


print("\n"+"="*76); print("A. 出来高ショック(vol/20日平均) → 翌日 市場超過リターン"); print("="*76)
print("  高ショックLong/低Short のL/S (高出来高銘柄が継続なら正)")
print(f"  {'期間':<10} {'IC':>9} {'L/S gross/日bps':>16} {'net(20bps)':>11} {'Sharpe(gross)':>13}")
for label, sub in [('全期間', px), ('IS', px[px.period=='IS']), ('OOS', px[px.period=='OOS'])]:
    ic, mean, sh, n = ic_ls(sub, 'vol_shock')
    print(f"  {label:<10} {ic:>9.4f} {mean:>16.1f} {mean-COST:>11.1f} {sh:>13.2f}")

print("\n"+"="*76); print("B. 出来高×当日方向: 大商い(shock>2)の陽線/陰線 → 翌日"); print("="*76)
big = px[px['vol_shock'] >= 2.0]
print(f"  大商い日 n={len(big):,}")
print(f"  {'当日方向':<14} {'期間':<8} {'n':>7} {'翌日xs_fwd1 bps':>15} {'勝率%':>7}")
for sign, slab in [(1, '陽線(up)'), (-1, '陰線(down)')]:
    sub0 = big[np.sign(big['day_ret']) == sign]
    for label, s in [('全期間', sub0), ('IS', sub0[sub0.period=='IS']), ('OOS', sub0[sub0.period=='OOS'])]:
        d = s['xs_fwd1']
        if len(d) < 10: continue
        print(f"  {slab:<14} {label:<8} {len(s):>7} {d.mean():>15.1f} {(d>0).mean()*100:>7.1f}")
    print()

print("="*76); print("C. Amihud非流動性(|ret|/売買代金) → 翌日 市場超過 (高=非流動)"); print("="*76)
print(f"  {'期間':<10} {'IC':>9} {'L/S gross/日bps':>16} {'net(20bps)':>11} {'Sharpe':>9}")
for label, sub in [('全期間', px), ('IS', px[px.period=='IS']), ('OOS', px[px.period=='OOS'])]:
    ic, mean, sh, n = ic_ls(sub, 'amihud_ma')
    print(f"  {label:<10} {ic:>9.4f} {mean:>16.1f} {mean-COST:>11.1f} {sh:>9.2f}")

print("\n"+"="*76); print("D. 出来高急減(枯れ vol_shock<0.7) → 翌日 市場超過"); print("="*76)
dry = px[px['vol_shock'] < 0.7]
for label, s in [('全期間', dry), ('IS', dry[dry.period=='IS']), ('OOS', dry[dry.period=='OOS'])]:
    d = s['xs_fwd1']
    print(f"  {label:<10} n={len(s):>7} 翌日xs_fwd1平均={d.mean():>7.1f}bps 勝率={(d>0).mean()*100:.1f}%")

# 出来高ショック分位の素の翌日リターン(単調性確認)
print("\n"+"="*76); print("E. 出来高ショック 五分位別 翌日市場超過 (単調性)"); print("="*76)
px['vq'] = pd.qcut(px['vol_shock'].rank(method='first'), 5, labels=False)
print(f"  {'分位':<6} {'shock中央':>10} {'翌日xs_fwd1 bps':>15} {'n':>8}")
for q, gg in px.groupby('vq'):
    print(f"  Q{int(q):<5} {gg['vol_shock'].median():>10.2f} {gg['xs_fwd1'].mean():>15.1f} {len(gg):>8}")

px[['code','date','vol_shock','amihud_ma','day_ret','xs_fwd1','period']].to_csv(
    os.path.join(HERE,'vol_obs.csv'), index=False)
print("\n  保存: vol_obs.csv\n完了")
