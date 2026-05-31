"""
投資部門別売買動向 海外フロー → 市場タイミング検証 (レポート柱3の核心)

レポート主張: 「市場の方向性は海外投資家フローに従属。海外は数週かけてVWAP執行
するため自己相関(連続性)が強くモメンタムを生む。個人は逆張り」。

決定的制約: 投資部門別売買動向は en_date(週末金) に対し pub_date(翌木) で公表。
公表ラグ ~4営業日。先週の海外フローを知るのは翌週木曜 → トレード可能性の肝。

検証:
  A. 海外フローの自己相関 (FrgnBal_t vs FrgnBal_{t+1..4}) = レポートの「連続性」主張
  B. 同時相関 (FrgnBal_t vs 同週index return) = 「市場は海外に従属」(非トレード可能)
  C. 予測力【トレード可能】: FrgnBal(pub_date時点で既知) vs 公表後forward index return
  D. レジーム戦略: 海外4週MAの符号でindexをlong/flat → Sharpe vs buy&hold
  E. 個人(IndBal)・投信(InvTrBal)の逆張り性

データ: investor_types(TSE1st 2016-22 + TSEPrime 2022-26 stitch), index_daily TOPIX(0000)
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


def fetch(sql, params=None):
    conn = psycopg2.connect(**PG); df = pd.read_sql(sql, conn, params=params); conn.close(); return df


print("=" * 76)
print("投資部門別 海外フロー → 市場タイミング検証 (レポート柱3核心)")
print("=" * 76)

# ---- 投資部門別 (主要板 stitch) ----
inv = fetch("""
    SELECT section, st_date, en_date, pub_date, payload
    FROM investor_types WHERE section IN ('TSE1st','TSEPrime')
    ORDER BY en_date
""")
for c in ['st_date', 'en_date', 'pub_date']:
    inv[c] = pd.to_datetime(inv[c])

def pk(p, k):
    p = p if isinstance(p, dict) else (json.loads(p) if p else {})
    try: return float(p.get(k))
    except (TypeError, ValueError): return np.nan

for k in ['FrgnBal', 'IndBal', 'InvTrBal', 'TotBal']:
    inv[k] = inv['payload'].apply(lambda p: pk(p, k))
inv = inv.drop(columns=['payload']).dropna(subset=['FrgnBal']).sort_values('en_date').reset_index(drop=True)
print(f"\n週次データ: {len(inv)} 週 ({inv['en_date'].min().date()}〜{inv['en_date'].max().date()})")
print(f"公表ラグ中央値: {(inv['pub_date']-inv['en_date']).dt.days.median():.0f}日")

# ---- index (TOPIX) ----
idx = fetch("SELECT date, close::float c FROM index_daily WHERE code='0000' ORDER BY date")
idx['date'] = pd.to_datetime(idx['date'])
idx = idx.set_index('date').sort_index()
print(f"指数(TOPIX 0000) 最新値: {idx['c'].iloc[-1]:.1f}")
tdays = idx.index.values

def ret_between(d0, d1):
    """d0以降最初の取引日のclose → d1以降最初の取引日のclose のリターン"""
    i0 = np.searchsorted(tdays, np.datetime64(d0))
    i1 = np.searchsorted(tdays, np.datetime64(d1))
    if i0 >= len(tdays) or i1 >= len(tdays) or i1 <= i0: return np.nan
    return idx['c'].iloc[i1] / idx['c'].iloc[i0] - 1

def ret_fwd(d0, ndays):
    i0 = np.searchsorted(tdays, np.datetime64(d0))
    i1 = i0 + ndays
    if i1 >= len(tdays) or i0 >= len(tdays): return np.nan
    return idx['c'].iloc[i1] / idx['c'].iloc[i0] - 1

# 各週のリターン定義
inv['ret_week'] = inv.apply(lambda r: ret_between(r['st_date'], r['en_date']), axis=1)       # 同時(その週)
inv['ret_pub_1w'] = inv['pub_date'].apply(lambda d: ret_fwd(d, 5))                             # 公表後1週(トレード可能)
inv['ret_pub_2w'] = inv['pub_date'].apply(lambda d: ret_fwd(d, 10))
inv['frgn_next'] = inv['FrgnBal'].shift(-1)
inv['date'] = inv['en_date']
inv['period'] = np.where(inv['en_date'] >= pd.Timestamp('2022-01-01'), 'OOS', 'IS')

def shp(x, ann=np.sqrt(52)):
    x = pd.Series(x).dropna()
    return float(x.mean()/x.std()*ann) if len(x) >= 10 and x.std() > 0 else float('nan')

# ============================================
print("\n" + "=" * 76)
print("A. 海外フローの自己相関 (レポートの『連続性』主張の検証)")
print("=" * 76)
for lag in [1, 2, 4]:
    a = inv['FrgnBal']; b = inv['FrgnBal'].shift(lag)
    s = pd.concat([a, b], axis=1).dropna()
    rho, p = spearmanr(s.iloc[:, 0], s.iloc[:, 1])
    print(f"  FrgnBal lag{lag}週: ρ={rho:+.3f} (p={p:.1e})")
streak = (np.sign(inv['FrgnBal']) == np.sign(inv['FrgnBal'].shift(1))).mean()
print(f"  符号継続率(連続買い/売り越し): {streak*100:.1f}%")

# ============================================
print("\n" + "=" * 76)
print("B. 同時相関 FrgnBal vs その週のindex return (非トレード可能・参考)")
print("=" * 76)
s = inv[['FrgnBal', 'ret_week']].dropna()
rho, p = spearmanr(s['FrgnBal'], s['ret_week'])
print(f"  ρ={rho:+.3f} (p={p:.1e})  → 海外買い越し週は市場上昇 (従属の確認、ただし取引不可)")

# ============================================
print("\n" + "=" * 76)
print("C. 予測力【トレード可能】FrgnBal(公表時既知) vs 公表後 forward return")
print("=" * 76)
for col, lab in [('ret_pub_1w', '公表後1週'), ('ret_pub_2w', '公表後2週')]:
    for plabel, sub in [('全期間', inv), ('IS(〜21)', inv[inv.period=='IS']), ('OOS(22〜)', inv[inv.period=='OOS'])]:
        s = sub[['FrgnBal', col]].dropna()
        if len(s) < 20: continue
        rho, p = spearmanr(s['FrgnBal'], s[col])
        # 上位/下位3分位
        q = pd.qcut(s['FrgnBal'].rank(method='first'), 3, labels=False)
        hi = s[q==2][col].mean()*100; lo = s[q==0][col].mean()*100
        print(f"  {lab} {plabel:<9}: ρ={rho:+.3f}(p={p:.2f})  海外買越多→{hi:+.2f}% / 売越→{lo:+.2f}% / 差={hi-lo:+.2f}%")
    print()

# ============================================
print("=" * 76)
print("D. レジーム戦略: 海外4週MA>0 で TOPIX long / それ以外flat (公表後執行)")
print("=" * 76)
inv['frgn_ma4'] = inv['FrgnBal'].rolling(4).mean()
inv['signal'] = (inv['frgn_ma4'] > 0).astype(int)  # 直近4週平均で買い越し → long
# 公表後1週リターンをシグナルに従って取る
inv['strat_ret'] = inv['signal'] * inv['ret_pub_1w']
bh = inv['ret_pub_1w']  # buy&hold (毎週1週保有)
for plabel, sub in [('全期間', inv), ('IS(〜21)', inv[inv.period=='IS']), ('OOS(22〜)', inv[inv.period=='OOS'])]:
    st = sub['strat_ret'].dropna(); b = sub['ret_pub_1w'].dropna()
    long_frac = sub['signal'].mean()*100
    print(f"  {plabel:<9}: 戦略Sharpe={shp(st):+.2f} (long率{long_frac:.0f}%) vs B&H Sharpe={shp(b):+.2f}  "
          f"戦略平均={st.mean()*100:+.3f}%/週 B&H={b.mean()*100:+.3f}%/週")

# ============================================
print("\n" + "=" * 76)
print("E. 個人(IndBal)・投信(InvTrBal) の逆張り性 vs 公表後forward")
print("=" * 76)
for k in ['IndBal', 'InvTrBal']:
    s = inv[[k, 'ret_pub_1w']].dropna()
    rho, p = spearmanr(s[k], s['ret_pub_1w'])
    sc = inv[[k, 'ret_week']].dropna()
    rho_c, _ = spearmanr(sc[k], sc['ret_week'])
    print(f"  {k:<10}: 同時ρ={rho_c:+.3f}  公表後1週ρ={rho:+.3f}(p={p:.2f})")

inv.to_csv(os.path.join(HERE, 'flow_obs.csv'), index=False)
print(f"\n  保存: flow_obs.csv")
print("\n完了")
