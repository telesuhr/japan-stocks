"""
新規戦略20本 一括スクリーニング・バックテスト

これまでの教訓を全織り込み:
  - 年率化は √(252/hold) (オーバーラップの幻回避)
  - IS/OOS分割 (2024-01)
  - コスト込み (リバランス毎に往復bps)
  - 市場(TOPIX)超過リターンで評価 (βの幻回避)
  - 5日保有・5日毎リバランス (非重複近似・ターンオーバー抑制)

本スクリプトは「クロスセクション系」と「タイミング/季節系」を一括検証。
ファンダ(会社予想)・マクロ(USDJPY)・ペアは別スクリプト(run_b.py)。

20アイデア (詳細はREADME):
  CS系 (クロスセクション5日L/S decile): #1-12
  タイミング系 (ユニバース等加重 long): #13-16
  別途(run_b): #17-20
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
HOLD = 5
COST_RT = 20.0  # 往復bps (decile L/S 5日毎リバランス)
ANN = np.sqrt(252 / HOLD)


def fetch(sql, p=None):
    c = psycopg2.connect(**PG); d = pd.read_sql(sql, c, params=p); c.close(); return d


print("="*78); print("新規戦略20本 一括スクリーニング (CS系+タイミング系)"); print("="*78)

uni = fetch("""SELECT code FROM stocks_daily WHERE date>='2024-05-01' AND turnover_value>0
  GROUP BY code ORDER BY AVG(turnover_value) DESC LIMIT 400""")['code'].tolist()
ph = ','.join(['%s']*len(uni))
px = fetch(f"""SELECT code,date,adj_open::float ao,adj_high::float ah,adj_low::float al,
  adj_close::float ac,volume::float vol,turnover_value::float tv
  FROM stocks_daily WHERE code IN ({ph}) AND date>='2018-06-01' AND adj_close>0 AND volume>0
  ORDER BY code,date""", tuple(uni))
px['date'] = pd.to_datetime(px['date'])
idx = fetch("SELECT date,close::float c FROM index_daily WHERE code='0000' ORDER BY date")
idx['date'] = pd.to_datetime(idx['date']); idx = idx.set_index('date')['c'].sort_index()
mkt_ret1 = idx.pct_change()
print(f"  ユニバース {len(uni)}, 行 {len(px):,}")

px = px.sort_values(['code','date'])
g = px.groupby('code')
px['ret1'] = g['ac'].pct_change()
px['ac_fwd'] = g['ac'].shift(-HOLD)
px['fwd5'] = px['ac_fwd']/px['ac'] - 1
px['mkt_fwd5'] = px['date'].map((idx.shift(-HOLD)/idx - 1))
px['xs_fwd5'] = px['fwd5'] - px['mkt_fwd5']
# 特徴量 (group shift で安全に)
px['ret5']  = px['ac']/g['ac'].shift(5) - 1
px['ret20'] = px['ac']/g['ac'].shift(20) - 1
px['ret60'] = px['ac']/g['ac'].shift(60) - 1
px['mom_skip'] = g['ac'].shift(5)/g['ac'].shift(65) - 1
px['max252'] = g['ac'].transform(lambda s: s.rolling(252).max())
px['prox52w'] = px['ac']/px['max252']
px['vol20'] = g['ret1'].transform(lambda s: s.rolling(20).std())
px['range_today'] = (px['ah']-px['al'])/px['ac']
px['gap'] = px['ao']/g['ac'].shift(1) - 1
px['overnight'] = px['ao']/g['ac'].shift(1) - 1
px['intraday'] = px['ac']/px['ao'] - 1
px['vol_ma20'] = g['vol'].transform(lambda s: s.rolling(20).mean())
px['vol_shock'] = px['vol']/px['vol_ma20']
px['tv_ma5'] = g['tv'].transform(lambda s: s.rolling(5).mean())
px['tv_ma20'] = g['tv'].transform(lambda s: s.rolling(20).mean())
px['turn_chg'] = px['tv_ma5']/px['tv_ma20'] - 1
px['skew20'] = g['ret1'].transform(lambda s: s.rolling(20).skew())

px['period'] = np.where(px['date']>=OOS,'OOS','IS')

# CS signals: (name, factor_col, direction) direction=+1: 高い値Long
CS = [
    ('#1 ST反転(5d loser)',      'ret5',     -1),
    ('#2 中期モメンタム60d',       'ret60',    +1),
    ('#3 12-1モメンタム(skip5)',   'mom_skip', +1),
    ('#4 52週高値接近',           'prox52w',  +1),
    ('#5 低ボラ(low vol)',        'vol20',    -1),
    ('#6 レンジ反転(高range loser)','range_today', -1),  # 後で符号調整
    ('#7 ギャップ反転',           'gap',      -1),
    ('#8 オーバーナイト継続',      'overnight',+1),
    ('#9 日中反転',               'intraday', -1),
    ('#10 出来高確認モメンタム',    'volmom',   +1),
    ('#11 売買代金トレンド',       'turn_chg', +1),
    ('#12 リターン歪度(低skew)',   'skew20',   -1),
]
px['volmom'] = np.sign(px['ret5'])*px['vol_shock']  # #10

def cs_test(fac, direction):
    sub = px.dropna(subset=[fac,'xs_fwd5']).copy()
    sub['f'] = sub[fac]*direction
    # 5日毎リバランス (日付を5刻みで間引き)
    days = sorted(sub['date'].unique())
    rb = set(days[::HOLD])
    sub = sub[sub['date'].isin(rb)]
    daily, ics = [], []
    for dt,gg in sub.groupby('date'):
        s = gg[['f','xs_fwd5']].dropna()
        if len(s)<40: continue
        ic,_ = spearmanr(s['f'],s['xs_fwd5']); ics.append(ic)
        k=max(1,int(len(s)*0.1)); r=s.sort_values('f')
        daily.append({'date':dt,'ls':(r.tail(k)['xs_fwd5'].mean()-r.head(k)['xs_fwd5'].mean())*1e4})
    d=pd.DataFrame(daily)
    if len(d)<10: return None
    d['period']=np.where(d['date']>=OOS,'OOS','IS')
    out={'ic':np.nanmean(ics)}
    for p,dd in [('all',d),('is',d[d.period=='IS']),('oos',d[d.period=='OOS'])]:
        net=dd['ls']-COST_RT
        sh=net.mean()/net.std()*ANN if len(net)>=8 and net.std()>0 else np.nan
        out[p]=sh; out[p+'_g']=dd['ls'].mean()
    return out

print("\n"+"="*78); print("CS系 (5日L/S decile, 市場超過, 往復20bps, 年率√50.4)"); print("="*78)
print(f"  {'戦略':<26}{'IC':>8}{'gross/RB':>9}{'net Sh(all)':>12}{'IS':>7}{'OOS':>7}  判定")
print("  "+"-"*72)
results=[]
for name,fac,dr in CS:
    r=cs_test(fac,dr)
    if r is None: print(f"  {name:<26} データ不足"); continue
    v='○' if (r['all']==r['all'] and r['all']>=1.0 and r['is']>0 and r['oos']>0) else \
      ('△' if r['all']==r['all'] and r['all']>0.5 else '✗')
    print(f"  {name:<26}{r['ic']:>8.3f}{r['all_g']:>9.1f}{r['all']:>12.2f}{r['is']:>7.2f}{r['oos']:>7.2f}  {v}")
    results.append({'strategy':name,'ic':round(r['ic'],4),'gross_bps':round(r['all_g'],1),
                    'net_sharpe':round(r['all'],2),'is':round(r['is'],2),'oos':round(r['oos'],2),'verdict':v})

# ============ タイミング系 ============
print("\n"+"="*78); print("タイミング系 (ユニバース等加重, TOPIX超過 or 絶対)"); print("="*78)

# ユニバース等加重日次リターン
ew = px.groupby('date')['ret1'].mean().dropna()
ew_xs = ew - mkt_ret1.reindex(ew.index)  # 超過(ほぼ0なので絶対も見る)
cal = pd.DataFrame({'ew':ew}); cal.index=pd.to_datetime(cal.index)
cal['mkt']=mkt_ret1.reindex(cal.index)
cal['period']=np.where(cal.index>=OOS,'OOS','IS')
cal['dom']=cal.index.day; cal['month']=cal.index.month; cal['dow']=cal.index.dayofweek
cal['mend_rank']=cal.groupby(cal.index.to_period('M')).cumcount(ascending=False)

def timing(mask,label,series='mkt'):
    s=cal[mask][series].dropna()
    if len(s)<20: print(f"  {label:<30} n<20"); return
    full=cal[series].dropna()
    def shp(x): return x.mean()/x.std()*np.sqrt(252) if x.std()>0 else np.nan
    is_s=cal[mask&(cal.period=='IS')][series]; oos_s=cal[mask&(cal.period=='OOS')][series]
    # マーケットを「条件日だけ保有」した時のSharpe vs 全日保有
    print(f"  {label:<30} n={len(s):>4} 条件日Sh={shp(s):>5.2f}(全日{shp(full):>5.2f}) IS={shp(is_s):>5.2f} OOS={shp(oos_s):>5.2f} 平均={s.mean()*1e4:>6.1f}bps")

print("  #13 ターン・オブ・マンス (月末2日+月初3日 long, TOPIX)")
timing((cal['mend_rank']<=1)|(cal['dom']<=3),'   TOM日 long')
print("  #14 曜日 (月曜/金曜, TOPIX)")
timing(cal['dow']==0,'   月曜 long'); timing(cal['dow']==4,'   金曜 long')
print("  #15 季節性 (11-4月 long / 5-10月, TOPIX)")
timing(cal['month'].isin([11,12,1,2,3,4]),'   11-4月 long'); timing(cal['month'].isin([5,6,7,8,9,10]),'   5-10月 long')
print("  #16 指数RSI過売り→翌日 (TOPIX RSI14<30の翌日 long, TOPIX絶対)")
d_idx=pd.DataFrame({'r':mkt_ret1}); delta=idx.diff()
up=delta.clip(lower=0).rolling(14).mean(); dn=(-delta.clip(upper=0)).rolling(14).mean()
rsi=100-100/(1+up/dn); d_idx['rsi_prev']=rsi.shift(1); d_idx['period']=np.where(d_idx.index>=OOS,'OOS','IS')
osub=d_idx[d_idx['rsi_prev']<30]['r'].dropna()
def shp(x): return x.mean()/x.std()*np.sqrt(252) if len(x)>=10 and x.std()>0 else np.nan
print(f"     RSI<30翌日 long: n={len(osub)} Sh={shp(osub):.2f} 平均={osub.mean()*1e4:.1f}bps (全日{shp(d_idx['r'].dropna()):.2f})")

pd.DataFrame(results).to_csv(os.path.join(HERE,'cs_results.csv'),index=False)
print(f"\n  保存: cs_results.csv")
print("\n[CS系+タイミング系 完了] ファンダ/マクロ/ペアは run_b.py へ")
