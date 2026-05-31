"""
新規戦略20本 一括スクリーニング — ファンダ/マクロ/ペア/イベント編 (#17-20 + 追加)

#17 会社予想 上方修正ドリフト (FOP今回 vs 前回FOP up → drift)
#18 配当予想 増配 → drift (FDivAnn up)
#19 USDJPY前日大幅変動 → 輸出株(自動車/電機)翌日
#20 ペア乖離 (8035 vs 6857 等 高相関ペア z-score 平均回帰)
追加#21 大幅安日リバウンド (ret1<=-5% → 翌日)
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
OOS = pd.Timestamp('2024-01-01')


def fetch(sql, p=None):
    c = psycopg2.connect(**PG); d = pd.read_sql(sql, c, params=p); c.close(); return d


def num(x):
    try: v=float(x); return v if np.isfinite(v) else np.nan
    except (TypeError,ValueError): return np.nan

def shp(x, ann):
    x=pd.Series(x).dropna()
    return float(x.mean()/x.std()*ann) if len(x)>=10 and x.std()>0 else float('nan')

print("="*78); print("新規戦略 ファンダ/マクロ/ペア編 (#17-21)"); print("="*78)

# 価格 (流動性上位500)
uni=fetch("""SELECT code FROM stocks_daily WHERE date>='2024-05-01' AND turnover_value>0
  GROUP BY code ORDER BY AVG(turnover_value) DESC LIMIT 500""")['code'].tolist()
ph=','.join(['%s']*len(uni))
px=fetch(f"""SELECT code,date,adj_open::float ao,adj_close::float ac FROM stocks_daily
  WHERE code IN ({ph}) AND date>='2020-06-01' AND adj_close>0 ORDER BY code,date""",tuple(uni))
px['date']=pd.to_datetime(px['date'])
idx=fetch("SELECT date,close::float c FROM index_daily WHERE code='0000' ORDER BY date")
idx['date']=pd.to_datetime(idx['date']); idx=idx.set_index('date')['c'].sort_index()

# 各銘柄の date->price 辞書 & 先行リターン関数
pxi=px.set_index(['code','date']).sort_index()
def fwd_xs(code, d0, hold):
    try:
        sd=pxi.loc[code]
    except KeyError: return np.nan
    fut=sd[sd.index>d0]
    if len(fut)<hold+1: return np.nan
    e=fut.iloc[0]['ao']; x=fut.iloc[hold]['ac']
    if e<=0 or x<=0: return np.nan
    i0=idx.index.searchsorted(fut.index[0]); i1=idx.index.searchsorted(fut.index[hold])
    if i1>=len(idx) or i0>=len(idx): return np.nan
    m=idx.iloc[i1]/idx.iloc[i0]-1
    return (x/e-1-m)*1e4

# ========== #17 会社予想 上方修正ドリフト ==========
print("\n#17 会社予想(FOP)上方修正 → ドリフト (d5/d20, 市場超過bps)")
ev=fetch("""SELECT DISTINCT ON (code,disc_date) code,disc_date,cur_per_type,payload
  FROM fin_summary WHERE disc_date>='2021-01-01' ORDER BY code,disc_date,disc_time""")
ev['disc_date']=pd.to_datetime(ev['disc_date'])
ev['fop']=ev['payload'].apply(lambda p:num((p if isinstance(p,dict) else json.loads(p)).get('FOP')))
ev=ev.dropna(subset=['fop']).sort_values(['code','disc_date'])
ev['fop_prev']=ev.groupby('code')['fop'].shift(1)
ev['rev']=(ev['fop']-ev['fop_prev'])/ev['fop_prev'].abs().clip(lower=1.0)
ev=ev[np.isfinite(ev['rev'])]
ev['code']=ev['code'].astype(str)
up=ev[ev['rev']>=0.05]   # 5%以上上方修正
dn=ev[ev['rev']<=-0.05]
for lab,sub in [('上方修正(≥+5%)',up),('下方修正(≤-5%)',dn)]:
    s=sub.copy(); s['d20']=[fwd_xs(c,d,20) for c,d in zip(s['code'],s['disc_date'])]
    s=s.dropna(subset=['d20']); s['period']=np.where(s['disc_date']>=OOS,'OOS','IS')
    if len(s)<20: print(f"  {lab}: n<20"); continue
    print(f"  {lab}: n={len(s):>4} d20平均={s['d20'].mean():>6.0f}bps 勝率={(s['d20']>0).mean()*100:.0f}% "
          f"IS={s[s.period=='IS']['d20'].mean():.0f} OOS={s[s.period=='OOS']['d20'].mean():.0f}")

# ========== #18 増配(FDivAnn上方) → ドリフト ==========
print("\n#18 配当予想 増配(FDivAnn up) → ドリフト (d20, 市場超過bps)")
ev['fdiv']=ev['payload'].apply(lambda p:num((p if isinstance(p,dict) else json.loads(p)).get('FDivAnn')))
ev2=ev.dropna(subset=['fdiv']).sort_values(['code','disc_date']).copy()
ev2['fdiv_prev']=ev2.groupby('code')['fdiv'].shift(1)
ev2['div_up']=ev2['fdiv']>ev2['fdiv_prev']*1.0001
inc=ev2[ev2['div_up']]
s=inc.copy(); s['d20']=[fwd_xs(c,d,20) for c,d in zip(s['code'],s['disc_date'])]
s=s.dropna(subset=['d20']); s['period']=np.where(s['disc_date']>=OOS,'OOS','IS')
if len(s)>=20:
    print(f"  増配: n={len(s):>4} d20平均={s['d20'].mean():>6.0f}bps 勝率={(s['d20']>0).mean()*100:.0f}% "
          f"IS={s[s.period=='IS']['d20'].mean():.0f} OOS={s[s.period=='OOS']['d20'].mean():.0f}")
else: print(f"  増配: n={len(s)}<20")

# ========== #19 USDJPY前日変動 → 輸出株翌日 ==========
print("\n#19 USDJPY前日変動 → 輸出株(自動車/電機)翌日リターン")
fx=fetch("SELECT trade_date AS date, close FROM macro.daily_ohlcv WHERE symbol='JPY=' ORDER BY trade_date")
if len(fx)==0:
    fx=fetch("SELECT date, close FROM macro.daily_ohlcv WHERE symbol ILIKE '%JPY%' ORDER BY date LIMIT 5")
    print(f"  JPY=なし。候補: {fx.to_dict('records')[:3] if len(fx) else 'macro.daily_ohlcv要確認'}")
else:
    fx['date']=pd.to_datetime(fx['date']); fx=fx.set_index('date')['close'].sort_index()
    fx_ret=fx.pct_change()
    EXP=['72030','72670','69020','69540','79740']  # トヨタ,ホンダ,デンソー,ファナック,任天堂
    ex=px[px['code'].astype(str).isin(EXP)].copy()
    ex['day_ret']=ex.groupby('code')['ac'].pct_change()
    ex['fx_prev']=ex['date'].map(fx_ret.shift(0))  # 当日FX(前日比) → 翌日株? 同日相関+予測力
    # 円安(fx_ret>0=JPY/USD? 確認)とのspearman
    s=ex.dropna(subset=['day_ret','fx_prev'])
    rho,p=spearmanr(s['fx_prev'],s['day_ret'])
    print(f"  輸出株 当日リターン vs 当日FX変動: ρ={rho:+.3f}(p={p:.1e}) ※同時相関")
    # 予測: 前日FX → 翌日株
    ex['fx_yest']=ex['date'].map(fx_ret.shift(1))
    s2=ex.dropna(subset=['day_ret','fx_yest'])
    rho2,p2=spearmanr(s2['fx_yest'],s2['day_ret'])
    print(f"  輸出株 当日リターン vs 前日FX変動(予測): ρ={rho2:+.3f}(p={p2:.2f})")

# ========== #20 ペア乖離 平均回帰 ==========
print("\n#20 ペア乖離 平均回帰 (高相関ペア z-score, 5日保有, bps)")
PAIRS=[('80350','68570'),('72030','72670'),('63670','65010'),('99840','99830')]  # TEL-アドテスト,トヨタ-ホンダ,ダイキン-日立,SBG-ファストリ
for a,b in PAIRS:
    da=pxi.loc[a]['ac'] if a in pxi.index.get_level_values(0) else None
    db=pxi.loc[b]['ac'] if b in pxi.index.get_level_values(0) else None
    if da is None or db is None: print(f"  {a}-{b}: データなし"); continue
    j=pd.DataFrame({'a':da,'b':db}).dropna()
    j['lr']=np.log(j['a'])-np.log(j['b'])
    j['z']=(j['lr']-j['lr'].rolling(60).mean())/j['lr'].rolling(60).std()
    j['fa']=j['a'].shift(-5)/j['a']-1; j['fb']=j['b'].shift(-5)/j['b']-1
    # z>2: aが割高→a売りb買い, z<-2: a買いb売り
    sig=j[j['z'].abs()>=2].copy()
    sig['ret']=np.where(sig['z']>0, -(sig['fa']-sig['fb']), (sig['fa']-sig['fb']))*1e4 - 16  # cost8bps×2
    sig=sig.dropna(subset=['ret'])
    if len(sig)<10: print(f"  {a}-{b}: n={len(sig)}<10"); continue
    sig['period']=np.where(sig.index>=OOS,'OOS','IS')
    print(f"  {a}-{b}: n={len(sig):>4} mean={sig['ret'].mean():>6.0f}bps Sh={shp(sig['ret'],np.sqrt(252/5)):>5.2f} "
          f"勝率={(sig['ret']>0).mean()*100:.0f}%")

# ========== #21 大幅安日リバウンド ==========
print("\n#21 大幅安日(当日≤-5%)→翌日リバウンド (市場超過bps)")
px['day_ret']=px.groupby('code')['ac'].pct_change()
# 翌日 open→close
px['nxt_oc']=px.groupby('code')['ac'].shift(-1)/px.groupby('code')['ao'].shift(-1)-1
crash=px[px['day_ret']<=-0.05].dropna(subset=['nxt_oc']).copy()
crash['period']=np.where(crash['date']>=OOS,'OOS','IS')
if len(crash)>=20:
    print(f"  大幅安翌日: n={len(crash):>5} 翌日o→c平均={crash['nxt_oc'].mean()*1e4:>6.1f}bps 勝率={(crash['nxt_oc']>0).mean()*100:.0f}% "
          f"IS={crash[crash.period=='IS']['nxt_oc'].mean()*1e4:.0f} OOS={crash[crash.period=='OOS']['nxt_oc'].mean()*1e4:.0f}")

print("\n[ファンダ/マクロ/ペア編 完了]")
