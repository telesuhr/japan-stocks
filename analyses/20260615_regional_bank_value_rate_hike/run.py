#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
利上げ局面で「ファンダ的に出遅れた地銀(低PBR)」を買うとワークするか。

仮説(教訓5: 仮説先行):
  日銀の利上げ(マイナス金利解除 2024-03〜)で銀行は利ざや改善 → 銀行株は再評価。
  特に地方銀行は解散価値割れ(PBR<1)が多く、出遅れた低PBR地銀ほど後から本源価値へ
  キャッチアップする(=利上げレジームで地銀バリュー効果が出る)。
  ただし低PBRは「万年割安=バリュートラップ」の可能性もあるので、
  クロスセクション(地銀の中で低PBR vs 高PBR)で前向きリターン差が出るかを検証する。

手法:
  - ユニバース: 銀行業(sector33) から メガ(Core30) / 大型(Large70=ゆうちょ/りそな/トラスト) / 日銀(8301) を除外 = 地銀
  - 各月末リバランス。PBR = 月末調整後終値 / 直近開示BPS(point-in-time, disc_date<=月末)。先読み無し。
  - 流動性: 直近20日平均売買代金 >= 0.3億円(地銀は薄いので緩め)。満たす銘柄のみ。
  - PBR3分位: 割安(Q1)/中位(Q2)/割高(Q3)。各等加重ポートを翌月リターンで評価。
  - ベンチ: 地銀等加重(セクター内中立) と TOPIX(0000)。
  - コスト: long-only 月次入替の往復20bp相当を控除(地銀は薄いので厚め)。
  - レジーム: pre(〜2024-02) / hike(2024-03〜)。hike内で IS(〜2025-06)/OOS(2025-07〜)。
  - 補助: 価格出遅れ(6カ月リターン下位)版、低PBR×ROE(質)版も比較。教訓1: 必ずforward。
"""
import sys, os
sys.stdout.reconfigure(line_buffering=True)
import numpy as np, pandas as pd, psycopg2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.font_manager as fm
for fp in ["/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
           "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
           "/mnt/c/Windows/Fonts/meiryo.ttc"]:
    if os.path.exists(fp):
        fm.fontManager.addfont(fp); plt.rcParams["font.family"]=fm.FontProperties(fname=fp).get_name(); break
plt.rcParams["axes.unicode_minus"]=False
PG={"host":os.environ.get("PGHOST","localhost"),"port":int(os.environ.get("PGPORT",5432)),
    "user":os.environ.get("PGUSER","postgres"),"dbname":os.environ.get("PGDATABASE","market_data")}
HERE=os.path.dirname(os.path.abspath(__file__))
COST=0.002          # 往復20bp(地銀は流動性薄め)
LIQ=0.3e8           # 0.3億/日
HIKE="2024-03-01"   # 日銀マイナス金利解除
OOS="2025-07-01"

conn=psycopg2.connect(**PG)
uni=pd.read_sql("""
  SELECT code5 code, code4, name_ja, scale_cat FROM symbol_master
  WHERE sector33_nm LIKE '%%銀行%%'
    AND scale_cat NOT IN ('TOPIX Core30','TOPIX Large70') AND code4<>'8301'
""", conn)
codes=tuple(uni['code'])
px=pd.read_sql("""
  SELECT code, date, adj_close c, turnover_value tv FROM stocks_daily
  WHERE code IN %s AND date>='2022-01-01' ORDER BY code,date
""", conn, params=(codes,))
fin=pd.read_sql("""
  SELECT code, disc_date, NULLIF(payload->>'BPS','')::float bps, NULLIF(payload->>'EPS','')::float eps,
         NULLIF(payload->>'NP','')::float np, NULLIF(payload->>'Eq','')::float eq,
         NULLIF(payload->>'DivAnn','')::float divann
  FROM fin_summary WHERE code IN %s AND NULLIF(payload->>'BPS','') IS NOT NULL ORDER BY code,disc_date
""", conn, params=(codes,))
topix=pd.read_sql("SELECT date, close FROM index_daily WHERE code='0000' AND close IS NOT NULL ORDER BY date", conn)
conn.close()
print(f"地銀ユニバース {len(uni)} 銘柄 / 価格 {len(px):,}行 / 財務 {len(fin):,}行")

px['date']=pd.to_datetime(px['date']); px['c']=px['c'].astype(float); px['tv']=px['tv'].astype(float)
fin['disc_date']=pd.to_datetime(fin['disc_date'])
topix['date']=pd.to_datetime(topix['date']); topix['close']=topix['close'].astype(float)

# 月末営業日
px['ym']=px['date'].dt.to_period('M')
month_end=px.groupby('ym')['date'].max().reset_index()
mdates=sorted(month_end['date'].tolist())

# 各銘柄: 月末close, 20日平均代金, 6カ月リターン
def per_code(x):
    x=x.sort_values('date').set_index('date')
    me=x.reindex(mdates, method='ffill')
    me['tv20']=x['tv'].rolling(20).mean().reindex(mdates, method='ffill')
    me['ret_fwd']=me['c'].shift(-1)/me['c']-1     # 翌月末までのリターン
    me['ret_6m']=me['c']/me['c'].shift(6)-1       # 過去6カ月(価格出遅れ用)
    me['code']=x['code'].iloc[0]
    return me.reset_index().rename(columns={'index':'date'})
me=pd.concat([per_code(g) for _,g in px.groupby('code')], ignore_index=True)
me=me.dropna(subset=['c'])

# point-in-time BPS/EPS/NP/Eq を月末にasof結合
fin_s=fin.sort_values('disc_date')
def asof_fin(row):
    f=fin_s[(fin_s['code']==row['code'])&(fin_s['disc_date']<=row['date'])]
    if len(f)==0: return pd.Series({'bps':np.nan,'eps':np.nan,'roe':np.nan,'divy':np.nan})
    r=f.iloc[-1]
    roe=(r['np']/r['eq']*100) if (r['eq'] and r['eq']>0) else np.nan
    return pd.Series({'bps':r['bps'],'eps':r['eps'],'roe':roe,'divy':(r['divann'] if r['divann'] else np.nan)})
me=me.join(me.apply(asof_fin,axis=1))
me['pbr']=me['c']/me['bps']
me['divy']=me['divy']/me['c']*100
me=me[(me['pbr']>0)&(me['pbr']<5)]               # 異常値除去

# リバランス可能母集団: 流動性 + forwardあり
def regime(d):
    if d<pd.Timestamp(HIKE): return 'pre'
    return 'hike_OOS' if d>=pd.Timestamp(OOS) else 'hike_IS'

rows=[]
port_ts={'割安Q1':{}, '割高Q3':{}, '地銀EW':{}, '価格出遅れ':{}, '低PBR×ROE':{}}
for d, g in me.groupby('date'):
    g=g[(g['tv20']>=LIQ)&g['ret_fwd'].notna()&g['pbr'].notna()].copy()
    if len(g)<9: continue
    g['q']=pd.qcut(g['pbr'],3,labels=['Q1','Q2','Q3'])           # Q1=割安
    cheap=g[g['q']=='Q1']; rich=g[g['q']=='Q3']
    ew=g
    # 価格出遅れ: 6カ月リターン下位3分の1
    gp=g[g['ret_6m'].notna()].copy()
    lag=gp[gp['ret_6m']<=gp['ret_6m'].quantile(1/3)] if len(gp)>=9 else g.iloc[0:0]
    # 低PBR×ROE: 割安(下位半分)の中でROE上位半分
    gq=g[g['roe'].notna()].copy()
    if len(gq)>=9:
        cheaphalf=gq[gq['pbr']<=gq['pbr'].median()]
        cq=cheaphalf[cheaphalf['roe']>=cheaphalf['roe'].median()]
    else: cq=g.iloc[0:0]
    for nm,sub in [('割安Q1',cheap),('割高Q3',rich),('地銀EW',ew),('価格出遅れ',lag),('低PBR×ROE',cq)]:
        if len(sub)>0:
            port_ts[nm][d]=sub['ret_fwd'].mean()-COST   # 月次入替コスト控除
    rows.append(dict(date=d, n=len(g), reg=regime(d),
                     cheap=cheap['ret_fwd'].mean(), rich=rich['ret_fwd'].mean(),
                     ew=ew['ret_fwd'].mean(),
                     cheap_pbr=cheap['pbr'].mean(), rich_pbr=rich['pbr'].mean()))

R=pd.DataFrame(rows).set_index('date')

def stats(s):
    s=s.dropna()
    if len(s)<6: return dict(n=len(s),ann=np.nan,sr=np.nan,hit=np.nan,cum=np.nan)
    return dict(n=len(s), ann=s.mean()*12*100, sr=s.mean()/s.std()*np.sqrt(12),
                hit=(s>0).mean()*100, cum=((1+s).prod()-1)*100)

print("\n=== 地銀バリュー(割安Q1)・コスト後・レジーム別 ===")
print(f"{'戦略/レジーム':18}{'N':>4}{'年率%':>8}{'SR':>6}{'勝率%':>7}{'累積%':>8}")
series={k:pd.Series(v) for k,v in port_ts.items()}
for nm in ['割安Q1','割高Q3','地銀EW','価格出遅れ','低PBR×ROE']:
    s=series[nm]
    for reg,lab in [(None,'全期間'),('hike','利上げ後'),('hike_IS','└IS'),('hike_OOS','└OOS')]:
        if reg is None: ss=s
        elif reg=='hike': ss=s[s.index>=pd.Timestamp(HIKE)]
        else: ss=s[[regime(d)==reg for d in s.index]]
        st=stats(ss)
        print(f"{nm+'/'+lab:18}{st['n']:>4}{st['ann']:>8.1f}{st['sr']:>6.2f}{st['hit']:>7.0f}{st['cum']:>8.1f}")
    print()

# 割安−割高スプレッド(セクター内の純粋なバリュー効果)
print("=== 割安Q1 − 割高Q3 スプレッド(地銀内バリュー効果・コスト前) ===")
sp=(R['cheap']-R['rich'])
for reg,lab in [(None,'全期間'),('hike','利上げ後'),('hike_IS','└IS'),('hike_OOS','└OOS')]:
    if reg is None: ss=sp
    elif reg=='hike': ss=sp[sp.index>=pd.Timestamp(HIKE)]
    else: ss=sp[[regime(d)==reg for d in sp.index]]
    st=stats(ss); print(f"  {lab:8} N={st['n']:>3} 月平均{ss.mean()*100:+.2f}% 年率{st['ann']:+.1f}% SR{st['sr']:+.2f} 勝率{st['hit']:.0f}%")

# 地銀セクター vs TOPIX(レジーム別・文脈)
topix=topix.set_index('date').reindex(mdates,method='ffill')
topix_ret=topix['close'].pct_change().shift(-1)  # 翌月
bank_ret=R['ew']
print("\n=== 文脈: 地銀EW vs TOPIX (翌月リターン平均・コスト前) ===")
for reg,lab in [('pre','利上げ前'),('hike','利上げ後')]:
    if reg=='pre': mask=R.index<pd.Timestamp(HIKE)
    else: mask=R.index>=pd.Timestamp(HIKE)
    b=bank_ret[mask].mean()*100; t=topix_ret.reindex(R.index)[mask].mean()*100
    print(f"  {lab}: 地銀{b:+.2f}%/月  TOPIX{t:+.2f}%/月  差{b-t:+.2f}%")

# 可視化: 累積(利上げ後)
fig,ax=plt.subplots(figsize=(12,6.75),dpi=100)
for nm,col in [('割安Q1','#2ca02c'),('低PBR×ROE','#1f6feb'),('地銀EW','#888'),('割高Q3','#d62728'),('価格出遅れ','#d29922')]:
    s=series[nm]; s=s[s.index>=pd.Timestamp(HIKE)].dropna().sort_index()
    if len(s)<3: continue
    ax.plot(s.index,((1+s).cumprod()-1)*100,label=nm,lw=2 if nm in('割安Q1','低PBR×ROE') else 1.3,color=col)
ax.axvline(pd.Timestamp(OOS),color='gray',ls='--',lw=1); ax.text(pd.Timestamp(OOS),ax.get_ylim()[1]," OOS→",color='gray',va='top')
ax.axhline(0,color='k',lw=.6); ax.set_title("利上げ局面の地銀バリュー: 戦略別 累積リターン(コスト後・2024-03〜)",fontsize=13)
ax.set_ylabel("累積リターン %"); ax.legend(fontsize=10); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(os.path.join(HERE,"result.png")); print("\nsaved result.png")

# 現在の候補リスト(最新月末・低PBR地銀 + 質)
latest=me[me['date']==me['date'].max()].copy()
latest=latest[(latest['tv20']>=LIQ)&latest['pbr'].notna()]
latest=latest.merge(uni[['code','code4','name_ja']],on='code',how='left')
latest['score']=latest['roe'].rank(ascending=False)-latest['pbr'].rank()  # 低PBR×高ROE
cand=latest.sort_values('pbr').head(15)
cand.to_csv(os.path.join(HERE,"candidates.csv"),index=False)
print("\n=== 現在の割安地銀 候補(PBR昇順・代金>=0.3億) ===")
print(f"{'code':5}{'銘柄':20}{'PBR':>6}{'PER':>7}{'ROE%':>7}{'配当%':>7}{'6M%':>7}{'代金億':>7}")
for _,r in cand.iterrows():
    print(f"{r['code4']:5}{str(r['name_ja'])[:18]:20}{r['pbr']:>6.2f}{(r['c']/r['eps'] if r['eps'] else float('nan')):>7.1f}{(r['roe'] if pd.notna(r['roe']) else float('nan')):>7.1f}{(r['divy'] if pd.notna(r['divy']) else float('nan')):>7.1f}{(r['ret_6m']*100 if pd.notna(r['ret_6m']) else float('nan')):>7.1f}{r['tv20']/1e8:>7.1f}")
print("done")
