"""
月曜イントラ深掘り：非鉄 vs 半導体AI
- 時間帯別波形
- 前場/後場の挙動と相関
- 月曜ON結果別の月曜イントラ挙動（リバーサルか継続か）
- 銘柄別ランキング
- ボラパターン
"""
import sys, os, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import psycopg2
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

PG_CONFIG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}

SEMI = {
    '69200': 'レーザーテック', '80350': '東京エレクトロン', '68570': 'アドバンテスト',
    '61460': 'ディスコ', '40630': '信越化学', '69630': 'ローム',
    '77350': 'SCREEN', '34360': 'SUMCO', '65260': 'ソシオネクスト', '99840': 'SBG',
}
NONFER = {
    '57130': '住友金属鉱山', '57110': '三菱マテリアル', '57060': '三井金属',
    '57140': 'DOWA', '57150': '古河機械金属', '57270': '東邦チタニウム',
}
ALL = {**SEMI, **NONFER}

START = '2025-11-01'
END   = '2026-05-18'

# ── イントラ取得 ─────────────────────
conn = psycopg2.connect(**PG_CONFIG)
placeholders = ','.join(f"'{c}'" for c in ALL)
intra = pd.read_sql(f"""
    SELECT code, ts, open, high, low, close, volume
    FROM stocks_intraday
    WHERE code IN ({placeholders})
      AND ts >= '{START}' AND ts < '{END}'
    ORDER BY code, ts
""", conn)

# 日足（ON gap計算用）
daily = pd.read_sql(f"""
    SELECT code, date, open, close
    FROM stocks_daily
    WHERE code IN ({placeholders})
      AND date >= '2025-10-25' AND date <= '{END}'
    ORDER BY code, date
""", conn)
conn.close()

intra['ts']     = pd.to_datetime(intra['ts'])
intra['date']   = intra['ts'].dt.date
intra['dow']    = intra['ts'].dt.dayofweek
intra['minute'] = intra['ts'].dt.hour * 60 + intra['ts'].dt.minute
intra = intra[intra['ts'].dt.time.between(pd.Timestamp('09:00').time(),
                                            pd.Timestamp('15:30').time())]
intra['sector'] = intra['code'].apply(lambda c: 'semi' if c in SEMI else 'nonfer')

# 月曜だけ抽出
mon = intra[intra['dow'] == 0].copy()
print(f"月曜1分足: {len(mon):,}行 / 営業日数: {mon['date'].nunique()}日")

# ── ON gap 計算 ─────────────────
daily['date'] = pd.to_datetime(daily['date']).dt.tz_localize(None)
daily = daily.sort_values(['code','date'])
daily['prev_close'] = daily.groupby('code')['close'].shift(1)
daily['on_gap']     = (daily['open'] / daily['prev_close'] - 1) * 100
daily['dow']        = daily['date'].dt.dayofweek

mon_on = daily[daily['dow'] == 0][['code','date','on_gap','open','close']].copy()
mon_on['date'] = mon_on['date'].dt.date
mon_on['mon_full'] = (mon_on['close'] / mon_on['open'] - 1) * 100  # 月曜寄→引

# ─── 1. セクター別 時間帯波形 ─────────
def intra_pattern(df):
    df = df.copy()
    day_opens = df.groupby(['code','date'])['open'].transform('first')
    df['ret_from_open'] = (df['close'] / day_opens - 1) * 100
    by_code = df.groupby(['code','minute'])['ret_from_open'].mean().reset_index()
    return by_code.groupby('minute')['ret_from_open'].mean()

mon_semi   = mon[mon['sector']=='semi']
mon_nonfer = mon[mon['sector']=='nonfer']

pattern_semi   = intra_pattern(mon_semi)
pattern_nonfer = intra_pattern(mon_nonfer)

print("\n【月曜イントラ 寄付比累積リターン (セクター平均)】")
key_times = [(9*60,'9:00'),(9*60+15,'9:15'),(9*60+30,'9:30'),(10*60,'10:00'),
             (10*60+30,'10:30'),(11*60,'11:00'),(11*60+30,'11:30'),
             (12*60+30,'12:30'),(13*60,'13:00'),(14*60,'14:00'),
             (15*60,'15:00'),(15*60+30,'15:30')]
print(f"  {'時刻':10} {'半導体':>10} {'非鉄':>10}")
for m, label in key_times:
    s = pattern_semi.get(m, np.nan)
    n = pattern_nonfer.get(m, np.nan)
    print(f"  {label:10} {s:>+9.3f}% {n:>+9.3f}%")

# ─── 2. 前場/後場 ─────────────
def session_rets(df):
    out = []
    for (code, date), g in df.groupby(['code','date']):
        am = g[g['minute'] <= 690]
        pm = g[g['minute'] >= 750]
        if len(am)<5 or len(pm)<5: continue
        am_ret = (am['close'].iloc[-1] / am['open'].iloc[0] - 1) * 100
        pm_ret = (pm['close'].iloc[-1] / pm['open'].iloc[0] - 1) * 100
        out.append({'code':code, 'date':date,
                    'am_ret':am_ret, 'pm_ret':pm_ret})
    return pd.DataFrame(out)

sess_semi   = session_rets(mon_semi)
sess_nonfer = session_rets(mon_nonfer)

print("\n【月曜 前場/後場 (セクター平均)】")
for name, s in [('半導体', sess_semi), ('非鉄', sess_nonfer)]:
    corr = s['am_ret'].corr(s['pm_ret'])
    print(f"  {name:6}: 前場{s['am_ret'].mean():+.3f}% / 後場{s['pm_ret'].mean():+.3f}% / "
          f"AM-PM相関{corr:+.3f}  (n={len(s)})")

# ─── 3. 月曜ON別の月曜イントラ挙動 ───
print("\n【月曜ON結果別 月曜イントラ挙動】")
print("（ON下げの後、寄→引はどう動く？）")
mon_on['sector'] = mon_on['code'].apply(lambda c: 'semi' if c in SEMI else 'nonfer')
for sec_code, sec_name in [('semi','半導体'), ('nonfer','非鉄')]:
    sub = mon_on[mon_on['sector']==sec_code].dropna(subset=['on_gap','mon_full'])
    on_down = sub[sub['on_gap'] < -0.5]
    on_flat = sub[sub['on_gap'].abs() <= 0.5]
    on_up   = sub[sub['on_gap'] > 0.5]
    print(f"\n  {sec_name}:")
    print(f"    ON下げ(<-0.5%) → 月曜寄→引 平均{on_down['mon_full'].mean():+.3f}%  勝率{(on_down['mon_full']>0).mean()*100:.0f}%  n={len(on_down)}")
    print(f"    ONフラット(±0.5%) → 月曜寄→引 平均{on_flat['mon_full'].mean():+.3f}%  勝率{(on_flat['mon_full']>0).mean()*100:.0f}%  n={len(on_flat)}")
    print(f"    ON上げ(>+0.5%) → 月曜寄→引 平均{on_up['mon_full'].mean():+.3f}%  勝率{(on_up['mon_full']>0).mean()*100:.0f}%  n={len(on_up)}")

# ─── 4. 銘柄別 月曜イントラ ─────────
print("\n【銘柄別 月曜イントラ (寄→引) ランキング】")
mon_open_close = mon.groupby(['code','date']).agg(
    day_open=('open','first'), day_close=('close','last')
).reset_index()
mon_open_close['ret'] = (mon_open_close['day_close']/mon_open_close['day_open']-1)*100
rank = mon_open_close.groupby('code').agg(
    avg=('ret','mean'),
    wr=('ret', lambda x: (x>0).mean()*100),
    std=('ret','std'),
    n=('ret','count')
).sort_values('avg', ascending=False)

print(f"  {'銘柄':14} {'セク':6} {'平均':>8} {'勝率':>6} {'σ':>6} {'Sharpe':>8} N")
for code, row in rank.iterrows():
    sec = '半導体' if code in SEMI else '非鉄'
    sh = row['avg']/row['std']*np.sqrt(252/5) if row['std']>0 else 0
    print(f"  {ALL[code]:14}{sec:6} {row['avg']:>+7.2f}% {row['wr']:>4.0f}% {row['std']:>5.2f}% {sh:>+7.2f}  {int(row['n'])}")

# ─── 5. ボラと出来高パターン ─────────
print("\n【月曜の時間帯別出来高 (各日normalize)】")
def vol_pattern(df):
    df = df.copy()
    day_vol = df.groupby(['code','date'])['volume'].transform('sum')
    df['vol_norm'] = df['volume'] / day_vol
    by_code = df.groupby(['code','minute'])['vol_norm'].mean().reset_index()
    return by_code.groupby('minute')['vol_norm'].mean()

vol_semi   = vol_pattern(mon_semi)
vol_nonfer = vol_pattern(mon_nonfer)

# 30分集計
print(f"  {'時間帯':14} {'半導体出来高%':>13} {'非鉄出来高%':>13}")
buckets = [(9*60, 9*60+30, '9:00-9:30'),
           (9*60+30, 10*60+30, '9:30-10:30'),
           (10*60+30, 11*60+30, '10:30-11:30'),
           (12*60+30, 13*60+30, '12:30-13:30'),
           (13*60+30, 14*60+30, '13:30-14:30'),
           (14*60+30, 15*60+30, '14:30-15:30')]
for s_min, e_min, label in buckets:
    s_pct = vol_semi.loc[(vol_semi.index>=s_min) & (vol_semi.index<e_min)].sum() * 100
    n_pct = vol_nonfer.loc[(vol_nonfer.index>=s_min) & (vol_nonfer.index<e_min)].sum() * 100
    print(f"  {label:14} {s_pct:>12.1f}% {n_pct:>12.1f}%")

# 保存
with open(os.path.join(os.path.dirname(__file__), 'results.pkl'), 'wb') as f:
    pickle.dump(dict(
        pattern_semi=pattern_semi, pattern_nonfer=pattern_nonfer,
        sess_semi=sess_semi, sess_nonfer=sess_nonfer,
        mon_on=mon_on, rank=rank,
        vol_semi=vol_semi, vol_nonfer=vol_nonfer,
        ALL=ALL, SEMI=SEMI, NONFER=NONFER,
        START=START, END=END,
    ), f)
print("\n→ results.pkl 保存完了")
