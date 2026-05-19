"""
マルチタイムフレーム (MTF) トレンド一致戦略 包括検証
=====================================================
分析日: 2026-05-15

【コンセプト】
  日足トレンド ∩ 前場 (分足相当) トレンドが一致したときだけエントリー

【検証構造】
  日足トレンド: close vs MA25/MA75, 20日リターン, スロープ
  前場トレンド: 前場(9:00-11:30)寄→引リターン, 前場高値引け率

  エントリー: 後場 12:30 寄り
  決済:       後場 15:30 引け (3時間保有)

【対象】 TOPIX Core30+Large70+Mid400 (493銘柄)
"""
import psycopg2
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

PG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}
COST_PCT_RT = 0.10

def q(sql):
    conn = psycopg2.connect(**PG)
    df = pd.read_sql(sql, conn)
    conn.close()
    return df

print("=" * 80)
print("  マルチタイムフレーム (MTF) トレンド一致戦略 包括検証")
print("=" * 80)
print("\n  日足データ取得中...")

# 日足 + 移動平均
sql_daily = """
WITH base AS (
    SELECT d.code, s.name_ja, s.sector33_nm AS sector, s.scale_cat,
           d.date, d.adj_open, d.adj_close, d.turnover_value
    FROM stocks_daily d
    JOIN symbol_master s ON s.code5 = d.code
    WHERE s.market='0111'
      AND s.scale_cat IN ('TOPIX Core30','TOPIX Large70','TOPIX Mid400')
      AND d.date BETWEEN '2023-10-01' AND '2026-05-14'
      AND d.adj_close > 0
),
ind1 AS (
    SELECT *,
        AVG(adj_close)  OVER w25 AS ma25,
        AVG(adj_close)  OVER w75 AS ma75,
        LAG(adj_close, 20) OVER (PARTITION BY code ORDER BY date) AS close_20d_ago
    FROM base
    WINDOW
        w25 AS (PARTITION BY code ORDER BY date ROWS BETWEEN 25 PRECEDING AND 1 PRECEDING),
        w75 AS (PARTITION BY code ORDER BY date ROWS BETWEEN 75 PRECEDING AND 1 PRECEDING)
),
ind2 AS (
    SELECT *,
        LAG(ma25, 5) OVER (PARTITION BY code ORDER BY date) AS ma25_5d_ago
    FROM ind1
)
SELECT * FROM ind2 WHERE date >= '2024-01-04' AND ma25 IS NOT NULL
"""
daily = q(sql_daily)
for c in ['adj_open','adj_close','turnover_value','ma25','ma75','close_20d_ago','ma25_5d_ago']:
    daily[c] = pd.to_numeric(daily[c], errors='coerce')
daily['date'] = pd.to_datetime(daily['date'])
daily = daily[daily['turnover_value'] >= 500_000_000]
print(f"  日足ロード: {len(daily):,}行")

print("\n  前場・後場OHLC 計算中 (stocks_intraday から集計)...")
# JST naive 時刻を使って前場/後場を判定
sql_intraday = """
WITH topix500 AS (
    SELECT code5 FROM symbol_master
    WHERE market='0111' AND scale_cat IN ('TOPIX Core30','TOPIX Large70','TOPIX Mid400')
),
filtered AS (
    SELECT i.code, i.ts::date AS date,
           CASE
             WHEN i.ts::time >= '09:00:00' AND i.ts::time <= '11:30:00' THEN 'morning'
             WHEN i.ts::time >= '12:30:00' AND i.ts::time <= '15:30:00' THEN 'afternoon'
             ELSE NULL
           END AS session,
           i.ts, i.open, i.high, i.low, i.close, i.volume
    FROM stocks_intraday i
    WHERE i.code IN (SELECT code5 FROM topix500)
      AND i.ts >= '2024-01-04' AND i.ts < '2026-05-15'
)
SELECT code, date, session,
       (array_agg(open  ORDER BY ts ASC))[1]                                AS sess_open,
       MAX(high)                                                            AS sess_high,
       MIN(low)                                                             AS sess_low,
       (array_agg(close ORDER BY ts DESC))[1]                               AS sess_close,
       SUM(volume)                                                          AS sess_volume,
       SUM(close * volume) / NULLIF(SUM(volume), 0)                         AS sess_vwap
FROM filtered
WHERE session IS NOT NULL
GROUP BY code, date, session
"""
intra = q(sql_intraday)
for c in ['sess_open','sess_high','sess_low','sess_close','sess_volume','sess_vwap']:
    intra[c] = pd.to_numeric(intra[c], errors='coerce')
intra['date'] = pd.to_datetime(intra['date'])
print(f"  前場・後場集計: {len(intra):,}行")

# pivot して前場/後場を横並びに
morn = intra[intra['session']=='morning'].set_index(['code','date'])[
    ['sess_open','sess_high','sess_low','sess_close','sess_vwap']].add_prefix('m_')
aft = intra[intra['session']=='afternoon'].set_index(['code','date'])[
    ['sess_open','sess_close','sess_high','sess_low']].add_prefix('a_')

# 日足とjoin
daily = daily.set_index(['code','date'])
df = daily.join(morn, how='inner').join(aft, how='inner').reset_index()
print(f"  日足×前場×後場 join: {len(df):,}行")

# 派生指標
df['dist_ma25']     = (df['adj_close'] / df['ma25'] - 1) * 100
df['ma25_slope_5d'] = (df['ma25'] / df['ma25_5d_ago'] - 1) * 100
df['ret_20d']       = (df['adj_close'] / df['close_20d_ago'] - 1) * 100
df['morning_ret']   = (df['m_sess_close'] / df['m_sess_open'] - 1) * 100
df['morning_strong'] = ((df['m_sess_close'] - df['m_sess_low']) /
                        (df['m_sess_high'] - df['m_sess_low']).clip(lower=0.01) > 0.85).astype(int)
df['morning_above_vwap'] = (df['m_sess_close'] > df['m_sess_vwap']).astype(int)
df['afternoon_ret'] = (df['a_sess_close'] / df['a_sess_open'] - 1) * 100

df = df.dropna(subset=['afternoon_ret','dist_ma25','morning_ret','ma75','ma25_slope_5d','ret_20d'])
print(f"  有効サンプル: {len(df):,}行 × {df['code'].nunique()}銘柄")


# ════════════════════════════════════════════════════════════════
# Step 1. ベースライン: 前場 vs 後場
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("【1】ベースライン: 前場リターン vs 後場リターン")
print("=" * 80)

print(f"\n  全体 N={len(df):,}")
print(f"  後場平均: {df['afternoon_ret'].mean():+.4f}%, 勝率: {(df['afternoon_ret']>0).mean()*100:.1f}%")

print(f"\n  ─ 前場リターン別 後場の動き (相関確認) ─")
print(f"  {'前場帯':<24}  {'N':>6}  {'後場平均':>9}  {'後場勝率':>9}  {'Sharpe':>7}")
buckets = [
    ('前場大陽線 (>+1.5%)',     df['morning_ret'] > 1.5),
    ('前場小陽線 (+0.3〜+1.5%)', (df['morning_ret']>0.3)&(df['morning_ret']<=1.5)),
    ('前場小動き (-0.3〜+0.3%)', df['morning_ret'].abs()<0.3),
    ('前場小陰線 (-1.5〜-0.3%)', (df['morning_ret']<-0.3)&(df['morning_ret']>=-1.5)),
    ('前場大陰線 (<-1.5%)',     df['morning_ret'] < -1.5),
]
for label, mask in buckets:
    sub = df[mask]
    if len(sub) < 50: continue
    r = sub['afternoon_ret']
    sharpe = r.mean()/r.std()*np.sqrt(252) if r.std()>0 else 0
    print(f"  {label:<24}  {len(sub):>6}  {r.mean():>+8.4f}%  "
          f"{(r>0).mean()*100:>8.1f}%  {sharpe:>+6.2f}")


# ════════════════════════════════════════════════════════════════
# Step 2. MTF一致テスト
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("【2】MTF一致テスト")
print("=" * 80)

daily_signals = {
    'D1_above_MA25':    df['adj_close'] > df['ma25'],
    'D1_below_MA25':    df['adj_close'] < df['ma25'],
    'D2_MA25slope>0.5%':df['ma25_slope_5d'] > 0.5,
    'D2_MA25slope<-0.5%':df['ma25_slope_5d'] < -0.5,
    'D3_20d_up>5%':     df['ret_20d'] > 5,
    'D3_20d_dn<-5%':    df['ret_20d'] < -5,
    'D4_above_MA75':    df['adj_close'] > df['ma75'],
    'D4_below_MA75':    df['adj_close'] < df['ma75'],
    'D5_PMcfg_up':      (df['adj_close'] > df['ma25']) & (df['ma25'] > df['ma75']),
    'D5_PMcfg_dn':      (df['adj_close'] < df['ma25']) & (df['ma25'] < df['ma75']),
}

morning_signals = {
    'I1_morn>+0.3%':    df['morning_ret'] > 0.3,
    'I1_morn<-0.3%':    df['morning_ret'] < -0.3,
    'I2_morn>+1.0%':    df['morning_ret'] > 1.0,
    'I2_morn<-1.0%':    df['morning_ret'] < -1.0,
    'I3_morn_strong_high':df['morning_strong'] == 1,
    'I4_morn>VWAP':     df['morning_above_vwap'] == 1,
    'I4_morn<VWAP':     df['morning_above_vwap'] == 0,
}

# LONG一致
print(f"\n  ─ LONG: 日足UP × 前場UP ─")
print(f"  {'日足':<22}  {'前場':<22}  {'N':>5}  {'net':>8}  {'勝率':>6}  {'Sharpe':>7}")
print("  " + "-" * 85)
results = []
long_combos = [
    ('D1_above_MA25','I1_morn>+0.3%'),
    ('D1_above_MA25','I2_morn>+1.0%'),
    ('D1_above_MA25','I3_morn_strong_high'),
    ('D1_above_MA25','I4_morn>VWAP'),
    ('D2_MA25slope>0.5%','I1_morn>+0.3%'),
    ('D2_MA25slope>0.5%','I2_morn>+1.0%'),
    ('D3_20d_up>5%','I1_morn>+0.3%'),
    ('D3_20d_up>5%','I2_morn>+1.0%'),
    ('D4_above_MA75','I1_morn>+0.3%'),
    ('D4_above_MA75','I2_morn>+1.0%'),
    ('D5_PMcfg_up','I1_morn>+0.3%'),
    ('D5_PMcfg_up','I2_morn>+1.0%'),
    ('D5_PMcfg_up','I3_morn_strong_high'),
    ('D5_PMcfg_up','I4_morn>VWAP'),
]
for d, m in long_combos:
    mask = daily_signals[d] & morning_signals[m]
    sub = df[mask]
    if len(sub) < 50: continue
    r = sub['afternoon_ret']
    net = r - COST_PCT_RT
    sharpe = net.mean()/r.std()*np.sqrt(252) if r.std()>0 else 0
    flag = " ★★" if sharpe>1.5 else (" ★" if sharpe>1.0 else "")
    print(f"  {d:<22}  {m:<22}  {len(sub):>5}  {net.mean():>+7.4f}%  "
          f"{(net>0).mean()*100:>5.1f}%  {sharpe:>+6.2f}{flag}")
    results.append({'side':'LONG','daily':d,'morning':m,'N':len(sub),
                    'net':net.mean(),'wr':(net>0).mean()*100,'sharpe':sharpe})

# SHORT一致
print(f"\n  ─ SHORT: 日足DOWN × 前場DOWN ─")
print(f"  {'日足':<22}  {'前場':<22}  {'N':>5}  {'net':>8}  {'勝率':>6}  {'Sharpe':>7}")
print("  " + "-" * 85)
short_combos = [
    ('D1_below_MA25','I1_morn<-0.3%'),
    ('D1_below_MA25','I2_morn<-1.0%'),
    ('D1_below_MA25','I4_morn<VWAP'),
    ('D2_MA25slope<-0.5%','I1_morn<-0.3%'),
    ('D2_MA25slope<-0.5%','I2_morn<-1.0%'),
    ('D3_20d_dn<-5%','I1_morn<-0.3%'),
    ('D3_20d_dn<-5%','I2_morn<-1.0%'),
    ('D4_below_MA75','I1_morn<-0.3%'),
    ('D4_below_MA75','I2_morn<-1.0%'),
    ('D5_PMcfg_dn','I1_morn<-0.3%'),
    ('D5_PMcfg_dn','I2_morn<-1.0%'),
    ('D5_PMcfg_dn','I4_morn<VWAP'),
]
for d, m in short_combos:
    mask = daily_signals[d] & morning_signals[m]
    sub = df[mask]
    if len(sub) < 50: continue
    r_short = -sub['afternoon_ret']
    net = r_short - COST_PCT_RT
    sharpe = net.mean()/r_short.std()*np.sqrt(252) if r_short.std()>0 else 0
    flag = " ★★" if sharpe>1.5 else (" ★" if sharpe>1.0 else "")
    print(f"  {d:<22}  {m:<22}  {len(sub):>5}  {net.mean():>+7.4f}%  "
          f"{(net>0).mean()*100:>5.1f}%  {sharpe:>+6.2f}{flag}")
    results.append({'side':'SHORT','daily':d,'morning':m,'N':len(sub),
                    'net':net.mean(),'wr':(net>0).mean()*100,'sharpe':sharpe})


# ════════════════════════════════════════════════════════════════
# Step 3. 不一致 (counter signal) 検証
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("【3】MTF不一致 (counter) — 逆張り想定")
print("=" * 80)
print(f"\n  {'日足':<20}  {'前場':<20}  {'想定':<20}  {'N':>5}  {'net':>8}  {'勝率':>6}  {'Sharpe':>7}")
print("  " + "-" * 100)

counter = [
    ('D1_above_MA25','I2_morn<-1.0%','LONG (上昇トレンド押し目)'),
    ('D5_PMcfg_up','I2_morn<-1.0%','LONG (PMcfg押し目)'),
    ('D5_PMcfg_up','I1_morn<-0.3%','LONG (PMcfg小幅押し目)'),
    ('D1_below_MA25','I2_morn>+1.0%','SHORT (下降トレンド戻し売り)'),
    ('D5_PMcfg_dn','I2_morn>+1.0%','SHORT (PMcfg戻し売り)'),
]
for d, m, hyp in counter:
    mask = daily_signals[d] & morning_signals[m]
    sub = df[mask]
    if len(sub) < 50: continue
    r = sub['afternoon_ret'] if 'LONG' in hyp else -sub['afternoon_ret']
    net = r - COST_PCT_RT
    sharpe = net.mean()/r.std()*np.sqrt(252) if r.std()>0 else 0
    flag = " ★★" if sharpe>1.5 else (" ★" if sharpe>1.0 else "")
    print(f"  {d:<20}  {m:<20}  {hyp:<20}  {len(sub):>5}  {net.mean():>+7.4f}%  "
          f"{(net>0).mean()*100:>5.1f}%  {sharpe:>+6.2f}{flag}")
    results.append({'side':'COUNTER','daily':d,'morning':m,'N':len(sub),
                    'net':net.mean(),'wr':(net>0).mean()*100,'sharpe':sharpe})


# ════════════════════════════════════════════════════════════════
# Step 4. TOP15ランキング
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 95)
print("【4】TOP15 (Sharpe順)")
print("=" * 95)
res_df = pd.DataFrame(results)
print(f"\n  {'side':<8}  {'日足':<22}  {'前場':<22}  {'N':>5}  {'net':>8}  {'勝率':>6}  {'Sharpe':>7}")
print("  " + "-" * 95)
for _, r in res_df.sort_values('sharpe', ascending=False).head(15).iterrows():
    flag = " ★★" if r['sharpe']>1.5 else (" ★" if r['sharpe']>1.0 else "")
    print(f"  {r['side']:<8}  {r['daily']:<22}  {r['morning']:<22}  {int(r['N']):>5}  "
          f"{r['net']:>+7.4f}%  {r['wr']:>5.1f}%  {r['sharpe']:>+6.2f}{flag}")


# ════════════════════════════════════════════════════════════════
# Step 5. ベスト設定の詳細
# ════════════════════════════════════════════════════════════════
best = res_df.sort_values('sharpe', ascending=False).iloc[0]
print("\n" + "=" * 80)
print(f"【5】ベスト: {best['side']} / {best['daily']} ∩ {best['morning']}")
print(f"     N={int(best['N'])}, net={best['net']:+.4f}%, Sharpe={best['sharpe']:+.2f}")
print("=" * 80)

mask_b = daily_signals[best['daily']] & morning_signals[best['morning']]
sub_b = df[mask_b].copy()
sub_b['year'] = sub_b['date'].dt.year
# side判定: SHORTカウンターか LONGカウンターかを判別
short_counter_combos = [('D1_below_MA25','I2_morn>+1.0%'),
                        ('D5_PMcfg_dn','I2_morn>+1.0%')]
is_short = (best['side']=='SHORT') or \
           (best['side']=='COUNTER' and (best['daily'],best['morning']) in short_counter_combos)
if is_short:
    sub_b['ret'] = -sub_b['afternoon_ret']
else:
    sub_b['ret'] = sub_b['afternoon_ret']
sub_b['net'] = sub_b['ret'] - COST_PCT_RT

print(f"\n  ─ 年次安定性 ─")
print(f"  {'年':>4}  {'N':>5}  {'net':>8}  {'勝率':>6}  {'Sharpe':>7}")
for year, g in sub_b.groupby('year'):
    sharpe = g['net'].mean()/g['ret'].std()*np.sqrt(252) if g['ret'].std()>0 else 0
    print(f"  {year}  {len(g):>5}  {g['net'].mean():>+7.4f}%  "
          f"{(g['net']>0).mean()*100:>5.1f}%  {sharpe:>+6.2f}")

# Walk-forward
train = sub_b[sub_b['year']==2024]
test  = sub_b[sub_b['year']>=2025]
print(f"\n  ─ Walk-forward ─")
for label, g in [("Train 2024", train), ("Test 2025-26", test)]:
    if len(g)==0: continue
    sharpe = g['net'].mean()/g['ret'].std()*np.sqrt(252) if g['ret'].std()>0 else 0
    print(f"  {label:<14}: N={len(g):>4} net={g['net'].mean():>+.4f}% Sharpe={sharpe:+.2f}")


# CSV出力
out_dir = '/Users/Yusuke/claude-code/japan-stocks/analyses/20260515_mtf_trend_alignment'
res_df.to_csv(f'{out_dir}/all_results.csv', index=False)
print(f"\n  ✅ 完了")
