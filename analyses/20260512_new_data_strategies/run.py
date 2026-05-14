"""
新データソース活用 トレーディング戦略 — 4仮説の包括検証
========================================================
分析日: 2026-05-12

【検証する4仮説】
  H1. 空売り残高 × 大商い陰線 (ショートスクイーズ強化版)
      → bank_absorption 強化版になるか
  H2. 信用買い残オーバーハング × 大商い陽線 → SHORT
      → 高値圏での信用利確売りを狙う
  H3. 信用買い残急増週後の Long
      → モメンタムフォロー
  H4. 決算前 5日ドリフト Long (earnings_calendar)
      → インサイダー的事前買いの追随

【期間】
  2024-01-04 〜 2026-05-12
  プライム市場銘柄 / 売買代金10億+

【判定基準】
  N≥100, net>0, Sharpe>1.5, t-stat>2, PF>1.3, 年次安定
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

print("=" * 78)
print("  新データソース活用 戦略 — 4仮説検証")
print("=" * 78)


# ════════════════════════════════════════════════════════════════
# H1: 空売り残高 × 大商い陰線 ショートスクイーズ強化
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("【H1】空売り残高 × 大商い陰線 → 翌寄り Long (5日保有)")
print("=" * 78)

# jquants_short_sale_report で銘柄別空売り残ratio を取得
# 各日付の最新報告で各銘柄の空売り残率
short_sql = """
WITH latest_short AS (
    SELECT code, calc_date,
           SUM(shrt_pos_to_so) AS total_short_ratio
    FROM jquants_short_sale_report
    WHERE calc_date >= '2024-01-01'
    GROUP BY code, calc_date
)
SELECT * FROM latest_short
"""
short_df = q(short_sql)
short_df['calc_date'] = pd.to_datetime(short_df['calc_date'])
short_df['total_short_ratio'] = pd.to_numeric(short_df['total_short_ratio'])
print(f"  空売り残データ: {len(short_df):,}行 × {short_df['code'].nunique()}銘柄")
print(f"  空売り残ratio分布:")
print(f"    25%分位: {short_df['total_short_ratio'].quantile(0.25)*100:.2f}%")
print(f"    50%分位: {short_df['total_short_ratio'].quantile(0.50)*100:.2f}%")
print(f"    75%分位: {short_df['total_short_ratio'].quantile(0.75)*100:.2f}%")
print(f"    90%分位: {short_df['total_short_ratio'].quantile(0.90)*100:.2f}%")

# 日次パネルを構築 (各日付の銘柄の空売り残ratio = forward fill)
short_pivot = short_df.set_index(['calc_date','code'])['total_short_ratio'].unstack()
short_daily = short_pivot.reindex(pd.date_range(short_pivot.index.min(), short_pivot.index.max())).ffill(limit=10)

# 出来高吸収シグナル + 空売り残高ratio
sig_sql = """
WITH base AS (
    SELECT d.code, s.name_ja, s.sector33_nm AS sector,
           d.date, d.adj_open, d.adj_close, d.adj_high, d.adj_low,
           d.adj_volume, d.turnover_value
    FROM stocks_daily d
    JOIN symbol_master s ON s.code5 = d.code
    WHERE s.market = '0111' AND d.date BETWEEN '2024-01-04' AND '2026-05-12'
      AND d.adj_close > 0 AND d.adj_open > 0
      AND d.turnover_value >= 1000000000
),
ind AS (
    SELECT *,
        AVG(adj_volume) OVER (PARTITION BY code ORDER BY date
                              ROWS BETWEEN 21 PRECEDING AND 2 PRECEDING) AS vol_ma20,
        LEAD(adj_open, 1)   OVER (PARTITION BY code ORDER BY date) AS next_open,
        LEAD(adj_close, 5)  OVER (PARTITION BY code ORDER BY date) AS close_5d
    FROM base
)
SELECT code, name_ja, sector, date,
       adj_volume, vol_ma20, adj_volume/NULLIF(vol_ma20,0) AS vol_ratio,
       (adj_close/adj_open - 1)*100 AS day_ret,
       next_open, close_5d,
       (close_5d/next_open - 1)*100 AS fwd_5d
FROM ind
WHERE vol_ma20 IS NOT NULL AND next_open IS NOT NULL AND close_5d IS NOT NULL
  AND adj_volume/vol_ma20 >= 1.5
  AND (adj_close/adj_open - 1) < 0      -- 陰線
"""
sig_df = q(sig_sql)
for c in ['vol_ratio','day_ret','fwd_5d','next_open','close_5d']:
    sig_df[c] = pd.to_numeric(sig_df[c], errors='coerce')
sig_df['date'] = pd.to_datetime(sig_df['date'])
print(f"\n  ベースシグナル (vol≥1.5x × 陰線): {len(sig_df):,}件")

# 各シグナル日の空売り残ratio を join
def get_short_ratio(row):
    d = row['date']
    code = row['code']
    if d not in short_daily.index or code not in short_daily.columns:
        return np.nan
    return short_daily.loc[d, code]

sig_df['short_ratio'] = sig_df.apply(get_short_ratio, axis=1)
sig_with_short = sig_df.dropna(subset=['short_ratio']).copy()
print(f"  うち空売り残ratio取得可: {len(sig_with_short):,}件 ({len(sig_with_short)/len(sig_df)*100:.1f}%)")

# 空売り残ratio別の翌5日リターン
print(f"\n  ─ 空売り残ratio別 翌5日リターン ─")
bins = [(0, 0.005, '<0.5%'), (0.005, 0.015, '0.5〜1.5%'),
        (0.015, 0.03, '1.5〜3%'), (0.03, 0.05, '3〜5%'),
        (0.05, 1.0, '5%+')]
print(f"  {'空売り残ratio':<14}  {'N':>5}  {'gross5d':>9}  {'net5d':>9}  {'勝率':>6}  {'Sharpe':>7}")
print("  " + "-" * 60)
h1_results = []
for lo, hi, label in bins:
    sub = sig_with_short[(sig_with_short['short_ratio'] >= lo) & (sig_with_short['short_ratio'] < hi)]
    if len(sub) < 30:
        continue
    r = sub['fwd_5d']
    net = r - COST_PCT_RT
    sharpe = net.mean() / r.std() * np.sqrt(252/5) if r.std() > 0 else 0
    flag = " ★" if sharpe > 1.5 else ""
    print(f"  {label:<14}  {len(sub):>5}  {r.mean():>+8.3f}%  {net.mean():>+8.3f}%  "
          f"{(r>0).mean()*100:>5.1f}%  {sharpe:>+6.2f}{flag}")
    h1_results.append({'short_ratio': label, 'N': len(sub), 'net': net.mean(),
                       'wr': (r>0).mean()*100, 'sharpe': sharpe})


# ════════════════════════════════════════════════════════════════
# H2: 信用買い残オーバーハング × 大商い陽線 → SHORT
# ════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 78)
print("【H2】信用買い残オーバーハング × 大商い陽線 → 翌寄り Short (5日保有)")
print("=" * 78)

# 信用残データ
margin_sql = """
SELECT code, date, long_vol, shrt_vol
FROM jquants_margin_interest
WHERE date >= '2024-01-01'
"""
mg = q(margin_sql)
mg['date'] = pd.to_datetime(mg['date'])
mg['long_vol'] = pd.to_numeric(mg['long_vol'])
mg['shrt_vol'] = pd.to_numeric(mg['shrt_vol'])
mg['long_short_ratio'] = mg['long_vol'] / (mg['shrt_vol'] + 1)  # +1で0除算回避
print(f"  信用残データ: {len(mg):,}行 × {mg['code'].nunique()}銘柄")

# 日次パネル化
mg_pivot = mg.set_index(['date','code'])['long_short_ratio'].unstack().ffill(limit=10)

# 大商い陽線シグナル
sig_sql_h2 = """
WITH base AS (
    SELECT d.code, s.name_ja, d.date,
           d.adj_open, d.adj_close, d.adj_volume, d.turnover_value
    FROM stocks_daily d
    JOIN symbol_master s ON s.code5 = d.code
    WHERE s.market = '0111' AND d.date BETWEEN '2024-01-04' AND '2026-05-12'
      AND d.adj_close > 0 AND d.adj_open > 0 AND d.turnover_value >= 1000000000
),
ind AS (
    SELECT *,
        AVG(adj_volume) OVER (PARTITION BY code ORDER BY date
                              ROWS BETWEEN 21 PRECEDING AND 2 PRECEDING) AS vol_ma20,
        LEAD(adj_open, 1)   OVER (PARTITION BY code ORDER BY date) AS next_open,
        LEAD(adj_close, 5)  OVER (PARTITION BY code ORDER BY date) AS close_5d
    FROM base
)
SELECT code, name_ja, date,
       adj_volume/NULLIF(vol_ma20,0) AS vol_ratio,
       (adj_close/adj_open - 1)*100 AS day_ret,
       (close_5d/next_open - 1)*100 AS fwd_5d
FROM ind
WHERE vol_ma20 IS NOT NULL AND next_open IS NOT NULL AND close_5d IS NOT NULL
  AND adj_volume/vol_ma20 >= 1.5
  AND (adj_close/adj_open - 1) > 1.0     -- 大陽線 (+1%超)
"""
sig_h2 = q(sig_sql_h2)
for c in ['vol_ratio','day_ret','fwd_5d']:
    sig_h2[c] = pd.to_numeric(sig_h2[c], errors='coerce')
sig_h2['date'] = pd.to_datetime(sig_h2['date'])

def get_margin_ratio(row):
    d = row['date']
    code = row['code']
    if d not in mg_pivot.index or code not in mg_pivot.columns:
        return np.nan
    return mg_pivot.loc[d, code]

sig_h2['margin_ratio'] = sig_h2.apply(get_margin_ratio, axis=1)
sig_h2 = sig_h2.dropna(subset=['margin_ratio'])
print(f"\n  ベースシグナル(vol≥1.5x×大陽線+1%超) × 信用残取得可: {len(sig_h2):,}件")

print(f"\n  ─ 信用買い残/売り残比別 翌5日リターン (SHORT想定 = -fwd_5d) ─")
mg_bins = [(0, 1, '<1 (売り残>=買い残)'), (1, 5, '1〜5'), (5, 20, '5〜20'),
           (20, 100, '20〜100'), (100, 1e9, '100+ (買い残過多)')]
print(f"  {'long/short':<22}  {'N':>5}  {'gross5d':>9}  {'SHORT net':>10}  {'勝率':>6}  {'Sharpe':>7}")
print("  " + "-" * 70)
h2_results = []
for lo, hi, label in mg_bins:
    sub = sig_h2[(sig_h2['margin_ratio'] >= lo) & (sig_h2['margin_ratio'] < hi)]
    if len(sub) < 30:
        continue
    r = -sub['fwd_5d']    # SHORTなので符号反転
    net = r - COST_PCT_RT
    sharpe = net.mean() / r.std() * np.sqrt(252/5) if r.std() > 0 else 0
    flag = " ★" if sharpe > 1.5 else ""
    print(f"  {label:<22}  {len(sub):>5}  {sub['fwd_5d'].mean():>+8.3f}%  "
          f"{net.mean():>+9.3f}%  {(r>0).mean()*100:>5.1f}%  {sharpe:>+6.2f}{flag}")
    h2_results.append({'margin_ratio': label, 'N': len(sub), 'net': net.mean(),
                       'wr': (r>0).mean()*100, 'sharpe': sharpe})


# ════════════════════════════════════════════════════════════════
# H3: 信用買い残急増週後の Long (週次変化率)
# ════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 78)
print("【H3】信用買い残 週次変化率別 翌週リターン")
print("=" * 78)

# 信用残の週次変化率
mg_weekly = mg.copy()
mg_weekly = mg_weekly.sort_values(['code','date'])
mg_weekly['long_vol_prev'] = mg_weekly.groupby('code')['long_vol'].shift(1)
mg_weekly['long_chg_pct'] = (mg_weekly['long_vol'] / mg_weekly['long_vol_prev'] - 1) * 100

# 各日付の信用残報告 (週次) を価格データに join
# 信用残は週1更新なので、各週次データに対し次の5営業日リターンを取得
prc_sql = """
SELECT d.code, d.date, d.adj_close,
       LEAD(d.adj_close, 5) OVER (PARTITION BY d.code ORDER BY d.date) AS close_5d_after,
       LEAD(d.adj_open, 1) OVER (PARTITION BY d.code ORDER BY d.date) AS next_open,
       d.turnover_value
FROM stocks_daily d
JOIN symbol_master s ON s.code5 = d.code
WHERE s.market = '0111' AND d.date >= '2024-01-01'
"""
prc = q(prc_sql)
prc['date'] = pd.to_datetime(prc['date'])
for c in ['adj_close','close_5d_after','next_open','turnover_value']:
    prc[c] = pd.to_numeric(prc[c], errors='coerce')

merged = mg_weekly.merge(prc, on=['code','date'], how='left')
merged = merged.dropna(subset=['long_chg_pct','close_5d_after','next_open','turnover_value'])
merged = merged[merged['turnover_value'] >= 500_000_000]
merged['fwd_5d'] = (merged['close_5d_after'] / merged['next_open'] - 1) * 100

print(f"\n  信用残×価格 joined: {len(merged):,}件")

# 週次変化率別
chg_bins = [(-99, -20, '<-20% (急減)'), (-20, -10, '-20〜-10%'), (-10, -5, '-10〜-5%'),
            (-5, 5, '-5〜+5% (横ばい)'), (5, 10, '+5〜+10%'),
            (10, 20, '+10〜+20%'), (20, 50, '+20〜+50%'), (50, 999, '+50%+(急増)')]

print(f"\n  ─ 信用買い残 週次変化率別 翌5日リターン (Long想定) ─")
print(f"  {'変化率':<22}  {'N':>6}  {'gross5d':>9}  {'net5d':>9}  {'勝率':>6}  {'Sharpe':>7}")
print("  " + "-" * 65)
h3_results = []
for lo, hi, label in chg_bins:
    sub = merged[(merged['long_chg_pct'] >= lo) & (merged['long_chg_pct'] < hi)]
    if len(sub) < 100:
        continue
    r = sub['fwd_5d']
    net = r - COST_PCT_RT
    sharpe = net.mean() / r.std() * np.sqrt(252/5) if r.std() > 0 else 0
    flag = " ★" if sharpe > 1.5 else ""
    print(f"  {label:<22}  {len(sub):>6}  {r.mean():>+8.3f}%  {net.mean():>+8.3f}%  "
          f"{(r>0).mean()*100:>5.1f}%  {sharpe:>+6.2f}{flag}")
    h3_results.append({'chg': label, 'N': len(sub), 'net': net.mean(),
                       'wr': (r>0).mean()*100, 'sharpe': sharpe})


# ════════════════════════════════════════════════════════════════
# H4: 決算前 5日ドリフト
# ════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 78)
print("【H4】決算前 5日のドリフト — 発表5日前にエントリー、発表前日にExit")
print("=" * 78)

# fin_summary の過去発表日を「予定日」として代用 (earnings_calendar は未来分のみ)
# 戦略: 発表日(T)の5営業日前(T-5)に Long、T-1 引けに Exit
event_sql = """
WITH events AS (
    SELECT DISTINCT ON (code, disc_date) code, disc_date, doc_type
    FROM fin_summary
    WHERE disc_date >= '2024-01-04'
    ORDER BY code, disc_date
)
SELECT * FROM events
"""
ev = q(event_sql)
ev['disc_date'] = pd.to_datetime(ev['disc_date'])

# 銘柄ごとの日付列
prc2 = prc.copy()
prc2 = prc2.sort_values(['code','date'])
code_dates = {c: g['date'].tolist() for c, g in prc2.groupby('code')}
prc2_idx = prc2.set_index(['code','date'])

results_h4 = []
for _, row in ev.iterrows():
    code = row['code']
    d_evt = row['disc_date']
    if code not in code_dates:
        continue
    dates = code_dates[code]
    try:
        idx = dates.index(d_evt)
    except ValueError:
        continue
    if idx < 5:
        continue
    d_entry = dates[idx - 5]
    d_exit  = dates[idx - 1]
    try:
        entry = prc2_idx.loc[(code, d_entry), 'adj_close']
        exit_ = prc2_idx.loc[(code, d_exit), 'adj_close']
        tv    = prc2_idx.loc[(code, d_entry), 'turnover_value']
    except KeyError:
        continue
    if pd.isna(entry) or pd.isna(exit_) or pd.isna(tv):
        continue
    if tv < 500_000_000:
        continue
    ret = (exit_ / entry - 1) * 100
    results_h4.append({'code': code, 'entry_date': d_entry, 'exit_date': d_exit,
                       'doc_type': row['doc_type'], 'pre_5d_ret': ret})

h4_df = pd.DataFrame(results_h4)
print(f"\n  決算イベント数 (5日前にエントリー可): {len(h4_df):,}")

r = h4_df['pre_5d_ret']
net = r - COST_PCT_RT
sharpe = net.mean() / r.std() * np.sqrt(252/5) if r.std() > 0 else 0
print(f"\n  全体:")
print(f"    gross 5日: {r.mean():>+.3f}%, net: {net.mean():>+.3f}%")
print(f"    勝率: {(r>0).mean()*100:.1f}%, Sharpe: {sharpe:+.2f}")

# doc_type別
print(f"\n  ─ doc_type別 ─")
print(f"  {'doc_type':<48}  {'N':>4}  {'net5d':>8}  {'勝率':>6}  {'Sharpe':>7}")
print("  " + "-" * 80)
h4_results = []
for dt, g in h4_df.groupby('doc_type'):
    if len(g) < 50:
        continue
    r = g['pre_5d_ret']
    net = r - COST_PCT_RT
    sharpe = net.mean() / r.std() * np.sqrt(252/5) if r.std() > 0 else 0
    flag = " ★" if sharpe > 1.5 else ""
    print(f"  {dt:<48}  {len(g):>4}  {net.mean():>+7.3f}%  "
          f"{(r>0).mean()*100:>5.1f}%  {sharpe:>+6.2f}{flag}")
    h4_results.append({'doc_type': dt, 'N': len(g), 'net': net.mean(),
                       'wr': (r>0).mean()*100, 'sharpe': sharpe})


# ════════════════════════════════════════════════════════════════
# 総括: 最も有望な仮説
# ════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 78)
print("【総括】4仮説の有望性ランキング")
print("=" * 78)

all_results = []
for r in h1_results: all_results.append({'H': 'H1空売り残高×吸収', **r})
for r in h2_results: all_results.append({'H': 'H2信用残×大商い→Short', **r})
for r in h3_results: all_results.append({'H': 'H3信用買残変化率', **r})
for r in h4_results: all_results.append({'H': 'H4決算前ドリフト', **r})

best_df = pd.DataFrame(all_results).sort_values('sharpe', ascending=False)
print(f"\n  TOP10 結果 (Sharpe降順):")
print(f"  {'仮説':<24}  {'バケット':<24}  {'N':>5}  {'net':>8}  {'Sharpe':>7}")
print("  " + "-" * 80)
for _, r in best_df.head(15).iterrows():
    bucket = r.get('short_ratio') or r.get('margin_ratio') or r.get('chg') or r.get('doc_type', '')
    print(f"  {r['H']:<24}  {str(bucket)[:24]:<24}  {int(r['N']):>5}  "
          f"{r['net']:>+7.3f}%  {r['sharpe']:>+6.2f}")

# CSV出力
out_dir = '/Users/Yusuke/claude-code/japan-stocks/analyses/20260512_new_data_strategies'
best_df.to_csv(f'{out_dir}/all_results.csv', index=False)
print(f"\n  ✅ 完了。次ステップ: ベスト仮説の詳細バックテスト・実SL/TP・Walk-forward")
