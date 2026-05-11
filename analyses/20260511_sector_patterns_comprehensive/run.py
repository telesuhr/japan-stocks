"""
セクター別 包括パターン分析（日次・ON・イントラ）
==================================================
分析日: 2026-05-11
対象:   プライム1,574銘柄 × 33セクター × 2024-01-04〜2026-05-08

【7軸の切り口】
  ① 日次リターン基本統計（mean / std / skew / 勝率）
  ② オーバーナイト(ON)特性 — 寄付きギャップの分布・ギャップフィル率
  ③ セッション分解 — 前場 vs 後場のリターン特性、AM→PM相関
  ④ 曜日効果 — Mon〜Fri のセクター別平均
  ⑤ モメンタム vs リバーサル — 1d/5d/20d自己相関
  ⑥ 出来高×リターン感応度 — vol_ratio別の翌日リターン
  ⑦ 時間帯別ボラティリティ（代表セクターのみ、1分足）
"""

import psycopg2
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

PG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}
START = '2024-01-04'
END   = '2026-05-08'

def q(sql):
    conn = psycopg2.connect(**PG)
    df = pd.read_sql(sql, conn)
    conn.close()
    return df

print("=" * 80)
print("  セクター別 包括パターン分析")
print("=" * 80)

# ════════════════════════════════════════════════════════════════
# 共通: 銘柄日次データ（プライム + sector33 + 派生指標）
# ════════════════════════════════════════════════════════════════
print("\n  共通データ準備中...")

base_sql = f"""
WITH base AS (
    SELECT d.code, s.sector33_nm AS sector, d.date,
           d.open, d.close, d.high, d.low, d.volume,
           d.adj_open, d.adj_close, d.adj_volume, d.turnover_value,
           d.morning_close, d.afternoon_open, d.afternoon_close,
           EXTRACT(DOW FROM d.date)::int AS dow
    FROM stocks_daily d
    JOIN symbol_master s ON s.code5 = d.code
    WHERE s.market = '0111' AND d.date BETWEEN '{START}' AND '{END}'
      AND d.adj_close > 0 AND d.adj_open > 0 AND d.adj_volume > 0
      AND d.turnover_value >= 500000000  -- 5億円フィルタ（流動性）
),
ind AS (
    SELECT *,
        LAG(adj_close, 1) OVER (PARTITION BY code ORDER BY date) AS prev_close,
        LAG(adj_close, 5) OVER (PARTITION BY code ORDER BY date) AS prev_close_5,
        LAG(adj_close,20) OVER (PARTITION BY code ORDER BY date) AS prev_close_20,
        LEAD(adj_close, 1) OVER (PARTITION BY code ORDER BY date) AS next_close,
        LEAD(adj_close, 5) OVER (PARTITION BY code ORDER BY date) AS next_close_5,
        LEAD(adj_open, 1) OVER (PARTITION BY code ORDER BY date) AS next_open,
        AVG(adj_volume) OVER (PARTITION BY code ORDER BY date
                              ROWS BETWEEN 21 PRECEDING AND 2 PRECEDING) AS vol_ma20
    FROM base
)
SELECT code, sector, date, dow,
       adj_open, adj_close, prev_close, next_open, next_close, next_close_5,
       prev_close_5, prev_close_20,
       morning_close, afternoon_open, afternoon_close,
       adj_volume, vol_ma20, turnover_value
FROM ind
WHERE prev_close IS NOT NULL AND prev_close > 0
"""

df = q(base_sql)
print(f"    取得行数: {len(df):,}")
print(f"    銘柄数:   {df['code'].nunique():,}")
print(f"    セクター: {df['sector'].nunique()}")

# 派生指標
for c in df.columns:
    if c not in ('code', 'sector', 'date', 'dow'):
        df[c] = pd.to_numeric(df[c], errors='coerce')

df['day_ret']     = (df['adj_close'] / df['adj_open'] - 1) * 100  # 日中（寄→引）
df['on_gap']      = (df['adj_open']  / df['prev_close'] - 1) * 100  # 前日終→寄
df['total_ret']   = (df['adj_close'] / df['prev_close']  - 1) * 100 # 1日リターン
# 前場・後場は日次テーブルに無いので、後で1分足から計算するため一旦スキップ
df['am_ret']      = np.nan
df['pm_ret']      = np.nan
df['fwd_1d']      = (df['next_close']     / df['adj_close']       - 1) * 100  # 引→翌引
df['fwd_5d']      = (df['next_close_5']   / df['adj_close']       - 1) * 100
df['ret_5d_past'] = (df['adj_close']      / df['prev_close_5']    - 1) * 100
df['ret_20d_past']= (df['adj_close']      / df['prev_close_20']   - 1) * 100
df['vol_ratio']   = df['adj_volume']      / df['vol_ma20']

df = df[df['vol_ratio'].notna()]
print(f"    最終サンプル: {len(df):,}")


# ════════════════════════════════════════════════════════════════
# ① 日次リターン基本統計
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("【①】セクター別 日次リターン基本統計")
print("=" * 80)

agg1 = df.groupby('sector').agg(
    N         =('total_ret', 'size'),
    mean      =('total_ret', 'mean'),
    std       =('total_ret', 'std'),
    skew      =('total_ret', 'skew'),
    p_win     =('total_ret', lambda x: (x > 0).mean() * 100),
    on_mean   =('on_gap',    'mean'),
    on_std    =('on_gap',    'std'),
    day_mean  =('day_ret',   'mean'),
    day_std   =('day_ret',   'std'),
).round(3).sort_values('mean', ascending=False)
agg1['sharpe_d'] = (agg1['mean'] / agg1['std'] * np.sqrt(252)).round(2)
agg1['on_share'] = (agg1['on_mean'].abs() / (agg1['on_mean'].abs() + agg1['day_mean'].abs()) * 100).round(1)

print(f"\n  {'セクター':<14}  {'N':>6}  {'1日Ret':>7}  {'σ':>5}  {'歪度':>5}  "
      f"{'勝率':>5}  {'ON平均':>7}  {'日中平均':>7}  {'ON寄与':>7}  {'年率Sharpe':>10}")
print("  " + "-" * 88)
for s, r in agg1.iterrows():
    print(f"  {s:<14}  {int(r['N']):>6,}  {r['mean']:>+6.3f}%  {r['std']:>4.2f}%  "
          f"{r['skew']:>+4.2f}  {r['p_win']:>4.1f}%  {r['on_mean']:>+6.3f}%  "
          f"{r['day_mean']:>+6.3f}%  {r['on_share']:>6.1f}%  {r['sharpe_d']:>+9.2f}")


# ════════════════════════════════════════════════════════════════
# ② オーバーナイト (ON) 特性 — ギャップフィル率
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("【②】セクター別 オーバーナイト・ギャップフィル率")
print("  GU = ギャップアップ後、日中に寄付きを下抜けた率（フィル成功）")
print("  GD = ギャップダウン後、日中に寄付きを上抜けた率")
print("=" * 80)

# ギャップフィル: GU(寄付>前日終)後、low<=prev_close = フィル
# 簡易近似: low/highデータがないので、引け値が寄付の反対側にきたかで判定
df['gu']      = (df['on_gap'] >  0.3).astype(int)
df['gd']      = (df['on_gap'] < -0.3).astype(int)
df['gu_fill'] = ((df['gu'] == 1) & (df['adj_close'] <= df['prev_close'])).astype(int)
df['gd_fill'] = ((df['gd'] == 1) & (df['adj_close'] >= df['prev_close'])).astype(int)

gap = df.groupby('sector').agg(
    n_gu     =('gu',      'sum'),
    gu_fill  =('gu_fill', 'sum'),
    n_gd     =('gd',      'sum'),
    gd_fill  =('gd_fill', 'sum'),
    gu_mean  =('on_gap',  lambda x: x[x > 0.3].mean() if (x > 0.3).any() else np.nan),
    gd_mean  =('on_gap',  lambda x: x[x < -0.3].mean() if (x < -0.3).any() else np.nan),
)
gap['gu_fill_rate'] = (gap['gu_fill'] / gap['n_gu'].replace(0, np.nan) * 100).round(1)
gap['gd_fill_rate'] = (gap['gd_fill'] / gap['n_gd'].replace(0, np.nan) * 100).round(1)
gap = gap.sort_values('gu_fill_rate', ascending=False)

print(f"\n  {'セクター':<14}  {'GU件数':>6}  {'GUフィル率':>10}  {'GU幅':>7}  "
      f"{'GD件数':>6}  {'GDフィル率':>10}  {'GD幅':>7}")
print("  " + "-" * 75)
for s, r in gap.iterrows():
    if r['n_gu'] < 50 or r['n_gd'] < 50:
        continue
    print(f"  {s:<14}  {int(r['n_gu']):>6}  {r['gu_fill_rate']:>9.1f}%  "
          f"{r['gu_mean']:>+6.2f}%  {int(r['n_gd']):>6}  "
          f"{r['gd_fill_rate']:>9.1f}%  {r['gd_mean']:>+6.2f}%")


# ════════════════════════════════════════════════════════════════
# ③ セッション分解 — 前場 vs 後場（1分足から計算）
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("【③】セッション分解 — 前場リターン vs 後場リターン（1分足ベース）")
print("  AM→PM相関 < 0 = リバーサル型 / > 0 = モメンタム型")
print("=" * 80)

print("  各セクター代表3銘柄の前場(9:00→11:30)・後場(12:30→15:30)集計中...")

# 代表銘柄（前段で取得しているのでここで先に取る）
rep_pre_sql = """
WITH ranked AS (
    SELECT d.code, s.sector33_nm,
           AVG(d.turnover_value) AS avg_tv,
           ROW_NUMBER() OVER (PARTITION BY s.sector33_nm ORDER BY AVG(d.turnover_value) DESC) AS rn
    FROM stocks_daily d
    JOIN symbol_master s ON s.code5 = d.code
    WHERE s.market = '0111' AND d.date >= '2025-01-01'
    GROUP BY d.code, s.sector33_nm
)
SELECT code, sector33_nm AS sector FROM ranked WHERE rn <= 3
"""
rep_pre = q(rep_pre_sql)
codes_pre = "','".join(rep_pre['code'].tolist())

sess_sql = f"""
SELECT code, DATE(ts) AS d,
       (MAX(CASE WHEN EXTRACT(HOUR FROM ts) < 11 OR (EXTRACT(HOUR FROM ts)=11 AND EXTRACT(MINUTE FROM ts)<=30)
                 THEN close END)
        / NULLIF(MIN(CASE WHEN EXTRACT(HOUR FROM ts)=9 AND EXTRACT(MINUTE FROM ts)=0 THEN open END), 0) - 1) * 100 AS am_ret,
       (MAX(CASE WHEN EXTRACT(HOUR FROM ts) >= 13 OR (EXTRACT(HOUR FROM ts)=12 AND EXTRACT(MINUTE FROM ts)>=30)
                 THEN close END)
        / NULLIF(MIN(CASE WHEN EXTRACT(HOUR FROM ts)=12 AND EXTRACT(MINUTE FROM ts)=30 THEN open END), 0) - 1) * 100 AS pm_ret
FROM stocks_intraday
WHERE code IN ('{codes_pre}') AND ts >= '2025-01-01'
GROUP BY code, DATE(ts)
"""
sess_raw = q(sess_sql)
sess_raw = sess_raw.merge(rep_pre[['code', 'sector']], on='code')
sess_raw['am_ret'] = pd.to_numeric(sess_raw['am_ret'], errors='coerce')
sess_raw['pm_ret'] = pd.to_numeric(sess_raw['pm_ret'], errors='coerce')
sess_df = sess_raw.dropna(subset=['am_ret', 'pm_ret'])

sess = sess_df.groupby('sector').agg(
    N        =('am_ret', 'size'),
    am_mean  =('am_ret', 'mean'),
    am_std   =('am_ret', 'std'),
    pm_mean  =('pm_ret', 'mean'),
    pm_std   =('pm_ret', 'std'),
).round(3)

corrs = []
for s, g in sess_df.groupby('sector'):
    if len(g) >= 100:
        corrs.append({'sector': s, 'am_pm_corr': g['am_ret'].corr(g['pm_ret'])})
corr_df = pd.DataFrame(corrs).set_index('sector')
sess = sess.join(corr_df).sort_values('am_pm_corr')

print(f"\n  {'セクター':<14}  {'N':>6}  {'前場mean':>8}  {'σ':>5}  "
      f"{'後場mean':>8}  {'σ':>5}  {'AM-PM相関':>10}  {'タイプ':<12}")
print("  " + "-" * 80)
for s, r in sess.iterrows():
    if pd.isna(r['am_pm_corr']):
        continue
    typ = "リバーサル" if r['am_pm_corr'] < -0.05 else \
          ("モメンタム"  if r['am_pm_corr'] >  0.05 else "ニュートラル")
    print(f"  {s:<14}  {int(r['N']):>6,}  {r['am_mean']:>+7.3f}%  {r['am_std']:>4.2f}%  "
          f"{r['pm_mean']:>+7.3f}%  {r['pm_std']:>4.2f}%  "
          f"{r['am_pm_corr']:>+9.3f}  {typ:<12}")


# ════════════════════════════════════════════════════════════════
# ④ 曜日効果
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("【④】曜日効果 — セクター別")
print("=" * 80)

dow_map = {1: '月', 2: '火', 3: '水', 4: '木', 5: '金'}
dow_piv = df[df['dow'].isin([1, 2, 3, 4, 5])].groupby(['sector', 'dow'])['total_ret'].mean().unstack(fill_value=np.nan)
dow_piv.columns = [dow_map[d] for d in dow_piv.columns]
dow_piv['avg'] = dow_piv.mean(axis=1)
dow_piv['range'] = dow_piv[['月','火','水','木','金']].max(axis=1) - dow_piv[['月','火','水','木','金']].min(axis=1)
dow_piv = dow_piv.sort_values('range', ascending=False)

print(f"\n  曜日変動の大きい順（max - min）")
print(f"  {'セクター':<14}  {'月':>6}  {'火':>6}  {'水':>6}  {'木':>6}  {'金':>6}  {'平均':>6}  {'幅':>6}")
print("  " + "-" * 75)
for s, r in dow_piv.head(15).iterrows():
    cells = "  ".join(f"{r[d]:>+5.2f}%" if not pd.isna(r[d]) else "  ---" for d in ['月','火','水','木','金'])
    print(f"  {s:<14}  {cells}  {r['avg']:>+5.2f}%  {r['range']:>5.2f}%")

# ベスト曜日・ワースト曜日
print(f"\n  ─ セクター別 最良曜日 / 最悪曜日 ─")
print(f"  {'セクター':<14}  {'最良':>4}  {'値':>7}  {'最悪':>4}  {'値':>7}")
print("  " + "-" * 50)
for s, r in dow_piv.iterrows():
    vals = r[['月','火','水','木','金']].dropna()
    if len(vals) < 5:
        continue
    print(f"  {s:<14}  {vals.idxmax():>4}  {vals.max():>+6.3f}%  "
          f"{vals.idxmin():>4}  {vals.min():>+6.3f}%")


# ════════════════════════════════════════════════════════════════
# ⑤ モメンタム vs リバーサル — 自己相関 1d/5d/20d
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("【⑤】モメンタム vs リバーサル — 過去Nd vs 翌5d 自己相関")
print("  > 0 = モメンタム / < 0 = 平均回帰")
print("=" * 80)

# 銘柄ごとに自己相関を計算してセクター平均
acs = []
for sector, g in df.groupby('sector'):
    sub = g.dropna(subset=['ret_5d_past', 'ret_20d_past', 'fwd_5d'])
    if len(sub) < 200:
        continue
    acs.append({
        'sector':    sector,
        'N':         len(sub),
        'ac_1d':     sub['total_ret'].corr(sub['fwd_1d']),
        'ac_5d':     sub['ret_5d_past'].corr(sub['fwd_5d']),
        'ac_20d':    sub['ret_20d_past'].corr(sub['fwd_5d']),
    })

ac_df = pd.DataFrame(acs).set_index('sector').sort_values('ac_5d', ascending=False)

print(f"\n  {'セクター':<14}  {'N':>6}  {'1d→1d':>8}  {'5d→5d':>8}  {'20d→5d':>8}  {'特性'}")
print("  " + "-" * 70)
for s, r in ac_df.iterrows():
    typ = "強モメンタム" if r['ac_5d'] > 0.05 else \
          "強リバーサル" if r['ac_5d'] < -0.05 else "中立"
    print(f"  {s:<14}  {int(r['N']):>6,}  {r['ac_1d']:>+7.3f}  {r['ac_5d']:>+7.3f}  "
          f"{r['ac_20d']:>+7.3f}  {typ}")


# ════════════════════════════════════════════════════════════════
# ⑥ 出来高×リターン感応度 — vol_ratio別の翌日リターン
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("【⑥】出来高 vol_ratio 別 翌日リターン — セクター別")
print("=" * 80)

vol_buckets = [
    ('低出来高',   0.0, 0.8),
    ('普通',       0.8, 1.5),
    ('大商い',     1.5, 3.0),
    ('超急増',     3.0, 99.0),
]

vol_res = []
for sector, g in df.groupby('sector'):
    if len(g) < 500:
        continue
    row = {'sector': sector}
    for name, lo, hi in vol_buckets:
        sub = g[(g['vol_ratio'] >= lo) & (g['vol_ratio'] < hi)].dropna(subset=['fwd_1d'])
        row[f'{name}_N']  = len(sub)
        row[f'{name}_r']  = sub['fwd_1d'].mean() if len(sub) > 20 else np.nan
        row[f'{name}_wr'] = (sub['fwd_1d'] > 0).mean() * 100 if len(sub) > 20 else np.nan
    vol_res.append(row)

vol_df = pd.DataFrame(vol_res).set_index('sector')
vol_df['edge_high_vs_low'] = vol_df['超急増_r'] - vol_df['低出来高_r']
vol_df = vol_df.sort_values('edge_high_vs_low', ascending=False)

print(f"\n  ↑ vol_ratio≥3x の翌日リターンが「低出来高日」より高い → 大商い順張り型")
print(f"  ↓ 逆 → 大商いリバーサル型")
print(f"\n  {'セクター':<14}  {'低出来高':>9}  {'普通':>8}  {'大商い':>8}  {'超急増':>8}  {'差(超-低)':>10}")
print("  " + "-" * 75)
for s, r in vol_df.iterrows():
    cells = []
    for name in ['低出来高', '普通', '大商い', '超急増']:
        v = r[f'{name}_r']
        cells.append(f"{v:>+7.3f}%" if not pd.isna(v) else "    ---")
    diff = r['edge_high_vs_low']
    diff_s = f"{diff:>+8.3f}%" if not pd.isna(diff) else "       ---"
    print(f"  {s:<14}  {cells[0]:>9}  {cells[1]:>8}  {cells[2]:>8}  {cells[3]:>8}  {diff_s:>10}")


# ════════════════════════════════════════════════════════════════
# ⑦ 時間帯別ボラティリティ（代表セクター・1分足）
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("【⑦】時間帯別ボラティリティ（代表セクター × 1分足）")
print("  対象: 各セクターの売買代金上位3銘柄")
print("=" * 80)

rep_sql = """
WITH ranked AS (
    SELECT d.code, s.sector33_nm,
           AVG(d.turnover_value) AS avg_tv,
           ROW_NUMBER() OVER (PARTITION BY s.sector33_nm ORDER BY AVG(d.turnover_value) DESC) AS rn
    FROM stocks_daily d
    JOIN symbol_master s ON s.code5 = d.code
    WHERE s.market = '0111' AND d.date >= '2025-01-01'
    GROUP BY d.code, s.sector33_nm
)
SELECT code, sector33_nm AS sector FROM ranked WHERE rn <= 3
"""
rep = q(rep_sql)
print(f"\n  代表銘柄数: {len(rep)} ({rep['sector'].nunique()}セクター × 3銘柄)")

# 1分足の集計：各分のσ（リターンの標準偏差）
print("  1分足のセクター別時間帯ボラティリティ計算中...")

# 時間帯バケット定義（JST）
time_buckets = [
    ('09:00-09:15', 9,  0,  9, 15),
    ('09:15-10:00', 9, 15, 10, 0),
    ('10:00-11:30', 10, 0, 11, 30),
    ('12:30-13:30', 12, 30, 13, 30),
    ('13:30-14:30', 13, 30, 14, 30),
    ('14:30-15:30', 14, 30, 15, 30),
]

# 全代表銘柄の1分足を一括取得し、セクター別集計
codes_list = "','".join(rep['code'].tolist())
intra_sql = f"""
SELECT i.code,
       EXTRACT(HOUR FROM i.ts)::int   AS h,
       EXTRACT(MINUTE FROM i.ts)::int AS m,
       LN(i.close::float / NULLIF(i.open::float, 0)) AS log_ret
FROM stocks_intraday i
WHERE i.code IN ('{codes_list}')
  AND i.ts >= '2025-01-01'
  AND i.open > 0 AND i.close > 0
"""
intra = q(intra_sql)
intra = intra.merge(rep[['code', 'sector']], on='code')
intra['log_ret'] = pd.to_numeric(intra['log_ret'], errors='coerce')
intra = intra.dropna(subset=['log_ret'])

print(f"  1分足ロード: {len(intra):,}行")

# 時間帯マッピング
def tb_label(h, m):
    for label, sh, sm, eh, em in time_buckets:
        if (h > sh or (h == sh and m >= sm)) and (h < eh or (h == eh and m < em)):
            return label
    return None

intra['tb'] = [tb_label(h, m) for h, m in zip(intra['h'], intra['m'])]
intra = intra.dropna(subset=['tb'])

vol_table = intra.groupby(['sector', 'tb'])['log_ret'].std() * 100  # %換算
vol_table = vol_table.unstack('tb').reindex(columns=[t[0] for t in time_buckets])
vol_table['日中平均'] = vol_table.mean(axis=1)
vol_table['U字度'] = (vol_table['09:00-09:15'] + vol_table['14:30-15:30']) / 2 - vol_table['12:30-13:30']
vol_table = vol_table.sort_values('U字度', ascending=False)

print(f"\n  セクター別 各時間帯の1分リターンσ (%)")
print(f"  {'セクター':<14}  " + "  ".join(f"{t[0]:>11}" for t in time_buckets) + f"  {'日中平均':>8}  {'U字度':>7}")
print("  " + "-" * 110)
for s, r in vol_table.iterrows():
    cells = "  ".join(f"{r[t[0]]:>10.3f}%" if not pd.isna(r[t[0]]) else "       ---" for t in time_buckets)
    avg = r['日中平均']
    u = r['U字度']
    print(f"  {s:<14}  {cells}  {avg:>7.3f}%  {u:>+6.3f}%")


# ════════════════════════════════════════════════════════════════
# まとめ
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("【まとめ】セクター別の特徴的傾向トップ発見")
print("=" * 80)

# 各分析からのハイライト抽出
print(f"""
  ■ 日次パフォーマンス上位3セクター（年率Sharpe）:
""")
top_sharpe = agg1.sort_values('sharpe_d', ascending=False).head(3)
for s, r in top_sharpe.iterrows():
    print(f"    {s:<12} Sharpe={r['sharpe_d']:+.2f} (年率)、勝率{r['p_win']:.1f}%、ON寄与{r['on_share']:.0f}%")

print(f"""
  ■ ON特性（GUフィル率が高い = ギャップ反転しやすい）TOP3:""")
for s, r in gap.head(3).iterrows():
    print(f"    {s:<12} GUフィル{r['gu_fill_rate']:.1f}%、GDフィル{r['gd_fill_rate']:.1f}%")

print(f"""
  ■ AM→PMリバーサル型セクター（強い）TOP3:""")
for s, r in sess.dropna(subset=['am_pm_corr']).head(3).iterrows():
    print(f"    {s:<12} 相関={r['am_pm_corr']:+.3f}、前場σ={r['am_std']:.2f}%、後場σ={r['pm_std']:.2f}%")

print(f"""
  ■ 強モメンタム型セクター（5d→5d自己相関）TOP3:""")
for s, r in ac_df.head(3).iterrows():
    print(f"    {s:<12} 5d自己相関={r['ac_5d']:+.3f}、1d={r['ac_1d']:+.3f}")

print(f"""
  ■ 強リバーサル型セクター TOP3:""")
for s, r in ac_df.tail(3).iterrows():
    print(f"    {s:<12} 5d自己相関={r['ac_5d']:+.3f}、1d={r['ac_1d']:+.3f}")

print(f"""
  ■ 大商い順張りエッジ大きい（超急増 - 低出来高 翌日Ret差）TOP3:""")
for s, r in vol_df.head(3).iterrows():
    print(f"    {s:<12} 差{r['edge_high_vs_low']:+.3f}%、超急増翌日{r['超急増_r']:+.3f}%")

print(f"""
  ■ 大商いリバーサル傾向（差が負）TOP3:""")
for s, r in vol_df.tail(3).iterrows():
    print(f"    {s:<12} 差{r['edge_high_vs_low']:+.3f}%、超急増翌日{r['超急増_r']:+.3f}%")

# CSV出力
out_dir = '/Users/Yusuke/claude-code/japan-stocks/analyses/20260511_sector_patterns_comprehensive'
agg1.to_csv(f'{out_dir}/daily_stats.csv')
gap.to_csv(f'{out_dir}/gap_fill.csv')
sess.to_csv(f'{out_dir}/session_breakdown.csv')
dow_piv.to_csv(f'{out_dir}/day_of_week.csv')
ac_df.to_csv(f'{out_dir}/autocorrelation.csv')
vol_df.to_csv(f'{out_dir}/volume_sensitivity.csv')
vol_table.to_csv(f'{out_dir}/intraday_volatility.csv')

print(f"\n  ✅ 分析完了。CSV 7本を {out_dir}/ に出力")
