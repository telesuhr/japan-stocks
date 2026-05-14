"""
TOPIX 500 限定 スイング戦略 包括検証
=====================================
分析日: 2026-05-15

【対象ユニバース】
  TOPIX Core30 (31銘柄) + Large70 (69銘柄) + Mid400 (393銘柄) = 493銘柄
  小型株 (Small 1, Small 2) は除外

【背景】
  小型株は「下落後の回復」に疑義あり (流動性消失/上場廃止リスク)
  中型〜超大型株のみで同じ仮説が成立するか検証

【検証する仮説】 (12個 × 3保有期間 × サイズ階層別)
  H1.  MA25乖離 (oversold reversal)
  H2.  20日下落率Worst → リバウンド
  H3.  20日上昇率Best → 継続
  H4.  ボリンジャーバンド -2σ近辺
  H5.  52週高値ブレイクアウト × 出来高
  H6.  5MA > 25MA ゴールデンクロス
  H7.  5日リターン極値 (oversold/overbought)
  H8.  出来高吸収逆張り (vol≥2x × 陰線)
  H9.  MA25押し目 (-5〜-10%)
  H10. 大商い陽線後の継続
  H11. 大商い陰線後のリバ (汎用)
  H12. ボリンジャー上抜け順張り

【サイズ階層】
  Core30 / Large70 / Mid400 / TOPIX500 (3層合算)
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
print("  TOPIX 500 限定 スイング戦略 包括検証")
print("=" * 80)
print("\n  対象: TOPIX Core30 + Large70 + Mid400 = 493銘柄")
print("  期間: 2024-01-04 〜 2026-05-14")
print("\n  データロード中...")

sql = """
WITH base AS (
    SELECT d.code, s.name_ja, s.sector33_nm AS sector,
           s.scale_cat,
           d.date, d.adj_open, d.adj_close, d.adj_high, d.adj_low,
           d.adj_volume, d.turnover_value
    FROM stocks_daily d
    JOIN symbol_master s ON s.code5 = d.code
    WHERE s.market = '0111' AND d.date BETWEEN '2023-10-01' AND '2026-05-14'
      AND d.adj_close > 0 AND d.adj_open > 0
      AND s.scale_cat IN ('TOPIX Core30','TOPIX Large70','TOPIX Mid400')
),
ind AS (
    SELECT *,
        AVG(adj_close)  OVER w25 AS ma25,
        AVG(adj_close)  OVER w5  AS ma5,
        AVG(adj_volume) OVER w20 AS vol_ma20,
        STDDEV(adj_close) OVER w20 AS sd20,
        MAX(adj_high) OVER w20 AS high_20d,
        MIN(adj_low)  OVER w20 AS low_20d,
        MAX(adj_high) OVER w_yr AS high_252d,
        LAG(adj_close, 20) OVER (PARTITION BY code ORDER BY date) AS close_20d_ago,
        LAG(adj_close, 5)  OVER (PARTITION BY code ORDER BY date) AS close_5d_ago,
        LEAD(adj_open, 1)   OVER (PARTITION BY code ORDER BY date) AS next_open,
        LEAD(adj_close, 5)  OVER (PARTITION BY code ORDER BY date) AS close_fwd5,
        LEAD(adj_close, 10) OVER (PARTITION BY code ORDER BY date) AS close_fwd10,
        LEAD(adj_close, 20) OVER (PARTITION BY code ORDER BY date) AS close_fwd20
    FROM base
    WINDOW
        w5  AS (PARTITION BY code ORDER BY date ROWS BETWEEN  5 PRECEDING AND 1 PRECEDING),
        w25 AS (PARTITION BY code ORDER BY date ROWS BETWEEN 25 PRECEDING AND 1 PRECEDING),
        w20 AS (PARTITION BY code ORDER BY date ROWS BETWEEN 21 PRECEDING AND 2 PRECEDING),
        w_yr AS (PARTITION BY code ORDER BY date ROWS BETWEEN 252 PRECEDING AND 1 PRECEDING)
)
SELECT * FROM ind WHERE date >= '2024-01-04' AND ma25 IS NOT NULL
"""
df = q(sql)
for c in ['adj_open','adj_close','adj_high','adj_low','adj_volume','turnover_value',
          'ma25','ma5','vol_ma20','sd20','high_20d','low_20d','high_252d',
          'close_20d_ago','close_5d_ago','next_open','close_fwd5','close_fwd10','close_fwd20']:
    df[c] = pd.to_numeric(df[c], errors='coerce')
df['date'] = pd.to_datetime(df['date'])

# 派生指標
df['day_ret']      = (df['adj_close'] / df['adj_open'] - 1) * 100
df['vol_ratio']    = df['adj_volume'] / df['vol_ma20']
df['dist_ma25']    = (df['adj_close'] / df['ma25'] - 1) * 100
df['ret_20d']      = (df['adj_close'] / df['close_20d_ago'] - 1) * 100
df['ret_5d']       = (df['adj_close'] / df['close_5d_ago'] - 1) * 100
df['bb_pos']       = (df['adj_close'] - df['ma25']) / (df['sd20'] * 2)
df['dd_from_high'] = (df['adj_close'] / df['high_252d'] - 1) * 100

for h in [5, 10, 20]:
    df[f'fwd_{h}d'] = (df[f'close_fwd{h}'] / df['next_open'] - 1) * 100

# 有効サンプル
df_valid = df.dropna(subset=['next_open','fwd_5d','fwd_10d','fwd_20d','dist_ma25','ret_20d'])
print(f"  ロード完了: {len(df_valid):,}行")
print(f"  scale_cat別:")
for cat in ['TOPIX Core30','TOPIX Large70','TOPIX Mid400']:
    n = (df_valid['scale_cat']==cat).sum()
    print(f"    {cat:<20}: {n:,}行")


# ════════════════════════════════════════════════════════════════
# 評価関数
# ════════════════════════════════════════════════════════════════
def evaluate_signal(sub, h, label, scale_label, results):
    if len(sub) < 100:
        return
    r = sub[f'fwd_{h}d'].dropna()
    if len(r) < 100:
        return
    net = r - COST_PCT_RT
    sharpe = net.mean() / r.std() * np.sqrt(252/h) if r.std() > 0 else 0
    t_stat = r.mean() / r.std() * np.sqrt(len(r))
    pf = net[net>0].sum() / abs(net[net<=0].sum()) if (net<=0).any() else 99
    results.append({
        'hypothesis': label, 'scale': scale_label, 'horizon': h,
        'N': len(r), 'gross': r.mean(), 'net': net.mean(),
        'wr': (r > 0).mean() * 100, 't_stat': t_stat,
        'pf': pf, 'sharpe': sharpe,
    })


# ════════════════════════════════════════════════════════════════
# 仮説定義
# ════════════════════════════════════════════════════════════════
def define_hypotheses(d):
    """各仮説の (label, mask) を返す"""
    return [
        # H1: MA25乖離 (深さ別)
        ('H1a_MA25<-15%',         d['dist_ma25'] < -15),
        ('H1b_MA25<-10%',         d['dist_ma25'] < -10),
        ('H1c_MA25 -10〜-5%',     (d['dist_ma25'] >= -10) & (d['dist_ma25'] < -5)),
        # H2: 20日下落率
        ('H2a_20d下落<-20%',      d['ret_20d'] < -20),
        ('H2b_20d下落<-10%',      d['ret_20d'] < -10),
        # H3: 20日上昇率Best
        ('H3a_20d上昇>+20%',      d['ret_20d'] > 20),
        ('H3b_20d上昇>+10%',      d['ret_20d'] > 10),
        # H4: ボリンジャー-2σ
        ('H4a_BB<-1.8σ',          d['bb_pos'] < -1.8),
        ('H4b_BB<-1.5σ',          d['bb_pos'] < -1.5),
        # H5: 52週高値ブレイクアウト + 出来高
        ('H5_52週高値×vol≥1.5x',  (d['adj_close'] >= d['high_252d'] * 0.999) & (d['vol_ratio'] >= 1.5)),
        # H6: 5MA > 25MAクロス
        ('H6_5MA>25MA',           (d['ma5'] > d['ma25']) & (d['ma5'].shift(1) <= d['ma25'].shift(1))),
        # H7: 5日リターン極値
        ('H7a_5d<-8%',            d['ret_5d'] < -8),
        ('H7b_5d>+10%',           d['ret_5d'] > 10),
        # H8: 出来高吸収逆張り
        ('H8a_vol≥2.0x×陰線',     (d['vol_ratio'] >= 2.0) & (d['day_ret'] < -0.3)),
        ('H8b_vol≥1.5x×陰線',     (d['vol_ratio'] >= 1.5) & (d['day_ret'] < -0.3)),
        # H9: MA25押し目
        ('H9a_MA25 -5〜0%',       (d['dist_ma25'] >= -5) & (d['dist_ma25'] < 0)),
        ('H9b_MA25 -10〜-5%',     (d['dist_ma25'] >= -10) & (d['dist_ma25'] < -5)),
        # H10: 大商い陽線
        ('H10_vol≥2.0x×大陽線',   (d['vol_ratio'] >= 2.0) & (d['day_ret'] > 1.0)),
        # H11: dd from 52週高値
        ('H11_52高値-30%超下',     d['dd_from_high'] < -30),
        # H12: 20日高値ブレイク
        ('H12_20日高値×vol≥1.5x', (d['adj_close'] >= d['high_20d'] * 0.999) & (d['vol_ratio'] >= 1.5)),
    ]


# ════════════════════════════════════════════════════════════════
# 検証実行
# ════════════════════════════════════════════════════════════════
print(f"\n  検証中... (12仮説 × 3保有 × 4サイズ階層 = 約240パターン)")
results = []

# サイズ階層別
scales = [
    ('TOPIX500',  df_valid),
    ('Core30',    df_valid[df_valid['scale_cat']=='TOPIX Core30']),
    ('Large70',   df_valid[df_valid['scale_cat']=='TOPIX Large70']),
    ('Mid400',    df_valid[df_valid['scale_cat']=='TOPIX Mid400']),
    ('Core+Large',df_valid[df_valid['scale_cat'].isin(['TOPIX Core30','TOPIX Large70'])]),
]

for scale_label, scale_df in scales:
    for h in [5, 10, 20]:
        for hyp_label, mask in define_hypotheses(scale_df):
            sub = scale_df[mask]
            evaluate_signal(sub, h, hyp_label, scale_label, results)

results_df = pd.DataFrame(results)


# ════════════════════════════════════════════════════════════════
# 1. TOPIX500 全体ランキング
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("【1】TOPIX500 全体 — TOP20 (Sharpe降順)")
print("=" * 90)
top500 = results_df[results_df['scale'] == 'TOPIX500'].sort_values('sharpe', ascending=False)
print(f"\n  {'仮説':<24}  {'H':>3}  {'N':>5}  {'gross':>7}  {'net':>7}  "
      f"{'勝率':>6}  {'t-stat':>7}  {'PF':>5}  {'Sharpe':>7}")
print("  " + "-" * 95)
for _, r in top500.head(20).iterrows():
    flag = " ★★★" if r['sharpe']>2.5 else (" ★★" if r['sharpe']>1.5 else (" ★" if r['sharpe']>1.0 else ""))
    print(f"  {r['hypothesis']:<24}  {int(r['horizon']):>2}d  {int(r['N']):>5,}  "
          f"{r['gross']:>+6.3f}%  {r['net']:>+6.3f}%  "
          f"{r['wr']:>5.1f}%  {r['t_stat']:>+6.2f}  {r['pf']:>4.2f}  {r['sharpe']:>+6.2f}{flag}")


# ════════════════════════════════════════════════════════════════
# 2. サイズ階層別比較 (上位仮説のみ)
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 95)
print("【2】サイズ階層別比較 — TOP仮説のスケール特性")
print("=" * 95)

# TOPIX500でSharpe>1.0の仮説を抽出
top_hyps = top500[top500['sharpe'] > 1.0]['hypothesis'].unique()[:8]

for hyp in top_hyps:
    print(f"\n  ─ {hyp} ─")
    print(f"    {'スケール':<14}  {'H':>3}  {'N':>5}  {'net':>7}  {'勝率':>6}  {'Sharpe':>7}")
    sub_hyp = results_df[results_df['hypothesis']==hyp].sort_values(['scale','horizon'])
    for _, r in sub_hyp.iterrows():
        flag = " ★" if r['sharpe']>1.5 else ""
        print(f"    {r['scale']:<14}  {int(r['horizon']):>2}d  {int(r['N']):>5,}  "
              f"{r['net']:>+6.3f}%  {r['wr']:>5.1f}%  {r['sharpe']:>+6.2f}{flag}")


# ════════════════════════════════════════════════════════════════
# 3. 大型株 (Core30+Large70) 専用ベストランキング
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("【3】Core30+Large70 (大型株のみ100銘柄) — TOP15")
print("=" * 90)
large = results_df[results_df['scale']=='Core+Large'].sort_values('sharpe', ascending=False)
print(f"\n  {'仮説':<24}  {'H':>3}  {'N':>5}  {'net':>7}  {'勝率':>6}  {'t-stat':>7}  {'Sharpe':>7}")
print("  " + "-" * 80)
for _, r in large.head(15).iterrows():
    if r['sharpe'] < 0.5: break
    flag = " ★★" if r['sharpe']>1.5 else (" ★" if r['sharpe']>1.0 else "")
    print(f"  {r['hypothesis']:<24}  {int(r['horizon']):>2}d  {int(r['N']):>5,}  "
          f"{r['net']:>+6.3f}%  {r['wr']:>5.1f}%  {r['t_stat']:>+6.2f}  {r['sharpe']:>+6.2f}{flag}")


# ════════════════════════════════════════════════════════════════
# 4. 中型株 (Mid400) 専用ベストランキング
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("【4】Mid400 (中型株のみ393銘柄) — TOP15")
print("=" * 90)
mid = results_df[results_df['scale']=='Mid400'].sort_values('sharpe', ascending=False)
print(f"\n  {'仮説':<24}  {'H':>3}  {'N':>5}  {'net':>7}  {'勝率':>6}  {'t-stat':>7}  {'Sharpe':>7}")
print("  " + "-" * 80)
for _, r in mid.head(15).iterrows():
    if r['sharpe'] < 0.5: break
    flag = " ★★" if r['sharpe']>1.5 else (" ★" if r['sharpe']>1.0 else "")
    print(f"  {r['hypothesis']:<24}  {int(r['horizon']):>2}d  {int(r['N']):>5,}  "
          f"{r['net']:>+6.3f}%  {r['wr']:>5.1f}%  {r['t_stat']:>+6.2f}  {r['sharpe']:>+6.2f}{flag}")


# ════════════════════════════════════════════════════════════════
# 5. サイズ別年次安定性 (TOP仮説のみ)
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("【5】サイズ別年次安定性 — MA25乖離戦略 (best across hypotheses)")
print("=" * 90)

# H1a (MA25<-15%) を 5日保有 で年次分解
for scale_label, scale_df in scales:
    sub = scale_df[scale_df['dist_ma25'] < -15].copy()
    if len(sub) < 50:
        continue
    sub['year'] = sub['date'].dt.year
    print(f"\n  ─ {scale_label} (MA25<-15% × 5日保有) ─")
    print(f"    {'年':>4}  {'N':>4}  {'gross':>7}  {'net':>7}  {'勝率':>6}  {'Sharpe':>7}")
    for year, g in sub.groupby('year'):
        if len(g) < 5: continue
        r = g['fwd_5d'].dropna()
        if len(r) < 5: continue
        net = r - COST_PCT_RT
        sharpe = net.mean()/r.std()*np.sqrt(252/5) if r.std()>0 else 0
        print(f"    {year}  {len(r):>4}  {r.mean():>+6.3f}%  {net.mean():>+6.3f}%  "
              f"{(r>0).mean()*100:>5.1f}%  {sharpe:>+6.2f}")


# ════════════════════════════════════════════════════════════════
# 6. セクター別検証 (TOPIX500 × MA25<-15%)
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("【6】TOPIX500 × MA25<-15% × 5日保有 — セクター別")
print("=" * 80)

best_sig = df_valid[df_valid['dist_ma25'] < -15]
print(f"\n  {'セクター':<14}  {'N':>4}  {'net':>7}  {'勝率':>6}  {'Sharpe':>7}")
print("  " + "-" * 55)
for sect, g in best_sig.groupby('sector'):
    if len(g) < 20: continue
    r = g['fwd_5d'].dropna()
    if len(r) < 20: continue
    net = r - COST_PCT_RT
    sharpe = net.mean()/r.std()*np.sqrt(252/5) if r.std()>0 else 0
    flag = " ★" if sharpe>1.5 else ""
    print(f"  {sect:<14}  {len(g):>4}  {net.mean():>+6.3f}%  "
          f"{(r>0).mean()*100:>5.1f}%  {sharpe:>+6.2f}{flag}")


# CSV出力
out_dir = '/Users/Yusuke/claude-code/japan-stocks/analyses/20260515_swing_topix500'
results_df.to_csv(f'{out_dir}/all_results.csv', index=False)
print(f"\n  ✅ 完了。CSV出力")
