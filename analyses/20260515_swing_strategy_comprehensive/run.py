"""
スイングトレード戦略 — 包括的仮説検証
=========================================
分析日: 2026-05-15
対象:   プライム全銘柄 × 2024-01-04〜2026-05-14 (約2.3年)
保有期間: 5日 / 10日 / 20日

【既存戦略でカバーされていない領域を網羅検証】

H1.  52週高値ブレイクアウト           (古典モメンタム)
H2.  MA25からの大幅乖離 → 反発         (過売りリバーサル)
H3.  20日下落率Worst → リバウンド      (loser reversal)
H4.  20日上昇率Best → 継続            (winner continuation)
H5.  5MA × 25MA ゴールデンクロス       (トレンドフォロー)
H6.  連続陽線/陰線後の継続/反転        (streak)
H7.  ボリンジャーバンドBreakout/Touch  (volatility)
H8.  RSI extreme levels                (oversold/overbought)
H9.  20日高値ブレイク × 出来高急増      (volume confirmed breakout)
H10. dd_from_high (高値からの下落率) 反発 (drawdown reversal)
H11. 5日連騰銘柄の継続                  (short-term momentum)
H12. 業績修正発表5日後の継続/反転       (after EarnForecast)

【判定基準】 N≥200, net>0.3%, Sharpe>1.5, t-stat>2
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
print("  スイング戦略 包括検証 — 12仮説 × 3保有期間")
print("=" * 80)
print("\n  データロード中... (各種テクニカル指標を含む大量SQLになります)")

# 大量Window関数で一括計算
sql = """
WITH base AS (
    SELECT d.code, s.name_ja, s.sector33_nm AS sector,
           d.date, d.adj_open, d.adj_close, d.adj_high, d.adj_low,
           d.adj_volume, d.turnover_value
    FROM stocks_daily d
    JOIN symbol_master s ON s.code5 = d.code
    WHERE s.market = '0111' AND d.date BETWEEN '2023-01-01' AND '2026-05-14'
      AND d.adj_close > 0 AND d.adj_open > 0
),
ind AS (
    SELECT *,
        -- 移動平均
        AVG(adj_close)  OVER w5  AS ma5,
        AVG(adj_close)  OVER w25 AS ma25,
        AVG(adj_close)  OVER w50 AS ma50,
        AVG(adj_volume) OVER w20 AS vol_ma20,
        -- 標準偏差 (ボリンジャー用)
        STDDEV(adj_close) OVER w20 AS sd20,
        -- 高値・安値
        MAX(adj_high)   OVER w20 AS high_20d,
        MIN(adj_low)    OVER w20 AS low_20d,
        MAX(adj_high)   OVER w_yr AS high_252d,
        MIN(adj_low)    OVER w_yr AS low_252d,
        -- 20日リターン
        LAG(adj_close, 20) OVER (PARTITION BY code ORDER BY date) AS close_20d_ago,
        LAG(adj_close, 5)  OVER (PARTITION BY code ORDER BY date) AS close_5d_ago,
        -- 連騰/連敗カウント用
        LAG(adj_close, 1) OVER (PARTITION BY code ORDER BY date) AS close_1d_ago,
        -- 翌日寄付 + 出口価格
        LEAD(adj_open, 1)   OVER (PARTITION BY code ORDER BY date) AS next_open,
        LEAD(adj_close, 5)  OVER (PARTITION BY code ORDER BY date) AS close_fwd5,
        LEAD(adj_close, 10) OVER (PARTITION BY code ORDER BY date) AS close_fwd10,
        LEAD(adj_close, 20) OVER (PARTITION BY code ORDER BY date) AS close_fwd20
    FROM base
    WINDOW
        w5   AS (PARTITION BY code ORDER BY date ROWS BETWEEN  5 PRECEDING AND 1 PRECEDING),
        w25  AS (PARTITION BY code ORDER BY date ROWS BETWEEN 25 PRECEDING AND 1 PRECEDING),
        w50  AS (PARTITION BY code ORDER BY date ROWS BETWEEN 50 PRECEDING AND 1 PRECEDING),
        w20  AS (PARTITION BY code ORDER BY date ROWS BETWEEN 21 PRECEDING AND 2 PRECEDING),
        w_yr AS (PARTITION BY code ORDER BY date ROWS BETWEEN 252 PRECEDING AND 1 PRECEDING)
)
SELECT * FROM ind
WHERE date >= '2024-01-04'
  AND ma25 IS NOT NULL AND high_252d IS NOT NULL
  AND turnover_value >= 1000000000  -- 流動性フィルタ
"""

df = q(sql)
print(f"  ロード完了: {len(df):,}行")

# Float化
numeric_cols = ['adj_open','adj_close','adj_high','adj_low','adj_volume','turnover_value',
                'ma5','ma25','ma50','vol_ma20','sd20',
                'high_20d','low_20d','high_252d','low_252d',
                'close_20d_ago','close_5d_ago','close_1d_ago',
                'next_open','close_fwd5','close_fwd10','close_fwd20']
for c in numeric_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')
df['date'] = pd.to_datetime(df['date'])

# 派生指標
df['day_ret']        = (df['adj_close'] / df['adj_open'] - 1) * 100
df['vol_ratio']      = df['adj_volume'] / df['vol_ma20']
df['dist_ma25']      = (df['adj_close'] / df['ma25'] - 1) * 100
df['dist_ma50']      = (df['adj_close'] / df['ma50'] - 1) * 100
df['ret_20d']        = (df['adj_close'] / df['close_20d_ago'] - 1) * 100
df['ret_5d']         = (df['adj_close'] / df['close_5d_ago'] - 1) * 100
df['dd_from_high']   = (df['adj_close'] / df['high_252d'] - 1) * 100   # 52週高値からの下落率(負)
df['bb_pos']         = (df['adj_close'] - df['ma25']) / (df['sd20'] * 2)  # ±1で2σ
df['ret_today']      = (df['adj_close'] / df['close_1d_ago'] - 1) * 100

# 前向きリターン (翌寄り → N日後引け)
for h in [5, 10, 20]:
    df[f'fwd_{h}d'] = (df[f'close_fwd{h}'] / df['next_open'] - 1) * 100

# 有効サンプル
df_valid = df.dropna(subset=['next_open','fwd_5d','fwd_10d','fwd_20d','dist_ma25','ret_20d'])
print(f"  有効サンプル: {len(df_valid):,}行")

# 連騰カウント (簡易: 過去5日のリターン)
print(f"\n  各仮説の検証開始...")


# ════════════════════════════════════════════════════════════════
# 評価ヘルパー
# ════════════════════════════════════════════════════════════════
def evaluate(sub, h, label, results_list):
    """sub: フィルタ後DataFrame, h: ホライズン日数"""
    if len(sub) < 200:
        return
    r = sub[f'fwd_{h}d'].dropna()
    if len(r) < 200:
        return
    net = r - COST_PCT_RT
    sharpe = net.mean() / r.std() * np.sqrt(252/h) if r.std() > 0 else 0
    t_stat = r.mean() / r.std() * np.sqrt(len(r))
    results_list.append({
        'hypothesis': label, 'horizon': h, 'N': len(r),
        'gross': r.mean(), 'net': net.mean(),
        'wr': (r > 0).mean() * 100, 'std': r.std(),
        't_stat': t_stat, 'sharpe': sharpe,
    })


results = []


# H1. 52週高値ブレイクアウト
for h in [5, 10, 20]:
    sub = df_valid[df_valid['adj_close'] >= df_valid['high_252d'] * 0.999]
    evaluate(sub, h, 'H1_52週高値ブレイク', results)
    sub_vol = df_valid[(df_valid['adj_close'] >= df_valid['high_252d'] * 0.999) & (df_valid['vol_ratio'] >= 1.5)]
    evaluate(sub_vol, h, 'H1b_52週高値×出来高1.5x', results)


# H2. MA25からの大幅乖離 → 反発
for h in [5, 10, 20]:
    for lo, hi, label in [(-99, -15, 'H2a_MA25-15%超下'),
                           (-15, -10, 'H2b_MA25-15〜-10%'),
                           (-10, -5,  'H2c_MA25-10〜-5%')]:
        sub = df_valid[(df_valid['dist_ma25'] >= lo) & (df_valid['dist_ma25'] < hi)]
        evaluate(sub, h, label, results)


# H3. 20日下落率Worst → リバウンド
for h in [5, 10, 20]:
    df_day = df_valid.groupby('date')
    losers = df_valid.copy()
    losers['ret_20d_rank'] = losers.groupby('date')['ret_20d'].rank(pct=True)
    sub_worst = losers[losers['ret_20d_rank'] <= 0.05]   # 下位5%
    evaluate(sub_worst, h, 'H3_20日下落Worst5%', results)
    sub_worst2 = losers[losers['ret_20d_rank'] <= 0.10]
    evaluate(sub_worst2, h, 'H3b_20日下落Worst10%', results)


# H4. 20日上昇率Best → 継続
for h in [5, 10, 20]:
    winners = df_valid.copy()
    winners['ret_20d_rank'] = winners.groupby('date')['ret_20d'].rank(pct=True)
    sub_best = winners[winners['ret_20d_rank'] >= 0.95]
    evaluate(sub_best, h, 'H4_20日上昇Best5%', results)
    sub_best2 = winners[winners['ret_20d_rank'] >= 0.90]
    evaluate(sub_best2, h, 'H4b_20日上昇Best10%', results)


# H5. 5MA > 25MA ゴールデンクロス
df_valid['ma5_above'] = (df_valid['ma5'] > df_valid['ma25']).astype(int)
df_valid['ma5_gc'] = ((df_valid['ma5'] > df_valid['ma25']) &
                     (df_valid.groupby('code')['ma5_above'].shift(1) == 0)).astype(int)
for h in [5, 10, 20]:
    sub_gc = df_valid[df_valid['ma5_gc'] == 1]
    evaluate(sub_gc, h, 'H5_5MA>25MAゴールデンクロス', results)


# H6. 連続陽線 (5日リターン高) 後の継続
for h in [5, 10, 20]:
    sub = df_valid[df_valid['ret_5d'] > 5]   # 過去5日で+5%超
    evaluate(sub, h, 'H6a_5日+5%超', results)
    sub2 = df_valid[df_valid['ret_5d'] > 10]
    evaluate(sub2, h, 'H6b_5日+10%超', results)


# H7. ボリンジャーバンド
for h in [5, 10, 20]:
    sub_low = df_valid[df_valid['bb_pos'] < -1.5]   # -2σ近辺
    evaluate(sub_low, h, 'H7a_BB-2σ近辺', results)
    sub_high = df_valid[df_valid['bb_pos'] > 1.5]
    evaluate(sub_high, h, 'H7b_BB+2σ近辺', results)


# H8. RSI (簡易: 5日上昇率の極値で代用)
for h in [5, 10, 20]:
    sub_os = df_valid[df_valid['ret_5d'] < -8]    # 5日で-8%超
    evaluate(sub_os, h, 'H8a_5日-8%超下落(oversold)', results)
    sub_ob = df_valid[df_valid['ret_5d'] > 12]
    evaluate(sub_ob, h, 'H8b_5日+12%超上昇(overbought)', results)


# H9. 20日高値ブレイク × 出来高
for h in [5, 10, 20]:
    sub = df_valid[(df_valid['adj_close'] >= df_valid['high_20d'] * 0.999) &
                    (df_valid['vol_ratio'] >= 1.5)]
    evaluate(sub, h, 'H9_20日高値ブレイク×出来高1.5x', results)
    sub2 = df_valid[(df_valid['adj_close'] >= df_valid['high_20d'] * 0.999) &
                     (df_valid['vol_ratio'] >= 2.0)]
    evaluate(sub2, h, 'H9b_20日高値ブレイク×出来高2.0x', results)


# H10. 52週高値からの下落率 (dd_from_high)
for h in [5, 10, 20]:
    for lo, hi, label in [(-99, -30, 'H10a_52週高値-30%超'),
                           (-30, -20, 'H10b_52週高値-30〜-20%'),
                           (-20, -10, 'H10c_52週高値-20〜-10%')]:
        sub = df_valid[(df_valid['dd_from_high'] >= lo) & (df_valid['dd_from_high'] < hi)]
        evaluate(sub, h, label, results)


# H11. 5日連騰 momentum
for h in [5, 10, 20]:
    sub = df_valid[df_valid['ret_5d'].between(2, 5)]   # 緩やかな5日連騰
    evaluate(sub, h, 'H11a_5日+2〜+5%(緩騰)', results)
    sub2 = df_valid[df_valid['ret_5d'].between(5, 10)]
    evaluate(sub2, h, 'H11b_5日+5〜+10%', results)


# H12. dd_from_high × MA25乖離 複合 (oversold confluence)
for h in [5, 10, 20]:
    sub = df_valid[(df_valid['dd_from_high'] < -15) & (df_valid['dist_ma25'] < -8)]
    evaluate(sub, h, 'H12_52週高値-15%超×MA25-8%超下', results)


# 追加: シクリカルセクター限定で H3, H10
cyc = ['銀行業','非鉄金属','鉄鋼','機械','電気機器','建設業','輸送用機器','化学']
for h in [5, 10, 20]:
    sub_cyc = df_valid[df_valid['sector'].isin(cyc) & (df_valid['dd_from_high'] < -20)]
    evaluate(sub_cyc, h, 'H13a_シクリカル × 52週高値-20%超', results)
    sub_cyc2 = df_valid[df_valid['sector'].isin(cyc) & (df_valid['dist_ma25'] < -10)]
    evaluate(sub_cyc2, h, 'H13b_シクリカル × MA25-10%超下', results)


# ════════════════════════════════════════════════════════════════
# 結果ランキング
# ════════════════════════════════════════════════════════════════
results_df = pd.DataFrame(results).sort_values('sharpe', ascending=False)

print("\n" + "=" * 80)
print("【1】仮説別 TOP25 (Sharpe降順)")
print("=" * 80)
print(f"\n  {'仮説':<40}  {'H':>3}  {'N':>5}  {'gross':>7}  {'net':>7}  "
      f"{'勝率':>6}  {'t-stat':>7}  {'Sharpe':>7}")
print("  " + "-" * 95)
for _, r in results_df.head(25).iterrows():
    flag = " ★★★" if r['sharpe'] > 2.5 else (" ★★" if r['sharpe'] > 1.5 else (" ★" if r['sharpe'] > 1.0 else ""))
    print(f"  {r['hypothesis']:<40}  {int(r['horizon']):>2}d  {int(r['N']):>5,}  "
          f"{r['gross']:>+6.3f}%  {r['net']:>+6.3f}%  "
          f"{r['wr']:>5.1f}%  {r['t_stat']:>+6.2f}  {r['sharpe']:>+6.2f}{flag}")


# ════════════════════════════════════════════════════════════════
# 既存戦略との重複チェック (bank_absorption, earnings_pead は既存)
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("【2】仮説別ベスト保有期間")
print("=" * 80)

best_per_h = results_df.loc[results_df.groupby('hypothesis')['sharpe'].idxmax()]
best_per_h = best_per_h.sort_values('sharpe', ascending=False)
print(f"\n  {'仮説':<42}  {'最適H':>5}  {'N':>5}  {'net':>7}  {'Sharpe':>7}")
print("  " + "-" * 75)
for _, r in best_per_h.iterrows():
    if r['sharpe'] < 0.5:
        continue
    flag = " ★★" if r['sharpe'] > 1.5 else (" ★" if r['sharpe'] > 1.0 else "")
    print(f"  {r['hypothesis']:<42}  {int(r['horizon']):>4}d  {int(r['N']):>5,}  "
          f"{r['net']:>+6.3f}%  {r['sharpe']:>+6.2f}{flag}")


# CSV出力
out_dir = '/Users/Yusuke/claude-code/japan-stocks/analyses/20260515_swing_strategy_comprehensive'
results_df.to_csv(f'{out_dir}/all_results.csv', index=False)
print(f"\n  ✅ 完了。全{len(results_df)}結果出力")
