"""
決算当日・翌日の動き 包括分析
==============================
分析日: 2026-05-12
対象:   fin_summary 発表データ × stocks_daily × 2024-01〜2026-05

【データ】
  - fin_summary: 25,000件超の決算発表（プライム）
  - 発表タイミング:
      AC (After Close)   = disc_time ≥ 15:00 (引け後発表、最多)
      BO (Before Open)   = disc_time < 9:00 (寄付前発表)
      DS (During Session)= 9:00 ≤ disc_time < 15:00 (場中発表)

【検証項目】
  ① 発表タイミング別の翌日反応分布
  ② 発表前(T日)のイントラ強さは翌日反応を予測するか？
  ③ PEAD (Post-Earnings Announcement Drift) - 翌日ギャップ × T+2〜T+5ドリフト
  ④ 発表前1週間の出来高動向は反応を予測するか？
  ⑤ doc_type別の反応の違い（決算 vs 業績修正 vs 配当修正）
  ⑥ セクター別の反応強さ
  ⑦ 戦略候補のバックテスト
"""
import psycopg2
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

PG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}
COST_PCT_RT = 0.10
START = '2024-01-04'
END   = '2026-05-08'

def q(sql):
    conn = psycopg2.connect(**PG)
    df = pd.read_sql(sql, conn)
    conn.close()
    return df

print("=" * 78)
print("  決算当日・翌日の動き 包括分析")
print("=" * 78)
print("\n  決算発表データ + 周辺価格を取得中...")

# 決算発表 + 当日と翌日5日の価格を取得
sql = f"""
WITH events AS (
    SELECT f.code, s.name_ja, s.sector33_nm AS sector,
           f.disc_date, f.disc_time, f.doc_type,
           CASE
             WHEN f.disc_time >= '15:00:00' THEN 'AC'   -- After Close
             WHEN f.disc_time < '09:00:00'  THEN 'BO'   -- Before Open
             ELSE 'DS'                                  -- During Session
           END AS ann_timing
    FROM fin_summary f
    JOIN symbol_master s ON s.code5 = f.code
    WHERE s.market = '0111'
      AND f.disc_date BETWEEN '{START}' AND '{END}'
      AND f.disc_time IS NOT NULL
)
SELECT * FROM events
"""

events = q(sql)
events['disc_date'] = pd.to_datetime(events['disc_date'])
print(f"  発表イベント数: {len(events):,}")
print(f"  発表タイミング分布:")
for tm, n in events['ann_timing'].value_counts().items():
    print(f"    {tm}: {n:,}")

# 価格データ取得（プライム全銘柄、2024-）
price_sql = f"""
SELECT d.code, d.date, d.adj_open, d.adj_close, d.adj_high, d.adj_low,
       d.adj_volume, d.turnover_value
FROM stocks_daily d
JOIN symbol_master s ON s.code5 = d.code
WHERE s.market = '0111' AND d.date BETWEEN '{START}'::date - INTERVAL '40 days' AND '{END}'::date + INTERVAL '40 days'
  AND d.adj_close > 0
"""
prices = q(price_sql)
for c in ['adj_open','adj_close','adj_high','adj_low','adj_volume','turnover_value']:
    prices[c] = pd.to_numeric(prices[c], errors='coerce')
prices['date'] = pd.to_datetime(prices['date'])
prices = prices.sort_values(['code', 'date']).reset_index(drop=True)
print(f"  価格データ: {len(prices):,}行")

# 各銘柄の日付インデックスを作る（高速ルックアップ）
price_dict = {}
for code, g in prices.groupby('code'):
    price_dict[code] = g.set_index('date').sort_index()

# 各イベントに対して、周辺価格を結合
def get_event_window(row):
    code = row['code']
    disc = row['disc_date']
    if code not in price_dict:
        return None
    p = price_dict[code]
    if disc not in p.index:
        return None

    # T = 発表日（AC=引け後発表なので T_close は発表前最終値）
    # T+1 = 翌営業日（AC発表の反応日）
    dates = p.index
    try:
        idx = dates.get_loc(disc)
    except KeyError:
        return None

    if idx < 5 or idx + 5 >= len(dates):
        return None

    # T-5 〜 T+5
    rec = {}
    rec['t_open']  = p['adj_open'].iloc[idx]
    rec['t_close'] = p['adj_close'].iloc[idx]
    rec['t_high']  = p['adj_high'].iloc[idx]
    rec['t_low']   = p['adj_low'].iloc[idx]
    rec['t_vol']   = p['adj_volume'].iloc[idx]
    rec['t_tv']    = p['turnover_value'].iloc[idx]

    # T-1 close
    rec['tm1_close'] = p['adj_close'].iloc[idx - 1]
    # T-5 close
    rec['tm5_close'] = p['adj_close'].iloc[idx - 5]
    # T+1 open / close
    rec['tp1_open']  = p['adj_open'].iloc[idx + 1]
    rec['tp1_close'] = p['adj_close'].iloc[idx + 1]
    rec['tp1_high']  = p['adj_high'].iloc[idx + 1]
    rec['tp1_low']   = p['adj_low'].iloc[idx + 1]
    # T+5 close
    rec['tp5_close'] = p['adj_close'].iloc[idx + 5]
    # 20日平均出来高（T時点）
    win = p['adj_volume'].iloc[max(0, idx - 21):idx - 1]
    rec['vol_ma20'] = win.mean() if len(win) > 5 else np.nan
    return rec

print("\n  イベント周辺価格を結合中...")
windows = events.apply(get_event_window, axis=1)
# None を除外して DataFrame 化
valid_mask = windows.notna()
events = events[valid_mask].reset_index(drop=True)
windows = windows[valid_mask].reset_index(drop=True)
win_df = pd.DataFrame(list(windows))
events = pd.concat([events, win_df], axis=1)
events = events.dropna(subset=['t_close', 'tp1_open', 'tp1_close', 'tp5_close'])
print(f"  結合後: {len(events):,}")

# 派生指標
events['t_intraday_ret']   = (events['t_close'] / events['t_open']  - 1) * 100  # T日 寄→引（発表前イントラ）
events['t_total_ret']      = (events['t_close'] / events['tm1_close'] - 1) * 100  # T日 前日比
events['t_5d_pre_ret']     = (events['tm1_close'] / events['tm5_close'] - 1) * 100  # T-5→T-1
events['gap_ret']          = (events['tp1_open']  / events['t_close']  - 1) * 100  # T_close→T+1 寄 (ON gap)
events['day_after_ret']    = (events['tp1_close'] / events['tp1_open'] - 1) * 100  # T+1 寄→引
events['react_total']      = (events['tp1_close'] / events['t_close']  - 1) * 100  # T_close→T+1_close
events['drift_5d']         = (events['tp5_close'] / events['tp1_close'] - 1) * 100  # T+1_close→T+5_close
events['t_vol_ratio']      = events['t_vol'] / events['vol_ma20']

# 流動性フィルタ
events_liq = events[events['t_tv'] >= 500_000_000].copy()
print(f"\n  流動性フィルタ後 (T日売買代金5億+): {len(events_liq):,}")


# ════════════════════════════════════════════════════════════════
# ① 発表タイミング別 反応分布
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("【①】発表タイミング別 翌日反応分布")
print("=" * 78)

print(f"\n  {'タイミング':<6}  {'N':>5}  {'gap':>8}  {'gap std':>8}  {'翌日中':>8}  {'翌日勝率':>9}  {'5d drift':>9}")
print("  " + "-" * 65)
for tm, g in events_liq.groupby('ann_timing'):
    if len(g) < 50:
        continue
    print(f"  {tm:<6}  {len(g):>5,}  "
          f"{g['gap_ret'].mean():>+7.3f}%  {g['gap_ret'].std():>7.2f}%  "
          f"{g['day_after_ret'].mean():>+7.3f}%  "
          f"{(g['react_total']>0).mean()*100:>8.1f}%  "
          f"{g['drift_5d'].mean():>+8.3f}%")


# ════════════════════════════════════════════════════════════════
# ② 発表前 (T日) のイントラ強さは翌日反応を予測するか
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("【②】発表前 (T日) イントラ強さ → 翌日反応の予測力")
print("  AC = 引け後発表のため、T_open→T_close が発表前イントラ")
print("=" * 78)

ac = events_liq[events_liq['ann_timing'] == 'AC'].copy()
print(f"\n  AC発表 N={len(ac):,}")

intra_buckets = [
    ('大陽線 (>+2%)',     ac['t_intraday_ret'] >  2.0),
    ('陽線 (+1〜+2%)',    (ac['t_intraday_ret'] >  1.0) & (ac['t_intraday_ret'] <= 2.0)),
    ('小陽線 (0〜+1%)',   (ac['t_intraday_ret'] >  0)   & (ac['t_intraday_ret'] <= 1.0)),
    ('小陰線 (-1〜0)',    (ac['t_intraday_ret'] >= -1.0)& (ac['t_intraday_ret'] < 0)),
    ('陰線 (-2〜-1%)',    (ac['t_intraday_ret'] >= -2.0)& (ac['t_intraday_ret'] < -1.0)),
    ('大陰線 (<-2%)',     ac['t_intraday_ret'] < -2.0),
]

print(f"\n  {'発表前イントラ':<18}  {'N':>5}  {'ON gap':>8}  {'翌日中':>8}  {'T_close→T+1引':>13}  {'5d drift':>9}")
print("  " + "-" * 75)
for label, mask in intra_buckets:
    sub = ac[mask]
    if len(sub) < 30:
        continue
    print(f"  {label:<18}  {len(sub):>5}  "
          f"{sub['gap_ret'].mean():>+7.3f}%  "
          f"{sub['day_after_ret'].mean():>+7.3f}%  "
          f"{sub['react_total'].mean():>+12.3f}%  "
          f"{sub['drift_5d'].mean():>+8.3f}%")

# 相関係数
corr_intra_gap = ac['t_intraday_ret'].corr(ac['gap_ret'])
corr_intra_react = ac['t_intraday_ret'].corr(ac['react_total'])
corr_intra_drift = ac['t_intraday_ret'].corr(ac['drift_5d'])
print(f"\n  相関係数:")
print(f"    発表前イントラ ↔ ON gap:        {corr_intra_gap:+.4f}")
print(f"    発表前イントラ ↔ T+1総合反応:    {corr_intra_react:+.4f}")
print(f"    発表前イントラ ↔ 5d drift:      {corr_intra_drift:+.4f}")


# ════════════════════════════════════════════════════════════════
# ③ PEAD — ON gap × 5日ドリフトの関係
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("【③】PEAD — 翌日ON gap (反応の大きさ) × その後5日ドリフト")
print("  大きく動いた銘柄は方向継続するか？")
print("=" * 78)

gap_buckets = [
    ('大幅GD (<-5%)',     ac['gap_ret'] < -5.0),
    ('GD (-5〜-2%)',     (ac['gap_ret'] >= -5.0) & (ac['gap_ret'] < -2.0)),
    ('小GD (-2〜-0.5%)', (ac['gap_ret'] >= -2.0) & (ac['gap_ret'] < -0.5)),
    ('フラット (-0.5〜+0.5%)',(ac['gap_ret'] >= -0.5) & (ac['gap_ret'] <= 0.5)),
    ('小GU (+0.5〜+2%)', (ac['gap_ret'] > 0.5) & (ac['gap_ret'] <= 2.0)),
    ('GU (+2〜+5%)',    (ac['gap_ret'] > 2.0) & (ac['gap_ret'] <= 5.0)),
    ('大幅GU (>+5%)',   ac['gap_ret'] > 5.0),
]

print(f"\n  {'翌日ON gap':<22}  {'N':>5}  {'gap mean':>9}  {'翌日中':>8}  {'5d drift':>9}  {'5d勝率':>7}")
print("  " + "-" * 75)
for label, mask in gap_buckets:
    sub = ac[mask]
    if len(sub) < 30:
        continue
    print(f"  {label:<22}  {len(sub):>5}  {sub['gap_ret'].mean():>+8.3f}%  "
          f"{sub['day_after_ret'].mean():>+7.3f}%  "
          f"{sub['drift_5d'].mean():>+8.3f}%  "
          f"{(sub['drift_5d']>0).mean()*100:>6.1f}%")


# ════════════════════════════════════════════════════════════════
# ④ 発表前1週間の出来高動向は予測するか
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("【④】発表前(T日)の出来高 × 反応")
print("=" * 78)

vol_buckets = [
    ('低出来高 (<0.8x)', (ac['t_vol_ratio'] >= 0)   & (ac['t_vol_ratio'] < 0.8)),
    ('普通 (0.8-1.2x)',  (ac['t_vol_ratio'] >= 0.8) & (ac['t_vol_ratio'] < 1.2)),
    ('やや増 (1.2-1.5x)',(ac['t_vol_ratio'] >= 1.2) & (ac['t_vol_ratio'] < 1.5)),
    ('大商い (1.5-2x)',  (ac['t_vol_ratio'] >= 1.5) & (ac['t_vol_ratio'] < 2.0)),
    ('急増 (2x+)',       (ac['t_vol_ratio'] >= 2.0)),
]
print(f"\n  {'発表前出来高':<18}  {'N':>5}  {'発表前イントラ':>12}  {'ON gap':>8}  {'T+1総合':>9}")
print("  " + "-" * 65)
for label, mask in vol_buckets:
    sub = ac[mask & ac['t_vol_ratio'].notna()]
    if len(sub) < 50:
        continue
    print(f"  {label:<18}  {len(sub):>5}  "
          f"{sub['t_intraday_ret'].mean():>+11.3f}%  "
          f"{sub['gap_ret'].mean():>+7.3f}%  "
          f"{sub['react_total'].mean():>+8.3f}%")


# ════════════════════════════════════════════════════════════════
# ⑤ doc_type別 反応
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("【⑤】発表種別 (doc_type) 別 反応")
print("=" * 78)

# 主要typeのみ
main_types = events_liq['doc_type'].value_counts().head(8).index.tolist()
print(f"\n  {'doc_type':<48}  {'N':>5}  {'ON gap':>8}  {'gap std':>8}  {'T+1勝率':>9}")
print("  " + "-" * 90)
for dt in main_types:
    sub = events_liq[(events_liq['doc_type'] == dt) & (events_liq['ann_timing'] == 'AC')]
    if len(sub) < 50:
        continue
    print(f"  {dt:<48}  {len(sub):>5}  "
          f"{sub['gap_ret'].mean():>+7.3f}%  "
          f"{sub['gap_ret'].std():>7.2f}%  "
          f"{(sub['react_total']>0).mean()*100:>8.1f}%")


# ════════════════════════════════════════════════════════════════
# ⑥ セクター別 反応
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("【⑥】セクター別 決算反応 (AC発表のみ)")
print("=" * 78)

sect_res = []
for sect, g in ac.groupby('sector'):
    if len(g) < 100:
        continue
    sect_res.append({
        'sector':       sect,
        'N':            len(g),
        'gap_mean':     g['gap_ret'].mean(),
        'gap_std':      g['gap_ret'].std(),
        'react_mean':   g['react_total'].mean(),
        'react_std':    g['react_total'].std(),
        'drift_mean':   g['drift_5d'].mean(),
        'react_wr':     (g['react_total'] > 0).mean() * 100,
    })

sect_df = pd.DataFrame(sect_res).sort_values('gap_mean', ascending=False)
print(f"\n  {'セクター':<14}  {'N':>5}  {'ON gap':>8}  {'gap std':>8}  {'T+1総合':>9}  {'勝率':>6}  {'5d drift':>9}")
print("  " + "-" * 80)
for _, r in sect_df.iterrows():
    print(f"  {r['sector']:<14}  {int(r['N']):>5}  "
          f"{r['gap_mean']:>+7.3f}%  {r['gap_std']:>7.2f}%  "
          f"{r['react_mean']:>+8.3f}%  {r['react_wr']:>5.1f}%  "
          f"{r['drift_mean']:>+8.3f}%")


# ════════════════════════════════════════════════════════════════
# ⑦ 戦略候補のバックテスト
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("【⑦】戦略候補バックテスト")
print("=" * 78)

# 候補1: 発表前イントラ強い銘柄 → T+1 寄り Long (リバーサル狙い)
# 候補2: 発表前イントラ弱い銘柄 → T+1 寄り Long (オーバーシュート狙い)
# 候補3: 大GU後 → T+1 寄り Short
# 候補4: 大GD後 → T+1 寄り Long (オーバーシュート)
# 候補5: PEAD drift (T+1引け→T+5引け、ギャップ方向に追随)

strategies = [
    # (label, signal, entry_field, exit_field, direction)
    ("発表前+1〜+2%強 → T+1寄り Long (引け決済)",
        (ac['t_intraday_ret'] > 1.0) & (ac['t_intraday_ret'] <= 2.0),
        'tp1_open', 'tp1_close', 'long'),
    ("発表前 大陽線 → T+1寄り Long (引け決済)",
        (ac['t_intraday_ret'] > 2.0),
        'tp1_open', 'tp1_close', 'long'),
    ("発表前 大陰線 → T+1寄り Long (リバ狙い)",
        (ac['t_intraday_ret'] < -2.0),
        'tp1_open', 'tp1_close', 'long'),
    ("大GU後 (>+5%) → T+1寄り Short",
        (ac['gap_ret'] > 5.0),
        'tp1_open', 'tp1_close', 'short'),
    ("大GD後 (<-5%) → T+1寄り Long (リバ狙い)",
        (ac['gap_ret'] < -5.0),
        'tp1_open', 'tp1_close', 'long'),
    ("PEAD Long: GU(+2〜+5%)後 → T+1引け Long、T+5引け決済",
        (ac['gap_ret'] > 2.0) & (ac['gap_ret'] <= 5.0),
        'tp1_close', 'tp5_close', 'long'),
    ("PEAD Short: GD(-5〜-2%)後 → T+1引け Short、T+5引け決済",
        (ac['gap_ret'] < -2.0) & (ac['gap_ret'] >= -5.0),
        'tp1_close', 'tp5_close', 'short'),
    ("大GU(>+5%) PEAD Long → T+5引け",
        (ac['gap_ret'] > 5.0),
        'tp1_close', 'tp5_close', 'long'),
    ("大GD(<-5%) PEAD Long (リバ) → T+5引け",
        (ac['gap_ret'] < -5.0),
        'tp1_close', 'tp5_close', 'long'),
    ("発表前イントラ強(+2%超) + 小GU(+0.5〜+2%) → T+1寄り Long",
        (ac['t_intraday_ret'] > 2.0) & (ac['gap_ret'] > 0.5) & (ac['gap_ret'] <= 2.0),
        'tp1_open', 'tp1_close', 'long'),
]

print(f"\n  {'戦略':<54}  {'N':>4}  {'gross':>7}  {'net':>7}  {'勝率':>6}  {'Sharpe':>7}  {'t-stat':>7}")
print("  " + "-" * 100)
for label, mask, entry_f, exit_f, direction in strategies:
    sub = ac[mask].dropna(subset=[entry_f, exit_f])
    if len(sub) < 20:
        continue
    raw = (sub[exit_f] / sub[entry_f] - 1) * 100
    if direction == 'short':
        raw = -raw
    net = raw - COST_PCT_RT
    sharpe = net.mean() / raw.std() * np.sqrt(252) if raw.std() > 0 else 0
    t_stat = raw.mean() / raw.std() * np.sqrt(len(raw))
    pf = net[net>0].sum() / abs(net[net<=0].sum()) if (net<=0).any() else 99
    flag = " ★★★" if sharpe > 2.5 else (" ★★" if sharpe > 1.5 else (" ★" if sharpe > 1.0 else ""))
    print(f"  {label:<54}  {len(raw):>4}  "
          f"{raw.mean():>+6.3f}%  {net.mean():>+6.3f}%  "
          f"{(net>0).mean()*100:>5.1f}%  {sharpe:>+6.2f}  {t_stat:>+6.2f}{flag}")


# ════════════════════════════════════════════════════════════════
# ⑧ 大型株フィルタ追加（売買代金10億+）
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("【⑧】大型株フィルタ (T日売買代金≥10億) で再検証")
print("=" * 78)

ac_big = ac[ac['t_tv'] >= 1_000_000_000]
print(f"\n  AC × 売買代金10億+: N={len(ac_big):,}")

big_strategies = [
    ("大GU(>+5%) → T+1寄り Short",
        (ac_big['gap_ret'] > 5.0),
        'tp1_open', 'tp1_close', 'short'),
    ("大GD(<-5%) → T+1寄り Long",
        (ac_big['gap_ret'] < -5.0),
        'tp1_open', 'tp1_close', 'long'),
    ("発表前 大陽線(>+2%) → T+1寄り Long",
        (ac_big['t_intraday_ret'] > 2.0),
        'tp1_open', 'tp1_close', 'long'),
    ("PEAD Long: GU(+2〜+5%) → T+5引け",
        (ac_big['gap_ret'] > 2.0) & (ac_big['gap_ret'] <= 5.0),
        'tp1_close', 'tp5_close', 'long'),
    ("PEAD Short: GD(-5〜-2%) → T+5引け",
        (ac_big['gap_ret'] < -2.0) & (ac_big['gap_ret'] >= -5.0),
        'tp1_close', 'tp5_close', 'short'),
]
print(f"\n  {'戦略':<46}  {'N':>4}  {'net':>7}  {'勝率':>6}  {'Sharpe':>7}  {'t-stat':>7}")
print("  " + "-" * 90)
for label, mask, entry_f, exit_f, direction in big_strategies:
    sub = ac_big[mask].dropna(subset=[entry_f, exit_f])
    if len(sub) < 20:
        continue
    raw = (sub[exit_f] / sub[entry_f] - 1) * 100
    if direction == 'short':
        raw = -raw
    net = raw - COST_PCT_RT
    sharpe = net.mean() / raw.std() * np.sqrt(252) if raw.std() > 0 else 0
    t_stat = raw.mean() / raw.std() * np.sqrt(len(raw))
    flag = " ★★★" if sharpe > 2.5 else (" ★★" if sharpe > 1.5 else (" ★" if sharpe > 1.0 else ""))
    print(f"  {label:<46}  {len(raw):>4}  "
          f"{net.mean():>+6.3f}%  {(net>0).mean()*100:>5.1f}%  "
          f"{sharpe:>+6.2f}  {t_stat:>+6.2f}{flag}")


# ════════════════════════════════════════════════════════════════
# ⑨ 最終まとめ
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("【⑨】まとめ — 決算戦略候補ランキング")
print("=" * 78)

# 全戦略を集約
all_strats = strategies + big_strategies
records = []
for label, mask, entry_f, exit_f, direction in strategies:
    sub = ac[mask].dropna(subset=[entry_f, exit_f])
    if len(sub) < 20:
        continue
    raw = (sub[exit_f] / sub[entry_f] - 1) * 100
    if direction == 'short':
        raw = -raw
    net = raw - COST_PCT_RT
    sharpe = net.mean() / raw.std() * np.sqrt(252) if raw.std() > 0 else 0
    records.append({
        'label': label, 'N': len(raw), 'net': net.mean(),
        'wr': (net>0).mean()*100, 'sharpe': sharpe,
        't': raw.mean()/raw.std()*np.sqrt(len(raw)),
    })
for label, mask, entry_f, exit_f, direction in big_strategies:
    sub = ac_big[mask].dropna(subset=[entry_f, exit_f])
    if len(sub) < 20:
        continue
    raw = (sub[exit_f] / sub[entry_f] - 1) * 100
    if direction == 'short':
        raw = -raw
    net = raw - COST_PCT_RT
    sharpe = net.mean() / raw.std() * np.sqrt(252) if raw.std() > 0 else 0
    records.append({
        'label': '[10億+] ' + label, 'N': len(raw), 'net': net.mean(),
        'wr': (net>0).mean()*100, 'sharpe': sharpe,
        't': raw.mean()/raw.std()*np.sqrt(len(raw)),
    })

rk = pd.DataFrame(records).sort_values('sharpe', ascending=False)
print(f"\n  {'順位':>2}  {'戦略':<60}  {'N':>4}  {'net':>7}  {'勝率':>6}  {'Sharpe':>7}")
print("  " + "-" * 100)
for i, (_, r) in enumerate(rk.head(10).iterrows(), 1):
    print(f"  {i:>2}  {r['label']:<60}  {int(r['N']):>4}  "
          f"{r['net']:>+6.3f}%  {r['wr']:>5.1f}%  {r['sharpe']:>+6.2f}")

# CSV出力
out_dir = '/Users/Yusuke/claude-code/japan-stocks/analyses/20260512_earnings_patterns'
sect_df.to_csv(f'{out_dir}/by_sector.csv', index=False)
rk.to_csv(f'{out_dir}/strategy_ranking.csv', index=False)
ac.to_csv(f'{out_dir}/ac_events.csv', index=False)

print(f"\n  ✅ 完了。CSV 3本出力")
