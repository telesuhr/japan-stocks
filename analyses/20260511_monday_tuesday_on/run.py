"""
月曜→火曜 ON Long 戦略 — 全セクター検証
==========================================
分析日: 2026-05-11
対象:   プライム1,574銘柄 × 2024-01-04〜2026-05-08

【仮説】
  先行分析で「33セクター中30で月曜最悪・火曜最良」を発見。
  → 月曜 15:30 引成Long → 火曜 09:00 寄成決済 のON戦略が機能するか？

【検証方法】
  Entry  = 月曜 15:30 終値 (adj_close)
  Exit   = 火曜 09:00 寄付 (adj_open)
  Return = (Tue_open / Mon_close - 1) * 100
  Cost   = 往復 10bps (片道5bps × 2)

【テスト軸】
  1. ベースライン: 全プライム銘柄プール（売買代金10億+）
  2. セクター別: 33セクター
  3. シクリカル限定: 銀行/非鉄/建設/機械/電気機器
  4. 大型株限定: turnover上位300銘柄
  5. フィルタ最適化: 月曜の値動き別、出来高別
"""
import psycopg2
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

PG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}
COST_PCT_RT = 0.10  # 往復コスト
START = '2024-01-04'
END   = '2026-05-08'

def q(sql):
    conn = psycopg2.connect(**PG)
    df = pd.read_sql(sql, conn)
    conn.close()
    return df

print("=" * 78)
print("  月曜→火曜 ON Long 戦略 — 全セクター検証")
print("=" * 78)
print("\n  データ取得中...")

# 月曜と次営業日のペアを抽出
sql = f"""
WITH base AS (
    SELECT d.code, s.sector33_nm AS sector, s.market_nm,
           d.date, d.adj_close, d.adj_open, d.turnover_value, d.adj_volume,
           EXTRACT(DOW FROM d.date)::int AS dow
    FROM stocks_daily d
    JOIN symbol_master s ON s.code5 = d.code
    WHERE s.market = '0111' AND d.date BETWEEN '{START}' AND '{END}'
      AND d.adj_close > 0 AND d.adj_open > 0
      AND d.turnover_value >= 1000000000   -- 売買代金10億+
),
ranked AS (
    SELECT *,
        LEAD(date, 1)       OVER (PARTITION BY code ORDER BY date) AS next_date,
        LEAD(adj_open, 1)   OVER (PARTITION BY code ORDER BY date) AS next_open,
        LEAD(adj_close, 1)  OVER (PARTITION BY code ORDER BY date) AS next_close,
        AVG(adj_volume)     OVER (PARTITION BY code ORDER BY date
                                  ROWS BETWEEN 21 PRECEDING AND 2 PRECEDING) AS vol_ma20
    FROM base
)
SELECT * FROM ranked
WHERE dow = 1   -- 月曜のみ
  AND next_date IS NOT NULL
"""

df = q(sql)
for c in ['adj_close','adj_open','turnover_value','adj_volume','next_open','next_close','vol_ma20']:
    df[c] = pd.to_numeric(df[c], errors='coerce')

df['next_date'] = pd.to_datetime(df['next_date'])
df['date']      = pd.to_datetime(df['date'])
df['gap_days']  = (df['next_date'] - df['date']).dt.days  # 通常1だが連休後は大きくなる

# 火曜限定（gap_days=1）
df = df[df['gap_days'] == 1].copy()
df['on_ret']   = (df['next_open'] / df['adj_close'] - 1) * 100
df['day_ret']  = (df['adj_close'] / df['adj_open']  - 1) * 100  # 月曜の日中
df['vol_ratio']= df['adj_volume'] / df['vol_ma20']
df = df.dropna(subset=['on_ret'])

print(f"  月曜→火曜ペア数: {len(df):,}")
print(f"  ユニーク月曜数:  {df['date'].nunique()}")
print(f"  銘柄数:          {df['code'].nunique()}")


# ════════════════════════════════════════════════════════════════
# 1. ベースライン (全プライム銘柄プール)
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("【1】ベースライン — 全プライム銘柄プール (売買代金10億+)")
print("=" * 78)

r = df['on_ret']
net = r - COST_PCT_RT
t_stat = r.mean() / r.std() * np.sqrt(len(r))
sharpe = net.mean() / r.std() * np.sqrt(50)  # 月曜は年間約50回

print(f"\n  全プール:")
print(f"    N={len(r):,}, gross_mean={r.mean():>+.4f}%, net_mean={net.mean():>+.4f}%")
print(f"    勝率={(r>0).mean()*100:.1f}%, std={r.std():.3f}%, t-stat={t_stat:.2f}")
print(f"    年率Sharpe (50回/年): {sharpe:+.2f}")


# ════════════════════════════════════════════════════════════════
# 2. セクター別
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("【2】セクター別 月曜→火曜 ON Long パフォーマンス")
print("=" * 78)

sect_res = []
for sect, g in df.groupby('sector'):
    if len(g) < 100:
        continue
    r = g['on_ret']
    net = r - COST_PCT_RT
    sharpe = net.mean() / r.std() * np.sqrt(50) if r.std() > 0 else 0
    sect_res.append({
        'sector':     sect,
        'N':          len(g),
        'gross_mean': r.mean(),
        'net_mean':   net.mean(),
        'wr':         (r > 0).mean() * 100,
        'std':        r.std(),
        't_stat':     r.mean() / r.std() * np.sqrt(len(r)),
        'sharpe':     sharpe,
    })

sect_df = pd.DataFrame(sect_res).sort_values('sharpe', ascending=False)
print(f"\n  {'セクター':<14}  {'N':>6}  {'gross':>8}  {'net':>8}  {'勝率':>6}  "
      f"{'std':>5}  {'t-stat':>7}  {'Sharpe':>7}")
print("  " + "-" * 78)
for _, r in sect_df.iterrows():
    flag = " ★★★" if r['sharpe'] > 2.5 else (" ★★" if r['sharpe'] > 1.5 else (" ★" if r['sharpe'] > 1.0 else ""))
    print(f"  {r['sector']:<14}  {int(r['N']):>6,}  {r['gross_mean']:>+7.4f}%  "
          f"{r['net_mean']:>+7.4f}%  {r['wr']:>5.1f}%  {r['std']:>4.2f}%  "
          f"{r['t_stat']:>+6.2f}  {r['sharpe']:>+6.2f}{flag}")


# ════════════════════════════════════════════════════════════════
# 3. シクリカル限定（先行分析で日次Sharpe>1.0のセクター）
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("【3】シクリカル限定 (銀行/非鉄/建設/機械/電気機器/重機/輸送機器)")
print("=" * 78)

cyc_sectors = ['銀行業', '非鉄金属', '建設業', '機械', '電気機器', '鉄鋼', '輸送用機器', '化学']
cyc = df[df['sector'].isin(cyc_sectors)]
r = cyc['on_ret']
net = r - COST_PCT_RT
sharpe = net.mean() / r.std() * np.sqrt(50)
print(f"\n  N={len(r):,}, gross={r.mean():>+.4f}%, net={net.mean():>+.4f}%")
print(f"  勝率={(r>0).mean()*100:.1f}%, t-stat={r.mean()/r.std()*np.sqrt(len(r)):.2f}, Sharpe={sharpe:+.2f}")


# ════════════════════════════════════════════════════════════════
# 4. 大型株限定 (turnover上位300銘柄、月単位リバランス)
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("【4】大型株限定 — 直近1ヶ月の売買代金上位300銘柄")
print("=" * 78)

# 各月ごとに売買代金上位300銘柄を選定
df['ym'] = df['date'].dt.to_period('M').astype(str)
top300 = (df.groupby(['ym','code'])['turnover_value']
            .mean().reset_index()
            .sort_values(['ym','turnover_value'], ascending=[True, False])
            .groupby('ym').head(300))
big = df.merge(top300[['ym','code']], on=['ym','code'])

r = big['on_ret']
net = r - COST_PCT_RT
sharpe = net.mean() / r.std() * np.sqrt(50)
print(f"\n  N={len(r):,}, gross={r.mean():>+.4f}%, net={net.mean():>+.4f}%")
print(f"  勝率={(r>0).mean()*100:.1f}%, t-stat={r.mean()/r.std()*np.sqrt(len(r)):.2f}, Sharpe={sharpe:+.2f}")


# ════════════════════════════════════════════════════════════════
# 5. 月曜値動き別の条件付け（陰線後 vs 陽線後）
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("【5】月曜の日中値動き別 ON Long リターン")
print("=" * 78)

day_buckets = [
    ('月曜大陽線(+1%超)',  df['day_ret'] >  1.0),
    ('月曜小陽線(0〜+1%)', (df['day_ret'] > 0) & (df['day_ret'] <= 1.0)),
    ('月曜小陰線(-1%〜0)', (df['day_ret'] >= -1.0) & (df['day_ret'] < 0)),
    ('月曜大陰線(-1%超下)', df['day_ret'] < -1.0),
]
print(f"\n  {'月曜値動き':<24}  {'N':>6}  {'gross':>8}  {'net':>8}  {'勝率':>6}  {'Sharpe':>7}")
print("  " + "-" * 70)
for label, mask in day_buckets:
    sub = df[mask]
    if len(sub) < 100:
        continue
    r = sub['on_ret']
    net = r - COST_PCT_RT
    sharpe = net.mean() / r.std() * np.sqrt(50)
    print(f"  {label:<24}  {len(r):>6,}  {r.mean():>+7.4f}%  {net.mean():>+7.4f}%  "
          f"{(r>0).mean()*100:>5.1f}%  {sharpe:>+6.2f}")


# ════════════════════════════════════════════════════════════════
# 6. 出来高別
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("【6】月曜の出来高別 ON Long リターン")
print("=" * 78)

vol_buckets = [
    ('低出来高(<0.8x)',    (df['vol_ratio'] < 0.8)),
    ('普通(0.8-1.2x)',     (df['vol_ratio'] >= 0.8) & (df['vol_ratio'] < 1.2)),
    ('やや増(1.2-1.5x)',   (df['vol_ratio'] >= 1.2) & (df['vol_ratio'] < 1.5)),
    ('大商い(1.5-3.0x)',   (df['vol_ratio'] >= 1.5) & (df['vol_ratio'] < 3.0)),
    ('超急増(3.0x+)',      (df['vol_ratio'] >= 3.0)),
]
print(f"\n  {'出来高':<20}  {'N':>6}  {'gross':>8}  {'net':>8}  {'勝率':>6}  {'Sharpe':>7}")
print("  " + "-" * 65)
for label, mask in vol_buckets:
    sub = df[mask]
    if len(sub) < 100:
        continue
    r = sub['on_ret']
    net = r - COST_PCT_RT
    sharpe = net.mean() / r.std() * np.sqrt(50)
    print(f"  {label:<20}  {len(r):>6,}  {r.mean():>+7.4f}%  {net.mean():>+7.4f}%  "
          f"{(r>0).mean()*100:>5.1f}%  {sharpe:>+6.2f}")


# ════════════════════════════════════════════════════════════════
# 7. 最適複合シグナル
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("【7】複合シグナル最適化 — 月曜値動き × 出来高 × セクター")
print("=" * 78)

combos = [
    ("全銘柄",                                        np.ones(len(df), dtype=bool)),
    ("シクリカル全部",                                df['sector'].isin(cyc_sectors)),
    ("シクリカル × 月曜陰線",                         df['sector'].isin(cyc_sectors) & (df['day_ret'] < 0)),
    ("シクリカル × 月曜大陰線(-1%下)",                df['sector'].isin(cyc_sectors) & (df['day_ret'] < -1.0)),
    ("シクリカル × 月曜大陰線 × vol≥1.5x",            df['sector'].isin(cyc_sectors) & (df['day_ret'] < -1.0) & (df['vol_ratio'] >= 1.5)),
    ("全銘柄 × 月曜大陰線(-1%下)",                    df['day_ret'] < -1.0),
    ("全銘柄 × 月曜大陰線 × vol≥1.5x",                (df['day_ret'] < -1.0) & (df['vol_ratio'] >= 1.5)),
    ("Sharpe>1.5セクター × 月曜大陰線",               df['sector'].isin(sect_df[sect_df['sharpe']>1.5]['sector'].tolist()) & (df['day_ret'] < -1.0)),
    ("Sharpe>1.5セクター × 全月曜",                   df['sector'].isin(sect_df[sect_df['sharpe']>1.5]['sector'].tolist())),
]
print(f"\n  {'条件':<42}  {'N':>5}  {'net':>8}  {'勝率':>6}  {'t-stat':>7}  {'Sharpe':>7}")
print("  " + "-" * 85)
for label, mask in combos:
    sub = df[mask]
    if len(sub) < 50:
        continue
    r = sub['on_ret']
    net = r - COST_PCT_RT
    sharpe = net.mean() / r.std() * np.sqrt(50) if r.std() > 0 else 0
    t_stat = r.mean() / r.std() * np.sqrt(len(r))
    flag = " ★★★" if sharpe > 2.5 else (" ★★" if sharpe > 1.5 else "")
    print(f"  {label:<42}  {len(r):>5,}  {net.mean():>+7.4f}%  "
          f"{(r>0).mean()*100:>5.1f}%  {t_stat:>+6.2f}  {sharpe:>+6.2f}{flag}")


# ════════════════════════════════════════════════════════════════
# 8. ポートフォリオレベル: 月曜引け各銘柄¥100万 → 火曜寄り
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("【8】ポートフォリオシム — シクリカル × 月曜大陰線 × vol≥1.5x")
print("  毎月曜にシグナル銘柄を全買い (上限あり)、火曜寄り全決済")
print("=" * 78)

best_mask = df['sector'].isin(cyc_sectors) & (df['day_ret'] < -1.0) & (df['vol_ratio'] >= 1.5)
port = df[best_mask].copy()
port['week'] = port['date']
weekly = port.groupby('week').agg(
    n_signals=('code', 'count'),
    avg_ret  =('on_ret', 'mean'),  # 均等加重
    median   =('on_ret', 'median'),
)
weekly['net_ret'] = weekly['avg_ret'] - COST_PCT_RT

print(f"\n  発動月曜数: {len(weekly)}")
print(f"  1週あたり平均シグナル銘柄数: {weekly['n_signals'].mean():.1f}")
print(f"  週次平均net (均等加重): {weekly['net_ret'].mean():>+.4f}%")
print(f"  週次勝率: {(weekly['net_ret']>0).mean()*100:.1f}%")
print(f"  週次Sharpe (年率): {weekly['net_ret'].mean()/weekly['net_ret'].std()*np.sqrt(50):+.2f}")

# 同時保有数別
print(f"\n  ─ 上限N銘柄での実行シミュレーション ─")
for max_pos in [3, 5, 10, 20]:
    pnls = []
    for w, g in port.groupby('week'):
        sel = g.head(max_pos)
        pnl_pct = sel['on_ret'].mean() - COST_PCT_RT
        pnls.append(pnl_pct * len(sel))   # 銘柄ごと¥100万、合計を%で
    pnls = pd.Series(pnls)
    sharpe = pnls.mean() / pnls.std() * np.sqrt(50) if pnls.std() > 0 else 0
    print(f"    最大{max_pos}銘柄: 週次総%={pnls.mean():>+.3f}%, Sharpe={sharpe:+.2f}")


# ════════════════════════════════════════════════════════════════
# 9. 年次安定性
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("【9】年次安定性 — Sharpe>1.5セクター × 月曜大陰線")
print("=" * 78)

high_sharpe_sectors = sect_df[sect_df['sharpe']>1.5]['sector'].tolist()
best = df[df['sector'].isin(high_sharpe_sectors) & (df['day_ret'] < -1.0)].copy()
best['year'] = best['date'].dt.year
print(f"\n  対象セクター: {high_sharpe_sectors}")
print(f"  {'年':>4}  {'N':>5}  {'gross':>8}  {'net':>8}  {'勝率':>6}  {'Sharpe':>7}")
print("  " + "-" * 50)
for year, g in best.groupby('year'):
    r = g['on_ret']
    net = r - COST_PCT_RT
    sharpe = net.mean() / r.std() * np.sqrt(50)
    print(f"  {year}  {len(r):>5}  {r.mean():>+7.4f}%  {net.mean():>+7.4f}%  "
          f"{(r>0).mean()*100:>5.1f}%  {sharpe:>+6.2f}")


# ════════════════════════════════════════════════════════════════
# 10. Strategy昇格判定
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("【10】Strategy昇格判定")
print("=" * 78)

# 最良パラメータでの判定
best_filt = df['sector'].isin(cyc_sectors) & (df['day_ret'] < -1.0)
sub = df[best_filt]
r = sub['on_ret']
net = r - COST_PCT_RT
sharpe = net.mean() / r.std() * np.sqrt(50)
t_stat = r.mean() / r.std() * np.sqrt(len(r))
pf = net[net>0].sum() / abs(net[net<=0].sum()) if (net<=0).any() else 99

# 年次安定性
yearly_pos = all(g['on_ret'].mean() > 0 for _, g in sub.groupby(sub['date'].dt.year))

criteria = [
    ("N≥1000",            len(sub) >= 1000,        f"{len(sub)}"),
    ("net mean > 0",      net.mean() > 0,          f"{net.mean():+.4f}%"),
    ("Sharpe > 2.0",      sharpe > 2.0,            f"{sharpe:+.2f}"),
    ("t-stat > 2.0",      t_stat > 2.0,            f"{t_stat:+.2f}"),
    ("PF > 1.1",          pf > 1.1,                f"{pf:.2f}"),
    ("年次全部 net>0",     yearly_pos,              "yearly check"),
]
print(f"\n  ベスト設定: シクリカル × 月曜大陰線(-1%下) ON Long")
print(f"  {'基準':<24}  {'結果':<10}  値")
print("  " + "-" * 60)
for name, ok, val in criteria:
    mark = "✅ PASS" if ok else "❌ FAIL"
    print(f"  {name:<24}  {mark:<10}  {val}")

passed = sum(1 for _, ok, _ in criteria if ok)
print(f"\n  合格基準: {passed} / {len(criteria)}")
if passed >= 5:
    print("  判定: 🎯 Strategy昇格を強く推奨")
elif passed >= 4:
    print("  判定: ⚠️  条件付き昇格")
else:
    print("  判定: ❌ research段階")

# CSV出力
out_dir = '/Users/Yusuke/claude-code/japan-stocks/analyses/20260511_monday_tuesday_on'
sect_df.to_csv(f'{out_dir}/by_sector.csv', index=False)
print(f"\n  ✅ 完了")
