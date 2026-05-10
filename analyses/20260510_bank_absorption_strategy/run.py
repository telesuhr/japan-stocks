"""
銀行株 出来高吸収逆張り戦略 — 実SL/TPバックテスト
=================================================
分析日: 2026-05-10
対象:   プライム銀行業 69銘柄 × 2024-05-10〜2026-05-08

【戦略仮説】
  「大商い × 価格下落」の翌営業日寄付きで買い、5日以内にSL/TPに引っかからなければ
  5日後の引けで決済する。 機関投資家による売り吸収後のリバを狙う。

【今回新規】
  - 日中の高値・安値を使った実SL/TPシミュレーション
    （前回のclip()ベース楽観バイアスを排除）
  - SL/TPが同日ヒットした場合は **SL優先（保守的）**
  - Walk-forward: 2024 → 2025 → 2026 で性能の頑健性を検証
  - ポートフォリオレベル（日次最大ポジション数制約）でのEquity curve
  - 現実的コスト（5bps片道、計10bps）

【検証パラメータ】
  - Entry condition: vol_ratio≥1.5 × day_ret<0 × turnover≥10億
  - SL grid: -1.5%, -2%, -3%
  - TP grid: なし, +3%, +5%, +7%
  - 保有期間: 5営業日
"""

import psycopg2
import pandas as pd
import numpy as np
from datetime import date, timedelta
import warnings
warnings.filterwarnings('ignore')

PG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}
COST_PCT_RT = 0.10   # 往復コスト (5bps × 2)
HOLD_DAYS   = 5
SECTOR      = '銀行業'
START_DATE  = '2024-05-10'   # 1分足が利用可能な開始日

def q(sql, params=None):
    conn = psycopg2.connect(**PG)
    df = pd.read_sql(sql, conn, params=params)
    conn.close()
    return df


# ════════════════════════════════════════════════════════════════
# Step 1. シグナル抽出
# ════════════════════════════════════════════════════════════════
print("=" * 78)
print("  銀行株 出来高吸収逆張り戦略 — 実SL/TPバックテスト")
print("=" * 78)
print(f"\n  Step 1: シグナル抽出（{SECTOR}・売買代金10億+・vol≥1.5x×価格↓）")

sig_sql = f"""
WITH base AS (
    SELECT d.code, s.name_ja, d.date,
           d.adj_open, d.adj_close, d.adj_high, d.adj_low, d.adj_volume,
           d.turnover_value
    FROM stocks_daily d
    JOIN symbol_master s ON s.code5 = d.code
    WHERE s.market = '0111' AND s.sector33_nm = '{SECTOR}'
      AND d.date >= '{START_DATE}'::date
      AND d.adj_close > 0 AND d.adj_open > 0 AND d.adj_volume > 0
),
ind AS (
    SELECT *,
        AVG(adj_volume) OVER (
            PARTITION BY code ORDER BY date
            ROWS BETWEEN 21 PRECEDING AND 2 PRECEDING
        ) AS vol_ma20,
        (adj_close/adj_open - 1)*100 AS day_ret,
        LEAD(adj_open, 1) OVER (PARTITION BY code ORDER BY date) AS next_open
    FROM base
)
SELECT code, name_ja, date, adj_close, adj_volume, vol_ma20, day_ret,
       turnover_value, next_open
FROM ind
WHERE vol_ma20 IS NOT NULL AND vol_ma20 > 0
  AND adj_volume / vol_ma20 >= 1.5
  AND day_ret < 0
  AND turnover_value >= 1000000000
  AND next_open IS NOT NULL
ORDER BY date, code
"""

signals = q(sig_sql)
for c in ['adj_close','adj_volume','vol_ma20','day_ret','turnover_value','next_open']:
    signals[c] = pd.to_numeric(signals[c], errors='coerce')
signals['vol_ratio'] = signals['adj_volume'] / signals['vol_ma20']
signals['date'] = pd.to_datetime(signals['date']).dt.date

print(f"  抽出シグナル数: {len(signals):,}")
print(f"  期間: {signals['date'].min()} 〜 {signals['date'].max()}")
print(f"  銘柄数: {signals['code'].nunique()}")


# ════════════════════════════════════════════════════════════════
# Step 2. 各シグナルの保有期間中の高値・安値を取得
# ════════════════════════════════════════════════════════════════
print(f"\n  Step 2: 保有期間中の日次高値・安値ロード（{HOLD_DAYS}日間）")

# 各シグナル日の翌営業日からN営業日分の (high/low/close) を取得
hl_sql = f"""
WITH ranked AS (
    SELECT d.code, d.date, d.adj_open, d.adj_close, d.adj_high, d.adj_low,
           ROW_NUMBER() OVER (PARTITION BY d.code ORDER BY d.date) AS day_idx
    FROM stocks_daily d
    JOIN symbol_master s ON s.code5 = d.code
    WHERE s.market='0111' AND s.sector33_nm='{SECTOR}'
      AND d.date >= '{START_DATE}'::date
      AND d.adj_close > 0
)
SELECT code, date, adj_open, adj_close, adj_high, adj_low, day_idx
FROM ranked
ORDER BY code, date
"""
hl_df = q(hl_sql)
for c in ['adj_open','adj_close','adj_high','adj_low']:
    hl_df[c] = pd.to_numeric(hl_df[c], errors='coerce')
hl_df['date'] = pd.to_datetime(hl_df['date']).dt.date
hl_df = hl_df.set_index(['code','date']).sort_index()

# 各銘柄の日付列
code_dates = {code: sorted(g.index.get_level_values(1))
              for code, g in hl_df.groupby(level=0)}

print(f"  日次OHLC: {len(hl_df):,}行")


# ════════════════════════════════════════════════════════════════
# Step 3. 実SL/TPバックテストエンジン
# ════════════════════════════════════════════════════════════════
def simulate_trade(code, sig_date, entry_price, sl_pct, tp_pct, hold_days):
    """
    エントリー日の翌営業日始値で買い、保有期間中に SL/TP がヒットしたら
    その価格で決済。同日ヒット時は SL を優先（保守的）。

    Returns:
        ret_pct: グロスリターン（%）
        exit_reason: 'SL' / 'TP' / 'TIME' / 'NA'
        exit_day: 何日目で決済したか（1=翌日, 5=5日目）
    """
    if code not in code_dates or pd.isna(entry_price):
        return None, 'NA', 0

    dates = code_dates[code]
    # シグナル日の翌営業日からN日
    try:
        sig_idx = dates.index(sig_date)
    except ValueError:
        return None, 'NA', 0

    if sig_idx + hold_days >= len(dates):
        return None, 'NA', 0

    sl_price = entry_price * (1 + sl_pct / 100) if sl_pct else None
    tp_price = entry_price * (1 + tp_pct / 100) if tp_pct else None

    for k in range(1, hold_days + 1):
        d = dates[sig_idx + k]
        try:
            row = hl_df.loc[(code, d)]
        except KeyError:
            continue
        hi = row['adj_high']
        lo = row['adj_low']
        cl = row['adj_close']

        sl_hit = sl_price and lo <= sl_price
        tp_hit = tp_price and hi >= tp_price

        if sl_hit and tp_hit:
            # 同日ヒット → SL優先（保守的）
            return (sl_price / entry_price - 1) * 100, 'SL', k
        elif sl_hit:
            return (sl_price / entry_price - 1) * 100, 'SL', k
        elif tp_hit:
            return (tp_price / entry_price - 1) * 100, 'TP', k

        if k == hold_days:
            return (cl / entry_price - 1) * 100, 'TIME', k

    return None, 'NA', 0


def run_backtest(signals_df, sl_pct, tp_pct):
    """全シグナルに対してバックテストを実行"""
    trades = []
    for _, sig in signals_df.iterrows():
        ret, reason, exit_day = simulate_trade(
            sig['code'], sig['date'], sig['next_open'],
            sl_pct, tp_pct, HOLD_DAYS
        )
        if ret is None:
            continue
        trades.append({
            'code':       sig['code'],
            'name':       sig['name_ja'],
            'sig_date':   sig['date'],
            'entry':      sig['next_open'],
            'gross_ret':  ret,
            'net_ret':    ret - COST_PCT_RT,
            'reason':     reason,
            'exit_day':   exit_day,
        })
    return pd.DataFrame(trades)


# ════════════════════════════════════════════════════════════════
# Step 4. SL/TPグリッドサーチ
# ════════════════════════════════════════════════════════════════
print(f"\n  Step 3: SL/TPグリッドサーチ（{len(signals):,}シグナル × 12組）")

grid_results = []
for sl_pct in [-1.5, -2.0, -3.0, None]:
    for tp_pct in [None, 3.0, 5.0, 7.0]:
        if sl_pct is None and tp_pct is None:
            label = "SL=なし TP=なし（5日保有）"
        else:
            label = f"SL={sl_pct}% TP={tp_pct or 'なし'}"
        trades = run_backtest(signals, sl_pct, tp_pct)
        if len(trades) == 0:
            continue
        net = trades['net_ret']
        gross = trades['gross_ret']
        sharpe = net.mean() / net.std() * np.sqrt(252 / HOLD_DAYS) if net.std() > 0 else 0
        win = (net > 0).sum()
        loss = (net <= 0).sum()
        pf = (net[net > 0].sum() / abs(net[net <= 0].sum())) if net[net <= 0].sum() != 0 else 99
        grid_results.append({
            'sl': sl_pct, 'tp': tp_pct, 'label': label,
            'N':         len(trades),
            'gross_avg': gross.mean(),
            'net_avg':   net.mean(),
            'wr':        win / len(net) * 100,
            'std':       net.std(),
            'sharpe':    sharpe,
            'pf':        pf,
            'sl_pct':    (trades['reason'] == 'SL').mean() * 100,
            'tp_pct':    (trades['reason'] == 'TP').mean() * 100,
            'time_pct':  (trades['reason'] == 'TIME').mean() * 100,
        })

grid_df = pd.DataFrame(grid_results).sort_values('sharpe', ascending=False)

print("\n" + "=" * 78)
print("【1】SL/TPグリッドサーチ結果")
print("=" * 78)
print(f"\n  {'パラメータ':<28} {'N':>5}  {'gross':>7}  {'net':>7}  {'勝率':>6}  {'Sharpe':>7}  {'PF':>5}  {'SL率':>5}  {'TP率':>5}  {'時間':>5}")
print("  " + "-" * 100)
for _, r in grid_df.iterrows():
    print(f"  {r['label']:<28} {r['N']:>5}  "
          f"{r['gross_avg']:>+6.3f}%  {r['net_avg']:>+6.3f}%  "
          f"{r['wr']:>5.1f}%  {r['sharpe']:>+6.2f}  "
          f"{r['pf']:>4.2f}  {r['sl_pct']:>4.1f}%  {r['tp_pct']:>4.1f}%  {r['time_pct']:>4.1f}%")


# ════════════════════════════════════════════════════════════════
# Step 5. ベスト組み合わせの詳細分析
# ════════════════════════════════════════════════════════════════
best_row = grid_df.iloc[0]
best_sl = best_row['sl']
best_tp = best_row['tp']

print(f"\n" + "=" * 78)
print(f"【2】ベスト組み合わせの詳細: {best_row['label']}")
print("=" * 78)

best_trades = run_backtest(signals, best_sl, best_tp)
best_trades['sig_date'] = pd.to_datetime(best_trades['sig_date'])
best_trades['year'] = best_trades['sig_date'].dt.year

# 年次性能
print(f"\n  ─ 年次パフォーマンス ─")
print(f"  {'年':>4}  {'N':>5}  {'gross':>7}  {'net':>7}  {'勝率':>6}  {'Sharpe':>7}  {'PF':>5}  {'最大利益':>9}  {'最大損失':>9}")
print("  " + "-" * 80)
for year, g in best_trades.groupby('year'):
    net = g['net_ret']
    sharpe = net.mean() / net.std() * np.sqrt(252/HOLD_DAYS) if net.std() > 0 else 0
    pf = net[net>0].sum() / abs(net[net<=0].sum()) if net[net<=0].sum() != 0 else 99
    print(f"  {year}  {len(net):>5}  {g['gross_ret'].mean():>+6.3f}%  {net.mean():>+6.3f}%  "
          f"{(net>0).mean()*100:>5.1f}%  {sharpe:>+6.2f}  {pf:>4.2f}  "
          f"{net.max():>+8.2f}%  {net.min():>+8.2f}%")

# Walk-forward (2024訓練 → 2025-2026テスト)
print(f"\n  ─ Walk-forward検証 ─")
train = best_trades[best_trades['year'] == 2024]
test  = best_trades[best_trades['year'] >= 2025]
for label, g in [('訓練 2024', train), ('テスト 2025-2026', test)]:
    if len(g) == 0:
        continue
    net = g['net_ret']
    sharpe = net.mean() / net.std() * np.sqrt(252/HOLD_DAYS) if net.std() > 0 else 0
    print(f"  {label:>16}: N={len(net):>4}  net={net.mean():>+.3f}%  "
          f"勝率={(net>0).mean()*100:.1f}%  Sharpe={sharpe:+.2f}")


# ════════════════════════════════════════════════════════════════
# Step 6. 銘柄別の効き方
# ════════════════════════════════════════════════════════════════
print(f"\n" + "=" * 78)
print("【3】銘柄別パフォーマンス（N≥10のみ）")
print("=" * 78)
print(f"\n  {'銘柄':<24}  {'N':>4}  {'net avg':>8}  {'勝率':>6}  {'Sharpe':>7}")
print("  " + "-" * 60)
sym_results = []
for code, g in best_trades.groupby('code'):
    if len(g) < 10:
        continue
    net = g['net_ret']
    sharpe = net.mean()/net.std() * np.sqrt(252/HOLD_DAYS) if net.std() > 0 else 0
    sym_results.append({
        'code': code, 'name': g['name'].iloc[0],
        'N': len(net), 'net_avg': net.mean(),
        'wr': (net>0).mean()*100, 'sharpe': sharpe,
    })
sym_df = pd.DataFrame(sym_results).sort_values('sharpe', ascending=False)
for _, r in sym_df.iterrows():
    flag = " ★" if r['sharpe'] > 1.0 else (" ◎" if r['sharpe'] > 0.5 else "")
    print(f"  {r['name']:<24}  {r['N']:>4}  {r['net_avg']:>+7.3f}%  "
          f"{r['wr']:>5.1f}%  {r['sharpe']:>+6.2f}{flag}")


# ════════════════════════════════════════════════════════════════
# Step 7. ポートフォリオシミュレーション（日次最大Nポジション制約）
# ════════════════════════════════════════════════════════════════
print(f"\n" + "=" * 78)
print("【4】ポートフォリオシミュレーション")
print(f"  ベスト設定 ({best_row['label']}) で 1ポジ100万円、最大同時保有数を変えて検証")
print("=" * 78)

best_trades_sorted = best_trades.sort_values('sig_date').reset_index(drop=True)
best_trades_sorted['exit_date'] = best_trades_sorted.apply(
    lambda r: r['sig_date'] + pd.tseries.offsets.BDay(int(r['exit_day'])), axis=1
)

POSITION_SIZE = 1_000_000

for max_pos in [3, 5, 10, 20]:
    open_positions = []
    pnl_list = []
    skipped = 0
    for _, t in best_trades_sorted.iterrows():
        # 期限切れポジション削除
        open_positions = [p for p in open_positions if p > t['sig_date']]
        if len(open_positions) >= max_pos:
            skipped += 1
            continue
        open_positions.append(t['exit_date'])
        pnl = POSITION_SIZE * t['net_ret'] / 100
        pnl_list.append({'date': t['sig_date'], 'pnl': pnl, 'ret': t['net_ret']})

    if not pnl_list:
        continue
    p = pd.DataFrame(pnl_list).set_index('date')
    p['cumulative'] = p['pnl'].cumsum()
    daily = p['pnl'].resample('B').sum().fillna(0)
    sharpe_d = daily.mean()/daily.std()*np.sqrt(252) if daily.std() > 0 else 0
    max_dd = (p['cumulative'] - p['cumulative'].cummax()).min()

    print(f"\n  最大同時保有 {max_pos}銘柄:")
    print(f"    執行 {len(p)}件 / スキップ {skipped}件")
    print(f"    総P&L: {p['pnl'].sum():>+12,.0f}円")
    print(f"    平均1トレード: {p['pnl'].mean():>+10,.0f}円")
    print(f"    最大DD: {max_dd:>+12,.0f}円")
    print(f"    日次Sharpe: {sharpe_d:>+.2f}")


# ════════════════════════════════════════════════════════════════
# Step 8. ベンチマーク比較（同期間TOPIX買い持ち）
# ════════════════════════════════════════════════════════════════
print(f"\n" + "=" * 78)
print("【5】ベンチマーク比較")
print("=" * 78)

topix = q(f"""
    SELECT date, close FROM index_daily
    WHERE code='0000' AND date BETWEEN '{START_DATE}' AND '2026-05-08'
    ORDER BY date
""")
topix['close'] = pd.to_numeric(topix['close'])
topix_ret = topix['close'].iloc[-1] / topix['close'].iloc[0] - 1
days = (topix['date'].iloc[-1] - topix['date'].iloc[0]).days
topix_ann = (1 + topix_ret) ** (365 / days) - 1

# 戦略の総リターン（最大10ポジ想定）
total_pnl = sum([t['net_ret'] for _, t in best_trades.iterrows()])
print(f"\n  期間: {topix['date'].iloc[0]} 〜 {topix['date'].iloc[-1]} ({days}日)")
print(f"  TOPIX買い持ち: {topix_ret*100:>+.2f}%  年率 {topix_ann*100:>+.2f}%")
print(f"  戦略総P&L (1トレード比例): {total_pnl:>+.2f}%")
print(f"  戦略 best年次 Sharpe: {best_row['sharpe']:>+.2f}")


# ════════════════════════════════════════════════════════════════
# Step 9. 総合判定
# ════════════════════════════════════════════════════════════════
print(f"\n" + "=" * 78)
print("【6】Strategy昇格判定")
print("=" * 78)

# 判定基準
test_sharpe = test['net_ret'].mean()/test['net_ret'].std()*np.sqrt(252/HOLD_DAYS) if len(test)>10 and test['net_ret'].std()>0 else 0
train_sharpe = train['net_ret'].mean()/train['net_ret'].std()*np.sqrt(252/HOLD_DAYS) if len(train)>10 and train['net_ret'].std()>0 else 0

criteria = [
    ("N≥1000",                    len(best_trades) >= 1000,            f"{len(best_trades)}"),
    ("net mean > 0",              best_trades['net_ret'].mean() > 0,   f"{best_trades['net_ret'].mean():+.3f}%"),
    ("Sharpe > 1.0",              best_row['sharpe'] > 1.0,            f"{best_row['sharpe']:+.2f}"),
    ("PF > 1.3",                  best_row['pf'] > 1.3,                f"{best_row['pf']:.2f}"),
    ("Walk-forward (test>0.5×train)",
       (test_sharpe > 0.5 * train_sharpe) if train_sharpe > 0 else False,
       f"train {train_sharpe:+.2f} / test {test_sharpe:+.2f}"),
    ("年次安定性 (全年 net>0)",
       all(g['net_ret'].mean() > 0 for _, g in best_trades.groupby('year')),
       "yearly check"),
]

print(f"\n  {'基準':<32}  {'結果':<10}  値")
print("  " + "-" * 60)
for name, ok, val in criteria:
    mark = "✅ PASS" if ok else "❌ FAIL"
    print(f"  {name:<32}  {mark:<10}  {val}")

passed = sum(1 for _, ok, _ in criteria if ok)
print(f"\n  合格基準: {passed} / {len(criteria)}")

if passed >= 5:
    verdict = "🎯 Strategy昇格を強く推奨"
elif passed >= 4:
    verdict = "⚠️  Strategy昇格は条件付き（追加検証が望ましい）"
else:
    verdict = "❌ Strategy昇格は時期尚早。research段階のまま追加検証へ"
print(f"\n  判定: {verdict}")

# CSV出力
best_trades.to_csv('/Users/Yusuke/claude-code/japan-stocks/analyses/20260510_bank_absorption_strategy/trades.csv', index=False)
grid_df.to_csv('/Users/Yusuke/claude-code/japan-stocks/analyses/20260510_bank_absorption_strategy/grid_results.csv', index=False)
print(f"\n  ✅ 完了。CSV出力: trades.csv, grid_results.csv")
