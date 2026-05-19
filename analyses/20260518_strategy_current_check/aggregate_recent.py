"""
9戦略の直近パフォーマンス集計
- 既存5戦略: run_6months.py の結果
- 新規4戦略: trades.csv から直近期間抽出
"""
import sys, os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 既存5戦略 (run_6months.py 実行済みの数値を貼り付け)
EXISTING_RESULTS = {
    'vwap_morning_meanrevert':  dict(N=24, WR=62.5, PF=2.13, sharpe=4.58, mean_bps=49.6, baseline=6.76),
    'bank_absorption':          dict(N=70, WR=61.4, PF=1.85, sharpe=3.95, mean_bps=129.5, baseline=1.84),
    'lasertec_ma25_support':    dict(N=5,  WR=60.0, PF=2.54, sharpe=6.96, mean_bps=618.5, baseline=7.57),
    'eneos_vwap_trend':         dict(N=43, WR=51.2, PF=1.44, sharpe=2.37, mean_bps=19.4, baseline=3.81),
    'orb_breakout_long':        dict(N=116,WR=50.0, PF=1.03, sharpe=0.15, mean_bps=1.6,  baseline=2.31),
}

# 新規4戦略をtrades.csvから集計
def aggregate_trades(csv_path, date_col_candidates, ret_col_candidates, cutoff='2025-11-15'):
    df = pd.read_csv(csv_path)
    # 日付列を検出
    date_col = None
    for c in date_col_candidates:
        if c in df.columns:
            date_col = c; break
    if date_col is None:
        print(f"  日付列なし: cols={list(df.columns)[:5]}")
        return None
    df[date_col] = pd.to_datetime(df[date_col])
    df = df[df[date_col] >= pd.Timestamp(cutoff)].copy()
    if len(df) == 0:
        return None
    # リターン列検出
    ret_col = None
    for c in ret_col_candidates:
        if c in df.columns:
            ret_col = c; break
    if ret_col is None:
        print(f"  リターン列なし: cols={list(df.columns)[:10]}")
        return None
    rets = df[ret_col].dropna().values
    if len(rets) == 0:
        return None
    wr = (rets > 0).mean() * 100
    pf = rets[rets > 0].sum() / abs(rets[rets <= 0].sum()) if (rets <= 0).any() else 99
    # Sharpeは1取引あたりリターンの平均/std * sqrt(252/平均保有日数)
    mean_ret = rets.mean()
    std_ret  = rets.std()
    # 平均保有日数を仮置き
    return dict(N=len(rets), WR=wr, PF=pf, mean_ret=mean_ret, std_ret=std_ret,
                rets=rets, dates=df[date_col].values)

# 新規戦略のtrades読み込み
new_strategies = {
    'oversold_ma25_reversal': dict(
        path='/Users/Yusuke/claude-code/japan-stocks/analyses/20260515_oversold_ma25_validation/trades.csv',
        dates=['exit_date','date','entry_date'],
        rets=['net_ret','ret','pnl_pct','pnl'],
        hold_days=5, baseline=3.21
    ),
    'large_cap_oversold_reversal': dict(
        path='/Users/Yusuke/claude-code/japan-stocks/analyses/20260515_large_cap_reversal_validation/trades.csv',
        dates=['exit_date','date','entry_date'],
        rets=['net_ret','ret','pnl_pct','pnl'],
        hold_days=5, baseline=2.92
    ),
    'earnings_pead': dict(
        path='/Users/Yusuke/claude-code/japan-stocks/analyses/20260512_earnings_pead_validation/best_trades.csv',
        dates=['exit_date','date','entry_date','disc_date'],
        rets=['net_ret','ret','pnl_pct','pnl'],
        hold_days=5, baseline=2.19
    ),
    'pre_earnings_drift': dict(
        path='/Users/Yusuke/claude-code/japan-stocks/analyses/20260512_pre_earnings_drift_validation/trades.csv',
        dates=['disc_date'],
        rets=['net_ret','ret'],
        hold_days=4, baseline=2.07
    ),
}

print("="*82)
print("【9戦略 直近6ヶ月パフォーマンス vs 5年検証ベースライン】")
print("="*82)
print(f"\n{'戦略':<32} {'N':>5} {'WR%':>6} {'PF':>6} {'Sharpe':>8} {'Base':>6} {'判定':>8}")
print("-"*82)

results = []

# 既存5戦略
for name, r in EXISTING_RESULTS.items():
    sh = r['sharpe']
    base = r['baseline']
    diff = sh - base
    if sh >= 2.0:
        judge = "✅継続" if diff >= -1.5 else "⚠️低下"
    elif sh >= 1.0:
        judge = "⚠️低下"
    else:
        judge = "❌劣化"
    print(f"{name:<32} {r['N']:>5} {r['WR']:>5.1f}% {r['PF']:>5.2f} {sh:>+7.2f} {base:>+5.2f} {judge:>8}")
    results.append({'name': name, **r, 'judge': judge})

# 新規4戦略
for name, cfg in new_strategies.items():
    res = aggregate_trades(cfg['path'], cfg['dates'], cfg['rets'])
    if res is None:
        print(f"{name:<32} {'--':>5} {'--':>6} {'--':>6} {'--':>8} {cfg['baseline']:>+5.2f} {'NoData':>8}")
        continue
    # Sharpe概算: トレードリターン平均/std * sqrt(252/hold_days)
    sh = (res['mean_ret']/res['std_ret']*np.sqrt(252/cfg['hold_days'])) if res['std_ret']>0 else 0
    base = cfg['baseline']
    diff = sh - base
    if sh >= 2.0:
        judge = "✅継続" if diff >= -1.5 else "⚠️低下"
    elif sh >= 1.0:
        judge = "⚠️低下"
    else:
        judge = "❌劣化"
    mean_bps = res['mean_ret'] * 100  # %→bps
    print(f"{name:<32} {res['N']:>5} {res['WR']:>5.1f}% {res['PF']:>5.2f} {sh:>+7.2f} {base:>+5.2f} {judge:>8}")
    results.append({'name': name, 'N': res['N'], 'WR': res['WR'], 'PF': res['PF'],
                    'sharpe': sh, 'baseline': base, 'mean_bps': mean_bps, 'judge': judge})

print("="*82)
print("\n判定凡例: ✅継続(Sharpe≥2.0かつベースから-1.5以内) / ⚠️低下 / ❌劣化")

# CSV保存
df_out = pd.DataFrame(results)
df_out.to_csv('/Users/Yusuke/claude-code/japan-stocks/analyses/20260518_strategy_current_check/summary.csv', index=False)
print(f"\n→ summary.csv 保存完了")
