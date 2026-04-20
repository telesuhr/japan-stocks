"""
戦略8: 日本株大型ギャップフェード
前日引け → 当日寄 のギャップが大きい銘柄に対し、
  A) Fade (逆張り): 大GUで売り / 大GDで買い
  B) Continuation (順張り): 大GUで買い / 大GDで売り
  → 当日 15:25 決済
コア5銘柄 + 半導体 + 海運 を対象に、ギャップ |>=2%| で検証
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '20260421_common')))
import mdutil as U
import pandas as pd
import numpy as np
from datetime import time as dtime
plt = U.matplotlib_jp()
import matplotlib.dates as mdates

TARGETS = U.CORE5 + U.SEMICON + U.SHIPPING
GAP_THRESHOLDS = [1.0, 1.5, 2.0, 3.0]


def load_jp_with_intraday(sym):
    """寄り/引けに加えて、日中も取る"""
    df = U.fetch_intraday(sym).dropna(subset=['open','close'])
    return df


def extract_daily_with_gap(df):
    out = []
    for d in sorted(set(df.index.date)):
        if d.weekday() >= 5: continue
        gd = df[df.index.date == d]
        h, m = gd.index.hour, gd.index.minute
        op = gd[(h==9)&(m<=5)]
        # 当日15:25引け付近
        cl = gd[(h==15)&(m>=20)&(m<=30)]
        if len(op) == 0 or len(cl) == 0: continue
        out.append({'date': d, 'open': op['open'].iloc[0], 'close': cl['close'].iloc[-1]})
    ddf = pd.DataFrame(out).set_index('date')
    ddf['prev_close'] = ddf['close'].shift(1)
    ddf['gap_pct'] = (ddf['open'] / ddf['prev_close'] - 1) * 100
    ddf['intraday_ret_pct'] = (ddf['close'] / ddf['open'] - 1) * 100
    return ddf


def backtest(ddf_dict, gap_th, direction):
    """direction: 'fade' or 'momentum'"""
    rows = []
    for sym, ddf in ddf_dict.items():
        for d, r in ddf.iterrows():
            if pd.isna(r['gap_pct']) or pd.isna(r['intraday_ret_pct']): continue
            g = r['gap_pct']
            if abs(g) < gap_th: continue
            if abs(r['intraday_ret_pct']) > U.OUTLIER_PCT: continue
            if direction == 'fade':
                # 大GUで売り、大GDで買い
                side = -1 if g > 0 else 1
            else:
                side = 1 if g > 0 else -1
            pnl = side * r['intraday_ret_pct'] * 100 - U.COST_BPS
            rows.append({'date': d, 'sym': sym, 'gap': g, 'intraday': r['intraday_ret_pct'], 'pnl_bps': pnl})
    return pd.DataFrame(rows)


def main():
    print("Loading intraday data...")
    ddf_dict = {}
    for sym, _ in TARGETS:
        df = load_jp_with_intraday(sym)
        ddf_dict[sym] = extract_daily_with_gap(df)
        print(f"  {sym}: {len(ddf_dict[sym])}日")

    results = {}
    for direction in ['fade', 'momentum']:
        results[direction] = {}
        print(f"\n=== Direction: {direction} ===")
        for gap_th in GAP_THRESHOLDS:
            tdf = backtest(ddf_dict, gap_th, direction)
            if len(tdf) == 0: continue
            tdf = tdf.sort_values('date')
            st = U.compute_stats(tdf['pnl_bps'].values)
            results[direction][gap_th] = {'tdf': tdf, 'stats': st}
            U.print_stats(f"gap>={gap_th}% {direction}", st)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    colors = {1.0: '#cccccc', 1.5: '#ff7f0e', 2.0: '#1f77b4', 3.0: '#d62728'}

    for idx, direction in enumerate(['fade', 'momentum']):
        ax = axes[0, idx]
        for gap_th in GAP_THRESHOLDS:
            if gap_th not in results[direction]: continue
            tdf = results[direction][gap_th]['tdf'].copy()
            tdf['cum'] = tdf['pnl_bps'].cumsum()
            st = results[direction][gap_th]['stats']
            ax.plot(pd.to_datetime(tdf['date']), tdf['cum'], lw=1.5, color=colors[gap_th],
                    label=f"gap>={gap_th}% N={st['n']} Shp={st['sharpe']:+.2f}")
        ax.set_title(f'{direction.upper()} 戦略累積PnL', fontweight='bold')
        ax.axhline(0, color='gray', lw=0.8); ax.legend(fontsize=9); ax.grid(alpha=0.3)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))

    ax3 = axes[1, 0]
    ths_s = [f"{t}%" for t in GAP_THRESHOLDS]
    fade_sh = [results['fade'][t]['stats']['sharpe'] if t in results['fade'] else 0 for t in GAP_THRESHOLDS]
    mom_sh = [results['momentum'][t]['stats']['sharpe'] if t in results['momentum'] else 0 for t in GAP_THRESHOLDS]
    x = np.arange(len(GAP_THRESHOLDS))
    ax3.bar(x-0.2, fade_sh, width=0.4, color='salmon', label='Fade', edgecolor='black')
    ax3.bar(x+0.2, mom_sh, width=0.4, color='steelblue', label='Momentum', edgecolor='black')
    ax3.set_xticks(x); ax3.set_xticklabels(ths_s); ax3.axhline(0, color='black', lw=0.8)
    ax3.set_title('Sharpe比較: Fade vs Momentum'); ax3.legend(); ax3.grid(alpha=0.3, axis='y')

    ax4 = axes[1, 1]
    # 全銘柄のgap vs intraday散布図
    all_g, all_i = [], []
    for ddf in ddf_dict.values():
        tmp = ddf.dropna(subset=['gap_pct', 'intraday_ret_pct'])
        tmp = tmp[(tmp['gap_pct'].abs() < 10) & (tmp['intraday_ret_pct'].abs() < 10)]
        all_g.extend(tmp['gap_pct'].values)
        all_i.extend(tmp['intraday_ret_pct'].values)
    ax4.scatter(all_g, all_i, alpha=0.3, s=8, color='#1f77b4')
    ax4.axhline(0, color='black', lw=0.5); ax4.axvline(0, color='black', lw=0.5)
    ax4.set_xlabel('寄りギャップ (%)'); ax4.set_ylabel('当日日中リターン (%)')
    ax4.set_title(f'ギャップ vs 日中リターン 散布図 (N={len(all_g)})', fontweight='bold')
    ax4.grid(alpha=0.3)
    # 相関係数
    if len(all_g) > 0:
        corr = np.corrcoef(all_g, all_i)[0, 1]
        ax4.text(0.05, 0.95, f'corr = {corr:+.3f}', transform=ax4.transAxes,
                 fontsize=11, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white'))

    plt.suptitle(f'戦略8: 大型ギャップ Fade vs Momentum (13銘柄, cost={U.COST_BPS}bps)',
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), 'result.png')
    plt.savefig(out, dpi=130, bbox_inches='tight')
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
