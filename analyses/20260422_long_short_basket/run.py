"""
戦略7: LMEアップ日に 景気敏感5銘柄Long + 内需5銘柄Short のLSバスケット
内需: 8267 イオン, 9020 JR東, 7974 任天堂, 6758 ソニー, 8411 みずほ
ロング: コア5 (5711/6501/7011/5016/4502)
コスト: 片側2bps×往復×2サイド = 8bps
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '20260421_common')))
import mdutil as U
import pandas as pd
import numpy as np
from datetime import time as dtime
plt = U.matplotlib_jp()
import matplotlib.dates as mdates

LONG = U.CORE5
SHORT = U.DOMESTIC_SHORT
COST_LS = 8.0
THRESHOLDS = [0.5, 0.8, 1.0, 1.5]


def load_lme_signals():
    df = U.fetch_intraday('CMCU3').dropna(subset=['close'])
    sig = []
    for d in sorted(set(df.index.date)):
        if d.weekday() >= 5: continue
        oh = 9 if U.is_bst(d) else 10
        ot = pd.Timestamp.combine(d, dtime(oh, 0))
        ct = pd.Timestamp.combine(d, dtime(15, 25))
        day = df[df.index.date == d]
        if len(day) == 0: continue
        after = day[day.index >= ot]; before = day[day.index <= ct]
        if len(after)==0 or len(before)==0: continue
        ob, cb = after.iloc[0], before.iloc[-1]
        if (ob.name-ot).total_seconds()>1800 or (ct-cb.name).total_seconds()>1800: continue
        sig.append({'date': d, 'move_pct': (cb['close']/ob['open']-1)*100})
    return pd.DataFrame(sig).set_index('date')


def backtest_ls(sig, long_jp, short_jp, th):
    """LMEアップ日にLong basket + Short basket"""
    per_date_l = {}; per_date_s = {}
    for sym, jp in long_jp.items():
        dates = sorted(jp.index)
        for i, d in enumerate(dates[:-1]):
            if d not in sig.index: continue
            if sig.loc[d, 'move_pct'] < th: continue
            nd = dates[i+1]
            ret = (jp.loc[nd, 'open']/jp.loc[d, 'close'] - 1) * 100
            if abs(ret) > U.OUTLIER_PCT: continue
            per_date_l.setdefault(d, []).append(ret * 100)
    for sym, jp in short_jp.items():
        dates = sorted(jp.index)
        for i, d in enumerate(dates[:-1]):
            if d not in sig.index: continue
            if sig.loc[d, 'move_pct'] < th: continue
            nd = dates[i+1]
            ret = (jp.loc[nd, 'open']/jp.loc[d, 'close'] - 1) * 100
            if abs(ret) > U.OUTLIER_PCT: continue
            per_date_s.setdefault(d, []).append(-ret * 100)  # short: invert
    all_dates = sorted(set(per_date_l.keys()) & set(per_date_s.keys()))
    rows = []
    for d in all_dates:
        l_pnl = np.mean(per_date_l[d])
        s_pnl = np.mean(per_date_s[d])
        total = (l_pnl + s_pnl) / 2 - COST_LS  # 等加重
        rows.append({'date': d, 'pnl_bps': total, 'long_bps': l_pnl, 'short_bps': s_pnl})
    return pd.DataFrame(rows).set_index('date') if rows else pd.DataFrame()


def main():
    sig = load_lme_signals()
    long_jp = {s: U.load_jp_daily(s) for s, _ in LONG}
    short_jp = {s: U.load_jp_daily(s) for s, _ in SHORT}
    print(f"LME N={len(sig)}")

    print("\n=== LSバスケット (Long 5銘柄 + Short 5銘柄, コスト8bps) ===")
    results = {}
    for th in THRESHOLDS:
        bdf = backtest_ls(sig, long_jp, short_jp, th)
        if len(bdf) == 0: continue
        bdf['cum_bps'] = bdf['pnl_bps'].cumsum()
        bdf['long_cum'] = bdf['long_bps'].cumsum()
        bdf['short_cum'] = bdf['short_bps'].cumsum()
        bdf['dd'] = bdf['cum_bps'] - bdf['cum_bps'].cummax()
        st = U.compute_stats(bdf['pnl_bps'].values)
        results[th] = {'bdf': bdf, 'stats': st}
        U.print_stats(f"LS  th={th}%", st)
        st_l = U.compute_stats(bdf['long_bps'].values)
        st_s = U.compute_stats(bdf['short_bps'].values)
        print(f"    Long leg  only: Mean={st_l['mean']:+.1f} Sharpe={st_l['sharpe']:+.2f}")
        print(f"    Short leg only: Mean={st_s['mean']:+.1f} Sharpe={st_s['sharpe']:+.2f}")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    colors = {0.5: '#cccccc', 0.8: '#ff7f0e', 1.0: '#1f77b4', 1.5: '#d62728'}

    ax1 = axes[0, 0]
    for th in THRESHOLDS:
        if th not in results: continue
        b = results[th]['bdf']; st = results[th]['stats']
        ax1.plot(pd.to_datetime(b.index), b['cum_bps'], lw=2, color=colors[th],
                 label=f"{th}% N={st['n']} Shp={st['sharpe']:+.2f}")
    ax1.set_title('LS Basket 累積PnL (net of 8bps)', fontweight='bold')
    ax1.axhline(0, color='gray', lw=0.8); ax1.legend(fontsize=9); ax1.grid(alpha=0.3)
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))

    ax2 = axes[0, 1]
    if 1.0 in results:
        b = results[1.0]['bdf']
        ax2.plot(pd.to_datetime(b.index), b['long_cum'], lw=2, color='steelblue', label='Long leg')
        ax2.plot(pd.to_datetime(b.index), b['short_cum'], lw=2, color='salmon', label='Short leg')
        ax2.plot(pd.to_datetime(b.index), b['cum_bps'], lw=2, color='black', label='LS合算 (net)')
    ax2.set_title('th=1.0% レッグ別累積 (gross)', fontweight='bold')
    ax2.axhline(0, color='gray', lw=0.8); ax2.legend(); ax2.grid(alpha=0.3)
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))

    ax3 = axes[1, 0]
    ths_s = [f"{t}%" for t in THRESHOLDS]
    sharpes = [results[t]['stats']['sharpe'] if t in results else 0 for t in THRESHOLDS]
    cols = [colors[t] for t in THRESHOLDS]
    bars = ax3.bar(ths_s, sharpes, color=cols, edgecolor='black')
    for b, v in zip(bars, sharpes):
        ax3.text(b.get_x()+b.get_width()/2, b.get_height()+0.1, f'{v:+.2f}',
                 ha='center', fontsize=10, fontweight='bold')
    ax3.axhline(0, color='black', lw=0.8)
    ax3.set_title('Sharpe (閾値別)', fontweight='bold'); ax3.grid(alpha=0.3, axis='y')

    ax4 = axes[1, 1]
    if 1.0 in results:
        b = results[1.0]['bdf']
        ax4.hist(b['pnl_bps'].values, bins=15, color='#9467bd', edgecolor='black', alpha=0.7)
        ax4.axvline(0, color='black', lw=1)
        ax4.axvline(b['pnl_bps'].mean(), color='red', lw=1.5, ls='--',
                    label=f'Mean={b.pnl_bps.mean():+.1f}')
        ax4.set_title(f'th=1.0% LS PnL分布 (net)', fontweight='bold')
        ax4.legend(); ax4.grid(alpha=0.3)

    plt.suptitle(f'戦略7: LMEアップ日 Long/Short Basket (cost={COST_LS}bps LS)',
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), 'result.png')
    plt.savefig(out, dpi=130, bbox_inches='tight')
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
