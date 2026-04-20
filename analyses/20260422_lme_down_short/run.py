"""
戦略1: LME銅ダウン日 (<=-1%) でコア5銘柄ショート → 翌朝寄で買戻
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '20260421_common')))
import mdutil as U
import pandas as pd
import numpy as np
from datetime import time as dtime
plt = U.matplotlib_jp()
import matplotlib.dates as mdates

THRESHOLDS = [-0.5, -0.8, -1.0, -1.5]
CORE = U.CORE5


def load_lme_signals():
    df = U.fetch_intraday('CMCU3')
    df = df.dropna(subset=['close'])
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


def backtest_short(sig, jp_all, th):
    """LMEが<=th%のとき5銘柄ショート→翌朝買戻(PnL=-(retpct))"""
    per_date = {}
    for sym, jp in jp_all.items():
        dates = sorted(jp.index)
        for i, d in enumerate(dates[:-1]):
            if d not in sig.index: continue
            if sig.loc[d, 'move_pct'] > th: continue
            nd = dates[i+1]
            ret = (jp.loc[nd, 'open']/jp.loc[d, 'close'] - 1) * 100
            if abs(ret) > U.OUTLIER_PCT: continue
            pnl = -ret * 100 - U.COST_BPS
            per_date.setdefault(d, []).append(pnl)
    rows = [{'date': d, 'pnl_bps': np.mean(v)} for d, v in sorted(per_date.items())]
    return pd.DataFrame(rows).set_index('date') if rows else pd.DataFrame()


def main():
    print("Loading LME signals...")
    sig = load_lme_signals()
    jp_all = {sym: U.load_jp_daily(sym) for sym, _ in CORE}

    print(f"LME N={len(sig)}, ダウン日数:")
    for th in THRESHOLDS:
        print(f"  <=  {th}%: {(sig.move_pct<=th).sum()}")

    print("\n" + "=" * 110)
    print("戦略1: LMEダウン日 5銘柄ショートバスケット")
    print("=" * 110)
    results = {}
    for th in THRESHOLDS:
        bdf = backtest_short(sig, jp_all, th)
        if len(bdf) == 0:
            results[th] = None; continue
        bdf['cum_bps'] = bdf['pnl_bps'].cumsum()
        bdf['dd'] = bdf['cum_bps'] - bdf['cum_bps'].cummax()
        st = U.compute_stats(bdf['pnl_bps'].values)
        results[th] = {'bdf': bdf, 'stats': st}
        U.print_stats(f"Short th={th}%", st)

    # Figure
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)
    colors = {-0.5: '#cccccc', -0.8: '#ff7f0e', -1.0: '#1f77b4', -1.5: '#d62728'}

    ax1 = fig.add_subplot(gs[0, :])
    for th in THRESHOLDS:
        r = results[th]
        if r is None: continue
        b = r['bdf']; st = r['stats']
        ax1.plot(pd.to_datetime(b.index), b['cum_bps'], lw=2, color=colors[th],
                 label=f"th={th}%  N={st['n']}  Sharpe={st['sharpe']:+.2f}  Total={st['total']:+.0f}bps")
    ax1.axhline(0, color='gray', lw=0.8)
    ax1.set_title('戦略1: LMEダウン日ショート エクイティカーブ', fontsize=13, fontweight='bold')
    ax1.set_ylabel('累積PnL (bps)'); ax1.grid(alpha=0.3); ax1.legend(fontsize=10)
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    ax2 = fig.add_subplot(gs[1, 0])
    ths_s = [f"{t}%" for t in THRESHOLDS]
    sharpes = [results[t]['stats']['sharpe'] if results[t] else 0 for t in THRESHOLDS]
    cols = [colors[t] for t in THRESHOLDS]
    bars = ax2.bar(ths_s, sharpes, color=cols, edgecolor='black')
    for b, v in zip(bars, sharpes):
        ax2.text(b.get_x()+b.get_width()/2, b.get_height()+0.1, f'{v:+.2f}',
                 ha='center', fontsize=10, fontweight='bold')
    ax2.axhline(0, color='black', lw=0.8)
    ax2.set_title('Sharpe (閾値別)', fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3, axis='y')

    ax3 = fig.add_subplot(gs[1, 1])
    ns = [results[t]['stats']['n'] if results[t] else 0 for t in THRESHOLDS]
    means = [results[t]['stats']['mean'] if results[t] else 0 for t in THRESHOLDS]
    ax3b = ax3.twinx()
    ax3.bar([i-0.2 for i in range(len(THRESHOLDS))], ns, width=0.4, color='lightblue', edgecolor='k', label='N')
    ax3b.bar([i+0.2 for i in range(len(THRESHOLDS))], means, width=0.4, color='salmon', edgecolor='k', label='Mean')
    ax3.set_xticks(range(len(THRESHOLDS))); ax3.set_xticklabels(ths_s)
    ax3.set_ylabel('N', color='steelblue'); ax3b.set_ylabel('Mean bps', color='red')
    ax3.set_title('頻度 vs 平均PnL', fontsize=12, fontweight='bold')

    plt.suptitle(f'戦略1: LME銅ダウン日ショート (コア5銘柄等加重, cost={U.COST_BPS}bps)',
                 fontsize=14, fontweight='bold', y=1.00)
    out = os.path.join(os.path.dirname(__file__), 'result.png')
    plt.savefig(out, dpi=130, bbox_inches='tight')
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
