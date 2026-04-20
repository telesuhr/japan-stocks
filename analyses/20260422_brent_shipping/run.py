"""
戦略3: Brent原油(LCOc1) NYセッション変化 → 海運3銘柄 翌日寄りギャップ
海運株は燃料コスト(逆相関)と世界需要(順相関)が混在するため、検証で方向性を見る。
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '20260421_common')))
import mdutil as U
import pandas as pd
import numpy as np
from datetime import time as dtime
plt = U.matplotlib_jp()
import matplotlib.dates as mdates

SHIPPING = U.SHIPPING
THRESHOLDS = [0.5, 1.0, 1.5, 2.0]


def load_brent_signals():
    df = U.fetch_intraday('LCOc1').dropna(subset=['close'])
    sig = []
    dates = sorted(set(df.index.date))
    for i in range(1, len(dates)):
        prev_d, d = dates[i-1], dates[i]
        if d.weekday() >= 5: continue
        t0 = pd.Timestamp.combine(prev_d, dtime(15, 30))
        t1 = pd.Timestamp.combine(d, dtime(8, 30))
        seg = df[(df.index >= t0) & (df.index <= t1)]
        if len(seg) < 10: continue
        s0, s1 = seg.iloc[0]['close'], seg.iloc[-1]['close']
        sig.append({'date': d, 'move_pct': (s1/s0-1)*100})
    return pd.DataFrame(sig).set_index('date')


def backtest(sig, jp_all, th, side='long'):
    per_date = {}
    for sym, jp in jp_all.items():
        dates = sorted(jp.index)
        for i, d in enumerate(dates[:-1]):
            if d not in sig.index: continue
            mv = sig.loc[d, 'move_pct']
            if side == 'long' and mv < th: continue
            if side == 'short' and mv > -th: continue
            nd = dates[i+1]
            ret = (jp.loc[nd, 'open']/jp.loc[d, 'close'] - 1) * 100
            if abs(ret) > U.OUTLIER_PCT: continue
            pnl = (ret if side=='long' else -ret) * 100 - U.COST_BPS
            per_date.setdefault(d, []).append(pnl)
    rows = [{'date': d, 'pnl_bps': np.mean(v)} for d, v in sorted(per_date.items())]
    return pd.DataFrame(rows).set_index('date') if rows else pd.DataFrame()


def main():
    sig = load_brent_signals()
    jp_all = {sym: U.load_jp_daily(sym) for sym, _ in SHIPPING}
    print(f"Brent N={len(sig)}, std={sig.move_pct.std():.2f}%")

    print("\n=== 海運3銘柄 (郵船/商船三井/川崎汽船) ===")
    results_l, results_s = {}, {}
    for th in THRESHOLDS:
        bl = backtest(sig, jp_all, th, 'long')
        bs = backtest(sig, jp_all, th, 'short')
        if len(bl) > 0:
            bl['cum_bps'] = bl['pnl_bps'].cumsum()
            st = U.compute_stats(bl['pnl_bps'].values)
            results_l[th] = {'bdf': bl, 'stats': st}
            U.print_stats(f"Long  th={th}%", st)
        if len(bs) > 0:
            bs['cum_bps'] = bs['pnl_bps'].cumsum()
            st = U.compute_stats(bs['pnl_bps'].values)
            results_s[th] = {'bdf': bs, 'stats': st}
            U.print_stats(f"Short th={th}%", st)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    colors = {0.5: '#cccccc', 1.0: '#1f77b4', 1.5: '#2ca02c', 2.0: '#d62728'}

    for ax, results, title in [(axes[0,0], results_l, 'Long (Brentアップ時買い)'),
                                (axes[0,1], results_s, 'Short (Brentダウン時売り)')]:
        for th in THRESHOLDS:
            if th not in results: continue
            b = results[th]['bdf']; st = results[th]['stats']
            ax.plot(pd.to_datetime(b.index), b['cum_bps'], lw=2, color=colors[th],
                    label=f"{th}% N={st['n']} Shp={st['sharpe']:+.2f}")
        ax.set_title(title, fontweight='bold')
        ax.axhline(0, color='gray', lw=0.8); ax.legend(fontsize=9); ax.grid(alpha=0.3)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))

    ax3 = axes[1, 0]
    ths_s = [f"{t}%" for t in THRESHOLDS]
    sl = [results_l[t]['stats']['sharpe'] if t in results_l else 0 for t in THRESHOLDS]
    ss = [results_s[t]['stats']['sharpe'] if t in results_s else 0 for t in THRESHOLDS]
    x = np.arange(len(THRESHOLDS))
    ax3.bar(x-0.2, sl, width=0.4, color='steelblue', label='Long', edgecolor='black')
    ax3.bar(x+0.2, ss, width=0.4, color='salmon', label='Short', edgecolor='black')
    ax3.set_xticks(x); ax3.set_xticklabels(ths_s)
    ax3.set_title('Sharpe 比較', fontweight='bold'); ax3.legend(); ax3.grid(alpha=0.3, axis='y')
    ax3.axhline(0, color='black', lw=0.8)

    ax4 = axes[1, 1]
    ax4.hist(sig['move_pct'].values, bins=30, color='#2ca02c', edgecolor='black', alpha=0.7)
    ax4.axvline(0, color='black', lw=1)
    ax4.set_title(f'Brent NYセッション変化率分布 (N={len(sig)})', fontweight='bold')
    ax4.set_xlabel('Move (%)'); ax4.grid(alpha=0.3)

    plt.suptitle(f'戦略3: Brent原油OVN→海運3銘柄 (cost={U.COST_BPS}bps)',
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), 'result.png')
    plt.savefig(out, dpi=130, bbox_inches='tight')
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
