"""
戦略5: TOPIX先物(.TOPX) 夜間 → 翌日寄りギャップ
同コア5銘柄でTOPIX先物夜間変化率シグナルを検証。
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '20260421_common')))
import mdutil as U
import pandas as pd
import numpy as np
from datetime import time as dtime
plt = U.matplotlib_jp()
import matplotlib.dates as mdates

CORE = U.CORE5
THRESHOLDS = [0.3, 0.5, 0.8, 1.0]


def load_topix_signals():
    df = U.fetch_intraday('.TOPX').dropna(subset=['close'])
    print(f"TOPIX raw rows: {len(df)}")
    sig = []
    dates = sorted(set(df.index.date))
    for i in range(1, len(dates)):
        prev_d, d = dates[i-1], dates[i]
        if d.weekday() >= 5: continue
        # TOPIX先物は昼間のみの可能性→前日クローズ→当日9:00直前で代替
        # JST 15:00~15:15のクローズ → 翌9:00直前
        prev_day = df[df.index.date == prev_d]
        today = df[df.index.date == d]
        if len(prev_day) == 0 or len(today) == 0: continue
        # 前日終値(最終)
        s0 = prev_day['close'].iloc[-1]
        # 当日9:00近辺
        morning = today[today.index.hour <= 9]
        if len(morning) == 0: continue
        s1 = morning['close'].iloc[0]
        sig.append({'date': d, 'move_pct': (s1/s0-1)*100,
                    't0': prev_day.index[-1], 't1': morning.index[0]})
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
    sig = load_topix_signals()
    jp_all = {sym: U.load_jp_daily(sym) for sym, _ in CORE}
    print(f"TOPIX夜間 N={len(sig)}, mean={sig.move_pct.mean():+.3f}%, std={sig.move_pct.std():.3f}%")

    print("\n=== コア5銘柄バスケット ===")
    results_l, results_s = {}, {}
    for th in THRESHOLDS:
        bl = backtest(sig, jp_all, th, 'long')
        bs = backtest(sig, jp_all, th, 'short')
        if len(bl) > 0:
            bl['cum_bps'] = bl['pnl_bps'].cumsum()
            st = U.compute_stats(bl['pnl_bps'].values); results_l[th] = {'bdf': bl, 'stats': st}
            U.print_stats(f"Long  th={th}%", st)
        if len(bs) > 0:
            bs['cum_bps'] = bs['pnl_bps'].cumsum()
            st = U.compute_stats(bs['pnl_bps'].values); results_s[th] = {'bdf': bs, 'stats': st}
            U.print_stats(f"Short th={th}%", st)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    colors = {0.3: '#cccccc', 0.5: '#ff7f0e', 0.8: '#2ca02c', 1.0: '#1f77b4'}
    for ax, results, title in [(axes[0,0], results_l, 'Long'), (axes[0,1], results_s, 'Short')]:
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
    ax3.set_xticks(x); ax3.set_xticklabels(ths_s); ax3.axhline(0, color='black', lw=0.8)
    ax3.set_title('Sharpe比較'); ax3.legend(); ax3.grid(alpha=0.3, axis='y')

    ax4 = axes[1, 1]
    ax4.hist(sig['move_pct'].values, bins=30, color='#8c564b', edgecolor='black', alpha=0.7)
    ax4.axvline(0, color='black', lw=1)
    ax4.set_title(f'TOPIX夜間変化率分布 (N={len(sig)})', fontweight='bold')
    ax4.set_xlabel('Move (%)'); ax4.grid(alpha=0.3)

    plt.suptitle(f'戦略5: TOPIX夜間→日本株寄り (cost={U.COST_BPS}bps)',
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), 'result.png')
    plt.savefig(out, dpi=130, bbox_inches='tight')
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
