"""
戦略9: オープンレンジブレイクアウト (ORB)
- レンジ期間: 9:00 - 9:30 (30分)
- ブレイク: 10:00までに上/下抜け検知
- エントリー: ブレイク価格で成行
- 決済: 11:30 前場引け
対象: 流動性が高い13銘柄
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
MIN_RANGE_BPS = [0, 50, 100]  # ORB幅が小さすぎる日はスキップ


def load_1min(sym):
    return U.fetch_intraday(sym).dropna(subset=['open','high','low','close'])


def orb_backtest(df, min_range_bps):
    rows = []
    for d in sorted(set(df.index.date)):
        if d.weekday() >= 5: continue
        day = df[df.index.date == d]
        h, m = day.index.hour, day.index.minute
        # 9:00-9:30 レンジ
        rng = day[((h==9)&(m<=30))]
        if len(rng) < 10: continue
        rh, rl = rng['high'].max(), rng['low'].min()
        open_p = rng.iloc[0]['open']
        range_bps = (rh/rl - 1) * 10000
        if range_bps < min_range_bps: continue

        # ブレイク検知 9:30-10:00
        after = day[((h==9)&(m>30)) | ((h==10)&(m<=0))]
        if len(after) == 0: continue
        entry = None; side = None
        for t, row in after.iterrows():
            if row['high'] > rh:
                entry = rh; side = 1; break
            if row['low'] < rl:
                entry = rl; side = -1; break
        if entry is None: continue

        # 前場引け 11:30近辺
        cls = day[((h==11)&(m>=25)&(m<=31))]
        if len(cls) == 0: continue
        exit_p = cls.iloc[-1]['close']
        ret = side * (exit_p/entry - 1) * 100
        if abs(ret) > 10: continue
        pnl = ret * 100 - U.COST_BPS
        rows.append({'date': d, 'range_bps': range_bps, 'side': side, 'pnl_bps': pnl})
    return pd.DataFrame(rows)


def main():
    print("ORB backtesting 13 symbols...")
    all_trades = {}
    for sym, name in TARGETS:
        df = load_1min(sym)
        print(f"  {sym} {name}: rows={len(df)}")
        for mr in MIN_RANGE_BPS:
            t = orb_backtest(df, mr)
            key = (sym, mr)
            all_trades[key] = t

    print("\n=== 個別銘柄 ORB (min_range=0) ===")
    print(f"{'Sym':<10} {'Name':<12} {'N':>5} {'Mean':>7} {'WR':>6} {'Sharpe':>7}")
    summary = []
    for sym, name in TARGETS:
        t = all_trades[(sym, 0)]
        if len(t) < 5: continue
        st = U.compute_stats(t['pnl_bps'].values)
        summary.append((sym, name, st))
        print(f"{sym:<10} {name:<12} {st['n']:>5} {st['mean']:>+6.1f} {st['wr']:>5.1f}% {st['sharpe']:>+6.2f}")

    print("\n=== 全銘柄集約 (各min_range) ===")
    for mr in MIN_RANGE_BPS:
        all_arr = []
        for sym, _ in TARGETS:
            all_arr.append(all_trades[(sym, mr)]['pnl_bps'].values)
        arr = np.concatenate(all_arr) if all_arr else np.array([])
        st = U.compute_stats(arr) if len(arr) else None
        if st:
            U.print_stats(f"min_range>={mr}bps", st)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. 個別銘柄Sharpe bar
    ax1 = axes[0, 0]
    summary_sorted = sorted(summary, key=lambda x: x[2]['sharpe'], reverse=True)
    labels = [s[0] for s in summary_sorted]
    sharpes = [s[2]['sharpe'] for s in summary_sorted]
    cols = ['#2ca02c' if v > 0 else '#d62728' for v in sharpes]
    ax1.barh(labels, sharpes, color=cols, edgecolor='black')
    ax1.axvline(0, color='black', lw=0.8)
    ax1.set_title('銘柄別 ORB Sharpe', fontweight='bold')
    ax1.grid(alpha=0.3, axis='x')

    # 2. Equity curve: 全銘柄合算
    ax2 = axes[0, 1]
    all_t = []
    for sym, _ in TARGETS:
        t = all_trades[(sym, 0)]
        if len(t) > 0:
            all_t.append(t)
    if all_t:
        big = pd.concat(all_t).sort_values('date')
        big['cum'] = big['pnl_bps'].cumsum()
        ax2.plot(pd.to_datetime(big['date']), big['cum'], lw=1, color='steelblue')
    ax2.axhline(0, color='gray', lw=0.8)
    ax2.set_title('全銘柄合算 累積PnL', fontweight='bold')
    ax2.grid(alpha=0.3)
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))

    # 3. min_range別Sharpe
    ax3 = axes[1, 0]
    ranges = []
    sharpes_r = []
    for mr in MIN_RANGE_BPS:
        arr_all = np.concatenate([all_trades[(s,mr)]['pnl_bps'].values for s,_ in TARGETS
                                   if len(all_trades[(s,mr)])>0])
        st = U.compute_stats(arr_all)
        if st:
            ranges.append(f"{mr}bps")
            sharpes_r.append(st['sharpe'])
    cols3 = ['#2ca02c' if v > 0 else '#d62728' for v in sharpes_r]
    ax3.bar(ranges, sharpes_r, color=cols3, edgecolor='black')
    ax3.axhline(0, color='black', lw=0.8)
    ax3.set_title('min_range フィルタ別 全体Sharpe', fontweight='bold')
    ax3.grid(alpha=0.3, axis='y')
    for i, v in enumerate(sharpes_r):
        ax3.text(i, v+0.02, f'{v:+.2f}', ha='center', fontsize=10, fontweight='bold')

    # 4. PnL分布
    ax4 = axes[1, 1]
    if all_t:
        arr = big['pnl_bps'].values
        ax4.hist(arr, bins=40, color='#1f77b4', edgecolor='black', alpha=0.7)
        ax4.axvline(0, color='black', lw=1)
        ax4.axvline(arr.mean(), color='red', lw=1.5, ls='--', label=f'Mean={arr.mean():+.1f}')
        ax4.set_title(f'全銘柄ORB PnL分布 N={len(arr)}', fontweight='bold')
        ax4.legend(); ax4.grid(alpha=0.3)

    plt.suptitle(f'戦略9: 9:00-9:30 ORB → 11:30決済 (13銘柄, cost={U.COST_BPS}bps)',
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), 'result.png')
    plt.savefig(out, dpi=130, bbox_inches='tight')
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
