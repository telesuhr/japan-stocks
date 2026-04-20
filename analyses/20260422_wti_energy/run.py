"""
戦略2: WTI原油(CLc1) のNYセッション変化 → 日本のエネルギー銘柄 翌日寄りギャップ
シグナル定義: 前日JST 15:30時点のWTI → 当日JST 8:30時点のWTI の変化率
エントリー: 日本株を前日引けでLong(or Short)、翌9:00寄決済
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '20260421_common')))
import mdutil as U
import pandas as pd
import numpy as np
from datetime import time as dtime, timedelta
plt = U.matplotlib_jp()
import matplotlib.dates as mdates

ENERGY = [("1605.T", "INPEX"), ("5016.T", "出光"), ("5020.T", "ENEOS")]
THRESHOLDS = [0.5, 1.0, 1.5, 2.0]


def load_wti_signals():
    df = U.fetch_intraday('CLc1').dropna(subset=['close'])
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
            # 同方向: WTIアップ→long, WTIダウン→short
            # 逆方向: oppositeトレード
            if side == 'long' and mv < th: continue
            if side == 'short' and mv > -th: continue
            nd = dates[i+1]
            ret = (jp.loc[nd, 'open']/jp.loc[d, 'close'] - 1) * 100
            if abs(ret) > U.OUTLIER_PCT: continue
            pnl = (ret if side == 'long' else -ret) * 100 - U.COST_BPS
            per_date.setdefault(d, []).append(pnl)
    rows = [{'date': d, 'pnl_bps': np.mean(v)} for d, v in sorted(per_date.items())]
    return pd.DataFrame(rows).set_index('date') if rows else pd.DataFrame()


def main():
    print("Loading WTI signals...")
    sig = load_wti_signals()
    print(f"WTI N={len(sig)}")
    print(f"  mean move: {sig.move_pct.mean():+.3f}%, std: {sig.move_pct.std():.3f}%")
    for th in THRESHOLDS:
        print(f"  |move|>={th}%: UP={(sig.move_pct>=th).sum()} DOWN={(sig.move_pct<=-th).sum()}")

    jp_all = {sym: U.load_jp_daily(sym) for sym, _ in ENERGY}

    print("\n=== エネルギー3銘柄 (INPEX/出光/ENEOS) ===")
    results_long = {}
    results_short = {}
    for th in THRESHOLDS:
        bdf_l = backtest(sig, jp_all, th, 'long')
        bdf_s = backtest(sig, jp_all, th, 'short')
        if len(bdf_l) > 0:
            bdf_l['cum_bps'] = bdf_l['pnl_bps'].cumsum()
            st_l = U.compute_stats(bdf_l['pnl_bps'].values)
            results_long[th] = {'bdf': bdf_l, 'stats': st_l}
            U.print_stats(f"Long th={th}%", st_l)
        if len(bdf_s) > 0:
            bdf_s['cum_bps'] = bdf_s['pnl_bps'].cumsum()
            st_s = U.compute_stats(bdf_s['pnl_bps'].values)
            results_short[th] = {'bdf': bdf_s, 'stats': st_s}
            U.print_stats(f"Short th={th}%", st_s)

    # Figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    colors = {0.5: '#cccccc', 1.0: '#1f77b4', 1.5: '#2ca02c', 2.0: '#d62728'}

    ax1 = axes[0, 0]
    for th in THRESHOLDS:
        if th not in results_long: continue
        b = results_long[th]['bdf']; st = results_long[th]['stats']
        ax1.plot(pd.to_datetime(b.index), b['cum_bps'], lw=2, color=colors[th],
                 label=f"{th}% N={st['n']} Shp={st['sharpe']:+.1f}")
    ax1.set_title('Long (WTIアップ時)', fontweight='bold')
    ax1.axhline(0, color='gray', lw=0.8); ax1.legend(fontsize=9); ax1.grid(alpha=0.3)
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))

    ax2 = axes[0, 1]
    for th in THRESHOLDS:
        if th not in results_short: continue
        b = results_short[th]['bdf']; st = results_short[th]['stats']
        ax2.plot(pd.to_datetime(b.index), b['cum_bps'], lw=2, color=colors[th],
                 label=f"{th}% N={st['n']} Shp={st['sharpe']:+.1f}")
    ax2.set_title('Short (WTIダウン時)', fontweight='bold')
    ax2.axhline(0, color='gray', lw=0.8); ax2.legend(fontsize=9); ax2.grid(alpha=0.3)
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))

    ax3 = axes[1, 0]
    ths_s = [f"{t}%" for t in THRESHOLDS]
    sl = [results_long[t]['stats']['sharpe'] if t in results_long else 0 for t in THRESHOLDS]
    ss = [results_short[t]['stats']['sharpe'] if t in results_short else 0 for t in THRESHOLDS]
    x = np.arange(len(THRESHOLDS))
    ax3.bar(x-0.2, sl, width=0.4, color='steelblue', label='Long', edgecolor='black')
    ax3.bar(x+0.2, ss, width=0.4, color='salmon', label='Short', edgecolor='black')
    ax3.set_xticks(x); ax3.set_xticklabels(ths_s)
    ax3.set_title('Sharpe 比較', fontweight='bold'); ax3.legend(); ax3.grid(alpha=0.3, axis='y')
    ax3.axhline(0, color='black', lw=0.8)

    ax4 = axes[1, 1]
    ax4.hist(sig['move_pct'].values, bins=30, color='#1f77b4', edgecolor='black', alpha=0.7)
    ax4.axvline(0, color='black', lw=1)
    ax4.set_title(f'WTI NYセッション変化率分布 (N={len(sig)})', fontweight='bold')
    ax4.set_xlabel('Move (%)'); ax4.grid(alpha=0.3)

    plt.suptitle(f'戦略2: WTI原油オーバーナイト→日本エネルギー3銘柄 (cost={U.COST_BPS}bps)',
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), 'result.png')
    plt.savefig(out, dpi=130, bbox_inches='tight')
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
