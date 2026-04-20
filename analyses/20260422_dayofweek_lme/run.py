"""
戦略6: LMEシグナルの曜日効果
LME銅アップ日(>=+1%)に対し、コア5銘柄のONリターンを曜日別に検証。
週明け (月曜エントリー→火曜寄) は特に強い? 金曜エントリー→月曜寄は?
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '20260421_common')))
import mdutil as U
import pandas as pd
import numpy as np
from datetime import time as dtime
plt = U.matplotlib_jp()

CORE = U.CORE5
THRESHOLD = 1.0
DAYNAMES = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']


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


def main():
    sig = load_lme_signals()
    jp_all = {sym: U.load_jp_daily(sym) for sym, _ in CORE}

    # LMEアップ日でコア5銘柄Longトレードを収集、曜日別分類
    per_dow = {i: [] for i in range(5)}
    per_dow_all = {i: [] for i in range(5)}  # 全日baseline
    for sym, jp in jp_all.items():
        dates = sorted(jp.index)
        for i, d in enumerate(dates[:-1]):
            if d not in sig.index: continue
            nd = dates[i+1]
            ret = (jp.loc[nd, 'open']/jp.loc[d, 'close'] - 1) * 100
            if abs(ret) > U.OUTLIER_PCT: continue
            dow = d.weekday()
            if dow > 4: continue
            pnl = ret * 100 - U.COST_BPS
            per_dow_all[dow].append(pnl)
            if sig.loc[d, 'move_pct'] >= THRESHOLD:
                per_dow[dow].append(pnl)

    print("\n=== LME>=+1%アップ日 Long コア5 平均加重 ===")
    print(f"{'Dow':<8} {'N':>5} {'Mean':>8} {'WR':>6} {'Sharpe':>8}")
    sig_stats = {}
    for i in range(5):
        arr = np.array(per_dow[i])
        st = U.compute_stats(arr) if len(arr) else None
        sig_stats[i] = st
        if st:
            print(f"{DAYNAMES[i]:<8} {st['n']:>5} {st['mean']:>+7.1f} {st['wr']:>5.1f}% {st['sharpe']:>+7.2f}")

    print("\n=== ベースライン: 全日 ON リターン (閾値なし) ===")
    print(f"{'Dow':<8} {'N':>5} {'Mean':>8} {'WR':>6} {'Sharpe':>8}")
    base_stats = {}
    for i in range(5):
        arr = np.array(per_dow_all[i])
        st = U.compute_stats(arr) if len(arr) else None
        base_stats[i] = st
        if st:
            print(f"{DAYNAMES[i]:<8} {st['n']:>5} {st['mean']:>+7.1f} {st['wr']:>5.1f}% {st['sharpe']:>+7.2f}")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    x = np.arange(5)

    ax1 = axes[0, 0]
    sig_means = [sig_stats[i]['mean'] if sig_stats[i] else 0 for i in range(5)]
    base_means = [base_stats[i]['mean'] if base_stats[i] else 0 for i in range(5)]
    ax1.bar(x-0.2, sig_means, width=0.4, color='steelblue', label='LME>=+1%日', edgecolor='black')
    ax1.bar(x+0.2, base_means, width=0.4, color='lightgray', label='全日ベース', edgecolor='black')
    ax1.set_xticks(x); ax1.set_xticklabels(DAYNAMES)
    ax1.axhline(0, color='black', lw=0.8)
    ax1.set_title('曜日別 平均PnL (bps)', fontweight='bold')
    ax1.legend(); ax1.grid(alpha=0.3, axis='y')
    for i, v in enumerate(sig_means):
        ax1.text(i-0.2, v+2, f'{v:+.0f}', ha='center', fontsize=9)

    ax2 = axes[0, 1]
    sig_sharpe = [sig_stats[i]['sharpe'] if sig_stats[i] else 0 for i in range(5)]
    base_sharpe = [base_stats[i]['sharpe'] if base_stats[i] else 0 for i in range(5)]
    ax2.bar(x-0.2, sig_sharpe, width=0.4, color='steelblue', label='LME>=+1%', edgecolor='black')
    ax2.bar(x+0.2, base_sharpe, width=0.4, color='lightgray', label='全日', edgecolor='black')
    ax2.set_xticks(x); ax2.set_xticklabels(DAYNAMES)
    ax2.axhline(0, color='black', lw=0.8)
    ax2.set_title('曜日別 Sharpe', fontweight='bold')
    ax2.legend(); ax2.grid(alpha=0.3, axis='y')

    ax3 = axes[1, 0]
    wrs = [sig_stats[i]['wr'] if sig_stats[i] else 0 for i in range(5)]
    bars = ax3.bar(x, wrs, color=['#1f77b4' if v>=50 else '#d62728' for v in wrs], edgecolor='black')
    ax3.axhline(50, color='black', lw=1, linestyle='--')
    ax3.set_xticks(x); ax3.set_xticklabels(DAYNAMES)
    ax3.set_ylabel('WR (%)')
    ax3.set_title('曜日別 勝率 (LME>=+1%日)', fontweight='bold')
    ax3.grid(alpha=0.3, axis='y')
    for b, v in zip(bars, wrs):
        ax3.text(b.get_x()+b.get_width()/2, v+1, f'{v:.0f}%', ha='center', fontsize=10)

    ax4 = axes[1, 1]
    ns = [sig_stats[i]['n'] if sig_stats[i] else 0 for i in range(5)]
    ax4.bar(x, ns, color='#2ca02c', edgecolor='black')
    ax4.set_xticks(x); ax4.set_xticklabels(DAYNAMES)
    ax4.set_title('曜日別 トレード数 (LME>=+1%日)', fontweight='bold')
    ax4.grid(alpha=0.3, axis='y')
    for i, v in enumerate(ns):
        ax4.text(i, v+0.5, str(v), ha='center', fontsize=10)

    plt.suptitle(f'戦略6: LMEシグナル曜日効果 (>=+1%, コア5銘柄Long, cost={U.COST_BPS}bps)',
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), 'result.png')
    plt.savefig(out, dpi=130, bbox_inches='tight')
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
