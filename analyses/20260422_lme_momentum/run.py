"""
戦略10: LME累積モメンタム
過去N日のLME銅の累積リターンがしきい値を超えたらコア5銘柄をLong。
LME単日シグナルではなく、トレンド継続を検証。
- lookback: 3, 5, 10日
- threshold: 累積>= +2%, +3%, +5%
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
LOOKBACKS = [3, 5, 10]
THRESHOLDS = [2.0, 3.0, 5.0]


def load_lme_daily():
    df = U.fetch_intraday('CMCU3').dropna(subset=['close'])
    daily = []
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
        daily.append({'date': d, 'open': ob['open'], 'close': cb['close']})
    ddf = pd.DataFrame(daily).set_index('date')
    ddf['intraday_pct'] = (ddf['close']/ddf['open']-1)*100
    return ddf


def backtest(lme, jp_all, lookback, th, side='long'):
    per_date = {}
    dates = sorted(lme.index)
    for i, d in enumerate(dates):
        if i < lookback: continue
        cum = lme['intraday_pct'].iloc[i-lookback:i].sum()
        ok = (cum >= th) if side=='long' else (cum <= -th)
        if not ok: continue
        # d から翌日 (d はシグナル判定日の引け後と同等)
        for sym, jp in jp_all.items():
            jp_dates = sorted(jp.index)
            if d not in jp.index: continue
            idx_in_jp = jp_dates.index(d)
            if idx_in_jp+1 >= len(jp_dates): continue
            nd = jp_dates[idx_in_jp+1]
            ret = (jp.loc[nd, 'open']/jp.loc[d, 'close'] - 1) * 100
            if abs(ret) > U.OUTLIER_PCT: continue
            pnl = (ret if side=='long' else -ret) * 100 - U.COST_BPS
            per_date.setdefault(d, []).append(pnl)
    rows = [{'date': d, 'pnl_bps': np.mean(v)} for d, v in sorted(per_date.items())]
    return pd.DataFrame(rows).set_index('date') if rows else pd.DataFrame()


def main():
    lme = load_lme_daily()
    jp_all = {sym: U.load_jp_daily(sym) for sym, _ in CORE}
    print(f"LME日次 N={len(lme)}, intraday mean={lme.intraday_pct.mean():+.2f}%")

    print("\n=== LME累積モメンタム (コア5銘柄Long) ===")
    results = {}
    print(f"{'LB':>3} {'Th':>5} {'N':>5} {'Mean':>7} {'WR':>6} {'Sharpe':>7}")
    for lb in LOOKBACKS:
        for th in THRESHOLDS:
            bdf = backtest(lme, jp_all, lb, th, 'long')
            if len(bdf) == 0:
                results[(lb, th)] = None; continue
            bdf['cum_bps'] = bdf['pnl_bps'].cumsum()
            st = U.compute_stats(bdf['pnl_bps'].values)
            results[(lb, th)] = {'bdf': bdf, 'stats': st}
            print(f"{lb:>3} {th:>5} {st['n']:>5} {st['mean']:>+6.1f} {st['wr']:>5.1f}% {st['sharpe']:>+6.2f}")

    # ネガティブ(モメンタム逆張り)も確認
    print("\n=== 逆張り: 累積下落 → Long (反発狙い) ===")
    results_rev = {}
    for lb in LOOKBACKS:
        for th in THRESHOLDS:
            bdf = backtest(lme, jp_all, lb, th, 'short')  # LME cumulative <= -th → reverse long
            # ここではside='short'で累積ダウン判定、Longポジ取るため pnl = -(-ret) = ret
            # 上記実装はshort→-ret、反転したい: cum<=-thでLong
            # なので再実装
            per_date = {}
            dates = sorted(lme.index)
            for i, d in enumerate(dates):
                if i < lb: continue
                cum = lme['intraday_pct'].iloc[i-lb:i].sum()
                if cum > -th: continue
                for sym, jp in jp_all.items():
                    jp_dates = sorted(jp.index)
                    if d not in jp.index: continue
                    idx_in_jp = jp_dates.index(d)
                    if idx_in_jp+1 >= len(jp_dates): continue
                    nd = jp_dates[idx_in_jp+1]
                    ret = (jp.loc[nd, 'open']/jp.loc[d, 'close'] - 1) * 100
                    if abs(ret) > U.OUTLIER_PCT: continue
                    pnl = ret * 100 - U.COST_BPS
                    per_date.setdefault(d, []).append(pnl)
            rows = [{'date': d, 'pnl_bps': np.mean(v)} for d, v in sorted(per_date.items())]
            if not rows:
                results_rev[(lb, th)] = None; continue
            bdf2 = pd.DataFrame(rows).set_index('date')
            bdf2['cum_bps'] = bdf2['pnl_bps'].cumsum()
            st2 = U.compute_stats(bdf2['pnl_bps'].values)
            results_rev[(lb, th)] = {'bdf': bdf2, 'stats': st2}
            print(f"LB={lb} cum<=-{th}% Long: N={st2['n']} Mean={st2['mean']:+.1f} Sharpe={st2['sharpe']:+.2f}")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Sharpe heatmap (momentum)
    ax1 = axes[0, 0]
    M = np.full((len(LOOKBACKS), len(THRESHOLDS)), np.nan)
    for i, lb in enumerate(LOOKBACKS):
        for j, th in enumerate(THRESHOLDS):
            r = results[(lb, th)]
            if r is not None and r['stats']['n'] >= 5:
                M[i, j] = r['stats']['sharpe']
    im = ax1.imshow(M, aspect='auto', cmap='RdYlGn', vmin=-5, vmax=5)
    ax1.set_xticks(range(len(THRESHOLDS))); ax1.set_xticklabels([f"+{t}%" for t in THRESHOLDS])
    ax1.set_yticks(range(len(LOOKBACKS))); ax1.set_yticklabels([f"{lb}d" for lb in LOOKBACKS])
    ax1.set_xlabel('累積閾値'); ax1.set_ylabel('Lookback')
    for i in range(len(LOOKBACKS)):
        for j in range(len(THRESHOLDS)):
            if not np.isnan(M[i, j]):
                r = results[(LOOKBACKS[i], THRESHOLDS[j])]
                ax1.text(j, i, f"{M[i,j]:+.1f}\nN={r['stats']['n']}", ha='center', va='center', fontsize=9)
    ax1.set_title('LME累積モメンタム Long Sharpe', fontweight='bold')
    plt.colorbar(im, ax=ax1)

    # 2. reverse (mean reversion)
    ax2 = axes[0, 1]
    M2 = np.full((len(LOOKBACKS), len(THRESHOLDS)), np.nan)
    for i, lb in enumerate(LOOKBACKS):
        for j, th in enumerate(THRESHOLDS):
            r = results_rev[(lb, th)]
            if r is not None and r['stats']['n'] >= 5:
                M2[i, j] = r['stats']['sharpe']
    im2 = ax2.imshow(M2, aspect='auto', cmap='RdYlGn', vmin=-5, vmax=5)
    ax2.set_xticks(range(len(THRESHOLDS))); ax2.set_xticklabels([f"-{t}%" for t in THRESHOLDS])
    ax2.set_yticks(range(len(LOOKBACKS))); ax2.set_yticklabels([f"{lb}d" for lb in LOOKBACKS])
    ax2.set_xlabel('累積閾値 (ダウン)'); ax2.set_ylabel('Lookback')
    for i in range(len(LOOKBACKS)):
        for j in range(len(THRESHOLDS)):
            if not np.isnan(M2[i, j]):
                r = results_rev[(LOOKBACKS[i], THRESHOLDS[j])]
                ax2.text(j, i, f"{M2[i,j]:+.1f}\nN={r['stats']['n']}", ha='center', va='center', fontsize=9)
    ax2.set_title('LME累積下落後 反発狙いLong Sharpe', fontweight='bold')
    plt.colorbar(im2, ax=ax2)

    # 3. 代表的な組み合わせのエクイティカーブ
    ax3 = axes[1, 0]
    for lb in LOOKBACKS:
        th = 3.0
        r = results[(lb, th)]
        if r is None: continue
        b = r['bdf']; st = r['stats']
        ax3.plot(pd.to_datetime(b.index), b['cum_bps'], lw=1.5,
                 label=f"LB={lb} +{th}% N={st['n']} Shp={st['sharpe']:+.1f}")
    ax3.set_title(f'モメンタム Long 累積PnL (cum>=+3%)', fontweight='bold')
    ax3.axhline(0, color='gray', lw=0.8); ax3.legend(fontsize=9); ax3.grid(alpha=0.3)
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))

    # 4. 反発狙いのエクイティカーブ
    ax4 = axes[1, 1]
    for lb in LOOKBACKS:
        th = 3.0
        r = results_rev[(lb, th)]
        if r is None: continue
        b = r['bdf']; st = r['stats']
        ax4.plot(pd.to_datetime(b.index), b['cum_bps'], lw=1.5,
                 label=f"LB={lb} -{th}% N={st['n']} Shp={st['sharpe']:+.1f}")
    ax4.set_title(f'反発狙いLong 累積PnL (cum<=-3%)', fontweight='bold')
    ax4.axhline(0, color='gray', lw=0.8); ax4.legend(fontsize=9); ax4.grid(alpha=0.3)
    ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))

    plt.suptitle(f'戦略10: LME累積モメンタム (コア5銘柄Long, cost={U.COST_BPS}bps)',
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), 'result.png')
    plt.savefig(out, dpi=130, bbox_inches='tight')
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
