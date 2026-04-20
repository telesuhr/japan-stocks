"""
コア5銘柄Long-onlyバスケット 閾値比較バックテスト
閾値: 0.3% / 0.5% / 0.8% / 1.0% / 1.5%
"""
import psycopg2
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
plt.rcParams['font.family'] = ['Hiragino Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
from datetime import date, time as dtime

PG_CONFIG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}
START = "2025-04-01"
END = "2026-04-21"
OUTLIER_PCT = 15.0
COST_BPS = 4.0
THRESHOLDS = [0.3, 0.5, 0.8, 1.0, 1.5]

CORE = [
    ("5711.T", "三菱マテリアル"),
    ("6501.T", "日立"),
    ("7011.T", "三菱重工"),
    ("5016.T", "出光"),
    ("4502.T", "武田"),
]

BST = [
    (date(2024, 3, 31), date(2024, 10, 27)),
    (date(2025, 3, 30), date(2025, 10, 26)),
    (date(2026, 3, 29), date(2026, 10, 25)),
]


def is_bst(d):
    return any(s <= d < e for s, e in BST)


def load_lme():
    conn = psycopg2.connect(**PG_CONFIG)
    q = f"SELECT timestamp, open, close FROM intraday_data WHERE symbol='CMCU3' AND timestamp>='{START}' AND timestamp<'{END}' ORDER BY timestamp"
    df = pd.read_sql(q, conn); conn.close()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['jst'] = df['timestamp'] + pd.Timedelta(hours=9)
    df = df.dropna(subset=['close']).set_index('jst').sort_index()
    sig = []
    for d in sorted(set(df.index.date)):
        if d.weekday() >= 5: continue
        oh = 9 if is_bst(d) else 10
        ot = pd.Timestamp.combine(d, dtime(oh, 0))
        ct = pd.Timestamp.combine(d, dtime(15, 25))
        day = df[df.index.date == d]
        if len(day) == 0: continue
        after = day[day.index >= ot]
        before = day[day.index <= ct]
        if len(after) == 0 or len(before) == 0: continue
        ob = after.iloc[0]; cb = before.iloc[-1]
        if (ob.name - ot).total_seconds() > 1800: continue
        if (ct - cb.name).total_seconds() > 1800: continue
        sig.append({'date': d, 'move_pct': (cb['close']/ob['open']-1)*100})
    return pd.DataFrame(sig).set_index('date')


def load_jp(sym):
    conn = psycopg2.connect(**PG_CONFIG)
    q = f"SELECT timestamp, open, close FROM intraday_data WHERE symbol='{sym}' AND timestamp>='{START}' AND timestamp<'{END}' ORDER BY timestamp"
    df = pd.read_sql(q, conn); conn.close()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['jst'] = df['timestamp'] + pd.Timedelta(hours=9)
    df = df.dropna(subset=['open','close']).set_index('jst').sort_index()
    h, m = df.index.hour, df.index.minute
    df = df[((h==9)&(m<=5))|((h==15)&(m>=20)&(m<=30))]
    out = []
    for d in sorted(set(df.index.date)):
        gd = df[df.index.date == d]
        h2, m2 = gd.index.hour, gd.index.minute
        cl = gd[(h2==15)&(m2>=20)]; op = gd[(h2==9)&(m2<=5)]
        if len(cl)==0 or len(op)==0: continue
        out.append({'date': d, 'jp_close': cl['close'].iloc[-1], 'jp_open': op['open'].iloc[0]})
    return pd.DataFrame(out).set_index('date')


def backtest_basket(sig, jp_all, th):
    """コア5銘柄等加重Long-onlyバスケット"""
    per_date = {}
    for sym in jp_all:
        jp = jp_all[sym]
        dates = sorted(jp.index)
        for i, d in enumerate(dates[:-1]):
            if d not in sig.index: continue
            mv = sig.loc[d, 'move_pct']
            if mv < th: continue
            nd = dates[i+1]
            entry = jp.loc[d, 'jp_close']; exit_ = jp.loc[nd, 'jp_open']
            ret = (exit_/entry - 1) * 100
            if abs(ret) > OUTLIER_PCT: continue
            pnl = ret * 100 - COST_BPS
            per_date.setdefault(d, []).append(pnl)
    rows = [{'date': d, 'pnl_bps': np.mean(v), 'n_stocks': len(v)}
            for d, v in sorted(per_date.items())]
    return pd.DataFrame(rows).set_index('date')


def compute_stats(bdf):
    if len(bdf) == 0: return None
    arr = bdf['pnl_bps'].values
    m = arr.mean(); s = arr.std()
    cum = arr.cumsum()
    dd = (cum - np.maximum.accumulate(cum)).min()
    pos = arr[arr > 0].sum(); neg = abs(arr[arr <= 0].sum())
    return {
        'n': len(arr),
        'mean': m,
        'total': arr.sum(),
        'wr': (arr > 0).mean() * 100,
        'sharpe': m/s*np.sqrt(252) if s > 0 else 0,
        'pf': pos/neg if neg > 0 else np.inf,
        'maxdd': dd,
    }


def main():
    print("データロード...")
    sig = load_lme()
    jp_all = {sym: load_jp(sym) for sym, _ in CORE}
    print(f"LMEシグナル日数: {len(sig)}")
    for th in THRESHOLDS:
        print(f"  >=+{th}%: {(sig.move_pct>=th).sum()}日")

    # 各閾値でバックテスト
    results = {}
    for th in THRESHOLDS:
        bdf = backtest_basket(sig, jp_all, th)
        bdf['cum_bps'] = bdf['pnl_bps'].cumsum()
        bdf['peak'] = bdf['cum_bps'].cummax()
        bdf['dd'] = bdf['cum_bps'] - bdf['peak']
        results[th] = {'bdf': bdf, 'stats': compute_stats(bdf)}

    # コンソール出力
    print("\n" + "=" * 110)
    print(f"{'Threshold':<10} {'N':>5} {'Mean(bps)':>10} {'Total(bps)':>11} {'WR%':>6} "
          f"{'PF':>7} {'Sharpe':>7} {'MaxDD':>8}")
    print("-" * 110)
    for th in THRESHOLDS:
        st = results[th]['stats']
        print(f"{th}%{'':<7} {st['n']:>5} {st['mean']:>+9.1f} {st['total']:>+10.0f} "
              f"{st['wr']:>5.1f}% {st['pf']:>6.2f} {st['sharpe']:>+6.2f} {st['maxdd']:>+7.0f}")

    # === Figure ===
    fig = plt.figure(figsize=(17, 11))
    gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.22)
    colors = {0.3: '#999999', 0.5: '#ff7f0e', 0.8: '#2ca02c', 1.0: '#1f77b4', 1.5: '#d62728'}

    # 1. Equity curves 重ね描き
    ax1 = fig.add_subplot(gs[0, :])
    for th in THRESHOLDS:
        bdf = results[th]['bdf']; st = results[th]['stats']
        ax1.plot(pd.to_datetime(bdf.index), bdf['cum_bps'],
                 color=colors[th], lw=2,
                 label=f"th={th}%  N={st['n']}  Sharpe={st['sharpe']:+.2f}  Total={st['total']:+.0f}bps")
    ax1.axhline(0, color='gray', lw=0.8)
    ax1.set_title(f'コア5銘柄Long-only バスケット エクイティカーブ (閾値比較, cost={COST_BPS}bps)',
                  fontsize=13, fontweight='bold')
    ax1.set_ylabel('累積PnL (bps)')
    ax1.grid(alpha=0.3)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # 2. Drawdown 重ね描き
    ax2 = fig.add_subplot(gs[1, 0])
    for th in THRESHOLDS:
        bdf = results[th]['bdf']
        ax2.plot(pd.to_datetime(bdf.index), bdf['dd'],
                 color=colors[th], lw=1.3, label=f'{th}%')
    ax2.set_title('ドローダウン (bps)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('DD (bps)')
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=9)
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))

    # 3. 主要指標バー比較
    ax3 = fig.add_subplot(gs[1, 1])
    ths_s = [f"{t}%" for t in THRESHOLDS]
    sharpes = [results[t]['stats']['sharpe'] for t in THRESHOLDS]
    cols = [colors[t] for t in THRESHOLDS]
    bars = ax3.bar(ths_s, sharpes, color=cols, edgecolor='black')
    for bar, v in zip(bars, sharpes):
        ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
                 f'{v:+.2f}', ha='center', fontsize=10, fontweight='bold')
    ax3.set_title('Sharpe (閾値別)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Sharpe')
    ax3.grid(alpha=0.3, axis='y')

    # 4. トレード数 vs Mean PnL散布図的バー
    ax4 = fig.add_subplot(gs[2, 0])
    ns = [results[t]['stats']['n'] for t in THRESHOLDS]
    means = [results[t]['stats']['mean'] for t in THRESHOLDS]
    ax4b = ax4.twinx()
    ax4.bar([i-0.2 for i in range(len(THRESHOLDS))], ns, width=0.4,
            color='lightblue', edgecolor='black', label='N')
    ax4b.bar([i+0.2 for i in range(len(THRESHOLDS))], means, width=0.4,
             color='salmon', edgecolor='black', label='Mean bps')
    ax4.set_xticks(range(len(THRESHOLDS)))
    ax4.set_xticklabels(ths_s)
    ax4.set_ylabel('トレード日数', color='steelblue')
    ax4b.set_ylabel('平均PnL (bps)', color='red')
    ax4.set_title('頻度 vs 1トレードあたりの大きさ', fontsize=12, fontweight='bold')
    for i, (n, m) in enumerate(zip(ns, means)):
        ax4.text(i-0.2, n+1, str(n), ha='center', fontsize=9)
        ax4b.text(i+0.2, m+5, f'{m:+.0f}', ha='center', fontsize=9)

    # 5. Win rate / PF比較
    ax5 = fig.add_subplot(gs[2, 1])
    wrs = [results[t]['stats']['wr'] for t in THRESHOLDS]
    pfs = [min(results[t]['stats']['pf'], 20) for t in THRESHOLDS]  # PF capped for viz
    x = np.arange(len(THRESHOLDS))
    ax5.bar(x-0.2, wrs, width=0.4, color='lightgreen', edgecolor='black', label='WR (%)')
    ax5b = ax5.twinx()
    ax5b.bar(x+0.2, pfs, width=0.4, color='gold', edgecolor='black', label='PF')
    ax5.set_xticks(x); ax5.set_xticklabels(ths_s)
    ax5.set_ylabel('勝率 (%)', color='green')
    ax5b.set_ylabel('PF (capped 20)', color='darkorange')
    ax5.set_title('勝率 / プロフィットファクター', fontsize=12, fontweight='bold')
    for i, (w, p) in enumerate(zip(wrs, pfs)):
        ax5.text(i-0.2, w+1, f'{w:.0f}%', ha='center', fontsize=9)
        ax5b.text(i+0.2, p+0.3, f'{p:.1f}', ha='center', fontsize=9)

    plt.suptitle(f'閾値比較バックテスト: コア5銘柄Long-only '
                 f'({START} ~ {END}, cost={COST_BPS}bps/trade)',
                 fontsize=14, fontweight='bold', y=0.998)
    out = '/Users/Yusuke/claude-code/japan-stocks/analyses/20260421_lme_copper_overnight/threshold_compare.png'
    plt.savefig(out, dpi=130, bbox_inches='tight')
    print(f"\n保存: {out}")


if __name__ == "__main__":
    main()
