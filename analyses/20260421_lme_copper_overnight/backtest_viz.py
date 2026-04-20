"""
コア5銘柄Long-only バスケット バックテスト & 可視化
構成: 5711 三菱マテリアル, 6501 日立, 7011 三菱重工, 5016 出光, 4502 武田
ルール: LME銅 open→JST15:25 変化が +1.0%以上 → 当該5銘柄を15:30引けで等加重Long → 翌朝9:00寄で決済
コスト: 片側2bps × 往復 = 4bps/銘柄/トレード
"""
import psycopg2
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['Hiragino Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
import matplotlib.dates as mdates
from datetime import date, time as dtime

PG_CONFIG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}
START = "2025-04-01"
END = "2026-04-21"
OUTLIER_PCT = 15.0
COST_BPS = 4.0
THRESHOLD = 1.0

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


def backtest_single(sig, jp, th):
    """ 1銘柄 Long-only: LMEアップ日(>=th)に引け買い→翌朝寄売り """
    jp_dates = sorted(jp.index)
    trades = []
    for i, d in enumerate(jp_dates[:-1]):
        if d not in sig.index: continue
        mv = sig.loc[d, 'move_pct']
        if mv < th: continue
        nd = jp_dates[i+1]
        entry = jp.loc[d, 'jp_close']
        exit_ = jp.loc[nd, 'jp_open']
        ret = (exit_/entry - 1) * 100
        if abs(ret) > OUTLIER_PCT: continue
        pnl_bps = ret * 100 - COST_BPS
        trades.append({'entry': d, 'exit': nd, 'ret_pct': ret, 'pnl_bps': pnl_bps})
    return pd.DataFrame(trades)


def main():
    sig = load_lme()
    print(f"LMEシグナル日数: {len(sig)}, アップ日(>=+{THRESHOLD}%): {(sig.move_pct>=THRESHOLD).sum()}")

    per_date = {}
    per_stock = {}
    for sym, name in CORE:
        jp = load_jp(sym)
        tdf = backtest_single(sig, jp, THRESHOLD)
        per_stock[sym] = tdf
        for _, r in tdf.iterrows():
            per_date.setdefault(r['entry'], []).append(r['pnl_bps'])

    basket_rows = [{'date': d, 'pnl_bps': np.mean(v), 'n_stocks': len(v)}
                   for d, v in sorted(per_date.items())]
    bdf = pd.DataFrame(basket_rows).set_index('date')
    bdf['cum_bps'] = bdf['pnl_bps'].cumsum()

    # 累積最高 → ドローダウン
    bdf['peak'] = bdf['cum_bps'].cummax()
    bdf['dd'] = bdf['cum_bps'] - bdf['peak']

    n = len(bdf)
    mean = bdf['pnl_bps'].mean()
    std = bdf['pnl_bps'].std()
    wr = (bdf['pnl_bps'] > 0).mean() * 100
    sharpe = mean / std * np.sqrt(252) if std > 0 else 0
    total = bdf['pnl_bps'].sum()
    maxdd = bdf['dd'].min()
    pos = bdf[bdf.pnl_bps > 0]['pnl_bps'].sum()
    neg = abs(bdf[bdf.pnl_bps <= 0]['pnl_bps'].sum())
    pf = pos / neg if neg > 0 else np.inf

    print(f"\n=== コア5銘柄Long-only バスケット (th={THRESHOLD}%) ===")
    print(f"取引日数: {n}, 期間: {bdf.index[0]} ~ {bdf.index[-1]}")
    print(f"平均PnL: {mean:+.1f} bps/日, 累積: {total:+.0f} bps")
    print(f"勝率: {wr:.1f}%, PF: {pf:.2f}, Sharpe: {sharpe:+.2f}")
    print(f"最大DD: {maxdd:.0f} bps")

    # === 可視化 ===
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.25)
    bdf_dt = pd.to_datetime(bdf.index)

    # 1. Equity curve
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(bdf_dt, bdf['cum_bps'], lw=2, color='#1f77b4', label='Cumulative PnL')
    ax1.fill_between(bdf_dt, bdf['cum_bps'], 0, alpha=0.2, color='#1f77b4')
    ax1.axhline(0, color='gray', lw=0.8)
    ax1.set_title(f'Equity Curve — Core5 Long-only Basket (th={THRESHOLD}%, cost={COST_BPS}bps)',
                  fontsize=13, fontweight='bold')
    ax1.set_ylabel('Cumulative PnL (bps)')
    ax1.grid(alpha=0.3)
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    txt = (f"N={n}  Total={total:+.0f}bps  Mean={mean:+.1f}bps/day  "
           f"WR={wr:.1f}%  PF={pf:.2f}  Sharpe={sharpe:+.2f}  MaxDD={maxdd:.0f}bps")
    ax1.text(0.01, 0.97, txt, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    # 2. Drawdown
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.fill_between(bdf_dt, bdf['dd'], 0, color='#d62728', alpha=0.5)
    ax2.plot(bdf_dt, bdf['dd'], color='#d62728', lw=1)
    ax2.set_title('Drawdown (bps)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('DD (bps)')
    ax2.grid(alpha=0.3)
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # 3. Monthly PnL bars
    ax3 = fig.add_subplot(gs[1, 1])
    bdf_m = bdf.copy()
    bdf_m.index = pd.to_datetime(bdf_m.index)
    monthly = bdf_m['pnl_bps'].resample('ME').sum()
    monthly = monthly[monthly != 0]
    colors = ['#2ca02c' if v > 0 else '#d62728' for v in monthly.values]
    ax3.bar(range(len(monthly)), monthly.values, color=colors, edgecolor='black')
    ax3.set_xticks(range(len(monthly)))
    ax3.set_xticklabels([d.strftime('%Y-%m') for d in monthly.index], rotation=45, ha='right')
    ax3.axhline(0, color='black', lw=0.8)
    ax3.set_title('Monthly PnL (bps)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('PnL (bps)')
    ax3.grid(alpha=0.3, axis='y')

    # 4. Per-trade PnL distribution
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.hist(bdf['pnl_bps'].values, bins=20, color='#1f77b4', edgecolor='black', alpha=0.75)
    ax4.axvline(0, color='black', lw=1.2)
    ax4.axvline(mean, color='red', lw=1.5, ls='--', label=f'Mean={mean:+.1f}')
    ax4.set_title('Per-day PnL Distribution', fontsize=12, fontweight='bold')
    ax4.set_xlabel('PnL (bps)')
    ax4.set_ylabel('Frequency')
    ax4.legend()
    ax4.grid(alpha=0.3)

    # 5. Per-stock sharpe bar
    ax5 = fig.add_subplot(gs[2, 1])
    stock_stats = []
    for sym, name in CORE:
        t = per_stock[sym]
        if len(t) == 0:
            stock_stats.append((f"{sym}\n{name}", 0, 0, 0)); continue
        m = t['pnl_bps'].mean(); s = t['pnl_bps'].std()
        sh = m/s*np.sqrt(252) if s > 0 else 0
        w = (t['pnl_bps'] > 0).mean() * 100
        stock_stats.append((f"{sym}\n{name}", m, sh, w))
    labels = [x[0] for x in stock_stats]
    sharpes = [x[2] for x in stock_stats]
    means = [x[1] for x in stock_stats]
    cols = ['#2ca02c' if v > 0 else '#d62728' for v in sharpes]
    bars = ax5.bar(labels, sharpes, color=cols, edgecolor='black')
    for bar, mv in zip(bars, means):
        ax5.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                 f'{mv:+.0f}bps', ha='center', fontsize=9)
    ax5.set_title('Per-stock Sharpe (annualized)', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Sharpe')
    ax5.grid(alpha=0.3, axis='y')
    ax5.axhline(0, color='black', lw=0.8)

    plt.suptitle(f'LME Copper → Japan Core5 Overnight Strategy Backtest\n'
                 f'{bdf.index[0]} ~ {bdf.index[-1]}  |  threshold={THRESHOLD}%, cost={COST_BPS}bps/trade',
                 fontsize=14, fontweight='bold', y=0.995)

    out = '/Users/Yusuke/claude-code/japan-stocks/analyses/20260421_lme_copper_overnight/backtest_viz.png'
    plt.savefig(out, dpi=130, bbox_inches='tight')
    print(f"\n保存: {out}")

    # CSV出力
    csv_out = '/Users/Yusuke/claude-code/japan-stocks/analyses/20260421_lme_copper_overnight/backtest_trades.csv'
    bdf.to_csv(csv_out)
    print(f"保存: {csv_out}")


if __name__ == "__main__":
    main()
