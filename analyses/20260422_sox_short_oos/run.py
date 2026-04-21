"""
.SOX < -2% → 翌日 TOPIX Short 戦略 Out-of-Sample 検証

元分析: ../20260422_us_jp_leadlag/
  全期間 (2015-2026, N=2758): N=314, Sharpe+2.11, t=+2.36

本分析の検証項目:
  [A] H1/H2 均等分割 (時系列OoS)
  [B] 3分割 (2015-2018 / 2019-2022 / 2023-2026)
  [C] 閾値感応度 (-1.5 / -2.0 / -2.5 / -3.0%)
  [D] VIXレジーム条件 (VIX < 20 vs >= 20)
  [E] 曜日条件 (木曜除外で改善するか)
  [F] 出口タイミング (寄→引 vs 寄→後場寄付)
  [G] .SOX vs ESc1 vs 同時条件 (AND) の比較
  [H] 累積リターン/ドローダウン可視化
  [I] 採用判定
"""
import os, sys
import numpy as np
import pandas as pd
import pymysql
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
plt.rcParams['font.family'] = ['Hiragino Sans','Arial Unicode MS','sans-serif']
plt.rcParams['axes.unicode_minus'] = False

OUT = os.path.dirname(os.path.abspath(__file__))
COST_BPS = 4.0
MARIA = dict(host='100.92.181.92', port=3306, user='rfnews',
             password='Bleach@924', database='refinitiv_news')

SYMS = ['.SOX','ESc1','NQc1','VXc1','.TOPX','NKc1']


def fetch():
    conn = pymysql.connect(**MARIA)
    ph = ','.join(['%s']*len(SYMS))
    q = f"""SELECT symbol, trade_date, open, high, low, close
           FROM daily_data WHERE symbol IN ({ph})
           ORDER BY symbol, trade_date"""
    df = pd.read_sql(q, conn, params=SYMS)
    conn.close()
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    out = {}
    for s in SYMS:
        d = df[df['symbol']==s].set_index('trade_date')[['open','close']].astype(float)
        d = d.dropna(subset=['close']).sort_index()
        d['ret'] = d['close'].pct_change()
        d['gap'] = d['open']/d['close'].shift(1) - 1
        d['intra'] = d['close']/d['open'] - 1
        out[s] = d
    return out


def asof_prev(src, target_idx, col='ret'):
    """target_idx の各日に対し、厳密にそれ以前の直近 src[col] を返す"""
    s = src[col].dropna()
    idx = s.index.values
    out = []
    for d in target_idx:
        mask = idx < np.datetime64(d)
        out.append(s.loc[pd.Timestamp(idx[mask].max())] if mask.any() else np.nan)
    return out


def build_panel(data):
    m = data['.TOPX'][['open','close','ret','gap','intra']].copy()
    m.columns = ['jp_open','jp_close','jp_ret','jp_gap','jp_intra']
    m['sox_ret'] = asof_prev(data['.SOX'], m.index, 'ret')
    m['es_ret']  = asof_prev(data['ESc1'], m.index, 'ret')
    m['nq_ret']  = asof_prev(data['NQc1'], m.index, 'ret')
    m['vix_lvl'] = asof_prev(data['VXc1'], m.index, 'close')
    m['dow'] = m.index.day_name()
    return m.dropna()


def stats(r_bps):
    n = len(r_bps)
    if n == 0:
        return dict(n=0, mean=0, std=0, wr=0, sharpe=0, tstat=0, total=0, maxdd=0)
    m, s = r_bps.mean(), r_bps.std()
    wr = (r_bps > 0).mean() * 100
    sharpe = (m/s)*np.sqrt(252) if s > 0 else 0
    tstat = m/(s/np.sqrt(n)) if s > 0 else 0
    cum = np.cumsum(r_bps.values)
    maxdd = (cum - np.maximum.accumulate(cum)).min()
    return dict(n=n, mean=m, std=s, wr=wr, sharpe=sharpe, tstat=tstat,
                total=cum[-1], maxdd=maxdd)


def run_signal(df, cond, side='short'):
    sel = df[cond].copy()
    if side == 'short':
        r = -sel['jp_intra']*10000 - COST_BPS
    else:
        r = sel['jp_intra']*10000 - COST_BPS
    sel['ret_bps'] = r
    return sel


def print_stats_row(label, st):
    print(f"  {label:30s}  N={st['n']:4d}  mean={st['mean']:+7.2f}bp  "
          f"std={st['std']:6.1f}  WR={st['wr']:5.1f}%  "
          f"Sharpe={st['sharpe']:+6.2f}  t={st['tstat']:+5.2f}  "
          f"total={st['total']:+.0f}bp  MaxDD={st['maxdd']:+.0f}bp")


def main():
    print("=" * 110)
    print("  .SOX < -2% → 翌日 TOPIX Short  OoS検証")
    print("=" * 110)

    data = fetch()
    df = build_panel(data)
    print(f"\nパネル: N={len(df)}  {df.index.min().date()} → {df.index.max().date()}")

    # 基本シグナル
    base_cond = df['sox_ret'] <= -0.02
    base = run_signal(df, base_cond)
    print(f"\n[Full] .SOX <= -2%:")
    print_stats_row("Full 2015-2026", stats(base['ret_bps']))

    # ===== [A] H1/H2 =====
    print("\n" + "=" * 110)
    print("[A] H1/H2 均等分割 (時系列 OoS)")
    print("=" * 110)
    mid = df.index[len(df)//2]
    h1 = df[df.index < mid]
    h2 = df[df.index >= mid]
    print(f"  H1: {h1.index.min().date()} → {h1.index.max().date()}  (N={len(h1)})")
    print(f"  H2: {h2.index.min().date()} → {h2.index.max().date()}  (N={len(h2)})")
    print()
    for label, part in [('H1 (In-Sample)', h1), ('H2 (Out-of-Sample)', h2)]:
        s = run_signal(part, part['sox_ret'] <= -0.02)
        print_stats_row(label, stats(s['ret_bps']))

    # ===== [B] 3分割 =====
    print("\n" + "=" * 110)
    print("[B] 3分割 (walk-forward風)")
    print("=" * 110)
    cut1 = pd.Timestamp('2018-12-31'); cut2 = pd.Timestamp('2022-12-31')
    for label, mask in [
        ('P1 2015-2018', df.index <= cut1),
        ('P2 2019-2022', (df.index > cut1) & (df.index <= cut2)),
        ('P3 2023-2026', df.index > cut2),
    ]:
        part = df[mask]
        s = run_signal(part, part['sox_ret'] <= -0.02)
        print_stats_row(label, stats(s['ret_bps']))

    # ===== [C] 閾値感応度 =====
    print("\n" + "=" * 110)
    print("[C] 閾値感応度 (.SOX の閾値ごと、H1/H2分割)")
    print("=" * 110)
    for th in [-0.015, -0.02, -0.025, -0.03]:
        print(f"\n  閾値: .SOX <= {th*100:.1f}%")
        for label, part in [('  Full ', df), ('  H1   ', h1), ('  H2   ', h2)]:
            s = run_signal(part, part['sox_ret'] <= th)
            print_stats_row(label, stats(s['ret_bps']))

    # ===== [D] VIXレジーム条件 =====
    print("\n" + "=" * 110)
    print("[D] VIXレジーム条件 (.SOX <= -2% かつ VIX条件)")
    print("=" * 110)
    cond_base = df['sox_ret'] <= -0.02
    for label, cond in [
        ('VIX < 15 (低ボラ時の反応)', cond_base & (df['vix_lvl'] < 15)),
        ('VIX 15-20',                cond_base & (df['vix_lvl'] >= 15) & (df['vix_lvl'] < 20)),
        ('VIX 20-25',                cond_base & (df['vix_lvl'] >= 20) & (df['vix_lvl'] < 25)),
        ('VIX 25-35',                cond_base & (df['vix_lvl'] >= 25) & (df['vix_lvl'] < 35)),
        ('VIX >= 35 (パニック)',     cond_base & (df['vix_lvl'] >= 35)),
        ('VIX < 20 (低中ボラ)',      cond_base & (df['vix_lvl'] < 20)),
        ('VIX >= 20 (ストレス)',     cond_base & (df['vix_lvl'] >= 20)),
    ]:
        s = run_signal(df, cond)
        print_stats_row(label, stats(s['ret_bps']))

    # ===== [E] 曜日条件 =====
    print("\n" + "=" * 110)
    print("[E] 曜日条件 (.SOX <= -2% かつ 曜日フィルタ)")
    print("=" * 110)
    for dow in ['Monday','Tuesday','Wednesday','Thursday','Friday']:
        cond = cond_base & (df['dow'] == dow)
        s = run_signal(df, cond)
        print_stats_row(f"  エントリー={dow[:3]}", stats(s['ret_bps']))
    print()
    # 木曜除外
    s = run_signal(df, cond_base & (df['dow'] != 'Thursday'))
    print_stats_row("木曜除外", stats(s['ret_bps']))

    # ===== [F] 出口タイミング =====
    print("\n" + "=" * 110)
    print("[F] 出口タイミング比較 (寄→引 vs 別出口)")
    print("=" * 110)
    # 寄→引 (既定)
    sel = df[cond_base].copy()
    r1 = -sel['jp_intra']*10000 - COST_BPS
    print_stats_row("寄→引 (既定)", stats(r1))

    # close-to-close (フル日)
    r2 = -sel['jp_ret']*10000 - COST_BPS
    print_stats_row("前日close→翌日close (Full)", stats(r2))

    # gapのみ (寄付で即決済=不可能だが参考)
    r3 = -sel['jp_gap']*10000 - COST_BPS
    print_stats_row("前日close→翌日open (Gap only参考)", stats(r3))

    # ===== [G] SOX vs ES vs AND =====
    print("\n" + "=" * 110)
    print("[G] .SOX vs ESc1 vs AND条件")
    print("=" * 110)
    for label, cond in [
        ('.SOX <= -2%',           df['sox_ret'] <= -0.02),
        ('ESc1 <= -1%',           df['es_ret'] <= -0.01),
        ('ESc1 <= -1.5%',         df['es_ret'] <= -0.015),
        ('NQc1 <= -2%',           df['nq_ret'] <= -0.02),
        ('.SOX<=-2% AND ES<=-1%', (df['sox_ret']<=-0.02) & (df['es_ret']<=-0.01)),
        ('.SOX<=-2% AND ES<=-1.5%', (df['sox_ret']<=-0.02) & (df['es_ret']<=-0.015)),
        ('.SOX<=-2% OR ES<=-1.5%',  (df['sox_ret']<=-0.02) | (df['es_ret']<=-0.015)),
    ]:
        s = run_signal(df, cond)
        print_stats_row(label, stats(s['ret_bps']))

    # ===== [H] 累積リターン可視化 =====
    print("\n" + "=" * 110)
    print("[H] 可視化")
    print("=" * 110)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. 累積リターン H1/H2
    ax = axes[0, 0]
    for label, part, color in [('H1 In-Sample', h1, 'steelblue'),
                                ('H2 Out-of-Sample', h2, 'crimson')]:
        s = run_signal(part, part['sox_ret'] <= -0.02).sort_index()
        if len(s) == 0: continue
        cum = s['ret_bps'].cumsum()
        ax.plot(cum.index, cum.values, lw=2, color=color, label=f"{label} (N={len(s)})")
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_title('.SOX<=-2% → TOPIX Short 累積リターン (H1/H2)')
    ax.set_ylabel('累積 bps')
    ax.legend(); ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # 2. 閾値感応度 (Sharpe, 3期間)
    ax = axes[0, 1]
    ths = [-0.015, -0.02, -0.025, -0.03]
    periods = {'Full': df, 'H1': h1, 'H2': h2}
    colors = {'Full':'black','H1':'steelblue','H2':'crimson'}
    x = np.arange(len(ths)); w = 0.27
    for i,(label, part) in enumerate(periods.items()):
        vals = []
        for th in ths:
            s = run_signal(part, part['sox_ret'] <= th)
            st = stats(s['ret_bps'])
            vals.append(st['sharpe'])
        ax.bar(x + (i-1)*w, vals, w, label=label, color=colors[label])
    ax.set_xticks(x); ax.set_xticklabels([f'{t*100:.1f}%' for t in ths])
    ax.axhline(0, color='black', lw=0.5); ax.axhline(2, color='green', lw=0.5, ls='--', label='採用基準 Sharpe=2')
    ax.set_title('閾値感応度 (Sharpe, 期間別)')
    ax.set_ylabel('Sharpe (per-trade × √252)')
    ax.legend(); ax.grid(alpha=0.3, axis='y')

    # 3. VIXレジーム別 Sharpe
    ax = axes[1, 0]
    vix_bins = [('<15', df['vix_lvl']<15),
                ('15-20', (df['vix_lvl']>=15) & (df['vix_lvl']<20)),
                ('20-25', (df['vix_lvl']>=20) & (df['vix_lvl']<25)),
                ('25-35', (df['vix_lvl']>=25) & (df['vix_lvl']<35)),
                ('>=35', df['vix_lvl']>=35)]
    labels, shs, ns = [], [], []
    for lab, m in vix_bins:
        s = run_signal(df, cond_base & m)
        st = stats(s['ret_bps'])
        labels.append(lab); shs.append(st['sharpe']); ns.append(st['n'])
    bars = ax.bar(labels, shs, color=['lightgreen','green','orange','crimson','darkred'])
    for b, n in zip(bars, ns):
        ax.text(b.get_x()+b.get_width()/2, b.get_height(),
                f'N={n}', ha='center', va='bottom', fontsize=9)
    ax.axhline(0, color='black', lw=0.5); ax.axhline(2, color='green', lw=0.5, ls='--')
    ax.set_title('VIXレジーム別 Sharpe (.SOX<=-2%発動時)')
    ax.set_ylabel('Sharpe'); ax.grid(alpha=0.3, axis='y')

    # 4. Drawdown + trade scatter
    ax = axes[1, 1]
    sel = df[cond_base].copy().sort_index()
    r = -sel['jp_intra']*10000 - COST_BPS
    cum = r.cumsum()
    peak = np.maximum.accumulate(cum.values)
    dd = cum.values - peak
    ax.fill_between(cum.index, dd, 0, color='crimson', alpha=0.4, label='Drawdown')
    ax.plot(cum.index, cum.values, color='darkgreen', lw=1.5, label='Cumulative bps')
    ax.axhline(0, color='black', lw=0.5)
    ax.set_title('全期間 累積リターン + DD')
    ax.legend(); ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.suptitle('.SOX<=-2% → 翌日 TOPIX Short  OoS検証',
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    path = os.path.join(OUT, 'result.png')
    plt.savefig(path, dpi=130, bbox_inches='tight')
    print(f"\nSaved: {path}")

    # ===== [I] 採用判定 =====
    print("\n" + "=" * 110)
    print("[I] 採用判定 (Sharpe ≥ 2.0 & N ≥ 30 & t-stat ≥ 2.0)")
    print("=" * 110)
    full = stats(run_signal(df, cond_base)['ret_bps'])
    h1s  = stats(run_signal(h1, h1['sox_ret']<=-0.02)['ret_bps'])
    h2s  = stats(run_signal(h2, h2['sox_ret']<=-0.02)['ret_bps'])
    print(f"\n    基本 (.SOX<=-2%) の期間別成績:")
    print(f"      Full: Sharpe{full['sharpe']:+.2f} t{full['tstat']:+.2f} N={full['n']}")
    print(f"      H1:   Sharpe{h1s['sharpe']:+.2f} t{h1s['tstat']:+.2f} N={h1s['n']}")
    print(f"      H2:   Sharpe{h2s['sharpe']:+.2f} t{h2s['tstat']:+.2f} N={h2s['n']}")

    def verdict(s):
        ok = (s['sharpe'] >= 2.0) and (s['n'] >= 30) and (s['tstat'] >= 2.0)
        return '○ PASS' if ok else '× FAIL'

    print(f"\n    採用基準チェック:")
    print(f"      Full: {verdict(full)}")
    print(f"      H1:   {verdict(h1s)}")
    print(f"      H2:   {verdict(h2s)}  ← 最重要 (未来データ)")

    # データセット保存
    df.to_csv(os.path.join(OUT, 'panel.csv'))


if __name__ == '__main__':
    main()
