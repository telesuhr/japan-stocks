"""
米国(前日) → 日本(翌日) 長期リードラグ分析 — NAS MariaDB版

データソース: refinitiv_news.daily_data (NAS MariaDB, 2015-01〜2026-04, ~2850日)
    US先物: ESc1 (S&P500先物), NQc1 (Nasdaq先物), VXc1 (VIX先物)
    US現物: .SOX (Philly半導体)
    JP現物: .TOPX
    JP先物: NKc1, JNIc1
    FX:    JPY=, EURJPY=

先物ベースの長所:
  - NYクローズ後もグローバルに取引されるため、日本寄付時点の最新情報が反映
  - 特に .SOX → 日本半導体 (SOXは半導体銘柄が多い日経225の先行指標) の検証に有用
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

US_SYMS = {
    'ESc1': 'SP500_fut',
    'NQc1': 'Nasdaq_fut',
    'VXc1': 'VIX_fut',
    '.SOX': 'SOX',
}
JP_SYMS = {
    '.TOPX': 'TOPIX',
    'NKc1':  'N225_fut',
}


def fetch_all():
    conn = pymysql.connect(**MARIA)
    syms = list(US_SYMS) + list(JP_SYMS)
    placeholders = ','.join(['%s']*len(syms))
    q = f"""SELECT symbol, trade_date, open, high, low, close
           FROM daily_data WHERE symbol IN ({placeholders})
           ORDER BY symbol, trade_date"""
    df = pd.read_sql(q, conn, params=syms)
    conn.close()
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    out = {}
    for sym in syms:
        d = df[df['symbol']==sym].set_index('trade_date')[['open','high','low','close']].astype(float)
        d = d.dropna(subset=['close']).sort_index()
        if len(d) > 0:
            d['ret']   = d['close'].pct_change()
            d['gap']   = d['open']/d['close'].shift(1) - 1
            d['intra'] = d['close']/d['open'] - 1
            out[sym] = d
    return out


def build_aligned(data):
    """JP .TOPX をマスターに、各US指数の「JP日 t より厳密に前」の直近営業日 ret を結合"""
    master = data['.TOPX'][['open','close','ret','gap','intra']].copy()
    master.columns = ['jp_open','jp_close','jp_ret','jp_gap','jp_intra']

    for sym, d in data.items():
        if sym == '.TOPX': continue
        col_ret = d['ret'].dropna()
        idx = col_ret.index.values
        vals = []
        for dt in master.index:
            mask = idx < np.datetime64(dt)
            if not mask.any():
                vals.append(np.nan)
            else:
                last = idx[mask].max()
                vals.append(col_ret.loc[pd.Timestamp(last)])
        name = US_SYMS.get(sym) or JP_SYMS.get(sym)
        master[f'us_{name}_ret' if sym in US_SYMS else f'{name}_ret'] = vals

    # VIX level
    vix = data['VXc1']['close'].dropna()
    idx = vix.index.values
    vl = []
    for dt in master.index:
        mask = idx < np.datetime64(dt)
        vl.append(vix.loc[pd.Timestamp(idx[mask].max())] if mask.any() else np.nan)
    master['us_VIX_level'] = vl

    master = master.dropna()
    return master


def correlation_table(df):
    print("\n===== 相関行列 (US前日 → JP翌日) =====")
    print(f"N = {len(df)}")
    us_cols = [c for c in df.columns if c.startswith('us_') and c.endswith('_ret')]
    print(f"  {'US指数':25s} {'JP Gap':>10s} {'JP Intra':>10s} {'JP Full':>10s}")
    rows=[]
    for c in us_cols:
        rg = df[c].corr(df['jp_gap'])
        ri = df[c].corr(df['jp_intra'])
        rf = df[c].corr(df['jp_ret'])
        label = c.replace('us_','').replace('_ret','')
        print(f"  {label:25s} {rg:+10.3f} {ri:+10.3f} {rf:+10.3f}")
        rows.append(dict(us=label, r_gap=rg, r_intra=ri, r_full=rf))
    return pd.DataFrame(rows)


def quintile(df, us_col='us_SP500_fut_ret'):
    print(f"\n===== 分位数分析 ({us_col}) =====")
    df = df.copy()
    df['q'] = pd.qcut(df[us_col], 5, labels=['Q1(底)','Q2','Q3','Q4','Q5(上)'])
    out = df.groupby('q', observed=True).agg(
        n=('jp_ret','count'),
        us_ret=(us_col, lambda x: x.mean()*100),
        jp_gap=('jp_gap', lambda x: x.mean()*100),
        jp_intra=('jp_intra', lambda x: x.mean()*100),
        jp_full=('jp_ret', lambda x: x.mean()*100),
        hit=('jp_ret', lambda x: (x>0).mean()*100),
    )
    print(out.round(3).to_string())
    return out


def threshold_bt(df, us_col, label):
    print(f"\n===== 閾値戦略 ({label}) ・JP寄→引 =====")
    print(f"  {'side':>6s} {'X':>6s} {'N':>5s} {'Mean':>8s} {'Std':>7s} {'WR':>6s} {'Sharpe':>8s} {'t-stat':>8s}")
    rows=[]
    for th in [0.5,1.0,1.5,2.0,2.5]:
        sel = df[df[us_col] >= th/100.0]
        if len(sel) >= 10:
            r = sel['jp_intra']*10000 - COST_BPS
            n,m,s = len(r), r.mean(), r.std()
            sh = (m/s)*np.sqrt(252) if s>0 else 0
            t  = m/(s/np.sqrt(n)) if s>0 else 0
            wr = (r>0).mean()*100
            print(f"  {'Long':>6s} +{th:4.1f}% {n:5d} {m:+7.1f}bp {s:7.1f} {wr:5.1f}% {sh:+8.2f} {t:+8.2f}")
            rows.append(dict(side='long', th=th, n=n, mean=m, sharpe=sh, tstat=t, wr=wr))
    for th in [0.5,1.0,1.5,2.0,2.5]:
        sel = df[df[us_col] <= -th/100.0]
        if len(sel) >= 10:
            r = -sel['jp_intra']*10000 - COST_BPS
            n,m,s = len(r), r.mean(), r.std()
            sh = (m/s)*np.sqrt(252) if s>0 else 0
            t  = m/(s/np.sqrt(n)) if s>0 else 0
            wr = (r>0).mean()*100
            print(f"  {'Short':>6s} -{th:4.1f}% {n:5d} {m:+7.1f}bp {s:7.1f} {wr:5.1f}% {sh:+8.2f} {t:+8.2f}")
            rows.append(dict(side='short', th=th, n=n, mean=m, sharpe=sh, tstat=t, wr=wr))
    return pd.DataFrame(rows)


def decomposition(df):
    print("\n===== Gap捕捉率 / Intra継続性 =====")
    for c, lab in [('us_SP500_fut_ret','ESc1'),('us_Nasdaq_fut_ret','NQc1'),
                    ('us_SOX_ret','.SOX'),('us_VIX_fut_ret','VXc1')]:
        if c not in df.columns: continue
        mp = df[c]>0; mn = df[c]<0
        cap_p = (df.loc[mp,'jp_gap']/df.loc[mp,c]).median()
        cap_n = (df.loc[mn,'jp_gap']/df.loc[mn,c]).median()
        intra_p = df.loc[mp,'jp_intra'].mean()*10000
        intra_n = df.loc[mn,'jp_intra'].mean()*10000
        cont_p = (df.loc[mp,'jp_intra']>0).mean()
        cont_n = (df.loc[mn,'jp_intra']<0).mean()
        print(f"  {lab:8s}  US>0 N={mp.sum()}: cap_med={cap_p:.1%} intra_mean={intra_p:+.1f}bp 継続率={cont_p:.1%}")
        print(f"  {lab:8s}  US<0 N={mn.sum()}: cap_med={cap_n:.1%} intra_mean={intra_n:+.1f}bp 継続率={cont_n:.1%}")


def vix_regime(df):
    print("\n===== VIXレジーム別 US→JP 感応度 =====")
    df = df.copy()
    df['vix_bin'] = pd.cut(df['us_VIX_level'], [0,15,20,25,35,200],
                            labels=['低<15','15-20','20-25','25-35','高>35'])
    out = df.groupby('vix_bin', observed=True).apply(
        lambda g: pd.Series(dict(
            N=len(g),
            r_gap=g['us_SP500_fut_ret'].corr(g['jp_gap']),
            r_full=g['us_SP500_fut_ret'].corr(g['jp_ret']),
            beta=np.polyfit(g['us_SP500_fut_ret'], g['jp_ret'], 1)[0] if len(g)>5 else np.nan,
        )), include_groups=False
    )
    print(out.round(3).to_string())
    return out


def sox_analysis(df):
    """SOX (Philly半導体) → JP半導体への特殊分析"""
    print("\n===== .SOX(US半導体) → JP長期感応度 =====")
    if 'us_SOX_ret' not in df.columns:
        print("  SOXデータなし"); return
    rg = df['us_SOX_ret'].corr(df['jp_gap'])
    ri = df['us_SOX_ret'].corr(df['jp_intra'])
    rf = df['us_SOX_ret'].corr(df['jp_ret'])
    rs = df['us_SOX_ret'].corr(df['us_SP500_fut_ret'])
    print(f"  SOX vs ESc1 相関 (同日情報の重複度) : {rs:+.3f}")
    print(f"  SOX → JP gap : {rg:+.3f}")
    print(f"  SOX → JP intra : {ri:+.3f}")
    print(f"  SOX → JP full : {rf:+.3f}")
    # SOX単独シグナル (ESc1の影響を差し引いた residual)
    x = df['us_SOX_ret']; z = df['us_SP500_fut_ret']; y = df['jp_ret']
    # y ~ a*z → residual
    a = np.polyfit(z, y, 1)[0]
    resid_y = y - a*z
    b = np.polyfit(z, x, 1)[0]
    resid_x = x - b*z
    r_resid = pd.Series(resid_x).corr(pd.Series(resid_y))
    print(f"  SOXのresidual (ESを差し引き) → JP residual 相関: {r_resid:+.3f}")


def rolling_beta(df, col='us_SP500_fut_ret', win=252):
    x = df[col]; y = df['jp_ret']
    cov = x.rolling(win).cov(y)
    var = x.rolling(win).var()
    return cov/var


def make_plots(df, corr, quint, bt_es, rb):
    fig, axes = plt.subplots(3, 2, figsize=(15,14))

    ax = axes[0,0]
    ax.scatter(df['us_SP500_fut_ret']*100, df['jp_gap']*100, s=4, alpha=0.25, color='steelblue')
    z = np.polyfit(df['us_SP500_fut_ret']*100, df['jp_gap']*100, 1)
    xx = np.linspace(df['us_SP500_fut_ret'].min()*100, df['us_SP500_fut_ret'].max()*100, 50)
    ax.plot(xx, np.polyval(z,xx), 'r-', lw=2, label=f'β={z[0]:.2f}')
    r = df['us_SP500_fut_ret'].corr(df['jp_gap'])
    ax.set_title(f'ESc1(前日) → TOPIX寄付ギャップ r={r:+.3f}')
    ax.set_xlabel('ESc1 close-to-close (%)'); ax.set_ylabel('TOPIX gap (%)')
    ax.axhline(0,color='gray',lw=.5); ax.axvline(0,color='gray',lw=.5)
    ax.legend(); ax.grid(alpha=.3)

    ax = axes[0,1]
    ax.scatter(df['us_SP500_fut_ret']*100, df['jp_intra']*100, s=4, alpha=0.25, color='coral')
    z2 = np.polyfit(df['us_SP500_fut_ret']*100, df['jp_intra']*100, 1)
    ax.plot(xx, np.polyval(z2,xx), 'r-', lw=2, label=f'β={z2[0]:.2f}')
    r2 = df['us_SP500_fut_ret'].corr(df['jp_intra'])
    ax.set_title(f'ESc1(前日) → TOPIX イントラ r={r2:+.3f}')
    ax.set_xlabel('ESc1 (%)'); ax.set_ylabel('TOPIX intra (%)')
    ax.axhline(0,color='gray',lw=.5); ax.axvline(0,color='gray',lw=.5)
    ax.legend(); ax.grid(alpha=.3)

    ax = axes[1,0]
    x = np.arange(len(quint)); w = 0.25
    ax.bar(x-w, quint['jp_gap'], w, label='JP Gap', color='steelblue')
    ax.bar(x,   quint['jp_intra'], w, label='JP Intra', color='coral')
    ax.bar(x+w, quint['jp_full'], w, label='JP Full', color='seagreen')
    ax.set_xticks(x); ax.set_xticklabels(quint.index)
    ax.set_title('ESc1 分位数別 TOPIXリターン分解')
    ax.set_ylabel('Mean (%)'); ax.axhline(0,color='black',lw=.8)
    ax.legend(); ax.grid(alpha=.3, axis='y')

    ax = axes[1,1]
    x = np.arange(len(corr)); w = 0.25
    ax.bar(x-w, corr['r_gap'], w, label='→ JP Gap', color='steelblue')
    ax.bar(x,   corr['r_intra'], w, label='→ JP Intra', color='coral')
    ax.bar(x+w, corr['r_full'], w, label='→ JP Full', color='seagreen')
    ax.set_xticks(x); ax.set_xticklabels(corr['us'], rotation=15, ha='right')
    ax.set_title('US指数別 翌日TOPIX感応度 (相関)')
    ax.axhline(0,color='black',lw=.8); ax.legend(); ax.grid(alpha=.3, axis='y')

    ax = axes[2,0]
    ax.plot(rb.index, rb.values, color='darkblue', lw=1.2)
    ax.axhline(rb.mean(), color='red', lw=.8, ls='--', label=f'平均 {rb.mean():.2f}')
    ax.set_title('ローリングβ (ESc1→TOPIX, 252日)')
    ax.set_ylabel('β'); ax.legend(); ax.grid(alpha=.3)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    ax = axes[2,1]
    if not bt_es.empty:
        longs  = bt_es[bt_es['side']=='long'].reset_index(drop=True)
        shorts = bt_es[bt_es['side']=='short'].reset_index(drop=True)
        xl = np.arange(len(longs))
        ax.bar(xl-0.2, longs['sharpe'].values, 0.4, label='Long (ES≥+X%)', color='green')
        xs = np.arange(len(shorts))
        ax.bar(xs+0.2, shorts['sharpe'].values, 0.4, label='Short (ES≤-X%)', color='crimson')
        ax.set_xticks(xl); ax.set_xticklabels([f'±{t}%' for t in longs['th']])
        ax.set_title('閾値戦略 Sharpe (ESc1前日→TOPIX寄引)')
        ax.axhline(0,color='black',lw=.8); ax.legend(); ax.grid(alpha=.3, axis='y')

    plt.suptitle(f'米(前日)→日(翌日) 長期リードラグ [NAS MariaDB]  '
                 f'N={len(df)}  {df.index.min().date()}→{df.index.max().date()}',
                 fontsize=13, fontweight='bold', y=1.00)
    plt.tight_layout()
    out = os.path.join(OUT, 'result.png')
    plt.savefig(out, dpi=130, bbox_inches='tight')
    print(f"\nSaved: {out}")


def main():
    print("[1/6] Fetching from NAS MariaDB ...")
    data = fetch_all()
    for s,d in data.items():
        print(f"  {s:8s}  N={len(d)}  {d.index.min().date()} → {d.index.max().date()}")

    print("\n[2/6] Aligning dataset ...")
    df = build_aligned(data)
    df.to_csv(os.path.join(OUT, 'dataset_maria.csv'))
    print(f"  N={len(df)}  {df.index.min().date()} → {df.index.max().date()}")

    print("\n[3/6] Correlations ...")
    corr = correlation_table(df)

    print("\n[4/6] Quintile & thresholds ...")
    q = quintile(df, 'us_SP500_fut_ret')
    bt_es = threshold_bt(df, 'us_SP500_fut_ret', 'ESc1')
    _     = threshold_bt(df, 'us_Nasdaq_fut_ret', 'NQc1')
    if 'us_SOX_ret' in df.columns:
        _ = threshold_bt(df, 'us_SOX_ret', '.SOX')

    print("\n[5/6] Decomposition & regimes ...")
    decomposition(df)
    vix_regime(df)
    sox_analysis(df)

    print("\n[6/6] Plotting ...")
    rb = rolling_beta(df)
    make_plots(df, corr, q, bt_es, rb)


if __name__ == '__main__':
    main()
