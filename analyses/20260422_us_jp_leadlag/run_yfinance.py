"""
米国株 → 翌日日本市場の長期リードラグ分析 (日次)

目的:
  前日の米国株の動きが翌日の日本市場 (寄付ギャップ / イントラ / 終値) にどう影響するか
  長期日次データで体系的に検証する。

データ:
  yfinance経由で ^GSPC(S&P500), ^IXIC(Nasdaq), ^DJI(Dow), ^VIX, ^N225, 1306.T(TOPIX ETF)
  を2010年以降で取得。

分析:
  1. 基本統計 / 相関行列
  2. US前日リターン → JP翌日の分解 (gap / intraday / full)
  3. 分位数 (5分位) 分析
  4. 閾値戦略バックテスト (US>±X% → JPロング/ショート)
  5. 曜日 / 規制 / VIX高低での層化
  6. 時系列相関の安定性 (ローリングβ)
"""
import os, sys
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
plt.rcParams['font.family'] = ['Hiragino Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

OUT = os.path.dirname(os.path.abspath(__file__))
START = '2010-01-01'
END = '2026-04-22'

US_TICKERS = {
    '^GSPC': 'SP500',
    '^IXIC': 'Nasdaq',
    '^DJI':  'Dow',
    '^VIX':  'VIX',
}
JP_TICKERS = {
    '^N225':  'N225',
    '1306.T': 'TOPIX_ETF',  # TOPIX ETF as TOPIX proxy (long history)
}

COST_BPS = 4.0  # round-trip cost for backtests


def fetch(ticker):
    df = yf.download(ticker, start=START, end=END, progress=False, auto_adjust=False)
    if df.empty: return None
    # Flatten MultiIndex if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[['Open','High','Low','Close']].dropna()
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    return df


def build_dataset():
    print("[1/6] Fetching data ...")
    us = {}
    for t, name in US_TICKERS.items():
        d = fetch(t)
        if d is None:
            print(f"  {t}: FAILED")
            continue
        us[name] = d
        print(f"  {t:10s} {name:10s}  N={len(d)}  {d.index.min().date()} → {d.index.max().date()}")
    jp = {}
    for t, name in JP_TICKERS.items():
        d = fetch(t)
        if d is None:
            print(f"  {t}: FAILED"); continue
        jp[name] = d
        print(f"  {t:10s} {name:10s}  N={len(d)}  {d.index.min().date()} → {d.index.max().date()}")

    # Build aligned daily frame
    # US-side: prev-day (日本時間翌日から見た「前日」) returns
    #   us_close_to_close: close_{t-1 US日付}/close_{t-2 US日付}
    # JP-side: day t
    #   jp_gap   = (open_t / close_{t-1 JP}) - 1
    #   jp_intra = (close_t / open_t) - 1
    #   jp_full  = (close_t / close_{t-1 JP}) - 1
    # Key: the JP day t is influenced by US day t-1 (NY close日 = JP t-1 の夜)
    #   日本市場 open (t 09:00 JST) の時点でUS NY close (t-1 16:00 ET = t 05:00 JST) が確定済み
    #   → JP.day(t) を US.day(t-1) とペアリング
    # yfinanceの ^GSPC は取引日インデックスなので、JP day(t) の calendar_date を使って
    # US で t より前の直近取引日のリターンを引けばよい。

    # 1) 各シリーズに log return と pct return を付ける
    for name, d in {**us, **jp}.items():
        d['ret'] = d['Close'].pct_change()
        d['gap'] = d['Open']/d['Close'].shift(1) - 1
        d['intra'] = d['Close']/d['Open'] - 1

    # 2) アラインメント: JPのトレード日をマスターに
    n225 = jp['N225'].copy()
    n225 = n225.rename(columns={'Open':'jp_open','High':'jp_high','Low':'jp_low','Close':'jp_close',
                                 'ret':'jp_ret','gap':'jp_gap','intra':'jp_intra'})

    # asof merge: 各JP日に対して、「その日より厳密に前」の直近US営業日のclose_to_closeリターン
    def asof_prev(us_df, jp_dates, col='ret'):
        us_s = us_df[col].dropna()
        idx = us_s.index.values
        out = []
        for d in jp_dates:
            mask = idx < np.datetime64(d)
            if not mask.any():
                out.append(np.nan)
            else:
                last_idx = idx[mask].max()
                out.append(us_s.loc[pd.Timestamp(last_idx)])
        return out

    master = n225[['jp_open','jp_close','jp_ret','jp_gap','jp_intra']].copy()
    for name, d in us.items():
        master[f'us_{name}_ret'] = asof_prev(d, master.index, 'ret')
        if name == 'VIX':
            # VIX level too (not just return)
            master[f'us_VIX_level'] = asof_prev(d, master.index, 'Close')

    master = master.dropna()
    print(f"\n[2/6] Aligned dataset: N={len(master)}  "
          f"{master.index.min().date()} → {master.index.max().date()}")
    return master


def stats_section(df):
    print("\n[3/6] ===== 基本統計・相関 =====")
    for c in ['jp_gap','jp_intra','jp_ret']:
        x = df[c]*100
        print(f"  {c:10s}  mean={x.mean():+.3f}%  std={x.std():.3f}%  N={len(x)}")

    print("\n  相関 (US前日リターン vs JP翌日):")
    print(f"  {'US→':20s} {'JP Gap':>10s} {'JP Intra':>10s} {'JP Full':>10s}")
    rows = []
    for us_col in ['us_SP500_ret','us_Nasdaq_ret','us_Dow_ret','us_VIX_ret']:
        r_gap = df[us_col].corr(df['jp_gap'])
        r_int = df[us_col].corr(df['jp_intra'])
        r_full = df[us_col].corr(df['jp_ret'])
        print(f"  {us_col:20s} {r_gap:+10.3f} {r_int:+10.3f} {r_full:+10.3f}")
        rows.append({'us': us_col, 'r_gap': r_gap, 'r_intra': r_int, 'r_full': r_full})
    return pd.DataFrame(rows)


def quintile_analysis(df, us_col='us_SP500_ret'):
    print(f"\n[4/6] ===== 分位数分析 ({us_col}) =====")
    df = df.copy()
    df['q'] = pd.qcut(df[us_col], 5, labels=['Q1(底)','Q2','Q3','Q4','Q5(上)'])
    out = df.groupby('q', observed=True).agg(
        n=('jp_ret','count'),
        us_ret=(us_col, lambda x: x.mean()*100),
        jp_gap=('jp_gap', lambda x: x.mean()*100),
        jp_intra=('jp_intra', lambda x: x.mean()*100),
        jp_full=('jp_ret', lambda x: x.mean()*100),
        hit_rate=('jp_ret', lambda x: (x>0).mean()*100),
    )
    print(out.round(3).to_string())
    return out


def threshold_backtest(df, us_col, thresholds=(0.5, 1.0, 1.5, 2.0)):
    """US前日リターン ≥ +X% の時だけJP寄付Long→引け決済 のシンプル戦略"""
    print(f"\n[5/6] ===== 閾値戦略 ({us_col}) =====")
    print(f"  ルール: US前日リターン >= +X% → JP翌日 open→close Long (往復{COST_BPS}bps)")
    print(f"  {'X':>6s} {'N':>5s} {'Mean':>8s} {'Std':>7s} {'WR':>6s} {'Sharpe':>8s} {'t-stat':>8s}")
    rows = []
    for th in thresholds:
        sel = df[df[us_col] >= th/100.0]
        if len(sel) < 5:
            continue
        # JP intra long = open→close
        ret_bps = sel['jp_intra']*10000 - COST_BPS
        n = len(ret_bps); m = ret_bps.mean(); s = ret_bps.std()
        wr = (ret_bps>0).mean()*100
        sharpe = (m/s)*np.sqrt(252) if s>0 else 0
        tstat  = m/(s/np.sqrt(n)) if s>0 else 0
        print(f"  +{th:4.1f}% {n:5d} {m:+7.1f}bp {s:7.1f} {wr:5.1f}% {sharpe:+8.2f} {tstat:+8.2f}")
        rows.append(dict(side='long', th=th, n=n, mean_bp=m, sharpe=sharpe, tstat=tstat, wr=wr))
    print()
    for th in thresholds:
        sel = df[df[us_col] <= -th/100.0]
        if len(sel) < 5: continue
        ret_bps = -sel['jp_intra']*10000 - COST_BPS   # short
        n = len(ret_bps); m = ret_bps.mean(); s = ret_bps.std()
        wr = (ret_bps>0).mean()*100
        sharpe = (m/s)*np.sqrt(252) if s>0 else 0
        tstat  = m/(s/np.sqrt(n)) if s>0 else 0
        print(f"  -{th:4.1f}%S {n:5d} {m:+7.1f}bp {s:7.1f} {wr:5.1f}% {sharpe:+8.2f} {tstat:+8.2f}")
        rows.append(dict(side='short', th=th, n=n, mean_bp=m, sharpe=sharpe, tstat=tstat, wr=wr))
    return pd.DataFrame(rows)


def gap_vs_intra_decomposition(df):
    """US前日リターンのうち、JPギャップで何%埋まり、intraでどう継続/反転するか"""
    print("\n===== US影響の分解: Gap捕捉率 / Intra継続性 =====")
    print("  【Gap捕捉率】JP寄付ギャップ / US前日リターン (同符号時の平均比)")
    print("  【Intra継続性】寄付以降、US方向に継続(+)か反転(-)か")
    res = {}
    for us_col in ['us_SP500_ret','us_Nasdaq_ret','us_Dow_ret']:
        # 同符号の日に限定
        df2 = df[np.sign(df[us_col])==np.sign(df[us_col])]  # all
        mask_pos = df[us_col]>0
        mask_neg = df[us_col]<0
        # Gap capture: 寄付ギャップ / USリターン の平均
        cap_pos = (df.loc[mask_pos,'jp_gap'] / df.loc[mask_pos,us_col]).median()
        cap_neg = (df.loc[mask_neg,'jp_gap'] / df.loc[mask_neg,us_col]).median()
        # Intra continuation: US>0時にJPイントラも>0の確率
        cont_pos = (df.loc[mask_pos,'jp_intra']>0).mean()
        cont_neg = (df.loc[mask_neg,'jp_intra']<0).mean()
        # Intra mean
        intra_pos = df.loc[mask_pos,'jp_intra'].mean()*10000
        intra_neg = df.loc[mask_neg,'jp_intra'].mean()*10000
        print(f"  {us_col}:")
        print(f"    US>0日 (N={mask_pos.sum()}): gap_capture_median={cap_pos:.2%}  "
              f"JP intra継続率={cont_pos:.1%}  JP intra mean={intra_pos:+.1f}bp")
        print(f"    US<0日 (N={mask_neg.sum()}): gap_capture_median={cap_neg:.2%}  "
              f"JP intra継続率={cont_neg:.1%}  JP intra mean={intra_neg:+.1f}bp")
        res[us_col] = dict(cap_pos=cap_pos, cap_neg=cap_neg,
                           cont_pos=cont_pos, cont_neg=cont_neg,
                           intra_pos=intra_pos, intra_neg=intra_neg)
    return res


def vix_regime_analysis(df):
    print("\n===== VIXレジーム別のJP感応度 =====")
    df = df.copy()
    vix = df['us_VIX_level']
    df['vix_bin'] = pd.cut(vix, bins=[0,15,20,25,35,200],
                           labels=['低<15','15-20','20-25','25-35','高>35'])
    out = df.groupby('vix_bin', observed=True).apply(
        lambda g: pd.Series({
            'N': len(g),
            'corr_US_JPgap': g['us_SP500_ret'].corr(g['jp_gap']),
            'corr_US_JPfull': g['us_SP500_ret'].corr(g['jp_ret']),
            'beta': np.polyfit(g['us_SP500_ret'], g['jp_ret'], 1)[0] if len(g)>5 else np.nan,
        }), include_groups=False
    )
    print(out.round(3).to_string())
    return out


def rolling_beta(df, window=252):
    x = df['us_SP500_ret']; y = df['jp_ret']
    cov = x.rolling(window).cov(y)
    var = x.rolling(window).var()
    beta = cov / var
    return beta


def make_plots(df, corr_df, quint, bt, rbeta):
    print("\n[6/6] Making plots ...")
    fig, axes = plt.subplots(3, 2, figsize=(15, 14))

    # 1. Scatter: US前日 S&P500 vs JP翌日 gap
    ax = axes[0,0]
    ax.scatter(df['us_SP500_ret']*100, df['jp_gap']*100, s=4, alpha=0.25, color='steelblue')
    z = np.polyfit(df['us_SP500_ret']*100, df['jp_gap']*100, 1)
    xx = np.linspace(df['us_SP500_ret'].min()*100, df['us_SP500_ret'].max()*100, 50)
    ax.plot(xx, np.polyval(z,xx), 'r-', lw=2, label=f'β={z[0]:.2f}')
    r = df['us_SP500_ret'].corr(df['jp_gap'])
    ax.set_title(f'US S&P500(前日) → JP寄付ギャップ  (r={r:+.3f})')
    ax.set_xlabel('US S&P500 前日 close-to-close (%)')
    ax.set_ylabel('JP 寄付ギャップ (%)')
    ax.axhline(0, color='gray', lw=0.5); ax.axvline(0, color='gray', lw=0.5)
    ax.legend(); ax.grid(alpha=0.3)

    # 2. Scatter: US前日 S&P500 vs JP翌日 intra
    ax = axes[0,1]
    ax.scatter(df['us_SP500_ret']*100, df['jp_intra']*100, s=4, alpha=0.25, color='coral')
    z2 = np.polyfit(df['us_SP500_ret']*100, df['jp_intra']*100, 1)
    ax.plot(xx, np.polyval(z2,xx), 'r-', lw=2, label=f'β={z2[0]:.2f}')
    r2 = df['us_SP500_ret'].corr(df['jp_intra'])
    ax.set_title(f'US S&P500(前日) → JP寄→引 イントラ  (r={r2:+.3f})')
    ax.set_xlabel('US S&P500 前日 (%)')
    ax.set_ylabel('JP イントラ (%)')
    ax.axhline(0, color='gray', lw=0.5); ax.axvline(0, color='gray', lw=0.5)
    ax.legend(); ax.grid(alpha=0.3)

    # 3. Quintile bars
    ax = axes[1,0]
    x = np.arange(len(quint))
    w = 0.25
    ax.bar(x-w, quint['jp_gap'], w, label='JP Gap', color='steelblue')
    ax.bar(x,   quint['jp_intra'], w, label='JP Intra', color='coral')
    ax.bar(x+w, quint['jp_full'], w, label='JP Full', color='seagreen')
    ax.set_xticks(x); ax.set_xticklabels(quint.index)
    ax.set_title('US前日 S&P500 分位数別 JPリターン分解')
    ax.set_ylabel('Mean (%)')
    ax.axhline(0, color='black', lw=0.8)
    ax.legend(); ax.grid(alpha=0.3, axis='y')

    # 4. Correlation bars across US indices
    ax = axes[1,1]
    x = np.arange(len(corr_df))
    ax.bar(x-w, corr_df['r_gap'], w, label='→ JP Gap', color='steelblue')
    ax.bar(x,   corr_df['r_intra'], w, label='→ JP Intra', color='coral')
    ax.bar(x+w, corr_df['r_full'], w, label='→ JP Full', color='seagreen')
    ax.set_xticks(x); ax.set_xticklabels([c.replace('us_','').replace('_ret','') for c in corr_df['us']])
    ax.set_title('US指数別の翌日JP感応度 (相関係数)')
    ax.set_ylabel('相関係数')
    ax.axhline(0, color='black', lw=0.8); ax.legend(); ax.grid(alpha=0.3, axis='y')

    # 5. Rolling beta
    ax = axes[2,0]
    ax.plot(rbeta.index, rbeta.values, color='darkblue', lw=1.2)
    ax.set_title(f'ローリングβ (US S&P500 → JP N225, 252日窓)')
    ax.axhline(rbeta.mean(), color='red', lw=0.8, ls='--', label=f'平均 {rbeta.mean():.2f}')
    ax.set_ylabel('β'); ax.legend(); ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # 6. Threshold strategy Sharpe bar
    ax = axes[2,1]
    if not bt.empty:
        longs = bt[bt['side']=='long']
        shorts = bt[bt['side']=='short']
        xl = np.arange(len(longs)); xs = np.arange(len(shorts))
        ax.bar(xl-0.2, longs['sharpe'].values, 0.4, label='Long (US≥+X%)', color='green')
        if len(shorts):
            ax.bar(xs+0.2, shorts['sharpe'].values, 0.4, label='Short (US≤-X%)', color='crimson')
        ax.set_xticks(xl); ax.set_xticklabels([f'±{t}%' for t in longs['th']])
        ax.set_title('閾値戦略 Sharpe (US前日→JP寄引き)')
        ax.axhline(0, color='black', lw=0.8); ax.legend(); ax.grid(alpha=0.3, axis='y')

    plt.suptitle(f'米国株(前日) → 日本市場(翌日) 長期リードラグ分析  '
                 f'N={len(df)}  {df.index.min().date()}→{df.index.max().date()}',
                 fontsize=13, fontweight='bold', y=1.00)
    plt.tight_layout()
    path = os.path.join(OUT, 'result.png')
    plt.savefig(path, dpi=130, bbox_inches='tight')
    print(f"Saved: {path}")


def main():
    df = build_dataset()
    df.to_csv(os.path.join(OUT, 'dataset.csv'))
    corr_df = stats_section(df)
    quint = quintile_analysis(df, 'us_SP500_ret')
    bt = threshold_backtest(df, 'us_SP500_ret')
    _ = threshold_backtest(df, 'us_Nasdaq_ret')
    gap_vs_intra_decomposition(df)
    vix_regime_analysis(df)
    rbeta = rolling_beta(df)
    make_plots(df, corr_df, quint, bt, rbeta)


if __name__ == '__main__':
    main()
