#!/usr/bin/env python3
"""
セクター別 US→JP 感応度分析 (2016-2026)

目的:
- 前日 .SOX / ESc1 / NQc1 急落時、日本のどのセクター/銘柄が最も反応するか?
- SOX Short 戦略のキャリア (1306 ETF) を差し替えて Sharpe 向上の余地はあるか?

データ:
- US指数: NAS MariaDB daily_data (.SOX, ESc1, NQc1, VXc1)
- JP個別株: yfinance (10年分)

出力:
- 前日 .SOX < -2% 時の Day N 寄→引 リターンを JP銘柄ごとに集計
- 銘柄別 Sharpe/t-stat ランキング
"""
import sys, os
from datetime import date
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['Hiragino Sans','Arial Unicode MS','sans-serif']
plt.rcParams['axes.unicode_minus'] = False

try:
    import pymysql
    import yfinance as yf
except ImportError as e:
    print(f"Missing: {e}")
    sys.exit(1)

MARIA = dict(host='100.92.181.92', port=3306, user='rfnews',
             password='Bleach@924', database='refinitiv_news')
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# 検証対象 JP 銘柄 (セクター代表)
JP_TICKERS = {
    # 半導体
    '8035.T':  ('東京エレクトロン', 'Semi'),
    '6857.T':  ('アドバンテスト',   'Semi'),
    '6146.T':  ('ディスコ',         'Semi'),
    '7735.T':  ('SCREEN HD',        'Semi'),
    '6920.T':  ('レーザーテック',   'Semi'),
    # テック/通信
    '6758.T':  ('ソニーG',          'Tech'),
    '6702.T':  ('富士通',           'Tech'),
    '9984.T':  ('ソフトバンクG',    'Tech'),
    # 自動車
    '7203.T':  ('トヨタ',           'Auto'),
    '7267.T':  ('ホンダ',           'Auto'),
    # 銀行・金融
    '8306.T':  ('MUFG',             'Bank'),
    '8316.T':  ('SMFG',             'Bank'),
    '8411.T':  ('みずほ',           'Bank'),
    # 商社・資源
    '5711.T':  ('三菱マテ',         'Metal'),
    '8058.T':  ('三菱商事',         'TradingCo'),
    # 鉄鋼・重工
    '5401.T':  ('日本製鉄',         'Steel'),
    '7011.T':  ('三菱重工',         'Heavy'),
    # 小売・ディフェンシブ
    '4502.T':  ('武田薬品',         'Pharma'),
    '9843.T':  ('ニトリ',           'Retail'),
    # ベンチマーク
    '1306.T':  ('TOPIX ETF',        'Index'),
    '1321.T':  ('日経225 ETF',      'Index'),
}

SOX_THRESHOLD = -2.0
COST_BPS = 4.0


def fetch_mariadb_sym(sym):
    conn = pymysql.connect(**MARIA, connect_timeout=10)
    q = "SELECT trade_date, close FROM daily_data WHERE symbol=%s ORDER BY trade_date"
    df = pd.read_sql(q, conn, params=[sym])
    conn.close()
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df = df.set_index('trade_date').sort_index()
    df['ret'] = df['close'].pct_change() * 100
    return df


def fetch_yf(ticker, start='2016-01-01'):
    df = yf.download(ticker, start=start, auto_adjust=False,
                     progress=False, multi_level_index=False)
    if df.empty:
        return None
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df[['Open', 'Close']].rename(columns={'Open':'open', 'Close':'close'})
    df['open_to_close_bps'] = (df['close'] / df['open'] - 1) * 10000
    df['close_to_close'] = df['close'].pct_change() * 100
    return df


def asof_prev(src_ret, target_dates):
    """target_date より strictly earlier な src の最新 ret を取得"""
    s = src_ret.dropna()
    idx = s.index.values
    out = []
    for d in target_dates:
        mask = idx < np.datetime64(d)
        out.append(s.loc[pd.Timestamp(idx[mask].max())] if mask.any() else np.nan)
    return np.array(out)


def stats(r_bps):
    r = r_bps.dropna().values
    n = len(r)
    if n == 0:
        return None
    m, s = r.mean(), r.std()
    wr = (r > 0).mean() * 100
    sharpe = (m/s) * np.sqrt(252) if s > 0 else 0
    tstat = m / (s/np.sqrt(n)) if s > 0 else 0
    return dict(n=n, mean=m, std=s, wr=wr, sharpe=sharpe, tstat=tstat)


def main():
    print("="*80)
    print("セクター別 US→JP 感応度分析")
    print("="*80)

    # 1) US指数取得 (MariaDB)
    print("\n[US指数取得 from MariaDB]")
    us = {}
    for sym in ['.SOX', 'ESc1', 'NQc1', 'VXc1']:
        df = fetch_mariadb_sym(sym)
        us[sym] = df
        print(f"  {sym}: N={len(df)}  {df.index.min().date()}〜{df.index.max().date()}")

    # 2) JP個別株取得 (yfinance)
    print("\n[JP個別株取得 from yfinance]")
    jp = {}
    for tk, (name, sector) in JP_TICKERS.items():
        df = fetch_yf(tk)
        if df is None or len(df) < 500:
            print(f"  {tk} {name}: ❌ データ不足")
            continue
        jp[tk] = df
        print(f"  {tk} {name} ({sector}): N={len(df)}  {df.index.min().date()}〜{df.index.max().date()}")

    # 3) 各JP銘柄について .SOX<-2% 条件下の Day N 寄→引 リターンを集計
    print("\n" + "="*80)
    print("[.SOX < -2% 条件下の翌営業日 寄→引 リターン]")
    print("="*80)
    print(f"\n{'銘柄':12} {'銘柄名':18} {'Sector':10} {'N':>5} {'Mean':>9} {'WR':>6} {'Sharpe':>8} {'t-stat':>8}")
    print("-"*90)

    rows = []
    sox_ret = us['.SOX']['ret']
    es_ret = us['ESc1']['ret']
    for tk, df in jp.items():
        name, sector = JP_TICKERS[tk]
        # Day N の寄→引リターン, 前日の .SOX ret
        dates = df.index
        prev_sox = asof_prev(sox_ret, dates)
        prev_es = asof_prev(es_ret, dates)
        # シグナル発動日のみ抽出
        mask = prev_sox <= SOX_THRESHOLD
        o2c = df['open_to_close_bps'].values
        # Short: 期待=下落 → 収益 = -o2c
        short_pnl = -o2c[mask] - COST_BPS
        if len(short_pnl) < 30:
            continue
        s = stats(pd.Series(short_pnl))
        s['ticker'] = tk
        s['name'] = name
        s['sector'] = sector
        rows.append(s)
        print(f"{tk:12} {name:18} {sector:10} {s['n']:>5} {s['mean']:>+8.1f}bp {s['wr']:>5.1f}% {s['sharpe']:>+7.2f} {s['tstat']:>+7.2f}")

    # ランキングDF
    rk = pd.DataFrame(rows).sort_values('sharpe', ascending=False)
    rk_path = os.path.join(OUT_DIR, 'sox_short_ranking.csv')
    rk.to_csv(rk_path, index=False)
    print(f"\nSaved ranking: {rk_path}")

    # 4) AND条件 (.SOX<-2% AND ES<-1%) の比較
    print("\n" + "="*80)
    print("[.SOX < -2% AND ES < -1% 条件下の結果]")
    print("="*80)
    print(f"\n{'銘柄':12} {'銘柄名':18} {'Sector':10} {'N':>5} {'Mean':>9} {'WR':>6} {'Sharpe':>8} {'t-stat':>8}")
    print("-"*90)

    rows2 = []
    for tk, df in jp.items():
        name, sector = JP_TICKERS[tk]
        dates = df.index
        prev_sox = asof_prev(sox_ret, dates)
        prev_es = asof_prev(es_ret, dates)
        mask = (prev_sox <= SOX_THRESHOLD) & (prev_es <= -1.0)
        o2c = df['open_to_close_bps'].values
        short_pnl = -o2c[mask] - COST_BPS
        if len(short_pnl) < 30:
            continue
        s = stats(pd.Series(short_pnl))
        s['ticker'] = tk
        s['name'] = name
        s['sector'] = sector
        rows2.append(s)
        print(f"{tk:12} {name:18} {sector:10} {s['n']:>5} {s['mean']:>+8.1f}bp {s['wr']:>5.1f}% {s['sharpe']:>+7.2f} {s['tstat']:>+7.2f}")

    rk2 = pd.DataFrame(rows2).sort_values('sharpe', ascending=False)
    rk2_path = os.path.join(OUT_DIR, 'sox_and_es_ranking.csv')
    rk2.to_csv(rk2_path, index=False)

    # 5) 可視化
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    for ax, df, title in [(axes[0], rk, '.SOX < -2% 単独'),
                          (axes[1], rk2, '.SOX<-2% AND ES<-1%')]:
        if df.empty:
            continue
        colors = ['#d62728' if s > 2 else '#ff9896' if s > 1 else '#aec7e8' if s > 0 else '#1f77b4'
                  for s in df['sharpe']]
        y_labels = [f"{t.ticker} {t.name}" for t in df.itertuples()]
        ax.barh(range(len(df)), df['sharpe'].values, color=colors)
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(y_labels, fontsize=9)
        ax.invert_yaxis()
        ax.axvline(2.0, color='red', linestyle='--', alpha=0.5, label='採用基準 Sharpe=2')
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_xlabel('Sharpe (annualized)')
        ax.set_title(f'翌日寄→引 Short Sharpe — {title}\n(N≥30, コスト4bps控除)')
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(axis='x', alpha=0.3)
        for i, (sh, n, tv) in enumerate(zip(df['sharpe'], df['n'], df['tstat'])):
            ax.text(sh + 0.05 if sh>=0 else sh-0.05, i, f"N={n} t={tv:+.1f}",
                    va='center', ha='left' if sh>=0 else 'right', fontsize=7)
    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, 'sector_sensitivity.png')
    plt.savefig(plot_path, dpi=120, bbox_inches='tight')
    print(f"Saved plot: {plot_path}")

    # 6) 1306 との相対比較 (採用判定)
    print("\n" + "="*80)
    print("[1306.T (TOPIX ETF) を基準とした相対比較]")
    print("="*80)
    base_1306 = rk[rk['ticker'] == '1306.T']
    if not base_1306.empty:
        base_sharpe = base_1306['sharpe'].values[0]
        print(f"\n  1306.T baseline Sharpe = {base_sharpe:+.2f}")
        better = rk[rk['sharpe'] > base_sharpe + 0.5].copy()
        if not better.empty:
            print(f"\n  1306 を +0.5 以上上回る銘柄 ({len(better)}件):")
            for r in better.itertuples():
                print(f"    {r.ticker} {r.name:15} ({r.sector:8}) Sharpe={r.sharpe:+.2f} (差分 {r.sharpe-base_sharpe:+.2f})")
        else:
            print("\n  1306 を大きく上回る銘柄なし → 現行の 1306.T ETF で妥当")


if __name__ == '__main__':
    main()
