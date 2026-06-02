"""
LME銅シグナル戦略 エッジ再検証 (2026-06-01)

前回分析(2026-04-22)の問題点を網羅的に検証:
  1. True OOS: 2026-04-22〜2026-05-21 (CMCU3データ終端まで)
  2. Walk-forward OOS (月次ロール)
  3. Permutation test (N不足・偶然性の定量化)
  4. Selection bias: コア5はIS期間で後出し選択された
  5. 取引コスト感応度 (4 / 8 / 12bps)
  6. 市場ベータ対比

データソース:
  シグナル : nas_archive.intraday_data (CMCU3, 5分足, 旧Refinitiv)
              ※前回と同一データソース。OOS期間(〜2026-05-21)まで有効
  日本株   : stocks_daily adj_close / adj_open (JQuants, 分割調整済)
              ※前回は intraday_data close (非調整) → 異常値フィルタが必要だった
              　今回は adj 価格を使うため分割調整不整合は発生しない

前回との差分:
  - 日本株を adj_close/adj_open に切り替え → outlier フィルタ不要
  - 真のOOS期間 (2026-04-22〜2026-05-21) を追加
  - Walk-forward / Permutation / Selection bias を追加
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
import os
import warnings
warnings.filterwarnings('ignore')

import psycopg2
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import date, time as dtime
from scipy import stats

# ─── 設定 ────────────────────────────────────────────
PG_CONFIG = {
    "host": os.environ.get("PGHOST", "192.168.0.118"),
    "port": int(os.environ.get("PGPORT", 5432)),
    "user": os.environ.get("PGUSER", "postgres"),
    "password": os.environ.get("PGPASSWORD", "postgres"),
    "dbname": os.environ.get("PGDATABASE", "market_data"),
}

FULL_START  = "2025-01-01"   # ウォームアップ含むデータ取得開始
IS_START    = "2025-04-01"   # In-sample 開始
IS_END      = "2026-04-21"   # In-sample 終了 (前回分析の最終日)
OOS_START   = "2026-04-22"   # True OOS 開始
OOS_END     = "2026-05-21"   # True OOS 終了 (CMCU3データ終端)
FULL_END    = OOS_END

LME_THRESHOLD = 1.0          # LMEシグナル閾値 (%)
COST_SCENARIOS = [4, 8, 12]  # 往復コスト候補 (bps)
PRIMARY_COST   = 4           # メイン分析に使うコスト

# BST期間 (UK夏時間: UTC+1)
BST_PERIODS = [
    (date(2025, 3, 30), date(2025, 10, 26)),
    (date(2026, 3, 29), date(2026, 10, 25)),
]

# コア5銘柄 (IS期間で後出し選択)
CORE5_RICS  = ["5711.T", "6501.T", "7011.T", "5016.T", "4502.T"]
CORE5_NAMES = {"5711.T": "三菱マテリアル", "6501.T": "日立",
               "7011.T": "三菱重工",       "5016.T": "出光",   "4502.T": "武田"}

# 広域スクリーニング (Selection bias 検証用)
SCREEN_RICS = [
    "5706.T","5711.T","5713.T","5714.T","5401.T","5411.T",
    "8035.T","6857.T","6920.T","6146.T","4063.T","6963.T",
    "7203.T","7267.T","7011.T","7012.T","7013.T",
    "8001.T","8002.T","8031.T","8053.T","8058.T",
    "8306.T","8316.T","8411.T",
    "9101.T","9104.T","9107.T",
    "1605.T","5020.T","5016.T",
    "4502.T","4503.T","4523.T",
    "6301.T","6305.T","6367.T",
    "9432.T","9433.T",
    "8801.T","8802.T",
    "8267.T","7974.T","6758.T",
    "9984.T","9983.T",
    "6501.T","6503.T","6506.T",
    "6702.T","6098.T","4578.T",
    "8604.T","8750.T","8766.T",
]

# ─── ヘルパー ─────────────────────────────────────────
def get_conn():
    return psycopg2.connect(**PG_CONFIG)

def is_bst(d):
    for s, e in BST_PERIODS:
        if s <= d < e:
            return True
    return False

# ─── CMCU3 シグナル計算 ───────────────────────────────
def load_lme_signals(start=FULL_START, end=FULL_END):
    """
    nas_archive.intraday_data の CMCU3 5分足から
    「LME電子市場オープン(JST 9:00/10:00) 〜 JST 15:25 の変化率」を計算
    ※ 前回の load_lme_signals() と同一ロジック
    """
    conn = get_conn()
    q = f"""
        SELECT timestamp, open, close
        FROM nas_archive.intraday_data
        WHERE symbol='CMCU3'
          AND timestamp >= '{start}'::timestamp - interval '1 day'
          AND timestamp < '{end}'::timestamp + interval '1 day'
        ORDER BY timestamp
    """
    df = pd.read_sql(q, conn)
    conn.close()

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['jst'] = df['timestamp'] + pd.Timedelta(hours=9)
    df = df.dropna(subset=['close']).set_index('jst').sort_index()

    signals = []
    for d in sorted(set(df.index.date)):
        if d.weekday() >= 5:
            continue
        open_hour = 9 if is_bst(d) else 10
        open_target  = pd.Timestamp.combine(d, dtime(open_hour, 0))
        close_target = pd.Timestamp.combine(d, dtime(15, 25))

        day = df[df.index.date == d]
        if len(day) == 0:
            continue

        # オープン: target から 30分以内に最初のバー
        after_open = day[day.index >= open_target]
        if len(after_open) == 0:
            continue
        ob = after_open.iloc[0]
        if (ob.name - open_target).total_seconds() > 1800:
            continue

        # クローズ: target 直前の最後のバー
        before_close = day[day.index <= close_target]
        if len(before_close) == 0:
            continue
        cb = before_close.iloc[-1]
        if (close_target - cb.name).total_seconds() > 1800:
            continue

        move_pct = (cb['close'] / ob['open'] - 1) * 100
        signals.append({'date': d, 'move_pct': move_pct})

    sig_df = pd.DataFrame(signals).set_index('date')
    return sig_df

# ─── 日本株 overnight return ─────────────────────────
def get_code5_map(rics):
    conn = get_conn()
    rics_str = "','".join(set(rics))
    q = f"SELECT ric, code5 FROM symbol_master WHERE ric IN ('{rics_str}')"
    df = pd.read_sql(q, conn)
    conn.close()
    return dict(zip(df['ric'], df['code5']))

def load_overnight_returns(code5s, start=FULL_START, end=FULL_END):
    """
    stocks_daily の adj_close / adj_open を使用
    overnight_ret_bps = (翌営業日 adj_open / 当日 adj_close - 1) × 10000
    分割調整済みのため outlier フィルタ不要
    """
    conn = get_conn()
    codes_str = "','".join(code5s)
    q = f"""
        SELECT code, date, adj_close, adj_open
        FROM stocks_daily
        WHERE code IN ('{codes_str}')
          AND date >= '{start}' AND date <= '{end}'
        ORDER BY code, date
    """
    df = pd.read_sql(q, conn, parse_dates=['date'])
    conn.close()
    df = df.sort_values(['code', 'date'])
    df['next_adj_open'] = df.groupby('code')['adj_open'].shift(-1)
    df = df.dropna(subset=['next_adj_open', 'adj_close'])
    # 調整済みでも極端な値(>50%)は除外 (上場廃止・合併等)
    df['overnight_ret_bps'] = (df['next_adj_open'] / df['adj_close'] - 1) * 10000
    df = df[df['overnight_ret_bps'].abs() <= 5000]
    return df

# ─── バックテスト ─────────────────────────────────────
def backtest_basket(sig_df, on_df, code5s, start, end, cost_bps=PRIMARY_COST, threshold=LME_THRESHOLD):
    """コア5等加重バスケットのオーバーナイトバックテスト"""
    period_sig = sig_df.loc[(sig_df.index >= pd.to_datetime(start).date()) &
                             (sig_df.index <= pd.to_datetime(end).date())]
    signal_days = period_sig[period_sig['move_pct'] >= threshold].index

    on_period = on_df[(on_df['date'] >= start) & (on_df['date'] <= end)]
    on_period = on_period[on_period['code'].isin(code5s)]
    on_pivot  = on_period.pivot(index='date', columns='code', values='overnight_ret_bps')

    trades = []
    for sig_date in signal_days:
        ts = pd.Timestamp(sig_date)
        if ts not in on_pivot.index:
            continue
        row = on_pivot.loc[ts, [c for c in code5s if c in on_pivot.columns]].dropna()
        if len(row) == 0:
            continue
        basket_ret = row.mean()
        trades.append({
            'date': ts,
            'lme_move_pct': period_sig.loc[sig_date, 'move_pct'],
            'gross_bps': basket_ret,
            'pnl_bps': basket_ret - cost_bps,
            'n_stocks': len(row),
        })
    return pd.DataFrame(trades)

def evaluate(tdf):
    if tdf is None or len(tdf) == 0:
        return {'n': 0, 'sharpe': np.nan, 'wr': np.nan,
                'mean_bps': np.nan, 'total_bps': 0, 'tstat': np.nan}
    arr = tdf['pnl_bps'].values
    n   = len(arr)
    mean = arr.mean()
    std  = arr.std(ddof=1) if n > 1 else 0
    sharpe = mean / std * np.sqrt(252) if std > 0 else np.nan
    wr     = (arr > 0).mean() * 100
    tstat  = stats.ttest_1samp(arr, 0).statistic if n >= 2 else np.nan
    return {'n': n, 'sharpe': sharpe, 'wr': wr,
            'mean_bps': mean, 'total_bps': arr.sum(), 'tstat': tstat}

# ─── 分析1: IS vs True OOS ───────────────────────────
def analysis_is_vs_oos(sig_df, on_df, code5s, cost_bps=PRIMARY_COST):
    print("\n" + "="*65)
    print("  [分析1] IS vs True OOS 直接比較")
    print("="*65)

    results = {}
    for label, s, e in [("IS  (2025-04〜2026-04)", IS_START, IS_END),
                         ("OOS (2026-04-22〜2026-05-21)", OOS_START, OOS_END)]:
        period_sig = sig_df.loc[(sig_df.index >= pd.to_datetime(s).date()) &
                                 (sig_df.index <= pd.to_datetime(e).date())]
        sig_n = int((period_sig['move_pct'] >= LME_THRESHOLD).sum())
        sig_days_all = len(period_sig)

        tdf = backtest_basket(sig_df, on_df, code5s, s, e, cost_bps)
        r = evaluate(tdf)
        results[label] = (tdf, r)

        print(f"\n  [{label}]")
        print(f"    LMEシグナル日: {sig_n}/{sig_days_all}日")
        if r['n'] > 0:
            sig_flag = '★ t>=2.0' if not np.isnan(r['tstat']) and abs(r['tstat']) >= 2.0 else '  t<2.0'
            print(f"    N={r['n']:2d}  WR={r['wr']:.1f}%  Mean={r['mean_bps']:+.1f}bps  "
                  f"Total={r['total_bps']:+.0f}bps")
            print(f"    Sharpe={r['sharpe']:+.2f}  t-stat={r['tstat']:+.2f}  {sig_flag}")
            if r['n'] < 8:
                print(f"    [!] N={r['n']} -- サンプル少。数値は参考値。")
        else:
            print(f"    トレードなし")

    return results["IS  (2025-04〜2026-04)"][0]

# ─── 分析2: Walk-forward ─────────────────────────────
def analysis_walkforward(sig_df, on_df, code5s, cost_bps=PRIMARY_COST):
    print("\n" + "="*65)
    print("  [分析2] Walk-forward OOS (月次ロール)")
    print("="*65)

    months = pd.date_range("2025-10-01", "2026-05-01", freq="MS")
    rows = []
    for oos_start in months:
        oos_end = oos_start + pd.DateOffset(months=1) - pd.Timedelta(days=1)
        oos_end = min(oos_end, pd.Timestamp(OOS_END))

        period_sig = sig_df.loc[(sig_df.index >= oos_start.date()) &
                                  (sig_df.index <= oos_end.date())]
        sig_n = int((period_sig['move_pct'] >= LME_THRESHOLD).sum())

        tdf = backtest_basket(sig_df, on_df, code5s,
                              str(oos_start.date()), str(oos_end.date()), cost_bps)
        r = evaluate(tdf)
        rows.append({
            'month': oos_start.strftime('%Y-%m'),
            'sig_n': sig_n,
            'n': r['n'],
            'mean_bps': r['mean_bps'],
            'total_bps': r['total_bps'],
        })

    wf_df = pd.DataFrame(rows)
    print(f"\n  {'月':^8} {'シグナル':>7} {'N':>4} {'Mean(bps)':>10} {'月次PnL(bps)':>12}")
    print("  " + "-"*45)
    for _, row in wf_df.iterrows():
        if row['n'] > 0:
            sign = '+' if row['total_bps'] > 0 else ' '
            print(f"  {row['month']:^8} {row['sig_n']:>7.0f} {row['n']:>4.0f} "
                  f"{row['mean_bps']:>+10.1f} {row['total_bps']:>+12.0f}")
        else:
            print(f"  {row['month']:^8} {row['sig_n']:>7.0f} {'--':>4} {'--':>10} {'--':>12}")

    active = wf_df[wf_df['n'] > 0]
    if len(active) > 0:
        n_pos = (active['total_bps'] > 0).sum()
        print(f"\n  取引あり月: {len(active)}ヶ月中 {n_pos}ヶ月が黒字  "
              f"({n_pos/len(active)*100:.0f}%)")
    return wf_df

# ─── 分析3: Permutation Test ─────────────────────────
def analysis_permutation(sig_df, on_df, code5s, cost_bps=PRIMARY_COST, n_perm=5000):
    print("\n" + "="*65)
    print(f"  [分析3] Permutation Test (シャッフル {n_perm:,}回)")
    print("="*65)
    print("  「シグナル日をランダムに選んでも同じSharpeが出るか?」を検定")

    tdf_real = backtest_basket(sig_df, on_df, code5s, IS_START, IS_END, cost_bps)
    r_real   = evaluate(tdf_real)
    N_signal = r_real['n']
    real_sharpe = r_real['sharpe']

    print(f"\n  実際のIS結果: N={N_signal}  Sharpe={real_sharpe:+.2f}  "
          f"Mean={r_real['mean_bps']:+.1f}bps")

    # IS期間の全日本株trading daysでon_pivot作成
    on_period = on_df[(on_df['date'] >= IS_START) & (on_df['date'] <= IS_END)]
    on_period = on_period[on_period['code'].isin(code5s)]
    on_pivot  = on_period.pivot(index='date', columns='code', values='overnight_ret_bps').dropna(how='all')
    all_dates = on_pivot.index.tolist()

    if N_signal < 2 or len(all_dates) < N_signal:
        print("  (サンプル不足のためテスト不可)")
        return np.array([]), real_sharpe

    rng = np.random.default_rng(42)
    null_sharpes = []
    for _ in range(n_perm):
        rand_dates = rng.choice(all_dates, size=N_signal, replace=False)
        rets = []
        for d in rand_dates:
            row = on_pivot.loc[d, [c for c in code5s if c in on_pivot.columns]].dropna()
            if len(row) > 0:
                rets.append(row.mean() - cost_bps)
        if len(rets) < 2:
            continue
        arr = np.array(rets)
        s = arr.mean() / arr.std(ddof=1) * np.sqrt(252) if arr.std(ddof=1) > 0 else 0
        null_sharpes.append(s)

    null_arr  = np.array(null_sharpes)
    p_value   = (null_arr >= real_sharpe).mean()
    pct_rank  = (null_arr < real_sharpe).mean() * 100

    print(f"  Null分布: mean={null_arr.mean():+.2f}  std={null_arr.std():.2f}  "
          f"95th={np.percentile(null_arr, 95):+.2f}  99th={np.percentile(null_arr, 99):+.2f}")
    print(f"  実Sharpe {real_sharpe:+.2f} は null の {pct_rank:.1f}%tile")
    print(f"  p-value (片側): {p_value:.4f}  "
          f"{'★ p<0.05 (統計的に有意)' if p_value < 0.05 else '  p>=0.05 (偶然の可能性を排除できない)'}")
    return null_arr, real_sharpe

# ─── 分析4: Selection Bias ────────────────────────────
def analysis_selection_bias(sig_df, on_df, cost_bps=PRIMARY_COST):
    print("\n" + "="*65)
    print("  [分析4] Selection Bias 検証")
    print("="*65)
    print("  IS期間で全スクリーニング銘柄を試した時のSharpe分布")

    ric_map   = get_code5_map(SCREEN_RICS)
    all_code5 = list(set(ric_map.values()))

    on_period = on_df[(on_df['date'] >= IS_START) & (on_df['date'] <= IS_END)]
    on_period = on_period[on_period['code'].isin(all_code5)]
    on_pivot  = on_period.pivot(index='date', columns='code', values='overnight_ret_bps')

    sig_dates = sig_df.loc[(sig_df.index >= pd.to_datetime(IS_START).date()) &
                            (sig_df.index <= pd.to_datetime(IS_END).date()) &
                            (sig_df['move_pct'] >= LME_THRESHOLD)].index
    sig_dates_ts = [pd.Timestamp(d) for d in sig_dates if pd.Timestamp(d) in on_pivot.index]

    stock_sharpes = {}
    for code in all_code5:
        if code not in on_pivot.columns:
            continue
        rets = [on_pivot.loc[d, code] - cost_bps for d in sig_dates_ts
                if not np.isnan(on_pivot.loc[d, code]) if code in on_pivot.columns]
        if len(rets) >= 5:
            arr = np.array(rets)
            s = arr.mean() / arr.std(ddof=1) * np.sqrt(252) if arr.std(ddof=1) > 0 else np.nan
            stock_sharpes[code] = {'sharpe': s, 'n': len(rets), 'mean': arr.mean()}

    if not stock_sharpes:
        print("  (データ不足)")
        return {}, np.array([])

    shp_df = pd.DataFrame(stock_sharpes).T.sort_values('sharpe', ascending=False)
    code5_to_ric = {v: k for k, v in ric_map.items()}
    shp_df['ric'] = shp_df.index.map(code5_to_ric)
    all_sharpes   = shp_df['sharpe'].dropna().values

    core5_map    = get_code5_map(CORE5_RICS)
    core5_code5s = list(core5_map.values())

    print(f"\n  全{len(shp_df)}銘柄 Sharpe分布: "
          f"mean={np.nanmean(all_sharpes):+.2f}  "
          f"std={np.nanstd(all_sharpes):.2f}  "
          f"median={np.nanmedian(all_sharpes):+.2f}")
    print(f"  Sharpe >= 2.0: {(all_sharpes>=2).sum()}銘柄  "
          f">= 5.0: {(all_sharpes>=5).sum()}銘柄  "
          f">= 8.0: {(all_sharpes>=8).sum()}銘柄")

    print(f"\n  Top10銘柄 (IS Sharpe):")
    print(f"  {'RIC':<10} {'N':>4} {'Mean(bps)':>10} {'Sharpe':>8}")
    for _, row in shp_df.head(10).iterrows():
        ric = row.get('ric', row.name)
        marker = ' ← core5' if row.name in core5_code5s else ''
        print(f"  {ric:<10} {row['n']:>4.0f} {row['mean']:>+10.1f} {row['sharpe']:>+8.2f}{marker}")

    print(f"\n  コア5銘柄のIS Sharpe 順位:")
    for c in core5_code5s:
        if c in shp_df.index:
            ric = code5_to_ric.get(c, c)
            s   = shp_df.loc[c, 'sharpe']
            pct = (all_sharpes < s).mean() * 100
            rank = int(shp_df.index.get_loc(c)) + 1
            print(f"    {ric:<10} Sharpe={s:+.2f}  順位{rank}/{len(shp_df)}  上位{100-pct:.0f}%tile")

    # コア5バスケット
    c5_in = [c for c in core5_code5s if c in on_pivot.columns]
    if c5_in:
        bret = [on_pivot.loc[d, c5_in].mean() - cost_bps for d in sig_dates_ts]
        bret = [r for r in bret if not np.isnan(r)]
        if len(bret) >= 2:
            arr = np.array(bret)
            bs  = arr.mean() / arr.std(ddof=1) * np.sqrt(252) if arr.std(ddof=1) > 0 else np.nan
            pct = (all_sharpes < bs).mean() * 100
            print(f"\n  コア5バスケット Sharpe={bs:+.2f}  上位{100-pct:.0f}%tile")

    print(f"\n  [判断] Sharpe>=5が多数なら選択バイアスが大きい。コア5だけが突出なら本物。")
    return shp_df, all_sharpes

# ─── 分析5: コスト感応度 ──────────────────────────────
def analysis_cost_sensitivity(sig_df, on_df, code5s):
    print("\n" + "="*65)
    print("  [分析5] コスト感応度 (IS期間, コア5バスケット)")
    print("="*65)
    print(f"  {'コスト':>6} {'N':>4} {'WR':>6} {'Mean(bps)':>10} {'Total(bps)':>11} {'Sharpe':>8}")
    print("  " + "-"*50)
    for cost in COST_SCENARIOS:
        tdf = backtest_basket(sig_df, on_df, code5s, IS_START, IS_END, cost)
        r = evaluate(tdf)
        if r['n'] > 0:
            print(f"  {cost:>4}bps {r['n']:>4} {r['wr']:>5.1f}% "
                  f"{r['mean_bps']:>+10.1f} {r['total_bps']:>+11.0f} {r['sharpe']:>+8.2f}")

# ─── 分析6: ベンチマーク対比 ─────────────────────────
def analysis_benchmark(sig_df, on_df, code5s, cost_bps=PRIMARY_COST):
    print("\n" + "="*65)
    print("  [分析6] 市場ベータ対比 (日経先物 JNIc1)")
    print("="*65)
    print("  「コア5のリターンは単なる市場ベータか?」")

    conn = get_conn()
    q = """SELECT trade_date AS date, close
           FROM macro.daily_ohlcv
           WHERE symbol='JNIc1'
             AND trade_date >= %s AND trade_date <= %s
           ORDER BY trade_date"""
    nk = pd.read_sql(q, conn, params=(IS_START, IS_END), parse_dates=['date'])
    conn.close()

    if len(nk) == 0:
        print("  (日経先物データなし)")
        return

    nk = nk.set_index('date')['close']
    # 日経先物の翌日オープン比リターン (日足closeは15:30前後)
    nk_overnight = nk.pct_change().shift(-1) * 10000  # bps (next-day close vs today's close as proxy)

    tdf = backtest_basket(sig_df, on_df, code5s, IS_START, IS_END, cost_bps)
    if len(tdf) == 0:
        return

    nk_rets = nk_overnight.reindex(tdf['date']).values
    basket_rets = tdf['pnl_bps'].values

    valid = ~(np.isnan(nk_rets) | np.isnan(basket_rets))
    if valid.sum() < 3:
        print("  (データ不足)")
        return

    br = basket_rets[valid]
    nr = nk_rets[valid]
    corr = np.corrcoef(br, nr)[0, 1]
    # 残差Sharpe (日経ベータ除去後)
    beta = np.cov(br, nr)[0, 1] / np.var(nr) if np.var(nr) > 0 else 0
    resid = br - beta * nr
    resid_sharpe = resid.mean() / resid.std(ddof=1) * np.sqrt(252) if resid.std(ddof=1) > 0 else np.nan

    print(f"\n  LMEシグナル日 (N={valid.sum()}) での比較:")
    print(f"  コア5バスケット: mean={br.mean():+.1f}bps  Sharpe={br.mean()/br.std()*np.sqrt(252):+.2f}")
    print(f"  日経先物(同日) : mean={nr.mean():+.1f}bps  Sharpe={nr.mean()/nr.std()*np.sqrt(252):+.2f}")
    print(f"  相関係数       : {corr:+.3f}")
    print(f"  β             : {beta:.3f}")
    print(f"  β除去後Sharpe  : {resid_sharpe:+.2f}"
          f"  ({'アルファ残存' if resid_sharpe > 1.0 else '市場ベータが主因'})")

# ─── 可視化 ──────────────────────────────────────────
def plot_results(tdf_is, wf_df, null_arr, real_sharpe, all_sharpes,
                 core5_sharpes, sig_df, on_df, code5s):
    fig = plt.figure(figsize=(14, 9), facecolor='white')
    plt.rcParams.update({
        'font.family': ['IPAexGothic', 'Noto Sans CJK JP', 'sans-serif'],
        'axes.unicode_minus': False,
        'axes.facecolor': '#f8f9fa',
        'grid.alpha': 0.3,
    })
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.38)

    # ── Panel 1: IS+OOS累積PnL ─────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    full_tdf = backtest_basket(sig_df, on_df, code5s, IS_START, OOS_END, PRIMARY_COST)
    if len(full_tdf) > 0:
        cum = full_tdf.set_index('date')['pnl_bps'].cumsum()
        # IS部分
        is_part  = cum[cum.index <= IS_END]
        oos_part = cum[cum.index >= OOS_START]
        ax1.step(is_part.index, is_part.values, color='#1565C0', lw=2,
                 where='post', label='IS期間')
        if len(oos_part) > 0:
            ax1.step(oos_part.index, oos_part.values, color='#E65100', lw=2.5,
                     where='post', label='OOS期間', linestyle='--')
    ax1.axvline(pd.Timestamp(OOS_START), color='gray', lw=1, linestyle='--', alpha=0.6)
    ax1.axhline(0, color='black', lw=0.5)
    ax1.text(pd.Timestamp(OOS_START), ax1.get_ylim()[0] if ax1.get_ylim()[0] != 0 else -50,
             ' OOS開始', fontsize=7, color='gray', va='bottom')
    ax1.set_title('コア5バスケット 累積PnL', fontsize=10, fontweight='bold')
    ax1.set_ylabel('累積PnL (bps)')
    ax1.legend(fontsize=8, loc='upper left')
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: Walk-forward 月次 ──────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    if wf_df is not None:
        active = wf_df[wf_df['n'] > 0].copy()
        if len(active) > 0:
            colors = ['#43A047' if v > 0 else '#E53935' for v in active['total_bps']]
            ax2.bar(range(len(active)), active['total_bps'].values, color=colors, alpha=0.85)
            ax2.set_xticks(range(len(active)))
            ax2.set_xticklabels(active['month'].tolist(), rotation=45, fontsize=7)
            ax2.axhline(0, color='black', lw=1)
            n_pos = (active['total_bps'] > 0).sum()
            ax2.set_title(f'Walk-forward 月次PnL\n({n_pos}/{len(active)}ヶ月が黒字)',
                          fontsize=10, fontweight='bold')
        else:
            ax2.set_title('Walk-forward 月次PnL\n(データなし)', fontsize=10)
    ax2.set_ylabel('月次PnL (bps)')
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: Permutation Test ───────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    if len(null_arr) > 0 and not np.isnan(real_sharpe):
        ax3.hist(null_arr, bins=60, color='#78909C', alpha=0.8,
                 edgecolor='white', label=f'Null ({len(null_arr):,}回)')
        ax3.axvline(real_sharpe, color='#E53935', lw=2.5,
                    label=f'実Sharpe={real_sharpe:+.2f}')
        p_val = (null_arr >= real_sharpe).mean()
        ax3.set_title(f'Permutation Test\np-value={p_val:.4f}', fontsize=10, fontweight='bold')
        ax3.set_xlabel('Sharpe')
        ax3.set_ylabel('頻度')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)

    # ── Panel 4: Selection Bias ─────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    if len(all_sharpes) > 0:
        valid_s = all_sharpes[~np.isnan(all_sharpes)]
        ax4.hist(valid_s, bins=20, color='#546E7A', alpha=0.8,
                 edgecolor='white', label=f'全銘柄 (N={len(valid_s)})')
        for s in core5_sharpes:
            if not np.isnan(s):
                ax4.axvline(s, color='#E53935', lw=1.2, alpha=0.8)
        if core5_sharpes:
            ax4.axvline(core5_sharpes[0], color='#E53935', lw=1.2,
                        alpha=0.8, label='コア5各銘柄')
        ax4.set_title('Selection Bias: 全銘柄Sharpe分布(IS)',
                      fontsize=10, fontweight='bold')
        ax4.set_xlabel('個別銘柄 Sharpe')
        ax4.set_ylabel('銘柄数')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)

    r_is  = evaluate(tdf_is)
    fig.suptitle(
        f'LME銅(CMCU3)シグナル戦略 エッジ再検証 — 2026-06-01\n'
        f'LME>=+{LME_THRESHOLD}% → コア5 Overnight Long  |  '
        f'IS Sharpe={r_is["sharpe"]:+.2f} (N={r_is["n"]})',
        fontsize=12, fontweight='bold', y=1.01
    )
    fig.text(0.99, -0.02,
             'Signal: nas_archive.intraday_data (CMCU3)  |  JP: stocks_daily adj_close/adj_open',
             ha='right', va='bottom', fontsize=7, color='gray')

    plt.savefig('result.png', dpi=100, bbox_inches='tight', facecolor='white')
    print("\n  -> result.png を保存")
    plt.close()

# ─── メイン ──────────────────────────────────────────
def main():
    print("=" * 65)
    print("  LME銅(CMCU3) シグナル戦略 エッジ再検証")
    print(f"  IS  : {IS_START} 〜 {IS_END}")
    print(f"  OOS : {OOS_START} 〜 {OOS_END}")
    print("=" * 65)

    print("\n[データ読み込み中...]")
    sig_df = load_lme_signals()
    print(f"  CMCU3シグナル計算完了: {len(sig_df)}日  "
          f"最新={sig_df.index[-1]}")

    ric_map     = get_code5_map(CORE5_RICS)
    code5s      = [ric_map[r] for r in CORE5_RICS if r in ric_map]
    print(f"  コア5 code5: {code5s}")

    screen_map  = get_code5_map(SCREEN_RICS)
    all_code5s  = list(set(code5s + list(screen_map.values())))
    on_df = load_overnight_returns(all_code5s)
    print(f"  overnight_returns: {len(on_df)}行  銘柄数={on_df['code'].nunique()}")

    # HGシグナル件数確認
    is_sig  = sig_df.loc[(sig_df.index >= pd.to_datetime(IS_START).date()) &
                          (sig_df.index <= pd.to_datetime(IS_END).date()) &
                          (sig_df['move_pct'] >= LME_THRESHOLD)]
    oos_sig = sig_df.loc[(sig_df.index >= pd.to_datetime(OOS_START).date()) &
                          (sig_df.index <= pd.to_datetime(OOS_END).date()) &
                          (sig_df['move_pct'] >= LME_THRESHOLD)]
    print(f"  CMCU3>=+{LME_THRESHOLD}%: IS={len(is_sig)}日  OOS={len(oos_sig)}日")
    if len(oos_sig) > 0:
        print(f"  OOSシグナル日: {oos_sig.index.tolist()}")

    # ─── 各分析 ─────────────────────────────────────
    tdf_is       = analysis_is_vs_oos(sig_df, on_df, code5s)
    wf_df        = analysis_walkforward(sig_df, on_df, code5s)
    null_arr, rs = analysis_permutation(sig_df, on_df, code5s, n_perm=5000)
    shp_df, all_sharpes = analysis_selection_bias(sig_df, on_df)
    analysis_cost_sensitivity(sig_df, on_df, code5s)
    analysis_benchmark(sig_df, on_df, code5s)

    # コア5各銘柄のIS Sharpeをplot用に取得
    c5_map = get_code5_map(CORE5_RICS)
    c5_code5s = list(c5_map.values())
    core5_sharpes = []
    if shp_df is not None and len(shp_df) > 0:
        for c in c5_code5s:
            if c in shp_df.index:
                core5_sharpes.append(float(shp_df.loc[c, 'sharpe']))

    # ─── 総合評価 ───────────────────────────────────
    print("\n" + "="*65)
    print("  総合評価まとめ")
    print("="*65)
    r_is = evaluate(tdf_is)
    tdf_oos = backtest_basket(sig_df, on_df, code5s, OOS_START, OOS_END, PRIMARY_COST)
    r_oos   = evaluate(tdf_oos)
    p_val_str = f"{(null_arr >= rs).mean():.4f}" if len(null_arr) > 0 else "N/A"

    print(f"""
  IS  : Sharpe={r_is['sharpe']:+.2f}  N={r_is['n']}  Mean={r_is['mean_bps']:+.1f}bps
  OOS : {'Sharpe='+f"{r_oos['sharpe']:+.2f}  N={r_oos['n']}  Total={r_oos['total_bps']:+.0f}bps" if r_oos['n']>0 else 'トレードなし'}
  Permutation p-value : {p_val_str}

  チェックリスト:
  [{'x' if r_is['sharpe'] >= 5.0 else ' '}] IS Sharpe >= 5.0 (実値: {r_is['sharpe']:+.2f})
  [{'x' if r_oos['total_bps'] > 0 else ' '}] OOS 正リターン (実値: {r_oos['total_bps']:+.0f}bps)
  [{'x' if len(null_arr)>0 and (null_arr>=rs).mean()<0.05 else ' '}] Permutation p < 0.05 ({p_val_str})
  [{'x' if len(all_sharpes)>0 and (all_sharpes>=5).sum()<=3 else ' '}] 高Sharpe銘柄が少ない (>=5.0が<=3銘柄)
  [{'x' if evaluate(backtest_basket(sig_df,on_df,code5s,IS_START,IS_END,8))['sharpe'] >= 3.0 else ' '}] コスト8bps でも Sharpe >= 3.0
""")

    plot_results(tdf_is, wf_df,
                 null_arr if null_arr is not None else np.array([]),
                 rs,
                 all_sharpes if all_sharpes is not None else np.array([]),
                 core5_sharpes,
                 sig_df, on_df, code5s)
    print("完了")


if __name__ == "__main__":
    main()
