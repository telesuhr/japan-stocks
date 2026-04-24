"""
日次 (multi-day) 出来高 → 方向性予測

3つのアプローチ:

A. 日次出来高レシオ → T+1, T+5, T+10 リターン
  daily_vol_ratio = 当日出来高 / 20日平均出来高
  翌日 open→close, T+5日累積, T+10日累積 とのSpearman相関
  5分位バケット分析

B. 日次 OBV ダイバージェンス (multi-day)
  OBV(t) = Σ volume * sign(close - prev_close)
  price_mom = 20日価格モメンタム (pct change)
  obv_mom = 20日OBVモメンタム (pct change of OBV level)
  diverg = price_mom_rank - obv_mom_rank  (銘柄内百分位)
  4象限 + 連続量で T+5, T+10 リターンと相関

C. 出来高急増日イベントスタディ
  スパイク日: volume >= 20日median * 3.0
  当日リターン符号で up_spike / down_spike に分岐
  T-5 〜 T+10 のカム平均リターンプロット
  仮説: up_spike = 買いクライマックス (反転) / 売りクライマックス (反転)

対象: 非鉄3 + 半導体5 = 8銘柄
期間: 2018-01-01 〜 2026-04-22 (約2,000営業日/銘柄)
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '20260421_common')))
import mdutil as U
import pandas as pd
import numpy as np
import psycopg2
from scipy import stats
plt = U.matplotlib_jp()

SYMS = U.NONFERROUS + U.SEMICON
OUT_DIR = os.path.dirname(__file__)
START = "2018-01-01"
END = "2026-04-23"


def load_daily(sym):
    conn = psycopg2.connect(**U.PG_CONFIG)
    q = (f"SELECT trade_date, open, high, low, close, volume FROM daily_data "
         f"WHERE symbol='{sym}' AND trade_date>='{START}' AND trade_date<'{END}' "
         f"ORDER BY trade_date")
    df = pd.read_sql(q, conn); conn.close()
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df = df.set_index('trade_date').sort_index()
    for c in ['open','high','low','close','volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['open','close','volume'])
    df = df[df['volume'] > 0]
    return df


def add_features(df):
    """日次特徴量を追加"""
    df = df.copy()
    # リターン
    df['ret_oc'] = df['close'] / df['open'] - 1  # open-to-close
    df['ret_cc'] = df['close'] / df['close'].shift(1) - 1  # close-to-close
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    # 出来高
    df['vol_ma20'] = df['volume'].rolling(20, min_periods=10).mean().shift(1)
    df['vol_med20'] = df['volume'].rolling(20, min_periods=10).median().shift(1)
    df['vol_ratio'] = df['volume'] / df['vol_ma20']
    df['vol_ratio_med'] = df['volume'] / df['vol_med20']
    # OBV
    sign = np.sign(df['close'] - df['close'].shift(1))
    df['signed_vol'] = df['volume'] * sign
    df['obv'] = df['signed_vol'].cumsum()
    # 20日モメンタム (pct)
    df['price_mom20'] = df['close'].pct_change(20)
    df['obv_mom20'] = df['obv'] - df['obv'].shift(20)  # 絶対変化
    # 銘柄内百分位 (ローリングではなく全期間)
    df['price_mom20_rank'] = df['price_mom20'].rank(pct=True)
    df['obv_mom20_rank'] = df['obv_mom20'].rank(pct=True)
    df['divergence'] = df['price_mom20_rank'] - df['obv_mom20_rank']
    # 前向きリターン
    for h in [1, 5, 10]:
        df[f'fwd_{h}'] = df['close'].shift(-h) / df['close'] - 1
    # T+1 open→close のみ別途
    df['fwd_1_oc'] = (df['close'].shift(-1) / df['open'].shift(-1)) - 1
    return df


def quintile_stats(x, y, ret_bps=True):
    """5分位バケット分析"""
    df = pd.DataFrame({'x': x, 'y': y}).dropna()
    if len(df) < 25: return None
    try:
        df['q'] = pd.qcut(df['x'], 5, labels=['Q1(低)','Q2','Q3','Q4','Q5(高)'],
                          duplicates='drop')
    except ValueError:
        return None
    scale = 10000 if ret_bps else 1
    rows = []
    for q in df['q'].cat.categories:
        sub = df[df['q'] == q]['y'] * scale
        if len(sub) == 0: continue
        m, s = sub.mean(), sub.std()
        t = m/(s/np.sqrt(len(sub))) if s>0 else 0
        rows.append({'bucket': str(q), 'N': len(sub),
                     'mean_bps': m, 'wr': (sub>0).mean()*100, 't_stat': t})
    return pd.DataFrame(rows)


def fmt_df(df):
    fs = {}
    for c in df.columns:
        if c in ('mean_bps',): fs[c] = '{:+.1f}'.format
        elif c == 'wr': fs[c] = '{:.1f}'.format
        elif c in ('t_stat','r','spearman_r'): fs[c] = '{:+.2f}'.format
        elif c == 'p': fs[c] = '{:.3f}'.format
    return df.to_string(index=False, formatters=fs)


# ============================================================
# A. 日次出来高レシオ → T+1/T+5/T+10
# ============================================================

def analyze_A(feats):
    """feats: {sym: df_with_features}"""
    print("=" * 70)
    print(" [A] 日次 vol_ratio → T+1, T+5, T+10 リターン")
    print("=" * 70)

    # 集約 pool
    pool = []
    for sym, df in feats.items():
        sub = df[['vol_ratio','fwd_1','fwd_5','fwd_10','fwd_1_oc']].copy()
        sub['sym'] = sym
        pool.append(sub)
    pool = pd.concat(pool).reset_index(drop=True)

    # 集約 Spearman
    print("\n== 集約 8銘柄 Spearman ==")
    for tgt in ['fwd_1','fwd_1_oc','fwd_5','fwd_10']:
        sub = pool.dropna(subset=['vol_ratio', tgt])
        sp = stats.spearmanr(sub['vol_ratio'], sub[tgt])
        print(f"  vol_ratio vs {tgt:<10}: N={len(sub):5d}  r={sp.correlation:+.3f}  p={sp.pvalue:.3f}")

    # 銘柄別
    print("\n== 銘柄別 Spearman (vol_ratio vs fwd_5) ==")
    sym_rows = []
    for sym, df in feats.items():
        sub = df.dropna(subset=['vol_ratio','fwd_5'])
        if len(sub) < 100: continue
        sp = stats.spearmanr(sub['vol_ratio'], sub['fwd_5'])
        sym_rows.append({'sym': sym, 'N': len(sub),
                        'spearman_r': sp.correlation, 'p': sp.pvalue})
    sym_df = pd.DataFrame(sym_rows)
    print(fmt_df(sym_df))

    # 5分位バケット (集約、target=fwd_5)
    print("\n== 集約 5分位バケット (target=fwd_5) ==")
    q5 = quintile_stats(pool['vol_ratio'], pool['fwd_5'])
    if q5 is not None: print(fmt_df(q5))

    print("\n== 集約 5分位バケット (target=fwd_10) ==")
    q10 = quintile_stats(pool['vol_ratio'], pool['fwd_10'])
    if q10 is not None: print(fmt_df(q10))

    return pool, q5, q10, sym_df


# ============================================================
# B. OBV ダイバージェンス
# ============================================================

def analyze_B(feats):
    print("\n" + "=" * 70)
    print(" [B] 日次 OBV × 価格モメンタム ダイバージェンス")
    print("=" * 70)

    pool = []
    for sym, df in feats.items():
        sub = df[['price_mom20','obv_mom20','divergence','fwd_1','fwd_5','fwd_10']].copy()
        sub['sym'] = sym
        pool.append(sub)
    pool = pd.concat(pool).reset_index(drop=True).dropna(
        subset=['price_mom20','obv_mom20','divergence'])

    # 連続変数 Spearman
    print("\n== 集約 Spearman: divergence vs forward ret ==")
    for tgt in ['fwd_1','fwd_5','fwd_10']:
        sub = pool.dropna(subset=['divergence', tgt])
        sp = stats.spearmanr(sub['divergence'], sub[tgt])
        print(f"  divergence vs {tgt:<10}: N={len(sub):5d}  r={sp.correlation:+.3f}  p={sp.pvalue:.3f}")

    # 4象限分類 (銘柄内 median split)
    p_med = pool.groupby('sym')['price_mom20'].transform('median')
    o_med = pool.groupby('sym')['obv_mom20'].transform('median')
    pool['p_sign'] = np.where(pool['price_mom20'] >= p_med, 'P+', 'P-')
    pool['o_sign'] = np.where(pool['obv_mom20'] >= o_med, 'O+', 'O-')
    pool['quad'] = pool['p_sign'] + pool['o_sign']

    print("\n== 4象限 → T+5 リターン ==")
    rows = []
    for q in ['P+O+','P+O-','P-O+','P-O-']:
        sub = pool[pool['quad']==q]['fwd_5'].dropna() * 10000
        if len(sub) < 50: continue
        m, s = sub.mean(), sub.std()
        t = m/(s/np.sqrt(len(sub))) if s>0 else 0
        rows.append({'quad': q, 'N': len(sub), 'mean_bps': m,
                     'wr': (sub>0).mean()*100, 't_stat': t})
    b_q5 = pd.DataFrame(rows)
    print(fmt_df(b_q5))

    # P+O- (弱気div) vs P-O+ (強気div)
    bu = pool[pool['quad']=='P-O+']['fwd_5'].dropna() * 10000
    be = pool[pool['quad']=='P+O-']['fwd_5'].dropna() * 10000
    if len(bu) >= 50 and len(be) >= 50:
        diff = bu.mean() - be.mean()
        t_test = stats.ttest_ind(bu, be, equal_var=False)
        print(f"\n強気div (P-O+) mean - 弱気div (P+O-) mean = {diff:+.2f} bps  "
              f"Welch t={t_test.statistic:+.2f} (p={t_test.pvalue:.3f})")

    print("\n== 4象限 → T+10 リターン ==")
    rows = []
    for q in ['P+O+','P+O-','P-O+','P-O-']:
        sub = pool[pool['quad']==q]['fwd_10'].dropna() * 10000
        if len(sub) < 50: continue
        m, s = sub.mean(), sub.std()
        t = m/(s/np.sqrt(len(sub))) if s>0 else 0
        rows.append({'quad': q, 'N': len(sub), 'mean_bps': m,
                     'wr': (sub>0).mean()*100, 't_stat': t})
    b_q10 = pd.DataFrame(rows)
    print(fmt_df(b_q10))

    # divergence 5分位
    print("\n== divergence 5分位 → T+5 ==")
    q5 = quintile_stats(pool['divergence'], pool['fwd_5'])
    if q5 is not None: print(fmt_df(q5))

    print("\n== divergence 5分位 → T+10 ==")
    q10 = quintile_stats(pool['divergence'], pool['fwd_10'])
    if q10 is not None: print(fmt_df(q10))

    return pool, b_q5, b_q10, q5, q10


# ============================================================
# C. 出来高急増イベントスタディ
# ============================================================

def analyze_C(feats, spike_mult=3.0):
    print("\n" + "=" * 70)
    print(f" [C] 出来高急増イベントスタディ (vol >= {spike_mult}x 20日中央値)")
    print("=" * 70)

    # 各銘柄で spike日を識別、T-5〜T+10 の 累積リターンパス
    all_paths_up = []
    all_paths_dn = []
    spike_count_up = 0
    spike_count_dn = 0
    for sym, df in feats.items():
        df = df.copy()
        df['is_spike'] = df['vol_ratio_med'] >= spike_mult
        # event: spike が当日リターン > +50bps → up, < -50bps → dn
        for i, (d, row) in enumerate(df.iterrows()):
            if not row['is_spike']: continue
            ret_oc = row['ret_oc']
            if pd.isna(ret_oc): continue
            # T-5 〜 T+10 の close を取得
            pos = df.index.get_loc(d)
            if pos < 5 or pos > len(df)-11: continue
            window = df.iloc[pos-5:pos+11]  # T-5..T+10 (16日)
            base_close = window.iloc[5]['close']  # T=0 (event day close)
            path = (window['close'].values / base_close - 1) * 10000
            if ret_oc > 0.005:  # +50bps
                all_paths_up.append(path)
                spike_count_up += 1
            elif ret_oc < -0.005:
                all_paths_dn.append(path)
                spike_count_dn += 1
    print(f"  up spike events: {spike_count_up}")
    print(f"  down spike events: {spike_count_dn}")

    if len(all_paths_up) > 20:
        arr_up = np.vstack(all_paths_up)
        mean_up = np.nanmean(arr_up, axis=0)
        std_up = np.nanstd(arr_up, axis=0)
        n_up = arr_up.shape[0]
    else:
        mean_up = None; std_up = None; n_up = 0

    if len(all_paths_dn) > 20:
        arr_dn = np.vstack(all_paths_dn)
        mean_dn = np.nanmean(arr_dn, axis=0)
        std_dn = np.nanstd(arr_dn, axis=0)
        n_dn = arr_dn.shape[0]
    else:
        mean_dn = None; std_dn = None; n_dn = 0

    print("\n== 平均リターンパス (bps 基準: T=0 終値) ==")
    print(f"  offset |   up_spike (N={n_up})   |  dn_spike (N={n_dn})")
    offsets = list(range(-5, 11))
    for i, off in enumerate(offsets):
        up = mean_up[i] if mean_up is not None else np.nan
        dn = mean_dn[i] if mean_dn is not None else np.nan
        print(f"    T{off:+3d}  |  mean={up:+8.1f}       |  mean={dn:+8.1f}")

    # T+1〜T+10 の統計的検定 (mean != 0)
    print("\n== t-stat of mean = 0 (T+1〜T+10) ==")
    if mean_up is not None:
        print("  [up_spike] T+1〜T+10 mean cum ret (T=0基準):")
        for i, off in enumerate(range(1, 11)):
            idx = 5 + off  # 0..10 offset, index 5 is T=0
            col = arr_up[:, idx] - arr_up[:, 5]  # difference vs T=0 (should be same as col since base=T0close)
            # Actually path is relative to T=0 close, so col at index 5 == 0 by construction.
            # So arr_up[:, idx] is already the cum return from T=0.
            col = arr_up[:, idx]
            col = col[~np.isnan(col)]
            m, s = col.mean(), col.std()
            t = m/(s/np.sqrt(len(col))) if s>0 else 0
            print(f"    T+{off:2d}: N={len(col):4d}  mean={m:+7.1f} bps  t={t:+5.2f}")
    if mean_dn is not None:
        print("  [dn_spike] T+1〜T+10 mean cum ret:")
        for i, off in enumerate(range(1, 11)):
            idx = 5 + off
            col = arr_dn[:, idx]
            col = col[~np.isnan(col)]
            m, s = col.mean(), col.std()
            t = m/(s/np.sqrt(len(col))) if s>0 else 0
            print(f"    T+{off:2d}: N={len(col):4d}  mean={m:+7.1f} bps  t={t:+5.2f}")

    return mean_up, std_up, n_up, mean_dn, std_dn, n_dn, offsets


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print(" 日次 (multi-day) 出来高 → 方向性予測 ")
    print("=" * 70)
    print(f"期間: {START} 〜 {END}")

    feats = {}
    for sym, name in SYMS:
        df = load_daily(sym)
        feats[sym] = add_features(df)
        print(f"  loaded {sym} {name}: {len(df)} days")
    print()

    # A
    a_pool, a_q5, a_q10, a_sym = analyze_A(feats)
    # B
    b_pool, b_q5_quad, b_q10_quad, b_q5_div, b_q10_div = analyze_B(feats)
    # C
    c_up_mean, c_up_std, c_up_n, c_dn_mean, c_dn_std, c_dn_n, c_offsets = analyze_C(feats, spike_mult=3.0)

    # ============================================================
    # 可視化
    # ============================================================
    print("\n可視化中...")
    fig = plt.figure(figsize=(18, 18))
    gs = fig.add_gridspec(4, 3, hspace=0.55, wspace=0.32)

    # (0,0) A: 銘柄別 Spearman (vol_ratio vs fwd_5)
    ax = fig.add_subplot(gs[0,0])
    colors = ['#d62728' if r<0 else '#2ca02c' for r in a_sym['spearman_r']]
    ax.barh(range(len(a_sym)), a_sym['spearman_r'], color=colors)
    ax.set_yticks(range(len(a_sym))); ax.set_yticklabels(a_sym['sym'], fontsize=9)
    ax.axvline(0, color='k', lw=0.5)
    for i, (r, p) in enumerate(zip(a_sym['spearman_r'], a_sym['p'])):
        sig = '*' if p<0.05 else ''
        ax.text(r, i, f" {r:+.2f}{sig}", va='center', fontsize=8)
    ax.invert_yaxis()
    ax.set_title('[A] 銘柄別 Spearman\nvol_ratio → fwd_5', fontweight='bold', fontsize=10)
    ax.set_xlabel('Spearman r (* p<0.05)')

    # (0,1) A: vol_ratio 5分位 → fwd_5
    ax = fig.add_subplot(gs[0,1])
    if a_q5 is not None:
        colors = ['#d62728' if v<0 else '#2ca02c' for v in a_q5['mean_bps']]
        ax.bar(range(len(a_q5)), a_q5['mean_bps'], color=colors)
        ax.set_xticks(range(len(a_q5)))
        ax.set_xticklabels(a_q5['bucket'], rotation=0, fontsize=8)
        ax.axhline(0, color='k', lw=0.5)
        for i, (m, n, t) in enumerate(zip(a_q5['mean_bps'], a_q5['N'], a_q5['t_stat'])):
            ax.text(i, m, f"{m:+.0f}\nt={t:+.1f}\nN={n}", ha='center',
                    va='bottom' if m>0 else 'top', fontsize=7)
    ax.set_title('[A] vol_ratio 5分位 → T+5 (bps)', fontweight='bold', fontsize=10)
    ax.set_ylabel('mean_bps'); ax.grid(alpha=0.3)

    # (0,2) A: vol_ratio 5分位 → fwd_10
    ax = fig.add_subplot(gs[0,2])
    if a_q10 is not None:
        colors = ['#d62728' if v<0 else '#2ca02c' for v in a_q10['mean_bps']]
        ax.bar(range(len(a_q10)), a_q10['mean_bps'], color=colors)
        ax.set_xticks(range(len(a_q10)))
        ax.set_xticklabels(a_q10['bucket'], rotation=0, fontsize=8)
        ax.axhline(0, color='k', lw=0.5)
        for i, (m, n, t) in enumerate(zip(a_q10['mean_bps'], a_q10['N'], a_q10['t_stat'])):
            ax.text(i, m, f"{m:+.0f}\nt={t:+.1f}\nN={n}", ha='center',
                    va='bottom' if m>0 else 'top', fontsize=7)
    ax.set_title('[A] vol_ratio 5分位 → T+10 (bps)', fontweight='bold', fontsize=10)
    ax.set_ylabel('mean_bps'); ax.grid(alpha=0.3)

    # (1,0) B: 4象限 → T+5
    ax = fig.add_subplot(gs[1,0])
    color_map = {'P+O+':'#2ca02c','P+O-':'#d62728','P-O+':'#1f77b4','P-O-':'#ff7f0e'}
    if not b_q5_quad.empty:
        ax.bar(b_q5_quad['quad'], b_q5_quad['mean_bps'],
               color=[color_map.get(q,'gray') for q in b_q5_quad['quad']])
        ax.axhline(0, color='k', lw=0.5)
        for i, (m, n, t) in enumerate(zip(b_q5_quad['mean_bps'], b_q5_quad['N'], b_q5_quad['t_stat'])):
            ax.text(i, m, f"{m:+.0f}\nt={t:+.1f}\nN={n}", ha='center',
                    va='bottom' if m>0 else 'top', fontsize=8)
    ax.set_title('[B] 4象限 (価格mom×OBV mom) → T+5', fontweight='bold', fontsize=10)
    ax.set_ylabel('mean_bps'); ax.grid(alpha=0.3)

    # (1,1) B: 4象限 → T+10
    ax = fig.add_subplot(gs[1,1])
    if not b_q10_quad.empty:
        ax.bar(b_q10_quad['quad'], b_q10_quad['mean_bps'],
               color=[color_map.get(q,'gray') for q in b_q10_quad['quad']])
        ax.axhline(0, color='k', lw=0.5)
        for i, (m, n, t) in enumerate(zip(b_q10_quad['mean_bps'], b_q10_quad['N'], b_q10_quad['t_stat'])):
            ax.text(i, m, f"{m:+.0f}\nt={t:+.1f}\nN={n}", ha='center',
                    va='bottom' if m>0 else 'top', fontsize=8)
    ax.set_title('[B] 4象限 → T+10', fontweight='bold', fontsize=10)
    ax.set_ylabel('mean_bps'); ax.grid(alpha=0.3)

    # (1,2) B: divergence 5分位 → T+5
    ax = fig.add_subplot(gs[1,2])
    if b_q5_div is not None:
        colors = ['#d62728' if v<0 else '#2ca02c' for v in b_q5_div['mean_bps']]
        ax.bar(range(len(b_q5_div)), b_q5_div['mean_bps'], color=colors)
        ax.set_xticks(range(len(b_q5_div))); ax.set_xticklabels(b_q5_div['bucket'], fontsize=8)
        ax.axhline(0, color='k', lw=0.5)
        for i, (m, n, t) in enumerate(zip(b_q5_div['mean_bps'], b_q5_div['N'], b_q5_div['t_stat'])):
            ax.text(i, m, f"{m:+.0f}\nt={t:+.1f}\nN={n}", ha='center',
                    va='bottom' if m>0 else 'top', fontsize=7)
    ax.set_title('[B] divergence 5分位 → T+5\n(Q1=強気div/Q5=弱気div)', fontweight='bold', fontsize=10)
    ax.set_ylabel('mean_bps'); ax.grid(alpha=0.3)

    # (2,0-2) C: 平均リターンパス
    ax = fig.add_subplot(gs[2,0:3])
    if c_up_mean is not None:
        ax.plot(c_offsets, c_up_mean, 'o-', color='#2ca02c', lw=2,
                label=f'up spike (+ret, N={c_up_n})')
    if c_dn_mean is not None:
        ax.plot(c_offsets, c_dn_mean, 'o-', color='#d62728', lw=2,
                label=f'dn spike (-ret, N={c_dn_n})')
    ax.axvline(0, color='gray', lw=0.8, ls='--', label='Event day')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('offset (days from event)'); ax.set_ylabel('cum_bps (base=T0 close)')
    ax.set_title('[C] 出来高急増(3x+)イベントスタディ: 平均リターンパス',
                 fontweight='bold', fontsize=11)
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_xticks(c_offsets)

    # (3,0) C: T+1〜T+10 の up_spike t-stat
    ax = fig.add_subplot(gs[3,0])
    # 再計算ここで (上記ではprintのみ)
    # パスからt統計量を抽出
    up_ts, dn_ts = [], []
    if c_up_mean is not None:
        # 実はパス自体のstdは入手可、手元にないのでプロットは mean だけで代替
        for off in range(1, 11):
            idx = 5 + off
            up_ts.append(c_up_mean[idx])
    if c_dn_mean is not None:
        for off in range(1, 11):
            idx = 5 + off
            dn_ts.append(c_dn_mean[idx])
    x = range(1, 11)
    if up_ts:
        ax.plot(list(x), up_ts, 'o-', color='#2ca02c', label=f'up_spike', lw=2)
    if dn_ts:
        ax.plot(list(x), dn_ts, 'o-', color='#d62728', label=f'dn_spike', lw=2)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('days after event'); ax.set_ylabel('cum mean bps')
    ax.set_title('[C] T+1〜T+10 累積リターン', fontweight='bold', fontsize=10)
    ax.legend(); ax.grid(alpha=0.3)

    # (3,1) サマリー
    ax = fig.add_subplot(gs[3,1]); ax.axis('off')
    _tmp = a_pool.dropna(subset=['vol_ratio','fwd_5'])
    pool_fwd5_sp = stats.spearmanr(_tmp['vol_ratio'], _tmp['fwd_5'])
    bu = b_pool[b_pool['quad']=='P-O+']['fwd_5'].dropna()*10000
    be = b_pool[b_pool['quad']=='P+O-']['fwd_5'].dropna()*10000
    diff = bu.mean() - be.mean() if (len(bu)>0 and len(be)>0) else 0
    summary = f"""
サマリー: 日次出来高 → 方向性

A. vol_ratio → fwd_5 (集約)
  Spearman = {pool_fwd5_sp.correlation:+.3f}
  p = {pool_fwd5_sp.pvalue:.3f}

B. OBV div → T+5
  強気div(P-O+): {bu.mean():+.1f} bps
  弱気div(P+O-): {be.mean():+.1f} bps
  差分: {diff:+.1f} bps

C. 出来高急増パターン:
  up_spike T+10: {c_up_mean[15] if c_up_mean is not None else 0:+.0f} bps
  dn_spike T+10: {c_dn_mean[15] if c_dn_mean is not None else 0:+.0f} bps

判定基準:
  |r|<0.05: 実質無相関
  |r|<0.10: 弱い示唆
  |r|>=0.10: 取引可能性あり
"""
    ax.text(0.02, 0.98, summary, family='monospace', fontsize=9,
            va='top', ha='left', transform=ax.transAxes)

    # (3,2) A 散布図 (サンプル)
    ax = fig.add_subplot(gs[3,2])
    x = a_pool['vol_ratio'].values
    y = a_pool['fwd_5'].values * 10000
    mask = ~(np.isnan(x) | np.isnan(y))
    ax.scatter(x[mask], y[mask], s=3, alpha=0.2, color='#1f77b4')
    ax.axhline(0, color='k', lw=0.5); ax.axvline(1.0, color='gray', lw=0.5, ls='--')
    ax.set_xlim(0, 5); ax.set_ylim(-1500, 1500)
    ax.set_xlabel('vol_ratio'); ax.set_ylabel('fwd_5 bps')
    ax.set_title('[A] 散布図 vol_ratio vs fwd_5', fontweight='bold', fontsize=10)
    ax.grid(alpha=0.3)

    plt.suptitle(f'日次出来高 → 方向性予測 (8銘柄, {START[:7]} 〜 {END[:7]})',
                 fontsize=14, fontweight='bold', y=0.995)
    out = os.path.join(OUT_DIR, 'result.png')
    plt.savefig(out, dpi=130, bbox_inches='tight')
    print(f"Saved: {out}")

    # CSV
    a_pool.to_csv(os.path.join(OUT_DIR, 'A_daily_panel.csv'), index=False)
    b_pool.to_csv(os.path.join(OUT_DIR, 'B_obv_divergence.csv'), index=False)
    a_sym.to_csv(os.path.join(OUT_DIR, 'A_per_symbol_spearman.csv'), index=False)
    if a_q5 is not None:
        a_q5.to_csv(os.path.join(OUT_DIR, 'A_vol_ratio_q5_fwd5.csv'), index=False)
    if a_q10 is not None:
        a_q10.to_csv(os.path.join(OUT_DIR, 'A_vol_ratio_q5_fwd10.csv'), index=False)
    if not b_q5_quad.empty:
        b_q5_quad.to_csv(os.path.join(OUT_DIR, 'B_quadrant_fwd5.csv'), index=False)
    if not b_q10_quad.empty:
        b_q10_quad.to_csv(os.path.join(OUT_DIR, 'B_quadrant_fwd10.csv'), index=False)
    print("CSV saved")

    print("\n" + "=" * 70)
    print(" 完了")
    print("=" * 70)


if __name__ == "__main__":
    main()
