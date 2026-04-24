"""
出来高 → 方向性 予測力の検証

仮説の一般化:
  出来高に関する単独シグナル(エントリー条件や価格水準と独立)で
  「上がるか下がるか」の方向予測ができるか?

既存の 20260422_orb_volume では「出来高フィルター付きORB」を検証し
「出来高単独の方向示唆は弱く、銘柄特性を増幅するだけ」と結論。
今回はより一般的な5つの出来高シグナルで方向予測力を検証する。

検証する5つのシグナル:
  S1. 寄付30分の出来高レシオ (当日/20日中央値) → 残り時間(9:30-15:30)の方向
  S2. 1分足出来高スパイク (20bar med の N倍超) → 直後5分/30分の方向
       更に分解: 上昇足スパイク vs 下降足スパイクで forward return は違うか
  S3. 寄付の出来高加速度 (9:15-9:30 / 9:00-9:15) → 9:30-15:30の方向
  S4. 前場累計出来高レシオ (9:00-11:30 / 20日中央値) → 後場(12:30-15:30)の方向
  S5. ギャップ × 出来高 (前日終値からの寄付ギャップ × 寄付30分出来高レシオ)
       継続するのか反転するのか、4象限で検証

対象: 非鉄3 + 半導体5 = 8銘柄 (既存分析と同一)
期間: 2025-04-01 〜 2026-04-21
評価: スピアマン相関、5分位バケット別 mean_bps & t-stat
コスト: 方向予測力の純粋検証のため、まずはコスト無視で見る
        +最終サマリーで 4bps 差し引き評価も併記
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '20260421_common')))
import mdutil as U
import pandas as pd
import numpy as np
from scipy import stats
plt = U.matplotlib_jp()

SYMS = U.NONFERROUS + U.SEMICON  # 8 銘柄
OUT_DIR = os.path.dirname(__file__)

# ---------- 共通ユーティリティ ----------

def load_1min(sym):
    df = U.fetch_intraday(sym).dropna(subset=['open','high','low','close'])
    h, m = df.index.hour, df.index.minute
    mask = (((h==9)) | ((h==10)) | ((h==11)&(m<=30)) |
            ((h==12)&(m>=30)) | ((h==13)|(h==14)) | ((h==15)&(m<=30)))
    df = df[mask].copy()
    df['mo'] = (df.index.hour - 9) * 60 + df.index.minute  # 9:00=0, 15:30=390
    return df


def quintile_stats(x, y, labels=('Q1(低)','Q2','Q3','Q4','Q5(高)')):
    """xの5分位でyをバケット分け。返り値: DataFrame"""
    df = pd.DataFrame({'x': x, 'y': y}).dropna()
    if len(df) < 25:
        return None
    try:
        df['q'] = pd.qcut(df['x'], 5, labels=labels, duplicates='drop')
    except ValueError:
        return None
    rows = []
    for q in df['q'].cat.categories:
        sub = df[df['q'] == q]['y']
        if len(sub) == 0: continue
        m, s = sub.mean(), sub.std()
        tstat = m / (s/np.sqrt(len(sub))) if s > 0 else 0
        rows.append({'bucket': str(q), 'N': len(sub),
                     'mean_bps': m, 'median_bps': float(sub.median()),
                     'wr': (sub>0).mean()*100, 't_stat': tstat})
    return pd.DataFrame(rows)


def fmt_df(df, floats=('mean_bps','median_bps','wr','t_stat')):
    fs = {}
    for c in floats:
        if c in df.columns:
            if c in ('mean_bps','median_bps'):
                fs[c] = '{:+.1f}'.format
            elif c in ('wr',):
                fs[c] = '{:.1f}'.format
            else:
                fs[c] = '{:+.2f}'.format
    return df.to_string(index=False, formatters=fs)


# ---------- シグナル1: 寄付30分出来高レシオ → 残り時間方向 ----------

def signal1_open_volratio(df, or_min=30):
    """
    per-day:
      vol_ratio = open_vol / median_20(open_vol)
      fwd_ret_bps = (close(15:30) - close(or_min)) / close(or_min) * 10000
    """
    rows = []
    for d in sorted(set(df.index.date)):
        day = df[df.index.date == d]
        if len(day) < 200: continue
        ow = day[day['mo'] < or_min]
        post = day[(day['mo'] >= or_min) & (day['mo'] <= 390)]
        if len(ow) < or_min*0.5 or len(post) < 30: continue
        open_vol = ow['volume'].sum()
        entry = ow['close'].iloc[-1]
        exit_ = post['close'].iloc[-1]
        if entry <= 0: continue
        rows.append({'date': d, 'open_vol': open_vol,
                     'fwd_bps': (exit_/entry - 1)*10000})
    tdf = pd.DataFrame(rows).sort_values('date').reset_index(drop=True)
    if tdf.empty: return tdf
    tdf['med20'] = tdf['open_vol'].rolling(20, min_periods=10).median().shift(1)
    tdf['vol_ratio'] = tdf['open_vol'] / tdf['med20']
    return tdf


# ---------- シグナル2: 1分足出来高スパイク → 直後方向 ----------

def signal2_spike(df, window=20, spike_mult=3.0, horizons=(5, 30)):
    """
    各分足について:
      med = 過去20bar(同日, 前時刻)の volume 中央値
      spike: volume >= med * spike_mult
    spike足のdirection (close>open: up, close<open: down, ==: neutral)
    forward return: spike足のclose → +h 分後のclose
    """
    out = []
    for d in sorted(set(df.index.date)):
        day = df[df.index.date == d].copy()
        if len(day) < 100: continue
        # 出来高のローリング中央値 (shift(1) で自分自身を除外)
        day['vol_med'] = day['volume'].rolling(window, min_periods=5).median().shift(1)
        day['spike'] = day['volume'] / day['vol_med']
        for h in horizons:
            day[f'fwd_{h}'] = day['close'].shift(-h) / day['close'] - 1
        # 方向分類: body 幅ベース
        body_bps = (day['close']/day['open'] - 1) * 10000
        day['bar_dir'] = np.where(body_bps > 1, 'up',
                                  np.where(body_bps < -1, 'down', 'flat'))
        hit = day[day['spike'] >= spike_mult].copy()
        hit['date'] = d
        out.append(hit[['date','mo','volume','vol_med','spike','bar_dir',
                        'open','close'] + [f'fwd_{h}' for h in horizons]])
    if not out: return pd.DataFrame()
    return pd.concat(out, ignore_index=True)


# ---------- シグナル3: 寄付出来高加速度 → 残り時間方向 ----------

def signal3_acceleration(df):
    """
    acc = vol(9:15-9:30) / vol(9:00-9:15)
    fwd: 9:30 close → 15:30 close
    """
    rows = []
    for d in sorted(set(df.index.date)):
        day = df[df.index.date == d]
        if len(day) < 200: continue
        a = day[(day['mo'] >= 0) & (day['mo'] < 15)]
        b = day[(day['mo'] >= 15) & (day['mo'] < 30)]
        post = day[(day['mo'] >= 30) & (day['mo'] <= 390)]
        if len(a) < 8 or len(b) < 8 or len(post) < 30: continue
        va, vb = a['volume'].sum(), b['volume'].sum()
        if va <= 0: continue
        acc = vb / va
        entry = b['close'].iloc[-1]
        exit_ = post['close'].iloc[-1]
        if entry <= 0: continue
        rows.append({'date': d, 'acc': acc,
                     'fwd_bps': (exit_/entry - 1)*10000})
    return pd.DataFrame(rows)


# ---------- シグナル4: 前場出来高 → 後場方向 ----------

def signal4_morning(df):
    """
    morning_vol_ratio = vol(9:00-11:30) / median_20(同)
    fwd: 12:30 close → 15:30 close
    """
    rows = []
    for d in sorted(set(df.index.date)):
        day = df[df.index.date == d]
        if len(day) < 200: continue
        morn = day[(day['mo'] >= 0) & (day['mo'] <= 150)]
        aft = day[(day['mo'] >= 210) & (day['mo'] <= 390)]
        if len(morn) < 100 or len(aft) < 100: continue
        mv = morn['volume'].sum()
        entry = aft['close'].iloc[0]   # 12:30 寄付近辺
        exit_ = aft['close'].iloc[-1]
        if entry <= 0: continue
        rows.append({'date': d, 'morning_vol': mv,
                     'fwd_bps': (exit_/entry - 1)*10000})
    tdf = pd.DataFrame(rows).sort_values('date').reset_index(drop=True)
    if tdf.empty: return tdf
    tdf['med20'] = tdf['morning_vol'].rolling(20, min_periods=10).median().shift(1)
    tdf['vol_ratio'] = tdf['morning_vol'] / tdf['med20']
    return tdf


# ---------- シグナル5: ギャップ × 出来高 象限 ----------

def signal5_gap_vol(df, or_min=30):
    """
    per-day:
      gap_bps = (open(9:00) - close(prev 15:30)) / close(prev 15:30) * 10000
      vol_ratio = open30min_vol / median_20
      fwd_bps = (close(15:30) - open(9:00)) / open(9:00) * 10000
    4象限:
      UG+HV (up-gap + high-vol) / UG+LV / DG+HV / DG+LV  (HV: vol_ratio>=1.3)
    """
    rows = []
    prev_close = None
    for d in sorted(set(df.index.date)):
        day = df[df.index.date == d]
        if len(day) < 200:
            prev_close = None; continue
        ow = day[day['mo'] < or_min]
        if len(ow) < or_min*0.5:
            prev_close = None; continue
        day_open = day[day['mo'] == 0]
        if len(day_open) == 0:
            prev_close = day['close'].iloc[-1]; continue
        open_px = day_open['open'].iloc[0]
        close_px = day['close'].iloc[-1]
        open_vol = ow['volume'].sum()
        if open_px <= 0:
            prev_close = close_px; continue
        gap_bps = (open_px/prev_close - 1)*10000 if prev_close else np.nan
        fwd = (close_px/open_px - 1)*10000
        rows.append({'date': d, 'open_px': open_px, 'open_vol': open_vol,
                     'gap_bps': gap_bps, 'fwd_bps': fwd})
        prev_close = close_px
    tdf = pd.DataFrame(rows).sort_values('date').reset_index(drop=True)
    if tdf.empty: return tdf
    tdf['med20'] = tdf['open_vol'].rolling(20, min_periods=10).median().shift(1)
    tdf['vol_ratio'] = tdf['open_vol'] / tdf['med20']
    return tdf


# ---------- メイン ----------

def corr_report(x, y, name=""):
    df = pd.DataFrame({'x': x, 'y': y}).dropna()
    if len(df) < 10:
        return None
    sp = stats.spearmanr(df['x'], df['y'])
    pe = stats.pearsonr(df['x'], df['y'])
    print(f"  {name}: N={len(df)}  Spearman={sp.correlation:+.3f} (p={sp.pvalue:.3f})  "
          f"Pearson={pe[0]:+.3f} (p={pe[1]:.3f})")
    return {'n': len(df), 'spearman_r': sp.correlation, 'spearman_p': sp.pvalue,
            'pearson_r': pe[0], 'pearson_p': pe[1]}


def main():
    print("=" * 70)
    print(" 出来高 → 方向性 予測力の検証 (5 signals)")
    print("=" * 70)
    print(f"対象 {len(SYMS)} 銘柄: " + ", ".join([s for s,_ in SYMS]))
    print()

    loaded = {}
    for sym, name in SYMS:
        loaded[sym] = load_1min(sym)
        print(f"  loaded {sym} {name}: {len(loaded[sym])} rows")
    print()

    # ========== S1: 寄付30分 vol_ratio → 残り時間 ==========
    print("=" * 70)
    print(" [S1] 寄付30分出来高レシオ → 9:30-15:30 方向")
    print("=" * 70)
    s1_all = []
    s1_corrs = []
    for sym, name in SYMS:
        tdf = signal1_open_volratio(loaded[sym])
        tdf['sym'] = sym
        s1_all.append(tdf)
        print(f"\n[{sym} {name}]")
        c = corr_report(tdf['vol_ratio'], tdf['fwd_bps'], 'vol_ratio vs fwd_bps')
        if c is not None:
            c.update({'sym': sym, 'name': name})
            s1_corrs.append(c)
    s1_all = pd.concat(s1_all, ignore_index=True)
    print(f"\n[集約 8銘柄] vol_ratio vs fwd_bps:")
    s1_agg_corr = corr_report(s1_all['vol_ratio'], s1_all['fwd_bps'], 'POOLED')
    print("\n[集約 5分位バケット分析]")
    q1 = quintile_stats(s1_all['vol_ratio'], s1_all['fwd_bps'])
    if q1 is not None:
        print(fmt_df(q1))

    # ========== S2: 1分足スパイク → 直後方向 ==========
    print("\n" + "=" * 70)
    print(" [S2] 1分足出来高スパイク (20bar中央値の3倍以上) → 直後方向")
    print("=" * 70)
    s2_all = []
    for sym, name in SYMS:
        hit = signal2_spike(loaded[sym], window=20, spike_mult=3.0,
                            horizons=(5, 30))
        if hit.empty: continue
        hit['sym'] = sym
        s2_all.append(hit)
        # 上昇足スパイク vs 下降足スパイク の forward
        for h in (5, 30):
            print(f"\n[{sym} {name}] horizon={h}min")
            for d in ('up','down','flat'):
                sub = hit[hit['bar_dir']==d]
                if len(sub) < 20: continue
                fw = sub[f'fwd_{h}'].dropna() * 10000
                if len(fw) == 0: continue
                m, s = fw.mean(), fw.std()
                t = m/(s/np.sqrt(len(fw))) if s>0 else 0
                print(f"  bar_dir={d:4s}  N={len(fw):4d}  mean_bps={m:+7.2f}  "
                      f"wr={(fw>0).mean()*100:5.1f}%  t={t:+5.2f}")
    s2_all = pd.concat(s2_all, ignore_index=True) if s2_all else pd.DataFrame()

    # 集約表: スパイク閾値別 forward return
    print("\n[集約] スパイク倍率別の forward 5min リターン (bar_dir=up)")
    if not s2_all.empty:
        up = s2_all[s2_all['bar_dir'] == 'up']
        rows = []
        for mult in [2.0, 3.0, 5.0, 8.0, 12.0]:
            sub = up[up['spike'] >= mult]['fwd_5'].dropna() * 10000
            if len(sub) < 10: continue
            m, s = sub.mean(), sub.std()
            t = m/(s/np.sqrt(len(sub))) if s>0 else 0
            rows.append({'spike>=': mult, 'N': len(sub),
                         'mean_bps': m, 'wr': (sub>0).mean()*100, 't_stat': t})
        if rows:
            print(fmt_df(pd.DataFrame(rows)))
        print("\n[集約] スパイク倍率別の forward 5min リターン (bar_dir=down)")
        dn = s2_all[s2_all['bar_dir'] == 'down']
        rows = []
        for mult in [2.0, 3.0, 5.0, 8.0, 12.0]:
            sub = dn[dn['spike'] >= mult]['fwd_5'].dropna() * 10000
            if len(sub) < 10: continue
            m, s = sub.mean(), sub.std()
            t = m/(s/np.sqrt(len(sub))) if s>0 else 0
            rows.append({'spike>=': mult, 'N': len(sub),
                         'mean_bps': m, 'wr': (sub>0).mean()*100, 't_stat': t})
        if rows:
            print(fmt_df(pd.DataFrame(rows)))

    # ========== S3: 寄付加速度 → 残り時間 ==========
    print("\n" + "=" * 70)
    print(" [S3] 寄付出来高加速度 (9:15-9:30 / 9:00-9:15) → 9:30-15:30")
    print("=" * 70)
    s3_all = []
    for sym, name in SYMS:
        tdf = signal3_acceleration(loaded[sym])
        tdf['sym'] = sym
        s3_all.append(tdf)
        print(f"\n[{sym} {name}]")
        corr_report(tdf['acc'], tdf['fwd_bps'], 'acc vs fwd_bps')
    s3_all = pd.concat(s3_all, ignore_index=True)
    print(f"\n[集約] acc vs fwd_bps:")
    s3_agg_corr = corr_report(s3_all['acc'], s3_all['fwd_bps'], 'POOLED')
    print("\n[集約 5分位バケット分析]")
    q3 = quintile_stats(s3_all['acc'], s3_all['fwd_bps'])
    if q3 is not None:
        print(fmt_df(q3))

    # ========== S4: 前場出来高 → 後場 ==========
    print("\n" + "=" * 70)
    print(" [S4] 前場累計出来高レシオ (9:00-11:30 / 20日中央値) → 後場 12:30-15:30")
    print("=" * 70)
    s4_all = []
    for sym, name in SYMS:
        tdf = signal4_morning(loaded[sym])
        tdf['sym'] = sym
        s4_all.append(tdf)
        print(f"\n[{sym} {name}]")
        corr_report(tdf['vol_ratio'], tdf['fwd_bps'], 'vol_ratio vs fwd_bps')
    s4_all = pd.concat(s4_all, ignore_index=True)
    print(f"\n[集約] vol_ratio vs fwd_bps:")
    s4_agg_corr = corr_report(s4_all['vol_ratio'], s4_all['fwd_bps'], 'POOLED')
    print("\n[集約 5分位バケット分析]")
    q4 = quintile_stats(s4_all['vol_ratio'], s4_all['fwd_bps'])
    if q4 is not None:
        print(fmt_df(q4))

    # ========== S5: ギャップ × 出来高 象限 ==========
    print("\n" + "=" * 70)
    print(" [S5] ギャップ × 出来高 4象限 → 寄付〜大引け方向")
    print("=" * 70)
    s5_all = []
    for sym, name in SYMS:
        tdf = signal5_gap_vol(loaded[sym])
        tdf['sym'] = sym
        s5_all.append(tdf)
    s5_all = pd.concat(s5_all, ignore_index=True).dropna(subset=['gap_bps','vol_ratio'])
    print(f"集約 N={len(s5_all)}")
    # 4象限: gap の符号 × vol_ratio >= 1.3 or not
    s5_all['gap_dir'] = np.where(s5_all['gap_bps'] > 0, 'UG', 'DG')
    s5_all['vol_dir'] = np.where(s5_all['vol_ratio'] >= 1.3, 'HV', 'LV')
    s5_all['quad'] = s5_all['gap_dir'] + '_' + s5_all['vol_dir']
    rows = []
    for q in ['UG_HV','UG_LV','DG_HV','DG_LV']:
        sub = s5_all[s5_all['quad'] == q]['fwd_bps']
        if len(sub) < 10: continue
        m, s = sub.mean(), sub.std()
        t = m/(s/np.sqrt(len(sub))) if s>0 else 0
        rows.append({'quad': q, 'N': len(sub), 'mean_bps': m,
                     'wr': (sub>0).mean()*100, 't_stat': t})
    print(fmt_df(pd.DataFrame(rows)))
    print("\n解釈のキー:")
    print("  UG_HV: UpGap + HighVol → mean+=継続 / mean-=反転")
    print("  DG_HV: DownGap + HighVol → 同上")
    print("  High/Low volの差が方向性のシグナルになりうるか?\n")

    # より細かく: ギャップ5分位 × vol_ratio 5分位 (上位下位のみ)
    print("[詳細] gap絶対値 + vol_ratio の相互作用")
    # 絶対gapが大きい日 (上下位20%) のみで、vol_ratio の方向効果
    gap_thr_hi = s5_all['gap_bps'].quantile(0.8)
    gap_thr_lo = s5_all['gap_bps'].quantile(0.2)
    big_ug = s5_all[s5_all['gap_bps'] >= gap_thr_hi]
    big_dg = s5_all[s5_all['gap_bps'] <= gap_thr_lo]
    print(f"\n大きなUpGap (>={gap_thr_hi:.0f}bps, N={len(big_ug)}): vol_ratio バケット別 fwd")
    q5u = quintile_stats(big_ug['vol_ratio'], big_ug['fwd_bps'])
    if q5u is not None: print(fmt_df(q5u))
    print(f"\n大きなDownGap (<={gap_thr_lo:.0f}bps, N={len(big_dg)}): vol_ratio バケット別 fwd")
    q5d = quintile_stats(big_dg['vol_ratio'], big_dg['fwd_bps'])
    if q5d is not None: print(fmt_df(q5d))

    # ========== 可視化 ==========
    print("\n可視化中...")
    fig = plt.figure(figsize=(18, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.55, wspace=0.35)

    # (0,0) S1: vol_ratio 5分位の mean_bps
    ax = fig.add_subplot(gs[0,0])
    if q1 is not None:
        ax.bar(range(len(q1)), q1['mean_bps'], color=[
            '#d62728' if v<0 else '#2ca02c' for v in q1['mean_bps']])
        ax.set_xticks(range(len(q1))); ax.set_xticklabels(q1['bucket'], rotation=0, fontsize=8)
        ax.set_ylabel('mean_bps'); ax.axhline(0, color='k', lw=0.5)
        for i, (m, n) in enumerate(zip(q1['mean_bps'], q1['N'])):
            ax.text(i, m, f"{m:+.1f}\nN={n}", ha='center', va='bottom' if m>0 else 'top', fontsize=7)
    ax.set_title('[S1] 寄付30分 vol_ratio バケット別\n残り時間 fwd_bps', fontweight='bold', fontsize=10)
    ax.grid(alpha=0.3)

    # (0,1) S1: 銘柄別 Spearman
    ax = fig.add_subplot(gs[0,1])
    if s1_corrs:
        sc = pd.DataFrame(s1_corrs)
        colors = ['#d62728' if r<0 else '#2ca02c' for r in sc['spearman_r']]
        ax.barh(range(len(sc)), sc['spearman_r'], color=colors)
        ax.set_yticks(range(len(sc))); ax.set_yticklabels([f"{r['sym']} {r['name']}" for _,r in sc.iterrows()], fontsize=8)
        ax.axvline(0, color='k', lw=0.5)
        for i, (r, p) in enumerate(zip(sc['spearman_r'], sc['spearman_p'])):
            sig = '*' if p<0.05 else ''
            ax.text(r, i, f" {r:+.2f}{sig}", va='center', fontsize=8)
    ax.set_title('[S1] 銘柄別 Spearman r', fontweight='bold', fontsize=10)
    ax.set_xlabel('Spearman相関 (* = p<0.05)')

    # (0,2) S1: 散布図 (集約)
    ax = fig.add_subplot(gs[0,2])
    x, y = s1_all['vol_ratio'].values, s1_all['fwd_bps'].values
    mask = ~(np.isnan(x) | np.isnan(y))
    ax.scatter(x[mask], y[mask], s=5, alpha=0.3, color='#1f77b4')
    ax.axhline(0, color='k', lw=0.5); ax.axvline(1.0, color='gray', lw=0.5, ls='--')
    ax.set_xlim(0, 5); ax.set_ylim(-500, 500)
    ax.set_xlabel('vol_ratio (寄付30分)'); ax.set_ylabel('fwd_bps (9:30-15:30)')
    ax.set_title('[S1] 散布図 (集約 8銘柄)', fontweight='bold', fontsize=10)
    ax.grid(alpha=0.3)

    # (1,0) S2: スパイク倍率 × 方向別 forward 5min
    ax = fig.add_subplot(gs[1,0])
    if not s2_all.empty:
        mults = [2.0, 3.0, 5.0, 8.0]
        up_means, dn_means = [], []
        up_ns, dn_ns = [], []
        for mult in mults:
            u = s2_all[(s2_all['bar_dir']=='up') & (s2_all['spike']>=mult)]['fwd_5'].dropna()*10000
            d = s2_all[(s2_all['bar_dir']=='down') & (s2_all['spike']>=mult)]['fwd_5'].dropna()*10000
            up_means.append(u.mean() if len(u)>0 else 0); up_ns.append(len(u))
            dn_means.append(d.mean() if len(d)>0 else 0); dn_ns.append(len(d))
        x = np.arange(len(mults))
        w = 0.4
        ax.bar(x-w/2, up_means, w, label='up spike', color='#2ca02c')
        ax.bar(x+w/2, dn_means, w, label='down spike', color='#d62728')
        ax.axhline(0, color='k', lw=0.5)
        ax.set_xticks(x); ax.set_xticklabels([f">={m}x" for m in mults])
        ax.set_ylabel('mean fwd_bps (5min)'); ax.legend()
        for i, (u, d, un, dn) in enumerate(zip(up_means, dn_means, up_ns, dn_ns)):
            ax.text(i-w/2, u, f"{u:+.1f}\nN={un}", ha='center',
                    va='bottom' if u>0 else 'top', fontsize=7)
            ax.text(i+w/2, d, f"{d:+.1f}\nN={dn}", ha='center',
                    va='bottom' if d>0 else 'top', fontsize=7)
    ax.set_title('[S2] スパイク倍率 × 足方向\n→ 直後5分 mean_bps', fontweight='bold', fontsize=10)
    ax.grid(alpha=0.3)

    # (1,1) S2: horizon 5min vs 30min (up spike)
    ax = fig.add_subplot(gs[1,1])
    if not s2_all.empty:
        up = s2_all[s2_all['bar_dir']=='up']
        mults = [2.0, 3.0, 5.0, 8.0]
        m5 = [up[up['spike']>=m]['fwd_5'].mean()*10000 for m in mults]
        m30 = [up[up['spike']>=m]['fwd_30'].mean()*10000 for m in mults]
        x = np.arange(len(mults))
        w = 0.4
        ax.bar(x-w/2, m5, w, label='+5min', color='#1f77b4')
        ax.bar(x+w/2, m30, w, label='+30min', color='#ff7f0e')
        ax.axhline(0, color='k', lw=0.5)
        ax.set_xticks(x); ax.set_xticklabels([f">={m}x" for m in mults])
        ax.set_ylabel('mean fwd_bps'); ax.legend()
    ax.set_title('[S2] up-spike: horizon 5 vs 30min', fontweight='bold', fontsize=10)
    ax.grid(alpha=0.3)

    # (1,2) S2: bar_dir 分布
    ax = fig.add_subplot(gs[1,2])
    if not s2_all.empty:
        cnt = s2_all['bar_dir'].value_counts()
        colors_map = {'up':'#2ca02c','down':'#d62728','flat':'#7f7f7f'}
        ax.bar(cnt.index, cnt.values, color=[colors_map.get(k,'gray') for k in cnt.index])
        for i, v in enumerate(cnt.values):
            ax.text(i, v, f"{v}", ha='center', va='bottom', fontsize=9)
    ax.set_title('[S2] スパイク足の方向分布', fontweight='bold', fontsize=10)
    ax.set_ylabel('count')

    # (2,0) S3: acc 5分位
    ax = fig.add_subplot(gs[2,0])
    if q3 is not None:
        ax.bar(range(len(q3)), q3['mean_bps'], color=[
            '#d62728' if v<0 else '#2ca02c' for v in q3['mean_bps']])
        ax.set_xticks(range(len(q3))); ax.set_xticklabels(q3['bucket'], rotation=0, fontsize=8)
        ax.axhline(0, color='k', lw=0.5); ax.set_ylabel('mean_bps')
        for i, (m, n) in enumerate(zip(q3['mean_bps'], q3['N'])):
            ax.text(i, m, f"{m:+.1f}\nN={n}", ha='center',
                    va='bottom' if m>0 else 'top', fontsize=7)
    ax.set_title('[S3] 寄付加速度 バケット別 fwd_bps', fontweight='bold', fontsize=10)
    ax.grid(alpha=0.3)

    # (2,1) S4: morning vol_ratio 5分位 → 後場
    ax = fig.add_subplot(gs[2,1])
    if q4 is not None:
        ax.bar(range(len(q4)), q4['mean_bps'], color=[
            '#d62728' if v<0 else '#2ca02c' for v in q4['mean_bps']])
        ax.set_xticks(range(len(q4))); ax.set_xticklabels(q4['bucket'], rotation=0, fontsize=8)
        ax.axhline(0, color='k', lw=0.5); ax.set_ylabel('mean_bps')
        for i, (m, n) in enumerate(zip(q4['mean_bps'], q4['N'])):
            ax.text(i, m, f"{m:+.1f}\nN={n}", ha='center',
                    va='bottom' if m>0 else 'top', fontsize=7)
    ax.set_title('[S4] 前場vol_ratio バケット別\n→ 後場 fwd_bps', fontweight='bold', fontsize=10)
    ax.grid(alpha=0.3)

    # (2,2) S5: 4象限 mean_bps
    ax = fig.add_subplot(gs[2,2])
    q5_rows = []
    for q in ['UG_HV','UG_LV','DG_HV','DG_LV']:
        sub = s5_all[s5_all['quad'] == q]['fwd_bps']
        if len(sub) < 10: continue
        q5_rows.append({'quad': q, 'N': len(sub), 'mean': sub.mean()})
    if q5_rows:
        q5df = pd.DataFrame(q5_rows)
        colors = {'UG_HV':'#2ca02c','UG_LV':'#98df8a','DG_HV':'#d62728','DG_LV':'#ff9896'}
        ax.bar(q5df['quad'], q5df['mean'],
               color=[colors.get(q,'gray') for q in q5df['quad']])
        ax.axhline(0, color='k', lw=0.5); ax.set_ylabel('mean fwd_bps')
        for i, (m, n) in enumerate(zip(q5df['mean'], q5df['N'])):
            ax.text(i, m, f"{m:+.1f}\nN={n}", ha='center',
                    va='bottom' if m>0 else 'top', fontsize=8)
    ax.set_title('[S5] ギャップ×出来高 4象限\n→ 寄付-大引け fwd_bps', fontweight='bold', fontsize=10)
    ax.grid(alpha=0.3)

    # (3,0) S5: big UG × vol_ratio 5分位
    ax = fig.add_subplot(gs[3,0])
    if q5u is not None:
        ax.bar(range(len(q5u)), q5u['mean_bps'], color=[
            '#d62728' if v<0 else '#2ca02c' for v in q5u['mean_bps']])
        ax.set_xticks(range(len(q5u))); ax.set_xticklabels(q5u['bucket'], rotation=0, fontsize=7)
        ax.axhline(0, color='k', lw=0.5); ax.set_ylabel('mean_bps')
        for i, (m, n) in enumerate(zip(q5u['mean_bps'], q5u['N'])):
            ax.text(i, m, f"{m:+.0f}\nN={n}", ha='center',
                    va='bottom' if m>0 else 'top', fontsize=7)
    ax.set_title('[S5] 大きなUpGap日: vol_ratio別\n→ fwd_bps', fontweight='bold', fontsize=10)
    ax.grid(alpha=0.3)

    # (3,1) S5: big DG × vol_ratio 5分位
    ax = fig.add_subplot(gs[3,1])
    if q5d is not None:
        ax.bar(range(len(q5d)), q5d['mean_bps'], color=[
            '#d62728' if v<0 else '#2ca02c' for v in q5d['mean_bps']])
        ax.set_xticks(range(len(q5d))); ax.set_xticklabels(q5d['bucket'], rotation=0, fontsize=7)
        ax.axhline(0, color='k', lw=0.5); ax.set_ylabel('mean_bps')
        for i, (m, n) in enumerate(zip(q5d['mean_bps'], q5d['N'])):
            ax.text(i, m, f"{m:+.0f}\nN={n}", ha='center',
                    va='bottom' if m>0 else 'top', fontsize=7)
    ax.set_title('[S5] 大きなDownGap日: vol_ratio別\n→ fwd_bps', fontweight='bold', fontsize=10)
    ax.grid(alpha=0.3)

    # (3,2) サマリー
    ax = fig.add_subplot(gs[3,2]); ax.axis('off')
    summary = f"""
サマリー: 出来高 → 方向性 予測力

S1 寄付30分vol_ratio:
  pooled Spearman r = {s1_agg_corr['spearman_r']:+.3f}
  p = {s1_agg_corr['spearman_p']:.3f}
  N = {s1_agg_corr['n']}

S3 寄付加速度:
  pooled Spearman r = {s3_agg_corr['spearman_r']:+.3f}
  p = {s3_agg_corr['spearman_p']:.3f}
  N = {s3_agg_corr['n']}

S4 前場vol_ratio:
  pooled Spearman r = {s4_agg_corr['spearman_r']:+.3f}
  p = {s4_agg_corr['spearman_p']:.3f}
  N = {s4_agg_corr['n']}

判定基準:
  |r|<0.05: 実質無相関
  |r|<0.10: 弱い示唆
  |r|>=0.10: 取引可能性あり
"""
    ax.text(0.02, 0.98, summary, family='monospace', fontsize=9,
            va='top', ha='left', transform=ax.transAxes)

    plt.suptitle('出来高シグナル → 方向性予測力の検証 (8銘柄, 2025-04〜2026-04)',
                 fontsize=14, fontweight='bold', y=0.995)
    out = os.path.join(OUT_DIR, 'result.png')
    plt.savefig(out, dpi=130, bbox_inches='tight')
    print(f"Saved: {out}")

    # ========== CSV 出力 ==========
    s1_all.to_csv(os.path.join(OUT_DIR, 's1_open_volratio.csv'), index=False)
    if not s2_all.empty:
        s2_all.to_csv(os.path.join(OUT_DIR, 's2_spikes.csv'), index=False)
    s3_all.to_csv(os.path.join(OUT_DIR, 's3_acceleration.csv'), index=False)
    s4_all.to_csv(os.path.join(OUT_DIR, 's4_morning.csv'), index=False)
    s5_all.to_csv(os.path.join(OUT_DIR, 's5_gap_vol.csv'), index=False)
    print("CSV saved: s1..s5_*.csv")

    # per-symbol Spearman をまとめて保存
    if s1_corrs:
        pd.DataFrame(s1_corrs).to_csv(
            os.path.join(OUT_DIR, 's1_per_symbol_corr.csv'), index=False)

    print("\n" + "=" * 70)
    print(" 完了")
    print("=" * 70)


if __name__ == "__main__":
    main()
