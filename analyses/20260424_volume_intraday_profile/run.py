"""
日中出来高推移 → 方向性予測

2つのアプローチ:

A. 時刻帯パネル分析 (slot-level Spearman)
  30分スロット 9:00-9:30, 9:30-10:00, ..., 15:00-15:30 で
  vol_ratio = vol(slot) / median_20d(vol(same slot at same time-of-day))
  次スロットの close→close リターン と相関

  → どの時刻の出来高増加が最も強い方向示唆を持つか?
  仮説: 後場(特に14:30以降)の vol_ratio は引けまでの方向を示す

B. 擬似累積デルタ vs 価格のダイバージェンス
  各1分足で signed_vol = volume * sign(close - open)
  累積して cum_delta(t) を作る
  価格変化 z_p(t) = (close(t) - open_9:00) / open_9:00
  累積デルタ z_d(t) = cum_delta(t) / total_day_vol  (-1..+1 近辺に正規化)

  評価時点 t ∈ {10:30, 11:30, 13:30, 14:30}:
    price_rank = 各銘柄内の z_p(t) 百分位
    delta_rank = 各銘柄内の z_d(t) 百分位
    diverg = price_rank - delta_rank
      正: 価格は強いのにデルタが弱い = 弱気ダイバージェンス
      負: 価格は弱いのにデルタが強い = 強気ダイバージェンス

    4象限分類 (median でsplit):
      P+D+: モメンタム強気 (継続?)
      P+D-: 弱気divergence (反転?)
      P-D-: モメンタム弱気 (継続?)
      P-D+: 強気divergence (反転?)

  残時間 (t → 15:30) のリターンを各象限で比較

対象: 非鉄3 + 半導体5 = 8銘柄
期間: 2025-04-01 〜 2026-04-21
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '20260421_common')))
import mdutil as U
import pandas as pd
import numpy as np
from scipy import stats
plt = U.matplotlib_jp()

SYMS = U.NONFERROUS + U.SEMICON
OUT_DIR = os.path.dirname(__file__)

# 30分スロット (mo基準、9:00=0)
# 前場: 0-30, 30-60, 60-90, 90-120, 120-150 (5スロット)
# 後場: 210-240, 240-270, 270-300, 300-330, 330-360, 360-390 (6スロット)
SLOTS = [
    ('09:00-09:30', 0, 30),
    ('09:30-10:00', 30, 60),
    ('10:00-10:30', 60, 90),
    ('10:30-11:00', 90, 120),
    ('11:00-11:30', 120, 150),
    ('12:30-13:00', 210, 240),
    ('13:00-13:30', 240, 270),
    ('13:30-14:00', 270, 300),
    ('14:00-14:30', 300, 330),
    ('14:30-15:00', 330, 360),
    ('15:00-15:30', 360, 390),
]


def load_1min(sym):
    df = U.fetch_intraday(sym).dropna(subset=['open','high','low','close'])
    h, m = df.index.hour, df.index.minute
    mask = (((h==9)) | ((h==10)) | ((h==11)&(m<=30)) |
            ((h==12)&(m>=30)) | ((h==13)|(h==14)) | ((h==15)&(m<=30)))
    df = df[mask].copy()
    df['mo'] = (df.index.hour - 9) * 60 + df.index.minute
    return df


# ========== A. 時刻帯パネル ==========

def build_slot_panel(df):
    """日 × スロット のパネル: vol_ratio と next-slot return"""
    rows = []
    for d in sorted(set(df.index.date)):
        day = df[df.index.date == d]
        if len(day) < 200: continue
        slot_data = {}
        for name, s, e in SLOTS:
            seg = day[(day['mo'] >= s) & (day['mo'] < e)]
            if len(seg) < (e-s)*0.5: continue
            slot_data[name] = {
                'vol': seg['volume'].sum(),
                'close': seg['close'].iloc[-1],
                'open': seg['open'].iloc[0],
            }
        if len(slot_data) < 8: continue
        # next-slot return: slot の close → next slot の close
        slot_names = [n for n,_,_ in SLOTS]
        for i, n in enumerate(slot_names[:-1]):
            if n not in slot_data or slot_names[i+1] not in slot_data: continue
            cur = slot_data[n]
            nxt = slot_data[slot_names[i+1]]
            if cur['close'] <= 0: continue
            fwd_bps = (nxt['close']/cur['close'] - 1) * 10000
            rows.append({'date': d, 'slot': n, 'vol': cur['vol'],
                         'fwd_bps': fwd_bps})
        # また各スロットから 15:30 までのリターンも計算 (残時間方向)
        if '15:00-15:30' in slot_data:
            final_close = slot_data['15:00-15:30']['close']
            for n in slot_names[:-1]:
                if n not in slot_data: continue
                cur = slot_data[n]
                if cur['close'] <= 0: continue
                fwd_to_close = (final_close/cur['close'] - 1) * 10000
                # 既存rowを探して追加、なければ新規
                # 簡単のため別listに
    # vol_ratio は後で slot ごとに rolling median で計算
    panel = pd.DataFrame(rows)
    if panel.empty: return panel
    panel = panel.sort_values(['slot','date']).reset_index(drop=True)
    panel['med20'] = panel.groupby('slot')['vol'].transform(
        lambda x: x.rolling(20, min_periods=10).median().shift(1))
    panel['vol_ratio'] = panel['vol'] / panel['med20']
    return panel


def build_slot_to_close_panel(df):
    """日 × スロット: vol_ratio と そのスロット終了時点 → 15:30 のリターン"""
    rows = []
    for d in sorted(set(df.index.date)):
        day = df[df.index.date == d]
        if len(day) < 200: continue
        # 大引け close
        close_seg = day[(day['mo'] >= 360) & (day['mo'] < 390)]
        if len(close_seg) < 10: continue
        final_close = close_seg['close'].iloc[-1]
        for name, s, e in SLOTS:
            if name == '15:00-15:30': continue
            seg = day[(day['mo'] >= s) & (day['mo'] < e)]
            if len(seg) < (e-s)*0.5: continue
            cur_close = seg['close'].iloc[-1]
            if cur_close <= 0: continue
            rows.append({
                'date': d, 'slot': name,
                'vol': seg['volume'].sum(),
                'close_to_1530_bps': (final_close/cur_close - 1) * 10000,
            })
    panel = pd.DataFrame(rows)
    if panel.empty: return panel
    panel = panel.sort_values(['slot','date']).reset_index(drop=True)
    panel['med20'] = panel.groupby('slot')['vol'].transform(
        lambda x: x.rolling(20, min_periods=10).median().shift(1))
    panel['vol_ratio'] = panel['vol'] / panel['med20']
    return panel


# ========== B. 擬似累積デルタ ==========

def compute_cum_delta(df):
    """
    per-day: signed_vol = volume * sign(close-open)
    cum_delta の 10:30/11:30/13:30/14:30 時点の値と、
    対応する価格変化・残時間リターンを返す
    """
    # 評価時点 (mo)
    EVAL_POINTS = [
        ('10:30', 90),
        ('11:30', 150),
        ('13:30', 270),
        ('14:30', 330),
    ]
    rows = []
    for d in sorted(set(df.index.date)):
        day = df[df.index.date == d].copy()
        if len(day) < 200: continue
        open_9 = day[day['mo'] == 0]
        if len(open_9) == 0: continue
        open_px = open_9['open'].iloc[0]
        if open_px <= 0: continue
        close_seg = day[(day['mo'] >= 360) & (day['mo'] < 390)]
        if len(close_seg) < 10: continue
        final_close = close_seg['close'].iloc[-1]
        total_vol = day['volume'].sum()
        if total_vol <= 0: continue

        # 符号つき出来高
        sig = np.sign(day['close'] - day['open'])
        day['signed_vol'] = day['volume'] * sig
        day['cum_delta'] = day['signed_vol'].cumsum()

        for label, mo in EVAL_POINTS:
            t = day[day['mo'] <= mo]
            if len(t) < 30: continue
            cur_close = t['close'].iloc[-1]
            cur_cum = t['cum_delta'].iloc[-1]
            cur_vol = t['volume'].sum()
            if cur_close <= 0 or cur_vol <= 0: continue
            z_p_bps = (cur_close/open_px - 1) * 10000
            z_d = cur_cum / cur_vol  # -1..+1
            fwd_bps = (final_close/cur_close - 1) * 10000
            rows.append({
                'date': d, 'eval': label, 'eval_mo': mo,
                'z_p_bps': z_p_bps, 'z_d': z_d,
                'fwd_bps': fwd_bps, 'cur_close': cur_close
            })
    return pd.DataFrame(rows)


# ========== 共通ツール ==========

def quintile_stats(x, y, labels=('Q1(低)','Q2','Q3','Q4','Q5(高)')):
    df = pd.DataFrame({'x': x, 'y': y}).dropna()
    if len(df) < 25: return None
    try:
        df['q'] = pd.qcut(df['x'], 5, labels=labels, duplicates='drop')
    except ValueError:
        return None
    rows = []
    for q in df['q'].cat.categories:
        sub = df[df['q'] == q]['y']
        if len(sub) == 0: continue
        m, s = sub.mean(), sub.std()
        t = m/(s/np.sqrt(len(sub))) if s>0 else 0
        rows.append({'bucket': str(q), 'N': len(sub),
                     'mean_bps': m, 'wr': (sub>0).mean()*100, 't_stat': t})
    return pd.DataFrame(rows)


def fmt_df(df):
    fs = {}
    for c in df.columns:
        if c in ('mean_bps','median_bps'): fs[c] = '{:+.1f}'.format
        elif c == 'wr': fs[c] = '{:.1f}'.format
        elif c in ('t_stat','spearman_r','pearson_r'): fs[c] = '{:+.2f}'.format
        elif c in ('spearman_p','pearson_p'): fs[c] = '{:.3f}'.format
        elif c in ('z_p_bps','z_d'): fs[c] = '{:+.3f}'.format
    return df.to_string(index=False, formatters=fs)


# ========== メイン ==========

def main():
    print("=" * 70)
    print(" 日中出来高推移 → 方向性予測 (A: 時刻帯パネル + B: 擬似累積デルタ)")
    print("=" * 70)

    loaded = {}
    for sym, name in SYMS:
        loaded[sym] = load_1min(sym)
        print(f"  loaded {sym} {name}: {len(loaded[sym])} rows")
    print()

    # ======== A. 時刻帯パネル: vol_ratio vs next-slot return ========
    print("=" * 70)
    print(" [A1] 時刻帯 vol_ratio → 次30分スロット リターン (銘柄集約)")
    print("=" * 70)

    panels_next = []
    panels_toclose = []
    for sym, name in SYMS:
        p1 = build_slot_panel(loaded[sym])
        p1['sym'] = sym
        panels_next.append(p1)
        p2 = build_slot_to_close_panel(loaded[sym])
        p2['sym'] = sym
        panels_toclose.append(p2)
    panel_next = pd.concat(panels_next, ignore_index=True)
    panel_tc = pd.concat(panels_toclose, ignore_index=True)

    # スロット別 Spearman (集約)
    print(f"\n{'slot':<14} {'N':>6} {'r_spearman':>11} {'p':>7} "
          f"{'Q1_mean':>8} {'Q5_mean':>8} {'Q1-Q5':>8}")
    rows_a1 = []
    for name, s, e in SLOTS[:-1]:
        sub = panel_next[panel_next['slot'] == name].dropna(subset=['vol_ratio','fwd_bps'])
        if len(sub) < 100: continue
        sp = stats.spearmanr(sub['vol_ratio'], sub['fwd_bps'])
        q = quintile_stats(sub['vol_ratio'], sub['fwd_bps'])
        if q is None: continue
        q1_mean = q.iloc[0]['mean_bps']; q5_mean = q.iloc[-1]['mean_bps']
        row = {'slot': name, 'N': len(sub), 'spearman_r': sp.correlation,
               'p': sp.pvalue, 'Q1_mean': q1_mean, 'Q5_mean': q5_mean,
               'Q1-Q5': q1_mean - q5_mean}
        rows_a1.append(row)
        print(f"{name:<14} {len(sub):>6d} {sp.correlation:>+11.3f} "
              f"{sp.pvalue:>7.3f} {q1_mean:>+8.1f} {q5_mean:>+8.1f} "
              f"{q1_mean - q5_mean:>+8.1f}")
    a1_df = pd.DataFrame(rows_a1)

    # ======== A2: vol_ratio → 15:30までの残り方向 ========
    print("\n" + "=" * 70)
    print(" [A2] 時刻帯 vol_ratio → 15:30までの残り時間 リターン")
    print("=" * 70)
    print(f"\n{'slot':<14} {'N':>6} {'r_spearman':>11} {'p':>7} "
          f"{'Q1_mean':>8} {'Q5_mean':>8} {'Q5-Q1':>8}")
    rows_a2 = []
    for name, s, e in SLOTS[:-1]:
        sub = panel_tc[panel_tc['slot'] == name].dropna(subset=['vol_ratio','close_to_1530_bps'])
        if len(sub) < 100: continue
        sp = stats.spearmanr(sub['vol_ratio'], sub['close_to_1530_bps'])
        q = quintile_stats(sub['vol_ratio'], sub['close_to_1530_bps'])
        if q is None: continue
        q1_mean = q.iloc[0]['mean_bps']; q5_mean = q.iloc[-1]['mean_bps']
        row = {'slot': name, 'N': len(sub), 'spearman_r': sp.correlation,
               'p': sp.pvalue, 'Q1_mean': q1_mean, 'Q5_mean': q5_mean,
               'Q5-Q1': q5_mean - q1_mean}
        rows_a2.append(row)
        print(f"{name:<14} {len(sub):>6d} {sp.correlation:>+11.3f} "
              f"{sp.pvalue:>7.3f} {q1_mean:>+8.1f} {q5_mean:>+8.1f} "
              f"{q5_mean - q1_mean:>+8.1f}")
    a2_df = pd.DataFrame(rows_a2)

    # ======== A3: 銘柄 × スロット の Spearman ヒートマップ ========
    print("\n[A3] 銘柄 × スロット Spearman (vol_ratio vs 次スロットret)")
    heat = np.full((len(SYMS), len(SLOTS)-1), np.nan)
    for i, (sym, name) in enumerate(SYMS):
        for j, (slot_name, s, e) in enumerate(SLOTS[:-1]):
            sub = panel_next[(panel_next['sym']==sym) & (panel_next['slot']==slot_name)]
            sub = sub.dropna(subset=['vol_ratio','fwd_bps'])
            if len(sub) < 30: continue
            sp = stats.spearmanr(sub['vol_ratio'], sub['fwd_bps'])
            heat[i,j] = sp.correlation

    # ======== B. 擬似累積デルタ ========
    print("\n" + "=" * 70)
    print(" [B] 擬似累積デルタ vs 価格 ダイバージェンス → 残時間方向")
    print("=" * 70)

    b_all = []
    for sym, name in SYMS:
        cd = compute_cum_delta(loaded[sym])
        cd['sym'] = sym
        b_all.append(cd)
    b_all = pd.concat(b_all, ignore_index=True)

    print("\n[B1] 評価時点 × 4象限 (price & delta の銘柄内 median split)")
    b_results = {}
    for eval_label in ['10:30','11:30','13:30','14:30']:
        sub = b_all[b_all['eval'] == eval_label].copy()
        if len(sub) < 100: continue
        # 銘柄ごとに median split
        sub['p_sign'] = sub.groupby('sym')['z_p_bps'].transform(
            lambda x: np.where(x >= x.median(), 'P+', 'P-'))
        sub['d_sign'] = sub.groupby('sym')['z_d'].transform(
            lambda x: np.where(x >= x.median(), 'D+', 'D-'))
        sub['quad'] = sub['p_sign'] + sub['d_sign']
        print(f"\n== {eval_label} 時点 == (残時間: {eval_label} → 15:30)")
        rows = []
        for q in ['P+D+','P+D-','P-D+','P-D-']:
            qs = sub[sub['quad']==q]['fwd_bps']
            if len(qs) < 30: continue
            m, s_ = qs.mean(), qs.std()
            t = m/(s_/np.sqrt(len(qs))) if s_>0 else 0
            rows.append({'quad': q, 'N': len(qs), 'mean_bps': m,
                         'wr': (qs>0).mean()*100, 't_stat': t})
        rdf = pd.DataFrame(rows)
        print(fmt_df(rdf))
        b_results[eval_label] = rdf

        # ダイバージェンス: 強気(P-D+) vs 弱気(P+D-)
        pd_bullish = sub[sub['quad']=='P-D+']['fwd_bps']
        pd_bearish = sub[sub['quad']=='P+D-']['fwd_bps']
        if len(pd_bullish) >= 30 and len(pd_bearish) >= 30:
            diff_m = pd_bullish.mean() - pd_bearish.mean()
            t_test = stats.ttest_ind(pd_bullish, pd_bearish, equal_var=False)
            print(f"  強気div (P-D+) mean - 弱気div (P+D-) mean = {diff_m:+.2f} bps  "
                  f"Welch t={t_test.statistic:+.2f} (p={t_test.pvalue:.3f})")

    # ======== B2: 連続変数 (diverg = p_rank - d_rank) で相関 ========
    print("\n[B2] ダイバージェンス度 (price_rank - delta_rank) vs 残時間 Spearman")
    rows_b2 = []
    for eval_label in ['10:30','11:30','13:30','14:30']:
        sub = b_all[b_all['eval']==eval_label].copy()
        if len(sub) < 100: continue
        sub['p_rank'] = sub.groupby('sym')['z_p_bps'].rank(pct=True)
        sub['d_rank'] = sub.groupby('sym')['z_d'].rank(pct=True)
        sub['diverg'] = sub['p_rank'] - sub['d_rank']
        # 弱気divergence (diverg > 0) = 価格強いのにデルタ弱い → 反転?
        sp = stats.spearmanr(sub['diverg'], sub['fwd_bps'])
        q = quintile_stats(sub['diverg'], sub['fwd_bps'])
        if q is not None:
            q1m = q.iloc[0]['mean_bps']; q5m = q.iloc[-1]['mean_bps']
            print(f"  {eval_label}: N={len(sub):4d} r={sp.correlation:+.3f} (p={sp.pvalue:.3f})  "
                  f"Q1(bullish_div)_mean={q1m:+.1f}  Q5(bearish_div)_mean={q5m:+.1f}  "
                  f"Q1-Q5={q1m-q5m:+.1f}")
            rows_b2.append({'eval': eval_label, 'N': len(sub),
                           'spearman_r': sp.correlation, 'p': sp.pvalue,
                           'Q1_mean': q1m, 'Q5_mean': q5m, 'Q1-Q5': q1m-q5m})
    b2_df = pd.DataFrame(rows_b2)

    # ======== 可視化 ========
    print("\n可視化中...")
    fig = plt.figure(figsize=(18, 17))
    gs = fig.add_gridspec(4, 3, hspace=0.55, wspace=0.35)

    # (0,0) A1: スロット別 Spearman (vol_ratio → 次スロットret)
    ax = fig.add_subplot(gs[0,0])
    if not a1_df.empty:
        colors = ['#d62728' if r<0 else '#2ca02c' for r in a1_df['spearman_r']]
        ax.barh(range(len(a1_df)), a1_df['spearman_r'], color=colors)
        ax.set_yticks(range(len(a1_df))); ax.set_yticklabels(a1_df['slot'], fontsize=8)
        ax.axvline(0, color='k', lw=0.5)
        for i, (r, p) in enumerate(zip(a1_df['spearman_r'], a1_df['p'])):
            sig = '*' if p<0.05 else ''
            ax.text(r, i, f" {r:+.3f}{sig}", va='center', fontsize=8)
        ax.invert_yaxis()
    ax.set_title('[A1] スロット別 Spearman\nvol_ratio → 次30分ret', fontweight='bold', fontsize=10)
    ax.set_xlabel('Spearman相関 (* = p<0.05)')

    # (0,1) A2: スロット別 Spearman (vol_ratio → 15:30残)
    ax = fig.add_subplot(gs[0,1])
    if not a2_df.empty:
        colors = ['#d62728' if r<0 else '#2ca02c' for r in a2_df['spearman_r']]
        ax.barh(range(len(a2_df)), a2_df['spearman_r'], color=colors)
        ax.set_yticks(range(len(a2_df))); ax.set_yticklabels(a2_df['slot'], fontsize=8)
        ax.axvline(0, color='k', lw=0.5)
        for i, (r, p) in enumerate(zip(a2_df['spearman_r'], a2_df['p'])):
            sig = '*' if p<0.05 else ''
            ax.text(r, i, f" {r:+.3f}{sig}", va='center', fontsize=8)
        ax.invert_yaxis()
    ax.set_title('[A2] スロット別 Spearman\nvol_ratio → 15:30残', fontweight='bold', fontsize=10)
    ax.set_xlabel('Spearman相関')

    # (0,2) A1: Q1-Q5 bar (大きいほど低出来高優位)
    ax = fig.add_subplot(gs[0,2])
    if not a1_df.empty:
        colors = ['#d62728' if v<0 else '#2ca02c' for v in a1_df['Q1-Q5']]
        ax.barh(range(len(a1_df)), a1_df['Q1-Q5'], color=colors)
        ax.set_yticks(range(len(a1_df))); ax.set_yticklabels(a1_df['slot'], fontsize=8)
        ax.axvline(0, color='k', lw=0.5)
        for i, v in enumerate(a1_df['Q1-Q5']):
            ax.text(v, i, f" {v:+.1f}", va='center', fontsize=8)
        ax.invert_yaxis()
    ax.set_title('[A1] Q1-Q5 差分 bps\n(正: 低出来高slotが強い)', fontweight='bold', fontsize=10)
    ax.set_xlabel('Q1_mean - Q5_mean (bps)')

    # (1,0) A3: ヒートマップ 銘柄×スロット
    ax = fig.add_subplot(gs[1,0:2])
    vmax = np.nanmax(np.abs(heat))
    im = ax.imshow(heat, aspect='auto', cmap='RdYlGn', vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(SLOTS)-1))
    ax.set_xticklabels([s[0] for s in SLOTS[:-1]], rotation=45, fontsize=8)
    ax.set_yticks(range(len(SYMS)))
    ax.set_yticklabels([f"{s} {n}" for s,n in SYMS], fontsize=9)
    ax.set_title('[A3] 銘柄 × スロット Spearman\n(vol_ratio vs 次スロットret)',
                 fontweight='bold', fontsize=10)
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            if not np.isnan(heat[i,j]):
                ax.text(j, i, f"{heat[i,j]:+.2f}", ha='center', va='center',
                        fontsize=7, color='black' if abs(heat[i,j])<0.15 else 'white')
    fig.colorbar(im, ax=ax, shrink=0.7)

    # (1,2) A2: Q5-Q1 差分 (15:30残時間の視点)
    ax = fig.add_subplot(gs[1,2])
    if not a2_df.empty:
        colors = ['#d62728' if v<0 else '#2ca02c' for v in a2_df['Q5-Q1']]
        ax.barh(range(len(a2_df)), a2_df['Q5-Q1'], color=colors)
        ax.set_yticks(range(len(a2_df))); ax.set_yticklabels(a2_df['slot'], fontsize=8)
        ax.axvline(0, color='k', lw=0.5)
        for i, v in enumerate(a2_df['Q5-Q1']):
            ax.text(v, i, f" {v:+.1f}", va='center', fontsize=8)
        ax.invert_yaxis()
    ax.set_title('[A2] Q5-Q1 差分 (残時間)\n(正: 高vol_ratio優位)', fontweight='bold', fontsize=10)
    ax.set_xlabel('Q5_mean - Q1_mean (bps)')

    # (2,0-3) B1: 評価時点ごとの4象限 bar
    for idx, eval_label in enumerate(['10:30','11:30','13:30','14:30']):
        r = 2 + idx//2; c = idx % 2
        ax = fig.add_subplot(gs[r, c])
        if eval_label not in b_results: continue
        rdf = b_results[eval_label]
        color_map = {'P+D+':'#2ca02c','P+D-':'#d62728','P-D+':'#1f77b4','P-D-':'#ff7f0e'}
        ax.bar(rdf['quad'], rdf['mean_bps'],
               color=[color_map.get(q,'gray') for q in rdf['quad']])
        ax.axhline(0, color='k', lw=0.5)
        for i, (m, n) in enumerate(zip(rdf['mean_bps'], rdf['N'])):
            ax.text(i, m, f"{m:+.1f}\nN={n}", ha='center',
                    va='bottom' if m>0 else 'top', fontsize=8)
        ax.set_title(f'[B1] {eval_label} 時点: 価格×デルタ 4象限\n→ 15:30残ret',
                     fontweight='bold', fontsize=10)
        ax.set_ylabel('mean bps'); ax.grid(alpha=0.3)

    # (3,2) B2: diverg vs fwd_bps の Spearman 推移
    ax = fig.add_subplot(gs[3,2])
    if not b2_df.empty:
        colors = ['#d62728' if r<0 else '#2ca02c' for r in b2_df['spearman_r']]
        ax.bar(range(len(b2_df)), b2_df['spearman_r'], color=colors)
        ax.set_xticks(range(len(b2_df))); ax.set_xticklabels(b2_df['eval'])
        ax.axhline(0, color='k', lw=0.5)
        for i, (r, p, q1q5) in enumerate(zip(b2_df['spearman_r'], b2_df['p'], b2_df['Q1-Q5'])):
            sig = '*' if p<0.05 else ''
            ax.text(i, r, f"{r:+.3f}{sig}\nQ1-Q5:{q1q5:+.1f}", ha='center',
                    va='bottom' if r>0 else 'top', fontsize=8)
    ax.set_title('[B2] divergence rank vs 残ret\nSpearman推移', fontweight='bold', fontsize=10)
    ax.set_ylabel('Spearman r (* = p<0.05)')
    ax.set_xlabel('評価時点'); ax.grid(alpha=0.3)

    plt.suptitle('日中出来高推移 → 方向性予測 (A: 時刻帯パネル + B: 擬似累積デルタ)',
                 fontsize=14, fontweight='bold', y=0.995)
    out = os.path.join(OUT_DIR, 'result.png')
    plt.savefig(out, dpi=130, bbox_inches='tight')
    print(f"Saved: {out}")

    # CSV
    panel_next.to_csv(os.path.join(OUT_DIR, 'A_slot_next.csv'), index=False)
    panel_tc.to_csv(os.path.join(OUT_DIR, 'A_slot_to_close.csv'), index=False)
    a1_df.to_csv(os.path.join(OUT_DIR, 'A1_slot_spearman_next.csv'), index=False)
    a2_df.to_csv(os.path.join(OUT_DIR, 'A2_slot_spearman_toclose.csv'), index=False)
    b_all.to_csv(os.path.join(OUT_DIR, 'B_cum_delta_eval.csv'), index=False)
    if not b2_df.empty:
        b2_df.to_csv(os.path.join(OUT_DIR, 'B2_divergence_spearman.csv'), index=False)
    print("CSV saved")

    print("\n" + "=" * 70)
    print(" 完了")
    print("=" * 70)


if __name__ == "__main__":
    main()
