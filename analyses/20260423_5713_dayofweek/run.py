"""
住友金属鉱山 (5713.T) 曜日別イントラデイ傾向分析

検証項目:
  1. 曜日 × セッション (ON/日中/全日) の期待値・t値・勝率 (全期間・レジーム別)
  2. 曜日 × 前場(9:00寄→11:30前引)/後場(12:30後寄→15:30引)/引け前30分(14:55→15:25)
  3. 14:45時点の損益方向 → 大引け方向の反転率 (曜日別)
  4. 月次変動チェック (曜日パターンが時期で安定か)

レジーム:
  Pre  : 〜2026-02-27
  War  : 2026-02-28 〜 2026-03-13
  Post : 2026-03-16 〜

出力: stdout 表, result.png, csv
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '20260421_common')))
import mdutil as U
import pandas as pd
import numpy as np
from datetime import date

plt = U.matplotlib_jp()

DOW_NAMES = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
SYM = "5713.T"
NAME = "住友金属鉱山"
OUT = os.path.dirname(__file__)

WAR_START = date(2026, 2, 28)
WAR_END = date(2026, 3, 13)
POST_START = date(2026, 3, 16)


def regime_of(d):
    if d < WAR_START: return 'Pre'
    if d <= WAR_END: return 'War'
    if d >= POST_START: return 'Post'
    return 'Gap'


def pick_bar(grp, h, m, col='close', tol=3):
    """最も近いバーを1本取る (分単位ズレ許容)"""
    bar_min = grp.index.hour * 60 + grp.index.minute
    target = h * 60 + m
    diff = np.abs(bar_min - target)
    idx = diff.argmin()
    if diff[idx] > tol: return np.nan
    return float(grp.iloc[idx][col])


def build_daily(intra):
    """イントラデイから 日次の各セッション価格を構築"""
    rows = []
    prev_close = np.nan
    for d in sorted(set(intra.index.date)):
        g = intra[intra.index.date == d].sort_index()
        if len(g) < 10: continue
        p_open = pick_bar(g, 9, 0, 'open')
        p_am_close = pick_bar(g, 11, 30, 'close', tol=5)
        p_pm_open = pick_bar(g, 12, 30, 'open', tol=5)
        p_1445 = pick_bar(g, 14, 45, 'close', tol=5)
        p_1455 = pick_bar(g, 14, 55, 'close', tol=5)
        p_1525 = pick_bar(g, 15, 25, 'close', tol=5)
        p_close = pick_bar(g, 15, 30, 'close', tol=5)
        if pd.isna(p_close):
            # 15:30ぴったりが無い場合、15:25以降のラスト close
            tail = g[(g.index.hour == 15) & (g.index.minute >= 20)]
            p_close = float(tail['close'].iloc[-1]) if len(tail) else np.nan
        if any(pd.isna(x) for x in [p_open, p_close]):
            continue
        rows.append({
            'date': d,
            'dow': pd.Timestamp(d).weekday(),
            'regime': regime_of(d),
            'prev_close': prev_close,
            'open': p_open,
            'am_close': p_am_close,
            'pm_open': p_pm_open,
            'p1445': p_1445,
            'p1455': p_1455,
            'p1525': p_1525,
            'close': p_close,
        })
        prev_close = p_close
    df = pd.DataFrame(rows).set_index('date')
    df['on_ret'] = (df['open'] / df['prev_close'] - 1) * 10000
    df['day_ret'] = (df['close'] / df['open'] - 1) * 10000
    df['full_ret'] = (df['close'] / df['prev_close'] - 1) * 10000
    df['am_ret'] = (df['am_close'] / df['open'] - 1) * 10000
    df['lunch_gap'] = (df['pm_open'] / df['am_close'] - 1) * 10000
    df['pm_ret'] = (df['close'] / df['pm_open'] - 1) * 10000
    df['last30_ret'] = (df['close'] / df['p1445'] - 1) * 10000
    df['last5_ret'] = (df['close'] / df['p1525'] - 1) * 10000
    df['ret_to_1445'] = (df['p1445'] / df['open'] - 1) * 10000
    return df


def agg_bps(df, col, groupby='dow'):
    """平均 (bps), N, t値, 勝率 を返す"""
    g = df.groupby(groupby)[col].agg(['mean', 'count', 'std'])
    g['t'] = g['mean'] / (g['std'] / np.sqrt(g['count']))
    g['wr'] = df.groupby(groupby)[col].apply(lambda x: (x > 0).mean() * 100)
    return g


def print_table(df, col, title, regime=None):
    sub = df if regime is None else df[df['regime'] == regime]
    sub = sub.dropna(subset=[col])
    if len(sub) == 0:
        print(f"\n--- {title} ---  (N=0)")
        return
    g = agg_bps(sub, col)
    print(f"\n--- {title}  (regime={regime or 'All'}, N_total={len(sub)}) ---")
    print(f"{'曜日':>4}  {'bps':>8}  {'N':>3}  {'t':>6}  {'勝率':>5}")
    for dow in range(5):
        if dow not in g.index:
            print(f"  {DOW_NAMES[dow]}:  {'--':>8}  {'0':>3}")
            continue
        r = g.loc[dow]
        marker = ''
        if abs(r['t']) >= 2: marker = ' ★'
        if abs(r['t']) >= 2.5: marker = ' ★★'
        if abs(r['t']) >= 3: marker = ' ★★★'
        print(f"  {DOW_NAMES[dow]}:  {r['mean']:+8.1f}  {int(r['count']):>3}  {r['t']:+6.2f}  {r['wr']:5.1f}%{marker}")


def save_summary_csv(df, path):
    rows = []
    for col, label in [('on_ret', 'ON'), ('day_ret', '日中'),
                       ('full_ret', '全日'), ('am_ret', '前場'),
                       ('pm_ret', '後場'), ('last30_ret', '引け前30分'),
                       ('last5_ret', 'ラスト5分')]:
        for regime in ['All', 'Pre', 'Post']:
            sub = df if regime == 'All' else df[df['regime'] == regime]
            sub = sub.dropna(subset=[col])
            if len(sub) == 0: continue
            g = agg_bps(sub, col)
            for dow in range(5):
                if dow not in g.index: continue
                r = g.loc[dow]
                rows.append({
                    'regime': regime, 'session': label, 'dow': DOW_NAMES[dow],
                    'n': int(r['count']), 'bps': round(r['mean'], 2),
                    't': round(r['t'], 3), 'wr': round(r['wr'], 1),
                })
    pd.DataFrame(rows).to_csv(path, index=False)


def main():
    print("=" * 70)
    print(f"住友金属鉱山 ({SYM}) 曜日別イントラデイ傾向分析")
    print("=" * 70)

    print("\n[1/3] イントラデイデータ取得中...")
    intra = U.fetch_intraday(SYM)
    print(f"  intraday rows: {len(intra):,}")

    print("\n[2/3] 日次セッション価格構築中...")
    daily = build_daily(intra)
    print(f"  daily rows: {len(daily)}")
    reg_count = daily['regime'].value_counts()
    print(f"  regime: Pre={reg_count.get('Pre',0)}  War={reg_count.get('War',0)}  "
          f"Post={reg_count.get('Post',0)}")
    print(f"  期間: {daily.index.min()} 〜 {daily.index.max()}")

    print("\n[3/3] 集計...")

    # ------------------------------------------------------------------
    # 全期間 (All)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("■ 全期間 曜日別セッションリターン (bps)")
    print("=" * 70)
    for col, label in [('on_ret', 'ON (前日引→当日寄)'),
                       ('day_ret', '日中 (寄→引)'),
                       ('full_ret', '全日 (前日引→当日引)'),
                       ('am_ret', '前場 (9:00寄→11:30前引)'),
                       ('pm_ret', '後場 (12:30後寄→15:30引)'),
                       ('last30_ret', '大引け前30分 (14:45→15:30)'),
                       ('last5_ret', 'ラスト5分 (15:25→15:30)')]:
        print_table(daily, col, label)

    # ------------------------------------------------------------------
    # Pre / Post 分割
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("■ Pre vs Post レジーム別 (2026-02-28 イラン戦争前後)")
    print("=" * 70)
    for col, label in [('on_ret', 'ON'), ('day_ret', '日中'),
                       ('full_ret', '全日'), ('am_ret', '前場'),
                       ('pm_ret', '後場'), ('last30_ret', '引け前30分')]:
        print_table(daily, col, label + ' [Pre]', regime='Pre')
        print_table(daily, col, label + ' [Post]', regime='Post')

    # ------------------------------------------------------------------
    # 14:45損益方向 → 大引け方向 反転率
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("■ 14:45までの損益 → 大引け前30分方向 反転率 (曜日別, 全期間)")
    print("=" * 70)
    sub = daily.dropna(subset=['ret_to_1445', 'last30_ret'])
    sub = sub[sub['ret_to_1445'].abs() >= 20]  # 寄→14:45が±20bps未満はノイズ除外
    print(f"N_total (|寄→14:45|>=20bps): {len(sub)}")
    print(f"{'曜日':>4}  {'N':>3}  {'一致率':>6}  {'反転率':>6}  {'14:45→引 bps':>12}")
    for dow in range(5):
        s = sub[sub['dow'] == dow]
        if len(s) == 0:
            print(f"  {DOW_NAMES[dow]}:  {'0':>3}  N/A")
            continue
        match = (np.sign(s['ret_to_1445']) == np.sign(s['last30_ret'])).mean() * 100
        mean_bp = s['last30_ret'].mean()
        print(f"  {DOW_NAMES[dow]}:  {len(s):>3}  {match:5.1f}%  {100-match:5.1f}%  {mean_bp:+12.1f}")

    # ------------------------------------------------------------------
    # 月次変動
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("■ 月次 曜日パターン (日中 bps) — 時期安定性チェック")
    print("=" * 70)
    daily2 = daily.copy()
    daily2.index = pd.to_datetime(daily2.index)
    daily2['ym'] = daily2.index.to_period('M')
    piv = daily2.pivot_table(index='ym', columns='dow', values='day_ret',
                             aggfunc='mean')
    piv.columns = [DOW_NAMES[c] for c in piv.columns]
    print(piv.round(1).to_string())

    # ------------------------------------------------------------------
    # CSV 保存
    # ------------------------------------------------------------------
    save_summary_csv(daily, os.path.join(OUT, 'summary.csv'))
    daily.to_csv(os.path.join(OUT, 'daily.csv'))
    print(f"\nSaved: summary.csv, daily.csv")

    # ------------------------------------------------------------------
    # 可視化
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35)

    # (0,0) 全期間セッション別 bps bar (ON/日中/全日)
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(5); w = 0.28
    for i, (col, label, color) in enumerate([
            ('on_ret', 'ON', '#4472c4'),
            ('day_ret', '日中', '#ed7d31'),
            ('full_ret', '全日', '#70ad47')]):
        g = agg_bps(daily.dropna(subset=[col]), col)
        vals = [g.loc[d, 'mean'] if d in g.index else 0 for d in range(5)]
        ax.bar(x + (i - 1) * w, vals, w, label=label, color=color, edgecolor='black')
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(DOW_NAMES)
    ax.set_ylabel('bps')
    ax.set_title('全期間 曜日×セッション リターン', fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3, axis='y')

    # (0,1) Pre vs Post 日中
    ax = fig.add_subplot(gs[0, 1])
    for i, (regime, color) in enumerate([('Pre', '#6baed6'), ('Post', '#fd8d3c')]):
        sub = daily[daily['regime'] == regime].dropna(subset=['day_ret'])
        if len(sub) == 0: continue
        g = agg_bps(sub, 'day_ret')
        vals = [g.loc[d, 'mean'] if d in g.index else 0 for d in range(5)]
        ax.bar(x + (i - 0.5) * 0.4, vals, 0.4, label=regime, color=color, edgecolor='black')
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(DOW_NAMES)
    ax.set_ylabel('bps')
    ax.set_title('日中 Pre vs Post', fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3, axis='y')

    # (0,2) Pre vs Post ON
    ax = fig.add_subplot(gs[0, 2])
    for i, (regime, color) in enumerate([('Pre', '#6baed6'), ('Post', '#fd8d3c')]):
        sub = daily[daily['regime'] == regime].dropna(subset=['on_ret'])
        if len(sub) == 0: continue
        g = agg_bps(sub, 'on_ret')
        vals = [g.loc[d, 'mean'] if d in g.index else 0 for d in range(5)]
        ax.bar(x + (i - 0.5) * 0.4, vals, 0.4, label=regime, color=color, edgecolor='black')
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(DOW_NAMES)
    ax.set_ylabel('bps')
    ax.set_title('ON Pre vs Post', fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3, axis='y')

    # (1,0) ヒートマップ: セッション × 曜日 (全期間 bps)
    ax = fig.add_subplot(gs[1, 0])
    session_cols = [('on_ret', 'ON'), ('am_ret', '前場'), ('pm_ret', '後場'),
                    ('last30_ret', '引前30'), ('full_ret', '全日')]
    M = np.full((len(session_cols), 5), np.nan)
    for i, (col, _) in enumerate(session_cols):
        g = agg_bps(daily.dropna(subset=[col]), col)
        for d in range(5):
            if d in g.index:
                M[i, d] = g.loc[d, 'mean']
    im = ax.imshow(M, aspect='auto', cmap='RdYlGn', vmin=-60, vmax=60)
    ax.set_xticks(range(5)); ax.set_xticklabels(DOW_NAMES)
    ax.set_yticks(range(len(session_cols)))
    ax.set_yticklabels([lbl for _, lbl in session_cols])
    ax.set_title('全期間 セッション×曜日 bps', fontweight='bold')
    for i in range(len(session_cols)):
        for j in range(5):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i,j]:+.0f}", ha='center', va='center', fontsize=9)
    plt.colorbar(im, ax=ax)

    # (1,1) t値ヒートマップ
    ax = fig.add_subplot(gs[1, 1])
    M = np.full((len(session_cols), 5), np.nan)
    for i, (col, _) in enumerate(session_cols):
        g = agg_bps(daily.dropna(subset=[col]), col)
        for d in range(5):
            if d in g.index:
                M[i, d] = g.loc[d, 't']
    im = ax.imshow(M, aspect='auto', cmap='RdYlGn', vmin=-3, vmax=3)
    ax.set_xticks(range(5)); ax.set_xticklabels(DOW_NAMES)
    ax.set_yticks(range(len(session_cols)))
    ax.set_yticklabels([lbl for _, lbl in session_cols])
    ax.set_title('全期間 セッション×曜日 t値', fontweight='bold')
    for i in range(len(session_cols)):
        for j in range(5):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i,j]:+.1f}", ha='center', va='center', fontsize=9)
    plt.colorbar(im, ax=ax)

    # (1,2) Post期間 t値ヒートマップ
    ax = fig.add_subplot(gs[1, 2])
    sub = daily[daily['regime'] == 'Post']
    M = np.full((len(session_cols), 5), np.nan)
    for i, (col, _) in enumerate(session_cols):
        s = sub.dropna(subset=[col])
        if len(s) == 0: continue
        g = agg_bps(s, col)
        for d in range(5):
            if d in g.index:
                M[i, d] = g.loc[d, 't']
    im = ax.imshow(M, aspect='auto', cmap='RdYlGn', vmin=-3, vmax=3)
    ax.set_xticks(range(5)); ax.set_xticklabels(DOW_NAMES)
    ax.set_yticks(range(len(session_cols)))
    ax.set_yticklabels([lbl for _, lbl in session_cols])
    ax.set_title('Post期間 t値', fontweight='bold')
    for i in range(len(session_cols)):
        for j in range(5):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i,j]:+.1f}", ha='center', va='center', fontsize=9)
    plt.colorbar(im, ax=ax)

    # (2,0) 累積PnL: 曜日別 (全期間 日中)
    ax = fig.add_subplot(gs[2, 0])
    daily3 = daily.copy()
    daily3.index = pd.to_datetime(daily3.index)
    colors = plt.cm.tab10.colors
    for dow in range(5):
        s = daily3[daily3['dow'] == dow].dropna(subset=['day_ret']).sort_index()
        if len(s) == 0: continue
        cum = s['day_ret'].cumsum()
        ax.plot(cum.index, cum, lw=1.5, label=f"{DOW_NAMES[dow]} ({cum.iloc[-1]:+.0f})",
                color=colors[dow])
    ax.axhline(0, color='black', lw=0.8)
    ax.set_ylabel('累積 bps'); ax.set_title('曜日別 累積日中PnL', fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # (2,1) 曜日別 累積全日PnL
    ax = fig.add_subplot(gs[2, 1])
    for dow in range(5):
        s = daily3[daily3['dow'] == dow].dropna(subset=['full_ret']).sort_index()
        if len(s) == 0: continue
        cum = s['full_ret'].cumsum()
        ax.plot(cum.index, cum, lw=1.5, label=f"{DOW_NAMES[dow]} ({cum.iloc[-1]:+.0f})",
                color=colors[dow])
    ax.axhline(0, color='black', lw=0.8)
    ax.set_ylabel('累積 bps'); ax.set_title('曜日別 累積全日PnL', fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # (2,2) 月次ヒートマップ (日中 bps)
    ax = fig.add_subplot(gs[2, 2])
    daily3['ym'] = daily3.index.to_period('M').astype(str)
    piv = daily3.pivot_table(index='ym', columns='dow', values='day_ret',
                             aggfunc='mean')
    piv = piv.reindex(columns=range(5))
    im = ax.imshow(piv.values, aspect='auto', cmap='RdYlGn',
                   vmin=-150, vmax=150, interpolation='nearest')
    ax.set_xticks(range(5)); ax.set_xticklabels(DOW_NAMES)
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index, fontsize=8)
    ax.set_title('月次 日中bps 推移', fontweight='bold')
    plt.colorbar(im, ax=ax)

    plt.suptitle(f'{SYM} {NAME}: 曜日別イントラデイ傾向分析',
                 fontsize=15, fontweight='bold', y=1.00)
    out_png = os.path.join(OUT, 'result.png')
    plt.savefig(out_png, dpi=130, bbox_inches='tight')
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
