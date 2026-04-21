"""
非鉄金属・半導体に特化した曜日別特性分析
レジーム: 2026年2月末イラン戦争 前後

非鉄 (NONFERROUS): 5711 三菱マテリアル / 5706 三井金属 / 5713 住友金属鉱山
半導体 (SEMICON) : 8035 TEL / 6857 アドバンテスト / 6146 ディスコ / 4063 信越化学 / 6963 ローム

検証項目:
  1. バスケット等加重: 曜日×セッション(ON/日中/全日) Pre vs Post
  2. 個別銘柄: 曜日×日中 Pre vs Post
  3. War期間(10日) の個別ドローダウン
  4. 両セクターの対比 (非鉄と半導体でパターンが違うか)
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '20260421_common')))
import mdutil as U
import pandas as pd
import numpy as np
from datetime import date
plt = U.matplotlib_jp()

DOW_NAMES = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
WAR_START = date(2026, 2, 28)
WAR_END = date(2026, 3, 13)
POST_START = date(2026, 3, 16)


def regime_of(d):
    if d < WAR_START: return 'Pre'
    if d <= WAR_END: return 'War'
    if d >= POST_START: return 'Post'
    return 'Gap'


def load_sym(sym):
    jp = U.load_jp_daily(sym)
    jp['prev_close'] = jp['close'].shift(1)
    jp['on_ret'] = (jp['open']/jp['prev_close']-1)*100
    jp['day_ret'] = (jp['close']/jp['open']-1)*100
    jp['full_ret'] = (jp['close']/jp['prev_close']-1)*100
    jp['dow'] = [d.weekday() for d in jp.index]
    jp['regime'] = [regime_of(d) for d in jp.index]
    return jp


def by_dow_regime(cat, col, regime):
    sub = cat[cat['regime']==regime]
    g = sub.groupby('dow')[col].agg(['mean','count','std'])
    g['bps'] = g['mean']*100
    g['t'] = g['mean']/(g['std']/np.sqrt(g['count']))
    return g


def main():
    NONFERROUS = U.NONFERROUS
    SEMICON = U.SEMICON

    sym_data = {}
    for sym, name in NONFERROUS + SEMICON:
        try:
            jp = load_sym(sym)
            sym_data[sym] = (name, jp)
            print(f"  {sym} {name}: N={len(jp)} (Pre={sum(jp.regime=='Pre')}/War={sum(jp.regime=='War')}/Post={sum(jp.regime=='Post')})")
        except Exception as e:
            print(f"  {sym} ERROR: {e}")

    # ========== バスケット等加重 ==========
    print("\n" + "="*60)
    print("【バスケット等加重】")
    print("="*60)

    for basket_name, syms in [('非鉄', NONFERROUS), ('半導体', SEMICON)]:
        allret = []
        for sym, _ in syms:
            if sym in sym_data:
                _, jp = sym_data[sym]
                allret.append(jp[['on_ret','day_ret','full_ret','dow','regime']].copy())
        cat = pd.concat(allret)

        print(f"\n### {basket_name}バスケット ({len(syms)}銘柄)")
        for col, label in [('on_ret','ON'), ('day_ret','日中'), ('full_ret','全日')]:
            g_pre = by_dow_regime(cat, col, 'Pre')
            g_post = by_dow_regime(cat, col, 'Post')
            print(f"\n--- {label} bps ---")
            print(f"{'曜日':>4} {'Pre':>17} {'Post':>22} {'Δ':>7}")
            for dow in range(5):
                pre_b = g_pre.loc[dow,'bps'] if dow in g_pre.index else np.nan
                pre_n = int(g_pre.loc[dow,'count']) if dow in g_pre.index else 0
                post_b = g_post.loc[dow,'bps'] if dow in g_post.index else np.nan
                post_n = int(g_post.loc[dow,'count']) if dow in g_post.index else 0
                post_t = g_post.loc[dow,'t'] if dow in g_post.index else np.nan
                delta = post_b - pre_b if not (pd.isna(pre_b) or pd.isna(post_b)) else np.nan
                print(f"  {DOW_NAMES[dow]}: {pre_b:+7.1f}(N={pre_n:3d})  {post_b:+7.1f}(N={post_n:3d},t={post_t:+.2f})  {delta:+7.1f}")

    # ========== 個別銘柄 日中 Pre/Post ==========
    print("\n" + "="*60)
    print("【個別銘柄 日中リターン Pre vs Post】")
    print("="*60)

    individual_post = {}  # sym -> {dow: (bps, n, t)}
    individual_pre = {}

    for sym, (name, jp) in sym_data.items():
        pre = jp[jp['regime']=='Pre']
        post = jp[jp['regime']=='Post']
        gp = pre.groupby('dow')['day_ret'].agg(['mean','count','std'])
        gq = post.groupby('dow')['day_ret'].agg(['mean','count','std'])
        gp['bps'] = gp['mean']*100; gp['t'] = gp['mean']/(gp['std']/np.sqrt(gp['count']))
        gq['bps'] = gq['mean']*100; gq['t'] = gq['mean']/(gq['std']/np.sqrt(gq['count']))
        individual_pre[sym] = gp
        individual_post[sym] = gq

        print(f"\n[{sym} {name}]")
        print(f"{'曜日':>4} {'Pre bps(N)':>17} {'Post bps(N,t)':>22}")
        for dow in range(5):
            pre_str = f"{gp.loc[dow,'bps']:+7.1f}(N={int(gp.loc[dow,'count']):3d})" if dow in gp.index else 'N/A'
            if dow in gq.index:
                post_str = f"{gq.loc[dow,'bps']:+7.1f}(N={int(gq.loc[dow,'count']):2d},t={gq.loc[dow,'t']:+.2f})"
            else:
                post_str = 'N/A'
            print(f"  {DOW_NAMES[dow]}: {pre_str}  {post_str}")

    # ========== War期間の個別DD ==========
    print("\n" + "="*60)
    print("【War期間 個別銘柄ドローダウン】")
    print("="*60)
    war_summary = []
    for sym, (name, jp) in sym_data.items():
        war = jp[jp['regime']=='War']
        if len(war)==0: continue
        cum = (war['full_ret']/100).cumsum()
        dd = ((1+cum) / (1+cum).cummax() - 1).min() * 100
        total = war['full_ret'].sum()
        war_summary.append({'sym': sym, 'name': name, 'total_bps': total*100,
                            'maxdd_pct': dd, 'worst_day_bps': war['full_ret'].min()*100,
                            'sector': '非鉄' if (sym,name) in NONFERROUS else '半導体'})
    wdf = pd.DataFrame(war_summary).sort_values('total_bps')
    print(f"\n{'Sym':>6} {'Name':<18} {'Sector':<4} {'War累積bps':>10} {'MaxDD%':>7} {'最悪日bps':>10}")
    for _, r in wdf.iterrows():
        print(f"  {r['sym']:>6} {r['name']:<18} {r['sector']:<4} {r['total_bps']:+10.0f} {r['maxdd_pct']:+7.2f} {r['worst_day_bps']:+10.0f}")

    # ========== 可視化 ==========
    fig = plt.figure(figsize=(18, 13))
    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35)

    # (0,0) 非鉄バスケット 日中 Pre/Post bar
    ax = fig.add_subplot(gs[0,0])
    allret_nf = pd.concat([sym_data[s][1] for s,_ in NONFERROUS if s in sym_data])
    g_pre = by_dow_regime(allret_nf, 'day_ret', 'Pre')
    g_post = by_dow_regime(allret_nf, 'day_ret', 'Post')
    x = np.arange(5); w = 0.35
    pre_v = [g_pre.loc[d,'bps'] if d in g_pre.index else 0 for d in range(5)]
    post_v = [g_post.loc[d,'bps'] if d in g_post.index else 0 for d in range(5)]
    ax.bar(x-w/2, pre_v, w, label='Pre', color='#6baed6', edgecolor='black')
    ax.bar(x+w/2, post_v, w, label='Post', color='#fd8d3c', edgecolor='black')
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(DOW_NAMES)
    ax.set_ylabel('bps/day'); ax.set_title('非鉄 日中リターン Pre vs Post', fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3, axis='y')

    # (0,1) 半導体 日中
    ax = fig.add_subplot(gs[0,1])
    allret_sc = pd.concat([sym_data[s][1] for s,_ in SEMICON if s in sym_data])
    g_pre = by_dow_regime(allret_sc, 'day_ret', 'Pre')
    g_post = by_dow_regime(allret_sc, 'day_ret', 'Post')
    pre_v = [g_pre.loc[d,'bps'] if d in g_pre.index else 0 for d in range(5)]
    post_v = [g_post.loc[d,'bps'] if d in g_post.index else 0 for d in range(5)]
    ax.bar(x-w/2, pre_v, w, label='Pre', color='#6baed6', edgecolor='black')
    ax.bar(x+w/2, post_v, w, label='Post', color='#fd8d3c', edgecolor='black')
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(DOW_NAMES)
    ax.set_ylabel('bps/day'); ax.set_title('半導体 日中リターン Pre vs Post', fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3, axis='y')

    # (0,2) 両バスケット Post比較
    ax = fig.add_subplot(gs[0,2])
    g_nf = by_dow_regime(allret_nf, 'day_ret', 'Post')
    g_sc = by_dow_regime(allret_sc, 'day_ret', 'Post')
    nf_v = [g_nf.loc[d,'bps'] if d in g_nf.index else 0 for d in range(5)]
    sc_v = [g_sc.loc[d,'bps'] if d in g_sc.index else 0 for d in range(5)]
    ax.bar(x-w/2, nf_v, w, label='非鉄', color='#d62728', edgecolor='black')
    ax.bar(x+w/2, sc_v, w, label='半導体', color='#1f77b4', edgecolor='black')
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(DOW_NAMES)
    ax.set_ylabel('bps/day'); ax.set_title('Post期間 セクター対比 日中', fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3, axis='y')

    # (1,0) 非鉄 個別×曜日 ヒートマップ Post
    ax = fig.add_subplot(gs[1,0])
    syms_nf = [s for s,_ in NONFERROUS if s in sym_data]
    M = np.full((len(syms_nf), 5), np.nan)
    for i, s in enumerate(syms_nf):
        gq = individual_post[s]
        for dow in range(5):
            if dow in gq.index:
                M[i, dow] = gq.loc[dow,'bps']
    im = ax.imshow(M, aspect='auto', cmap='RdYlGn', vmin=-150, vmax=150)
    ax.set_xticks(range(5)); ax.set_xticklabels(DOW_NAMES)
    ax.set_yticks(range(len(syms_nf)))
    ax.set_yticklabels([f"{s}\n{sym_data[s][0]}" for s in syms_nf], fontsize=8)
    ax.set_title('非鉄個別×曜日 Post日中 bps', fontweight='bold')
    for i in range(len(syms_nf)):
        for j in range(5):
            if not np.isnan(M[i,j]):
                ax.text(j, i, f"{M[i,j]:+.0f}", ha='center', va='center', fontsize=9)
    plt.colorbar(im, ax=ax)

    # (1,1) 半導体個別
    ax = fig.add_subplot(gs[1,1])
    syms_sc = [s for s,_ in SEMICON if s in sym_data]
    M = np.full((len(syms_sc), 5), np.nan)
    for i, s in enumerate(syms_sc):
        gq = individual_post[s]
        for dow in range(5):
            if dow in gq.index:
                M[i, dow] = gq.loc[dow,'bps']
    im = ax.imshow(M, aspect='auto', cmap='RdYlGn', vmin=-150, vmax=150)
    ax.set_xticks(range(5)); ax.set_xticklabels(DOW_NAMES)
    ax.set_yticks(range(len(syms_sc)))
    ax.set_yticklabels([f"{s}\n{sym_data[s][0]}" for s in syms_sc], fontsize=8)
    ax.set_title('半導体個別×曜日 Post日中 bps', fontweight='bold')
    for i in range(len(syms_sc)):
        for j in range(5):
            if not np.isnan(M[i,j]):
                ax.text(j, i, f"{M[i,j]:+.0f}", ha='center', va='center', fontsize=9)
    plt.colorbar(im, ax=ax)

    # (1,2) Post t値ヒートマップ (全8銘柄)
    ax = fig.add_subplot(gs[1,2])
    all_syms = syms_nf + syms_sc
    M = np.full((len(all_syms), 5), np.nan)
    for i, s in enumerate(all_syms):
        gq = individual_post[s]
        for dow in range(5):
            if dow in gq.index:
                M[i, dow] = gq.loc[dow,'t']
    im = ax.imshow(M, aspect='auto', cmap='RdYlGn', vmin=-3, vmax=3)
    ax.set_xticks(range(5)); ax.set_xticklabels(DOW_NAMES)
    ax.set_yticks(range(len(all_syms)))
    ax.set_yticklabels([f"{s}" for s in all_syms], fontsize=8)
    ax.set_title('個別×曜日 Post t-stat', fontweight='bold')
    for i in range(len(all_syms)):
        for j in range(5):
            if not np.isnan(M[i,j]):
                ax.text(j, i, f"{M[i,j]:+.1f}", ha='center', va='center', fontsize=8)
    plt.colorbar(im, ax=ax)

    # (2,0) War期間 累積PnL 個別
    ax = fig.add_subplot(gs[2,0])
    colors = plt.cm.tab10.colors
    max_war_dates = []
    for i, (sym, (name, jp)) in enumerate(sym_data.items()):
        war = jp[jp['regime']=='War']
        if len(war)==0: continue
        if len(war.index) > len(max_war_dates):
            max_war_dates = list(war.index)
        cum = war['full_ret'].cumsum()
        ls = '-' if sym in [s for s,_ in NONFERROUS] else '--'
        ax.plot(pd.to_datetime(cum.index), cum*100, lw=1.5, ls=ls, marker='o', ms=3,
                label=f"{sym} {name[:6]} ({cum.iloc[-1]*100:+.0f})",
                color=colors[i % 10])
    ax.axhline(0, color='black', lw=0.8)
    ax.set_ylabel('累積 bps'); ax.set_title('War期間 累積PnL (実線=非鉄, 破線=半導体)', fontweight='bold')
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)

    # (2,1) War累積・最悪日 バー
    ax = fig.add_subplot(gs[2,1])
    wdf2 = wdf.copy()
    cols = ['#d62728' if s=='非鉄' else '#1f77b4' for s in wdf2['sector']]
    ax.barh(wdf2['sym']+' '+wdf2['name'].str[:6], wdf2['total_bps'], color=cols, edgecolor='black')
    ax.axvline(0, color='black', lw=0.8)
    ax.set_xlabel('War累積 bps'); ax.set_title('War期間 銘柄別累積 (赤=非鉄, 青=半導体)', fontweight='bold')
    ax.grid(alpha=0.3, axis='x')

    # (2,2) バスケット比較: Post累積日中PnL 曜日別
    ax = fig.add_subplot(gs[2,2])
    cat_nf_post = allret_nf[allret_nf['regime']=='Post']
    cat_sc_post = allret_sc[allret_sc['regime']=='Post']
    # 曜日別に累積
    for basket, data, color in [('非鉄', cat_nf_post, '#d62728'), ('半導体', cat_sc_post, '#1f77b4')]:
        daily = data.groupby([data.index if hasattr(data,'index') else data.date, 'dow'])
        # 日付で平均してから累積
        daily_mean = data.groupby(level=0)['day_ret'].mean()
        cum = daily_mean.cumsum() * 100
        ax.plot(pd.to_datetime(cum.index), cum, lw=1.8, label=f"{basket} ({cum.iloc[-1]:+.0f}bps)", color=color)
    ax.axhline(0, color='black', lw=0.8)
    ax.set_ylabel('累積bps'); ax.set_title('Post期間 バスケット等加重 累積日中PnL', fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3)

    plt.suptitle('非鉄金属 vs 半導体 曜日分析: 2026年2月末イラン戦争 前後',
                 fontsize=15, fontweight='bold', y=1.00)
    out = os.path.join(os.path.dirname(__file__), 'result.png')
    plt.savefig(out, dpi=130, bbox_inches='tight')
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
