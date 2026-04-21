"""
日本株 曜日別特性: 2026年2月末イラン戦争 前後レジーム比較

背景:
  2026年2月末に勃発したイラン戦争を境にしたレジーム転換を検証。
  前回の2025-06イスラエル-イラン12日戦争は別イベントとして扱い、
  今回はより直近の開戦ショックとその後の展開を見る。

期間定義:
  Pre : 2025-04-17 〜 2026-02-27 (約220営業日)
  War : 2026-02-28 〜 2026-03-13 (2週間を想定, 除外)
  Post: 2026-03-16 〜 2026-04-20 (約25-30営業日, 標本小)

注意: Post期間の標本が小さいため、t値と併せてNを必ず確認。
      曜日あたり5-6サンプルしかないので過学習リスク極大。
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


def load_topix_daily():
    df = U.fetch_intraday('.TOPX').dropna(subset=['open','close'])
    h, m = df.index.hour, df.index.minute
    df = df[((h==9)&(m<=5)) | ((h==15)&(m>=0)&(m<=30))]
    out = []
    for d in sorted(set(df.index.date)):
        if d.weekday() >= 5: continue
        g = df[df.index.date == d]
        h2, m2 = g.index.hour, g.index.minute
        op = g[(h2==9)&(m2<=5)]; cl = g[(h2==15)]
        if len(op)==0 or len(cl)==0: continue
        out.append({'date': d, 'open': op['open'].iloc[0], 'close': cl['close'].iloc[-1]})
    ddf = pd.DataFrame(out).set_index('date').sort_index()
    ddf['prev_close'] = ddf['close'].shift(1)
    ddf['on_ret'] = (ddf['open']/ddf['prev_close']-1)*100
    ddf['day_ret'] = (ddf['close']/ddf['open']-1)*100
    ddf['full_ret'] = (ddf['close']/ddf['prev_close']-1)*100
    ddf['dow'] = [d.weekday() for d in ddf.index]
    ddf['regime'] = [regime_of(d) for d in ddf.index]
    return ddf


def dow_table(ddf, col):
    rows = []
    for rg in ['Pre', 'War', 'Post']:
        sub = ddf[ddf['regime']==rg]
        for dow in range(5):
            x = sub[sub['dow']==dow][col].dropna().values
            if len(x) == 0:
                rows.append({'regime':rg, 'dow':DOW_NAMES[dow], 'n':0, 'mean':np.nan, 't':np.nan, 'wr':np.nan})
                continue
            m, s = x.mean(), x.std()
            rows.append({
                'regime': rg, 'dow': DOW_NAMES[dow], 'n': len(x),
                'mean': m*100,
                't': m/(s/np.sqrt(len(x))) if s>0 else 0,
                'wr': (x>0).mean()*100,
            })
    return pd.DataFrame(rows)


def main():
    print("=== TOPIX 曜日別レジーム分析 (2026年2月末イラン戦争) ===")
    top = load_topix_daily()
    print(f"TOPIX days: {len(top)}")
    print(f"  Pre  (〜2026-02-27): {sum(top.regime=='Pre')}日")
    print(f"  War  (02-28〜03-13): {sum(top.regime=='War')}日")
    print(f"  Post (03-16〜)     : {sum(top.regime=='Post')}日")
    print(f"Date range: {top.index[0]} → {top.index[-1]}")

    # War期間のショック確認
    war = top[top['regime']=='War']
    if len(war) > 0:
        print(f"\n--- War期間の日次リターン (TOPIX) ---")
        for _, r in war.iterrows():
            print(f"  {r.name} ({DOW_NAMES[int(r.dow)]}): ON={r.on_ret*100:+6.1f} Day={r.day_ret*100:+6.1f} Full={r.full_ret*100:+6.1f}bps")
        print(f"  War累積: {war['full_ret'].sum()*100:+.0f}bps")

    tbl_full = dow_table(top, 'full_ret')
    tbl_day = dow_table(top, 'day_ret')
    tbl_on = dow_table(top, 'on_ret')

    for label, tbl in [('全日(close→close)', tbl_full), ('日中(open→close)', tbl_day), ('ON(prev_close→open)', tbl_on)]:
        print(f"\n--- TOPIX {label} bps ---")
        pv = tbl.pivot(index='dow', columns='regime', values='mean').reindex(DOW_NAMES)
        nv = tbl.pivot(index='dow', columns='regime', values='n').reindex(DOW_NAMES)
        # Pre/Post に限定して表示、Nも併記
        for dow in DOW_NAMES:
            pre_m = pv.loc[dow, 'Pre'] if 'Pre' in pv.columns else np.nan
            pre_n = int(nv.loc[dow, 'Pre']) if 'Pre' in nv.columns else 0
            post_m = pv.loc[dow, 'Post'] if 'Post' in pv.columns else np.nan
            post_n = int(nv.loc[dow, 'Post']) if 'Post' in nv.columns else 0
            delta = post_m - pre_m if not (pd.isna(pre_m) or pd.isna(post_m)) else np.nan
            print(f"  {dow}: Pre {pre_m:+6.1f}(N={pre_n:3d})  Post {post_m:+7.1f}(N={post_n:2d})  Δ={delta:+6.1f}")

    # セクター別 Pre/Post 日中
    print("\n=== セクター×曜日 (Post期間 日中, bps) ===")
    print("※Post期間は標本小、参考程度")
    sectors = {
        'コア5': U.CORE5, '半導体': U.SEMICON, '海運': U.SHIPPING,
        '非鉄': U.NONFERROUS, 'エネルギー': U.ENERGY,
    }
    sector_post = {}
    sector_pre = {}
    for name, syms in sectors.items():
        allret = []
        for sym, _ in syms:
            try:
                jp = U.load_jp_daily(sym)
            except Exception: continue
            jp['day_ret'] = (jp['close']/jp['open']-1)*100
            jp['dow'] = [d.weekday() for d in jp.index]
            jp['regime'] = [regime_of(d) for d in jp.index]
            allret.append(jp[['day_ret','dow','regime']])
        if not allret: continue
        cat = pd.concat(allret)
        post = cat[cat['regime']=='Post']
        pre = cat[cat['regime']=='Pre']
        by_post = post.groupby('dow')['day_ret'].agg(['mean','count','std'])
        by_pre = pre.groupby('dow')['day_ret'].agg(['mean','count','std'])
        by_post['bps'] = by_post['mean']*100
        by_post['t'] = by_post['mean']/(by_post['std']/np.sqrt(by_post['count']))
        by_pre['bps'] = by_pre['mean']*100
        by_pre['t'] = by_pre['mean']/(by_pre['std']/np.sqrt(by_pre['count']))
        sector_post[name] = by_post
        sector_pre[name] = by_pre
        print(f"\n[{name}]")
        for dow in range(5):
            ppre = by_pre.loc[dow] if dow in by_pre.index else None
            ppost = by_post.loc[dow] if dow in by_post.index else None
            pre_str = f"{ppre['bps']:+6.1f}(N={int(ppre['count']):3d})" if ppre is not None else 'N/A'
            post_str = f"{ppost['bps']:+7.1f}(N={int(ppost['count']):3d},t={ppost['t']:+.2f})" if ppost is not None else 'N/A'
            print(f"  {DOW_NAMES[dow]}: Pre {pre_str}  Post {post_str}")

    # 可視化
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))

    # (0,0) TOPIX full Pre/Post
    ax = axes[0,0]
    pv_full = tbl_full.pivot(index='dow', columns='regime', values='mean').reindex(DOW_NAMES)
    cols = [c for c in ['Pre','Post'] if c in pv_full.columns]
    M = pv_full[cols].values
    im = ax.imshow(M, aspect='auto', cmap='RdYlGn', vmin=-100, vmax=100)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols)
    ax.set_yticks(range(5)); ax.set_yticklabels(DOW_NAMES)
    ax.set_title('TOPIX 全日(close→close) bps', fontweight='bold')
    for i in range(5):
        for j in range(len(cols)):
            if not np.isnan(M[i,j]):
                ax.text(j, i, f"{M[i,j]:+.1f}", ha='center', va='center', fontsize=9)
    plt.colorbar(im, ax=ax)

    # (0,1) TOPIX day Pre/Post
    ax = axes[0,1]
    pv_day = tbl_day.pivot(index='dow', columns='regime', values='mean').reindex(DOW_NAMES)
    M = pv_day[cols].values
    im = ax.imshow(M, aspect='auto', cmap='RdYlGn', vmin=-100, vmax=100)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols)
    ax.set_yticks(range(5)); ax.set_yticklabels(DOW_NAMES)
    ax.set_title('TOPIX 日中(open→close) bps', fontweight='bold')
    for i in range(5):
        for j in range(len(cols)):
            if not np.isnan(M[i,j]):
                ax.text(j, i, f"{M[i,j]:+.1f}", ha='center', va='center', fontsize=9)
    plt.colorbar(im, ax=ax)

    # (0,2) ON
    ax = axes[0,2]
    pv_on = tbl_on.pivot(index='dow', columns='regime', values='mean').reindex(DOW_NAMES)
    M = pv_on[cols].values
    im = ax.imshow(M, aspect='auto', cmap='RdYlGn', vmin=-100, vmax=100)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols)
    ax.set_yticks(range(5)); ax.set_yticklabels(DOW_NAMES)
    ax.set_title('TOPIX ON(prev_close→open) bps', fontweight='bold')
    for i in range(5):
        for j in range(len(cols)):
            if not np.isnan(M[i,j]):
                ax.text(j, i, f"{M[i,j]:+.1f}", ha='center', va='center', fontsize=9)
    plt.colorbar(im, ax=ax)

    # (1,0) Pre/Post 日中 曜日別bar
    ax = axes[1,0]
    x = np.arange(5); w = 0.35
    pre_vals = [pv_day.loc[d,'Pre'] if 'Pre' in pv_day.columns and not pd.isna(pv_day.loc[d,'Pre']) else 0 for d in DOW_NAMES]
    post_vals = [pv_day.loc[d,'Post'] if 'Post' in pv_day.columns and not pd.isna(pv_day.loc[d,'Post']) else 0 for d in DOW_NAMES]
    ax.bar(x-w/2, pre_vals, w, label=f'Pre (~2/27, N~{sum(top.regime=="Pre")})', color='#6baed6', edgecolor='black')
    ax.bar(x+w/2, post_vals, w, label=f'Post (3/16~, N={sum(top.regime=="Post")})', color='#fd8d3c', edgecolor='black')
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(DOW_NAMES)
    ax.set_ylabel('Mean bps/day')
    ax.set_title('TOPIX日中リターン: Pre vs Post', fontweight='bold')
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')

    # (1,1) セクター×曜日 Post期間
    ax = axes[1,1]
    sector_names = list(sector_post.keys())
    M2 = np.full((len(sector_names), 5), np.nan)
    for i, sn in enumerate(sector_names):
        for dow in range(5):
            if dow in sector_post[sn].index:
                M2[i, dow] = sector_post[sn].loc[dow, 'bps']
    im = ax.imshow(M2, aspect='auto', cmap='RdYlGn', vmin=-150, vmax=150)
    ax.set_xticks(range(5)); ax.set_xticklabels(DOW_NAMES)
    ax.set_yticks(range(len(sector_names))); ax.set_yticklabels(sector_names)
    ax.set_title(f'セクター×曜日 Post (日中bps)\n※N小・参考程度', fontweight='bold')
    for i in range(len(sector_names)):
        for j in range(5):
            if not np.isnan(M2[i,j]):
                ax.text(j, i, f"{M2[i,j]:+.0f}", ha='center', va='center', fontsize=9)
    plt.colorbar(im, ax=ax)

    # (1,2) TOPIX War期間の日次バー
    ax = axes[1,2]
    if len(war) > 0:
        ax.bar(range(len(war)), war['full_ret']*100,
               color=['#d62728' if v<0 else '#2ca02c' for v in war['full_ret']*100],
               edgecolor='black')
        ax.set_xticks(range(len(war)))
        ax.set_xticklabels([f"{d.month}/{d.day}\n{DOW_NAMES[d.weekday()]}" for d in war.index], fontsize=8)
        ax.axhline(0, color='black', lw=0.8)
        ax.set_ylabel('bps')
        ax.set_title(f'War期間 日次TOPIX全日リターン (累積{war.full_ret.sum()*100:+.0f}bps)', fontweight='bold')
        ax.grid(alpha=0.3, axis='y')

    plt.suptitle('日本株 曜日別特性: 2026年2月末イラン戦争 前後レジーム比較',
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), 'result.png')
    plt.savefig(out, dpi=130, bbox_inches='tight')
    print(f"\nSaved: {out}")

    tbl_day.to_csv(os.path.join(os.path.dirname(__file__), 'dow_day_ret.csv'), index=False)
    tbl_full.to_csv(os.path.join(os.path.dirname(__file__), 'dow_full_ret.csv'), index=False)
    tbl_on.to_csv(os.path.join(os.path.dirname(__file__), 'dow_on_ret.csv'), index=False)
    print("CSVs saved.")


if __name__ == "__main__":
    main()
