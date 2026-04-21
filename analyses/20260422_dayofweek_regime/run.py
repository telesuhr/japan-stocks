"""
日本株の曜日別特性分析 (レジーム比較: イラン戦争前/後)

背景:
  2025年6月13日にイスラエルがイランを攻撃、6月24日に停戦 (12日戦争)。
  原油急騰→急落、リスクオフ→リスクオンの大きなレジーム転換が発生。
  この前後で日本株の曜日別特性がどう変化したかを検証。

検証対象:
  1. TOPIX (.TOPX) / 日経先物 (JNIc1) の曜日別リターン
  2. セッション分解: ON (前日close→当日open), 日中 (9→15:30)
  3. セクター別曜日性 (コア5/半導体/海運)
  4. レジーム: Pre (〜2025-06-12), War (06-13〜06-24), Post (2025-07-01〜)

出力: result.png, regime_table.csv
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '20260421_common')))
import mdutil as U
import pandas as pd
import numpy as np
from datetime import date, time as dtime
plt = U.matplotlib_jp()

DOW_NAMES = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
WAR_START = date(2025, 6, 13)
WAR_END = date(2025, 6, 24)
POST_START = date(2025, 7, 1)


def regime_of(d):
    if d < WAR_START: return 'Pre'
    if d <= WAR_END: return 'War'
    if d >= POST_START: return 'Post'
    return 'Gap'


def load_topix_daily():
    """TOPIX: ON(前日close→当日open) と intraday(open→close) を作る"""
    df = U.fetch_intraday('.TOPX').dropna(subset=['open','close'])
    # 9:00寄りと15:00〜15:30引け
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
    """曜日×レジーム別統計"""
    rows = []
    for rg in ['Pre', 'Post', 'All']:
        sub = ddf if rg == 'All' else ddf[ddf['regime']==rg]
        for dow in range(5):
            x = sub[sub['dow']==dow][col].dropna().values
            if len(x) == 0:
                rows.append({'regime':rg, 'dow':DOW_NAMES[dow], 'n':0, 'mean':np.nan, 't':np.nan, 'wr':np.nan})
                continue
            m, s = x.mean(), x.std()
            rows.append({
                'regime': rg, 'dow': DOW_NAMES[dow], 'n': len(x),
                'mean': m*100,  # bps
                't': m/(s/np.sqrt(len(x))) if s>0 else 0,
                'wr': (x>0).mean()*100,
            })
    return pd.DataFrame(rows)


def main():
    print("=== TOPIX 曜日別レジーム分析 ===")
    top = load_topix_daily()
    print(f"TOPIX days: {len(top)}, Pre={sum(top.regime=='Pre')}, War={sum(top.regime=='War')}, Post={sum(top.regime=='Post')}")
    print(f"Date range: {top.index[0]} → {top.index[-1]}")

    # 1. 全日リターンの曜日効果 (Pre/Post)
    tbl_full = dow_table(top, 'full_ret')
    tbl_day = dow_table(top, 'day_ret')
    tbl_on = dow_table(top, 'on_ret')

    print("\n--- TOPIX 全日リターン (close→close) bps ---")
    print(tbl_full.pivot(index='dow', columns='regime', values='mean').reindex(DOW_NAMES).round(1))
    print("\n--- TOPIX 日中リターン (open→close) bps ---")
    print(tbl_day.pivot(index='dow', columns='regime', values='mean').reindex(DOW_NAMES).round(1))
    print("\n--- TOPIX ONリターン (prev_close→open) bps ---")
    print(tbl_on.pivot(index='dow', columns='regime', values='mean').reindex(DOW_NAMES).round(1))

    # 2. セクター曜日効果 (Pre/Post)
    print("\n=== セクター曜日効果 (day_ret, Post期間のみ) ===")
    sectors = {
        'コア5': U.CORE5, '半導体': U.SEMICON, '海運': U.SHIPPING,
        '非鉄': U.NONFERROUS, 'エネルギー': U.ENERGY,
    }
    sector_post = {}
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
        cat = cat[cat['regime']=='Post']
        by_dow = cat.groupby('dow')['day_ret'].agg(['mean','count','std'])
        by_dow['bps'] = by_dow['mean']*100
        by_dow['t'] = by_dow['mean']/(by_dow['std']/np.sqrt(by_dow['count']))
        sector_post[name] = by_dow
        print(f"\n[{name}] Post期間 日中リターン bps (t-stat)")
        for dow in range(5):
            if dow in by_dow.index:
                r = by_dow.loc[dow]
                print(f"  {DOW_NAMES[dow]}: {r['bps']:+6.1f}bps (N={int(r['count'])}, t={r['t']:+.2f})")

    # 3. Pre vs Post 差分の顕著なものを抽出
    print("\n=== Pre vs Post: TOPIX 日中リターン差 (bps) ===")
    pv = tbl_day.pivot(index='dow', columns='regime', values='mean').reindex(DOW_NAMES)
    pv['Δ(Post-Pre)'] = pv['Post'] - pv['Pre']
    print(pv.round(1))

    # --- 可視化 ---
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    cmap_rb = plt.get_cmap('RdYlGn')

    # (0,0) TOPIX full_ret heatmap Pre/Post
    ax = axes[0,0]
    M = tbl_full.pivot(index='dow', columns='regime', values='mean').reindex(DOW_NAMES)[['Pre','Post']].values
    im = ax.imshow(M, aspect='auto', cmap='RdYlGn', vmin=-50, vmax=50)
    ax.set_xticks([0,1]); ax.set_xticklabels(['Pre','Post'])
    ax.set_yticks(range(5)); ax.set_yticklabels(DOW_NAMES)
    ax.set_title('TOPIX 全日リターン (bps) - レジーム別', fontweight='bold')
    for i in range(5):
        for j in range(2):
            if not np.isnan(M[i,j]):
                ax.text(j, i, f"{M[i,j]:+.1f}", ha='center', va='center', fontsize=10)
    plt.colorbar(im, ax=ax)

    # (0,1) TOPIX day_ret
    ax = axes[0,1]
    M = tbl_day.pivot(index='dow', columns='regime', values='mean').reindex(DOW_NAMES)[['Pre','Post']].values
    im = ax.imshow(M, aspect='auto', cmap='RdYlGn', vmin=-50, vmax=50)
    ax.set_xticks([0,1]); ax.set_xticklabels(['Pre','Post'])
    ax.set_yticks(range(5)); ax.set_yticklabels(DOW_NAMES)
    ax.set_title('TOPIX 日中 (9→15:30) bps', fontweight='bold')
    for i in range(5):
        for j in range(2):
            if not np.isnan(M[i,j]):
                ax.text(j, i, f"{M[i,j]:+.1f}", ha='center', va='center', fontsize=10)
    plt.colorbar(im, ax=ax)

    # (0,2) TOPIX on_ret
    ax = axes[0,2]
    M = tbl_on.pivot(index='dow', columns='regime', values='mean').reindex(DOW_NAMES)[['Pre','Post']].values
    im = ax.imshow(M, aspect='auto', cmap='RdYlGn', vmin=-50, vmax=50)
    ax.set_xticks([0,1]); ax.set_xticklabels(['Pre','Post'])
    ax.set_yticks(range(5)); ax.set_yticklabels(DOW_NAMES)
    ax.set_title('TOPIX ON (prev_close→open) bps', fontweight='bold')
    for i in range(5):
        for j in range(2):
            if not np.isnan(M[i,j]):
                ax.text(j, i, f"{M[i,j]:+.1f}", ha='center', va='center', fontsize=10)
    plt.colorbar(im, ax=ax)

    # (1,0) Pre vs Post 曜日別bar (日中)
    ax = axes[1,0]
    x = np.arange(5); w = 0.35
    pre_vals = [pv.loc[d,'Pre'] if not pd.isna(pv.loc[d,'Pre']) else 0 for d in DOW_NAMES]
    post_vals = [pv.loc[d,'Post'] if not pd.isna(pv.loc[d,'Post']) else 0 for d in DOW_NAMES]
    ax.bar(x-w/2, pre_vals, w, label='Pre (~6/12)', color='#6baed6', edgecolor='black')
    ax.bar(x+w/2, post_vals, w, label='Post (7/1~)', color='#fd8d3c', edgecolor='black')
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(DOW_NAMES)
    ax.set_ylabel('Mean bps/day')
    ax.set_title('TOPIX日中リターン: Pre vs Post', fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3, axis='y')

    # (1,1) セクター×曜日 Post期間ヒートマップ
    ax = axes[1,1]
    sector_names = list(sector_post.keys())
    M2 = np.full((len(sector_names), 5), np.nan)
    for i, sn in enumerate(sector_names):
        for dow in range(5):
            if dow in sector_post[sn].index:
                M2[i, dow] = sector_post[sn].loc[dow, 'bps']
    im = ax.imshow(M2, aspect='auto', cmap='RdYlGn', vmin=-80, vmax=80)
    ax.set_xticks(range(5)); ax.set_xticklabels(DOW_NAMES)
    ax.set_yticks(range(len(sector_names))); ax.set_yticklabels(sector_names)
    ax.set_title('セクター×曜日 (Post期間, 日中bps)', fontweight='bold')
    for i in range(len(sector_names)):
        for j in range(5):
            if not np.isnan(M2[i,j]):
                ax.text(j, i, f"{M2[i,j]:+.0f}", ha='center', va='center', fontsize=9)
    plt.colorbar(im, ax=ax)

    # (1,2) TOPIX累積PnL(曜日別, Post期間)
    ax = axes[1,2]
    post = top[top['regime']=='Post'].copy()
    for dow in range(5):
        sub = post[post['dow']==dow].copy()
        sub['cum'] = (sub['day_ret']*100).cumsum()
        if len(sub)>0:
            ax.plot(pd.to_datetime(sub.index), sub['cum'], lw=1.5, marker='o', ms=3,
                    label=f"{DOW_NAMES[dow]} N={len(sub)} end={sub['cum'].iloc[-1]:+.0f}")
    ax.axhline(0, color='gray', lw=0.8)
    ax.set_title('TOPIX Post期間 曜日別累積日中bps', fontweight='bold')
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)

    plt.suptitle('日本株 曜日別特性: イラン戦争前後レジーム比較',
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), 'result.png')
    plt.savefig(out, dpi=130, bbox_inches='tight')
    print(f"\nSaved: {out}")

    # CSV 出力
    tbl_day.to_csv(os.path.join(os.path.dirname(__file__), 'dow_day_ret.csv'), index=False)
    tbl_full.to_csv(os.path.join(os.path.dirname(__file__), 'dow_full_ret.csv'), index=False)
    print("CSVs saved.")


if __name__ == "__main__":
    main()
