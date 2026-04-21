"""
非鉄金属3銘柄 イントラデイ典型パターン分析

銘柄: 5711 三菱マテリアル / 5706 三井金属 / 5713 住友金属鉱山

検証項目:
  1. 時間帯別リターンプロファイル (9:00基準の累積ドリフト)
  2. 時間帯別ボラティリティ (1分足標準偏差)
  3. 時間帯別出来高プロファイル (U字型?)
  4. 寄付30分の方向性 → 残り時間の関係 (モメンタム or リバーサル)
  5. 前場引け → 後場寄りのギャップ (昼休みジャンプ)
  6. 引け30分の挙動 (クロージングオークション効果)
  7. 寄付ギャップフィル率 (前日引→当日寄のギャップが埋まるか)
  8. 当日レンジ形成時刻 (高値・安値がいつ付くか)
  9. トレンド日 vs レンジ日の分類
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '20260421_common')))
import mdutil as U
import pandas as pd
import numpy as np
from datetime import time as dtime
plt = U.matplotlib_jp()
import matplotlib.dates as mdates

SYMS = [('5711.T', '三菱マテリアル'),
        ('5706.T', '三井金属'),
        ('5713.T', '住友金属鉱山')]


def load_1min(sym):
    df = U.fetch_intraday(sym).dropna(subset=['open','high','low','close'])
    # 取引時間のみ残す
    h, m = df.index.hour, df.index.minute
    mask = (((h==9)) | ((h==10)) | ((h==11)&(m<=30)) |
            ((h==12)&(m>=30)) | ((h==13)|(h==14)) | ((h==15)&(m<=30)))
    return df[mask].copy()


def minute_of_day(idx):
    """9:00=0 として分オフセットを返す (昼休み 11:30-12:30=60分は詰めない)"""
    h, m = idx.hour, idx.minute
    return (h - 9) * 60 + m


def time_label(mo):
    h = 9 + mo // 60
    m = mo % 60
    return f"{h:02d}:{m:02d}"


def main():
    print("=== 非鉄金属3銘柄 イントラデイ典型パターン分析 ===\n")
    dfs = {}
    for sym, name in SYMS:
        df = load_1min(sym)
        df['ret_1m_bps'] = df['close'].pct_change() * 10000  # bps
        df['mo'] = minute_of_day(df.index)
        # 日別寄付 (9:00 or 最初) 価格
        dfs[sym] = df
        print(f"{sym} {name}: {len(df):>7} rows, days={len(set(df.index.date))}")

    # ========== 1. 時間帯別リターンプロファイル (9:00基準累積) ==========
    print("\n=== 1. 9:00基準 累積ドリフト (各日の寄り価格比, bps) ===")
    profile_cum = {}
    for sym, name in SYMS:
        df = dfs[sym]
        by_min = []
        for d in sorted(set(df.index.date)):
            day = df[df.index.date == d]
            if len(day) < 200: continue
            open_p = day['open'].iloc[0]
            day = day.copy()
            day['cum_bps'] = (day['close']/open_p - 1) * 10000
            by_min.append(day[['mo','cum_bps']])
        if not by_min: continue
        allb = pd.concat(by_min)
        prof = allb.groupby('mo')['cum_bps'].agg(['mean','std','count'])
        profile_cum[sym] = prof
        # ピーク/底を出す
        peak = prof['mean'].idxmax()
        trough = prof['mean'].idxmin()
        print(f"  {sym} {name}: peak {time_label(peak)} {prof.loc[peak,'mean']:+.1f}bps, "
              f"trough {time_label(trough)} {prof.loc[trough,'mean']:+.1f}bps, "
              f"EOD {prof['mean'].iloc[-1]:+.1f}bps")

    # ========== 2. 時間帯別 1分ボラティリティ ==========
    print("\n=== 2. 時間帯別 1分足ボラティリティ (bps std) ===")
    vol_by_min = {}
    for sym, name in SYMS:
        df = dfs[sym].dropna(subset=['ret_1m_bps'])
        # 株価分割等の外れ値除去: |ret|>300bps=3% 除外
        df = df[df['ret_1m_bps'].abs() < 300]
        vol = df.groupby('mo')['ret_1m_bps'].std()
        vol_by_min[sym] = vol
        # 各時間帯の代表値
        key_times = [0, 15, 30, 60, 120, 150, 210, 270, 330, 390]  # 9:00, 9:15, 9:30, 10:00, 11:00, 11:30, 12:30, 13:30, 14:30, 15:30
        print(f"  {sym}:", end=' ')
        for kt in key_times:
            if kt in vol.index:
                print(f"{time_label(kt)}={vol.loc[kt]:.1f}", end=' ')
        print()

    # ========== 3. 時間帯別 出来高プロファイル ==========
    print("\n=== 3. 時間帯別 出来高プロファイル (各日の1日合計に対する比率, %) ===")
    volume_profile = {}
    for sym, name in SYMS:
        df = dfs[sym]
        rows = []
        for d in sorted(set(df.index.date)):
            day = df[df.index.date == d].copy()
            total = day['volume'].sum()
            if total <= 0: continue
            day['vol_pct'] = day['volume'] / total * 100
            rows.append(day[['mo','vol_pct']])
        if not rows: continue
        allv = pd.concat(rows)
        vp = allv.groupby('mo')['vol_pct'].mean()
        volume_profile[sym] = vp
        # ピーク時間帯
        top3 = vp.nlargest(3)
        print(f"  {sym}: top3 times = " + ", ".join([f"{time_label(i)}({v:.2f}%)" for i,v in top3.items()]))

    # ========== 4. 寄付30分 → 残り時間 ==========
    print("\n=== 4. 寄付30分リターン → 残り時間リターン の関係 ===")
    for sym, name in SYMS:
        df = dfs[sym]
        rows = []
        for d in sorted(set(df.index.date)):
            day = df[df.index.date == d]
            if len(day) < 200: continue
            # 9:00〜9:30 (mo<=30)
            first30 = day[day['mo']<=30]
            rest = day[day['mo']>30]
            if len(first30)<10 or len(rest)<50: continue
            o = first30['open'].iloc[0]
            m30_close = first30['close'].iloc[-1]
            eod = rest['close'].iloc[-1]
            r_first = (m30_close/o - 1)*10000
            r_rest = (eod/m30_close - 1)*10000
            rows.append({'date': d, 'r_first30': r_first, 'r_rest': r_rest})
        if not rows: continue
        rdf = pd.DataFrame(rows).set_index('date')
        # 相関
        c = rdf['r_first30'].corr(rdf['r_rest'])
        # 寄付30分が±X bps超のサブセット別
        up = rdf[rdf['r_first30'] > 50]
        down = rdf[rdf['r_first30'] < -50]
        print(f"  {sym} {name}: corr={c:+.3f}")
        print(f"    first30 > +50bps (N={len(up):>3}): rest mean={up['r_rest'].mean():+.1f}bps, WR={(up['r_rest']>0).mean()*100:.1f}%")
        print(f"    first30 < -50bps (N={len(down):>3}): rest mean={down['r_rest'].mean():+.1f}bps, WR={(down['r_rest']>0).mean()*100:.1f}%")

    # ========== 5. 昼休みジャンプ ==========
    print("\n=== 5. 前場引け → 後場寄り (昼休みジャンプ, bps) ===")
    lunch = {}
    for sym, name in SYMS:
        df = dfs[sym]
        rows = []
        for d in sorted(set(df.index.date)):
            day = df[df.index.date == d]
            am_close = day[day['mo']==150]['close']   # 11:30
            pm_open = day[day['mo']==210]['open']     # 12:30
            if len(am_close)==0 or len(pm_open)==0:
                # 最も近い時刻で近似
                am = day[day['mo']<=150].tail(1)
                pm = day[day['mo']>=210].head(1)
                if len(am)==0 or len(pm)==0: continue
                am_c = am['close'].iloc[0]; pm_o = pm['open'].iloc[0]
            else:
                am_c = am_close.iloc[0]; pm_o = pm_open.iloc[0]
            rows.append({'date': d, 'jump_bps': (pm_o/am_c-1)*10000})
        ldf = pd.DataFrame(rows).set_index('date')
        ldf = ldf[ldf['jump_bps'].abs() < 500]  # 外れ値除去
        lunch[sym] = ldf
        print(f"  {sym}: mean={ldf['jump_bps'].mean():+.2f}bps, median={ldf['jump_bps'].median():+.2f}bps, "
              f"std={ldf['jump_bps'].std():.2f}, |jump|>20bps={((ldf['jump_bps'].abs()>20).mean()*100):.1f}%")

    # ========== 6. 引け30分の挙動 ==========
    print("\n=== 6. 引け30分 (15:00-15:30) リターン ===")
    for sym, name in SYMS:
        df = dfs[sym]
        rows = []
        for d in sorted(set(df.index.date)):
            day = df[df.index.date == d]
            at_1500 = day[day['mo']>=360].head(1)  # 15:00
            at_close = day[day['mo']>=390].tail(1) if len(day[day['mo']>=390])>0 else day.tail(1)
            if len(at_1500)==0: continue
            open_p = day['open'].iloc[0]
            r_late = (at_close['close'].iloc[0]/at_1500['close'].iloc[0]-1)*10000
            r_day = (at_1500['close'].iloc[0]/open_p-1)*10000
            rows.append({'date': d, 'r_morning_to_1500': r_day, 'r_close30min': r_late})
        edf = pd.DataFrame(rows).set_index('date')
        edf = edf[edf['r_close30min'].abs()<300]
        # 引け30分の方向性が午前と一致するか
        trend_up = edf[edf['r_morning_to_1500']>0]
        trend_down = edf[edf['r_morning_to_1500']<0]
        print(f"  {sym}: close30 mean={edf['r_close30min'].mean():+.1f}bps, std={edf['r_close30min'].std():.1f}")
        print(f"    朝〜15:00 up (N={len(trend_up):>3}): close30 mean={trend_up['r_close30min'].mean():+.1f}, 継続率={(trend_up['r_close30min']>0).mean()*100:.1f}%")
        print(f"    朝〜15:00 dn (N={len(trend_down):>3}): close30 mean={trend_down['r_close30min'].mean():+.1f}, 継続率={(trend_down['r_close30min']<0).mean()*100:.1f}%")

    # ========== 7. ギャップフィル率 ==========
    print("\n=== 7. 寄付ギャップフィル率 (前日引→当日寄のギャップが日中に埋まるか) ===")
    for sym, name in SYMS:
        # 日次ベースで
        jp = U.load_jp_daily(sym)
        jp['prev_close'] = jp['close'].shift(1)
        jp['gap_pct'] = (jp['open']/jp['prev_close']-1)*100
        jp = jp.dropna(subset=['gap_pct'])
        # 低値/高値で埋まったか
        df = dfs[sym]
        rows = []
        for d, row in jp.iterrows():
            day = df[df.index.date == d]
            if len(day) < 200: continue
            dl = day['low'].min(); dh = day['high'].max()
            gap = row['gap_pct']
            if gap > 0.3:  # up gap
                # 日中安値が prev_close 以下なら埋まった
                filled = dl <= row['prev_close']
                rows.append({'gap_pct': gap, 'direction': 'up', 'filled': filled})
            elif gap < -0.3:
                filled = dh >= row['prev_close']
                rows.append({'gap_pct': gap, 'direction': 'down', 'filled': filled})
        if not rows: continue
        gdf = pd.DataFrame(rows)
        up = gdf[gdf['direction']=='up']
        dn = gdf[gdf['direction']=='down']
        print(f"  {sym}: up-gap N={len(up)} fill={up['filled'].mean()*100:.1f}%, "
              f"down-gap N={len(dn)} fill={dn['filled'].mean()*100:.1f}%")

    # ========== 8. レンジ形成時刻 (高値・安値がいつ付くか) ==========
    print("\n=== 8. 日中高値・安値の到達時刻分布 ===")
    for sym, name in SYMS:
        df = dfs[sym]
        hi_times = []; lo_times = []
        for d in sorted(set(df.index.date)):
            day = df[df.index.date == d]
            if len(day) < 200: continue
            hi_times.append(day['mo'].iloc[day['high'].argmax()])
            lo_times.append(day['mo'].iloc[day['low'].argmin()])
        hi_times = np.array(hi_times); lo_times = np.array(lo_times)
        # 時間帯を [0-30, 30-60, 60-150, 150-210(昼), 210-300, 300-390, 390+] に分類
        def bucket(arr):
            bins = [0, 30, 60, 150, 210, 300, 390, 500]
            labels = ['9:00-9:30','9:30-10:00','10:00-11:30','11:30-12:30','12:30-14:00','14:00-15:00','15:00-15:30']
            counts = []
            for i in range(len(bins)-1):
                counts.append(((arr>=bins[i])&(arr<bins[i+1])).sum())
            return dict(zip(labels, counts))
        hbk = bucket(hi_times); lbk = bucket(lo_times)
        print(f"  {sym} 高値時刻分布:", {k:f"{v/len(hi_times)*100:.0f}%" for k,v in hbk.items() if v>0})
        print(f"  {sym} 安値時刻分布:", {k:f"{v/len(lo_times)*100:.0f}%" for k,v in lbk.items() if v>0})

    # ========== 可視化 ==========
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(4, 3, hspace=0.5, wspace=0.3)
    colors = ['#d62728', '#1f77b4', '#2ca02c']

    # (0,0-2) 累積ドリフト 3銘柄
    for i, (sym, name) in enumerate(SYMS):
        ax = fig.add_subplot(gs[0,i])
        if sym in profile_cum:
            p = profile_cum[sym]
            mo = p.index.values
            mean = p['mean'].values
            sd_err = p['std'].values / np.sqrt(p['count'].values)
            ax.plot(mo, mean, color=colors[i], lw=1.5)
            ax.fill_between(mo, mean-sd_err, mean+sd_err, color=colors[i], alpha=0.2)
            ax.axhline(0, color='gray', lw=0.8)
            # 昼休み塗り
            ax.axvspan(150, 210, color='gray', alpha=0.15)
        ax.set_title(f"{sym} {name}\n累積ドリフト (9:00基準, bps)", fontweight='bold')
        xticks = [0, 60, 150, 210, 300, 390]
        ax.set_xticks(xticks)
        ax.set_xticklabels([time_label(t) for t in xticks], rotation=30, fontsize=8)
        ax.grid(alpha=0.3)

    # (1,0) ボラティリティ 3銘柄重ね
    ax = fig.add_subplot(gs[1,0])
    for i, (sym, _) in enumerate(SYMS):
        if sym in vol_by_min:
            v = vol_by_min[sym]
            ax.plot(v.index, v.values, color=colors[i], lw=1.2, label=sym, alpha=0.8)
    ax.axvspan(150, 210, color='gray', alpha=0.15)
    ax.set_title('時間帯別 1分ボラ (std bps)', fontweight='bold')
    xticks = [0, 60, 150, 210, 300, 390]
    ax.set_xticks(xticks); ax.set_xticklabels([time_label(t) for t in xticks], rotation=30, fontsize=8)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (1,1) 出来高プロファイル
    ax = fig.add_subplot(gs[1,1])
    for i, (sym, _) in enumerate(SYMS):
        if sym in volume_profile:
            v = volume_profile[sym]
            ax.plot(v.index, v.values, color=colors[i], lw=1.2, label=sym, alpha=0.8)
    ax.axvspan(150, 210, color='gray', alpha=0.15)
    ax.set_title('時間帯別 出来高比率 (%)', fontweight='bold')
    ax.set_xticks(xticks); ax.set_xticklabels([time_label(t) for t in xticks], rotation=30, fontsize=8)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (1,2) 昼休みジャンプ分布
    ax = fig.add_subplot(gs[1,2])
    for i, (sym, _) in enumerate(SYMS):
        if sym in lunch:
            j = lunch[sym]['jump_bps'].values
            ax.hist(j, bins=30, alpha=0.5, color=colors[i], label=f"{sym} (μ={j.mean():+.1f})")
    ax.axvline(0, color='black', lw=0.8)
    ax.set_title('昼休みジャンプ分布', fontweight='bold')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (2,0-2) 寄付30分 vs 残り 散布図
    for i, (sym, name) in enumerate(SYMS):
        ax = fig.add_subplot(gs[2,i])
        df = dfs[sym]
        rows = []
        for d in sorted(set(df.index.date)):
            day = df[df.index.date == d]
            if len(day) < 200: continue
            first30 = day[day['mo']<=30]
            rest = day[day['mo']>30]
            if len(first30)<10 or len(rest)<50: continue
            o = first30['open'].iloc[0]
            m30 = first30['close'].iloc[-1]
            eod = rest['close'].iloc[-1]
            rows.append({'r_first30': (m30/o-1)*10000, 'r_rest': (eod/m30-1)*10000})
        r = pd.DataFrame(rows)
        r = r[(r['r_first30'].abs()<300) & (r['r_rest'].abs()<300)]
        ax.scatter(r['r_first30'], r['r_rest'], alpha=0.5, s=15, color=colors[i])
        ax.axhline(0, color='gray', lw=0.5); ax.axvline(0, color='gray', lw=0.5)
        # 回帰
        if len(r) > 5:
            b, a = np.polyfit(r['r_first30'], r['r_rest'], 1)
            xs = np.linspace(r['r_first30'].min(), r['r_first30'].max(), 50)
            ax.plot(xs, a+b*xs, color='black', lw=1)
            c = r['r_first30'].corr(r['r_rest'])
            ax.text(0.05, 0.95, f"slope={b:+.2f}\ncorr={c:+.3f}\nN={len(r)}",
                    transform=ax.transAxes, va='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.set_xlabel('寄付30分 bps'); ax.set_ylabel('残り時間 bps')
        ax.set_title(f'{sym} 寄付30分 → 残り時間', fontweight='bold')
        ax.grid(alpha=0.3)

    # (3,0-2) 高値・安値時刻ヒストグラム (3銘柄重ね)
    ax = fig.add_subplot(gs[3,0])
    for i, (sym, _) in enumerate(SYMS):
        df = dfs[sym]
        hi_times = []
        for d in sorted(set(df.index.date)):
            day = df[df.index.date == d]
            if len(day) < 200: continue
            hi_times.append(day['mo'].iloc[day['high'].argmax()])
        ax.hist(hi_times, bins=25, alpha=0.5, color=colors[i], label=sym)
    ax.axvspan(150, 210, color='gray', alpha=0.2)
    ax.set_title('日中高値到達時刻', fontweight='bold')
    xticks = [0, 60, 150, 210, 300, 390]
    ax.set_xticks(xticks); ax.set_xticklabels([time_label(t) for t in xticks], rotation=30, fontsize=8)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[3,1])
    for i, (sym, _) in enumerate(SYMS):
        df = dfs[sym]
        lo_times = []
        for d in sorted(set(df.index.date)):
            day = df[df.index.date == d]
            if len(day) < 200: continue
            lo_times.append(day['mo'].iloc[day['low'].argmin()])
        ax.hist(lo_times, bins=25, alpha=0.5, color=colors[i], label=sym)
    ax.axvspan(150, 210, color='gray', alpha=0.2)
    ax.set_title('日中安値到達時刻', fontweight='bold')
    ax.set_xticks(xticks); ax.set_xticklabels([time_label(t) for t in xticks], rotation=30, fontsize=8)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (3,2) 寄付30分 方向性 → EOD方向 (全3銘柄集約)
    ax = fig.add_subplot(gs[3,2])
    summary = []
    for sym, name in SYMS:
        df = dfs[sym]
        u_rest = []; d_rest = []
        for d in sorted(set(df.index.date)):
            day = df[df.index.date == d]
            if len(day) < 200: continue
            first30 = day[day['mo']<=30]
            rest = day[day['mo']>30]
            if len(first30)<10 or len(rest)<50: continue
            o = first30['open'].iloc[0]
            m30 = first30['close'].iloc[-1]
            eod = rest['close'].iloc[-1]
            r_first = (m30/o-1)*10000
            r_rest = (eod/m30-1)*10000
            if r_first > 50: u_rest.append(r_rest)
            elif r_first < -50: d_rest.append(r_rest)
        summary.append((sym, np.mean(u_rest) if u_rest else 0, np.mean(d_rest) if d_rest else 0))
    xs = np.arange(len(summary))
    w = 0.35
    ax.bar(xs-w/2, [s[1] for s in summary], w, color='#2ca02c', label='first30>+50 の残り平均', edgecolor='black')
    ax.bar(xs+w/2, [s[2] for s in summary], w, color='#d62728', label='first30<-50 の残り平均', edgecolor='black')
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xticks(xs); ax.set_xticklabels([s[0] for s in summary])
    ax.set_ylabel('残り時間 平均bps')
    ax.set_title('寄付30分方向 → 残り時間の追随', fontweight='bold')
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')

    plt.suptitle('非鉄金属3銘柄 イントラデイ典型パターン分析',
                 fontsize=15, fontweight='bold', y=1.00)
    out = os.path.join(os.path.dirname(__file__), 'result.png')
    plt.savefig(out, dpi=130, bbox_inches='tight')
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
