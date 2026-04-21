"""
半導体5銘柄 イントラデイ典型パターン分析

銘柄:
  8035 東京エレクトロン (TEL)
  6857 アドバンテスト
  6146 ディスコ
  4063 信越化学
  6963 ローム

検証項目:
  1. 時間帯別リターンプロファイル (9:00基準の累積ドリフト)
  2. 時間帯別ボラティリティ (1分足標準偏差)
  3. 時間帯別出来高プロファイル (U字型?)
  4. 寄付30分の方向性 → 残り時間の関係 (モメンタム or リバーサル)
  5. 前場引け → 後場寄りのギャップ (昼休みジャンプ)
  6. 引け30分の挙動 (クロージングオークション効果)
  7. 寄付ギャップフィル率 (前日引→当日寄のギャップが埋まるか)
  8. 当日レンジ形成時刻 (高値・安値がいつ付くか)
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '20260421_common')))
import mdutil as U
import pandas as pd
import numpy as np
plt = U.matplotlib_jp()

SYMS = [('8035.T', '東京エレクトロン'),
        ('6857.T', 'アドバンテスト'),
        ('6146.T', 'ディスコ'),
        ('4063.T', '信越化学'),
        ('6963.T', 'ローム')]


def load_1min(sym):
    df = U.fetch_intraday(sym).dropna(subset=['open','high','low','close'])
    h, m = df.index.hour, df.index.minute
    mask = (((h==9)) | ((h==10)) | ((h==11)&(m<=30)) |
            ((h==12)&(m>=30)) | ((h==13)|(h==14)) | ((h==15)&(m<=30)))
    return df[mask].copy()


def minute_of_day(idx):
    h, m = idx.hour, idx.minute
    return (h - 9) * 60 + m


def time_label(mo):
    h = 9 + mo // 60
    m = mo % 60
    return f"{h:02d}:{m:02d}"


def main():
    print("=== 半導体5銘柄 イントラデイ典型パターン分析 ===\n")
    dfs = {}
    for sym, name in SYMS:
        df = load_1min(sym)
        df['ret_1m_bps'] = df['close'].pct_change() * 10000
        df['mo'] = minute_of_day(df.index)
        dfs[sym] = df
        print(f"{sym} {name}: {len(df):>7} rows, days={len(set(df.index.date))}")

    # 1. 累積ドリフト
    print("\n=== 1. 9:00基準 累積ドリフト (bps) ===")
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
        peak = prof['mean'].idxmax()
        trough = prof['mean'].idxmin()
        print(f"  {sym} {name}: peak {time_label(peak)} {prof.loc[peak,'mean']:+.1f}bps, "
              f"trough {time_label(trough)} {prof.loc[trough,'mean']:+.1f}bps, "
              f"EOD {prof['mean'].iloc[-1]:+.1f}bps")

    # 2. ボラ
    print("\n=== 2. 時間帯別 1分足ボラ (bps std) ===")
    vol_by_min = {}
    for sym, name in SYMS:
        df = dfs[sym].dropna(subset=['ret_1m_bps'])
        df = df[df['ret_1m_bps'].abs() < 300]
        vol = df.groupby('mo')['ret_1m_bps'].std()
        vol_by_min[sym] = vol
        key_times = [0, 15, 30, 60, 120, 150, 210, 270, 330, 390]
        print(f"  {sym}:", end=' ')
        for kt in key_times:
            if kt in vol.index:
                print(f"{time_label(kt)}={vol.loc[kt]:.1f}", end=' ')
        print()

    # 3. 出来高
    print("\n=== 3. 出来高プロファイル (%) ===")
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
        top3 = vp.nlargest(3)
        print(f"  {sym}: top3 = " + ", ".join([f"{time_label(i)}({v:.2f}%)" for i,v in top3.items()]))

    # 4. 寄付30分 → 残り
    print("\n=== 4. 寄付30分 → 残り時間 ===")
    first30_effect = {}
    for sym, name in SYMS:
        df = dfs[sym]
        rows = []
        for d in sorted(set(df.index.date)):
            day = df[df.index.date == d]
            if len(day) < 200: continue
            first30 = day[day['mo']<=30]
            rest = day[day['mo']>30]
            if len(first30)<10 or len(rest)<50: continue
            o = first30['open'].iloc[0]
            m30_close = first30['close'].iloc[-1]
            eod = rest['close'].iloc[-1]
            rows.append({'r_first30': (m30_close/o-1)*10000, 'r_rest': (eod/m30_close-1)*10000})
        if not rows: continue
        rdf = pd.DataFrame(rows)
        rdf = rdf[(rdf['r_first30'].abs()<300) & (rdf['r_rest'].abs()<300)]
        c = rdf['r_first30'].corr(rdf['r_rest'])
        up = rdf[rdf['r_first30'] > 50]
        down = rdf[rdf['r_first30'] < -50]
        first30_effect[sym] = rdf
        print(f"  {sym} {name}: corr={c:+.3f}")
        print(f"    first30 > +50bps (N={len(up):>3}): rest mean={up['r_rest'].mean():+.1f}bps, WR={(up['r_rest']>0).mean()*100:.1f}%")
        print(f"    first30 < -50bps (N={len(down):>3}): rest mean={down['r_rest'].mean():+.1f}bps, WR={(down['r_rest']>0).mean()*100:.1f}%")

    # 5. 昼休みジャンプ
    print("\n=== 5. 昼休みジャンプ (bps) ===")
    lunch = {}
    for sym, name in SYMS:
        df = dfs[sym]
        rows = []
        for d in sorted(set(df.index.date)):
            day = df[df.index.date == d]
            am_close = day[day['mo']==150]['close']
            pm_open = day[day['mo']==210]['open']
            if len(am_close)==0 or len(pm_open)==0:
                am = day[day['mo']<=150].tail(1)
                pm = day[day['mo']>=210].head(1)
                if len(am)==0 or len(pm)==0: continue
                am_c = am['close'].iloc[0]; pm_o = pm['open'].iloc[0]
            else:
                am_c = am_close.iloc[0]; pm_o = pm_open.iloc[0]
            rows.append({'date': d, 'jump_bps': (pm_o/am_c-1)*10000})
        ldf = pd.DataFrame(rows).set_index('date')
        ldf = ldf[ldf['jump_bps'].abs() < 500]
        lunch[sym] = ldf
        print(f"  {sym}: mean={ldf['jump_bps'].mean():+.2f}bps, median={ldf['jump_bps'].median():+.2f}bps, "
              f"std={ldf['jump_bps'].std():.2f}, |jump|>20bps={((ldf['jump_bps'].abs()>20).mean()*100):.1f}%")

    # 6. 引け30分
    print("\n=== 6. 引け30分 (15:00-15:30) ===")
    for sym, name in SYMS:
        df = dfs[sym]
        rows = []
        for d in sorted(set(df.index.date)):
            day = df[df.index.date == d]
            at_1500 = day[day['mo']>=360].head(1)
            at_close = day[day['mo']>=390].tail(1) if len(day[day['mo']>=390])>0 else day.tail(1)
            if len(at_1500)==0: continue
            open_p = day['open'].iloc[0]
            r_late = (at_close['close'].iloc[0]/at_1500['close'].iloc[0]-1)*10000
            r_day = (at_1500['close'].iloc[0]/open_p-1)*10000
            rows.append({'date': d, 'r_morning_to_1500': r_day, 'r_close30min': r_late})
        edf = pd.DataFrame(rows).set_index('date')
        edf = edf[edf['r_close30min'].abs()<300]
        trend_up = edf[edf['r_morning_to_1500']>0]
        trend_down = edf[edf['r_morning_to_1500']<0]
        print(f"  {sym}: close30 mean={edf['r_close30min'].mean():+.1f}bps, std={edf['r_close30min'].std():.1f}")
        print(f"    up日 (N={len(trend_up):>3}): close30 mean={trend_up['r_close30min'].mean():+.1f}, 継続率={(trend_up['r_close30min']>0).mean()*100:.1f}%")
        print(f"    dn日 (N={len(trend_down):>3}): close30 mean={trend_down['r_close30min'].mean():+.1f}, 継続率={(trend_down['r_close30min']<0).mean()*100:.1f}%")

    # 7. ギャップフィル
    print("\n=== 7. 寄付ギャップフィル率 ===")
    for sym, name in SYMS:
        jp = U.load_jp_daily(sym)
        jp['prev_close'] = jp['close'].shift(1)
        jp['gap_pct'] = (jp['open']/jp['prev_close']-1)*100
        jp = jp.dropna(subset=['gap_pct'])
        df = dfs[sym]
        rows = []
        for d, row in jp.iterrows():
            day = df[df.index.date == d]
            if len(day) < 200: continue
            dl = day['low'].min(); dh = day['high'].max()
            gap = row['gap_pct']
            if gap > 0.3:
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

    # 8. 高値・安値時刻
    print("\n=== 8. 高値・安値時刻分布 ===")
    for sym, name in SYMS:
        df = dfs[sym]
        hi_times = []; lo_times = []
        for d in sorted(set(df.index.date)):
            day = df[df.index.date == d]
            if len(day) < 200: continue
            hi_times.append(day['mo'].iloc[day['high'].argmax()])
            lo_times.append(day['mo'].iloc[day['low'].argmin()])
        hi_times = np.array(hi_times); lo_times = np.array(lo_times)
        def bucket(arr):
            bins = [0, 30, 60, 150, 210, 300, 390, 500]
            labels = ['9:00-9:30','9:30-10:00','10:00-11:30','11:30-12:30','12:30-14:00','14:00-15:00','15:00-15:30']
            counts = []
            for i in range(len(bins)-1):
                counts.append(((arr>=bins[i])&(arr<bins[i+1])).sum())
            return dict(zip(labels, counts))
        hbk = bucket(hi_times); lbk = bucket(lo_times)
        print(f"  {sym} 高値:", {k:f"{v/len(hi_times)*100:.0f}%" for k,v in hbk.items() if v>0})
        print(f"  {sym} 安値:", {k:f"{v/len(lo_times)*100:.0f}%" for k,v in lbk.items() if v>0})

    # === 可視化: 4rows × 5cols ===
    fig = plt.figure(figsize=(22, 15))
    gs = fig.add_gridspec(4, 5, hspace=0.55, wspace=0.35)
    colors = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd']
    xticks = [0, 60, 150, 210, 300, 390]

    # row 0: 累積ドリフト 5銘柄
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
            ax.axvspan(150, 210, color='gray', alpha=0.15)
        ax.set_title(f"{sym} {name}\n累積ドリフト (bps)", fontweight='bold', fontsize=10)
        ax.set_xticks(xticks)
        ax.set_xticklabels([time_label(t) for t in xticks], rotation=30, fontsize=7)
        ax.grid(alpha=0.3)

    # row 1: 集約パネル (vol / volume / lunch hist / corr bar / 朝方向効果)
    ax = fig.add_subplot(gs[1,0])
    for i, (sym, _) in enumerate(SYMS):
        if sym in vol_by_min:
            v = vol_by_min[sym]
            ax.plot(v.index, v.values, color=colors[i], lw=1.2, label=sym, alpha=0.8)
    ax.axvspan(150, 210, color='gray', alpha=0.15)
    ax.set_title('1分ボラ (std bps)', fontweight='bold')
    ax.set_xticks(xticks); ax.set_xticklabels([time_label(t) for t in xticks], rotation=30, fontsize=7)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1,1])
    for i, (sym, _) in enumerate(SYMS):
        if sym in volume_profile:
            v = volume_profile[sym]
            ax.plot(v.index, v.values, color=colors[i], lw=1.2, label=sym, alpha=0.8)
    ax.axvspan(150, 210, color='gray', alpha=0.15)
    ax.set_title('出来高比率 (%)', fontweight='bold')
    ax.set_xticks(xticks); ax.set_xticklabels([time_label(t) for t in xticks], rotation=30, fontsize=7)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1,2])
    for i, (sym, _) in enumerate(SYMS):
        if sym in lunch:
            j = lunch[sym]['jump_bps'].values
            ax.hist(j, bins=30, alpha=0.4, color=colors[i], label=f"{sym} (μ={j.mean():+.1f})")
    ax.axvline(0, color='black', lw=0.8)
    ax.set_title('昼休みジャンプ分布', fontweight='bold')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # first30 effect bar chart
    ax = fig.add_subplot(gs[1,3])
    summary = []
    for sym, name in SYMS:
        if sym not in first30_effect: continue
        r = first30_effect[sym]
        up = r[r['r_first30']>50]
        dn = r[r['r_first30']<-50]
        summary.append((sym, up['r_rest'].mean() if len(up)>0 else 0,
                        dn['r_rest'].mean() if len(dn)>0 else 0))
    xs = np.arange(len(summary)); w = 0.35
    ax.bar(xs-w/2, [s[1] for s in summary], w, color='#2ca02c', label='first30>+50→rest', edgecolor='black')
    ax.bar(xs+w/2, [s[2] for s in summary], w, color='#d62728', label='first30<-50→rest', edgecolor='black')
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xticks(xs); ax.set_xticklabels([s[0] for s in summary], rotation=30, fontsize=8)
    ax.set_ylabel('残り時間 平均bps')
    ax.set_title('寄付30分方向 → 残り時間', fontweight='bold')
    ax.legend(fontsize=7); ax.grid(alpha=0.3, axis='y')

    # corr bar
    ax = fig.add_subplot(gs[1,4])
    corrs = []
    for sym, _ in SYMS:
        if sym not in first30_effect: continue
        r = first30_effect[sym]
        corrs.append((sym, r['r_first30'].corr(r['r_rest'])))
    xs = np.arange(len(corrs))
    bar_colors = ['#2ca02c' if c[1]>0 else '#d62728' for c in corrs]
    ax.bar(xs, [c[1] for c in corrs], color=bar_colors, edgecolor='black')
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xticks(xs); ax.set_xticklabels([c[0] for c in corrs], rotation=30, fontsize=8)
    ax.set_title('寄付30分 vs 残り 相関', fontweight='bold')
    ax.grid(alpha=0.3, axis='y')

    # row 2: 散布図 5銘柄
    for i, (sym, name) in enumerate(SYMS):
        ax = fig.add_subplot(gs[2,i])
        if sym not in first30_effect: continue
        r = first30_effect[sym]
        ax.scatter(r['r_first30'], r['r_rest'], alpha=0.5, s=12, color=colors[i])
        ax.axhline(0, color='gray', lw=0.5); ax.axvline(0, color='gray', lw=0.5)
        if len(r) > 5:
            b, a = np.polyfit(r['r_first30'], r['r_rest'], 1)
            xs = np.linspace(r['r_first30'].min(), r['r_first30'].max(), 50)
            ax.plot(xs, a+b*xs, color='black', lw=1)
            c = r['r_first30'].corr(r['r_rest'])
            ax.text(0.05, 0.95, f"slope={b:+.2f}\ncorr={c:+.3f}\nN={len(r)}",
                    transform=ax.transAxes, va='top', fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.set_xlabel('寄付30分 bps', fontsize=8); ax.set_ylabel('残り時間 bps', fontsize=8)
        ax.set_title(f'{sym} 寄付30分→残り', fontweight='bold', fontsize=10)
        ax.grid(alpha=0.3)

    # row 3: 高値/安値時刻ヒストグラム + ギャップフィル棒グラフ
    ax = fig.add_subplot(gs[3,0])
    for i, (sym, _) in enumerate(SYMS):
        df = dfs[sym]
        hi_times = []
        for d in sorted(set(df.index.date)):
            day = df[df.index.date == d]
            if len(day) < 200: continue
            hi_times.append(day['mo'].iloc[day['high'].argmax()])
        ax.hist(hi_times, bins=25, alpha=0.4, color=colors[i], label=sym)
    ax.axvspan(150, 210, color='gray', alpha=0.2)
    ax.set_title('日中高値到達時刻', fontweight='bold')
    ax.set_xticks(xticks); ax.set_xticklabels([time_label(t) for t in xticks], rotation=30, fontsize=7)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[3,1])
    for i, (sym, _) in enumerate(SYMS):
        df = dfs[sym]
        lo_times = []
        for d in sorted(set(df.index.date)):
            day = df[df.index.date == d]
            if len(day) < 200: continue
            lo_times.append(day['mo'].iloc[day['low'].argmin()])
        ax.hist(lo_times, bins=25, alpha=0.4, color=colors[i], label=sym)
    ax.axvspan(150, 210, color='gray', alpha=0.2)
    ax.set_title('日中安値到達時刻', fontweight='bold')
    ax.set_xticks(xticks); ax.set_xticklabels([time_label(t) for t in xticks], rotation=30, fontsize=7)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # ギャップフィル率の比較
    ax = fig.add_subplot(gs[3,2])
    gap_summary = []
    for sym, _ in SYMS:
        jp = U.load_jp_daily(sym)
        jp['prev_close'] = jp['close'].shift(1)
        jp['gap_pct'] = (jp['open']/jp['prev_close']-1)*100
        jp = jp.dropna(subset=['gap_pct'])
        df = dfs[sym]
        u_f=0; u_n=0; d_f=0; d_n=0
        for d, row in jp.iterrows():
            day = df[df.index.date == d]
            if len(day) < 200: continue
            dl = day['low'].min(); dh = day['high'].max()
            if row['gap_pct'] > 0.3:
                u_n += 1; u_f += int(dl <= row['prev_close'])
            elif row['gap_pct'] < -0.3:
                d_n += 1; d_f += int(dh >= row['prev_close'])
        gap_summary.append((sym, u_f/u_n*100 if u_n>0 else 0, d_f/d_n*100 if d_n>0 else 0))
    xs = np.arange(len(gap_summary)); w = 0.35
    ax.bar(xs-w/2, [g[1] for g in gap_summary], w, color='#2ca02c', label='Up-gap fill', edgecolor='black')
    ax.bar(xs+w/2, [g[2] for g in gap_summary], w, color='#d62728', label='Down-gap fill', edgecolor='black')
    ax.axhline(50, color='black', lw=0.8, linestyle='--')
    ax.set_xticks(xs); ax.set_xticklabels([g[0] for g in gap_summary], rotation=30, fontsize=8)
    ax.set_ylabel('フィル率 (%)')
    ax.set_title('寄付ギャップフィル率', fontweight='bold')
    ax.legend(fontsize=7); ax.grid(alpha=0.3, axis='y')

    # EODドリフト比較
    ax = fig.add_subplot(gs[3,3])
    eod_vals = []
    for sym, _ in SYMS:
        if sym in profile_cum:
            eod_vals.append((sym, profile_cum[sym]['mean'].iloc[-1]))
    xs = np.arange(len(eod_vals))
    bar_colors = ['#2ca02c' if e[1]>0 else '#d62728' for e in eod_vals]
    ax.bar(xs, [e[1] for e in eod_vals], color=bar_colors, edgecolor='black')
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xticks(xs); ax.set_xticklabels([e[0] for e in eod_vals], rotation=30, fontsize=8)
    ax.set_ylabel('EOD 平均リターン (bps)')
    ax.set_title('9:00 → 15:30 平均ドリフト', fontweight='bold')
    ax.grid(alpha=0.3, axis='y')

    # 寄付ボラ比較
    ax = fig.add_subplot(gs[3,4])
    vol_vals = []
    for sym, _ in SYMS:
        if sym in vol_by_min and 0 in vol_by_min[sym].index:
            vol_vals.append((sym, vol_by_min[sym].loc[0]))
    xs = np.arange(len(vol_vals))
    ax.bar(xs, [v[1] for v in vol_vals], color=colors[:len(vol_vals)], edgecolor='black')
    ax.set_xticks(xs); ax.set_xticklabels([v[0] for v in vol_vals], rotation=30, fontsize=8)
    ax.set_ylabel('9:00分足 bps std')
    ax.set_title('寄付ボラ (9:00分足)', fontweight='bold')
    ax.grid(alpha=0.3, axis='y')

    plt.suptitle('半導体5銘柄 イントラデイ典型パターン分析',
                 fontsize=16, fontweight='bold', y=1.00)
    out = os.path.join(os.path.dirname(__file__), 'result.png')
    plt.savefig(out, dpi=130, bbox_inches='tight')
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
