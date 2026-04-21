"""
出来高フィルター付き ORB (Opening Range Breakout) 戦略の最適化

仮説:
  寄付30分 (9:00-9:30) の出来高が通常より多い日は、情報の流入・機関の動きがあり、
  その後のブレイクアウトが持続しやすい。単純ORBよりエッジが出るはず。

対象銘柄: 非鉄3 + 半導体5 = 8銘柄
  5711 三菱マテリアル / 5706 三井金属 / 5713 住友金属鉱山
  8035 TEL / 6857 アドバンテスト / 6146 ディスコ / 4063 信越化学 / 6963 ローム

設計:
  OR (Opening Range): 9:00-OR_MIN の High / Low / Volume
    - OR_MIN ∈ {15, 30, 60}
  出来高フィルター: 当日OR期間のvolume / 直近N日同時刻出来高中央値 >= vol_ratio
    - vol_ratio ∈ {1.0 (無条件), 1.3, 1.5, 2.0, 3.0}
    - N=20 日 (同時刻 9:00-OR_MINの累計出来高)
  エントリー: OR期間後の分足でOR H 上抜け → Long, OR L 下抜け → Short
    - OR_MIN以降〜11:30 の間の最初のブレイクのみ (1日1トレード/方向)
  決済: 11:30 (前場引け) / 15:30 (大引け) / stop=反対側OR
    - EXIT ∈ {'1130', '1530', 'stop_1130', 'stop_1530'}
  Long/Short/両方:
    - DIR ∈ {'long', 'short', 'both'}
  コスト: 4bps (片側2bps × 往復)

評価:
  銘柄横断、vol_ratio × OR_MIN × EXIT × DIR でグリッド
  Sharpe >= +2.0 & N >= 30 & t_stat >= +2.0 を合格条件とする
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '20260421_common')))
import mdutil as U
import pandas as pd
import numpy as np
plt = U.matplotlib_jp()

SYMS = U.NONFERROUS + U.SEMICON  # 8 銘柄


def load_1min(sym):
    df = U.fetch_intraday(sym).dropna(subset=['open','high','low','close'])
    h, m = df.index.hour, df.index.minute
    mask = (((h==9)) | ((h==10)) | ((h==11)&(m<=30)) |
            ((h==12)&(m>=30)) | ((h==13)|(h==14)) | ((h==15)&(m<=30)))
    df = df[mask].copy()
    df['mo'] = (df.index.hour - 9) * 60 + df.index.minute  # 9:00=0
    return df


def simulate_day(day, or_min, exit_mode):
    """
    day: 当日分足 DataFrame (sorted, with 'mo','open','high','low','close','volume')
    or_min: OR期間 (minutes from 9:00)
    exit_mode: '1130' | '1530' | 'stop_1130' | 'stop_1530'
    returns: dict(direction, entry_price, exit_price, ret_bps, or_high, or_low, or_vol)
             or None
    """
    or_window = day[day['mo'] < or_min]  # 9:00 - (9:00+or_min)
    post = day[day['mo'] >= or_min]
    if len(or_window) < or_min * 0.5 or len(post) < 30:
        return None
    or_high = or_window['high'].max()
    or_low = or_window['low'].min()
    or_vol = or_window['volume'].sum()
    if or_high <= or_low:
        return None

    # 後場以降でOR上抜け/下抜けを探す。ブレイクした最初の足でエントリー、終値でエントリーと見なす
    # 決済時刻
    if exit_mode in ('1130', 'stop_1130'):
        exit_window = post[post['mo'] <= 150]  # 11:30 (mo=150)
    else:
        exit_window = post[post['mo'] <= 390]  # 15:30 (mo=390)
    if len(exit_window) < 5:
        return None

    trades = []
    for direction, level, is_up in [('long', or_high, True), ('short', or_low, False)]:
        # Break bar: high > or_high (long) or low < or_low (short)
        if is_up:
            brk = exit_window[exit_window['high'] > or_high]
        else:
            brk = exit_window[exit_window['low'] < or_low]
        if len(brk) == 0: continue
        brk_row = brk.iloc[0]
        entry_price = level  # 閾値そのもので約定 (保守的: ブレイク足のcloseでなく水準)
        entry_time = brk_row['mo']
        # 決済
        after = exit_window[exit_window['mo'] > entry_time]
        if len(after) == 0: continue
        if exit_mode.startswith('stop'):
            # 反対側OR到達で損切
            opp = or_low if is_up else or_high
            if is_up:
                stop_hit = after[after['low'] <= opp]
            else:
                stop_hit = after[after['high'] >= opp]
            if len(stop_hit) > 0:
                exit_price = opp
            else:
                exit_price = after['close'].iloc[-1]
        else:
            exit_price = after['close'].iloc[-1]
        ret = (exit_price / entry_price - 1) * 10000
        if not is_up: ret = -ret
        trades.append({
            'direction': direction, 'entry': entry_price, 'exit': exit_price,
            'ret_bps': ret, 'or_high': or_high, 'or_low': or_low, 'or_vol': or_vol,
            'entry_mo': entry_time,
        })
    return trades


def build_tradelist(df, or_min, exit_mode):
    """1銘柄の全日についてトレード候補を生成 (vol_ratio フィルタ前)"""
    rows = []
    dates = sorted(set(df.index.date))
    for d in dates:
        day = df[df.index.date == d]
        if len(day) < 200: continue
        trades = simulate_day(day, or_min, exit_mode)
        if trades is None: continue
        for t in trades:
            t['date'] = d
            rows.append(t)
    if not rows: return pd.DataFrame()
    tdf = pd.DataFrame(rows)
    # 同時刻OR期間の直近20日出来高中央値
    # date別に or_vol を取り出し、20日rolling median
    daily_vol = tdf.drop_duplicates(subset=['date'])[['date','or_vol']].set_index('date').sort_index()
    daily_vol['or_vol_med20'] = daily_vol['or_vol'].rolling(20, min_periods=10).median().shift(1)
    daily_vol['vol_ratio'] = daily_vol['or_vol'] / daily_vol['or_vol_med20']
    tdf = tdf.merge(daily_vol[['or_vol_med20','vol_ratio']], left_on='date', right_index=True, how='left')
    return tdf


def eval_strategy(tdf, vol_ratio_min, direction, cost_bps=U.COST_BPS):
    """フィルタ適用して統計計算"""
    if len(tdf) == 0: return None, None
    sub = tdf[tdf['vol_ratio'] >= vol_ratio_min].copy()
    if direction != 'both':
        sub = sub[sub['direction'] == direction]
    if len(sub) < 5: return sub, None
    sub['net_bps'] = sub['ret_bps'] - cost_bps
    st = U.compute_stats(sub['net_bps'].values)
    return sub, st


def main():
    print("=== 出来高フィルター付き ORB 最適化 ===\n")
    print(f"対象 {len(SYMS)} 銘柄: " + ", ".join([s for s,_ in SYMS]))
    print(f"コスト: {U.COST_BPS}bps (片側2bps×往復)\n")

    OR_MINS = [15, 30, 60]
    VOL_RATIOS = [1.0, 1.3, 1.5, 2.0, 3.0]
    EXITS = ['1130', '1530', 'stop_1130', 'stop_1530']
    DIRS = ['long', 'short', 'both']

    # 銘柄ごとに load + build_tradelist (OR_MIN × EXIT 組)
    all_trades = {}  # (sym, or_min, exit) -> DataFrame
    for sym, name in SYMS:
        df = load_1min(sym)
        for or_min in OR_MINS:
            for ex in EXITS:
                tdf = build_tradelist(df, or_min, ex)
                if not tdf.empty:
                    tdf['symbol'] = sym
                all_trades[(sym, or_min, ex)] = tdf
        print(f"  {sym} {name}: {len(df)} rows loaded")

    # ========== グリッド評価: 全銘柄集約 ==========
    print("\n=== グリッド結果 (全8銘柄集約、Sharpe降順Top25) ===")
    results = []
    for or_min in OR_MINS:
        for ex in EXITS:
            pool = pd.concat([all_trades[(s, or_min, ex)]
                              for s,_ in SYMS if not all_trades[(s, or_min, ex)].empty],
                             ignore_index=True)
            for vr in VOL_RATIOS:
                for dr in DIRS:
                    sub, st = eval_strategy(pool, vr, dr)
                    if st is None or st['n'] < 30: continue
                    results.append({
                        'or_min': or_min, 'exit': ex, 'vol_ratio>=': vr, 'dir': dr,
                        'N': st['n'], 'mean_bps': st['mean'], 'wr': st['wr'],
                        'pf': st['pf'], 'sharpe': st['sharpe'], 'maxdd': st['maxdd'],
                        't_stat': st['t_stat']
                    })
    res = pd.DataFrame(results).sort_values('sharpe', ascending=False)
    print(res.head(25).to_string(index=False,
          formatters={'mean_bps': '{:+.1f}'.format, 'wr': '{:.1f}'.format,
                      'pf': '{:.2f}'.format, 'sharpe': '{:+.2f}'.format,
                      'maxdd': '{:+.0f}'.format, 't_stat': '{:+.2f}'.format}))

    # 合格条件: Sharpe>=2.0, N>=30, t>=2.0
    winners = res[(res['sharpe']>=2.0) & (res['N']>=30) & (res['t_stat']>=2.0)]
    print(f"\n合格戦略 (Sharpe>=2.0 & N>=30 & t>=2.0): {len(winners)} 件")

    # ========== 銘柄別: 最優秀パラメータでの成績 ==========
    print("\n=== 銘柄別 Top3 戦略 (Sharpeベース) ===")
    per_sym_best = {}
    for sym, name in SYMS:
        per_rows = []
        for or_min in OR_MINS:
            for ex in EXITS:
                tdf = all_trades[(sym, or_min, ex)]
                if tdf.empty: continue
                for vr in VOL_RATIOS:
                    for dr in DIRS:
                        sub, st = eval_strategy(tdf, vr, dr)
                        if st is None or st['n'] < 15: continue
                        per_rows.append({
                            'sym': sym, 'or': or_min, 'exit': ex,
                            'vr': vr, 'dir': dr, 'N': st['n'],
                            'mean_bps': st['mean'], 'wr': st['wr'], 'pf': st['pf'],
                            'sharpe': st['sharpe'], 't': st['t_stat']
                        })
        if not per_rows: continue
        pr = pd.DataFrame(per_rows).sort_values('sharpe', ascending=False)
        per_sym_best[sym] = pr
        print(f"\n{sym} {name}:")
        print(pr.head(3).to_string(index=False,
              formatters={'mean_bps': '{:+.1f}'.format, 'wr': '{:.1f}'.format,
                          'pf': '{:.2f}'.format, 'sharpe': '{:+.2f}'.format,
                          't': '{:+.2f}'.format}))

    # ========== vol_ratio による単調性検証 (OR=30min, EXIT=1530) ==========
    print("\n=== vol_ratio 単調性 (OR=30, EXIT=1530, DIR=both) ===")
    pool = pd.concat([all_trades[(s, 30, '1530')]
                      for s,_ in SYMS if not all_trades[(s, 30, '1530')].empty],
                     ignore_index=True)
    mono = []
    for vr in [1.0, 1.2, 1.5, 2.0, 3.0, 5.0]:
        sub, st = eval_strategy(pool, vr, 'both')
        if st is None: continue
        mono.append({'vol>=': vr, 'N': st['n'], 'mean': st['mean'],
                     'wr': st['wr'], 'sharpe': st['sharpe']})
    mono_df = pd.DataFrame(mono)
    print(mono_df.to_string(index=False,
          formatters={'mean': '{:+.1f}'.format, 'wr': '{:.1f}'.format, 'sharpe': '{:+.2f}'.format}))

    # ========== 可視化 ==========
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.3)

    # (0,0) Sharpe ヒートマップ: vol_ratio × OR_MIN (EXIT=1530, DIR=both)
    ax = fig.add_subplot(gs[0,0])
    heat = np.full((len(VOL_RATIOS), len(OR_MINS)), np.nan)
    for i, vr in enumerate(VOL_RATIOS):
        for j, or_min in enumerate(OR_MINS):
            pool = pd.concat([all_trades[(s, or_min, '1530')]
                              for s,_ in SYMS if not all_trades[(s, or_min, '1530')].empty],
                             ignore_index=True)
            sub, st = eval_strategy(pool, vr, 'both')
            if st is not None and st['n']>=30:
                heat[i,j] = st['sharpe']
    im = ax.imshow(heat, cmap='RdYlGn', aspect='auto', vmin=-3, vmax=3)
    ax.set_xticks(range(len(OR_MINS))); ax.set_xticklabels([f"{m}min" for m in OR_MINS])
    ax.set_yticks(range(len(VOL_RATIOS))); ax.set_yticklabels([f">={v}" for v in VOL_RATIOS])
    ax.set_xlabel('OR期間'); ax.set_ylabel('vol_ratio')
    ax.set_title('Sharpe (EXIT=1530, DIR=both)', fontweight='bold')
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            if not np.isnan(heat[i,j]):
                ax.text(j, i, f"{heat[i,j]:+.1f}", ha='center', va='center', fontsize=9,
                        color='black' if abs(heat[i,j])<2 else 'white')
    fig.colorbar(im, ax=ax, shrink=0.8)

    # (0,1) Sharpe ヒートマップ: vol_ratio × EXIT (OR=30, DIR=both)
    ax = fig.add_subplot(gs[0,1])
    heat = np.full((len(VOL_RATIOS), len(EXITS)), np.nan)
    for i, vr in enumerate(VOL_RATIOS):
        for j, ex in enumerate(EXITS):
            pool = pd.concat([all_trades[(s, 30, ex)]
                              for s,_ in SYMS if not all_trades[(s, 30, ex)].empty],
                             ignore_index=True)
            sub, st = eval_strategy(pool, vr, 'both')
            if st is not None and st['n']>=30:
                heat[i,j] = st['sharpe']
    im = ax.imshow(heat, cmap='RdYlGn', aspect='auto', vmin=-3, vmax=3)
    ax.set_xticks(range(len(EXITS))); ax.set_xticklabels(EXITS, rotation=30)
    ax.set_yticks(range(len(VOL_RATIOS))); ax.set_yticklabels([f">={v}" for v in VOL_RATIOS])
    ax.set_xlabel('決済'); ax.set_ylabel('vol_ratio')
    ax.set_title('Sharpe (OR=30min, DIR=both)', fontweight='bold')
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            if not np.isnan(heat[i,j]):
                ax.text(j, i, f"{heat[i,j]:+.1f}", ha='center', va='center', fontsize=9,
                        color='black' if abs(heat[i,j])<2 else 'white')
    fig.colorbar(im, ax=ax, shrink=0.8)

    # (0,2) vol_ratio 単調性ライン
    ax = fig.add_subplot(gs[0,2])
    ax.plot(mono_df['vol>='].values, mono_df['sharpe'].values, 'o-', color='#1f77b4', lw=2)
    ax.axhline(0, color='gray', lw=0.8)
    ax.set_xlabel('vol_ratio >=')
    ax.set_ylabel('Sharpe')
    ax.set_title('vol_ratio 閾値 vs Sharpe\n(OR=30, EXIT=1530, both)', fontweight='bold')
    for _, r in mono_df.iterrows():
        ax.annotate(f"N={int(r['N'])}", (r['vol>='], r['sharpe']),
                    textcoords="offset points", xytext=(5,5), fontsize=8)
    ax.grid(alpha=0.3)

    # (1,0-2) & (2,0-2): 銘柄別 best 戦略の累積損益
    top_syms = [s for s,_ in SYMS]
    for idx, sym in enumerate(top_syms[:6]):
        r = idx // 3; c = idx % 3
        ax = fig.add_subplot(gs[1+r, c])
        if sym not in per_sym_best or per_sym_best[sym].empty:
            ax.set_title(f"{sym} N/A"); continue
        best = per_sym_best[sym].iloc[0]
        tdf = all_trades[(sym, int(best['or']), best['exit'])]
        sub, st = eval_strategy(tdf, best['vr'], best['dir'])
        if sub is None or len(sub) == 0: continue
        sub = sub.sort_values('date')
        sub['cum'] = (sub['ret_bps'] - U.COST_BPS).cumsum()
        ax.plot(pd.to_datetime(sub['date']), sub['cum'].values, color='#1f77b4', lw=1.3)
        ax.axhline(0, color='gray', lw=0.8)
        ax.set_title(f"{sym} best: OR={int(best['or'])}, vr>={best['vr']}, "
                     f"{best['exit']}, {best['dir']}\n"
                     f"N={st['n']}, Sharpe={st['sharpe']:+.2f}, WR={st['wr']:.0f}%",
                     fontsize=9, fontweight='bold')
        ax.tick_params(axis='x', labelrotation=30, labelsize=7)
        ax.set_ylabel('累積 bps')
        ax.grid(alpha=0.3)

    plt.suptitle('出来高フィルター付き ORB 戦略最適化',
                 fontsize=15, fontweight='bold', y=1.00)
    out = os.path.join(os.path.dirname(__file__), 'result.png')
    plt.savefig(out, dpi=130, bbox_inches='tight')
    print(f"\nSaved: {out}")

    # ========== 最終推奨 ==========
    if len(winners) > 0:
        print("\n=== 合格戦略 (全リスト) ===")
        print(winners.to_string(index=False,
              formatters={'mean_bps': '{:+.1f}'.format, 'wr': '{:.1f}'.format,
                          'pf': '{:.2f}'.format, 'sharpe': '{:+.2f}'.format,
                          'maxdd': '{:+.0f}'.format, 't_stat': '{:+.2f}'.format}))

    # CSV 保存
    res.to_csv(os.path.join(os.path.dirname(__file__), 'grid_results.csv'), index=False)
    per_all = []
    for sym, pr in per_sym_best.items():
        per_all.append(pr.head(3))
    if per_all:
        pd.concat(per_all, ignore_index=True).to_csv(
            os.path.join(os.path.dirname(__file__), 'per_symbol_top3.csv'), index=False)
    print(f"\nCSV saved: grid_results.csv, per_symbol_top3.csv")


if __name__ == "__main__":
    main()
