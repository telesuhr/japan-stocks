"""
VWAP 徹底分析 — 全22銘柄 × 多角的アプローチ

検証する仮説:
  (A) Attraction (引き寄せ): |dev|>=X で乖離した後、価格はVWAPに戻るか？
  (B) Breakout (ブレイク継続): 一定時間下に居て上抜けたら、そのまま上に走るか？
  (C) Reversion (逆張り): 決定時刻での |dev| シグナルで逆張り→大引け決済
  (D) Trend (順張り): 同シグナルで順張り→大引け決済
  (E) Slope: VWAP自体の傾きが方向シグナルになるか？
  (F) Session close vs VWAP: 引けはVWAPより上/下、に規則性があるか？

対象: 22銘柄 (CORE5 / NONFERROUS / ENERGY / SHIPPING / SEMICON / DOMESTIC_SHORT からユニーク抽出)
コスト: 4bps (往復, 片側2bps)
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '20260421_common')))
import mdutil as U
import pandas as pd
import numpy as np
plt = U.matplotlib_jp()


# === 銘柄リスト (ユニーク) ===
def uniq_syms():
    pool = U.CORE5 + U.NONFERROUS + U.ENERGY + U.SHIPPING + U.SEMICON + U.DOMESTIC_SHORT
    seen, out = set(), []
    for s, n in pool:
        if s not in seen:
            out.append((s, n)); seen.add(s)
    return out


SYMS = uniq_syms()
DECISION_TIMES_MO = {
    '9:30': 30, '10:00': 60, '11:00': 120,
    '11:30': 150, '13:30': 270, '14:00': 300,
}


# ============================== データロード ==============================
def load_1min(sym):
    df = U.fetch_intraday(sym).dropna(subset=['open', 'high', 'low', 'close'])
    h, m = df.index.hour, df.index.minute
    mask = ((h == 9) | (h == 10) | ((h == 11) & (m <= 30)) |
            ((h == 12) & (m >= 30)) | (h == 13) | (h == 14) | ((h == 15) & (m <= 30)))
    df = df[mask].copy()
    df['mo'] = (df.index.hour - 9) * 60 + df.index.minute
    # 12:00-12:30昼休みをmoで連続化: 11:30=150, 12:30=180 のまま
    return df


def compute_session_vwap(day):
    tp = (day['high'] + day['low'] + day['close']) / 3
    v = day['volume'].fillna(0)
    pv = tp * v
    cum_v = v.cumsum()
    vwap = pv.cumsum() / cum_v.replace(0, np.nan)
    return vwap.ffill().fillna(tp)


# ============================== 日別特徴量 ==============================
def build_day_features(day):
    """1日分から特徴量dict。day_frame(mo, close, vwap, dev_bps)も返す"""
    if len(day) < 200:
        return None, None
    day = day.copy()
    day['vwap'] = compute_session_vwap(day)
    day['dev_bps'] = (day['close'] / day['vwap'] - 1) * 10000
    day['above'] = (day['dev_bps'] > 0).astype(int)
    open_p = day['open'].iloc[0]
    eod = day['close'].iloc[-1]

    row = {'date': day.index[0].date(), 'open': open_p, 'eod_close': eod,
           'eod_ret_bps': (eod / open_p - 1) * 10000}

    # 時刻別 dev と remaining return
    for tname, tmo in DECISION_TIMES_MO.items():
        at = day[day['mo'] >= tmo].head(1)
        if len(at) == 0:
            continue
        row[f'dev_{tname}'] = at['dev_bps'].iloc[0]
        row[f'close_{tname}'] = at['close'].iloc[0]
        row[f'vwap_{tname}'] = at['vwap'].iloc[0]
        row[f'ret_to_eod_{tname}'] = (eod / at['close'].iloc[0] - 1) * 10000
        # VWAP slope (直近15min)
        past = day[(day['mo'] <= tmo) & (day['mo'] > tmo - 15)]
        if len(past) >= 5:
            row[f'vwap_slope_{tname}'] = (past['vwap'].iloc[-1] / past['vwap'].iloc[0] - 1) * 10000
        else:
            row[f'vwap_slope_{tname}'] = 0.0
        # この時刻までに上にいた比率
        upto = day[day['mo'] <= tmo]
        row[f'above_pct_{tname}'] = upto['above'].mean() * 100

    # VWAP最初の交差(9:30以降): 最初に符号が反転したmo
    post930 = day[day['mo'] >= 30].copy()
    if len(post930) > 0:
        first_above = post930['above'].iloc[0]
        crossed = post930[post930['above'] != first_above]
        row['first_cross_mo'] = crossed['mo'].iloc[0] if len(crossed) > 0 else np.nan
        row['first_cross_dir'] = (1 if first_above == 0 else -1) if len(crossed) > 0 else 0
    # セッション全体の上にいた比率
    row['above_pct_all'] = day['above'].mean() * 100
    # 15:30 引けがVWAPより上か
    row['close_above_vwap'] = int(day['dev_bps'].iloc[-1] > 0)
    row['final_dev_bps'] = day['dev_bps'].iloc[-1]

    return row, day[['mo', 'close', 'vwap', 'dev_bps', 'above']]


def analyze_attraction(day_frames, threshold_bps=50, within_min=60):
    """
    Attraction test:
    毎分 |dev| が threshold を上向き横切った瞬間を記録
    → その後 within_min 以内に |dev|<5 で VWAP に戻ったか？
    戻った場合は到達時間も記録
    """
    events_up = []   # +側乖離 (上方向に突っ込んだイベント)
    events_dn = []   # -側乖離
    for df in day_frames:
        dev = df['dev_bps'].values
        mo = df['mo'].values
        n = len(dev)
        # 上向きクロス: |dev|>=th 初到達
        # up event: dev crosses >= +th
        for i in range(1, n):
            if dev[i - 1] < threshold_bps and dev[i] >= threshold_bps:
                # 以降 within_min 以内に |dev|<5 に戻るか
                end_idx = i
                touched = False
                t_touch = None
                while end_idx < n and (mo[end_idx] - mo[i]) <= within_min:
                    if abs(dev[end_idx]) < 5:
                        touched = True
                        t_touch = mo[end_idx] - mo[i]
                        break
                    end_idx += 1
                events_up.append({'mo_event': mo[i], 'touched': int(touched),
                                  't_touch': t_touch if touched else within_min + 1,
                                  'dev_at_event': dev[i]})
            if dev[i - 1] > -threshold_bps and dev[i] <= -threshold_bps:
                end_idx = i
                touched = False
                t_touch = None
                while end_idx < n and (mo[end_idx] - mo[i]) <= within_min:
                    if abs(dev[end_idx]) < 5:
                        touched = True
                        t_touch = mo[end_idx] - mo[i]
                        break
                    end_idx += 1
                events_dn.append({'mo_event': mo[i], 'touched': int(touched),
                                  't_touch': t_touch if touched else within_min + 1,
                                  'dev_at_event': dev[i]})
    return pd.DataFrame(events_up), pd.DataFrame(events_dn)


def backtest_breakout(day_frames, min_persistence=30, hold_min=60, direction='above'):
    """
    (B) VWAP ブレイク継続:
    'above': min_persistence 以上連続で下にいた後、VWAPを上抜け → Long, hold_min 後決済
    'below': 逆
    """
    trades = []
    for df in day_frames:
        dev = df['dev_bps'].values
        mo = df['mo'].values
        above = df['above'].values
        close_arr = df['close'].values
        n = len(dev)
        # 連続カウント
        run = 0
        last_sign = 0  # 1=above, -1=below
        for i in range(n):
            sign = 1 if above[i] == 1 else -1
            if sign == last_sign:
                run += 1
            else:
                run = 1
                last_sign = sign
            # breakout判定: 前までbelowで、今aboveに切り替わった
            if i > 0 and above[i] != above[i - 1]:
                if direction == 'above' and above[i] == 1:
                    # 直前まで何連続でbelowだったか
                    prev_run = 0
                    j = i - 1
                    while j >= 0 and above[j] == 0:
                        prev_run += 1
                        j -= 1
                    if prev_run >= min_persistence:
                        # entry at close[i], exit at mo[i]+hold or EOD
                        entry = close_arr[i]
                        # find exit
                        tgt_mo = mo[i] + hold_min
                        exit_idx = n - 1
                        for k in range(i + 1, n):
                            if mo[k] >= tgt_mo:
                                exit_idx = k
                                break
                        exit_p = close_arr[exit_idx]
                        trades.append({
                            'mo_entry': mo[i], 'mo_exit': mo[exit_idx],
                            'direction': +1,
                            'gross_bps': (exit_p / entry - 1) * 10000,
                            'run_before': prev_run,
                        })
                elif direction == 'below' and above[i] == 0:
                    prev_run = 0
                    j = i - 1
                    while j >= 0 and above[j] == 1:
                        prev_run += 1
                        j -= 1
                    if prev_run >= min_persistence:
                        entry = close_arr[i]
                        tgt_mo = mo[i] + hold_min
                        exit_idx = n - 1
                        for k in range(i + 1, n):
                            if mo[k] >= tgt_mo:
                                exit_idx = k
                                break
                        exit_p = close_arr[exit_idx]
                        trades.append({
                            'mo_entry': mo[i], 'mo_exit': mo[exit_idx],
                            'direction': -1,
                            'gross_bps': (entry / exit_p - 1) * 10000,  # short pnl
                            'run_before': prev_run,
                        })
    return pd.DataFrame(trades)


# ============================== メイン ==============================
def main():
    print("=" * 80)
    print(f"VWAP 徹底分析 - 全{len(SYMS)}銘柄")
    print("=" * 80)

    # ----- 各銘柄: 日別特徴量と日別フレーム -----
    feats = {}
    day_frames_all = {}
    for sym, name in SYMS:
        try:
            df = load_1min(sym)
        except Exception as e:
            print(f"  {sym}: load error {e}")
            continue
        dates = sorted(set(df.index.date))
        rows = []
        frames = []
        for d in dates:
            day = df[df.index.date == d]
            row, frame = build_day_features(day)
            if row is None:
                continue
            rows.append(row)
            if frame is not None:
                frames.append(frame)
        if not rows:
            continue
        feats[sym] = pd.DataFrame(rows).set_index('date')
        day_frames_all[sym] = frames
        print(f"  {sym} {name}: {len(rows)} 日")

    # ========== 1. 記述統計 ==========
    print("\n" + "=" * 80)
    print("1. 記述統計 (VWAP基礎指標)")
    print("=" * 80)
    desc = []
    for sym, name in SYMS:
        if sym not in feats:
            continue
        f = feats[sym]
        row = {
            'sym': sym, 'name': name, 'days': len(f),
            'mean|dev_1130|': f['dev_11:30'].abs().mean() if 'dev_11:30' in f else np.nan,
            'std_dev_1130': f['dev_11:30'].std() if 'dev_11:30' in f else np.nan,
            'mean_above_pct': f['above_pct_all'].mean(),
            'P(close_above)': f['close_above_vwap'].mean() * 100,
            'mean_final_dev': f['final_dev_bps'].mean(),
            'med_first_cross': f['first_cross_mo'].median() if 'first_cross_mo' in f else np.nan,
        }
        desc.append(row)
    desc_df = pd.DataFrame(desc)
    print(desc_df.to_string(index=False, float_format='{:.1f}'.format))
    desc_df.to_csv(os.path.join(os.path.dirname(__file__), 'descriptive.csv'), index=False)

    # ========== 2. 相関 (dev vs ret_to_eod) ==========
    print("\n" + "=" * 80)
    print("2. VWAP乖離 vs 残り時間リターン 相関 (負=Reversion, 正=Trend)")
    print("=" * 80)
    corr_rows = []
    for sym, _ in SYMS:
        if sym not in feats:
            continue
        f = feats[sym]
        for tname in DECISION_TIMES_MO.keys():
            dev_c = f'dev_{tname}'; ret_c = f'ret_to_eod_{tname}'
            if dev_c not in f.columns:
                continue
            sub = f[[dev_c, ret_c]].dropna()
            if len(sub) < 30:
                continue
            corr_rows.append({'sym': sym, 'time': tname, 'N': len(sub),
                              'corr': sub[dev_c].corr(sub[ret_c]),
                              'dev_std': sub[dev_c].std()})
    corr_df = pd.DataFrame(corr_rows)
    pivot_c = corr_df.pivot(index='sym', columns='time', values='corr')[list(DECISION_TIMES_MO.keys())]
    print(pivot_c.to_string(float_format='{:+.3f}'.format))
    corr_df.to_csv(os.path.join(os.path.dirname(__file__), 'corr_table.csv'), index=False)

    # ========== 3. Reversion/Trend grid ==========
    print("\n" + "=" * 80)
    print("3. Reversion/Trend バックテスト")
    print("=" * 80)
    rev_rows, tr_rows = [], []
    for sym, _ in SYMS:
        if sym not in feats:
            continue
        f = feats[sym]
        for tname in DECISION_TIMES_MO.keys():
            dev_c = f'dev_{tname}'; ret_c = f'ret_to_eod_{tname}'
            if dev_c not in f.columns:
                continue
            sub = f[[dev_c, ret_c]].dropna()
            for th in [20, 30, 50, 80, 120]:
                # reversion
                d = np.where(sub[dev_c] >= th, -1, np.where(sub[dev_c] <= -th, +1, 0))
                mask = d != 0
                if mask.sum() < 15:
                    pass
                else:
                    net = d[mask] * sub[ret_c].values[mask] - U.COST_BPS
                    st = U.compute_stats(net)
                    rev_rows.append({'sym': sym, 'time': tname, 'th': th, 'N': st['n'],
                                     'mean': st['mean'], 'wr': st['wr'], 'pf': st['pf'],
                                     'sharpe': st['sharpe'], 't': st['t_stat']})
                # trend
                d = np.where(sub[dev_c] >= th, +1, np.where(sub[dev_c] <= -th, -1, 0))
                mask = d != 0
                if mask.sum() < 15:
                    continue
                net = d[mask] * sub[ret_c].values[mask] - U.COST_BPS
                st = U.compute_stats(net)
                tr_rows.append({'sym': sym, 'time': tname, 'th': th, 'N': st['n'],
                                'mean': st['mean'], 'wr': st['wr'], 'pf': st['pf'],
                                'sharpe': st['sharpe'], 't': st['t_stat']})
    rev_df = pd.DataFrame(rev_rows)
    tr_df = pd.DataFrame(tr_rows)

    fmt = {'mean': '{:+.1f}'.format, 'wr': '{:.1f}'.format, 'pf': '{:.2f}'.format,
           'sharpe': '{:+.2f}'.format, 't': '{:+.2f}'.format}
    print("\n--- Reversion Top 20 (Sharpe) ---")
    print(rev_df.sort_values('sharpe', ascending=False).head(20).to_string(index=False, formatters=fmt))
    print("\n--- Trend Top 20 (Sharpe) ---")
    print(tr_df.sort_values('sharpe', ascending=False).head(20).to_string(index=False, formatters=fmt))
    rev_df.to_csv(os.path.join(os.path.dirname(__file__), 'reversion_grid.csv'), index=False)
    tr_df.to_csv(os.path.join(os.path.dirname(__file__), 'trend_grid.csv'), index=False)

    # 合格基準
    rev_pass = rev_df[(rev_df['sharpe'] >= 2.0) & (rev_df['N'] >= 30) & (rev_df['t'] >= 2.0)]
    tr_pass = tr_df[(tr_df['sharpe'] >= 2.0) & (tr_df['N'] >= 30) & (tr_df['t'] >= 2.0)]
    print(f"\n合格 Reversion: {len(rev_pass)}件 / Trend: {len(tr_pass)}件")
    if len(rev_pass):
        print("\n--- Reversion 合格 ---")
        print(rev_pass.sort_values('sharpe', ascending=False).to_string(index=False, formatters=fmt))
    if len(tr_pass):
        print("\n--- Trend 合格 ---")
        print(tr_pass.sort_values('sharpe', ascending=False).to_string(index=False, formatters=fmt))

    # ========== 4. Attraction test ==========
    print("\n" + "=" * 80)
    print("4. VWAP Attraction (引き寄せ) テスト")
    print("=" * 80)
    print("  |dev|>=Xbps を横切った後、60min以内にVWAP(|dev|<5bps)に戻った割合")
    attract_rows = []
    for sym, _ in SYMS:
        if sym not in day_frames_all:
            continue
        frames = day_frames_all[sym]
        for th in [30, 50, 80]:
            up, dn = analyze_attraction(frames, threshold_bps=th, within_min=60)
            attract_rows.append({
                'sym': sym, 'th': th,
                'up_N': len(up), 'up_returnP': up['touched'].mean() * 100 if len(up) else np.nan,
                'up_med_ttr': up[up['touched'] == 1]['t_touch'].median() if (len(up) and (up['touched'] == 1).any()) else np.nan,
                'dn_N': len(dn), 'dn_returnP': dn['touched'].mean() * 100 if len(dn) else np.nan,
                'dn_med_ttr': dn[dn['touched'] == 1]['t_touch'].median() if (len(dn) and (dn['touched'] == 1).any()) else np.nan,
            })
    attract_df = pd.DataFrame(attract_rows)
    print(attract_df.to_string(index=False, float_format='{:.1f}'.format))
    attract_df.to_csv(os.path.join(os.path.dirname(__file__), 'attraction.csv'), index=False)

    # ========== 5. Breakout continuation ==========
    print("\n" + "=" * 80)
    print("5. VWAP Breakout 継続テスト")
    print("=" * 80)
    print("  連続30min以上 下(上)にいた後、VWAPを上(下)抜け → 60min後決済")
    bo_rows = []
    for sym, _ in SYMS:
        if sym not in day_frames_all:
            continue
        frames = day_frames_all[sym]
        for pers in [20, 30, 60]:
            for hold in [30, 60, 120]:
                tr_up = backtest_breakout(frames, min_persistence=pers, hold_min=hold, direction='above')
                tr_dn = backtest_breakout(frames, min_persistence=pers, hold_min=hold, direction='below')
                for lbl, tr in [('up', tr_up), ('down', tr_dn)]:
                    if len(tr) < 15:
                        continue
                    net = tr['gross_bps'].values - U.COST_BPS
                    st = U.compute_stats(net)
                    bo_rows.append({
                        'sym': sym, 'break_dir': lbl, 'persist': pers, 'hold': hold,
                        'N': st['n'], 'mean': st['mean'], 'wr': st['wr'],
                        'pf': st['pf'], 'sharpe': st['sharpe'], 't': st['t_stat']
                    })
    bo_df = pd.DataFrame(bo_rows)
    print("\n--- Breakout Top 20 (Sharpe) ---")
    print(bo_df.sort_values('sharpe', ascending=False).head(20).to_string(index=False, formatters=fmt))
    bo_df.to_csv(os.path.join(os.path.dirname(__file__), 'breakout.csv'), index=False)
    bo_pass = bo_df[(bo_df['sharpe'] >= 2.0) & (bo_df['N'] >= 30) & (bo_df['t'] >= 2.0)]
    print(f"\nBreakout 合格: {len(bo_pass)}件")
    if len(bo_pass):
        print(bo_pass.sort_values('sharpe', ascending=False).to_string(index=False, formatters=fmt))

    # ========== 6. Slope signal ==========
    print("\n" + "=" * 80)
    print("6. VWAP Slope シグナル (直近15minのVWAP傾きで方向判定)")
    print("=" * 80)
    slope_rows = []
    for sym, _ in SYMS:
        if sym not in feats:
            continue
        f = feats[sym]
        for tname in DECISION_TIMES_MO.keys():
            sl_c = f'vwap_slope_{tname}'; ret_c = f'ret_to_eod_{tname}'
            if sl_c not in f.columns:
                continue
            sub = f[[sl_c, ret_c]].dropna()
            if len(sub) < 30:
                continue
            for th in [5, 10, 20]:
                d = np.where(sub[sl_c] >= th, +1, np.where(sub[sl_c] <= -th, -1, 0))
                mask = d != 0
                if mask.sum() < 20:
                    continue
                net = d[mask] * sub[ret_c].values[mask] - U.COST_BPS
                st = U.compute_stats(net)
                slope_rows.append({'sym': sym, 'time': tname, 'th': th, 'N': st['n'],
                                   'mean': st['mean'], 'wr': st['wr'], 'pf': st['pf'],
                                   'sharpe': st['sharpe'], 't': st['t_stat']})
    slope_df = pd.DataFrame(slope_rows)
    print("\n--- Slope Top 20 (Sharpe) ---")
    print(slope_df.sort_values('sharpe', ascending=False).head(20).to_string(index=False, formatters=fmt))
    slope_df.to_csv(os.path.join(os.path.dirname(__file__), 'slope_grid.csv'), index=False)
    slope_pass = slope_df[(slope_df['sharpe'] >= 2.0) & (slope_df['N'] >= 30) & (slope_df['t'] >= 2.0)]
    print(f"\nSlope 合格: {len(slope_pass)}件")
    if len(slope_pass):
        print(slope_pass.sort_values('sharpe', ascending=False).to_string(index=False, formatters=fmt))

    # ========== 7. 銘柄分類 (Reversion / Trend / Breakout型) ==========
    print("\n" + "=" * 80)
    print("7. 銘柄特性分類 (最優秀戦略 Sharpe ベース)")
    print("=" * 80)
    classification = []
    for sym, name in SYMS:
        if sym not in feats:
            continue
        best = []
        sub_rev = rev_df[(rev_df['sym'] == sym) & (rev_df['N'] >= 25)]
        sub_tr = tr_df[(tr_df['sym'] == sym) & (tr_df['N'] >= 25)]
        sub_bo = bo_df[(bo_df['sym'] == sym) & (bo_df['N'] >= 25)]
        sub_sl = slope_df[(slope_df['sym'] == sym) & (slope_df['N'] >= 25)]
        for label, dfa in [('Reversion', sub_rev), ('Trend', sub_tr),
                           ('Breakout', sub_bo), ('Slope', sub_sl)]:
            if len(dfa) == 0:
                continue
            r = dfa.sort_values('sharpe', ascending=False).iloc[0]
            best.append((label, r['sharpe'], r['N'], r.get('t', 0), r.to_dict()))
        if not best:
            continue
        best.sort(key=lambda x: -x[1])
        top = best[0]
        classification.append({
            'sym': sym, 'name': name, 'best_type': top[0],
            'best_sharpe': top[1], 'best_N': top[2], 'best_t': top[3],
        })
    cls_df = pd.DataFrame(classification).sort_values('best_sharpe', ascending=False)
    print(cls_df.to_string(index=False, float_format='{:.2f}'.format))
    cls_df.to_csv(os.path.join(os.path.dirname(__file__), 'classification.csv'), index=False)

    # ========== 可視化 ==========
    fig = plt.figure(figsize=(22, 18))
    gs = fig.add_gridspec(4, 3, hspace=0.65, wspace=0.35)

    # (0,0) 相関ヒートマップ
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(pivot_c.values, cmap='RdBu_r', aspect='auto', vmin=-0.3, vmax=0.3)
    ax.set_xticks(range(len(pivot_c.columns))); ax.set_xticklabels(pivot_c.columns, rotation=30)
    ax.set_yticks(range(len(pivot_c.index))); ax.set_yticklabels(pivot_c.index, fontsize=8)
    ax.set_title('corr(dev, ret_to_eod) 銘柄×時刻', fontweight='bold')
    fig.colorbar(im, ax=ax, shrink=0.8)

    # (0,1) Attraction: 60min以内戻り率 (th=50)
    ax = fig.add_subplot(gs[0, 1])
    sub = attract_df[attract_df['th'] == 50].set_index('sym')
    if len(sub):
        xs = np.arange(len(sub))
        w = 0.35
        ax.bar(xs - w / 2, sub['up_returnP'].values, w, label='上乖離→下戻り', color='#d62728')
        ax.bar(xs + w / 2, sub['dn_returnP'].values, w, label='下乖離→上戻り', color='#1f77b4')
        ax.set_xticks(xs); ax.set_xticklabels(sub.index, rotation=90, fontsize=7)
        ax.axhline(50, color='gray', ls='--', alpha=0.5)
        ax.set_ylabel('60min以内VWAP戻り率 (%)')
        ax.set_title('Attraction: |dev|>=50bps から VWAP復帰率', fontweight='bold')
        ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')

    # (0,2) P(15:30引けがVWAP上)
    ax = fig.add_subplot(gs[0, 2])
    if len(desc_df):
        sub = desc_df.set_index('sym').sort_values('P(close_above)', ascending=False)
        ax.barh(range(len(sub)), sub['P(close_above)'].values, color='#2ca02c')
        ax.axvline(50, color='red', ls='--', alpha=0.6)
        ax.set_yticks(range(len(sub))); ax.set_yticklabels(sub.index, fontsize=7)
        ax.set_xlabel('P(15:30 close > VWAP)  (%)')
        ax.set_title('引けがVWAPより上に終わる確率', fontweight='bold')
        ax.grid(alpha=0.3, axis='x')

    # (1,*) Reversion / Trend / Breakout / Slope 最良 Sharpe 銘柄別
    def per_sym_best(df_a, ax, title):
        if df_a is None or df_a.empty:
            ax.set_title(title + ' (no data)'); return
        best = df_a[df_a['N'] >= 25].groupby('sym')['sharpe'].max().sort_values(ascending=False)
        colors = ['#2ca02c' if v >= 2 else ('#ff7f0e' if v >= 1 else '#aaaaaa') for v in best.values]
        ax.barh(range(len(best)), best.values, color=colors)
        ax.set_yticks(range(len(best))); ax.set_yticklabels(best.index, fontsize=7)
        ax.axvline(2, color='red', ls='--', alpha=0.6, label='Sharpe=2 合格線')
        ax.set_xlabel('Sharpe (銘柄別最良)')
        ax.set_title(title, fontweight='bold')
        ax.grid(alpha=0.3, axis='x')

    per_sym_best(rev_df, fig.add_subplot(gs[1, 0]), 'Reversion 最良 Sharpe/銘柄')
    per_sym_best(tr_df, fig.add_subplot(gs[1, 1]), 'Trend 最良 Sharpe/銘柄')
    per_sym_best(bo_df, fig.add_subplot(gs[1, 2]), 'Breakout 最良 Sharpe/銘柄')

    # (2,0) Slope 最良
    per_sym_best(slope_df, fig.add_subplot(gs[2, 0]), 'Slope 最良 Sharpe/銘柄')

    # (2,1) Attraction th=80
    ax = fig.add_subplot(gs[2, 1])
    sub = attract_df[attract_df['th'] == 80].set_index('sym')
    if len(sub):
        xs = np.arange(len(sub))
        w = 0.35
        ax.bar(xs - w / 2, sub['up_returnP'].fillna(0).values, w, label='上→戻り', color='#d62728')
        ax.bar(xs + w / 2, sub['dn_returnP'].fillna(0).values, w, label='下→戻り', color='#1f77b4')
        ax.set_xticks(xs); ax.set_xticklabels(sub.index, rotation=90, fontsize=7)
        ax.axhline(50, color='gray', ls='--', alpha=0.5)
        ax.set_ylabel('戻り率 (%)'); ax.set_title('Attraction: |dev|>=80bps', fontweight='bold')
        ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')

    # (2,2) First cross median time
    ax = fig.add_subplot(gs[2, 2])
    if len(desc_df):
        sub = desc_df.set_index('sym').sort_values('med_first_cross', ascending=True)
        ax.barh(range(len(sub)), sub['med_first_cross'].values, color='#9467bd')
        ax.set_yticks(range(len(sub))); ax.set_yticklabels(sub.index, fontsize=7)
        ax.set_xlabel('中央値 (9:00からの分)')
        ax.set_title('VWAP初交差時刻の中央値', fontweight='bold')
        ax.grid(alpha=0.3, axis='x')

    # (3,0-2) Top3銘柄の累積損益 (分類ベース)
    if len(cls_df) > 0:
        top3 = cls_df.head(3).to_dict('records')
        for i, rec in enumerate(top3):
            ax = fig.add_subplot(gs[3, i])
            sym = rec['sym']; kind = rec['best_type']
            if kind == 'Reversion':
                ss = rev_df[(rev_df['sym'] == sym) & (rev_df['N'] >= 25)].sort_values('sharpe', ascending=False)
                if len(ss) == 0:
                    continue
                r = ss.iloc[0]
                f = feats[sym]; dev_c = f'dev_{r["time"]}'; ret_c = f'ret_to_eod_{r["time"]}'
                sub = f[[dev_c, ret_c]].dropna()
                d = np.where(sub[dev_c] >= r['th'], -1, np.where(sub[dev_c] <= -r['th'], +1, 0))
                mask = d != 0
                net = d[mask] * sub[ret_c].values[mask] - U.COST_BPS
                idx = sub.index[mask]
                cum = pd.Series(net, index=idx).sort_index().cumsum()
            elif kind == 'Trend':
                ss = tr_df[(tr_df['sym'] == sym) & (tr_df['N'] >= 25)].sort_values('sharpe', ascending=False)
                if len(ss) == 0:
                    continue
                r = ss.iloc[0]
                f = feats[sym]; dev_c = f'dev_{r["time"]}'; ret_c = f'ret_to_eod_{r["time"]}'
                sub = f[[dev_c, ret_c]].dropna()
                d = np.where(sub[dev_c] >= r['th'], +1, np.where(sub[dev_c] <= -r['th'], -1, 0))
                mask = d != 0
                net = d[mask] * sub[ret_c].values[mask] - U.COST_BPS
                idx = sub.index[mask]
                cum = pd.Series(net, index=idx).sort_index().cumsum()
            elif kind == 'Breakout':
                ss = bo_df[(bo_df['sym'] == sym) & (bo_df['N'] >= 25)].sort_values('sharpe', ascending=False)
                if len(ss) == 0:
                    continue
                r = ss.iloc[0]
                frames = day_frames_all[sym]
                tr_res = backtest_breakout(frames, min_persistence=r['persist'],
                                           hold_min=r['hold'], direction=r['break_dir'])
                net = tr_res['gross_bps'].values - U.COST_BPS
                cum = pd.Series(net).cumsum()
            else:
                continue
            ax.plot(range(len(cum)), cum.values if hasattr(cum, 'values') else cum, color='#1f77b4', lw=1.3)
            ax.axhline(0, color='gray', lw=0.8)
            ax.set_title(f"{sym} / {kind}\nSh={rec['best_sharpe']:+.2f}, N={int(rec['best_N'])}",
                         fontsize=10, fontweight='bold')
            ax.set_xlabel('トレード順'); ax.set_ylabel('累積 bps')
            ax.grid(alpha=0.3)

    plt.suptitle('VWAP 徹底分析 — 全22銘柄 × 多角的検証', fontsize=16, fontweight='bold', y=1.00)
    outp = os.path.join(os.path.dirname(__file__), 'result.png')
    plt.savefig(outp, dpi=120, bbox_inches='tight')
    print(f"\nSaved: {outp}")


if __name__ == "__main__":
    main()
