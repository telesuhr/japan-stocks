"""
VWAP ベースのエントリー方向判定 - 徹底検証

仮説:
  (A) VWAP reversion: 価格がVWAPから大きく乖離したら戻り方向へエントリー (逆張り)
  (B) VWAP trend:     価格がVWAPの上/下で持続していたら継続方向へエントリー (順張り)
  (C) VWAP slope:     VWAP自体の傾きを方向シグナルとして使う

対象: 非鉄3 + 半導体5 = 8銘柄
検証ポイント: 9:30, 10:00, 11:00, 11:30, 13:30, 14:00 でのVWAP乖離
ターゲット: その時刻 → 15:30 大引けまでのリターン
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '20260421_common')))
import mdutil as U
import pandas as pd
import numpy as np
plt = U.matplotlib_jp()

SYMS = U.NONFERROUS + U.SEMICON

# 決定時刻 (9:00=0分)
DECISION_TIMES_MO = {
    '9:30': 30, '10:00': 60, '11:00': 120,
    '11:30': 150, '13:30': 270, '14:00': 300,
}


def load_1min(sym):
    df = U.fetch_intraday(sym).dropna(subset=['open','high','low','close'])
    h, m = df.index.hour, df.index.minute
    mask = (((h==9)) | ((h==10)) | ((h==11)&(m<=30)) |
            ((h==12)&(m>=30)) | ((h==13)|(h==14)) | ((h==15)&(m<=30)))
    df = df[mask].copy()
    df['mo'] = (df.index.hour - 9) * 60 + df.index.minute
    return df


def compute_session_vwap(day):
    """session VWAP: cumulative (typical_price * volume) / cumulative volume"""
    tp = (day['high'] + day['low'] + day['close']) / 3
    pv = tp * day['volume'].fillna(0)
    cum_pv = pv.cumsum()
    cum_v = day['volume'].fillna(0).cumsum()
    vwap = cum_pv / cum_v.replace(0, np.nan)
    # volume=0で最初のsetpを埋める
    vwap = vwap.ffill().fillna(tp)
    return vwap


def build_daily_features(df):
    """日別にVWAP特徴量を計算。各決定時刻と決済時刻のリターンを記録"""
    rows = []
    dates = sorted(set(df.index.date))
    for d in dates:
        day = df[df.index.date == d].copy()
        if len(day) < 200: continue
        day['vwap'] = compute_session_vwap(day)
        # 決済時刻の close (15:30最後)
        eod_close = day['close'].iloc[-1]
        open_p = day['open'].iloc[0]
        row = {'date': d, 'open': open_p, 'eod_close': eod_close,
               'eod_ret_bps': (eod_close/open_p-1)*10000}
        for tname, tmo in DECISION_TIMES_MO.items():
            at = day[day['mo'] == tmo]
            if len(at) == 0:
                # 最も近い (>=tmo)
                at = day[day['mo'] >= tmo].head(1)
            if len(at) == 0: continue
            cl = at['close'].iloc[0]
            vw = at['vwap'].iloc[0]
            # VWAP乖離 (bps)
            row[f'dev_{tname}'] = (cl/vw - 1) * 10000
            row[f'close_{tname}'] = cl
            row[f'vwap_{tname}'] = vw
            # その時刻→15:30のリターン
            row[f'ret_to_eod_{tname}'] = (eod_close/cl - 1) * 10000
            # VWAPの傾き (直近5分間のVWAP変化)
            past = day[(day['mo'] <= tmo) & (day['mo'] > tmo-5)]
            if len(past) >= 3:
                row[f'vwap_slope_{tname}'] = (past['vwap'].iloc[-1] - past['vwap'].iloc[0])
            else:
                row[f'vwap_slope_{tname}'] = 0
        rows.append(row)
    return pd.DataFrame(rows).set_index('date')


def analyze_dev_vs_rest(feat, tname, sym):
    """VWAP乖離 vs その後のリターンの関係"""
    dev_col = f'dev_{tname}'
    ret_col = f'ret_to_eod_{tname}'
    if dev_col not in feat.columns: return None
    sub = feat[[dev_col, ret_col]].dropna()
    if len(sub) < 30: return None
    # 全体相関
    corr = sub[dev_col].corr(sub[ret_col])
    # 分位数別
    qs = pd.qcut(sub[dev_col], 5, duplicates='drop')
    q_mean = sub.groupby(qs, observed=True)[ret_col].agg(['mean','count'])
    # 両サイド threshold
    thresholds = [20, 30, 50, 80]
    th_results = []
    for th in thresholds:
        up = sub[sub[dev_col] > th]
        dn = sub[sub[dev_col] < -th]
        th_results.append({
            'th': th,
            'up_N': len(up), 'up_mean_rest': up[ret_col].mean() if len(up)>0 else 0,
            'up_wr': (up[ret_col]>0).mean()*100 if len(up)>0 else 0,
            'dn_N': len(dn), 'dn_mean_rest': dn[ret_col].mean() if len(dn)>0 else 0,
            'dn_wr': (dn[ret_col]<0).mean()*100 if len(dn)>0 else 0,  # short勝率なので<0
        })
    return {'sym': sym, 'time': tname, 'corr': corr, 'N': len(sub),
            'thresholds': th_results}


def backtest_vwap_reversion(feat, tname, th_bps, cost=U.COST_BPS):
    """
    VWAP reversion: |dev| >= th の日、符号反対にエントリー、15:30決済
    """
    dev_col = f'dev_{tname}'
    ret_col = f'ret_to_eod_{tname}'
    if dev_col not in feat.columns: return None
    sub = feat[[dev_col, ret_col]].dropna().copy()
    sub['direction'] = np.where(sub[dev_col] >= th_bps, -1,
                                 np.where(sub[dev_col] <= -th_bps, +1, 0))
    sub = sub[sub['direction'] != 0].copy()
    if len(sub) < 10: return None
    sub['net_bps'] = sub['direction'] * sub[ret_col] - cost
    st = U.compute_stats(sub['net_bps'].values)
    st['N_up_entries'] = (sub['direction']==-1).sum()  # Short (上乖離→下戻り)
    st['N_dn_entries'] = (sub['direction']==+1).sum()  # Long (下乖離→上戻り)
    return sub, st


def backtest_vwap_trend(feat, tname, th_bps, cost=U.COST_BPS):
    """
    VWAP trend: |dev| >= th の日、符号と同方向にエントリー (順張り)
    """
    dev_col = f'dev_{tname}'
    ret_col = f'ret_to_eod_{tname}'
    if dev_col not in feat.columns: return None
    sub = feat[[dev_col, ret_col]].dropna().copy()
    sub['direction'] = np.where(sub[dev_col] >= th_bps, +1,
                                 np.where(sub[dev_col] <= -th_bps, -1, 0))
    sub = sub[sub['direction'] != 0].copy()
    if len(sub) < 10: return None
    sub['net_bps'] = sub['direction'] * sub[ret_col] - cost
    st = U.compute_stats(sub['net_bps'].values)
    return sub, st


def main():
    print("=== VWAP ベース方向判定 徹底検証 ===\n")
    print(f"対象: {len(SYMS)}銘柄, コスト={U.COST_BPS}bps\n")

    # 各銘柄の日別特徴量
    feats = {}
    for sym, name in SYMS:
        df = load_1min(sym)
        feat = build_daily_features(df)
        feats[sym] = feat
        print(f"  {sym} {name}: {len(feat)} 日分の特徴量")

    # ========== 1. VWAP乖離と残り時間リターンの関係 ==========
    print("\n=== 1. VWAP乖離 → 残り時間リターンの関係 (相関 by 銘柄×時刻) ===")
    print(f"{'sym':<8} {'time':<6} {'N':>5} {'corr':>8} {'dev_std':>8}")
    corr_table = []
    for sym, _ in SYMS:
        feat = feats[sym]
        for tname in DECISION_TIMES_MO.keys():
            dev_col = f'dev_{tname}'
            ret_col = f'ret_to_eod_{tname}'
            if dev_col not in feat.columns: continue
            sub = feat[[dev_col, ret_col]].dropna()
            if len(sub) < 30: continue
            c = sub[dev_col].corr(sub[ret_col])
            sd = sub[dev_col].std()
            corr_table.append({'sym': sym, 'time': tname, 'N': len(sub),
                               'corr': c, 'dev_std': sd})
    corr_df = pd.DataFrame(corr_table)
    # ピボット: 銘柄×時刻
    pivot_c = corr_df.pivot(index='sym', columns='time', values='corr')
    pivot_c = pivot_c[list(DECISION_TIMES_MO.keys())]
    print("\n相関ピボット (負=reversion, 正=trend):")
    print(pivot_c.to_string(float_format='{:+.3f}'.format))
    pivot_s = corr_df.pivot(index='sym', columns='time', values='dev_std')
    pivot_s = pivot_s[list(DECISION_TIMES_MO.keys())]
    print("\n乖離幅 std (bps):")
    print(pivot_s.to_string(float_format='{:.1f}'.format))

    # ========== 2. 閾値別 reversion バックテスト ==========
    print("\n=== 2. VWAP Reversion (逆張り) バックテスト ===")
    rev_results = []
    for sym, _ in SYMS:
        for tname in DECISION_TIMES_MO.keys():
            for th in [20, 30, 50, 80, 120]:
                out = backtest_vwap_reversion(feats[sym], tname, th)
                if out is None: continue
                sub, st = out
                if st['n'] < 15: continue
                rev_results.append({
                    'sym': sym, 'time': tname, 'th': th,
                    'N': st['n'], 'mean': st['mean'], 'wr': st['wr'],
                    'pf': st['pf'], 'sharpe': st['sharpe'], 't': st['t_stat']
                })
    rev_df = pd.DataFrame(rev_results)
    if not rev_df.empty:
        print("\nTop15 (Sharpe降順):")
        top = rev_df.sort_values('sharpe', ascending=False).head(15)
        print(top.to_string(index=False,
              formatters={'mean':'{:+.1f}'.format,'wr':'{:.1f}'.format,
                          'pf':'{:.2f}'.format,'sharpe':'{:+.2f}'.format,
                          't':'{:+.2f}'.format}))

    # ========== 3. 閾値別 trend バックテスト ==========
    print("\n=== 3. VWAP Trend (順張り) バックテスト ===")
    trend_results = []
    for sym, _ in SYMS:
        for tname in DECISION_TIMES_MO.keys():
            for th in [20, 30, 50, 80, 120]:
                out = backtest_vwap_trend(feats[sym], tname, th)
                if out is None: continue
                sub, st = out
                if st['n'] < 15: continue
                trend_results.append({
                    'sym': sym, 'time': tname, 'th': th,
                    'N': st['n'], 'mean': st['mean'], 'wr': st['wr'],
                    'pf': st['pf'], 'sharpe': st['sharpe'], 't': st['t_stat']
                })
    tr_df = pd.DataFrame(trend_results)
    if not tr_df.empty:
        print("\nTop15 (Sharpe降順):")
        top = tr_df.sort_values('sharpe', ascending=False).head(15)
        print(top.to_string(index=False,
              formatters={'mean':'{:+.1f}'.format,'wr':'{:.1f}'.format,
                          'pf':'{:.2f}'.format,'sharpe':'{:+.2f}'.format,
                          't':'{:+.2f}'.format}))

    # ========== 4. 全銘柄集約 (プール) ==========
    print("\n=== 4. 全8銘柄プール Reversion/Trend ===")
    for tname in DECISION_TIMES_MO.keys():
        pool_rows = []
        for sym, _ in SYMS:
            feat = feats[sym]
            dev_col = f'dev_{tname}'
            ret_col = f'ret_to_eod_{tname}'
            if dev_col not in feat.columns: continue
            sub = feat[[dev_col, ret_col]].dropna().copy()
            sub['sym'] = sym
            pool_rows.append(sub)
        if not pool_rows: continue
        pool = pd.concat(pool_rows)
        c = pool[dev_col].corr(pool[ret_col])
        # reversion at |th|=50
        rev = pool.copy()
        rev['direction'] = np.where(rev[dev_col] >= 50, -1,
                                     np.where(rev[dev_col] <= -50, +1, 0))
        rev = rev[rev['direction'] != 0]
        rev['net'] = rev['direction']*rev[ret_col] - U.COST_BPS
        st_rev = U.compute_stats(rev['net'].values) if len(rev)>=10 else None
        # trend at |th|=50
        tr = pool.copy()
        tr['direction'] = np.where(tr[dev_col] >= 50, +1,
                                    np.where(tr[dev_col] <= -50, -1, 0))
        tr = tr[tr['direction'] != 0]
        tr['net'] = tr['direction']*tr[ret_col] - U.COST_BPS
        st_tr = U.compute_stats(tr['net'].values) if len(tr)>=10 else None
        print(f"{tname:>5}: corr={c:+.3f}  "
              f"Rev(th=50): N={st_rev['n'] if st_rev else 0} "
              f"Sh={st_rev['sharpe']:+.2f} " if st_rev else f"{tname:>5}: corr={c:+.3f}  Rev: N/A  ", end='')
        if st_tr:
            print(f"Trend(th=50): N={st_tr['n']} Sh={st_tr['sharpe']:+.2f}")
        else:
            print("Trend: N/A")

    # ========== 5. 合格戦略のリスト ==========
    print("\n=== 5. 合格戦略 (Sharpe>=2, N>=30, t>=2) ===")
    winners_rev = rev_df[(rev_df['sharpe']>=2.0)&(rev_df['N']>=30)&(rev_df['t']>=2.0)]
    winners_tr = tr_df[(tr_df['sharpe']>=2.0)&(tr_df['N']>=30)&(tr_df['t']>=2.0)]
    print(f"Reversion: {len(winners_rev)}件")
    if len(winners_rev) > 0:
        print(winners_rev.sort_values('sharpe', ascending=False).to_string(index=False,
              formatters={'mean':'{:+.1f}'.format,'wr':'{:.1f}'.format,
                          'pf':'{:.2f}'.format,'sharpe':'{:+.2f}'.format,
                          't':'{:+.2f}'.format}))
    print(f"\nTrend: {len(winners_tr)}件")
    if len(winners_tr) > 0:
        print(winners_tr.sort_values('sharpe', ascending=False).to_string(index=False,
              formatters={'mean':'{:+.1f}'.format,'wr':'{:.1f}'.format,
                          'pf':'{:.2f}'.format,'sharpe':'{:+.2f}'.format,
                          't':'{:+.2f}'.format}))

    # ========== 可視化 ==========
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.5, wspace=0.35)

    # (0,0) 銘柄×時刻 相関ヒートマップ
    ax = fig.add_subplot(gs[0,0])
    im = ax.imshow(pivot_c.values, cmap='RdBu_r', aspect='auto', vmin=-0.3, vmax=0.3)
    ax.set_xticks(range(len(pivot_c.columns))); ax.set_xticklabels(pivot_c.columns, rotation=30)
    ax.set_yticks(range(len(pivot_c.index))); ax.set_yticklabels(pivot_c.index)
    ax.set_title('VWAP乖離 vs 残り時間リターン 相関', fontweight='bold')
    for i in range(pivot_c.shape[0]):
        for j in range(pivot_c.shape[1]):
            v = pivot_c.values[i,j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.2f}", ha='center', va='center', fontsize=8,
                        color='white' if abs(v)>0.15 else 'black')
    fig.colorbar(im, ax=ax, shrink=0.8)

    # (0,1) 乖離幅 std
    ax = fig.add_subplot(gs[0,1])
    im = ax.imshow(pivot_s.values, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(len(pivot_s.columns))); ax.set_xticklabels(pivot_s.columns, rotation=30)
    ax.set_yticks(range(len(pivot_s.index))); ax.set_yticklabels(pivot_s.index)
    ax.set_title('VWAP乖離 std (bps)', fontweight='bold')
    for i in range(pivot_s.shape[0]):
        for j in range(pivot_s.shape[1]):
            v = pivot_s.values[i,j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}", ha='center', va='center', fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)

    # (0,2) 時刻別 Reversion@th=50 プール Sharpe
    ax = fig.add_subplot(gs[0,2])
    tnames = list(DECISION_TIMES_MO.keys())
    rev_sh = []; tr_sh = []
    for tname in tnames:
        pool_rows = []
        for sym, _ in SYMS:
            feat = feats[sym]
            dev_col = f'dev_{tname}'
            ret_col = f'ret_to_eod_{tname}'
            if dev_col not in feat.columns: continue
            pool_rows.append(feat[[dev_col, ret_col]].dropna())
        if not pool_rows:
            rev_sh.append(0); tr_sh.append(0); continue
        pool = pd.concat(pool_rows)
        for mode, arr in [('rev',rev_sh),('tr',tr_sh)]:
            sign_coef = -1 if mode=='rev' else +1
            x = pool.copy()
            x['dir'] = np.where(x[dev_col] >= 50, sign_coef,
                                  np.where(x[dev_col] <= -50, -sign_coef, 0))
            x = x[x['dir']!=0]
            if len(x)<10:
                arr.append(0); continue
            net = x['dir']*x[ret_col] - U.COST_BPS
            st = U.compute_stats(net.values)
            arr.append(st['sharpe'] if st else 0)
    xs = np.arange(len(tnames)); w=0.35
    ax.bar(xs-w/2, rev_sh, w, label='Reversion', color='#1f77b4', edgecolor='black')
    ax.bar(xs+w/2, tr_sh, w, label='Trend', color='#d62728', edgecolor='black')
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xticks(xs); ax.set_xticklabels(tnames, rotation=30)
    ax.set_ylabel('Sharpe (pool, th=50, cost=4bps)')
    ax.set_title('VWAP戦略 時刻別 (全銘柄プール)', fontweight='bold')
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')

    # (1,0-2) 銘柄別ベスト戦略の累積損益
    best_list = []
    for sym, _ in SYMS:
        sub_rev = rev_df[rev_df['sym']==sym]
        sub_tr = tr_df[tr_df['sym']==sym]
        all_sub = pd.concat([sub_rev.assign(mode='rev'), sub_tr.assign(mode='tr')], ignore_index=True)
        if all_sub.empty: continue
        # N>=25の中でSharpe最大
        cand = all_sub[all_sub['N']>=25]
        if cand.empty: cand = all_sub
        best = cand.sort_values('sharpe', ascending=False).iloc[0]
        best_list.append((sym, best))

    for i, (sym, best) in enumerate(best_list[:6]):
        r = 1 + i // 3; c = i % 3
        ax = fig.add_subplot(gs[r, c])
        # 該当戦略を再実行
        if best['mode'] == 'rev':
            out = backtest_vwap_reversion(feats[sym], best['time'], best['th'])
        else:
            out = backtest_vwap_trend(feats[sym], best['time'], best['th'])
        if out is None: continue
        sub, st = out
        sub = sub.sort_index()
        cum = sub['net_bps'].cumsum()
        ax.plot(pd.to_datetime(sub.index), cum.values, color='#1f77b4', lw=1.3)
        ax.axhline(0, color='gray', lw=0.8)
        ax.set_title(f"{sym} best: {best['mode']}, {best['time']}, th={best['th']}\n"
                     f"N={st['n']}, Sh={st['sharpe']:+.2f}, WR={st['wr']:.0f}%",
                     fontsize=9, fontweight='bold')
        ax.tick_params(axis='x', labelrotation=30, labelsize=7)
        ax.set_ylabel('累積 bps')
        ax.grid(alpha=0.3)

    plt.suptitle('VWAP ベース方向判定 徹底検証', fontsize=15, fontweight='bold', y=1.00)
    out = os.path.join(os.path.dirname(__file__), 'result.png')
    plt.savefig(out, dpi=130, bbox_inches='tight')
    print(f"\nSaved: {out}")

    # CSV
    rev_df.to_csv(os.path.join(os.path.dirname(__file__), 'reversion_grid.csv'), index=False)
    tr_df.to_csv(os.path.join(os.path.dirname(__file__), 'trend_grid.csv'), index=False)
    corr_df.to_csv(os.path.join(os.path.dirname(__file__), 'corr_table.csv'), index=False)
    print("CSV: reversion_grid.csv, trend_grid.csv, corr_table.csv")


if __name__ == "__main__":
    main()
