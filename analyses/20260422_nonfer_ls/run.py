"""
非鉄金属 3銘柄のロングショート戦略検証

銘柄: 5711 三菱マテリアル / 5706 三井金属 / 5713 住友金属鉱山
検証: 日次 + イントラデイ両方

1. 相関・リードラグ (イントラデイ 1分足)
2. ペアスプレッドのZスコア平均回帰 (日次 / イントラデイ)
3. ランキングLS (前日までのN日リターンで最弱Long-最強Short)
4. セクター対指数LS (非鉄個別 vs 非鉄バスケット平均)
5. 寄付ダイバージェンス (9:00寄付時点の乖離 → 日中で収束)

コスト: 片側2bps × 往復 × 2銘柄 = 8bps/トレード
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '20260421_common')))
import mdutil as U
import pandas as pd
import numpy as np
from datetime import time as dtime
from itertools import combinations
plt = U.matplotlib_jp()

SYMS = [('5711.T', '三菱マテリアル'), ('5706.T', '三井金属'), ('5713.T', '住友金属鉱山')]
PAIRS = list(combinations([s for s,_ in SYMS], 2))
COST_LS = 8.0  # bps/トレード (LS=2銘柄×往復)


def load_1min(sym):
    df = U.fetch_intraday(sym).dropna(subset=['open','close'])
    return df


def load_daily(sym):
    jp = U.load_jp_daily(sym)
    jp['ret'] = jp['close'].pct_change() * 100
    return jp


def morning_1min_returns(df, d):
    """当日 9:00 を0として、分単位のリターン系列"""
    day = df[df.index.date == d]
    if len(day) == 0: return None
    morning = day[((day.index.hour==9)) | ((day.index.hour==10)) | ((day.index.hour==11)&(day.index.minute<=30))]
    if len(morning) < 30: return None
    p0 = morning['open'].iloc[0]
    morning = morning.copy()
    morning['ret_pct'] = (morning['close']/p0 - 1)*100
    return morning[['ret_pct']]


def compute_stats_ls(arr):
    arr = np.asarray(arr)
    if len(arr) == 0: return None
    m, s = arr.mean(), arr.std()
    cum = arr.cumsum()
    dd = (cum - np.maximum.accumulate(cum)).min()
    pos = arr[arr > 0].sum(); neg = abs(arr[arr <= 0].sum())
    return {
        'n': len(arr), 'mean': m, 'std': s, 'total': arr.sum(),
        'wr': (arr>0).mean()*100,
        'pf': pos/neg if neg > 0 else np.inf,
        'sharpe': m/s*np.sqrt(252) if s > 0 else 0,
        'maxdd': dd,
        't': m/(s/np.sqrt(len(arr))) if s > 0 else 0,
    }


def main():
    # ========== データロード ==========
    print("=== 非鉄金属 3銘柄 LS戦略検証 ===\n")
    daily = {sym: load_daily(sym) for sym, _ in SYMS}
    print("Daily data:")
    for sym, _ in SYMS:
        print(f"  {sym}: N={len(daily[sym])}  range {daily[sym].index[0]} to {daily[sym].index[-1]}")

    # 日次 close リターン DataFrame
    close_df = pd.DataFrame({sym: daily[sym]['close'] for sym, _ in SYMS})
    ret_df = close_df.pct_change() * 100
    ret_df = ret_df.dropna()

    # ========== 1. 日次相関 ==========
    print("\n=== 1. 日次リターン相関 ===")
    corr = ret_df.corr()
    print(corr.round(3))

    # ========== 2. ペアスプレッド Z-score 平均回帰 (日次) ==========
    print("\n=== 2. 日次ペアスプレッド Zスコア平均回帰LS ===")
    print("ルール: Z>+Z_in でA Short/B Long、Z<-Z_in で A Long/B Short、|Z|<0.5で決済 (保有日数上限10日)")

    pair_results = {}
    for a, b in PAIRS:
        spread = ret_df[a] - ret_df[b]  # 日次リターン差
        # 累積スプレッド (価格比)
        log_ratio = np.log(close_df[a] / close_df[b])
        log_ratio = log_ratio.dropna()
        for window in [20, 40]:
            for z_in in [1.5, 2.0]:
                mu = log_ratio.rolling(window).mean()
                sd = log_ratio.rolling(window).std()
                z = (log_ratio - mu) / sd
                # トレード生成
                trades = []
                pos = 0  # +1: AShort/BLong, -1: ALong/BShort
                entry_idx = None; entry_z = None
                for i, dt in enumerate(log_ratio.index):
                    zv = z.iloc[i]
                    if pd.isna(zv): continue
                    if pos == 0:
                        if zv > z_in:
                            pos = +1; entry_idx = i; entry_z = zv
                        elif zv < -z_in:
                            pos = -1; entry_idx = i; entry_z = zv
                    else:
                        held = i - entry_idx
                        if abs(zv) < 0.5 or held >= 10:
                            # 決済: スプレッド変化
                            ret_a = (close_df[a].iloc[i]/close_df[a].iloc[entry_idx]-1)*100
                            ret_b = (close_df[b].iloc[i]/close_df[b].iloc[entry_idx]-1)*100
                            # pos=+1: A Short / B Long → pnl = -ret_a + ret_b
                            pnl = (-pos) * (ret_a - ret_b) * 100 - COST_LS  # bps
                            trades.append({'exit': dt, 'z_in': entry_z, 'z_out': zv, 'held': held, 'pnl_bps': pnl})
                            pos = 0
                if len(trades) == 0: continue
                tdf = pd.DataFrame(trades)
                st = compute_stats_ls(tdf['pnl_bps'].values)
                pair_results[(a, b, window, z_in)] = {'df': tdf, 'stats': st}
                if st['n'] >= 3:
                    print(f"  {a[:5]}-{b[:5]} W={window:>2} Z={z_in}: N={st['n']:>3} Mean={st['mean']:>+6.1f} "
                          f"WR={st['wr']:>5.1f}% Shp={st['sharpe']:>+5.2f} PF={st['pf']:>4.2f} MaxDD={st['maxdd']:>+6.0f}")

    # ========== 3. ランキングLS (日次) ==========
    print("\n=== 3. 前日N日リターン順位によるLS (最弱Long / 最強Short) ===")
    ranking_results = {}
    for lb in [5, 10, 20]:
        # 過去lb日のリターン合計で順位
        past = ret_df.rolling(lb).sum()
        pnl_list = []
        for i in range(lb, len(ret_df)-1):
            ranks = past.iloc[i].rank()  # 1=最弱, 3=最強
            weakest = ranks.idxmin()
            strongest = ranks.idxmax()
            # 翌日リターン: Long weakest - Short strongest
            next_w = ret_df[weakest].iloc[i+1]
            next_s = ret_df[strongest].iloc[i+1]
            pnl_list.append((next_w - next_s) * 100 - COST_LS)
        st = compute_stats_ls(pnl_list)
        ranking_results[lb] = st
        if st:
            print(f"  lookback={lb:>2}d: N={st['n']:>4} Mean={st['mean']:>+6.1f} "
                  f"WR={st['wr']:>5.1f}% Shp={st['sharpe']:>+5.2f} PF={st['pf']:>4.2f} MaxDD={st['maxdd']:>+7.0f}")

    # 逆ランキング (最強Long/最弱Short = モメンタム)
    print("\n=== 3b. ランキングLS モメンタム版 (最強Long / 最弱Short) ===")
    ranking_mom_results = {}
    for lb in [5, 10, 20]:
        past = ret_df.rolling(lb).sum()
        pnl_list = []
        for i in range(lb, len(ret_df)-1):
            ranks = past.iloc[i].rank()
            weakest = ranks.idxmin()
            strongest = ranks.idxmax()
            next_w = ret_df[weakest].iloc[i+1]
            next_s = ret_df[strongest].iloc[i+1]
            pnl_list.append((next_s - next_w) * 100 - COST_LS)
        st = compute_stats_ls(pnl_list)
        ranking_mom_results[lb] = st
        if st:
            print(f"  lookback={lb:>2}d: N={st['n']:>4} Mean={st['mean']:>+6.1f} "
                  f"WR={st['wr']:>5.1f}% Shp={st['sharpe']:>+5.2f} PF={st['pf']:>4.2f} MaxDD={st['maxdd']:>+7.0f}")

    # ========== 4. イントラデイ 1分足 リードラグ ==========
    print("\n=== 4. イントラデイ 1分足 リードラグ相関 ===")
    print("LAG k: Aの時刻 t のリターン vs Bの時刻 t+k のリターン")
    intra = {}
    for sym, _ in SYMS:
        df = load_1min(sym)
        # 前場のみ(9:00-11:30), 後場(12:30-15:30)
        h, m = df.index.hour, df.index.minute
        mask = ((h==9)&(m>=0)) | ((h==10)) | ((h==11)&(m<=30)) | ((h==12)&(m>=30)) | ((h==13)|(h==14)) | ((h==15)&(m<=30))
        df = df[mask].copy()
        df['ret_1m'] = df['close'].pct_change() * 10000  # bps
        intra[sym] = df

    # 同時相関
    print("\n  同時 1分足リターン相関:")
    aligned = pd.DataFrame({sym: intra[sym]['ret_1m'] for sym, _ in SYMS})
    aligned = aligned.dropna()
    print(aligned.corr().round(3))

    # リードラグ (5711を基準に)
    print("\n  リードラグ相関 (行=他銘柄 vs 5711.T の lag):")
    print(f"  {'pair':<20} {'lag-3':>7} {'lag-2':>7} {'lag-1':>7} {'lag=0':>7} {'lag+1':>7} {'lag+2':>7} {'lag+3':>7}")
    base = '5711.T'
    lag_results = {}
    for sym, _ in SYMS:
        if sym == base: continue
        row = []
        for lag in range(-3, 4):
            if lag < 0:
                c = aligned[sym].corr(aligned[base].shift(-lag))
            else:
                c = aligned[sym].shift(lag).corr(aligned[base])
            row.append(c)
        lag_results[sym] = row
        print(f"  {sym} vs {base}: {row[0]:+7.3f} {row[1]:+7.3f} {row[2]:+7.3f} {row[3]:+7.3f} {row[4]:+7.3f} {row[5]:+7.3f} {row[6]:+7.3f}")

    # ========== 5. イントラデイ ペアZスコア (1分足) ==========
    print("\n=== 5. イントラデイ 1分足 ペアZスコア スキャルピングLS ===")
    print("ルール: 各日 9:00寄付からのリターン差Zスコア |Z|>Z_in で逆張り、|Z|<0.3で決済、引けまで全決済")

    # 日別に処理
    intra_ls_results = {}
    for a, b in PAIRS:
        all_trades = []
        common_dates = sorted(set(intra[a].index.date) & set(intra[b].index.date))
        for d in common_dates:
            if d.weekday() >= 5: continue
            da = morning_1min_returns(intra[a], d)
            db = morning_1min_returns(intra[b], d)
            if da is None or db is None: continue
            # 時刻で揃える
            df = da.join(db, how='inner', lsuffix='_a', rsuffix='_b').dropna()
            if len(df) < 60: continue
            df['spread'] = df['ret_pct_a'] - df['ret_pct_b']  # 寄付比 差分 (%)
            # 当日内のローリング (30分=30)
            mu = df['spread'].rolling(30).mean()
            sd = df['spread'].rolling(30).std()
            df['z'] = (df['spread'] - mu) / sd
            pos = 0; entry_idx = None
            for i in range(30, len(df)):
                zv = df['z'].iloc[i]
                if pd.isna(zv): continue
                if pos == 0:
                    if zv > 2.0:
                        pos = +1; entry_idx = i   # A Short / B Long
                    elif zv < -2.0:
                        pos = -1; entry_idx = i
                else:
                    if abs(zv) < 0.3 or i == len(df)-1:
                        sp_e = df['spread'].iloc[entry_idx]
                        sp_x = df['spread'].iloc[i]
                        pnl = (-pos) * (sp_x - sp_e) * 100 - COST_LS  # bps
                        all_trades.append({'date': d, 'held': i-entry_idx, 'pnl_bps': pnl})
                        pos = 0
        if all_trades:
            tdf = pd.DataFrame(all_trades)
            st = compute_stats_ls(tdf['pnl_bps'].values)
            intra_ls_results[(a, b)] = {'df': tdf, 'stats': st}
            print(f"  {a[:5]}-{b[:5]}: N={st['n']:>4} Mean={st['mean']:>+6.1f} WR={st['wr']:>5.1f}% "
                  f"Shp={st['sharpe']:>+5.2f} PF={st['pf']:>4.2f} MaxDD={st['maxdd']:>+7.0f}")

    # ========== 6. 寄付ダイバージェンスLS ==========
    print("\n=== 6. 寄付ダイバージェンスLS (9:00寄付時点の相対乖離 → 収束狙い) ===")
    print("ルール: 9:00寄時点でON差が大きいペアを逆張り (大寄せたほうをShort, 小寄せたほうをLong)、15:30引で決済")

    # 各銘柄のON (前日close→当日open)
    on_rets = {}
    for sym, _ in SYMS:
        d = daily[sym].copy()
        d['prev_close'] = d['close'].shift(1)
        d['on'] = (d['open']/d['prev_close']-1)*100
        d['day'] = (d['close']/d['open']-1)*100
        on_rets[sym] = d[['on','day']]

    div_results = {}
    for a, b in PAIRS:
        ja = on_rets[a]; jb = on_rets[b]
        idx = ja.index.intersection(jb.index)
        ja = ja.loc[idx]; jb = jb.loc[idx]
        diff = ja['on'] - jb['on']  # A のON - BのON
        # 閾値: |diff|>th%
        for th in [0.5, 1.0, 1.5]:
            trades = []
            for i, dt in enumerate(idx):
                dv = diff.iloc[i]
                if pd.isna(dv) or abs(dv) < th: continue
                # A が大きく寄せた → A Short / B Long
                sign = -1 if dv > 0 else +1  # sign=+1なら A Long
                day_a = ja['day'].iloc[i]; day_b = jb['day'].iloc[i]
                pnl = sign * (day_a - day_b) * 100 - COST_LS
                trades.append({'date': dt, 'diff': dv, 'pnl_bps': pnl})
            if not trades: continue
            tdf = pd.DataFrame(trades)
            st = compute_stats_ls(tdf['pnl_bps'].values)
            div_results[(a, b, th)] = {'df': tdf, 'stats': st}
            print(f"  {a[:5]}-{b[:5]} th={th}%: N={st['n']:>4} Mean={st['mean']:>+6.1f} "
                  f"WR={st['wr']:>5.1f}% Shp={st['sharpe']:>+5.2f} PF={st['pf']:>4.2f}")

    # ========== 可視化 ==========
    fig = plt.figure(figsize=(18, 13))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.35)

    # (0,0) 日次相関ヒートマップ
    ax = fig.add_subplot(gs[0,0])
    im = ax.imshow(corr.values, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(3)); ax.set_xticklabels([s for s,_ in SYMS], rotation=30)
    ax.set_yticks(range(3)); ax.set_yticklabels([s for s,_ in SYMS])
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{corr.values[i,j]:.2f}", ha='center', va='center', fontsize=10)
    ax.set_title('日次リターン 相関', fontweight='bold')
    plt.colorbar(im, ax=ax)

    # (0,1) 日次Zスコア LS Sharpe
    ax = fig.add_subplot(gs[0,1])
    labels_z = []; sharpes_z = []
    for key, r in pair_results.items():
        a,b,w,z = key
        if r['stats']['n'] >= 5:
            labels_z.append(f"{a[:4]}-{b[:4]}\nW{w}Z{z}")
            sharpes_z.append(r['stats']['sharpe'])
    cols = ['#2ca02c' if v>0 else '#d62728' for v in sharpes_z]
    ax.barh(labels_z, sharpes_z, color=cols, edgecolor='black')
    ax.axvline(0, color='black', lw=0.8)
    ax.set_title('日次ペアZスコアLS Sharpe', fontweight='bold')
    ax.grid(alpha=0.3, axis='x')

    # (0,2) ランキングLS (逆張り vs モメンタム)
    ax = fig.add_subplot(gs[0,2])
    lbs = [5, 10, 20]
    x = np.arange(3); w = 0.35
    mr_v = [ranking_results[lb]['sharpe'] if ranking_results[lb] else 0 for lb in lbs]
    mo_v = [ranking_mom_results[lb]['sharpe'] if ranking_mom_results[lb] else 0 for lb in lbs]
    ax.bar(x-w/2, mr_v, w, label='逆張り(最弱Long)', color='#2ca02c', edgecolor='black')
    ax.bar(x+w/2, mo_v, w, label='モメンタム(最強Long)', color='#d62728', edgecolor='black')
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([f"{lb}d" for lb in lbs])
    ax.set_ylabel('Sharpe')
    ax.set_title('ランキングLS Sharpe (lookback別)', fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3, axis='y')

    # (1,0) 1分足相関
    ax = fig.add_subplot(gs[1,0])
    cc = aligned.corr()
    im = ax.imshow(cc.values, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(3)); ax.set_xticklabels([s for s,_ in SYMS], rotation=30)
    ax.set_yticks(range(3)); ax.set_yticklabels([s for s,_ in SYMS])
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cc.values[i,j]:.2f}", ha='center', va='center', fontsize=10)
    ax.set_title('1分足リターン同時相関', fontweight='bold')
    plt.colorbar(im, ax=ax)

    # (1,1) リードラグ折れ線
    ax = fig.add_subplot(gs[1,1])
    lags = list(range(-3, 4))
    for sym, row in lag_results.items():
        ax.plot(lags, row, marker='o', lw=1.5, label=f"{sym} vs 5711.T")
    ax.axvline(0, color='gray', lw=0.8, ls='--')
    ax.axhline(0, color='gray', lw=0.8)
    ax.set_xlabel('lag (min)'); ax.set_ylabel('相関')
    ax.set_title('リードラグ相関 (5711基準)', fontweight='bold')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (1,2) イントラデイZスコアLS Sharpe
    ax = fig.add_subplot(gs[1,2])
    il = []; iv = []; ist = []
    for (a,b), r in intra_ls_results.items():
        il.append(f"{a[:4]}-{b[:4]}")
        iv.append(r['stats']['sharpe'])
        ist.append(r['stats']['n'])
    cols = ['#2ca02c' if v>0 else '#d62728' for v in iv]
    ax.barh(il, iv, color=cols, edgecolor='black')
    ax.axvline(0, color='black', lw=0.8)
    for i, (v, n) in enumerate(zip(iv, ist)):
        ax.text(v, i, f" N={n}", va='center', fontsize=9)
    ax.set_title('イントラデイZスコアLS Sharpe', fontweight='bold')
    ax.grid(alpha=0.3, axis='x')

    # (2,0) 寄付ダイバージェンス Sharpe
    ax = fig.add_subplot(gs[2,0])
    dl = []; dv = []; dn = []
    for key, r in div_results.items():
        a,b,th = key
        dl.append(f"{a[:4]}-{b[:4]}\nth{th}%")
        dv.append(r['stats']['sharpe'])
        dn.append(r['stats']['n'])
    cols = ['#2ca02c' if v>0 else '#d62728' for v in dv]
    ax.barh(dl, dv, color=cols, edgecolor='black')
    ax.axvline(0, color='black', lw=0.8)
    for i, (v, n) in enumerate(zip(dv, dn)):
        ax.text(v, i, f" N={n}", va='center', fontsize=8)
    ax.set_title('寄付ダイバージェンスLS Sharpe', fontweight='bold')
    ax.grid(alpha=0.3, axis='x')

    # (2,1) 最良のペアZスコア 累積PnL
    ax = fig.add_subplot(gs[2,1])
    best_key = max(pair_results.items(), key=lambda x: x[1]['stats']['sharpe'] if x[1]['stats']['n']>=5 else -99)[0]
    best = pair_results[best_key]['df']
    a,b,w,z = best_key
    cum = best['pnl_bps'].cumsum()
    ax.plot(range(len(cum)), cum, lw=1.5, marker='o', ms=3, color='steelblue')
    ax.axhline(0, color='gray', lw=0.8)
    ax.set_title(f'最良日次Zスコア 累積PnL\n{a[:5]}-{b[:5]} W={w} Z={z}', fontweight='bold')
    ax.set_xlabel('取引#'); ax.set_ylabel('累積bps')
    ax.grid(alpha=0.3)

    # (2,2) 最良イントラデイペア 累積PnL
    ax = fig.add_subplot(gs[2,2])
    if intra_ls_results:
        best_ik = max(intra_ls_results.items(), key=lambda x: x[1]['stats']['sharpe'])[0]
        bestI = intra_ls_results[best_ik]['df']
        cum = bestI['pnl_bps'].cumsum()
        ax.plot(range(len(cum)), cum, lw=1, color='purple')
        ax.axhline(0, color='gray', lw=0.8)
        ax.set_title(f'最良イントラデイ累積PnL\n{best_ik[0][:5]}-{best_ik[1][:5]}', fontweight='bold')
        ax.set_xlabel('取引#'); ax.set_ylabel('累積bps')
        ax.grid(alpha=0.3)

    plt.suptitle('非鉄金属3銘柄 ロングショート戦略検証 (日次+イントラデイ)',
                 fontsize=15, fontweight='bold', y=1.00)
    out = os.path.join(os.path.dirname(__file__), 'result.png')
    plt.savefig(out, dpi=130, bbox_inches='tight')
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
