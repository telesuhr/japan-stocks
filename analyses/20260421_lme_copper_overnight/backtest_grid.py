"""
銘柄別 × 閾値別 バックテスト & 可視化
- 22銘柄 × 4閾値(0.5/0.8/1.0/1.5%) のSharpe/WR/Mean/Nヒートマップ
- 各銘柄の閾値別エクイティカーブ(small multiples)
"""
import psycopg2
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['Hiragino Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
import matplotlib.dates as mdates
from datetime import date, time as dtime

PG_CONFIG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}
START = "2025-04-01"
END = "2026-04-21"
OUTLIER_PCT = 15.0
COST_BPS = 4.0
THRESHOLDS = [0.5, 0.8, 1.0, 1.5]

STOCKS = [
    ("285A.T", "Kioxia"),
    ("4502.T", "武田"),
    ("6501.T", "日立"),
    ("5016.T", "出光"),
    ("5711.T", "三菱マテリアル"),
    ("7011.T", "三菱重工"),
    ("6963.T", "ローム"),
    ("4063.T", "信越化学"),
    ("5706.T", "三井金属"),
    ("1605.T", "INPEX"),
    ("6146.T", "ディスコ"),
    ("6857.T", "アドバンテスト"),
    ("8306.T", "三菱UFJ"),
    ("4503.T", "アステラス"),
    ("5713.T", "住友金属鉱山"),
    ("8035.T", "TEL"),
    ("6305.T", "日立建機"),
    ("5332.T", "TOTO"),
    ("9101.T", "日本郵船"),
    ("6098.T", "リクルート"),
    ("4578.T", "大塚HD"),
    ("6702.T", "富士通"),
]

BST = [
    (date(2024, 3, 31), date(2024, 10, 27)),
    (date(2025, 3, 30), date(2025, 10, 26)),
    (date(2026, 3, 29), date(2026, 10, 25)),
]


def is_bst(d):
    return any(s <= d < e for s, e in BST)


def load_lme():
    conn = psycopg2.connect(**PG_CONFIG)
    q = f"SELECT timestamp, open, close FROM intraday_data WHERE symbol='CMCU3' AND timestamp>='{START}' AND timestamp<'{END}' ORDER BY timestamp"
    df = pd.read_sql(q, conn); conn.close()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['jst'] = df['timestamp'] + pd.Timedelta(hours=9)
    df = df.dropna(subset=['close']).set_index('jst').sort_index()
    sig = []
    for d in sorted(set(df.index.date)):
        if d.weekday() >= 5: continue
        oh = 9 if is_bst(d) else 10
        ot = pd.Timestamp.combine(d, dtime(oh, 0))
        ct = pd.Timestamp.combine(d, dtime(15, 25))
        day = df[df.index.date == d]
        if len(day) == 0: continue
        after = day[day.index >= ot]
        before = day[day.index <= ct]
        if len(after) == 0 or len(before) == 0: continue
        ob = after.iloc[0]; cb = before.iloc[-1]
        if (ob.name - ot).total_seconds() > 1800: continue
        if (ct - cb.name).total_seconds() > 1800: continue
        sig.append({'date': d, 'move_pct': (cb['close']/ob['open']-1)*100})
    return pd.DataFrame(sig).set_index('date')


def load_jp(sym):
    conn = psycopg2.connect(**PG_CONFIG)
    q = f"SELECT timestamp, open, close FROM intraday_data WHERE symbol='{sym}' AND timestamp>='{START}' AND timestamp<'{END}' ORDER BY timestamp"
    df = pd.read_sql(q, conn); conn.close()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['jst'] = df['timestamp'] + pd.Timedelta(hours=9)
    df = df.dropna(subset=['open','close']).set_index('jst').sort_index()
    h, m = df.index.hour, df.index.minute
    df = df[((h==9)&(m<=5))|((h==15)&(m>=20)&(m<=30))]
    out = []
    for d in sorted(set(df.index.date)):
        gd = df[df.index.date == d]
        h2, m2 = gd.index.hour, gd.index.minute
        cl = gd[(h2==15)&(m2>=20)]; op = gd[(h2==9)&(m2<=5)]
        if len(cl)==0 or len(op)==0: continue
        out.append({'date': d, 'jp_close': cl['close'].iloc[-1], 'jp_open': op['open'].iloc[0]})
    return pd.DataFrame(out).set_index('date')


def backtest(sig, jp, th):
    dates = sorted(jp.index)
    rows = []
    for i, d in enumerate(dates[:-1]):
        if d not in sig.index: continue
        mv = sig.loc[d, 'move_pct']
        if mv < th: continue
        nd = dates[i+1]
        entry = jp.loc[d, 'jp_close']; exit_ = jp.loc[nd, 'jp_open']
        ret = (exit_/entry - 1) * 100
        if abs(ret) > OUTLIER_PCT: continue
        pnl = ret * 100 - COST_BPS
        rows.append({'entry': d, 'pnl_bps': pnl})
    return pd.DataFrame(rows)


def stats(tdf):
    if len(tdf) == 0: return None
    arr = tdf['pnl_bps'].values
    m = arr.mean(); s = arr.std()
    return {
        'n': len(arr),
        'mean': m,
        'total': arr.sum(),
        'wr': (arr > 0).mean() * 100,
        'sharpe': m/s*np.sqrt(252) if s > 0 else 0,
        'pf': arr[arr>0].sum()/abs(arr[arr<=0].sum()) if (arr<=0).any() and abs(arr[arr<=0].sum())>0 else np.inf,
        'maxdd': (pd.Series(arr).cumsum() - pd.Series(arr).cumsum().cummax()).min(),
    }


def main():
    print("LMEシグナルロード中...")
    sig = load_lme()
    print(f"シグナル日数: {len(sig)}")
    for th in THRESHOLDS:
        print(f"  |move|>=+{th}%: {(sig.move_pct>=th).sum()}")

    # 結果格納
    results = {}  # (sym, th) -> stats
    trades_all = {}  # (sym, th) -> tdf

    print("\nバックテスト実行中...")
    for sym, name in STOCKS:
        jp = load_jp(sym)
        for th in THRESHOLDS:
            tdf = backtest(sig, jp, th)
            results[(sym, th)] = stats(tdf)
            trades_all[(sym, th)] = tdf
        print(f"  {sym} {name}: N={len(jp)}日")

    # 行列化
    syms = [s for s, _ in STOCKS]
    labels = [f"{s}\n{n}" for s, n in STOCKS]

    def mat(key):
        M = np.full((len(syms), len(THRESHOLDS)), np.nan)
        for i, sym in enumerate(syms):
            for j, th in enumerate(THRESHOLDS):
                st = results[(sym, th)]
                if st is not None and st['n'] >= 3:
                    M[i, j] = st[key]
        return M

    M_sharpe = mat('sharpe')
    M_mean = mat('mean')
    M_wr = mat('wr')
    M_n = mat('n')

    # === Figure 1: ヒートマップ 4枚 ===
    fig, axes = plt.subplots(1, 4, figsize=(22, 10))

    def draw_heatmap(ax, M, title, fmt, cmap, vmin=None, vmax=None, center=0):
        if vmin is None:
            vmin = np.nanmin(M); vmax = np.nanmax(M)
            if center == 0:
                v = max(abs(vmin), abs(vmax))
                vmin, vmax = -v, v
        im = ax.imshow(M, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(THRESHOLDS)))
        ax.set_xticklabels([f'{t}%' for t in THRESHOLDS])
        ax.set_yticks(range(len(syms)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(title, fontsize=12, fontweight='bold')
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                if not np.isnan(v):
                    color = 'white' if (cmap == 'RdYlGn' and (v < vmin*0.5 or v > vmax*0.5)) else 'black'
                    ax.text(j, i, fmt.format(v), ha='center', va='center',
                            fontsize=7.5, color=color)
                else:
                    ax.text(j, i, '-', ha='center', va='center', fontsize=8, color='gray')
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        ax.set_xlabel('Threshold')

    draw_heatmap(axes[0], M_sharpe, 'Sharpe (annualized)', '{:+.1f}', 'RdYlGn')
    draw_heatmap(axes[1], M_mean, 'Mean PnL (bps/trade)', '{:+.0f}', 'RdYlGn')
    draw_heatmap(axes[2], M_wr, 'Win Rate (%)', '{:.0f}', 'RdYlGn',
                 vmin=30, vmax=80, center=None)
    draw_heatmap(axes[3], M_n, 'N trades', '{:.0f}', 'Blues',
                 vmin=0, vmax=np.nanmax(M_n), center=None)

    plt.suptitle(f'LME銅→日本株 ON戦略 銘柄×閾値グリッド '
                 f'(2025-04 ~ 2026-04, cost={COST_BPS}bps, N>=3のみ表示)',
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    out1 = '/Users/Yusuke/claude-code/japan-stocks/analyses/20260421_lme_copper_overnight/backtest_grid_heatmap.png'
    plt.savefig(out1, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"\n保存: {out1}")

    # === Figure 2: 銘柄別エクイティカーブ (閾値で色分け) ===
    ncol = 4
    nrow = int(np.ceil(len(STOCKS) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(18, 3*nrow), sharex=True)
    axes = axes.flatten()
    colors = {0.5: '#cccccc', 0.8: '#66c2a5', 1.0: '#1f77b4', 1.5: '#d62728'}

    for idx, (sym, name) in enumerate(STOCKS):
        ax = axes[idx]
        for th in THRESHOLDS:
            tdf = trades_all[(sym, th)]
            if len(tdf) == 0: continue
            tdf = tdf.sort_values('entry')
            tdf['cum'] = tdf['pnl_bps'].cumsum()
            st = results[(sym, th)]
            lbl = f"{th}% (N={st['n']}, Shp={st['sharpe']:+.1f})"
            ax.plot(pd.to_datetime(tdf['entry']), tdf['cum'],
                    color=colors[th], lw=1.5, label=lbl, marker='o', markersize=3)
        ax.axhline(0, color='gray', lw=0.5)
        ax.set_title(f"{sym} {name}", fontsize=10, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6, loc='best')
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))
        ax.tick_params(axis='x', rotation=45, labelsize=7)
        ax.tick_params(axis='y', labelsize=8)
    for k in range(len(STOCKS), len(axes)):
        axes[k].axis('off')

    plt.suptitle(f'銘柄別 エクイティカーブ (閾値別, cost={COST_BPS}bps)',
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    out2 = '/Users/Yusuke/claude-code/japan-stocks/analyses/20260421_lme_copper_overnight/backtest_grid_equity.png'
    plt.savefig(out2, dpi=110, bbox_inches='tight')
    plt.close()
    print(f"保存: {out2}")

    # === CSV 出力 ===
    rows = []
    for sym, name in STOCKS:
        for th in THRESHOLDS:
            st = results[(sym, th)]
            if st is None:
                rows.append({'sym': sym, 'name': name, 'th': th, 'n': 0})
            else:
                rows.append({'sym': sym, 'name': name, 'th': th, **st})
    dfres = pd.DataFrame(rows)
    csv_out = '/Users/Yusuke/claude-code/japan-stocks/analyses/20260421_lme_copper_overnight/backtest_grid.csv'
    dfres.to_csv(csv_out, index=False)
    print(f"保存: {csv_out}")

    # コンソール出力: th=1.0% Sharpe降順 Top10
    print("\n--- th=1.0% Sharpe Top10 ---")
    d10 = dfres[dfres.th == 1.0].dropna(subset=['sharpe']).sort_values('sharpe', ascending=False).head(10)
    for _, r in d10.iterrows():
        print(f"  {r['sym']:<10} {r['name']:<12} N={int(r['n']):>3} "
              f"WR={r['wr']:>5.1f}% PF={r['pf']:>5.2f} "
              f"Mean={r['mean']:>+6.0f}bps Sharpe={r['sharpe']:>+6.2f}")


if __name__ == "__main__":
    main()
