#!/usr/bin/env python3
"""
非鉄・電気機器（半導体）イントラデイ LS戦略探索
対象: sector33 IN ('3500','3650')、期間: 2026-01〜2026-06
仮説: ORB15/30, Gap Follow/Fade, AM Momentum
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
import os, io, psycopg2, psycopg2.extras
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント
try:
    import matplotlib.font_manager as fm
    fp = fm.FontProperties(fname="/root/.fonts/NotoSansJP.ttf")
    plt.rcParams["font.family"] = fp.get_name()
except Exception:
    pass

PG = dict(
    host=os.environ.get('PGHOST', 'localhost'),
    port=int(os.environ.get('PGPORT', 5432)),
    user=os.environ.get('PGUSER', 'postgres'),
    password=os.environ.get('PGPASSWORD', 'postgres'),
    dbname=os.environ.get('PGDATABASE', 'market_data'),
)

SECTORS       = ['3500', '3650']
PERIOD_FROM   = '2026-01-01'
PERIOD_TO     = '2026-06-19'
COST_BPS      = 4     # 往復コスト (片道2bps × 2)
COST_SLIP_BPS = 8     # スリッページ込み保守的コスト
TURNOVER_MIN  = 500_000_000  # 5億円/日
GAP_THRESH    = 0.3   # ギャップ判定閾値 (%)
AM_THRESH     = 0.3   # 前場モメンタム閾値 (%)


# ─────────────────────────────────────────────────
# データ取得
# ─────────────────────────────────────────────────

def get_liquid_codes(conn):
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        SELECT sd.code, sm.name_ja, sm.sector33, sm.sector33_nm,
               AVG(sd.close * sd.volume) AS avg_turnover
        FROM stocks_daily sd
        JOIN symbol_master sm ON sm.code5 = sd.code
        WHERE sm.sector33 = ANY(%s)
          AND sd.date BETWEEN %s AND %s
          AND sd.volume > 0 AND sd.close > 0
        GROUP BY sd.code, sm.name_ja, sm.sector33, sm.sector33_nm
        HAVING AVG(sd.close * sd.volume) >= %s
        ORDER BY avg_turnover DESC
    """, (SECTORS, PERIOD_FROM, PERIOD_TO, TURNOVER_MIN))
    rows = cur.fetchall()
    cur.close()
    return pd.DataFrame(rows)


def get_intraday(conn, codes):
    """1分足取得 - 50銘柄ずつ"""
    all_rows = []
    chunk_size = 50
    for i in range(0, len(codes), chunk_size):
        sub = codes[i:i+chunk_size]
        cur = conn.cursor()
        cur.execute("""
            SELECT ts, code, open, high, low, close, volume
            FROM stocks_intraday
            WHERE code = ANY(%s)
              AND ts >= %s::timestamp AND ts < %s::timestamp
            ORDER BY code, ts
        """, (sub, f'{PERIOD_FROM} 09:00:00', f'{PERIOD_TO} 16:00:00'))
        rows = cur.fetchall()
        all_rows.extend(rows)
        cur.close()
        print(f"  取得 {i+len(sub)}/{len(codes)} 銘柄 ({len(all_rows)} rows)")
    df = pd.DataFrame(all_rows, columns=['ts','code','open','high','low','close','volume'])
    return df


# ─────────────────────────────────────────────────
# 日次特徴量構築
# ─────────────────────────────────────────────────

def build_daily_features(df_raw):
    df = df_raw.copy()
    df['ts']   = pd.to_datetime(df['ts'])
    df['date'] = df['ts'].dt.date
    df['hm']   = df['ts'].dt.hour * 100 + df['ts'].dt.minute

    rows = []
    for (code, d), g in df.groupby(['code', 'date']):
        g = g.sort_values('ts').reset_index(drop=True)

        # ── 寄り付き ──────────────────────────────
        first_bar = g[g['hm'] == 900]
        if first_bar.empty:
            first_bar = g.iloc[[0]]
        open_p = float(first_bar['open'].iloc[0])
        if open_p <= 0:
            continue

        # ── ORB15 (9:00-9:14) ────────────────────
        orb15 = g[g['hm'] < 915]
        if len(orb15) < 3:
            continue
        orb15_high = float(orb15['high'].max())
        orb15_low  = float(orb15['low'].min())

        # ── ORB30 (9:00-9:29) ────────────────────
        orb30 = g[g['hm'] < 930]
        orb30_high = float(orb30['high'].max())
        orb30_low  = float(orb30['low'].min())

        # ── 各時点の価格 ─────────────────────────
        def get_price(hm_val, col='close'):
            b = g[g['hm'] == hm_val]
            return float(b[col].iloc[-1]) if not b.empty else None

        p915  = get_price(915)
        p930  = get_price(930)
        p1130 = get_price(1130)  # 前場終値
        p1230 = get_price(1230)  # 後場始値

        # 引け (15:30 or 最後のバー)
        eod_bar = g[g['hm'] >= 1525]
        if eod_bar.empty:
            eod_bar = g[g['hm'] >= 1500]
        if eod_bar.empty:
            continue
        eod_p = float(eod_bar['close'].iloc[-1])
        if eod_p <= 0:
            continue

        rows.append({
            'code':       code,
            'date':       d,
            'open':       open_p,
            'orb15_hi':  orb15_high,
            'orb15_lo':  orb15_low,
            'orb30_hi':  orb30_high,
            'orb30_lo':  orb30_low,
            'p915':      p915,
            'p930':      p930,
            'p1130':     p1130,
            'p1230':     p1230,
            'eod':       eod_p,
        })

    feat = pd.DataFrame(rows)
    if feat.empty:
        return feat

    # 前日終値 (gap計算用)
    feat = feat.sort_values(['code', 'date'])
    feat['prev_eod'] = feat.groupby('code')['eod'].shift(1)
    feat['gap_pct']  = (feat['open'] / feat['prev_eod'] - 1) * 100

    return feat


# ─────────────────────────────────────────────────
# シグナル生成
# ─────────────────────────────────────────────────

def add_signals(feat):
    f = feat.copy()

    # ORB15: 9:15終値がORBを抜けたか
    f['sig_orb15'] = 0
    mask_up   = f['p915'].notna() & (f['p915'] > f['orb15_hi'])
    mask_down = f['p915'].notna() & (f['p915'] < f['orb15_lo'])
    f.loc[mask_up,   'sig_orb15'] =  1
    f.loc[mask_down, 'sig_orb15'] = -1

    # ORB30: 9:30終値がORBを抜けたか
    f['sig_orb30'] = 0
    mask_up   = f['p930'].notna() & (f['p930'] > f['orb30_hi'])
    mask_down = f['p930'].notna() & (f['p930'] < f['orb30_lo'])
    f.loc[mask_up,   'sig_orb30'] =  1
    f.loc[mask_down, 'sig_orb30'] = -1

    # Gap Follow (9:30エントリー)
    f['sig_gap_follow'] = 0
    f.loc[f['gap_pct'] >=  GAP_THRESH, 'sig_gap_follow'] =  1
    f.loc[f['gap_pct'] <= -GAP_THRESH, 'sig_gap_follow'] = -1

    # Gap Fade (逆張り)
    f['sig_gap_fade'] = -f['sig_gap_follow']

    # AM Momentum: 前場リターンで後場エントリー
    f['am_ret_pct'] = (f['p1130'] / f['open'] - 1) * 100
    f['sig_am_mom'] = 0
    f.loc[f['am_ret_pct'] >=  AM_THRESH, 'sig_am_mom'] =  1
    f.loc[f['am_ret_pct'] <= -AM_THRESH, 'sig_am_mom'] = -1

    return f


# ─────────────────────────────────────────────────
# バックテスト
# ─────────────────────────────────────────────────

STRATEGIES = {
    'ORB15':       ('p915',  'eod', 'sig_orb15'),
    'ORB30':       ('p930',  'eod', 'sig_orb30'),
    'Gap_Follow':  ('p930',  'eod', 'sig_gap_follow'),
    'Gap_Fade':    ('p930',  'eod', 'sig_gap_fade'),
    'AM_Momentum': ('p1230', 'eod', 'sig_am_mom'),
}


def simulate(feat, entry_col, exit_col, sig_col, cost_bps=COST_BPS):
    df = feat.dropna(subset=[entry_col, exit_col]).copy()
    df = df[df[sig_col] != 0].copy()
    if df.empty:
        return df
    df['ret'] = df[sig_col] * (df[exit_col] / df[entry_col] - 1) - cost_bps / 10000
    return df


def evaluate(df):
    if len(df) == 0:
        return {'n': 0, 'win_rate': 0, 'pf': 0, 'mean_ret_bps': 0, 'sr': 0, 'total_pct': 0}
    r = df['ret']
    wins   = r[r > 0].sum()
    losses = -r[r < 0].sum()
    return {
        'n':           len(r),
        'win_rate':    float((r > 0).mean()),
        'pf':          float(wins / losses) if losses > 0 else float('inf'),
        'mean_ret_bps':float(r.mean() * 10000),
        'sr':          float(r.mean() / r.std()) if r.std() > 0 else 0,
        'total_pct':   float(r.sum() * 100),
    }


# ─────────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────────

def main():
    conn = psycopg2.connect(**PG)

    # 1. 銘柄選定
    print("=== 1. 流動性フィルタ ===")
    liq = get_liquid_codes(conn)
    liq['name_ja'] = liq['name_ja'].apply(lambda x: x if isinstance(x, str) else str(x))
    print(f"対象: {len(liq)} 銘柄 (売買代金>=5億円/日)")
    print(liq[['code','name_ja','sector33','avg_turnover']].head(30).to_string(index=False))
    codes = liq['code'].tolist()

    # 2. イントラデイ取得
    print("\n=== 2. イントラデイデータ取得 ===")
    raw = get_intraday(conn, codes)
    conn.close()
    print(f"  総行数: {len(raw):,}")

    # 3. 日次特徴量
    print("\n=== 3. 日次特徴量構築 ===")
    feat = build_daily_features(raw)
    feat = add_signals(feat)
    feat = feat.merge(liq[['code','name_ja','sector33']], on='code', how='left')
    print(f"  {len(feat)} 銘柄-日, {feat['code'].nunique()} 銘柄, {feat['date'].nunique()} 営業日")

    # 4. 全戦略サマリー
    print("\n=== 4. 戦略サマリー（往復コスト 4bps込み） ===")
    all_res = {}
    header = f"{'戦略':<15} {'n':>5} {'勝率':>7} {'PF':>6} {'平均(bps)':>10} {'SR':>7} {'累計%':>8}"
    print(header)
    print('-' * len(header))
    for name, (ec, xc, sc) in STRATEGIES.items():
        df_sim = simulate(feat, ec, xc, sc)
        ev = evaluate(df_sim)
        all_res[name] = (df_sim, ev)
        print(f"{name:<15} {ev['n']:>5} {ev['win_rate']:>7.1%} {ev['pf']:>6.2f} "
              f"{ev['mean_ret_bps']:>10.2f} {ev['sr']:>7.3f} {ev['total_pct']:>8.2f}%")

    # 5. 保守コスト(8bps)での評価
    print("\n=== 5. 保守コスト 8bps（スリッページ込み）===")
    print(header)
    print('-' * len(header))
    for name, (ec, xc, sc) in STRATEGIES.items():
        df_sim = simulate(feat, ec, xc, sc, cost_bps=COST_SLIP_BPS)
        ev = evaluate(df_sim)
        print(f"{name:<15} {ev['n']:>5} {ev['win_rate']:>7.1%} {ev['pf']:>6.2f} "
              f"{ev['mean_ret_bps']:>10.2f} {ev['sr']:>7.3f} {ev['total_pct']:>8.2f}%")

    # 6. セクター別比較（最良戦略で）
    print("\n=== 6. セクター別比較 ===")
    best_strat = max(all_res.items(), key=lambda x: x[1][1]['sr'])
    best_name  = best_strat[0]
    ec, xc, sc = STRATEGIES[best_name]
    print(f"最良戦略: {best_name}")
    for sec in SECTORS:
        sub = feat[feat['sector33'] == sec]
        ev = evaluate(simulate(sub, ec, xc, sc))
        nm = '非鉄(3500)' if sec == '3500' else '電気機器(3650)'
        print(f"  {nm}: n={ev['n']}, 勝率={ev['win_rate']:.1%}, PF={ev['pf']:.2f}, "
              f"平均={ev['mean_ret_bps']:.2f}bps, SR={ev['sr']:.3f}")

    # 7. 月別推移（Gap Follow / ORB30）
    print("\n=== 7. 月別リターン推移 ===")
    feat['month'] = pd.to_datetime(feat['date']).dt.to_period('M')
    for check_name in ['Gap_Follow', 'ORB30', 'AM_Momentum']:
        if check_name not in STRATEGIES:
            continue
        ec, xc, sc = STRATEGIES[check_name]
        df_sim = simulate(feat, ec, xc, sc)
        if df_sim.empty:
            continue
        df_sim['month'] = pd.to_datetime(df_sim['date']).dt.to_period('M')
        monthly = df_sim.groupby('month')['ret'].mean() * 10000
        print(f"\n  {check_name} 月別平均(bps):")
        for m, v in monthly.items():
            bar = '█' * int(abs(v) / 2) if abs(v) < 60 else '█' * 30
            sign = '+' if v >= 0 else '-'
            print(f"    {m}: {sign}{abs(v):5.1f}  {bar}")

    # 8. 銘柄別ランキング（SR上位）
    print("\n=== 8. 銘柄別 Gap_Follow SR ランキング ===")
    ec, xc, sc = STRATEGIES['Gap_Follow']
    code_evs = []
    for code, g in feat.groupby('code'):
        df_sim = simulate(g, ec, xc, sc)
        ev = evaluate(df_sim)
        if ev['n'] >= 10:
            nm = liq[liq['code']==code]['name_ja'].values
            sec = liq[liq['code']==code]['sector33'].values
            code_evs.append({
                'code': code[:4],
                'name': nm[0] if len(nm) else '?',
                'sec':  sec[0] if len(sec) else '?',
                **ev
            })
    cr_df = pd.DataFrame(code_evs).sort_values('sr', ascending=False)
    print(cr_df[['code','name','sec','n','win_rate','pf','mean_ret_bps','sr','total_pct']].head(25).to_string(index=False))

    # ─────────────────────────────────────────────────
    # 可視化
    # ─────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 12))
    gs  = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35)

    strat_list = list(STRATEGIES.items())

    # 累積リターン曲線 (各戦略)
    for i, (name, (ec, xc, sc)) in enumerate(strat_list):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        df_sim = all_res[name][0]
        ev     = all_res[name][1]
        if df_sim.empty:
            ax.set_title(f'{name}\n(データなし)')
            continue
        df_sim_s = df_sim.sort_values('date')
        daily_ret = df_sim_s.groupby('date')['ret'].mean()
        cum = daily_ret.cumsum() * 100
        dates = [pd.Timestamp(d) for d in cum.index]
        color = 'steelblue' if ev['total_pct'] >= 0 else 'tomato'
        ax.plot(dates, cum.values, color=color, linewidth=1.5)
        ax.axhline(0, color='k', linewidth=0.5, linestyle='--')
        ax.set_title(f"{name}\nn={ev['n']}, PF={ev['pf']:.2f}, SR={ev['sr']:.2f}, 計={ev['total_pct']:.1f}%",
                     fontsize=9)
        ax.set_ylabel('累積リターン (%)', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
        ax.tick_params(labelsize=7)

    # 全戦略比較
    ax6 = fig.add_subplot(gs[1, 2])
    colors_map = ['steelblue','royalblue','green','red','darkorange']
    for i, (name, (ec, xc, sc)) in enumerate(strat_list):
        df_sim = all_res[name][0]
        if df_sim.empty:
            continue
        df_sim_s = df_sim.sort_values('date')
        daily_ret = df_sim_s.groupby('date')['ret'].mean()
        cum = daily_ret.cumsum() * 100
        dates = [pd.Timestamp(d) for d in cum.index]
        ax6.plot(dates, cum.values, label=name, color=colors_map[i], linewidth=1.2)
    ax6.axhline(0, color='k', linewidth=0.5, linestyle='--')
    ax6.set_title('全戦略比較', fontsize=9)
    ax6.legend(fontsize=7)
    ax6.grid(True, alpha=0.3)
    ax6.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
    ax6.tick_params(labelsize=7)

    # Gap Follow 銘柄別SR棒グラフ (上位15)
    ax7 = fig.add_subplot(gs[2, :2])
    top15 = cr_df.head(15)
    colors_bar = ['steelblue' if v >= 0 else 'tomato' for v in top15['sr']]
    ax7.barh(top15['name'], top15['sr'], color=colors_bar, edgecolor='gray', linewidth=0.3)
    ax7.axvline(0, color='k', linewidth=0.8)
    ax7.set_title('Gap_Follow 銘柄別SR（上位15）', fontsize=9)
    ax7.set_xlabel('SR', fontsize=8)
    ax7.tick_params(labelsize=7)
    ax7.grid(True, alpha=0.3, axis='x')

    # 月別ヒートマップ (Gap Follow)
    ax8 = fig.add_subplot(gs[2, 2])
    ec, xc, sc = STRATEGIES['Gap_Follow']
    df_sim = simulate(feat, ec, xc, sc)
    if not df_sim.empty:
        df_sim['month'] = pd.to_datetime(df_sim['date']).dt.to_period('M')
        monthly = df_sim.groupby('month')['ret'].mean() * 10000
        months  = [str(m) for m in monthly.index]
        vals    = monthly.values
        colors_m = ['steelblue' if v >= 0 else 'tomato' for v in vals]
        ax8.barh(months, vals, color=colors_m, edgecolor='gray', linewidth=0.3)
        ax8.axvline(0, color='k', linewidth=0.8)
        ax8.set_title('Gap_Follow 月別平均(bps)', fontsize=9)
        ax8.set_xlabel('bps', fontsize=8)
        ax8.tick_params(labelsize=7)
        ax8.grid(True, alpha=0.3, axis='x')

    fig.suptitle(
        f'非鉄・電気機器 イントラデイ戦略検証 ({PERIOD_FROM}〜{PERIOD_TO})\n'
        f'往復コスト {COST_BPS}bps込み | 銘柄数: {feat["code"].nunique()} | 期間: 約{feat["date"].nunique()}営業日',
        fontsize=11, fontweight='bold'
    )

    out_path = os.path.join(os.path.dirname(__file__), 'result.png')
    fig.savefig(out_path, dpi=100, bbox_inches='tight')
    print(f"\n保存: {out_path}")


if __name__ == '__main__':
    main()
