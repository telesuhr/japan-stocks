#!/usr/bin/env python3
"""
VWAPベースのイントラデイ戦略追加検証
前回run.pyの続編

仮説:
  VWAP_Cross_930 : 9:30時点で価格>VWAP → ロング (モメンタム)
  VWAP_Fade_930  : 9:30時点でVWAP偏差 >= 閾値 → 逆張り → 引けイグジット
  VWAP_Fade_1000 : 10:00時点でVWAP偏差 >= 閾値 → 逆張り → 引けイグジット
  VWAP_Fade_1030 : 10:30時点でVWAP偏差 >= 閾値 → 逆張り → 引けイグジット
  Gap×VWAP複合   : ギャップ逆張り AND VWAP方向も一致のみエントリー
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
import os, psycopg2, psycopg2.extras
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

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

SECTORS     = ['3500', '3650']
PERIOD_FROM = '2026-01-01'
PERIOD_TO   = '2026-06-19'
COST_BPS    = 4
TURNOVER_MIN = 500_000_000
GAP_THRESH  = 0.3   # ギャップ閾値(%)


# ─────────────────────────────────────────────────
# データ取得（run.pyと共通）
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
        print(f"  {i+len(sub)}/{len(codes)} 銘柄 ({len(all_rows):,} rows)")
    df = pd.DataFrame(all_rows, columns=['ts','code','open','high','low','close','volume'])
    for col in ['open','high','low','close','volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


# ─────────────────────────────────────────────────
# 日次特徴量 + VWAP
# ─────────────────────────────────────────────────

def build_daily_features_vwap(df_raw):
    df = df_raw.copy()
    df['ts']      = pd.to_datetime(df['ts'])
    df['date']    = df['ts'].dt.date
    df['hm']      = df['ts'].dt.hour * 100 + df['ts'].dt.minute
    df['typical'] = (df['high'] + df['low'] + df['close']) / 3.0

    rows = []
    for (code, d), g in df.groupby(['code', 'date']):
        g = g.sort_values('ts').reset_index(drop=True)
        if len(g) < 10:
            continue

        # VWAP累積計算
        g['tp_vol']      = g['typical'] * g['volume']
        g['cum_tp_vol']  = g['tp_vol'].cumsum()
        g['cum_vol']     = g['volume'].cumsum()
        g['vwap']        = np.where(g['cum_vol'] > 0,
                                    g['cum_tp_vol'] / g['cum_vol'], np.nan)

        # 寄り付き
        first = g[g['hm'] == 900]
        if first.empty:
            first = g.iloc[[0]]
        open_p = float(first['open'].iloc[0])
        if open_p <= 0:
            continue

        def snap(hm_val):
            """指定時刻のclose・VWAP・偏差を返す"""
            b = g[g['hm'] == hm_val]
            if b.empty:
                return None, None, None
            p = float(b['close'].iloc[-1])
            v = float(b['vwap'].iloc[-1])
            if v > 0 and p > 0:
                dev = (p / v - 1) * 100
            else:
                dev = None
            return p, v, dev

        p930,  vwap_930,  dev_930  = snap(930)
        p1000, vwap_1000, dev_1000 = snap(1000)
        p1030, vwap_1030, dev_1030 = snap(1030)
        p1130, _,         _        = snap(1130)
        p1230, _,         _        = snap(1230)

        eod_bar = g[g['hm'] >= 1525]
        if eod_bar.empty:
            eod_bar = g[g['hm'] >= 1500]
        if eod_bar.empty:
            continue
        eod_p = float(eod_bar['close'].iloc[-1])
        if eod_p <= 0:
            continue

        rows.append({
            'code':     code,
            'date':     d,
            'open':     open_p,
            'p930':     p930,
            'p1000':    p1000,
            'p1030':    p1030,
            'p1130':    p1130,
            'p1230':    p1230,
            'eod':      eod_p,
            'dev_930':  dev_930,
            'dev_1000': dev_1000,
            'dev_1030': dev_1030,
        })

    feat = pd.DataFrame(rows)
    if feat.empty:
        return feat

    feat = feat.sort_values(['code', 'date'])
    feat['prev_eod'] = feat.groupby('code')['eod'].shift(1)
    feat['gap_pct']  = (feat['open'] / feat['prev_eod'] - 1) * 100
    return feat


# ─────────────────────────────────────────────────
# シグナル
# ─────────────────────────────────────────────────

DEV_THRESHOLDS = [0.2, 0.5, 1.0]  # VWAP偏差の閾値(%)

def add_signals(feat):
    f = feat.copy()

    # VWAP Cross: 価格 > VWAP → ロング（モメンタム）
    f['sig_vwap_cross_930'] = np.sign(f['dev_930'].fillna(0)).astype(int)

    # VWAP Fade（各時刻×各閾値）
    for t in DEV_THRESHOLDS:
        tstr = str(t).replace('.', '')
        for hm, dev_col in [('930', 'dev_930'), ('1000', 'dev_1000'), ('1030', 'dev_1030')]:
            sig = f'sig_vwap_fade_{hm}_{tstr}'
            f[sig] = 0
            f.loc[f[dev_col] >=  t, sig] = -1   # 上振れ → ショート
            f.loc[f[dev_col] <= -t, sig] =  1   # 下振れ → ロング

    # Gap逆張りベース (比較用・再掲)
    f['sig_gap_fade'] = 0
    f.loc[f['gap_pct'] >=  GAP_THRESH, 'sig_gap_fade'] = -1
    f.loc[f['gap_pct'] <= -GAP_THRESH, 'sig_gap_fade'] =  1

    # Gap × VWAP Fade 複合（両方逆張り方向が一致するもの）
    # ギャップダウン & VWAP下 → ロング / ギャップアップ & VWAP上 → ショート
    f['sig_gap_and_vwap_930_05'] = 0
    mask_long  = (f['gap_pct'] <= -GAP_THRESH) & (f['dev_930'] <= -0.5)
    mask_short = (f['gap_pct'] >=  GAP_THRESH) & (f['dev_930'] >=  0.5)
    f.loc[mask_long,  'sig_gap_and_vwap_930_05'] =  1
    f.loc[mask_short, 'sig_gap_and_vwap_930_05'] = -1

    # VWAP Fade 逆 (VWAP Cross延長: 偏差がある方向に追随)
    for t in DEV_THRESHOLDS:
        tstr = str(t).replace('.', '')
        for hm, dev_col in [('930', 'dev_930'), ('1000', 'dev_1000')]:
            sig = f'sig_vwap_follow_{hm}_{tstr}'
            f[sig] = 0
            f.loc[f[dev_col] >=  t, sig] =  1   # 上振れ → ロング（追随）
            f.loc[f[dev_col] <= -t, sig] = -1

    return f


# ─────────────────────────────────────────────────
# バックテスト評価
# ─────────────────────────────────────────────────

def simulate(feat, entry_col, exit_col, sig_col, cost_bps=COST_BPS):
    df = feat.dropna(subset=[entry_col, exit_col, sig_col]).copy()
    df = df[df[sig_col] != 0].copy()
    if df.empty:
        return df
    df['ret'] = df[sig_col] * (df[exit_col] / df[entry_col] - 1) - cost_bps / 10000
    return df


def evaluate(df):
    if len(df) == 0:
        return {'n': 0, 'win_rate': 0, 'pf': 0, 'mean_bps': 0, 'sr': 0, 'total_pct': 0}
    r = df['ret']
    wins   = r[r > 0].sum()
    losses = -r[r < 0].sum()
    return {
        'n':         len(r),
        'win_rate':  float((r > 0).mean()),
        'pf':        float(wins / losses) if losses > 0 else float('inf'),
        'mean_bps':  float(r.mean() * 10000),
        'sr':        float(r.mean() / r.std()) if r.std() > 0 else 0,
        'total_pct': float(r.sum() * 100),
    }


# ─────────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────────

def main():
    conn = psycopg2.connect(**PG)

    print("=== 銘柄選定 ===")
    liq = get_liquid_codes(conn)
    print(f"対象: {len(liq)} 銘柄")
    codes = liq['code'].tolist()

    print("\n=== イントラデイデータ取得 ===")
    raw = get_intraday(conn, codes)
    conn.close()
    print(f"総行数: {len(raw):,}")

    print("\n=== VWAP特徴量構築（銘柄×日ループ） ===")
    feat = build_daily_features_vwap(raw)
    feat = add_signals(feat)
    feat = feat.merge(liq[['code','name_ja','sector33']], on='code', how='left')
    print(f"銘柄-日: {len(feat)}, 銘柄数: {feat['code'].nunique()}, 営業日: {feat['date'].nunique()}")

    # ─────────────────────────────────────────────────
    # 1. VWAP Cross（モメンタム）vs Fade（逆張り）
    # ─────────────────────────────────────────────────
    print("\n=== VWAP Cross vs Fade サマリー（4bps込み） ===")
    summary_strategies = {
        'VWAP_Cross_930':       ('p930',  'eod', 'sig_vwap_cross_930'),
        'Gap_Fade (比較)':      ('p930',  'eod', 'sig_gap_fade'),
    }
    for t in DEV_THRESHOLDS:
        tstr = str(t).replace('.', '')
        for hm in ['930', '1000', '1030']:
            sig = f'sig_vwap_fade_{hm}_{tstr}'
            entry = f'p{hm}'
            summary_strategies[f'VWAP_Fade_{hm}_{t}%'] = (entry, 'eod', sig)

    hdr = f"{'戦略':<30} {'n':>5} {'勝率':>6} {'PF':>5} {'bps':>7} {'SR':>7} {'累計%':>7}"
    print(hdr)
    print('-' * len(hdr))
    all_ev = {}
    for name, (ec, xc, sc) in summary_strategies.items():
        df_sim = simulate(feat, ec, xc, sc)
        ev = evaluate(df_sim)
        all_ev[name] = (df_sim, ev)
        marker = ' <<<' if ev['sr'] > 0.05 else ''
        print(f"{name:<30} {ev['n']:>5} {ev['win_rate']:>6.1%} {ev['pf']:>5.2f} "
              f"{ev['mean_bps']:>7.2f} {ev['sr']:>7.3f} {ev['total_pct']:>7.2f}%{marker}")

    # ─────────────────────────────────────────────────
    # 2. VWAP Follow（VWAP方向に順張り）
    # ─────────────────────────────────────────────────
    print("\n=== VWAP Follow（閾値越えたら方向追随）===")
    for t in DEV_THRESHOLDS:
        tstr = str(t).replace('.', '')
        for hm in ['930', '1000']:
            sig  = f'sig_vwap_follow_{hm}_{tstr}'
            entry = f'p{hm}'
            df_sim = simulate(feat, entry, 'eod', sig)
            ev = evaluate(df_sim)
            marker = ' <<<' if ev['sr'] > 0.05 else ''
            print(f"  Follow_{hm}_{t}%: n={ev['n']:>4}, 勝率={ev['win_rate']:.1%}, "
                  f"bps={ev['mean_bps']:>7.2f}, SR={ev['sr']:>7.3f}{marker}")

    # ─────────────────────────────────────────────────
    # 3. Gap × VWAP 複合
    # ─────────────────────────────────────────────────
    print("\n=== Gap_Fade × VWAP_Fade 複合フィルタ ===")
    df_gap  = simulate(feat, 'p930', 'eod', 'sig_gap_fade')
    df_comb = simulate(feat, 'p930', 'eod', 'sig_gap_and_vwap_930_05')
    ev_gap  = evaluate(df_gap)
    ev_comb = evaluate(df_comb)
    print(f"  Gap_Fade 単独:      n={ev_gap['n']:>4}, SR={ev_gap['sr']:.3f}, bps={ev_gap['mean_bps']:.2f}")
    print(f"  Gap × VWAP>=0.5%:  n={ev_comb['n']:>4}, SR={ev_comb['sr']:.3f}, bps={ev_comb['mean_bps']:.2f}")

    # ─────────────────────────────────────────────────
    # 4. 最良VWAP戦略の月別・銘柄別
    # ─────────────────────────────────────────────────
    best_name = max(all_ev.items(), key=lambda x: x[1][1]['sr'])
    best_n    = best_name[0]
    print(f"\n=== 最良戦略: {best_n} ===")
    best_ec, best_xc, best_sc = summary_strategies[best_n]

    df_best = all_ev[best_n][0]
    if not df_best.empty:
        df_best['month'] = pd.to_datetime(df_best['date']).dt.to_period('M')
        monthly = df_best.groupby('month')['ret'].mean() * 10000
        print("  月別平均(bps):")
        for m, v in monthly.items():
            bar = '█' * min(int(abs(v)/3), 25)
            sign = '+' if v >= 0 else '-'
            print(f"    {m}: {sign}{abs(v):5.1f}  {bar}")

    # セクター別
    print(f"\n  セクター別:")
    for sec in SECTORS:
        sub = feat[feat['sector33'] == sec]
        ev  = evaluate(simulate(sub, best_ec, best_xc, best_sc))
        nm  = '非鉄(3500)' if sec == '3500' else '電気機器(3650)'
        print(f"    {nm}: n={ev['n']}, SR={ev['sr']:.3f}, bps={ev['mean_bps']:.2f}")

    # 銘柄別ランキング
    print(f"\n  銘柄別SR上位（n>=10）:")
    code_evs = []
    for code, g in feat.groupby('code'):
        ev = evaluate(simulate(g, best_ec, best_xc, best_sc))
        if ev['n'] >= 10:
            nm  = liq[liq['code']==code]['name_ja'].values
            sec = liq[liq['code']==code]['sector33'].values
            code_evs.append({'code': code[:4], 'name': nm[0] if len(nm) else '?',
                              'sec': sec[0] if len(sec) else '?', **ev})
    cr = pd.DataFrame(code_evs).sort_values('sr', ascending=False)
    print(cr[['code','name','sec','n','win_rate','pf','mean_bps','sr']].head(20).to_string(index=False))

    # ─────────────────────────────────────────────────
    # VWAP偏差の分布確認（スナップショット）
    # ─────────────────────────────────────────────────
    print("\n=== VWAP偏差 統計（9:30時点） ===")
    dev = feat['dev_930'].dropna()
    print(f"  mean={dev.mean():.3f}%, std={dev.std():.3f}%, "
          f"±0.5%以上: {(dev.abs() >= 0.5).mean():.1%}, "
          f"±1.0%以上: {(dev.abs() >= 1.0).mean():.1%}")

    # ─────────────────────────────────────────────────
    # 可視化
    # ─────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 12))
    gs  = fig.add_gridspec(3, 3, hspace=0.5, wspace=0.38)

    # Panel 1-5: 主要戦略の累積リターン曲線
    key_strategies = [
        ('VWAP_Cross_930',       'p930', 'sig_vwap_cross_930'),
        ('VWAP_Fade_930_0.5%',   'p930', 'sig_vwap_fade_930_05'),
        ('VWAP_Fade_1000_0.5%',  'p1000','sig_vwap_fade_1000_05'),
        ('VWAP_Fade_1030_0.5%',  'p1030','sig_vwap_fade_1030_05'),
        ('Gap_Fade (前回)',       'p930', 'sig_gap_fade'),
    ]
    colors_line = ['steelblue','green','darkorange','purple','red']

    ax_compare = fig.add_subplot(gs[0, :2])
    for i, (name, ec, sc) in enumerate(key_strategies):
        df_s = simulate(feat, ec, 'eod', sc)
        if df_s.empty:
            continue
        ev = evaluate(df_s)
        cum = df_s.sort_values('date').groupby('date')['ret'].mean().cumsum() * 100
        dates = [pd.Timestamp(d) for d in cum.index]
        ax_compare.plot(dates, cum.values, label=f"{name} SR={ev['sr']:.3f}",
                        color=colors_line[i], linewidth=1.4)
    ax_compare.axhline(0, color='k', linewidth=0.5, linestyle='--')
    ax_compare.set_title('主要VWAP戦略 累積リターン比較（4bps込み）', fontsize=9)
    ax_compare.legend(fontsize=7)
    ax_compare.grid(True, alpha=0.3)
    ax_compare.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
    ax_compare.set_ylabel('累積リターン (%)', fontsize=8)

    # Panel 右上: VWAP偏差ヒストグラム
    ax_hist = fig.add_subplot(gs[0, 2])
    dev930 = feat['dev_930'].dropna()
    ax_hist.hist(dev930, bins=80, color='steelblue', edgecolor='none', alpha=0.7)
    ax_hist.axvline( 0.5, color='red',   linewidth=1, linestyle='--', label='±0.5%')
    ax_hist.axvline(-0.5, color='red',   linewidth=1, linestyle='--')
    ax_hist.axvline( 1.0, color='orange',linewidth=1, linestyle='--', label='±1.0%')
    ax_hist.axvline(-1.0, color='orange',linewidth=1, linestyle='--')
    ax_hist.axvline(0, color='k', linewidth=0.8)
    ax_hist.set_xlim(-5, 5)
    ax_hist.set_title('9:30 VWAP偏差 分布', fontsize=9)
    ax_hist.set_xlabel('偏差 (%)', fontsize=8)
    ax_hist.legend(fontsize=7)
    ax_hist.grid(True, alpha=0.3)

    # Panel 2: VWAP Fade 閾値別比較（ヒートマップ風）
    ax_heat = fig.add_subplot(gs[1, :2])
    hm_vals = ['930', '1000', '1030']
    x = np.arange(len(DEV_THRESHOLDS))
    width = 0.25
    for j, hm in enumerate(hm_vals):
        srs = []
        for t in DEV_THRESHOLDS:
            tstr = str(t).replace('.', '')
            sc = f'sig_vwap_fade_{hm}_{tstr}'
            entry = f'p{hm}'
            ev = evaluate(simulate(feat, entry, 'eod', sc))
            srs.append(ev['sr'])
        bars = ax_heat.bar(x + j*width, srs, width, label=f'{hm}時点',
                           color=['steelblue','darkorange','green'][j], alpha=0.8)
        for bar, val in zip(bars, srs):
            ax_heat.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                         f'{val:.3f}', ha='center', va='bottom', fontsize=7)
    ax_heat.axhline(0, color='k', linewidth=0.8)
    ax_heat.set_xticks(x + width)
    ax_heat.set_xticklabels([f'閾値{t}%' for t in DEV_THRESHOLDS])
    ax_heat.set_title('VWAP_Fade SR — 時点×閾値', fontsize=9)
    ax_heat.set_ylabel('SR', fontsize=8)
    ax_heat.legend(fontsize=8)
    ax_heat.grid(True, alpha=0.3, axis='y')

    # Panel 右中: Gap×VWAPフィルタ比較
    ax_combo = fig.add_subplot(gs[1, 2])
    combo_labels = ['Gap_Fade\n単独', 'Gap×VWAP\n>=0.5%']
    combo_srs    = [ev_gap['sr'], ev_comb['sr']]
    combo_bps    = [ev_gap['mean_bps'], ev_comb['mean_bps']]
    colors_c = ['steelblue' if v >= 0 else 'tomato' for v in combo_srs]
    bars = ax_combo.bar(combo_labels, combo_srs, color=colors_c, edgecolor='gray')
    ax_combo.axhline(0, color='k', linewidth=0.8)
    for bar, val, bps in zip(bars, combo_srs, combo_bps):
        ax_combo.text(bar.get_x() + bar.get_width()/2,
                      bar.get_height() + 0.001 if val >= 0 else bar.get_height() - 0.005,
                      f'SR={val:.3f}\n{bps:.1f}bps', ha='center', va='bottom', fontsize=8)
    ax_combo.set_title('Gap×VWAP複合フィルタ', fontsize=9)
    ax_combo.set_ylabel('SR', fontsize=8)
    ax_combo.grid(True, alpha=0.3, axis='y')

    # Panel 下段左: 最良戦略 銘柄別SR
    ax_code = fig.add_subplot(gs[2, :2])
    top20 = cr.head(20)
    colors_bar = ['steelblue' if v >= 0 else 'tomato' for v in top20['sr']]
    ax_code.barh(top20['name'], top20['sr'], color=colors_bar, edgecolor='gray', linewidth=0.3)
    ax_code.axvline(0, color='k', linewidth=0.8)
    ax_code.set_title(f'銘柄別SR — {best_n}（上位20）', fontsize=9)
    ax_code.set_xlabel('SR', fontsize=8)
    ax_code.tick_params(labelsize=7)
    ax_code.grid(True, alpha=0.3, axis='x')

    # Panel 下右: 最良戦略 月別リターン
    ax_mon = fig.add_subplot(gs[2, 2])
    if not df_best.empty:
        monthly_vals = df_best.groupby('month')['ret'].mean() * 10000
        months = [str(m) for m in monthly_vals.index]
        vals   = monthly_vals.values
        colors_m = ['steelblue' if v >= 0 else 'tomato' for v in vals]
        ax_mon.barh(months, vals, color=colors_m, edgecolor='gray', linewidth=0.3)
        ax_mon.axvline(0, color='k', linewidth=0.8)
        ax_mon.set_title(f'{best_n}\n月別平均(bps)', fontsize=9)
        ax_mon.set_xlabel('bps', fontsize=8)
        ax_mon.tick_params(labelsize=8)
        ax_mon.grid(True, alpha=0.3, axis='x')

    fig.suptitle(
        f'非鉄・電気機器 VWAPイントラデイ戦略検証 ({PERIOD_FROM}〜{PERIOD_TO})\n'
        f'往復コスト {COST_BPS}bps込み | {feat["code"].nunique()} 銘柄 | {feat["date"].nunique()} 営業日',
        fontsize=11, fontweight='bold'
    )

    out = os.path.join(os.path.dirname(__file__), 'result_vwap.png')
    fig.savefig(out, dpi=100, bbox_inches='tight')
    print(f"\n保存: {out}")


if __name__ == '__main__':
    main()
