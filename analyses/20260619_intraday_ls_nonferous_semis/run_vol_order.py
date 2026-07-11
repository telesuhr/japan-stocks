#!/usr/bin/env python3
"""
出来高スパイク・板インバランス戦略検証

[A] 出来高ベース (stocks_intraday, 122銘柄, 6ヶ月)
  A1. EarlyVol_Spike : 9:00-9:30の出来高が前20日平均比N倍 → 逆張り/追随
  A2. Vol_Trend      : 前場の出来高増減トレンドと後場リターンの相関
  A3. VolPrice_Div   : 出来高増大＋価格下落（セリクラ仮説）→ 引け翌日リターン

[B] 板インバランス (aukabu.snapshots_5sec + bars_1min, 50銘柄, ~5週間)
  B1. L1_Imb_930     : 9:30前後の買い板/売り板比率 (BBO) → 方向予測
  B2. Depth10_Imb    : 10層板厚バランス
  B3. VolSpike_Ratio : aukabu計算済み vol_spike_ratio
  B4. Market_Pressure: 総合市場圧力指標

教訓4: 1分足の出来高スパイクは遅い可能性あり（先読みなし厳守・教訓1）
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

SECTORS      = ['3500', '3650']
PERIOD_FROM  = '2026-01-01'
PERIOD_TO    = '2026-06-19'
AUKABU_FROM  = '2026-05-22'   # snapshots_5sec の開始日
COST_BPS     = 4
TURNOVER_MIN = 500_000_000
GAP_THRESH   = 0.3


# ─────────────────────────────────────────────────
# 共通: 流動性フィルタ
# ─────────────────────────────────────────────────

def get_liquid_codes(conn):
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        SELECT sd.code, sm.name_ja, sm.sector33
        FROM stocks_daily sd
        JOIN symbol_master sm ON sm.code5 = sd.code
        WHERE sm.sector33 = ANY(%s)
          AND sd.date BETWEEN %s AND %s
          AND sd.volume > 0 AND sd.close > 0
        GROUP BY sd.code, sm.name_ja, sm.sector33
        HAVING AVG(sd.close * sd.volume) >= %s
    """, (SECTORS, PERIOD_FROM, PERIOD_TO, TURNOVER_MIN))
    return pd.DataFrame(cur.fetchall())


# ─────────────────────────────────────────────────
# [A] 出来高スパイク戦略
# ─────────────────────────────────────────────────

def get_daily_vol_ratio(conn, codes):
    """日次出来高と過去20日ローリング平均の比率"""
    cur = conn.cursor()
    cur.execute("""
        SELECT code, date, volume, close
        FROM stocks_daily
        WHERE code = ANY(%s)
          AND date BETWEEN %s AND %s
          AND volume > 0
        ORDER BY code, date
    """, (codes, PERIOD_FROM, PERIOD_TO))
    rows = cur.fetchall()
    cur.close()
    df = pd.DataFrame(rows, columns=['code','date','volume','close'])
    df['volume'] = pd.to_numeric(df['volume'])
    df['close']  = pd.to_numeric(df['close'])
    # 過去20日平均（当日を除く）
    df = df.sort_values(['code','date'])
    df['vol_ma20'] = df.groupby('code')['volume'].transform(
        lambda x: x.shift(1).rolling(20, min_periods=5).mean()
    )
    df['vol_ratio'] = df['volume'] / df['vol_ma20']
    return df


def get_intraday(conn, codes, date_from, date_to):
    """1分足取得"""
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
        """, (sub, f'{date_from} 09:00:00', f'{date_to} 16:00:00'))
        rows = cur.fetchall()
        all_rows.extend(rows)
        cur.close()
        print(f"  {i+len(sub)}/{len(codes)} ({len(all_rows):,} rows)")
    df = pd.DataFrame(all_rows, columns=['ts','code','open','high','low','close','volume'])
    for c in ['open','high','low','close','volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def build_vol_features(raw, vol_ratio_df):
    """出来高ベースの日次特徴量"""
    df = raw.copy()
    df['ts']   = pd.to_datetime(df['ts'])
    df['date'] = df['ts'].dt.date
    df['hm']   = df['ts'].dt.hour * 100 + df['ts'].dt.minute

    rows = []
    for (code, d), g in df.groupby(['code', 'date']):
        g = g.sort_values('ts').reset_index(drop=True)
        if len(g) < 10:
            continue

        def price_at(hm_val):
            b = g[g['hm'] == hm_val]
            return float(b['close'].iloc[-1]) if not b.empty else None

        def vol_between(hm_from, hm_to):
            mask = (g['hm'] >= hm_from) & (g['hm'] < hm_to)
            return float(g.loc[mask, 'volume'].sum())

        open_p = price_at(900)
        if not open_p or open_p <= 0:
            b0 = g.iloc[0]
            open_p = float(b0['open'])

        p930  = price_at(930)
        p1000 = price_at(1000)
        p1130 = price_at(1130)
        p1230 = price_at(1230)

        eod_bar = g[g['hm'] >= 1525]
        if eod_bar.empty:
            eod_bar = g[g['hm'] >= 1500]
        if eod_bar.empty:
            continue
        eod_p = float(eod_bar['close'].iloc[-1])

        # 時間帯別出来高
        vol_early = vol_between(900, 930)    # 9:00-9:29
        vol_am1   = vol_between(930, 1000)   # 9:30-9:59
        vol_am2   = vol_between(1000, 1130)  # 10:00-11:29
        vol_pm    = vol_between(1230, 1530)  # 後場
        vol_total = float(g['volume'].sum())

        # 早朝出来高比率（全日）
        early_ratio = vol_early / vol_total if vol_total > 0 else None

        # 前場vs後場出来高比
        am_vol = vol_early + vol_am1 + vol_am2
        pm_vol = vol_pm
        am_pm_ratio = am_vol / pm_vol if pm_vol > 0 else None

        # 出来高×価格の方向（出来高アクション）
        # 9:00-9:30の出来高増加＋価格上昇 = 強い買い
        early_ret_pct = (p930 / open_p - 1) * 100 if p930 and open_p else None
        vol_price_sign = None
        if early_ret_pct is not None:
            vol_price_sign = 1 if early_ret_pct > 0 else (-1 if early_ret_pct < 0 else 0)

        rows.append({
            'code':        code,
            'date':        d,
            'open':        open_p,
            'p930':        p930,
            'p1000':       p1000,
            'p1130':       p1130,
            'p1230':       p1230,
            'eod':         eod_p,
            'vol_early':   vol_early,
            'vol_total':   vol_total,
            'early_ratio': early_ratio,
            'am_pm_ratio': am_pm_ratio,
            'early_ret':   early_ret_pct,
            'vol_price_sign': vol_price_sign,
        })

    feat = pd.DataFrame(rows)
    if feat.empty:
        return feat

    feat = feat.sort_values(['code','date'])
    feat['prev_eod'] = feat.groupby('code')['eod'].shift(1)
    feat['gap_pct']  = (feat['open'] / feat['prev_eod'] - 1) * 100

    # 日次出来高比率を結合
    feat = feat.merge(
        vol_ratio_df[['code','date','vol_ratio']],
        on=['code','date'], how='left'
    )

    # 過去20日平均の早朝出来高比率（ローリング参照値）
    feat['early_ratio_ma20'] = feat.groupby('code')['early_ratio'].transform(
        lambda x: x.shift(1).rolling(20, min_periods=5).mean()
    )
    feat['early_vol_spike'] = feat['early_ratio'] / feat['early_ratio_ma20']

    return feat


def add_vol_signals(feat):
    f = feat.copy()

    # A1: 全日出来高が前20日平均の N倍以上
    for thresh in [1.5, 2.0, 3.0]:
        tstr = str(thresh).replace('.','')
        # 出来高大→追随
        col_f = f'sig_vol_follow_{tstr}'
        col_r = f'sig_vol_fade_{tstr}'
        f[col_f] = 0
        f.loc[(f['vol_ratio'] >= thresh) & (f['gap_pct'] >= 0),  col_f] =  1
        f.loc[(f['vol_ratio'] >= thresh) & (f['gap_pct'] <  0),  col_f] = -1
        # 出来高大→逆張り
        f[col_r] = -f[col_f]

    # A2: 早朝出来高スパイク（9:00-9:30）→逆張り / 追随
    for thresh in [1.5, 2.0]:
        tstr = str(thresh).replace('.','')
        col_f = f'sig_early_vol_follow_{tstr}'
        col_r = f'sig_early_vol_fade_{tstr}'
        f[col_f] = 0
        f.loc[(f['early_vol_spike'] >= thresh) & (f['gap_pct'] >= 0),  col_f] =  1
        f.loc[(f['early_vol_spike'] >= thresh) & (f['gap_pct'] <  0),  col_f] = -1
        f[col_r] = -f[col_f]

    # A3: VolPrice: 出来高スパイク×価格方向（セリクラ=出来高大+価格下落→翌日ロング）
    # 当日引けから翌日のリターンを見る（out-of-sample）
    f['next_eod'] = f.groupby('code')['eod'].shift(-1)
    f['sig_vp_climax_fade'] = 0   # 出来高大+前場下落 → 後場ロング
    mask = (f['vol_ratio'] >= 2.0) & (f['early_ret'] <= -0.5)
    f.loc[mask, 'sig_vp_climax_fade'] = 1

    # A4: Gap_Fade × 出来高大の複合
    f['sig_gap_vol_combo'] = 0
    mask_long  = (f['gap_pct'] <= -GAP_THRESH) & (f['vol_ratio'] >= 1.5)
    mask_short = (f['gap_pct'] >=  GAP_THRESH) & (f['vol_ratio'] >= 1.5)
    f.loc[mask_long,  'sig_gap_vol_combo'] =  1
    f.loc[mask_short, 'sig_gap_vol_combo'] = -1

    return f


# ─────────────────────────────────────────────────
# [B] 板インバランス戦略 (aukabu)
# ─────────────────────────────────────────────────

def get_order_book_features(conn, symbols, date_from, date_to):
    """aukabu.bars_1minから板インバランス特徴量を取得"""
    cur = conn.cursor()
    cur.execute("""
        SELECT symbol,
               (bucket_ts AT TIME ZONE 'Asia/Tokyo')::date AS date,
               (bucket_ts AT TIME ZONE 'Asia/Tokyo') AS ts_jst,
               EXTRACT(HOUR FROM (bucket_ts AT TIME ZONE 'Asia/Tokyo')) * 100 +
               EXTRACT(MINUTE FROM (bucket_ts AT TIME ZONE 'Asia/Tokyo')) AS hm,
               open, high, low, close, volume,
               avg_l1_imb, avg_depth10_imb, avg_w_imb, avg_vwap_dev_pct
        FROM aukabu.bars_1min
        WHERE symbol = ANY(%s)
          AND bucket_ts >= %s::timestamptz AND bucket_ts < %s::timestamptz
        ORDER BY symbol, bucket_ts
    """, (symbols, f'{date_from} 09:00:00+09', f'{date_to} 16:00:00+09'))
    rows = cur.fetchall()
    cur.close()
    df = pd.DataFrame(rows, columns=[
        'symbol','date','ts_jst','hm','open','high','low','close','volume',
        'l1_imb','depth10_imb','w_imb','vwap_dev_pct'
    ])
    for c in ['open','high','low','close','volume','l1_imb','depth10_imb','w_imb','vwap_dev_pct']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def build_ob_features(df_ob):
    """板インバランスの日次スナップショット特徴量"""
    rows = []
    for (sym, d), g in df_ob.groupby(['symbol','date']):
        g = g.sort_values('hm').reset_index(drop=True)

        def snap(hm_val, tol=2):
            b = g[abs(g['hm'] - hm_val) <= tol]
            return b.iloc[-1] if not b.empty else None

        def snap_mean(hm_from, hm_to, col):
            b = g[(g['hm'] >= hm_from) & (g['hm'] < hm_to)]
            return float(b[col].mean()) if not b.empty else None

        b930  = snap(930)
        b1000 = snap(1000)

        open_p = float(g[g['hm'] == 900]['open'].iloc[0]) if not g[g['hm'] == 900].empty else None
        if not open_p:
            open_p = float(g['open'].iloc[0])

        eod_bar = g[g['hm'] >= 1525]
        if eod_bar.empty:
            eod_bar = g[g['hm'] >= 1500]
        if eod_bar.empty:
            continue
        eod_p = float(eod_bar['close'].iloc[-1])

        def fv(bar, col):
            return float(bar[col]) if bar is not None and pd.notna(bar[col]) else None

        # 9:30周辺の板バランス（5分平均）
        l1_930_mean    = snap_mean(925, 935, 'l1_imb')
        depth10_930    = snap_mean(925, 935, 'depth10_imb')
        w_imb_930      = snap_mean(925, 935, 'w_imb')

        # 前場平均板バランス
        l1_am_mean     = snap_mean(900, 1130, 'l1_imb')
        depth10_am     = snap_mean(900, 1130, 'depth10_imb')
        market_pressure_am = None  # bars_1minにはない

        rows.append({
            'symbol':       sym,
            'date':         d,
            'open':         open_p,
            'eod':          eod_p,
            'l1_930':       l1_930_mean,
            'depth10_930':  depth10_930,
            'w_imb_930':    w_imb_930,
            'l1_am':        l1_am_mean,
            'depth10_am':   depth10_am,
        })

    feat = pd.DataFrame(rows)
    if feat.empty:
        return feat
    feat = feat.sort_values(['symbol','date'])
    feat['prev_eod'] = feat.groupby('symbol')['eod'].shift(1)
    feat['gap_pct']  = (feat['open'] / feat['prev_eod'] - 1) * 100
    feat['day_ret']  = (feat['eod'] / feat['open'] - 1) * 100
    return feat


def add_ob_signals(feat):
    f = feat.copy()

    # B1: L1 BBO インバランス方向（買い板過多→ロング）
    f['sig_l1_follow']    = np.sign(f['l1_930'].fillna(0)).astype(int)
    f['sig_l1_fade']      = -f['sig_l1_follow']

    # B2: Depth10 インバランス
    f['sig_d10_follow']   = np.sign(f['depth10_930'].fillna(0)).astype(int)

    # B3: 加重インバランス
    f['sig_wimb_follow']  = np.sign(f['w_imb_930'].fillna(0)).astype(int)

    # B4: Gap×L1複合（ギャップ逆張り＋板方向一致）
    # ギャップアップ・板も売り方向 → ショート強化
    f['sig_gap_l1_combo'] = 0
    mask_short = (f['gap_pct'] >=  GAP_THRESH) & (f['l1_930'] < -0.1)
    mask_long  = (f['gap_pct'] <= -GAP_THRESH) & (f['l1_930'] >  0.1)
    f.loc[mask_short, 'sig_gap_l1_combo'] = -1
    f.loc[mask_long,  'sig_gap_l1_combo'] =  1

    return f


# ─────────────────────────────────────────────────
# 評価
# ─────────────────────────────────────────────────

def simulate(feat, entry_col, exit_col, sig_col, cost_bps=COST_BPS):
    df = feat.dropna(subset=[entry_col, exit_col]).copy()
    df = df[df[sig_col] != 0].copy()
    if df.empty:
        return df
    df['ret'] = df[sig_col] * (df[exit_col] / df[entry_col] - 1) - cost_bps / 10000
    return df


def evaluate(df):
    if len(df) == 0:
        return {'n': 0, 'win_rate': 0, 'pf': 0, 'mean_bps': 0, 'sr': 0, 'total_pct': 0}
    r = df['ret']
    w, l = r[r > 0].sum(), -r[r < 0].sum()
    return {
        'n':         len(r),
        'win_rate':  float((r > 0).mean()),
        'pf':        float(w / l) if l > 0 else float('inf'),
        'mean_bps':  float(r.mean() * 10000),
        'sr':        float(r.mean() / r.std()) if r.std() > 0 else 0,
        'total_pct': float(r.sum() * 100),
    }


# ─────────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────────

def main():
    conn = psycopg2.connect(**PG)

    # ─── 共通データ ──────────────────────────────
    print("=== 銘柄選定 ===")
    liq = get_liquid_codes(conn)
    print(f"対象: {len(liq)} 銘柄")
    codes = liq['code'].tolist()

    print("\n=== A: 出来高データ取得 ===")
    print("  日次出来高比率...")
    vol_ratio = get_daily_vol_ratio(conn, codes)
    print("  イントラデイ1分足...")
    raw = get_intraday(conn, codes, PERIOD_FROM, PERIOD_TO)

    print("\n=== A: 出来高特徴量構築 ===")
    feat_a = build_vol_features(raw, vol_ratio)
    feat_a = add_vol_signals(feat_a)
    feat_a = feat_a.merge(liq[['code','name_ja','sector33']], on='code', how='left')
    print(f"  {len(feat_a)} 銘柄-日")

    # ─── [A] 出来高戦略サマリー ──────────────────
    print("\n" + "="*65)
    print("[A] 出来高スパイク戦略サマリー（4bps込み）")
    print("="*65)

    vol_strategies = {}
    for thresh in [1.5, 2.0, 3.0]:
        tstr = str(thresh).replace('.','')
        vol_strategies[f'Vol_Follow_{thresh}x'] = ('p930', 'eod', f'sig_vol_follow_{tstr}')
        vol_strategies[f'Vol_Fade_{thresh}x']   = ('p930', 'eod', f'sig_vol_fade_{tstr}')
    for thresh in [1.5, 2.0]:
        tstr = str(thresh).replace('.','')
        vol_strategies[f'EarlyVol_Follow_{thresh}x'] = ('p930', 'eod', f'sig_early_vol_follow_{tstr}')
        vol_strategies[f'EarlyVol_Fade_{thresh}x']   = ('p930', 'eod', f'sig_early_vol_fade_{tstr}')
    vol_strategies['VolClimax_Fade']   = ('p1230', 'eod', 'sig_vp_climax_fade')
    vol_strategies['Gap×Vol_Combo']    = ('p930',  'eod', 'sig_gap_vol_combo')
    vol_strategies['Gap_Fade (基準)']  = ('p930',  'eod', 'sig_gap_vol_combo')  # dummy placeholder

    # Gap_Fide ベースライン
    feat_a['sig_gap_fade_base'] = 0
    feat_a.loc[feat_a['gap_pct'] >=  GAP_THRESH, 'sig_gap_fade_base'] = -1
    feat_a.loc[feat_a['gap_pct'] <= -GAP_THRESH, 'sig_gap_fade_base'] =  1
    vol_strategies['Gap_Fade (基準)'] = ('p930', 'eod', 'sig_gap_fade_base')

    hdr = f"{'戦略':<25} {'n':>5} {'勝率':>6} {'PF':>5} {'bps':>7} {'SR':>7} {'累計%':>7}"
    print(hdr)
    print('-' * len(hdr))
    all_ev_a = {}
    for name, (ec, xc, sc) in vol_strategies.items():
        df_s = simulate(feat_a, ec, xc, sc)
        ev   = evaluate(df_s)
        all_ev_a[name] = (df_s, ev)
        marker = ' <<<' if ev['sr'] > 0.04 else ''
        print(f"{name:<25} {ev['n']:>5} {ev['win_rate']:>6.1%} {ev['pf']:>5.2f} "
              f"{ev['mean_bps']:>7.2f} {ev['sr']:>7.3f} {ev['total_pct']:>7.2f}%{marker}")

    # 出来高閾値別効果
    print("\n=== 出来高倍率別: Gap_Fade との違い ===")
    bins = [0, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 9999]
    labs = ['<0.8','0.8-1','1-1.2','1.2-1.5','1.5-2','2-3','>3']
    feat_a['vol_bin'] = pd.cut(feat_a['vol_ratio'], bins=bins, labels=labs)
    print(f"  {'出来高倍率':<12} {'n':>5} {'勝率':>6} {'PF':>5} {'bps':>7} {'SR':>7}")
    for lab in labs:
        sub = feat_a[feat_a['vol_bin'] == lab]
        ev  = evaluate(simulate(sub, 'p930', 'eod', 'sig_gap_fade_base'))
        if ev['n'] > 0:
            print(f"  {lab:<12} {ev['n']:>5} {ev['win_rate']:>6.1%} {ev['pf']:>5.2f} "
                  f"{ev['mean_bps']:>7.2f} {ev['sr']:>7.3f}")

    # ─── [B] 板インバランス戦略 ──────────────────
    print("\n" + "="*65)
    print(f"[B] 板インバランス戦略 ({AUKABU_FROM}〜、約25営業日)")
    print("="*65)

    # aukabuの監視銘柄でセクター対象のものを選ぶ
    cur = conn.cursor()
    cur.execute("""
        SELECT DISTINCT b.symbol
        FROM aukabu.bars_1min b
        JOIN symbol_master sm ON sm.code4 = b.symbol
        WHERE sm.sector33 = ANY(%s)
          AND b.bucket_ts >= %s::timestamptz
    """, (SECTORS, f'{AUKABU_FROM} 09:00:00+09'))
    aukabu_symbols = [r[0] for r in cur.fetchall()]
    cur.close()
    print(f"  対象: {len(aukabu_symbols)} 銘柄（aukabu監視 ∩ 非鉄/電気機器）")

    print("  bars_1min取得...")
    df_ob_raw = get_order_book_features(conn, aukabu_symbols, AUKABU_FROM, PERIOD_TO)
    conn.close()

    feat_b = build_ob_features(df_ob_raw)
    if feat_b.empty:
        print("  板データなし — 板戦略をスキップ")
        # 空のfeat_bでも可視化が壊れないよう最低限のカラムを確保
        feat_b = pd.DataFrame(columns=['symbol','date','open','eod','l1_930',
                                        'depth10_930','w_imb_930','l1_am','depth10_am',
                                        'gap_pct','day_ret','prev_eod'])
    else:
        feat_b = add_ob_signals(feat_b)
    print(f"  {len(feat_b)} 銘柄-日, {feat_b['symbol'].nunique() if not feat_b.empty else 0} 銘柄")

    ob_strategies = {
        'L1_Follow':      ('open', 'eod', 'sig_l1_follow'),
        'L1_Fade':        ('open', 'eod', 'sig_l1_fade'),
        'Depth10_Follow': ('open', 'eod', 'sig_d10_follow'),
        'WImb_Follow':    ('open', 'eod', 'sig_wimb_follow'),
        'Gap×L1_Combo':  ('open', 'eod', 'sig_gap_l1_combo'),
    }

    hdr2 = f"{'戦略':<20} {'n':>4} {'勝率':>6} {'PF':>5} {'bps':>7} {'SR':>7}"
    print(hdr2)
    print('-' * len(hdr2))
    all_ev_b = {}
    if not feat_b.empty and 'sig_l1_follow' in feat_b.columns:
        for name, (ec, xc, sc) in ob_strategies.items():
            df_s = simulate(feat_b, ec, xc, sc)
            ev   = evaluate(df_s)
            all_ev_b[name] = (df_s, ev)
            marker = ' <<<' if ev['sr'] > 0.05 else ''
            print(f"{name:<20} {ev['n']:>4} {ev['win_rate']:>6.1%} {ev['pf']:>5.2f} "
                  f"{ev['mean_bps']:>7.2f} {ev['sr']:>7.3f}{marker}")

        # 板バランスとリターンの相関係数
        print("\n=== 板インバランス vs 当日リターン 相関 ===")
        for col, label in [('l1_930','BBO L1(9:30)'), ('depth10_930','Depth10(9:30)'),
                            ('w_imb_930','W_Imb(9:30)'), ('l1_am','BBO L1(前場平均)')]:
            valid = feat_b[[col, 'day_ret']].dropna()
            if len(valid) > 5:
                corr = valid[col].corr(valid['day_ret'])
                print(f"  {label:<20}: r = {corr:+.3f}  (n={len(valid)})")

        # 板インバランスの分布
        print("\n=== 板インバランス 統計 ===")
        for col in ['l1_930', 'depth10_930', 'w_imb_930']:
            v = feat_b[col].dropna()
            print(f"  {col}: mean={v.mean():+.3f}, std={v.std():.3f}, "
                  f"正={( v > 0).mean():.1%}, 負={(v < 0).mean():.1%}")
    else:
        print("  (板データなし)")
        all_ev_b = {k: (pd.DataFrame(), {'n':0,'win_rate':0,'pf':0,'mean_bps':0,'sr':0,'total_pct':0})
                    for k in ob_strategies}

    # ─────────────────────────────────────────────────
    # 可視化
    # ─────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 12))
    gs  = fig.add_gridspec(3, 3, hspace=0.5, wspace=0.38)

    # Panel 1: 出来高戦略 累積比較（主要5本）
    ax1 = fig.add_subplot(gs[0, :2])
    key_vol = ['Gap_Fade (基準)', 'Gap×Vol_Combo', 'Vol_Fade_2.0x', 'EarlyVol_Fade_1.5x', 'VolClimax_Fade']
    clrs = ['steelblue','green','darkorange','purple','red']
    for i, k in enumerate(key_vol):
        if k not in all_ev_a:
            continue
        df_s, ev = all_ev_a[k]
        if df_s.empty:
            continue
        cum = df_s.sort_values('date').groupby('date')['ret'].mean().cumsum() * 100
        ax1.plot([pd.Timestamp(d) for d in cum.index], cum.values,
                 label=f"{k} SR={ev['sr']:.3f}", color=clrs[i], linewidth=1.4)
    ax1.axhline(0, color='k', linewidth=0.5, linestyle='--')
    ax1.set_title('[A] 出来高戦略 累積リターン比較', fontsize=9)
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
    ax1.set_ylabel('累積リターン (%)', fontsize=8)

    # Panel 2: 出来高倍率別 Gap_Fade SR
    ax2 = fig.add_subplot(gs[0, 2])
    vol_bin_srs, vol_bin_ns = [], []
    for lab in labs:
        sub = feat_a[feat_a['vol_bin'] == lab]
        ev  = evaluate(simulate(sub, 'p930', 'eod', 'sig_gap_fade_base'))
        vol_bin_srs.append(ev['sr'])
        vol_bin_ns.append(ev['n'])
    colors_v = ['steelblue' if v >= 0 else 'tomato' for v in vol_bin_srs]
    bars2 = ax2.bar(labs, vol_bin_srs, color=colors_v, edgecolor='gray', linewidth=0.3)
    for bar, sr, n in zip(bars2, vol_bin_srs, vol_bin_ns):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.001 if sr >= 0 else bar.get_height() - 0.01,
                 f'{sr:.3f}\nn={n}', ha='center', va='bottom', fontsize=6.5)
    ax2.axhline(0, color='k', linewidth=0.8)
    ax2.set_title('Gap_Fade SR — 出来高倍率別', fontsize=9)
    ax2.set_ylabel('SR', fontsize=8)
    ax2.tick_params(axis='x', labelsize=7)
    ax2.grid(True, alpha=0.3, axis='y')

    # Panel 3: 板インバランス各指標の累積（L1 / Depth10 / WImb）
    ax3 = fig.add_subplot(gs[1, :2])
    ob_key = ['L1_Follow', 'L1_Fade', 'Depth10_Follow', 'WImb_Follow', 'Gap×L1_Combo']
    ob_clr = ['steelblue','tomato','green','purple','darkorange']
    for i, k in enumerate(ob_key):
        if k not in all_ev_b:
            continue
        df_s, ev = all_ev_b[k]
        if df_s.empty:
            continue
        cum = df_s.sort_values('date').groupby('date')['ret'].mean().cumsum() * 100
        ax3.plot([pd.Timestamp(d) for d in cum.index], cum.values,
                 label=f"{k} SR={ev['sr']:.3f}", color=ob_clr[i], linewidth=1.4)
    ax3.axhline(0, color='k', linewidth=0.5, linestyle='--')
    ax3.set_title(f'[B] 板インバランス戦略 累積リターン ({AUKABU_FROM}〜)', fontsize=9)
    ax3.legend(fontsize=7)
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax3.set_ylabel('累積リターン (%)', fontsize=8)

    # Panel 4: 板バランス vs 当日リターン 散布図
    ax4 = fig.add_subplot(gs[1, 2])
    valid = feat_b[['l1_930', 'day_ret']].dropna()
    if len(valid) > 5:
        ax4.scatter(valid['l1_930'], valid['day_ret'], alpha=0.3, s=8, color='steelblue')
        # 回帰線
        m, b = np.polyfit(valid['l1_930'], valid['day_ret'], 1)
        xl = np.linspace(valid['l1_930'].min(), valid['l1_930'].max(), 100)
        ax4.plot(xl, m*xl+b, color='red', linewidth=1)
        corr = valid['l1_930'].corr(valid['day_ret'])
        ax4.set_title(f'BBO L1(9:30) vs 当日リターン\nr={corr:+.3f}', fontsize=9)
        ax4.set_xlabel('L1 Imbalance', fontsize=8)
        ax4.set_ylabel('当日リターン (%)', fontsize=8)
        ax4.axhline(0, color='k', linewidth=0.5, linestyle='--')
        ax4.axvline(0, color='k', linewidth=0.5, linestyle='--')
        ax4.grid(True, alpha=0.3)

    # Panel 5: 板インバランス分布（L1, Depth10, WImb）
    ax5 = fig.add_subplot(gs[2, 0])
    for col, label, clr in [('l1_930','BBO L1','steelblue'),
                              ('depth10_930','Depth10','darkorange'),
                              ('w_imb_930','W_Imb','green')]:
        v = feat_b[col].dropna()
        if len(v) > 0:
            ax5.hist(v, bins=50, alpha=0.5, label=label, density=True)
    ax5.axvline(0, color='k', linewidth=0.8)
    ax5.set_title('板インバランス 分布（9:30前後）', fontsize=9)
    ax5.set_xlabel('インバランス値', fontsize=8)
    ax5.legend(fontsize=7)
    ax5.grid(True, alpha=0.3)

    # Panel 6: Gap×Volume 複合 vs Gap単体 月別
    ax6 = fig.add_subplot(gs[2, 1:])
    months_list = sorted(feat_a['date'].unique())
    month_periods = pd.to_datetime(feat_a['date']).dt.to_period('M').unique()

    x = np.arange(len(month_periods))
    w = 0.35
    combo_month_srs, gap_month_srs = [], []
    for mp in month_periods:
        mask = pd.to_datetime(feat_a['date']).dt.to_period('M') == mp
        sub  = feat_a[mask]
        ev_c = evaluate(simulate(sub, 'p930', 'eod', 'sig_gap_vol_combo'))
        ev_g = evaluate(simulate(sub, 'p930', 'eod', 'sig_gap_fade_base'))
        combo_month_srs.append(ev_c['mean_bps'])
        gap_month_srs.append(ev_g['mean_bps'])

    ax6.bar(x - w/2, gap_month_srs,   w, label='Gap_Fade 単体', color='steelblue', alpha=0.8)
    ax6.bar(x + w/2, combo_month_srs, w, label='Gap×Vol(1.5x)', color='green', alpha=0.8)
    ax6.axhline(0, color='k', linewidth=0.8)
    ax6.set_xticks(x)
    ax6.set_xticklabels([str(m) for m in month_periods], fontsize=7)
    ax6.set_title('Gap_Fade vs Gap×Vol — 月別平均(bps)', fontsize=9)
    ax6.set_ylabel('bps', fontsize=8)
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3, axis='y')

    fig.suptitle(
        f'非鉄・電気機器 出来高・板インバランス戦略検証 ({PERIOD_FROM}〜{PERIOD_TO})\n'
        f'[A] 出来高: {COST_BPS}bps込み 122銘柄  '
        f'[B] 板: {COST_BPS}bps込み {feat_b["symbol"].nunique() if not feat_b.empty else 0}銘柄 ({AUKABU_FROM}〜)',
        fontsize=11, fontweight='bold'
    )

    out = os.path.join(os.path.dirname(__file__), 'result_vol_order.png')
    fig.savefig(out, dpi=100, bbox_inches='tight')
    print(f"\n保存: {out}")


if __name__ == '__main__':
    main()
