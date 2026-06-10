"""
20戦略エッジ分析 - 直近3ヶ月 vs 1年比較
===========================================
仮説: 2年全期間ではレジームが混在してエッジが埋もれるが、
      直近3ヶ月のセクター内で絞ると有効なエッジが見つかるか

戦略グループ:
  A. マクロ連動 (SOX・銅・USD/JPY・NVDA)
  B. セクター内モメンタム
  C. 平均回帰
  D. 曜日・カレンダー
  E. 信用需給
  F. クロスアセット (ADRリード)

実行: python analyze.py
出力: results_detail.csv, results_summary.csv
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import pandas as pd
import numpy as np
import psycopg2
from scipy import stats
from datetime import date, timedelta
import warnings
warnings.filterwarnings('ignore')

DB_URL = "postgresql://postgres@localhost/market_data"
OUT_DIR = "/mnt/d/Root/ClaudeCode/01_Trading/japan-stocks/analyses/20260605_sector_edge_research"

# ─── ユニバース ────────────────────────────────────────────────────────────
SECTORS = {
    'SEMI':       ['80350','68570','69200','61460','77350','69630','65260',
                   '285A0','99840','40620','40630','67230','65250','77410','34360'],
    'NONFERROUS': ['57130','57110','57060','57140','50160','58010','58020','58030'],
    'ELEC_PARTS': ['69810','67620','69710','69760'],
    'INDUSTRIAL': ['70110','70130','70120','65030'],
    'AUTO':       ['72030','72670'],
    'BANK':       ['83060','83160','84110'],
    'SHOSHA':     ['80580','80310'],
}
ALL_CODES = sorted({c for codes in SECTORS.values() for c in codes})

PERIODS = {'3M': 63, '1Y': 252, '2Y': 504}

# ─── DB ───────────────────────────────────────────────────────────────────
def get_conn():
    return psycopg2.connect(DB_URL)

def load_daily(lookback=600):
    code_list = ",".join(f"'{c}'" for c in ALL_CODES)
    sql = f"""
        SELECT code, date, adj_open AS open, adj_high AS high,
               adj_low AS low, adj_close AS close, adj_volume AS volume
        FROM public.stocks_daily
        WHERE code IN ({code_list})
          AND date >= CURRENT_DATE - {lookback}
        ORDER BY code, date
    """
    with get_conn() as conn:
        return pd.read_sql(sql, conn, parse_dates=['date'])

def load_macro(symbols, lookback=600):
    sym_list = ",".join(f"'{s}'" for s in symbols)
    sql = f"""
        SELECT symbol, trade_date AS date, close
        FROM macro.daily_ohlcv
        WHERE symbol IN ({sym_list})
          AND trade_date >= CURRENT_DATE - {lookback}
        ORDER BY symbol, trade_date
    """
    with get_conn() as conn:
        return pd.read_sql(sql, conn, parse_dates=['date'])

def load_margin(lookback=600):
    code_list = ",".join(f"'{c}'" for c in ALL_CODES)
    sql = f"""
        SELECT code, date,
               long_vol::float AS long_vol,
               shrt_vol::float AS shrt_vol,
               ROUND(long_vol::numeric / NULLIF(shrt_vol,0), 3)::float AS bairitu
        FROM public.jquants_margin_interest
        WHERE code IN ({code_list})
          AND date >= CURRENT_DATE - {lookback}
        ORDER BY code, date
    """
    with get_conn() as conn:
        return pd.read_sql(sql, conn, parse_dates=['date'])

def load_names():
    code_list = ",".join(f"'{c}'" for c in ALL_CODES)
    sql = f"SELECT code5, name_ja FROM public.symbol_master WHERE code5 IN ({code_list})"
    with get_conn() as conn:
        df = pd.read_sql(sql, conn)
    return dict(zip(df['code5'], df['name_ja']))

# ─── 統計 ─────────────────────────────────────────────────────────────────
def calc_stats(trade_rets):
    r = pd.Series(trade_rets).dropna()
    n = len(r)
    if n < 8:
        return {'n': n, 'win_rate': np.nan, 'mean_ret': np.nan,
                'sharpe': np.nan, 't_stat': np.nan, 'p_value': np.nan,
                'edge_class': 'insufficient'}
    wr   = (r > 0).mean()
    mr   = r.mean()
    std  = r.std()
    sh   = mr / std * np.sqrt(252) if std > 0 else np.nan
    t, p = stats.ttest_1samp(r, 0)
    return {'n': n, 'win_rate': wr, 'mean_ret': mr, 'sharpe': sh,
            't_stat': t, 'p_value': p, 'edge_class': classify(t, mr)}

def classify(t, mr):
    if pd.isna(t): return 'insufficient'
    if t >  2.0 and mr > 0: return 'strong_pos'
    if t >  1.5 and mr > 0: return 'weak_pos'
    if t < -2.0 and mr < 0: return 'strong_neg'
    if t < -1.5 and mr < 0: return 'weak_neg'
    return 'noise'

# ─── データロード ──────────────────────────────────────────────────────────
print("データ取得中...")
daily = load_daily()
macro_raw = load_macro(['.SOX','HGc1','JPY=','VXc1','NQc1',
                        'NVDA','TSM','ASML',
                        'ADR_8035','ADR_6920','ADR_6857',
                        'US10YT=RR'])
margin_df = load_margin()
CODE_NAME = load_names()
print(f"  日足: {len(daily)}行 {daily['code'].nunique()}銘柄  "
      f"期間: {daily['date'].min().date()} ~ {daily['date'].max().date()}")

macro = macro_raw.pivot(index='date', columns='symbol', values='close').sort_index().ffill()

# pivot
px_close = daily.pivot(index='date', columns='code', values='close')
px_open  = daily.pivot(index='date', columns='code', values='open')

semi_codes = [c for c in SECTORS['SEMI']       if c in px_close.columns]
nf_codes   = [c for c in SECTORS['NONFERROUS'] if c in px_close.columns]
ep_codes   = [c for c in SECTORS['ELEC_PARTS'] if c in px_close.columns]
ind_codes  = [c for c in SECTORS['INDUSTRIAL'] if c in px_close.columns]
auto_codes = [c for c in SECTORS['AUTO']       if c in px_close.columns]
bank_codes = [c for c in SECTORS['BANK']       if c in px_close.columns]
print(f"  SEMI:{len(semi_codes)} NF:{len(nf_codes)} EP:{len(ep_codes)} "
      f"IND:{len(ind_codes)} AUTO:{len(auto_codes)} BANK:{len(bank_codes)}")

# リターン計算
co_ret      = px_close.pct_change()              # close→close 当日
co_ret_next = co_ret.shift(-1)                   # close→close 翌日
on_ret      = (px_close - px_open) / px_open     # open→close 当日
on_ret_next = on_ret.shift(-1)                   # open→close 翌日

# セクター平均
semi_co_next = co_ret_next[semi_codes].mean(axis=1)
semi_on_next = on_ret_next[semi_codes].mean(axis=1)
nf_co_next   = co_ret_next[nf_codes].mean(axis=1)
nf_on_next   = on_ret_next[nf_codes].mean(axis=1)
auto_co_next = co_ret_next[auto_codes].mean(axis=1)

# マクロリターン (米国時間差を吸収するため reindex+ffill)
def macro_ret(sym):
    if sym not in macro.columns: return None
    return macro[sym].pct_change().reindex(px_close.index, method='ffill')

sox_ret  = macro_ret('.SOX')
cu_ret   = macro_ret('HGc1')
jpy_ret  = macro_ret('JPY=')
nvda_ret = macro_ret('NVDA')
adr_8035_ret = macro_ret('ADR_8035')
adr_6920_ret = macro_ret('ADR_6920')

# ─── 戦略実行ヘルパー ─────────────────────────────────────────────────────
results = []

def run(sid, name, group, target_ret, signal_bool):
    """signal_bool: pd.Series[bool], target_ret: pd.Series[float]"""
    for label, days in PERIODS.items():
        cutoff = px_close.index.max() - pd.Timedelta(days=int(days * 1.5))
        sig = signal_bool[signal_bool.index >= cutoff]
        ret = target_ret.reindex(sig.index)
        trade_rets = ret[sig == True].dropna().tolist()
        s = calc_stats(trade_rets)
        results.append({'id': sid, 'name': name, 'group': group,
                        'period': label, **s})

def rolling_window_top(ret_matrix, codes, n_roll, n_top, fwd_ret, period_days):
    """各日: n_roll日リターン上位n_top銘柄に翌日エントリー"""
    cutoff = px_close.index.max() - pd.Timedelta(days=int(period_days * 1.5))
    roll = ret_matrix[codes].rolling(n_roll).sum()
    trade_rets = []
    for dt in roll.index[roll.index >= cutoff][:-1]:
        row = roll.loc[dt].dropna()
        if len(row) < n_top + 1: continue
        top = row.nlargest(n_top).index.tolist()
        next_mask = fwd_ret.index > dt
        if not next_mask.any(): continue
        nd = fwd_ret.index[next_mask][0]
        for c in top:
            if c in fwd_ret.columns:
                v = fwd_ret.loc[nd, c]
                if pd.notna(v): trade_rets.append(v)
    return trade_rets

def per_stock_condition(codes, cond_func, fwd_ret_matrix, period_days):
    """各日・各銘柄で cond_func(ret_series_up_to_dt) → bool を評価"""
    cutoff = px_close.index.max() - pd.Timedelta(days=int(period_days * 1.5))
    trade_rets = []
    for dt in co_ret.index[co_ret.index >= cutoff][:-1]:
        next_mask = fwd_ret_matrix.index > dt
        if not next_mask.any(): continue
        nd = fwd_ret_matrix.index[next_mask][0]
        for c in codes:
            if c not in co_ret.columns: continue
            r_today = co_ret.loc[dt, c]
            if not cond_func(r_today): continue
            v = fwd_ret_matrix.loc[nd, c] if c in fwd_ret_matrix.columns else np.nan
            if pd.notna(v): trade_rets.append(v)
    return trade_rets

# =====================================================================
# A. マクロ連動
# =====================================================================
print("\n[A] マクロ連動")

# A1: SOX前日+1%超 → 半導体翌日CO
if sox_ret is not None:
    run('A1', 'SOX>+1%→半導体翌日CO', 'A',
        semi_co_next, sox_ret > 0.01)

# A2: SOX3日連続陽線 → 半導体翌日CO
if sox_ret is not None:
    run('A2', 'SOX3日連続陽線→半導体翌日CO', 'A',
        semi_co_next, (sox_ret > 0).rolling(3).sum() == 3)

# A3: SOX前日-1%超下落 → 半導体翌日リバウンド
if sox_ret is not None:
    run('A3', 'SOX<-1%→半導体翌日リバウンド', 'A',
        semi_co_next, sox_ret < -0.01)

# A4: 銅3日連続上昇 → 非鉄翌日CO
if cu_ret is not None:
    run('A4', '銅3日連続上昇→非鉄翌日CO', 'A',
        nf_co_next, (cu_ret > 0).rolling(3).sum() == 3)

# A5: 円高3日連続(JPY=上昇) → 自動車翌日CO (マイナス検証)
if jpy_ret is not None:
    run('A5', '円高3日連続→自動車翌日CO', 'A',
        auto_co_next, (jpy_ret > 0).rolling(3).sum() == 3)

# A6: NVDA前日+2%超 → 半導体翌日CO
if nvda_ret is not None:
    run('A6', 'NVDA>+2%→半導体翌日CO', 'A',
        semi_co_next, nvda_ret > 0.02)

print(f"  A1-A6 done")

# =====================================================================
# B. セクター内モメンタム
# =====================================================================
print("\n[B] セクター内モメンタム")

# B7: 半導体5日MOM上位2 → 翌日CO
for label, days in PERIODS.items():
    tr = rolling_window_top(co_ret, semi_codes, 5, 2, co_ret_next, days)
    s = calc_stats(tr)
    results.append({'id':'B7','name':'半導体5日MOM上位2→翌日CO','group':'B','period':label,**s})

# B8: 半導体20日MOM上位2 → 翌週CO
for label, days in PERIODS.items():
    tr = rolling_window_top(co_ret, semi_codes, 20, 2,
                            co_ret[semi_codes].rolling(5).sum().shift(-5), days)
    s = calc_stats(tr)
    results.append({'id':'B8','name':'半導体20日MOM上位2→翌5日CO','group':'B','period':label,**s})

# B9: 非鉄5日MOM上位2 → 翌日CO
for label, days in PERIODS.items():
    tr = rolling_window_top(co_ret, nf_codes, 5, 2, co_ret_next, days)
    s = calc_stats(tr)
    results.append({'id':'B9','name':'非鉄5日MOM上位2→翌日CO','group':'B','period':label,**s})

# B10: 半導体セクター平均3日+2%超 → 翌日継続
semi_avg = co_ret[semi_codes].mean(axis=1)
run('B10', '半導体3日+2%超→翌日継続', 'B',
    semi_co_next, semi_avg.rolling(3).sum() > 0.02)

print("  B7-B10 done")

# =====================================================================
# C. 平均回帰
# =====================================================================
print("\n[C] 平均回帰")

# C11: 半導体内Zスコア下位2 → 翌5日リバウンド
semi_z = (co_ret[semi_codes] - co_ret[semi_codes].rolling(20).mean()) \
       / co_ret[semi_codes].rolling(20).std().replace(0, np.nan)
fwd5 = co_ret[semi_codes].rolling(5).sum().shift(-5)
for label, days in PERIODS.items():
    cutoff = px_close.index.max() - pd.Timedelta(days=int(days * 1.5))
    tr = []
    for dt in semi_z.index[semi_z.index >= cutoff][:-5]:
        row = semi_z.loc[dt].dropna()
        if len(row) < 4: continue
        for c in row.nsmallest(2).index:
            if c in fwd5.columns and dt in fwd5.index:
                v = fwd5.loc[dt, c]
                if pd.notna(v): tr.append(v)
    results.append({'id':'C11','name':'半導体Zスコア下位2→翌5日リバ','group':'C','period':label,**calc_stats(tr)})

# C12: 半導体個別-3%以上(セクターフラット) → 翌日リバ
semi_avg_abs = co_ret[semi_codes].mean(axis=1)
for label, days in PERIODS.items():
    cutoff = px_close.index.max() - pd.Timedelta(days=int(days * 1.5))
    tr = []
    dates = co_ret.index[co_ret.index >= cutoff][:-1]
    for dt in dates:
        if abs(semi_avg_abs.get(dt, np.nan)) > 0.01: continue   # セクターが動いている日は除外
        for c in semi_codes:
            if c not in co_ret.columns: continue
            r = co_ret.loc[dt, c] if dt in co_ret.index else np.nan
            if pd.notna(r) and r < -0.03:
                v = co_ret_next.loc[dt, c] if (dt in co_ret_next.index and c in co_ret_next.columns) else np.nan
                if pd.notna(v): tr.append(v)
    results.append({'id':'C12','name':'半導体個別-3%(セクターフラット)→翌日リバ','group':'C','period':label,**calc_stats(tr)})

# C13: 非鉄個別-3%以上 → 翌日リバ
nf_avg_abs = co_ret[nf_codes].mean(axis=1)
for label, days in PERIODS.items():
    cutoff = px_close.index.max() - pd.Timedelta(days=int(days * 1.5))
    tr = []
    for dt in co_ret.index[co_ret.index >= cutoff][:-1]:
        if abs(nf_avg_abs.get(dt, np.nan)) > 0.015: continue
        for c in nf_codes:
            if c not in co_ret.columns: continue
            r = co_ret.loc[dt, c] if dt in co_ret.index else np.nan
            if pd.notna(r) and r < -0.03:
                v = co_ret_next.loc[dt, c] if (dt in co_ret_next.index and c in co_ret_next.columns) else np.nan
                if pd.notna(v): tr.append(v)
    results.append({'id':'C13','name':'非鉄個別-3%(セクターフラット)→翌日リバ','group':'C','period':label,**calc_stats(tr)})

print("  C11-C13 done")

# =====================================================================
# D. 曜日パターン
# =====================================================================
print("\n[D] 曜日パターン")

dow = pd.Series(px_close.index.dayofweek, index=px_close.index)
DOW_NAMES = {0:'月', 1:'火', 2:'水', 3:'木', 4:'金'}

for d, dn in DOW_NAMES.items():
    run(f'D{14+d}', f'{dn}曜日の半導体翌日ON', 'D', semi_on_next, dow == d)

# 非鉄の月・金
run('D19', '月曜日の非鉄翌日ON', 'D', nf_on_next, dow == 0)
run('D20', '金曜日の非鉄翌日ON', 'D', nf_on_next, dow == 4)
print("  D14-D20 done")

# =====================================================================
# E. 信用需給
# =====================================================================
print("\n[E] 信用需給")

if not margin_df.empty:
    margin_pivot = margin_df.pivot(index='date', columns='code', values='bairitu')
    margin_pivot = margin_pivot.reindex(px_close.index, method='ffill')
    all_margin_codes = [c for c in (semi_codes + nf_codes) if c in margin_pivot.columns]
    fwd5_all = co_ret[semi_codes + nf_codes].rolling(5).sum().shift(-5)

    for label, days in PERIODS.items():
        cutoff = px_close.index.max() - pd.Timedelta(days=int(days * 1.5))
        tr_high, tr_low = [], []
        for dt in margin_pivot.index[margin_pivot.index >= cutoff][:-5]:
            row = margin_pivot.loc[dt, all_margin_codes].dropna()
            for c, b in row.items():
                v = fwd5_all.loc[dt, c] if (dt in fwd5_all.index and c in fwd5_all.columns) else np.nan
                if pd.isna(v): continue
                if b > 5:   tr_high.append(v)
                if b < 1.5: tr_low.append(v)
        results.append({'id':'E21','name':'信用倍率>5倍→翌5日CO','group':'E','period':label,**calc_stats(tr_high)})
        results.append({'id':'E22','name':'信用倍率<1.5倍→翌5日CO','group':'E','period':label,**calc_stats(tr_low)})
    print("  E21-E22 done")

# =====================================================================
# F. ADRリード
# =====================================================================
print("\n[F] ADRリード")

if adr_8035_ret is not None:
    code_8035_on = on_ret_next['80350'] if '80350' in on_ret_next.columns else None
    if code_8035_on is not None:
        run('F23', 'ADR_8035>+1%→8035翌日ON', 'F',
            code_8035_on, adr_8035_ret > 0.01)
        run('F24', 'ADR_8035<-1%→8035翌日ON', 'F',
            code_8035_on, adr_8035_ret < -0.01)

if adr_6920_ret is not None:
    code_6920_on = on_ret_next['69200'] if '69200' in on_ret_next.columns else None
    if code_6920_on is not None:
        run('F25', 'ADR_6920>+1%→6920翌日ON', 'F',
            code_6920_on, adr_6920_ret > 0.01)

if nvda_ret is not None:
    run('F26', 'NVDA>+1.5%→半導体翌日ON', 'F',
        semi_on_next, nvda_ret > 0.015)
    run('F27', 'NVDA<-1.5%→半導体翌日ON', 'F',
        semi_on_next, nvda_ret < -0.015)

print("  F23-F27 done")

# ─── 結果整形・表示 ────────────────────────────────────────────────────────
df = pd.DataFrame(results)

MARKER = {
    'strong_pos':   '★★ 強↑',
    'weak_pos':     '★  弱↑',
    'noise':        '   中立',
    'weak_neg':     '▼  弱↓',
    'strong_neg':   '▼▼ 強↓',
    'insufficient': '   N不足',
}

def fmt_row(row):
    n  = int(row['n'])  if pd.notna(row['n'])       else 0
    wr = f"{row['win_rate']*100:5.1f}%" if pd.notna(row['win_rate']) else '    –'
    mr = f"{row['mean_ret']*100:+.3f}%" if pd.notna(row['mean_ret']) else '     –'
    t  = f"{row['t_stat']:+.2f}"        if pd.notna(row['t_stat'])  else '    –'
    sh = f"{row['sharpe']:+.2f}"        if pd.notna(row.get('sharpe')) else '    –'
    ec = MARKER.get(row['edge_class'], row['edge_class'])
    return n, wr, mr, t, sh, ec

print("\n" + "="*90)
print("  20戦略エッジ分析結果   （N=サンプル数、期待値=1トレード平均リターン）")
print("="*90)
print(f"  {'ID':<4}  {'戦略':<42}  {'期間':>3}  {'N':>4}  {'勝率':>6}  {'期待値':>8}  {'t値':>5}  {'Sharpe':>6}  判定")
print("-"*90)

for sid in sorted(df['id'].unique()):
    rows = df[df['id'] == sid].set_index('period')
    name = rows.iloc[0]['name']
    first = True
    for plbl in ['3M','1Y','2Y']:
        if plbl not in rows.index: continue
        row = rows.loc[plbl]
        n, wr, mr, t, sh, ec = fmt_row(row)
        prefix = f"  {sid:<4}  {name:<42}" if first else f"  {'':4}  {'':42}"
        print(f"{prefix}  {plbl:>3}  {n:>4}  {wr:>6}  {mr:>8}  {t:>5}  {sh:>6}  {ec}")
        first = False
    print()

# ─── 有望戦略サマリー ─────────────────────────────────────────────────────
print("="*90)
print("  有望戦略（3Mで weak_pos 以上）")
print("="*90)
df_3m = df[df['period'] == '3M']
promising = df_3m[df_3m['edge_class'].isin(['strong_pos','weak_pos','strong_neg','weak_neg'])] \
              .sort_values('t_stat', ascending=False)

if promising.empty:
    print("  なし（全戦略がnoise / N不足）")
else:
    for _, row in promising.iterrows():
        n, wr, mr, t, sh, ec = fmt_row(row)
        print(f"  [{row['id']}] {row['name']:<44} N={n:>3} 勝率={wr} 期待値={mr} t={t} {ec}")

# ─── レジーム依存エッジ（3M有意、1Y noise）─────────────────────────────
print("\n" + "="*90)
print("  レジーム依存エッジ（3Mで有意 / 1Yでnoise） ← 今の相場特有")
print("="*90)
df_by_period = df.set_index(['id','period'])
regime = []
for sid in df['id'].unique():
    try:
        ec3 = df_by_period.loc[(sid,'3M'), 'edge_class']
        ec1 = df_by_period.loc[(sid,'1Y'), 'edge_class']
        if ec3 in ('strong_pos','weak_pos','strong_neg','weak_neg') and ec1 == 'noise':
            regime.append(sid)
    except KeyError:
        pass

if not regime:
    print("  なし")
else:
    for sid in regime:
        r3 = df_by_period.loc[(sid,'3M')]
        r1 = df_by_period.loc[(sid,'1Y')]
        _, wr3, mr3, t3, sh3, ec3 = fmt_row(r3)
        _, wr1, mr1, t1, sh1, ec1 = fmt_row(r1)
        print(f"  [{sid}] {r3['name']}")
        print(f"       3M: 勝率={wr3} 期待値={mr3} t={t3} Sharpe={sh3}  {ec3}")
        print(f"       1Y: 勝率={wr1} 期待値={mr1} t={t1} Sharpe={sh1}  {ec1}")
        print()

# ─── CSV保存 ─────────────────────────────────────────────────────────────
df.to_csv(f"{OUT_DIR}/results_detail.csv", index=False)
print(f"\n保存: {OUT_DIR}/results_detail.csv")
print("完了")
