"""
時間軸別ポートフォリオ シミュレーション
=========================================
イントラ / 日次 / スイング の3層に分けて
・各戦略の日次P&L系列を構築
・戦略間相関行列
・最適資金配分（1/4 Kelly + リスクパリティ）
・合成ポートフォリオのパフォーマンス統計

実行: python3 portfolio_sim.py
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import pandas as pd
import numpy as np
import psycopg2
from scipy import stats, optimize
import warnings
warnings.filterwarnings('ignore')

DB_URL  = "postgresql://postgres@localhost/market_data"
OUT_DIR = "/mnt/d/Root/ClaudeCode/01_Trading/japan-stocks/analyses/20260605_sector_edge_research"

# ─── ユニバース ────────────────────────────────────────────────────────────
SEMI_C5  = ['80350','68570','69200','61460','77350','69630','65260',
            '285A0','99840','40620','40630','67230','65250','77410','34360']
NF_C5    = ['57130','57110','57060','57140','50160','58010','58020','58030']
ALL_C5   = sorted(set(SEMI_C5 + NF_C5))

# ─── データ ───────────────────────────────────────────────────────────────
def get_conn(): return psycopg2.connect(DB_URL)

def load_daily():
    cl = ",".join(f"'{c}'" for c in ALL_C5)
    sql = f"""
        SELECT code, date, adj_open AS open, adj_close AS close
        FROM public.stocks_daily
        WHERE code IN ({cl}) AND date >= CURRENT_DATE - 600
        ORDER BY code, date
    """
    with get_conn() as c: return pd.read_sql(sql, c, parse_dates=['date'])

def load_macro():
    sql = """
        SELECT symbol, trade_date AS date, close
        FROM macro.daily_ohlcv
        WHERE symbol IN ('.SOX','HGc1','JPY=','NVDA','ADR_8035','ADR_6920')
          AND trade_date >= CURRENT_DATE - 600
        ORDER BY symbol, trade_date
    """
    with get_conn() as c: return pd.read_sql(sql, c, parse_dates=['date'])

def load_margin():
    cl = ",".join(f"'{c}'" for c in ALL_C5)
    sql = f"""
        SELECT code, date,
               ROUND(long_vol::numeric / NULLIF(shrt_vol,0), 2)::float AS bairitu
        FROM public.jquants_margin_interest
        WHERE code IN ({cl}) AND date >= CURRENT_DATE - 600
        ORDER BY code, date
    """
    with get_conn() as c: return pd.read_sql(sql, c, parse_dates=['date'])

print("データ取得中...")
daily  = load_daily()
mac_r  = load_macro()
margin = load_margin()

px_c = daily.pivot(index='date', columns='code', values='close')
px_o = daily.pivot(index='date', columns='code', values='open')
dates = px_c.index

semi_c = [c for c in SEMI_C5 if c in px_c.columns]
nf_c   = [c for c in NF_C5   if c in px_c.columns]

co_ret      = px_c.pct_change()
co_ret_next = co_ret.shift(-1)
on_ret_next = ((px_c - px_o) / px_o).shift(-1)   # 翌日の寄→引

macro = mac_r.pivot(index='date', columns='symbol', values='close').ffill()
sox   = macro['.SOX'].pct_change().reindex(dates, method='ffill') if '.SOX' in macro.columns else None
nvda  = macro['NVDA'].pct_change().reindex(dates, method='ffill') if 'NVDA'  in macro.columns else None
cu    = macro['HGc1'].pct_change().reindex(dates, method='ffill') if 'HGc1'  in macro.columns else None

margin_piv = margin.pivot(index='date', columns='code', values='bairitu') \
                   .reindex(dates, method='ffill')

# ─── 戦略別 日次P&L 系列 ─────────────────────────────────────────────────
# 各戦略: シグナル日の「平均リターン」を1単位として記録
# 実際の取引ではポジションサイズで調整する

def series_signal_ret(signal_bool, target_ret_series):
    """シグナル日に target_ret_series を受け取る、それ以外は0"""
    sig = signal_bool.reindex(target_ret_series.index).fillna(False)
    return target_ret_series.where(sig, 0).fillna(0)

def top_n_daily_ret(ret_mat, codes, roll, n, fwd_ret_mat):
    """各日: roll日リターン上位n銘柄の翌日平均リターン"""
    rolling_r = ret_mat[codes].rolling(roll).sum()
    out = pd.Series(0.0, index=ret_mat.index)
    for dt in rolling_r.index[:-1]:
        row = rolling_r.loc[dt].dropna()
        if len(row) < n+1: continue
        top = row.nlargest(n).index.tolist()
        next_mask = fwd_ret_mat.index > dt
        if not next_mask.any(): continue
        nd = fwd_ret_mat.index[next_mask][0]
        vals = [fwd_ret_mat.loc[nd, c] for c in top if c in fwd_ret_mat.columns and pd.notna(fwd_ret_mat.loc[nd, c])]
        if vals: out.loc[dt] = np.mean(vals)
    return out

def zscore_bottom_n_weekly(ret_mat, codes, n, fwd5_mat):
    """Zスコア下位n銘柄の翌5日平均リターン (5日毎に更新)"""
    roll20_m = ret_mat[codes].rolling(20).mean()
    roll20_s = ret_mat[codes].rolling(20).std().replace(0, np.nan)
    z = (ret_mat[codes] - roll20_m) / roll20_s
    fwd5 = fwd5_mat[codes].copy()
    out = pd.Series(0.0, index=ret_mat.index)
    prev_entry = None
    for dt in z.index[:-5]:
        if prev_entry and (dt - prev_entry).days < 5: continue
        row = z.loc[dt].dropna()
        if len(row) < n+1: continue
        bot = row.nsmallest(n).index.tolist()
        vals = [fwd5.loc[dt, c] for c in bot if c in fwd5.columns and pd.notna(fwd5.loc[dt, c])]
        if vals:
            out.loc[dt] = np.mean(vals)
            prev_entry = dt
    return out

print("戦略シミュレーション中...")

semi_co_avg  = co_ret_next[semi_c].mean(axis=1)
semi_on_avg  = on_ret_next[semi_c].mean(axis=1)
nf_co_avg    = co_ret_next[nf_c].mean(axis=1)
semi_avg_co  = co_ret[semi_c].mean(axis=1)
nf_avg_co    = co_ret[nf_c].mean(axis=1)

fwd5_semi    = co_ret[semi_c].rolling(5).sum().shift(-5)
fwd5_all     = co_ret[semi_c+nf_c].rolling(5).sum().shift(-5)

dow = pd.Series(dates.dayofweek, index=dates)

# ======================================================================
# ① イントラデイ層  (翌日ON: 寄り→引け, 当日中でポジション完結)
# ======================================================================
strats_intra = {}

# I-1: 金曜日の半導体ON
strats_intra['I1_金曜半導体ON'] = series_signal_ret(dow == 4, semi_on_avg)

# I-2: NVDA大幅安→翌日半導体ON (NVDAが-1.5%以下の翌日)
if nvda is not None:
    strats_intra['I2_NVDA安→半導体ON'] = series_signal_ret(nvda < -0.015, semi_on_avg)

# I-3: SOX大幅安翌日は売り継続 (A3の逆利用: ショート)
# SOX<-1% → 翌日 -semi_on_avg (ショートで利益)
if sox is not None:
    strats_intra['I3_SOX安→半導体ONショート'] = series_signal_ret(sox < -0.01, -semi_on_avg)

# ======================================================================
# ② 日次層  (翌日CO: 前日引け→翌日引け, 1日保有)
# ======================================================================
strats_daily = {}

# D-1: SOX>+1% → 翌日半導体CO
if sox is not None:
    strats_daily['D1_SOX高→半導体CO'] = series_signal_ret(sox > 0.01, semi_co_avg)

# D-2: SOX3日連続陽線 → 翌日半導体CO
if sox is not None:
    strats_daily['D2_SOX3連陽→半導体CO'] = series_signal_ret(
        (sox > 0).rolling(3).sum() == 3, semi_co_avg)

# D-3: 半導体個別-3%(セクターフラット) → 翌日リバ
# セクター平均が±1%以内の日に-3%超えた銘柄の平均リバウンド
def individual_drop_rebound(co_ret_mat, codes, avg_series, threshold=-0.03, sector_flat=0.01):
    out = pd.Series(0.0, index=co_ret_mat.index)
    for i, dt in enumerate(co_ret_mat.index[:-1]):
        if abs(avg_series.get(dt, np.nan)) > sector_flat: continue
        vals = []
        for c in codes:
            r = co_ret_mat.loc[dt, c] if c in co_ret_mat.columns else np.nan
            if pd.notna(r) and r < threshold:
                nd = co_ret_mat.index[i+1]
                v = co_ret_next.loc[dt, c] if (dt in co_ret_next.index and c in co_ret_next.columns) else np.nan
                if pd.notna(v): vals.append(v)
        if vals: out.loc[dt] = np.mean(vals)
    return out

strats_daily['D3_半導体個別-3%リバ'] = individual_drop_rebound(co_ret, semi_c, semi_avg_co)

# D-4: 銅3日連続上昇 → 翌日非鉄CO
if cu is not None:
    strats_daily['D4_銅3連上→非鉄CO'] = series_signal_ret(
        (cu > 0).rolling(3).sum() == 3, nf_co_avg)

# ======================================================================
# ③ スイング層  (5日保有, 週次リバランス)
# ======================================================================
strats_swing = {}

# S-1: 半導体20日MOM上位2 → 翌5日CO
strats_swing['S1_半導体MOM5日'] = top_n_daily_ret(co_ret, semi_c, 20, 2, fwd5_semi)

# S-2: 半導体Zスコア下位2 → 翌5日リバ
strats_swing['S2_半導体Zリバ5日'] = zscore_bottom_n_weekly(co_ret, semi_c, 2, fwd5_semi)

# S-3: 信用倍率>5倍 → 翌5日CO (モメンタム銘柄に乗る)
def margin_signal_5d(piv, codes, threshold, direction=1):
    out = pd.Series(0.0, index=piv.index)
    prev_entry = None
    for dt in piv.index[:-5]:
        if prev_entry and (dt - prev_entry).days < 5: continue
        row = piv.loc[dt, [c for c in codes if c in piv.columns]].dropna()
        hits = row[row > threshold].index.tolist() if direction > 0 else row[row < threshold].index.tolist()
        if not hits: continue
        vals = []
        for c in hits:
            if c in fwd5_all.columns and dt in fwd5_all.index:
                v = fwd5_all.loc[dt, c]
                if pd.notna(v): vals.append(v)
        if vals:
            out.loc[dt] = np.mean(vals)
            prev_entry = dt
    return out

strats_swing['S3_信用倍率>5MOМ5日'] = margin_signal_5d(margin_piv, semi_c+nf_c, 5, +1)

# ─── 合成 ─────────────────────────────────────────────────────────────────
all_strats = {**strats_intra, **strats_daily, **strats_swing}
df_ret = pd.DataFrame(all_strats).dropna(how='all').fillna(0)

# 直近3M, 1Yのみ対象
cutoff_3m = df_ret.index.max() - pd.Timedelta(days=95)
cutoff_1y = df_ret.index.max() - pd.Timedelta(days=380)
df_3m = df_ret[df_ret.index >= cutoff_3m]
df_1y = df_ret[df_ret.index >= cutoff_1y]

# ─── 統計サマリー ─────────────────────────────────────────────────────────
def strategy_stats(df, label):
    rows = []
    for col in df.columns:
        s = df[col]
        active = s[s != 0]
        if len(active) < 5:
            rows.append({'strategy': col, 'period': label, 'n_trade_days': 0})
            continue
        mu = active.mean()
        sd = active.std()
        wr = (active > 0).mean()
        t,p = stats.ttest_1samp(active, 0)
        # Kelly (simplified)
        b  = abs(active[active > 0].mean()) / abs(active[active < 0].mean()) if (active < 0).any() else 10
        kelly_f = (wr * b - (1-wr)) / b if b > 0 else 0
        kelly_f = max(0, min(kelly_f, 1))  # clip [0,1]
        quarter_kelly = kelly_f / 4
        freq = len(active)
        annual_trades = freq / (len(df)/252)
        rows.append({
            'strategy': col, 'period': label,
            'n_trade_days': freq, 'annual_freq': round(annual_trades),
            'win_rate': wr, 'mean_ret': mu,
            'sharpe_daily': mu/sd*np.sqrt(252) if sd>0 else np.nan,
            't_stat': t, 'p_value': p,
            'kelly_f': kelly_f, 'quarter_kelly': quarter_kelly,
        })
    return pd.DataFrame(rows)

stats_3m = strategy_stats(df_3m, '3M')
stats_1y = strategy_stats(df_1y, '1Y')
stats_all = pd.concat([stats_3m, stats_1y])

# ─── 相関行列（全期間）─────────────────────────────────────────────────────
active_days = (df_ret != 0)
# 少なくとも片方がアクティブな日で相関
corr_mat = df_ret.where(active_days.any(axis=1)).dropna(how='all').corr()

# ─── 資金配分設計 ─────────────────────────────────────────────────────────
# 3M統計を優先、1/4 Kellyで個別比率を算出
# 時間軸でバケット分け後、バケット内で正規化

LAYER_MAP = {
    'I1_金曜半導体ON':       'イントラ',
    'I2_NVDA安→半導体ON':   'イントラ',
    'I3_SOX安→半導体ONショート': 'イントラ',
    'D1_SOX高→半導体CO':    '日次',
    'D2_SOX3連陽→半導体CO': '日次',
    'D3_半導体個別-3%リバ':  '日次',
    'D4_銅3連上→非鉄CO':    '日次',
    'S1_半導体MOM5日':       'スイング',
    'S2_半導体Zリバ5日':     'スイング',
    'S3_信用倍率>5MOМ5日':  'スイング',
}

# 時間軸ごとのバジェット配分（仮: イントラ20%, 日次30%, スイング50%）
LAYER_BUDGET = {'イントラ': 0.20, '日次': 0.30, 'スイング': 0.50}

s3m = stats_3m.set_index('strategy')

alloc = {}
for layer, budget in LAYER_BUDGET.items():
    layer_strats = [k for k,v in LAYER_MAP.items() if v == layer and k in s3m.index]
    weights = {}
    for s in layer_strats:
        row = s3m.loc[s]
        qk = row.get('quarter_kelly', 0)
        if pd.isna(qk) or qk <= 0: qk = 0
        weights[s] = qk
    total_w = sum(weights.values())
    for s in layer_strats:
        w = weights[s] / total_w if total_w > 0 else 1/len(layer_strats)
        alloc[s] = budget * w

# ─── 合成ポートフォリオシミュレーション（3M）─────────────────────────────
# 各戦略に alloc[s] の割合で資金配分、毎日P&L合算
port_pnl = pd.Series(0.0, index=df_3m.index)
for s, w in alloc.items():
    if s in df_3m.columns:
        port_pnl += df_3m[s] * w

# 累積リターン
port_cum = (1 + port_pnl).cumprod()
port_dd  = port_cum / port_cum.cummax() - 1

port_mu  = port_pnl.mean()
port_sd  = port_pnl.std()
port_sh  = port_mu / port_sd * np.sqrt(252) if port_sd > 0 else np.nan
port_mdd = port_dd.min()
port_ann_ret = (port_cum.iloc[-1] ** (252/len(port_pnl))) - 1
t_port, p_port = stats.ttest_1samp(port_pnl, 0)

# ─── 出力 ─────────────────────────────────────────────────────────────────
print("\n" + "="*85)
print("  時間軸別 戦略統計 (3M / 1Y)")
print("="*85)
print(f"  {'戦略':<28} {'層':<8} {'期間':>3}  {'回数':>4}  {'年頻度':>5}  "
      f"{'勝率':>6}  {'期待値':>8}  {'t値':>5}  {'1/4Kelly':>8}")
print("-"*85)

for layer in ['イントラ', '日次', 'スイング']:
    print(f"\n  ── {layer} ──")
    layer_strats = [k for k,v in LAYER_MAP.items() if v == layer]
    for s in layer_strats:
        for period, sdf in [('3M', stats_3m), ('1Y', stats_1y)]:
            row = sdf[sdf['strategy'] == s]
            if row.empty: continue
            row = row.iloc[0]
            n  = int(row['n_trade_days'])
            af = int(row.get('annual_freq', 0)) if pd.notna(row.get('annual_freq')) else 0
            wr = f"{row['win_rate']*100:5.1f}%" if pd.notna(row.get('win_rate')) else '    –'
            mr = f"{row['mean_ret']*100:+.3f}%" if pd.notna(row.get('mean_ret')) else '     –'
            tv = f"{row['t_stat']:+.2f}"        if pd.notna(row.get('t_stat'))   else '    –'
            qk = f"{row['quarter_kelly']*100:.1f}%" if pd.notna(row.get('quarter_kelly')) else '   –'
            nm = s if period == '3M' else ''
            ly = layer if period == '3M' else ''
            print(f"  {nm:<28} {ly:<8} {period:>3}  {n:>4}  {af:>5}  {wr}  {mr}  {tv}  {qk:>8}")

# ─── 相関行列 ─────────────────────────────────────────────────────────────
print("\n" + "="*85)
print("  戦略間相関行列（シグナル発動日ベース）")
print("="*85)
cols = [c for c in LAYER_MAP if c in corr_mat.columns]
cr = corr_mat.loc[cols, cols]
header = f"  {'':28}" + "".join(f"{c[:8]:>10}" for c in cols)
print(header)
for r in cols:
    row_str = f"  {r:<28}" + "".join(
        f"{cr.loc[r,c]:>10.2f}" if pd.notna(cr.loc[r,c]) else f"{'–':>10}"
        for c in cols
    )
    print(row_str)

# ─── 資金配分テーブル ─────────────────────────────────────────────────────
print("\n" + "="*85)
print("  推奨資金配分（1/4 Kelly + 層バジェット）")
print("="*85)
print(f"\n  層バジェット: イントラ20% / 日次30% / スイング50%")
print(f"\n  {'戦略':<28} {'層':<8} {'配分':>6}   根拠")
print("-"*65)
for layer in ['イントラ', '日次', 'スイング']:
    layer_strats = [k for k,v in LAYER_MAP.items() if v == layer]
    for s in layer_strats:
        w = alloc.get(s, 0)
        r3 = s3m.loc[s] if s in s3m.index else None
        if r3 is not None and pd.notna(r3.get('t_stat')):
            t = r3['t_stat']
            wr = r3['win_rate']
            basis = f"t={t:+.2f} 勝率={wr*100:.0f}%"
        else:
            basis = "N不足"
        print(f"  {s:<28} {layer:<8} {w*100:5.1f}%   {basis}")

# ─── 合成ポートフォリオ統計 ───────────────────────────────────────────────
print("\n" + "="*85)
print("  合成ポートフォリオ統計（直近3M、各戦略に資金配分適用後）")
print("="*85)
print(f"  年率換算リターン: {port_ann_ret*100:+.2f}%")
print(f"  日次Sharpe比:    {port_sh:+.2f}")
print(f"  最大DD:          {port_mdd*100:.2f}%")
print(f"  t値(日次):       {t_port:+.2f}  p={p_port:.4f}")
print(f"  シグナル発動日数: {(port_pnl!=0).sum()} / {len(port_pnl)}営業日")

# ─── 実装ガイド ───────────────────────────────────────────────────────────
print("\n" + "="*85)
print("  実装ガイド（1000万円ポートフォリオ想定）")
print("="*85)
total_cap = 10_000_000
print(f"""
  【イントラデイ層 ¥{total_cap*0.20/10000:.0f}万円 (20%)】
  ・I1_金曜半導体ON : 毎週金曜の寄り付きで半導体2-3銘柄ロング → 引けで全決済
  ・I2_NVDA安→半導体ON : NVDA前日-1.5%以下の翌日寄りエントリー → 引け決済
  ・I3_SOX安→半導体ショート : SOX前日-1%以下の翌日寄りショート → 引け決済
  → 特徴: 1日で完結、翌日リスクなし。信用取引必須（ショートあり）
  → 注意: 発動頻度が低い（月3-4回程度）。空振りは待機

  【日次層 ¥{total_cap*0.30/10000:.0f}万円 (30%)】
  ・D1_SOX高→半導体CO : 朝にSOX確認→高ければ寄り付きロング、翌朝決済
  ・D2_SOX3連陽→半導体CO : より強いシグナル。D1と重複する日は1回のみ
  ・D3_半導体個別-3%リバ : 急落個別を翌日リバ狙いで翌朝決済
  ・D4_銅3連上→非鉄CO : 銅価格確認後エントリー
  → 特徴: 当日夜にシグナル確認、翌朝執行。1泊リスク
  → 注意: SOX系は相関高いので同日発動時は合算扱い（D1+D2同時は1トレード）

  【スイング層 ¥{total_cap*0.50/10000:.0f}万円 (50%)】
  ・S1_半導体MOM5日 : 毎週月曜、直近20日リターン上位2銘柄ロング → 5日後決済
  ・S2_半導体Zリバ5日 : Zスコア下位2銘柄 → 5日後決済（S1とは逆方向になることも）
  ・S3_信用倍率>5MOМ5日 : 信用倍率高=人気銘柄にモメンタム乗り → 5日後決済
  → 特徴: 週1回リバランス。保有中の値動きに惑わされない
  → 注意: S1とS2が同じ銘柄を逆方向に指示することがある → S1優先

  【リスク管理共通ルール】
  ・イントラ: ストップ -1.5%（寄り付き価格基準）
  ・日次:     ストップ -3.0%（エントリー価格基準）
  ・スイング: ストップ -5.0%（エントリー価格基準）、週次でトレーリング
  ・全体DD -8%超でスイング層を半分に縮小
  ・全体DD -15%超で全戦略一時停止
""")

# ─── CSV出力 ─────────────────────────────────────────────────────────────
stats_all.to_csv(f"{OUT_DIR}/portfolio_stats.csv", index=False)
cr.to_csv(f"{OUT_DIR}/correlation_matrix.csv")
port_pnl_df = pd.DataFrame({'date': port_pnl.index, 'pnl': port_pnl.values,
                             'cum': port_cum.values, 'dd': port_dd.values})
port_pnl_df.to_csv(f"{OUT_DIR}/portfolio_pnl.csv", index=False)
print(f"  CSV保存: {OUT_DIR}/")
print("  完了")
