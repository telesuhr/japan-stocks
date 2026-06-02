"""
バッチ3: 決算イベント 5本 (#13-17)

サプライズ定義 = 当期FY実績 NP が「前回FY決算で会社が出した翌期予想 NxFNp」をどれだけ上回ったか
  SUE = (NP_actual - NxFNp_prior) / (|NxFNp_prior| + eps)   ← 会社ガイダンスに対するビート/ミス
イベント = FY本決算(連結 JP/IFRS)。エントリ = 開示翌営業日の引け。ドリフト = +5/+20/+60日の
超過リターン AR(=銘柄 - 流動性ユニバース等加重指数, 同期間)。

#13 会社予想SUE PEAD          SUE上位 vs 下位 の d20 AR (PEAD強度)
#14 増配ドリフト 対照群        翌期予想増配(NxFDivFY>DivFY) vs 非増配 の d20 AR (対照群比較)
#15 決算ドリフト セクター内相対  SUE を sector33 内デミーン → 上位/下位 d20 AR
#16 決算サプライズ×出来高       SUE上位 を 開示翌日出来高急増(確認)で絞った d20 AR
#17 銘柄別ドリフト持続性        同一銘柄の連続イベント AR の自己相関(前回ビート→今回もドリフト?)

IS 2022-2023 / OOS 2024- / EVAL 2021-10-01。流動性上位800。
"""
from __future__ import annotations
import os, sys, warnings, json
import psycopg2, pandas as pd, numpy as np
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

DB = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost/market_data")
IS_S, IS_E, OOS_S, EVAL_S = (pd.Timestamp(x) for x in ("2022-01-01", "2023-12-31", "2024-01-01", "2021-10-01"))
NUNI = 800


def fetch(sql):
    c = psycopg2.connect(DB); df = pd.read_sql(sql, c); c.close(); return df


def f(v):
    try:
        x = float(v); return x
    except (TypeError, ValueError):
        return np.nan


def tstat(x):
    x = pd.Series(x).dropna()
    return float(x.mean()/x.std()*np.sqrt(len(x))) if len(x) >= 8 and x.std() > 0 else float('nan')


print("="*76); print("バッチ3: 決算イベント 5本 (#13-17)"); print("="*76)

print("\n[ユニバース・価格]")
uni = fetch(f"""SELECT code FROM stocks_daily WHERE date>='2021-01-01' AND turnover_value>0
    GROUP BY code HAVING count(*)>900 ORDER BY avg(turnover_value) DESC LIMIT {NUNI}""")
codes = uni['code'].tolist()
sec = fetch("SELECT code5, sector33 FROM symbol_master")
sec_map = dict(zip(sec['code5'], sec['sector33']))
px = fetch(f"""SELECT code, date, adj_close::float ac, volume::float v FROM stocks_daily
    WHERE code IN ({','.join(f"'{x}'" for x in codes)}) AND date>='2021-01-01' AND adj_close>0 ORDER BY code,date""")
px['date'] = pd.to_datetime(px['date'])
px = px.sort_values(['code', 'date'])
px['ret'] = px.groupby('code')['ac'].pct_change()
px['v20'] = px.groupby('code')['v'].transform(lambda s: s.rolling(20).mean())
# 流動性ユニバース等加重日次指数(市場中立化用)
idx = px.groupby('date')['ret'].mean().fillna(0)
cumidx = (1+idx).cumprod()
all_dates = sorted(px['date'].unique())
dpos = {d: i for i, d in enumerate(all_dates)}
# 銘柄ごとに date->ac, date->v配列
pxg = {c: g.set_index('date') for c, g in px.groupby('code')}

print("[FY本決算イベント取得・サプライズ計算]")
ev = fetch(f"""SELECT code, disc_date, disc_time, payload FROM fin_summary
    WHERE doc_type IN ('FYFinancialStatements_Consolidated_JP','FYFinancialStatements_Consolidated_IFRS')
      AND code IN ({','.join(f"'{x}'" for x in codes)}) AND disc_date>='2021-01-01' ORDER BY code, disc_date""")
ev['disc_date'] = pd.to_datetime(ev['disc_date'])
rows = []
for _, r in ev.iterrows():
    p = r['payload']
    if not isinstance(p, dict):
        p = json.loads(p)
    rows.append({'code': r['code'], 'disc_date': r['disc_date'], 'disc_time': r['disc_time'],
                 'NP': f(p.get('NP')), 'Sales': f(p.get('Sales')), 'OP': f(p.get('OP')),
                 'NxFNp': f(p.get('NxFNp')), 'NxFSales': f(p.get('NxFSales')),
                 'DivFY': f(p.get('DivFY')), 'NxFDivFY': f(p.get('NxFDivFY'))})
e = pd.DataFrame(rows).sort_values(['code', 'disc_date'])
# 前回FY決算の翌期予想 = 今期の予想ベンチ
e['prevF_NP'] = e.groupby('code')['NxFNp'].shift(1)
e['SUE'] = (e['NP'] - e['prevF_NP']) / (e['prevF_NP'].abs() + 1e6)
e['SUE'] = e['SUE'].clip(-3, 3)
e['div_raise'] = (e['NxFDivFY'] > e['DivFY']).astype(int)
e['sector'] = e['code'].map(sec_map)


def drift(code, disc_date, H):
    """開示翌営業日引けエントリ → +H営業日 の超過リターン (銘柄 - 指数)"""
    fut = [d for d in all_dates if d > disc_date]
    if len(fut) < H + 1:
        return np.nan
    entry = fut[0]
    ei = dpos[entry]
    if ei + H >= len(all_dates):
        return np.nan
    exit_d = all_dates[ei + H]
    gp = pxg.get(code)
    if gp is None or entry not in gp.index or exit_d not in gp.index:
        return np.nan
    sret = gp.loc[exit_d, 'ac'] / gp.loc[entry, 'ac'] - 1
    mret = cumidx[exit_d] / cumidx[entry] - 1
    # エントリ翌日出来高急増フラグ(確認シグナル用)
    return sret - mret


def vol_confirm(code, disc_date):
    fut = [d for d in all_dates if d > disc_date]
    if not fut: return np.nan
    entry = fut[0]
    gp = pxg.get(code)
    if gp is None or entry not in gp.index: return np.nan
    v, v20 = gp.loc[entry, 'v'], gp.loc[entry, 'v20']
    return v/v20 if v20 and v20 > 0 else np.nan


for H in (5, 20, 60):
    e[f'ar{H}'] = [drift(c, d, H) for c, d in zip(e['code'], e['disc_date'])]
e['vconf'] = [vol_confirm(c, d) for c, d in zip(e['code'], e['disc_date'])]
e = e[e['disc_date'] >= EVAL_S].dropna(subset=['SUE', 'ar20'])
print(f"  イベント数 N={len(e)} (SUE算出可・ドリフト計測可)")


def seg(df, col):
    return {'全': df, 'IS': df[(df['disc_date'] >= IS_S) & (df['disc_date'] <= IS_E)],
            'OOS': df[df['disc_date'] >= OOS_S]}


# ---- #13 会社予想SUE PEAD ----
print("\n--- #13 会社予想SUE PEAD (上位/下位5分位 d20 AR) ---")
for H in (5, 20, 60):
    d = e.dropna(subset=[f'ar{H}'])
    q = d['SUE'].rank(pct=True)
    top = d[q >= 0.8][f'ar{H}']; bot = d[q <= 0.2][f'ar{H}']
    ls = top.mean() - bot.mean()
    rho, _ = spearmanr(d['SUE'], d[f'ar{H}'])
    print(f"  d{H:<2}: ρ(SUE,AR)={rho:+.3f}  上位AR={top.mean()*1e4:+.0f}bps 下位AR={bot.mean()*1e4:+.0f}bps "
          f"L/S={ls*1e4:+.0f}bps t(上位)={tstat(top):+.2f}")
d20 = e.dropna(subset=['ar20'])
s = seg(d20, 'ar20')
for lab, sub in s.items():
    q = sub['SUE'].rank(pct=True)
    ls = sub[q >= 0.8]['ar20'].mean() - sub[q <= 0.2]['ar20'].mean()
    print(f"    d20 L/S {lab}: {ls*1e4:+.0f}bps (N={len(sub)})")

# ---- #14 増配ドリフト 対照群 ----
print("\n--- #14 増配(翌期予想)ドリフト 対照群比較 ---")
for H in (20, 60):
    d = e.dropna(subset=[f'ar{H}'])
    raise_ar = d[d['div_raise'] == 1][f'ar{H}']
    flat_ar = d[d['div_raise'] == 0][f'ar{H}']
    print(f"  d{H:<2}: 増配 AR={raise_ar.mean()*1e4:+.0f}bps(N={len(raise_ar)},t={tstat(raise_ar):+.2f}) "
          f"非増配 AR={flat_ar.mean()*1e4:+.0f}bps(N={len(flat_ar)}) "
          f"差={(raise_ar.mean()-flat_ar.mean())*1e4:+.0f}bps")

# ---- #15 決算ドリフト セクター内相対 ----
print("\n--- #15 SUE セクター中立 (sector33内デミーン → 上位/下位 d20 AR) ---")
d = e.dropna(subset=['ar20', 'sector']).copy()
d['SUE_sn'] = d.groupby(['sector'])['SUE'].transform(lambda x: x - x.mean())
for lab, sub in seg(d, 'ar20').items():
    q = sub['SUE_sn'].rank(pct=True)
    ls = sub[q >= 0.8]['ar20'].mean() - sub[q <= 0.2]['ar20'].mean()
    rho, _ = spearmanr(sub['SUE_sn'], sub['ar20'])
    print(f"  {lab}: ρ={rho:+.3f} L/S d20={ls*1e4:+.0f}bps (N={len(sub)})")

# ---- #16 決算サプライズ×出来高確認 ----
print("\n--- #16 SUE上位 × 開示翌日出来高急増(確認) d20 AR ---")
d = e.dropna(subset=['ar20', 'vconf'])
q = d['SUE'].rank(pct=True)
hi = d[q >= 0.8]
for thr in (1.0, 1.5, 2.0):
    conf = hi[hi['vconf'] >= thr]['ar20']
    unconf = hi[hi['vconf'] < thr]['ar20']
    print(f"  出来高≥{thr}x: 確認群 AR={conf.mean()*1e4:+.0f}bps(N={len(conf)},t={tstat(conf):+.2f}) "
          f"非確認 AR={unconf.mean()*1e4:+.0f}bps(N={len(unconf)})")

# ---- #17 銘柄別ドリフト持続性 ----
print("\n--- #17 銘柄別ドリフト持続性 (連続イベント AR の自己相関) ---")
d = e.dropna(subset=['ar20']).sort_values(['code', 'disc_date']).copy()
d['ar20_prev'] = d.groupby('code')['ar20'].shift(1)
dd = d.dropna(subset=['ar20_prev'])
rho_all, p_all = spearmanr(dd['ar20_prev'], dd['ar20'])
# 前回プラスドリフト銘柄の今回 vs 前回マイナスの今回
pos_next = dd[dd['ar20_prev'] > 0]['ar20']
neg_next = dd[dd['ar20_prev'] <= 0]['ar20']
print(f"  前回AR vs 今回AR: ρ={rho_all:+.3f} (p={p_all:.3f}, N={len(dd)})")
print(f"  前回+ドリフト銘柄の今回AR={pos_next.mean()*1e4:+.0f}bps / 前回-の今回AR={neg_next.mean()*1e4:+.0f}bps")

print("\n完了")
