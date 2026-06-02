"""
V6モメンタムスコア 50銘柄ユニバース バックテスト

morning_dashboard.html に実装した V6 スコア
    r20_adj = r20 / vol20        (vol20 = 直近20日の日次リターン標準偏差 × √252, 母集団std)
    d75     = 現値 / MA75 - 1
    V6_raw  = r20_adj + 0.5 * d75 * 10
を、22銘柄 → 50銘柄に拡張して検証する。

研究 20260531_v6score_comprehensive の次アクション:
  #2 ユニバース拡張: PORTFOLIO_ALL (50銘柄) でL/Sを試す
  #4 V6 L/S 本格バックテスト

ダッシュボードの計算式を「そのまま」再現する (population std, 直近20リターンに当日含む)。
研究版 run.py とは ddof と窓の取り方が僅かに異なる点に注意 — ここでは実装デプロイ済みの
ロジックをバックテストするのが目的。

評価:
  A. V6_raw の クロスセクショナル Spearman IC (全/IS/OOS)
  B. Long-only Sharpe (上位20%エントリー, 20日保有)
  C. クロスセクショナル L/S (上位5 Long / 下位5 Short, 20日保有)
  D. 22銘柄 vs 50銘柄 比較
  E. 日次リバランス L/S エクイティカーブ (翌日リターン, 図示用)

IS: 2022-01-01〜2023-12-31  /  OOS: 2024-01-01〜
N225 60日 -3% 割れでベアゲート (ロング手仕舞い → ロングサイドを現金化)
"""
from __future__ import annotations

import os
import sys
import psycopg2
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

sys.stdout.reconfigure(line_buffering=True)

DB_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost/market_data")

# 主戦22銘柄 (半導体14 + 非鉄8)
UNI22 = [
    '80350', '68570', '69200', '61460', '77350', '67230', '69630', '65260',
    '40620', '34360', '40630', '77410', '99840', '285A0',
    '58030', '50160', '58010', '58020', '57130', '57060', '57110', '57140',
]
# 追加28銘柄 (大型主力)
ADD28 = [
    '16050', '40040', '60980', '65010', '65030', '65250', '67580', '67620',
    '68610', '69540', '69710', '69760', '69810', '70110', '70120', '70130',
    '72030', '72670', '79740', '80310', '80580', '83060', '83160', '84110',
    '87660', '94320', '94330', '99830',
]
UNI50 = UNI22 + ADD28
CODES4_50 = [c[:4] for c in UNI50]
CODES4_22 = [c[:4] for c in UNI22]
CODE_LIST = ','.join(f"'{c}'" for c in UNI50)

IS_START = pd.Timestamp("2022-01-01")
IS_END = pd.Timestamp("2023-12-31")
OOS_START = pd.Timestamp("2024-01-01")
EVAL_START = pd.Timestamp("2021-10-01")  # MA75 + 余裕を見て指標安定後から評価
GATE = -0.03
HOLD = 20


def fetch(sql: str) -> pd.DataFrame:
    conn = psycopg2.connect(DB_URL)
    df = pd.read_sql(sql, conn)
    conn.close()
    return df


def sharpe(rets: pd.Series, ann: int = 252) -> float:
    r = rets.dropna()
    if len(r) < 10 or r.std() == 0:
        return float("nan")
    return float(r.mean() / r.std() * np.sqrt(ann))


def icir(ic: pd.Series) -> float:
    ic_ = ic.dropna()
    if len(ic_) < 10 or ic_.std() == 0:
        return float("nan")
    return float(ic_.mean() / ic_.std() * np.sqrt(252))


def ic_series(d: pd.DataFrame, fac: str, ret: str) -> pd.Series:
    ics = []
    for _, g in d.groupby('date'):
        sub = g[[fac, ret]].dropna()
        if len(sub) < 5:
            continue
        ic, _ = spearmanr(sub[fac], sub[ret])
        ics.append(ic)
    return pd.Series(ics)


print("=" * 76)
print("V6モメンタムスコア 50銘柄バックテスト")
print("=" * 76)
print("\n[データ取得中]")

prices = fetch(f"""
    SELECT LEFT(code,4) c, date, adj_close::float ac
    FROM stocks_daily
    WHERE code IN ({CODE_LIST}) AND date >= '2021-01-01' AND adj_close > 0
    ORDER BY code, date
""")
n225 = fetch("""
    SELECT date, close::float c FROM index_daily
    WHERE code='N225' AND date >= '2021-01-01' ORDER BY date
""")
prices['date'] = pd.to_datetime(prices['date'])
n225['date'] = pd.to_datetime(n225['date'])
n225 = n225.set_index('date').sort_index()

# code -> 価格パネル (date index)
pivot = prices.pivot(index='date', columns='c', values='ac').sort_index()
all_dates = list(pivot.index)
print(f"  銘柄: {pivot.shape[1]} / 営業日: {len(all_dates)}")

# N225 60日リターン (asof)
n225_aligned = n225['c'].reindex(all_dates).ffill()
n225_r60 = n225_aligned / n225_aligned.shift(60) - 1


def compute_v6_panel(codes4: list[str]) -> pd.DataFrame:
    """各日 × 各銘柄 の V6_raw と 20日フォワードリターン・翌日リターンを計算 (ダッシュボード式)"""
    rows = []
    cols = [c for c in codes4 if c in pivot.columns]
    for i, dt in enumerate(all_dates):
        if dt < EVAL_START:
            continue
        gate_on = bool(n225_r60.loc[dt] < GATE) if pd.notna(n225_r60.loc[dt]) else False
        for c in cols:
            s = pivot[c]
            # asof までの非欠損系列
            hist = s.iloc[:i + 1].dropna()
            if len(hist) < 76:
                continue
            a = hist.values
            last = a[-1]
            r20 = last / a[-21] - 1
            rets = a[-20:] / a[-21:-1] - 1  # 直近20本の日次リターン (当日含む)
            sd = float(np.std(rets))         # 母集団std (ダッシュボードと一致)
            vol20 = sd * np.sqrt(252)
            r20adj = r20 / vol20 if vol20 > 0 else 0.0
            ma75 = a[-75:].mean()
            d75 = last / ma75 - 1
            v6 = r20adj + 0.5 * d75 * 10
            # フォワード20日 / 翌日リターン (生の adj_close, asof日以降)
            fwd20 = nxt = np.nan
            if i + HOLD < len(all_dates):
                pf = s.iloc[i + HOLD]
                if pd.notna(pf):
                    fwd20 = pf / last - 1
            if i + 1 < len(all_dates):
                p1 = s.iloc[i + 1]
                if pd.notna(p1):
                    nxt = p1 / last - 1
            rows.append({'date': dt, 'code': c, 'r20adj': r20adj, 'd75': d75,
                         'v6': v6, 'gate': gate_on, 'fwd20': fwd20, 'nxt': nxt})
    return pd.DataFrame(rows)


print("\n[因子計算中] 50銘柄 ...")
df = compute_v6_panel(CODES4_50)
df50 = df.copy()
df22 = df[df['code'].isin(CODES4_22)].copy()
print(f"  サンプル: 50銘柄={len(df50):,} / 22銘柄={len(df22):,}")

# 市場超過 (xs) フォワード
for d in (df50, df22):
    d['xs20'] = d['fwd20'] - d.groupby('date')['fwd20'].transform('mean')


def eval_universe(d: pd.DataFrame, label: str, n_side: int):
    print("\n" + "=" * 76)
    print(f"{label} (各サイド{n_side}銘柄)")
    print("=" * 76)

    # --- A. IC ---
    print("\n  A. V6_raw クロスセクショナル Spearman IC (vs 市場超過20日)")
    for plabel, mask in [
        ("全期間", d['date'] >= EVAL_START),
        ("IS(22-23)", (d['date'] >= IS_START) & (d['date'] <= IS_END)),
        ("OOS(24-)", d['date'] >= OOS_START),
    ]:
        sub = d[mask][['date', 'v6', 'xs20']].dropna()
        ic = ic_series(sub, 'v6', 'xs20')
        print(f"    {plabel:<12} IC mean={ic.mean():+.4f}  ICIR={icir(ic):.2f}  (N日={len(ic)})")

    # --- B. Long-only 上位20% ---
    print("\n  B. Long-only Sharpe (各日 上位20%エントリー, 20日保有・ゲート適用)")
    res_b = {}
    for plabel, mask in [
        ("全期間", d['date'] >= EVAL_START),
        ("IS(22-23)", (d['date'] >= IS_START) & (d['date'] <= IS_END)),
        ("OOS(24-)", d['date'] >= OOS_START),
    ]:
        picks = []
        for dt, g in d[mask].dropna(subset=['fwd20']).groupby('date'):
            gg = g.copy()
            # ゲート: ベア相場ではロング見送り
            if gg['gate'].iloc[0]:
                continue
            th = gg['v6'].quantile(0.8)
            picks.append(gg[gg['v6'] >= th]['fwd20'])
        allp = pd.concat(picks) if picks else pd.Series(dtype=float)
        # 20日保有なので √(252/20) で年率化
        sh = sharpe(allp, ann=252 // HOLD)
        res_b[plabel] = sh
        mean_ = allp.mean() * 100 if len(allp) else float('nan')
        print(f"    {plabel:<12} n={len(allp):<6} mean(20d)={mean_:+.2f}%  Sharpe={sh:.2f}")

    # --- C. L/S top/bottom ---
    print(f"\n  C. クロスセクショナル L/S (上位{n_side} Long / 下位{n_side} Short, 20日保有)")
    res_c = {}
    for plabel, mask in [
        ("全期間", d['date'] >= EVAL_START),
        ("IS(22-23)", (d['date'] >= IS_START) & (d['date'] <= IS_END)),
        ("OOS(24-)", d['date'] >= OOS_START),
    ]:
        ls = []
        for dt, g in d[mask].dropna(subset=['fwd20']).groupby('date'):
            if len(g) < n_side * 2:
                continue
            r = g.sort_values('v6', ascending=False)
            lng = r.head(n_side)['fwd20'].mean()
            sht = r.tail(n_side)['fwd20'].mean()
            ls.append({'date': dt, 'ls': lng - sht, 'long': lng, 'short': sht})
        lsd = pd.DataFrame(ls).set_index('date').sort_index()
        sh_ls = sharpe(lsd['ls'], ann=252 // HOLD)
        sh_l = sharpe(lsd['long'], ann=252 // HOLD)
        sh_s = sharpe(lsd['short'], ann=252 // HOLD)
        res_c[plabel] = sh_ls
        print(f"    {plabel:<12} L/S Sharpe={sh_ls:.2f}  mean(20d)={lsd['ls'].mean()*100:+.2f}%"
              f"  | Long Sh={sh_l:.2f}  Short(原) Sh={sh_s:.2f}")
    return res_b, res_c


b50, c50 = eval_universe(df50, "D-1. 50銘柄ユニバース", n_side=5)
b22, c22 = eval_universe(df22, "D-2. 22銘柄ユニバース (主戦のみ・各サイド3)", n_side=3)

# ========================
# E. 日次リバランス L/S エクイティカーブ (翌日リターン, 図示用)
# ========================
print("\n" + "=" * 76)
print("E. 日次リバランス L/S エクイティカーブ (翌日リターン, 50銘柄, 上位5/下位5)")
print("=" * 76)

daily = []
for dt, g in df50.dropna(subset=['nxt']).groupby('date'):
    if len(g) < 10:
        continue
    r = g.sort_values('v6', ascending=False)
    lng = r.head(5)['nxt'].mean()
    sht = r.tail(5)['nxt'].mean()
    gate_on = bool(g['gate'].iloc[0])
    daily.append({'date': dt, 'long': lng, 'short': sht,
                  'ls': lng - sht,
                  'long_gated': 0.0 if gate_on else lng})  # ゲート時ロングは現金
dd = pd.DataFrame(daily).set_index('date').sort_index()
dd['ls_eq'] = (1 + dd['ls']).cumprod()
dd['long_eq'] = (1 + dd['long']).cumprod()
dd['longg_eq'] = (1 + dd['long_gated']).cumprod()

for plabel, mask in [
    ("全期間", dd.index >= EVAL_START),
    ("IS(22-23)", (dd.index >= IS_START) & (dd.index <= IS_END)),
    ("OOS(24-)", dd.index >= OOS_START),
]:
    sub = dd[mask]
    print(f"  {plabel:<12} L/S日次Sharpe={sharpe(sub['ls']):.2f}  "
          f"Long(ゲート無)={sharpe(sub['long']):.2f}  Long(ゲート有)={sharpe(sub['long_gated']):.2f}")

tot_ret = (dd['ls_eq'].iloc[-1] - 1) * 100
dmax = (dd['ls_eq'] / dd['ls_eq'].cummax() - 1).min() * 100
print(f"\n  L/S 累積リターン={tot_ret:+.1f}%  最大DD={dmax:.1f}%  期間={dd.index[0].date()}〜{dd.index[-1].date()}")

# ========================
# 図
# ========================
print("\n[作図中]")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

for fp in ['/root/.fonts/NotoSansJP.ttf',
           '/usr/share/fonts/truetype/ipaexg/ipaexg.ttf',
           '/usr/share/fonts/opentype/ipaexfont-gothic/ipaexg.ttf',
           '/usr/share/fonts/truetype/fonts-japanese-gothic.ttf']:
    if os.path.exists(fp):
        fm.fontManager.addfont(fp)
        plt.rcParams['font.family'] = fm.FontProperties(fname=fp).get_name()
        break
plt.rcParams['axes.unicode_minus'] = False

fig = plt.figure(figsize=(12, 6.75), facecolor='white')
gs = fig.add_gridspec(2, 2, height_ratios=[1.6, 1], hspace=0.42, wspace=0.22,
                      left=0.07, right=0.97, top=0.86, bottom=0.13)

# 上段: エクイティカーブ
ax = fig.add_subplot(gs[0, :])
ax.plot(dd.index, dd['ls_eq'], color='#1f6feb', lw=2, label='L/S (上位5ロング − 下位5ショート)')
ax.plot(dd.index, dd['longg_eq'], color='#2ea043', lw=1.6, label='ロングのみ (ベアゲート現金化)')
ax.plot(dd.index, dd['long_eq'], color='#999999', lw=1, ls='--', label='ロングのみ (ゲート無)')
ax.axvspan(IS_START, IS_END, color='#ffd8a8', alpha=0.25)
ax.axvline(OOS_START, color='#cf222e', ls=':', lw=1)
ax.text(IS_START + (IS_END - IS_START) / 2, ax.get_ylim()[1], 'IS', ha='center', va='top', fontsize=9, color='#bf8700')
ax.set_ylabel('累積成長 (×)')
ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
ax.grid(alpha=0.3)
ax.set_title('V6モメンタムスコア L/S — 日次リバランス・翌日リターン (50銘柄)', fontsize=12, pad=8)

# 下段左: IS/OOS L/S Sharpe 22 vs 50
ax2 = fig.add_subplot(gs[1, 0])
periods = ['IS(22-23)', 'OOS(24-)', '全期間']
x = np.arange(len(periods))
v50 = [c50[p] for p in periods]
v22 = [c22[p] for p in periods]
ax2.bar(x - 0.2, v22, 0.4, label='22銘柄(各3)', color='#8b949e')
ax2.bar(x + 0.2, v50, 0.4, label='50銘柄(各5)', color='#1f6feb')
ax2.set_xticks(x)
ax2.set_xticklabels(periods, fontsize=8)
ax2.set_ylabel('L/S Sharpe (20日保有)')
ax2.set_title('ユニバース拡張効果: L/S Sharpe', fontsize=10)
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3, axis='y')
for xi, v in zip(x - 0.2, v22):
    ax2.text(xi, v, f'{v:.1f}', ha='center', va='bottom' if v >= 0 else 'top', fontsize=7)
for xi, v in zip(x + 0.2, v50):
    ax2.text(xi, v, f'{v:.1f}', ha='center', va='bottom' if v >= 0 else 'top', fontsize=7)

# 下段右: Long-only Sharpe 22 vs 50
ax3 = fig.add_subplot(gs[1, 1])
lo50 = [b50[p] for p in periods]
lo22 = [b22[p] for p in periods]
ax3.bar(x - 0.2, lo22, 0.4, label='22銘柄', color='#8b949e')
ax3.bar(x + 0.2, lo50, 0.4, label='50銘柄', color='#2ea043')
ax3.set_xticks(x)
ax3.set_xticklabels(periods, fontsize=8)
ax3.set_ylabel('Long-only Sharpe')
ax3.set_title('Long-only (上位20%): Sharpe', fontsize=10)
ax3.legend(fontsize=8)
ax3.grid(alpha=0.3, axis='y')
for xi, v in zip(x - 0.2, lo22):
    ax3.text(xi, v, f'{v:.1f}', ha='center', va='bottom', fontsize=7)
for xi, v in zip(x + 0.2, lo50):
    ax3.text(xi, v, f'{v:.1f}', ha='center', va='bottom', fontsize=7)

fig.suptitle('V6モメンタムスコア 50銘柄バックテスト  r20_adj + 0.5×MA75乖離', fontsize=14, fontweight='bold', y=0.96)
fig.text(0.99, 0.01, f'データ: {dd.index[0].date()}〜{dd.index[-1].date()} / 日本株日足(JQuants) / IS橙帯・OOS赤点線',
         ha='right', va='bottom', fontsize=8, color='gray')

out = os.path.dirname(__file__)
plt.savefig(os.path.join(out, 'result.png'), dpi=100, facecolor='white')
print(f"  保存: result.png")

# CSV
dd[['long', 'short', 'ls', 'long_gated', 'ls_eq', 'longg_eq']].to_csv(os.path.join(out, 'ls_daily.csv'))
summary = pd.DataFrame({
    'period': periods,
    'ls_sharpe_50': [c50[p] for p in periods],
    'ls_sharpe_22': [c22[p] for p in periods],
    'longonly_sharpe_50': [b50[p] for p in periods],
    'longonly_sharpe_22': [b22[p] for p in periods],
})
summary.to_csv(os.path.join(out, 'sharpe_summary.csv'), index=False)
print(f"  保存: ls_daily.csv, sharpe_summary.csv")
print("\n完了")
