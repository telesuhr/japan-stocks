"""
V6スコア ロング特化バックテスト (ショート無し) — コスト込み・非重複・ベンチ超過

問い: ショートを諦め「ロング特化スコアリング」だけならV6は実用になるか?

検証:
  - 集中度: 上位 3 / 5 / 8 / 10銘柄 (等加重ロングのみ)
  - 保有: 20日・非重複(全20位相を平均)で √(252/20) 年率化 ← 過大評価を排除
  - コスト: 往復 0 / 10 / 20bps
  - ベアゲート: N225 60日 -3%割れで当該エントリーを現金(リターン0・コストのみ控除)
  - **ベンチ超過**: 50銘柄等加重の同期間20日リターンを引いた「純選択アルファ」も測る
    (2024-26は半導体メガキャップ自体が爆騰 → buy&hold超過で見ないと相場βを掴むだけ)

ダッシュボード実装の V6_raw = r20_adj + 0.5*d75*10 をそのまま使用。
"""
from __future__ import annotations

import os
import sys
import psycopg2
import pandas as pd
import numpy as np

sys.stdout.reconfigure(line_buffering=True)

DB_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost/market_data")

UNI22 = ['80350', '68570', '69200', '61460', '77350', '67230', '69630', '65260',
         '40620', '34360', '40630', '77410', '99840', '285A0',
         '58030', '50160', '58010', '58020', '57130', '57060', '57110', '57140']
ADD28 = ['16050', '40040', '60980', '65010', '65030', '65250', '67580', '67620',
         '68610', '69540', '69710', '69760', '69810', '70110', '70120', '70130',
         '72030', '72670', '79740', '80310', '80580', '83060', '83160', '84110',
         '87660', '94320', '94330', '99830']
UNI50 = UNI22 + ADD28
CODES4 = [c[:4] for c in UNI50]
CODE_LIST = ','.join(f"'{c}'" for c in UNI50)

IS_START = pd.Timestamp("2022-01-01")
IS_END = pd.Timestamp("2023-12-31")
OOS_START = pd.Timestamp("2024-01-01")
EVAL_START = pd.Timestamp("2021-10-01")
GATE = -0.03
HOLD = 20
ANN = 252 / HOLD


def fetch(sql):
    conn = psycopg2.connect(DB_URL)
    df = pd.read_sql(sql, conn)
    conn.close()
    return df


def sharpe(r, ann=ANN):
    r = pd.Series(r).dropna()
    if len(r) < 8 or r.std() == 0:
        return float('nan')
    return float(r.mean() / r.std() * np.sqrt(ann))


print("=" * 76)
print("V6 ロング特化バックテスト (コスト込み・非重複・ベンチ超過)")
print("=" * 76)
print("\n[データ取得]")
prices = fetch(f"""
    SELECT LEFT(code,4) c, date, adj_close::float ac FROM stocks_daily
    WHERE code IN ({CODE_LIST}) AND date >= '2021-01-01' AND adj_close > 0
    ORDER BY code, date
""")
n225 = fetch("SELECT date, close::float c FROM index_daily WHERE code='N225' AND date>='2021-01-01' ORDER BY date")
prices['date'] = pd.to_datetime(prices['date'])
n225['date'] = pd.to_datetime(n225['date'])
pivot = prices.pivot(index='date', columns='c', values='ac').sort_index()
dates = list(pivot.index)
n225s = n225.set_index('date')['c'].reindex(dates).ffill()
n225_r60 = n225s / n225s.shift(60) - 1
print(f"  銘柄{pivot.shape[1]} / 日{len(dates)}")

# V6パネル: date×code の v6, fwd20, gate
print("[V6計算]")
rows = []
cols = [c for c in CODES4 if c in pivot.columns]
for i, dt in enumerate(dates):
    if dt < EVAL_START:
        continue
    gate_on = bool(n225_r60.iloc[i] < GATE) if pd.notna(n225_r60.iloc[i]) else False
    fwd_i = i + HOLD
    for c in cols:
        s = pivot[c]
        hist = s.iloc[:i + 1].dropna()
        if len(hist) < 76:
            continue
        a = hist.values
        last = a[-1]
        r20 = last / a[-21] - 1
        rets = a[-20:] / a[-21:-1] - 1
        vol20 = float(np.std(rets)) * np.sqrt(252)
        r20adj = r20 / vol20 if vol20 > 0 else 0.0
        d75 = last / a[-75:].mean() - 1
        v6 = r20adj + 0.5 * d75 * 10
        fwd20 = np.nan
        if fwd_i < len(dates):
            pf = s.iloc[fwd_i]
            if pd.notna(pf):
                fwd20 = pf / last - 1
        rows.append({'date': dt, 'di': i, 'code': c, 'v6': v6, 'gate': gate_on, 'fwd20': fwd20})
df = pd.DataFrame(rows)
df['bench'] = df.groupby('date')['fwd20'].transform('mean')  # 50銘柄等加重の20日先リターン
print(f"  サンプル {len(df):,}")


def period_mask(s, key):
    if key == 'IS':
        return (s >= IS_START) & (s <= IS_END)
    if key == 'OOS':
        return s >= OOS_START
    return s >= EVAL_START


def longonly_nonoverlap(topN, cost_bps, gated=True, excess=False):
    """非重複20日・全位相平均。集合: 各エントリー日に上位topNを等加重ロング。
       gated: ゲート発動日はリターン0(コストも掛けない=見送り)。
       excess: ベンチ(50等加重)超過リターンで評価。
    戻り: {period: (sharpe, mean%, n)}"""
    cost = cost_bps / 10000.0
    di_list = sorted(df['di'].unique())
    by_di = {di: g for di, g in df.groupby('di')}
    out = {k: [] for k in ('all', 'IS', 'OOS')}  # 位相ごとのSharpeを集める
    means = {k: [] for k in ('all', 'IS', 'OOS')}
    counts = {k: 0 for k in ('all', 'IS', 'OOS')}
    for phase in range(HOLD):
        entries = di_list[phase::HOLD]
        recs = []
        for di in entries:
            g = by_di.get(di)
            if g is None:
                continue
            gg = g.dropna(subset=['fwd20'])
            if len(gg) < topN:
                continue
            dt = gg['date'].iloc[0]
            if gated and bool(gg['gate'].iloc[0]):
                ret = 0.0  # 現金
            else:
                top = gg.sort_values('v6', ascending=False).head(topN)
                ret = top['fwd20'].mean() - cost  # 往復コスト
                if excess:
                    ret = top['fwd20'].mean() - gg['bench'].iloc[0] - cost
            recs.append({'date': dt, 'ret': ret})
        if not recs:
            continue
        rd = pd.DataFrame(recs)
        for k in ('all', 'IS', 'OOS'):
            sub = rd[period_mask(rd['date'], k)]['ret']
            if len(sub) >= 8:
                sh = sharpe(sub)
                if not np.isnan(sh):
                    out[k].append(sh)
                    means[k].append(sub.mean() * 100)
                    counts[k] = max(counts[k], len(sub))
    res = {}
    for k in ('all', 'IS', 'OOS'):
        res[k] = (np.mean(out[k]) if out[k] else float('nan'),
                  np.mean(means[k]) if means[k] else float('nan'),
                  counts[k])
    return res


# ========================
# A. 集中度別 (絶対リターン, ゲート有, コスト別)
# ========================
print("\n" + "=" * 76)
print("A. ロング特化 絶対リターン Sharpe (非重複20日, ゲート有)")
print("=" * 76)
for cost in (0, 10, 20):
    print(f"\n  --- 往復コスト {cost}bps ---")
    print(f"  {'上位N':<8}{'全期間':<22}{'IS(22-23)':<22}{'OOS(24-)'}")
    for topN in (3, 5, 8, 10):
        r = longonly_nonoverlap(topN, cost, gated=True, excess=False)
        def fmt(t): return f"Sh{t[0]:.2f} ({t[1]:+.1f}%/20d)"
        print(f"  top{topN:<5}{fmt(r['all']):<22}{fmt(r['IS']):<22}{fmt(r['OOS'])}")

# ========================
# B. ベンチ超過 (50銘柄等加重 buy&hold を引いた純選択アルファ)
# ========================
print("\n" + "=" * 76)
print("B. ベンチ超過アルファ Sharpe (50銘柄等加重20日リターンを控除, ゲート無=選択力のみ)")
print("=" * 76)
print("   ※半導体メガキャップ相場βを除いた『V6が等加重より良い銘柄を選べるか』")
for cost in (0, 10):
    print(f"\n  --- 往復コスト {cost}bps ---")
    print(f"  {'上位N':<8}{'全期間':<22}{'IS(22-23)':<22}{'OOS(24-)'}")
    for topN in (3, 5, 8, 10):
        r = longonly_nonoverlap(topN, cost, gated=False, excess=True)
        def fmt(t): return f"Sh{t[0]:.2f} ({t[1]:+.2f}%/20d)"
        print(f"  top{topN:<5}{fmt(r['all']):<22}{fmt(r['IS']):<22}{fmt(r['OOS'])}")

# ========================
# C. ベンチマーク自体 (50銘柄等加重 buy&hold) の Sharpe
# ========================
print("\n" + "=" * 76)
print("C. ベンチマーク: 50銘柄等加重 buy&hold (非重複20日)")
print("=" * 76)
di_list = sorted(df['di'].unique())
by_di = {di: g for di, g in df.groupby('di')}
bench = {}
for k in ('all', 'IS', 'OOS'):
    phase_sh = []
    for phase in range(HOLD):
        recs = []
        for di in di_list[phase::HOLD]:
            g = by_di.get(di)
            if g is None:
                continue
            gg = g.dropna(subset=['fwd20'])
            if len(gg) < 10:
                continue
            recs.append({'date': gg['date'].iloc[0], 'ret': gg['fwd20'].mean()})
        if not recs:
            continue
        rd = pd.DataFrame(recs)
        sub = rd[period_mask(rd['date'], k)]['ret']
        if len(sub) >= 8:
            phase_sh.append(sharpe(sub))
    bench[k] = np.nanmean(phase_sh)
    print(f"  {k:<6} Sharpe={bench[k]:.2f}")

# ========================
# 図
# ========================
print("\n[作図]")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
for fp in ['/root/.fonts/NotoSansJP.ttf', '/usr/share/fonts/truetype/ipaexg/ipaexg.ttf']:
    if os.path.exists(fp):
        fm.fontManager.addfont(fp)
        plt.rcParams['font.family'] = fm.FontProperties(fname=fp).get_name()
        break
plt.rcParams['axes.unicode_minus'] = False

la = longonly_nonoverlap(5, 10, gated=True, excess=False)   # 絶対(top5,10bps,ゲート)
ex = longonly_nonoverlap(3, 10, gated=False, excess=True)    # 超過(top3,10bps)
periods = ['IS(22-23)', 'OOS(24-)', '全期間']
keys = ['IS', 'OOS', 'all']
abs_v = [la[k][0] for k in keys]
bm_v = [bench[k] for k in keys]
ex_v = [ex[k][0] for k in keys]

fig, ax = plt.subplots(figsize=(12, 6.75), facecolor='white')
x = np.arange(len(periods))
w = 0.26
b1 = ax.bar(x - w, abs_v, w, label='V6ロング特化 top5 (絶対, 10bps, ゲート有)', color='#2ea043')
b2 = ax.bar(x, bm_v, w, label='ベンチ: 50銘柄等加重 buy&hold', color='#8b949e')
b3 = ax.bar(x + w, ex_v, w, label='V6超過アルファ top3 (等加重控除=選択力)', color='#1f6feb')
for bars in (b1, b2, b3):
    for r in bars:
        ax.text(r.get_x() + r.get_width() / 2, r.get_height(), f'{r.get_height():.2f}',
                ha='center', va='bottom', fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(periods, fontsize=11)
ax.set_ylabel('Sharpe (非重複20日, √(252/20)年率)')
ax.axhline(2.0, color='#cf222e', ls=':', lw=1)
ax.text(0.01, 2.0, '昇格基準2.0', color='#cf222e', fontsize=9, va='bottom', transform=ax.get_yaxis_transform())
ax.legend(loc='upper left', fontsize=9.5, framealpha=0.95)
ax.grid(alpha=0.3, axis='y')
ax.set_title('V6ロング特化は buy&hold を超えるか? — 絶対 vs ベンチ vs 純選択アルファ',
             fontsize=13, fontweight='bold', pad=10)
fig.text(0.5, 0.90,
         '絶対Sharpeは等加重buy&holdと同程度=拾っているのは半導体メガキャップ相場β。'
         '超過アルファ(選択力)は正だがβヘッジ(ショート/先物)無しでは埋もれる。',
         ha='center', fontsize=10, color='#555')
fig.text(0.99, 0.01, 'データ: 2021-10〜2026-06 / 日本株日足(JQuants) / 非重複20日・往復10bps',
         ha='right', va='bottom', fontsize=8, color='gray')
fig.subplots_adjust(top=0.84, bottom=0.1, left=0.08, right=0.97)
plt.savefig(os.path.join(os.path.dirname(__file__), 'result_longonly.png'), dpi=100, facecolor='white')
print("  保存: result_longonly.png")
print("\n完了")
