"""
V6スコア + 早期利食い(テイクプロフィット)検証

問い: 「一定ライン行ったらすぐ利食いをするようなスコアとしても使えないか?」
  = 固定20日保有ではなく、エントリー後に含み益が閾値(+2/+3/+5/+8%)に達したら
    即座に手仕舞いする運用にすると、V6 は実用的なスコアになるか?

設計:
  - 各営業日、V6 上位 N 銘柄をロングエントリー(N225 60日-3%ベアゲート適用)
  - エントリー後、終値ベースで含み益が target に到達した最初の日に利食い
  - 到達しなければ MAX_HOLD 日で手仕舞い
  - 比較対象: 同じエントリーの「固定20日保有」(利食いなし)

評価:
  per-trade: 平均リターン / 勝率 / 平均保有日数 / 日あたりリターン / √(252/平均保有)年率Sharpe
  portfolio: 非重複(全位相平均)の日次ポートフォリオ Sharpe (資本リサイクルあり)

往復コスト(0/10/20bps)も適用。すべてグロス→ネット併記で過大主張を避ける。
"""
from __future__ import annotations

import os
import sys
import psycopg2
import pandas as pd
import numpy as np

sys.stdout.reconfigure(line_buffering=True)

DB_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost/market_data")

UNI22 = [
    '80350', '68570', '69200', '61460', '77350', '67230', '69630', '65260',
    '40620', '34360', '40630', '77410', '99840', '285A0',
    '58030', '50160', '58010', '58020', '57130', '57060', '57110', '57140',
]
ADD28 = [
    '16050', '40040', '60980', '65010', '65030', '65250', '67580', '67620',
    '68610', '69540', '69710', '69760', '69810', '70110', '70120', '70130',
    '72030', '72670', '79740', '80310', '80580', '83060', '83160', '84110',
    '87660', '94320', '94330', '99830',
]
UNI50 = UNI22 + ADD28
CODES4_50 = [c[:4] for c in UNI50]
CODE_LIST = ','.join(f"'{c}'" for c in UNI50)

IS_START = pd.Timestamp("2022-01-01")
IS_END = pd.Timestamp("2023-12-31")
OOS_START = pd.Timestamp("2024-01-01")
EVAL_START = pd.Timestamp("2021-10-01")
GATE = -0.03
MAX_HOLD = 20
TOPN = 5
TARGETS = [0.02, 0.03, 0.05, 0.08]  # 利食い閾値


def fetch(sql: str) -> pd.DataFrame:
    conn = psycopg2.connect(DB_URL)
    df = pd.read_sql(sql, conn)
    conn.close()
    return df


def sharpe(rets, ann: float) -> float:
    r = pd.Series(rets).dropna()
    if len(r) < 10 or r.std() == 0:
        return float("nan")
    return float(r.mean() / r.std() * np.sqrt(ann))


print("=" * 76)
print("V6スコア + 早期利食い(テイクプロフィット) 検証")
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

pivot = prices.pivot(index='date', columns='c', values='ac').sort_index()
all_dates = list(pivot.index)
cols = [c for c in CODES4_50 if c in pivot.columns]
print(f"  銘柄: {len(cols)} / 営業日: {len(all_dates)}")

n225_aligned = n225['c'].reindex(all_dates).ffill()
n225_r60 = n225_aligned / n225_aligned.shift(60) - 1


# ============ V6 パネル (インデックス i 保持) ============
def compute_panel() -> pd.DataFrame:
    rows = []
    arr = {c: pivot[c].values for c in cols}
    for i, dt in enumerate(all_dates):
        if dt < EVAL_START:
            continue
        gate_on = bool(n225_r60.iloc[i] < GATE) if pd.notna(n225_r60.iloc[i]) else False
        for c in cols:
            a_full = arr[c]
            # asof までの非欠損
            seg = a_full[:i + 1]
            seg = seg[~np.isnan(seg)]
            if len(seg) < 76:
                continue
            last = seg[-1]
            r20 = last / seg[-21] - 1
            rets = seg[-20:] / seg[-21:-1] - 1
            sd = float(np.std(rets))
            vol20 = sd * np.sqrt(252)
            r20adj = r20 / vol20 if vol20 > 0 else 0.0
            ma75 = seg[-75:].mean()
            d75 = last / ma75 - 1
            v6 = r20adj + 0.5 * d75 * 10
            rows.append({'i': i, 'date': dt, 'code': c, 'v6': v6,
                         'gate': gate_on, 'entry': last})
    return pd.DataFrame(rows)


print("\n[因子計算中] ...")
panel = compute_panel()
print(f"  サンプル: {len(panel):,}")


def exit_return(code: str, i_entry: int, entry_px: float, target: float):
    """エントリー翌日以降の終値で利食い判定。
       戻り値: (実現リターン, 保有日数)。利食い未達なら MAX_HOLD 日終値で清算。"""
    a = pivot[code].values
    last_k = min(MAX_HOLD, len(a) - 1 - i_entry)
    if last_k < 1:
        return None
    realized, held = None, last_k
    for k in range(1, last_k + 1):
        px = a[i_entry + k]
        if np.isnan(px):
            continue
        ret = px / entry_px - 1
        if ret >= target:
            realized, held = ret, k
            break
    if realized is None:
        # MAX_HOLD 日終値で清算
        px = a[i_entry + last_k]
        if np.isnan(px):
            return None
        realized, held = px / entry_px - 1, last_k
    return realized, held


def fixed_return(code: str, i_entry: int, entry_px: float):
    a = pivot[code].values
    last_k = min(MAX_HOLD, len(a) - 1 - i_entry)
    if last_k < 1:
        return None
    px = a[i_entry + last_k]
    if np.isnan(px):
        return None
    return px / entry_px - 1, last_k


# ============ 各日 top-N エントリーのトレード生成 ============
def gen_trades(target):
    """target が None なら固定保有。トレードリスト(dict)を返す。"""
    trades = []
    for dt, g in panel.groupby('date'):
        if g['gate'].iloc[0]:
            continue
        top = g.sort_values('v6', ascending=False).head(TOPN)
        for _, row in top.iterrows():
            if target is None:
                res = fixed_return(row['code'], row['i'], row['entry'])
            else:
                res = exit_return(row['code'], row['i'], row['entry'], target)
            if res is None:
                continue
            ret, held = res
            trades.append({'date': dt, 'i': int(row['i']), 'code': row['code'],
                           'ret': ret, 'held': held})
    return pd.DataFrame(trades)


def period_mask(s, plabel):
    if plabel == "全期間":
        return s >= EVAL_START
    if plabel == "IS":
        return (s >= IS_START) & (s <= IS_END)
    return s >= OOS_START


PERIODS = ["全期間", "IS", "OOS"]


def per_trade_stats(tr, cost_bps=0.0):
    """per-trade 統計。Sharpe は √(252/平均保有日数) で年率化。"""
    out = {}
    cost = cost_bps / 1e4  # 往復
    for p in PERIODS:
        sub = tr[period_mask(tr['date'], p)].copy()
        if len(sub) < 10:
            out[p] = None
            continue
        net = sub['ret'] - cost
        avg_hold = sub['held'].mean()
        ann = 252 / avg_hold
        out[p] = {
            'n': len(sub),
            'mean': net.mean() * 100,
            'win': (net > 0).mean() * 100,
            'hold': avg_hold,
            'per_day_bps': net.mean() / avg_hold * 1e4,
            'sharpe': sharpe(net, ann),
        }
    return out


# ============ 非重複ポートフォリオ Sharpe (資本リサイクル) ============
def portfolio_sharpe2(target, cost_bps=0.0):
    """日次ポートフォリオ: 毎日 top-N 新規エントリーすると重複するので、
       MAX_HOLD 日ごとにリバランスする位相を 0..MAX_HOLD-1 まで回し平均。
       各リバランスで top-N を等加重で建て、各銘柄は利食い/最大保有で個別に手仕舞い、
       手仕舞い後〜次リバランスまでは現金(0)。同一日付の複数位相は平均して日次Sharpe。"""
    cost = cost_bps / 1e4
    by_i = {i: g for i, g in panel.groupby('i')}
    eval_idx = sorted(by_i.keys())
    records = []  # (date, ret)
    for phase in range(MAX_HOLD):
        rb_points = [ix for ix in eval_idx if (ix - eval_idx[0]) % MAX_HOLD == phase]
        for rb in rb_points:
            g = by_i[rb]
            span = min(MAX_HOLD, len(all_dates) - 1 - rb)
            if span < 1:
                continue
            if g['gate'].iloc[0]:
                for k in range(1, span + 1):
                    records.append((all_dates[rb + k], 0.0))
                continue
            top = g.sort_values('v6', ascending=False).head(TOPN)
            slot_daily = np.zeros((TOPN, span))
            for s_idx, (_, row) in enumerate(top.iterrows()):
                a = pivot[row['code']].values
                entry_px = row['entry']
                prev_px = entry_px
                exited = False
                for k in range(1, span + 1):
                    if exited:
                        continue
                    px = a[rb + k]
                    if np.isnan(px):
                        continue
                    day_ret = px / prev_px - 1
                    prev_px = px
                    cumret = px / entry_px - 1
                    slot_daily[s_idx, k - 1] = day_ret
                    if (target is not None and cumret >= target) or k == span:
                        slot_daily[s_idx, k - 1] -= cost
                        exited = True
            port = slot_daily.mean(axis=0)
            for k in range(1, span + 1):
                records.append((all_dates[rb + k], port[k - 1]))
    rec = pd.DataFrame(records, columns=['date', 'ret'])
    # 同一日付に複数位相が重なる→平均(独立位相の平均ポートフォリオ)
    daily = rec.groupby('date')['ret'].mean()
    res = {}
    for p in PERIODS:
        res[p] = sharpe(daily[period_mask(daily.index.to_series(), p)], 252)
    return res


# ============ 実行 ============
print("\n" + "=" * 76)
print(f"A. per-trade 比較 (top{TOPN} エントリー, 最大{MAX_HOLD}日, グロス)")
print("=" * 76)

fixed_tr = gen_trades(None)
print(f"\n  [固定{MAX_HOLD}日保有] (利食いなし)")
st = per_trade_stats(fixed_tr)
print(f"    {'期間':<8}{'n':>6}{'平均%':>9}{'勝率%':>8}{'保有日':>8}{'日bps':>8}{'Sharpe':>9}")
for p in PERIODS:
    s = st[p]
    if s:
        print(f"    {p:<8}{s['n']:>6}{s['mean']:>9.2f}{s['win']:>8.1f}"
              f"{s['hold']:>8.1f}{s['per_day_bps']:>8.1f}{s['sharpe']:>9.2f}")

tp_trades = {}
for tg in TARGETS:
    tr = gen_trades(tg)
    tp_trades[tg] = tr
    print(f"\n  [利食い +{tg*100:.0f}%] (到達で即手仕舞い・未達は{MAX_HOLD}日)")
    st = per_trade_stats(tr)
    print(f"    {'期間':<8}{'n':>6}{'平均%':>9}{'勝率%':>8}{'保有日':>8}{'日bps':>8}{'Sharpe':>9}")
    for p in PERIODS:
        s = st[p]
        if s:
            print(f"    {p:<8}{s['n']:>6}{s['mean']:>9.2f}{s['win']:>8.1f}"
                  f"{s['hold']:>8.1f}{s['per_day_bps']:>8.1f}{s['sharpe']:>9.2f}")

# コスト感応 (全期間, 10bps)
print("\n" + "=" * 76)
print("B. コスト込み per-trade Sharpe (全期間, 往復10bps)")
print("=" * 76)
print(f"  固定{MAX_HOLD}日: ", end="")
s = per_trade_stats(fixed_tr, 10)["全期間"]
print(f"Sharpe={s['sharpe']:.2f} 平均{s['mean']:+.2f}% 保有{s['hold']:.1f}日")
for tg in TARGETS:
    s = per_trade_stats(tp_trades[tg], 10)["全期間"]
    print(f"  利食い+{tg*100:.0f}%: Sharpe={s['sharpe']:.2f} 平均{s['mean']:+.2f}% "
          f"保有{s['hold']:.1f}日 勝率{s['win']:.0f}%")

# ポートフォリオ Sharpe (資本リサイクル, 非重複位相平均)
print("\n" + "=" * 76)
print(f"C. 非重複ポートフォリオ日次Sharpe (top{TOPN}等加重・利食い後現金・10bps)")
print("=" * 76)
ps_fixed = portfolio_sharpe2(None, 10)
print(f"  {'戦略':<14}{'全期間':>9}{'IS':>9}{'OOS':>9}")
print(f"  {'固定'+str(MAX_HOLD)+'日':<14}"
      f"{ps_fixed['全期間']:>9.2f}{ps_fixed['IS']:>9.2f}{ps_fixed['OOS']:>9.2f}")
ps_tp = {}
for tg in TARGETS:
    ps = portfolio_sharpe2(tg, 10)
    ps_tp[tg] = ps
    print(f"  {'利食い+'+str(int(tg*100))+'%':<14}"
          f"{ps['全期間']:>9.2f}{ps['IS']:>9.2f}{ps['OOS']:>9.2f}")

# ============ 図 ============
print("\n[作図中]")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

for fp in ['/root/.fonts/NotoSansJP.ttf',
           '/usr/share/fonts/truetype/ipaexg/ipaexg.ttf',
           '/usr/share/fonts/opentype/ipaexfont-gothic/ipaexg.ttf']:
    if os.path.exists(fp):
        fm.fontManager.addfont(fp)
        plt.rcParams['font.family'] = fm.FontProperties(fname=fp).get_name()
        break
plt.rcParams['axes.unicode_minus'] = False

fig = plt.figure(figsize=(12, 6.75), facecolor='white')
gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.26,
                      left=0.08, right=0.97, top=0.85, bottom=0.12)

labels = [f'固定{MAX_HOLD}日'] + [f'+{int(t*100)}%' for t in TARGETS]

# 左上: 平均トレードリターン (全期間)
ax1 = fig.add_subplot(gs[0, 0])
means = [per_trade_stats(fixed_tr)["全期間"]['mean']] + \
        [per_trade_stats(tp_trades[t])["全期間"]['mean'] for t in TARGETS]
colors = ['#8b949e'] + ['#1f6feb'] * len(TARGETS)
ax1.bar(labels, means, color=colors)
for i, v in enumerate(means):
    ax1.text(i, v, f'{v:.1f}%', ha='center', va='bottom', fontsize=8)
ax1.set_ylabel('平均トレードリターン %')
ax1.set_title('① 早期利食いは平均リターンを削る', fontsize=10)
ax1.grid(alpha=0.3, axis='y')

# 右上: 勝率 vs 平均保有日数
ax2 = fig.add_subplot(gs[0, 1])
wins = [per_trade_stats(fixed_tr)["全期間"]['win']] + \
       [per_trade_stats(tp_trades[t])["全期間"]['win'] for t in TARGETS]
holds = [per_trade_stats(fixed_tr)["全期間"]['hold']] + \
        [per_trade_stats(tp_trades[t])["全期間"]['hold'] for t in TARGETS]
axb = ax2.twinx()
ax2.bar(labels, wins, color='#2ea043', alpha=0.7, label='勝率%')
axb.plot(labels, holds, color='#cf222e', marker='o', lw=2, label='平均保有日')
ax2.set_ylabel('勝率 %', color='#2ea043')
axb.set_ylabel('平均保有日数', color='#cf222e')
ax2.set_title('② 利食いで勝率↑保有日数↓', fontsize=10)
ax2.grid(alpha=0.3, axis='y')

# 左下: per-trade Sharpe (コスト10bps, 全期間)
ax3 = fig.add_subplot(gs[1, 0])
shs = [per_trade_stats(fixed_tr, 10)["全期間"]['sharpe']] + \
      [per_trade_stats(tp_trades[t], 10)["全期間"]['sharpe'] for t in TARGETS]
ax3.bar(labels, shs, color=colors)
for i, v in enumerate(shs):
    ax3.text(i, v, f'{v:.2f}', ha='center', va='bottom', fontsize=8)
ax3.set_ylabel('per-trade Sharpe')
ax3.set_title('③ per-trade Sharpe (10bps, √(252/保有)年率)', fontsize=10)
ax3.grid(alpha=0.3, axis='y')

# 右下: ポートフォリオ Sharpe OOS (資本リサイクル)
ax4 = fig.add_subplot(gs[1, 1])
x = np.arange(len(labels))
allv = [ps_fixed['全期間']] + [ps_tp[t]['全期間'] for t in TARGETS]
oosv = [ps_fixed['OOS']] + [ps_tp[t]['OOS'] for t in TARGETS]
ax4.bar(x - 0.2, allv, 0.4, label='全期間', color='#8b949e')
ax4.bar(x + 0.2, oosv, 0.4, label='OOS', color='#1f6feb')
ax4.set_xticks(x)
ax4.set_xticklabels(labels, fontsize=8)
ax4.set_ylabel('ポートフォリオ日次Sharpe')
ax4.set_title('④ 非重複ポートフォリオSharpe (資本リサイクル)', fontsize=10)
ax4.legend(fontsize=8)
ax4.grid(alpha=0.3, axis='y')

fig.suptitle('V6スコア + 早期利食い検証 — 「勝ち逃げ」は機能するか?',
             fontsize=14, fontweight='bold', y=0.95)
fig.text(0.99, 0.01,
         f'データ: {EVAL_START.date()}〜{all_dates[-1].date()} / 日本株日足(JQuants) / top{TOPN}・最大{MAX_HOLD}日保有・ベアゲート適用',
         ha='right', va='bottom', fontsize=8, color='gray')

out = os.path.dirname(__file__)
plt.savefig(os.path.join(out, 'result_takeprofit.png'), dpi=100, facecolor='white')
print(f"  保存: result_takeprofit.png")
print("\n完了")
