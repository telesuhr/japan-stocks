"""
寄り前シグナル → 同日イントラ(寄り→引け)継続性の検証

問い: 朝のオープン前に分かる情報で「今日イントラで強い銘柄」を当てられるか?
  対象 = 主戦22銘柄(半導体14 + 非鉄8)
  「強さ」の定義(ユーザー選択) = 直近モメンタム + 高値更新・ブレイク

寄り前に使えるシグナル(すべて t-1 終値時点までの情報):
  mom5     直近5日リターン      close[t-1]/close[t-6]-1
  mom20    直近20日リターン     close[t-1]/close[t-21]-1
  hi20_gap 20日高値接近度        close[t-1]/max(high[t-20..t-1])-1   (0=高値, 負=下)
  new_hi20 20日高値ブレイク      close[t-1] >= 直近20日高値
  d25      MA25乖離             close[t-1]/MA25-1
  ycls_pos 前日引けの強さ        (close-low)/(high-low) [t-1]  (1=高値引け)
  sox_ovn  SOXオーバーナイト     .SOX 当日寄り前の最新リターン (半導体地合い)
  cu_ovn   銅オーバーナイト      HGc1 (非鉄地合い)

ターゲット(day t, 寄りエントリー→引け/前引け決済):
  intra    close[t]/open[t]-1      寄り→大引け (メイン)
  am       morning_close[t]/open[t]-1  寄り→前引け (前場デイトレ)

評価:
  A. クロスセクショナル Spearman IC (各シグナル vs 同日intra/am), 全/IS/OOS
  B. top5 Long (寄り→引け, 日次√252, コスト0/10/20bps), 全/IS/OOS
  C. day-level: SOXオーバーナイトで「今日はGOか」を当てられるか(ユニバース平均intra)

IS 2022-01-01〜2023-12-31 / OOS 2024-01-01〜 / EVAL 2021-10-01〜
日次・寄り→引けは非重複 → √252 年率化は正しい(オーバーラップの幻なし)
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

# 主戦22銘柄 (5桁 canonical) — 半導体14 + 非鉄8
SEMI = ['80350', '68570', '69200', '61460', '77350', '67230', '69630', '65260',
        '40620', '34360', '40630', '77410', '99840', '285A0']
NONFE = ['58030', '50160', '58010', '58020', '57130', '57060', '57110', '57140']
UNI22 = SEMI + NONFE
CODE_LIST = ','.join(f"'{c}'" for c in UNI22)
SEMI_SET = set(SEMI)

IS_START = pd.Timestamp("2022-01-01")
IS_END = pd.Timestamp("2023-12-31")
OOS_START = pd.Timestamp("2024-01-01")
EVAL_START = pd.Timestamp("2021-10-01")


def fetch(sql: str) -> pd.DataFrame:
    conn = psycopg2.connect(DB_URL)
    df = pd.read_sql(sql, conn)
    conn.close()
    return df


def sharpe(rets, ann: float = 252) -> float:
    r = pd.Series(rets).dropna()
    if len(r) < 10 or r.std() == 0:
        return float("nan")
    return float(r.mean() / r.std() * np.sqrt(ann))


def ic_series(d: pd.DataFrame, fac: str, ret: str) -> pd.Series:
    ics = []
    for _, g in d.groupby('date'):
        sub = g[[fac, ret]].dropna()
        if len(sub) < 5:
            continue
        ic, _ = spearmanr(sub[fac], sub[ret])
        ics.append(ic)
    return pd.Series(ics)


def icir(ic: pd.Series) -> float:
    ic_ = ic.dropna()
    if len(ic_) < 10 or ic_.std() == 0:
        return float("nan")
    return float(ic_.mean() / ic_.std() * np.sqrt(252))


print("=" * 76)
print("寄り前シグナル → 同日イントラ(寄り→引け)継続性の検証 (主戦22銘柄)")
print("=" * 76)
print("\n[データ取得中]")

px = fetch(f"""
    SELECT code, date, open::float o, high::float h, low::float l, close::float c,
           morning_close::float mc, adj_close::float ac
    FROM stocks_daily
    WHERE code IN ({CODE_LIST}) AND date >= '2021-01-01' AND close > 0 AND open > 0
    ORDER BY code, date
""")
px['date'] = pd.to_datetime(px['date'])
for col in ['o', 'h', 'l', 'c', 'mc', 'ac']:
    px[col] = pd.to_numeric(px[col], errors='coerce')

sox = fetch("SELECT trade_date d, close::float c FROM macro.daily_ohlcv WHERE symbol='.SOX' ORDER BY trade_date")
cu = fetch("SELECT trade_date d, close::float c FROM macro.daily_ohlcv WHERE symbol='HGc1' ORDER BY trade_date")
for m in (sox, cu):
    m['d'] = pd.to_datetime(m['d'])
sox['ret'] = sox['c'].pct_change()
cu['ret'] = cu['c'].pct_change()
sox = sox.set_index('d')['ret']
cu = cu.set_index('d')['ret']
print(f"  銘柄数: {px['code'].nunique()} / 行数: {len(px):,}")

# ---- シグナルとターゲットを銘柄ごとに計算 ----
rows = []
for code, g in px.groupby('code'):
    g = g.sort_values('date').reset_index(drop=True)
    c = g['c'].values
    h = g['h'].values
    l = g['l'].values
    o = g['o'].values
    mc = g['mc'].values
    dts = g['date'].values
    n = len(g)
    for i in range(26, n):
        if pd.Timestamp(dts[i]) < EVAL_START:
            continue
        cm1 = c[i - 1]
        # シグナル (t-1 まで)
        mom5 = cm1 / c[i - 6] - 1
        mom20 = cm1 / c[i - 21] - 1
        hh20 = h[i - 20:i].max()
        hi20_gap = cm1 / hh20 - 1
        new_hi20 = 1.0 if cm1 >= hh20 else 0.0
        ma25 = c[i - 25:i].mean()
        d25 = cm1 / ma25 - 1
        rng = h[i - 1] - l[i - 1]
        ycls_pos = (c[i - 1] - l[i - 1]) / rng if rng > 0 else 0.5
        # ターゲット (day t, 寄りエントリー)
        if o[i] <= 0:
            continue
        intra = c[i] / o[i] - 1
        am = mc[i] / o[i] - 1 if (not np.isnan(mc[i]) and mc[i] > 0) else np.nan
        rows.append({'code': code, 'date': pd.Timestamp(dts[i]),
                     'mom5': mom5, 'mom20': mom20, 'hi20_gap': hi20_gap,
                     'new_hi20': new_hi20, 'd25': d25, 'ycls_pos': ycls_pos,
                     'intra': intra, 'am': am,
                     'semi': 1 if code in SEMI_SET else 0})

df = pd.DataFrame(rows)
# day-level overnight (as-of: jp date の最新 <= t-1)
jp_dates = sorted(df['date'].unique())
sox_ovn = {}
cu_ovn = {}
for t in jp_dates:
    prior = t - pd.Timedelta(days=1)
    s = sox[sox.index <= prior]
    cv = cu[cu.index <= prior]
    sox_ovn[t] = s.iloc[-1] if len(s) else np.nan
    cu_ovn[t] = cv.iloc[-1] if len(cv) else np.nan
df['sox_ovn'] = df['date'].map(sox_ovn)
df['cu_ovn'] = df['date'].map(cu_ovn)
print(f"  サンプル(銘柄×日): {len(df):,}  期間: {df['date'].min().date()}〜{df['date'].max().date()}")

SIGNALS = ['mom5', 'mom20', 'hi20_gap', 'new_hi20', 'd25', 'ycls_pos']
PERIODS = [("全期間", df['date'] >= EVAL_START),
           ("IS(22-23)", (df['date'] >= IS_START) & (df['date'] <= IS_END)),
           ("OOS(24-)", df['date'] >= OOS_START)]


# ===== A. クロスセクショナル IC =====
print("\n" + "=" * 76)
print("A. クロスセクショナル Spearman IC (各シグナル vs 同日 寄り→引け intra)")
print("=" * 76)
print(f"  {'シグナル':<10}{'全期間IC':>10}{'ICIR':>7}{'IS_IC':>9}{'OOS_IC':>9}")
for sig in SIGNALS:
    line = f"  {sig:<10}"
    icir_all = None
    vals = {}
    for plabel, mask in PERIODS:
        sub = df[mask][['date', sig, 'intra']].dropna()
        ic = ic_series(sub, sig, 'intra')
        vals[plabel] = ic
    icall = vals['全期間']
    line += f"{icall.mean():>+10.4f}{icir(icall):>7.2f}"
    line += f"{vals['IS(22-23)'].mean():>+9.4f}{vals['OOS(24-)'].mean():>+9.4f}"
    print(line)
print("  注: stocks_daily の morning_close は未投入のため寄り→前引け(am)は日足から算出不可")


# ===== B. top5 Long (寄り→引け) =====
print("\n" + "=" * 76)
print("B. top5 Long (寄り前シグナル上位5 を寄りで買い→引けで決済, 日次√252)")
print("=" * 76)


def topn_long(sig, mask, topn=5, cost_bps=0.0):
    cost = cost_bps / 1e4
    daily = []
    for dt, g in df[mask].dropna(subset=[sig, 'intra']).groupby('date'):
        if len(g) < 10:
            continue
        top = g.sort_values(sig, ascending=False).head(topn)
        daily.append(top['intra'].mean() - cost)
    return pd.Series(daily)


print(f"  {'シグナル':<10}{'全(0bps)':>10}{'全(10)':>9}{'全(20)':>9}{'IS(10)':>9}{'OOS(10)':>9}")
best_sig, best_oos = None, -99
for sig in SIGNALS:
    s_all0 = sharpe(topn_long(sig, df['date'] >= EVAL_START, 5, 0))
    s_all10 = sharpe(topn_long(sig, df['date'] >= EVAL_START, 5, 10))
    s_all20 = sharpe(topn_long(sig, df['date'] >= EVAL_START, 5, 20))
    s_is = sharpe(topn_long(sig, (df['date'] >= IS_START) & (df['date'] <= IS_END), 5, 10))
    s_oos = sharpe(topn_long(sig, df['date'] >= OOS_START, 5, 10))
    print(f"  {sig:<10}{s_all0:>10.2f}{s_all10:>9.2f}{s_all20:>9.2f}{s_is:>9.2f}{s_oos:>9.2f}")
    if not np.isnan(s_oos) and s_oos > best_oos:
        best_oos, best_sig = s_oos, sig

# ベンチ: 22銘柄を寄りで全部買って引けで売る (毎日)
bench = df[df['date'] >= EVAL_START].groupby('date')['intra'].mean()
print(f"\n  ベンチ: 22銘柄等加重 寄り→引け 毎日  Sharpe(10bps)="
      f"{sharpe(bench - 10/1e4):.2f}  平均={bench.mean()*1e4:+.1f}bps/日")
print(f"  → top5が等加重ベンチを超えるか(=銘柄選択が効くか)が肝")


# ===== C. day-level: SOXオーバーナイトで今日はGOか =====
print("\n" + "=" * 76)
print("C. day-level: SOXオーバーナイトで「今日は半導体GOか」を当てられるか")
print("=" * 76)
semi_day = df[(df['semi'] == 1) & (df['date'] >= EVAL_START)].groupby('date').agg(
    intra=('intra', 'mean'), sox_ovn=('sox_ovn', 'first')).dropna()
rho, p = spearmanr(semi_day['sox_ovn'], semi_day['intra'])
print(f"  半導体ユニバース平均(寄り→引け) vs SOXオーバーナイト  ρ={rho:+.3f} (p={p:.3f}, N={len(semi_day)})")
hi = semi_day[semi_day['sox_ovn'] > 0]['intra']
lo = semi_day[semi_day['sox_ovn'] <= 0]['intra']
print(f"  SOX陽線翌日: 半導体 寄り→引け 平均={hi.mean()*1e4:+.1f}bps (N={len(hi)})")
print(f"  SOX陰線翌日: 半導体 寄り→引け 平均={lo.mean()*1e4:+.1f}bps (N={len(lo)})")
# 非鉄 vs 銅
nf_day = df[(df['semi'] == 0) & (df['date'] >= EVAL_START)].groupby('date').agg(
    intra=('intra', 'mean'), cu_ovn=('cu_ovn', 'first')).dropna()
rho2, p2 = spearmanr(nf_day['cu_ovn'], nf_day['intra'])
print(f"  非鉄ユニバース平均(寄り→引け) vs 銅オーバーナイト  ρ={rho2:+.3f} (p={p2:.3f}, N={len(nf_day)})")

print(f"\n  >>> best OOS signal: {best_sig} (top5 OOS Sharpe 10bps={best_oos:.2f})")

# ===== 図 =====
print("\n[作図中]")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
for fp in ['/root/.fonts/NotoSansJP.ttf']:
    if os.path.exists(fp):
        fm.fontManager.addfont(fp)
        plt.rcParams['font.family'] = fm.FontProperties(fname=fp).get_name()
        break
plt.rcParams['axes.unicode_minus'] = False

fig = plt.figure(figsize=(12, 6.75), facecolor='white')
gs = fig.add_gridspec(2, 2, hspace=0.5, wspace=0.28, left=0.09, right=0.97, top=0.85, bottom=0.13)

# 左上: 各シグナルの全期間IC
ax1 = fig.add_subplot(gs[0, 0])
ics_all = []
for sig in SIGNALS:
    sub = df[df['date'] >= EVAL_START][['date', sig, 'intra']].dropna()
    ics_all.append(ic_series(sub, sig, 'intra').mean())
colors = ['#cf222e' if v < 0 else '#1f6feb' for v in ics_all]
ax1.barh(SIGNALS, ics_all, color=colors)
ax1.axvline(0, color='#333', lw=0.8)
for i, v in enumerate(ics_all):
    ax1.text(v, i, f'{v:+.3f}', va='center',
             ha='left' if v >= 0 else 'right', fontsize=8)
ax1.set_title('① 寄り前シグナル vs 同日寄り→引け IC', fontsize=10)
ax1.set_xlabel('Spearman IC (全期間)')

# 右上: top5 Long Sharpe (10bps) IS/OOS, best signal vs bench
ax2 = fig.add_subplot(gs[0, 1])
labels_p = ['IS', 'OOS', '全期間']
bs_sharpe = [
    sharpe(topn_long(best_sig, (df['date'] >= IS_START) & (df['date'] <= IS_END), 5, 10)),
    sharpe(topn_long(best_sig, df['date'] >= OOS_START, 5, 10)),
    sharpe(topn_long(best_sig, df['date'] >= EVAL_START, 5, 10)),
]
bench_sharpe = [
    sharpe(df[(df['date'] >= IS_START) & (df['date'] <= IS_END)].groupby('date')['intra'].mean() - 10/1e4),
    sharpe(df[df['date'] >= OOS_START].groupby('date')['intra'].mean() - 10/1e4),
    sharpe(bench - 10/1e4),
]
x = np.arange(3)
ax2.bar(x - 0.2, bs_sharpe, 0.4, label=f'top5({best_sig})', color='#1f6feb')
ax2.bar(x + 0.2, bench_sharpe, 0.4, label='22銘柄等加重', color='#8b949e')
ax2.set_xticks(x); ax2.set_xticklabels(labels_p)
ax2.axhline(0, color='#333', lw=0.8)
ax2.set_title('② top5 Long vs 等加重 (寄→引, 10bps)', fontsize=10)
ax2.set_ylabel('日次Sharpe (√252)'); ax2.legend(fontsize=8)
ax2.grid(alpha=0.3, axis='y')

# 左下: SOX条件付き半導体イントラ
ax3 = fig.add_subplot(gs[1, 0])
ax3.bar(['SOX陽線\n翌日', 'SOX陰線\n翌日'], [hi.mean()*1e4, lo.mean()*1e4],
        color=['#2ea043', '#cf222e'])
for i, v in enumerate([hi.mean()*1e4, lo.mean()*1e4]):
    ax3.text(i, v, f'{v:+.1f}bps', ha='center', va='bottom' if v >= 0 else 'top', fontsize=9)
ax3.axhline(0, color='#333', lw=0.8)
ax3.set_title(f'③ SOXオーバーナイト→半導体 寄→引 (ρ={rho:+.2f})', fontsize=10)
ax3.set_ylabel('平均 bps/日')

# 右下: best signal の累積エクイティ (寄→引, 10bps)
ax4 = fig.add_subplot(gs[1, 1])
eq_top = topn_long(best_sig, df['date'] >= EVAL_START, 5, 10)
dts_eq = [dt for dt, g in df[df['date'] >= EVAL_START].dropna(subset=[best_sig, 'intra']).groupby('date') if len(g) >= 10]
eq = (1 + eq_top).cumprod()
benchcum = (1 + (bench - 10/1e4)).cumprod()
ax4.plot(dts_eq, eq.values, color='#1f6feb', lw=1.6, label=f'top5 {best_sig}')
ax4.plot(bench.index, benchcum.values, color='#8b949e', lw=1.2, ls='--', label='22等加重')
ax4.axvline(OOS_START, color='#cf222e', ls=':', lw=1)
ax4.set_title('④ 累積成長 (寄→引・10bps)', fontsize=10)
ax4.set_ylabel('×'); ax4.legend(fontsize=8); ax4.grid(alpha=0.3)

fig.suptitle('寄り前シグナルで「今日イントラで強い銘柄」を当てられるか — 主戦22銘柄',
             fontsize=14, fontweight='bold', y=0.95)
fig.text(0.99, 0.01,
         f'データ: {df["date"].min().date()}〜{df["date"].max().date()} / 日本株日足(JQuants) 寄→引 / SOX・銅 overnight / IS-OOS=2024',
         ha='right', va='bottom', fontsize=8, color='gray')
out = os.path.dirname(__file__)
plt.savefig(os.path.join(out, 'result.png'), dpi=100, facecolor='white')
print("  保存: result.png")
print("\n完了")
