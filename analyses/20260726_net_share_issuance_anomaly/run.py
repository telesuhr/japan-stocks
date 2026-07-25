import sys
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, "/mnt/d/Root/ClaudeCode/01_Trading/japan-stocks")
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from jstock import db, costs, stats

# ------------------------------------------------------------------
# #25 自社株数減少（net share issuance, NSI）アノマリー検証
#   外部: Pontiff-Woodgate 2008 / Japan pre-registered study (Pac-Basin 2025)
#   仮説: 分割調整後のネット発行済株数(ShOutFY-TrShFY)がYoYで減る銘柄(=自社株買い/消却)
#         がその後1年アウトパフォーム。増える銘柄(=希薄化)がアンダーパフォーム。
#   規律: 分割調整 / コスト両側 / IS-OOS / サイズ・セクター中立 / 年次非重複
# ------------------------------------------------------------------

# ---------- 1. FY 株数レコード抽出 ----------
def getf(p, k):
    if isinstance(p, str):
        p = json.loads(p)
    v = p.get(k, '')
    try:
        return float(v)
    except Exception:
        return np.nan

def load_fy_shares():
    rows = db.read_sql("""
        SELECT code, disc_date, cur_per_en, payload
        FROM fin_summary
        WHERE cur_per_type='FY'
          AND doc_type IN ('FYFinancialStatements_Consolidated_JP',
                           'FYFinancialStatements_Consolidated_IFRS',
                           'FYFinancialStatements_Consolidated_US',
                           'FYFinancialStatements_NonConsolidated_JP')
        ORDER BY code, cur_per_en""", [])
    rows['ShOut'] = rows['payload'].apply(lambda p: getf(p, 'ShOutFY'))
    rows['TrSh'] = rows['payload'].apply(lambda p: getf(p, 'TrShFY'))
    rows = rows.drop(columns=['payload'])
    rows['net'] = rows['ShOut'] - rows['TrSh'].fillna(0.0)
    rows = rows.dropna(subset=['ShOut'])
    rows = rows[rows['net'] > 0]
    rows['disc_date'] = pd.to_datetime(rows['disc_date'])
    rows['cur_per_en'] = pd.to_datetime(rows['cur_per_en'])
    # code,FY末で最終開示(訂正反映)を採用
    rows = rows.sort_values('disc_date').groupby(['code', 'cur_per_en'], as_index=False).last()
    return rows

# ---------- 2. 価格 + 分割ファクター(rho=close/adj_close) ----------
def load_px():
    px = db.read_sql("""
        SELECT code, date, close, adj_close, turnover_value
        FROM stocks_daily WHERE date>='2016-01-01'
        ORDER BY code, date""", [])
    px['date'] = pd.to_datetime(px['date'])
    px['rho'] = px['close'] / px['adj_close']          # d 以後の累積分割係数
    return px

def build_code_arrays(px):
    """code -> dict(dates, rho, turn) の事前計算配列(searchsorted用)。"""
    d = {}
    for code, g in px.groupby('code'):
        g = g.sort_values('date')
        d[code] = {
            'dates': g['date'].values.astype('datetime64[ns]'),
            'rho': g['rho'].values.astype(float),
            'turn': g['turnover_value'].values.astype(float),
        }
    return d

def rho_at(CA, code, dt):
    a = CA.get(code)
    if a is None:
        return np.nan
    i = np.searchsorted(a['dates'], np.datetime64(dt), side='right') - 1
    return a['rho'][i] if i >= 0 else np.nan

def adv_at(CA, code, dt, win=60):
    a = CA.get(code)
    if a is None:
        return np.nan
    i = np.searchsorted(a['dates'], np.datetime64(dt), side='right')
    if i < 20:
        return np.nan
    return np.nanmean(a['turn'][max(0, i - win):i]) / 1e8

# ---------- 3. 分割調整 NSI ----------
def build_nsi(fy, CA):
    fy = fy.sort_values(['code', 'cur_per_en']).copy()
    fy['net_prev'] = fy.groupby('code')['net'].shift(1)
    fy['en_prev'] = fy.groupby('code')['cur_per_en'].shift(1)
    fy = fy.dropna(subset=['net_prev'])
    fy['rho_cur'] = [rho_at(CA, c, d) for c, d in zip(fy['code'], fy['cur_per_en'])]
    fy['rho_prev'] = [rho_at(CA, c, d) for c, d in zip(fy['code'], fy['en_prev'])]
    # 区間内分割 = rho_prev/rho_cur。前期株数を当期ベースへ。
    fy['split'] = fy['rho_prev'] / fy['rho_cur']
    fy['net_prev_adj'] = fy['net_prev'] * fy['split']
    fy['nsi'] = fy['net'] / fy['net_prev_adj'] - 1.0
    fy = fy.dropna(subset=['nsi'])
    # 分割誤検知の外れ値を軽くクリップ(±60%)。真の巨大増資/消却は稀
    fy = fy[fy['nsi'].abs() < 0.6]
    return fy

# ---------- 4. 年次L/Sポートフォリオ ----------
def daily_ret_panel(px):
    px = px.sort_values(['code', 'date'])
    px['ret'] = px.groupby('code')['adj_close'].pct_change()
    return px.pivot_table(index='date', columns='code', values='ret')

def run_portfolios(fy, px, CA, adv_min=5.0):
    retp = daily_ret_panel(px)
    # サイズ(時価総額代理=net*close)も取得用に close panel
    closep = px.pivot_table(index='date', columns='code', values='close')
    forms = []
    for Y in range(2018, 2026):
        fdate = pd.Timestamp(f'{Y}-07-01')
        end = pd.Timestamp(f'{Y+1}-07-01')
        # formation時点で利用可能な最新FY(過去12ヶ月以内に開示)
        avail = fy[(fy['disc_date'] <= fdate) & (fy['disc_date'] > fdate - pd.Timedelta(days=400))]
        avail = avail.sort_values('disc_date').groupby('code', as_index=False).last()
        # 流動性フィルタ
        codes = [c for c in avail['code']
                 if (adv_at(CA, c, fdate) or np.nan) >= adv_min]
        sub = avail[avail['code'].isin(codes)].copy()
        if len(sub) < 100:
            continue
        # サイズ代理(log mktcap) でサイズ中立化用の残差NSIも作る
        px_form = closep[closep.index <= fdate]
        last_close = px_form.iloc[-1] if len(px_form) else pd.Series(dtype=float)
        sub['close'] = sub['code'].map(last_close)
        sub['mktcap'] = sub['net'] * sub['close']
        sub['lnmc'] = np.log(sub['mktcap'].replace(0, np.nan))
        sub = sub.dropna(subset=['lnmc'])
        # 生NSIのクインタイル
        sub['q'] = pd.qcut(sub['nsi'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5]).astype(int)
        # サイズ中立: lnmc で回帰した残差NSI
        x = sub['lnmc'].values
        y = sub['nsi'].values
        b = np.polyfit(x, y, 1)
        sub['nsi_resid'] = y - (b[0] * x + b[1])
        sub['q_sn'] = pd.qcut(sub['nsi_resid'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5]).astype(int)
        forms.append((fdate, end, sub))
    return forms, retp

LS_RT_BPS = 8.0  # L/S 往復コスト(bps)。年次リバランスにつき各formationで1回だけ控除

def basket_daily(forms, retp, qcol, apply_cost=False):
    """Q1(減少=Long) - Q5(増加=Short) の日次L/S系列。
    apply_cost=True で各formation初日に往復8bpsを1回だけ控除(年次リバランス)。"""
    ls = []
    for fdate, end, sub in forms:
        win = retp[(retp.index > fdate) & (retp.index <= end)]
        longc = [c for c in sub[sub[qcol] == 1]['code'] if c in retp.columns]
        shortc = [c for c in sub[sub[qcol] == 5]['code'] if c in retp.columns]
        if not longc or not shortc:
            continue
        seg = (win[longc].mean(axis=1) - win[shortc].mean(axis=1)).rename('ls')
        if apply_cost and len(seg):
            seg.iloc[0] = seg.iloc[0] - LS_RT_BPS / 10_000.0
        ls.append(seg)
    s = pd.concat(ls).sort_index()
    return s

def q_spread_check(forms, retp, qcol):
    """各クインタイルの年次平均リターン(単調性確認・生)。"""
    out = {q: [] for q in [1, 2, 3, 4, 5]}
    for fdate, end, sub in forms:
        win = retp[(retp.index > fdate) & (retp.index <= end)]
        for q in [1, 2, 3, 4, 5]:
            cs = [c for c in sub[sub[qcol] == q]['code'] if c in retp.columns]
            if cs:
                out[q].append((1 + win[cs].mean(axis=1)).prod() - 1)
    return {q: np.mean(v) for q, v in out.items()}


if __name__ == '__main__':
    print('loading FY shares...')
    fy0 = load_fy_shares()
    print('loading prices...')
    px = load_px()
    print('building code arrays...')
    CA = build_code_arrays(px)
    print('building split-adjusted NSI...')
    fy = build_nsi(fy0, CA)
    print('NSI obs after split-adj & clip:', len(fy))
    print(fy['nsi'].describe(percentiles=[.05, .1, .25, .5, .75, .9, .95]).to_string())

    forms, retp = run_portfolios(fy, px, CA, adv_min=5.0)
    print('\nformation years used:', [f[0].year for f in forms])
    for f in forms:
        print(' ', f[0].date(), 'N=', len(f[2]))

    results = {}
    for label, qcol in [('生NSI', 'q'), ('サイズ中立NSI', 'q_sn')]:
        ls = basket_daily(forms, retp, qcol, apply_cost=False)
        net = basket_daily(forms, retp, qcol, apply_cost=True)   # 各年formationで往復8bps×1
        rep_g = stats.summary(ls, f'{label} gross')
        rep_n = stats.summary(net, f'{label} net')
        qsp = q_spread_check(forms, retp, qcol)
        results[label] = dict(ls=ls, net=net, gross=rep_g, netrep=rep_n, qsp=qsp)
        print(f'\n===== {label} =====')
        print('quintile 年次平均リターン(生, Q1=株数減少 .. Q5=増加):')
        for q in [1, 2, 3, 4, 5]:
            print(f'  Q{q}: {qsp[q]*100:+.2f}%')
        print('gross:', rep_g)
        print('net  :', rep_n)

    # IS/OOS 分割 (生NSI)
    net = results['生NSI']['net']
    is_mask = net.index < pd.Timestamp('2022-01-01')
    print('\n===== IS/OOS (生NSI net) =====')
    print('IS  (~2021):', stats.summary(net[is_mask], 'IS'))
    print('OOS (2022~):', stats.summary(net[~is_mask], 'OOS'))
    net_sn = results['サイズ中立NSI']['net']
    print('IS  SN:', stats.summary(net_sn[net_sn.index < pd.Timestamp('2022-01-01')], 'IS_SN'))
    print('OOS SN:', stats.summary(net_sn[net_sn.index >= pd.Timestamp('2022-01-01')], 'OOS_SN'))

    # 保存
    save = {}
    for label in results:
        save[label + '_gross'] = results[label]['gross']
        save[label + '_net'] = results[label]['netrep']
    pd.DataFrame(save).T.to_csv('results.csv', encoding='utf-8-sig')
    results['生NSI']['ls'].to_frame('ls_raw').join(
        results['サイズ中立NSI']['ls'].to_frame('ls_sizeneu'), how='outer').to_csv('daily_ls.csv')

    # ---------- 可視化 ----------
    try:
        fm.fontManager.addfont('/root/.fonts/NotoSansJP.ttf')
        plt.rcParams['font.family'] = 'Noto Sans JP'
    except Exception:
        pass
    plt.rcParams['axes.unicode_minus'] = False
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.2), facecolor='white')

    # 左: クインタイル単調性
    for label, col in [('生NSI', '#d62728'), ('サイズ中立NSI', '#1f77b4')]:
        qsp = results[label]['qsp']
        ax1.plot([1, 2, 3, 4, 5], [qsp[q] * 100 for q in [1, 2, 3, 4, 5]],
                 'o-', color=col, label=label)
    ax1.axhline(0, color='gray', lw=0.8)
    ax1.set_xlabel('NSIクインタイル (Q1=株数減少/自社株買い ← → Q5=増加/希薄化)')
    ax1.set_ylabel('翌1年 平均リターン %')
    ax1.set_title('株数減少クインタイルほど高リターンか?')
    ax1.legend(); ax1.grid(alpha=0.3)

    # 右: L/S 累積
    for label, col in [('生NSI', '#d62728'), ('サイズ中立NSI', '#1f77b4')]:
        net = results[label]['net']
        cum = (1 + net).cumprod()
        ax2.plot(cum.index, cum.values, color=col,
                 label=f"{label} (net Sh {results[label]['netrep'].get('sharpe', float('nan')):.2f})")
    ax2.axhline(1, color='gray', lw=0.8)
    ax2.set_ylabel('累積 (Q1 Long − Q5 Short, コスト後)')
    ax2.set_title('NSI L/S 累積損益')
    ax2.legend(); ax2.grid(alpha=0.3)

    fig.suptitle('自社株数減少(NSI)アノマリー検証 — 日本株 2018-2026', fontsize=15)
    fig.text(0.99, 0.01, 'データ: fin_summary(FY株数)×stocks_daily / 分割調整・往復8bps / 年次リバランス',
             ha='right', va='bottom', fontsize=8, color='gray')
    fig.savefig('result.png', dpi=100, bbox_inches='tight', facecolor='white')
    print('\nsaved result.png / results.csv / daily_ls.csv')
