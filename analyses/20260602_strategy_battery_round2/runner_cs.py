"""
バッチ2: クロスセクション条件付き 4本 (#9-12)

#9  残余モメンタム (セクター中立)   12-1ヶ月モメンタムをsector33内でデミーン → 上位Long/下位Short
#10 52週高値×出来高急増             52週高値接近 AND 出来高急増 を満たす銘柄のフォワード
#11 モメンタム加速 (5d vs 20d)      z(r5) - z(r20) が大きい=加速 銘柄のフォワード
#12 セクター別空売り比率極値反転     セクター短売比率が極値→翌週セクター平均が反転するか

共通: 流動性上位~500銘柄。月次(20営業日)リバランス・非重複。
ターゲット = フォワード20日 adj_close リターン。L/S はセクター中立デミーン後の上位/下位5分位。
IS 2022-2023 / OOS 2024- / EVAL 2021-10-01。コスト L/S 往復20bps/リバランス。√(252/20)年率化。
すべて signal は close[t] までの情報のみ(ルックアヘッドなし)、フォワードは t+1以降。
"""
from __future__ import annotations
import os, sys, warnings
import psycopg2, pandas as pd, numpy as np
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

DB = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost/market_data")
IS_S, IS_E, OOS_S, EVAL_S = (pd.Timestamp(x) for x in ("2022-01-01", "2023-12-31", "2024-01-01", "2021-10-01"))
H = 20                 # フォワード保有日数 / リバランス間隔
COST = 20 / 1e4        # L/S 往復/リバランス
NTOP = 500             # 流動性ユニバース


def fetch(sql):
    c = psycopg2.connect(DB); df = pd.read_sql(sql, c); c.close(); return df


def sharpe(r, ann_factor):
    r = pd.Series(r).dropna()
    return float(r.mean()/r.std()*np.sqrt(ann_factor)) if len(r) >= 8 and r.std() > 0 else float('nan')


print("="*76); print("バッチ2: クロスセクション条件付き 4本 (#9-12)"); print("="*76)

# ---- 流動性ユニバース ----
print("\n[ユニバース選定: 流動性上位500]")
uni = fetch(f"""SELECT code FROM stocks_daily WHERE date>='2021-01-01' AND turnover_value>0
    GROUP BY code HAVING count(*)>900 ORDER BY avg(turnover_value) DESC LIMIT {NTOP}""")
codes = uni['code'].tolist()
print(f"  N={len(codes)}銘柄")

sec = fetch("SELECT code5, sector33 FROM symbol_master")
sec_map = dict(zip(sec['code5'], sec['sector33']))

print("[価格取得]")
px = fetch(f"""SELECT code, date, adj_close::float ac, volume::float v FROM stocks_daily
    WHERE code IN ({','.join(f"'{x}'" for x in codes)}) AND date>='2020-06-01' AND adj_close>0
    ORDER BY code,date""")
px['date'] = pd.to_datetime(px['date'])
px['sector'] = px['code'].map(sec_map)
px = px.sort_values(['code', 'date'])
g = px.groupby('code')
px['r5'] = g['ac'].pct_change(5)
px['r20'] = g['ac'].pct_change(20)
px['r252'] = g['ac'].pct_change(252)
px['r21'] = g['ac'].pct_change(21)
px['mom121'] = (1+px['r252'])/(1+px['r21']) - 1          # 12-1ヶ月 (直近1ヶ月除外)
px['vol20'] = g['ac'].pct_change().rolling(20).std().reset_index(0, drop=True)
px['max252'] = g['ac'].transform(lambda s: s.rolling(252).max())
px['hi_prox'] = px['ac']/px['max252']                    # 52週高値接近度 (1.0=高値)
px['avgv20'] = g['v'].transform(lambda s: s.rolling(20).mean())
px['avgv60'] = g['v'].transform(lambda s: s.rolling(60).mean())
px['vsurge'] = px['avgv20']/px['avgv60']                 # 出来高急増
# フォワード20日リターン (t+1 から計測: ac[t+H]/ac[t]-1, エントリーは翌日想定で近似)
px['fwd'] = g['ac'].shift(-H)/px['ac'] - 1

dates = sorted(px['date'].unique())
dates = [d for d in dates if d >= EVAL_S]
reb_dates = dates[::H]   # 非重複リバランス


def zscore(s):
    return (s - s.mean())/s.std() if s.std() > 0 else s*0


def eval_cs(name, sigcol, neutral_sector=True, cond=None):
    """sigcol を月次CSランク, セクター中立デミーン, 上位/下位5分位L/S。IC も。"""
    ics, ls_ret, mkt = [], [], []
    seg = {'IS': [], 'OOS': []}
    for dt in reb_dates:
        d = px[px['date'] == dt].dropna(subset=[sigcol, 'fwd', 'sector']).copy()
        if cond is not None:
            d = d[cond(d)]
        if len(d) < 40:
            continue
        s = d[sigcol].astype(float)
        if neutral_sector:
            s = s.groupby(d['sector']).transform(lambda x: x - x.mean())
        d['sig'] = s
        rho, _ = spearmanr(d['sig'], d['fwd'])
        ics.append(rho)
        q = d['sig'].rank(pct=True)
        lng = d[q >= 0.8]['fwd'].mean()
        sht = d[q <= 0.2]['fwd'].mean()
        ret = (lng - sht) - COST
        ls_ret.append((dt, ret))
        mkt.append(d['fwd'].mean())
        if dt <= IS_E and dt >= IS_S: seg['IS'].append(ret)
        elif dt >= OOS_S: seg['OOS'].append(ret)
    ann = 252/H
    lsd = pd.Series(dict(ls_ret))
    ic = np.nanmean(ics); icir = ic/np.nanstd(ics)*np.sqrt(ann) if np.nanstd(ics) > 0 else np.nan
    print(f"  {name:<30} reb={len(lsd):<3} IC={ic:+.3f} ICIR={icir:+.2f}  "
          f"L/S Sh(20bps) 全{sharpe(lsd, ann):+.2f}/IS{sharpe(seg['IS'], ann):+.2f}/OOS{sharpe(seg['OOS'], ann):+.2f}  "
          f"L/S月平均={lsd.mean()*1e4:+.0f}bps")


print(f"\n--- 月次({H}営業日)リバランス・セクター中立L/S (上位/下位5分位) ---")
eval_cs("#9 残余モメンタム(12-1,sec中立)", 'mom121', neutral_sector=True)
# #10 52週高値接近 AND 出来高急増 を条件に、その中で高値接近度をシグナルに
eval_cs("#10 52週高値×出来高急増", 'hi_prox', neutral_sector=True,
        cond=lambda d: (d['vsurge'] > 1.2))
# #11 モメンタム加速: z(r5)-z(r20)
px['accel'] = np.nan
for dt in reb_dates:
    m = px['date'] == dt
    sub = px.loc[m]
    px.loc[m, 'accel'] = (zscore(sub['r5']) - zscore(sub['r20'])).values
eval_cs("#11 モメンタム加速(z r5 - z r20)", 'accel', neutral_sector=True)


# ---- #12 セクター別空売り比率極値反転 ----
print("\n--- #12 セクター別 空売り比率 極値反転 ---")
sr = fetch("""SELECT date, s33, sell_ex_short_va::float a, shrt_with_res_va::float b, shrt_no_res_va::float c
    FROM jquants_short_ratio WHERE date>='2021-01-01' ORDER BY date""")
sr['date'] = pd.to_datetime(sr['date'])
sr['sratio'] = (sr['b']+sr['c'])/(sr['a']+sr['b']+sr['c'])
# セクター平均フォワード20日(銘柄等加重)
secfwd = px.dropna(subset=['fwd', 'sector']).groupby(['date', 'sector'])['fwd'].mean().reset_index()
secfwd = secfwd.rename(columns={'sector': 's33'})
m = sr.merge(secfwd, on=['date', 's33'], how='inner')
# 各日: セクター横断で sratio をランク, 極値高(短売多)=反発狙いLong / 極値低=Short, 月次
ls12, seg12 = [], {'IS': [], 'OOS': []}
for dt in sorted(m['date'].unique())[::H]:
    d = m[m['date'] == dt].dropna(subset=['sratio', 'fwd'])
    if len(d) < 10: continue
    q = d['sratio'].rank(pct=True)
    lng = d[q >= 0.8]['fwd'].mean()   # 短売比率高セクター → 反発Long
    sht = d[q <= 0.2]['fwd'].mean()
    ret = (lng - sht) - COST
    ls12.append((dt, ret))
    if IS_S <= dt <= IS_E: seg12['IS'].append(ret)
    elif dt >= OOS_S: seg12['OOS'].append(ret)
ann = 252/H
ls12d = pd.Series(dict(ls12))
print(f"  #12 短売比率極値→反発Long/低Short  reb={len(ls12d)}  "
      f"Sh(20bps) 全{sharpe(ls12d, ann):+.2f}/IS{sharpe(seg12['IS'], ann):+.2f}/OOS{sharpe(seg12['OOS'], ann):+.2f}  "
      f"月平均={ls12d.mean()*1e4:+.0f}bps")

print("\n完了")
