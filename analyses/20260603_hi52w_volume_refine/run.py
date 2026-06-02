"""
#10「52週高値×出来高急増」精密化 — 昇格(Sharpe≥2.0)を狙えるか

第三十弾 batch2 #10 (IC+0.053/ICIR+1.26/L/S Sh0.85, +122bps月) の押上げ検証。
要素分解で何が効くかを切り分ける:
  (1) Long/Short どちら側が効くか (top/bottom分位 vs 市場 の市場超過)
  (2) 出来高急増 閾値スイープ (vsurge条件 / vsurge連続シグナル化)
  (3) 保有期間 H=10/20/40 スイープ
  (4) ボラ調整 (シグナルのボラ正規化 / 1/vol ウェイト)
  (5) 複合スコア (zscore(hi_prox)+zscore(log vsurge)) vs 単体
  (6) 年別Sharpe (レジーム依存チェック) / コスト感度

流動性上位500・月次非重複・セクター中立デミーン・√(252/H)年率化。
IS 2022-2023 / OOS 2024- / EVAL 2021-10-01。
すべてシグナルは close[t] までの情報のみ、フォワードは t以降のadj_close。
"""
from __future__ import annotations
import os, sys, warnings
import psycopg2, pandas as pd, numpy as np
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

DB = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost/market_data")
IS_S, IS_E, OOS_S, EVAL_S = (pd.Timestamp(x) for x in ("2022-01-01", "2023-12-31", "2024-01-01", "2021-10-01"))
NTOP = 500


def fetch(sql):
    c = psycopg2.connect(DB); df = pd.read_sql(sql, c); c.close(); return df


def sharpe(r, ann):
    r = pd.Series(r).dropna()
    return float(r.mean()/r.std()*np.sqrt(ann)) if len(r) >= 8 and r.std() > 0 else float('nan')


print("="*78); print("#10 52週高値×出来高急増 精密化"); print("="*78)

uni = fetch(f"""SELECT code FROM stocks_daily WHERE date>='2021-01-01' AND turnover_value>0
    GROUP BY code HAVING count(*)>900 ORDER BY avg(turnover_value) DESC LIMIT {NTOP}""")
codes = uni['code'].tolist()
sec = fetch("SELECT code5, sector33 FROM symbol_master")
sec_map = dict(zip(sec['code5'], sec['sector33']))
print(f"ユニバース N={len(codes)}")

px = fetch(f"""SELECT code, date, adj_close::float ac, volume::float v FROM stocks_daily
    WHERE code IN ({','.join(f"'{x}'" for x in codes)}) AND date>='2020-06-01' AND adj_close>0
    ORDER BY code,date""")
px['date'] = pd.to_datetime(px['date'])
px['sector'] = px['code'].map(sec_map)
px = px.sort_values(['code', 'date'])
g = px.groupby('code')
px['ret1'] = g['ac'].pct_change()
px['vol20'] = g['ret1'].transform(lambda s: s.rolling(20).std())
px['max252'] = g['ac'].transform(lambda s: s.rolling(252).max())
px['hi_prox'] = px['ac']/px['max252']                       # 52週高値接近 (1.0=高値)
px['avgv20'] = g['v'].transform(lambda s: s.rolling(20).mean())
px['avgv60'] = g['v'].transform(lambda s: s.rolling(60).mean())
px['vsurge'] = px['avgv20']/px['avgv60']
px['lvsurge'] = np.log(px['vsurge'].clip(0.2, 5))
# ボラ調整: 高値からの距離をボラで割る (低ボラで高値接近を優遇)
px['hi_voladj'] = (px['hi_prox'] - 1)/px['vol20']           # 0に近い=高値かつ低ボラ

dates = sorted(px['date'].unique())
dates_eval = [d for d in dates if d >= EVAL_S]


def zscore(s):
    sd = s.std()
    return (s - s.mean())/sd if sd > 0 else s*0


def run(name, signal_fn, H=20, cost=20/1e4, vthr=None, neutral=True, volwt=False, verbose=False):
    """signal_fn(df)->Series。月次非重複L/S(上位/下位5分位) + Long-only市場超過。"""
    reb = dates_eval[::H]
    ls, lo, ics, yrs = [], [], [], {}
    seg = {'IS': [], 'OOS': []}
    for dt in reb:
        d = px[px['date'] == dt].copy()
        # フォワードH日
        future = px[px['code'].isin(d['code'])]
        # 各codeのac[t+H]
        idxmap = {c: gg for c, gg in future.groupby('code')}
        d = d.dropna(subset=['hi_prox', 'vol20', 'vsurge', 'sector'])
        if vthr is not None:
            d = d[d['vsurge'] >= vthr]
        if len(d) < 40:
            continue
        fwd = []
        for _, r in d.iterrows():
            gg = idxmap[r['code']]
            pos = gg.index[gg['date'] == dt]
            if len(pos) == 0:
                fwd.append(np.nan); continue
            arr = gg['ac'].values
            i = gg.index.get_loc(pos[0])
            fwd.append(arr[i+H]/arr[i]-1 if i+H < len(arr) else np.nan)
        d['fwd'] = fwd
        d = d.dropna(subset=['fwd'])
        if len(d) < 40:
            continue
        s = signal_fn(d).astype(float)
        if neutral:
            s = s.groupby(d['sector']).transform(lambda x: x - x.mean())
        d['sig'] = s
        d = d.dropna(subset=['sig'])
        rho, _ = spearmanr(d['sig'], d['fwd'])
        ics.append(rho)
        mkt = d['fwd'].mean()
        q = d['sig'].rank(pct=True)
        topd, botd = d[q >= 0.8], d[q <= 0.2]
        if volwt:
            wt = 1/topd['vol20']; lng = np.average(topd['fwd'], weights=wt)
            wb = 1/botd['vol20']; sht = np.average(botd['fwd'], weights=wb)
        else:
            lng, sht = topd['fwd'].mean(), botd['fwd'].mean()
        ls_r = (lng - sht) - cost
        lo_r = (lng - mkt) - cost/2
        ls.append((dt, ls_r)); lo.append((dt, lo_r))
        yrs.setdefault(dt.year, []).append(ls_r)
        if IS_S <= dt <= IS_E: seg['IS'].append(ls_r)
        elif dt >= OOS_S: seg['OOS'].append(ls_r)
    ann = 252/H
    lsd = pd.Series(dict(ls)); lod = pd.Series(dict(lo))
    ic = np.nanmean(ics); icir = ic/np.nanstd(ics)*np.sqrt(ann) if np.nanstd(ics) > 0 else np.nan
    print(f"\n[{name}] H={H} cost={cost*1e4:.0f}bps vthr={vthr} neutral={neutral} volwt={volwt}")
    print(f"  IC={ic:+.3f} ICIR={icir:+.2f}  reb={len(lsd)}")
    print(f"  L/S Sh 全{sharpe(lsd,ann):+.2f}/IS{sharpe(seg['IS'],ann):+.2f}/OOS{sharpe(seg['OOS'],ann):+.2f}  月平均{lsd.mean()*1e4:+.0f}bps")
    print(f"  Long-only(市場超過) Sh 全{sharpe(lod,ann):+.2f}  月平均{lod.mean()*1e4:+.0f}bps")
    if verbose:
        yr = {y: sharpe(v, ann) for y, v in sorted(yrs.items())}
        print("  年別L/S Sh:", " ".join(f"{y}:{s:+.2f}" for y, s in yr.items()))
    return lsd, lod


# ===== (A) ベースライン再現 + Long/Short分解 =====
print("\n" + "-"*78); print("(A) ベースライン再現 + Long/Short分解"); print("-"*78)
run("base: hi_prox (vsurge>1.2)", lambda d: d['hi_prox'], vthr=1.2, verbose=True)

# ===== (B) 出来高閾値スイープ =====
print("\n" + "-"*78); print("(B) 出来高急増 閾値スイープ (hi_prox シグナル)"); print("-"*78)
for vt in [None, 1.0, 1.2, 1.5, 2.0]:
    run(f"hi_prox vthr={vt}", lambda d: d['hi_prox'], vthr=vt)

# ===== (C) シグナル設計: 単体 vs 複合 vs ボラ調整 =====
print("\n" + "-"*78); print("(C) シグナル設計比較 (vthr=1.2)"); print("-"*78)
run("hi_prox 単体", lambda d: d['hi_prox'], vthr=1.2)
run("vsurge 単体", lambda d: d['lvsurge'], vthr=None)
run("複合 z(hi)+z(lvsurge)", lambda d: zscore(d['hi_prox'])+zscore(d['lvsurge']), vthr=None, verbose=True)
run("hi_voladj (高値×低ボラ)", lambda d: d['hi_voladj'], vthr=1.2)
run("複合×ボラ調整 z(hi_voladj)+z(lvsurge)", lambda d: zscore(d['hi_voladj'])+zscore(d['lvsurge']), vthr=None, verbose=True)

# ===== (D) 保有期間スイープ (複合シグナル) =====
print("\n" + "-"*78); print("(D) 保有期間スイープ (複合 z(hi)+z(lvsurge))"); print("-"*78)
sig_comp = lambda d: zscore(d['hi_prox'])+zscore(d['lvsurge'])
for H in [10, 20, 40]:
    run(f"複合 H={H}", sig_comp, H=H)

# ===== (E) ボラ加重L/S + コスト感度 (最良候補) =====
print("\n" + "-"*78); print("(E) ボラ加重 & コスト感度 (複合 H=20)"); print("-"*78)
for cost in [10/1e4, 20/1e4, 30/1e4]:
    run(f"複合 volwt cost={cost*1e4:.0f}", sig_comp, H=20, cost=cost, volwt=True, verbose=(cost==20/1e4))

# ===== (F) 最良候補 hi_voladj の追い込み =====
print("\n" + "-"*78); print("(F) 最良候補 hi_voladj 追い込み (年別/H/vthr/Long-only)"); print("-"*78)
sig_va = lambda d: d['hi_voladj']
run("hi_voladj vthr=1.2 H=20 (年別)", sig_va, H=20, vthr=1.2, verbose=True)
for H in [20, 40, 60]:
    run(f"hi_voladj H={H} vthr=1.2", sig_va, H=H, vthr=1.2)
for vt in [1.0, 1.1, 1.3]:
    run(f"hi_voladj vthr={vt} H=20", sig_va, H=20, vthr=vt)
print("\n  --- Long-only(市場超過) コスト感度: 実運用=高値接近・低ボラ銘柄を買い ---")
for cost in [5/1e4, 10/1e4, 20/1e4]:
    run(f"hi_voladj LO cost={cost*1e4:.0f}", sig_va, H=20, vthr=1.2, cost=cost)

print("\n完了")
