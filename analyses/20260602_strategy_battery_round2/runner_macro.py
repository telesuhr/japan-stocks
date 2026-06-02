"""
バッチ1: 海外オーバーナイト・リードラグ 8本 (#1-8)

重要な執行制約: 海外オーバーナイトは JP 寄り前に判明する。
  → JPギャップ(前日終値→寄り)は事前に売買できず取れない(織込度の記述に留める)。
  → トレード可能なのは「寄りで見てから 寄り→引け(intra=close/open-1)」を取る形。
主ターゲット = intra (寄りエントリー→引け決済)。c2c(前日終値→当日終値)も参考表示。

#1 ADR→原株 (per-name timing)        ADR_xxxx overnight → 原株 intra
#2 ADR→原株 (cross-sectional L/S)     ADR16をランクし上位原株Long/下位Short intra
#3 米半導体(NVDA等6)→JP半導体          composite overnight → semis intra (timing)
#4 銅(HGc1)→非鉄                       overnight → 非鉄 intra (timing)
#5 USDJPY(JPY=)→輸出株                 overnight(円安=正) → 輸出株 intra (timing)
#6 WTI原油(CLc1)→商社/エネ             overnight → intra (timing)
#7 US10Y金利→邦銀                      yield change → 邦銀 intra (timing)
#8 VIX(VXc1)→日本株                    VIX change → 全universe intra (risk-off)

すべて signal は as-of(jp date t の前日以前で最新) = ルックアヘッドなし。
IS 2022-2023 / OOS 2024- / EVAL 2021-10-01。intra日次は非重複→√252正。コスト往復10bps。
"""
from __future__ import annotations
import os, sys, warnings
import psycopg2, pandas as pd, numpy as np
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

DB = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost/market_data")
IS_S, IS_E, OOS_S, EVAL_S = (pd.Timestamp(x) for x in ("2022-01-01", "2023-12-31", "2024-01-01", "2021-10-01"))
COST = 10 / 1e4  # 往復

# ---- ユニバース (5桁) ----
SEMI = ['80350','68570','69200','61460','77350','67230','69630','65260','40620','34360',
        '40630','77410','99840','285A0','65250','69540','67620','69760','69810']
NONFE = ['58030','50160','58010','58020','57130','57060','57110','57140']
EXPORT = ['72030','72670','67580','67620','68610','69710','69810','79740']
SHOSHA = ['16050','80310','80580']
BANK = ['83060','83160','84110']
ADR_MAP = {  # 原株5桁 -> ADRシンボル
    '65010':'ADR_6501','67580':'ADR_6758','67620':'ADR_6762','69020':'ADR_6902',
    '69200':'ADR_6920','72030':'ADR_7203','72670':'ADR_7267','77410':'ADR_7741',
    '79740':'ADR_7974','80350':'ADR_8035','83060':'ADR_8306','83160':'ADR_8316',
    '84110':'ADR_8411','94320':'ADR_9432','99830':'ADR_9983','99840':'ADR_9984'}
ALL_JP = sorted(set(SEMI+NONFE+EXPORT+SHOSHA+BANK+list(ADR_MAP)))
US_SEMI = ['NVDA','AMD','ASML','TSM','MU','AVGO']
MACRO = ['.SOX','HGc1','JPY=','CLc1','US10YT=RR','VXc1'] + US_SEMI + list(set(ADR_MAP.values()))


def fetch(sql):
    c = psycopg2.connect(DB); df = pd.read_sql(sql, c); c.close(); return df


def sharpe(r, ann=252):
    r = pd.Series(r).dropna()
    return float(r.mean()/r.std()*np.sqrt(ann)) if len(r) >= 10 and r.std() > 0 else float('nan')


print("="*76); print("バッチ1: 海外オーバーナイト・リードラグ 8本"); print("="*76)
print("\n[データ取得]")
jp = fetch(f"""SELECT code, date, open::float o, close::float c FROM stocks_daily
    WHERE code IN ({','.join(f"'{x}'" for x in ALL_JP)}) AND date>='2021-01-01' AND open>0 AND close>0
    ORDER BY code,date""")
jp['date'] = pd.to_datetime(jp['date'])
jp['o'] = pd.to_numeric(jp['o']); jp['c'] = pd.to_numeric(jp['c'])
jp = jp.sort_values(['code','date'])
jp['pc'] = jp.groupby('code')['c'].shift(1)
jp['intra'] = jp['c']/jp['o'] - 1          # 寄→引 (tradeable)
jp['gap'] = jp['o']/jp['pc'] - 1           # 前終→寄 (織込, not tradeable)
jp['c2c'] = jp['c']/jp['pc'] - 1
jp = jp[jp['date'] >= EVAL_S].dropna(subset=['intra','pc'])

mac = fetch(f"""SELECT symbol, trade_date d, close::float c FROM macro.daily_ohlcv
    WHERE symbol IN ({','.join(f"'{x}'" for x in MACRO)}) AND trade_date>='2021-01-01' ORDER BY symbol,trade_date""")
mac['d'] = pd.to_datetime(mac['d']); mac['c'] = pd.to_numeric(mac['c'])
# 各シンボルの overnight return 系列 (US10Yは水準=利回りなので階差)
sig = {}
for s, g in mac.groupby('symbol'):
    g = g.sort_values('d').set_index('d')['c']
    sig[s] = g.diff() if s == 'US10YT=RR' else g.pct_change()
jp_dates = sorted(jp['date'].unique())


def asof(sym_series):
    """jp date t に対し 前日以前で最新の値 (ルックアヘッドなし)"""
    out = {}
    s = sym_series.dropna()
    for t in jp_dates:
        v = s[s.index <= (t - pd.Timedelta(days=1))]
        out[t] = v.iloc[-1] if len(v) else np.nan
    return pd.Series(out)


def composite(syms):
    df = pd.concat([sig[s] for s in syms], axis=1)
    return df.mean(axis=1)


# ---- timing評価: signal(day-level) で universe平均intra を取れるか ----
def eval_timing(name, sigser, codes, flip=False):
    u = jp[jp['code'].isin(codes)].groupby('date').agg(intra=('intra','mean'),
                                                       gap=('gap','mean')).dropna()
    sg = asof(sigser).reindex(u.index)
    d = pd.DataFrame({'sig': sg, 'intra': u['intra'], 'gap': u['gap']}).dropna()
    if flip:
        d['sig'] = -d['sig']
    def stats(mask):
        sub = d[mask]
        if len(sub) < 20: return (np.nan,)*3
        rho, _ = spearmanr(sub['sig'], sub['intra'])
        # シグナル符号方向に寄→引を取る (long if sig>0, short if sig<0), コスト込み
        pos = np.sign(sub['sig']) * sub['intra'] - COST
        return rho, sharpe(pos), sub['intra'].mean()*1e4
    r_all, sh_all, m_all = stats(d.index >= EVAL_S)
    _, sh_is, _ = stats((d.index >= IS_S) & (d.index <= IS_E))
    _, sh_oos, _ = stats(d.index >= OOS_S)
    # 織込度: signal が gap をどれだけ説明するか
    rho_gap, _ = spearmanr(d['sig'], d['gap']) if len(d) > 20 else (np.nan, np.nan)
    print(f"  {name:<26} N={len(d):<5} ρ(sig,intra)={r_all:+.3f}  "
          f"Sh(符号方向,10bps) 全{sh_all:+.2f}/IS{sh_is:+.2f}/OOS{sh_oos:+.2f}  "
          f"ρ(sig,gap)={rho_gap:+.2f}")
    return sh_all, sh_oos


print("\n--- timing系 (#3-8) ---")
print("  ρ(sig,intra): 寄→引の継続性 / ρ(sig,gap): 寄りギャップへの織込度")
res = {}
res['#3 USsemi→JP半導体'] = eval_timing("#3 USsemi→JP半導体", composite(US_SEMI), SEMI)
res['#3b SOX→JP半導体'] = eval_timing("#3b SOX→JP半導体", sig['.SOX'], SEMI)
res['#4 銅→非鉄'] = eval_timing("#4 銅→非鉄", sig['HGc1'], NONFE)
res['#5 USDJPY→輸出株'] = eval_timing("#5 USDJPY→輸出株", sig['JPY='], EXPORT)
res['#6 WTI→商社'] = eval_timing("#6 WTI→商社/エネ", sig['CLc1'], SHOSHA)
res['#7 US10Y→邦銀'] = eval_timing("#7 US10Y利回り→邦銀", sig['US10YT=RR'], BANK)
res['#8 VIX→日本株(flip)'] = eval_timing("#8 VIX→日本株(リスクオフ)", sig['VXc1'], ALL_JP, flip=True)


# ---- #1 ADR→原株 per-name timing / #2 cross-sectional L/S ----
print("\n--- ADRリードラグ (#1 per-name, #2 cross-sectional) ---")
# 各原株について ADR overnight をas-of結合、intra/gap を見る
rows = []
adr_asof = {code: asof(sig[adr]) for code, adr in ADR_MAP.items()}
for code, adr in ADR_MAP.items():
    sub = jp[jp['code'] == code].set_index('date')
    s = adr_asof[code].reindex(sub.index)
    dd = pd.DataFrame({'adr': s, 'intra': sub['intra'], 'gap': sub['gap'], 'code': code}).dropna()
    rows.append(dd)
adf = pd.concat(rows).reset_index().rename(columns={'index': 'date'})

# #1 per-name: 全名プールで ADR overnight vs intra
rho1, _ = spearmanr(adf['adr'], adf['intra'])
rho1g, _ = spearmanr(adf['adr'], adf['gap'])
# 符号方向 intra (long if adr>0)
pos1 = np.sign(adf['adr']) * adf['intra'] - COST
print(f"  #1 ADR→原株(全名プール) N={len(adf)} ρ(adr,intra)={rho1:+.3f} "
      f"ρ(adr,gap)={rho1g:+.3f}(織込) 符号方向Sh={sharpe(pos1):+.2f}")

# #2 cross-sectional: 各日 ADR overnight でランク, 上位3原株Long/下位3Short の intra
ls = []
for dt, g in adf.groupby('date'):
    if len(g) < 8: continue
    r = g.sort_values('adr', ascending=False)
    lng = r.head(3)['intra'].mean(); sht = r.tail(3)['intra'].mean()
    ls.append({'date': dt, 'ls': lng - sht - 2*COST, 'long': lng, 'short': sht})
lsd = pd.DataFrame(ls).set_index('date').sort_index()
for lab, m in [('全', lsd.index >= EVAL_S), ('IS', (lsd.index >= IS_S)&(lsd.index <= IS_E)), ('OOS', lsd.index >= OOS_S)]:
    pass
print(f"  #2 ADR cross-sectional L/S(上位3原株Long/下位3Short intra, 10bps×2): "
      f"全{sharpe(lsd[lsd.index>=EVAL_S]['ls']):+.2f}/"
      f"IS{sharpe(lsd[(lsd.index>=IS_S)&(lsd.index<=IS_E)]['ls']):+.2f}/"
      f"OOS{sharpe(lsd[lsd.index>=OOS_S]['ls']):+.2f}")

print("\n[結論] 海外オーバーナイトは寄りギャップに織込まれ、寄→引(intra)の継続性は弱い/逆 が想定")
print("完了")
