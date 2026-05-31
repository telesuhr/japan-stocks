"""
クロージングオークション下落側Long — 実約定リアリズム精査

精密版で「引けで売られた銘柄(jump<=-50bps)を引け板寄せ買い→翌朝決済」が
net Sharpe2.0-2.8と判明。本分析は実約定の現実性を精査:

分解で判明: エッジ本体=オーバーナイトギャップ(close30→翌open +21.7bps)。
翌日中は中央値-14bpsで剥落 → 寄付き約定スピードが死活的。

検証:
  A. 翌日 exit時刻別の減衰カーブ (09:00寄り→09:05→09:15→09:30→10:00→11:30)
     どこまでにexitすればエッジが残るか
  B. 各exit時刻の Sharpe・コスト後 (寄り遅延のコスト)
  C. entry現実性: close30(板寄せ)で買えるか = MOC買い。jump大きいほど約定確実だが
     反発も大きい。entryを翌寄りにしたら(MOC使えない場合)エッジ消えるか
  D. 銘柄サイズ別 (ADV) でエッジ差

データ: stocks_intraday 15:30(entry) + 翌日 09:00/09:05/09:15/09:30/10:00/11:30(exit)
"""
from __future__ import annotations
import os, sys
import psycopg2
import pandas as pd
import numpy as np

sys.stdout.reconfigure(line_buffering=True)
PG = dict(host='localhost', port=5432, user='postgres', dbname='market_data')
HERE = os.path.dirname(__file__)
OBS = os.path.join(HERE, '..', '20260531_closing_auction_reversion', 'observations.csv')
OOS = pd.Timestamp('2025-08-05')


def fetch(sql, params=None):
    conn = psycopg2.connect(**PG); df = pd.read_sql(sql, conn, params=params); conn.close(); return df


print("="*76); print("クロージングオークション下落側Long — 実約定リアリズム精査"); print("="*76)

# 下落側イベント (jump<=-50bps)
obs = pd.read_csv(OBS); obs['date'] = pd.to_datetime(obs['date'])
obs = obs[obs['overnight'].abs() <= 0.10]
down = obs[obs['close_jump'] <= -0.005].copy()
codes = sorted(down['code'].astype(str).unique())
print(f"\n下落側イベント n={len(down)}, codes={len(codes)}")

ph = ','.join(['%s']*len(codes))
EXIT_TIMES = ['09:00:00','09:05:00','09:15:00','09:30:00','10:00:00','11:30:00','15:30:00']
bars = fetch(f"""
    SELECT code, ts, open, close FROM stocks_intraday
    WHERE code IN ({ph}) AND ts >= '2024-11-05'
      AND ts::time IN ('15:30:00','09:00:00','09:05:00','09:15:00','09:30:00','10:00:00','11:30:00')
    ORDER BY code, ts
""", tuple(codes))
bars['ts'] = pd.to_datetime(bars['ts'])
bars['date'] = bars['ts'].dt.normalize()
bars['t'] = bars['ts'].dt.strftime('%H:%M')
print(f"  bars {len(bars):,}")

# entry: auction日の 15:30 close
entry = bars[bars['t']=='15:30'][['code','date','close']].rename(columns={'close':'c30'})
# exit: 翌営業日の各時刻
piv = bars.pivot_table(index=['code','date'], columns='t', values=['open','close'], aggfunc='first')
piv.columns = [f'{a}_{b}' for a,b in piv.columns]; piv = piv.reset_index().sort_values(['code','date'])
# 翌営業日のバーを当該イベント行に紐付け: shift(-1)
piv['next_date'] = piv.groupby('code')['date'].shift(-1)
for col in ['open_09:00','close_09:05','close_09:15','close_09:30','close_10:00','close_11:30','close_15:30']:
    if col in piv.columns:
        piv['n_'+col] = piv.groupby('code')[col].shift(-1)

ev = down[['code','date','period']].copy()
ev['code'] = ev['code'].astype(str)
m = ev.merge(entry, on=['code','date'], how='left').merge(
    piv[['code','date','n_open_09:00','n_close_09:05','n_close_09:15','n_close_09:30',
         'n_close_10:00','n_close_11:30','n_close_15:30']], on=['code','date'], how='left')
m = m.dropna(subset=['c30','n_open_09:00'])
print(f"  結合後 n={len(m)}")

EXITS = [('翌09:00寄', 'n_open_09:00'), ('翌09:05', 'n_close_09:05'), ('翌09:15', 'n_close_09:15'),
         ('翌09:30', 'n_close_09:30'), ('翌10:00', 'n_close_10:00'), ('翌11:30', 'n_close_11:30'),
         ('翌15:30引', 'n_close_15:30')]

def shp(x, ann=np.sqrt(252)):
    x = pd.Series(x).dropna()
    return float(x.mean()/x.std()*ann) if len(x)>=10 and x.std()>0 else float('nan')

print("\n"+"="*76); print("A/B. exit時刻別 減衰カーブ (entry=引け板寄せ close30)"); print("="*76)
print(f"\n  {'exit時刻':<12} {'平均bps':>9} {'中央bps':>9} {'勝率%':>7} {'日次Sharpe':>11} {'net10bps_Sh':>12}")
print("  "+"-"*64)
for lab, col in EXITS:
    if col not in m.columns: continue
    r = (m[col]/m['c30']-1)*1e4
    r = r.dropna()
    # 日次平均系列でSharpe
    tmp = m.assign(r=(m[col]/m['c30']-1)*1e4).dropna(subset=['r'])
    daily = tmp.groupby(tmp['date'])['r'].mean()  # auction日基準でグループ
    print(f"  {lab:<12} {r.mean():>9.1f} {r.median():>9.1f} {(r>0).mean()*100:>7.1f} "
          f"{shp(daily):>11.2f} {shp(daily-10):>12.2f}")

print("\n"+"="*76); print("C. entry現実性: MOC不可で翌寄りentryにするとエッジ消えるか"); print("="*76)
# entry=翌09:00寄, exit=翌引け or 翌10:00 (引け買い前提が崩れた場合)
m['alt'] = (m['n_close_10:00']/m['n_open_09:00']-1)*1e4
print(f"  翌寄entry→翌10:00exit: 平均={m['alt'].mean():.1f}bps (引け板寄せ買いができない場合)")
print(f"  → 引け板寄せ(close30)entryが本質。MOC注文必須。")

print("\n"+"="*76); print("D. IS/OOS別 (entry=close30, exit=翌09:00寄)"); print("="*76)
m['ret_open'] = (m['n_open_09:00']/m['c30']-1)*1e4
for label, s in [('全期間', m), ('IS', m[m.period=='IS']), ('OOS', m[m.period=='OOS'])]:
    daily = s.groupby('date')['ret_open'].mean()
    print(f"  {label:<8} n={len(s):>5} 平均={s['ret_open'].mean():>6.1f}bps 勝率={(s['ret_open']>0).mean()*100:.1f}% "
          f"Sharpe={shp(daily):.2f} net10={shp(daily-10):.2f}")

m.to_csv(os.path.join(HERE,'exec_obs.csv'), index=False)
print(f"\n  保存: exec_obs.csv\n完了")
