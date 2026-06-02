"""
バッチ4: 日中構造 3本 (#18-20)  ※intradayは2024-05〜のみ存在 → 実質OOS期間のみ

#18 ORB15分→引け      寄り後15分(9:00-9:15)の高安レンジをブレイクした方向に引けまで保有
#19 前場→後場         前場リターン(寄→11:30)が後場リターン(12:30→引け)を予測するか(継続/反転)
#20 VWAP終日乖離→回帰  正午(12:30)時点のVWAP乖離が引けへの反転を予測するか

ユニバース = 主戦22(半導体14+非鉄8)。1分足 stocks_intraday。
日次=非重複 → √252。コスト往復10bps。期間 2024-05-10〜2026-06-02。
"""
from __future__ import annotations
import os, sys, warnings
import psycopg2, pandas as pd, numpy as np
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

DB = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost/market_data")
COST = 10 / 1e4
SEMI = ['80350','68570','69200','61460','77350','67230','69630','65260','40620','34360',
        '40630','77410','99840','285A0','65250','69540','67620','69760','69810']
NONFE = ['58030','50160','58010','58020','57130','57060','57110','57140']
UNI = sorted(set(SEMI + NONFE))


def fetch(sql):
    c = psycopg2.connect(DB); df = pd.read_sql(sql, c); c.close(); return df


def sharpe(r):
    r = pd.Series(r).dropna()
    return float(r.mean()/r.std()*np.sqrt(252)) if len(r) >= 20 and r.std() > 0 else float('nan')


print("="*76); print("バッチ4: 日中構造 3本 (#18-20)  [intraday=2024-05〜のみ→OOS実質]"); print("="*76)

orb_rows, am_pm_rows, vwap_rows = [], [], []
for code in UNI:
    df = fetch(f"""SELECT ts, open::float o, high::float h, low::float l, close::float c, volume::float v
        FROM stocks_intraday WHERE code='{code}' AND ts>='2024-05-01' ORDER BY ts""")
    if df.empty:
        continue
    df['ts'] = pd.to_datetime(df['ts'])
    df['date'] = df['ts'].dt.normalize()
    df['t'] = df['ts'].dt.time
    for d, g in df.groupby('date'):
        g = g.sort_values('ts')
        if len(g) < 60:
            continue
        op = g.iloc[0]['o']; cl = g.iloc[-1]['c']
        if op <= 0 or cl <= 0:
            continue
        # --- #18 ORB 15分 ---
        oran = g[g['t'] <= pd.Timestamp('09:15').time()]
        rest = g[g['t'] > pd.Timestamp('09:15').time()]
        if len(oran) >= 5 and len(rest) >= 10:
            orh = oran['h'].max(); orl = oran['l'].min()
            up_idx = rest.index[rest['h'] >= orh]
            dn_idx = rest.index[rest['l'] <= orl]
            up_t = rest.loc[up_idx[0], 'ts'] if len(up_idx) else None
            dn_t = rest.loc[dn_idx[0], 'ts'] if len(dn_idx) else None
            ret = np.nan
            if up_t is not None and (dn_t is None or up_t <= dn_t):
                ret = cl/orh - 1 - COST           # 上抜けLong → 引け
            elif dn_t is not None:
                ret = -(cl/orl - 1) - COST         # 下抜けShort → 引け
            if not np.isnan(ret):
                orb_rows.append({'date': d, 'code': code, 'ret': ret})
        # --- #19 前場→後場 ---
        am = g[g['t'] <= pd.Timestamp('11:30').time()]
        pm = g[g['t'] >= pd.Timestamp('12:30').time()]
        if len(am) >= 10 and len(pm) >= 10:
            am_ret = am.iloc[-1]['c']/am.iloc[0]['o'] - 1
            pm_open = pm.iloc[0]['o']
            pm_ret = cl/pm_open - 1
            am_pm_rows.append({'date': d, 'code': code, 'am': am_ret, 'pm': pm_ret})
        # --- #20 VWAP乖離(12:30)→引け回帰 ---
        g2 = g.copy()
        g2['pv'] = (g2['c']*g2['v'])
        upto = g2[g2['t'] <= pd.Timestamp('12:30').time()]
        if len(upto) >= 30 and upto['v'].sum() > 0:
            vwap = upto['pv'].sum()/upto['v'].sum()
            px_mid = upto.iloc[-1]['c']
            if vwap > 0 and px_mid > 0:
                dev = px_mid/vwap - 1
                ret_close = cl/px_mid - 1
                vwap_rows.append({'date': d, 'code': code, 'dev': dev, 'ret_close': ret_close})

orb = pd.DataFrame(orb_rows); ap = pd.DataFrame(am_pm_rows); vw = pd.DataFrame(vwap_rows)
print(f"\nサンプル: ORB={len(orb)} 前後場={len(ap)} VWAP={len(vw)} (銘柄×日)")

# ---- #18 ORB ----
print("\n--- #18 ORB15分→引け (ブレイク方向に引けまで, 10bps) ---")
dly = orb.groupby('date')['ret'].mean()   # 各日ユニバース等加重
print(f"  日次平均ret={dly.mean()*1e4:+.1f}bps Sharpe={sharpe(dly):+.2f} 勝率={ (orb['ret']>0).mean()*100:.1f}% "
      f"(全trade平均={orb['ret'].mean()*1e4:+.1f}bps)")

# ---- #19 前場→後場 ----
print("\n--- #19 前場→後場 (継続/反転, 後場12:30→引け) ---")
rho, p = spearmanr(ap['am'], ap['pm'])
print(f"  ρ(前場,後場)={rho:+.3f}(p={p:.3f}) 平均|前場|={ap['am'].abs().mean()*1e4:.0f}bps")
for c_bps in (10, 20, 30):
    cst = c_bps/1e4
    rev = -np.sign(ap['am'])*ap['pm'] - cst       # 前場と逆方向に後場保有(反転)
    rev_dly = ap.assign(r=rev).groupby('date')['r'].mean()
    h24 = rev_dly[rev_dly.index < '2025-01-01']; h25 = rev_dly[rev_dly.index >= '2025-01-01']
    print(f"    反転 {c_bps}bps: Sharpe全={sharpe(rev_dly):+.2f} 24={sharpe(h24):+.2f} 25-={sharpe(h25):+.2f} "
          f"trade平均={rev.mean()*1e4:+.1f}bps")

# ---- #20 VWAP乖離→引け回帰 ----
print("\n--- #20 VWAP(12:30)乖離→引け回帰 ---")
rho2, p2 = spearmanr(vw['dev'], vw['ret_close'])
rev = -np.sign(vw['dev'])*vw['ret_close'] - COST  # dev>0=割高→Short
rev_dly2 = vw.assign(r=rev).groupby('date')['r'].mean()
print(f"  ρ(乖離,引けret)={rho2:+.3f}(p={p2:.3f})  反転Sharpe(10bps)={sharpe(rev_dly2):+.2f} "
      f"(負ρ=回帰=反転で取れる)")

print("\n完了")
