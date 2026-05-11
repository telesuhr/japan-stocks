#!/usr/bin/env python3
"""
全7採用戦略 JQuantsベース継続性バックテスト (2026-05-11)

新DB (stocks_intraday/stocks_daily/index_daily) を使い、
strategies/ にある7戦略を実データで再検証する。

対象:
  1. topix_overnight       (TOPIX gap≥+0.3% → CORE5 ON Long)
  2. eneos_vwap_trend      (5020 9:30 VWAP±50bps)
  3. vwap_morning_meanrevert (TEL/ディスコ/レーザー 10-11:30 VWAP±275bps)
  4. orb_breakout_long     (三井金属30分OR / ディスコ60分OR)
  5. lasertec_ma25_support (6920 dd20≤-5% + MA25接触 + 上昇 + 10日CD)
  6. bank_absorption       (銀行22銘柄 出来高吸収 5日保有)
  7. pair_portfolio        (18ペア Z-score平均回帰)

検証期間:
  - イントラデイ系: 2024-05-09 〜 2026-05-10 (2年)
  - 日足系:        過去5年 (2021-05-09 〜 2026-05-08)

コスト: 2bps片道 (往復4bps) = 0.04%
"""
import warnings; warnings.filterwarnings('ignore')
import sys, csv
from datetime import date, datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import psycopg2

PG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}
COST_BPS = 2.0
INTRADAY_START = "2024-05-09"
DAILY_START = "2021-05-09"

# ──────────────────────────────────────────────
# 共通ユーティリティ
# ──────────────────────────────────────────────
def sharpe(arr):
    arr = np.asarray([x for x in arr if not np.isnan(x) and not np.isinf(x)], dtype=float)
    if len(arr) < 2 or arr.std() == 0:
        return np.nan
    return arr.mean() / arr.std() * np.sqrt(252)

def summarize(name, trades):
    arr = np.asarray([x for x in trades if not np.isnan(x) and not np.isinf(x)], dtype=float)
    n = len(arr)
    if n == 0:
        print(f"\n[{name}] N=0"); return dict(name=name, N=0)
    wr = (arr > 0).mean() * 100
    pos = arr[arr > 0].sum(); neg = abs(arr[arr <= 0].sum())
    pf = pos/neg if neg > 0 and pos > 0 else np.nan
    sh = sharpe(arr)
    print(f"  [{name}] N={n} WR={wr:.1f}% PF={pf:.2f} Sharpe={sh:.2f} 平均={arr.mean():+.1f}bps 合計={arr.sum():+.0f}bps")
    return dict(name=name, N=n, WR=wr, PF=pf, Sharpe=sh, mean_bps=arr.mean(), sum_bps=arr.sum())

def load_intraday(codes, start, end=None):
    end = end or '2027-01-01'
    conn = psycopg2.connect(**PG)
    placeholders = ','.join(['%s']*len(codes))
    df = pd.read_sql(
        f"SELECT code, ts, open, high, low, close, volume FROM stocks_intraday "
        f"WHERE code IN ({placeholders}) AND ts >= %s AND ts < %s ORDER BY code, ts",
        conn, params=tuple(codes)+(start, end))
    conn.close()
    df['ts'] = pd.to_datetime(df['ts'])
    df['dt'] = df['ts'].dt.date
    return df

def load_daily(codes, start=DAILY_START):
    conn = psycopg2.connect(**PG)
    placeholders = ','.join(['%s']*len(codes))
    df = pd.read_sql(
        f"SELECT code, date, open, high, low, close, volume, turnover_value, "
        f"       adj_open, adj_high, adj_low, adj_close, adj_volume "
        f"FROM stocks_daily WHERE code IN ({placeholders}) AND date >= %s "
        f"ORDER BY code, date",
        conn, params=tuple(codes)+(start,))
    conn.close()
    df['date'] = pd.to_datetime(df['date']).dt.date
    return df

# ──────────────────────────────────────────────
# 1. topix_overnight
# ──────────────────────────────────────────────
def bt_topix_overnight():
    print("\n" + "="*70 + "\n[1] topix_overnight: TOPIXギャップ ≥+0.3% → CORE5 ON Long\n" + "="*70)
    CORE5 = ["57110","65010","70110","50160","45020"]
    conn = psycopg2.connect(**PG)
    topix = pd.read_sql(
        "SELECT date, open, close FROM index_daily WHERE code='0000' AND date >= %s ORDER BY date",
        conn, params=(DAILY_START,))
    core = pd.read_sql(
        f"SELECT code, date, open, close FROM stocks_daily "
        f"WHERE code IN ({','.join(['%s']*5)}) AND date >= %s ORDER BY code, date",
        conn, params=tuple(CORE5)+(DAILY_START,))
    conn.close()
    topix['date'] = pd.to_datetime(topix['date']).dt.date
    core['date'] = pd.to_datetime(core['date']).dt.date
    topix['prev_close'] = topix['close'].shift(1)
    topix['gap_pct'] = (topix['open'] / topix['prev_close'] - 1) * 100

    trades = []
    sig_dates = []
    for _, r in topix.iterrows():
        dt = r['date']
        if pd.isna(r['gap_pct']) or r['gap_pct'] < 0.3:
            continue
        if dt.weekday() == 3 or dt.weekday() >= 5:
            continue
        # CORE5 ON: dt close → 翌営業日 open
        rets = []
        for sym in CORE5:
            sd = core[core['code']==sym].set_index('date')
            if dt not in sd.index: continue
            entry = float(sd.loc[dt, 'close'])
            future = sd[sd.index > dt]
            if future.empty: continue
            exit_open = float(future.iloc[0]['open'])
            if entry<=0 or exit_open<=0: continue
            rets.append((exit_open/entry - 1) * 10000 - COST_BPS*2)
        if rets:
            trades.append(np.mean(rets))
            sig_dates.append(dt)
    print(f"  シグナル発動: {len(sig_dates)} 件 ({sig_dates[0]} 〜 {sig_dates[-1] if sig_dates else 'N/A'})")
    return summarize("topix_overnight", trades)

# ──────────────────────────────────────────────
# 2. eneos_vwap_trend
# ──────────────────────────────────────────────
def bt_eneos_vwap():
    print("\n" + "="*70 + "\n[2] eneos_vwap_trend: ENEOS 9:30 VWAP±50bps\n" + "="*70)
    df = load_intraday(['50200'], INTRADAY_START)
    trades, longs, shorts = [], [], []
    for dt, g in df.groupby('dt'):
        if dt.weekday() >= 5: continue
        g = g.sort_values('ts')
        morning = g[g['ts'].dt.hour >= 9]
        if len(morning) < 5: continue
        vol = morning['volume'].fillna(0).clip(lower=1)
        vwap_s = (morning['close']*vol).cumsum() / vol.cumsum()
        bar = morning[(morning['ts'].dt.hour==9)&(morning['ts'].dt.minute==30)]
        if bar.empty: continue
        c = float(bar['close'].iloc[-1]); v = float(vwap_s.loc[bar.index[-1]])
        if v<=0: continue
        dev = (c/v-1)*10000
        if abs(dev) < 50: continue
        d = 1 if dev > 0 else -1
        hhmm = g['ts'].dt.hour*60 + g['ts'].dt.minute
        after = g[hhmm > 9*60+30]; ex = g[hhmm >= 15*60+20]
        if after.empty or ex.empty: continue
        ep = float(after['close'].iloc[0]); xp = float(ex['close'].iloc[-1])
        if ep<=0 or xp<=0: continue
        ret = d*(xp/ep-1)*10000 - COST_BPS*2
        trades.append(ret)
        (longs if d==1 else shorts).append(ret)
    print(f"  Long N={len(longs)} Sharpe={sharpe(longs):.2f} / Short N={len(shorts)} Sharpe={sharpe(shorts):.2f}")
    return summarize("eneos_vwap_trend", trades)

# ──────────────────────────────────────────────
# 3. vwap_morning_meanrevert
# ──────────────────────────────────────────────
def bt_vwap_meanrevert():
    print("\n" + "="*70 + "\n[3] vwap_morning_meanrevert: TEL/ディスコ/レーザー 10-11:30 VWAP±275bps\n" + "="*70)
    TARGETS = {'80350':'TEL','61460':'ディスコ','69200':'レーザー'}
    df_all = load_intraday(list(TARGETS), INTRADAY_START)
    trades = []
    per = {s: [] for s in TARGETS}
    for sym in TARGETS:
        df = df_all[df_all['code']==sym].copy()
        for dt, g in df.groupby('dt'):
            if dt.weekday() >= 5: continue
            g = g.sort_values('ts').copy()
            h, m = g['ts'].dt.hour, g['ts'].dt.minute
            session = g[((h==9)|(h==10)|((h==11)&(m<=30))|((h==12)&(m>=30))|(h==13)|(h==14)|((h==15)&(m<=30)))]
            if len(session) < 10: continue
            vol = session['volume'].fillna(0).clip(lower=1)
            session = session.copy()
            session['vwap'] = ((session['close']*vol).cumsum() / vol.cumsum()).values
            session['dev'] = (session['close']/session['vwap']-1)*10000
            session['hhmm'] = session['ts'].dt.hour*60 + session['ts'].dt.minute
            window = session[(session['hhmm']>=10*60)&(session['hhmm']<=11*60+30)]
            trig = window[window['dev'].abs()>=275]
            if trig.empty: continue
            row = trig.iloc[0]
            dev = float(row['dev'])
            d = -1 if dev>0 else 1
            ep = float(row['close'])
            if ep<=0: continue
            stop = ep*(1+400/10000) if d==-1 else ep*(1-400/10000)
            after = session[session['ts']>=row['ts']]
            xp = None
            for _, b in after.iterrows():
                if d==-1 and b['high']>=stop: xp=stop; break
                if d==1 and b['low']<=stop: xp=stop; break
            if xp is None:
                ex = session[session['hhmm']>=15*60+20]
                xp = float(ex['close'].iloc[-1]) if not ex.empty else float(session['close'].iloc[-1])
            ret = d*(xp/ep-1)*10000 - COST_BPS*2
            trades.append(ret); per[sym].append(ret)
    for sym, name in TARGETS.items():
        arr = np.asarray(per[sym])
        if len(arr)>0:
            print(f"  {sym} {name}: N={len(arr)} WR={(arr>0).mean()*100:.1f}% Sharpe={sharpe(arr):.2f} 平均={arr.mean():+.1f}bps")
    return summarize("vwap_morning_meanrevert", trades)

# ──────────────────────────────────────────────
# 4. orb_breakout_long
# ──────────────────────────────────────────────
def bt_orb_breakout():
    print("\n" + "="*70 + "\n[4] orb_breakout_long: 三井金属30分OR / ディスコ60分OR\n" + "="*70)
    TARGETS = [('57060','三井金属',30),('61460','ディスコ',60)]
    df_all = load_intraday([s for s,_,_ in TARGETS], INTRADAY_START)
    trades = []
    per = {s:[] for s,_,_ in TARGETS}
    for sym, name, or_min in TARGETS:
        df = df_all[df_all['code']==sym].copy()
        for dt, g in df.groupby('dt'):
            if dt.weekday() >= 5: continue
            g = g.sort_values('ts').copy()
            hhmm = g['ts'].dt.hour*60 + g['ts'].dt.minute
            or_bars = g[hhmm < 9*60+or_min]
            if len(or_bars) < max(3, or_min//2): continue
            or_high = or_bars['high'].max(); or_low = or_bars['low'].min()
            # ディスコは vwap_meanrevert発動日除外
            if sym == '61460':
                session = g[g['ts'].dt.hour>=9].copy()
                if not session.empty:
                    vol = session['volume'].fillna(0).clip(lower=1)
                    vwap = (session['close']*vol).cumsum() / vol.cumsum()
                    dev = (session['close']/vwap-1)*10000
                    h2 = session['ts'].dt.hour*60 + session['ts'].dt.minute
                    win = (h2>=10*60)&(h2<=11*60+30)
                    if (dev[win].abs()>=275).any(): continue
            post = g[hhmm >= 9*60+or_min]
            entry_px = None; entry_time = None
            for _, b in post.iterrows():
                if b['high'] > or_high:
                    entry_px = or_high; entry_time = b['ts']; break
            if entry_px is None or entry_px<=0: continue
            after = post[post['ts']>=entry_time]
            xp = None
            for _, b in after.iterrows():
                if b['low'] <= or_low: xp = or_low; break
            if xp is None:
                ah = after['ts'].dt.hour*60 + after['ts'].dt.minute
                ex = after[ah >= 15*60+20]
                xp = float(ex['close'].iloc[0]) if not ex.empty else float(after['close'].iloc[-1])
            ret = (xp/entry_px-1)*10000 - COST_BPS*2
            trades.append(ret); per[sym].append(ret)
    for sym, name, _ in TARGETS:
        arr = np.asarray(per[sym])
        if len(arr)>0:
            print(f"  {sym} {name}: N={len(arr)} WR={(arr>0).mean()*100:.1f}% Sharpe={sharpe(arr):.2f} 平均={arr.mean():+.1f}bps")
    return summarize("orb_breakout_long", trades)

# ──────────────────────────────────────────────
# 5. lasertec_ma25_support
# ──────────────────────────────────────────────
def bt_lasertec_ma25():
    print("\n" + "="*70 + "\n[5] lasertec_ma25_support: 6920 MA25接触 + dd20≤-5% + MA25↑ (10日CD)\n" + "="*70)
    df = load_daily(['69200'], start=DAILY_START)
    df = df.set_index('date').sort_index()
    df = df.astype({c: float for c in ['open','high','low','close']})
    df['ma25'] = df['close'].rolling(25).mean()
    df['ma25_5d_ago'] = df['ma25'].shift(5)
    df['hh20'] = df['close'].rolling(20).max()
    df['dd20'] = (df['close']/df['hh20']-1)*100
    lo, hi, ma = df['low'], df['high'], df['ma25']
    df['touched'] = (lo<=ma*1.01)&(hi>=ma*0.99)
    df['signal'] = df['touched'] & (df['dd20']<=-5.0) & (df['ma25']>df['ma25_5d_ago']) & df['ma25'].notna()

    HOLD = 10; CD = 10
    trades = []; sig_dates = []; last_entry = None
    for idx in range(len(df)):
        if not df.iloc[idx]['signal']: continue
        sig_dt = df.index[idx]
        if last_entry is not None and (sig_dt - last_entry).days < CD: continue
        if idx+1 >= len(df): continue
        entry = float(df.iloc[idx+1]['open'])
        if entry<=0 or np.isnan(entry): continue
        stop_level = entry * 0.90
        last_entry = df.index[idx+1]
        sig_dates.append(sig_dt)
        fut = df.iloc[idx+1:idx+2+HOLD]
        xp = None
        for _, r in fut.iloc[1:].iterrows():
            if r['low'] <= stop_level: xp=stop_level; break
        if xp is None:
            xp = float(fut.iloc[HOLD]['close']) if len(fut) > HOLD else float(fut.iloc[-1]['close'])
        ret = (xp/entry-1)*10000 - COST_BPS*2
        trades.append(ret)
    print(f"  シグナル発動 (CD適用後): {len(sig_dates)} 件 [{sig_dates[0] if sig_dates else 'N/A'} 〜 {sig_dates[-1] if sig_dates else 'N/A'}]")
    return summarize("lasertec_ma25_support", trades)

# ──────────────────────────────────────────────
# 6. bank_absorption
# ──────────────────────────────────────────────
def bt_bank_absorption():
    print("\n" + "="*70 + "\n[6] bank_absorption: 銀行22銘柄 出来高≥1.5×平均 + 陰線 + 売買代金≥10億\n" + "="*70)
    wl_path = Path("/Users/Yusuke/claude-code/japan-stocks/.claude/worktrees/stupefied-visvesvaraya-f58465/strategies/bank_absorption/whitelist.csv")
    if not wl_path.exists():
        print(f"  whitelist not found: {wl_path}")
        return dict(name="bank_absorption", N=0)
    codes = []
    with wl_path.open() as f:
        for r in csv.DictReader(f):
            c = r.get('code') or r.get('code5') or list(r.values())[0]
            c = str(c).strip()
            if c and c.isdigit():
                if len(c) == 4: c = c + '0'
                codes.append(c)
    print(f"  whitelist 銘柄数: {len(codes)}")
    df = load_daily(codes, start=DAILY_START)
    df = df.sort_values(['code','date']).reset_index(drop=True)
    trades = []
    for sym in codes:
        sd = df[df['code']==sym].copy().reset_index(drop=True)
        if len(sd) < 30: continue
        sd['vol_ma20'] = sd['volume'].rolling(20).mean()
        sd['adj_ret'] = (sd['adj_close']/sd['adj_open']-1)
        for i in range(20, len(sd)-6):
            r = sd.iloc[i]
            if pd.isna(r['vol_ma20']) or r['vol_ma20']<=0: continue
            if r['volume'] < 1.5*r['vol_ma20']: continue
            if r['adj_ret'] >= 0: continue  # 陰線でないと不発
            if r['turnover_value'] < 10e8: continue
            # Day N+1 寄成 Long → Day N+5 引成
            entry_row = sd.iloc[i+1]
            exit_row = sd.iloc[i+5] if i+5 < len(sd) else sd.iloc[-1]
            ep = float(entry_row['adj_open']); xp = float(exit_row['adj_close'])
            if ep<=0 or xp<=0: continue
            ret = (xp/ep-1)*10000 - COST_BPS*2
            trades.append(ret)
    return summarize("bank_absorption", trades)

# ──────────────────────────────────────────────
# 7. pair_portfolio
# ──────────────────────────────────────────────
def bt_pair_portfolio():
    print("\n" + "="*70 + "\n[7] pair_portfolio: 18ペア Z-score平均回帰 EW\n" + "="*70)
    PAIRS = [
        ("70110","70130","重工MHI-IHI",2.0,20,30,0.5,4.0),
        ("83060","84110","銀MUFG-みずほ",2.0,60,30,0.5,4.0),
        ("83060","83160","銀MUFG-SMFG",2.0,60,20,0.5,4.0),
        ("61460","63230","半ディスコ-ローツェ",2.0,40,10,0.5,4.0),
        ("94320","94330","通NTT-KDDI",2.5,60,10,0.6,4.0),
        ("80020","80310","商丸紅-三井物産",2.5,60,10,0.6,4.0),
        ("57110","57130","非鉄マテ-住友金鉱",2.5,20,30,0.6,4.0),
        ("57110","57060","非鉄マテ-三井金",2.5,20,30,0.6,4.0),
        ("72030","72670","車トヨタ-ホンダ",1.5,20,30,0.4,4.0),
        ("65010","65030","電機日立-三菱電",2.5,20,10,0.6,4.0),
        ("72700","72690","車スバル-スズキ",2.5,40,10,0.6,4.0),
        ("90200","90220","鉄JR東-JR東海",2.5,40,30,0.6,4.0),
        ("45030","45780","薬アステラス-大塚",2.5,40,10,0.6,4.0),
        ("67580","67020","電機ソニー-富士通",2.0,40,10,0.5,4.0),
        ("58020","58010","電線住電-古河",1.5,40,20,0.4,4.0),
        ("69200","68570","半レーザー-アドバン",2.5,40,10,0.6,4.0),
        ("80350","69200","半TEL-レーザー",2.5,40,10,0.6,4.0),
        ("16050","50200","エINPEX-ENEOS",2.5,20,30,0.6,4.0),
    ]
    BETA_WINDOW = 60; COST_PAIR_BPS = 8.0
    all_codes = sorted({c for p in PAIRS for c in (p[0], p[1])})
    df_all = load_daily(all_codes, start=DAILY_START)
    df_all = df_all[['code','date','adj_close']].copy()
    df_all['date'] = pd.to_datetime(df_all['date'])
    px = df_all.pivot(index='date', columns='code', values='adj_close').astype(float)

    pair_results = []
    all_trades = []
    for p1, p2, label, ez, zw, mh, xz, sz in PAIRS:
        if p1 not in px.columns or p2 not in px.columns:
            print(f"  {label}: 銘柄不足 skip"); continue
        d = px[[p1, p2]].dropna()
        if len(d) < BETA_WINDOW + zw + 10:
            print(f"  {label}: データ不足 N={len(d)}"); continue
        lp = np.log(d)
        # 60日ローリングβ (OLS)
        import statsmodels.api as sm
        betas, spreads, zs = [], [], []
        for i in range(BETA_WINDOW, len(lp)):
            window = lp.iloc[i-BETA_WINDOW:i]
            X = sm.add_constant(window[p2].values)
            try:
                res = sm.OLS(window[p1].values, X).fit()
                beta = res.params[1]
            except Exception:
                beta = np.nan
            betas.append(beta)
            spread = lp.iloc[i][p1] - beta * lp.iloc[i][p2] if not np.isnan(beta) else np.nan
            spreads.append(spread)
        spread_s = pd.Series(spreads, index=lp.index[BETA_WINDOW:])
        z_s = (spread_s - spread_s.rolling(zw).mean()) / spread_s.rolling(zw).std()
        # トレードシミュレーション
        trades = []; pos = None
        for i in range(zw, len(z_s)):
            z = z_s.iloc[i]
            if pd.isna(z): continue
            dt = z_s.index[i]
            # exit
            if pos is not None:
                hold = (dt - pos['entry_dt']).days
                # MR
                if abs(z) < xz:
                    p1_now = px.loc[dt, p1]; p2_now = px.loc[dt, p2]
                    if not pd.isna(p1_now) and not pd.isna(p2_now):
                        ret_p1 = (p1_now/pos['p1_px']-1)*10000
                        ret_p2 = (p2_now/pos['p2_px']-1)*10000
                        ret = pos['dir']*(ret_p1 - pos['beta']*ret_p2) - COST_PAIR_BPS
                        trades.append(ret); pos = None
                        continue
                # Stop
                if abs(z) > sz:
                    p1_now = px.loc[dt, p1]; p2_now = px.loc[dt, p2]
                    ret_p1 = (p1_now/pos['p1_px']-1)*10000
                    ret_p2 = (p2_now/pos['p2_px']-1)*10000
                    ret = pos['dir']*(ret_p1 - pos['beta']*ret_p2) - COST_PAIR_BPS
                    trades.append(ret); pos = None
                    continue
                # Time
                if hold >= mh:
                    p1_now = px.loc[dt, p1]; p2_now = px.loc[dt, p2]
                    ret_p1 = (p1_now/pos['p1_px']-1)*10000
                    ret_p2 = (p2_now/pos['p2_px']-1)*10000
                    ret = pos['dir']*(ret_p1 - pos['beta']*ret_p2) - COST_PAIR_BPS
                    trades.append(ret); pos = None
                    continue
            # entry
            if pos is None and abs(z) >= ez:
                d_dir = -1 if z > 0 else 1
                pos = dict(entry_dt=dt, dir=d_dir, p1_px=px.loc[dt,p1], p2_px=px.loc[dt,p2], beta=betas[i])
        arr = np.asarray(trades)
        sh = sharpe(arr)
        wr = (arr>0).mean()*100 if len(arr)>0 else 0
        pair_results.append((label, len(arr), wr, sh, arr.mean() if len(arr)>0 else np.nan))
        all_trades.extend(trades)
        print(f"  {label}: N={len(arr)} WR={wr:.1f}% Sharpe={sh:.2f} 平均={arr.mean() if len(arr)>0 else 0:+.1f}bps")
    return summarize("pair_portfolio (sum)", all_trades)

# ──────────────────────────────────────────────
# main
# ──────────────────────────────────────────────
def main():
    print("="*70)
    print(f"全7戦略 JQuants継続性検証 (実行日 {date.today()})")
    print("検証期間: イントラ 2024-05-09〜2026-05-10 (2年) / 日足 2021-05-09〜 (5年)")
    print("コスト: 2bps片道 (往復4bps)")
    print("="*70)

    results = []
    results.append(bt_topix_overnight())
    results.append(bt_eneos_vwap())
    results.append(bt_vwap_meanrevert())
    results.append(bt_orb_breakout())
    results.append(bt_lasertec_ma25())
    results.append(bt_bank_absorption())
    results.append(bt_pair_portfolio())

    REF = {'topix_overnight':6.27,'eneos_vwap_trend':3.81,'vwap_morning_meanrevert':6.76,
           'orb_breakout_long':2.31,'lasertec_ma25_support':7.57,
           'bank_absorption':1.84,'pair_portfolio (sum)':1.37}

    print("\n\n" + "="*70)
    print("★ JQuants継続性検証 サマリー")
    print("="*70)
    print(f"{'戦略':<32} {'N':>5} {'WR%':>6} {'PF':>6} {'Sharpe':>8} {'平均bps':>9} {'参照':>7} {'判定':>6}")
    print("-"*70)
    for r in results:
        n = r.get('N',0)
        ref = REF.get(r['name'], None)
        if isinstance(n,int) and n>0:
            wr = f"{r.get('WR',0):.1f}%"
            pf = r.get('PF',float('nan')); pf_s = f"{pf:.2f}" if not np.isnan(pf) else " - "
            sh = r.get('Sharpe',float('nan')); sh_s = f"{sh:.2f}" if not np.isnan(sh) else " - "
            mb = f"{r.get('mean_bps',0):+.1f}"
            ref_s = f"{ref:.2f}" if ref else " - "
            if ref and not np.isnan(sh):
                if sh >= ref*0.7: v = "✅継続"
                elif sh >= 1.0:   v = "⚠️低下"
                else:             v = "❌劣化"
            else:
                v = " - "
        else:
            wr=pf_s=sh_s=mb=ref_s=" - "; v=" - "
        print(f"  {r['name']:<30} {str(n):>5} {wr:>6} {pf_s:>6} {sh_s:>8} {mb:>9} {ref_s:>7} {v:>6}")
    print("="*70)

    # CSV保存
    df = pd.DataFrame(results)
    df.to_csv('results.csv', index=False)
    print("\n→ results.csv 保存完了")

if __name__ == "__main__":
    main()
