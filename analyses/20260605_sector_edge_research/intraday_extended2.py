"""
イントラデイ戦略拡張2
- PM寄り方向フォロー
- ORBショート専用
- 前場方向 → 後場初動
- 早朝モメンタム (9:00-9:15方向)
- 時間帯別モメンタム窓
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
import psycopg2
import pandas as pd
import numpy as np
from scipy import stats
import csv

PG = dict(host="localhost", port=5432, user="postgres", dbname="market_data")

SYMBOLS = ['7203','6758','9984','7974','8306','9433','6861','7267','6902','6857',
           '4063','4661','8035','6594','7741','9022','3382','2914','4519','6098','7751','9766']

def load_intraday():
    conn = psycopg2.connect(**PG)
    tables = ['stocks_intraday_202405','stocks_intraday_202406','stocks_intraday_202407',
              'stocks_intraday_202408','stocks_intraday_202409','stocks_intraday_202410',
              'stocks_intraday_202411','stocks_intraday_202412','stocks_intraday_202501',
              'stocks_intraday_202502','stocks_intraday_202503','stocks_intraday_202504',
              'stocks_intraday_202505','stocks_intraday_202506']
    cur = conn.cursor()
    codes5 = []
    for c in SYMBOLS:
        cur.execute("SELECT code5 FROM symbol_master WHERE code4=%s LIMIT 1", (c,))
        r = cur.fetchone()
        if r: codes5.append(r[0])
    dfs = []
    for t in tables:
        try:
            cur.execute(f"SELECT code,ts,open,high,low,close,volume FROM public.{t} WHERE code=ANY(%s)", (codes5,))
            rows = cur.fetchall()
            if rows:
                dfs.append(pd.DataFrame(rows, columns=['code','ts','open','high','low','close','volume']))
        except Exception:
            conn.rollback()
    cur.execute("SELECT code5, code4 FROM symbol_master WHERE code4=ANY(%s)", (SYMBOLS,))
    mapping = {r[0]: r[1] for r in cur.fetchall()}
    conn.close()
    df = pd.concat(dfs, ignore_index=True)
    df['ts'] = pd.to_datetime(df['ts'])
    df['date'] = df['ts'].dt.date
    df['hm'] = df['ts'].dt.hour * 60 + df['ts'].dt.minute
    for c in ['open','high','low','close','volume']:
        df[c] = pd.to_numeric(df[c])
    df['code4'] = df['code'].map(mapping)
    df = df.dropna(subset=['code4'])
    print(f"  1分足:{len(df):,}行 期間:{df['date'].min()}~{df['date'].max()}")
    return df

def summarize(rets, name):
    if len(rets) < 20:
        return (name, 0, 0.0, 0.0, 0.0, 0.0, "データ不足")
    arr = np.array(rets)
    n = len(arr)
    mean = arr.mean()
    std = arr.std()
    t = stats.ttest_1samp(arr, 0).statistic
    sharpe = mean / std * np.sqrt(252) if std > 0 else 0
    win = (arr > 0).mean()
    if t >= 3.0: cls = "★★強↑"
    elif t >= 2.0: cls = "★弱↑"
    elif t <= -3.0: cls = "▼▼強↓"
    elif t <= -2.0: cls = "▼弱↓"
    else: cls = "中立"
    return (name, n, win, mean * 100, t, sharpe, cls)

def main():
    print("ロード中...")
    df = load_intraday()
    results = []

    # ORB5 ショート専用
    print("\n[ORB5ショート専用]")
    short_rets = []
    for (code, date), g in df.groupby(['code4', 'date']):
        g = g.sort_values('hm')
        bar5 = g[g['hm'] == 9 * 60]
        bar_close = g[g['hm'] == 14 * 60 + 50]
        if bar5.empty or bar_close.empty: continue
        b = bar5.iloc[0]
        orb_high, orb_low = b['high'], b['low']
        if orb_high <= orb_low: continue
        subsequent = g[g['hm'] > 9 * 60].sort_values('hm')
        for _, row in subsequent.iterrows():
            if row['high'] > orb_high: break  # ロング先行
            if row['low'] < orb_low:
                entry = orb_low
                exit_p = bar_close.iloc[0]['close']
                short_rets.append((entry - exit_p) / entry)
                break
    results.append(summarize(short_rets, "ORB5ショート専用→14:50"))

    # PM寄り方向フォロー
    print("\n[PM寄り方向フォロー]")
    pm30_rets, pm60_rets = [], []
    for (code, date), g in df.groupby(['code4', 'date']):
        g = g.sort_values('hm')
        pm_start = g[g['hm'] == 12 * 60 + 30]
        pm30 = g[g['hm'] == 13 * 60]
        pm60 = g[g['hm'] == 13 * 60 + 30]
        if pm_start.empty: continue
        pm_open = pm_start.iloc[0]['open']
        pm_close1 = pm_start.iloc[0]['close']
        if pm_open <= 0: continue
        pm_dir = 1 if pm_close1 > pm_open else -1
        if not pm30.empty:
            pm30_rets.append(pm_dir * (pm30.iloc[0]['close'] - pm_close1) / pm_close1)
        if not pm60.empty:
            pm60_rets.append(pm_dir * (pm60.iloc[0]['close'] - pm_close1) / pm_close1)
    results.append(summarize(pm30_rets, "後場寄り方向→30分"))
    results.append(summarize(pm60_rets, "後場寄り方向→60分"))

    # 前場方向→後場60分
    print("\n[前場方向→後場]")
    am_pm_rets = []
    for (code, date), g in df.groupby(['code4', 'date']):
        g = g.sort_values('hm')
        open_b = g[g['hm'] == 9 * 60]
        am_end = g[g['hm'] == 11 * 60 + 25]
        pm_start = g[g['hm'] == 12 * 60 + 30]
        pm_exit = g[g['hm'] == 13 * 60 + 30]
        if any(x.empty for x in [open_b, am_end, pm_start, pm_exit]): continue
        day_open = open_b.iloc[0]['open']
        am_ret = (am_end.iloc[0]['close'] - day_open) / day_open
        am_dir = 1 if am_ret > 0 else -1
        pm_entry = pm_start.iloc[0]['open']
        ret = am_dir * (pm_exit.iloc[0]['close'] - pm_entry) / pm_entry
        am_pm_rets.append(ret)
    results.append(summarize(am_pm_rets, "前場方向継続→後場60分"))

    # 早朝モメンタム (9:00-9:15 → 継続)
    print("\n[早朝モメンタム]")
    early15_rets, early30_rets = [], []
    for (code, date), g in df.groupby(['code4', 'date']):
        g = g.sort_values('hm')
        open_b = g[g['hm'] == 9 * 60]
        b15 = g[g['hm'] == 9 * 60 + 15]
        b30 = g[g['hm'] == 9 * 60 + 30]
        b45 = g[g['hm'] == 9 * 60 + 45]
        if open_b.empty or b15.empty: continue
        p0 = open_b.iloc[0]['open']
        p15 = b15.iloc[0]['close']
        d = 1 if p15 > p0 else -1
        if not b30.empty:
            early15_rets.append(d * (b30.iloc[0]['close'] - p15) / p15)
        if not b45.empty:
            early30_rets.append(d * (b45.iloc[0]['close'] - p15) / p15)
    results.append(summarize(early15_rets, "早朝9:00-9:15→9:30継続"))
    results.append(summarize(early30_rets, "早朝9:00-9:15→9:45継続"))

    # 10:00ピボット逆張り
    print("\n[10:00ピボット逆張り]")
    pivot_rets = []
    for (code, date), g in df.groupby(['code4', 'date']):
        g = g.sort_values('hm')
        am = g[(g['hm'] >= 9 * 60) & (g['hm'] < 10 * 60)]
        b1000 = g[g['hm'] == 10 * 60]
        b1100 = g[g['hm'] == 11 * 60]
        if am.empty or b1000.empty or b1100.empty: continue
        am_high = am['high'].max()
        am_low = am['low'].min()
        rng = am_high - am_low
        if rng <= 0: continue
        p1000 = b1000.iloc[0]['close']
        p1100 = b1100.iloc[0]['close']
        pos = (p1000 - am_low) / rng
        if pos > 0.8:
            pivot_rets.append(-(p1100 - p1000) / p1000)
        elif pos < 0.2:
            pivot_rets.append((p1100 - p1000) / p1000)
    results.append(summarize(pivot_rets, "10:00高安値圏逆張り→11:00"))

    # 前場±1.5%後場逆張り
    print("\n[前場急騰→後場逆張り]")
    pm_rev30, pm_rev60 = [], []
    for (code, date), g in df.groupby(['code4', 'date']):
        g = g.sort_values('hm')
        open_b = g[g['hm'] == 9 * 60]
        am_end = g[g['hm'] == 11 * 60 + 25]
        pm_s = g[g['hm'] == 12 * 60 + 30]
        pm30 = g[g['hm'] == 13 * 60]
        pm60 = g[g['hm'] == 13 * 60 + 30]
        if any(x.empty for x in [open_b, am_end, pm_s]): continue
        am_ret = (am_end.iloc[0]['close'] - open_b.iloc[0]['open']) / open_b.iloc[0]['open']
        if abs(am_ret) < 0.015: continue
        pm_entry = pm_s.iloc[0]['open']
        rev = -1 if am_ret > 0 else 1
        if not pm30.empty:
            pm_rev30.append(rev * (pm30.iloc[0]['close'] - pm_entry) / pm_entry)
        if not pm60.empty:
            pm_rev60.append(rev * (pm60.iloc[0]['close'] - pm_entry) / pm_entry)
    results.append(summarize(pm_rev30, "前場±1.5%→後場逆張り30分"))
    results.append(summarize(pm_rev60, "前場±1.5%→後場逆張り60分"))

    # 14:30引け前モメンタム
    print("\n[14:30引け前モメンタム]")
    prclose_rets = []
    for (code, date), g in df.groupby(['code4', 'date']):
        g = g.sort_values('hm')
        b1430 = g[g['hm'] == 14 * 60 + 30]
        b1435 = g[g['hm'] == 14 * 60 + 35]
        b1450 = g[g['hm'] == 14 * 60 + 50]
        if any(x.empty for x in [b1430, b1435, b1450]): continue
        p1430 = b1430.iloc[0]['close']
        p1435 = b1435.iloc[0]['close']
        p1450 = b1450.iloc[0]['close']
        d = 1 if p1435 > p1430 else -1
        prclose_rets.append(d * (p1450 - p1435) / p1435)
    results.append(summarize(prclose_rets, "14:30→14:35方向→14:50継続"))

    # 市場同方向ORB5フィルター
    print("\n[市場同方向ORB5]")
    # まず各(code,date)のORB方向と入口を収集
    orb_info = {}  # (code,date) -> (dir, entry)
    for (code, date), g in df.groupby(['code4', 'date']):
        g = g.sort_values('hm')
        bar5 = g[g['hm'] == 9 * 60]
        if bar5.empty: continue
        b = bar5.iloc[0]
        oh, ol = b['high'], b['low']
        if oh <= ol: continue
        subsequent = g[g['hm'] > 9 * 60].sort_values('hm')
        for _, row in subsequent.iterrows():
            if row['high'] > oh:
                orb_info[(code, date)] = ('long', oh)
                break
            elif row['low'] < ol:
                orb_info[(code, date)] = ('short', ol)
                break

    # 日付ごとの方向集計
    date_dirs = {}
    for (code, date), (d, _) in orb_info.items():
        if date not in date_dirs: date_dirs[date] = []
        date_dirs[date].append(d)

    market_rets = []
    for (code, date), g in df.groupby(['code4', 'date']):
        if (code, date) not in orb_info: continue
        direction, entry = orb_info[(code, date)]
        dirs = date_dirs.get(date, [])
        if len(dirs) == 0: continue
        long_pct = dirs.count('long') / len(dirs)
        if direction == 'long' and long_pct < 0.7: continue
        if direction == 'short' and (1 - long_pct) < 0.7: continue
        g = g.sort_values('hm')
        bar_close = g[g['hm'] == 14 * 60 + 50]
        if bar_close.empty: continue
        exit_p = bar_close.iloc[0]['close']
        mul = 1 if direction == 'long' else -1
        market_rets.append(mul * (exit_p - entry) / entry)
    results.append(summarize(market_rets, "ORB5+市場70%同方向フィルター→14:50"))

    # 出力
    print("\n" + "=" * 85)
    print("  イントラデイ拡張戦略2 結果")
    print("=" * 85)
    print(f"  {'戦略':<52} {'N':>5}  {'勝率':>6}  {'期待値':>8}  {'t値':>7}  {'Sharpe':>7}  判定")
    print("-" * 85)
    positive = []
    for r in results:
        name, n, win, mean, t, sharpe, cls = r
        print(f"  {name:<52} {n:>5}  {win*100:>5.1f}%  {mean:>+8.3f}%  {t:>+7.2f}  {sharpe:>+7.2f}  {cls}")
        if t >= 2.0:
            positive.append(r)

    print(f"\n  ✅ プラスエッジ: {len(positive)}個")
    for r in positive:
        print(f"     {r[0]}: t={r[4]:+.2f} 勝率={r[2]*100:.1f}% 期待値={r[3]:+.3f}%")

    outpath = "/mnt/d/Root/ClaudeCode/01_Trading/japan-stocks/analyses/20260605_sector_edge_research/intraday_extended2.csv"
    with open(outpath, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['strategy', 'N', 'win_rate', 'mean_ret', 't_stat', 'sharpe', 'cls'])
        for r in results:
            w.writerow(r)
    print(f"\n保存: {outpath}  完了")

if __name__ == '__main__':
    main()
