"""
イントラ最終2個探し
I9: 10:30方向→12:00 (前場後半モメンタム)
I10: 14:00方向→14:50 (引け前モメンタム)
I11: 前場急騰+2%→後場逆張り(フィルタ強化)
I12: ORBワイドレンジ>2%
I13: ORB5+SOX当日方向一致
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
    return df.dropna(subset=['code4'])

def load_sox():
    conn = psycopg2.connect(**PG)
    cur = conn.cursor()
    cur.execute("SELECT trade_date, close FROM macro.daily_ohlcv WHERE symbol='.SOX' AND trade_date>='2024-01-01' ORDER BY trade_date")
    rows = cur.fetchall()
    conn.close()
    df = pd.DataFrame(rows, columns=['date','close'])
    df['date'] = pd.to_datetime(df['date']).dt.date
    df['close'] = pd.to_numeric(df['close'])
    df['sox_ret'] = df['close'].pct_change()
    sox_up = set(df[df['sox_ret'] > 0.01]['date'])
    sox_dn = set(df[df['sox_ret'] < -0.01]['date'])
    return sox_up, sox_dn

def summarize(rets, name):
    if len(rets) < 20:
        return (name, 0, 0.0, 0.0, 0.0, 0.0, "データ不足")
    arr = np.array(rets)
    n, mean, std = len(arr), arr.mean(), arr.std()
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
    print("イントラ最終")
    print("ロード中...")
    df = load_intraday()
    sox_up, sox_dn = load_sox()
    print(f"  1分足:{len(df):,}行")
    results = []

    # 10:30方向→12:00
    print("\n[前場後半モメンタム]")
    mid_rets = []
    for (code, date), g in df.groupby(['code4', 'date']):
        g = g.sort_values('hm')
        b1030 = g[g['hm'] == 10 * 60 + 30]
        b1025 = g[g['hm'] == 10 * 60 + 25]
        b1130 = g[g['hm'] == 11 * 60 + 30]
        if b1030.empty or b1025.empty or b1130.empty: continue
        dir_1030 = 1 if b1030.iloc[0]['close'] > b1025.iloc[0]['close'] else -1
        ret = dir_1030 * (b1130.iloc[0]['close'] - b1030.iloc[0]['close']) / b1030.iloc[0]['close']
        mid_rets.append(ret)
    results.append(summarize(mid_rets, "前場後半10:30方向→11:30"))

    # 14:00方向→14:50
    print("\n[引け前モメンタム]")
    prclose_rets, prclose30_rets = [], []
    for (code, date), g in df.groupby(['code4', 'date']):
        g = g.sort_values('hm')
        b1355 = g[g['hm'] == 13 * 60 + 55]
        b1400 = g[g['hm'] == 14 * 60]
        b1430 = g[g['hm'] == 14 * 60 + 30]
        b1450 = g[g['hm'] == 14 * 60 + 50]
        if b1400.empty or b1355.empty or b1450.empty: continue
        dir_1400 = 1 if b1400.iloc[0]['close'] > b1355.iloc[0]['close'] else -1
        prclose_rets.append(dir_1400 * (b1450.iloc[0]['close'] - b1400.iloc[0]['close']) / b1400.iloc[0]['close'])
        if not b1430.empty:
            prclose30_rets.append(dir_1400 * (b1430.iloc[0]['close'] - b1400.iloc[0]['close']) / b1400.iloc[0]['close'])
    results.append(summarize(prclose_rets, "14:00方向→14:50(引け前50分)"))
    results.append(summarize(prclose30_rets, "14:00方向→14:30(引け前30分)"))

    # 前場急騰+2%以上 → 後場逆張り (フィルタ強化)
    print("\n[前場+2%→後場逆張り]")
    pm_rev_rets = []
    for (code, date), g in df.groupby(['code4', 'date']):
        g = g.sort_values('hm')
        open_b = g[g['hm'] == 9 * 60]
        am_end = g[g['hm'] == 11 * 60 + 25]
        pm_s = g[g['hm'] == 12 * 60 + 30]
        pm30 = g[g['hm'] == 13 * 60]
        if any(x.empty for x in [open_b, am_end, pm_s, pm30]): continue
        am_ret = (am_end.iloc[0]['close'] - open_b.iloc[0]['open']) / open_b.iloc[0]['open']
        if abs(am_ret) < 0.02: continue  # ±2%以上
        pm_entry = pm_s.iloc[0]['open']
        rev = -1 if am_ret > 0 else 1
        pm_rev_rets.append(rev * (pm30.iloc[0]['close'] - pm_entry) / pm_entry)
    results.append(summarize(pm_rev_rets, "前場±2%→後場逆張り30分"))

    # ORBワイドレンジ>2%
    print("\n[ORBワイドレンジ>2%]")
    wide_rets = []
    for (code, date), g in df.groupby(['code4', 'date']):
        g = g.sort_values('hm')
        bar5 = g[g['hm'] == 9 * 60]
        bar1450 = g[g['hm'] == 14 * 60 + 50]
        if bar5.empty or bar1450.empty: continue
        b = bar5.iloc[0]
        oh, ol = b['high'], b['low']
        if oh <= ol: continue
        orb_pct = (oh - ol) / ol
        if orb_pct < 0.02: continue  # ORBレンジ2%以上
        subsequent = g[g['hm'] > 9 * 60].sort_values('hm')
        exit_p = bar1450.iloc[0]['close']
        for _, row in subsequent.iterrows():
            if row['high'] > oh:
                wide_rets.append((exit_p - oh) / oh)
                break
            elif row['low'] < ol:
                wide_rets.append((ol - exit_p) / ol)
                break
    results.append(summarize(wide_rets, "ORB5ワイドレンジ>2%→14:50"))

    # ORB5+SOX同方向
    print("\n[ORB5+SOX当日方向]")
    # SOXは米国夜の話で当日はわからないが、前日SOXを使う
    sox_orb_up, sox_orb_dn = [], []
    for (code, date), g in df.groupby(['code4', 'date']):
        g = g.sort_values('hm')
        bar5 = g[g['hm'] == 9 * 60]
        bar1450 = g[g['hm'] == 14 * 60 + 50]
        if bar5.empty or bar1450.empty: continue
        b = bar5.iloc[0]
        oh, ol = b['high'], b['low']
        if oh <= ol: continue
        subsequent = g[g['hm'] > 9 * 60].sort_values('hm')
        exit_p = bar1450.iloc[0]['close']
        for _, row in subsequent.iterrows():
            if row['high'] > oh:
                ret = (exit_p - oh) / oh
                if date in sox_up:
                    sox_orb_up.append(ret)  # SOX陽線日のORBロング
                break
            elif row['low'] < ol:
                ret = (ol - exit_p) / ol
                if date in sox_dn:
                    sox_orb_dn.append(ret)  # SOX陰線日のORBショート
                break
    results.append(summarize(sox_orb_up, "ORB5ロング+前日SOX>+1%→14:50"))
    results.append(summarize(sox_orb_dn, "ORB5ショート+前日SOX<-1%→14:50"))

    # 出力
    print("\n" + "=" * 85)
    print("  イントラ最終テスト 結果")
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

    outpath = "/mnt/d/Root/ClaudeCode/01_Trading/japan-stocks/analyses/20260605_sector_edge_research/intraday_final.csv"
    with open(outpath, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['strategy', 'N', 'win_rate', 'mean_ret', 't_stat', 'sharpe', 'cls'])
        for r in results:
            w.writerow(r)
    print(f"\n保存: {outpath}  完了")

if __name__ == '__main__':
    main()
