"""
第3ラウンド戦略検証
イントラ:
  I7: PM-ORB (後場12:30-12:34ブレイク→14:50)
  I8: ORB5+前日出来高増加フィルター
  I9: ORB5+前日近辺高値 (モメンタム銘柄)
  I10: ORB3+ORB5両方同方向 (ダブル確認)
デイリー:
  D9: ADR方向→翌日個別株
  D10: NK225 前日大陰線→翌日リバ
スイング:
  S9: VIX>25時 売られ過ぎ個別株 (RSI<35) 10日
  S10: ADR連続2日↑→翌週5日継続
  S11: 52週高値ブレイク10日後
  S12: 5日MOM+出来高確認
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
    return df

def load_daily():
    conn = psycopg2.connect(**PG)
    cur = conn.cursor()
    ph = ','.join(['%s'] * len(SYMBOLS))
    cur.execute(f"""
        SELECT sd.code, sd.date, sd.open, sd.high, sd.low, sd.close, sd.volume,
               sd.adj_open, sd.adj_close, sm.code4
        FROM public.stocks_daily sd
        JOIN public.symbol_master sm ON sm.code5 = sd.code
        WHERE sm.code4 IN ({ph})
          AND sd.date >= '2022-01-01'
        ORDER BY sd.code, sd.date
    """, SYMBOLS)
    rows = cur.fetchall()
    conn.close()
    df = pd.DataFrame(rows, columns=['code5','date','open','high','low','close','volume','adj_open','adj_close','code4'])
    df['date'] = pd.to_datetime(df['date'])
    for c in ['open','high','low','close','volume','adj_open','adj_close']:
        df[c] = pd.to_numeric(df[c])
    return df

def load_macro():
    conn = psycopg2.connect(**PG)
    cur = conn.cursor()
    cur.execute("""
        SELECT trade_date, symbol, open, high, low, close
        FROM macro.daily_ohlcv
        WHERE symbol IN ('.SOX', 'NKc1', 'VXc1', 'ADR_8035', 'ADR_6857', 'ADR_6758', 'ADR_9984',
                         'ADR_7203', 'ADR_6902', 'ADR_4063', 'ADR_7974', 'ADR_7741', 'ADR_6901')
          AND trade_date >= '2022-01-01'
        ORDER BY trade_date
    """)
    rows = cur.fetchall()
    conn.close()
    df = pd.DataFrame(rows, columns=['date','symbol','open','high','low','close'])
    df['date'] = pd.to_datetime(df['date'])
    for c in ['open','high','low','close']:
        df[c] = pd.to_numeric(df[c])
    return df

def summarize(rets, name, period_days=1):
    if len(rets) < 20:
        return (name, 0, 0.0, 0.0, 0.0, 0.0, "データ不足")
    arr = np.array(rets)
    n = len(arr)
    mean = arr.mean()
    std = arr.std()
    t = stats.ttest_1samp(arr, 0).statistic
    trade_days = 252 / max(period_days, 1)
    sharpe = mean / std * np.sqrt(trade_days) if std > 0 else 0
    win = (arr > 0).mean()
    if t >= 3.0: cls = "★★強↑"
    elif t >= 2.0: cls = "★弱↑"
    elif t <= -3.0: cls = "▼▼強↓"
    elif t <= -2.0: cls = "▼弱↓"
    else: cls = "中立"
    return (name, n, win, mean * 100, t, sharpe, cls)

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def main():
    print("第3ラウンド 戦略検証")
    print("ロード中...")
    intra = load_intraday()
    daily = load_daily()
    macro = load_macro()
    print(f"  イントラ:{len(intra):,}行, 日次:{len(daily):,}行, マクロ:{len(macro):,}行")

    results_intra = []
    results_daily = []
    results_swing = []

    # --- イントラ戦略 ---

    # I7: PM-ORB (12:30-12:34のレンジブレイク→14:50)
    print("\n[I7: PM-ORB]")
    pm_orb_rets = []
    for (code, date), g in intra.groupby(['code4', 'date']):
        g = g.sort_values('hm')
        pm_bar = g[g['hm'] == 12 * 60 + 30]
        bar1450 = g[g['hm'] == 14 * 60 + 50]
        if pm_bar.empty or bar1450.empty: continue
        b = pm_bar.iloc[0]
        pm_high, pm_low = b['high'], b['low']
        if pm_high <= pm_low: continue

        subsequent = g[g['hm'] > 12 * 60 + 30].sort_values('hm')
        subsequent = subsequent[subsequent['hm'] < 14 * 60 + 51]
        exit_price = bar1450.iloc[0]['close']

        for _, row in subsequent.iterrows():
            if row['high'] > pm_high:
                pm_orb_rets.append((exit_price - pm_high) / pm_high)
                break
            elif row['low'] < pm_low:
                pm_orb_rets.append((pm_low - exit_price) / pm_low)
                break

    results_intra.append(summarize(pm_orb_rets, "I7: PM-ORB(12:30-12:34→14:50)"))

    # I8: ORB5 + 前日出来高増加 (1.3倍以上)
    print("\n[I8: ORB5+前日出来高増加]")
    # まず日次の出来高比率を計算
    daily_vol_ratio = {}
    for sym, g in daily.groupby('code4'):
        g = g.sort_values('date').reset_index(drop=True)
        g['vol_ma10'] = g['volume'].rolling(10).mean().shift(1)
        g['vol_ratio'] = g['volume'] / g['vol_ma10']
        for _, row in g.iterrows():
            if pd.isna(row['vol_ratio']): continue
            daily_vol_ratio[(sym, pd.Timestamp(row['date']).date())] = row['vol_ratio']

    orb5_vol_rets = []
    for (code, date), g in intra.groupby(['code4', 'date']):
        from datetime import timedelta
        prev_date = date - timedelta(days=1)
        # find previous trading day
        for d in range(1, 8):
            pd_date = date - timedelta(days=d)
            if (code, pd_date) in daily_vol_ratio:
                prev_vol_ratio = daily_vol_ratio[(code, pd_date)]
                break
        else:
            continue
        if prev_vol_ratio < 1.3: continue  # 前日出来高が平均の1.3倍以上

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
                orb5_vol_rets.append((exit_p - oh) / oh)
                break
            elif row['low'] < ol:
                orb5_vol_rets.append((ol - exit_p) / ol)
                break

    results_intra.append(summarize(orb5_vol_rets, "I8: ORB5+前日出来高1.3倍増→14:50"))

    # I9: ORB5 + 前日上昇 (モメンタム銘柄のみ)
    print("\n[I9: ORB5+前日上昇]")
    prev_ret_map = {}
    for sym, g in daily.groupby('code4'):
        g = g.sort_values('date').reset_index(drop=True)
        for i in range(1, len(g)):
            d = pd.Timestamp(g.iloc[i]['date']).date()
            ret = (g.iloc[i]['adj_close'] - g.iloc[i-1]['adj_close']) / g.iloc[i-1]['adj_close']
            prev_ret_map[(sym, d)] = ret

    orb5_mom_long, orb5_mom_short = [], []
    for (code, date), g in intra.groupby(['code4', 'date']):
        from datetime import timedelta
        prev_ret = None
        for d in range(1, 8):
            pd_date = date - timedelta(days=d)
            if (code, pd_date) in prev_ret_map:
                prev_ret = prev_ret_map[(code, pd_date)]
                break
        if prev_ret is None: continue

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
                if prev_ret > 0.005:  # 前日+0.5%以上
                    orb5_mom_long.append(ret)
                break
            elif row['low'] < ol:
                ret = (ol - exit_p) / ol
                if prev_ret < -0.005:  # 前日-0.5%以下
                    orb5_mom_short.append(ret)
                break

    results_intra.append(summarize(orb5_mom_long, "I9a: ORB5ロング+前日上昇銘柄→14:50"))
    results_intra.append(summarize(orb5_mom_short, "I9b: ORB5ショート+前日下落銘柄→14:50"))

    # I10: ORB3とORB5が同方向に一致した場合
    print("\n[I10: ORB3+ORB5ダブル確認]")
    orb3_dir = {}  # (code,date) -> dir
    for (code, date), g in intra.groupby(['code4', 'date']):
        g = g.sort_values('hm')
        bar3 = g[g['hm'] == 9 * 60]
        if bar3.empty: continue
        b = bar3.iloc[0]
        oh, ol = b['high'], b['low']
        if oh <= ol: continue
        subsequent = g[(g['hm'] > 9 * 60) & (g['hm'] <= 9 * 60 + 3)].sort_values('hm')
        # 9:01-9:03で先にブレイクした方向
        for _, row in subsequent.iterrows():
            if row['high'] > oh:
                orb3_dir[(code, date)] = 'long'
                break
            elif row['low'] < ol:
                orb3_dir[(code, date)] = 'short'
                break

    double_rets = []
    for (code, date), g in intra.groupby(['code4', 'date']):
        if (code, date) not in orb3_dir: continue
        d3 = orb3_dir[(code, date)]

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
                if d3 == 'long':  # ORB3とORB5が両方ロング
                    double_rets.append((exit_p - oh) / oh)
                break
            elif row['low'] < ol:
                if d3 == 'short':  # ORB3とORB5が両方ショート
                    double_rets.append((ol - exit_p) / ol)
                break

    results_intra.append(summarize(double_rets, "I10: ORB3+ORB5両方同方向→14:50"))

    # --- 日次戦略 ---

    # D9: ADR方向→翌日個別株
    print("\n[D9: ADR方向→翌日]")
    # ADRの前日比
    adr_map = {}  # (symbol, date) -> ret
    for sym, g in macro.groupby('symbol'):
        if not sym.startswith('ADR_'): continue
        code4 = sym[4:]  # 'ADR_8035' -> '8035'
        g = g.sort_values('date').reset_index(drop=True)
        for i in range(1, len(g)):
            d = pd.Timestamp(g.iloc[i]['date']).date()
            if g.iloc[i-1]['close'] > 0:
                r = (g.iloc[i]['close'] - g.iloc[i-1]['close']) / g.iloc[i-1]['close']
                adr_map[(code4, d)] = r

    adr_up_rets, adr_dn_rets = [], []
    from datetime import timedelta
    for sym, g in daily.groupby('code4'):
        g = g.sort_values('date').reset_index(drop=True)
        for i in range(1, len(g)):
            today = pd.Timestamp(g.iloc[i]['date']).date()
            prev = pd.Timestamp(g.iloc[i-1]['date']).date()
            # ADRは前営業日のデータを使う
            adr_ret = adr_map.get((sym, prev))
            if adr_ret is None: continue
            ret = (g.iloc[i]['adj_close'] - g.iloc[i-1]['adj_close']) / g.iloc[i-1]['adj_close']
            if adr_ret > 0.01:
                adr_up_rets.append(ret)
            elif adr_ret < -0.01:
                adr_dn_rets.append(ret)

    results_daily.append(summarize(adr_up_rets, "D9a: ADR+1%以上→翌日ロング"))
    results_daily.append(summarize(adr_dn_rets, "D9b: ADR-1%以下→翌日逆張りロング"))

    # D10: NK225 前日大陰線→翌日リバ
    print("\n[D10: NK225前日大陰線翌日]")
    nk = macro[macro['symbol'] == 'NKc1'].copy().sort_values('date').reset_index(drop=True)
    nk['ret'] = (nk['close'] - nk['close'].shift(1)) / nk['close'].shift(1)
    nk_down_dates = set(nk[nk['ret'] < -0.015]['date'].dt.date)

    nk_rev_rets = []
    for sym, g in daily.groupby('code4'):
        g = g.sort_values('date').reset_index(drop=True)
        for i in range(1, len(g)):
            prev_d = pd.Timestamp(g.iloc[i-1]['date']).date()
            if prev_d not in nk_down_dates: continue
            ret = (g.iloc[i]['adj_close'] - g.iloc[i-1]['adj_close']) / g.iloc[i-1]['adj_close']
            nk_rev_rets.append(ret)
    results_daily.append(summarize(nk_rev_rets, "D10: NK225前日-1.5%→翌日リバウンド"))

    # D11: NK225 前日大陽線→翌日継続
    nk_up_dates = set(nk[nk['ret'] > 0.015]['date'].dt.date)
    nk_mom_rets = []
    for sym, g in daily.groupby('code4'):
        g = g.sort_values('date').reset_index(drop=True)
        for i in range(1, len(g)):
            prev_d = pd.Timestamp(g.iloc[i-1]['date']).date()
            if prev_d not in nk_up_dates: continue
            ret = (g.iloc[i]['adj_close'] - g.iloc[i-1]['adj_close']) / g.iloc[i-1]['adj_close']
            nk_mom_rets.append(ret)
    results_daily.append(summarize(nk_mom_rets, "D11: NK225前日+1.5%→翌日継続"))

    # --- スイング戦略 ---

    # S9: VIX>25時の売られ過ぎ (RSI<35) 10日
    print("\n[S9: VIX高時RSI売られ過ぎ]")
    vx = macro[macro['symbol'] == 'VXc1'].copy().sort_values('date')
    vx_high_dates = set(vx[vx['close'] > 25]['date'].dt.date)

    vx_rets = []
    for sym, g in daily.groupby('code4'):
        g = g.sort_values('date').reset_index(drop=True)
        g['rsi14'] = compute_rsi(g['adj_close'])
        for i in range(14, len(g) - 11):
            d = pd.Timestamp(g.iloc[i]['date']).date()
            if d not in vx_high_dates: continue
            if pd.isna(g.iloc[i]['rsi14']): continue
            if g.iloc[i]['rsi14'] >= 35: continue
            entry = g.iloc[i]['adj_close']
            exit_p = g.iloc[i+10]['adj_close']
            vx_rets.append((exit_p - entry) / entry)
    results_swing.append(summarize(vx_rets, "S9: VIX>25+RSI<35→10日後", period_days=10))

    # S10: ADR連続2日↑→翌週5日継続
    print("\n[S10: ADR2日連続↑→5日]")
    adr_consec = {}  # (code4, date) -> 連続上昇日数
    for sym, g in macro.groupby('symbol'):
        if not sym.startswith('ADR_'): continue
        code4 = sym[4:]
        g = g.sort_values('date').reset_index(drop=True)
        consec = 0
        for i in range(1, len(g)):
            d = pd.Timestamp(g.iloc[i]['date']).date()
            if g.iloc[i-1]['close'] > 0:
                if g.iloc[i]['close'] > g.iloc[i-1]['close']:
                    consec += 1
                else:
                    consec = 0
            adr_consec[(code4, d)] = consec

    adr2_rets = []
    for sym, g in daily.groupby('code4'):
        g = g.sort_values('date').reset_index(drop=True)
        for i in range(1, len(g) - 6):
            prev_d = pd.Timestamp(g.iloc[i]['date']).date()
            consec = adr_consec.get((sym, prev_d), 0)
            if consec < 2: continue
            entry = g.iloc[i]['adj_close']
            exit_p = g.iloc[i+5]['adj_close']
            adr2_rets.append((exit_p - entry) / entry)
    results_swing.append(summarize(adr2_rets, "S10: ADR2日連続↑→翌5日継続", period_days=5))

    # S11: 52週高値ブレイク後10日
    print("\n[S11: 52週高値→10日]")
    s11_rets = []
    for sym, g in daily.groupby('code4'):
        g = g.sort_values('date').reset_index(drop=True)
        g['high52w'] = g['high'].rolling(252, min_periods=100).max().shift(1)
        for i in range(len(g) - 11):
            if pd.isna(g.iloc[i]['high52w']): continue
            if g.iloc[i]['high'] <= g.iloc[i]['high52w']: continue
            entry = g.iloc[i]['adj_close']
            exit_p = g.iloc[i+10]['adj_close']
            s11_rets.append((exit_p - entry) / entry)
    results_swing.append(summarize(s11_rets, "S11: 52週高値ブレイク→10日後", period_days=10))

    # S12: 5日MOM+出来高確認 (前5日上昇かつ出来高増加)
    print("\n[S12: 5日MOM+出来高確認]")
    s12_rets = []
    for sym, g in daily.groupby('code4'):
        g = g.sort_values('date').reset_index(drop=True)
        g['vol_ma10'] = g['volume'].rolling(10).mean()
        for i in range(10, len(g) - 6):
            mom5 = (g.iloc[i]['adj_close'] - g.iloc[i-5]['adj_close']) / g.iloc[i-5]['adj_close']
            if mom5 < 0.02: continue  # 5日+2%以上
            if g.iloc[i-1]['vol_ma10'] <= 0: continue
            vol_ratio = g.iloc[i]['volume'] / g.iloc[i-1]['vol_ma10']
            if vol_ratio < 1.2: continue  # 出来高も平均の1.2倍以上
            entry = g.iloc[i]['adj_close']
            exit_p = g.iloc[i+5]['adj_close']
            s12_rets.append((exit_p - entry) / entry)
    results_swing.append(summarize(s12_rets, "S12: 5日MOM+2%+出来高1.2倍→5日", period_days=5))

    # --- 出力 ---
    all_results = results_intra + results_daily + results_swing

    print("\n" + "=" * 90)
    print("  第3ラウンド 戦略結果")
    print("=" * 90)
    print(f"  {'戦略':<56} {'N':>5}  {'勝率':>6}  {'期待値':>8}  {'t値':>7}  {'Sharpe':>7}  判定")
    print("-" * 90)
    positive = []
    for r in all_results:
        name, n, win, mean, t, sharpe, cls = r
        print(f"  {name:<56} {n:>5}  {win*100:>5.1f}%  {mean:>+8.3f}%  {t:>+7.2f}  {sharpe:>+7.2f}  {cls}")
        if t >= 2.0:
            positive.append(r)

    print(f"\n  ✅ プラスエッジ: {len(positive)}個")
    for r in positive:
        print(f"     {r[0]}: t={r[4]:+.2f} 勝率={r[2]*100:.1f}% 期待値={r[3]:+.3f}%")

    outpath = "/mnt/d/Root/ClaudeCode/01_Trading/japan-stocks/analyses/20260605_sector_edge_research/extended3.csv"
    with open(outpath, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['strategy', 'N', 'win_rate', 'mean_ret', 't_stat', 'sharpe', 'cls'])
        for r in all_results:
            w.writerow(r)
    print(f"\n保存: {outpath}  完了")

if __name__ == '__main__':
    main()
