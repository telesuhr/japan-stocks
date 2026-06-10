"""
日次・スイング戦略拡張
Daily:
  D4: NK225 RSI<30 バウンス
  D5: 52週高値ブレイク翌日
  D6: ADR↑→半導体翌日
  D7: 前日大幅安リバウンド (個別-2%以上)
  D8: 月末リバランス (月末2日前→月初3日)
  D9: 曜日効果 (月曜は弱い?)
  D10: ギャップアップ後の当日続伸
Swing:
  S4: RSI(14)<30 5日後
  S5: 52週高値ブレイク後 20日
  S6: 5/20日移動平均ゴールデンクロス
  S7: 月次最低リターンセクター反転
  S8: 出来高急増翌日から5日
  S9: 前月-10%以上銘柄の翌月反発
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

def load_daily(symbols):
    conn = psycopg2.connect(**PG)
    cur = conn.cursor()
    ph = ','.join(['%s'] * len(symbols))
    cur.execute(f"""
        SELECT sd.code, sd.date, sd.open, sd.high, sd.low, sd.close, sd.volume,
               sd.adj_open, sd.adj_close
        FROM public.stocks_daily sd
        JOIN public.symbol_master sm ON sm.code5 = sd.code
        WHERE sm.code4 IN ({ph})
          AND sd.date >= '2022-01-01'
        ORDER BY sd.code, sd.date
    """, symbols)
    rows = cur.fetchall()
    conn.close()
    df = pd.DataFrame(rows, columns=['code5','date','open','high','low','close','volume','adj_open','adj_close'])
    df['date'] = pd.to_datetime(df['date'])
    for c in ['open','high','low','close','volume','adj_open','adj_close']:
        df[c] = pd.to_numeric(df[c])
    return df

def load_macro_daily():
    """NK225, SOX的なマクロデータ"""
    conn = psycopg2.connect(**PG)
    cur = conn.cursor()
    cur.execute("""
        SELECT trade_date, symbol, close
        FROM macro.daily_ohlcv
        WHERE symbol IN ('N225','SOX','^N225','NKY')
          AND trade_date >= '2022-01-01'
        ORDER BY trade_date
    """)
    rows = cur.fetchall()
    conn.close()
    df = pd.DataFrame(rows, columns=['date','symbol','close'])
    df['date'] = pd.to_datetime(df['date'])
    df['close'] = pd.to_numeric(df['close'])
    return df

def load_symbol_map(symbols):
    conn = psycopg2.connect(**PG)
    cur = conn.cursor()
    ph = ','.join(['%s'] * len(symbols))
    cur.execute(f"SELECT code5, code4 FROM symbol_master WHERE code4 IN ({ph})", symbols)
    mapping = {r[0]: r[1] for r in cur.fetchall()}
    conn.close()
    return mapping

def summarize(rets, name, period_days=1):
    if len(rets) < 20:
        return (name, 0, 0.0, 0.0, 0.0, 0.0, "データ不足")
    arr = np.array(rets)
    n = len(arr)
    mean = arr.mean()
    std = arr.std()
    t = stats.ttest_1samp(arr, 0).statistic
    trade_days = 252 / period_days
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
    rsi = 100 - 100 / (1 + rs)
    return rsi

def main():
    print("日次・スイング戦略拡張")
    print("ロード中...")
    sym_map = load_symbol_map(SYMBOLS)
    code4_map = {v: k for k, v in sym_map.items()}  # code4 -> code5

    daily = load_daily(SYMBOLS)
    daily['code4'] = daily['code5'].map(sym_map)
    daily = daily.dropna(subset=['code4'])
    print(f"  日次:{len(daily):,}行 期間:{daily['date'].min().date()}~{daily['date'].max().date()}")

    macro = load_macro_daily()
    print(f"  マクロ:{len(macro):,}行 symbols:{macro['symbol'].unique()}")

    results = []

    # NK225取得
    nk_sym = None
    for s in ['^N225', 'N225', 'NKY']:
        sub = macro[macro['symbol'] == s].copy()
        if len(sub) > 100:
            nk_sym = s
            break

    # =================================================================
    # 日次戦略
    # =================================================================

    # D4: NK225 RSI<30バウンス → 翌日ロング
    print("\n[D4: NK225 RSI<30バウンス]")
    d4_rets = []
    if nk_sym:
        nk = macro[macro['symbol'] == nk_sym].copy().sort_values('date')
        nk['rsi14'] = compute_rsi(nk['close'])
        nk = nk.dropna(subset=['rsi14'])
        for sym, g in daily.groupby('code4'):
            g = g.sort_values('date').reset_index(drop=True)
            for i in range(1, len(g) - 2):
                d = g.iloc[i]['date']
                nk_day = nk[nk['date'] == d]
                if nk_day.empty: continue
                if nk_day.iloc[0]['rsi14'] >= 30: continue
                # 翌日の変化率
                ret = (g.iloc[i+1]['adj_close'] - g.iloc[i]['adj_close']) / g.iloc[i]['adj_close']
                d4_rets.append(ret)
    results.append(summarize(d4_rets, "D4: NK225 RSI<30翌日ロング"))

    # D5: 52週高値ブレイク翌日
    print("\n[D5: 52週高値ブレイク翌日]")
    d5_rets = []
    for sym, g in daily.groupby('code4'):
        g = g.sort_values('date').reset_index(drop=True)
        g['high52w'] = g['high'].rolling(252, min_periods=100).max().shift(1)
        for i in range(1, len(g) - 1):
            row = g.iloc[i]
            if pd.isna(row['high52w']): continue
            if row['high'] > row['high52w']:  # 52週高値ブレイク
                ret = (g.iloc[i+1]['adj_close'] - row['adj_close']) / row['adj_close']
                d5_rets.append(ret)
    results.append(summarize(d5_rets, "D5: 52週高値ブレイク翌日"))

    # D6: 個別銘柄-2%以上 → 翌日リバウンド
    print("\n[D6: 個別-2%翌日リバウンド]")
    d6_rets = []
    for sym, g in daily.groupby('code4'):
        g = g.sort_values('date').reset_index(drop=True)
        for i in range(1, len(g) - 1):
            ret_today = (g.iloc[i]['adj_close'] - g.iloc[i-1]['adj_close']) / g.iloc[i-1]['adj_close']
            if ret_today > -0.02: continue
            ret = (g.iloc[i+1]['adj_close'] - g.iloc[i]['adj_close']) / g.iloc[i]['adj_close']
            d6_rets.append(ret)
    results.append(summarize(d6_rets, "D6: 個別-2%以上翌日リバウンド"))

    # D7: 月末効果 (月末3~5営業日前に買い → 月初1営業日)
    print("\n[D7: 月末リバランス効果]")
    d7_rets = []
    for sym, g in daily.groupby('code4'):
        g = g.sort_values('date').reset_index(drop=True)
        g['ym'] = g['date'].dt.to_period('M')
        for ym, mg in g.groupby('ym'):
            idx = mg.index.tolist()
            if len(idx) < 5: continue
            # 月末3日前の日に買い、最終日に売り
            entry_idx = idx[-3]
            exit_idx = idx[-1]
            ep = g.loc[entry_idx, 'adj_close']
            xp = g.loc[exit_idx, 'adj_close']
            if ep > 0:
                d7_rets.append((xp - ep) / ep)
    results.append(summarize(d7_rets, "D7: 月末3日前→月末最終日", period_days=3))

    # D8: 曜日効果 (月曜日のリターン)
    print("\n[D8: 曜日別リターン]")
    for dow in range(5):
        dow_rets = []
        dow_name = ['月','火','水','木','金'][dow]
        for sym, g in daily.groupby('code4'):
            g = g.sort_values('date').reset_index(drop=True)
            for i in range(1, len(g)):
                if g.iloc[i]['date'].weekday() != dow: continue
                ret = (g.iloc[i]['adj_close'] - g.iloc[i-1]['adj_close']) / g.iloc[i-1]['adj_close']
                dow_rets.append(ret)
        results.append(summarize(dow_rets, f"D8: {dow_name}曜日リターン"))

    # D9: ギャップアップ当日続伸
    print("\n[D9: ギャップアップ続伸]")
    gap_up_rets, gap_down_rets = [], []
    for sym, g in daily.groupby('code4'):
        g = g.sort_values('date').reset_index(drop=True)
        for i in range(1, len(g)):
            prev_close = g.iloc[i-1]['adj_close']
            today_open = g.iloc[i]['adj_open']
            if prev_close <= 0: continue
            gap = (today_open - prev_close) / prev_close
            ret = (g.iloc[i]['adj_close'] - today_open) / today_open
            if gap > 0.01:  # +1%以上ギャップアップ
                gap_up_rets.append(ret)
            elif gap < -0.01:  # -1%以上ギャップダウン
                gap_down_rets.append(ret)
    results.append(summarize(gap_up_rets, "D9: ギャップアップ+1%→当日継続"))
    results.append(summarize(gap_down_rets, "D9b: ギャップダウン-1%→当日逆張り(ロング)"))

    # =================================================================
    # スイング戦略
    # =================================================================

    # S4: RSI(14)<30 → 5日後
    print("\n[S4: RSI<30 5日バウンス]")
    s4_rets = []
    for sym, g in daily.groupby('code4'):
        g = g.sort_values('date').reset_index(drop=True)
        g['rsi14'] = compute_rsi(g['adj_close'])
        for i in range(len(g) - 6):
            if pd.isna(g.iloc[i]['rsi14']): continue
            if g.iloc[i]['rsi14'] >= 30: continue
            entry = g.iloc[i]['adj_close']
            exit_p = g.iloc[i+5]['adj_close']
            s4_rets.append((exit_p - entry) / entry)
    results.append(summarize(s4_rets, "S4: RSI<30→5日後", period_days=5))

    # S5: 52週高値ブレイク後20日
    print("\n[S5: 52週高値ブレイク後20日]")
    s5_rets = []
    for sym, g in daily.groupby('code4'):
        g = g.sort_values('date').reset_index(drop=True)
        g['high52w'] = g['high'].rolling(252, min_periods=100).max().shift(1)
        for i in range(len(g) - 21):
            row = g.iloc[i]
            if pd.isna(row['high52w']): continue
            if row['high'] <= row['high52w']: continue
            entry = row['adj_close']
            exit_p = g.iloc[i+20]['adj_close']
            s5_rets.append((exit_p - entry) / entry)
    results.append(summarize(s5_rets, "S5: 52週高値ブレイク→20日後", period_days=20))

    # S6: 5/20日ゴールデンクロス→5日
    print("\n[S6: 5/20日GC→5日]")
    s6_rets = []
    for sym, g in daily.groupby('code4'):
        g = g.sort_values('date').reset_index(drop=True)
        g['ma5'] = g['adj_close'].rolling(5).mean()
        g['ma20'] = g['adj_close'].rolling(20).mean()
        for i in range(21, len(g) - 6):
            # ゴールデンクロス検出 (前日はma5<ma20, 今日はma5>ma20)
            prev = g.iloc[i-1]
            curr = g.iloc[i]
            if pd.isna(prev['ma5']) or pd.isna(curr['ma20']): continue
            if prev['ma5'] >= prev['ma20']: continue  # 前日はすでに上
            if curr['ma5'] <= curr['ma20']: continue  # 今日もクロスしてない
            entry = curr['adj_close']
            exit_p = g.iloc[i+5]['adj_close']
            s6_rets.append((exit_p - entry) / entry)
    results.append(summarize(s6_rets, "S6: 5/20日GC→5日後", period_days=5))

    # S7: 前月-10%以上の銘柄→翌月初
    print("\n[S7: 前月-10%翌月リバウンド]")
    s7_rets = []
    for sym, g in daily.groupby('code4'):
        g = g.sort_values('date').reset_index(drop=True)
        g['ym'] = g['date'].dt.to_period('M')
        periods = sorted(g['ym'].unique())
        for k in range(1, len(periods) - 1):
            prev_m = periods[k-1]
            curr_m = periods[k]
            prev_g = g[g['ym'] == prev_m]
            curr_g = g[g['ym'] == curr_m]
            if len(prev_g) < 5 or len(curr_g) < 10: continue
            prev_ret = (prev_g.iloc[-1]['adj_close'] - prev_g.iloc[0]['adj_close']) / prev_g.iloc[0]['adj_close']
            if prev_ret > -0.10: continue  # 前月-10%未満はスキップ
            entry = curr_g.iloc[0]['adj_close']
            exit_p = curr_g.iloc[min(9, len(curr_g)-1)]['adj_close']
            s7_rets.append((exit_p - entry) / entry)
    results.append(summarize(s7_rets, "S7: 前月-10%→翌月初10日", period_days=10))

    # S8: 出来高急増(2倍以上)翌日から5日
    print("\n[S8: 出来高2倍急増→5日]")
    s8_long, s8_short = [], []
    for sym, g in daily.groupby('code4'):
        g = g.sort_values('date').reset_index(drop=True)
        g['vol_ma20'] = g['volume'].rolling(20).mean()
        for i in range(20, len(g) - 6):
            row = g.iloc[i]
            if pd.isna(row['vol_ma20']) or row['vol_ma20'] == 0: continue
            if row['volume'] < 2 * row['vol_ma20']: continue
            # 当日の方向
            day_ret = (row['adj_close'] - g.iloc[i-1]['adj_close']) / g.iloc[i-1]['adj_close']
            entry = row['adj_close']
            exit_p = g.iloc[i+5]['adj_close']
            if day_ret > 0:
                s8_long.append((exit_p - entry) / entry)
            else:
                s8_short.append((exit_p - entry) / entry)
    results.append(summarize(s8_long, "S8: 出来高2倍+上昇→5日継続", period_days=5))
    results.append(summarize(s8_short, "S8b: 出来高2倍+下落→5日逆張りロング", period_days=5))

    # S9: RSI(14)>70 → 5日後 (過熱後の反落)
    print("\n[S9: RSI>70過熱→5日反落]")
    s9_rets = []
    for sym, g in daily.groupby('code4'):
        g = g.sort_values('date').reset_index(drop=True)
        g['rsi14'] = compute_rsi(g['adj_close'])
        for i in range(len(g) - 6):
            if pd.isna(g.iloc[i]['rsi14']): continue
            if g.iloc[i]['rsi14'] <= 70: continue
            entry = g.iloc[i]['adj_close']
            exit_p = g.iloc[i+5]['adj_close']
            s9_rets.append(-(exit_p - entry) / entry)  # ショート
    results.append(summarize(s9_rets, "S9: RSI>70→5日後ショート", period_days=5))

    # S10: ボリンジャーバンド±2σタッチ後10日
    print("\n[S10: BB±2σタッチ後]")
    bb_up_rets, bb_down_rets = [], []
    for sym, g in daily.groupby('code4'):
        g = g.sort_values('date').reset_index(drop=True)
        g['ma20'] = g['adj_close'].rolling(20).mean()
        g['std20'] = g['adj_close'].rolling(20).std()
        g['bb_up'] = g['ma20'] + 2 * g['std20']
        g['bb_dn'] = g['ma20'] - 2 * g['std20']
        for i in range(20, len(g) - 11):
            row = g.iloc[i]
            if any(pd.isna([row['bb_up'], row['bb_dn']])): continue
            entry = row['adj_close']
            exit_p = g.iloc[i+10]['adj_close']
            if row['close'] >= row['bb_up']:
                bb_up_rets.append(-(exit_p - entry) / entry)  # 上バンドタッチ→逆張り売り
            elif row['close'] <= row['bb_dn']:
                bb_down_rets.append((exit_p - entry) / entry)  # 下バンドタッチ→逆張り買い
    results.append(summarize(bb_up_rets, "S10a: BB+2σタッチ→10日逆張りショート", period_days=10))
    results.append(summarize(bb_down_rets, "S10b: BB-2σタッチ→10日逆張りロング", period_days=10))

    # 出力
    print("\n" + "=" * 90)
    print("  日次・スイング拡張戦略 結果")
    print("=" * 90)
    print(f"  {'戦略':<56} {'N':>5}  {'勝率':>6}  {'期待値':>8}  {'t値':>7}  {'Sharpe':>7}  判定")
    print("-" * 90)
    positive = []
    daily_pos = []
    swing_pos = []
    for r in results:
        name, n, win, mean, t, sharpe, cls = r
        print(f"  {name:<56} {n:>5}  {win*100:>5.1f}%  {mean:>+8.3f}%  {t:>+7.2f}  {sharpe:>+7.2f}  {cls}")
        if t >= 2.0:
            positive.append(r)
            if name.startswith('D'):
                daily_pos.append(r)
            else:
                swing_pos.append(r)

    print(f"\n  ✅ プラスエッジ: {len(positive)}個")
    for r in positive:
        print(f"     {r[0]}: t={r[4]:+.2f} 勝率={r[2]*100:.1f}% 期待値={r[3]:+.3f}%")

    outpath = "/mnt/d/Root/ClaudeCode/01_Trading/japan-stocks/analyses/20260605_sector_edge_research/daily_swing_extended.csv"
    with open(outpath, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['strategy', 'N', 'win_rate', 'mean_ret', 't_stat', 'sharpe', 'cls'])
        for r in results:
            w.writerow(r)
    print(f"\n保存: {outpath}  完了")

if __name__ == '__main__':
    main()
