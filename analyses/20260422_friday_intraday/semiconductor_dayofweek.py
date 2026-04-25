"""
半導体銘柄 曜日別イントラデイ傾向分析
分析日: 2026-04-22

非鉄と同様の軸で：
1. 曜日別日次リターン詳細（平均/中央値/勝率/σ）
2. 各銘柄の時間帯別累積リターン（曜日別）
3. 火曜の弱さ・水曜/金曜の強さを非鉄と定量比較
4. 前場/後場パターン × 曜日
5. 引け前30分の動き（曜日別）
"""

import psycopg2
import pandas as pd
import numpy as np

PG_CONFIG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}

SEMI = {
    "6857.T": "アドバンテスト",
    "6920.T": "レーザーテック",
    "6146.T": "ディスコ",
    "6861.T": "キーエンス",
}
NONFER = {
    "5713.T": "住友鉱山",
    "5711.T": "三菱マテ",
    "5706.T": "三井金属",
}
DAY = {0: "月", 1: "火", 2: "水", 3: "木", 4: "金"}


def load_data(sym):
    conn = psycopg2.connect(**PG_CONFIG)
    df = pd.read_sql(
        f"SELECT timestamp, open, high, low, close, volume FROM intraday_data "
        f"WHERE symbol='{sym}' ORDER BY timestamp", conn)
    conn.close()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['jst'] = df['timestamp'] + pd.Timedelta(hours=9)
    df = df.dropna(subset=['close']).set_index('jst').sort_index()
    return df


def trading_hours(df):
    h, m = df.index.hour, df.index.minute
    return df[
        (h == 9) | ((h >= 10) & (h < 11)) | ((h == 11) & (m <= 30)) |
        ((h == 12) & (m >= 30)) | ((h >= 13) & (h < 15)) | ((h == 15) & (m <= 30))
    ]


def get_daily_summary(df):
    rows = []
    for dt, g in df.groupby(df.index.date):
        g_day = trading_hours(g)
        if len(g_day) < 20:
            continue
        dow = pd.Timestamp(dt).dayofweek
        op = g_day['open'].iloc[0]
        cl = g_day['close'].iloc[-1]
        mae = g_day[(g_day.index.hour < 12)]
        go  = g_day[(g_day.index.hour >= 12)]
        if op <= 0:
            continue
        mae_ret = (mae['close'].iloc[-1] / op - 1) * 100 if len(mae) > 0 else np.nan
        go_ret  = (cl / mae['close'].iloc[-1] - 1) * 100 if len(mae) > 0 and len(go) > 0 and mae['close'].iloc[-1] > 0 else np.nan
        rows.append({'date': dt, 'dow': dow,
                     'day_ret': (cl/op-1)*100, 'mae_ret': mae_ret, 'go_ret': go_ret})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# 1. 曜日別統計テーブル（詳細）
# ─────────────────────────────────────────────
def dow_stats_detail(all_daily, symbols):
    print("="*70)
    print("1. 曜日別日次リターン詳細（平均 / 中央値 / σ / 勝率）")
    print("="*70)

    for sym, name in symbols.items():
        if sym not in all_daily:
            continue
        df = all_daily[sym]
        print(f"\n  【{name}】")
        print(f"  {'曜日':<4}  {'N':>4}  {'平均':>7}  {'中央値':>7}  {'σ':>6}  {'勝率':>6}  {'前場平均':>8}  {'後場平均':>8}")
        print("  " + "-" * 60)
        for d in range(5):
            sub = df[df['dow'] == d]
            r = sub['day_ret'].dropna()
            if len(r) < 3:
                continue
            wr = (r > 0).mean() * 100
            mret = sub['mae_ret'].mean()
            gret = sub['go_ret'].mean()
            print(f"  {DAY[d]}曜   {len(r):>4}  {r.mean():>+6.3f}%  {r.median():>+6.3f}%  {r.std():>6.3f}%  {wr:>5.1f}%  {mret:>+7.3f}%  {gret:>+7.3f}%")


# ─────────────────────────────────────────────
# 2. 時間帯別累積リターン（全曜日）
# ─────────────────────────────────────────────
def cumret_by_dow(df, name):
    checkpoints = [
        ("09:05", pd.Timestamp("2000-01-01 09:05").time()),
        ("09:15", pd.Timestamp("2000-01-01 09:15").time()),
        ("09:30", pd.Timestamp("2000-01-01 09:30").time()),
        ("10:00", pd.Timestamp("2000-01-01 10:00").time()),
        ("10:30", pd.Timestamp("2000-01-01 10:30").time()),
        ("11:00", pd.Timestamp("2000-01-01 11:00").time()),
        ("11:30", pd.Timestamp("2000-01-01 11:30").time()),
        ("12:30", pd.Timestamp("2000-01-01 12:30").time()),
        ("13:30", pd.Timestamp("2000-01-01 13:30").time()),
        ("14:00", pd.Timestamp("2000-01-01 14:00").time()),
        ("14:30", pd.Timestamp("2000-01-01 14:30").time()),
        ("15:00", pd.Timestamp("2000-01-01 15:00").time()),
        ("15:30", pd.Timestamp("2000-01-01 15:30").time()),
    ]

    rows_by_dow = {d: [] for d in range(5)}
    for dt, g in df.groupby(df.index.date):
        g_day = trading_hours(g).copy()
        if len(g_day) < 20:
            continue
        op = g_day['open'].iloc[0]
        if op <= 0:
            continue
        dow = pd.Timestamp(dt).dayofweek
        g_day['cumret'] = (g_day['close'] / op - 1) * 100
        g_day['time'] = g_day.index.time
        rows_by_dow[dow].append(g_day.set_index('time')['cumret'])

    print(f"\n  【{name}】 時間帯別累積リターン（寄=0%）")
    print(f"  {'時刻':<7}", end="")
    for d in range(5):
        print(f"   {DAY[d]}曜", end="")
    print()
    print("  " + "-" * 37)

    for label, cp in checkpoints:
        print(f"  {label:<7}", end="")
        for d in range(5):
            vals = [s[cp] for s in rows_by_dow[d] if cp in s.index]
            if vals:
                print(f"  {np.mean(vals):>+5.3f}", end="")
            else:
                print(f"  {'---':>6}", end="")
        print()

    return rows_by_dow


# ─────────────────────────────────────────────
# 3. 火曜弱さ・水曜/金曜強さの定量比較（非鉄 vs 半導体）
# ─────────────────────────────────────────────
def sector_dow_compare(all_daily):
    print("\n" + "="*70)
    print("3. 非鉄 vs 半導体 — 火曜/水曜/金曜リターン差の比較")
    print("="*70)

    rows = []
    for sym, name in {**NONFER, **SEMI}.items():
        if sym not in all_daily:
            continue
        df = all_daily[sym]
        avg = {d: df[df['dow']==d]['day_ret'].mean() for d in range(5)}
        wr  = {d: (df[df['dow']==d]['day_ret'] > 0).mean()*100 for d in range(5)}
        sector = "非鉄" if sym in NONFER else "半導体"
        rows.append({'名前': name, 'セクター': sector,
                     '月': avg[0], '火': avg[1], '水': avg[2], '木': avg[3], '金': avg[4],
                     '火-月〜木差': avg[1] - np.mean([avg[d] for d in [0,2,3,4]]),
                     '金曜勝率': wr[4]})

    tbl = pd.DataFrame(rows)
    print(f"\n  {'銘柄':<12}  {'月':>6}  {'火':>6}  {'水':>6}  {'木':>6}  {'金':>6}  {'火の乖離':>8}  {'金勝率':>7}")
    print("  " + "-" * 70)
    for _, r in tbl.iterrows():
        mark = "◀非鉄" if r['セクター'] == "非鉄" else "◀半導"
        print(f"  {r['名前']:<12}  {r['月']:>+5.2f}%  {r['火']:>+5.2f}%  {r['水']:>+5.2f}%  {r['木']:>+5.2f}%  {r['金']:>+5.2f}%  {r['火-月〜木差']:>+7.3f}%  {r['金曜勝率']:>6.1f}%  {mark}")


# ─────────────────────────────────────────────
# 4. 前場/後場パターン × 曜日（半導体4銘柄）
# ─────────────────────────────────────────────
def mae_go_pattern_by_dow(all_daily):
    print("\n" + "="*70)
    print("4. 前場/後場パターン × 曜日（半導体）")
    print("="*70)

    pattern_labels = {
        '前高後高': lambda r: (r['mae_ret'] > 0) & (r['go_ret'] > 0),
        '前高後安': lambda r: (r['mae_ret'] > 0) & (r['go_ret'] < 0),
        '前安後高': lambda r: (r['mae_ret'] < 0) & (r['go_ret'] > 0),
        '前安後安': lambda r: (r['mae_ret'] < 0) & (r['go_ret'] < 0),
    }

    for sym, name in SEMI.items():
        if sym not in all_daily:
            continue
        df = all_daily[sym]
        print(f"\n  【{name}】 前場/後場パターン × 曜日（件数 / 日次平均リターン）")
        print(f"  {'':12}", end="")
        for d in range(5):
            print(f"   {DAY[d]}曜(N/avg)", end="")
        print()
        print("  " + "-" * 65)

        for pname, cond in pattern_labels.items():
            print(f"  {pname:<12}", end="")
            for d in range(5):
                sub = df[df['dow'] == d].dropna(subset=['mae_ret', 'go_ret'])
                mask = cond(sub)
                cnt = mask.sum()
                avg = sub.loc[mask, 'day_ret'].mean() if cnt > 0 else np.nan
                if cnt > 0:
                    print(f"  {cnt:>2}/{avg:>+5.2f}%", end="")
                else:
                    print(f"  {'--/---':>9}", end="")
            print()


# ─────────────────────────────────────────────
# 5. 引け前30分の動き（曜日別）
# ─────────────────────────────────────────────
def last30min_by_dow(all_data, all_daily, symbols):
    print("\n" + "="*70)
    print("5. 引け前30分（15:00→15:30）の動き — 曜日別")
    print("="*70)

    for sym, name in symbols.items():
        if sym not in all_data:
            continue
        df = all_data[sym]
        daily = all_daily[sym]

        print(f"\n  【{name}】")
        print(f"  {'曜日':<4}  {'N':>4}  {'平均':>7}  {'σ':>6}  {'勝率':>6}")
        print("  " + "-" * 35)

        for d in range(5):
            dates = daily[daily['dow'] == d]['date'].tolist()
            rets = []
            for dt in dates:
                g = trading_hours(df[df.index.date == dt])
                b1500 = g[(g.index.hour == 15) & (g.index.minute == 0)]
                b1530 = g[(g.index.hour == 15) & (g.index.minute == 30)]
                if len(b1500) > 0 and len(b1530) > 0:
                    p0 = b1500['close'].iloc[0]
                    p1 = b1530['close'].iloc[-1]
                    if p0 > 0:
                        rets.append((p1/p0 - 1)*100)
            if rets:
                arr = np.array(rets)
                wr = (arr > 0).mean() * 100
                print(f"  {DAY[d]}曜   {len(arr):>4}  {arr.mean():>+6.3f}%  {arr.std():>6.3f}%  {wr:>5.1f}%")


# ─────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print("=== データロード ===")
    all_data   = {}
    all_daily  = {}
    all_syms   = {**SEMI, **NONFER}
    for sym, name in all_syms.items():
        try:
            df = load_data(sym)
            all_data[sym]  = df
            all_daily[sym] = get_daily_summary(df)
            print(f"  {sym} ({name}): {len(df)}バー")
        except Exception as e:
            print(f"  {sym}: 失敗 ({e})")

    # 1. 曜日別詳細（半導体）
    dow_stats_detail(all_daily, SEMI)

    # 2. 時間帯別累積リターン
    print("\n" + "="*70)
    print("2. 半導体 時間帯別累積リターン（寄=0%基準）")
    print("="*70)
    for sym, name in SEMI.items():
        if sym in all_data:
            cumret_by_dow(all_data[sym], name)

    # 3. セクター比較
    sector_dow_compare(all_daily)

    # 4. 前場/後場パターン × 曜日
    mae_go_pattern_by_dow(all_daily)

    # 5. 引け前30分
    last30min_by_dow(all_data, all_daily, SEMI)

    print("\n分析完了!")
