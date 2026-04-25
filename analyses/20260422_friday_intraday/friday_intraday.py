"""
非鉄金属・半導体 金曜イントラデイ傾向分析
分析日: 2026-04-22

分析内容:
1. 曜日別の寄付→引け日次リターン比較
2. 金曜の時間帯別累積リターン（引け前の売りパターン確認）
3. 金曜の寄付ギャップ傾向（月〜木との比較）
4. 住山(5713.T)フォーカス：金曜の時間帯別詳細
5. 金曜の分時別ボラティリティカーブ
"""

import psycopg2
import pandas as pd
import numpy as np

PG_CONFIG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}

SYMBOLS = {
    # 非鉄金属
    "5713.T": "住友鉱山",
    "5711.T": "三菱マテ",
    "5706.T": "三井金属",
    "5803.T": "フジクラ",
    "5802.T": "住友電工",
    # 半導体
    "6857.T": "アドバンテスト",
    "6920.T": "レーザーテック",
    "6146.T": "ディスコ",
    "6861.T": "キーエンス",
}

DAY_NAMES = {0: "月", 1: "火", 2: "水", 3: "木", 4: "金"}


def load_data(sym):
    conn = psycopg2.connect(**PG_CONFIG)
    df = pd.read_sql(
        f"SELECT timestamp, open, high, low, close, volume FROM intraday_data "
        f"WHERE symbol = '{sym}' ORDER BY timestamp",
        conn
    )
    conn.close()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['jst'] = df['timestamp'] + pd.Timedelta(hours=9)
    df = df.dropna(subset=['close']).set_index('jst').sort_index()
    return df


def trading_hours(df):
    h = df.index.hour
    m = df.index.minute
    return df[
        (h == 9) |
        ((h >= 10) & (h < 11)) |
        ((h == 11) & (m <= 30)) |
        ((h == 12) & (m >= 30)) |
        ((h >= 13) & (h < 15)) |
        ((h == 15) & (m <= 30))
    ]


def get_daily_summary(df):
    """日ごとに寄付・引け・最高・最安値、日次リターンをまとめる"""
    rows = []
    for dt, g in df.groupby(df.index.date):
        g_day = trading_hours(g)
        if len(g_day) < 20:
            continue
        dow = pd.Timestamp(dt).dayofweek  # 0=月,4=金
        op = g_day['open'].iloc[0]
        cl = g_day['close'].iloc[-1]
        hi = g_day['high'].max()
        lo = g_day['low'].min()
        vol = g_day['volume'].sum() if 'volume' in g_day.columns else np.nan

        # 前場/後場分割
        mae = g_day[(g_day.index.hour < 11) | ((g_day.index.hour == 11) & (g_day.index.minute <= 30))]
        go = g_day[(g_day.index.hour >= 12) | ((g_day.index.hour == 12) & (g_day.index.minute >= 30))]

        mae_ret = (mae['close'].iloc[-1] / op - 1) * 100 if len(mae) > 0 and op > 0 else np.nan
        go_ret = (cl / mae['close'].iloc[-1] - 1) * 100 if len(mae) > 0 and len(go) > 0 and mae['close'].iloc[-1] > 0 else np.nan
        day_ret = (cl / op - 1) * 100 if op > 0 else np.nan

        rows.append({
            'date': dt, 'dow': dow,
            'open': op, 'close': cl, 'high': hi, 'low': lo,
            'day_ret': day_ret, 'mae_ret': mae_ret, 'go_ret': go_ret,
            'volume': vol, 'n_bars': len(g_day)
        })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────
# 1. 曜日別・日次リターン統計
# ──────────────────────────────────────────────────
def dayofweek_returns(all_daily):
    print("="*65)
    print("1. 曜日別・日次リターン平均（寄→引け）")
    print("="*65)

    print(f"\n{'銘柄':<12}", end="")
    for d in range(5):
        print(f"  {DAY_NAMES[d]}曜", end="")
    print()
    print("-" * 40)

    for sym, name in SYMBOLS.items():
        if sym not in all_daily:
            continue
        df = all_daily[sym]
        print(f"{name:<12}", end="")
        for d in range(5):
            vals = df[df['dow'] == d]['day_ret'].dropna()
            if len(vals) >= 3:
                print(f"  {vals.mean():>+5.2f}%", end="")
            else:
                print(f"  {'N/A':>6}", end="")
        print()

    # 金曜だけ詳細
    print(f"\n{'銘柄':<12}  {'金曜N':>5}  {'金曜平均':>8}  {'金曜中央':>8}  {'金曜σ':>6}  {'勝率':>6}")
    print("-" * 55)
    for sym, name in SYMBOLS.items():
        if sym not in all_daily:
            continue
        fri = all_daily[sym][all_daily[sym]['dow'] == 4]['day_ret'].dropna()
        if len(fri) < 3:
            continue
        wr = (fri > 0).mean() * 100
        print(f"{name:<12}  {len(fri):>5}  {fri.mean():>+7.3f}%  {fri.median():>+7.3f}%  {fri.std():>6.3f}%  {wr:>5.1f}%")


# ──────────────────────────────────────────────────
# 2. 金曜の時間帯別累積リターン（分単位）
# ──────────────────────────────────────────────────
def friday_intraday_cumret(df, name, syms_focus=None):
    """
    金曜日の各バーについて、寄付からの累積リターンを平均する。
    全曜日との比較も出力。
    """
    print("\n" + "="*65)
    print(f"2. {name} — 曜日別・時間帯別累積リターン（寄=0%基準）")
    print("="*65)

    # 1分足で時刻ごとの「寄付比」を計算
    time_keys = []
    # 9:01〜11:30, 12:30〜15:30
    for h in range(9, 16):
        for m in range(0, 60):
            t = pd.Timestamp("2000-01-01") + pd.Timedelta(hours=h, minutes=m)
            t_time = t.time()
            if ((h == 9 and m >= 1) or (h >= 10 and h < 11) or
                    (h == 11 and m <= 30) or (h == 12 and m >= 30) or
                    (h >= 13 and h < 15) or (h == 15 and m <= 30)):
                time_keys.append(t_time)

    # 日ごとに寄付比リターン系列を計算
    rows_by_dow = {d: [] for d in range(5)}
    for dt, g in df.groupby(df.index.date):
        g_day = trading_hours(g).copy()
        if len(g_day) < 20:
            continue
        dow = pd.Timestamp(dt).dayofweek
        op = g_day['open'].iloc[0]
        if op <= 0:
            continue
        # 各バーの寄付比
        g_day['cumret'] = (g_day['close'] / op - 1) * 100
        # 時刻をキーにする
        g_day['time'] = g_day.index.time
        g_by_time = g_day.set_index('time')['cumret']
        rows_by_dow[dow].append(g_by_time)

    # チェックポイント時刻の設定（見やすくするため間引き）
    checkpoints = [
        pd.Timestamp("2000-01-01 09:15").time(),
        pd.Timestamp("2000-01-01 09:30").time(),
        pd.Timestamp("2000-01-01 10:00").time(),
        pd.Timestamp("2000-01-01 10:30").time(),
        pd.Timestamp("2000-01-01 11:00").time(),
        pd.Timestamp("2000-01-01 11:30").time(),
        pd.Timestamp("2000-01-01 12:30").time(),
        pd.Timestamp("2000-01-01 13:00").time(),
        pd.Timestamp("2000-01-01 14:00").time(),
        pd.Timestamp("2000-01-01 14:30").time(),
        pd.Timestamp("2000-01-01 15:00").time(),
        pd.Timestamp("2000-01-01 15:30").time(),
    ]

    print(f"\n  {'時刻':<7}", end="")
    for d in range(5):
        print(f"  {DAY_NAMES[d]}曜", end="")
    print()
    print("  " + "-" * 37)

    for cp in checkpoints:
        print(f"  {cp.strftime('%H:%M'):<7}", end="")
        for d in range(5):
            series_list = rows_by_dow[d]
            vals = [s[cp] for s in series_list if cp in s.index]
            if vals:
                print(f"  {np.mean(vals):>+5.3f}", end="")
            else:
                print(f"  {'---':>6}", end="")
        print()


# ──────────────────────────────────────────────────
# 3. 住山フォーカス：金曜詳細分析
# ──────────────────────────────────────────────────
def sumitomo_friday_focus(df, daily):
    print("\n" + "="*65)
    print("3. 住山(5713.T) 金曜フォーカス詳細")
    print("="*65)

    fri_daily = daily[daily['dow'] == 4].copy()
    non_fri = daily[daily['dow'] != 4].copy()

    print(f"\n  金曜N={len(fri_daily)}, 非金曜N={len(non_fri)}")
    print(f"  {'':20}  {'金曜':>8}  {'月〜木':>8}")
    print("  " + "-" * 42)

    metrics = [
        ('日次リターン平均', 'day_ret', '{:+.3f}%'),
        ('日次リターン中央値', 'day_ret', None),
        ('前場リターン平均', 'mae_ret', '{:+.3f}%'),
        ('後場リターン平均', 'go_ret', '{:+.3f}%'),
        ('日次σ', 'day_ret', '{:.3f}%'),
        ('勝率', None, '{:.1f}%'),
    ]

    for label, col, fmt in metrics:
        if col is None:
            fri_v = (fri_daily['day_ret'] > 0).mean() * 100
            non_v = (non_fri['day_ret'] > 0).mean() * 100
            print(f"  {label:<20}  {fri_v:>7.1f}%  {non_v:>7.1f}%")
        elif label.endswith('中央値'):
            fri_v = fri_daily[col].median()
            non_v = non_fri[col].median()
            print(f"  {label:<20}  {fri_v:>+7.3f}%  {non_v:>+7.3f}%")
        elif label.endswith('σ'):
            fri_v = fri_daily[col].std()
            non_v = non_fri[col].std()
            print(f"  {label:<20}  {fri_v:>7.3f}%  {non_v:>7.3f}%")
        else:
            fri_v = fri_daily[col].mean()
            non_v = non_fri[col].mean()
            print(f"  {label:<20}  {fri_v:>+7.3f}%  {non_v:>+7.3f}%")

    # 金曜の前場・後場パターン
    print("\n  --- 金曜の前場/後場パターン別 ---")
    patterns = {
        '前高後高（追随）': (fri_daily['mae_ret'] > 0) & (fri_daily['go_ret'] > 0),
        '前高後安（手仕舞い）': (fri_daily['mae_ret'] > 0) & (fri_daily['go_ret'] < 0),
        '前安後高（V字）': (fri_daily['mae_ret'] < 0) & (fri_daily['go_ret'] > 0),
        '前安後安（弱い）': (fri_daily['mae_ret'] < 0) & (fri_daily['go_ret'] < 0),
    }
    for pname, mask in patterns.items():
        cnt = mask.sum()
        avg = fri_daily.loc[mask, 'day_ret'].mean() if cnt > 0 else np.nan
        print(f"  {pname:<18}  N={cnt:>3}  ({cnt/len(fri_daily)*100:>4.1f}%)  日次平均: {avg:>+6.3f}%" if not np.isnan(avg) else f"  {pname:<18}  N={cnt:>3}")

    # 引け前30分の傾向（金曜 vs 平日）
    print("\n  --- 引け前30分の動き（15:00→15:30）---")
    for label, target_daily in [("金曜", fri_daily), ("月〜木", non_fri)]:
        dates = target_daily['date'].tolist()
        last30_rets = []
        for dt in dates:
            g_day = trading_hours(df[df.index.date == dt])
            bars_1500 = g_day[(g_day.index.hour == 15) & (g_day.index.minute == 0)]
            bars_1530 = g_day[(g_day.index.hour == 15) & (g_day.index.minute == 30)]
            if len(bars_1500) > 0 and len(bars_1530) > 0:
                p1500 = bars_1500['close'].iloc[0]
                p1530 = bars_1530['close'].iloc[-1]
                if p1500 > 0:
                    last30_rets.append((p1530 / p1500 - 1) * 100)
        if last30_rets:
            arr = np.array(last30_rets)
            wr = (arr > 0).mean() * 100
            print(f"  {label:<6}  平均{arr.mean():>+6.3f}%  σ={arr.std():.3f}%  勝率{wr:.1f}%  N={len(arr)}")


# ──────────────────────────────────────────────────
# 4. 半導体 vs 非鉄 金曜パターン比較
# ──────────────────────────────────────────────────
def sector_friday_compare(all_daily):
    print("\n" + "="*65)
    print("4. 半導体 vs 非鉄 金曜 vs 月〜木 比較")
    print("="*65)

    nonfer = ["5713.T", "5711.T", "5706.T", "5803.T", "5802.T"]
    semi = ["6857.T", "6920.T", "6146.T", "6861.T"]

    print(f"\n{'銘柄':<12}  {'金曜平均':>8}  {'月〜木平均':>9}  {'差':>6}  {'金曜σ':>6}  {'金曜勝率':>8}  {'引け前30分':>10}")
    print("-" * 70)

    for group_name, syms in [("--- 非鉄 ---", nonfer), ("--- 半導体 ---", semi)]:
        print(f"\n  {group_name}")
        for sym in syms:
            if sym not in all_daily or sym not in SYMBOLS:
                continue
            name = SYMBOLS[sym]
            df = all_daily[sym]
            fri = df[df['dow'] == 4]['day_ret'].dropna()
            non = df[df['dow'] != 4]['day_ret'].dropna()
            if len(fri) < 3:
                continue
            diff = fri.mean() - non.mean()
            wr = (fri > 0).mean() * 100
            # 引け前30分は後で計算するため空欄
            print(f"  {name:<12}  {fri.mean():>+7.3f}%  {non.mean():>+8.3f}%  {diff:>+5.3f}%  {fri.std():>6.3f}%  {wr:>7.1f}%")


# ──────────────────────────────────────────────────
# 5. 住山 金曜の分足累積リターン（詳細版）
# ──────────────────────────────────────────────────
def sumitomo_minute_cumret_detail(df):
    print("\n" + "="*65)
    print("5. 住山 金曜 vs 月〜木 — 分足累積リターン詳細")
    print("="*65)

    fri_rets = []
    non_rets = []

    for dt, g in df.groupby(df.index.date):
        g_day = trading_hours(g).copy()
        if len(g_day) < 20:
            continue
        dow = pd.Timestamp(dt).dayofweek
        op = g_day['open'].iloc[0]
        if op <= 0:
            continue
        g_day['cumret'] = (g_day['close'] / op - 1) * 100
        g_day['time'] = g_day.index.time
        s = g_day.set_index('time')['cumret']
        if dow == 4:
            fri_rets.append(s)
        else:
            non_rets.append(s)

    checkpoints = [
        ("09:05", pd.Timestamp("2000-01-01 09:05").time()),
        ("09:15", pd.Timestamp("2000-01-01 09:15").time()),
        ("09:30", pd.Timestamp("2000-01-01 09:30").time()),
        ("10:00", pd.Timestamp("2000-01-01 10:00").time()),
        ("10:30", pd.Timestamp("2000-01-01 10:30").time()),
        ("11:00", pd.Timestamp("2000-01-01 11:00").time()),
        ("11:30", pd.Timestamp("2000-01-01 11:30").time()),
        ("12:30", pd.Timestamp("2000-01-01 12:30").time()),
        ("13:00", pd.Timestamp("2000-01-01 13:00").time()),
        ("13:30", pd.Timestamp("2000-01-01 13:30").time()),
        ("14:00", pd.Timestamp("2000-01-01 14:00").time()),
        ("14:30", pd.Timestamp("2000-01-01 14:30").time()),
        ("15:00", pd.Timestamp("2000-01-01 15:00").time()),
        ("15:30", pd.Timestamp("2000-01-01 15:30").time()),
    ]

    print(f"\n  {'時刻':<7}  {'金曜平均':>8}  {'月〜木平均':>9}  {'差':>6}  {'金曜勝率':>8}")
    print("  " + "-" * 48)

    for label, cp in checkpoints:
        fri_v = [s[cp] for s in fri_rets if cp in s.index]
        non_v = [s[cp] for s in non_rets if cp in s.index]
        if fri_v and non_v:
            fmean = np.mean(fri_v)
            nmean = np.mean(non_v)
            diff = fmean - nmean
            fwr = (np.array(fri_v) > 0).mean() * 100
            print(f"  {label:<7}  {fmean:>+7.3f}%  {nmean:>+8.3f}%  {diff:>+5.3f}%  {fwr:>7.1f}%")


# ──────────────────────────────────────────────────
# メイン
# ──────────────────────────────────────────────────
if __name__ == '__main__':
    print("=== データロード ===")
    all_data = {}
    all_daily = {}
    for sym, name in SYMBOLS.items():
        try:
            df = load_data(sym)
            all_data[sym] = df
            all_daily[sym] = get_daily_summary(df)
            n_fri = (all_daily[sym]['dow'] == 4).sum()
            print(f"  {sym} ({name}): {len(df)}バー / 金曜{n_fri}日")
        except Exception as e:
            print(f"  {sym}: ロード失敗 ({e})")

    dayofweek_returns(all_daily)
    friday_intraday_cumret(all_data['5713.T'], "住山(5713.T)")
    sumitomo_friday_focus(all_data['5713.T'], all_daily['5713.T'])
    sector_friday_compare(all_daily)
    sumitomo_minute_cumret_detail(all_data['5713.T'])

    print("\n分析完了!")
