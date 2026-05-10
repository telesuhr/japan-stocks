#!/usr/bin/env python3
"""
ENEOS VWAP Trend シグナル自動判定スクリプト (v1.0)

9:30 頃に実行:
  python3 signal_check.py

シグナル = 5020.T (ENEOS) の 9:30時点VWAP乖離 ≥ ±50bps
発動 → 9:31〜 成行エントリー → 15:30 引成決済
"""
import sys
from datetime import date, datetime
import psycopg2
import pandas as pd

PG_CONFIG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}

SYMBOL    = "5020.T"
NAME      = "ENEOS"
THRESHOLD = 50  # bps
SIGNAL_HOUR = 9
SIGNAL_MIN  = 30


def fetch_vwap_dev():
    """9:00〜9:30の1分足からVWAPと乖離を計算"""
    today = date.today()
    conn = psycopg2.connect(**PG_CONFIG)
    q = f"""
        SELECT timestamp, open, high, low, close, volume
        FROM archive.intraday_data
        WHERE symbol = '{SYMBOL}'
          AND DATE(timestamp + INTERVAL '9 hours') = '{today}'
          AND close IS NOT NULL
        ORDER BY timestamp
    """
    df = pd.read_sql(q, conn)
    conn.close()

    if df.empty:
        return None

    df['jst'] = pd.to_datetime(df['timestamp']) + pd.Timedelta(hours=9)
    df = df.sort_values('jst').set_index('jst')

    # 9:00以降のデータのみ
    morning = df[df.index.hour >= 9]
    if morning.empty:
        return None

    # VWAP計算（累積）
    vol = morning['volume'].fillna(0)
    vol = vol.where(vol > 0, 1.0)
    cum_pv  = (morning['close'] * vol).cumsum()
    cum_vol = vol.cumsum()
    vwap_series = cum_pv / cum_vol

    # 9:30バーを厳密に要求（minute == 30 のバーが存在しない場合はスキップ）
    bar_930 = morning[
        (morning.index.hour == SIGNAL_HOUR) & (morning.index.minute == SIGNAL_MIN)
    ]
    if bar_930.empty:
        return None  # 9:30バー欠損 → 判定不能としてスキップ

    latest = bar_930.iloc[-1]
    close_930 = float(latest['close'])
    vwap_930  = float(vwap_series.loc[bar_930.index[-1]])
    dev_bps   = (close_930 / vwap_930 - 1) * 10000

    return {
        'time': bar_930.index[-1].strftime('%H:%M'),
        'close': close_930,
        'vwap': vwap_930,
        'dev_bps': dev_bps,
    }


def check_signal():
    print("=" * 65)
    print(f"ENEOS ({SYMBOL}) VWAP Trend シグナル判定")
    print(f"実行時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S JST')}")
    print("=" * 65)

    today = date.today()
    weekday = today.strftime('%A')
    print(f"\n本日: {today} ({weekday})")

    if today.weekday() >= 5:
        print("\n🚫 土日のためスキップ")
        return 0

    try:
        g = fetch_vwap_dev()
    except Exception as e:
        print(f"\n❌ データ取得エラー: {e}")
        return 1

    if g is None:
        print("\n❌ データ不足 (9:30バーが取得できない)")
        return 1

    print(f"\n[VWAP乖離 @ {g['time']}]")
    print(f"  現値:  ¥{g['close']:,.1f}")
    print(f"  VWAP:  ¥{g['vwap']:,.1f}")
    print(f"  乖離:  {g['dev_bps']:+.1f} bps  (閾値 ±{THRESHOLD}bps)")

    if g['dev_bps'] >= THRESHOLD:
        direction = "Long"
        signal = True
    elif g['dev_bps'] <= -THRESHOLD:
        direction = "Short"
        signal = True
    else:
        direction = None
        signal = False

    print(f"  判定:  {'✅ 通過 → ' + direction if signal else '❌ 不足 → スキップ'}")

    if not signal:
        print("\n" + "=" * 65)
        print("🚫 本日はシグナル不発 → スキップ")
        print("   skipped_reason=dev_below_threshold")
        print("=" * 65)
        return 0

    print("\n" + "=" * 65)
    print(f"🟢 シグナル発動 → {direction} エントリー")
    print("=" * 65)

    shares_1000 = int(10_000_000 / g['close'] / 100) * 100
    print(f"\n[発注目安 ¥1,000万の場合]")
    print(f"  株数: {shares_1000:,} 株 @ ¥{g['close']:,.1f}")
    print(f"  金額: ¥{shares_1000 * g['close']:,.0f}")

    print(f"\n[発注手順]")
    print(f"  1. 銘柄: 5020 ENEOS")
    print(f"  2. 売買: {'新規買い (Long)' if direction == 'Long' else '新規売り (Short)'}")
    print(f"  3. 種別: 成行")
    print(f"  4. 数量: [ポジションサイズ ÷ ¥{g['close']:,.0f}] 株")
    print(f"\n[決済]")
    print(f"  15:25-15:29: 引成 (CLO) で全量{'売り' if direction == 'Long' else '買い戻し'}")

    print(f"\n[確認事項]")
    print(f"  □ 本日の決算発表なし")
    print(f"  □ 日経225先物が急変動中でない (±3%以内)")

    return 0


if __name__ == "__main__":
    sys.exit(check_signal())
