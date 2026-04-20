"""
株式分割・異常値検出スクリプト
各銘柄で |overnight return| > 20% のイベントを抽出 → 株式分割/無償配当の可能性
"""
import psycopg2
import pandas as pd
import numpy as np

PG_CONFIG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}
START = "2025-04-01"
END = "2026-04-21"


def detect_splits():
    conn = psycopg2.connect(**PG_CONFIG)
    cur = conn.cursor()
    cur.execute(f"""SELECT symbol FROM intraday_data
                    WHERE symbol LIKE '%.T' AND timestamp >= '{START}'
                    GROUP BY symbol HAVING COUNT(*) > 50000""")
    symbols = [r[0] for r in cur.fetchall()]

    print(f"検証対象: {len(symbols)}銘柄")
    print(f"{'Symbol':<10} {'Date':<12} {'Close':>10} {'NextOpen':>10} {'ON Ret':>8}")
    print("-" * 60)

    all_splits = {}
    for sym in symbols:
        q = f"""SELECT timestamp, open, close FROM intraday_data
                WHERE symbol='{sym}' AND timestamp >= '{START}' AND timestamp < '{END}'
                ORDER BY timestamp"""
        df = pd.read_sql(q, conn)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['jst'] = df['timestamp'] + pd.Timedelta(hours=9)
        df = df.dropna(subset=['open', 'close']).set_index('jst').sort_index()
        h, m = df.index.hour, df.index.minute
        df = df[((h == 9) & (m <= 5)) | ((h == 15) & (m >= 20) & (m <= 30))]

        daily = []
        for d in sorted(set(df.index.date)):
            gd = df[df.index.date == d]
            h2, m2 = gd.index.hour, gd.index.minute
            closes = gd[(h2 == 15) & (m2 >= 20)]
            opens = gd[(h2 == 9) & (m2 <= 5)]
            if len(closes) == 0 or len(opens) == 0:
                continue
            daily.append({'date': d, 'jp_close': closes['close'].iloc[-1],
                          'jp_open': opens['open'].iloc[0]})
        dd = pd.DataFrame(daily).set_index('date')
        if len(dd) < 2:
            continue
        # overnight return: 前日close → 当日open
        dd['overnight'] = dd['jp_open'] / dd['jp_close'].shift(1) - 1
        extreme = dd[dd['overnight'].abs() > 0.20]  # |return| > 20%
        if len(extreme) > 0:
            for d, row in extreme.iterrows():
                ret = row['overnight'] * 100
                prev_close = dd['jp_close'].shift(1).loc[d]
                print(f"{sym:<10} {str(d):<12} {prev_close:>10.1f} {row['jp_open']:>10.1f} {ret:>+7.1f}%")
                all_splits.setdefault(sym, []).append({'date': d, 'return': ret})

    conn.close()
    print(f"\n合計: {len(all_splits)} 銘柄で異常値検出")
    return all_splits


if __name__ == "__main__":
    detect_splits()
