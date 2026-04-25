"""
共通ユーティリティ: DB ロード, JSTタイムスタンプ, バックテスト補助

接続優先順位:
  1. ローカル PostgreSQL (localhost:5432)
  2. NAS MariaDB (100.92.181.92:3306) ← PGが落ちているときの自動フォールバック
"""
import pandas as pd
import numpy as np
from datetime import date, time as dtime

PG_CONFIG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}
NAS_CONFIG = {"host": "100.92.181.92", "port": 3306, "user": "rfnews",
              "password": "Bleach@924", "database": "refinitiv_news"}

START = "2025-04-01"
END = "2026-04-24"
OUTLIER_PCT = 15.0
COST_BPS = 4.0

BST_PERIODS = [
    (date(2024, 3, 31), date(2024, 10, 27)),
    (date(2025, 3, 30), date(2025, 10, 26)),
    (date(2026, 3, 29), date(2026, 10, 25)),
]


def is_bst(d):
    return any(s <= d < e for s, e in BST_PERIODS)


def _connect_pg():
    import psycopg2
    return psycopg2.connect(**PG_CONFIG)

def _connect_nas():
    import pymysql
    return pymysql.connect(**NAS_CONFIG)

def fetch_intraday(symbol, start=START, end=END):
    """ローカルPG → NAS MariaDB の順で接続を試みる"""
    q = (f"SELECT timestamp, open, high, low, close, volume FROM intraday_data "
         f"WHERE symbol='{symbol}' AND timestamp>='{start}' AND timestamp<'{end}' ORDER BY timestamp")
    try:
        conn = _connect_pg()
        df = pd.read_sql(q, conn); conn.close()
    except Exception:
        conn = _connect_nas()
        df = pd.read_sql(q, conn); conn.close()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['jst'] = df['timestamp'] + pd.Timedelta(hours=9)
    return df.set_index('jst').sort_index()


def load_jp_daily(symbol):
    """日本株の日次open/close(9:00寄りと15:20-30引け)"""
    df = fetch_intraday(symbol)
    df = df.dropna(subset=['open','close'])
    h, m = df.index.hour, df.index.minute
    df = df[((h==9)&(m<=5)) | ((h==15)&(m>=20)&(m<=30))]
    out = []
    for d in sorted(set(df.index.date)):
        gd = df[df.index.date == d]
        h2, m2 = gd.index.hour, gd.index.minute
        op = gd[(h2==9)&(m2<=5)]; cl = gd[(h2==15)&(m2>=20)]
        if len(op) == 0 or len(cl) == 0: continue
        out.append({'date': d, 'open': op['open'].iloc[0], 'close': cl['close'].iloc[-1]})
    return pd.DataFrame(out).set_index('date')


def compute_stats(arr, ann=252):
    arr = np.asarray(arr)
    if len(arr) == 0: return None
    m, s = arr.mean(), arr.std()
    cum = arr.cumsum()
    dd = (cum - np.maximum.accumulate(cum)).min()
    pos = arr[arr > 0].sum(); neg = abs(arr[arr <= 0].sum())
    return {
        'n': len(arr), 'mean': m, 'std': s, 'total': arr.sum(),
        'median': float(np.median(arr)), 'wr': (arr>0).mean()*100,
        'pf': pos/neg if neg > 0 else np.inf,
        'sharpe': m/s*np.sqrt(ann) if s > 0 else 0,
        'maxdd': dd, 't_stat': m/(s/np.sqrt(len(arr))) if s > 0 else 0,
    }


def print_stats(label, st):
    if st is None:
        print(f"{label}: N=0")
        return
    print(f"{label}: N={st['n']:>4} Mean={st['mean']:>+7.1f} Total={st['total']:>+7.0f} "
          f"WR={st['wr']:>5.1f}% PF={st['pf']:>5.2f} Sharpe={st['sharpe']:>+6.2f} "
          f"MaxDD={st['maxdd']:>+6.0f} t={st['t_stat']:>+5.2f}")


def matplotlib_jp():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = ['Hiragino Sans', 'Arial Unicode MS', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    return plt


CORE5 = [
    ("5711.T", "三菱マテリアル"),
    ("6501.T", "日立"),
    ("7011.T", "三菱重工"),
    ("5016.T", "出光"),
    ("4502.T", "武田"),
]

NONFERROUS = [
    ("5711.T", "三菱マテリアル"), ("5706.T", "三井金属"), ("5713.T", "住友金属鉱山"),
]

ENERGY = [
    ("1605.T", "INPEX"), ("5016.T", "出光"), ("5020.T", "ENEOS"),
]

SHIPPING = [
    ("9101.T", "日本郵船"), ("9104.T", "商船三井"), ("9107.T", "川崎汽船"),
]

SEMICON = [
    ("8035.T", "TEL"), ("6857.T", "アドバンテスト"), ("6146.T", "ディスコ"),
    ("4063.T", "信越化学"), ("6963.T", "ローム"),
]

DOMESTIC_SHORT = [
    ("8267.T", "イオン"), ("9020.T", "JR東日本"), ("7974.T", "任天堂"),
    ("6758.T", "ソニー"), ("8411.T", "みずほ"),
]
