"""共通データローダ"""
import psycopg2
import pandas as pd
import numpy as np

PG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}

NONFERROUS = [
    "5711.T",  # 三菱マテリアル
    "5706.T",  # 三井金属
    "5713.T",  # 住友金属鉱山
    "5714.T",  # DOWA
    "5016.T",  # JX 金属
    "5801.T",  # 古河電工
    "5802.T",  # 住友電工
    "5803.T",  # フジクラ
]
SEMI = ["8035.T", "6146.T", "6920.T", "6857.T", "6323.T",
        "6963.T", "6526.T", "3436.T", "6525.T"]
TELECOM = ["9434.T", "9984.T"]  # ソフトバンク / ソフトバンクグループ
ALL_SYMBOLS = NONFERROUS + SEMI + TELECOM

SECTOR = {s: "nonferrous" for s in NONFERROUS}
SECTOR.update({s: "semi" for s in SEMI})
SECTOR.update({s: "telecom" for s in TELECOM})


def load_symbol(symbol: str) -> pd.DataFrame:
    conn = psycopg2.connect(**PG)
    df = pd.read_sql(
        f"SELECT timestamp,open,high,low,close,volume FROM intraday_data "
        f"WHERE symbol='{symbol}' ORDER BY timestamp", conn)
    conn.close()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['jst'] = df['timestamp'] + pd.Timedelta(hours=9)
    df = df.dropna(subset=['open']).set_index('jst').sort_index()
    # 株式時間帯のみ (9:00-11:30, 12:30-15:30)
    h, m = df.index.hour, df.index.minute
    mask = (((h == 9)) | (h == 10) | ((h == 11) & (m <= 30)) |
            ((h == 12) & (m >= 30)) | (h == 13) | (h == 14) |
            ((h == 15) & (m <= 30)))
    return df[mask]


def load_all(symbols=ALL_SYMBOLS):
    return {s: load_symbol(s) for s in symbols}


def perf(pnl_bps: np.ndarray, label="") -> dict:
    """1トレードあたり bps の配列 → 統計"""
    a = np.asarray(pnl_bps, dtype=float)
    a = a[~np.isnan(a)]
    n = len(a)
    if n < 5:
        return {"label": label, "N": n, "mean": np.nan, "sharpe": np.nan,
                "t": np.nan, "wr": np.nan, "pf": np.nan, "sum": np.nan,
                "mdd": np.nan, "h1": np.nan, "h2": np.nan}
    mean = a.mean()
    sd = a.std(ddof=1)
    sharpe = mean / sd * np.sqrt(252) if sd > 0 else 0
    t = mean / (sd / np.sqrt(n)) if sd > 0 else 0
    wr = (a > 0).mean() * 100
    pos = a[a > 0].sum()
    neg = -a[a <= 0].sum()
    pf = pos / neg if neg > 0 else np.inf
    cum = np.cumsum(a)
    mdd = (np.maximum.accumulate(cum) - cum).max()
    mid = n // 2
    h1 = a[:mid].mean() / a[:mid].std(ddof=1) * np.sqrt(252) if a[:mid].std(ddof=1) > 0 else 0
    h2 = a[mid:].mean() / a[mid:].std(ddof=1) * np.sqrt(252) if a[mid:].std(ddof=1) > 0 else 0
    return {"label": label, "N": n, "mean": mean, "sharpe": sharpe, "t": t,
            "wr": wr, "pf": pf, "sum": a.sum(), "mdd": mdd, "h1": h1, "h2": h2}


def print_perf(rows):
    print(f"{'戦略':<40} {'N':>6} {'mean':>8} {'Sharpe':>8} {'t':>7} {'WR%':>6} "
          f"{'PF':>6} {'累計':>9} {'H1':>7} {'H2':>7}")
    print("-" * 120)
    for r in rows:
        if np.isnan(r["mean"]):
            print(f"{r['label']:<40} {r['N']:>6} (insufficient)")
            continue
        print(f"{r['label']:<40} {r['N']:>6} {r['mean']:>+8.1f} {r['sharpe']:>+8.2f} "
              f"{r['t']:>+7.2f} {r['wr']:>6.1f} {r['pf']:>6.2f} "
              f"{r['sum']:>+9.0f} {r['h1']:>+7.2f} {r['h2']:>+7.2f}")
