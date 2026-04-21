"""
データ成果物生成スクリプト

目的:
  レビュー用にデータベースから以下を生成:
  - fingerprint.json: 各シンボルの範囲・件数・ハッシュ (完全データの同一性検証用)
  - sample/*.csv: 極小サンプル (5日×銘柄, 1MB以下/ファイル)
  - summary.md: 生成済みサンプルの概要

実行:
  python3 data/generate_artifacts.py
"""
import sys, os, json, hashlib
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'analyses', '20260421_common')))
import mdutil as U
import pandas as pd
from datetime import date

SYMS = [
    # 景気敏感コア5
    '5711.T', '6501.T', '7011.T', '5016.T', '4502.T',
    # 非鉄
    '5706.T', '5713.T',
    # 半導体
    '8035.T', '6857.T', '6146.T', '4063.T', '6963.T',
    # 海運
    '9101.T', '9104.T', '9107.T',
    # エネルギー
    '1605.T', '5020.T',
    # 内需Short基礎銘柄
    '8267.T', '9020.T', '7974.T', '6758.T', '8411.T',
    # 外部指標
    '.TOPX', 'JNIc1', 'CMCU3', 'CLc1', 'LCOc1',
]

SAMPLE_DAYS = 5  # サンプルに含める営業日数
SAMPLE_START = date(2026, 3, 16)  # Post期間開始
SAMPLE_END = date(2026, 3, 23)    # 1週間分


def fetch_full_stats(symbol):
    """DBから完全データの統計情報を取得"""
    import psycopg2
    conn = psycopg2.connect(**U.PG_CONFIG)
    cur = conn.cursor()
    cur.execute(
        "SELECT MIN(timestamp)::date, MAX(timestamp)::date, COUNT(*), "
        "       ROUND(AVG(close)::numeric, 2), ROUND(STDDEV(close)::numeric, 2) "
        "FROM intraday_data WHERE symbol=%s", (symbol,)
    )
    row = cur.fetchone()
    # ハッシュ計算: timestamp+close のペアのみで軽量チェックサム
    cur.execute(
        "SELECT timestamp, close FROM intraday_data WHERE symbol=%s "
        "ORDER BY timestamp", (symbol,)
    )
    h = hashlib.md5()
    n = 0
    for ts, cl in cur:
        if cl is None: continue
        h.update(f"{ts.isoformat()}|{float(cl):.4f}|".encode())
        n += 1
    conn.close()
    return {
        'start_date': str(row[0]),
        'end_date': str(row[1]),
        'row_count': int(row[2]),
        'valid_rows': n,
        'mean_close': float(row[3]) if row[3] else None,
        'std_close': float(row[4]) if row[4] else None,
        'md5_ts_close': h.hexdigest(),
    }


def export_sample(symbol, start, end, out_path):
    """指定期間の1分足をサンプルCSVに出力"""
    import psycopg2
    conn = psycopg2.connect(**U.PG_CONFIG)
    q = ("SELECT timestamp, open, high, low, close, volume FROM intraday_data "
         f"WHERE symbol=%s AND timestamp>=%s AND timestamp<%s ORDER BY timestamp")
    df = pd.read_sql(q, conn, params=(symbol, start, end))
    conn.close()
    if len(df) == 0:
        return 0
    df.to_csv(out_path, index=False)
    return len(df)


def main():
    here = os.path.dirname(__file__)
    fp_path = os.path.join(here, 'fingerprint.json')
    sample_dir = os.path.join(here, 'sample')
    os.makedirs(sample_dir, exist_ok=True)

    print("=== 1. fingerprint.json 生成 ===")
    fingerprints = {}
    for sym in SYMS:
        try:
            st = fetch_full_stats(sym)
            fingerprints[sym] = st
            print(f"  {sym}: {st['row_count']:>7d} rows, "
                  f"{st['start_date']} → {st['end_date']}, md5={st['md5_ts_close'][:12]}...")
        except Exception as e:
            print(f"  {sym}: ERROR {e}")
            fingerprints[sym] = {'error': str(e)}

    with open(fp_path, 'w') as f:
        json.dump({
            'generated_at': pd.Timestamp.now().isoformat(),
            'db': 'market_data.intraday_data',
            'timezone_note': 'timestamps are stored in UTC; add +9h for JST',
            'symbols': fingerprints,
        }, f, indent=2, ensure_ascii=False)
    print(f"  → {fp_path}")

    print(f"\n=== 2. サンプルCSV生成 ({SAMPLE_START} 〜 {SAMPLE_END}) ===")
    total_bytes = 0
    sample_summary = []
    for sym in SYMS:
        safe = sym.replace('.', '_').replace(':', '_')
        out = os.path.join(sample_dir, f"{safe}.csv")
        try:
            n = export_sample(sym, SAMPLE_START, SAMPLE_END, out)
            size = os.path.getsize(out) if os.path.exists(out) and n > 0 else 0
            total_bytes += size
            sample_summary.append({'symbol': sym, 'rows': n, 'bytes': size, 'file': os.path.basename(out)})
            print(f"  {sym}: {n:>5} rows, {size/1024:>5.1f} KB")
        except Exception as e:
            print(f"  {sym}: ERROR {e}")
            sample_summary.append({'symbol': sym, 'error': str(e)})
    print(f"\n総サイズ: {total_bytes/1024:.1f} KB")

    # サマリーMD
    summary_md = os.path.join(here, 'SAMPLE_SUMMARY.md')
    with open(summary_md, 'w') as f:
        f.write(f"# データサンプル サマリー\n\n")
        f.write(f"期間: **{SAMPLE_START} 〜 {SAMPLE_END}** (イラン戦争後Post期間の1週間)\n\n")
        f.write(f"用途: コードの型・ロジック検証用。統計分析は元プロジェクトの集計CSV/PNGを参照。\n\n")
        f.write(f"| シンボル | 行数 | サイズ | ファイル |\n|---|---|---|---|\n")
        for s in sample_summary:
            if 'error' in s:
                f.write(f"| {s['symbol']} | ERROR | - | {s['error'][:40]} |\n")
            else:
                f.write(f"| {s['symbol']} | {s['rows']} | {s['bytes']/1024:.1f} KB | `{s['file']}` |\n")
        f.write(f"\n**総サイズ: {total_bytes/1024:.1f} KB**\n\n")
        f.write(f"## 列定義\n")
        f.write(f"- `timestamp`: UTC時刻 (JSTは +9h)\n")
        f.write(f"- `open`, `high`, `low`, `close`: 1分足OHLC\n")
        f.write(f"- `volume`: 出来高 (指数/先物は NULL か 0 のケースあり)\n")
    print(f"  → {summary_md}")


if __name__ == "__main__":
    main()
