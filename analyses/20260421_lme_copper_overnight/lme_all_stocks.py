"""
LME銅 → 日本株 全銘柄 オーバーナイト戦略検証
景気敏感株全般が反応する仮説を検証

戦略:
  LME銅のオープン(夏時間JST9:00/冬時間JST10:00)〜JST15:25の値動き幅が
  閾値を超えたら、同方向で日本株を15:30引けエントリー、翌朝9:00オープンでクローズ

全日本株(N>=100)でスクリーニング → ランキング
"""
import psycopg2
import pandas as pd
import numpy as np
from datetime import date, time as dtime

PG_CONFIG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}

START = "2025-04-01"
END = "2026-04-21"
COST_BPS = 4  # 往復

BST_PERIODS = [
    (date(2024, 3, 31), date(2024, 10, 27)),
    (date(2025, 3, 30), date(2025, 10, 26)),
    (date(2026, 3, 29), date(2026, 10, 25)),
]

# 銘柄名ラベル (主要銘柄のみ、他は symbol のまま表示)
SYMBOL_LABELS = {
    "5706.T": "三井金属", "5711.T": "三菱マテリアル", "5713.T": "住友金属鉱山",
    "5714.T": "DOWA", "5801.T": "古河電工", "5802.T": "住友電工", "5803.T": "フジクラ",
    "5401.T": "日本製鉄", "5411.T": "JFE", "5332.T": "TOTO",
    "8035.T": "TEL", "6857.T": "アドバンテスト", "6920.T": "レーザーテック",
    "6146.T": "ディスコ", "6723.T": "ルネサス", "4063.T": "信越化学",
    "7203.T": "トヨタ", "7201.T": "日産", "7267.T": "ホンダ", "7269.T": "スズキ", "7270.T": "SUBARU",
    "7011.T": "三菱重工", "7012.T": "川崎重工", "7013.T": "IHI",
    "8001.T": "伊藤忠", "8002.T": "丸紅", "8031.T": "三井物産", "8053.T": "住友商事", "8058.T": "三菱商事",
    "8306.T": "三菱UFJ", "8316.T": "三井住友FG", "8411.T": "みずほFG",
    "9501.T": "東京電力", "9502.T": "中部電力", "9503.T": "関西電力",
    "9984.T": "ソフトバンクG", "9983.T": "ファーストリテイリング",
    "6758.T": "ソニー", "7974.T": "任天堂",
    "6501.T": "日立", "6503.T": "三菱電機", "6502.T": "東芝",
    "1605.T": "INPEX", "5020.T": "ENEOS", "5016.T": "出光",
    "4502.T": "武田", "4503.T": "アステラス", "4523.T": "エーザイ",
    "6301.T": "コマツ", "6305.T": "日立建機", "6367.T": "ダイキン",
    "9432.T": "NTT", "9433.T": "KDDI", "9434.T": "SBG",
    "9020.T": "JR東日本", "9022.T": "JR東海", "9023.T": "JR西日本",
    "9101.T": "日本郵船", "9104.T": "商船三井", "9107.T": "川崎汽船",
    "7741.T": "HOYA", "6954.T": "ファナック", "6861.T": "キーエンス",
    "8801.T": "三井不動産", "8802.T": "三菱地所",
    "8267.T": "イオン", "8113.T": "ユニ・チャーム",
    "4661.T": "OLC", "6273.T": "SMC",
    "6098.T": "リクルート",
    "2502.T": "アサヒ", "2503.T": "キリン", "2801.T": "キッコーマン", "2802.T": "味の素",
}


def is_bst(d):
    for start, end in BST_PERIODS:
        if start <= d < end:
            return True
    return False


def load_lme_signals():
    conn = psycopg2.connect(**PG_CONFIG)
    q = f"""SELECT timestamp, open, close FROM intraday_data
            WHERE symbol='CMCU3' AND timestamp >= '{START}' AND timestamp < '{END}'
            ORDER BY timestamp"""
    df = pd.read_sql(q, conn)
    conn.close()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['jst'] = df['timestamp'] + pd.Timedelta(hours=9)
    df = df.dropna(subset=['close']).set_index('jst').sort_index()

    signals = []
    for d in sorted(set(df.index.date)):
        if d.weekday() >= 5:
            continue
        open_hour = 9 if is_bst(d) else 10
        open_target = pd.Timestamp.combine(d, dtime(open_hour, 0))
        close_target = pd.Timestamp.combine(d, dtime(15, 25))
        day = df[df.index.date == d]
        if len(day) == 0:
            continue
        after = day[day.index >= open_target]
        if len(after) == 0:
            continue
        ob = after.iloc[0]
        if (ob.name - open_target).total_seconds() > 1800:
            continue
        before = day[day.index <= close_target]
        if len(before) == 0:
            continue
        cb = before.iloc[-1]
        if (close_target - cb.name).total_seconds() > 1800:
            continue
        signals.append({
            'date': d, 'move_pct': (cb['close'] / ob['open'] - 1) * 100
        })
    return pd.DataFrame(signals).set_index('date')


def load_all_jp_daily():
    """全日本株の日次 close(15:30直前) / open(9:00) を一括取得"""
    conn = psycopg2.connect(**PG_CONFIG)
    # シンボルリスト取得
    cur = conn.cursor()
    cur.execute(f"""SELECT symbol FROM intraday_data
                    WHERE symbol LIKE '%.T' AND timestamp >= '{START}'
                    GROUP BY symbol HAVING COUNT(*) > 50000""")
    symbols = [r[0] for r in cur.fetchall()]

    result = {}
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
        if len(df) == 0:
            continue
        df_date = df.index.date
        daily = []
        for d in sorted(set(df_date)):
            gd = df[df_date == d]
            h2, m2 = gd.index.hour, gd.index.minute
            closes = gd[(h2 == 15) & (m2 >= 20)]
            opens = gd[(h2 == 9) & (m2 <= 5)]
            if len(closes) == 0 or len(opens) == 0:
                continue
            daily.append({'date': d, 'jp_close': closes['close'].iloc[-1],
                          'jp_open': opens['open'].iloc[0]})
        if daily:
            result[sym] = pd.DataFrame(daily).set_index('date')
    conn.close()
    return result


def backtest(signals, jp_daily, threshold_pct):
    trades = []
    jp_dates = sorted(jp_daily.index)
    for i, d in enumerate(jp_dates[:-1]):
        if d not in signals.index:
            continue
        m = signals.loc[d, 'move_pct']
        if abs(m) < threshold_pct:
            continue
        direction = np.sign(m)
        entry = jp_daily.loc[d, 'jp_close']
        next_d = jp_dates[i + 1]
        exit_p = jp_daily.loc[next_d, 'jp_open']
        gross = (exit_p / entry - 1) * direction * 10000
        trades.append({
            'entry_date': d, 'exit_date': next_d, 'lme_move_pct': m,
            'direction': int(direction), 'pnl_bps': gross - COST_BPS, 'gross_bps': gross,
        })
    return pd.DataFrame(trades)


def evaluate(tdf):
    if len(tdf) == 0:
        return None
    arr = tdf['pnl_bps'].values
    wr = (arr > 0).mean() * 100
    pos = arr[arr > 0].sum()
    neg = abs(arr[arr <= 0].sum())
    pf = pos / neg if neg > 0 else np.inf
    mean = arr.mean()
    total = arr.sum()
    sharpe = mean / arr.std() * np.sqrt(252) if arr.std() > 0 else 0
    return {'n': len(tdf), 'wr': wr, 'pf': pf, 'mean_bps': mean, 'total_bps': total, 'sharpe': sharpe}


def lbl(sym):
    return SYMBOL_LABELS.get(sym, '')


def main():
    print("=" * 110)
    print("LME銅 → 日本株 全銘柄スクリーニング")
    print(f"期間: {START} 〜 {END} / コスト往復{COST_BPS}bps")
    print("=" * 110)

    print("\n[1] LMEシグナル計算")
    signals = load_lme_signals()
    print(f"  有効シグナル日数: {len(signals)}")

    print("\n[2] 全日本株データロード")
    jp_data = load_all_jp_daily()
    print(f"  対象銘柄数: {len(jp_data)}")

    # 閾値別ランキング
    for threshold, min_n in [(0.5, 15), (0.8, 12), (1.0, 10), (1.5, 5)]:
        n_sig = (signals['move_pct'].abs() >= threshold).sum()
        print(f"\n\n{'=' * 110}")
        print(f"[3] threshold = {threshold}% (シグナル日数: {n_sig})")
        print("=" * 110)

        results = []
        for sym, jp_daily in jp_data.items():
            tdf = backtest(signals, jp_daily, threshold)
            r = evaluate(tdf)
            if r and r['n'] >= min_n:
                # 方向別
                long_r = evaluate(tdf[tdf.direction == 1])
                short_r = evaluate(tdf[tdf.direction == -1])
                results.append({
                    'symbol': sym, 'name': lbl(sym),
                    **r,
                    'long_n': long_r['n'] if long_r else 0,
                    'long_wr': long_r['wr'] if long_r else 0,
                    'long_pf': long_r['pf'] if long_r else 0,
                    'long_mean': long_r['mean_bps'] if long_r else 0,
                    'short_n': short_r['n'] if short_r else 0,
                    'short_wr': short_r['wr'] if short_r else 0,
                    'short_pf': short_r['pf'] if short_r else 0,
                    'short_mean': short_r['mean_bps'] if short_r else 0,
                })

        rdf = pd.DataFrame(results)
        if len(rdf) == 0:
            print("  (該当銘柄なし)")
            continue
        rdf_sorted = rdf.sort_values('sharpe', ascending=False)

        # Top 20 by Sharpe
        print(f"\n-- 両方向 (Long+Short) Sharpe Top 20 --")
        print(f"  {'Sym':<8} {'Name':<15} {'N':>4} {'WR':>6} {'PF':>5} {'Mean':>7} {'Total':>8} {'Shp':>5}")
        for _, r in rdf_sorted.head(20).iterrows():
            print(f"  {r['symbol']:<8} {r['name']:<15} {r['n']:>4} {r['wr']:>5.1f}% {r['pf']:>5.2f} "
                  f"{r['mean_bps']:>+6.1f} {r['total_bps']:>+7.0f} {r['sharpe']:>+4.2f}")

        print(f"\n-- 両方向 Sharpe Bottom 10 (逆張り候補) --")
        print(f"  {'Sym':<8} {'Name':<15} {'N':>4} {'WR':>6} {'PF':>5} {'Mean':>7} {'Total':>8} {'Shp':>5}")
        for _, r in rdf_sorted.tail(10).iterrows():
            print(f"  {r['symbol']:<8} {r['name']:<15} {r['n']:>4} {r['wr']:>5.1f}% {r['pf']:>5.2f} "
                  f"{r['mean_bps']:>+6.1f} {r['total_bps']:>+7.0f} {r['sharpe']:>+4.2f}")

        # Long-only ranking (Long>=10 trades)
        long_df = rdf[rdf.long_n >= 10].copy()
        long_df['long_sharpe'] = long_df['long_mean'] / 100  # 簡易
        long_df = long_df.sort_values('long_mean', ascending=False)
        print(f"\n-- Long only Mean Top 15 (Long N>=10) --")
        print(f"  {'Sym':<8} {'Name':<15} {'N':>4} {'WR':>6} {'PF':>5} {'Mean':>7}")
        for _, r in long_df.head(15).iterrows():
            print(f"  {r['symbol']:<8} {r['name']:<15} {int(r['long_n']):>4} {r['long_wr']:>5.1f}% {r['long_pf']:>5.2f} "
                  f"{r['long_mean']:>+6.1f}")

    # STEP4: セクター別平均 (th=1.0%)
    print(f"\n\n{'=' * 110}")
    print("[4] セクター別平均 (threshold=1.0%)")
    print("=" * 110)
    SECTORS = {
        '非鉄金属': ['5706.T', '5711.T', '5713.T', '5714.T'],
        '電線': ['5801.T', '5802.T', '5803.T'],
        '鉄鋼': ['5401.T', '5411.T'],
        '商社': ['8001.T', '8002.T', '8031.T', '8053.T', '8058.T'],
        '機械': ['6301.T', '6305.T', '6367.T', '6954.T'],
        '自動車': ['7203.T', '7201.T', '7267.T', '7269.T', '7270.T'],
        '重工': ['7011.T', '7012.T', '7013.T'],
        '銀行': ['8306.T', '8316.T', '8411.T'],
        '半導体': ['8035.T', '6857.T', '6920.T', '6146.T', '6723.T', '4063.T'],
        '海運': ['9101.T', '9104.T', '9107.T'],
        'エネルギー': ['1605.T', '5020.T', '5016.T'],
        'ディフェンシブ(医薬)': ['4502.T', '4503.T', '4523.T'],
        '通信': ['9432.T', '9433.T'],
        '電力': ['9501.T', '9502.T', '9503.T'],
    }
    print(f"  {'Sector':<25} {'N銘柄':>6} {'Avg Sharpe':>11} {'Avg Mean':>10} {'Avg WR':>8} {'Avg PF':>8}")
    for sector, syms in SECTORS.items():
        rs = []
        for sym in syms:
            if sym in jp_data:
                tdf = backtest(signals, jp_data[sym], 1.0)
                r = evaluate(tdf)
                if r:
                    rs.append(r)
        if rs:
            df = pd.DataFrame(rs)
            print(f"  {sector:<25} {len(rs):>6} {df['sharpe'].mean():>+10.2f} "
                  f"{df['mean_bps'].mean():>+9.1f} {df['wr'].mean():>7.1f}% {df['pf'].mean():>7.2f}")


if __name__ == "__main__":
    main()
