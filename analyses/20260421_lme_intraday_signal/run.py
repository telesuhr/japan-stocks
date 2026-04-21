"""
LME銅 東京時間変化率 閾値 0.3% → オーバーナイト戦略バックテスト

lme_on_copper と同じ構造:
  シグナル: LME銅 東京オープン → 15:25 の変化率 ≥ threshold
  エントリー: Day N 15:30 引成 Long (Core5 / 非鉄バスケット)
  決済:       Day N+1 09:00 寄成

目的: 閾値0.3% (他Claude提唱) が有効かどうかを検証
比較: 0.3% / 0.5% / 0.8% / 1.0% (既存採用値)
"""
import psycopg2
import pandas as pd
import numpy as np
from datetime import date, time as dtime
import warnings
warnings.filterwarnings("ignore")

PG_CONFIG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}

START = "2025-04-01"
END   = "2026-04-21"
COST_BPS = 4
OUTLIER_PCT = 15.0

BST_PERIODS = [
    (date(2024, 3, 31), date(2024, 10, 27)),
    (date(2025, 3, 30), date(2025, 10, 26)),
    (date(2026, 3, 29), date(2026, 10, 25)),
]

CORE5 = {"5711.T": "三菱マテリアル", "6501.T": "日立",
         "7011.T": "三菱重工",    "5016.T": "出光", "4502.T": "武田"}
NONFER = {"5711.T": "三菱マテリアル", "5713.T": "住友金属鉱山",
          "5706.T": "三井金属",    "5714.T": "DOWA",
          "5801.T": "古河電工",    "5802.T": "住友電工"}
ALL_STOCKS = {**CORE5, **NONFER}
THRESHOLDS = [0.3, 0.5, 0.8, 1.0]


def is_bst(d):
    for s, e in BST_PERIODS:
        if s <= d < e:
            return True
    return False


def load_lme_signals(exclude_thursday=True):
    conn = psycopg2.connect(**PG_CONFIG)
    q = f"""SELECT timestamp, open, close FROM intraday_data
            WHERE symbol='CMCU3' AND timestamp >= '{START}' AND timestamp < '{END}'
            ORDER BY timestamp"""
    df = pd.read_sql(q, conn); conn.close()
    df['jst'] = pd.to_datetime(df['timestamp']) + pd.Timedelta(hours=9)
    df = df.dropna(subset=['close']).set_index('jst').sort_index()
    signals = []
    for d in sorted(set(df.index.date)):
        if d.weekday() >= 5: continue
        if exclude_thursday and d.weekday() == 3: continue
        open_hour = 9 if is_bst(d) else 10
        ot = pd.Timestamp.combine(d, dtime(open_hour, 0))
        ct = pd.Timestamp.combine(d, dtime(15, 25))
        day = df[df.index.date == d]
        after = day[day.index >= ot]
        if after.empty or (after.index[0] - ot).total_seconds() > 1800: continue
        before = day[day.index <= ct]
        if before.empty or (ct - before.index[-1]).total_seconds() > 1800: continue
        signals.append({'date': d,
                        'move_pct': (before['close'].iloc[-1] / after['open'].iloc[0] - 1) * 100})
    return pd.DataFrame(signals).set_index('date')


def load_stock(sym):
    conn = psycopg2.connect(**PG_CONFIG)
    q = f"""SELECT timestamp, open, close FROM intraday_data
            WHERE symbol='{sym}' AND timestamp >= '{START}' AND timestamp < '{END}'
            ORDER BY timestamp"""
    df = pd.read_sql(q, conn); conn.close()
    df['jst'] = pd.to_datetime(df['timestamp']) + pd.Timedelta(hours=9)
    df = df.dropna(subset=['open','close']).set_index('jst').sort_index()
    h, m = df.index.hour, df.index.minute
    df = df[((h == 9) & (m <= 5)) | ((h == 15) & (m >= 20) & (m <= 30))]
    daily = []
    for d in sorted(set(df.index.date)):
        gd = df[df.index.date == d]
        h2, m2 = gd.index.hour, gd.index.minute
        closes = gd[(h2 == 15) & (m2 >= 20)]
        opens  = gd[(h2 == 9)  & (m2 <= 5)]
        if closes.empty or opens.empty: continue
        daily.append({'date': d, 'close15': closes['close'].iloc[-1],
                      'open9': opens['open'].iloc[0]})
    return pd.DataFrame(daily).set_index('date')


def backtest_single(signals, jp, threshold):
    trades = []
    dates = sorted(jp.index)
    for i, d in enumerate(dates[:-1]):
        if d not in signals.index: continue
        if signals.loc[d, 'move_pct'] < threshold: continue
        entry = jp.loc[d, 'close15']
        next_d = dates[i + 1]
        exit_p = jp.loc[next_d, 'open9']
        ret_pct = (exit_p / entry - 1) * 100
        if abs(ret_pct) > OUTLIER_PCT: continue
        trades.append({'date': d, 'pnl_bps': ret_pct * 100 - COST_BPS})
    return pd.DataFrame(trades)


def evaluate(tdf):
    if len(tdf) == 0: return None
    arr = tdf['pnl_bps'].values
    wr  = (arr > 0).mean() * 100
    pf  = arr[arr>0].sum() / abs(arr[arr<=0].sum()) if arr[arr<=0].sum() != 0 else np.inf
    mean = arr.mean(); std = arr.std()
    sharpe = mean / std * np.sqrt(252) if std > 0 else 0
    t_stat = mean / (std / np.sqrt(len(arr))) if std > 0 else 0
    return {'n': len(tdf), 'mean': mean, 'wr': wr, 'pf': pf,
            'sharpe': sharpe, 't_stat': t_stat, 'total': arr.sum()}


def basket_eval(signals, stock_data, syms, threshold):
    per_stock = {}
    for sym in syms:
        if sym not in stock_data: continue
        tdf = backtest_single(signals, stock_data[sym], threshold)
        if not tdf.empty:
            per_stock[sym] = tdf.set_index('date')['pnl_bps']
    common = None
    for s in per_stock.values():
        common = set(s.index) if common is None else common & set(s.index)
    if not common: return None
    basket = [{'date': d, 'pnl_bps': np.mean([per_stock[s][d] for s in per_stock if d in per_stock[s].index])}
              for d in sorted(common)]
    return evaluate(pd.DataFrame(basket))


def main():
    print("=" * 75)
    print("LME銅 東京時間変化率 → ONホールド戦略 閾値比較 (木曜除外)")
    print(f"期間: {START} 〜 {END}  コスト往復{COST_BPS}bps")
    print("=" * 75)

    signals = load_lme_signals(exclude_thursday=True)
    n_all = len(signals)
    print(f"\n有効シグナル日数 (木曜除外後): {n_all} 日")
    for th in THRESHOLDS:
        n = (signals['move_pct'] >= th).sum()
        print(f"  ≥{th:.1f}%: {n}日 (月{n/13*4:.1f}回)")

    print("\n[株式データロード中...]")
    stock_data = {}
    for sym, name in ALL_STOCKS.items():
        df = load_stock(sym)
        if not df.empty:
            stock_data[sym] = df
            print(f"  {sym} {name}: {len(df)}日")

    # ── Core5バスケット ──────────────────────────────────────────────────
    print(f"\n{'=' * 75}")
    print("▼ Core5 バスケット (等加重)")
    print(f"{'=' * 75}")
    print(f"  {'th':>5}  {'N':>4} {'Mean(bps)':>9} {'WR':>6} {'PF':>5} {'Sharpe':>7} {'t-stat':>7}")
    print(f"  {'-'*5}  {'-'*4} {'-'*9} {'-'*6} {'-'*5} {'-'*7} {'-'*7}")
    core5_rows = []
    for th in THRESHOLDS:
        r = basket_eval(signals, stock_data, list(CORE5.keys()), th)
        tag = " ← 既存採用" if th == 1.0 else (" ← 検証対象" if th == 0.3 else "")
        if r:
            core5_rows.append({'threshold': th, **r})
            print(f"  {th:>5.1f}  {r['n']:>4} {r['mean']:>+8.1f} {r['wr']:>5.1f}% "
                  f"{r['pf']:>5.2f} {r['sharpe']:>+7.2f} {r['t_stat']:>+7.2f}{tag}")
        else:
            print(f"  {th:>5.1f}  N/A{tag}")

    # ── 非鉄バスケット ───────────────────────────────────────────────────
    print(f"\n{'=' * 75}")
    print("▼ 非鉄6銘柄 バスケット (等加重)")
    print(f"{'=' * 75}")
    print(f"  {'th':>5}  {'N':>4} {'Mean(bps)':>9} {'WR':>6} {'PF':>5} {'Sharpe':>7} {'t-stat':>7}")
    print(f"  {'-'*5}  {'-'*4} {'-'*9} {'-'*6} {'-'*5} {'-'*7} {'-'*7}")
    for th in THRESHOLDS:
        r = basket_eval(signals, stock_data, list(NONFER.keys()), th)
        if r:
            print(f"  {th:>5.1f}  {r['n']:>4} {r['mean']:>+8.1f} {r['wr']:>5.1f}% "
                  f"{r['pf']:>5.2f} {r['sharpe']:>+7.2f} {r['t_stat']:>+7.2f}")
        else:
            print(f"  {th:>5.1f}  N/A")

    # ── 個別銘柄 閾値0.3% ─────────────────────────────────────────────
    print(f"\n{'=' * 75}")
    print("▼ 個別銘柄 (threshold=0.3%, 木曜除外)")
    print(f"{'=' * 75}")
    print(f"  {'Sym':<8} {'Name':<12} {'N':>4} {'Mean':>9} {'WR':>6} {'PF':>5} {'Sharpe':>7} {'t':>6}")
    for sym, name in ALL_STOCKS.items():
        if sym not in stock_data: continue
        tdf = backtest_single(signals, stock_data[sym], 0.3)
        r = evaluate(tdf)
        if r:
            print(f"  {sym:<8} {name:<12} {r['n']:>4} {r['mean']:>+8.1f} "
                  f"{r['wr']:>5.1f}% {r['pf']:>5.2f} {r['sharpe']:>+7.2f} {r['t_stat']:>+6.2f}")

    if core5_rows:
        pd.DataFrame(core5_rows).to_csv("core5_threshold_compare.csv", index=False)
    print("\n[出力] core5_threshold_compare.csv")


if __name__ == "__main__":
    main()
