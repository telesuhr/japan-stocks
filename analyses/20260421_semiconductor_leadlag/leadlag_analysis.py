"""
半導体セクター リードラグ分析 + バックテスト
対象: 8035.T, 6857.T, 6920.T, 6146.T, 6723.T, 4063.T
期間: 直近1年 (2025-04-21 ~ 2026-04-21)
"""
import psycopg2
import pandas as pd
import numpy as np
from itertools import product

PG_CONFIG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}

SYMBOLS = {
    "8035.T": "TEL",
    "6857.T": "Advantest",
    "6920.T": "Lasertec",
    "6146.T": "DISCO",
    "6723.T": "Renesas",
    "4063.T": "ShinEtsu",
}
START = "2025-04-21"
END = "2026-04-21"


def load_data():
    conn = psycopg2.connect(**PG_CONFIG)
    frames = {}
    for sym in SYMBOLS:
        q = f"""SELECT timestamp, open, high, low, close, volume
                FROM intraday_data
                WHERE symbol='{sym}' AND timestamp >= '{START}' AND timestamp < '{END}'
                ORDER BY timestamp"""
        df = pd.read_sql(q, conn)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['jst'] = df['timestamp'] + pd.Timedelta(hours=9)
        df = df.dropna(subset=['open']).set_index('jst').sort_index()
        # 取引時間帯のみ (9:00-11:30, 12:30-15:30)
        h = df.index.hour
        m = df.index.minute
        morning = (h == 9) | (h == 10) | ((h == 11) & (m <= 30))
        afternoon = ((h == 12) & (m >= 30)) | (h == 13) | (h == 14) | ((h == 15) & (m <= 30))
        df = df[morning | afternoon]
        frames[sym] = df
        print(f"{sym} ({SYMBOLS[sym]}): {len(df):,} rows, {df.index.min()} ~ {df.index.max()}")
    conn.close()
    return frames


def build_returns_panel(frames):
    """各銘柄の1分足リターンを共通タイムインデックスで結合"""
    rets = {}
    for sym, df in frames.items():
        r = df['close'].pct_change()
        rets[sym] = r
    panel = pd.concat(rets, axis=1)
    panel.columns = list(SYMBOLS.keys())
    # 全銘柄データある行のみ
    panel = panel.dropna()
    print(f"\nパネル: {len(panel):,} rows, {panel.index.min()} ~ {panel.index.max()}")
    return panel


def leadlag_matrix(panel, lags=range(-5, 6)):
    """ラグ別クロス相関 corr(X_t, Y_{t+lag})
    lag>0: Xがリード, Y追随"""
    syms = list(panel.columns)
    results = []
    for x in syms:
        for y in syms:
            if x == y:
                continue
            for lag in lags:
                if lag == 0:
                    c = panel[x].corr(panel[y])
                else:
                    c = panel[x].corr(panel[y].shift(-lag))
                results.append({'leader': x, 'follower': y, 'lag': lag, 'corr': c})
    return pd.DataFrame(results)


def summarize_leadlag(ll_df):
    """各ペアで最も相関が強いラグを抽出"""
    syms = list(SYMBOLS.keys())
    print("\n=== ペア別 最適ラグ (|corr|最大) ===")
    print(f"{'Leader':>10} {'Follower':>10} {'Lag':>5} {'Corr':>8} {'Corr@0':>8}")
    rows = []
    for x in syms:
        for y in syms:
            if x == y:
                continue
            sub = ll_df[(ll_df.leader == x) & (ll_df.follower == y)]
            best = sub.loc[sub['corr'].abs().idxmax()]
            c0 = sub[sub.lag == 0]['corr'].values[0]
            rows.append({'leader': x, 'follower': y, 'best_lag': int(best.lag), 'best_corr': best['corr'], 'corr_lag0': c0})
    summary = pd.DataFrame(rows)
    # リード強度: lag>0で相関が最大の銘柄がリーダー
    for x in syms:
        pos_lag = summary[(summary.leader == x) & (summary.best_lag > 0)]
        neg_lag = summary[(summary.leader == x) & (summary.best_lag < 0)]
        print(f"{x}({SYMBOLS[x]}): lead={len(pos_lag)} pairs, lag={len(neg_lag)} pairs, "
              f"mean_lag={summary[summary.leader==x].best_lag.mean():+.2f}")
    return summary


def leadlag_by_session(panel):
    """前場 vs 後場でリード関係変化"""
    h = panel.index.hour
    morning = panel[(h >= 9) & (h < 12)]
    afternoon = panel[(h >= 12) & (h < 16)]
    print(f"\n=== 前場 ({len(morning):,} rows) リードラグ ===")
    m_ll = leadlag_matrix(morning, lags=range(-3, 4))
    m_sum = summarize_pair_leaders(m_ll)
    print(f"\n=== 後場 ({len(afternoon):,} rows) リードラグ ===")
    a_ll = leadlag_matrix(afternoon, lags=range(-3, 4))
    a_sum = summarize_pair_leaders(a_ll)
    return m_sum, a_sum


def summarize_pair_leaders(ll_df):
    syms = list(SYMBOLS.keys())
    # ラグ1での相関マトリクス
    lag1 = ll_df[ll_df.lag == 1].pivot(index='leader', columns='follower', values='corr')
    print("\nCorr @ lag=1 (行=Leader, 列=Follower, t+1):")
    print(lag1.round(3).to_string())
    return lag1


def backtest_following(panel, leader, follower, signal_threshold=0.003, hold_minutes=3, cost_bps=2):
    """リード銘柄のリターンがthresholdを超えたら、follower銘柄を同方向に追随
    hold_minutes分保持して決済"""
    lead_r = panel[leader]
    fol = panel[follower]

    # シグナル: leaderの1分リターン
    signal = lead_r.copy()

    trades = []
    i = 0
    lead_arr = signal.values
    n = len(signal)
    idx = signal.index

    # follower価格系列を元データから引くのは大変なのでリターンから復元
    # followerのt+1 ~ t+hold_minutesの累積リターン
    fol_future = fol.shift(-1).rolling(window=hold_minutes).sum().shift(-(hold_minutes - 1))

    while i < n - hold_minutes - 1:
        s = lead_arr[i]
        if abs(s) >= signal_threshold:
            # 同じセッション内かチェック（日をまたがない）
            t = idx[i]
            t_exit_idx = i + hold_minutes
            if t_exit_idx >= n:
                break
            t_exit = idx[t_exit_idx]
            # 同日同セッションか
            if t.date() == t_exit.date():
                direction = np.sign(s)
                # follower t+1からhold期間累積
                fut_ret = panel[follower].iloc[i+1:i+1+hold_minutes].sum()
                pnl_bps = direction * fut_ret * 10000 - cost_bps * 2  # 往復
                trades.append({
                    'entry_time': t,
                    'leader_ret': s,
                    'direction': direction,
                    'fol_ret': fut_ret,
                    'pnl_bps': pnl_bps,
                })
                i = i + hold_minutes  # 重複エントリー回避
                continue
        i += 1

    if not trades:
        return None
    tdf = pd.DataFrame(trades)
    return tdf


def evaluate_trades(tdf, label):
    if tdf is None or len(tdf) == 0:
        print(f"{label}: トレードなし")
        return None
    arr = tdf['pnl_bps'].values
    wr = (arr > 0).mean() * 100
    pos = arr[arr > 0].sum()
    neg = abs(arr[arr <= 0].sum())
    pf = pos / neg if neg > 0 else np.inf
    total = arr.sum()
    mean = arr.mean()
    sharpe = mean / arr.std() * np.sqrt(252) if arr.std() > 0 else 0
    print(f"{label}: N={len(tdf):>5} | WR={wr:5.1f}% | PF={pf:4.2f} | mean={mean:+6.2f}bps | total={total:+8.0f}bps | Sharpe={sharpe:+.2f}")
    return {'label': label, 'n': len(tdf), 'wr': wr, 'pf': pf, 'mean_bps': mean, 'total_bps': total, 'sharpe': sharpe}


def main():
    print("=" * 80)
    print("半導体セクター リードラグ分析")
    print("=" * 80)

    frames = load_data()
    panel = build_returns_panel(frames)

    # STEP 1: 全期間ラグ別相関
    print("\n" + "=" * 80)
    print("STEP 1: 全期間 クロス相関 (ラグ -5 ~ +5分)")
    print("=" * 80)
    ll = leadlag_matrix(panel)
    summary = summarize_leadlag(ll)

    # lag=1 相関マトリクス
    lag1 = ll[ll.lag == 1].pivot(index='leader', columns='follower', values='corr')
    print("\n全期間 Corr @ lag=1 (行=Leader t, 列=Follower t+1):")
    print(lag1.round(4).to_string())

    # 行平均（どの銘柄が他をよくリードするか）
    print("\nリード強度 (lag=1 行平均):")
    print(lag1.mean(axis=1).sort_values(ascending=False).round(4).to_string())

    # STEP 2: セッション別
    print("\n" + "=" * 80)
    print("STEP 2: セッション別リードラグ")
    print("=" * 80)
    leadlag_by_session(panel)

    # STEP 3: バックテスト
    print("\n" + "=" * 80)
    print("STEP 3: 追随戦略バックテスト (コスト2bps/片側, 4bps/往復)")
    print("=" * 80)

    # lag=1相関が高いペアを選定 (対角除く)
    lag1_masked = lag1.copy()
    np.fill_diagonal(lag1_masked.values, np.nan)
    top_pairs = []
    stacked = lag1_masked.stack().sort_values(ascending=False)
    print("\n上位5ペア (lag=1 相関):")
    for (lead, fol), c in stacked.head(5).items():
        print(f"  {lead}({SYMBOLS[lead]}) -> {fol}({SYMBOLS[fol]}): {c:.4f}")
        top_pairs.append((lead, fol))

    # パラメータ感応度
    print("\n--- パラメータ感応度 (上位3ペア, threshold x hold_min) ---")
    results = []
    for lead, fol in top_pairs[:3]:
        print(f"\n[{lead}({SYMBOLS[lead]}) -> {fol}({SYMBOLS[fol]})]")
        for th in [0.002, 0.003, 0.005, 0.008]:
            for hm in [2, 3, 5]:
                tdf = backtest_following(panel, lead, fol, signal_threshold=th, hold_minutes=hm)
                r = evaluate_trades(tdf, f"  th={th:.3f} hm={hm}")
                if r:
                    r['pair'] = f"{lead}->{fol}"
                    r['threshold'] = th
                    r['hold'] = hm
                    results.append(r)

    # ベストパラメータ
    if results:
        rdf = pd.DataFrame(results)
        # N>=50 かつ PF>1 のもので総利益順
        valid = rdf[rdf.n >= 50]
        if len(valid) > 0:
            best = valid.sort_values('total_bps', ascending=False).head(10)
            print("\n=== Top 10 Best (N>=50) ===")
            print(best[['pair', 'threshold', 'hold', 'n', 'wr', 'pf', 'mean_bps', 'total_bps', 'sharpe']].to_string(index=False))


if __name__ == "__main__":
    main()
