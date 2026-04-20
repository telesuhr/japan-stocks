"""
バイアスチェック: 株価上昇相場に便乗しているだけでないかを検証

検証項目:
  1. 全日ベースライン: 無条件オーバーナイトリターン (buy & hold for ON) の平均
  2. LME UP日 vs DOWN日の非対称性
  3. Long only vs Short only の勝率差
  4. シグナル vs ランダム日での比較 (プラセボテスト)
  5. LMEムーブの符号と日本株ON収益の相関
"""
import psycopg2
import pandas as pd
import numpy as np
from datetime import date, time as dtime

PG_CONFIG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}
START = "2025-04-01"
END = "2026-04-21"
OUTLIER_PCT = 15.0

BST_PERIODS = [
    (date(2024, 3, 31), date(2024, 10, 27)),
    (date(2025, 3, 30), date(2025, 10, 26)),
    (date(2026, 3, 29), date(2026, 10, 25)),
]


def is_bst(d):
    return any(s <= d < e for s, e in BST_PERIODS)


def load_lme_signals():
    conn = psycopg2.connect(**PG_CONFIG)
    q = f"SELECT timestamp, open, close FROM intraday_data WHERE symbol='CMCU3' AND timestamp >= '{START}' AND timestamp < '{END}' ORDER BY timestamp"
    df = pd.read_sql(q, conn); conn.close()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['jst'] = df['timestamp'] + pd.Timedelta(hours=9)
    df = df.dropna(subset=['close']).set_index('jst').sort_index()
    signals = []
    for d in sorted(set(df.index.date)):
        if d.weekday() >= 5: continue
        oh = 9 if is_bst(d) else 10
        ot = pd.Timestamp.combine(d, dtime(oh, 0))
        ct = pd.Timestamp.combine(d, dtime(15, 25))
        day = df[df.index.date == d]
        if len(day) == 0: continue
        after = day[day.index >= ot]
        if len(after) == 0: continue
        ob = after.iloc[0]
        if (ob.name - ot).total_seconds() > 1800: continue
        before = day[day.index <= ct]
        if len(before) == 0: continue
        cb = before.iloc[-1]
        if (ct - cb.name).total_seconds() > 1800: continue
        signals.append({'date': d, 'move_pct': (cb['close']/ob['open']-1)*100})
    return pd.DataFrame(signals).set_index('date')


def load_jp_daily(sym):
    conn = psycopg2.connect(**PG_CONFIG)
    q = f"SELECT timestamp, open, close FROM intraday_data WHERE symbol='{sym}' AND timestamp >= '{START}' AND timestamp < '{END}' ORDER BY timestamp"
    df = pd.read_sql(q, conn); conn.close()
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
        if len(closes) == 0 or len(opens) == 0: continue
        daily.append({'date': d, 'jp_close': closes['close'].iloc[-1], 'jp_open': opens['open'].iloc[0]})
    return pd.DataFrame(daily).set_index('date')


def compute_overnight(jp):
    """jp_close(day t) → jp_open(day t+1) のリターン(%)"""
    dates = sorted(jp.index)
    records = []
    for i in range(len(dates)-1):
        d = dates[i]; nd = dates[i+1]
        entry = jp.loc[d, 'jp_close']
        exit_ = jp.loc[nd, 'jp_open']
        ret = (exit_/entry - 1) * 100
        if abs(ret) > OUTLIER_PCT: continue
        records.append({'entry_date': d, 'exit_date': nd, 'on_ret_pct': ret})
    return pd.DataFrame(records).set_index('entry_date')


def stats(arr):
    if len(arr) == 0:
        return None
    return {'n': len(arr), 'mean': arr.mean(), 'median': np.median(arr),
            'std': arr.std(), 'wr': (arr > 0).mean() * 100, 'sum': arr.sum(),
            't': arr.mean() / (arr.std() / np.sqrt(len(arr))) if arr.std() > 0 else 0}


def main():
    print("=" * 100)
    print("バイアスチェック: LMEシグナルは本物か、上昇相場便乗か")
    print("=" * 100)

    sig = load_lme_signals()
    print(f"\nLMEシグナル統計:")
    print(f"  全期間: N={len(sig)}, 平均move={sig['move_pct'].mean():+.3f}%, median={sig['move_pct'].median():+.3f}%")
    print(f"  UP日数 (move>0):  {(sig['move_pct']>0).sum()} ({(sig['move_pct']>0).mean()*100:.1f}%)")
    print(f"  DOWN日数 (move<0): {(sig['move_pct']<0).sum()} ({(sig['move_pct']<0).mean()*100:.1f}%)")
    print(f"  UP mean:   {sig[sig.move_pct>0]['move_pct'].mean():+.3f}%")
    print(f"  DOWN mean: {sig[sig.move_pct<0]['move_pct'].mean():+.3f}%")

    # LME銅そのもののトレンド
    print(f"\nLME銅の累積変化 (期間内の日次moveの合計): {sig['move_pct'].sum():+.1f}%")

    # 代表的な上位銘柄でチェック
    TARGETS = [
        ('5711.T', '三菱マテリアル'),
        ('5706.T', '三井金属'),
        ('6501.T', '日立'),
        ('6857.T', 'アドバンテスト'),
        ('8035.T', 'TEL'),
        ('4502.T', '武田'),
        ('7011.T', '三菱重工'),
        ('5016.T', '出光'),
        ('1605.T', 'INPEX'),
        ('9101.T', '日本郵船'),
    ]

    print("\n" + "=" * 100)
    print("各銘柄 バイアス検証 (th=1.0%, ONリターン%単位)")
    print("=" * 100)
    print(f"{'Symbol':<18} {'BaseN':>5} {'BaseMean':>9} {'BaseWR':>7} | "
          f"{'LongN':>5} {'LongMn':>7} {'LongWR':>7} {'t':>5} | "
          f"{'ShortN':>6} {'ShortMn':>8} {'ShortWR':>7} {'t':>5}")
    print("-" * 140)

    for sym, name in TARGETS:
        jp = load_jp_daily(sym)
        on = compute_overnight(jp)
        # ベースライン: 全日ON
        base = on['on_ret_pct'].values
        b = stats(base)

        # シグナル日 th=1.0%
        sig_days_up = sig[sig.move_pct >= 1.0].index
        sig_days_dn = sig[sig.move_pct <= -1.0].index
        long_ret = on.loc[on.index.intersection(sig_days_up)]['on_ret_pct'].values
        # Short: LMEダウン日に空売り → ON収益のマイナス = ショート利益
        short_ret_raw = on.loc[on.index.intersection(sig_days_dn)]['on_ret_pct'].values
        short_ret = -short_ret_raw  # ショート視点

        l = stats(long_ret)
        s = stats(short_ret)

        def fmt(d, keys):
            return " ".join([f"{d[k]:>+6.3f}" if isinstance(d[k], float) else f"{d[k]:>5}" for k in keys])

        label = f"{sym}{name[:8]}"
        line = f"{label:<18}"
        line += f" {b['n']:>5} {b['mean']:>+8.3f}% {b['wr']:>6.1f}% |"
        if l:
            line += f" {l['n']:>5} {l['mean']:>+6.3f}% {l['wr']:>6.1f}% {l['t']:>+5.2f} |"
        else:
            line += f" {'--':>5} {'--':>7} {'--':>7} {'--':>5} |"
        if s:
            line += f" {s['n']:>6} {s['mean']:>+7.3f}% {s['wr']:>6.1f}% {s['t']:>+5.2f}"
        else:
            line += f" {'--':>6} {'--':>8} {'--':>7} {'--':>5}"
        print(line)

    # 集約検証: ベースラインvsシグナル日
    print("\n" + "=" * 100)
    print("集約統計 (10銘柄平均) — シグナル日 vs 非シグナル日")
    print("=" * 100)

    all_base = []
    all_sig_long = []      # LMEアップ日の生ON (マッチ方向=Long)
    all_sig_short = []     # LMEダウン日の生ON
    all_nonsig = []        # |move|<1.0% の日の生ON

    for sym, name in TARGETS:
        jp = load_jp_daily(sym)
        on = compute_overnight(jp)
        all_base.append(on['on_ret_pct'].values)

        sig_up = sig[sig.move_pct >= 1.0].index
        sig_dn = sig[sig.move_pct <= -1.0].index
        non_sig = sig[sig.move_pct.abs() < 1.0].index

        all_sig_long.append(on.loc[on.index.intersection(sig_up)]['on_ret_pct'].values)
        all_sig_short.append(on.loc[on.index.intersection(sig_dn)]['on_ret_pct'].values)
        all_nonsig.append(on.loc[on.index.intersection(non_sig)]['on_ret_pct'].values)

    base = np.concatenate(all_base)
    sl = np.concatenate(all_sig_long)
    ss = np.concatenate(all_sig_short)
    ns = np.concatenate(all_nonsig)

    print(f"\n全トレード合算 (10銘柄):")
    print(f"  {'Bucket':<30} {'N':>6} {'Mean%':>8} {'Median%':>9} {'WR%':>6} {'t-stat':>7}")
    for lbl, arr in [('全日 baseline ON', base),
                     ('LMEアップ日 (>=+1%) ON', sl),
                     ('LMEダウン日 (<=-1%) ON', ss),
                     ('非シグナル日 (|m|<1%) ON', ns)]:
        st = stats(arr)
        if st:
            print(f"  {lbl:<30} {st['n']:>6} {st['mean']:>+7.3f}% {st['median']:>+8.3f}% "
                  f"{st['wr']:>5.1f}% {st['t']:>+6.2f}")

    # プラセボ: ランダム日と同等数のシグナル日を比較
    print(f"\n--- プラセボテスト ---")
    print(f"LMEアップ日ON平均: {sl.mean():+.3f}% (N={len(sl)}, t={stats(sl)['t']:+.2f})")
    print(f"LMEダウン日ON平均: {ss.mean():+.3f}% (N={len(ss)}, t={stats(ss)['t']:+.2f})")
    print(f"差分 (UP - DOWN): {sl.mean() - ss.mean():+.3f}% ← これが大きければシグナルに意味あり")

    # 両側t検定的
    diff = sl.mean() - ss.mean()
    se_diff = np.sqrt(sl.var()/len(sl) + ss.var()/len(ss))
    t_stat = diff / se_diff if se_diff > 0 else 0
    print(f"UP vs DOWN t-stat (Welch): {t_stat:+.2f}")

    # ベースライン vs LMEアップ日 (ここが「上昇相場便乗」を見る指標)
    diff2 = sl.mean() - base.mean()
    se_diff2 = np.sqrt(sl.var()/len(sl) + base.var()/len(base))
    t2 = diff2 / se_diff2 if se_diff2 > 0 else 0
    print(f"\nLMEアップ日ON vs 全日ベースライン:")
    print(f"  アップ日:     {sl.mean():+.3f}%")
    print(f"  ベースライン: {base.mean():+.3f}%")
    print(f"  超過収益:   {diff2:+.3f}% (t={t2:+.2f})")
    print(f"  → これが正で有意 → ベースライン超えのエッジあり")
    print(f"  → これがゼロ近辺 → 単に上昇相場便乗")


if __name__ == "__main__":
    main()
