"""
本日 (2026-05-13) 時点での包括分析

データ取得状況:
  - stocks_daily: 5/14まで取得済み (4,449銘柄)
  - ティック: 5/11〜5/14 取得済み
  - investor_types: 5/12公表 (覆う週: 4/24-5/1) 取得済み
  - index_daily / 空売り報告: 5/8まで

分析項目:
  1. 5/11暴落の市場全体への影響 (TOPIX/N225/セクター)
  2. 5/12-5/14 のリバウンド分析
  3. 投資部門別シグナル (5/12公表): 最新シグナルは何か?
  4. B戦略 5/12-5/14 のシグナルと結果 (SBG)
  5. 5/11 で勝った銘柄/負けた銘柄 ランキング
"""
import psycopg2
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
from pathlib import Path
import sys
sys.path.insert(0, '/Users/Yusuke/claude-code/DataFetcher')
import duckdb
from src.ticks import _file_globs, _to_date, _norm5, _norm4
import warnings
warnings.filterwarnings('ignore')

PG_CONFIG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}


def main():
    import time
    t0 = time.time()

    fig = plt.figure(figsize=(16, 12), facecolor='white')
    plt.rcParams.update({
        'font.family': ['Hiragino Sans', 'IPAexGothic', 'sans-serif'],
        'axes.unicode_minus': False,
    })
    fig.suptitle('2026-05-13 包括分析: 5/11暴落と直後3日間 (5/12-14)',
                 fontsize=14, fontweight='bold', y=0.99)

    # =====================================================
    # 1. 直近のTOPIX/N225動向
    # =====================================================
    print("=" * 70)
    print("【1】 直近の市場全体動向")
    print("=" * 70)
    conn = psycopg2.connect(**PG_CONFIG)
    # stocks_daily からTOPIXは集計できないので、stocks_daily の市場ベンチマーク代用
    # 全銘柄の単純平均リターン
    daily_market = pd.read_sql("""
        WITH r AS (
          SELECT date, code,
                 (adj_close::float / NULLIF(adj_open::float, 0) - 1) * 10000 AS intraday,
                 (adj_close::float / LAG(adj_close::float) OVER (PARTITION BY code ORDER BY date) - 1) * 10000 AS daily
          FROM stocks_daily
          WHERE date >= '2026-04-15' AND adj_close IS NOT NULL
        )
        SELECT date,
               COUNT(*) AS n_codes,
               AVG(intraday) AS avg_intraday,
               AVG(daily) AS avg_daily
        FROM r
        GROUP BY date ORDER BY date
    """, conn)
    print("\n日次市場リターン (全銘柄平均):")
    print(daily_market.to_string(index=False))

    # 5/11 のセクター別ダメージ
    print("\n--- 5/11セクター別パフォーマンス (上位/下位) ---")
    sec_511 = pd.read_sql("""
        SELECT s.code,
               (s.adj_close::float / s.adj_open::float - 1) * 10000 AS intraday_bps,
               (s.adj_close::float / s.adj_open::float - 1) * 100 AS intraday_pct,
               s.adj_close
        FROM stocks_daily s
        WHERE s.date = '2026-05-11' AND s.adj_open > 0 AND s.adj_close > 0
        ORDER BY intraday_bps
    """, conn)
    print(f"\n5/11 全銘柄数: {len(sec_511)}")
    print(f"  平均下落: {sec_511['intraday_pct'].mean():.2f}%")
    print(f"  中央値: {sec_511['intraday_pct'].median():.2f}%")
    print(f"  下落銘柄数: {(sec_511['intraday_bps'] < 0).sum()} ({100*(sec_511['intraday_bps']<0).mean():.1f}%)")
    print(f"\n最大下落 Top10:")
    print(sec_511.head(10)[['code', 'intraday_pct']].to_string(index=False))
    print(f"\n最大上昇 Top10:")
    print(sec_511.tail(10)[['code', 'intraday_pct']].to_string(index=False))

    # =====================================================
    # 2. 5/11 → 5/12-14 リバウンド分析
    # =====================================================
    print("\n" + "=" * 70)
    print("【2】 5/11→5/12-14 リバウンド分析")
    print("=" * 70)
    rebound = pd.read_sql("""
        WITH d_511 AS (
          SELECT code, adj_open AS open_511, adj_close AS close_511
          FROM stocks_daily WHERE date = '2026-05-11' AND adj_close IS NOT NULL
        ),
        d_512 AS (
          SELECT code, adj_close AS close_512
          FROM stocks_daily WHERE date = '2026-05-12' AND adj_close IS NOT NULL
        ),
        d_514 AS (
          SELECT code, adj_close AS close_514
          FROM stocks_daily WHERE date = '2026-05-14' AND adj_close IS NOT NULL
        )
        SELECT d_511.code,
               (d_511.close_511 / d_511.open_511 - 1) * 10000 AS ret_511,
               (d_512.close_512 / d_511.close_511 - 1) * 10000 AS ret_512,
               (d_514.close_514 / d_511.close_511 - 1) * 10000 AS ret_511_to_514
        FROM d_511
        JOIN d_512 ON d_511.code = d_512.code
        JOIN d_514 ON d_511.code = d_514.code
    """, conn)
    print(f"\n対象銘柄: {len(rebound)}")
    print(f"5/11 平均: {rebound['ret_511'].mean()/100:.2f}%")
    print(f"5/12 (1日) 平均: {rebound['ret_512'].mean()/100:.2f}%")
    print(f"5/11→5/14 (3日累計) 平均: {rebound['ret_511_to_514'].mean()/100:.2f}%")

    # 5/11で大暴落した銘柄が翌日反発したか
    crashed = rebound[rebound['ret_511'] <= -500]  # 5%以上下落
    print(f"\n5/11で-5%以上下落: {len(crashed)}銘柄")
    print(f"  そのうち翌日5/12のリバウンド平均: {crashed['ret_512'].mean()/100:.2f}%")
    print(f"  3日後5/14までの累積: {crashed['ret_511_to_514'].mean()/100:.2f}%")

    # 急落 → 翌日反発の相関
    corr = rebound[['ret_511', 'ret_512']].corr().iloc[0, 1]
    print(f"\n5/11リターン vs 5/12リターン相関: {corr:.3f}")
    print(f"  → {'リバウンド有り (負相関)' if corr < -0.1 else 'モメンタム継続 (正)' if corr > 0.1 else 'ランダム'}")

    # =====================================================
    # 3. 投資部門別 5/12公表シグナル
    # =====================================================
    print("\n" + "=" * 70)
    print("【3】 投資部門別 5/12公表シグナル")
    print("=" * 70)
    import json
    inv = pd.read_sql("""
        SELECT pub_date, st_date, en_date, payload
        FROM investor_types
        WHERE section = 'TSEPrime' AND pub_date >= '2025-11-01'
        ORDER BY pub_date
    """, conn)
    records = []
    for _, r in inv.iterrows():
        p = r['payload'] if isinstance(r['payload'], dict) else json.loads(r['payload'])
        records.append({
            'pub_date': r['pub_date'],
            'en_date': r['en_date'],
            'FrgnBal': p.get('FrgnBal', np.nan) / 1e8,
            'IndBal': p.get('IndBal', np.nan) / 1e8,
            'InvTrBal': p.get('InvTrBal', np.nan) / 1e8,
            'TrstBnkBal': p.get('TrstBnkBal', np.nan) / 1e8,
            'PropBal': p.get('PropBal', np.nan) / 1e8,
        })
    inv_df = pd.DataFrame(records)
    print("\n投資部門別 直近6か月の週次データ (億円):")
    print(inv_df.to_string(index=False))

    # 5/12公表分の Z-score 計算
    # 全期間 (2016以降) の26週ローリングZ
    inv_full = pd.read_sql("""
        SELECT pub_date, payload
        FROM investor_types
        WHERE section IN ('TSEPrime', 'TSE1st') AND pub_date >= '2016-01-01'
        ORDER BY pub_date
    """, conn)
    conn.close()
    full_recs = []
    for _, r in inv_full.iterrows():
        p = r['payload'] if isinstance(r['payload'], dict) else json.loads(r['payload'])
        full_recs.append({
            'pub_date': r['pub_date'],
            'FrgnBal': p.get('FrgnBal', np.nan) / 1e8,
            'IndBal': p.get('IndBal', np.nan) / 1e8,
            'InvTrBal': p.get('InvTrBal', np.nan) / 1e8,
        })
    full_df = pd.DataFrame(full_recs).drop_duplicates('pub_date', keep='last').sort_values('pub_date').reset_index(drop=True)
    full_df['pub_date'] = pd.to_datetime(full_df['pub_date'])

    for c in ['FrgnBal', 'IndBal', 'InvTrBal']:
        ma = full_df[c].shift(1).rolling(26, min_periods=8).mean()
        sd = full_df[c].shift(1).rolling(26, min_periods=8).std()
        full_df[f'{c}_z'] = (full_df[c] - ma) / sd.replace(0, np.nan)

    full_df['FvI'] = full_df['FrgnBal'] - full_df['IndBal']
    full_df['FvI_z'] = ((full_df['FvI'] - full_df['FvI'].shift(1).rolling(26).mean()) /
                        full_df['FvI'].shift(1).rolling(26).std().replace(0, np.nan))

    latest = full_df.iloc[-1]
    print(f"\n=== 最新シグナル (公表日: {latest['pub_date'].strftime('%Y-%m-%d')}) ===")
    print(f"  対象週: {full_df.iloc[-1]['pub_date']}")
    print(f"  海外勢 Bal: {latest['FrgnBal']:.0f}億 (Z={latest['FrgnBal_z']:.2f})")
    print(f"  個人 Bal: {latest['IndBal']:.0f}億 (Z={latest['IndBal_z']:.2f})")
    print(f"  投信 Bal: {latest['InvTrBal']:.0f}億 (Z={latest['InvTrBal_z']:.2f})")
    print(f"  海外-個人 スプレッド: {latest['FvI']:.0f}億 (Z={latest['FvI_z']:.2f})")

    # シグナル判定
    signals_active = []
    if latest['FrgnBal_z'] > 1.5:
        signals_active.append(f"海外勢 Z>1.5 (Z={latest['FrgnBal_z']:.2f}) ロング ✓")
    elif latest['FrgnBal_z'] > 1.0:
        signals_active.append(f"海外勢 Z>1.0緩 (Z={latest['FrgnBal_z']:.2f}) ロング △")
    if latest['InvTrBal_z'] > 1.5:
        signals_active.append(f"投信 Z>1.5 (Z={latest['InvTrBal_z']:.2f}) ロング ✓")
    if latest['FvI_z'] > 1.5:
        signals_active.append(f"海外-個人 Z>1.5 (Z={latest['FvI_z']:.2f}) ロング ✓")
    if latest['FrgnBal_z'] < -1.0:
        signals_active.append(f"海外勢 Z<-1.0 弱気 ✗")
    print(f"\n  発動シグナル: {signals_active if signals_active else 'なし (中立)'}")

    # =====================================================
    # 4. B戦略 5/12-5/14 SBG OOS
    # =====================================================
    print("\n" + "=" * 70)
    print("【4】 B戦略 5/12-5/14 SBG OOS")
    print("=" * 70)
    print("SBGティック取得 (5/12-5/14)...")
    s = pd.Timestamp('2026-05-12').date()
    e = pd.Timestamp('2026-05-14').date()
    root = Path.home() / "Data" / "jquants_trades" / "equities" / "trades"
    files = _file_globs(root, s, e)
    if files:
        files_arr = "[" + ", ".join(f"'{f}'" for f in files) + "]"
        csv_part = (
            f"read_csv({files_arr}, header=true, "
            "columns={'Date':'DATE','Code':'VARCHAR','Time':'VARCHAR',"
            "'SessionDistinction':'VARCHAR','Price':'DOUBLE','TradingVolume':'BIGINT',"
            "'TransactionId':'VARCHAR'})"
        )
        sql = f"""
            SELECT Date AS d, Time AS tm, Price AS price, TradingVolume AS vol,
                   date_add(CAST(Date AS TIMESTAMP),
                            INTERVAL (
                                EXTRACT(HOUR FROM CAST(Time AS TIME)) * 3600
                              + EXTRACT(MINUTE FROM CAST(Time AS TIME)) * 60
                              + EXTRACT(SECOND FROM CAST(Time AS TIME))
                            ) SECOND) AS ts
            FROM {csv_part}
            WHERE Date BETWEEN ? AND ? AND Code IN (?, ?)
            ORDER BY ts
        """
        con = duckdb.connect()
        con.execute("PRAGMA threads=4")
        ticks = con.execute(sql, [s, e, _norm5('99840'), _norm4('99840')]).df()
        con.close()
        print(f"  {len(ticks):,} ticks")

        # 大口買いシグナル (vol≥8200 & uptick & 9:00-10:30)
        ticks = ticks.sort_values('ts').reset_index(drop=True)
        ticks['diff'] = ticks['price'] - ticks['price'].shift(1)
        ticks['date'] = pd.to_datetime(ticks['d']).dt.date
        ticks['hour'] = ticks['ts'].dt.hour
        ticks['minute'] = ticks['ts'].dt.minute

        sig = ticks[
            (ticks['vol'] >= 8200) &
            (ticks['diff'] > 0) &
            ((ticks['hour'] == 9) | ((ticks['hour'] == 10) & (ticks['minute'] < 30)))
        ].copy()
        print(f"\n  B-Optimized シグナル (9:00-10:30 大口買い): {len(sig)}件")

        # 60分後リターン計算
        ticks_us = ticks['ts'].astype('datetime64[s]').astype('int64').values
        prices_arr = ticks['price'].values.astype(float)
        dates_arr = ticks['date'].values

        sig['fwd_60min'] = np.nan
        sig['fwd_60min_pct'] = np.nan
        for i in sig.index:
            t0_s = ticks['ts'].iloc[i].timestamp()
            # +1秒レイテンシ
            t_entry = int(t0_s + 1)
            j_entry = np.searchsorted(ticks_us, t_entry)
            if j_entry >= len(prices_arr):
                continue
            sig_date = sig.at[i, 'date']
            if dates_arr[j_entry] != sig_date:
                continue
            entry_p = prices_arr[j_entry]
            t_exit = ticks_us[j_entry] + 3600
            j_exit = np.searchsorted(ticks_us, t_exit)
            if j_exit >= len(prices_arr) or dates_arr[j_exit] != sig_date:
                # 同日内最終
                last_same = j_entry
                while last_same + 1 < len(prices_arr) and dates_arr[last_same + 1] == sig_date:
                    last_same += 1
                if last_same <= j_entry:
                    continue
                exit_p = prices_arr[last_same]
            else:
                exit_p = prices_arr[j_exit]
            ret_bps = (exit_p / entry_p - 1) * 10000
            sig.at[i, 'fwd_60min'] = ret_bps
            sig.at[i, 'fwd_60min_pct'] = ret_bps / 100

        # 日別集計
        for d in sorted(sig['date'].unique()):
            sub = sig[sig['date'] == d].dropna(subset=['fwd_60min'])
            n = len(sub)
            if n > 0:
                avg = sub['fwd_60min'].mean()
                wr = (sub['fwd_60min'] > 4).mean() * 100  # コスト後勝ち
                total = (sub['fwd_60min'] - 4).sum()
                print(f"  {d}: N={n}, mean(raw)={avg:.1f}bps, "
                      f"net={avg-4:.1f}bps, 勝率={wr:.0f}%, 合計net={total:.0f}bps")

    # =====================================================
    # 5. 5/11で大ダメージだった銘柄ランキング + 直後リバウンド
    # =====================================================
    print("\n" + "=" * 70)
    print("【5】 5/11ワースト銘柄 と そのリバウンド")
    print("=" * 70)
    worst = rebound.sort_values('ret_511').head(20).copy()
    worst['ret_511_pct'] = worst['ret_511'] / 100
    worst['ret_512_pct'] = worst['ret_512'] / 100
    worst['ret_511_to_514_pct'] = worst['ret_511_to_514'] / 100
    print(worst[['code', 'ret_511_pct', 'ret_512_pct', 'ret_511_to_514_pct']].to_string(index=False))

    # =====================================================
    # 図
    # =====================================================
    # 上左: 市場全体の日次推移
    ax1 = fig.add_axes([0.04, 0.55, 0.30, 0.38])
    daily_market_df = daily_market.copy()
    daily_market_df['date'] = pd.to_datetime(daily_market_df['date'])
    cols = ['#43A047' if v >= 0 else '#E53935' for v in daily_market_df['avg_intraday']]
    ax1.bar(daily_market_df['date'], daily_market_df['avg_intraday'].values / 100,
            color=cols, alpha=0.85)
    ax1.axvline(pd.Timestamp('2026-05-11'), color='red', linestyle='--', alpha=0.5, lw=1)
    ax1.axhline(0, color='black', lw=0.6)
    ax1.set_ylabel('全銘柄平均 当日リターン (%)', fontsize=9)
    ax1.set_title('日次市場全体リターン (4/15-5/14)', fontsize=10, fontweight='bold')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    plt.setp(ax1.get_xticklabels(), rotation=30, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # 上中: 5/11リターン分布
    ax2 = fig.add_axes([0.38, 0.55, 0.30, 0.38])
    pct_511 = sec_511['intraday_pct'].clip(-15, 5)
    ax2.hist(pct_511, bins=60, color='#E53935', alpha=0.7, edgecolor='none')
    ax2.axvline(0, color='black', lw=1)
    ax2.axvline(sec_511['intraday_pct'].mean(), color='blue', linestyle='--',
                lw=1.5, label=f'平均 {sec_511["intraday_pct"].mean():.2f}%')
    ax2.set_xlabel('5/11 日中リターン (%)', fontsize=9)
    ax2.set_ylabel('銘柄数', fontsize=9)
    ax2.set_title(f'5/11 全銘柄リターン分布 (N={len(sec_511)})',
                  fontsize=10, fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # 上右: 5/11 vs 5/12 散布図 (リバウンド検証)
    ax3 = fig.add_axes([0.72, 0.55, 0.25, 0.38])
    ax3.scatter(rebound['ret_511']/100, rebound['ret_512']/100,
                s=2, alpha=0.3, c='#666')
    ax3.axhline(0, color='black', lw=0.6)
    ax3.axvline(0, color='black', lw=0.6)
    # 回帰線
    x = rebound['ret_511'].values / 100
    y = rebound['ret_512'].values / 100
    mask = np.isfinite(x) & np.isfinite(y)
    z = np.polyfit(x[mask], y[mask], 1)
    p = np.poly1d(z)
    xs = np.linspace(-15, 5, 50)
    ax3.plot(xs, p(xs), color='red', lw=1.5, label=f'傾き={z[0]:.2f}, r={corr:.3f}')
    ax3.set_xlabel('5/11 リターン (%)', fontsize=9)
    ax3.set_ylabel('5/12 リターン (%)', fontsize=9)
    ax3.set_title('5/11→5/12 リバウンド有無', fontsize=10, fontweight='bold')
    ax3.legend(fontsize=8)
    ax3.set_xlim(-15, 5)
    ax3.set_ylim(-10, 10)
    ax3.grid(alpha=0.3)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    # 下左: 投資部門別 直近の動き
    ax4 = fig.add_axes([0.04, 0.07, 0.40, 0.42])
    recent_df = full_df.tail(30).copy()
    ax4.plot(recent_df['pub_date'], recent_df['FrgnBal_z'], 'o-', color='#1565C0',
             lw=1.5, label='海外勢 Z')
    ax4.plot(recent_df['pub_date'], recent_df['InvTrBal_z'], 's-', color='#FF9800',
             lw=1.5, alpha=0.7, label='投信 Z')
    ax4.plot(recent_df['pub_date'], recent_df['FvI_z'], '^-', color='#43A047',
             lw=1.5, alpha=0.7, label='海外-個人 Z')
    ax4.axhline(1.5, color='gray', linestyle='--', lw=0.6)
    ax4.axhline(-1.5, color='gray', linestyle='--', lw=0.6)
    ax4.axhline(0, color='black', lw=0.6)
    # 5/12公表分を強調
    latest_z = full_df.iloc[-1]
    ax4.scatter([latest_z['pub_date']], [latest_z['FrgnBal_z']],
                s=200, marker='*', color='red', zorder=10,
                label=f"5/12最新 海外Z={latest_z['FrgnBal_z']:.2f}")
    ax4.set_xlabel('公表日', fontsize=9)
    ax4.set_ylabel('Z-score (26週ローリング)', fontsize=9)
    ax4.set_title('投資部門別シグナル 直近の推移', fontsize=10, fontweight='bold')
    ax4.legend(fontsize=8, loc='best')
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    plt.setp(ax4.get_xticklabels(), rotation=30, ha='right')
    ax4.grid(alpha=0.3)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)

    # 下右: 5/11ワースト20銘柄 + リバウンド
    ax5 = fig.add_axes([0.48, 0.07, 0.50, 0.42])
    worst20 = rebound.sort_values('ret_511').head(20).copy()
    ys = range(len(worst20))
    ax5.barh(list(ys), worst20['ret_511']/100, color='#E53935', alpha=0.7,
             label='5/11 リターン (%)')
    ax5.barh(list(ys), worst20['ret_511_to_514']/100, color='#43A047',
             alpha=0.5, label='5/11→5/14 累積 (%)')
    ax5.set_yticks(list(ys))
    ax5.set_yticklabels(worst20['code'], fontsize=7)
    ax5.axvline(0, color='black', lw=0.6)
    ax5.set_xlabel('リターン (%)', fontsize=9)
    ax5.set_title('5/11 ワースト20銘柄: 暴落 vs 5/11→5/14 累積',
                  fontsize=10, fontweight='bold')
    ax5.legend(fontsize=8, loc='best')
    ax5.grid(axis='x', alpha=0.3)
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)

    fig.text(0.99, 0.005,
             'データ: 2026-04-15〜2026-05-14 / stocks_daily 4449銘柄 + ティック5/11-14 + investor_types 5/12公表',
             ha='right', va='bottom', fontsize=7, color='gray')
    plt.savefig('result.png', dpi=100, bbox_inches='tight', facecolor='white')

    # CSV保存
    daily_market.to_csv('daily_market.csv', index=False)
    rebound.to_csv('rebound.csv', index=False)
    inv_df.to_csv('investor_recent.csv', index=False)

    print(f"\nresult.png 保存完了 ({time.time()-t0:.1f}秒)")


if __name__ == '__main__':
    main()
