"""
イントラデイ ルール検証 — 5分足・10分足レベル
平均回帰 / モメンタム / 出来高分析

全ルールでルック・アヘッドなし:
  - バーt までの情報でバーt+1 以降を予測
  - ret_1b  = 次の5分 (1バー先)
  - ret_3b  = 次の15分 (3バー先)
  - ret_6b  = 次の30分 (6バー先)
  - ret_eod = 今から大引けまで

コスト: 往復4bps
"""

import psycopg2
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

PG_CONFIG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}
COST_BPS = 4

# 対象: データ量が多い主要銘柄に絞る (計算速度のため)
TARGET_SYMBOLS = [
    # 半導体
    '6920.T', '6857.T', '8035.T', '6146.T', '6273.T', '6861.T',
    # 非鉄
    '5713.T', '5711.T', '5706.T', '5801.T', '5802.T', '5803.T',
    # 海運
    '9104.T', '9107.T', '9101.T',
    # 重工・防衛
    '7011.T', '7012.T', '7013.T',
    # 自動車
    '7203.T', '7267.T', '7270.T',
    # 商社
    '8053.T', '8058.T', '8001.T',
    # 銀行
    '8306.T', '8316.T', '8411.T',
    # IT
    '9984.T', '6702.T',
    # 電機
    '6762.T', '6752.T', '6954.T',
    # 鉄鋼
    '5401.T', '5411.T',
]

def load_data():
    syms = "','".join(TARGET_SYMBOLS)
    conn = psycopg2.connect(**PG_CONFIG)
    q = f"""
        SELECT symbol, timestamp, open, high, low, close, volume
        FROM intraday_data
        WHERE interval = '1min'
          AND symbol IN ('{syms}')
          AND timestamp >= '2025-04-26'
        ORDER BY symbol, timestamp
    """
    # ショック後のみ (有効性が確認された期間)
    print("データ読み込み中 (ショック後2025/4/26~)...")
    df = pd.read_sql(q, conn)
    conn.close()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['jst'] = df['timestamp'] + pd.Timedelta(hours=9)
    return df.dropna(subset=['close'])


def make_5min_bars(df1m: pd.DataFrame) -> pd.DataFrame:
    """1分足 → 5分足に変換"""
    df = df1m.copy()
    df = df.set_index('jst').sort_index()

    # 5分バーのラベル (floor to 5min)
    df['bar'] = df.index.floor('5min')

    bars = df.groupby(['symbol', 'bar']).agg(
        open=('open', 'first'),
        high=('high', 'max'),
        low=('low', 'min'),
        close=('close', 'last'),
        volume=('volume', 'sum'),
    ).reset_index()

    # 取引時間内のみ (9:00-11:30, 12:30-15:00)
    h = bars['bar'].dt.hour
    m = bars['bar'].dt.minute
    in_session = (
        ((h == 9) | (h == 10) | ((h == 11) & (m <= 25))) |
        (((h == 12) & (m >= 30)) | (h == 13) | (h == 14))
    )
    bars = bars[in_session].copy()
    bars['date'] = bars['bar'].dt.date
    bars['time_min'] = bars['bar'].dt.hour * 60 + bars['bar'].dt.minute  # 分単位の時刻
    return bars.reset_index(drop=True)


def add_features(bars: pd.DataFrame) -> pd.DataFrame:
    """バーごとの特徴量を計算 (look-ahead なし)"""
    result = []

    for (symbol, date), g in bars.groupby(['symbol', 'date']):
        g = g.sort_values('bar').reset_index(drop=True)
        n = len(g)
        if n < 10:
            continue

        closes = g['close'].values
        volumes = g['volume'].values
        opens   = g['open'].values
        highs   = g['high'].values
        lows    = g['low'].values
        times   = g['time_min'].values

        # VWAP (累積)
        cum_vol  = volumes.cumsum()
        cum_pv   = (closes * volumes).cumsum()
        vwap     = np.where(cum_vol > 0, cum_pv / cum_vol, closes)

        # 各バーのリターン (%)
        bar_ret = np.zeros(n)
        bar_ret[1:] = (closes[1:] / closes[:-1] - 1) * 10000  # bps

        # 移動平均 (20バー = 100分)
        ma20 = pd.Series(closes).rolling(20, min_periods=5).mean().values

        # 出来高移動平均 (20バー)
        vol20 = pd.Series(volumes).rolling(20, min_periods=5).mean().values

        # 当日始値
        day_open = opens[0]

        for i in range(5, n):  # 最低5バーの情報が必要
            c   = closes[i]
            v   = volumes[i]
            t   = times[i]
            vw  = vwap[i]
            m20 = ma20[i]
            v20 = vol20[i]

            # ---- 特徴量 ----
            # 直近N バーの累積リターン
            cum3 = (c / closes[i-3] - 1) * 10000 if i >= 3 else 0   # 15分
            cum5 = (c / closes[i-5] - 1) * 10000 if i >= 5 else 0   # 25分
            cum10= (c / closes[max(0,i-10)] - 1) * 10000             # 50分

            # 当日寄付からの累積
            cum_from_open = (c / day_open - 1) * 10000

            # VWAPからの乖離 (bps)
            vwap_dev = (c / vw - 1) * 10000 if vw > 0 else 0

            # MA20からの乖離 (bps)
            ma20_dev = (c / m20 - 1) * 10000 if (m20 and m20 > 0) else 0

            # 出来高比 (直近バー/20バー平均)
            vol_ratio = v / v20 if (v20 and v20 > 0) else 1.0

            # 直近3バーの出来高トレンド (増加してるか)
            vol_trend = (volumes[i] / volumes[i-3] - 1) if (i >= 3 and volumes[i-3] > 0) else 0

            # 直近バーのリターン
            ret_now = bar_ret[i]

            # 出来高×リターン (方向×強さ)
            vol_ret_signal = np.sign(cum3) * vol_ratio

            # 時間帯カテゴリー
            if t <= 9*60+30:
                time_cat = 'open_rush'      # 9:00-9:30
            elif t <= 10*60:
                time_cat = 'mid_morning'    # 9:30-10:00
            elif t <= 11*60:
                time_cat = 'late_morning'   # 10:00-11:00
            elif t <= 11*60+30:
                time_cat = 'pre_lunch'      # 11:00-11:30
            elif t <= 13*60:
                time_cat = 'aft_open'       # 12:30-13:00
            elif t <= 14*60:
                time_cat = 'mid_aft'        # 13:00-14:00
            else:
                time_cat = 'close_rush'     # 14:00-15:00

            # ---- ターゲット (look-ahead) ----
            ret_1b  = (closes[i+1]/c - 1)*10000 if i+1 < n else np.nan  # 次の5分
            ret_3b  = (closes[min(i+3,n-1)]/c - 1)*10000 if i+3 < n else np.nan
            ret_6b  = (closes[min(i+6,n-1)]/c - 1)*10000 if i+6 < n else np.nan
            eod_close = closes[-1]
            ret_eod = (eod_close/c - 1)*10000 if i < n-3 else np.nan   # 今から大引け

            result.append({
                'symbol': symbol,
                'date': date,
                'bar': g['bar'].iloc[i],
                'time_min': t,
                'time_cat': time_cat,
                'close': c,
                'volume': v,
                'cum3': cum3,
                'cum5': cum5,
                'cum10': cum10,
                'cum_from_open': cum_from_open,
                'vwap_dev': vwap_dev,
                'ma20_dev': ma20_dev,
                'vol_ratio': vol_ratio,
                'vol_trend': vol_trend,
                'ret_now': ret_now,
                'vol_ret_signal': vol_ret_signal,
                'ret_1b': ret_1b,
                'ret_3b': ret_3b,
                'ret_6b': ret_6b,
                'ret_eod': ret_eod,
            })

    return pd.DataFrame(result)


def test_rule(df, cond, target, label, direction=1):
    sub = df[cond & df[target].notna()]
    if len(sub) < 100:
        return None
    arr = sub[target].values * direction
    net = arr - COST_BPS
    t, p = stats.ttest_1samp(arr, 0)
    wr = (arr > 0).mean() * 100
    neg = abs(arr[arr<=0].sum())
    pf = arr[arr>0].sum()/neg if neg > 0 else np.inf
    sharpe = net.mean()/arr.std()*np.sqrt(252*66) if arr.std() > 0 else 0  # 66バー/日
    return dict(rule=label, target=target, dir=direction, N=len(sub),
                mean_raw=round(arr.mean(),2), mean_net=round(net.mean(),2),
                std=round(arr.std(),2), t_stat=round(t,2), p_val=round(p,4),
                win_rate=round(wr,1), profit_factor=round(pf,2),
                sharpe=round(sharpe,2),
                sig=(p<0.05 and net.mean()>0))


def run_rules(df):
    res = []
    def add(r):
        if r: res.append(r)

    # ============================================================
    # A. 平均回帰系 (Mean Reversion)
    # ============================================================
    # A1: 直近15分で急落 (<-30bps) → 次の5分で反転買い
    add(test_rule(df, df['cum3'] <= -30, 'ret_1b', 'MR_A1_Drop15m_Buy1b'))
    # A2: 直近15分で急落 → 次の30分で反転買い
    add(test_rule(df, df['cum3'] <= -30, 'ret_6b', 'MR_A2_Drop15m_Buy6b'))
    # A3: 直近15分で急騰 (>+30bps) → 次の5分で反転売り
    add(test_rule(df, df['cum3'] >= 30, 'ret_1b', 'MR_A3_Rise15m_Sell1b', direction=-1))
    # A4: 直近15分で急騰 → 次の30分で反転売り
    add(test_rule(df, df['cum3'] >= 30, 'ret_6b', 'MR_A4_Rise15m_Sell6b', direction=-1))
    # A5: 直近25分で急落 (<-50bps) → 次の30分で反転買い
    add(test_rule(df, df['cum5'] <= -50, 'ret_6b', 'MR_A5_Drop25m_Buy6b'))
    # A6: 直近25分で急騰 (>+50bps) → 次の30分で反転売り
    add(test_rule(df, df['cum5'] >= 50, 'ret_6b', 'MR_A6_Rise25m_Sell6b', direction=-1))
    # A7: 当日寄付から-100bps以上下落 → 今から大引けで買い
    add(test_rule(df, df['cum_from_open'] <= -100, 'ret_eod', 'MR_A7_CumDrop100_EodBuy'))
    # A8: 当日寄付から+100bps以上上昇 → 今から大引けで売り
    add(test_rule(df, df['cum_from_open'] >= 100, 'ret_eod', 'MR_A8_CumRise100_EodSell', direction=-1))

    # A9: VWAP より大幅下方 (<-30bps) → 次の5分買い
    add(test_rule(df, df['vwap_dev'] <= -30, 'ret_1b', 'MR_A9_BelowVWAP30_Buy1b'))
    # A10: VWAP より大幅下方 → 次の30分買い
    add(test_rule(df, df['vwap_dev'] <= -30, 'ret_6b', 'MR_A10_BelowVWAP30_Buy6b'))
    # A11: VWAP より大幅上方 → 次の5分売り
    add(test_rule(df, df['vwap_dev'] >= 30, 'ret_1b', 'MR_A11_AboveVWAP30_Sell1b', direction=-1))
    # A12: VWAP より大幅上方 → 次の30分売り
    add(test_rule(df, df['vwap_dev'] >= 30, 'ret_6b', 'MR_A12_AboveVWAP30_Sell6b', direction=-1))

    # A13: MA20より下方 (<-50bps) → 次の30分買い
    add(test_rule(df, df['ma20_dev'] <= -50, 'ret_6b', 'MR_A13_BelowMA20_50_Buy6b'))
    # A14: MA20より上方 → 次の30分売り
    add(test_rule(df, df['ma20_dev'] >= 50, 'ret_6b', 'MR_A14_AboveMA20_50_Sell6b', direction=-1))

    # ============================================================
    # B. モメンタム系 (Momentum)
    # ============================================================
    # B1: 直近15分で上昇(>+20bps) → 次の5分継続
    add(test_rule(df, df['cum3'] >= 20, 'ret_1b', 'MOM_B1_Rise15m_Buy1b'))
    # B2: 直近15分で下落(<-20bps) → 次の5分継続
    add(test_rule(df, df['cum3'] <= -20, 'ret_1b', 'MOM_B2_Drop15m_Sell1b', direction=-1))
    # B3: 直近50分で上昇(>+50bps) → 次の30分継続
    add(test_rule(df, df['cum10'] >= 50, 'ret_6b', 'MOM_B3_Rise50m_Buy6b'))
    # B4: 直近50分で下落(<-50bps) → 次の30分継続
    add(test_rule(df, df['cum10'] <= -50, 'ret_6b', 'MOM_B4_Drop50m_Sell6b', direction=-1))
    # B5: 当日モメンタム(寄付+50bps以上) → 今から大引け買い
    add(test_rule(df, df['cum_from_open'] >= 50, 'ret_eod', 'MOM_B5_StrongDay_EodBuy'))
    # B6: 当日モメンタム(寄付-50bps以下) → 今から大引け売り
    add(test_rule(df, df['cum_from_open'] <= -50, 'ret_eod', 'MOM_B6_WeakDay_EodSell', direction=-1))
    # B7: VWAPを上方クロス (vwap_dev が0付近かつ cum3>0) → 継続
    add(test_rule(df, (df['vwap_dev'] >= 0) & (df['vwap_dev'] <= 15) & (df['cum3'] > 0),
                  'ret_6b', 'MOM_B7_VWAPCrossUp_Buy6b'))
    # B8: VWAPを下方クロス → 継続
    add(test_rule(df, (df['vwap_dev'] <= 0) & (df['vwap_dev'] >= -15) & (df['cum3'] < 0),
                  'ret_6b', 'MOM_B8_VWAPCrossDown_Sell6b', direction=-1))

    # ============================================================
    # C. 出来高分析系 (Volume)
    # ============================================================
    # C1: 出来高スパイク(>2倍) × 価格上昇 → 次の5分買い継続
    add(test_rule(df, (df['vol_ratio'] >= 2.0) & (df['cum3'] > 0), 'ret_1b',
                  'VOL_C1_VolSpike_Up_Buy1b'))
    # C2: 出来高スパイク × 価格下落 → 次の5分売り継続
    add(test_rule(df, (df['vol_ratio'] >= 2.0) & (df['cum3'] < 0), 'ret_1b',
                  'VOL_C2_VolSpike_Down_Sell1b', direction=-1))
    # C3: 出来高スパイク × 価格上昇 → 次の30分買い
    add(test_rule(df, (df['vol_ratio'] >= 2.0) & (df['cum3'] > 0), 'ret_6b',
                  'VOL_C3_VolSpike_Up_Buy6b'))
    # C4: 出来高スパイク × 価格下落 → 次の30分売り
    add(test_rule(df, (df['vol_ratio'] >= 2.0) & (df['cum3'] < 0), 'ret_6b',
                  'VOL_C4_VolSpike_Down_Sell6b', direction=-1))
    # C5: 出来高収縮(0.5倍以下) × 価格上昇 → 次5分は反転売り (フェイク抜け?)
    add(test_rule(df, (df['vol_ratio'] <= 0.5) & (df['cum3'] > 0), 'ret_1b',
                  'VOL_C5_LowVol_Up_Sell1b', direction=-1))
    # C6: 出来高収縮 × 価格下落 → 次5分は反転買い
    add(test_rule(df, (df['vol_ratio'] <= 0.5) & (df['cum3'] < 0), 'ret_1b',
                  'VOL_C6_LowVol_Down_Buy1b'))
    # C7: 出来高増加トレンド(前3バーで増加) × 価格上昇 → モメンタム
    add(test_rule(df, (df['vol_trend'] >= 0.5) & (df['cum3'] > 0), 'ret_3b',
                  'VOL_C7_VolAccel_Up_Buy3b'))
    # C8: 出来高増加トレンド × 価格下落 → モメンタム
    add(test_rule(df, (df['vol_trend'] >= 0.5) & (df['cum3'] < 0), 'ret_3b',
                  'VOL_C8_VolAccel_Down_Sell3b', direction=-1))
    # C9: 出来高スパイク × VWAP 下方 → リバーサル買い
    add(test_rule(df, (df['vol_ratio'] >= 2.0) & (df['vwap_dev'] <= -20), 'ret_6b',
                  'VOL_C9_VolSpike_BelowVWAP_Buy6b'))
    # C10: 出来高スパイク × VWAP 上方 → リバーサル売り
    add(test_rule(df, (df['vol_ratio'] >= 2.0) & (df['vwap_dev'] >= 20), 'ret_6b',
                  'VOL_C10_VolSpike_AboveVWAP_Sell6b', direction=-1))

    # ============================================================
    # D. 時間帯×モメンタム/リバーサル
    # ============================================================
    for tcat, tlabel in [('open_rush','寄付'), ('mid_morning','前場中盤'),
                         ('aft_open','後場序盤'), ('close_rush','大引前')]:
        sub = df[df['time_cat'] == tcat]
        # モメンタム
        add(test_rule(sub, sub['cum3'] >= 20, 'ret_3b',
                      f'TIME_D_MOM_Up_{tlabel}'))
        # リバーサル
        add(test_rule(sub, sub['cum3'] <= -20, 'ret_3b',
                      f'TIME_D_MR_Down_Buy_{tlabel}'))

    return pd.DataFrame(res)


if __name__ == '__main__':
    import time
    t0 = time.time()

    raw = load_data()
    print(f"  {len(raw):,}行, {raw['symbol'].nunique()}銘柄")

    print("5分バー生成中...")
    bars = make_5min_bars(raw)
    print(f"  {len(bars):,}バー")

    print("特徴量計算中...")
    feats = add_features(bars)
    print(f"  {len(feats):,}レコード")

    print("ルール検証中...")
    results = run_rules(feats)
    results = results.sort_values('t_stat', ascending=False)

    print(f"\n=== 全ルール (t値降順) ===")
    cols = ['rule','target','N','mean_raw','mean_net','t_stat','p_val','win_rate','sharpe','sig']
    print(results[cols].to_string(index=False))

    sig = results[results['sig']]
    print(f"\n=== 有望ルール (p<0.05 かつ net>0): {len(sig)}件 ===")
    print(sig[cols].to_string(index=False))

    results.to_csv('results_all.csv', index=False)
    sig.to_csv('results_significant.csv', index=False)

    # ---- 可視化 ----
    fig = plt.figure(figsize=(16, 10), facecolor='white')
    plt.rcParams.update({
        'font.family': ['Hiragino Sans', 'IPAexGothic', 'sans-serif'],
        'axes.unicode_minus': False,
    })
    fig.suptitle('イントラデイ ルール検証 (5分足)\n平均回帰・モメンタム・出来高分析 — 36銘柄 2025/4/26~',
                 fontsize=13, fontweight='bold', y=0.99)

    # カテゴリー色
    cat_color = {'MR': '#E53935', 'MOM': '#1E88E5', 'VOL': '#43A047', 'TIME': '#8E24AA'}
    def get_color(rule):
        for k, v in cat_color.items():
            if rule.startswith(k):
                return v
        return 'gray'

    # --- 左: t統計量バープロット ---
    ax1 = fig.add_axes([0.04, 0.10, 0.40, 0.84])
    top = results.head(40)
    ys = range(len(top))
    colors = [get_color(r) for r in top['rule']]
    ax1.barh(list(ys), top['t_stat'], color=colors, alpha=0.8, height=0.7)
    ax1.set_yticks(list(ys))
    ax1.set_yticklabels(top['rule'], fontsize=6.5)
    ax1.axvline(0, color='black', lw=0.8)
    ax1.axvline(1.96, color='gray', lw=0.8, linestyle='--', alpha=0.7, label='t=1.96')
    ax1.axvline(-1.96, color='gray', lw=0.8, linestyle='--', alpha=0.7)
    ax1.set_xlabel('t統計量', fontsize=9)
    ax1.set_title('全ルール t値ランキング (上位40)', fontsize=10, fontweight='bold')
    patches = [mpatches.Patch(color=v, label=k, alpha=0.8) for k, v in cat_color.items()]
    ax1.legend(handles=patches, fontsize=8, loc='lower right')
    ax1.grid(axis='x', alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # --- 右上: 有望ルール mean_net ---
    ax2 = fig.add_axes([0.50, 0.55, 0.47, 0.39])
    if len(sig) > 0:
        sig_s = sig.sort_values('mean_net', ascending=True)
        ys2 = range(len(sig_s))
        c2 = [get_color(r) for r in sig_s['rule']]
        ax2.barh(list(ys2), sig_s['mean_net'], color=c2, alpha=0.8, height=0.6)
        ax2.set_yticks(list(ys2))
        ax2.set_yticklabels(sig_s['rule'], fontsize=7.5)
        ax2.axvline(0, color='black', lw=0.8)
        for i, (_, row) in enumerate(sig_s.iterrows()):
            ax2.text(row['mean_net'] + 0.02, i,
                     f"N={row['N']:,} WR={row['win_rate']:.0f}%",
                     va='center', fontsize=6.5)
    ax2.set_xlabel('コスト後平均リターン (bps)', fontsize=9)
    ax2.set_title(f'有望ルール (p<0.05, net>0): {len(sig)}件', fontsize=10, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # --- 右下: スキャッター (t_stat vs mean_net) ---
    ax3 = fig.add_axes([0.50, 0.10, 0.47, 0.38])
    for _, row in results.iterrows():
        c3 = get_color(row['rule'])
        m = '*' if row['sig'] else 'o'
        s = 80 if row['sig'] else 25
        ax3.scatter(row['t_stat'], row['mean_net'], c=c3, marker=m, s=s,
                    alpha=0.85 if row['sig'] else 0.35, zorder=5 if row['sig'] else 3)
    ax3.axhline(0, color='black', lw=0.8)
    ax3.axvline(0, color='black', lw=0.8)
    ax3.axvline(1.96, color='gray', lw=0.8, linestyle='--', alpha=0.5)
    ax3.axvline(-1.96, color='gray', lw=0.8, linestyle='--', alpha=0.5)
    ax3.set_xlabel('t統計量', fontsize=9)
    ax3.set_ylabel('コスト後平均リターン (bps)', fontsize=9)
    ax3.set_title('全ルール スキャッター (★=有意)', fontsize=10, fontweight='bold')
    ax3.legend(handles=patches, fontsize=7)
    ax3.grid(alpha=0.3)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    fig.text(0.99, 0.005,
             'データ: 2025/4/26〜2026/5/7 / 日本株36銘柄5分足 | コスト4bps往復 | 取引時間内バーのみ',
             ha='right', va='bottom', fontsize=7, color='gray')
    plt.savefig('result.png', dpi=100, bbox_inches='tight', facecolor='white')
    print(f"\nresult.png 保存完了 ({time.time()-t0:.1f}秒)")
