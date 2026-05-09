"""
幅広いアルゴリズムトレーディングルール検証 (ルック・アヘッドバイアス修正版)
日本個別株 1分足データ (2025-01-01 ~ 2026-05-07)

予測対象と特徴量の時間的分離を厳守:
  - aft_ret  : 後場(12:30~15:00)。前場/ORB/ギャップで予測可
  - on_ret   : 当日引け→翌日寄付(ON)。当日全データで予測可
  - next_ret : 翌日全日(寄付~引け)。当日全データで予測可

コスト前提: 片側2bps×往復 = 4bps (アウトライト)
"""

import psycopg2
import pandas as pd
import numpy as np
import warnings
from scipy import stats

warnings.filterwarnings('ignore')

PG_CONFIG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}

SECTOR_MAP = {
    '6920.T': '半導体', '6857.T': '半導体', '8035.T': '半導体',
    '6146.T': '半導体', '6273.T': '半導体', '6861.T': '半導体',
    '6762.T': '電機', '6752.T': '電機', '6503.T': '電機',
    '6954.T': '電機', '6501.T': '電機', '6594.T': '電機',
    '6702.T': 'IT', '6981.T': '電機',
    '5713.T': '非鉄', '5711.T': '非鉄', '5706.T': '非鉄',
    '5714.T': '非鉄', '5801.T': '非鉄', '5802.T': '非鉄', '5803.T': '非鉄',
    '9104.T': '海運', '9107.T': '海運', '9101.T': '海運',
    '1605.T': 'エネルギー', '5020.T': 'エネルギー', '5016.T': 'エネルギー',
    '7203.T': '自動車', '7267.T': '自動車', '7201.T': '自動車', '7270.T': '自動車',
    '5401.T': '鉄鋼', '5411.T': '鉄鋼', '4063.T': '化学',
    '8306.T': '銀行', '8316.T': '銀行', '8411.T': '銀行',
    '8604.T': '証券', '8308.T': '銀行',
    '8053.T': '商社', '8058.T': '商社', '8001.T': '商社',
    '8015.T': '商社', '8002.T': '商社', '8031.T': '商社',
    '9432.T': '通信', '9433.T': '通信', '9434.T': '通信',
    '7011.T': '重工', '7012.T': '重工', '7013.T': '重工',
    '9984.T': 'IT', '9983.T': '小売',
    '2914.T': '食品', '2503.T': '食品', '2502.T': '食品', '2801.T': '食品',
    '4661.T': 'レジャー', '6098.T': 'サービス',
    '8113.T': '生活用品', '9023.T': '鉄道',
    '4502.T': '製薬', '4503.T': '製薬', '4523.T': '製薬',
}


def load_all_stocks():
    conn = psycopg2.connect(**PG_CONFIG)
    query = """
        SELECT symbol, timestamp, open, high, low, close, volume
        FROM intraday_data
        WHERE interval = '1min' AND symbol LIKE '%.T'
          AND timestamp >= '2025-01-01'
        ORDER BY symbol, timestamp
    """
    print("データ読み込み中...")
    df = pd.read_sql(query, conn)
    conn.close()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['jst'] = df['timestamp'] + pd.Timedelta(hours=9)
    df = df.dropna(subset=['open', 'close'])
    print(f"  {len(df):,}行, {df['symbol'].nunique()}銘柄")
    return df


def build_daily_features(raw: pd.DataFrame) -> pd.DataFrame:
    """1分足から日次フィーチャーを構築 (look-ahead厳守版)"""
    records = []

    for symbol, g in raw.groupby('symbol'):
        g = g.set_index('jst').sort_index()

        for date, day in g.groupby(g.index.date):
            if len(day) < 30:
                continue

            # 前場(9:00-11:30)
            morning = day[
                (day.index.hour >= 9) & (
                    (day.index.hour < 11) |
                    ((day.index.hour == 11) & (day.index.minute <= 30))
                )
            ]
            # 後場(12:30-15:00)
            afternoon = day[
                ((day.index.hour == 12) & (day.index.minute >= 30)) |
                ((day.index.hour >= 13) & (day.index.hour < 15))
            ]

            if len(morning) < 10 or len(afternoon) < 5:
                continue

            # 基本
            open_price = morning['open'].iloc[0]
            day_close = day['close'].iloc[-1]
            morning_close = morning['close'].iloc[-1]
            aft_open = afternoon['open'].iloc[0]
            aft_close = afternoon['close'].iloc[-1]

            # ORB (最初15分/30分) — 前場データのみ
            orb15 = morning[morning.index <= morning.index[0] + pd.Timedelta(minutes=15)]
            orb30 = morning[morning.index <= morning.index[0] + pd.Timedelta(minutes=30)]
            orb15_high = orb15['high'].max() if len(orb15) > 0 else np.nan
            orb15_low = orb15['low'].min() if len(orb15) > 0 else np.nan
            orb30_high = orb30['high'].max() if len(orb30) > 0 else np.nan
            orb30_low = orb30['low'].min() if len(orb30) > 0 else np.nan

            # ---- リターン定義 (contamination-free) ----
            # gap_ret: prev_close→open (前日引けから寄付) — 前日データが必要なため後で計算
            # morning_ret: open→前場引け (前場のみ)
            morning_ret = (morning_close / open_price - 1) * 10000 if open_price > 0 else np.nan
            # aft_ret: 後場寄付→後場引け (完全に後場のみ)
            aft_ret = (aft_close / aft_open - 1) * 10000 if aft_open > 0 else np.nan
            # fullday_ret: 寄付→大引け (当日全日) — on_ret計算用。ルール予測には使わない
            fullday_ret = (day_close / open_price - 1) * 10000 if open_price > 0 else np.nan

            # ORBシグナル (前場中の動き — 後場予測に使う)
            # 前場の最高値 vs ORB高値を超えたか
            orb15_break_up = (morning_close > orb15_high) if not np.isnan(orb15_high) else False
            orb15_break_down = (morning_close < orb15_low) if not np.isnan(orb15_low) else False
            orb30_break_up = (morning_close > orb30_high) if not np.isnan(orb30_high) else False
            orb30_break_down = (morning_close < orb30_low) if not np.isnan(orb30_low) else False
            orb15_range = (orb15_high - orb15_low) / open_price * 10000 if open_price > 0 else np.nan
            orb30_range = (orb30_high - orb30_low) / open_price * 10000 if open_price > 0 else np.nan

            # 出来高 (前場のみ使用 — 後場を含まない)
            morning_vol = morning['volume'].sum()
            aft_vol = afternoon['volume'].sum()
            total_vol = day['volume'].sum()

            # 前場ボラティリティ (前場5分足)
            morning['min5'] = (morning.index.hour * 60 + morning.index.minute) // 5
            bar5 = morning.groupby('min5')['close'].last()
            morning_vol_bps = bar5.pct_change().std() * 10000 if len(bar5) > 3 else np.nan

            records.append({
                'symbol': symbol,
                'date': pd.Timestamp(date),
                'open': open_price,
                'close': day_close,
                'morning_close': morning_close,
                'aft_open': aft_open,
                'aft_close': aft_close,
                'morning_ret': morning_ret,
                'aft_ret': aft_ret,
                'fullday_ret': fullday_ret,
                'morning_vol': morning_vol,
                'aft_vol': aft_vol,
                'total_vol': total_vol,
                'orb15_high': orb15_high,
                'orb15_low': orb15_low,
                'orb30_high': orb30_high,
                'orb30_low': orb30_low,
                'orb15_break_up': orb15_break_up,
                'orb15_break_down': orb15_break_down,
                'orb30_break_up': orb30_break_up,
                'orb30_break_down': orb30_break_down,
                'orb15_range': orb15_range,
                'orb30_range': orb30_range,
                'morning_vol_bps': morning_vol_bps,
            })

    return pd.DataFrame(records)


def add_lagged_features(df: pd.DataFrame) -> pd.DataFrame:
    """前日フィーチャーを追加 + ON/翌日リターン計算"""
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)

    for sym, g in df.groupby('symbol'):
        idx = g.index
        df.loc[idx, 'prev_close'] = g['close'].shift(1).values
        df.loc[idx, 'prev_fullday_ret'] = g['fullday_ret'].shift(1).values
        df.loc[idx, 'prev_morning_ret'] = g['morning_ret'].shift(1).values
        df.loc[idx, 'prev_aft_ret'] = g['aft_ret'].shift(1).values
        df.loc[idx, 'prev_morning_vol'] = g['morning_vol'].shift(1).values
        df.loc[idx, 'prev_vol5d'] = g['total_vol'].shift(1).rolling(5, min_periods=3).mean().values
        df.loc[idx, 'prev_morning_vol_bps'] = g['morning_vol_bps'].shift(1).values
        df.loc[idx, 'vol5d_bps'] = g['morning_vol_bps'].shift(1).rolling(5, min_periods=3).mean().values

        # ON_ret: 当日引け → 翌日寄付
        df.loc[idx, 'on_ret'] = (g['open'].shift(-1) / g['close'] - 1).values * 10000
        # next_fullday_ret: 翌日全日
        df.loc[idx, 'next_fullday_ret'] = g['fullday_ret'].shift(-1).values
        # next_morning_ret: 翌日前場
        df.loc[idx, 'next_morning_ret'] = g['morning_ret'].shift(-1).values
        # next_aft_ret: 翌日後場
        df.loc[idx, 'next_aft_ret'] = g['aft_ret'].shift(-1).values

    # ギャップリターン (前日引け→当日寄付) — ルール特徴量として使用可
    df['gap_ret'] = (df['open'] / df['prev_close'] - 1) * 10000

    # 出来高比 (当日前場 vs 前5日平均 — 前場段階で計算可)
    df['morning_vol_ratio'] = df['morning_vol'] / df['prev_vol5d'].replace(0, np.nan)

    # 前場ボラ比
    df['morning_vol_bps_ratio'] = df['morning_vol_bps'] / df['vol5d_bps'].replace(0, np.nan)

    # 前場レンジ幅 (open~morning_close)
    df['morning_range'] = abs(df['morning_ret'])

    return df.dropna(subset=['prev_close', 'gap_ret'])


def test_rule(df, condition, ret_col, rule_name, cost_bps=4, direction=1):
    """単一ルールのバックテスト"""
    sub = df[condition & df[ret_col].notna()].copy()
    if len(sub) < 50:
        return None

    returns = sub[ret_col].values * direction
    net_returns = returns - cost_bps

    n = len(returns)
    mean_raw = returns.mean()
    mean_net = net_returns.mean()
    std = returns.std()
    t_stat, p_val = stats.ttest_1samp(returns, 0)
    wr = (returns > 0).mean() * 100
    neg_sum = abs(returns[returns <= 0].sum())
    pf = returns[returns > 0].sum() / neg_sum if neg_sum > 0 else np.inf
    sharpe = mean_net / std * np.sqrt(252) if std > 0 else 0

    return {
        'rule': rule_name,
        'target': ret_col,
        'N': n,
        'mean_raw_bps': round(mean_raw, 1),
        'mean_net_bps': round(mean_net, 1),
        'std_bps': round(std, 1),
        't_stat': round(t_stat, 2),
        'p_val': round(p_val, 3),
        'win_rate': round(wr, 1),
        'profit_factor': round(pf, 2),
        'sharpe_net': round(sharpe, 2),
        'significant': (p_val < 0.05) and (mean_net > 0),
    }


def run_all_rules(df):
    results = []

    def add(r):
        if r:
            results.append(r)

    # ===========================================================
    # A. Gap系 — gap_ret (前日引け→寄付) を特徴量に使う
    #    target: aft_ret (後場、look-aheadなし)
    # ===========================================================
    # A1: ギャップアップ大(≥+50bps) → 後場買い (モメンタム)
    add(test_rule(df, df['gap_ret'] >= 50, 'aft_ret', 'A1_GapUp_Lg_Aft_Momentum'))

    # A2: ギャップアップ大 → 後場売り (リバーサル)
    add(test_rule(df, df['gap_ret'] >= 50, 'aft_ret', 'A2_GapUp_Lg_Aft_Reversal', direction=-1))

    # A3: ギャップアップ小(+15~+50bps) → 後場買い
    add(test_rule(df, (df['gap_ret'] >= 15) & (df['gap_ret'] < 50), 'aft_ret', 'A3_GapUp_Sm_Aft_Buy'))

    # A4: ギャップダウン大(≤-50bps) → 後場売り (モメンタム)
    add(test_rule(df, df['gap_ret'] <= -50, 'aft_ret', 'A4_GapDown_Lg_Aft_Short', direction=-1))

    # A5: ギャップダウン大 → 後場買い (リバーサル)
    add(test_rule(df, df['gap_ret'] <= -50, 'aft_ret', 'A5_GapDown_Lg_Aft_Reversal'))

    # A6: ギャップアップ大 → 翌日全日買い
    add(test_rule(df, df['gap_ret'] >= 50, 'next_fullday_ret', 'A6_GapUp_Lg_NextDay'))

    # A7: ギャップダウン大 → 翌日全日買い (リバーサル)
    add(test_rule(df, df['gap_ret'] <= -50, 'next_fullday_ret', 'A7_GapDown_Lg_NextDay_Buy'))

    # A8: ギャップアップ大 → ONリターン(当日引け→翌日寄付)
    add(test_rule(df, df['gap_ret'] >= 50, 'on_ret', 'A8_GapUp_Lg_ON'))

    # ===========================================================
    # B. 前日価格モメンタム/リバーサル
    #    target: next_fullday_ret / on_ret / next_morning_ret
    # ===========================================================
    # B1: 前日強(≥+100bps) → 翌日全日買い
    add(test_rule(df, df['prev_fullday_ret'] >= 100, 'next_fullday_ret', 'B1_PrevStrong_NextBuy'))

    # B2: 前日強 → 翌日全日売り (mean reversion)
    add(test_rule(df, df['prev_fullday_ret'] >= 100, 'next_fullday_ret', 'B2_PrevStrong_NextShort', direction=-1))

    # B3: 前日弱(≤-100bps) → 翌日全日買い (mean reversion)
    add(test_rule(df, df['prev_fullday_ret'] <= -100, 'next_fullday_ret', 'B3_PrevWeak_NextBuy'))

    # B4: 前日弱 → 翌日全日売り (モメンタム)
    add(test_rule(df, df['prev_fullday_ret'] <= -100, 'next_fullday_ret', 'B4_PrevWeak_NextShort', direction=-1))

    # B5: 前日強 → 翌日ON買い
    add(test_rule(df, df['prev_fullday_ret'] >= 100, 'on_ret', 'B5_PrevStrong_ON_Buy'))

    # B6: 前日弱 → 翌日ON買い
    add(test_rule(df, df['prev_fullday_ret'] <= -100, 'on_ret', 'B6_PrevWeak_ON_Reversal'))

    # B7: 前日前場強(≥+50bps) → 翌日全日
    add(test_rule(df, df['prev_morning_ret'] >= 50, 'next_fullday_ret', 'B7_PrevMorningStrong_NextBuy'))

    # B8: 前日後場強 → 翌日全日
    add(test_rule(df, df['prev_aft_ret'] >= 50, 'next_fullday_ret', 'B8_PrevAftStrong_NextBuy'))

    # B9: 前日後場弱 → 翌日全日買い (reversal)
    add(test_rule(df, df['prev_aft_ret'] <= -50, 'next_fullday_ret', 'B9_PrevAftWeak_NextBuy'))

    # ===========================================================
    # C. 前場→後場引き継ぎ
    #    target: aft_ret (後場。前場情報のみ使用)
    # ===========================================================
    # C1: 前場強(≥+30bps) → 後場買い (モメンタム)
    add(test_rule(df, df['morning_ret'] >= 30, 'aft_ret', 'C1_MorningStrong_AftBuy'))

    # C2: 前場強 → 後場売り (ランチリバーサル)
    add(test_rule(df, df['morning_ret'] >= 30, 'aft_ret', 'C2_MorningStrong_AftShort', direction=-1))

    # C3: 前場弱(≤-30bps) → 後場買い (午後リバーサル)
    add(test_rule(df, df['morning_ret'] <= -30, 'aft_ret', 'C3_MorningWeak_AftBuy'))

    # C4: 前場弱 → 後場売り (モメンタム継続)
    add(test_rule(df, df['morning_ret'] <= -30, 'aft_ret', 'C4_MorningWeak_AftShort', direction=-1))

    # C5: 前場レンジ小(±10bps以内) → 後場方向性
    add(test_rule(df, df['morning_range'] <= 10, 'aft_ret', 'C5_MorningFlat_AftAny'))

    # C6: ギャップアップ × 前場さらに上昇 → 後場買い
    add(test_rule(df, (df['gap_ret'] >= 20) & (df['morning_ret'] >= 20), 'aft_ret',
                  'C6_GapUp_MornUp_AftBuy'))

    # C7: ギャップアップ × 前場反落 → 後場売り (ギャップ埋め)
    add(test_rule(df, (df['gap_ret'] >= 20) & (df['morning_ret'] <= -10), 'aft_ret',
                  'C7_GapUp_MornDown_GapFill_Short', direction=-1))

    # C8: ギャップダウン × 前場上昇 → 後場買い (ギャップ埋め)
    add(test_rule(df, (df['gap_ret'] <= -20) & (df['morning_ret'] >= 10), 'aft_ret',
                  'C8_GapDown_MornUp_GapFill_Buy'))

    # C9: ギャップダウン × 前場も下落 → 後場売り
    add(test_rule(df, (df['gap_ret'] <= -20) & (df['morning_ret'] <= -10), 'aft_ret',
                  'C9_GapDown_MornDown_AftShort', direction=-1))

    # C10: 前場強 → 翌日前場
    add(test_rule(df, df['morning_ret'] >= 50, 'next_morning_ret', 'C10_MorningStrong_NextMorning'))

    # ===========================================================
    # D. ORB系 — オープンレンジ後の後場エントリー
    #    target: aft_ret
    # ===========================================================
    # D1: ORB15 上抜け(前場引け > ORB高値) → 後場買い
    add(test_rule(df, df['orb15_break_up'], 'aft_ret', 'D1_ORB15_Up_AftBuy'))

    # D2: ORB15 下抜け → 後場売り
    add(test_rule(df, df['orb15_break_down'], 'aft_ret', 'D2_ORB15_Down_AftShort', direction=-1))

    # D3: ORB30 上抜け → 後場買い
    add(test_rule(df, df['orb30_break_up'], 'aft_ret', 'D3_ORB30_Up_AftBuy'))

    # D4: ORB30 下抜け → 後場売り
    add(test_rule(df, df['orb30_break_down'], 'aft_ret', 'D4_ORB30_Down_AftShort', direction=-1))

    # D5: ORB15 レンジ広い(≥30bps) × 上抜け → 後場買い (強いシグナル)
    add(test_rule(df, (df['orb15_range'] >= 30) & df['orb15_break_up'], 'aft_ret',
                  'D5_ORB15_Wide_Up_AftBuy'))

    # D6: ORB15 レンジ広い × 下抜け → 後場売り
    add(test_rule(df, (df['orb15_range'] >= 30) & df['orb15_break_down'], 'aft_ret',
                  'D6_ORB15_Wide_Down_AftShort', direction=-1))

    # D7: ORB15 レンジ狭い(≤10bps) → 後場どちらへ動くか
    add(test_rule(df, df['orb15_range'] <= 10, 'aft_ret', 'D7_ORB15_Narrow_AftAny'))

    # D8: ORB30上抜け → ON買い (翌日持ち越し)
    add(test_rule(df, df['orb30_break_up'], 'on_ret', 'D8_ORB30_Up_ON_Buy'))

    # ===========================================================
    # E. 出来高シグナル (前場出来高のみ使用)
    # ===========================================================
    # E1: 前場出来高2倍超 × 前場上昇 → 後場買い
    add(test_rule(df, (df['morning_vol_ratio'] >= 2.0) & (df['morning_ret'] > 0), 'aft_ret',
                  'E1_HighVol_MornUp_AftBuy'))

    # E2: 前場出来高2倍超 × 前場下落 → 後場買い (強いリバーサル?)
    add(test_rule(df, (df['morning_vol_ratio'] >= 2.0) & (df['morning_ret'] < 0), 'aft_ret',
                  'E2_HighVol_MornDown_AftBuy'))

    # E3: 前場出来高2倍超 × 前場下落 → 後場売り
    add(test_rule(df, (df['morning_vol_ratio'] >= 2.0) & (df['morning_ret'] < 0), 'aft_ret',
                  'E3_HighVol_MornDown_AftShort', direction=-1))

    # E4: 前場出来高2倍超 → 翌日全日
    add(test_rule(df, df['morning_vol_ratio'] >= 2.0, 'next_fullday_ret',
                  'E4_HighVol_NextDay'))

    # E5: 前場出来高少ない(0.5以下) → 後場
    add(test_rule(df, df['morning_vol_ratio'] <= 0.5, 'aft_ret', 'E5_LowVol_AftRet'))

    # E6: ギャップアップ × 前場出来高急増 → 後場買い
    add(test_rule(df, (df['gap_ret'] >= 20) & (df['morning_vol_ratio'] >= 2.0), 'aft_ret',
                  'E6_GapUp_HighVol_AftBuy'))

    # E7: 前場ボラ急増(前5日比2倍) → 後場
    add(test_rule(df, df['morning_vol_bps_ratio'] >= 2.0, 'aft_ret',
                  'E7_HighVolBps_AftRet'))

    # ===========================================================
    # F. ON (オーバーナイト)戦略
    #    target: on_ret (当日引け→翌日寄付)
    # ===========================================================
    # F1: 前場強 → ON買い
    add(test_rule(df, df['morning_ret'] >= 50, 'on_ret', 'F1_MorningStrong_ON_Buy'))

    # F2: 前場弱 → ON買い (reversal)
    add(test_rule(df, df['morning_ret'] <= -50, 'on_ret', 'F2_MorningWeak_ON_Buy'))

    # F3: ORB上抜け + 前場強 → ON買い
    add(test_rule(df, df['orb30_break_up'] & (df['morning_ret'] >= 20), 'on_ret',
                  'F3_ORB30Up_MornStrong_ON'))

    # F4: 前日弱 × ギャップ下落 → ON買い (2日連続下落後反転)
    add(test_rule(df, (df['prev_fullday_ret'] <= -100) & (df['gap_ret'] <= -20), 'on_ret',
                  'F4_PrevWeak_GapDown_ON_Buy'))

    # F5: 後場強(≥+30bps) → ON買い
    add(test_rule(df, df['aft_ret'] >= 30, 'on_ret', 'F5_AftStrong_ON_Buy'))

    # F6: 後場弱(≤-30bps) → ON買い (反転)
    add(test_rule(df, df['aft_ret'] <= -30, 'on_ret', 'F6_AftWeak_ON_Buy'))

    # ===========================================================
    # G. 複合条件ルール
    # ===========================================================
    # G1: ギャップアップ大 × ORB上抜け × 前場出来高増 → 後場買い (全条件揃い)
    add(test_rule(df, (df['gap_ret'] >= 30) & df['orb15_break_up'] & (df['morning_vol_ratio'] >= 1.5),
                  'aft_ret', 'G1_GapUp_ORB_HighVol_AftBuy'))

    # G2: 前日弱 × ギャップダウン → 後場反転買い
    add(test_rule(df, (df['prev_fullday_ret'] <= -100) & (df['gap_ret'] <= -30), 'aft_ret',
                  'G2_PrevWeak_GapDown_AftBuy'))

    # G3: 前日弱 × ギャップダウン → 翌日全日買い
    add(test_rule(df, (df['prev_fullday_ret'] <= -100) & (df['gap_ret'] <= -30), 'next_fullday_ret',
                  'G3_PrevWeak_GapDown_NextBuy'))

    # G4: 連続下落後(前日 and ギャップ) × ORB下抜け → 後場売り
    add(test_rule(df, (df['prev_fullday_ret'] <= -50) & df['orb30_break_down'], 'aft_ret',
                  'G4_PrevWeak_ORBDown_AftShort', direction=-1))

    return pd.DataFrame(results)


def rank_rules(results: pd.DataFrame) -> pd.DataFrame:
    res = results.copy()
    # スコア: mean_net + t_stat重視
    t_std = res['t_stat'].std()
    m_std = res['mean_net_bps'].std()
    s_std = res['sharpe_net'].abs().max()
    res['score'] = (
        (res['mean_net_bps'] / m_std if m_std > 0 else 0) * 0.3 +
        (res['t_stat'] / t_std if t_std > 0 else 0) * 0.4 +
        (res['sharpe_net'] / s_std if s_std > 0 else 0) * 0.3
    )
    return res.sort_values('score', ascending=False)


if __name__ == '__main__':
    import time
    t0 = time.time()

    raw = load_all_stocks()

    print("日次フィーチャー構築中...")
    daily = build_daily_features(raw)
    print(f"  {len(daily):,}日次レコード, {daily['symbol'].nunique()}銘柄")

    print("ラグフィーチャー追加中...")
    daily = add_lagged_features(daily)
    daily['sector'] = daily['symbol'].map(SECTOR_MAP).fillna('その他')
    print(f"  有効レコード: {len(daily):,}")

    print("ルール検証中...")
    results = run_all_rules(daily)
    ranked = rank_rules(results)

    print(f"\n=== 全ルール結果 (コスト4bps差引後) ===")
    cols = ['rule', 'target', 'N', 'mean_raw_bps', 'mean_net_bps', 't_stat', 'p_val',
            'win_rate', 'sharpe_net', 'significant']
    print(ranked[cols].to_string(index=False))

    print(f"\n=== 有望ルール (p<0.05 かつ net>0) ===")
    sig = ranked[ranked['significant'] == True]
    print(sig[cols].to_string(index=False))

    ranked.to_csv('results_all_rules.csv', index=False)
    sig.to_csv('results_significant.csv', index=False)

    elapsed = time.time() - t0
    print(f"\n完了 ({elapsed:.1f}秒)")
    print(f"ルール数: {len(results)}, 有望ルール: {len(sig)}")
