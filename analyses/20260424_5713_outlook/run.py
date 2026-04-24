"""
住友金属鉱山 (5713.T) の今後の動き分析

これまでの出来高分析 (20260424_*) で得た知見を住友金属鉱山に適用:

1. C1スパイク戦略 (最強シグナル): vol>=20日median*3 かつ OC<-0.5% → T+10 +306 bps (集約)
   → 5713に絞った過去成績を検証 + 直近該当イベントの有無
2. 日次 vol_ratio と OBV の推移
3. 直近60日の価格・出来高パターンと現在のシグナル状態
4. イントラデイ検証: 最新日のギャップ×出来高 (S5) と 11:30 div

出力:
- 直近60営業日のチャート (価格、出来高、OBV)
- 過去8年の 5713 C1 スパイクイベント一覧 + 成績
- 現時点のシグナル状態と今後10日間の見通し
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '20260421_common')))
import mdutil as U
import pandas as pd
import numpy as np
import psycopg2
from datetime import timedelta
plt = U.matplotlib_jp()

SYM = "5713.T"
NAME = "住友金属鉱山"
OUT_DIR = os.path.dirname(__file__)
START = "2018-01-01"
END = "2026-04-23"


def load_daily(sym):
    conn = psycopg2.connect(**U.PG_CONFIG)
    q = (f"SELECT trade_date, open, high, low, close, volume FROM daily_data "
         f"WHERE symbol='{sym}' AND trade_date>='{START}' AND trade_date<='{END}' "
         f"ORDER BY trade_date")
    df = pd.read_sql(q, conn); conn.close()
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df = df.set_index('trade_date').sort_index()
    for c in ['open','high','low','close','volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df.dropna(subset=['open','close','volume'])


def add_features(df):
    df = df.copy()
    df['ret_oc'] = df['close']/df['open'] - 1
    df['ret_cc'] = df['close']/df['close'].shift(1) - 1
    # 出来高
    df['vol_ma20'] = df['volume'].rolling(20, min_periods=10).mean().shift(1)
    df['vol_med20'] = df['volume'].rolling(20, min_periods=10).median().shift(1)
    df['vol_ratio'] = df['volume'] / df['vol_ma20']
    df['vol_ratio_med'] = df['volume'] / df['vol_med20']
    # OBV
    sign = np.sign(df['close'] - df['close'].shift(1))
    df['signed_vol'] = df['volume'] * sign
    df['obv'] = df['signed_vol'].cumsum()
    df['obv_mom20'] = df['obv'] - df['obv'].shift(20)
    df['price_mom20'] = df['close'].pct_change(20)
    # スパイク判定
    df['is_spike_3x'] = df['vol_ratio_med'] >= 3.0
    df['is_spike_2x'] = df['vol_ratio_med'] >= 2.0
    # 前向きリターン
    for h in [1,3,5,10,20]:
        df[f'fwd_{h}'] = df['close'].shift(-h) / df['close'] - 1
    return df


# ============ 1. C1 スパイク戦略の 5713 単独成績 ============

def backtest_C1(df, vol_mult=3.0, oc_threshold=-0.005, hold=10):
    """C1: vol_spike + 下落 → T+1 open Long, T+10 close 決済"""
    trades = []
    for i, (d, row) in enumerate(df.iterrows()):
        if row['vol_ratio_med'] < vol_mult: continue
        if pd.isna(row['ret_oc']) or row['ret_oc'] > oc_threshold: continue
        pos = df.index.get_loc(d)
        if pos >= len(df) - hold - 1: continue
        entry = df.iloc[pos+1]['open']
        exit_ = df.iloc[pos+hold]['close']
        if pd.isna(entry) or pd.isna(exit_) or entry <= 0: continue
        ret_bps = (exit_/entry - 1) * 10000
        trades.append({
            'event_date': d, 'event_close': row['close'],
            'event_ret_oc_bps': row['ret_oc']*10000,
            'vol': row['volume'], 'vol_ratio': row['vol_ratio_med'],
            'entry_date': df.index[pos+1], 'entry_price': entry,
            'exit_date': df.index[pos+hold], 'exit_price': exit_,
            'ret_bps': ret_bps,
        })
    return pd.DataFrame(trades)


def backtest_C2(df, vol_mult=3.0, oc_threshold=0.005, hold=10):
    """上昇スパイク版: T+1 open Long, T+10 close 決済"""
    trades = []
    for i, (d, row) in enumerate(df.iterrows()):
        if row['vol_ratio_med'] < vol_mult: continue
        if pd.isna(row['ret_oc']) or row['ret_oc'] < oc_threshold: continue
        pos = df.index.get_loc(d)
        if pos >= len(df) - hold - 1: continue
        entry = df.iloc[pos+1]['open']
        exit_ = df.iloc[pos+hold]['close']
        if pd.isna(entry) or pd.isna(exit_) or entry <= 0: continue
        ret_bps = (exit_/entry - 1) * 10000
        trades.append({
            'event_date': d, 'event_close': row['close'],
            'event_ret_oc_bps': row['ret_oc']*10000,
            'vol': row['volume'], 'vol_ratio': row['vol_ratio_med'],
            'entry_date': df.index[pos+1], 'entry_price': entry,
            'exit_date': df.index[pos+hold], 'exit_price': exit_,
            'ret_bps': ret_bps,
        })
    return pd.DataFrame(trades)


# ============ 2. 現時点のシグナル評価 ============

def evaluate_current(df, today=None):
    """最終営業日のシグナル状態を評価"""
    if today is None:
        today = df.index[-1]
    row = df.loc[today]
    report = []
    report.append(f"=== {today.strftime('%Y-%m-%d')} 時点のシグナル評価 ({NAME}) ===")
    report.append(f"  終値: {row['close']:,.0f}円")
    report.append(f"  出来高: {row['volume']:,.0f}")
    report.append(f"  当日ret_oc: {row['ret_oc']*10000:+.1f} bps")
    report.append(f"  vol_ratio (/20日中央値): {row['vol_ratio_med']:.2f}x")
    report.append(f"  vol_ratio (/20日平均): {row['vol_ratio']:.2f}x")
    report.append(f"  20日価格モメンタム: {row['price_mom20']*10000:+.0f} bps")
    report.append(f"  20日OBVモメンタム: {row['obv_mom20']:,.0f}")

    # C1 条件チェック
    spike_c1 = row['vol_ratio_med'] >= 3.0 and row['ret_oc'] <= -0.005
    spike_c2 = row['vol_ratio_med'] >= 3.0 and row['ret_oc'] >= 0.005
    spike_soft = row['vol_ratio_med'] >= 2.0
    report.append(f"\n  [C1 条件 (3x + -0.5%)]: {'🔥 発動' if spike_c1 else '未発動'}")
    report.append(f"  [C2 条件 (3x + +0.5%)]: {'🔥 発動' if spike_c2 else '未発動'}")
    report.append(f"  [緩和条件 (2x)]: {'注意' if spike_soft else '通常'}")

    # 過去20日のC1/C2該当日
    recent = df.tail(20)
    recent_c1 = recent[(recent['vol_ratio_med'] >= 3.0) & (recent['ret_oc'] <= -0.005)]
    recent_c2 = recent[(recent['vol_ratio_med'] >= 3.0) & (recent['ret_oc'] >= 0.005)]
    recent_2x = recent[recent['vol_ratio_med'] >= 2.0]
    report.append(f"\n  直近20営業日 C1 該当: {len(recent_c1)}回")
    if len(recent_c1) > 0:
        for d, r in recent_c1.iterrows():
            report.append(f"    {d.strftime('%Y-%m-%d')}: close={r['close']:,.0f} "
                         f"vol_ratio={r['vol_ratio_med']:.2f}x ret_oc={r['ret_oc']*10000:+.0f}bps")
    report.append(f"  直近20営業日 C2 該当: {len(recent_c2)}回")
    if len(recent_c2) > 0:
        for d, r in recent_c2.iterrows():
            report.append(f"    {d.strftime('%Y-%m-%d')}: close={r['close']:,.0f} "
                         f"vol_ratio={r['vol_ratio_med']:.2f}x ret_oc={r['ret_oc']*10000:+.0f}bps")
    report.append(f"  直近20営業日 2x+ 出来高日: {len(recent_2x)}回")

    return "\n".join(report)


def main():
    print("=" * 70)
    print(f" {NAME} ({SYM}) 今後の動き分析")
    print("=" * 70)
    df = load_daily(SYM)
    df = add_features(df)
    print(f"データ: {df.index[0].date()} 〜 {df.index[-1].date()}, {len(df)} days")

    # === 1. C1/C2 スパイク戦略の過去成績 ===
    print("\n" + "=" * 70)
    print(" [1] 過去8年 C1/C2 スパイク戦略の 5713 単独成績")
    print("=" * 70)

    c1 = backtest_C1(df, vol_mult=3.0, oc_threshold=-0.005, hold=10)
    c2 = backtest_C2(df, vol_mult=3.0, oc_threshold=+0.005, hold=10)

    def stats_of(t, label):
        if len(t) == 0:
            print(f"  {label}: N=0 (該当なし)")
            return None
        r = t['ret_bps'].values
        m, s = r.mean(), r.std()
        tstat = m/(s/np.sqrt(len(r))) if s>0 else 0
        pf_pos = r[r>0].sum()
        pf_neg = abs(r[r<=0].sum())
        pf = pf_pos/pf_neg if pf_neg>0 else np.inf
        print(f"  {label}: N={len(r)}  mean={m:+.0f}bps  std={s:.0f}  "
              f"WR={(r>0).mean()*100:.0f}%  PF={pf:.2f}  t={tstat:+.2f}  "
              f"best={r.max():+.0f} worst={r.min():+.0f}")

    stats_of(c1, "C1 (3x + 下落)    T+10 hold")
    stats_of(c2, "C2 (3x + 上昇)    T+10 hold")

    # 緩和条件 (2x)
    print("\n  [緩和] 2x+ 閾値版:")
    c1_soft = backtest_C1(df, vol_mult=2.0, oc_threshold=-0.005, hold=10)
    c2_soft = backtest_C2(df, vol_mult=2.0, oc_threshold=+0.005, hold=10)
    stats_of(c1_soft, "C1 (2x + 下落)    T+10 hold")
    stats_of(c2_soft, "C2 (2x + 上昇)    T+10 hold")

    # 保有期間変化
    print("\n  [保有期間感応度] C1 (2x+下落):")
    for h in [3, 5, 10, 15, 20]:
        c_h = backtest_C1(df, vol_mult=2.0, oc_threshold=-0.005, hold=h)
        stats_of(c_h, f"   T+{h} hold")

    print("\n  [C1 発生イベント一覧 (3x+下落)]:")
    if len(c1) > 0:
        print(c1[['event_date','event_close','event_ret_oc_bps','vol_ratio','entry_price','exit_price','ret_bps']].to_string(index=False, formatters={
            'event_date': lambda x: pd.Timestamp(x).strftime('%Y-%m-%d'),
            'event_close': '{:,.0f}'.format,
            'event_ret_oc_bps': '{:+.0f}'.format,
            'vol_ratio': '{:.2f}'.format,
            'entry_price': '{:,.0f}'.format,
            'exit_price': '{:,.0f}'.format,
            'ret_bps': '{:+.0f}'.format,
        }))

    # === 2. 現時点のシグナル評価 ===
    print("\n" + "=" * 70)
    print(" [2] 現時点のシグナル評価")
    print("=" * 70)
    print(evaluate_current(df))

    # === 3. 直近60営業日の pattern ===
    print("\n" + "=" * 70)
    print(" [3] 直近60営業日のサマリー")
    print("=" * 70)
    recent60 = df.tail(60).copy()
    print(f"  期間: {recent60.index[0].date()} 〜 {recent60.index[-1].date()}")
    print(f"  高値: {recent60['high'].max():,.0f} ({recent60['high'].idxmax().date()})")
    print(f"  安値: {recent60['low'].min():,.0f} ({recent60['low'].idxmin().date()})")
    print(f"  期間リターン: {(recent60['close'].iloc[-1]/recent60['close'].iloc[0]-1)*100:+.1f}%")
    print(f"  高値 → 現在: {(recent60['close'].iloc[-1]/recent60['high'].max()-1)*100:+.1f}%")
    print(f"  安値 → 現在: {(recent60['close'].iloc[-1]/recent60['low'].min()-1)*100:+.1f}%")
    spike_days_60 = recent60[recent60['vol_ratio_med']>=2.0]
    print(f"  2x+ 出来高日: {len(spike_days_60)} 回")
    c1_days_60 = recent60[(recent60['vol_ratio_med']>=3.0) & (recent60['ret_oc']<=-0.005)]
    c2_days_60 = recent60[(recent60['vol_ratio_med']>=3.0) & (recent60['ret_oc']>=0.005)]
    print(f"  C1 (3x+下落) 日: {len(c1_days_60)}")
    print(f"  C2 (3x+上昇) 日: {len(c2_days_60)}")
    if len(c1_days_60) > 0:
        print("  C1 日の内訳:")
        for d, r in c1_days_60.iterrows():
            fwd10 = df.loc[d, 'fwd_10'] if 'fwd_10' in df.columns else np.nan
            print(f"    {d.date()}: close={r['close']:,.0f} vol_ratio={r['vol_ratio_med']:.1f}x "
                 f"ret_oc={r['ret_oc']*10000:+.0f}bps → T+10 fwd={fwd10*10000 if not pd.isna(fwd10) else float('nan'):+.0f}bps")

    # === 4. 可視化 ===
    print("\n可視化中...")
    fig = plt.figure(figsize=(18, 16))
    gs = fig.add_gridspec(4, 2, hspace=0.45, wspace=0.28)

    recent120 = df.tail(120).copy()

    # (0,:) 価格 + スパイクマーカー
    ax = fig.add_subplot(gs[0, :])
    ax.plot(recent120.index, recent120['close'], color='#1f77b4', lw=1.5, label='close')
    # 移動平均
    ax.plot(recent120.index, recent120['close'].rolling(20).mean(),
            color='#ff7f0e', lw=1.0, alpha=0.6, label='MA20')
    # スパイク日
    spike_c1 = recent120[(recent120['vol_ratio_med']>=3.0) & (recent120['ret_oc']<=-0.005)]
    spike_c2 = recent120[(recent120['vol_ratio_med']>=3.0) & (recent120['ret_oc']>=0.005)]
    spike_2x_up = recent120[(recent120['vol_ratio_med']>=2.0) & (recent120['vol_ratio_med']<3.0) & (recent120['ret_oc']>=0.005)]
    spike_2x_dn = recent120[(recent120['vol_ratio_med']>=2.0) & (recent120['vol_ratio_med']<3.0) & (recent120['ret_oc']<=-0.005)]
    ax.scatter(spike_c1.index, spike_c1['close'], color='red', s=100, marker='v',
               label=f'C1 (3x+下落) N={len(spike_c1)}', zorder=5, edgecolors='black')
    ax.scatter(spike_c2.index, spike_c2['close'], color='green', s=100, marker='^',
               label=f'C2 (3x+上昇) N={len(spike_c2)}', zorder=5, edgecolors='black')
    ax.scatter(spike_2x_dn.index, spike_2x_dn['close'], color='orange', s=40, marker='v',
               label=f'2x+下落 N={len(spike_2x_dn)}', alpha=0.6)
    ax.scatter(spike_2x_up.index, spike_2x_up['close'], color='lightgreen', s=40, marker='^',
               label=f'2x+上昇 N={len(spike_2x_up)}', alpha=0.6)
    # 現在地
    last = recent120.iloc[-1]
    ax.annotate(f"現在: {last['close']:,.0f}円",
                xy=(recent120.index[-1], last['close']),
                xytext=(10, 10), textcoords='offset points',
                fontsize=11, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='black'))
    ax.set_title(f'{NAME} ({SYM}) 直近120営業日 close + 出来高スパイク日',
                 fontsize=12, fontweight='bold')
    ax.set_ylabel('価格 (円)'); ax.legend(loc='lower left', fontsize=9); ax.grid(alpha=0.3)

    # (1,:) 出来高 + vol_ratio
    ax = fig.add_subplot(gs[1, :])
    colors = ['#d62728' if r < 0 else '#2ca02c' for r in recent120['ret_oc']]
    ax.bar(recent120.index, recent120['volume'], color=colors, alpha=0.6, width=0.8)
    ax2 = ax.twinx()
    ax2.plot(recent120.index, recent120['vol_ratio_med'], color='purple', lw=1.5, label='vol_ratio')
    ax2.axhline(3.0, color='red', ls='--', lw=0.8, alpha=0.7, label='C1/C2 threshold (3x)')
    ax2.axhline(2.0, color='orange', ls='--', lw=0.8, alpha=0.5, label='2x threshold')
    ax2.axhline(1.0, color='gray', ls='-', lw=0.5)
    ax.set_title('出来高 (陽線=緑/陰線=赤) と vol_ratio (紫)', fontsize=11, fontweight='bold')
    ax.set_ylabel('出来高'); ax2.set_ylabel('vol_ratio (vs 20日中央値)')
    ax2.legend(loc='upper right', fontsize=9); ax.grid(alpha=0.3)

    # (2,0) OBV
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(recent120.index, recent120['obv'], color='#2ca02c', lw=1.5, label='OBV')
    ax2 = ax.twinx()
    ax2.plot(recent120.index, recent120['close'], color='#1f77b4', lw=1.0, label='close', alpha=0.7)
    ax.set_title('OBV vs 価格', fontsize=11, fontweight='bold')
    ax.set_ylabel('OBV'); ax2.set_ylabel('close')
    ax.grid(alpha=0.3)

    # (2,1) 20日モメンタム (price vs OBV)
    ax = fig.add_subplot(gs[2, 1])
    ax.plot(recent120.index, recent120['price_mom20']*100, color='#1f77b4', lw=1.5, label='price_mom20 %')
    ax2 = ax.twinx()
    ax2.plot(recent120.index, recent120['obv_mom20']/1e6, color='#2ca02c', lw=1.5, label='obv_mom20 (M)')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_title('20日モメンタム (price % vs OBV M)', fontsize=11, fontweight='bold')
    ax.set_ylabel('price_mom20 (%)'); ax2.set_ylabel('obv_mom20 (million)')
    ax.legend(loc='upper left', fontsize=9); ax2.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3)

    # (3,0) C1 イベント別 T+10 リターン分布 (過去 8 年)
    ax = fig.add_subplot(gs[3, 0])
    if len(c1) > 0:
        ax.hist(c1['ret_bps'], bins=10, color='#d62728', alpha=0.7, edgecolor='black')
        ax.axvline(c1['ret_bps'].mean(), color='black', lw=2, ls='--',
                   label=f'mean={c1["ret_bps"].mean():+.0f}bps')
        ax.axvline(0, color='gray', lw=1)
    ax.set_title(f'C1 (3x+下落) T+10 分布 (過去8年 N={len(c1)})',
                 fontsize=10, fontweight='bold')
    ax.set_xlabel('T+10 ret_bps'); ax.set_ylabel('count')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # (3,1) C1 累積損益カーブ (過去 8 年)
    ax = fig.add_subplot(gs[3, 1])
    if len(c1) > 0:
        c1_sorted = c1.sort_values('event_date').copy()
        c1_sorted['cum_bps'] = c1_sorted['ret_bps'].cumsum()
        ax.plot(c1_sorted['event_date'], c1_sorted['cum_bps'], 'o-',
                color='#d62728', lw=1.5, markersize=4)
        ax.axhline(0, color='k', lw=0.5)
    ax.set_title(f'C1 累積bps (過去8年 N={len(c1)})',
                 fontsize=10, fontweight='bold')
    ax.set_xlabel('event date'); ax.set_ylabel('cum bps')
    ax.grid(alpha=0.3); ax.tick_params(axis='x', rotation=30)

    plt.suptitle(f'{NAME} ({SYM}) 今後見通し分析',
                 fontsize=14, fontweight='bold', y=0.995)
    out = os.path.join(OUT_DIR, 'result.png')
    plt.savefig(out, dpi=130, bbox_inches='tight')
    print(f"Saved: {out}")

    # CSV 保存
    if len(c1) > 0:
        c1.to_csv(os.path.join(OUT_DIR, 'C1_events.csv'), index=False)
    if len(c2) > 0:
        c2.to_csv(os.path.join(OUT_DIR, 'C2_events.csv'), index=False)
    df.tail(120).to_csv(os.path.join(OUT_DIR, 'recent_120d.csv'))
    print("CSV saved")

    print("\n" + "=" * 70)
    print(" 完了")
    print("=" * 70)


if __name__ == "__main__":
    main()
