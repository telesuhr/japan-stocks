"""
マルチタイムフレーム (MTF) 戦略

複数の時間軸を組み合わせたエントリー戦略を検証。

時間軸:
  - 週足: 過去5日リターン, 過去20日リターン (中期トレンド)
  - 日足: 前日リターン, 寄付ギャップ, RSI
  - 時間足/分足: 当日寄付以降の累積リターン (10:00, 13:00時点)

シグナルパターン:
  ◆ Confluence (整合): すべての時間軸が同方向 → モメンタム
  ◆ Pullback (押し目): 大TFは上昇 + 小TFは調整 → 反発狙い
  ◆ Counter (逆張り): 大TF過熱 + 小TF反転兆候 → 反転狙い
  ◆ Trend-Reversal: 大TF反転 + 小TF確認 → トレンド転換

対象:
  - 14銘柄 (既存キャッシュ: 非鉄7 + AI半導体7)
  - 2024-05〜2026-05 (約2年)
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

PG_CONFIG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}
COST_BPS = 4

SYMBOLS = {
    '57060': '三井金属', '57110': '三菱マテ', '57130': '住友鉱山',
    '57140': 'DOWA', '58010': '古河電工', '58020': '住友電工', '58030': 'フジクラ',
    '61460': 'ディスコ', '65260': 'ソシオネクスト', '68570': 'アドバンテスト',
    '69200': 'レーザーテック', '69630': 'ローム', '69760': '太陽誘電', '80350': '東京エレクトロン',
}

CACHE_BARS = Path('/Users/Yusuke/claude-code/japan-stocks/.claude/worktrees/vibrant-mccarthy-d4c865/analyses/20260513_sector_ls_intraday/intraday_bars.parquet')
CACHE_MTF = Path('mtf_features.parquet')


def compute_rsi(prices, period=14):
    """簡易RSI"""
    deltas = prices.diff()
    gain = deltas.where(deltas > 0, 0).rolling(period).mean()
    loss = (-deltas.where(deltas < 0, 0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def build_mtf_features():
    if CACHE_MTF.exists():
        print(f"MTFキャッシュからロード: {CACHE_MTF}")
        return pd.read_parquet(CACHE_MTF)

    print(f"1分足ロード: {CACHE_BARS}")
    df = pd.read_parquet(CACHE_BARS)
    df['ts'] = pd.to_datetime(df['ts'])
    df['date'] = df['ts'].dt.date
    df['time_str'] = df['ts'].dt.strftime('%H:%M:%S')

    print("日次集計 + イントラ時点切出...")
    records = []
    for (d, code), g in df.groupby(['date', 'code']):
        g = g.sort_values('ts').reset_index(drop=True)
        if len(g) < 30:
            continue
        morning = g[g['time_str'] >= '09:00:00']
        if morning.empty:
            continue
        rec = {
            'date': d, 'code': code,
            'day_open': morning.iloc[0]['open'],
            'day_high': g['high'].max(),
            'day_low': g['low'].min(),
            'day_close': g.iloc[-1]['close'],
            'day_volume': g['volume'].sum(),
        }
        # 各時点 close
        for t in ['10:00:00', '10:30:00', '11:00:00', '12:30:00', '13:00:00', '14:00:00']:
            sub = g[g['time_str'] <= t]
            rec[f'p_{t}'] = sub.iloc[-1]['close'] if not sub.empty else np.nan
        records.append(rec)

    daily = pd.DataFrame(records)
    daily = daily.sort_values(['code', 'date']).reset_index(drop=True)

    # 各銘柄ごとに日次特徴量計算
    print("MTF特徴量計算...")
    out = []
    for code, g in daily.groupby('code'):
        g = g.sort_values('date').reset_index(drop=True)
        # 前日close (= dt-1 day close)
        g['prev_close'] = g['day_close'].shift(1)
        # 寄付ギャップ
        g['gap_pct'] = (g['day_open'] / g['prev_close'] - 1) * 10000
        # 日足リターン (前日比)
        g['daily_ret'] = (g['day_close'] / g['prev_close'] - 1) * 10000
        # 5日リターン (週次相当)
        g['week_ret'] = (g['day_close'] / g['day_close'].shift(5) - 1) * 10000
        # 20日リターン (月次相当)
        g['month_ret'] = (g['day_close'] / g['day_close'].shift(20) - 1) * 10000
        # 50日リターン (中期トレンド)
        g['long_ret'] = (g['day_close'] / g['day_close'].shift(50) - 1) * 10000
        # 50日MAとの距離
        ma50 = g['day_close'].rolling(50).mean()
        g['dist_ma50'] = (g['day_close'] / ma50 - 1) * 10000
        # 20日MAとの距離
        ma20 = g['day_close'].rolling(20).mean()
        g['dist_ma20'] = (g['day_close'] / ma20 - 1) * 10000
        # 日足RSI
        g['rsi14'] = compute_rsi(g['day_close'], 14)
        # 20日ボラ
        g['vol_20d'] = g['daily_ret'].rolling(20).std()
        # 当日寄付以降の累積リターン (時間足相当)
        for t in ['10:00:00', '10:30:00', '11:00:00', '12:30:00', '13:00:00', '14:00:00']:
            g[f'cum_ret_{t}'] = (g[f'p_{t}'] / g['day_open'] - 1) * 10000
        # 翌日全日リターン
        g['next_day_ret'] = (g['day_close'].shift(-1) / g['day_close'].shift(-1).pipe(lambda x: x * 0 + g['day_close'].shift(-1)) - 1) * 10000
        # 翌日リターン (overnight + 当日)
        g['next_day_full'] = (g['day_close'].shift(-1) / g['day_close'] - 1) * 10000
        # 翌日寄付ギャップ
        g['next_open'] = g['day_open'].shift(-1)
        g['next_on_ret'] = (g['next_open'] / g['day_close'] - 1) * 10000
        # 翌日寄付→引け
        g['next_intra'] = (g['day_close'].shift(-1) / g['next_open'] - 1) * 10000
        # 当日各時点から引けまでのリターン
        for t in ['10:00:00', '10:30:00', '11:00:00', '12:30:00', '13:00:00', '14:00:00']:
            g[f'fwd_close_{t}'] = (g['day_close'] / g[f'p_{t}'] - 1) * 10000

        out.append(g)

    mtf = pd.concat(out, ignore_index=True)
    mtf.to_parquet(CACHE_MTF)
    print(f"  保存: {CACHE_MTF} ({len(mtf):,} 行)")
    return mtf


def test_rule(df, condition, target_col, label, direction=1):
    sub = df[condition & df[target_col].notna()].copy()
    if len(sub) < 30:
        return None
    arr = sub[target_col].values * direction
    net = arr - COST_BPS
    t, p = stats.ttest_1samp(arr, 0)
    sh = net.mean() / arr.std() * np.sqrt(252) if arr.std() > 0 else 0
    return dict(rule=label, N=len(arr),
                mean_raw=round(arr.mean(), 1),
                mean_net=round(net.mean(), 1),
                t=round(t, 2), p=round(p, 4),
                wr=round((arr > 0).mean() * 100, 1),
                sharpe=round(sh, 2),
                sig=(p < 0.05 and net.mean() > 0))


def main():
    import time
    t0 = time.time()

    df = build_mtf_features()
    print(f"\n対象: {len(df):,} (date,code) | {df['code'].nunique()}銘柄")

    print("\n" + "="*70)
    print("MTFルール検証")
    print("="*70)

    results = []

    # ========== A. Confluence (整合) ==========
    # A1: 週足up + 日足up + 当日寄付gap up → 当日続伸 (寄付→引け)
    cond = (df['week_ret'] > 200) & (df['daily_ret'] > 0) & (df['gap_pct'] > 50)
    r = test_rule(df, cond, 'fwd_close_10:00:00', 'A1: 週up+日up+ギャップup → 10:00から引け Long')
    if r: results.append(r)

    # A2: 週足up強 + 当日10:00時点もup → 続伸
    cond = (df['week_ret'] > 300) & (df['cum_ret_10:00:00'] > 30)
    r = test_rule(df, cond, 'fwd_close_10:00:00', 'A2: 週強up + 10:00 up → 引けまで Long')
    if r: results.append(r)

    # A3: 週足up + 日足up + 当日13:00時点もup → 後場続伸
    cond = (df['week_ret'] > 200) & (df['daily_ret'] > 0) & (df['cum_ret_13:00:00'] > 0)
    r = test_rule(df, cond, 'fwd_close_13:00:00', 'A3: 週up+日up+13:00 up → 引けまで Long')
    if r: results.append(r)

    # A4: 全部下 → 続落 (Short)
    cond = (df['week_ret'] < -200) & (df['daily_ret'] < 0) & (df['gap_pct'] < -50)
    r = test_rule(df, cond, 'fwd_close_10:00:00', 'A4: 週down+日down+ギャップdown → Short', direction=-1)
    if r: results.append(r)

    # ========== B. Pullback (押し目) ==========
    # B1: 週足up + 前日大幅下落 + 当日朝反発 → 続伸
    cond = (df['week_ret'] > 200) & (df['daily_ret'] < -150) & (df['cum_ret_10:00:00'] > 0)
    r = test_rule(df, cond, 'fwd_close_10:00:00', 'B1: 週up + 前日下落 + 朝反発 → Long')
    if r: results.append(r)

    # B2: 週足up + 当日ギャップダウン → 当日埋め (10:00から引け)
    cond = (df['week_ret'] > 200) & (df['gap_pct'] < -50)
    r = test_rule(df, cond, 'fwd_close_10:00:00', 'B2: 週up + 当日ギャップ↓ → 引けまで Long')
    if r: results.append(r)

    # B3: 50日MA上 + RSI<30 + 当日反発 → 続伸
    cond = (df['dist_ma50'] > 0) & (df['rsi14'] < 30) & (df['cum_ret_10:00:00'] > 0)
    r = test_rule(df, cond, 'fwd_close_10:00:00', 'B3: 50日MA上 + RSI<30 + 朝反発 → Long')
    if r: results.append(r)

    # ========== C. Counter (逆張り) ==========
    # C1: 週足過熱 (週ret > 500bps) + 当日ギャップアップ → ショート
    cond = (df['week_ret'] > 500) & (df['gap_pct'] > 100)
    r = test_rule(df, cond, 'fwd_close_10:00:00', 'C1: 週過熱 + ギャップ↑ → 10:00 Short', direction=-1)
    if r: results.append(r)

    # C2: 月足過熱 (20日 +5%以上) + 当日ギャップアップ → ショート
    cond = (df['month_ret'] > 500) & (df['gap_pct'] > 100)
    r = test_rule(df, cond, 'fwd_close_10:00:00', 'C2: 月過熱 + ギャップ↑ → Short', direction=-1)
    if r: results.append(r)

    # C3: RSI > 70 + 当日朝ピーク → ショート
    cond = (df['rsi14'] > 70) & (df['cum_ret_10:00:00'] > 50)
    r = test_rule(df, cond, 'fwd_close_10:00:00', 'C3: RSI>70 + 朝強い → Short', direction=-1)
    if r: results.append(r)

    # C4: 週足深売り + 当日朝ギャップダウン → ロング (逆張り)
    cond = (df['week_ret'] < -500) & (df['gap_pct'] < -100)
    r = test_rule(df, cond, 'fwd_close_10:00:00', 'C4: 週深売り + ギャップ↓ → Long (逆張)')
    if r: results.append(r)

    # ========== D. Trend Reversal ==========
    # D1: 週足下落 + 日足プラス転換 + 寄付ギャップアップ → 続伸
    cond = (df['week_ret'] < -200) & (df['daily_ret'] > 100) & (df['gap_pct'] > 50)
    r = test_rule(df, cond, 'fwd_close_10:00:00', 'D1: 週下落 + 日反転 + ギャップ↑ → Long')
    if r: results.append(r)

    # D2: 月足下落 + 日足反転 → Long
    cond = (df['month_ret'] < -500) & (df['daily_ret'] > 200)
    r = test_rule(df, cond, 'fwd_close_10:00:00', 'D2: 月下落 + 日反転 → Long')
    if r: results.append(r)

    # ========== E. 翌日リターン予測 ==========
    # E1: 週足up + 日足up + 引けまで up → 翌日も up
    cond = (df['week_ret'] > 200) & (df['daily_ret'] > 100)
    r = test_rule(df, cond, 'next_day_full', 'E1: 週up + 日up → 翌日 Long')
    if r: results.append(r)

    # E2: 週足下落 + 日足下落 + 引け弱い → 翌日も下落
    cond = (df['week_ret'] < -200) & (df['daily_ret'] < -100)
    r = test_rule(df, cond, 'next_day_full', 'E2: 週down + 日down → 翌日 Short', direction=-1)
    if r: results.append(r)

    # E3: 週足up + 日足下落 (押し目) → 翌日反発
    cond = (df['week_ret'] > 200) & (df['daily_ret'] < -100)
    r = test_rule(df, cond, 'next_day_full', 'E3: 週up + 日down (押し目) → 翌日 Long')
    if r: results.append(r)

    # E4: 週足下落 + 日足上昇 (戻し失敗) → 翌日下落
    cond = (df['week_ret'] < -200) & (df['daily_ret'] > 100)
    r = test_rule(df, cond, 'next_day_full', 'E4: 週down + 日up → 翌日 Short', direction=-1)
    if r: results.append(r)

    # E5: 50日MA上 + RSI<30 → 翌日反発
    cond = (df['dist_ma50'] > 0) & (df['rsi14'] < 30)
    r = test_rule(df, cond, 'next_day_full', 'E5: 50日MA上 + RSI<30 → 翌日 Long')
    if r: results.append(r)

    # E6: 寄付ギャップ正常時 (-50〜+50) で週/日同方向 → 翌日
    cond = (df['week_ret'].abs() > 200) & (df['gap_pct'].abs() < 50) & (np.sign(df['week_ret']) == np.sign(df['daily_ret']))
    r = test_rule(df, cond, 'next_day_full', 'E6: 週日同方向 + ギャップ小 → 翌日順方向')
    if r:
        # 方向を週方向に
        r2 = r.copy()
        # 戦略リターン: 直接計算
        sub = df[cond & df['next_day_full'].notna()].copy()
        sub['signal_ret'] = np.sign(sub['week_ret']) * sub['next_day_full']
        arr = sub['signal_ret'].values
        net = arr - COST_BPS
        t, _ = stats.ttest_1samp(arr, 0)
        sh = net.mean() / arr.std() * np.sqrt(252) if arr.std() > 0 else 0
        r2['mean_raw'] = round(arr.mean(), 1)
        r2['mean_net'] = round(net.mean(), 1)
        r2['t'] = round(t, 2)
        r2['sharpe'] = round(sh, 2)
        r2['wr'] = round((arr > 0).mean() * 100, 1)
        r2['rule'] = 'E6: 週日同方向 + ギャップ小 → 翌日順方向 (Direction-aware)'
        results.append(r2)

    # ========== F. 高度な組み合わせ ==========
    # F1: 週up + 月up + 日下落 + 朝反発 → ペア押し目
    cond = (df['week_ret'] > 200) & (df['month_ret'] > 0) & (df['daily_ret'] < -100) & (df['cum_ret_10:00:00'] > 0)
    r = test_rule(df, cond, 'fwd_close_10:00:00', 'F1: 週up+月up+日下落+朝反発 → Long')
    if r: results.append(r)

    # F2: 週down + 月down + 日上昇 + 朝失速 → ショート
    cond = (df['week_ret'] < -200) & (df['month_ret'] < 0) & (df['daily_ret'] > 100) & (df['cum_ret_10:00:00'] < 0)
    r = test_rule(df, cond, 'fwd_close_10:00:00', 'F2: 週down+月down+日上昇+朝失速 → Short', direction=-1)
    if r: results.append(r)

    # 結果
    res_df = pd.DataFrame(results)
    res_df_sorted = res_df.sort_values('t', ascending=False)

    print("\n=== 全ルール結果 (t値降順) ===")
    print(res_df_sorted[['rule','N','mean_raw','mean_net','t','p','wr','sharpe','sig']].to_string(index=False))

    sig_rules = res_df[res_df['sig']]
    print(f"\n=== 有望ルール (p<0.05, net>0): {len(sig_rules)}件 ===")
    if len(sig_rules) > 0:
        print(sig_rules.sort_values('sharpe', ascending=False)[
            ['rule','N','mean_net','t','wr','sharpe']
        ].to_string(index=False))

    res_df.to_csv('mtf_results.csv', index=False)

    # =====================================================================
    # 図
    # =====================================================================
    fig = plt.figure(figsize=(16, 11), facecolor='white')
    plt.rcParams.update({
        'font.family': ['Hiragino Sans', 'IPAexGothic', 'sans-serif'],
        'axes.unicode_minus': False,
    })
    fig.suptitle('マルチタイムフレーム (MTF) 戦略検証\n週足/日足/時間足/分足の組み合わせ — 14銘柄 2024/5-2026/5',
                 fontsize=13, fontweight='bold', y=0.99)

    # 左: 全ルール t値ランキング
    ax1 = fig.add_axes([0.05, 0.05, 0.50, 0.90])
    plot_df = res_df_sorted.copy()
    ys = range(len(plot_df))
    colors = []
    for _, row in plot_df.iterrows():
        if row['p'] < 0.05 and row['mean_net'] > 0:
            colors.append('#43A047')
        elif row['p'] < 0.05 and row['mean_net'] < 0:
            colors.append('#E53935')
        elif row['t'] > 1.0 and row['mean_net'] > 0:
            colors.append('#FF9800')
        else:
            colors.append('#9E9E9E')
    ax1.barh(list(ys), plot_df['t'].values, color=colors, alpha=0.85, height=0.7)
    ax1.set_yticks(list(ys))
    ax1.set_yticklabels([s[:38] for s in plot_df['rule']], fontsize=7.5)
    ax1.axvline(0, color='black', lw=0.7)
    ax1.axvline(1.96, color='gray', lw=0.6, linestyle='--', alpha=0.6)
    ax1.axvline(-1.96, color='gray', lw=0.6, linestyle='--', alpha=0.6)
    ax1.set_xlabel('t統計量', fontsize=9)
    ax1.set_title('全ルール t値ランキング (緑=有意プラス, 赤=有意マイナス)',
                  fontsize=10, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # 右上: 有望ルール詳細
    ax2 = fig.add_axes([0.60, 0.55, 0.37, 0.40])
    ax2.axis('off')
    if len(sig_rules) > 0:
        tbl = sig_rules.sort_values('sharpe', ascending=False)[
            ['rule','N','mean_net','t','wr','sharpe']
        ].copy()
        tbl.columns = ['ルール','N','net(bps)','t値','勝率%','Sharpe']
        tbl['ルール'] = tbl['ルール'].str[:30]
        table = ax2.table(cellText=tbl.values, colLabels=tbl.columns,
                          cellLoc='center', loc='upper center',
                          bbox=[0, 0.1, 1, 0.85])
        table.auto_set_font_size(False)
        table.set_fontsize(8.5)
        for (r, c), cell in table.get_celld().items():
            if r == 0:
                cell.set_facecolor('#1565C0')
                cell.set_text_props(color='white', fontweight='bold')
            elif r % 2 == 0:
                cell.set_facecolor('#E3F2FD')
            cell.set_edgecolor('#BDBDBD')
        ax2.set_title(f'有望ルール: {len(sig_rules)}件 (Sharpe降順)',
                      fontsize=10, fontweight='bold', y=0.96)
    else:
        ax2.text(0.5, 0.5, '有望ルール: 0件', ha='center', va='center',
                 fontsize=14, fontweight='bold', color='red')

    # 右下: スキャッター (t vs mean_net)
    ax3 = fig.add_axes([0.60, 0.05, 0.37, 0.40])
    for _, row in res_df.iterrows():
        is_sig = row['p'] < 0.05 and row['mean_net'] > 0
        c = '#43A047' if is_sig else ('#FF9800' if row['t'] > 1 and row['mean_net'] > 0 else '#9E9E9E')
        m = '*' if is_sig else 'o'
        s = 150 if is_sig else 40
        ax3.scatter(row['t'], row['mean_net'], color=c, marker=m, s=s,
                    alpha=0.85, edgecolor='black', lw=0.4)
    ax3.axhline(0, color='black', lw=0.6)
    ax3.axvline(0, color='black', lw=0.6)
    ax3.axvline(1.96, color='gray', lw=0.6, linestyle='--')
    ax3.set_xlabel('t統計量', fontsize=9)
    ax3.set_ylabel('mean_net (bps)', fontsize=9)
    ax3.set_title('t値 vs net期待値 (★=有意プラス)', fontsize=10, fontweight='bold')
    ax3.grid(alpha=0.3)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    fig.text(0.99, 0.005,
             '14銘柄 (非鉄7+AI半導体7) | 2024-05〜2026-05 | コスト4bps (片方向)',
             ha='right', va='bottom', fontsize=7, color='gray')
    plt.savefig('result.png', dpi=100, bbox_inches='tight', facecolor='white')
    print(f"\nresult.png 保存完了 ({time.time()-t0:.1f}秒)")


if __name__ == '__main__':
    main()
