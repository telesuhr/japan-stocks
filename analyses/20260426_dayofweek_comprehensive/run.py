"""
全セクター × 曜日別 × ON / イントラ / 全日 包括的分析
20260426_dayofweek_comprehensive/run.py

出力:
  result.png        -- X投稿用: ヒートマップ3枚 (ON/Intra/Full)
  result_detail.png -- セクター別棒グラフ詳細版
  dayofweek_stats.csv
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '20260421_common'))
import mdutil as U
import pandas as pd
import numpy as np
from scipy import stats

# ── 設定 ────────────────────────────────────────────────
DAYS_EN = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
DAYS_JP = ['月', '火', '水', '木', '金']

SECTORS = {
    'CORE5':      U.CORE5,
    'NONFER':     U.NONFERROUS,
    'SEMICON':    U.SEMICON,
    'ENERGY':     U.ENERGY,
    'SHIPPING':   U.SHIPPING,
    'DOMESTIC':   U.DOMESTIC_SHORT,
}

SECTOR_JP = {
    'CORE5':    'コア5\n(景気敏感)',
    'NONFER':   '非鉄\n(銅連動)',
    'SEMICON':  '半導体\n(SOX連動)',
    'ENERGY':   'エネルギー\n(原油連動)',
    'SHIPPING': '海運\n(BDI連動)',
    'DOMESTIC': '内需\n(LS Short)',
}

RETURN_TYPES = ['on', 'intra', 'full']
RETURN_LABELS = {
    'on':    'オーバーナイト\n(前日引→寄)',
    'intra': 'イントラデイ\n(寄→引)',
    'full':  '全日\n(前日引→引)',
}


# ── データロード ────────────────────────────────────────
def load_sector(basket: list) -> pd.DataFrame | None:
    frames = []
    for sym, name in basket:
        try:
            df = U.load_jp_daily(sym)
            df = df.dropna(subset=['open', 'close'])
            df['on']    = (df['open']  / df['close'].shift(1) - 1) * 10_000
            df['intra'] = (df['close'] / df['open']           - 1) * 10_000
            df['full']  = (df['close'] / df['close'].shift(1) - 1) * 10_000
            df['dow']   = pd.to_datetime(df.index).dayofweek  # 0=Mon
            df = df[['on', 'intra', 'full', 'dow']].dropna()
            # 外れ値除去 (|return| > 1500bps = 15%)
            for col in RETURN_TYPES:
                df = df[df[col].abs() <= 1500]
            df['symbol'] = sym
            frames.append(df)
        except Exception as e:
            print(f"  {sym} スキップ: {e}")
    if not frames:
        return None
    return pd.concat(frames)


# ── 統計計算 ────────────────────────────────────────────
def compute_dow_stats(df: pd.DataFrame, ret_col: str) -> pd.DataFrame:
    rows = []
    for dow in range(5):
        arr = df.loc[df['dow'] == dow, ret_col].values
        if len(arr) < 5:
            rows.append({'dow': dow, 'mean': np.nan, 't': np.nan, 'n': len(arr)})
            continue
        t, p = stats.ttest_1samp(arr, 0)
        rows.append({
            'dow': dow,
            'mean': arr.mean(),
            't': t,
            'n': len(arr),
            'wr': (arr > 0).mean() * 100,
        })
    return pd.DataFrame(rows).set_index('dow')


# ── メイン処理 ──────────────────────────────────────────
print("データロード中...")
sector_data = {}
for sec, basket in SECTORS.items():
    print(f"  {sec} ...", end=' ')
    df = load_sector(basket)
    if df is not None:
        sector_data[sec] = df
        print(f"N={len(df)}")
    else:
        print("スキップ")

# 統計テーブル構築
records = []
for sec, df in sector_data.items():
    for ret in RETURN_TYPES:
        st = compute_dow_stats(df, ret)
        for dow in range(5):
            if dow in st.index:
                row = st.loc[dow]
                records.append({
                    'sector': sec,
                    'return_type': ret,
                    'dow': dow,
                    'dow_en': DAYS_EN[dow],
                    'mean_bps': row['mean'],
                    't_stat': row['t'],
                    'n': row['n'],
                    'wr_pct': row.get('wr', np.nan),
                })

stats_df = pd.DataFrame(records)
stats_df.to_csv(os.path.join(os.path.dirname(__file__), 'dayofweek_stats.csv'), index=False, float_format='%.2f')
print("dayofweek_stats.csv 保存完了")


# ── 可視化 (result.png: X投稿用) ────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt_mod
import matplotlib.colors as mcolors
plt_mod.rcParams['font.family'] = ['Yu Gothic', 'Meiryo', 'MS Gothic', 'sans-serif']
plt_mod.rcParams['axes.unicode_minus'] = False

fig = plt_mod.figure(figsize=(12, 6.75), facecolor='white')
fig.patch.set_facecolor('white')

# タイトル
fig.text(0.5, 0.97, '日本株 全セクター × 曜日別リターン分析',
         ha='center', va='top', fontsize=15, fontweight='bold')
fig.text(0.5, 0.93, 'ON（前日引→寄） / イントラ（寄→引） / 全日（前日引→引）',
         ha='center', va='top', fontsize=10, color='#444444')

# カラーマップ: t統計量ベース (-3 〜 +3)
cmap = plt_mod.cm.RdYlGn
norm = mcolors.TwoSlopeNorm(vmin=-3, vcenter=0, vmax=3)

sector_keys = list(sector_data.keys())
n_sec = len(sector_keys)

axes = []
for i, ret in enumerate(RETURN_TYPES):
    ax = fig.add_axes([0.06 + i * 0.315, 0.10, 0.28, 0.78])
    axes.append(ax)

    # ヒートマップデータ構築 (5 DoW × n_sec)
    mat_t    = np.full((5, n_sec), np.nan)
    mat_mean = np.full((5, n_sec), np.nan)
    mat_n    = np.full((5, n_sec), np.nan)

    for j, sec in enumerate(sector_keys):
        sub = stats_df[(stats_df['sector'] == sec) & (stats_df['return_type'] == ret)]
        for _, row in sub.iterrows():
            d = int(row['dow'])
            mat_t[d, j]    = row['t_stat']
            mat_mean[d, j] = row['mean_bps']
            mat_n[d, j]    = row['n']

    im = ax.imshow(mat_t, cmap=cmap, norm=norm, aspect='auto')

    # セル注釈
    for d in range(5):
        for j in range(n_sec):
            m = mat_mean[d, j]
            t = mat_t[d, j]
            if np.isnan(m):
                continue
            sign = '★' if abs(t) >= 2.0 else ('△' if abs(t) >= 1.5 else '')
            color = 'white' if abs(t) >= 2.5 else 'black'
            ax.text(j, d, f'{m:+.0f}\n{sign}',
                    ha='center', va='center', fontsize=7.5,
                    fontweight='bold' if abs(t) >= 2.0 else 'normal',
                    color=color)

    # 軸ラベル
    ax.set_xticks(range(n_sec))
    ax.set_xticklabels([SECTOR_JP[s].replace('\n', '\n') for s in sector_keys],
                       fontsize=7, ha='center')
    ax.set_yticks(range(5))
    if i == 0:
        ax.set_yticklabels(DAYS_JP, fontsize=9)
    else:
        ax.set_yticklabels([])

    ax.set_title(RETURN_LABELS[ret], fontsize=9, pad=4)
    ax.tick_params(length=0)

    # グリッド
    for d in range(6):
        ax.axhline(d - 0.5, color='white', linewidth=1.5)
    for j in range(n_sec + 1):
        ax.axvline(j - 0.5, color='white', linewidth=1.5)

# カラーバー
cbar_ax = fig.add_axes([0.955, 0.15, 0.012, 0.65])
sm = plt_mod.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label('t統計量', fontsize=8)
cbar.set_ticks([-3, -2, -1, 0, 1, 2, 3])

# 凡例
fig.text(0.5, 0.04,
         '色: t統計量 (緑=有意にプラス / 赤=有意にマイナス)  |  数値: 平均リターン (bps)  |  ★: |t|≥2.0  △: |t|≥1.5',
         ha='center', va='bottom', fontsize=8, color='#555555')
fig.text(0.99, 0.01,
         f'データ: {U.START}〜{U.END} / 日本株1分足 (Refinitiv) / 6セクター27銘柄',
         ha='right', va='bottom', fontsize=7, color='gray')

out_path = os.path.join(os.path.dirname(__file__), 'result.png')
plt_mod.savefig(out_path, dpi=100, bbox_inches='tight', facecolor='white')
plt_mod.close()
print(f"result.png 保存完了")


# ── 詳細グラフ (result_detail.png: セクター別棒グラフ) ──
fig2, axes2 = plt_mod.subplots(3, n_sec, figsize=(14, 9))
fig2.patch.set_facecolor('white')
fig2.suptitle('セクター × 曜日 × リターン種別 詳細', fontsize=13, fontweight='bold', y=0.99)

colors_bar = ['#e74c3c', '#f39c12', '#27ae60', '#c0392b', '#2980b9']  # Mon-Fri

for row_i, ret in enumerate(RETURN_TYPES):
    for col_j, sec in enumerate(sector_keys):
        ax = axes2[row_i, col_j]
        sub = stats_df[(stats_df['sector'] == sec) & (stats_df['return_type'] == ret)]
        means = []
        ts    = []
        for d in range(5):
            r = sub[sub['dow'] == d]
            means.append(r['mean_bps'].values[0] if len(r) > 0 else 0)
            ts.append(r['t_stat'].values[0] if len(r) > 0 else 0)

        for d, (m, t) in enumerate(zip(means, ts)):
            alp = min(1.0, 0.4 + abs(t) * 0.2)
            ax.bar(d, m, color='#2ecc71' if m > 0 else '#e74c3c',
                   alpha=alp, width=0.7)
        ax.axhline(0, color='black', linewidth=0.7)

        # t≥2.0 に ★
        for d, (m, t) in enumerate(zip(means, ts)):
            if abs(t) >= 2.0 and m != 0:
                ax.text(d, m + (5 if m >= 0 else -5), '★',
                        ha='center', va='bottom' if m >= 0 else 'top',
                        fontsize=9, color='black')

        ax.set_xticks(range(5))
        ax.set_xticklabels(DAYS_JP, fontsize=8)
        ax.tick_params(axis='y', labelsize=7)
        ax.grid(axis='y', alpha=0.3)

        if row_i == 0:
            ax.set_title(sec, fontsize=9, fontweight='bold')
        if col_j == 0:
            ax.set_ylabel(RETURN_LABELS[ret].replace('\n', ' '), fontsize=7)

fig2.text(0.99, 0.005,
          f'データ: {U.START}〜{U.END} / ★: |t|≥2.0 / アルファは有意度比例',
          ha='right', va='bottom', fontsize=7, color='gray')
plt_mod.tight_layout()
out2 = os.path.join(os.path.dirname(__file__), 'result_detail.png')
plt_mod.savefig(out2, dpi=100, bbox_inches='tight', facecolor='white')
plt_mod.close()
print(f"result_detail.png 保存完了")


# ── サマリー出力 ────────────────────────────────────────
print("\n" + "="*65)
print("[sig] |t|>=2.0 signals")
print("="*65)
sig = stats_df[stats_df['t_stat'].abs() >= 2.0].sort_values('t_stat', key=abs, ascending=False)
for _, r in sig.iterrows():
    direction = "Long" if r['mean_bps'] > 0 else "Short"
    print(f"  [{r['sector']:8s}] {r['dow_en']} {r['return_type']:5s} "
          f"mean={r['mean_bps']:+7.1f}bps  t={r['t_stat']:+5.2f}  N={r['n']:.0f}  -> {direction}")

print("\nThursday filter (all sectors, ON):")
thu = stats_df[(stats_df['dow'] == 3) & (stats_df['return_type'] == 'on')]
for _, r in thu.iterrows():
    print(f"  {r['sector']:8s}: {r['mean_bps']:+7.1f}bps  t={r['t_stat']:+5.2f}")
