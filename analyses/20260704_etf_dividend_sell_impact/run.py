"""
ETF配当金捻出売りの需給インパクト分析

ETF決算日の大引け前後でTOPIX/日経に有意なパターンがあるかを検証。
捻出売り推定日: 毎年6〜7月（本決算ETFの配当支払い月）
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, psycopg2
from datetime import date, timedelta

PG = dict(host=os.environ.get("PGHOST","localhost"),
          port=int(os.environ.get("PGPORT",5432)),
          user=os.environ.get("PGUSER","postgres"),
          dbname=os.environ.get("PGDATABASE","market_data"))

# =====================================================================
# ETF配当金捻出売り 推定決算日リスト
# ※ 大半のETFは6月末決算 → 7月上旬に分配金支払い → その前日大引けで売り
# 主要ETF決算日（8日・10日前後が多い）を年度別に手動整理
# 売り規模は各年の日経・Bloomberg報道ベースの概算
# =====================================================================
ETF_EVENTS = [
    # (決算日, 売り推定規模 億円)
    # 2019年
    ("2019-07-10", 4500),
    # 2020年
    ("2020-07-09", 5200),
    ("2020-07-13", 3800),
    # 2021年
    ("2021-07-08", 6000),
    ("2021-07-12", 4500),
    # 2022年
    ("2022-07-11", 5800),
    ("2022-07-13", 4200),
    # 2023年
    ("2023-07-10", 7500),
    ("2023-07-12", 5500),
    # 2024年
    ("2024-07-08", 7000),
    ("2024-07-10", 8500),
    # 2025年
    ("2025-07-08", 8000),
    ("2025-07-10", 9500),
    # 2026年（今回の注目イベント）
    ("2026-07-08", 6000),
    ("2026-07-10", 9000),
]

def load_index(code: str) -> pd.DataFrame:
    conn = psycopg2.connect(**PG)
    cur = conn.cursor()
    cur.execute("""
        SELECT date, open::float, high::float, low::float, close::float
        FROM index_daily
        WHERE code = %s ORDER BY date
    """, (code,))
    rows = cur.fetchall(); conn.close()
    df = pd.DataFrame(rows, columns=["date","open","high","low","close"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df["ret"] = df["close"].pct_change()
    return df

print("データ読み込み中...")
topix = load_index("0000")
print(f"TOPIX: {topix.index[0].date()} 〜 {topix.index[-1].date()}, {len(topix)}日")

# 取引日一覧
trading_days = topix.index.tolist()

def get_offset_day(event_date: str, offset: int):
    """イベント日からoffset営業日ずらした日付を返す"""
    ed = pd.Timestamp(event_date)
    if ed not in trading_days:
        # 最寄り取引日を探す
        idx = topix.index.searchsorted(ed)
        if idx >= len(trading_days): return None
        ed = trading_days[idx]
    pos = trading_days.index(ed)
    target = pos + offset
    if 0 <= target < len(trading_days):
        return trading_days[target]
    return None

# =====================================================================
# D-5〜D+5 のリターンを集計
# =====================================================================
WINDOW = range(-5, 6)
rows = []
for event_date, sell_size in ETF_EVENTS:
    ed = pd.Timestamp(event_date)
    if ed > topix.index[-1]: continue  # 将来日はスキップ（2026年分）
    row = {"event_date": event_date, "sell_size": sell_size}
    for offset in WINDOW:
        target = get_offset_day(event_date, offset)
        if target and target in topix.index:
            row[f"D{offset:+d}"] = topix.loc[target, "ret"] * 100
        else:
            row[f"D{offset:+d}"] = np.nan
    rows.append(row)

df_events = pd.DataFrame(rows)
print(f"\nイベント数: {len(df_events)}")
print(df_events[["event_date","sell_size"]].to_string(index=False))

# =====================================================================
# 集計
# =====================================================================
day_cols = [f"D{o:+d}" for o in WINDOW]
mean_ret = df_events[day_cols].mean()
std_ret  = df_events[day_cols].std()
n        = df_events[day_cols].count()
t_stat   = mean_ret / (std_ret / np.sqrt(n))

print(f"\n\n{'='*60}")
print("ETF捻出売りイベント前後 TOPIX リターン (平均%)")
print(f"{'='*60}")
print(f"{'日':>6}  {'平均%':>8}  {'t値':>7}  {'N':>4}")
for col in day_cols:
    marker = "★" if col == "D+0" else ("  " if col not in ["D-1","D+1"] else "→")
    print(f"  {col:>5}{marker}  {mean_ret[col]:>+7.3f}%  {t_stat[col]:>+6.2f}   {n[col]:>4}")

# 累積
print(f"\n  累積 D-3〜D-1: {df_events[['D-3','D-2','D-1']].sum(axis=1).mean():>+.3f}%")
print(f"  累積 D+1〜D+3: {df_events[['D+1','D+2','D+3']].sum(axis=1).mean():>+.3f}%")

# =====================================================================
# 売り規模別 (大/中/小 3分位)
# =====================================================================
df_events["size_group"] = pd.qcut(df_events["sell_size"], 3, labels=["小","中","大"])
print(f"\n\n{'='*60}")
print("売り規模別 D+0 リターン")
print(f"{'='*60}")
for g in ["小","中","大"]:
    sub = df_events[df_events["size_group"]==g]["D+0"]
    print(f"  {g} (n={len(sub)}): 平均 {sub.mean():>+.3f}%  std {sub.std():.3f}%")

# =====================================================================
# 可視化
# =====================================================================
try:
    import matplotlib.font_manager as fm
    fp = fm.FontProperties(fname="/root/.fonts/NotoSansJP.ttf")
    plt.rcParams["font.family"] = fp.get_name()
except Exception:
    pass

fig, axes = plt.subplots(1, 2, figsize=(12, 6.75), facecolor='white')

# (1) イベント前後の平均リターン
ax = axes[0]
colors = ['#d62728' if v < 0 else '#2ca02c' for v in mean_ret.values]
bars = ax.bar(range(len(WINDOW)), mean_ret.values, color=colors, alpha=0.75, width=0.7)
ax.axhline(0, color='black', lw=0.8)
ax.axvline(5, color='red', lw=1.5, ls='--', label='イベント日 (D+0)')
ax.set_xticks(range(len(WINDOW)))
ax.set_xticklabels([f"D{o:+d}" for o in WINDOW], fontsize=8)
ax.set_ylabel('TOPIX リターン (%)', fontsize=10)
ax.set_title('ETF配当捻出売り\nイベント前後の平均リターン', fontsize=11)
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)
for bar, t in zip(bars, t_stat.values):
    if abs(t) > 1.0:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(),
                f't={t:.1f}', ha='center', va='bottom' if bar.get_height()>0 else 'top',
                fontsize=7)

# (2) 個別イベントのD-1/D+0/D+1散布
ax2 = axes[1]
x = df_events["D-1"].values
y = df_events["D+1"].values
sz = (df_events["sell_size"] / 500).values
ax2.scatter(x, y, s=sz*20, alpha=0.6, c='steelblue', edgecolors='navy', linewidth=0.5)
for _, r in df_events.iterrows():
    ax2.annotate(r["event_date"][:7], (r["D-1"], r["D+1"]), fontsize=6.5, alpha=0.7)
ax2.axhline(0, color='black', lw=0.8, ls='--')
ax2.axvline(0, color='black', lw=0.8, ls='--')
ax2.set_xlabel('D-1 リターン (%)', fontsize=10)
ax2.set_ylabel('D+1 リターン (%)', fontsize=10)
ax2.set_title('前日(D-1) vs 翌日(D+1)\n（円の大きさ=売り規模）', fontsize=11)
ax2.grid(alpha=0.3)

fig.text(0.5, 0.01, 'データ: index_daily TOPIX / ETF捻出売り推定日は報道ベース',
         ha='center', fontsize=8, color='gray')
plt.tight_layout(rect=[0,0.03,1,1])
plt.savefig("result.png", dpi=100, bbox_inches='tight', facecolor='white')
print("\nsaved result.png")
