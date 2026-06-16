import os, sys
sys.stdout.reconfigure(line_buffering=True)
import psycopg2, pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path

PG_CONFIG = {
    "host": os.environ.get("PGHOST", "omen"),
    "port": int(os.environ.get("PGPORT", 5432)),
    "user": os.environ.get("PGUSER", "postgres"),
    "dbname": os.environ.get("PGDATABASE", "market_data"),
}

# ── FOMC 発表日リスト (FRB公式スケジュール) ──────────────────────────────────
# 二日間会合の2日目(発表日) = JST翌早朝3時頃に声明・会見
FOMC_DATES = pd.to_datetime([
    # 2020
    "2020-01-29", "2020-03-03", "2020-03-15", "2020-04-29",
    "2020-06-10", "2020-07-29", "2020-09-16", "2020-11-05", "2020-12-16",
    # 2021
    "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16",
    "2021-07-28", "2021-09-22", "2021-11-03", "2021-12-15",
    # 2022
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15",
    "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
    # 2023
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14",
    "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    # 2024
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
    "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    # 2025
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-11-05", "2025-12-17",
    # 2026
    "2026-01-28", "2026-03-19", "2026-05-07", "2026-06-18",
])

# ── DB からインデックス日足を取得 ──────────────────────────────────────────────
print("DB接続中...")
conn = psycopg2.connect(**PG_CONFIG)

# TOPIX (0020) と 日経225 (0010)
idx_sql = """
    SELECT date, code, close
    FROM index_daily
    WHERE code IN ('0000', 'N225')
      AND date >= '2020-01-01' AND date <= '2026-06-15'
    ORDER BY date
"""
idx_df = pd.read_sql(idx_sql, conn, parse_dates=["date"])
conn.close()
print(f"取得: {len(idx_df)} rows, date range {idx_df.date.min().date()} ~ {idx_df.date.max().date()}")

# ピボット → リターン
idx_pivot = idx_df.pivot(index="date", columns="code", values="close")
idx_pivot.columns = ["nk225", "topix"]  # N225=日経, 0000=TOPIX
idx_pivot = idx_pivot.sort_index()
ret = idx_pivot.pct_change()  # 日次リターン

# ── 営業日カレンダー上の「FOMC前日」算出 ──────────────────────────────────────
trading_days = idx_pivot.index  # DB上の営業日が正

def prev_trading_day(d, tdays):
    """FOMCd の前営業日を返す（日本株市場が開いている日）"""
    before = tdays[tdays < d]
    return before[-1] if len(before) else None

fomc_pre = []
fomc_on  = []
for fd in FOMC_DATES:
    prev = prev_trading_day(fd, trading_days)
    if prev is not None and fd in trading_days:
        fomc_pre.append(prev)
        fomc_on.append(fd)

fomc_pre = pd.DatetimeIndex(fomc_pre)
fomc_on  = pd.DatetimeIndex(fomc_on)
print(f"FOMC前日サンプル: {len(fomc_pre)} 件")

# ── リターン分類 ───────────────────────────────────────────────────────────────
ret_pre  = ret.loc[ret.index.isin(fomc_pre)].copy()
ret_on   = ret.loc[ret.index.isin(fomc_on)].copy()
ret_other = ret.loc[~ret.index.isin(fomc_pre) & ~ret.index.isin(fomc_on)].copy()

def describe(df, label):
    r = df[["topix","nk225"]].describe()
    print(f"\n=== {label} (n={len(df)}) ===")
    print(r.loc[["mean","std","50%"]].T)
    return r

describe(ret_pre, "FOMC前日")
describe(ret_on, "FOMC当日(JST翌朝発表→当日市場)")
describe(ret_other, "通常日")

# ── t検定 ─────────────────────────────────────────────────────────────────────
results = {}
for col in ["topix", "nk225"]:
    x = ret_pre[col].dropna()
    y = ret_other[col].dropna()
    t, p = stats.ttest_ind(x, y, equal_var=False)
    d = (x.mean() - y.mean()) / np.sqrt((x.std()**2 + y.std()**2)/2)
    results[col] = dict(
        pre_mean=x.mean()*100, pre_std=x.std()*100,
        other_mean=y.mean()*100, other_std=y.std()*100,
        t=t, p=p, d=d, n_pre=len(x), n_other=len(y)
    )
    print(f"\n{col}: 前日mean={x.mean()*100:.3f}% vs 通常{y.mean()*100:.3f}%, t={t:.2f}, p={p:.3f}, Cohen's d={d:.3f}")

# ── レジーム分解 ──────────────────────────────────────────────────────────────
# ゼロ金利: 〜2022-02 / 利上げ: 2022-03〜2023-07 / 据え置き: 2023-08〜2024-08 / 利下げ: 2024-09〜
def regime(d):
    if d < pd.Timestamp("2022-03-16"): return "ゼロ金利"
    elif d < pd.Timestamp("2023-08-01"): return "利上げ"
    elif d < pd.Timestamp("2024-09-18"): return "据え置き"
    else: return "利下げ"

ret_pre["regime"] = ret_pre.index.map(regime)
regime_stats = ret_pre.groupby("regime")[["topix","nk225"]].agg(["mean","std","count"]) * 100
print("\n=== FOMC前日 レジーム別 (%) ===")
print(regime_stats.round(3))

# ── FOMC前日→当日 連続性 ────────────────────────────────────────────────────
# 前日陽線 → 当日どうなるか
aligned = pd.DataFrame({
    "pre_topix": ret_pre["topix"].values,
    "on_topix":  ret_on["topix"].values,
}, index=fomc_pre[:len(ret_on)])
aligned["pre_up"] = aligned.pre_topix > 0
followthrough = aligned.groupby("pre_up").on_topix.mean() * 100
print("\n=== 前日方向 → 当日 (TOPIX, %) ===")
print(followthrough.rename({True:"前日↑", False:"前日↓"}))

# ── 可視化 ─────────────────────────────────────────────────────────────────────
try:
    font_path = Path("/root/.fonts/NotoSansJP.ttf")
    if not font_path.exists():
        font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if font_path.exists():
        fp = fm.FontProperties(fname=str(font_path))
        plt.rcParams["font.family"] = fp.get_name()
except Exception:
    pass

plt.rcParams.update({
    "axes.unicode_minus": False,
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f9fa",
    "grid.alpha": 0.3,
})

fig = plt.figure(figsize=(12, 6.75), facecolor="white")
fig.suptitle("FOMC前日の日本株リターン (2020〜2026年)", fontsize=16, fontweight="bold", y=0.98)

axes = fig.subplots(1, 3)

# 1. 箱ひげ（TOPIX）
ax = axes[0]
data_box = [
    ret_pre["topix"].dropna()*100,
    ret_other["topix"].dropna()*100,
]
bp = ax.boxplot(data_box, labels=["FOMC前日", "通常日"], patch_artist=True,
                medianprops=dict(color="red", linewidth=2))
bp["boxes"][0].set_facecolor("#4C9BE8")
bp["boxes"][1].set_facecolor("#A8D5A2")
ax.axhline(0, color="gray", lw=0.8, ls="--")
ax.set_title("TOPIX日次リターン分布", fontsize=11)
ax.set_ylabel("リターン (%)")
r = results["topix"]
ax.text(0.5, 0.02, f"前日: {r['pre_mean']:.3f}%±{r['pre_std']:.2f}%\n通常: {r['other_mean']:.3f}%±{r['other_std']:.2f}%\nt={r['t']:.2f}, p={r['p']:.3f}",
        transform=ax.transAxes, ha="center", va="bottom", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

# 2. レジーム別平均
ax = axes[1]
reg_mean = ret_pre.groupby("regime")["topix"].mean() * 100
order = ["ゼロ金利", "利上げ", "据え置き", "利下げ"]
reg_mean = reg_mean.reindex([r for r in order if r in reg_mean.index])
colors = ["#4C9BE8" if v >= 0 else "#E87D4C" for v in reg_mean.values]
bars = ax.bar(reg_mean.index, reg_mean.values, color=colors)
ax.axhline(0, color="gray", lw=0.8, ls="--")
for b, v in zip(bars, reg_mean.values):
    ax.text(b.get_x()+b.get_width()/2, v + (0.03 if v>=0 else -0.08),
            f"{v:.3f}%", ha="center", va="bottom" if v>=0 else "top", fontsize=9)
counts = ret_pre.groupby("regime").size().reindex(reg_mean.index)
ax.set_title("FOMC前日 レジーム別平均 (TOPIX)", fontsize=11)
ax.set_ylabel("平均リターン (%)")
ax.set_xticklabels(reg_mean.index, fontsize=9)
# n数を注記
for i, (idx, cnt) in enumerate(counts.items()):
    ax.text(i, -0.15, f"n={cnt}", ha="center", fontsize=8, color="gray",
            transform=ax.get_xaxis_transform())

# 3. 前日方向→当日 連動
ax = axes[2]
fup = aligned[aligned.pre_up]["on_topix"].dropna() * 100
fdn = aligned[~aligned.pre_up]["on_topix"].dropna() * 100
bp2 = ax.boxplot([fup, fdn], labels=["前日↑", "前日↓"], patch_artist=True,
                 medianprops=dict(color="red", linewidth=2))
bp2["boxes"][0].set_facecolor("#4C9BE8")
bp2["boxes"][1].set_facecolor("#E87D4C")
ax.axhline(0, color="gray", lw=0.8, ls="--")
ax.set_title("前日方向 → 発表当日TOPIX", fontsize=11)
ax.set_ylabel("当日リターン (%)")
ax.text(0.5, 0.02, f"前日↑翌日: {fup.mean():.3f}%  前日↓翌日: {fdn.mean():.3f}%",
        transform=ax.transAxes, ha="center", va="bottom", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

fig.text(0.99, 0.01, "データ: JQuants index_daily (2020-01〜2026-06) / FOMC日程: FRB公式",
         ha="right", va="bottom", fontsize=8, color="gray")
plt.tight_layout(rect=[0, 0.03, 1, 0.96])
out = Path(__file__).parent / "result.png"
fig.savefig(out, dpi=100, bbox_inches="tight", facecolor="white")
print(f"\nsaved {out}")

# ── サマリ出力 ────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
for col, name in [("topix","TOPIX"), ("nk225","日経225")]:
    r = results[col]
    sig = "有意" if r["p"] < 0.05 else ("弱い傾向" if r["p"] < 0.1 else "非有意")
    print(f"{name}: FOMC前日 {r['pre_mean']:+.3f}% vs 通常 {r['other_mean']:+.3f}%  "
          f"t={r['t']:.2f} p={r['p']:.3f} → {sig}")
print(f"\n本日は {pd.Timestamp.today().date()} = 次回FOMC 2026-06-18 の前日(-2)")
