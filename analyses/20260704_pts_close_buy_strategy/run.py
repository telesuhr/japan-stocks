import sys
sys.stdout.reconfigure(line_buffering=True)
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from jstock import db, stats

# ── データ取得 ────────────────────────────────────────────
# 株式分割汚染（edge_pct < -10）を除外
df = db.read_sql("""
    SELECT trade_date, code, name, snap_hm,
           close, pts_last, edge_pct
    FROM aukabu.pts_quotes
    WHERE pts_last IS NOT NULL
      AND edge_pct IS NOT NULL
      AND edge_pct BETWEEN -10 AND 20
    ORDER BY trade_date, snap_hm, code
""")
df["trade_date"] = pd.to_datetime(df["trade_date"])
print(f"レコード数: {len(df)}  対象日: {df['trade_date'].nunique()}日")

# ── 時間帯別分布 ──────────────────────────────────────────
hm_stats = (df.groupby("snap_hm")["edge_pct"]
              .agg(avg="mean", median="median",
                   p25=lambda x: x.quantile(0.25),
                   p75=lambda x: x.quantile(0.75),
                   pct_pos=lambda x: (x > 0).mean() * 100,
                   n="count")
              .reset_index())
print("\n── 時間帯別 edge_pct ──")
print(hm_stats.to_string(index=False))

# ── 17:00 コスト控除後シミュレーション ───────────────────
COST = 0.20  # 往復コスト(%)
df17 = df[df["snap_hm"] == "17:00"].copy()
df17["net"] = df17["edge_pct"] - COST
df17["win"] = df17["net"] > 0

daily = (df17.groupby("trade_date")
             .agg(avg_net=("net", "mean"),
                  median_net=("net", "median"),
                  win_rate=("win", "mean"),
                  n=("net", "count"))
             .reset_index())
daily["win_rate_pct"] = daily["win_rate"] * 100
print("\n── 17:00売り コスト0.2%控除後 日別 ──")
print(daily[["trade_date","avg_net","median_net","win_rate_pct","n"]].to_string(index=False))

# ── 可視化 ────────────────────────────────────────────────
try:
    import matplotlib.font_manager as fm
    fp = fm.FontProperties(fname="/root/.fonts/NotoSansJP.ttf")
    plt.rcParams["font.family"] = fp.get_name()
except Exception:
    pass

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
fig.suptitle("PTS引け買い戦略検証（50銘柄・10日間）", fontsize=13)

# 左: 時間帯別中央値
ax = axes[0]
ax.bar(hm_stats["snap_hm"], hm_stats["median"], color="steelblue", alpha=0.8, label="中央値")
ax.axhline(0, color="black", lw=0.8)
ax.axhline(-COST, color="red", lw=1, ls="--", label=f"コスト閾値 -{COST}%")
ax.set_title("時間帯別 edge_pct 中央値")
ax.set_xlabel("PTS時刻")
ax.set_ylabel("edge_pct (%)")
ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.2f"))
ax.tick_params(axis="x", rotation=45)
ax.legend()

# 右: 日別ネット勝率(17:00)
ax2 = axes[1]
colors = ["green" if v >= 50 else "red" for v in daily["win_rate_pct"]]
ax2.bar(daily["trade_date"].dt.strftime("%m/%d"), daily["win_rate_pct"],
        color=colors, alpha=0.8)
ax2.axhline(50, color="black", lw=0.8, ls="--", label="勝率50%")
ax2.set_title("17:00売り コスト控除後 日別勝率")
ax2.set_xlabel("日付")
ax2.set_ylabel("ネット勝率 (%)")
ax2.set_ylim(0, 100)
ax2.tick_params(axis="x", rotation=45)
ax2.legend()

plt.tight_layout()
fig.savefig("result.png", dpi=100, bbox_inches="tight")
print("\nsaved result.png")
print("\n結論: 中央値はほぼ0、コスト控除後は一貫して負け越し → 戦略不成立")
