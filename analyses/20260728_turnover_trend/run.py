import sys
sys.stdout.reconfigure(line_buffering=True)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from jstock import db  # 共通ライブラリ (PG_CONFIG手書き禁止)

# ============================================================
# 日本株 売買代金（全上場銘柄 close×volume 合計）の推移
#   - 母集団: stocks_daily 全上場銘柄。東証プライム公表値より母集団が広く水準は高め。
#     ただし推移・倍率のトレンドは正確。
#   - 単位: 兆円 / 表示は「1営業日あたり平均売買代金」
# ============================================================

MONTHLY_SQL = """
SELECT to_char(date,'YYYY-MM') ym,
       COUNT(DISTINCT date)                                   ndays,
       SUM(close*volume)/1e12                                 tot_tril,
       SUM(close*volume)/COUNT(DISTINCT date)/1e12            avg_daily_tril
FROM stocks_daily
WHERE date>='2023-01-01' AND close IS NOT NULL AND volume IS NOT NULL
GROUP BY 1 ORDER BY 1
"""

m = db.read_sql(MONTHLY_SQL)
m["ym_dt"] = pd.to_datetime(m["ym"] + "-01")
m.to_csv("monthly_turnover.csv", index=False)

# 年平均（1日あたり）
m["year"] = m["ym_dt"].dt.year
yearly = m.groupby("year").apply(
    lambda g: g["tot_tril"].sum() / g["ndays"].sum()
).round(2)
print("=== 年平均 1日売買代金（兆円）===")
print(yearly.to_string())
print("\n=== 2026 月次（1日平均・兆円）===")
print(m[m.year == 2026][["ym", "ndays", "avg_daily_tril"]].round(2).to_string(index=False))

# --- 可視化（日本語フォント・1200x675）---
try:
    import matplotlib.font_manager as fm
    fm.fontManager.addfont("/root/.fonts/NotoSansJP.ttf")
    plt.rcParams["font.family"] = fm.FontProperties(
        fname="/root/.fonts/NotoSansJP.ttf").get_name()
except Exception:
    pass
plt.rcParams["axes.unicode_minus"] = False

fig = plt.figure(figsize=(12, 6.75), facecolor="white")
ax = fig.add_subplot(111)
ax.set_facecolor("#f8f9fa")

colors = ["#c0392b" if y == 2026 else "#5b6b7a" for y in m["year"]]
ax.bar(m["ym_dt"], m["avg_daily_tril"], width=22, color=colors, alpha=0.9)

# 12ヶ月移動平均線
m_sorted = m.sort_values("ym_dt")
ma = m_sorted["avg_daily_tril"].rolling(12, min_periods=3).mean()
ax.plot(m_sorted["ym_dt"], ma, color="#e67e22", lw=2.2, label="12ヶ月移動平均")

ax.set_title("日本株 1日あたり平均売買代金の推移（月次）\n2026年に構造的ジャンプ：3〜6兆円 → 10〜13兆円",
             fontsize=16, fontweight="bold", pad=14)
ax.set_ylabel("1営業日あたり売買代金（兆円）", fontsize=11)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}"))
ax.grid(axis="y", alpha=0.3)
ax.legend(loc="upper left", fontsize=10)

# 直近値の注記
last = m_sorted.iloc[-1]
ax.annotate(f"{last['ym']}\n{last['avg_daily_tril']:.1f}兆円",
            xy=(last["ym_dt"], last["avg_daily_tril"]),
            xytext=(-10, 20), textcoords="offset points",
            fontsize=9, ha="right", color="#c0392b", fontweight="bold")

fig.text(0.99, 0.01,
         "データ: JQuants stocks_daily 全上場銘柄 close×volume 合計 / 2023-01〜2026-07",
         ha="right", va="bottom", fontsize=8, color="gray")

fig.savefig("result.png", dpi=100, bbox_inches="tight", facecolor="white")
print("\nsaved result.png")
