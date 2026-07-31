"""
追加検証: 「ストップ高で引けた銘柄」の翌日以降。
本体(run.py)は「ギャップアップ反発日の追随は報われない」と結論したが、
ストップ高は"買い需要が値幅制限で未消化のまま翌日に持ち越される"別現象の可能性がある。

H5: ストップ高引けの翌日は、同程度の大幅高(ストップ高でない)より強い(未消化需要の持ち越し)
H6: その強さは「寄りのギャップ」に集約され、寄→引は取れない(本体H3と同じ形)
H7: 全面高の日(市場が急騰し多数がストップ高)のストップ高は、個別材料の単独ストップ高より弱い
    (=セクター踏み上げの産物で、銘柄固有の新情報ではない)
"""
import sys; sys.stdout.reconfigure(line_buffering=True)
from pathlib import Path
import pandas as pd, numpy as np
from scipy import stats as sps
import matplotlib.pyplot as plt
from jstock import db

HERE = Path(__file__).resolve().parent

# 大幅高イベント(ストップ高 / 非ストップ高)を流動性込みで抽出
ev = db.read_sql("""
WITH u AS (
  SELECT code, date, adj_open, adj_close, upper_limit, turnover_value,
    LAG(adj_close) OVER (PARTITION BY code ORDER BY date) pc,
    AVG(turnover_value) OVER (PARTITION BY code ORDER BY date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING) adv20,
    LEAD(adj_open,  1) OVER (PARTITION BY code ORDER BY date) o1,
    LEAD(adj_close, 1) OVER (PARTITION BY code ORDER BY date) c1,
    LEAD(adj_close, 2) OVER (PARTITION BY code ORDER BY date) c2,
    LEAD(adj_close, 3) OVER (PARTITION BY code ORDER BY date) c3,
    LEAD(adj_close, 5) OVER (PARTITION BY code ORDER BY date) c5,
    LEAD(adj_close,10) OVER (PARTITION BY code ORDER BY date) c10
  FROM stocks_daily WHERE date >= '2016-01-01'
)
SELECT code, date, upper_limit, adv20,
  (adj_close/pc - 1)*100 AS ret,
  (o1/adj_close - 1)*100 AS nx_gap,      -- 翌日ギャップ
  (c1/o1 - 1)*100        AS nx_o2c,      -- 翌日 寄→引
  (c1/adj_close - 1)*100 AS d1,
  (c2/adj_close - 1)*100 AS d2,
  (c3/adj_close - 1)*100 AS d3,
  (c5/adj_close - 1)*100 AS d5,
  (c10/adj_close - 1)*100 AS d10
FROM u
WHERE pc IS NOT NULL AND o1 > 0 AND adv20 >= 5e8   -- ADV>=5億(SBG/太陽誘電クラスの流動性)
  AND (adj_close/pc - 1)*100 >= 8                   -- 大幅高のみ比較
""", [])
for c in ["ret", "nx_gap", "nx_o2c", "d1", "d2", "d3", "d5", "d10", "adv20"]:
    ev[c] = ev[c].astype(float)
ev["date"] = pd.to_datetime(ev["date"])

# 市場全体の急騰日(=今日のような全面高)フラグ
mkt = pd.read_csv(HERE / "market_daily.csv", parse_dates=["date"]).set_index("date")
ev = ev.join(mkt["ew_ret"].rename("mkt_ret"), on="date")
ev["broad_rally"] = ev["mkt_ret"] >= 2.0

lim = ev[ev["upper_limit"]]
non = ev[~ev["upper_limit"]]
print(f"ADV>=5億・当日+8%以上の大幅高: n={len(ev)}  うちストップ高引け n={len(lim)} / 非ストップ高 n={len(non)}")
print(f"  ストップ高の当日上昇率 平均 {lim['ret'].mean():.1f}% / 非ストップ高 {non['ret'].mean():.1f}%")

HS = ["d1", "d2", "d3", "d5", "d10"]

def line(lbl, g):
    if len(g) == 0:
        return
    t1 = sps.ttest_1samp(g["d1"].dropna(), 0)[0] if len(g) > 2 else np.nan
    print(f"  {lbl:34s} n={len(g):5d} | 翌ギャップ {g['nx_gap'].mean():+6.2f}  翌寄→引 {g['nx_o2c'].mean():+6.2f} | "
          + "  ".join(f"{h.upper()} {g[h].mean():+6.2f}" for h in HS)
          + f" | D+1勝率 {(g['d1'].dropna()>0).mean()*100:3.0f}%  t {t1:+5.2f}")

print("\n=== H5: ストップ高 vs 同程度の大幅高(非ストップ高) 単位:% ===")
line("ストップ高引け", lim)
line("非ストップ高(+8%以上)", non)

print("\n=== H6: 内訳(翌日ギャップ vs 寄→引) ===")
print(f"  ストップ高: 翌日D+1 {lim['d1'].mean():+.2f}% のうち ギャップ {lim['nx_gap'].mean():+.2f}% / "
      f"寄→引 {lim['nx_o2c'].mean():+.2f}%  (寄→引の勝率 {(lim['nx_o2c'].dropna()>0).mean()*100:.0f}%)")
print(f"  → 翌日の上げのギャップ寄与 {lim['nx_gap'].mean()/lim['d1'].mean()*100:.0f}%")

print("\n=== H7: 全面高の日のストップ高か、単独のストップ高か ===")
line("ストップ高 × 全面高日(市場+2%↑)", lim[lim["broad_rally"]])
line("ストップ高 × 通常日", lim[~lim["broad_rally"]])

print("\n=== 参考: ストップ高の当日上昇率で層別 ===")
for lo, hi in [(8, 12), (12, 17), (17, 100)]:
    line(f"ストップ高 {lo}〜{hi}%高", lim[(lim["ret"] >= lo) & (lim["ret"] < hi)])

print("\n=== 参考: 超大型(ADV>=100億)のストップ高 = SBG級 ===")
line("ストップ高 × ADV>=100億", lim[lim["adv20"] >= 1e10])
line("ストップ高 × ADV 5〜100億", lim[lim["adv20"] < 1e10])

ev.to_csv(HERE / "limit_up_events.csv", index=False)

# ---------- 可視化 ----------
import matplotlib.font_manager as fm
fm.fontManager.addfont("/root/.fonts/NotoSansJP.ttf")
plt.rcParams["font.family"] = fm.FontProperties(fname="/root/.fonts/NotoSansJP.ttf").get_name()
plt.rcParams["axes.unicode_minus"] = False

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.4), facecolor="white")
xs = [0, 1, 2, 3, 5, 10]
for lbl, g, c in [("ストップ高引け", lim, "#c0392b"),
                  ("非ストップ高の大幅高(+8%↑)", non, "#8fa9bf")]:
    ax1.plot(xs, [0] + [g[h].mean() for h in HS], marker="o", lw=2.2, color=c, label=f"{lbl} (n={len(g)})")
ax1.axhline(0, color="#333", lw=0.8); ax1.grid(alpha=0.3)
ax1.set_xlabel("経過営業日"); ax1.set_ylabel("平均リターン(%)"); ax1.legend(fontsize=9)
ax1.set_title("ストップ高は「未消化の買い需要」が翌日に残るか", fontsize=12, fontweight="bold")

g1, g2 = lim[lim["broad_rally"]], lim[~lim["broad_rally"]]
w = 0.36; x = np.arange(2)
ax2.bar(x - w/2, [g1["nx_gap"].mean(), g2["nx_gap"].mean()], w, color="#2e7d32", label="翌日ギャップ")
ax2.bar(x + w/2, [g1["nx_o2c"].mean(), g2["nx_o2c"].mean()], w, color="#c0392b", label="翌日 寄→引")
ax2.set_xticks(x)
ax2.set_xticklabels([f"全面高日のS高\n(n={len(g1)}) ←今日", f"単独材料のS高\n(n={len(g2)})"])
ax2.axhline(0, color="#333", lw=0.8); ax2.grid(axis="y", alpha=0.3)
ax2.set_ylabel("翌日リターン(%)"); ax2.legend(fontsize=9)
ax2.set_title("同じストップ高でも「全面高の産物」は続かない", fontsize=12, fontweight="bold")

fig.suptitle("ストップ高で引けた翌日はどうなるか — ADV≥5億・2016-2026", fontsize=13, fontweight="bold")
fig.text(0.99, 0.005, "データ: JQuants stocks_daily (upper_limitフラグ)", ha="right", fontsize=8, color="gray")
fig.tight_layout()
fig.savefig(HERE / "result_limit_up.png", dpi=100, bbox_inches="tight", facecolor="white")
print("\nsaved result_limit_up.png")
