"""S高翌日のギャップは「市場が下げて始まる日」でも残るか。

背景: 2026-07-31 に SBG / 太陽誘電 / イビデン / アドテスト / SUMCO が S高引け。
     #20260731 の検証で「S高は持ち越して翌営業日の寄りで売るのが最良」
     (翌ギャップ +3.66%, 超大型は +3.79%) と結論した。
     ところが CME 日経先物は 62,925 と現物 64,299 に対し -2.14%。
     ベースレートは無条件平均であり、市場全体が大きく下げて始まる日には
     当てはまらない可能性がある。ここを条件付きで検算する。

仮説（事前登録）:
  H1: S高翌日の個別ギャップは、市場ギャップが下がるほど小さくなる（当然）が、
      市場ギャップを控除した「超過ギャップ」は市場ギャップに依存せず安定して正。
      → 棄却条件: 市場ギャップ ≤ -1.5% の層で超過ギャップの t < 2 または符号が負。
  H2: 市場ギャップ ≤ -1.5% の層でも、生ギャップ（実際に手にする値段）の
      中央値は正 = 「寄りで売る」が前日終値より有利。
      → 棄却条件: 生ギャップの中央値 ≤ 0。
  H3: S高当日に出来高が枯れた銘柄（買い気配張り付き = 需要未消化）ほど翌ギャップが大きい。
      → 棄却条件: 出来高比 (当日出来高/ADV20) 下位層と上位層の差が t < 2。
      ※ 2026-07-31 の太陽誘電(前日比 -97%)・イビデン(-87%)がまさにこの型。
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sps

from jstock import db

ADV_MIN = 3e8  # 流動性フィルタ 3億円

SQL = """
WITH px AS (
  SELECT code, date, adj_close, adj_open, upper_limit, turnover_value,
         AVG(turnover_value) OVER (PARTITION BY code ORDER BY date
             ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING) AS adv20,
         LEAD(date)      OVER w AS d1,
         LEAD(adj_open)  OVER w AS o1,
         LEAD(adj_close) OVER w AS c1
  FROM stocks_daily
  WHERE date >= '2016-01-01'
  WINDOW w AS (PARTITION BY code ORDER BY date)
)
SELECT p.code, m.name_ja, m.sector33_nm, p.date, p.adj_close, p.turnover_value,
       p.adv20, p.d1, p.o1, p.c1
FROM px p JOIN symbol_master m ON m.code5 = p.code
WHERE p.upper_limit = TRUE AND p.adv20 >= %s AND p.o1 IS NOT NULL
"""

ev = db.read_sql(SQL, [ADV_MIN])
print(f"S高イベント: {len(ev):,}件  {ev['date'].min()} 〜 {ev['date'].max()}")

# 市場（TOPIX）のギャップ
ix = db.read_sql(
    "SELECT date, open, close FROM index_daily WHERE code='0000' ORDER BY date", []
)
ix["pc"] = ix["close"].shift(1)
ix["mkt_gap"] = (ix["open"] / ix["pc"] - 1) * 100
ix["mkt_o2c"] = (ix["close"] / ix["open"] - 1) * 100

ev = ev.merge(ix[["date", "mkt_gap", "mkt_o2c"]], left_on="d1", right_on="date",
              how="inner", suffixes=("", "_ix")).drop(columns=["date_ix"])

ev["gap"] = (ev["o1"] / ev["adj_close"] - 1) * 100          # 翌日の生ギャップ
ev["exgap"] = ev["gap"] - ev["mkt_gap"]                      # 市場控除後の超過ギャップ
ev["o2c"] = (ev["c1"] / ev["o1"] - 1) * 100                  # 翌日の寄→引
ev["volr"] = ev["turnover_value"] / ev["adv20"]              # 当日出来高 / ADV20
ev = ev.dropna(subset=["gap", "mkt_gap"])
print(f"市場ギャップ結合後: {len(ev):,}件")


def desc(s):
    s = s.dropna()
    if len(s) < 5:
        return dict(n=len(s), mean=np.nan, med=np.nan, t=np.nan, win=np.nan)
    return dict(n=len(s), mean=s.mean(), med=s.median(),
                t=sps.ttest_1samp(s, 0).statistic, win=(s > 0).mean() * 100)


print("\n" + "=" * 78)
print("【H1/H2】翌日の市場ギャップ別 — S高銘柄の翌ギャップ")
print("=" * 78)
bins = [-99, -1.5, -0.5, 0.5, 1.5, 99]
labels = ["市場≤-1.5%", "-1.5〜-0.5%", "-0.5〜+0.5%", "+0.5〜+1.5%", "市場>+1.5%"]
ev["mkt_bin"] = pd.cut(ev["mkt_gap"], bins=bins, labels=labels)

rows = []
for lab in labels:
    sub = ev[ev["mkt_bin"] == lab]
    g, x = desc(sub["gap"]), desc(sub["exgap"])
    rows.append(dict(層=lab, n=g["n"],
                     生ギャップ平均=g["mean"], 生ギャップ中央値=g["med"],
                     生勝率=g["win"], 超過平均=x["mean"], 超過t=x["t"],
                     翌日寄引=desc(sub["o2c"])["mean"]))
tbl = pd.DataFrame(rows)
print(tbl.to_string(index=False, float_format=lambda v: f"{v:7.2f}"))

print("\n【全体】")
for k, v in {"生ギャップ": "gap", "超過ギャップ": "exgap", "翌日寄→引": "o2c"}.items():
    d = desc(ev[v])
    print(f"  {k:<10} n={d['n']:5d} 平均{d['mean']:+.2f}% 中央値{d['med']:+.2f}% "
          f"t={d['t']:+.2f} 勝率{d['win']:.0f}%")

print("\n" + "=" * 78)
print("【H3】S高当日の出来高比（当日売買代金/ADV20）別 — 買い気配張り付きほど翌ギャップ大か")
print("=" * 78)
vb = [0, 0.5, 1.0, 2.0, 5.0, 1e9]
vl = ["<0.5x(気配張付)", "0.5-1x", "1-2x", "2-5x", ">5x(大商い)"]
ev["vol_bin"] = pd.cut(ev["volr"], bins=vb, labels=vl)
rows = []
for lab in vl:
    sub = ev[ev["vol_bin"] == lab]
    g, x = desc(sub["gap"]), desc(sub["exgap"])
    rows.append(dict(出来高比=lab, n=g["n"], 生ギャップ平均=g["mean"],
                     生ギャップ中央値=g["med"], 生勝率=g["win"],
                     超過平均=x["mean"], 超過t=x["t"]))
vtbl = pd.DataFrame(rows)
print(vtbl.to_string(index=False, float_format=lambda v: f"{v:7.2f}"))

lo = ev[ev["volr"] < 0.5]["exgap"].dropna()
hi = ev[ev["volr"] >= 2.0]["exgap"].dropna()
tt = sps.ttest_ind(lo, hi, equal_var=False)
print(f"\n  気配張付(<0.5x, n={len(lo)}) {lo.mean():+.2f}%  vs  "
      f"大商い(>=2x, n={len(hi)}) {hi.mean():+.2f}%  "
      f"差 {lo.mean()-hi.mean():+.2f}pt  t={tt.statistic:+.2f} p={tt.pvalue:.4f}")

print("\n" + "=" * 78)
print("【核心】市場が大きく下げて始まる日(市場ギャップ≤-1.5%)の S高銘柄")
print("=" * 78)
crash = ev[ev["mkt_gap"] <= -1.5]
d = desc(crash["gap"])
print(f"  生ギャップ    n={d['n']} 平均{d['mean']:+.2f}% 中央値{d['med']:+.2f}% "
      f"t={d['t']:+.2f} 勝率{d['win']:.0f}%")
x = desc(crash["exgap"])
print(f"  超過ギャップ  平均{x['mean']:+.2f}% 中央値{x['med']:+.2f}% t={x['t']:+.2f}")
for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
    print(f"    {int(q*100):>3d}%タイル: {crash['gap'].quantile(q):+.2f}%")
neg = (crash["gap"] < 0).mean() * 100
bad = (crash["gap"] < -3).mean() * 100
print(f"  ギャップがマイナス: {neg:.0f}%  /  -3%以下: {bad:.0f}%")

# 超大型（ADV>=100億）に限定
big = crash[crash["adv20"] >= 1e10]
if len(big) >= 5:
    d = desc(big["gap"])
    print(f"\n  うち超大型(ADV>=100億) n={d['n']} 平均{d['mean']:+.2f}% "
          f"中央値{d['med']:+.2f}% 勝率{d['win']:.0f}%")

# 気配張り付き × 市場下落 のクロス（太陽誘電・イビデンの型）
cross = crash[crash["volr"] < 0.5]
if len(cross) >= 5:
    d = desc(cross["gap"])
    print(f"  うち気配張付(<0.5x)  n={d['n']} 平均{d['mean']:+.2f}% "
          f"中央値{d['med']:+.2f}% 勝率{d['win']:.0f}%")

ev.to_csv("limitup_events.csv", index=False)
tbl.to_csv("by_market_gap.csv", index=False)
vtbl.to_csv("by_volume_ratio.csv", index=False)

# ---------------- 可視化 ----------------
try:
    import matplotlib.font_manager as fm
    fm.fontManager.addfont("/root/.fonts/NotoSansJP.ttf")
    plt.rcParams["font.family"] = fm.FontProperties(
        fname="/root/.fonts/NotoSansJP.ttf").get_name()
except Exception:
    pass
plt.rcParams["axes.unicode_minus"] = False

fig = plt.figure(figsize=(12, 6.75), facecolor="white")
fig.suptitle("S高の翌ギャップは「市場が下げて始まる日」でも残るか",
             fontsize=16, fontweight="bold", y=0.985)

ax = fig.add_subplot(131)
xs = np.arange(len(tbl))
ax.bar(xs - 0.2, tbl["生ギャップ平均"], 0.4, color="#1f77b4", label="生ギャップ")
ax.bar(xs + 0.2, tbl["超過平均"], 0.4, color="#ff7f0e", label="超過(市場控除後)")
ax.set_xticks(xs)
ax.set_xticklabels([l.replace("市場", "") for l in tbl["層"]], rotation=45,
                   ha="right", fontsize=8)
ax.axhline(0, color="k", lw=0.8)
ax.set_title("翌日の市場ギャップ別", fontsize=11)
ax.set_ylabel("S高銘柄の翌ギャップ (%)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y")
for i, (n, v) in enumerate(zip(tbl["n"], tbl["生ギャップ平均"])):
    ax.text(i - 0.2, v + 0.15, f"n{n}", ha="center", fontsize=7)

ax = fig.add_subplot(132)
xs = np.arange(len(vtbl))
ax.bar(xs, vtbl["生ギャップ平均"], 0.6,
       color=["#d62728" if l.startswith("<0.5") else "#7f9fbf" for l in vtbl["出来高比"]])
ax.set_xticks(xs)
ax.set_xticklabels(vtbl["出来高比"], rotation=45, ha="right", fontsize=8)
ax.axhline(0, color="k", lw=0.8)
ax.set_title("S高当日の出来高比別\n(赤=買い気配張り付き)", fontsize=11)
ax.set_ylabel("翌ギャップ (%)")
ax.grid(alpha=0.3, axis="y")
for i, (n, v) in enumerate(zip(vtbl["n"], vtbl["生ギャップ平均"])):
    ax.text(i, v + 0.15, f"n{n}", ha="center", fontsize=7)

ax = fig.add_subplot(133)
ax.hist(crash["gap"].clip(-15, 20), bins=35, color="#2ca02c", alpha=0.8)
ax.axvline(0, color="k", lw=1.2)
ax.axvline(crash["gap"].median(), color="red", lw=1.6, ls="--",
           label=f"中央値 {crash['gap'].median():+.2f}%")
ax.set_title(f"市場ギャップ≤-1.5%の日\nS高銘柄の翌ギャップ分布 (n={len(crash)})",
             fontsize=11)
ax.set_xlabel("翌ギャップ (%)")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

fig.text(0.99, 0.005,
         f"データ: stocks_daily / index_daily(TOPIX) 2016-01〜2026-07 "
         f"・S高引け{len(ev):,}件(ADV≥3億) / 日本株",
         ha="right", va="bottom", fontsize=8, color="gray")
fig.tight_layout(rect=[0, 0.02, 1, 0.96])
fig.savefig("result.png", dpi=100, bbox_inches="tight", facecolor="white")
print("\nsaved result.png")
