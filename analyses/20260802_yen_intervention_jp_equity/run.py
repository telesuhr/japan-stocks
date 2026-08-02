"""円買い介入 / 急激な円高ショックの後、日本株はどうなるか。

背景: 2026-07-30〜31 に日米が円買い協調介入（報道ベース。協調は1998-06以来28年ぶり）。
     推計6兆円超。USDJPY 163.66 → NYクローズ157.40（-3.8%）。
     「協調介入だから効果が持続する」「円高で日本株は下がる」という2つの通説を検算する。

仮説（事前登録）:
  H1: 円買い介入後、円高は持続する（介入日から d20 まで USDJPY は戻さない）。
      → 棄却条件: d20 の USDJPY が介入日終値を上回る（=円安に戻る）ケースが過半。
  H2: 円高ショック後、日本株はアンダーパフォームする。
      → 棄却条件: d5/d20 の日本株リターンの符号が正、または t < -2 に届かない。
  H3: 円高ショック後は「輸出 < 内需」のセクター分岐が起きる。
      → 棄却条件: 輸出−内需スプレッドが d20 で t > -2（有意な負にならない）。

注意: 「協調介入」の先例は本データ範囲(2000〜)で 2011-03-18 のG7協調のみ = n=1。
     協調介入そのもののベースレートは統計的に構築不能。ここは正直に n=1 と明示し、
     より母数の取れる「円高ショック」で代替する。
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

HORIZONS = [1, 5, 10, 20]

# ---- 公表ベースの円買い介入日（財務省「外国為替平衡操作の実施状況」＋報道） ----
# 円売り介入(2003-04等)は方向が逆なので除外。
INTERVENTIONS = {
    "2010-09-15": "単独（6年半ぶり・2.1兆円）",
    "2011-03-18": "★G7協調（震災後の急激な円高）",
    "2011-08-04": "単独（4.5兆円）",
    "2011-10-31": "単独（過去最大 8.1兆円）",
    "2022-09-22": "単独（24年ぶり・2.8兆円）",
    "2022-10-21": "単独（覆面・5.6兆円）",
    "2024-04-29": "単独（覆面・5.9兆円）",
    "2024-07-11": "単独（覆面・3.2兆円）",
}

fx = db.read_sql(
    "SELECT trade_date AS d, close FROM macro.daily_ohlcv "
    "WHERE symbol='JPY=' ORDER BY trade_date", []).set_index("d")["close"]
nk = db.read_sql(
    "SELECT trade_date AS d, close FROM macro.daily_ohlcv "
    "WHERE symbol='JNIc1' ORDER BY trade_date", []).set_index("d")["close"]
print(f"JPY=  {fx.index.min()} 〜 {fx.index.max()}  n={len(fx):,}")
print(f"JNIc1 {nk.index.min()} 〜 {nk.index.max()}  n={len(nk):,}")

fx.index = pd.to_datetime(fx.index)
nk.index = pd.to_datetime(nk.index)


def fwd(series, dt, h):
    """dt 以降の営業日で h 日先のリターン(%)。dt が無ければ直近の過去日に寄せる。"""
    idx = series.index
    pos = idx.searchsorted(pd.Timestamp(dt))
    if pos >= len(idx):
        return np.nan
    if pos + h >= len(idx):
        return np.nan
    return (series.iloc[pos + h] / series.iloc[pos] - 1) * 100


print("\n" + "=" * 84)
print("【H1】円買い介入後、円高は持続したか（USDJPY のその後。負=円高が進む）")
print("=" * 84)
rows = []
for dt, label in INTERVENTIONS.items():
    r = {"介入日": dt, "内容": label}
    for h in HORIZONS:
        r[f"USDJPY d{h}"] = fwd(fx, dt, h)
    rows.append(r)
ivfx = pd.DataFrame(rows)
print(ivfx.to_string(index=False, float_format=lambda v: f"{v:6.2f}"))
for h in HORIZONS:
    c = ivfx[f"USDJPY d{h}"].dropna()
    print(f"  d{h:<2}: 平均{c.mean():+.2f}%  中央値{c.median():+.2f}%  "
          f"円安に戻った割合 {(c > 0).mean()*100:.0f}%  (n={len(c)})")

print("\n" + "=" * 84)
print("【H2-a】同じ介入日の後、日本株(日経225先物 JNIc1)はどうだったか")
print("=" * 84)
rows = []
for dt, label in INTERVENTIONS.items():
    r = {"介入日": dt, "内容": label}
    for h in HORIZONS:
        r[f"日経 d{h}"] = fwd(nk, dt, h)
    rows.append(r)
ivnk = pd.DataFrame(rows)
print(ivnk.to_string(index=False, float_format=lambda v: f"{v:6.2f}"))
for h in HORIZONS:
    c = ivnk[f"日経 d{h}"].dropna()
    if len(c) >= 3:
        t = sps.ttest_1samp(c, 0).statistic
        print(f"  d{h:<2}: 平均{c.mean():+.2f}%  中央値{c.median():+.2f}%  "
              f"勝率{(c > 0).mean()*100:.0f}%  t={t:+.2f}  (n={len(c)})")

print("\n※ 協調介入の先例は 2011-03-18 の G7 協調のみ = n=1。個別に表示:")
for h in HORIZONS:
    print(f"    2011-03-18(G7協調) d{h:<2}: USDJPY {fwd(fx,'2011-03-18',h):+6.2f}%  "
          f"日経 {fwd(nk,'2011-03-18',h):+6.2f}%")

print("\n" + "=" * 84)
print("【H2-b】母数を取る: 3日累計の円高ショック（JPY= が3日で -3%以上）後の日本株")
print("=" * 84)
fx3 = (fx / fx.shift(3) - 1) * 100
shock_dates = fx3[fx3 <= -3.0].index
# 連続する日は先頭のみ採用（重複イベント除去）
keep, last = [], None
for d in shock_dates:
    if last is None or (d - last).days > 20:
        keep.append(d)
    last = d
print(f"円高ショック(3日で-3%以下) 独立イベント: {len(keep)}件  "
      f"{keep[0].date()} 〜 {keep[-1].date()}")

rows = []
for d in keep:
    r = {"日付": d.date(), "3日変化": fx3.loc[d]}
    for h in HORIZONS:
        r[f"日経 d{h}"] = fwd(nk, d, h)
        r[f"FX d{h}"] = fwd(fx, d, h)
    rows.append(r)
sh = pd.DataFrame(rows)
print("\n【日本株(日経先物)】")
for h in HORIZONS:
    c = sh[f"日経 d{h}"].dropna()
    t = sps.ttest_1samp(c, 0).statistic
    print(f"  d{h:<2}: 平均{c.mean():+.2f}%  中央値{c.median():+.2f}%  "
          f"勝率{(c > 0).mean()*100:3.0f}%  t={t:+.2f}  (n={len(c)})")
print("【USDJPY（負=円高継続 / 正=円安に戻す）】")
for h in HORIZONS:
    c = sh[f"FX d{h}"].dropna()
    print(f"  d{h:<2}: 平均{c.mean():+.2f}%  中央値{c.median():+.2f}%  "
          f"円安に戻った割合 {(c > 0).mean()*100:3.0f}%  (n={len(c)})")

# 2008年の金融危機を除いた頑健性（円高ショックの半分が2008年に集中するため）
sh["year"] = pd.to_datetime(sh["日付"]).dt.year
ex08 = sh[sh["year"] != 2008]
print(f"\n【2008年を除く n={len(ex08)}】")
for h in HORIZONS:
    c = ex08[f"日経 d{h}"].dropna()
    if len(c) >= 3:
        t = sps.ttest_1samp(c, 0).statistic
        print(f"  日経 d{h:<2}: 平均{c.mean():+.2f}%  中央値{c.median():+.2f}%  "
              f"勝率{(c > 0).mean()*100:3.0f}%  t={t:+.2f}  (n={len(c)})")

print("\n" + "=" * 84)
print("【H3】円高ショック後のセクター分岐（輸出 vs 内需、2016-05以降の個別株）")
print("=" * 84)
EXPORT = ["輸送用機器", "電気機器", "精密機器", "機械"]
DOMESTIC = ["陸運業", "電気・ガス業", "小売業", "食料品", "建設業",
            "情報・通信業", "サービス業", "銀行業"]

sec = db.read_sql("""
    SELECT s.date, m.sector33_nm AS sec, s.code,
           s.adj_close, s.turnover_value
    FROM stocks_daily s JOIN symbol_master m ON m.code5 = s.code
    WHERE s.date >= '2016-05-01' AND m.sector33_nm = ANY(%s)
""", [EXPORT + DOMESTIC])
sec["date"] = pd.to_datetime(sec["date"])
sec = sec.sort_values(["code", "date"])
sec["ret"] = sec.groupby("code")["adj_close"].pct_change() * 100
# 流動性フィルタ: 直近20日平均売買代金 >= 5億
sec["adv20"] = (sec.groupby("code")["turnover_value"]
                .transform(lambda s: s.rolling(20, min_periods=10).mean().shift(1)))
sec = sec[sec["adv20"] >= 5e8]
sec["grp"] = np.where(sec["sec"].isin(EXPORT), "輸出", "内需")

# グループ別の等加重日次リターン
gd = sec.groupby(["date", "grp"])["ret"].mean().unstack()
gd["スプレッド"] = gd["輸出"] - gd["内需"]
print(f"日次系列: {gd.index.min().date()} 〜 {gd.index.max().date()} "
      f"({len(gd)}営業日) / 輸出・内需の等加重(ADV≥5億)")

keep16 = [d for d in keep if d >= pd.Timestamp("2016-06-01")]
print(f"対象イベント（2016-06以降の円高ショック）: {len(keep16)}件 "
      f"{[str(d.date()) for d in keep16]}")

rows = []
for d in keep16:
    pos = gd.index.searchsorted(d)
    r = {"日付": d.date()}
    for h in HORIZONS:
        if pos + h < len(gd):
            w = gd.iloc[pos + 1: pos + 1 + h]
            r[f"輸出 d{h}"] = w["輸出"].sum()
            r[f"内需 d{h}"] = w["内需"].sum()
            r[f"差 d{h}"] = w["スプレッド"].sum()
    rows.append(r)
sp = pd.DataFrame(rows)
print(sp.to_string(index=False, float_format=lambda v: f"{v:6.2f}"))
print()
for h in HORIZONS:
    c = sp[f"差 d{h}"].dropna()
    if len(c) >= 3:
        t = sps.ttest_1samp(c, 0).statistic
        print(f"  輸出−内需 d{h:<2}: 平均{c.mean():+.2f}pt  中央値{c.median():+.2f}pt  "
              f"輸出劣後の割合{(c < 0).mean()*100:3.0f}%  t={t:+.2f}  (n={len(c)})")

# 全期間ベースライン（円高ショックと関係ない普段のスプレッド）
for h in HORIZONS:
    base = gd["スプレッド"].rolling(h).sum().dropna()
    print(f"    参考ベースライン d{h:<2}: 平均{base.mean():+.2f}pt (全{len(base)}窓)")

sh.to_csv("yen_shock_events.csv", index=False)
sp.to_csv("sector_spread.csv", index=False)
ivfx.merge(ivnk, on=["介入日", "内容"]).to_csv("intervention_events.csv", index=False)

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
fig.suptitle("円買い介入・円高ショックの後、日本株はどうなったか",
             fontsize=16, fontweight="bold", y=0.985)

ax = fig.add_subplot(131)
xs = np.arange(len(HORIZONS))
fxm = [ivfx[f"USDJPY d{h}"].mean() for h in HORIZONS]
back = [(ivfx[f"USDJPY d{h}"] > 0).mean() * 100 for h in HORIZONS]
ax.bar(xs, fxm, 0.6, color="#d62728")
ax.set_xticks(xs); ax.set_xticklabels([f"d{h}" for h in HORIZONS])
ax.axhline(0, color="k", lw=0.8)
ax.set_title("介入後のUSDJPY (n=8)\n正=円安に戻った", fontsize=11)
ax.set_ylabel("USDJPY 変化率 (%)")
ax.grid(alpha=0.3, axis="y")
for i, (v, b) in enumerate(zip(fxm, back)):
    ax.text(i, v + 0.08, f"戻り{b:.0f}%", ha="center", fontsize=8)

ax = fig.add_subplot(132)
nkm = [sh[f"日経 d{h}"].mean() for h in HORIZONS]
nkx = [ex08[f"日経 d{h}"].mean() for h in HORIZONS]
ax.bar(xs - 0.2, nkm, 0.4, color="#7f7f7f", label=f"全期間 n={len(sh)}")
ax.bar(xs + 0.2, nkx, 0.4, color="#1f77b4", label=f"2008年除く n={len(ex08)}")
ax.set_xticks(xs); ax.set_xticklabels([f"d{h}" for h in HORIZONS])
ax.axhline(0, color="k", lw=0.8)
ax.set_title("円高ショック(3日で-3%)後の日経", fontsize=11)
ax.set_ylabel("日経225先物 リターン (%)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y")

ax = fig.add_subplot(133)
ex = [sp[f"輸出 d{h}"].mean() for h in HORIZONS]
dm = [sp[f"内需 d{h}"].mean() for h in HORIZONS]
ax.bar(xs - 0.2, ex, 0.4, color="#ff7f0e", label="輸出")
ax.bar(xs + 0.2, dm, 0.4, color="#2ca02c", label="内需")
ax.set_xticks(xs); ax.set_xticklabels([f"d{h}" for h in HORIZONS])
ax.axhline(0, color="k", lw=0.8)
ax.set_title(f"円高ショック後のセクター\n(2016-06以降 n={len(sp)})", fontsize=11)
ax.set_ylabel("累積リターン (%)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y")

fig.text(0.99, 0.005,
         "データ: macro.daily_ohlcv (JPY=, JNIc1) 2000-2026 / stocks_daily 2016-05〜 "
         "・介入日は財務省公表＋報道ベース",
         ha="right", va="bottom", fontsize=8, color="gray")
fig.tight_layout(rect=[0, 0.02, 1, 0.96])
fig.savefig("result.png", dpi=100, bbox_inches="tight", facecolor="white")
print("\nsaved result.png")
