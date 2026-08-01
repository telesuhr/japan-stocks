"""
earnings_pead × 会社予想ガイダンス修正 の層別
=============================================
採用戦略 earnings_pead (決算翌日ギャップ+7%以上 → 当日引成Long → D+5引成, SL-5%, Sh+2.19)
に対し、「決算と同時に会社が業績予想を修正したか／どちら向きにどれだけ修正したか」
でフィルタをかけると改善するかを検証する。

背景（既出2件の緊張関係）:
 - `20260512_earnings_pead_validation` / 採用戦略: 大幅ギャップは買い（ドリフトが取れる）
 - `20260613_earnings_forecast_revision_drift`: 最大級の上方修正(Q5)は寄り後にアンダー
   = overshoot-reversal（寄り高を作りすぎて戻す）
   → 「大幅ギャップ」かつ「大幅上方修正」はどちらの力が勝つのか未検証。

仮説（検証前に固定 — 教訓5）:
 H1: 同時に上方修正した決算ギャップはドリフトが強い（good news の裏付け）
     → 修正なし群・下方修正群よりリターン高い
 H2: ただし最大級の上方修正は overshoot-reversal でドリフトが消える → 修正幅に対し逆U字
 H3: ギャップアップなのに下方修正/据置 = 材料出尽くし → 最弱
 H4: 最良層でフィルタすると earnings_pead の net Sharpe が改善する（実用判定）

棄却条件: 層別間の差が t<2、かつ最良層のSharpe改善が +0.3 未満ならフィルタ無効として棄却。

規律: 往復4bps控除 / 日次バスケット系列で√252年率化(非重複) / IS-OOS分割 /
      先読みなし(修正はDay N-1引け後開示＝Day N 15:30エントリー時点で既知)
"""
import sys; sys.stdout.reconfigure(line_buffering=True)
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats as sps
import matplotlib.pyplot as plt
from jstock import db

HERE = Path(__file__).resolve().parent
COST_BPS = 4.0          # 往復 (片側2bps)
SL_PCT = -5.0
HOLD = 5
GAP_MIN = 7.0
ADV_MIN = 5e8

# ---------------------------------------------------------------- 1. イベント抽出
# 決算開示(D-1 15:00以降 or D 9:00前) → D寄りギャップ>=+7% → D引成Long → D+5引成
SQL = """
WITH fin AS (   -- 決算開示を「反映される取引日」に正規化
  SELECT DISTINCT code,
    CASE WHEN disc_time >= '15:00' THEN disc_date ELSE disc_date - 1 END AS prev_date,
    disc_date AS fin_disc_date
  FROM fin_summary
  WHERE disc_date >= '2016-01-01'
),
px AS (
  SELECT code, date, adj_open, adj_close, adj_low, turnover_value,
    LAG(adj_close) OVER w  AS pc,
    LAG(date)      OVER w  AS pdate,
    AVG(turnover_value) OVER (PARTITION BY code ORDER BY date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING) AS adv20,
    LEAD(date,1) OVER w d1, LEAD(date,2) OVER w d2, LEAD(date,3) OVER w d3,
    LEAD(date,4) OVER w d4, LEAD(date,5) OVER w d5,
    LEAD(adj_close,1) OVER w c1, LEAD(adj_close,2) OVER w c2, LEAD(adj_close,3) OVER w c3,
    LEAD(adj_close,4) OVER w c4, LEAD(adj_close,5) OVER w c5,
    LEAD(adj_low,1) OVER w l1, LEAD(adj_low,2) OVER w l2, LEAD(adj_low,3) OVER w l3,
    LEAD(adj_low,4) OVER w l4, LEAD(adj_low,5) OVER w l5
  FROM stocks_daily
  WHERE date >= '2016-01-01'
  WINDOW w AS (PARTITION BY code ORDER BY date)
)
SELECT p.code, p.date, f.fin_disc_date,
  (p.adj_open/p.pc - 1)*100 AS gap, p.adv20, p.adj_close AS c0,
  p.d1,p.d2,p.d3,p.d4,p.d5, p.c1,p.c2,p.c3,p.c4,p.c5, p.l1,p.l2,p.l3,p.l4,p.l5
FROM px p
JOIN fin f ON f.code = p.code AND f.prev_date = p.pdate
WHERE p.pc > 0 AND p.adv20 >= %s
  AND (p.adj_open/p.pc - 1)*100 >= %s
  AND p.c5 IS NOT NULL
"""
ev = db.read_sql(SQL, [ADV_MIN, GAP_MIN])
for c in ["gap", "adv20", "c0", "c1", "c2", "c3", "c4", "c5", "l1", "l2", "l3", "l4", "l5"]:
    ev[c] = ev[c].astype(float)
ev["date"] = pd.to_datetime(ev["date"])
ev = ev.drop_duplicates(subset=["code", "date"])
print(f"PEADシグナル(決算翌日ギャップ>=+{GAP_MIN}% / ADV>={ADV_MIN/1e8:.0f}億): n={len(ev):,}  "
      f"{ev['date'].min():%Y-%m} 〜 {ev['date'].max():%Y-%m}")

# ---------------------------------------------------------------- 2. ガイダンス修正を結合
rev = db.read_sql("""
  SELECT code, disc_date AS fin_disc_date,
         rev_op_pct, rev_np_pct, div_change, direction
  FROM earnings_forecast_revisions WHERE disc_date >= '2016-01-01'
""", [])
rev["rev_op_pct"] = pd.to_numeric(rev["rev_op_pct"], errors="coerce")
# 同日複数開示は営業利益改定率の絶対値最大を採用
rev = (rev.assign(_a=rev["rev_op_pct"].abs())
          .sort_values("_a", ascending=False)
          .drop_duplicates(subset=["code", "fin_disc_date"]))
ev = ev.merge(rev.drop(columns="_a"), on=["code", "fin_disc_date"], how="left")

n_rev = ev["rev_op_pct"].notna().sum()
print(f"  うち決算と同時にガイダンス修正あり: {n_rev:,} ({n_rev/len(ev)*100:.0f}%) / 修正なし {len(ev)-n_rev:,}")

# ---------------------------------------------------------------- 3. トレード収益(SL込み)
entry = ev["c0"].values
legs_c = ev[["c1", "c2", "c3", "c4", "c5"]].values
legs_l = ev[["l1", "l2", "l3", "l4", "l5"]].values
sl_px = entry * (1 + SL_PCT / 100)

# SL: 安値が初めてSL価格を割った日で -5% 決済（その日以降はフラット）
hit = legs_l <= sl_px[:, None]
first_hit = np.where(hit.any(axis=1), hit.argmax(axis=1), HOLD)   # 0-4 = D+1..D+5, 5 = 未発動

daily = np.empty((len(ev), HOLD))                                  # 各保有日の日次リターン(%)
prev = entry.copy()
for k in range(HOLD):
    px_k = np.where(k < first_hit, legs_c[:, k],
           np.where(k == first_hit, sl_px, np.nan))                # SL日はSL価格、以降NaN(手仕舞い済)
    daily[:, k] = (px_k / prev - 1) * 100
    prev = np.where(np.isnan(px_k), prev, px_k)
ev["gross"] = np.where(first_hit < HOLD, SL_PCT, (legs_c[:, -1] / entry - 1) * 100)
ev["net"] = ev["gross"] - COST_BPS / 100
ev["sl_hit"] = first_hit < HOLD

# ---------------------------------------------------------------- 4. 層別
def bucket(r):
    if pd.isna(r):       return "0_修正なし"
    if r < 0:            return "1_下方修正"
    if r < 10:           return "2_上方 0-10%"
    if r < 30:           return "3_上方 10-30%"
    return "4_上方 30%+"
ev["grp"] = ev["rev_op_pct"].apply(bucket)
ev["yr"] = ev["date"].dt.year
IS_END = 2022
ev["seg"] = np.where(ev["yr"] <= IS_END, "IS", "OOS")

def stat(g):
    x = g["net"].dropna()
    if len(x) < 5: return None
    t = sps.ttest_1samp(x, 0)[0]
    return dict(n=len(x), mean=x.mean(), med=x.median(), win=(x > 0).mean()*100,
                t=t, sl=g["sl_hit"].mean()*100)

print(f"\n=== 層別: 1トレードあたり net リターン(%) 往復{COST_BPS}bps控除 ===")
print(f"{'層':16s} {'n':>5s} {'平均':>7s} {'中央':>7s} {'勝率':>6s} {'t':>6s} {'SL率':>6s} | {'IS平均':>7s} {'OOS平均':>8s}")
rows = []
for g, d in ev.groupby("grp"):
    s = stat(d)
    if not s: continue
    si = stat(d[d.seg == "IS"]); so = stat(d[d.seg == "OOS"])
    print(f"{g:16s} {s['n']:5d} {s['mean']:+7.2f} {s['med']:+7.2f} {s['win']:5.1f}% {s['t']:+6.2f} {s['sl']:5.1f}% | "
          f"{(si['mean'] if si else np.nan):+7.2f} {(so['mean'] if so else np.nan):+8.2f}")
    rows.append(dict(grp=g, **s, is_mean=si['mean'] if si else np.nan,
                     oos_mean=so['mean'] if so else np.nan))
res = pd.DataFrame(rows)

# 修正あり vs なし の差の検定
a = ev.loc[ev["rev_op_pct"].notna(), "net"].dropna()
b = ev.loc[ev["rev_op_pct"].isna(),  "net"].dropna()
tt = sps.ttest_ind(a, b, equal_var=False)
print(f"\n[H1] 修正あり({len(a)}) {a.mean():+.2f}%  vs  修正なし({len(b)}) {b.mean():+.2f}%   "
      f"差 {a.mean()-b.mean():+.2f}pt  t={tt[0]:+.2f} p={tt[1]:.3f}")

# ---------------------------------------------------------------- 5. 日次バスケット系列 → Sharpe
def daily_series(sub):
    """保有中ポジションの等加重日次リターン系列（重複保有を正しく扱う→√252年率化が妥当）"""
    idx = sub.index.to_numpy()
    recs = []
    dcols = ["d1", "d2", "d3", "d4", "d5"]
    for j, i in enumerate(idx):
        row = sub.loc[i]
        for k in range(HOLD):
            dt = row[dcols[k]]
            r = daily[ev.index.get_loc(i), k]
            if pd.isna(dt) or pd.isna(r): continue
            recs.append((pd.Timestamp(dt), r))
    if not recs: return pd.Series(dtype=float)
    s = pd.DataFrame(recs, columns=["date", "r"]).groupby("date")["r"].mean()
    # コストは建て・落ちの2日に案分せず、トレード単位の総コストを保有日数で割って日次控除
    return (s / 100) - (COST_BPS / 100 / 100) / HOLD

print(f"\n=== 日次バスケットSharpe(√252・コスト込) ===")
print(f"{'構成':28s} {'n_trade':>7s} {'日数':>5s} {'年率%':>7s} {'Sharpe':>7s} {'IS':>6s} {'OOS':>6s} {'MDD%':>7s}")
def sh(s):
    if len(s) < 20: return np.nan, np.nan, np.nan
    ann = s.mean() * 252 * 100
    shr = s.mean() / s.std() * np.sqrt(252)
    eq = (1 + s).cumprod()
    mdd = ((eq / eq.cummax()) - 1).min() * 100
    return ann, shr, mdd

variants = {
    "全シグナル(現行earnings_pead)": ev,
    "修正あり のみ":               ev[ev["rev_op_pct"].notna()],
    "上方修正 のみ":               ev[ev["rev_op_pct"] > 0],
    "上方 0-30% のみ":             ev[(ev["rev_op_pct"] > 0) & (ev["rev_op_pct"] < 30)],
    "上方 30%+ のみ":              ev[ev["rev_op_pct"] >= 30],
    "修正なし のみ":               ev[ev["rev_op_pct"].isna()],
    "下方修正 のみ":               ev[ev["rev_op_pct"] < 0],
}
curves = {}
for name, sub in variants.items():
    s = daily_series(sub)
    curves[name] = s
    ann, shr, mdd = sh(s)
    si = s[s.index.year <= IS_END]; so = s[s.index.year > IS_END]
    _, shi, _ = sh(si); _, sho, _ = sh(so)
    print(f"{name:28s} {len(sub):7d} {len(s):5d} {ann:+7.1f} {shr:+7.2f} {shi:+6.2f} {sho:+6.2f} {mdd:+7.1f}")

ev.drop(columns=[c for c in ev.columns if c.startswith(("l", "c")) and c[1:].isdigit()]) \
  .to_csv(HERE / "pead_guidance_events.csv", index=False)
res.to_csv(HERE / "group_stats.csv", index=False)

# ---------------------------------------------------------------- 6. 可視化
import matplotlib.font_manager as fm
fm.fontManager.addfont("/root/.fonts/NotoSansJP.ttf")
plt.rcParams["font.family"] = fm.FontProperties(fname="/root/.fonts/NotoSansJP.ttf").get_name()
plt.rcParams["axes.unicode_minus"] = False

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 6.2), facecolor="white")
o = res.sort_values("grp")
cols = ["#8fa9bf" if g.startswith("0") else "#c0392b" if g.startswith("1") else "#2e7d32" for g in o["grp"]]
ax1.bar(range(len(o)), o["mean"], color=cols)
for i, (m, n) in enumerate(zip(o["mean"], o["n"])):
    ax1.text(i, m + (0.05 if m >= 0 else -0.12), f"{m:+.2f}%\nn={n}", ha="center", fontsize=8.5)
ax1.set_xticks(range(len(o)))
ax1.set_xticklabels([g[2:] for g in o["grp"]], fontsize=9)
ax1.axhline(0, color="#333", lw=0.8); ax1.grid(axis="y", alpha=0.3)
ax1.set_ylabel("1トレード net リターン(%)")
ax1.set_title("決算ギャップ+7%後のドリフトを\n「同時のガイダンス修正」で層別", fontsize=12, fontweight="bold")

for name, c in [("全シグナル(現行earnings_pead)", "#333"), ("上方修正 のみ", "#2e7d32"),
                ("修正なし のみ", "#8fa9bf"), ("下方修正 のみ", "#c0392b")]:
    s = curves[name]
    if len(s) > 20:
        ax2.plot(s.index, (1 + s).cumprod(), lw=1.8, color=c, label=name)
ax2.axvline(pd.Timestamp(f"{IS_END}-12-31"), color="gray", ls="--", lw=1)
ax2.text(pd.Timestamp(f"{IS_END}-12-31"), ax2.get_ylim()[1], " OOS→", fontsize=8, color="gray", va="top")
ax2.grid(alpha=0.3); ax2.legend(fontsize=8.5); ax2.set_ylabel("累積（日次バスケット・コスト込）")
ax2.set_title("フィルタ別 エクイティカーブ", fontsize=12, fontweight="bold")

fig.suptitle("PEAD × 会社予想ガイダンス修正 — 決算ギャップ買いはガイダンスで選別できるか",
             fontsize=13, fontweight="bold")
fig.text(0.99, 0.005, f"データ: JQuants fin_summary × earnings_forecast_revisions × stocks_daily "
                      f"2016-2026 / 往復{COST_BPS}bps控除", ha="right", fontsize=8, color="gray")
fig.tight_layout()
fig.savefig(HERE / "result.png", dpi=100, bbox_inches="tight", facecolor="white")
print("\nsaved result.png")
