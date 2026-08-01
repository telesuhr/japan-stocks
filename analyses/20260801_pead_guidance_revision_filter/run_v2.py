"""
v2: ガイダンス修正を「決算短信の会社予想(FOP/NxFOP)の変化」で再構成する。

v1 の弱点: `earnings_forecast_revisions` は "予想修正の単独開示(EarnForecastRevision)" のみで、
決算ギャップ事象の **8% しかカバーしない** → フィルタとして母集団が足りず、✗の結論が弱い。

v2: `fin_summary.payload` の
  FOP    = 当期(CurFYEn)の会社予想営業利益
  NxFOP  = 次期(NxtFYEn)の会社予想営業利益
から「ある決算期(fy_end)に対する会社予想の時系列」を作り、各開示での前回比変化率を
ガイダンス改定率(guid_pct)として算出する。これで決算ギャップ事象のほぼ全件に値が付く。

仮説はv1と同一(H1〜H4)。先読みなし: 改定はDay N-1引け後の開示 → Day N 15:30エントリー時点で既知。
"""
import sys; sys.stdout.reconfigure(line_buffering=True)
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats as sps
import matplotlib.pyplot as plt
from jstock import db

HERE = Path(__file__).resolve().parent
COST_BPS, SL_PCT, HOLD, GAP_MIN, ADV_MIN = 4.0, -5.0, 5, 7.0, 5e8
IS_END = 2022

# ------------------------------------------------- 1. 会社予想(営業利益)の時系列を再構成
g = db.read_sql("""
SELECT code, disc_date, disc_time,
       (payload->>'CurFYEn') AS cur_fy, (payload->>'NxtFYEn') AS nxt_fy,
       NULLIF(payload->>'FOP','')   AS fop,
       NULLIF(payload->>'NxFOP','') AS nxfop
FROM fin_summary WHERE disc_date >= '2015-04-01'
""", [])
print(f"fin_summary 開示: {len(g):,}")

long = []
for fy_col, v_col in [("cur_fy", "fop"), ("nxt_fy", "nxfop")]:
    d = g[["code", "disc_date", "disc_time", fy_col, v_col]].copy()
    d.columns = ["code", "disc_date", "disc_time", "fy_end", "guid"]
    long.append(d)
gl = pd.concat(long, ignore_index=True)
gl["guid"] = pd.to_numeric(gl["guid"], errors="coerce")
gl["fy_end"] = gl["fy_end"].str[:10]
gl = gl.dropna(subset=["guid", "fy_end"])
gl = gl[gl["fy_end"] != "None"]
gl["disc_date"] = pd.to_datetime(gl["disc_date"])

# 同一(code,fy_end,disc_date)は最後の開示を採用 → fy_end系列で前回比
gl = (gl.sort_values(["code", "fy_end", "disc_date", "disc_time"])
        .drop_duplicates(subset=["code", "fy_end", "disc_date"], keep="last"))
gl["prev_guid"] = gl.groupby(["code", "fy_end"])["guid"].shift(1)
gl["guid_pct"] = np.where(gl["prev_guid"] > 0, (gl["guid"] / gl["prev_guid"] - 1) * 100, np.nan)
rev = (gl.dropna(subset=["guid_pct"])
         .rename(columns={"disc_date": "fin_disc_date"})[["code", "fin_disc_date", "fy_end", "guid_pct"]])
# 1開示で当期・次期の両方が改定される場合は絶対値の大きい方
rev = (rev.assign(_a=rev["guid_pct"].abs()).sort_values("_a", ascending=False)
          .drop_duplicates(subset=["code", "fin_disc_date"]).drop(columns="_a"))
print(f"会社予想の改定観測: {len(rev):,}  ({rev['fin_disc_date'].min():%Y-%m}〜{rev['fin_disc_date'].max():%Y-%m})")

# ------------------------------------------------- 2. PEADイベント (v1と同一)
SQL = """
WITH fin AS (
  SELECT DISTINCT code,
    CASE WHEN disc_time >= '15:00' THEN disc_date ELSE disc_date - 1 END AS prev_date,
    disc_date AS fin_disc_date
  FROM fin_summary WHERE disc_date >= '2016-01-01'
),
px AS (
  SELECT code, date, adj_open, adj_close, adj_low, turnover_value,
    LAG(adj_close) OVER w AS pc, LAG(date) OVER w AS pdate,
    AVG(turnover_value) OVER (PARTITION BY code ORDER BY date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING) adv20,
    LEAD(date,1) OVER w d1, LEAD(date,2) OVER w d2, LEAD(date,3) OVER w d3,
    LEAD(date,4) OVER w d4, LEAD(date,5) OVER w d5,
    LEAD(adj_close,1) OVER w c1, LEAD(adj_close,2) OVER w c2, LEAD(adj_close,3) OVER w c3,
    LEAD(adj_close,4) OVER w c4, LEAD(adj_close,5) OVER w c5,
    LEAD(adj_low,1) OVER w l1, LEAD(adj_low,2) OVER w l2, LEAD(adj_low,3) OVER w l3,
    LEAD(adj_low,4) OVER w l4, LEAD(adj_low,5) OVER w l5
  FROM stocks_daily WHERE date >= '2016-01-01'
  WINDOW w AS (PARTITION BY code ORDER BY date)
)
SELECT p.code, p.date, f.fin_disc_date, (p.adj_open/p.pc-1)*100 AS gap, p.adv20,
  p.adj_close AS c0, p.d1,p.d2,p.d3,p.d4,p.d5, p.c1,p.c2,p.c3,p.c4,p.c5, p.l1,p.l2,p.l3,p.l4,p.l5
FROM px p JOIN fin f ON f.code=p.code AND f.prev_date=p.pdate
WHERE p.pc>0 AND p.adv20>=%s AND (p.adj_open/p.pc-1)*100>=%s AND p.c5 IS NOT NULL
"""
ev = db.read_sql(SQL, [ADV_MIN, GAP_MIN])
for c in ["gap", "adv20", "c0", "c1", "c2", "c3", "c4", "c5", "l1", "l2", "l3", "l4", "l5"]:
    ev[c] = ev[c].astype(float)
ev["date"] = pd.to_datetime(ev["date"])
ev["fin_disc_date"] = pd.to_datetime(ev["fin_disc_date"])
ev = ev.drop_duplicates(subset=["code", "date"]).reset_index(drop=True)
ev = ev.merge(rev, on=["code", "fin_disc_date"], how="left")
cov = ev["guid_pct"].notna().mean() * 100
print(f"PEADシグナル n={len(ev):,} / ガイダンス改定率のカバレッジ {cov:.0f}%  (v1は8%)")

# ------------------------------------------------- 3. トレード収益
entry, legs_c = ev["c0"].values, ev[["c1", "c2", "c3", "c4", "c5"]].values
legs_l = ev[["l1", "l2", "l3", "l4", "l5"]].values
sl_px = entry * (1 + SL_PCT / 100)
hit = legs_l <= sl_px[:, None]
first_hit = np.where(hit.any(axis=1), hit.argmax(axis=1), HOLD)

daily = np.empty((len(ev), HOLD)); prev = entry.copy()
for k in range(HOLD):
    px_k = np.where(k < first_hit, legs_c[:, k], np.where(k == first_hit, sl_px, np.nan))
    daily[:, k] = (px_k / prev - 1) * 100
    prev = np.where(np.isnan(px_k), prev, px_k)
ev["gross"] = np.where(first_hit < HOLD, SL_PCT, (legs_c[:, -1] / entry - 1) * 100)
ev["net"] = ev["gross"] - COST_BPS / 100
ev["sl_hit"] = first_hit < HOLD
ev["yr"] = ev["date"].dt.year
ev["seg"] = np.where(ev["yr"] <= IS_END, "IS", "OOS")

# ------------------------------------------------- 4. 層別（5分位＋据置＋不明）
def bucket(r):
    if pd.isna(r):        return "5_不明"
    if abs(r) < 0.5:      return "0_据置(±0.5%)"
    if r <= -10:          return "1_下方 -10%超"
    if r < 0:             return "2_下方 -10%以内"
    if r < 20:            return "3_上方 +20%以内"
    return "4_上方 +20%超"
ev["grp"] = ev["guid_pct"].apply(bucket)

def stat(g):
    x = g["net"].dropna()
    if len(x) < 5: return None
    return dict(n=len(x), mean=x.mean(), med=x.median(), win=(x > 0).mean()*100,
                t=sps.ttest_1samp(x, 0)[0], sl=g["sl_hit"].mean()*100)

print(f"\n=== 層別: 1トレード net(%) 往復{COST_BPS}bps控除 ===")
print(f"{'層':18s} {'n':>5s} {'平均':>7s} {'中央':>7s} {'勝率':>6s} {'t':>6s} {'SL率':>6s} | {'IS':>7s} {'OOS':>7s}")
rows = []
for gname, d in sorted(ev.groupby("grp")):
    s = stat(d)
    if not s: continue
    si, so = stat(d[d.seg == "IS"]), stat(d[d.seg == "OOS"])
    print(f"{gname:18s} {s['n']:5d} {s['mean']:+7.2f} {s['med']:+7.2f} {s['win']:5.1f}% {s['t']:+6.2f} {s['sl']:5.1f}% | "
          f"{(si['mean'] if si else np.nan):+7.2f} {(so['mean'] if so else np.nan):+7.2f}")
    rows.append(dict(grp=gname, **s, is_mean=si['mean'] if si else np.nan, oos_mean=so['mean'] if so else np.nan))
res = pd.DataFrame(rows)

known = ev[ev["guid_pct"].notna()]
up, dn = known[known["guid_pct"] > 0.5], known[known["guid_pct"] < -0.5]
tt = sps.ttest_ind(up["net"], dn["net"], equal_var=False)
print(f"\n[H1] 上方({len(up)}) {up['net'].mean():+.2f}%  vs  下方({len(dn)}) {dn['net'].mean():+.2f}%  "
      f"差 {up['net'].mean()-dn['net'].mean():+.2f}pt  t={tt[0]:+.2f} p={tt[1]:.3f}")
# 連続量での相関（層別の恣意性を排除）
k2 = known.dropna(subset=["net"])
r_s = sps.spearmanr(k2["guid_pct"], k2["net"])
print(f"[H2] 改定率 vs net の順位相関 rho={r_s[0]:+.3f} p={r_s[1]:.3f}  (n={len(k2)})")

# ------------------------------------------------- 5. 日次バスケットSharpe
dcols = ["d1", "d2", "d3", "d4", "d5"]
def daily_series(sub):
    recs = []
    for i in sub.index:
        for k in range(HOLD):
            dt, r = ev.at[i, dcols[k]], daily[i, k]
            if pd.isna(dt) or pd.isna(r): continue
            recs.append((pd.Timestamp(dt), r))
    if not recs: return pd.Series(dtype=float)
    s = pd.DataFrame(recs, columns=["date", "r"]).groupby("date")["r"].mean()
    return (s / 100) - (COST_BPS / 100 / 100) / HOLD

def sh(s):
    if len(s) < 20: return (np.nan,) * 3
    eq = (1 + s).cumprod()
    return s.mean()*252*100, s.mean()/s.std()*np.sqrt(252), ((eq/eq.cummax())-1).min()*100

print(f"\n=== 日次バスケットSharpe(√252・コスト込) ===")
print(f"{'構成':26s} {'n_tr':>5s} {'日数':>5s} {'年率%':>7s} {'Sharpe':>7s} {'IS':>6s} {'OOS':>6s} {'MDD%':>7s}")
variants = {
    "全シグナル(現行)":      ev,
    "上方改定 のみ":         ev[ev["guid_pct"] > 0.5],
    "上方 +20%超 のみ":      ev[ev["guid_pct"] > 20],
    "据置(±0.5%) のみ":      ev[ev["guid_pct"].abs() <= 0.5],
    "下方改定 のみ":         ev[ev["guid_pct"] < -0.5],
    "改定情報なし のみ":     ev[ev["guid_pct"].isna()],
    "下方を除外":            ev[~(ev["guid_pct"] < -0.5)],
    "★|改定|>=20% (事後)":   ev[ev["guid_pct"].abs() >= 20],
    "★|改定|>=20% or 情報なし": ev[(ev["guid_pct"].abs() >= 20) | ev["guid_pct"].isna()],
}
curves = {}
for name, sub in variants.items():
    s = daily_series(sub); curves[name] = s
    ann, shr, mdd = sh(s)
    _, shi, _ = sh(s[s.index.year <= IS_END]); _, sho, _ = sh(s[s.index.year > IS_END])
    print(f"{name:26s} {len(sub):5d} {len(s):5d} {ann:+7.1f} {shr:+7.2f} {shi:+6.2f} {sho:+6.2f} {mdd:+7.1f}")

# ------------------------------------------------- 5b. 事後(post-hoc)チェック
# H1/H2が「方向」で全滅した一方、層別平均が据置<小幅<大幅と非単調に見えた。
# → 方向ではなく「改定の大きさ(=情報量)」が効くのでは、という**事前登録していない**仮説。
# データを見た後の仮説なので、IS/OOS両方で独立に再現するかを唯一の合格条件とする。
print(f"\n=== [事後] 方向でなく『改定の大きさ』か（※事前登録していない探索的検証）===")
k2 = k2.assign(ab=k2["guid_pct"].abs())
rs = sps.spearmanr(k2["ab"], k2["net"])
print(f"  |改定率| vs net 順位相関 rho={rs[0]:+.3f} p={rs[1]:.3f} (n={len(k2)})")
print(f"  {'|改定率|':14s} {'n':>5s} {'平均':>7s} {'t':>6s} | {'IS':>7s} {'nIS':>5s} {'OOS':>7s} {'nOOS':>5s}")
for lo, hi, lab in [(0, .5, '据置 <0.5%'), (.5, 5, '0.5-5%'), (5, 20, '5-20%'), (20, 50, '20-50%'), (50, 1e9, '50%+')]:
    gg = k2[(k2.ab >= lo) & (k2.ab < hi)]
    if len(gg) < 5: continue
    gi, go = gg[gg.seg == "IS"], gg[gg.seg == "OOS"]
    print(f"  {lab:14s} {len(gg):5d} {gg.net.mean():+7.2f} {sps.ttest_1samp(gg.net,0)[0]:+6.2f} | "
          f"{gi.net.mean():+7.2f} {len(gi):5d} {go.net.mean():+7.2f} {len(go):5d}")
big, sml = k2[k2.ab >= 20], k2[k2.ab < 20]
tt2 = sps.ttest_ind(big.net, sml.net, equal_var=False)
print(f"  大改定|r|>=20%({len(big)}) {big.net.mean():+.2f}% vs 小改定({len(sml)}) {sml.net.mean():+.2f}%  "
      f"差 {big.net.mean()-sml.net.mean():+.2f}pt t={tt2[0]:+.2f} p={tt2[1]:.4f}")
for s in ["IS", "OOS"]:
    b, m = big[big.seg == s], sml[sml.seg == s]
    print(f"    {s:3s}: 大 {b.net.mean():+.2f}%(n={len(b)}) vs 小 {m.net.mean():+.2f}%(n={len(m)})  "
          f"差 {b.net.mean()-m.net.mean():+.2f}pt t={sps.ttest_ind(b.net,m.net,equal_var=False)[0]:+.2f}")

ev.drop(columns=[c for c in ev.columns if len(c) == 2 and c[0] in "cl" and c[1].isdigit()]) \
  .to_csv(HERE / "pead_guidance_events_v2.csv", index=False)
res.to_csv(HERE / "group_stats_v2.csv", index=False)

# ------------------------------------------------- 6. 可視化
import matplotlib.font_manager as fm
fm.fontManager.addfont("/root/.fonts/NotoSansJP.ttf")
plt.rcParams["font.family"] = fm.FontProperties(fname="/root/.fonts/NotoSansJP.ttf").get_name()
plt.rcParams["axes.unicode_minus"] = False
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16.5, 5.8), facecolor="white")

# --- (1) 方向で層別 → 効かない（事前仮説H1-H3の棄却） ---
order = ["1_下方 -10%超", "2_下方 -10%以内", "0_据置(±0.5%)", "3_上方 +20%以内", "4_上方 +20%超"]
o = res.set_index("grp").reindex([g for g in order if g in set(res.grp)])
ax1.bar(range(len(o)), o["mean"], color="#8fa9bf", edgecolor="#5b7183")
for i, (m, n) in enumerate(zip(o["mean"], o["n"])):
    ax1.text(i, m + 0.08, f"{m:+.2f}%\nn={int(n)}", ha="center", fontsize=8.5)
ax1.set_xticks(range(len(o))); ax1.set_xticklabels([g[2:] for g in o.index], fontsize=8, rotation=15)
ax1.axhline(0, color="#333", lw=0.8); ax1.grid(axis="y", alpha=0.3)
ax1.set_ylabel("1トレード net リターン(%)")
ax1.set_title(f"× 改定の「方向」では説明できない\n順位相関 rho={r_s[0]:+.3f} (p={r_s[1]:.2f})",
              fontsize=11.5, fontweight="bold", color="#c0392b")

# --- (2) 大きさで層別 → 効く（事後発見・IS/OOS並記） ---
labs, mi, mo, ns = [], [], [], []
for lo, hi, lab in [(0, .5, '据置\n<0.5%'), (.5, 5, '0.5-5%'), (5, 20, '5-20%'), (20, 50, '20-50%'), (50, 1e9, '50%+')]:
    gg = k2[(k2.ab >= lo) & (k2.ab < hi)]
    labs.append(lab); ns.append(len(gg))
    mi.append(gg[gg.seg == "IS"].net.mean()); mo.append(gg[gg.seg == "OOS"].net.mean())
x = np.arange(len(labs)); w = 0.38
ax2.bar(x - w/2, mi, w, color="#5b7183", label="IS (2016-22)")
ax2.bar(x + w/2, mo, w, color="#2e7d32", label="OOS (2023-26)")
for i, n in enumerate(ns):
    ax2.text(i, min(mi[i], mo[i], 0) - 0.55, f"n={n}", ha="center", fontsize=8, color="gray")
ax2.set_xticks(x); ax2.set_xticklabels(labs, fontsize=8.5)
ax2.axhline(0, color="#333", lw=0.8); ax2.grid(axis="y", alpha=0.3); ax2.legend(fontsize=8.5)
ax2.set_xlabel("会社予想 営業利益の改定率 |絶対値|"); ax2.set_ylabel("1トレード net リターン(%)")
ax2.set_title("○ 効くのは「方向」でなく「大きさ」\n|改定|≥20%: +3.87% vs <20%: +0.99% (t=3.6)",
              fontsize=11.5, fontweight="bold", color="#2e7d32")

# --- (3) エクイティカーブ 現行 vs 提案フィルタ ---
for name, c, lw in [("全シグナル(現行)", "#8fa9bf", 1.8), ("★|改定|>=20% or 情報なし", "#c0392b", 2.2)]:
    s = curves[name]
    ann, shr, mdd = sh(s)
    lbl = ("現行 earnings_pead" if "現行" in name else "小幅改定を除外")
    ax3.plot(s.index, (1+s).cumprod(), lw=lw, color=c, label=f"{lbl}  Sh{shr:+.2f} MDD{mdd:.0f}%")
ax3.set_yscale("log"); ax3.axvline(pd.Timestamp(f"{IS_END}-12-31"), color="gray", ls="--", lw=1)
ax3.text(pd.Timestamp(f"{IS_END}-12-31"), ax3.get_ylim()[1]*0.6, " OOS→", fontsize=8, color="gray")
ax3.grid(alpha=0.3); ax3.legend(fontsize=9, loc="upper left")
ax3.set_ylabel("累積(対数・日次バスケット・コスト込)")
ax3.set_title("提案フィルタ: 小幅改定(|r|<20%)を除外\n※事後仮説・要 再検証", fontsize=11.5, fontweight="bold")

fig.suptitle("PEAD × 会社予想ガイダンス改定 — 決算ギャップ買いはガイダンスで選別できるか",
             fontsize=13, fontweight="bold")
fig.text(0.99, 0.005, f"データ: JQuants fin_summary(FOP/NxFOP) × stocks_daily 2016-2026 / 往復{COST_BPS}bps控除",
         ha="right", fontsize=8, color="gray")
fig.tight_layout()
fig.savefig(HERE / "result_v2.png", dpi=100, bbox_inches="tight", facecolor="white")
print("\nsaved result_v2.png")
