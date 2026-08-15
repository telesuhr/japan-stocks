"""
市場中立L/S候補 #12 GapReversal2.5% / #13 6Mモメンタム の再検証。
元検証の2バイアス（生存者バイアス・稼働日無視の年率化）を修正し、
pre_earnings_drift（βエンジン）との相関・合成効果まで測る。
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as sstats
from jstock import db

COST_LS = 0.0008           # L/S往復8bps
COST_1W = 2.0 / 10000.0    # 片道2bps（pre_earnings_drift用）
ADV_MIN = 5e8
ADV_WIN = 60
Q_LS = 0.20
GAP_THR = 0.025
IS_END = pd.Timestamp("2021-06-30")

# ============================================================
# 1. データ（ユニバースを先に絞らない＝生存者バイアス排除）
# ============================================================
print("[1] stocks_daily 読み込み...")
raw = db.read_sql("""
    SELECT code, date, adj_open, adj_close, turnover_value
    FROM stocks_daily
    WHERE date >= '2015-01-01' AND adj_close > 0 AND adj_open > 0
""")
raw["date"] = pd.to_datetime(raw["date"])
print(f"  {raw['code'].nunique():,}銘柄 / {len(raw):,}行")

AO = raw.pivot(index="date", columns="code", values="adj_open").sort_index()
AC = raw.pivot(index="date", columns="code", values="adj_close").sort_index()
TV = raw.pivot(index="date", columns="code", values="turnover_value").sort_index()

# point-in-time ユニバース: 直近60営業日平均売買代金 >= 5億（当日は含めずshift(1)）
ADV = TV.rolling(ADV_WIN, min_periods=40).mean().shift(1)
UNIV = ADV >= ADV_MIN
print(f"  PITユニバース 平均銘柄数: {UNIV.sum(axis=1).mean():.0f}（元検証は全期間固定838銘柄）")

ret_OC = AC / AO - 1.0
gap = AO / AC.shift(1) - 1.0
cal = AC.index


def sh(s, ann=252):
    s = s.dropna()
    if len(s) < 10 or s.std() == 0:
        return np.nan
    return float(s.mean() / s.std() * np.sqrt(ann))


def mdd(s):
    c = (1 + s.dropna()).cumprod()
    return float((c / c.cummax() - 1).min())


def rep(s, ann=252, label=""):
    s = s.dropna()
    t, p = sstats.ttest_1samp(s, 0) if len(s) >= 5 else (np.nan, np.nan)
    return dict(label=label, N=len(s), sharpe=sh(s, ann), ann_pct=s.mean() * ann * 100,
                mdd=mdd(s) * 100, cum=((1 + s).prod() - 1) * 100,
                wr=(s > 0).mean() * 100, t=float(t))


# ============================================================
# 2. #12 GapReversal 2.5%（PITユニバース・稼働日ベース年率化）
# ============================================================
print("\n[2] #12 GapReversal 2.5%...")
sig = (-gap).where((gap.abs() >= GAP_THR) & UNIV & ret_OC.notna())

rows = []
for d in cal:
    s = sig.loc[d].dropna()
    if len(s) < 20:
        continue
    r = ret_OC.loc[d, s.index]
    ok = r.notna()
    s, r = s[ok], r[ok]
    if len(s) < 20:
        continue
    n = max(3, int(len(s) * Q_LS))
    o = s.sort_values().index
    lr = r[o[-n:]].mean()   # sig大 = gap最小(下ギャップ) → ロング
    srt = r[o[:n]].mean()
    rows.append(dict(date=d, ret=lr - srt - COST_LS, spread=lr - srt, n=len(s)))
gapdf = pd.DataFrame(rows).set_index("date").sort_index()
gap_active = gapdf["ret"]
# 全営業日系列（非稼働日=0）＝資金を寝かせている日を織り込む
gap_full = gap_active.reindex(cal).fillna(0.0)
gap_full = gap_full[gap_full.index >= "2016-01-01"]
gap_active = gap_active[gap_active.index >= "2016-01-01"]
print(f"  シグナル日 {len(gap_active)}日 / 年{len(gap_active)/((cal[-1]-pd.Timestamp('2016-01-01')).days/365.25):.0f}日")

# ============================================================
# 3. #13 6Mモメンタム（寄成→引成の実執行形）
# ============================================================
print("\n[3] #13 6Mモメンタム...")
me = pd.Series(cal, index=cal).groupby([cal.year, cal.month]).last().values   # 各月末営業日
ms = pd.Series(cal, index=cal).groupby([cal.year, cal.month]).first().values  # 各月初営業日
me = pd.DatetimeIndex(me); ms = pd.DatetimeIndex(ms)

mom_rows = []
for i in range(len(me) - 1):
    sd = me[i]                      # シグナル日（月末）
    j = np.searchsorted(cal, sd)
    k = j - 120                     # 約6ヶ月前
    if k < 0:
        continue
    entry, exitd = ms[i + 1], me[i + 1]
    if entry <= sd:
        continue
    m6 = AC.loc[sd] / AC.iloc[k] - 1.0
    u = UNIV.loc[sd] & m6.notna() & AO.loc[entry].notna() & AC.loc[exitd].notna()
    m6 = m6[u]
    if len(m6) < 50:
        continue
    fwd = AC.loc[exitd, m6.index] / AO.loc[entry, m6.index] - 1.0
    n = max(3, int(len(m6) * Q_LS))
    o = m6.sort_values().index
    lr = fwd[o[-n:]].mean(); srt = fwd[o[:n]].mean()
    mom_rows.append(dict(date=exitd, entry=entry, ret=lr - srt - COST_LS,
                         spread=lr - srt, long=lr, short=srt, n=len(m6)))
momdf = pd.DataFrame(mom_rows).set_index("date").sort_index()
mom = momdf["ret"]
mom = mom[mom.index >= "2016-01-01"]
print(f"  月次サイクル {len(mom)}件 / 平均ユニバース {momdf['n'].mean():.0f}銘柄")

# ============================================================
# 4. pre_earnings_drift 日次系列（βエンジン・姉妹分析と同一定義）
# ============================================================
print("\n[4] pre_earnings_drift 日次系列...")
ann_df = db.read_sql("""
    WITH fs AS (
        SELECT code, cur_per_type, cur_per_en, MIN(disc_date) AS disc_date
        FROM fin_summary
        WHERE doc_type LIKE '%%FinancialStatements%%'
          AND cur_per_type IN ('FY','2Q','3Q')
        GROUP BY code, cur_per_type, cur_per_en
    )
    SELECT fs.code, fs.cur_per_type, fs.disc_date
    FROM fs JOIN symbol_master sm ON sm.code5 = fs.code
    WHERE sm.market_nm = 'プライム'
      AND (sm.sector33_nm IS NULL OR sm.sector33_nm NOT IN ('医薬品','陸運業'))
""")
ann_df["disc_date"] = pd.to_datetime(ann_df["disc_date"])
LEAD = {"FY": 5, "2Q": 3, "3Q": 3}
pos = {d: i for i, d in enumerate(cal)}
a = ann_df[ann_df["disc_date"].isin(pos)].copy()
a["e"] = a["disc_date"].map(pos).astype(int)
a["si"] = a["e"] - a["cur_per_type"].map(LEAD).astype(int)
a["eni"] = a["si"] + 1; a["xi"] = a["e"] - 1
a = a[(a["si"] >= 0) & (a["eni"] < a["xi"])].copy()

TVv = TV.values; AOv = AO.values; ACv = AC.values
colpos = {c: i for i, c in enumerate(AC.columns)}
a = a[a["code"].isin(colpos)].copy()
a["ci"] = a["code"].map(colpos).astype(int)
a = a[(TVv[a["si"].values, a["ci"].values] >= 5e8)]
a = a[np.isfinite(AOv[a["eni"].values, a["ci"].values])]
print(f"  有効トレード {len(a):,}件")

prev_close = AC.shift(1)
PCv = prev_close.values
ped = {}
for si_, ci, ei, xi in zip(a["si"].values, a["ci"].values, a["eni"].values, a["xi"].values):
    for k in range(ei, xi + 1):
        if k == ei:
            base = AOv[k, ci]
            r = ACv[k, ci] / base - 1.0 - COST_1W if base > 0 else np.nan
        else:
            base = PCv[k, ci]
            r = ACv[k, ci] / base - 1.0 if base > 0 else np.nan
        if not np.isfinite(r):
            continue
        if k == xi:
            r -= COST_1W
        ped.setdefault(k, []).append(r)
ped_s = pd.Series({cal[k]: np.mean(v) for k, v in ped.items()}).sort_index()
ped_s = ped_s[ped_s.index >= "2016-01-01"]

# ============================================================
# 5. 集計
# ============================================================
def seg(s, name, ann=252):
    out = []
    for lbl, sub in [("IS", s[s.index <= IS_END]), ("OOS", s[s.index > IS_END]), ("ALL", s)]:
        r = rep(sub, ann, f"{name}/{lbl}")
        out.append(r)
    return out

print("\n" + "=" * 86)
print("主要指標（コスト後）  ※GapRevは全営業日系列（非稼働日=0）で年率化＝稼働日を織り込む")
print("=" * 86)
print(f"{'戦略/期間':<34}{'N':>6}{'Sharpe':>8}{'年率%':>8}{'MDD%':>8}{'累積%':>9}{'勝率%':>7}{'t':>7}")
allrows = []
for name, s, ann in [("#12 GapRev(全営業日)", gap_full, 252),
                     ("#12 GapRev(稼働日のみ*誇張)", gap_active, 252),
                     ("#13 6Mモメンタム(月次)", mom, 12),
                     ("pre_earnings_drift(β)", ped_s, 252)]:
    for r in seg(s, name, ann):
        allrows.append(r)
        print(f"{r['label']:<34}{r['N']:>6}{r['sharpe']:>8.2f}{r['ann_pct']:>8.1f}"
              f"{r['mdd']:>8.1f}{r['cum']:>9.1f}{r['wr']:>7.0f}{r['t']:>7.2f}")
pd.DataFrame(allrows).to_csv("metrics.csv", index=False)

# ============================================================
# 6. 相関（H3）と合成バスケット
# ============================================================
print("\n" + "=" * 86)
print("相関 & 合成（H3）")
print("=" * 86)
D = pd.DataFrame({"gap": gap_full, "ped": ped_s}).dropna()
print(f"日次相関 #12 GapRev × pre_earnings_drift : {D['gap'].corr(D['ped']):+.3f}  (N={len(D)})")

# 月次に落として3者相関
def to_m(s):
    return (1 + s).resample("ME").prod() - 1
M = pd.DataFrame({"gap": to_m(gap_full), "mom": mom.resample("ME").sum(),
                  "ped": to_m(ped_s)}).dropna()
print("\n月次相関行列:")
print(M.corr().round(3).to_string())

print("\n月次 単独 vs 合成（等ウェイト）  ※IS=〜2021-06 / OOS=2021-07〜")
combos = {"ped単独": ["ped"], "gap単独": ["gap"], "mom単独": ["mom"],
          "ped+gap": ["ped", "gap"], "ped+mom": ["ped", "mom"],
          "gap+mom": ["gap", "mom"], "3本合成": ["ped", "gap", "mom"]}
crows = []
for lbl, cols in combos.items():
    s = M[cols].mean(axis=1)
    for sg, sub in [("IS", s[s.index <= IS_END]), ("OOS", s[s.index > IS_END]), ("ALL", s)]:
        r = rep(sub, 12, f"{lbl}/{sg}")
        crows.append(r)
    o = {sg: rep(sub, 12) for sg, sub in
         [("IS", s[s.index <= IS_END]), ("OOS", s[s.index > IS_END]), ("ALL", s)]}
    print(f"  {lbl:<10} IS Sh{o['IS']['sharpe']:+.2f}(MDD{o['IS']['mdd']:6.1f}%)  "
          f"OOS Sh{o['OOS']['sharpe']:+.2f}(MDD{o['OOS']['mdd']:6.1f}%)  "
          f"ALL Sh{o['ALL']['sharpe']:+.2f} 年率{o['ALL']['ann_pct']:+6.1f}%")
pd.DataFrame(crows).to_csv("combo_metrics.csv", index=False)

# 年次
print("\n年次リターン%（月次複利）")
yr = pd.DataFrame({k: (1 + M[c].mean(axis=1) if False else None) for k, c in []}) if False else None
yrows = []
for lbl, cols in [("gap", ["gap"]), ("mom", ["mom"]), ("ped", ["ped"]), ("3本合成", ["ped", "gap", "mom"])]:
    s = M[cols].mean(axis=1)
    y = ((1 + s).groupby(s.index.year).prod() - 1) * 100
    yrows.append(y.rename(lbl))
Y = pd.concat(yrows, axis=1)
print(Y.round(1).to_string())
Y.to_csv("per_year.csv")

# ============================================================
# 7. 可視化
# ============================================================
try:
    import matplotlib.font_manager as fm
    fm.fontManager.addfont("/root/.fonts/NotoSansJP.ttf")
    plt.rcParams["font.family"] = fm.FontProperties(fname="/root/.fonts/NotoSansJP.ttf").get_name()
    plt.rcParams["axes.unicode_minus"] = False
except Exception:
    pass
fig, axes = plt.subplots(1, 2, figsize=(12, 6.75), facecolor="white")
for c, lbl, col in [("ped", "pre_earnings_drift(βエンジン)", "#8250df"),
                    ("gap", "#12 GapRev2.5%", "#0969da"),
                    ("mom", "#13 6Mモメンタム", "#2da44e")]:
    axes[0].plot(M.index, (1 + M[c]).cumprod(), label=lbl, color=col, lw=1.4)
mix = M[["ped", "gap", "mom"]].mean(axis=1)
axes[0].plot(M.index, (1 + mix).cumprod(), label="3本等ウェイト合成", color="#cf222e", lw=2.2)
axes[0].axvline(IS_END, color="gray", ls="--", lw=1)
axes[0].set_yscale("log"); axes[0].grid(alpha=0.3); axes[0].legend(fontsize=8)
axes[0].set_title("月次累積（対数軸）点線=IS|OOS境界")
cm = M.corr()
im = axes[1].imshow(cm, cmap="RdBu_r", vmin=-1, vmax=1)
axes[1].set_xticks(range(3)); axes[1].set_yticks(range(3))
axes[1].set_xticklabels(["GapRev", "6Mモメ", "決算前"], fontsize=9)
axes[1].set_yticklabels(["GapRev", "6Mモメ", "決算前"], fontsize=9)
for i in range(3):
    for j in range(3):
        axes[1].text(j, i, f"{cm.iloc[i,j]:+.2f}", ha="center", va="center", fontsize=11)
axes[1].set_title("月次リターン相関（分散寄与の確認）")
fig.suptitle("市場中立L/S候補 vs βエンジン — 相関と合成効果", fontsize=13)
fig.text(0.99, 0.01, "データ: stocks_daily 2016-2026 / PITユニバース(ADV60≥5億) / L/S往復8bps",
         ha="right", va="bottom", fontsize=8, color="gray")
fig.savefig("result.png", dpi=100, bbox_inches="tight", facecolor="white")
print("\nsaved result.png / metrics.csv / combo_metrics.csv / per_year.csv")

# 系列を保存（H4スクリプト・後続分析用）
pd.DataFrame({"gap_full": gap_full}).to_csv("series_gap_daily.csv")
gapdf.to_csv("gap_signal_days.csv")
mom.rename("mom").to_csv("series_mom_monthly.csv")
ped_s.rename("ped").to_csv("series_ped_daily.csv")
