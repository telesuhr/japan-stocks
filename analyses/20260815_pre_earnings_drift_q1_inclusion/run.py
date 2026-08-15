"""
pre_earnings_drift への 1Q 追加の複数年IS/OOS検証。
決算発表日は fin_summary(disc_date) から10年分を再構成。現行 pre_earnings_drift の
エントリー規則(signal=発表-lead, entry=signal+1寄成, exit=発表-1引成)に忠実。
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from jstock import db

COST_1W = 2.0 / 10000.0          # 片道2bps
LEAD = {"FY": 5, "1Q": 3, "2Q": 3, "3Q": 3}
EXCLUDE_SEC = ("医薬品", "陸運業")
TURN_MIN = 5e8
IS_END_YEAR = 2021               # IS=2016-2021 / OOS=2022-2026

# ---------- データ取得 ----------
print("決算発表(fin_summary)取得...")
ann = db.read_sql("""
    WITH fs AS (
        SELECT code, cur_per_type, cur_per_en,
               MIN(disc_date) AS disc_date
        FROM fin_summary
        WHERE doc_type LIKE '%%FinancialStatements%%'
          AND cur_per_type IN ('FY','1Q','2Q','3Q')
        GROUP BY code, cur_per_type, cur_per_en
    )
    SELECT fs.code, fs.cur_per_type, fs.disc_date,
           sm.sector33_nm, sm.market_nm
    FROM fs
    JOIN symbol_master sm ON sm.code5 = fs.code
    WHERE sm.market_nm = 'プライム'
      AND (sm.sector33_nm IS NULL OR sm.sector33_nm NOT IN %(exc)s)
""", {"exc": EXCLUDE_SEC})
ann["disc_date"] = pd.to_datetime(ann["disc_date"])
print(f"  発表イベント {len(ann):,}件  (期間 {ann.disc_date.min().date()}〜{ann.disc_date.max().date()})")

print("株価(stocks_daily)取得...")
daily = db.read_sql("""
    SELECT code, date, adj_open, adj_close, turnover_value
    FROM stocks_daily WHERE adj_close > 0
""")
daily["date"] = pd.to_datetime(daily["date"])
daily = daily.sort_values(["code", "date"]).reset_index(drop=True)
print(f"  日足 {len(daily):,}行")

# ---------- 営業日カレンダー & トレード生成 ----------
cal = np.array(sorted(daily["date"].unique()))
cal_pos = {d: i for i, d in enumerate(cal)}

# 各銘柄の日付→行参照（open/close/turnover 引き当て用）
key_open = daily.set_index(["code", "date"])["adj_open"]
key_close = daily.set_index(["code", "date"])["adj_close"]
key_turn = daily.set_index(["code", "date"])["turnover_value"]

ann = ann[ann["disc_date"].isin(cal_pos)].copy()
ann["e_idx"] = ann["disc_date"].map(cal_pos).astype(int)
ann["lead"] = ann["cur_per_type"].map(LEAD).astype(int)
ann["sig_idx"] = ann["e_idx"] - ann["lead"]
ann["ent_idx"] = ann["sig_idx"] + 1
ann["ex_idx"] = ann["e_idx"] - 1
ann = ann[(ann["sig_idx"] >= 0) & (ann["ent_idx"] < ann["ex_idx"])].copy()
ann["sig_date"] = cal[ann["sig_idx"].values]
ann["ent_date"] = cal[ann["ent_idx"].values]
ann["ex_date"] = cal[ann["ex_idx"].values]

def lookup(series, codes, dates):
    idx = pd.MultiIndex.from_arrays([codes, dates])
    return series.reindex(idx).values

ann["turn"] = lookup(key_turn, ann["code"].values, ann["sig_date"].values)
ann["p_in"] = lookup(key_open, ann["code"].values, ann["ent_date"].values)
ann["p_out"] = lookup(key_close, ann["code"].values, ann["ex_date"].values)
tr = ann.dropna(subset=["turn", "p_in", "p_out"])
tr = tr[(tr["turn"] >= TURN_MIN) & (tr["p_in"] > 0)].copy()
tr["gross"] = tr["p_out"] / tr["p_in"] - 1.0
tr["net"] = tr["gross"] - 2 * COST_1W
tr["net10"] = tr["gross"] - 2 * (5.0 / 10000.0)   # 往復10bps感度
tr["year"] = tr["disc_date"].dt.year
tr["grp"] = np.where(tr["cur_per_type"] == "1Q", "1Q", "base")
tr["seg"] = np.where(tr["year"] <= IS_END_YEAR, "IS", "OOS")
print(f"  有効トレード {len(tr):,}件  (base={sum(tr.grp=='base'):,} / 1Q={sum(tr.grp=='1Q'):,})")

# ---------- per-trade 統計 ----------
def pstat(s):
    s = s.dropna()
    if len(s) == 0:
        return dict(N=0, mean=np.nan, med=np.nan, win=np.nan, t=np.nan)
    return dict(N=len(s), mean=s.mean()*100, med=s.median()*100,
                win=(s > 0).mean()*100, t=s.mean()/s.std()*np.sqrt(len(s)))

print("\n" + "="*78)
print("per-trade（コスト後4bps）  variant × IS/OOS")
print("="*78)
print(f"{'variant':<10}{'seg':<5}{'N':>7}{'平均%':>9}{'中央%':>9}{'勝率%':>8}{'t':>7}")
rows_pt = []
for grp, label in [("base", "base(FY/2Q/3Q)"), ("1Q", "1Q単独"), ("all", "all(base+1Q)")]:
    for seg in ["IS", "OOS", "ALL"]:
        sub = tr if grp == "all" else tr[tr.grp == grp]
        if seg != "ALL":
            sub = sub[sub.seg == seg]
        st = pstat(sub["net"])
        rows_pt.append(dict(variant=label, seg=seg, **st))
        print(f"{label:<10}{seg:<5}{st['N']:>7}{st['mean']:>9.3f}{st['med']:>9.3f}{st['win']:>8.1f}{st['t']:>7.2f}")

# 10bps感度（OOSのみ）
print("\n[往復10bps感度] OOS 平均net%:")
for grp, label in [("base","base"),("1Q","1Q"),("all","all")]:
    sub = tr if grp=="all" else tr[tr.grp==grp]
    sub = sub[sub.seg=="OOS"]
    print(f"  {label:<5} {sub['net10'].mean()*100:+.3f}%  (t={sub['net10'].mean()/sub['net10'].std()*np.sqrt(len(sub)):.2f})")

# ---------- 日次EWポートフォリオ Sharpe（honest）----------
# 保有パス再構成: entry日=寄→引(建てコスト), 以降=前日引→引, exit日に決済コスト
daily2 = daily.copy()
daily2["prev_close"] = daily2.groupby("code")["adj_close"].shift(1)
dret_cc = daily2.set_index(["code","date"]).eval("adj_close/prev_close - 1")
dret_oc = daily2.set_index(["code","date"]).eval("adj_close/adj_open - 1")

def pf_series(sub):
    """トレード集合 → 日次EWポートフォリオ収益系列"""
    recs = {}
    for _, t in sub.iterrows():
        c = t["code"]; ei = int(t["ent_idx"]); xi = int(t["ex_idx"])
        for k in range(ei, xi+1):
            d = cal[k]
            if k == ei:
                r = dret_oc.get((c, d), np.nan) - COST_1W
            else:
                r = dret_cc.get((c, d), np.nan)
            if k == xi:
                r = (r if not np.isnan(r) else 0.0) - COST_1W
            if not np.isnan(r):
                recs.setdefault(d, []).append(r)
    if not recs:
        return pd.Series(dtype=float)
    return pd.Series({d: np.mean(v) for d, v in recs.items()}).sort_index()

def sh(ser):
    ser = ser.dropna()
    if len(ser) < 5: return dict(N=len(ser), sharpe=np.nan, ann=np.nan, mdd=np.nan, cum=np.nan)
    s = ser.mean()/ser.std()*np.sqrt(252)
    cum = (1+ser).prod()-1
    mdd = (((1+ser).cumprod()/(1+ser).cumprod().cummax())-1).min()
    return dict(N=len(ser), sharpe=s, ann=ser.mean()*252*100, mdd=mdd*100, cum=cum*100)

print("\n" + "="*78)
print("日次EWポートフォリオ Sharpe（honest・コスト後4bps）")
print("="*78)
print(f"{'variant':<16}{'seg':<5}{'稼働日':>7}{'Sharpe':>8}{'年率%':>8}{'MDD%':>8}{'累積%':>9}")
rows_pf = []
variants = {"base(FY/2Q/3Q)": tr[tr.grp=="base"], "all(base+1Q)": tr}
for label, sub in variants.items():
    for seg in ["IS","OOS","ALL"]:
        s2 = sub if seg=="ALL" else sub[sub.seg==seg]
        r = sh(pf_series(s2))
        rows_pf.append(dict(variant=label, seg=seg, **r))
        print(f"{label:<16}{seg:<5}{r['N']:>7}{r['sharpe']:>8.2f}{r['ann']:>8.1f}{r['mdd']:>8.1f}{r['cum']:>9.1f}")

# 稼働日数（低タッチ観点: 年間何日ポジションを持つか）
for label, sub in variants.items():
    ser = pf_series(sub[sub.seg=="OOS"])
    yrs = sub[sub.seg=="OOS"]["year"].nunique()
    print(f"  {label}: OOS稼働 {len(ser)}日 / {yrs}年 = 年 {len(ser)/max(yrs,1):.0f}日")

pd.DataFrame(rows_pt).to_csv("per_trade_stats.csv", index=False)
pd.DataFrame(rows_pf).to_csv("pf_sharpe.csv", index=False)

# ---------- 年次 per-trade（レジーム安定性の核心）----------
yr_base = tr[tr.grp=="base"].groupby("year")["net"].mean()*100
yr_1q   = tr[tr.grp=="1Q"].groupby("year")["net"].mean()*100
pd.DataFrame({"base_mean_pct": yr_base, "q1_mean_pct": yr_1q}).to_csv("per_year_mean.csv")

# ---------- 可視化 ----------
try:
    import matplotlib.font_manager as fm
    fm.fontManager.addfont("/root/.fonts/NotoSansJP.ttf")
    plt.rcParams["font.family"] = fm.FontProperties(fname="/root/.fonts/NotoSansJP.ttf").get_name()
    plt.rcParams["axes.unicode_minus"] = False
except Exception:
    pass
fig, axes = plt.subplots(1, 2, figsize=(12, 6.75), facecolor="white")
# 左: 年次平均net%（base vs 1Q）— レジーム不安定性が一目で分かる
yrs = sorted(set(yr_base.index) | set(yr_1q.index))
x = np.arange(len(yrs)); w = 0.4
axes[0].bar(x-w/2, [yr_base.get(y, 0) for y in yrs], w, label="base(FY/2Q/3Q)", color="#2da44e")
axes[0].bar(x+w/2, [yr_1q.get(y, 0) for y in yrs], w, label="1Q", color="#cf222e")
axes[0].axhline(0, color="black", lw=0.8)
axes[0].axvline(5.5, color="gray", ls="--", lw=1)  # IS|OOS 境界 (2021|2022)
axes[0].set_xticks(x); axes[0].set_xticklabels(yrs, rotation=45)
axes[0].set_title("年次 平均net%/trade（決算前ドリフト）\n左=IS(2016-21) 右=OOS(2022-26)")
axes[0].set_ylabel("平均net% / trade"); axes[0].legend(); axes[0].grid(alpha=0.3, axis="y")
# 右: OOS 累積エクイティ base vs all
for label, sub in variants.items():
    ser = pf_series(sub[sub.seg=="OOS"])
    axes[1].plot(ser.index, (1+ser).cumprod(), label=label)
axes[1].set_title("日次EWポートフォリオ 累積（OOS 2022-2026・コスト後4bps）")
axes[1].legend(); axes[1].grid(alpha=0.3)
fig.suptitle("pre_earnings_drift への1Q追加検証 — エッジは直近レジーム依存", fontsize=13)
fig.text(0.99, 0.01, "データ: fin_summary(2016-2026)決算発表 × stocks_daily / コスト往復4bps",
         ha="right", va="bottom", fontsize=8, color="gray")
fig.savefig("result.png", dpi=100, bbox_inches="tight", facecolor="white")
print("\nsaved result.png / per_trade_stats.csv / pf_sharpe.csv / per_year_mean.csv")
