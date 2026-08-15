"""
pre_earnings_drift の market-neutral 版検証。
base(FY/2Q/3Q) ロング決算前ドリフトに、保有中 1306 TOPIX ETF を等額(β=1)ショート。
ロングオンリー版と 日次PF Sharpe / MDD / 年次リターン を比較。
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from jstock import db

COST_1W = 2.0 / 10000.0
LEAD = {"FY": 5, "2Q": 3, "3Q": 3}       # base のみ（1Qは姉妹分析で見送り）
EXCLUDE_SEC = ("医薬品", "陸運業")
TURN_MIN = 5e8
IS_END_YEAR = 2021
HEDGE = "13060"   # 1306 TOPIX ETF

# ---------- 発表イベント & 株価 ----------
print("データ取得...")
ann = db.read_sql("""
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
      AND (sm.sector33_nm IS NULL OR sm.sector33_nm NOT IN %(exc)s)
""", {"exc": EXCLUDE_SEC})
ann["disc_date"] = pd.to_datetime(ann["disc_date"])

daily = db.read_sql("""
    SELECT code, date, adj_open, adj_close, turnover_value
    FROM stocks_daily WHERE adj_close > 0
""")
daily["date"] = pd.to_datetime(daily["date"])
daily = daily.sort_values(["code", "date"]).reset_index(drop=True)
daily["prev_close"] = daily.groupby("code")["adj_close"].shift(1)
print(f"  発表 {len(ann):,}件 / 日足 {len(daily):,}行")

cal = np.array(sorted(daily["date"].unique()))
pos = {d: i for i, d in enumerate(cal)}
ko = daily.set_index(["code", "date"])["adj_open"]
kc = daily.set_index(["code", "date"])["adj_close"]
kt = daily.set_index(["code", "date"])["turnover_value"]
ret_oc = daily.set_index(["code", "date"]).eval("adj_close/adj_open - 1")
ret_cc = daily.set_index(["code", "date"]).eval("adj_close/prev_close - 1")

# ヘッジ(1306)の日次リターン辞書
h = daily[daily.code == HEDGE].set_index("date")
h_oc = (h["adj_close"] / h["adj_open"] - 1).to_dict()
h_cc = (h["adj_close"] / h["prev_close"] - 1).to_dict()

# ---------- トレード生成 ----------
ann = ann[ann["disc_date"].isin(pos)].copy()
ann["e"] = ann["disc_date"].map(pos).astype(int)
ann["ld"] = ann["cur_per_type"].map(LEAD).astype(int)
ann["si"] = ann["e"] - ann["ld"]; ann["eni"] = ann["si"] + 1; ann["xi"] = ann["e"] - 1
ann = ann[(ann["si"] >= 0) & (ann["eni"] < ann["xi"])].copy()
for c, i in [("sd", "si"), ("ed", "eni"), ("xd", "xi")]:
    ann[c] = cal[ann[i].values]
def lk(s, co, da): return s.reindex(pd.MultiIndex.from_arrays([co, da])).values
ann["turn"] = lk(kt, ann["code"].values, ann["sd"].values)
ann["pin"] = lk(ko, ann["code"].values, ann["ed"].values)
tr = ann.dropna(subset=["turn", "pin"])
tr = tr[(tr["turn"] >= TURN_MIN) & (tr["pin"] > 0)].copy()
tr["year"] = tr["disc_date"].dt.year
tr["seg"] = np.where(tr["year"] <= IS_END_YEAR, "IS", "OOS")
print(f"  有効トレード {len(tr):,}件")

# ---------- 日次PF系列（ロングオンリー / market-neutral）----------
def pf_series(sub, hedged):
    lo, mn = {}, {}
    for _, t in sub.iterrows():
        c = t["code"]; ei = int(t["eni"]); xi = int(t["xi"])
        for k in range(ei, xi + 1):
            d = cal[k]
            if k == ei:
                r = ret_oc.get((c, d), np.nan); hr = h_oc.get(d, np.nan)
                lc = COST_1W                       # ロング建てコスト
                sc = COST_1W                       # ショート建てコスト
            else:
                r = ret_cc.get((c, d), np.nan); hr = h_cc.get(d, np.nan)
                lc = sc = 0.0
            if np.isnan(r):
                continue
            # ロングオンリー
            lr = r - lc - (COST_1W if k == xi else 0.0)
            lo.setdefault(d, []).append(lr)
            # market-neutral: ロング - ヘッジ、両脚コスト
            if not np.isnan(hr):
                mr = (r - hr) - lc - sc - (2 * COST_1W if k == xi else 0.0)
                mn.setdefault(d, []).append(mr)
    to_ser = lambda dd: pd.Series({d: np.mean(v) for d, v in dd.items()}).sort_index()
    return to_ser(lo), to_ser(mn)

def stat(ser):
    ser = ser.dropna()
    if len(ser) < 5:
        return dict(N=len(ser), sharpe=np.nan, ann=np.nan, mdd=np.nan, cum=np.nan)
    return dict(N=len(ser), sharpe=ser.mean()/ser.std()*np.sqrt(252),
                ann=ser.mean()*252*100,
                mdd=(((1+ser).cumprod()/(1+ser).cumprod().cummax())-1).min()*100,
                cum=((1+ser).prod()-1)*100)

print("\n" + "="*74)
print("日次EWポートフォリオ（コスト後）  ロングオンリー vs market-neutral(1306等額S)")
print("="*74)
print(f"{'版':<18}{'seg':<5}{'稼働':>6}{'Sharpe':>8}{'年率%':>8}{'MDD%':>8}{'累積%':>9}")
series = {}
for seg in ["IS", "OOS", "ALL"]:
    sub = tr if seg == "ALL" else tr[tr.seg == seg]
    lo, mn = pf_series(sub, True)
    series[("lo", seg)] = lo; series[("mn", seg)] = mn
    for tag, ser in [("ロングオンリー", lo), ("market-neutral", mn)]:
        r = stat(ser)
        print(f"{tag:<18}{seg:<5}{r['N']:>6}{r['sharpe']:>8.2f}{r['ann']:>8.1f}{r['mdd']:>8.1f}{r['cum']:>9.1f}")

# ---------- 年次リターン（負け年の改善を見る）----------
print("\n年次リターン%（日次PF複利・コスト後）  ロングオンリー / market-neutral")
rows = []
for y in sorted(tr.year.unique()):
    sub = tr[tr.year == y]
    lo, mn = pf_series(sub, True)
    lo_r = ((1+lo).prod()-1)*100 if len(lo) else np.nan
    mn_r = ((1+mn).prod()-1)*100 if len(mn) else np.nan
    rows.append(dict(year=y, long_only=lo_r, market_neutral=mn_r))
    mark = " ←負け年改善" if (lo_r < 0 and mn_r > lo_r) else ""
    print(f"  {y}: LO {lo_r:+6.1f}%   MN {mn_r:+6.1f}%{mark}")
pd.DataFrame(rows).to_csv("per_year_return.csv", index=False)

# ---------- 可視化 ----------
try:
    import matplotlib.font_manager as fm
    fm.fontManager.addfont("/root/.fonts/NotoSansJP.ttf")
    plt.rcParams["font.family"] = fm.FontProperties(fname="/root/.fonts/NotoSansJP.ttf").get_name()
    plt.rcParams["axes.unicode_minus"] = False
except Exception:
    pass
fig, axes = plt.subplots(1, 2, figsize=(12, 6.75), facecolor="white")
lo_all, mn_all = series[("lo", "ALL")], series[("mn", "ALL")]
axes[0].plot(lo_all.index, (1+lo_all).cumprod(), label="ロングオンリー", color="#8250df")
axes[0].plot(mn_all.index, (1+mn_all).cumprod(), label="market-neutral(1306等額S)", color="#2da44e")
axes[0].axvline(pd.Timestamp("2022-01-01"), color="gray", ls="--", lw=1)
axes[0].set_yscale("log"); axes[0].set_title("累積エクイティ 全期間2016-2026（対数軸）\n点線=IS|OOS境界")
axes[0].legend(); axes[0].grid(alpha=0.3)
dfy = pd.DataFrame(rows).set_index("year")
x = np.arange(len(dfy)); w = 0.4
axes[1].bar(x-w/2, dfy["long_only"], w, label="ロングオンリー", color="#8250df")
axes[1].bar(x+w/2, dfy["market_neutral"], w, label="market-neutral", color="#2da44e")
axes[1].axhline(0, color="black", lw=0.8)
axes[1].set_xticks(x); axes[1].set_xticklabels(dfy.index, rotation=45)
axes[1].set_title("年次リターン%（負け年の改善を確認）")
axes[1].legend(); axes[1].grid(alpha=0.3, axis="y")
fig.suptitle("pre_earnings_drift の β中立化（1306等額ショート）効果", fontsize=13)
fig.text(0.99, 0.01, "データ: fin_summary(2016-2026) × stocks_daily / L/S往復8bps",
         ha="right", va="bottom", fontsize=8, color="gray")
fig.savefig("result.png", dpi=100, bbox_inches="tight", facecolor="white")
print("\nsaved result.png / per_year_return.csv")
