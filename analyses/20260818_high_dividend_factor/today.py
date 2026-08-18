"""検証した「高配当×質」ルールを最新日に適用した実際の保有候補30銘柄＋結果図。"""
import sys
sys.stdout.reconfigure(line_buffering=True)
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from jstock import db

HERE = Path(__file__).resolve().parent
try:
    import matplotlib.font_manager as fm
    fm.fontManager.addfont("/root/.fonts/NotoSansJP.ttf")
    plt.rcParams["font.family"] = fm.FontProperties(fname="/root/.fonts/NotoSansJP.ttf").get_name()
    plt.rcParams["axes.unicode_minus"] = False
except Exception:
    pass

# ---------------- 最新日のスクリーン ----------------
px = db.read_sql("""SELECT code, date, close, adj_close, turnover_value
                    FROM stocks_daily WHERE date >= '2026-01-01' AND close>0 AND adj_close>0""")
px["date"] = pd.to_datetime(px["date"])
g = px.groupby("code")
snap = pd.DataFrame({"close": g["close"].last(), "adjc": g["adj_close"].last(),
                     "adv": g["turnover_value"].apply(lambda s: s.tail(60).mean())})
snap["r_now"] = snap["adjc"] / snap["close"]
LAST = px["date"].max()
rr = px.assign(r=px["adj_close"] / px["close"])[["code", "date", "r"]]

fin = db.read_sql("""
    SELECT code, disc_date, NULLIF(payload->>'FDivAnn','')::float fdiv,
           NULLIF(payload->>'FEPS','')::float feps, NULLIF(payload->>'FNP','')::float fnp,
           NULLIF(payload->>'TA','')::float ta, NULLIF(payload->>'EqAR','')::float eqar,
           NULLIF(payload->>'Eq','')::float eq, NULLIF(payload->>'ShOutFY','')::float shout,
           NULLIF(payload->>'TrShFY','')::float trsh
    FROM fin_summary WHERE doc_type LIKE '%%FinancialStatements%%' AND disc_date >= '2025-01-01'
""")
fin["disc_date"] = pd.to_datetime(fin["disc_date"])
d = fin.sort_values("disc_date").groupby("code").tail(1).set_index("code")
own = d["ta"] * d["eqar"]
d["eq_own"] = np.where(np.isfinite(own) & (own > 0) & (own <= d["eq"]), own, d["eq"])
d = d.join(snap, how="inner")

rj = pd.merge_asof(d.reset_index()[["code", "disc_date"]].sort_values("disc_date"),
                   rr.sort_values("date"), by="code", left_on="disc_date",
                   right_on="date", direction="forward")
d["k"] = (d["r_now"] / rj.set_index("code")["r"].reindex(d.index)).replace([np.inf, -np.inf], np.nan)
d.loc[~np.isfinite(d["k"]) | (d["k"] <= 0), "k"] = 1.0
d["shares"] = (d["shout"] - d["trsh"].fillna(0)) * d["k"]
d["yield"] = (d["fdiv"] / d["k"]) / d["close"] * 100
d["payout"] = np.where(d["feps"] > 0, d["fdiv"] / d["feps"] * 100, np.nan)
d["roe"] = np.where(d["eq_own"] > 0, d["fnp"] / d["eq_own"] * 100, np.nan)
d["per"] = np.where(d["feps"] > 0, d["close"] / (d["feps"] / d["k"]), np.nan)
d["pbr"] = d["close"] * d["shares"] / d["eq_own"]
d["mktcap"] = d["close"] * d["shares"]
sm = db.read_sql("SELECT code5 code, name_ja, sector33_nm, market_nm FROM symbol_master")
d = d.join(sm.set_index("code"), how="left")

SEL = ((d["adv"] >= 3e8) & d["yield"].between(0.01, 20) & (d["fnp"] > 0)
       & d["payout"].between(20, 80) & (d["roe"] >= 8.0)
       & d["market_nm"].isin(["プライム", "スタンダード"]))
top = d[SEL].nlargest(30, "yield")

O = pd.DataFrame({
    "コード": top.index.str[:4], "銘柄": top["name_ja"], "業種": top["sector33_nm"],
    "株価": top["close"].round(0).astype(int), "予想利回%": top["yield"].round(2),
    "予想PER": top["per"].round(1), "PBR": top["pbr"].round(2),
    "予想ROE%": top["roe"].round(1), "配当性向%": top["payout"].round(0),
    "時価総額億": (top["mktcap"] / 1e8).round(0).astype(int),
}).reset_index(drop=True)
print("=" * 110)
print(f"高配当×質 上位30銘柄（{LAST.date()}終値・予想利回り順）")
print("  条件: ADV≥3億 / 配当性向20-80% / 予想ROE≥8% / 予想最終黒字 / プライム・スタンダード")
print("=" * 110)
print(O.to_string())
O.to_csv(HERE / "today_top30.csv", index=False, encoding="utf-8-sig")
print(f"\n平均予想利回り {top['yield'].mean():.2f}% / 平均PER {top['per'].median():.1f} / "
      f"業種数 {top['sector33_nm'].nunique()}")
print(top["sector33_nm"].value_counts().head(6).to_string())

# ---------------- 図 ----------------
P = pd.read_csv(HERE / "summary_plain.csv")
Y = pd.read_csv(HERE / "yearly.csv", index_col=0)
S30 = pd.read_csv(HERE / "top30_annual.csv", index_col=0, parse_dates=True).iloc[:, 0]

fig, axes = plt.subplots(1, 2, figsize=(12, 6.75), facecolor="white")

ax = axes[0]
lab = {"Q1": "Q1\n低利回り", "Q2": "Q2", "Q3": "Q3", "Q4": "Q4", "Q5": "Q5\n高配当",
       "無配": "無配", "市場EW": "市場\nEW"}
P["lb"] = P["分位"].map(lab)
col = ["#8c959f"] * 5 + ["#cf222e", "#bf8700"]
b = ax.bar(P["lb"], P["年率"], color=col[:len(P)])
b[4].set_color("#0969da")
for i, v in enumerate(P["年率"]):
    ax.text(i, v + (0.5 if v >= 0 else -1.6), f"{v:.1f}%", ha="center", fontsize=9,
            fontweight="bold" if P["分位"].iloc[i] in ("Q5", "無配") else "normal")
ax.axhline(0, color="black", lw=0.9)
ax.set_ylabel("年率リターン %（配当込み・コスト込み）")
ax.set_title("予想配当利回り5分位 2016-2026\n単調に効く。効果の半分は「無配を避ける」こと", fontsize=11)
ax.grid(alpha=0.3, axis="y")

ax = axes[1]
x = np.arange(len(Y))
ax.bar(x - 0.22, Y["高配当Q5"], 0.44, label="高配当Q5", color="#0969da")
ax.bar(x + 0.22, Y["市場EW"], 0.44, label="市場EW", color="#bf8700")
ax.axhline(0, color="black", lw=0.9)
ax.axvline(4.5, color="#cf222e", ls="--", lw=1.4)
ax.text(4.6, ax.get_ylim()[1] * 0.86, "←IS: 市場に負けていた\n  OOS: 大幅超過→",
        fontsize=8.5, color="#cf222e")
ax.set_xticks(x)
ax.set_xticklabels(Y.index, rotation=45, fontsize=8)
ax.set_ylabel("年次リターン %")
ax.set_title("超過は2021年以降に集中している\nIS(〜2020) 年率3.9% vs 市場4.7% ＝ 当時は勝てていない", fontsize=11)
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis="y")

fig.suptitle("「高配当を買っておけばいいか」— 10年検証", fontsize=14)
fig.text(0.99, 0.005, "データ: JQuants stocks_daily + fin_summary 2016-2026 / PIT・生存者バイアス排除 / "
                      "配当を足し戻し・売買コスト込み", ha="right", va="bottom", fontsize=8, color="gray")
fig.savefig(HERE / "result.png", dpi=100, bbox_inches="tight", facecolor="white")
print("\nsaved today_top30.csv / result.png")
