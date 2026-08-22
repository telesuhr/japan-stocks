"""検証で生き残ったルール（高配当×質・上位30銘柄・制約なし）を最新日に適用した最終リスト。

build.py の結論: 一過性利益フィルタ・業種上限はいずれも守るべきリスクを減らさず年率だけ削った → 不採用。
hold.py の結論: 一度買って3〜5年放置しても年率11〜13%・5年保有では負けなし・市場に86%勝ち → 放置可。

放置運用に固有の情報として「過去5年の減配回数」を付ける（5年持つなら減配歴は見るべき）。
"""
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
           NULLIF(payload->>'FOP','')::float fop, NULLIF(payload->>'TA','')::float ta,
           NULLIF(payload->>'EqAR','')::float eqar, NULLIF(payload->>'Eq','')::float eq,
           NULLIF(payload->>'ShOutFY','')::float shout, NULLIF(payload->>'TrShFY','')::float trsh
    FROM fin_summary WHERE doc_type LIKE '%%FinancialStatements%%' AND disc_date >= '2020-01-01'
""")
fin["disc_date"] = pd.to_datetime(fin["disc_date"])

# 過去5年の減配回数（年1回の年間予想配当の推移で判定）
hist = fin.dropna(subset=["fdiv"]).query("fdiv > 0").sort_values("disc_date")
hist["yr"] = hist["disc_date"].dt.year
ann_div = hist.groupby(["code", "yr"])["fdiv"].last().reset_index()
cuts = (ann_div.sort_values(["code", "yr"]).groupby("code")["fdiv"]
        .apply(lambda s: int(((s.pct_change() < -0.01)).sum())))

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
d["oneoff"] = np.where(d["fop"] > 0, d["fnp"] / d["fop"], np.nan)
d["cuts5y"] = cuts.reindex(d.index).fillna(0).astype(int)
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
    "5年減配回数": top["cuts5y"], "純利/営利": top["oneoff"].round(2),
}).reset_index(drop=True)
O.index = O.index + 1
print("=" * 120)
print(f"【買って放置】高配当×質 上位30銘柄  {LAST.date()}終値・予想利回り順")
print("  ルール: ADV≥3億 / 配当性向20-80% / 予想ROE≥8% / 予想最終黒字 / プライム・スタンダード")
print("=" * 120)
print(O.to_string())
O.to_csv(HERE / "final_list.csv", encoding="utf-8-sig")
print(f"\n平均予想利回り {top['yield'].mean():.2f}% / PER中央 {top['per'].median():.1f} / "
      f"PBR中央 {top['pbr'].median():.2f} / 業種数 {top['sector33_nm'].nunique()}")
print(f"5年で一度も減配なし: {(top['cuts5y']==0).sum()}/30銘柄")
print(top["sector33_nm"].value_counts().head(6).to_string())

# ---------------- 図 ----------------
V = pd.read_csv(HERE / "variants.csv")
H = pd.read_csv(HERE / "hold_years.csv")

fig, axes = plt.subplots(1, 2, figsize=(12, 6.75), facecolor="white")

ax = axes[0]
x = np.arange(len(H))
ax.bar(x - 0.2, H["年率中央"], 0.4, label="高配当×質30銘柄", color="#0969da")
ax.bar(x + 0.2, H["市場EW年率中央"], 0.4, label="市場EW", color="#bf8700")
for i, r in H.iterrows():
    ax.text(i - 0.2, r["年率中央"] + 0.3, f"{r['年率中央']:.1f}%", ha="center", fontsize=9, fontweight="bold")
    ax.text(i, -1.9, f"負け{r['負けた割合']:.0f}%", ha="center", fontsize=8, color="#cf222e")
ax.set_xticks(x)
ax.set_xticklabels([f"{int(y)}年放置\n(n={int(n)})" for y, n in zip(H["放置年数"], H["標本"])], fontsize=9)
ax.set_ylabel("年率リターン中央値 %（配当込み・退場銘柄も反映）")
ax.set_ylim(-3, 18)
ax.axhline(0, color="black", lw=0.9)
ax.set_title("一度買って放置しても壊れない\n5年保有では負けなし・市場に86%勝ち", fontsize=11)
ax.legend(fontsize=9, loc="upper right")
ax.grid(alpha=0.3, axis="y")

ax = axes[1]
lb = ["検証済み\n(制約なし)", "+一過性\n利益除外", "+業種\n上限3", "20銘柄\nに集約"]
c = ["#0969da", "#8c959f", "#8c959f", "#8c959f"]
b = ax.bar(lb, V["年率"], color=c)
for i, v in enumerate(V["年率"]):
    ax.text(i, v + 0.2, f"{v:.1f}%", ha="center", fontsize=9,
            fontweight="bold" if i == 0 else "normal")
ax.set_ylabel("年率リターン %（年1回入替・共通116ヶ月）")
ax.set_ylim(0, 15)
ax.set_title("『良かれと思った制約』は全て性能を削った\n減配も下落も防がず、利回りだけ捨てていた", fontsize=11)
ax.grid(alpha=0.3, axis="y")

fig.suptitle("買って放置する現物ポートフォリオ — 何を買い、何年持つか", fontsize=14)
fig.text(0.99, 0.005, "データ: JQuants stocks_daily + fin_summary 2016-2026 / PIT・生存者バイアス排除 / "
                      "配当込み・コスト込み", ha="right", va="bottom", fontsize=8, color="gray")
fig.savefig(HERE / "result.png", dpi=100, bbox_inches="tight", facecolor="white")
print("\nsaved final_list.csv / result.png")
