"""
PBR1倍割れは買いか売りか（全市場・クロスセクション）。
月末リバランス／PIT BPS(先読み無)／コスト後／2023-03東証要請でIS-OOS。
"""
import sys; sys.stdout.reconfigure(line_buffering=True)
from pathlib import Path
import numpy as np, pandas as pd
from jstock import db, costs, stats

HERE = Path(__file__).resolve().parent
TSE_REQUEST = "2023-03-31"          # 東証「資本コスト・株価を意識した経営」要請
LIQ = 1e8                            # 月次平均売買代金 >= 1億円
FIN_SECTORS = ("銀行業", "保険業", "証券･商品先物取引業", "その他金融業")

# ---------- データ取得 ----------
# 月末: raw close, adj_close, 月次平均売買代金（1クエリ・window）
px = db.read_sql("""
  WITH m AS (
    SELECT code, date, close, adj_close, turnover_value,
           date_trunc('month',date)::date mo,
           row_number() OVER (PARTITION BY code, date_trunc('month',date) ORDER BY date DESC) rn,
           avg(turnover_value) OVER (PARTITION BY code, date_trunc('month',date)) tv_avg
    FROM stocks_daily WHERE date>='2016-01-01' AND adj_close>0 AND close>0
  )
  SELECT code, mo, date me_date, close rawc, adj_close adjc, tv_avg
  FROM m WHERE rn=1 ORDER BY code, mo
""", [])
fin = db.read_sql("""
  SELECT code, disc_date,
         NULLIF(payload->>'BPS','')::float bps,
         NULLIF(payload->>'NP','')::float np,
         NULLIF(payload->>'Eq','')::float eq
  FROM fin_summary WHERE NULLIF(payload->>'BPS','') IS NOT NULL AND (payload->>'BPS')::float>0
  ORDER BY disc_date
""", [])
sm = db.read_sql("SELECT code5 code, name_ja, sector33_nm FROM symbol_master", [])
print(f"月末パネル {len(px):,}行 / 財務 {len(fin):,}行 / master {len(sm):,}")

px["me_date"] = pd.to_datetime(px["me_date"]); px["mo"] = pd.to_datetime(px["mo"])
fin["disc_date"] = pd.to_datetime(fin["disc_date"])
for c in ["rawc","adjc","tv_avg"]: px[c] = px[c].astype(float)

# ---------- PIT BPS を merge_asof で結合（銘柄別・backward） ----------
px = px.sort_values("me_date")
fin = fin.sort_values("disc_date")
merged = pd.merge_asof(px, fin, by="code", left_on="me_date", right_on="disc_date",
                       direction="backward")
merged = merged.merge(sm, on="code", how="left")

# PBR = raw close / 開示BPS（教科書）。ROE = NP/Eq。
merged["pbr"] = merged["rawc"] / merged["bps"]
merged["roe"] = np.where(merged["eq"] > 0, merged["np"] / merged["eq"] * 100, np.nan)

# forwardリターンは adj_close チェーン（銘柄内で翌月）
merged = merged.sort_values(["code","mo"])
merged["fwd"] = merged.groupby("code")["adjc"].shift(-1) / merged["adjc"] - 1

# ---------- 月次バスケット構築 ----------
def monthly_baskets(df, label):
    df = df[(df["pbr"] > 0) & (df["pbr"] < 10) & (df["tv_avg"] >= LIQ) & df["fwd"].notna()].copy()
    recs = []
    for mo, g in df.groupby("mo"):
        if len(g) < 30:      # 分位を切れる最小母集団
            continue
        g = g.copy()
        g["q"] = pd.qcut(g["pbr"], 5, labels=["Q1","Q2","Q3","Q4","Q5"], duplicates="drop")
        row = {"mo": mo, "n": len(g), "mkt_ew": g["fwd"].mean()}
        for q in ["Q1","Q2","Q3","Q4","Q5"]:
            sub = g[g["q"] == q]
            row[q] = sub["fwd"].mean() if len(sub) else np.nan
        # PBR<1 vs >=1
        row["below1"] = g.loc[g["pbr"] < 1, "fwd"].mean()
        row["above1"] = g.loc[g["pbr"] >= 1, "fwd"].mean()
        row["n_below1"] = int((g["pbr"] < 1).sum())
        # 質オーバーレイ: 最割安2分位(下位40%) の中で ROE 上/下
        cheap = g[g["pbr"] <= g["pbr"].quantile(0.4)]
        cq = cheap[cheap["roe"].notna()]
        if len(cq) >= 10:
            med = cq["roe"].median()
            row["cheap_hiROE"] = cq.loc[cq["roe"] > med, "fwd"].mean()
            row["cheap_loROE"] = cq.loc[cq["roe"] <= med, "fwd"].mean()
        recs.append(row)
    b = pd.DataFrame(recs).set_index("mo").sort_index()
    b["Q1_Q5"] = b["Q1"] - b["Q5"]                 # 割安ロング/割高ショート(L/S)
    b["below_ex"] = b["below1"] - b["mkt_ew"]      # PBR<1 の対市場EW超過
    b["label"] = label
    return b

def msum(series, label):
    r = pd.Series(series).dropna()
    if len(r) < 3:
        return {"label": label, "n": len(r), "ann%": np.nan, "sharpe": np.nan, "t": np.nan, "win%": np.nan}
    return {"label": label, "n": len(r),
            "ann%": r.mean()*12*100, "sharpe": stats.sharpe(r, ann=12),
            "t": stats.t_stat(r), "win%": (r>0).mean()*100}

def report(b, title):
    print(f"\n{'='*82}\n★ {title}  （{b.index.min().date()}〜{b.index.max().date()}, {len(b)}ヶ月, 月平均 {b['n'].mean():.0f}銘柄, うちPBR<1 {b['n_below1'].mean():.0f}銘柄）")
    # ----- H2 分位（コスト後 long-only 往復1回/月） -----
    print("\n[H2] PBR5分位 等加重 long-only（コスト後・年率）  Q1=最割安 … Q5=最割高")
    rows = []
    for q in ["Q1","Q2","Q3","Q4","Q5","mkt_ew"]:
        net = costs.net_returns(b[q], round_trips=1)
        rows.append({**msum(net, q), "gross_ann%": b[q].mean()*12*100})
    R = pd.DataFrame(rows)[["label","n","gross_ann%","ann%","sharpe","t","win%"]]
    print(R.to_string(index=False, float_format=lambda x: f"{x:7.2f}"))
    # ----- H1 PBR<1 basket & 対市場超過 -----
    print("\n[H1] PBR<1 を買う（対 全市場EW）")
    for nm, ser, ls in [("PBR<1 (net)", costs.net_returns(b["below1"], round_trips=1), False),
                        ("PBR>=1 (net)", costs.net_returns(b["above1"], round_trips=1), False),
                        ("市場EW (net)", costs.net_returns(b["mkt_ew"], round_trips=1), False),
                        ("PBR<1 − 市場EW 超過(gross)", b["below_ex"], False)]:
        s = msum(ser, nm); print(f"  {nm:26} ann={s['ann%']:6.2f}% Sharpe={s['sharpe']:5.2f} t={s['t']:5.2f} win={s['win%']:5.1f}%")
    # ----- Q1-Q5 L/S -----
    ls_net = costs.net_returns(b["Q1_Q5"], ls=True)
    s = msum(ls_net, "Q1-Q5 L/S(net 8bp)")
    print(f"\n[L/S] 割安Q1ロング − 割高Q5ショート（コスト後8bp/月）: ann={s['ann%']:6.2f}% Sharpe={s['sharpe']:5.2f} t={s['t']:5.2f} win={s['win%']:5.1f}%")
    # ----- H3 質オーバーレイ -----
    if "cheap_hiROE" in b:
        hi = msum(costs.net_returns(b["cheap_hiROE"], round_trips=1), "cheap×高ROE")
        lo = msum(costs.net_returns(b["cheap_loROE"], round_trips=1), "cheap×低ROE")
        print(f"\n[H3] 割安(下位40%)内 質オーバーレイ(net): 高ROE ann={hi['ann%']:6.2f}% Sh={hi['sharpe']:.2f} | 低ROE ann={lo['ann%']:6.2f}% Sh={lo['sharpe']:.2f}")
    return R

# 本体: 金融除外
core = merged[~merged["sector33_nm"].isin(FIN_SECTORS)]
b_core = monthly_baskets(core, "非金融")
report(b_core, "全市場（金融除外）")

# H4 レジーム IS/OOS
b_is = b_core[b_core.index <= TSE_REQUEST]
b_oos = b_core[b_core.index >= "2023-04-01"]
print(f"\n{'#'*82}\n[H4] 東証PBR改善要請(2023-03-31) 前後  Q1-Q5 L/S(net) & PBR<1超過")
for nm, bb in [("IS(〜2023-03)", b_is), ("OOS(2023-04〜)", b_oos)]:
    ls = msum(costs.net_returns(bb["Q1_Q5"], ls=True), nm)
    ex = msum(bb["below_ex"], nm)
    print(f"  {nm:16} Q1-Q5 L/S: ann={ls['ann%']:6.2f}% Sh={ls['sharpe']:5.2f} t={ls['t']:5.2f} | PBR<1超過 ann={ex['ann%']:6.2f}% t={ex['t']:5.2f}")

# 補助: 金融込み
b_all = monthly_baskets(merged, "全業種")
report(b_all, "全市場（金融込み・参考）")

# ---------- 可視化 ----------
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.font_manager as fm
try:
    fm.fontManager.addfont("/root/.fonts/NotoSansJP.ttf")
    plt.rcParams["font.family"] = "Noto Sans JP"
    plt.rcParams["axes.unicode_minus"] = False
except Exception:
    pass

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.75), facecolor="white",
                               gridspec_kw={"width_ratios":[1.6,1]})
# 左: 累積カーブ(コスト後)
for q, col, lab in [("Q1","#2ca02c","割安Q1"),("Q5","#d62728","割高Q5"),
                    ("below1","#1f6feb","PBR<1"),("mkt_ew","#888","市場EW")]:
    s = costs.net_returns(b_core[q], round_trips=1).dropna()
    ax1.plot(s.index, (1+s).cumprod().values, label=lab, lw=2, color=col)
ax1.axvline(pd.Timestamp(TSE_REQUEST), color="k", ls=":", lw=1)
ax1.text(pd.Timestamp(TSE_REQUEST), ax1.get_ylim()[1], " 東証要請", fontsize=9, va="top")
ax1.set_title("PBR分位 等加重 累積(コスト後・金融除外)"); ax1.set_ylabel("成長率(倍)")
ax1.legend(); ax1.grid(alpha=0.3); ax1.set_yscale("log")
# 右: 分位別 年率(コスト後)
qs = ["Q1","Q2","Q3","Q4","Q5"]
anns = [costs.net_returns(b_core[q], round_trips=1).mean()*12*100 for q in qs]
ax2.bar(qs, anns, color=["#2ca02c","#7ac07a","#bbb","#e08a8a","#d62728"])
ax2.axhline(0, color="k", lw=0.8)
ax2.set_title("分位別 年率リターン(コスト後%)\nQ1=最割安 → Q5=最割高")
for i,v in enumerate(anns): ax2.text(i, v, f"{v:.1f}", ha="center", va="bottom" if v>=0 else "top", fontsize=9)
ax2.grid(alpha=0.3, axis="y")
fig.suptitle("PBR1倍割れは買いか売りか（全市場・金融除外・2016-2026）", fontsize=15)
fig.tight_layout()
fig.savefig(HERE/"result.png", dpi=100, bbox_inches="tight", facecolor="white")
print("\nsaved result.png")
