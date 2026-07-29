"""
足元の「割安株への資金シフト」は実在するか＋乗り遅れバリュー株の抽出。

検証:
  H1(2026-01→06) と 足元(2026-07以降) で、PBR分位(PIT)のリターンを比較。
  割安Q1が割高Q5を「足元で」上回っていればローテーション実在。
抽出:
  現時点(2026-06-30形成)の割安×質(ROE>0)×流動性の中で、足元まだ上がってない
  (=乗り遅れ)銘柄を返す。
"""
import sys; sys.path.insert(0, '.'); sys.stdout.reconfigure(line_buffering=True)
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from jstock import db

HERE = Path(__file__).resolve().parent
LIQ = 3e8                                # 足元ADV >= 3億円
FIN = ("銀行業", "保険業", "証券･商品先物取引業", "その他金融業")

# ---------- 月末パネル(PIT) ----------
px = db.read_sql("""
  WITH m AS (
    SELECT code, date, close, adj_close, turnover_value,
           date_trunc('month',date)::date mo,
           row_number() OVER (PARTITION BY code, date_trunc('month',date) ORDER BY date DESC) rn,
           avg(turnover_value) OVER (PARTITION BY code, date_trunc('month',date)) tv
    FROM stocks_daily WHERE date>='2025-11-01' AND adj_close>0 AND close>0
  )
  SELECT code, mo, date me_date, close rawc, adj_close adjc, tv
  FROM m WHERE rn=1 ORDER BY code, mo
""", [])
fin = db.read_sql("""
  SELECT code, disc_date,
         NULLIF(payload->>'BPS','')::float bps,
         NULLIF(payload->>'NP','')::float np,
         NULLIF(payload->>'Eq','')::float eq
  FROM fin_summary WHERE NULLIF(payload->>'BPS','')::float>0 ORDER BY disc_date
""", [])
sm = db.read_sql("SELECT code5 code, name_ja, sector33_nm FROM symbol_master", [])

px["me_date"] = pd.to_datetime(px["me_date"]); px["mo"] = pd.to_datetime(px["mo"])
fin["disc_date"] = pd.to_datetime(fin["disc_date"])
for c in ["rawc", "adjc", "tv"]: px[c] = px[c].astype(float)

# PIT BPS を月末ごとに backward結合
p = px.sort_values("me_date")
f = fin.sort_values("disc_date")
mg = pd.merge_asof(p, f, by="code", left_on="me_date", right_on="disc_date", direction="backward")
mg = mg.merge(sm, on="code", how="left")
mg["pbr"] = mg["rawc"] / mg["bps"]
mg["roe"] = np.where(mg["eq"] > 0, mg["np"] / mg["eq"] * 100, np.nan)

# ワイド化: 各銘柄の月末 adj_close
wide = mg.pivot_table(index="code", columns="mo", values="adjc")
months = sorted(mg["mo"].unique())
M_1225 = pd.Timestamp("2025-12-01"); M_0626 = pd.Timestamp("2026-06-01"); M_0726 = pd.Timestamp("2026-07-01")

def snapshot(mo):
    """指定月末のPIT断面(pbr/roe/流動性/非金融)"""
    g = mg[(mg["mo"] == mo) & (mg["pbr"] > 0.1) & (mg["pbr"] < 10)
           & (mg["tv"] >= LIQ) & (~mg["sector33_nm"].isin(FIN))].copy()
    return g

def window_spread(form_mo, ret_from, ret_to, label):
    """form_moでPBR5分位を形成し、ret_from->ret_toのEWリターンを分位別に測る"""
    g = snapshot(form_mo)
    r = wide[ret_to] / wide[ret_from] - 1
    g = g.assign(ret=g["code"].map(r)).dropna(subset=["ret"])
    g["q"] = pd.qcut(g["pbr"], 5, labels=["Q1割安","Q2","Q3","Q4","Q5割高"])
    qret = g.groupby("q")["ret"].mean() * 100
    spread = qret["Q1割安"] - qret["Q5割高"]
    print(f"\n[{label}] 形成={form_mo.date()} 期間={ret_from.date()}->{ret_to.date()} N={len(g)}")
    print(qret.round(2).to_string())
    print(f"  Q1割安 - Q5割高 スプレッド = {spread:+.2f}%  (市場EW {g['ret'].mean()*100:+.2f}%)")
    return qret, spread, g

# ===== 検証: H1 vs 足元 =====
print("="*64)
qH1, sH1, _   = window_spread(M_1225, M_1225, M_0626, "H1 2026 (1月→6月)")
qNow, sNow, gNow = window_spread(M_0626, M_0626, M_0726, "足元 (6月末→7/28)")

# ===== 日次: Q1割安 vs Q5割高 の累積(6/30形成) =====
gform = snapshot(M_0626).copy()
gform["q"] = pd.qcut(gform["pbr"], 5, labels=["Q1","Q2","Q3","Q4","Q5"])
q1codes = gform.loc[gform["q"] == "Q1", "code"].tolist()
q5codes = gform.loc[gform["q"] == "Q5", "code"].tolist()
codes = q1codes + q5codes
dpx = db.read_sql("""
  SELECT code, date, adj_close FROM stocks_daily
  WHERE code = ANY(%s) AND date >= '2026-06-27' AND adj_close>0 ORDER BY date
""", [codes])
dpx["date"] = pd.to_datetime(dpx["date"])
dw = dpx.pivot_table(index="date", columns="code", values="adj_close")
base = dw.iloc[0]
cum = (dw / base)
q1cum = cum[[c for c in q1codes if c in cum.columns]].mean(axis=1)
q5cum = cum[[c for c in q5codes if c in cum.columns]].mean(axis=1)

# ---------- 乗り遅れバリュー株の抽出 ----------
cut40 = gNow["pbr"].quantile(0.40)   # 割安=PBR下位40%

def fmt(df):
    o = df[["code","name_ja","sector33_nm","pbr","roe","ret","rawc","tv"]].copy()
    o["足元%"] = (o["ret"]*100).round(1); o["ADV億"] = (o["tv"]/1e8).round(1)
    o["pbr"] = o["pbr"].round(2); o["roe"] = o["roe"].round(1)
    return o.drop(columns=["ret","tv"]).rename(columns={
        "name_ja":"銘柄","sector33_nm":"業種","rawc":"株価"})

# (a) 見せかけの割安 = グロース崩れ(落ちるナイフ)。足元-15%超の急落 → 乗り遅れではなく売られてる側
crash = gNow[(gNow["pbr"] <= cut40) & (gNow["ret"] <= -0.15)].sort_values("ret")
print("\n" + "="*64)
print(f"【除外】見せかけの割安=グロース崩れ(足元-15%超) {len(crash)}件。乗り遅れではなく売られてる側:")
print(fmt(crash).head(8).to_string(index=False))

# (b) 本命の乗り遅れ = 割安×質(ROE>=8)×流動性(ADV>=5億)×まだ動いてない(-10%<=足元<=+3%)
band = gNow[(gNow["pbr"] <= cut40) & (gNow["roe"] >= 8.0) & (gNow["tv"] >= 5e8)
            & (gNow["ret"] >= -0.10) & (gNow["ret"] <= 0.03)].copy()
band = band.sort_values("pbr")   # 最も割安な質バリューから
lag_out = fmt(band)
lag_out.to_csv(HERE/"laggard_value_candidates.csv", index=False, encoding="utf-8-sig")
print("\n" + "="*64)
print(f"【本命】乗り遅れ割安×質 (PBR下位40%×ROE≥8%×ADV≥5億×足元-10〜+3%) 全{len(lag_out)}件・割安順 上位25")
print(lag_out.head(25).to_string(index=False))
# 業種の偏り
print("\n乗り遅れ候補の業種分布 上位:")
print(band["sector33_nm"].value_counts().head(8).to_string())

# ---------- 可視化 ----------
import matplotlib.font_manager as fm
fm.fontManager.addfont("/root/.fonts/NotoSansJP.ttf")
plt.rcParams["font.family"] = fm.FontProperties(fname="/root/.fonts/NotoSansJP.ttf").get_name()
plt.rcParams["axes.unicode_minus"] = False

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 6.2), facecolor="white",
                               gridspec_kw={"width_ratios":[1.05,1]})
# 左: H1 vs 足元 の分位リターン
x = np.arange(5); w = 0.38
ax1.bar(x-w/2, qH1.values, w, label="H1 (1→6月)", color="#8fa9bf")
ax1.bar(x+w/2, qNow.values, w, label="足元 (6末→7/28)", color="#c0392b")
ax1.set_xticks(x); ax1.set_xticklabels(["Q1\n割安","Q2","Q3","Q4","Q5\n割高"])
ax1.axhline(0, color="#333", lw=0.8)
ax1.set_ylabel("EW平均リターン (%)")
ax1.set_title(f"PBR分位リターン: H1 vs 足元\nQ1-Q5スプレッド  H1={sH1:+.1f}%  →  足元={sNow:+.1f}%",
              fontsize=13, fontweight="bold")
ax1.legend(); ax1.grid(axis="y", alpha=0.3)
# 右: 日次累積 Q1 vs Q5
ax2.plot(q1cum.index, (q1cum-1)*100, color="#c0392b", lw=2.2, label="Q1 割安 (EW)")
ax2.plot(q5cum.index, (q5cum-1)*100, color="#5b6b7a", lw=2.2, label="Q5 割高 (EW)")
ax2.axhline(0, color="#333", lw=0.8)
ax2.set_ylabel("累積リターン (%)"); ax2.set_title("6/30形成 割安 vs 割高 の日次累積", fontsize=13, fontweight="bold")
ax2.legend(); ax2.grid(alpha=0.3)
for lbl in ax2.get_xticklabels(): lbl.set_rotation(30); lbl.set_ha("right")

fig.suptitle("割安株ローテーションの検証 (全市場・PIT PBR・非金融・ADV≥3億)", fontsize=15, fontweight="bold")
fig.text(0.99, 0.005, "データ: JQuants stocks_daily + fin_summary(PIT BPS) / 2026", ha="right", fontsize=8, color="gray")
fig.tight_layout(rect=[0,0.01,1,0.96])
fig.savefig(HERE/"result.png", dpi=100, bbox_inches="tight", facecolor="white")
print("\nsaved result.png")
