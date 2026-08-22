"""勤務先制約（不動産・銀行を除外）を課した最新リスト。

restricted.py は形成日(月末)基準の利回りで並べたため、直近で急騰した銘柄の利回りが
古いまま上位に来ていた（例: トーメンデバイス 7/30 17,150円 → 8/21 28,350円 の+65%で
形成日利回り7.75%・現値利回り5.78%）。ここでは final.py と同じく**最新終値で統一**して再計算する。
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
from pathlib import Path
import numpy as np
import pandas as pd
from jstock import db

HERE = Path(__file__).resolve().parent
EX_A = ["不動産業", "銀行業"]
EX_B = EX_A + ["証券･商品先物取引業", "その他金融業", "保険業"]

px = db.read_sql("""SELECT code, date, close, adj_close, turnover_value
                    FROM stocks_daily WHERE date >= '2026-01-01' AND close>0 AND adj_close>0""")
px["date"] = pd.to_datetime(px["date"])
g = px.groupby("code")
snap = pd.DataFrame({"close": g["close"].last(), "adjc": g["adj_close"].last(),
                     "adv": g["turnover_value"].apply(lambda s: s.tail(60).mean()),
                     "p60": g["close"].apply(lambda s: s.iloc[-60] if len(s) >= 60 else np.nan),
                     "p20": g["close"].apply(lambda s: s.iloc[-20] if len(s) >= 20 else np.nan)})
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
hist = fin.dropna(subset=["fdiv"]).query("fdiv > 0").sort_values("disc_date")
hist["yr"] = hist["disc_date"].dt.year
cuts = (hist.groupby(["code", "yr"])["fdiv"].last().reset_index().sort_values(["code", "yr"])
        .groupby("code")["fdiv"].apply(lambda s: int((s.pct_change() < -0.01).sum())))

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
d["ret3m%"] = (d["close"] / d["p60"] - 1) * 100
d["ret1m%"] = (d["close"] / d["p20"] - 1) * 100
sm = db.read_sql("SELECT code5 code, name_ja, sector33_nm, market_nm FROM symbol_master")
d = d.join(sm.set_index("code"), how="left")

BASE = ((d["adv"] >= 3e8) & d["yield"].between(0.01, 20) & (d["fnp"] > 0)
        & d["payout"].between(20, 80) & (d["roe"] >= 8.0)
        & d["market_nm"].isin(["プライム", "スタンダード"]))

for tag, ex in [("A", EX_A), ("B", EX_B)]:
    sel = BASE & ~d["sector33_nm"].isin(ex)
    top = d[sel].nlargest(30, "yield")
    O = pd.DataFrame({
        "コード": top.index.str[:4], "銘柄": top["name_ja"], "業種": top["sector33_nm"],
        "株価": top["close"].round(0).astype(int), "予想利回%": top["yield"].round(2),
        "予想PER": top["per"].round(1), "PBR": top["pbr"].round(2),
        "予想ROE%": top["roe"].round(1), "配当性向%": top["payout"].round(0),
        "時価総額億": (top["mktcap"] / 1e8).round(0).astype(int),
        "5年減配": top["cuts5y"], "純利/営利": top["oneoff"].round(2),
        "3ヶ月騰落%": top["ret3m%"].round(1),
    }).reset_index(drop=True)
    O.index = O.index + 1
    print("=" * 124)
    print(f"【{tag}】{LAST.date()}終値・予想利回り順  除外業種: {'・'.join(ex)}")
    print("=" * 124)
    print(O.to_string())
    print(f"  平均予想利回り {top['yield'].mean():.2f}% / PER中央 {top['per'].median():.1f} / "
          f"業種数 {top['sector33_nm'].nunique()} / 単元合計 {(top['close']*100).sum():,.0f}円")
    print("  業種:", top["sector33_nm"].value_counts().head(5).to_dict())
    O.to_csv(HERE / f"list_{tag}.csv", encoding="utf-8-sig")
print("\nsaved list_A.csv / list_B.csv")
