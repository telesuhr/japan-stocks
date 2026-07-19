"""
最新月末スクリーン: PBR<1 × 高ROE(質) × 流動性≥1億 の候補リスト。
検証で「割安×質は小型ほど効く／低PBR単体で概ね効く」を踏まえ、
バリュートラップ(ROE<0の低PBR)を除外して低PBR×健全ROEを抽出。
"""
import sys; sys.stdout.reconfigure(line_buffering=True)
from pathlib import Path
import numpy as np, pandas as pd
from jstock import db

HERE = Path(__file__).resolve().parent
LIQ = 1e8
FIN_SECTORS = ("銀行業", "保険業", "証券･商品先物取引業", "その他金融業")

px = db.read_sql("""
  WITH m AS (
    SELECT code, date, close, turnover_value,
           date_trunc('month',date)::date mo,
           row_number() OVER (PARTITION BY code, date_trunc('month',date) ORDER BY date DESC) rn,
           avg(turnover_value) OVER (PARTITION BY code, date_trunc('month',date)) tv_avg
    FROM stocks_daily WHERE date>='2026-01-01' AND close>0)
  SELECT code, mo, date me_date, close rawc, tv_avg
  FROM m WHERE rn=1 ORDER BY code, mo""", [])
px["me_date"]=pd.to_datetime(px["me_date"]); px["mo"]=pd.to_datetime(px["mo"])
latest_mo = px["mo"].max()
px = px[px["mo"]==latest_mo].copy()
for c in ["rawc","tv_avg"]: px[c]=px[c].astype(float)

fin = db.read_sql("""
  SELECT DISTINCT ON (code) code, disc_date,
         NULLIF(payload->>'BPS','')::float bps, NULLIF(payload->>'EPS','')::float eps,
         NULLIF(payload->>'NP','')::float np, NULLIF(payload->>'Eq','')::float eq,
         NULLIF(payload->>'DivAnn','')::float divann
  FROM fin_summary WHERE NULLIF(payload->>'BPS','') IS NOT NULL AND (payload->>'BPS')::float>0
  ORDER BY code, disc_date DESC""", [])
sm = db.read_sql("SELECT code5 code, code4, name_ja, sector33_nm, market_nm FROM symbol_master", [])

d = px.merge(fin, on="code", how="left").merge(sm, on="code", how="left")
d["pbr"]=d["rawc"]/d["bps"]
d["roe"]=np.where(d["eq"]>0, d["np"]/d["eq"]*100, np.nan)
d["per"]=np.where(d["eps"]>0, d["rawc"]/d["eps"], np.nan)
d["divy"]=np.where(d["divann"].notna(), d["divann"]/d["rawc"]*100, np.nan)
d["mcap"]=np.where(d["bps"]>0, d["rawc"]*d["eq"]/d["bps"]/1e8, np.nan)  # 億円

# スクリーン: 非金融・PBR<1・ROE>=8%(健全)・流動性≥1億・PBR異常値除外
c = d[(~d["sector33_nm"].isin(FIN_SECTORS)) & (d["pbr"]>0.1) & (d["pbr"]<1.0)
      & (d["roe"]>=8) & (d["tv_avg"]>=LIQ)].copy()
# 複合スコア: 低PBR順位 + 高ROE順位（小さいほど良い割安、大きいほど良い質）
c["score"] = c["roe"].rank(ascending=False) + c["pbr"].rank(ascending=True)
c = c.sort_values("score")

print(f"最新月末: {latest_mo.date()} / 母集団(非金融・流動性≥1億) {len(d[(~d['sector33_nm'].isin(FIN_SECTORS))&(d['tv_avg']>=LIQ)])}銘柄")
print(f"スクリーン該当(PBR<1 & ROE>=8% & 代金≥1億): {len(c)}銘柄\n")
cols=["code4","name_ja","sector33_nm","pbr","per","roe","divy","mcap","tv_avg"]
show=c[cols].head(30).copy()
show["tv_avg"]=show["tv_avg"]/1e8
print(f"{'code':5}{'銘柄':16}{'業種':12}{'PBR':>6}{'PER':>6}{'ROE%':>6}{'配当%':>6}{'時価億':>8}{'代金億':>7}")
for _,r in show.iterrows():
    nm=str(r['name_ja'])[:14]; sec=str(r['sector33_nm'])[:10]
    print(f"{r['code4']:5}{nm:16}{sec:12}{r['pbr']:6.2f}{(r['per'] if pd.notna(r['per']) else 0):6.1f}"
          f"{r['roe']:6.1f}{(r['divy'] if pd.notna(r['divy']) else 0):6.1f}{(r['mcap'] if pd.notna(r['mcap']) else 0):8.0f}{r['tv_avg']:7.1f}")

c[cols].to_csv(HERE/"candidates_pbr_below1.csv", index=False)
print(f"\nsaved candidates_pbr_below1.csv ({len(c)}銘柄)")
