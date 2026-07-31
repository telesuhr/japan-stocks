"""7/29の乗り遅れバリュー32銘柄が、その後(7/28→7/30)どう動いたかの追跡＋ローテーション継続確認。"""
import sys; sys.stdout.reconfigure(line_buffering=True)
import pandas as pd
from jstock import db

cand = pd.read_csv("laggard_value_candidates.csv", encoding="utf-8-sig")
cand["code"] = cand["code"].astype(str).str.zfill(5)
codes = cand["code"].tolist()

px = db.read_sql("""
SELECT code, date, adj_close, turnover_value FROM stocks_daily
WHERE code = ANY(%s) AND date >= '2026-07-25' ORDER BY code, date""", [codes])
px["adj_close"] = px["adj_close"].astype(float)
p = px.pivot(index="date", columns="code", values="adj_close")
print("取得日:", list(p.index))
ret = (p.loc[p.index[-1]] / p.loc[p.index[0]] - 1) * 100   # 7/28→7/30
cand = cand.merge(ret.rename("since728").reset_index(), on="code", how="left")

print(f"\n=== 乗り遅れバリュー32銘柄 その後(7/28→7/30) ===")
print(f"  等加重 {cand['since728'].mean():+.2f}%  中央値 {cand['since728'].median():+.2f}%  "
      f"プラス {int((cand['since728']>0).sum())}/{len(cand)}銘柄")

# 市場ベンチ(EW・ADV1億以上)
mkt = db.read_sql("""
WITH u AS (SELECT code,date,adj_close,
   LAG(adj_close) OVER(PARTITION BY code ORDER BY date) pc,
   AVG(turnover_value) OVER(PARTITION BY code ORDER BY date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING) adv
 FROM stocks_daily WHERE date>='2026-06-01')
SELECT date, AVG((adj_close/pc-1)*100) ew FROM u
WHERE pc IS NOT NULL AND adv>=1e8 AND date>='2026-07-28' GROUP BY date ORDER BY date""", [])
mkt["ew"] = mkt["ew"].astype(float)
mb = ((1 + mkt["ew"] / 100).prod() - 1) * 100
print(f"  市場EW(同期間) {mb:+.2f}%  → 超過 {cand['since728'].mean()-mb:+.2f}%")
print("\n  [個別] 割安順")
print(cand.sort_values("pbr")[["code", "銘柄", "業種", "pbr", "roe", "足元%", "since728", "ADV億"]]
      .head(20).to_string(index=False))
cand.to_csv("laggard_followup_20260731.csv", index=False, encoding="utf-8-sig")
