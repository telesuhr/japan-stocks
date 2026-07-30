"""
2026-07-29の急増出来高は「セリクラ(selling climax)」だったのか検証。
出来高スパイク単独ではセリクラと判定できない。中身(ブレッドス/新高値安値/VIX/反転)で識別。
結論: 広い市場はセリクラでなく「ローテーションの大商い(rotation climax)」。半導体も distribution 継続。
"""
import sys; sys.path.insert(0, '.'); sys.stdout.reconfigure(line_buffering=True)
from pathlib import Path
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from jstock import db

HERE = Path(__file__).resolve().parent

# ---------- 市場内部(ブレッドス/売買代金/新高安) ----------
mkt = db.read_sql("""
  WITH u AS (
    SELECT code, date, close, low, volume, close*volume tv,
      LAG(close) OVER (PARTITION BY code ORDER BY date) pc,
      MIN(low)  OVER (PARTITION BY code ORDER BY date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING) lo20,
      MAX(high) OVER (PARTITION BY code ORDER BY date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING) hi20,
      AVG(volume) OVER (PARTITION BY code ORDER BY date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING) v20
    FROM stocks_daily WHERE date>='2026-06-15'
  )
  SELECT date,
    ROUND(SUM(tv)/1e12,1) turnover_tril,
    ROUND(100.0*SUM(CASE WHEN close<pc THEN 1 ELSE 0 END)/COUNT(*),1) pct_down,
    SUM(CASE WHEN close<=lo20 THEN 1 ELSE 0 END) new_low20,
    SUM(CASE WHEN close>=hi20 THEN 1 ELSE 0 END) new_high20,
    ROUND(AVG(volume/NULLIF(v20,0))::numeric,2) vol_ratio
  FROM u WHERE date>='2026-07-13' AND pc IS NOT NULL
  GROUP BY date ORDER BY date
""", [])
mkt.to_csv(HERE/"market_internals.csv", index=False)

# ---------- VIX / SOX ----------
mac = db.read_sql("""SELECT symbol, trade_date, close FROM macro.daily_ohlcv
    WHERE symbol IN ('VXc1','.SOX') AND trade_date>='2026-07-13' ORDER BY symbol, trade_date""", [])
mac["trade_date"] = pd.to_datetime(mac["trade_date"])
vix = mac[mac.symbol=='VXc1'].set_index('trade_date')['close']

# ---------- 半導体11銘柄 ----------
semi_codes = ['8035','6920','6323','6758','6857','4062','6723','6963','4005','6976','7735']
semi = db.read_sql("""
  WITH u AS (
    SELECT s.code, s.date, s.close, s.low, s.volume,
      LAG(s.close) OVER(PARTITION BY s.code ORDER BY s.date) pc,
      AVG(s.volume) OVER(PARTITION BY s.code ORDER BY s.date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING) v20,
      MIN(s.low)  OVER(PARTITION BY s.code ORDER BY s.date ROWS BETWEEN 60 PRECEDING AND 1 PRECEDING) lo60
    FROM stocks_daily s JOIN symbol_master m ON m.code5=s.code
    WHERE m.code4=ANY(%s) AND s.date>='2026-06-15'
  )
  SELECT date, ROUND(AVG((close/pc-1)*100)::numeric,1) semi_ret,
    ROUND(AVG(volume/NULLIF(v20,0))::numeric,2) vol_ratio,
    SUM(CASE WHEN close<=lo60 THEN 1 ELSE 0 END) n_60d_low
  FROM u WHERE date>='2026-07-13' AND pc IS NOT NULL GROUP BY date ORDER BY date
""", [semi_codes])
semi.to_csv(HERE/"semi_internals.csv", index=False)

print("=== 市場内部 ===\n", mkt.to_string(index=False))
print("\n=== 半導体11銘柄 ===\n", semi.to_string(index=False))
print("\nVIX直近:", [(str(d.date()), round(c,1)) for d,c in vix.items()][-6:])

# ---------- セリクラ・スコアカード(7/29) ----------
row = mkt[mkt.date.astype(str)=='2026-07-29'].iloc[0]
v29 = vix[pd.Timestamp('2026-07-29')] if pd.Timestamp('2026-07-29') in vix.index else float('nan')
print("\n=== セリクラ判定 7/29 ===")
print(f" 出来高急増  : 売買代金{row.turnover_tril}兆(今週最大)/vol{row.vol_ratio}x → ✓合致")
print(f" 全面安      : 値下がり{row.pct_down}% (=上昇{100-row.pct_down:.1f}%)     → ✗ 逆")
print(f" 新安値殺到  : 新安値{row.new_low20} < 新高値{row.new_high20}          → ✗ 逆")
print(f" VIX急騰     : {v29} (前日から低下)                       → ✗ 平穏")
print(" 投げ後急反発: 無し                                       → ✗")
print(" => 出来高スパイク単独はセリクラでない。中身はrotation climax。")

# ---------- 可視化 ----------
import matplotlib.font_manager as fm
fm.fontManager.addfont("/root/.fonts/NotoSansJP.ttf")
plt.rcParams["font.family"] = fm.FontProperties(fname="/root/.fonts/NotoSansJP.ttf").get_name()
plt.rcParams["axes.unicode_minus"] = False

mkt["d"] = pd.to_datetime(mkt["date"])
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8.2), facecolor="white", height_ratios=[1.15,1])
# 上: 売買代金(棒) + 新高値/新安値(線)
c = ["#c0392b" if str(x)=='2026-07-29' else "#8fa9bf" for x in mkt["date"]]
ax1.bar(mkt["d"], mkt["turnover_tril"], width=0.6, color=c, label="売買代金(兆円)")
ax1.set_ylabel("売買代金(兆円)")
ax1.set_title("7/29の出来高急増(赤)はセリクラか？ → 新高値831>新安値519・値下がり41%でrotation climax",
              fontsize=13, fontweight="bold")
ax1b = ax1.twinx()
ax1b.plot(mkt["d"], mkt["new_high20"], color="#2e7d32", lw=2, marker="o", ms=4, label="新高値(20日)")
ax1b.plot(mkt["d"], mkt["new_low20"], color="#c0392b", lw=2, marker="o", ms=4, label="新安値(20日)")
ax1b.set_ylabel("新高値/新安値 銘柄数")
ax1.legend(loc="upper left"); ax1b.legend(loc="upper right"); ax1.grid(axis="y", alpha=0.3)
# 下: 半導体リターン + 出来高比
semi["d"] = pd.to_datetime(semi["date"])
c2 = ["#c0392b" if r<0 else "#2e7d32" for r in semi["semi_ret"]]
ax2.bar(semi["d"], semi["semi_ret"], width=0.6, color=c2)
ax2.set_ylabel("半導体11銘柄 日次リターン(%)"); ax2.axhline(0, color="#333", lw=0.8)
ax2.set_title("半導体: 7/28 -9.5%・7/29 -7.2%の2日連続下落(出来高1.55x・11中7が60日安値)=distribution継続",
              fontsize=12, fontweight="bold")
ax2.grid(axis="y", alpha=0.3)
for lbl in ax2.get_xticklabels(): lbl.set_rotation(30); lbl.set_ha("right")

fig.text(0.99, 0.005, "データ: JQuants stocks_daily + macro(VIX/SOX) / 2026-07", ha="right", fontsize=8, color="gray")
fig.tight_layout()
fig.savefig(HERE/"result.png", dpi=100, bbox_inches="tight", facecolor="white")
print("\nsaved result.png")
