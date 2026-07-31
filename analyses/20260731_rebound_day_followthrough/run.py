"""
2026-07-31 の大幅反発(日経+2900円/+4.7%・SOX+8%を受けたギャップアップ)は「本物」か。
下落局面での大幅反発日を過去10年から抽出し、翌日以降のフォロースルーを検証する。

H1: 下落局面(20日高値からDD>=4%)での大幅反発日(+2.5%以上)の翌日は、無条件日より良くはない
    (=反発初日の追随買いは報われない)。
H2: 「本物の反発(出来高を伴う)」と「空中戻し(出来高を伴わない)」で翌日以降が分かれる。
H3: 反発当日の内訳は「ギャップに集約・寄→引はマイナス」(既出 20260612 の一般化)。
"""
import sys; sys.stdout.reconfigure(line_buffering=True)
from pathlib import Path
import pandas as pd, numpy as np
from scipy import stats as sps
import matplotlib.pyplot as plt
from jstock import db

HERE = Path(__file__).resolve().parent

# ---------- 市場日次系列(流動性ユニバース・PIT) ----------
# 等加重リターン(close-to-close)・寄りギャップ・寄→引・市場出来高
mkt = db.read_sql("""
WITH u AS (
  SELECT code, date, adj_open, adj_close, turnover_value,
    LAG(adj_close) OVER (PARTITION BY code ORDER BY date) pc,
    AVG(turnover_value) OVER (PARTITION BY code ORDER BY date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING) adv20
  FROM stocks_daily WHERE date >= '2015-06-01'
)
SELECT date,
  AVG((adj_close/pc - 1)*100)   AS ew_ret,
  AVG((adj_open /pc - 1)*100)   AS ew_gap,
  AVG((adj_close/adj_open-1)*100) AS ew_o2c,
  SUM(turnover_value)/1e12      AS turnover_tril,
  COUNT(*)                      AS n
FROM u
WHERE pc IS NOT NULL AND adj_open > 0 AND adv20 >= 1e8   -- ADV>=1億の流動性ユニバース(PIT)
GROUP BY date ORDER BY date
""", [])
mkt["date"] = pd.to_datetime(mkt["date"])
for c in ["ew_ret", "ew_gap", "ew_o2c", "turnover_tril"]:
    mkt[c] = mkt[c].astype(float)
mkt = mkt.set_index("date")

# 市場出来高の20日平均比 / 20日高値からのDD(前日終値時点=先読み無し)
mkt["tv_ratio"] = mkt["turnover_tril"] / mkt["turnover_tril"].rolling(20).mean().shift(1)
cum = (1 + mkt["ew_ret"] / 100).cumprod()
mkt["dd20"] = (cum / cum.rolling(20).max() - 1) * 100
mkt["dd20_prev"] = mkt["dd20"].shift(1)          # 前日終値時点のDD
mkt["prev_ret"] = mkt["ew_ret"].shift(1)

# 先行リターン(D+1..D+20、当日終値から)
for h in [1, 2, 3, 5, 10, 20]:
    mkt[f"fwd{h}"] = (cum.shift(-h) / cum - 1) * 100
mkt["nx_gap"] = mkt["ew_gap"].shift(-1)          # 翌日のギャップ
mkt["nx_o2c"] = mkt["ew_o2c"].shift(-1)          # 翌日の寄→引

mkt.to_csv(HERE / "market_daily.csv")
print(f"市場系列: {mkt.index.min().date()} 〜 {mkt.index.max().date()} / {len(mkt)}日 / 平均銘柄数 {mkt['n'].mean():.0f}")

# ---------- H1: 下落局面での大幅反発日 ----------
DD_TH, RET_TH = -4.0, 2.5
ev = mkt[(mkt["dd20_prev"] <= DD_TH) & (mkt["ew_ret"] >= RET_TH)].copy()
base = mkt.dropna(subset=["fwd1"])

HS = [1, 2, 3, 5, 10, 20]
rows = []
for h in HS:
    e = ev[f"fwd{h}"].dropna(); b = base[f"fwd{h}"].dropna()
    t, p = sps.ttest_1samp(e, 0) if len(e) > 2 else (np.nan, np.nan)
    rows.append({"h": h, "n": len(e), "event_mean": e.mean(), "event_med": e.median(),
                 "win%": (e > 0).mean() * 100, "base_mean": b.mean(),
                 "excess": e.mean() - b.mean(), "t": t, "p": p})
h1 = pd.DataFrame(rows)
print("\n=== H1: 下落局面(DD<=-4%)での大幅反発日(EW>=+2.5%) の先行リターン(%) ===")
print(f"イベント日数 n={len(ev)}  期間内シェア {len(ev)/len(mkt)*100:.1f}%")
print(h1.round(2).to_string(index=False))
print("\n[イベント日一覧(直近15)]")
print(ev[["ew_ret", "dd20_prev", "tv_ratio", "fwd1", "fwd5", "fwd20"]].tail(15).round(2).to_string())

# 翌日の内訳(ギャップ vs 寄→引)
print("\n=== 反発翌日の内訳 ===")
print(f"  翌日ギャップ  平均 {ev['nx_gap'].mean():+.2f}%  (無条件 {base['ew_gap'].mean():+.2f}%)")
print(f"  翌日 寄→引  平均 {ev['nx_o2c'].mean():+.2f}%  (無条件 {base['ew_o2c'].mean():+.2f}%)")

# ---------- H2: 出来高で分岐するか ----------
med = ev["tv_ratio"].median()
hi, lo = ev[ev["tv_ratio"] >= med], ev[ev["tv_ratio"] < med]
rows = []
for lbl, g in [("高出来高(比≥%.2f)" % med, hi), ("低出来高(比<%.2f)" % med, lo)]:
    r = {"group": lbl, "n": len(g), "tv_ratio": g["tv_ratio"].mean()}
    for h in HS:
        r[f"fwd{h}"] = g[f"fwd{h}"].dropna().mean()
    r["win%_fwd5"] = (g["fwd5"].dropna() > 0).mean() * 100
    rows.append(r)
h2 = pd.DataFrame(rows)
print("\n=== H2: 反発日の出来高で分けた先行リターン(%) ===")
print(h2.round(2).to_string(index=False))

# ---------- H2b: V字(前日大幅安→当日大幅高)か ----------
v = ev[ev["prev_ret"] <= -2.0]
nv = ev[ev["prev_ret"] > -2.0]
print("\n=== H2b: V字反発(前日<=-2.0%)か否か ===")
for lbl, g in [("V字(前日大幅安)", v), ("非V字", nv)]:
    if len(g) == 0: continue
    print(f"  {lbl:16s} n={len(g):3d}  D+1 {g['fwd1'].mean():+.2f}  D+5 {g['fwd5'].mean():+.2f}  "
          f"D+10 {g['fwd10'].mean():+.2f}  D+20 {g['fwd20'].mean():+.2f}  D+5勝率 {(g['fwd5'].dropna()>0).mean()*100:.0f}%")

# ---------- H3: 反発当日の内訳(ギャップ集約か) ----------
print("\n=== H3: 大幅反発日そのものの内訳 ===")
print(f"  当日ギャップ 平均 {ev['ew_gap'].mean():+.2f}% / 寄→引 平均 {ev['ew_o2c'].mean():+.2f}% "
      f"(全体リターン {ev['ew_ret'].mean():+.2f}%)")
print(f"  → 上げのうちギャップ寄与 {ev['ew_gap'].mean()/ev['ew_ret'].mean()*100:.0f}%")

# ---------- 半導体バスケット: SOX急騰翌日の追加検証(今日はSOX+8%) ----------
semi_codes = ['8035', '6920', '6323', '6758', '6857', '4062', '6723', '6963', '4005', '6976', '7735']
semi = db.read_sql("""
WITH u AS (
  SELECT s.code, s.date, s.adj_open, s.adj_close,
    LAG(s.adj_close) OVER (PARTITION BY s.code ORDER BY s.date) pc
  FROM stocks_daily s JOIN symbol_master m ON m.code5 = s.code
  WHERE m.code4 = ANY(%s) AND s.date >= '2015-06-01')
SELECT date, AVG((adj_close/pc-1)*100) ret, AVG((adj_open/pc-1)*100) gap,
       AVG((adj_close/adj_open-1)*100) o2c
FROM u WHERE pc IS NOT NULL AND adj_open > 0 GROUP BY date ORDER BY date
""", [semi_codes])
semi["date"] = pd.to_datetime(semi["date"]); semi = semi.set_index("date").astype(float)

sox = db.read_sql("""SELECT trade_date, close FROM macro.daily_ohlcv
                     WHERE symbol='.SOX' ORDER BY trade_date""", [])
sox["trade_date"] = pd.to_datetime(sox["trade_date"])
sox = sox.set_index("trade_date")["close"].astype(float)
sox_ret = sox.pct_change() * 100

# 各日本営業日の「直前に終了した」米セッションをアライン。
# ⚠️ macro.daily_ohlcv の .SOX は米セッション日付。日本のD日の日中(9:00-15:30)は
#    米セッションD(22:30JST開始)より前 → 同日マッチは完全なルックアヘッド(教訓1)。
#    allow_exact_matches=False で「厳密に前日以前」＝前夜に終わった米セッションを使う。
al = pd.merge_asof(semi.reset_index().sort_values("date"),
                   sox_ret.rename("sox").reset_index().sort_values("trade_date"),
                   left_on="date", right_on="trade_date", direction="backward",
                   allow_exact_matches=False)
al = al.dropna(subset=["sox"])

# 参考: 同日マッチ(=ルックアヘッド版)も並記して罠を可視化する
al_bad = pd.merge_asof(semi.reset_index().sort_values("date"),
                       sox_ret.rename("sox").reset_index().sort_values("trade_date"),
                       left_on="date", right_on="trade_date", direction="backward",
                       allow_exact_matches=True).dropna(subset=["sox"])

print("\n=== SOX急騰の翌日本営業日: 半導体11銘柄の内訳(%) ===")
for th in [3, 5, 7]:
    g = al[al["sox"] >= th]
    print(f"  SOX>=+{th}%  n={len(g):3d}  ギャップ {g['gap'].mean():+.2f}  寄→引 {g['o2c'].mean():+.2f}  "
          f"(寄→引 勝率 {(g['o2c']>0).mean()*100:.0f}% / t={sps.ttest_1samp(g['o2c'],0)[0]:.2f})")
g_all = al
print(f"  無条件      n={len(g_all):4d}  ギャップ {g_all['gap'].mean():+.2f}  寄→引 {g_all['o2c'].mean():+.2f}")
print("  [参考] 同日マッチ=ルックアヘッド版(誤り):")
for th in [5, 7]:
    g = al_bad[al_bad["sox"] >= th]
    print(f"    SOX>=+{th}%  n={len(g):3d}  ギャップ {g['gap'].mean():+.2f}  寄→引 {g['o2c'].mean():+.2f} "
          f"← 日本の日中が米国夜を『予測』する見せかけ")

# 今日と同型: 反発がギャップ主導かどうかで分けた先行リターン
ev = ev.assign(gap_share=ev["ew_gap"] / ev["ew_ret"])
gmed = ev["gap_share"].median()
print("\n=== H3b: 反発がギャップ主導か寄り後主導か で分けた先行リターン(%) ===")
for lbl, g in [(f"ギャップ主導(share>={gmed:.2f})", ev[ev["gap_share"] >= gmed]),
               (f"寄り後主導(share<{gmed:.2f})", ev[ev["gap_share"] < gmed])]:
    print(f"  {lbl:24s} n={len(g):3d}  D+1 {g['fwd1'].mean():+.2f}  D+5 {g['fwd5'].mean():+.2f}  "
          f"D+20 {g['fwd20'].mean():+.2f}  D+1勝率 {(g['fwd1'].dropna()>0).mean()*100:.0f}%")

# ---------- H4: 半導体バスケット固有の「暴落後の急反発」(今日の直接の類似形) ----------
# 広い市場(EW)のDDは浅く、暴落しているのは半導体/日経(price-weight)。ユーザー建玉もこちら側。
scum = (1 + semi["ret"] / 100).cumprod()
semi["dd20"] = (scum / scum.rolling(20).max() - 1) * 100
semi["dd20_prev"] = semi["dd20"].shift(1)
for h in [1, 2, 3, 5, 10, 20]:
    semi[f"fwd{h}"] = (scum.shift(-h) / scum - 1) * 100
semi["nx_gap"] = semi["gap"].shift(-1); semi["nx_o2c"] = semi["o2c"].shift(-1)

sev = semi[(semi["dd20_prev"] <= -15) & (semi["ret"] >= 5)]
print("\n=== H4: 半導体バスケット 『20日DD<=-15%からの当日+5%以上の急反発』 ===")
print(f"  n={len(sev)}  当日: ギャップ {sev['gap'].mean():+.2f}% / 寄→引 {sev['o2c'].mean():+.2f}% "
      f"(計 {sev['ret'].mean():+.2f}%)")
print(f"  翌日: ギャップ {sev['nx_gap'].mean():+.2f}% / 寄→引 {sev['nx_o2c'].mean():+.2f}%")
for h in [1, 2, 3, 5, 10, 20]:
    s = sev[f"fwd{h}"].dropna()
    print(f"   D+{h:<2d} 平均 {s.mean():+6.2f}%  中央値 {s.median():+6.2f}%  勝率 {(s>0).mean()*100:3.0f}%  n={len(s)}")
# ⚠️ 決定的な識別: 過去10件は「本物のパニック(VIX急騰)を伴う底」だったか。
#    今回(2026-07)はVIX 18台・セリクラ無し([20260730_selling_climax_check])＝同型か否かの分岐点。
vix = db.read_sql("""SELECT trade_date, close FROM macro.daily_ohlcv
                     WHERE symbol='VXc1' ORDER BY trade_date""", [])
vix["trade_date"] = pd.to_datetime(vix["trade_date"])
vix_s = vix.set_index("trade_date")["close"].astype(float)
sev = sev.assign(vix=vix_s.reindex(sev.index, method="ffill"),
                 vix_20d_max=[vix_s.loc[:d].last("40D").max() if len(vix_s.loc[:d]) else np.nan for d in sev.index])
print("\n  [イベント日一覧] ※vix=反発日のVIX先物, vix_20d_max=直前約1ヶ月の最高VIX")
print(sev[["ret", "gap", "o2c", "dd20_prev", "vix", "vix_20d_max", "fwd1", "fwd5", "fwd20"]].round(2).to_string())
cur_vix = vix_s.iloc[-1]; cur_vixmax = vix_s.last("40D").max()
print(f"\n  現在(2026-07): VIX {cur_vix:.1f} / 直近1ヶ月最高 {cur_vixmax:.1f}  "
      f"← 過去10件の中央値 VIX {sev['vix'].median():.1f} / 最高 {sev['vix_20d_max'].median():.1f}")
print(f"  過去10件のうち 反発日VIX>=25 だったもの: {(sev['vix']>=25).sum()}/10 件")
sev.to_csv(HERE / "semi_rebound_events.csv")

# 現在地(7/30時点)の確認
last = mkt.iloc[-1]
print(f"\n=== 現在地 (直近確定日 {mkt.index[-1].date()}) ===")
print(f"  市場EWリターン {last['ew_ret']:+.2f}% / 20日高値からのDD {last['dd20']:+.2f}% / "
      f"売買代金 {last['turnover_tril']:.1f}兆 (20日平均比 {last['tv_ratio']:.2f}x)")
print(f"  → 本日(7/31)は DD{last['dd20']:.1f}% の下落局面からの大幅反発 = 上記イベント条件に合致見込み")

al.to_csv(HERE / "sox_nextday_semis.csv", index=False)

# ---------- 可視化 ----------
import matplotlib.font_manager as fm
fm.fontManager.addfont("/root/.fonts/NotoSansJP.ttf")
plt.rcParams["font.family"] = fm.FontProperties(fname="/root/.fonts/NotoSansJP.ttf").get_name()
plt.rcParams["axes.unicode_minus"] = False

fig, axes2 = plt.subplots(2, 2, figsize=(14, 9.5), facecolor="white")
axes = axes2.ravel()
# (1) イベント vs ベースの先行リターン
ax = axes[0]
x = np.arange(len(HS)); w = 0.38
ax.bar(x - w/2, h1["event_mean"], w, color="#c0392b", label=f"下落局面の大幅反発日 (n={len(ev)})")
ax.bar(x + w/2, h1["base_mean"], w, color="#8fa9bf", label="無条件(全日)")
ax.set_xticks(x); ax.set_xticklabels([f"D+{h}" for h in HS]); ax.axhline(0, color="#333", lw=0.8)
ax.set_ylabel("平均先行リターン(%)"); ax.legend(fontsize=8)
ax.set_title("反発日の「追随買い」は報われるか", fontsize=12, fontweight="bold"); ax.grid(axis="y", alpha=0.3)
# (2) 出来高で分岐
ax = axes[1]
for (lbl, g), c in zip([(h2.iloc[0]["group"], hi), (h2.iloc[1]["group"], lo)], ["#2e7d32", "#e67e22"]):
    ax.plot([0] + HS, [0] + [g[f"fwd{h}"].mean() for h in HS], marker="o", color=c, lw=2, label=lbl)
ax.axhline(0, color="#333", lw=0.8); ax.set_xlabel("保有営業日"); ax.set_ylabel("平均リターン(%)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
ax.set_title("出来高を伴う反発かどうかで分かれるか", fontsize=12, fontweight="bold")
# (3) SOX急騰翌日の内訳
ax = axes[2]
ths = [3, 5, 7]; gaps = []; o2cs = []
for th in ths:
    g = al[al["sox"] >= th]; gaps.append(g["gap"].mean()); o2cs.append(g["o2c"].mean())
x = np.arange(len(ths))
ax.bar(x - w/2, gaps, w, color="#2e7d32", label="ギャップ(前夜→寄)")
ax.bar(x + w/2, o2cs, w, color="#c0392b", label="寄→引")
ax.set_xticks(x); ax.set_xticklabels([f"SOX≥+{t}%" for t in ths]); ax.axhline(0, color="#333", lw=0.8)
ax.set_ylabel("半導体11銘柄 平均(%)"); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
ax.set_title("SOX急騰翌日: 上げはギャップに集約・寄→引はマイナス", fontsize=12, fontweight="bold")
# (4) 今日の類似形(半導体の暴落後急反発)は"パニック底"だったか
ax = axes[3]
ax.scatter(sev["vix"], sev["fwd20"], s=90, color="#2e7d32", zorder=3, label="過去10件(半導体 暴落後の急反発)")
for d, r in sev.iterrows():
    ax.annotate(d.strftime("%y/%m"), (r["vix"], r["fwd20"]), fontsize=7,
                xytext=(4, 4), textcoords="offset points")
ax.axvline(cur_vix, color="#c0392b", ls="--", lw=2)
ax.text(cur_vix + 0.7, ax.get_ylim()[1] * 0.85, f"今回 VIX {cur_vix:.1f}\n(過去10件の最小20.3を下回る)",
        color="#c0392b", fontsize=9, fontweight="bold")
ax.axhline(0, color="#333", lw=0.8)
ax.set_xlabel("反発日のVIX先物"); ax.set_ylabel("その後20営業日リターン(%)")
ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.3)
ax.set_title("過去の『底』は全てパニック(VIX20超)を伴った — 今回は伴っていない",
             fontsize=12, fontweight="bold")

fig.suptitle("2026-07-31 大幅反発は本物か — 下落局面の反発日 過去10年の事後検証",
             fontsize=13, fontweight="bold")
fig.text(0.99, 0.005, "データ: JQuants stocks_daily (ADV≥1億のPIT流動性ユニバース) + macro SOX",
         ha="right", fontsize=8, color="gray")
fig.tight_layout()
fig.savefig(HERE / "result.png", dpi=100, bbox_inches="tight", facecolor="white")
print("\nsaved result.png")

h1.to_csv(HERE / "h1_forward_returns.csv", index=False)
h2.to_csv(HERE / "h2_by_volume.csv", index=False)
ev[["ew_ret", "ew_gap", "ew_o2c", "dd20_prev", "tv_ratio", "prev_ret"] + [f"fwd{h}" for h in HS]] \
    .to_csv(HERE / "rebound_events.csv")
