#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
watchlist ダッシュボードの多角テクニカル・スクリーンが将来リターンを生むかを検証。

仮説(教訓5: 仮説先行):
  「今日の注目銘柄」が抽出する各テクニカル条件(52週高値ブレイク/出来高急増/GC/ギャップアップ/
   押し目/強モメンタム)は、翌営業日寄りで買って数日保有したとき、
   同日の流動性ユニバース平均(=ただ流動株をロングする)を上回る超過リターンを残すか?
   newlow(52週安値) は逆に下振れ(空売り側のエッジ)があるか?

手法:
  - ユニバース: 各日 point-in-time で「直近20日平均売買代金 >= 1億円」(watchlist と同じ)
  - シグナルは day T の調整後株価から計算(先読み無し)
  - entry = T+1 寄り(adj_open)、exit = T+h 引け(adj_close)、h∈{1,5,10,20}
  - ベンチ: 同じ entry/exit 日のユニバース等加重平均リターン → abnormal = 個別 − ユニバース平均
  - コスト: long-only 往復 10bp(片道5bp)を控除(教訓2)
  - 統計: シグナル日ごとに等加重ポートフォリオの平均超過を取り、その日次系列で平均・t値
          (同日多発のクラスタを考慮し pseudo-replication を回避)
  - IS(<2025-01-01) / OOS(>=2025-01-01) 分割で過学習チェック(教訓1: 同時点相関≠予測 を回避するため必ず forward)
"""
import sys, os
sys.stdout.reconfigure(line_buffering=True)
import numpy as np, pandas as pd, psycopg2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 日本語フォント
for fp in ["/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
           "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
           "/mnt/c/Windows/Fonts/meiryo.ttc", "/mnt/c/Windows/Fonts/YuGothM.ttc"]:
    if os.path.exists(fp):
        fm.fontManager.addfont(fp); plt.rcParams["font.family"] = fm.FontProperties(fname=fp).get_name(); break
plt.rcParams["axes.unicode_minus"] = False

PG = {"host": os.environ.get("PGHOST", "localhost"), "port": int(os.environ.get("PGPORT", 5432)),
      "user": os.environ.get("PGUSER", "postgres"), "dbname": os.environ.get("PGDATABASE", "market_data")}

LIQ = 1e8           # 売買代金 1億円/日
COST = 0.001        # 往復 10bp
OOS_START = "2025-01-01"
HORIZONS = [1, 5, 10, 20]
START = "2022-07-01"  # 200MA ウォームアップ込み
HERE = os.path.dirname(os.path.abspath(__file__))

print("== データ取得 ==")
conn = psycopg2.connect(**PG)
sql = """
WITH days AS (SELECT DISTINCT date FROM stocks_daily WHERE date>=%s),
liq AS (
  SELECT code FROM stocks_daily
  WHERE date >= (SELECT max(date)-interval '40 days' FROM stocks_daily)
  GROUP BY code HAVING avg(turnover_value) >= %s
)
SELECT s.code, s.date, s.adj_open o, s.adj_high h, s.adj_low l, s.adj_close c,
       s.volume v, s.turnover_value tv
FROM stocks_daily s
WHERE s.date >= %s AND s.code IN (SELECT code FROM liq)
ORDER BY s.code, s.date
"""
# 注: liq は「現在の流動株」での母集団確定だが、エントリ可否は各日の point-in-time turnover で再判定する
df = pd.read_sql(sql, conn, params=(START, LIQ, START))
conn.close()
df["date"] = pd.to_datetime(df["date"])
for cc in ["o", "h", "l", "c", "tv", "v"]:
    df[cc] = df[cc].astype(float)
print(f"  rows={len(df):,}  codes={df['code'].nunique():,}  期間 {df['date'].min()}..{df['date'].max()}")

print("== 特徴量・シグナル計算 ==")
def feat(x):
    x = x.sort_values("date")
    c, h, l, v = x["c"], x["h"], x["l"], x["v"]
    x["ma25"] = c.rolling(25).mean(); x["ma75"] = c.rolling(75).mean()
    x["ma200"] = c.rolling(200, min_periods=120).mean()
    x["vol20"] = v.rolling(20).mean()
    x["hi52"] = h.rolling(250, min_periods=60).max()
    x["lo52"] = l.rolling(250, min_periods=60).min()
    d = c.diff(); up = d.clip(lower=0); dn = (-d).clip(lower=0)
    x["rsi"] = 100 - 100 / (1 + up.rolling(14).mean() / dn.rolling(14).mean())
    x["ret5"] = c / c.shift(5) - 1
    x["ret20"] = c / c.shift(20) - 1
    x["vr"] = v / x["vol20"]
    x["gap"] = x["o"] / c.shift(1) - 1
    x["ma25p"] = x["ma25"].shift(1); x["ma75p"] = x["ma75"].shift(1)
    x["liq20"] = x["tv"].rolling(20).mean()
    # 将来リターン: entry=T+1 open, exit=T+h close
    nopen = x["o"].shift(-1)
    for hh in HORIZONS:
        x[f"ret_{hh}"] = x["c"].shift(-hh) / nopen - 1
    return x

df = pd.concat([feat(x) for _, x in df.groupby("code", sort=False)], ignore_index=True)

# シグナル定義(watchlist と同一ロジック)
sig = {
    "newhigh":     (df["c"] >= df["hi52"] * 0.999),
    "volsurge":    (df["vr"] >= 2) & (df["ret5"] > 0) & (df["c"] > df["ma25"]),
    "goldencross": (df["ma25"] > df["ma75"]) & (df["ma25p"] <= df["ma75p"]),
    "gapup":       (df["gap"] >= 0.03) & (df["c"] >= df["o"]),
    "pullback":    (df["c"] > df["ma200"]) & (df["ma25"] > df["ma75"]) & (df["rsi"] <= 35),
    "momentum":    (df["c"] > df["ma25"]) & (df["ma25"] > df["ma75"]) & (df["ret20"] > 0.05),
    "newlow":      (df["c"] <= df["lo52"] * 1.001),
}

# エントリ可能母集団 = point-in-time 流動性
elig = df["liq20"] >= LIQ
df = df[elig].copy()
for k in sig:
    sig[k] = sig[k][elig]

# ユニバース平均(各シグナル日 T の forward を等加重平均) → ベンチ
mkt = {hh: df.groupby("date")[f"ret_{hh}"].transform("mean") for hh in HORIZONS}

print("== 集計 ==")
rows = []
def agg(mask, name):
    for hh in HORIZONS:
        r = df[f"ret_{hh}"]
        m = mask & r.notna()
        sub = pd.DataFrame({"date": df["date"][m], "raw": r[m], "abn": (r - mkt[hh])[m]})
        if len(sub) == 0:
            continue
        for split, dd in [("全", sub),
                          ("IS", sub[sub["date"] < pd.Timestamp(OOS_START)]),
                          ("OOS", sub[sub["date"] >= pd.Timestamp(OOS_START)])]:
            if len(dd) < 20:
                rows.append(dict(sig=name, h=hh, split=split, n=len(dd), ndays=dd["date"].nunique(),
                                 raw_bp=np.nan, abn_bp=np.nan, t=np.nan, hit=np.nan, net_abn_bp=np.nan))
                continue
            # 日次ポートフォリオ(等加重)→日次系列でt値(クラスタ考慮)
            daily = dd.groupby("date")["abn"].mean()
            t = daily.mean() / (daily.std(ddof=1) / np.sqrt(len(daily))) if daily.std(ddof=1) > 0 else np.nan
            rows.append(dict(
                sig=name, h=hh, split=split, n=len(dd), ndays=int(daily.shape[0]),
                raw_bp=dd["raw"].mean() * 1e4,
                abn_bp=dd["abn"].mean() * 1e4,
                t=t, hit=(dd["abn"] > 0).mean(),
                net_abn_bp=(dd["abn"].mean() - COST) * 1e4,
            ))

for k, m in sig.items():
    agg(m, k)

res = pd.DataFrame(rows)
res.to_csv(os.path.join(HERE, "results.csv"), index=False)

# 表示
pd.set_option("display.width", 160, "display.max_rows", 300)
def fmt(v, d=1):
    return "" if pd.isna(v) else f"{v:.{d}f}"
print("\n=== 結果: シグナル別 超過リターン(ユニバース等加重対比) ===")
print(f"{'sig':12}{'h':>3}{'split':>5}{'n':>6}{'days':>5}{'raw_bp':>9}{'abn_bp':>9}{'t':>7}{'hit':>6}{'net_abn_bp':>11}")
for _, r in res.iterrows():
    print(f"{r['sig']:12}{int(r['h']):>3}{r['split']:>5}{int(r['n']):>6}{int(r['ndays']):>5}"
          f"{fmt(r['raw_bp']):>9}{fmt(r['abn_bp']):>9}{fmt(r['t'],2):>7}{fmt(r['hit'],2):>6}{fmt(r['net_abn_bp']):>11}")

# 可視化: h=5 の超過リターン累積(日次平均abnの累積和) シグナル別
print("\n== 可視化 ==")
fig, ax = plt.subplots(figsize=(12, 6.75), dpi=100)
hh = 5
for k, m in sig.items():
    r = df[f"ret_{hh}"]; mm = m & r.notna()
    sub = pd.DataFrame({"date": df["date"][mm], "abn": (r - mkt[hh])[mm]})
    if len(sub) < 20:
        continue
    daily = sub.groupby("date")["abn"].mean().sort_index()
    ax.plot(daily.index, (daily.cumsum()) * 100, label=f"{k} (n={len(sub)})", lw=1.4)
ax.axvline(pd.Timestamp(OOS_START), color="gray", ls="--", lw=1)
ax.text(pd.Timestamp(OOS_START), ax.get_ylim()[1], " OOS→", color="gray", va="top", fontsize=9)
ax.axhline(0, color="black", lw=0.6)
ax.set_title(f"テクニカル・スクリーン別 累積超過リターン (h={hh}日・ユニバース対比・コスト前)", fontsize=13)
ax.set_ylabel("累積 超過リターン (%・日次平均の累積和)")
ax.legend(fontsize=9, ncol=2); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "result.png")); print("  saved result.png")
print("done")
