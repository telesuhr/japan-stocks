"""台湾外国人フロー(T86) → 翌日 日本半導体 超過リターン リードラグ検証。

仮説: 台湾の外国人ネット買い(日t, 大引け後14:30JST公表) は日本半導体の超過リターン(日t+1)を先行予測。
反証: SOXオーバーナイトへの同時反応の代理にすぎない → SOX制御で独自寄与を確認(教訓1,3)。
タイミング: 日本D 寄→引(intraday, ルックアヘッド無し), シグナルは tw_date < D の最新。コスト往復10bp(教訓2)。
"""
import os, sys
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
import pandas as pd
import psycopg2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PG = {
    "host": os.environ.get("PGHOST", "localhost"),
    "port": int(os.environ.get("PGPORT", 5432)),
    "user": os.environ.get("PGUSER", "postgres"),
    "password": os.environ.get("PGPASSWORD", "postgres"),
    "dbname": os.environ.get("PGDATABASE", "market_data"),
}
conn = psycopg2.connect(**PG)

JP_SEMIS = ["80350","68570","61460","69200","77350","77290","67230","69630","40630","34360","40620"]
TW_BASKET = ["2330","2454","2379","3034","3711","2303"]   # TSMC, MediaTek, Realtek, Novatek, ASE, UMC
TW_PRIMARY = "2330"
START = "2022-10-01"
COST_RT = 0.0010          # 中立ロング往復コスト 10bp (バスケット~4 + TOPIXヘッジ~2 + スリッページ)
OOS_START = pd.Timestamp("2025-01-01")

# ---------- データ取得 ----------
jp = pd.read_sql("SELECT code,date,adj_open,adj_close FROM stocks_daily WHERE code=ANY(%s) AND date>=%s",
                 conn, params=[JP_SEMIS, START])
top = pd.read_sql("SELECT date,open,close FROM index_daily WHERE code='0000' AND date>=%s", conn, params=[START])
tw = pd.read_sql("SELECT trade_date,code,foreign_net FROM macro.tw_foreign_flow WHERE code=ANY(%s) AND trade_date>=%s",
                 conn, params=[TW_BASKET, START])
sox = pd.read_sql("SELECT trade_date,close FROM macro.daily_ohlcv WHERE symbol='.SOX' AND trade_date>=%s",
                  conn, params=[START])
conn.close()

for d in (jp, top, tw, sox):
    c = "date" if "date" in d.columns else "trade_date"
    d[c] = pd.to_datetime(d[c])

# ---------- 日本半導体バスケット 寄→引 超過 ----------
jp["oc"] = jp["adj_close"].astype(float) / jp["adj_open"].astype(float) - 1.0
basket = jp.groupby("date")["oc"].mean().rename("basket_oc")             # 等加重
top = top.set_index("date").sort_index()
top["topix_oc"] = top["close"].astype(float) / top["open"].astype(float) - 1.0
panel = pd.concat([basket, top["topix_oc"]], axis=1).dropna()
panel["excess"] = panel["basket_oc"] - panel["topix_oc"]                 # 市場中立 寄→引 超過
panel["excess_lag"] = panel["excess"].shift(1)                          # モメンタム制御用
panel = panel.reset_index().rename(columns={"index": "date"})

# ---------- 台湾シグナル: 外国人ネットの 60d z-score ----------
tw = tw.sort_values(["code","trade_date"])
def zscore(g):
    m = g["foreign_net"].rolling(60, min_periods=30).mean()
    s = g["foreign_net"].rolling(60, min_periods=30).std()
    g["z"] = (g["foreign_net"] - m) / s
    return g
tw = tw.groupby("code", group_keys=False).apply(zscore)
tsmc_z = tw[tw["code"] == TW_PRIMARY][["trade_date","z"]].rename(columns={"z":"tsmc_z"})
basket_z = tw.groupby("trade_date")["z"].mean().reset_index().rename(columns={"z":"basket_z"})
tw_sig = tsmc_z.merge(basket_z, on="trade_date", how="outer").sort_values("trade_date").dropna()

# ---------- SOX オーバーナイト(終値前日比) ----------
sox = sox.sort_values("trade_date")
sox["sox_ret"] = sox["close"].astype(float).pct_change()
sox_sig = sox[["trade_date","sox_ret"]].dropna()

# ---------- ルックアヘッド排除マージ (tw_date < D, sox_date < D) ----------
panel = panel.sort_values("date")
m = pd.merge_asof(panel, tw_sig.sort_values("trade_date"),
                  left_on="date", right_on="trade_date", direction="backward", allow_exact_matches=False)
m = pd.merge_asof(m.sort_values("date"), sox_sig.sort_values("trade_date"),
                  left_on="date", right_on="trade_date", direction="backward", allow_exact_matches=False,
                  suffixes=("", "_sox"))
m = m.dropna(subset=["excess","tsmc_z","sox_ret","excess_lag"]).reset_index(drop=True)
print(f"panel n={len(m)}  {m['date'].min().date()} ~ {m['date'].max().date()}")

# ---------- 1) 単変量: z五分位ごとの翌日超過 ----------
print("\n=== TSMC z 五分位 → 翌日(寄→引)超過リターン ===")
m["q"] = pd.qcut(m["tsmc_z"], 5, labels=[1,2,3,4,5])
qt = m.groupby("q")["excess"].agg(["mean","count"])
qt["mean_bp"] = qt["mean"] * 1e4
print(qt[["mean_bp","count"]].round(2).to_string())
spread = (qt.loc[5,"mean"] - qt.loc[1,"mean"]) * 1e4
corr = m["tsmc_z"].corr(m["excess"])
print(f"Q5-Q1 spread = {spread:.2f} bp   corr(z, excess) = {corr:.4f}")

# ---------- 2) 多変量 OLS (SOX・モメンタム制御) ----------
def ols(y, X, names):
    X = np.column_stack([np.ones(len(X))] + [X[c].values for c in X.columns])
    names = ["const"] + names
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    dof = len(y) - X.shape[1]
    s2 = (e @ e) / dof
    cov = s2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    t = b / se
    r2 = 1 - (e @ e) / (((y - y.mean()) ** 2).sum())
    return pd.DataFrame({"coef": b, "t": t}, index=names), r2

print("\n=== OLS: excess[D] ~ TSMC_z + SOX_overnight + excess[D-1] ===")
res, r2 = ols(m["excess"].values,
              m[["tsmc_z","sox_ret","excess_lag"]].assign(tsmc_z=m["tsmc_z"]),
              ["tsmc_z","sox_ret","excess_lag"])
# 係数はbp表示 (tsmc_z は1標準偏差あたりのbp)
print(res.assign(coef_bp=res["coef"] * 1e4)[["coef_bp","t"]].round(3).to_string())
print(f"R^2 = {r2:.4f}")

# ---------- 3) 戦略: z上位ターシル → 中立ロング 寄→引 (コスト込み, IS/OOS) ----------
def strat_stats(df, thr):
    pos = (df["tsmc_z"] >= thr).astype(int)
    gross = pos * df["excess"]
    net = gross - pos * COST_RT
    traded = net[pos == 1]
    if len(traded) == 0:
        return None
    ann = traded.mean() * 252
    sharpe = (traded.mean() / traded.std()) * np.sqrt(252) if traded.std() > 0 else np.nan
    return dict(n=int(pos.sum()), gross_bp=gross[pos==1].mean()*1e4, net_bp=traded.mean()*1e4,
                ann_pct=ann*100, sharpe=sharpe, hit=(traded > 0).mean()*100)

thr = m["tsmc_z"].quantile(2/3)   # 上位ターシル閾値(全期間)
print(f"\n=== 戦略: TSMC_z >= {thr:.2f}(上位1/3) で 半導体バスケット−TOPIX 寄→引, 往復{COST_RT*1e4:.0f}bp ===")
for label, df in [("全期間", m), ("IS(〜2024)", m[m["date"] < OOS_START]), ("OOS(2025〜)", m[m["date"] >= OOS_START])]:
    s = strat_stats(df, thr)
    if s:
        print(f"  {label:12} n={s['n']:4d}  gross={s['gross_bp']:6.2f}bp  net={s['net_bp']:6.2f}bp  "
              f"年率={s['ann_pct']:5.1f}%  Sharpe={s['sharpe']:4.2f}  勝率={s['hit']:4.1f}%")
# ベースライン: 無条件 中立ロング(毎日)
base_net = (m["excess"] - COST_RT)
print(f"  {'ベースライン毎日':12} n={len(m):4d}  net={base_net.mean()*1e4:6.2f}bp  "
      f"年率={base_net.mean()*252*100:5.1f}%  Sharpe={base_net.mean()/base_net.std()*np.sqrt(252):4.2f}")

# ---------- 可視化 ----------
try:
    import matplotlib.font_manager as fm
    fm.fontManager.addfont("/root/.fonts/NotoSansJP.ttf")
    plt.rcParams["font.family"] = "Noto Sans JP"
    plt.rcParams["axes.unicode_minus"] = False
except Exception:
    pass
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2))

# 五分位バー
colors = ["#c0392b","#e67e22","#95a5a6","#2980b9","#27ae60"]
ax1.bar(qt.index.astype(int), qt["mean_bp"], color=colors)
ax1.axhline(0, color="k", lw=0.8)
ax1.set_xlabel("TSMC 外国人ネット z-score 五分位 (1=売り … 5=買い)")
ax1.set_ylabel("翌日 半導体超過リターン (bp, 寄→引)")
ax1.set_title(f"台湾外国人フロー → 翌日 日本半導体超過\nQ5-Q1={spread:.1f}bp  corr={corr:.3f}  n={len(m)}")

# 累積ネットP&L
m2 = m.copy()
pos = (m2["tsmc_z"] >= thr).astype(int)
m2["strat_net"] = pos * m2["excess"] - pos * COST_RT
m2["base_net"] = m2["excess"] - COST_RT
ax2.plot(m2["date"], (m2["strat_net"]).cumsum()*100, label="戦略(z上位1/3で中立ロング)", color="#27ae60", lw=1.6)
ax2.plot(m2["date"], (m2["base_net"]).cumsum()*100, label="毎日 中立ロング", color="#95a5a6", lw=1.2)
ax2.axvline(OOS_START, color="red", ls="--", lw=1, alpha=0.7)
ax2.text(OOS_START, ax2.get_ylim()[1], " OOS", color="red", va="top", fontsize=9)
ax2.axhline(0, color="k", lw=0.8)
ax2.set_ylabel("累積ネットリターン (%, コスト込)")
ax2.set_title("累積P&L (寄→引・往復10bp)")
ax2.legend(fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(os.path.dirname(__file__), "result.png"), dpi=100, bbox_inches="tight")
print("\nsaved result.png")
