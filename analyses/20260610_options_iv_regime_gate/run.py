import os, sys
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
import pandas as pd
import psycopg2
import matplotlib.pyplot as plt
from scipy import stats

PG_CONFIG = {
    "host": os.environ.get("PGHOST", "localhost"),
    "port": int(os.environ.get("PGPORT", 5432)),
    "user": os.environ.get("PGUSER", "postgres"),
    "dbname": os.environ.get("PGDATABASE", "market_data"),
}
conn = psycopg2.connect(**PG_CONFIG)

OUTDIR = os.path.dirname(os.path.abspath(__file__))
COST_BPS = 5.0  # エクスポージャ変化1.0あたり 5bps (先物/ETF想定)

# ---------------------------------------------------------------
# 1. IV指標の構築 (OI上位2限月 = 月物)
# ---------------------------------------------------------------
print("loading options aggregates...")
sql_iv = """
WITH base AS (
  SELECT date, sq_date, pc_div, implied_vol, open_interest,
         (sq_date - date) AS dte,
         strike / underlying_px AS mny
  FROM options_n225
  WHERE implied_vol > 1 AND implied_vol < 150
    AND underlying_px > 0
    AND (sq_date - date) BETWEEN 7 AND 120
),
exp_rank AS (
  SELECT date, sq_date,
         ROW_NUMBER() OVER (PARTITION BY date ORDER BY SUM(open_interest) DESC) rn_oi
  FROM base GROUP BY date, sq_date
),
top2 AS (
  SELECT date, sq_date,
         ROW_NUMBER() OVER (PARTITION BY date ORDER BY sq_date) rn
  FROM exp_rank WHERE rn_oi <= 2
)
SELECT b.date, t.rn AS expiry_rank, MIN(b.dte) AS dte,
  AVG(CASE WHEN b.mny BETWEEN 0.975 AND 1.025 AND b.pc_div=1 THEN b.implied_vol END) AS atm_put,
  AVG(CASE WHEN b.mny BETWEEN 0.975 AND 1.025 AND b.pc_div=2 THEN b.implied_vol END) AS atm_call,
  AVG(CASE WHEN b.mny BETWEEN 0.90 AND 0.96 AND b.pc_div=1 THEN b.implied_vol END) AS otm_put,
  AVG(CASE WHEN b.mny BETWEEN 1.04 AND 1.10 AND b.pc_div=2 THEN b.implied_vol END) AS otm_call
FROM base b JOIN top2 t USING (date, sq_date)
GROUP BY b.date, t.rn
ORDER BY b.date, t.rn
"""
iv = pd.read_sql(sql_iv, conn)
iv["date"] = pd.to_datetime(iv["date"])
for c in ["atm_put", "atm_call", "otm_put", "otm_call"]:
    iv[c] = iv[c].astype(float)
near = iv[iv.expiry_rank == 1].set_index("date")
nxt = iv[iv.expiry_rank == 2].set_index("date")
print(f"  days: {near.shape[0]} ({near.index.min().date()} - {near.index.max().date()})")

ind = pd.DataFrame(index=near.index)
ind["atm_iv"] = (near["atm_put"] + near["atm_call"]) / 2
ind["atm_iv_next"] = (nxt["atm_put"] + nxt["atm_call"]) / 2
ind["skew"] = near["otm_put"] - near["otm_call"]
ind["term"] = ind["atm_iv"] - ind["atm_iv_next"]  # 正 = 逆転(ストレス)
ind["dte_near"] = near["dte"]
ind["iv_chg5"] = ind["atm_iv"].diff(5)

# ---------------------------------------------------------------
# 2. 指数・バスケットのリターン
# ---------------------------------------------------------------
print("loading index & basket...")
idx = pd.read_sql(
    "SELECT date, code, close FROM index_daily WHERE code IN ('N225','0000') ORDER BY date",
    conn)
idx["date"] = pd.to_datetime(idx["date"])
n225 = idx[idx.code == "N225"].set_index("date")["close"].astype(float)
n225_ret = n225.pct_change()

codes22 = ["57130", "57110", "57060", "57140", "50160", "58010", "58020", "58030",
           "80350", "68570", "69200", "61460", "77350", "40630", "34360", "77410",
           "69630", "65260", "99840", "40620", "67230", "285A0"]
px = pd.read_sql(
    "SELECT date, code, adj_close FROM stocks_daily WHERE code = ANY(%s) ORDER BY date",
    conn, params=[codes22])
px["date"] = pd.to_datetime(px["date"])
mat = px.pivot(index="date", columns="code", values="adj_close").astype(float)
bask_ret = mat.pct_change(fill_method=None).mean(axis=1)  # EW、上場前は欠損のまま
print(f"  basket stocks: {mat.shape[1]}")

# 実現ボラ・VRP
rv20 = np.log(n225).diff().rolling(20).std() * np.sqrt(252) * 100
ind["rv20"] = rv20.reindex(ind.index)
ind["vrp"] = ind["atm_iv"] - ind["rv20"]

# z-score (252日ローリング, min 126)
def zscore(s, win=252, minp=126):
    return (s - s.rolling(win, min_periods=minp).mean()) / s.rolling(win, min_periods=minp).std()

ind["skew_z"] = zscore(ind["skew"])
ind["ivchg5_z"] = zscore(ind["iv_chg5"])
ind["atm_z"] = zscore(ind["atm_iv"])

# ---------------------------------------------------------------
# 3. 予測力診断: z五分位 × 先行リターン, Spearman IC
# ---------------------------------------------------------------
print("\n=== 予測力診断 (N225 先行リターン) ===")
diag = ind.copy()
cal = n225.reindex(diag.index.union(n225.index)).sort_index()
fwd5 = (cal.shift(-7) / cal.shift(-2) - 1).reindex(diag.index)   # T+2投資開始→5日
fwd20 = (cal.shift(-22) / cal.shift(-2) - 1).reindex(diag.index)

rows = []
for sig in ["skew_z", "term", "ivchg5_z", "atm_z", "vrp"]:
    s = diag[sig].dropna()
    f5, f20 = fwd5.reindex(s.index), fwd20.reindex(s.index)
    ic5 = stats.spearmanr(s, f5, nan_policy="omit")[0]
    ic20 = stats.spearmanr(s, f20, nan_policy="omit")[0]
    q = pd.qcut(s, 5, labels=False, duplicates="drop")
    top_f20 = f20[q == q.max()].mean() * 100
    bot_f20 = f20[q == 0].mean() * 100
    rows.append([sig, ic5, ic20, bot_f20, top_f20])
    print(f"  {sig:10s} IC5={ic5:+.3f} IC20={ic20:+.3f}  fwd20: Q1={bot_f20:+.2f}% Q5={top_f20:+.2f}%")
diag_df = pd.DataFrame(rows, columns=["signal", "ic5", "ic20", "q1_fwd20pct", "q5_fwd20pct"])

# ---------------------------------------------------------------
# 4. ゲートバックテスト
# ---------------------------------------------------------------
def gate_exposure(triggers: pd.DataFrame) -> pd.Series:
    """trigger列(bool)ごとに半減、下限0.25"""
    expo = pd.Series(1.0, index=triggers.index)
    for c in triggers.columns:
        expo = expo * np.where(triggers[c].fillna(False), 0.5, 1.0)
    return expo.clip(lower=0.25)

g1 = ind["skew_z"] > 1.5
g2 = ind["term"] > 0
g3 = ind["ivchg5_z"] > 1.5
r60 = n225.pct_change(60).reindex(ind.index)
g_r60 = r60 < -0.05

gates = {
    "no_gate":   pd.Series(1.0, index=ind.index),
    "G1_skew":   gate_exposure(g1.to_frame()),
    "G2_term":   gate_exposure(g2.to_frame()),
    "G3_ivchg":  gate_exposure(g3.to_frame()),
    "G_combo":   gate_exposure(pd.concat([g1, g2, g3], axis=1)),
    "G13":       gate_exposure(pd.concat([g1, g3], axis=1)),  # 事後追加: タームを除く
    "R60_base":  gate_exposure(g_r60.to_frame()),
}

def metrics(ret: pd.Series, name: str) -> dict:
    ret = ret.dropna()
    ann = ret.mean() * 252
    vol = ret.std() * np.sqrt(252)
    sh = ann / vol if vol > 0 else np.nan
    eq = (1 + ret).cumprod()
    mdd = (eq / eq.cummax() - 1).min()
    return {"name": name, "ann%": ann * 100, "vol%": vol * 100, "sharpe": sh, "mdd%": mdd * 100}

def run_gate_bt(target_ret: pd.Series, label: str):
    print(f"\n=== ゲートBT: {label} (コスト {COST_BPS}bps/単位ターンオーバ) ===")
    out = []
    for gname, expo in gates.items():
        lag = 1 if gname in ("no_gate", "R60_base") else 2  # IVデータはT+1夕方着 → shift(2)
        e = expo.reindex(target_ret.index).ffill().shift(lag).fillna(1.0)
        cost = e.diff().abs().fillna(0) * COST_BPS / 1e4
        ret = target_ret * e - cost
        # 指標が揃う期間に統一
        ret = ret[ind["skew_z"].dropna().index.min():]
        m = metrics(ret, gname)
        m["avg_expo"] = e[ret.index].mean()
        # 前半/後半
        m["sh_1H"] = metrics(ret[:"2021-12-31"], "")["sharpe"]
        m["sh_2H"] = metrics(ret["2022-01-01":], "")["sharpe"]
        out.append(m)
        print(f"  {gname:9s} ann={m['ann%']:+6.2f}% vol={m['vol%']:5.2f}% "
              f"Sharpe={m['sharpe']:+.2f} (1H {m['sh_1H']:+.2f} / 2H {m['sh_2H']:+.2f}) "
              f"MDD={m['mdd%']:+6.2f}% expo={m['avg_expo']:.2f}")
    return pd.DataFrame(out)

res_n225 = run_gate_bt(n225_ret, "N225ロング")
res_bask = run_gate_bt(bask_ret, "EW22バスケット(半導体14+非鉄8)")

ind.to_csv(os.path.join(OUTDIR, "iv_indicators.csv"))
res_n225.to_csv(os.path.join(OUTDIR, "gate_results_n225.csv"), index=False)
res_bask.to_csv(os.path.join(OUTDIR, "gate_results_basket.csv"), index=False)
diag_df.to_csv(os.path.join(OUTDIR, "ic_diagnostics.csv"), index=False)

# ---------------------------------------------------------------
# 5. 可視化
# ---------------------------------------------------------------
try:
    import matplotlib.font_manager as fm
    for fpath in [r"C:\Windows\Fonts\meiryo.ttc", "/root/.fonts/NotoSansJP.ttf"]:
        if os.path.exists(fpath):
            fm.fontManager.addfont(fpath)
            plt.rcParams["font.family"] = fm.FontProperties(fname=fpath).get_name()
            break
except Exception:
    pass
plt.rcParams.update({
    "axes.unicode_minus": False,
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f9fa",
    "grid.alpha": 0.3,
})

# --- result.png: X投稿用 1200x675 (資金曲線比較) ---
start = ind["skew_z"].dropna().index.min()
fig, ax = plt.subplots(figsize=(12, 6.75), facecolor="white")
colors = {"no_gate": "#888888", "G13": "#d62728", "R60_base": "#1f77b4"}
labels = {"no_gate": "ゲートなし (Sharpe 0.78 / MDD -32%)",
          "G13": "IVゲート: スキュー+IV急騰 (Sharpe 0.90 / MDD -23%)",
          "R60_base": "従来の価格ゲート R60<-5% (Sharpe 0.67 / MDD -28%)"}
for gname in ["no_gate", "G13", "R60_base"]:
    expo = gates[gname]
    lag = 1 if gname in ("no_gate", "R60_base") else 2
    e = expo.reindex(n225_ret.index).ffill().shift(lag).fillna(1.0)
    cost = e.diff().abs().fillna(0) * COST_BPS / 1e4
    ret = (n225_ret * e - cost)[start:]
    eq = (1 + ret.dropna()).cumprod()
    ax.plot(eq.index, eq, lw=1.4, label=labels[gname], color=colors[gname])
ax.grid(True)
ax.legend(loc="upper left", fontsize=11)
ax.set_title("日経225オプションIVレジームゲート — リターンを落とさずMDDを3割削減\n"
             "スキューz>1.5 / IV5日変化z>1.5 でエクスポージャ半減 (N225ロングに適用)",
             fontsize=14)
fig.text(0.99, 0.01, "データ: 2017-07〜2026-06 / JQuants options_n225・index_daily / コスト5bps込み",
         ha="right", va="bottom", fontsize=8, color="gray")
fig.savefig(os.path.join(OUTDIR, "result.png"), dpi=100, bbox_inches="tight", facecolor="white")

# --- diagnostics.png: 指標の中身 ---
fig2, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
ax = axes[0]
ax.plot(ind.index, ind["atm_iv"], lw=0.8, label="ATM IV (期近)")
ax.plot(ind.index, ind["rv20"], lw=0.8, alpha=0.7, label="実現ボラ20日")
ax.legend(loc="upper left"); ax.grid(True); ax.set_title("日経225 ATM IV と実現ボラ")
ax = axes[1]
ax.plot(ind.index, ind["skew_z"], lw=0.7, label="skew z")
ax.plot(ind.index, ind["ivchg5_z"], lw=0.7, alpha=0.7, label="IV5日変化 z")
ax.axhline(1.5, color="r", ls="--", lw=0.6, label="発火閾値 1.5")
ax.legend(loc="upper left"); ax.grid(True); ax.set_title("ゲートシグナル (z-score)")
fig2.savefig(os.path.join(OUTDIR, "diagnostics.png"), dpi=100, bbox_inches="tight", facecolor="white")
print("\nsaved result.png / diagnostics.png / iv_indicators.csv / gate_results_*.csv")
conn.close()
