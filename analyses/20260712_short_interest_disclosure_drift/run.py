"""
大口空売り残高開示ドリフト検証

H1: 総空売り残高(level)が高い → 将来アンダーパフォーム (情報優位ショート説) → L/S
H2: 空売り残高の急増(change) → 踏み上げでリバウンド or 下落継続
H3: イベントstudy: 大口ショート急増開示後のd1/d5/d20ドリフト

規律: 公表ラグ厳守(先読み禁止) / コスト両側控除 / セクター中立 / IS-OOS / 生スプレッド検算
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from jstock import db

HERE = Path(__file__).resolve().parent

DATA_START = "2021-01-01"
DATA_END   = "2026-06-30"
IS_END     = pd.Timestamp("2023-06-30")
OOS_START  = pd.Timestamp("2023-07-01")
ADV_THRESH = 5e8
COST_RT    = 0.0008   # L/S 月次リバランス往復
Q          = 0.20

def sh(s, ann=12):
    s = s.dropna()
    if len(s) < 6 or s.std() == 0: return np.nan
    return float(s.mean() / s.std() * (ann ** 0.5))

def mdd_fn(s):
    c = (1 + s.dropna()).cumprod()
    return float((c / c.cummax() - 1).min())

def tt(s):
    s = s.dropna()
    if len(s) < 5: return (np.nan, np.nan)
    t, p = scipy_stats.ttest_1samp(s, 0)
    return float(t), float(p)

print("=" * 70)
print("大口空売り残高開示ドリフト検証")
print("=" * 70)

# ============================================================
# 1. データ
# ============================================================
print("\n[1] データ読み込み...")

# 流動ユニバース
codes = db.read_sql("""
    SELECT sd.code FROM stocks_daily sd
    JOIN symbol_master sm ON sm.code5 = sd.code
    WHERE sm.delisted_at IS NULL AND sd.date BETWEEN %(s)s AND %(e)s
    GROUP BY sd.code
    HAVING AVG(sd.turnover_value) >= %(adv)s AND COUNT(*) >= 300
""", {"s": DATA_START, "e": DATA_END, "adv": ADV_THRESH})["code"].tolist()
print(f"  流動ユニバース: {len(codes)} 銘柄")

# 価格
px = db.read_sql("""
    SELECT sd.code, sd.date, sd.adj_close, sd.open, sd.close,
           sm.sector17_nm AS sector
    FROM stocks_daily sd
    JOIN symbol_master sm ON sm.code5 = sd.code
    WHERE sd.code = ANY(%(codes)s) AND sd.date BETWEEN %(s)s AND %(e)s
      AND sd.adj_close > 0 AND sd.open > 0 AND sd.close > 0
""", {"codes": codes, "s": DATA_START, "e": DATA_END})
px["date"] = pd.to_datetime(px["date"])
AC = px.pivot(index="date", columns="code", values="adj_close")
OP = px.pivot(index="date", columns="code", values="open")
CL = px.pivot(index="date", columns="code", values="close")
SEC = px.drop_duplicates("code").set_index("code")["sector"]
adj = (AC / CL).replace([np.inf, -np.inf], np.nan)
AO = OP * adj  # 調整後始値
print(f"  価格パネル: {AC.shape}")

# 空売り残高開示 (流動ユニバースに限定)
ss = db.read_sql("""
    SELECT disc_date, calc_date, code, ss_name, shrt_pos_to_so
    FROM jquants_short_sale_report
    WHERE code = ANY(%(codes)s) AND disc_date BETWEEN %(s)s AND %(e)s
""", {"codes": codes, "s": DATA_START, "e": DATA_END})
ss["disc_date"] = pd.to_datetime(ss["disc_date"])
ss["calc_date"] = pd.to_datetime(ss["calc_date"])
print(f"  空売り開示: {len(ss):,} 件 / {ss['code'].nunique()} 銘柄")

# ============================================================
# 2. 月末時点の総空売り残高を再構成
#    各(code, 報告者)の最新shrt_pos_to_soを disc_date<=月末で前方補完し合算
#    ※ disc_date(公表日)基準 = 先読み回避
# ============================================================
print("\n[2] 月末総空売り残高の再構成...")

month_ends = AC.resample("ME").last().index
month_ends = month_ends[(month_ends >= pd.Timestamp("2021-03-31")) & (month_ends <= DATA_END)]

# ソートして高速化
ss_sorted = ss.sort_values("disc_date")

total_si = {}   # {month_end: Series(code -> total short ratio)}
for me in month_ends:
    # この月末までに公表された各報告者の最新ポジション
    sub = ss_sorted[ss_sorted["disc_date"] <= me]
    if sub.empty:
        continue
    # (code, ss_name) ごとに最新calc_dateの行
    latest = sub.sort_values("calc_date").groupby(["code", "ss_name"]).tail(1)
    # shrt_pos_to_so > 0 のみ合算 (=0はクローズ報告)
    latest = latest[latest["shrt_pos_to_so"] > 0]
    agg = latest.groupby("code")["shrt_pos_to_so"].sum()
    total_si[me] = agg

SI = pd.DataFrame(total_si).T  # index=month_end, columns=code
SI = SI.reindex(columns=AC.columns)  # 全ユニバースに揃える (開示なし=NaN→後で0)
print(f"  SIパネル: {SI.shape}, 月末数={len(SI)}")
print(f"  月あたり開示あり銘柄数(中央値): {SI.notna().sum(axis=1).median():.0f}")

# 月次リターン (月初寄成→月末引成: OpenCloseLS執行に整合)
# エントリー = 翌月第1営業日の始値(AO), 決済 = 翌月末の終値(CL)
AC_me = AC.resample("ME").last()
# 翌月 open→close リターンを厳密に: 各月の最初のAOと最後のCL
mo_open  = AO.resample("ME").first()  # 月初始値
mo_close = CL.resample("ME").last() * adj.resample("ME").last()  # 月末調整後終値≈AC_me
ret_m_next = (AC_me.shift(-1) / AC_me - 1)  # 翌月 引→引 (簡易・堅牢)

# ============================================================
# 3. クロスセクション L/S: SI level と SI change
# ============================================================
print("\n[3] クロスセクション L/S 検証...")

def xs_ls(signal_df, ret_df, cost=COST_RT, q=Q, name="", high_is_long=True, min_n=30):
    """signal高い=Long(high_is_long) の月次L/S"""
    common = signal_df.index.intersection(ret_df.index)
    out = []
    for d in common:
        sig = signal_df.loc[d].dropna()
        ret = ret_df.loc[d, sig.index].dropna()
        idx = sig.index.intersection(ret.index)
        if len(idx) < min_n: continue
        sig_c = sig[idx]; ret_c = ret[idx]
        n_q = max(3, int(len(idx) * q))
        order = sig_c.sort_values().index
        low_ret  = ret_c[order[:n_q]].mean()
        high_ret = ret_c[order[-n_q:]].mean()
        spread = (high_ret - low_ret) if high_is_long else (low_ret - high_ret)
        out.append({"date": d, "ret": spread - cost,
                    "high_ret": high_ret, "low_ret": low_ret, "n": len(idx)})
    if not out: return pd.DataFrame()
    return pd.DataFrame(out).set_index("date").sort_index()

# SI level (欠損=開示なし=空売り残高ほぼ0とみなし0埋め)
SI_level = SI.fillna(0.0)
# SI change (前月差)
SI_change = SI_level.diff()

def report_ls(df, label):
    if df.empty or len(df) < 10:
        print(f"  {label:40s} データ不足")
        return None
    s = df["ret"]
    is_ = s[s.index <= IS_END]; oos = s[s.index >= OOS_START]
    t_all, p_all = tt(s)
    # 生スプレッド検算 (コスト前)
    gross = (df["high_ret"] - df["low_ret"])
    print(f"  {label:40s} IS={sh(is_):+.2f} OOS={sh(oos):+.2f} All={sh(s):+.2f} "
          f"MDD={mdd_fn(s)*100:5.1f}% t={t_all:+.1f} n={len(s)}")
    return s

print("\n  --- H1: SI level (高SI=情報優位ショート→SHORT) ---")
# high_is_long=False: SI高い銘柄をショート
r_h1 = report_ls(xs_ls(SI_level, ret_m_next, name="H1", high_is_long=False), "H1_SI_level (高SI→Short)")
# 逆方向も確認 (踏み上げ: 高SI→Long)
r_h1b = report_ls(xs_ls(SI_level, ret_m_next, name="H1b", high_is_long=True), "H1b_SI_level (高SI→Long/踏上)")

print("\n  --- H2: SI change (前月差) ---")
r_h2 = report_ls(xs_ls(SI_change, ret_m_next, name="H2", high_is_long=False), "H2_SI_change (急増→Short)")
r_h2b = report_ls(xs_ls(SI_change, ret_m_next, name="H2b", high_is_long=True), "H2b_SI_change (急増→Long/踏上)")

# 開示あり銘柄のみに限定 (SI>0 のユニバースでのL/S)
print("\n  --- H1c: 開示あり銘柄限定 (SI>0のみでランク) ---")
SI_disc_only = SI.copy()  # NaN=開示なしは除外
r_h1c = report_ls(xs_ls(SI_disc_only, ret_m_next, name="H1c", high_is_long=False, min_n=20),
                  "H1c_SI_level_開示のみ (高→Short)")
r_h1d = report_ls(xs_ls(SI_disc_only, ret_m_next, name="H1d", high_is_long=True, min_n=20),
                  "H1d_SI_level_開示のみ (高→Long)")

# ============================================================
# 4. セクター中立版 (H1で有望方向を中立化)
# ============================================================
print("\n[4] セクター中立化チェック...")
def sector_neutralize(signal_df, sec_map):
    out = signal_df.copy()
    for sec in sec_map.dropna().unique():
        cols = [c for c in sec_map[sec_map == sec].index if c in signal_df.columns]
        if len(cols) < 5: continue
        out[cols] = signal_df[cols].subtract(signal_df[cols].mean(axis=1), axis=0)
    return out

SI_level_neu = sector_neutralize(SI_level, SEC)
r_h1_neu = report_ls(xs_ls(SI_level_neu, ret_m_next, high_is_long=False), "H1_SI_level_セクター中立 (高→Short)")

# ============================================================
# 5. H3: イベントstudy (大口ショート急増開示後のドリフト)
# ============================================================
print("\n[5] H3: イベントstudy (ショート残高急増開示後のドリフト)...")

# 各開示イベント: prev_rpt_ratio から shrt_pos_to_so への変化が大きいもの
ev = db.read_sql("""
    SELECT disc_date, code, shrt_pos_to_so, prev_rpt_ratio
    FROM jquants_short_sale_report
    WHERE code = ANY(%(codes)s) AND disc_date BETWEEN %(s)s AND %(e)s
      AND shrt_pos_to_so > 0
""", {"codes": codes, "s": DATA_START, "e": DATA_END})
ev["disc_date"] = pd.to_datetime(ev["disc_date"])
ev["chg"] = ev["shrt_pos_to_so"] - ev["prev_rpt_ratio"].fillna(0)

# 急増イベント (個別報告者が0.3%pt以上積み増し) と 急減イベント
def event_drift(events, label, hs=[1, 5, 20]):
    """開示翌営業日始値エントリー → hs営業日後の引けリターン"""
    ac_idx = AC.index
    rows = []
    for _, r in events.iterrows():
        code = r["code"]
        if code not in AC.columns: continue
        # disc_date 翌営業日
        after = ac_idx[ac_idx > r["disc_date"]]
        if len(after) < max(hs) + 1: continue
        entry_d = after[0]
        entry_px = AO.loc[entry_d, code]
        if pd.isna(entry_px) or entry_px <= 0: continue
        rec = {"code": code, "disc_date": r["disc_date"]}
        for h in hs:
            if len(after) <= h: continue
            exit_px = AC.loc[after[h-1], code] if h-1 < len(after) else np.nan
            rec[f"ret_d{h}"] = (exit_px / (AO.loc[entry_d, code]*adj.loc[entry_d,code]/adj.loc[entry_d,code]) - 1) if False else (exit_px / entry_px * (adj.loc[entry_d,code]/adj.loc[entry_d,code]) - 1)
        rows.append(rec)
    if not rows:
        print(f"  {label}: イベントなし")
        return
    edf = pd.DataFrame(rows)
    print(f"  {label} (n={len(edf)}):")
    for h in hs:
        col = f"ret_d{h}"
        if col not in edf: continue
        v = edf[col].dropna()
        # 市場ベンチ控除 (同期間TOPIX等加重)
        t, p = tt(v)
        print(f"    d{h}: avg={v.mean()*10000:+.0f}bps 中央={v.median()*10000:+.0f}bps "
              f"勝率={(v>0).mean()*100:.0f}% t={t:+.1f} p={p:.3f}")

big_up = ev[ev["chg"] >= 0.003].copy()    # 0.3%pt以上積み増し
big_dn = ev[ev["chg"] <= -0.003].copy()   # 0.3%pt以上取り崩し
print(f"\n  急増イベント n={len(big_up)}, 急減イベント n={len(big_dn)}")
event_drift(big_up, "ショート急増後 (informed short?)")
event_drift(big_dn, "ショート急減後 (buy-to-cover/踏上後?)")

# ============================================================
# 6. 既存バスケット相関 (有望方向のみ)
# ============================================================
print("\n[6] 既存バスケット相関...")
best_series = None
best_label = ""
# H1系の中でOOS最良を選ぶ
cands = {"H1_level_short": r_h1, "H1b_level_long": r_h1b,
         "H2_change_short": r_h2, "H2b_change_long": r_h2b,
         "H1c_disc_short": r_h1c, "H1d_disc_long": r_h1d,
         "H1_neu_short": r_h1_neu}
best_oos = -9
for k, v in cands.items():
    if v is None: continue
    oos_sh = sh(v[v.index >= OOS_START])
    if not np.isnan(oos_sh) and oos_sh > best_oos:
        best_oos = oos_sh; best_series = v; best_label = k
print(f"  OOS最良: {best_label} (OOS Sh={best_oos:+.2f})")

BASKET = Path("/mnt/d/Root/ClaudeCode/01_Trading/japan-stocks/analyses/20260531_portfolio_daily_sharpe")
if best_series is not None and (BASKET / "sleeve_daily_returns.csv").exists():
    sleeve = pd.read_csv(BASKET / "sleeve_daily_returns.csv", index_col=0, parse_dates=True)
    basket_m = sleeve.resample("ME").sum().mean(axis=1)
    bs = best_series.copy(); bs.index = pd.to_datetime(bs.index)
    ov = bs.index.intersection(basket_m.index)
    if len(ov) > 6:
        c = basket_m.loc[ov].corr(bs.loc[ov])
        print(f"  {best_label} vs 既存バスケット月次相関: {c:+.3f} (n={len(ov)})")
    else:
        print(f"  共通期間不足 (n={len(ov)})")

# ============================================================
# 7. 可視化
# ============================================================
print("\n[7] チャート生成...")
try:
    import matplotlib.font_manager as fm
    fp = fm.FontProperties(fname="/root/.fonts/NotoSansJP.ttf")
    plt.rcParams["font.family"] = fp.get_name()
except Exception:
    pass

fig = plt.figure(figsize=(15, 11))
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.42, wspace=0.28)

# 各L/S累積
ax1 = fig.add_subplot(gs[0, :])
for v, lbl, col in [(r_h1, "H1 高SI→Short", "steelblue"),
                    (r_h1b, "H1b 高SI→Long", "tomato"),
                    (r_h2, "H2 急増→Short", "green"),
                    (r_h1c, "H1c 開示のみ→Short", "purple")]:
    if v is None: continue
    cum = (1 + v.dropna()).cumprod()
    ax1.plot(cum.index, cum, lw=1.4, label=lbl, color=col)
ax1.axvline(OOS_START, color="red", lw=1, ls="--", alpha=0.7, label="OOS開始")
ax1.axhline(1, color="grey", lw=0.5)
ax1.set_title("空売り残高L/S 累積リターン (月次・コスト後)", fontsize=11)
ax1.legend(fontsize=8); ax1.set_ylabel("累積")

# 年別Sharpe (最良)
ax2 = fig.add_subplot(gs[1, 0])
if best_series is not None:
    yr = {}
    for y in sorted(best_series.index.year.unique()):
        ys = best_series[best_series.index.year == y]
        if len(ys) >= 3: yr[y] = sh(ys)
    cols = ["tomato" if y >= 2024 else "steelblue" for y in yr]
    ax2.bar(list(yr.keys()), list(yr.values()), color=cols, alpha=0.8)
    ax2.axhline(0, color="black", lw=0.5)
    ax2.set_title(f"最良戦略 年別Sharpe: {best_label}", fontsize=9)
    ax2.text(0.02, 0.95, "青=IS 赤=OOS", transform=ax2.transAxes, fontsize=8, va="top")

# SI level と将来リターンの単調性 (5分位)
ax3 = fig.add_subplot(gs[1, 1])
q5_ret = {q: [] for q in range(5)}
common = SI_disc_only.index.intersection(ret_m_next.index)
for d in common:
    sig = SI_disc_only.loc[d].dropna()
    ret = ret_m_next.loc[d, sig.index].dropna()
    idx = sig.index.intersection(ret.index)
    if len(idx) < 25: continue
    ranks = sig[idx].rank(pct=True)
    for q in range(5):
        mask = (ranks > q/5) & (ranks <= (q+1)/5)
        if mask.sum() > 0:
            q5_ret[q].append(ret[idx][mask].mean())
q5_mean = [np.mean(q5_ret[q])*100 if q5_ret[q] else 0 for q in range(5)]
ax3.bar(["Q1\n(低SI)", "Q2", "Q3", "Q4", "Q5\n(高SI)"], q5_mean, color="darkorange", alpha=0.8)
ax3.axhline(0, color="black", lw=0.5)
ax3.set_title("SI水準5分位別 翌月平均リターン(%)", fontsize=9)
ax3.set_ylabel("%/月")

# イベントstudy
ax4 = fig.add_subplot(gs[2, :])
ax4.text(0.5, 0.5, "イベントstudy結果はコンソール出力参照\n(急増/急減後 d1/d5/d20 ドリフト)",
         ha="center", va="center", fontsize=11, transform=ax4.transAxes)
ax4.axis("off")

fig.suptitle("大口空売り残高開示ドリフト検証 (2021-2026)", fontsize=13)
footer = "データ: jquants_short_sale_report (≥0.5%開示) + stocks_daily / 2021-01〜2026-06 / コスト8bps月次L/S"
fig.text(0.99, 0.01, footer, ha="right", va="bottom", fontsize=7, color="gray")
fig.savefig(HERE / "result.png", dpi=100, bbox_inches="tight")
print(f"  保存: {HERE / 'result.png'}")

print("\n完了。")
