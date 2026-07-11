"""
既存戦略の有効性確認 + セクターMOM L/S 組み合わせ効果の検証

1. PED / PEAD / ON-LS の直近リターンを DB から再計算して健全性チェック
2. sleeve_daily_returns.csv (4戦略) の足元パフォーマンス
3. セクターMOM TOPIX-ヘッジ L/S を加えた場合のポートフォリオ改善を定量化
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from jstock import db

HERE = Path(__file__).resolve().parent
PREV = HERE.parent / "20260706_sector_rotation_promotion"
BASKET = HERE.parent / "20260531_portfolio_daily_sharpe"

# ------------------------------------------------------------------
# ユーティリティ
# ------------------------------------------------------------------
def sharpe(s, ann=12):
    s = s.dropna()
    if len(s) < 3 or s.std() == 0:
        return float("nan")
    return float(s.mean() / s.std() * (ann ** 0.5))

def max_dd(s):
    cum = (1 + s).cumprod()
    return float((cum / cum.cummax() - 1).min())

def wr(s):
    return float((s.dropna() > 0).mean())

def block(label, s, ann=252):
    s = s.dropna()
    sh = sharpe(s, ann)
    return dict(label=label, n=len(s), ann_ret=float(s.mean()*ann),
                sharpe=sh, mdd=max_dd(s), wr=wr(s))

# ------------------------------------------------------------------
# 1. 既存戦略: sleeve daily returns (4戦略)
# ------------------------------------------------------------------
print("=" * 60)
print("1. 既存4戦略 — sleeve daily returns 足元パフォーマンス")
print("=" * 60)
sleeve = pd.read_csv(BASKET / "sleeve_daily_returns.csv", index_col=0, parse_dates=True)
baselines = {"eneos_vwap_trend": 0.82, "vwap_morning_meanrevert": 1.78,
             "lasertec_ma25_support": 2.97, "bank_absorption": 1.20}
print(f"\n期間: {sleeve.index[0].date()} 〜 {sleeve.index[-1].date()} ({len(sleeve)}日)")
rows = []
for col in sleeve.columns:
    s_all = sleeve[col]
    s_3m  = sleeve[sleeve.index >= sleeve.index[-1] - pd.Timedelta(days=90)][col]
    s_6m  = sleeve[sleeve.index >= sleeve.index[-1] - pd.Timedelta(days=180)][col]
    sh_all = sharpe(s_all[s_all != 0])
    sh_3m  = sharpe(s_3m[s_3m != 0])
    sh_6m  = sharpe(s_6m[s_6m != 0])
    base   = baselines.get(col, 1.5)
    thr    = base * 0.6
    if sh_all >= thr:
        status = "✅"
    elif sh_all > 0:
        status = "⚠️"
    else:
        status = "🚨"
    print(f"  {status} {col:30s}  全期間Sh={sh_all:+.2f}  6M Sh={sh_6m:+.2f}  3M Sh={sh_3m:+.2f}  基準={base:.2f}")
    rows.append(dict(name=col, sh_all=sh_all, sh_6m=sh_6m, sh_3m=sh_3m))

sleeve_all = sleeve.mean(axis=1)  # 等加重バスケット
print(f"\n  等加重バスケット全体  Sh(年率)={sharpe(sleeve_all):.2f}  MDD={max_dd(sleeve_all)*100:.1f}%")

# ------------------------------------------------------------------
# 2. PED (決算前4日ドリフト) の直近リターン
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("2. PED 直近リターン (DB再計算)")
print("=" * 60)
try:
    ped_raw = db.read_sql("""
        WITH earn AS (
            SELECT DISTINCT code, date AS earn_date
            FROM earnings_calendar
            WHERE date BETWEEN '2025-01-01' AND '2026-07-10'
        ),
        d4 AS (
            SELECT e.code, e.earn_date,
                   sd.date AS entry_date, sd.adj_close AS entry_px
            FROM earn e
            JOIN stocks_daily sd ON sd.code = e.code
            WHERE sd.date = (
                SELECT d2.date FROM stocks_daily d2
                WHERE d2.code = e.code AND d2.date < e.earn_date
                ORDER BY d2.date DESC LIMIT 1 OFFSET 3
            )
              AND sd.adj_close > 0
        ),
        d1 AS (
            SELECT e.code, e.earn_date,
                   sd.date AS exit_date, sd.adj_close AS exit_px
            FROM earn e
            JOIN stocks_daily sd ON sd.code = e.code
            WHERE sd.date = (
                SELECT d2.date FROM stocks_daily d2
                WHERE d2.code = e.code AND d2.date < e.earn_date
                ORDER BY d2.date DESC LIMIT 1
            )
              AND sd.adj_close > 0
        )
        SELECT d4.entry_date, x.exit_date,
               (x.exit_px - d4.entry_px) / d4.entry_px AS gross_ret
        FROM d4 JOIN d1 x ON x.code = d4.code AND x.earn_date = d4.earn_date
        ORDER BY d4.entry_date
    """)
    ped_raw["net"] = ped_raw["gross_ret"] - 0.0004
    ped_m = ped_raw.set_index("entry_date")["net"].resample("MS").mean().dropna()
    ped_sh = sharpe(ped_m, 12)
    ped_wr = wr(ped_m)
    print(f"  PED n={len(ped_raw)} trades ({ped_raw['entry_date'].min()} 〜 {ped_raw['entry_date'].max()})")
    print(f"  月次Sharpe={ped_sh:.2f}  WR={ped_wr*100:.0f}%  月次avg={ped_m.mean()*10000:.0f}bps")
    status = "✅" if ped_sh >= 0.94 else ("⚠️" if ped_sh > 0 else "🚨")  # baseline 1.56 × 0.6
    print(f"  {status} 基準Sharpe 1.56 × 0.6 = 0.94")
except Exception as e:
    print(f"  PED再計算エラー: {e}")
    ped_m = None

# ------------------------------------------------------------------
# 3. ON-LS ペーパートレード
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("3. ON-LS ペーパートレード (2026-06-12〜)")
print("=" * 60)
onls_path = HERE.parent.parent.parent / "ON_LongShort" / "live" / "out" / "paper_log.csv"
if onls_path.exists():
    onls = pd.read_csv(onls_path)
    net_s = onls["net_realized_bp"].dropna() / 10000  # bp → 小数
    onls_sh = sharpe(net_s, 252)
    onls_wr = wr(net_s)
    cum_bp = onls["net_realized_bp"].sum()
    print(f"  ON-LS n={len(onls)}日  累積={cum_bp:.0f}bp  日次Sharpe(年率)={onls_sh:.2f}  WR={onls_wr*100:.0f}%")
    print(f"  ✅ ゲートON・運用中（バスケット外・モニタリング段階）")

# ------------------------------------------------------------------
# 4. セクターMOM L/S との組み合わせ効果
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("4. セクターMOM TOPIX-ヘッジ L/S 追加の分散効果")
print("=" * 60)

# 月次リターン系列
excess_m = pd.read_csv(PREV / "strategy_monthly_excess.csv", index_col=0).iloc[:, 0]
abs_m    = pd.read_csv(PREV / "strategy_monthly_abs.csv",    index_col=0).iloc[:, 0]
topix_m  = abs_m - excess_m
ls_net   = excess_m - 0.0010  # L/S net (-10bps/月)
ls_net.index = pd.to_datetime(pd.Series(ls_net.index) + "-01").values

# 既存バスケット月次
sleeve_m = sleeve.resample("MS").sum()

# 共通期間
common = ls_net.index.intersection(sleeve_m.index)
ls_c   = ls_net.loc[common]
sl_c   = sleeve_m.loc[common]
sl_eq  = sl_c.mean(axis=1)  # 等加重バスケット月次

print(f"\n共通期間: {common[0].date()} 〜 {common[-1].date()} (n={len(common)})")

print("\n--- 等加重バスケット (4戦略) ---")
print(f"  Sh(年率)={sharpe(sl_eq, 12):.2f}  MDD={max_dd(sl_eq)*100:.1f}%  WR={wr(sl_eq)*100:.0f}%")

# 相関確認
corr_with_ls = sl_c.corrwith(ls_c)
print(f"\n各戦略とセクターMOM L/Sの相関:")
for k, v in corr_with_ls.items():
    print(f"  {k}: {v:+.3f}")
print(f"  平均相関: {corr_with_ls.mean():+.3f}")

# 混合ポートフォリオ (バスケット + セクターMOM L/Sをw%追加)
print("\n--- 混合ポートフォリオ (basket + セクターMOM L/S @ w%) ---")
print(f"  {'w%':>5s}  {'Sh':>7s}  {'MDD':>8s}  {'WR':>6s}  {'月次avg':>10s}")
best_sh = -9
best_w = 0
results_blend = []
for w in [0, 0.1, 0.2, 0.3, 0.4, 0.5]:
    mixed = (1 - w) * sl_eq + w * ls_c
    sh_m = sharpe(mixed, 12)
    mdd_m = max_dd(mixed)
    wr_m = wr(mixed)
    avg_m = mixed.mean()
    print(f"  {w*100:>5.0f}%  {sh_m:>+7.2f}  {mdd_m*100:>+7.1f}%  {wr_m*100:>5.0f}%  {avg_m*100:>+9.2f}%/月")
    results_blend.append(dict(w=w, sh=sh_m, mdd=mdd_m, wr=wr_m))
    if sh_m > best_sh:
        best_sh = sh_m; best_w = w
print(f"\n  → 最適ウェイト: {best_w*100:.0f}% でSharpe最大 ({best_sh:.2f})")

# ------------------------------------------------------------------
# 5. 可視化
# ------------------------------------------------------------------
fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

# 既存4戦略の累積
ax1 = fig.add_subplot(gs[0, :])
for col in sleeve.columns:
    s = sleeve[col]
    s_nz = s.copy(); s_nz[s_nz == 0] = np.nan
    cum = (1 + s_nz).cumprod()
    ax1.plot(sleeve.index, cum / cum.iloc[0], label=col, lw=1.2, alpha=0.8)
ax1.plot(sleeve.index, (1+sleeve_all).cumprod() / (1+sleeve_all).cumprod().iloc[0],
         color='black', lw=2, linestyle='--', label='EqualWeight Basket')
ax1.set_title("Existing 4 Strategies - Cumulative Returns")
ax1.set_ylabel("Cumulative (norm.)")
ax1.legend(fontsize=8, loc='upper left')
ax1.axhline(1, color='grey', lw=0.5)

# 混合ポートフォリオ Sharpe推移
ax2 = fig.add_subplot(gs[1, 0])
blend_df = pd.DataFrame(results_blend)
ax2.bar([f"{int(w*100)}%" for w in blend_df.w], blend_df.sh, color='steelblue', alpha=0.8)
ax2.axhline(blend_df[blend_df.w==0]['sh'].values[0], color='red', lw=1, linestyle='--', label='Basket only')
ax2.set_title("Portfolio Sharpe vs Sector MOM L/S Weight")
ax2.set_xlabel("Sector MOM L/S Weight")
ax2.set_ylabel("Annual Sharpe")
ax2.legend(fontsize=9)

# 混合ポートフォリオ MDD推移
ax3 = fig.add_subplot(gs[1, 1])
ax3.bar([f"{int(w*100)}%" for w in blend_df.w], [-x*100 for x in blend_df.mdd],
        color='tomato', alpha=0.8)
ax3.axhline(-blend_df[blend_df.w==0]['mdd'].values[0]*100, color='red', lw=1, linestyle='--', label='Basket only')
ax3.set_title("Max Drawdown vs Sector MOM L/S Weight")
ax3.set_xlabel("Sector MOM L/S Weight")
ax3.set_ylabel("Max DD (%)")
ax3.legend(fontsize=9)

# 累積比較 (共通期間)
ax4 = fig.add_subplot(gs[2, :])
for w, col, ls_style in [(0, 'Basket only', '-'), (best_w, f'Basket+{best_w*100:.0f}% SectorMOM', '--')]:
    mixed = (1-w)*sl_eq + w*ls_c
    cum = (1+mixed).cumprod()
    ax4.plot(common, cum, label=col, lw=1.8, linestyle=ls_style)
cum_ls = (1+ls_c).cumprod()
ax4.plot(common, cum_ls / cum_ls.iloc[0], label='SectorMOM L/S net (standalone)',
         lw=1.2, color='green', alpha=0.7, linestyle=':')
ax4.set_title(f"Combined Portfolio vs Components (overlap period, n={len(common)})")
ax4.set_ylabel("Cumulative")
ax4.legend(fontsize=9)
ax4.axhline(1, color='grey', lw=0.5)

fig.suptitle("Strategy Health & Sector MOM L/S Portfolio Impact", fontsize=12)
fig.savefig(HERE / "result_portfolio.png", dpi=100, bbox_inches="tight")
print("\nsaved result_portfolio.png")
