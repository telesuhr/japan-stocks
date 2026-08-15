"""
H4: GapReversal の執行タイミング感度。
寄成(9:00の始値)では約定できない（ギャップは寄って初めて確定する）。
1分足で 9:05 / 9:10 / 9:30 建てに置き換え、エッジがどれだけ残るかを測る。
対象: 2024-01〜2026-08（直近の強い局面＝最も有利な条件でのテスト）。
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
import pandas as pd
from jstock import db

COST_LS = 0.0008
ADV_MIN, ADV_WIN, Q_LS, GAP_THR = 5e8, 60, 0.20, 0.025
START = "2023-11-01"   # ADV助走込み
SIG_FROM = "2024-05-10"   # stocks_intraday の開始

print("[1] 日足...")
raw = db.read_sql("""
    SELECT code, date, adj_open, adj_close, turnover_value
    FROM stocks_daily WHERE date >= %(s)s AND adj_close > 0 AND adj_open > 0
""", {"s": START})
raw["date"] = pd.to_datetime(raw["date"])
AO = raw.pivot(index="date", columns="code", values="adj_open").sort_index()
AC = raw.pivot(index="date", columns="code", values="adj_close").sort_index()
TV = raw.pivot(index="date", columns="code", values="turnover_value").sort_index()
UNIV = TV.rolling(ADV_WIN, min_periods=40).mean().shift(1) >= ADV_MIN
gap = AO / AC.shift(1) - 1.0
sig = (-gap).where((gap.abs() >= GAP_THR) & UNIV & AC.notna())

picks = []
for d in AO.index[AO.index >= SIG_FROM]:
    s = sig.loc[d].dropna()
    if len(s) < 20:
        continue
    n = max(3, int(len(s) * Q_LS))
    o = s.sort_values().index
    for c in o[-n:]:
        picks.append(dict(date=d, code=c, side=+1))
    for c in o[:n]:
        picks.append(dict(date=d, code=c, side=-1))
P = pd.DataFrame(picks)
print(f"  シグナル日 {P.date.nunique()}日 / 建玉 {len(P):,}件 / 銘柄 {P.code.nunique()}")

print("[2] 1分足（寄り30分＋大引け）— シグナル日のみ月次で取得...")
chunks = []
for (yy, mm), grp in P.groupby([P.date.dt.year, P.date.dt.month]):
    c = sorted(grp.code.unique())
    d0 = grp.date.min()
    d1 = grp.date.max() + pd.Timedelta(days=1)
    part = db.read_sql("""
        SELECT code, ts, close FROM stocks_intraday
        WHERE code = ANY(%(c)s) AND ts >= %(a)s AND ts < %(b)s
          AND (ts::time <= '09:35' OR ts::time >= '14:50')
    """, {"c": c, "a": d0.date(), "b": d1.date()})
    chunks.append(part)
    print(f"    {yy}-{mm:02d}: {len(c)}銘柄 {len(part):,}行")
bars = pd.concat(chunks, ignore_index=True)
bars["ts"] = pd.to_datetime(bars["ts"])
bars["date"] = bars["ts"].dt.normalize()
bars["t"] = bars["ts"].dt.strftime("%H:%M")
print(f"  {len(bars):,}行")

# 各(code,date)の 時刻別価格 と 引け値
px = bars.pivot_table(index=["code", "date"], columns="t", values="close", aggfunc="last")
last = bars.sort_values("ts").groupby(["code", "date"])["close"].last().rename("EOD")
px = px.join(last)

TIMES = {"09:00寄成(元検証)": None, "09:05": "09:05", "09:10": "09:10",
         "09:20": "09:20", "09:30": "09:30"}
P = P.set_index(["code", "date"])
P["EOD"] = last.reindex(P.index)
P["open_adj"] = [AO.at[d, c] for c, d in P.index]
P["close_adj"] = [AC.at[d, c] for c, d in P.index]
# adj係数（EODは生値なので、生値ベースでリターンを取れば調整不要）
P["open_raw_ratio"] = P["close_adj"] / P["open_adj"]   # = 生close/生open

rows = []
series = {}
for lbl, t in TIMES.items():
    if t is None:
        r = P["open_raw_ratio"] - 1.0
    else:
        p0 = px[t].reindex(P.index) if t in px.columns else pd.Series(np.nan, index=P.index)
        r = P["EOD"] / p0 - 1.0
    d = pd.DataFrame({"date": P.index.get_level_values("date").values,
                      "side": P["side"].values, "r": r.values}).dropna()
    leg = d.groupby(["date", "side"])["r"].mean().unstack()
    if not {1, -1}.issubset(leg.columns):
        continue
    daily = (leg[1] - leg[-1]).dropna()
    net = daily - COST_LS
    series[lbl] = net
    cov = d.groupby("date").size().mean()
    rows.append(dict(entry=lbl, N=len(daily), cover=cov,
                     spread=daily.mean() * 100, net=net.mean() * 100,
                     med=net.median() * 100,
                     sharpe_act=net.mean() / net.std() * np.sqrt(252) if net.std() > 0 else np.nan))

R = pd.DataFrame(rows)
base = R.loc[R.entry.str.startswith("09:00"), "spread"].iloc[0]
R["残存率%"] = R["spread"] / base * 100
print("\n" + "=" * 82)
print("H4: 執行タイミング別 L/Sスプレッド（2024-01〜2026-08・コスト8bps後）")
print("=" * 82)
print(f"{'建てタイミング':<20}{'日数':>6}{'銘柄/日':>8}{'総spread%':>11}{'net%':>8}{'中央値%':>9}{'残存率%':>9}")
for _, x in R.iterrows():
    print(f"{x['entry']:<20}{x['N']:>6}{x['cover']:>8.0f}{x['spread']:>11.3f}"
          f"{x['net']:>8.3f}{x['med']:>9.3f}{x['残存率%']:>9.0f}")
R.to_csv("h4_execution_timing.csv", index=False)

# 利益集中度・年次（外れ値依存の確認）
print("\n利益集中度と年次（コスト後net）")
print(f"{'建て':<20}{'上位5日の利益シェア%':>20}{'2024H2':>9}{'2025':>9}{'2026':>9}")
for lbl, s in series.items():
    tot = s.sum()
    top5 = s.nlargest(5).sum() / tot * 100 if tot > 0 else np.nan
    yv = {y: ((1 + g).prod() - 1) * 100 for y, g in s.groupby(s.index.year)}
    print(f"{lbl:<20}{top5:>20.0f}{yv.get(2024, np.nan):>9.1f}"
          f"{yv.get(2025, np.nan):>9.1f}{yv.get(2026, np.nan):>9.1f}")
pd.DataFrame(series).to_csv("h4_daily_series.csv")
print("\nsaved h4_execution_timing.csv / h4_daily_series.csv")
