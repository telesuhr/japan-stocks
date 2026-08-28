"""closing_auction_rebound を「低タッチで執行できるか」の一点で判定する。

背景:
  SUMMARY.md で「昇格判断フェーズ」のまま3ヶ月止まっている唯一の候補。
  この1本の可否が J-Quants 分足アドオン(月5,500円)の継続可否に直結する。

検証する3点（事前宣言・教訓5）:
  H1: バックテストのエッジ(net Sharpe 2.00)は、ペーパー期間(2026-05-29〜)でも生きているか
  H2: 決済を「翌09:00-09:15の窓」から**純粋な寄成(MOO)**に置き換えてもエッジは残るか
      → 低タッチでは窓執行は不可能。寄成で消えるなら低タッチ戦略として成立しない
  H3: そもそもシグナルは**15:25のMOC発注期限までに観測可能か**
      → jump は 15:30 板寄せ後に確定する。JQuants分足に 15:25-15:29 のバーが存在するかで判定

昨日の教訓を適用:
  - 日次ポートフォリオ系列(シグナルゼロの日=0.0)で年率化。per-trade×√252 は禁止
  - コストは往復で明示的に控除
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
from pathlib import Path
import numpy as np
import pandas as pd
from jstock import db

HERE = Path(__file__).resolve().parent
ANN = 252
THRESHOLD_BPS = -50.0
TOP_N = 200
IS_END = "2026-05-28"      # 元バックテストの終端
OOS_START = "2026-05-29"   # ペーパー記録の開始

print("=" * 92)
print("STEP0: H3 — シグナルはMOC発注期限(15:25)までに観測可能か")
print("=" * 92)
bars = db.read_sql("""
    SELECT ts::time AS t, COUNT(*) n
    FROM stocks_intraday
    WHERE ts >= %s AND ts < %s
    GROUP BY 1 ORDER BY 1
""", ["2026-08-28 15:20", "2026-08-28 15:35"])
print(bars.to_string(index=False))
has_preclose = bars["t"].astype(str).between("15:25:00", "15:29:59").any()
print(f"\n  15:25-15:29 のバー: {'あり' if has_preclose else '**なし**'}")
print("  → jump は 15:30 板寄せ後にしか確定しない。MOC は 15:25 までに入れる必要がある。")
print("  → JQuants分足だけでは**事前に発注できない**（プレクロージング気配の別フィードが必要）。")

# ---------------------------------------------------------------- ユニバース
print("\n" + "=" * 92)
print("STEP1: PITユニバース（trailing 400日ADV上位200・月次更新）")
print("=" * 92)
months = db.read_sql("""
    SELECT DISTINCT date_trunc('month', ts)::date AS m
    FROM stocks_intraday ORDER BY 1
""")["m"].tolist()
print(f"  対象月: {months[0]} 〜 {months[-1]} ({len(months)}ヶ月)")

univ = {}
for m in months:
    u = db.read_sql("""
        SELECT code FROM stocks_daily
        WHERE date < %s AND date >= %s::date - INTERVAL '400 days' AND turnover_value > 0
        GROUP BY code ORDER BY AVG(turnover_value) DESC LIMIT %s
    """, [m, m, TOP_N])["code"].tolist()
    univ[m] = set(u)
all_codes = sorted(set().union(*univ.values()))
print(f"  延べ銘柄数(和集合): {len(all_codes)}")

# ---------------------------------------------------------------- 価格取得
print("\n" + "=" * 92)
print("STEP2: 必要な時刻のバーだけ取得 (15:24 / 15:30 / 翌09:00 / 09:05 / 09:15)")
print("=" * 92)
px = db.read_sql("""
    SELECT ts::date AS d, code, ts::time AS t, open, close
    FROM stocks_intraday
    WHERE code = ANY(%s)
      AND ts::time IN ('15:24:00','15:30:00','09:00:00','09:05:00','09:15:00')
""", [all_codes])
px["t"] = px["t"].astype(str)
print(f"  取得: {len(px):,} 行")

wide = px.pivot_table(index=["d", "code"], columns="t",
                      values=["open", "close"], aggfunc="last")
wide.columns = [f"{a}_{b}" for a, b in wide.columns]
wide = wide.reset_index()
wide["d"] = pd.to_datetime(wide["d"])

# ---------------------------------------------------------------- シグナル
print("\n" + "=" * 92)
print("STEP3: シグナル生成 (close_jump = 引値15:30 / 15:24終値 − 1 ≤ −50bps)")
print("=" * 92)
w = wide.dropna(subset=["close_15:24:00", "close_15:30:00"]).copy()
w["in_univ"] = [c in univ.get(d.replace(day=1).date(), set())
                for d, c in zip(w["d"], w["code"])]
w = w[w["in_univ"]]
ENTRY_RAW = "close_15:30:00"
w["jump_bps"] = (w[ENTRY_RAW] / w["close_15:24:00"] - 1) * 1e4
sig = w[w["jump_bps"] <= THRESHOLD_BPS].copy()
print(f"  シグナル総数: {len(sig):,}  (対象日数 {w['d'].nunique()})")
print(f"  1日あたり中央値: {sig.groupby('d').size().median():.0f} 銘柄")

# ---------------------------------------------------------------- 翌日価格を結合
nxt = wide[["d", "code", "open_09:00:00", "close_09:05:00",
            "close_09:15:00", "close_15:30:00"]].copy()
nxt.columns = ["d", "code", "nx_open0900", "nx_c0905", "nx_c0915", "nx_close"]
days = sorted(wide["d"].unique())
nextday = {d: n for d, n in zip(days[:-1], days[1:])}
sig["nd"] = sig["d"].map(nextday)
sig = sig.merge(nxt, left_on=["nd", "code"], right_on=["d", "code"],
                how="left", suffixes=("", "_y")).drop(columns=["d_y"])

# --- 分割調整（stocks_intraday は無調整の生値。オーバーナイト保有は必ず割れる）---
# JQuants の AdjustmentFactor は「その日より前の価格に掛ける係数」。
# 翌営業日に分割があれば、建値(前日引値)に adj_factor を掛けて同じ土俵に乗せる。
af = db.read_sql("""
    SELECT date AS nd, code, adj_factor FROM stocks_daily
    WHERE code = ANY(%s) AND adj_factor IS NOT NULL
""", [all_codes])
af["nd"] = pd.to_datetime(af["nd"])
sig = sig.merge(af, on=["nd", "code"], how="left")
sig["adj_factor"] = sig["adj_factor"].astype(float).fillna(1.0)
n_split = int((sig["adj_factor"] != 1.0).sum())
sig["entry_adj"] = sig[ENTRY_RAW] * sig["adj_factor"]
print(f"  翌営業日に分割/併合があったシグナル: {n_split} 件 → 建値を調整")

# ---------------------------------------------------------------- 損益
EXITS = {
    "翌09:00 寄成(MOO)": "nx_open0900",       # ★低タッチで唯一実行可能
    "翌09:05": "nx_c0905",
    "翌09:15": "nx_c0915",
    "翌引け 引成(MOC)": "nx_close",           # ★低タッチで実行可能
}
ENTRY = "entry_adj"          # 分割調整済みの建値


def daily_series(df, exitcol, cost_bps):
    x = df.dropna(subset=[ENTRY, exitcol]).copy()
    x["gross"] = x[exitcol] / x[ENTRY] - 1
    x["net"] = x["gross"] - cost_bps / 1e4
    g = x.groupby("d")["net"].mean()
    cal = pd.Series(0.0, index=pd.DatetimeIndex(days))
    cal.loc[g.index] = g.values
    return cal, x


def rep(s, lb, n_trades):
    s = s.dropna()
    eq = (1 + s).cumprod()
    yrs = len(s) / ANN
    sh = s.mean() / s.std() * np.sqrt(ANN) if s.std() > 0 else np.nan
    return {"決済": lb, "年率%": (eq.iloc[-1] ** (1 / yrs) - 1) * 100, "Sharpe": sh,
            "MDD%": (eq / eq.cummax() - 1).min() * 100, "N取引": n_trades,
            "N日": len(s)}


for cost in (10.0, 4.0):
    print("\n" + "=" * 92)
    print(f"STEP4: 全期間 決済タイミング別（往復 {cost:.0f}bps 控除・日次PF系列・√252）")
    print("=" * 92)
    rows = []
    for lb, col in EXITS.items():
        s, x = daily_series(sig, col, cost)
        rows.append(rep(s, lb, len(x)))
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:8.2f}"))

# ---------------------------------------------------------------- IS/OOS
print("\n" + "=" * 92)
print(f"STEP5: IS(〜{IS_END}) vs OOS/ペーパー期間({OOS_START}〜) — 往復10bps")
print("=" * 92)
rows = []
for lb, col in EXITS.items():
    s, x = daily_series(sig, col, 10.0)
    for tag, sub in [("IS", s[s.index <= IS_END]), ("OOS", s[s.index >= OOS_START])]:
        nt = len(x[(x["d"] <= IS_END)]) if tag == "IS" else len(x[x["d"] >= OOS_START])
        r = rep(sub, f"{lb} [{tag}]", nt)
        rows.append(r)
out = pd.DataFrame(rows)
print(out.to_string(index=False, float_format=lambda v: f"{v:8.2f}"))
out.to_csv(HERE / "is_oos.csv", index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------- 窓→寄成の減衰
print("\n" + "=" * 92)
sane = int(((sig["nx_open0900"] / sig["entry_adj"] - 1).abs() > 0.20).sum())
print(f"\n  調整後に |翌寄りリターン| > 20% の残存: {sane} 件（0 なら分割は潰せている）")

print("STEP6: H2 — 窓執行を寄成に落としたときの劣化（gross bps/取引）")
print("=" * 92)
g = []
for lb, col in EXITS.items():
    x = sig.dropna(subset=[ENTRY, col])
    gr = (x[col] / x[ENTRY] - 1) * 1e4
    g.append({"決済": lb, "gross bps/取引": gr.mean(), "勝率%": (gr > 0).mean() * 100,
              "中央値bps": gr.median(), "N": len(gr)})
G = pd.DataFrame(g)
print(G.to_string(index=False, float_format=lambda v: f"{v:8.2f}"))
G.to_csv(HERE / "exit_decay.csv", index=False, encoding="utf-8-sig")

sig.to_csv(HERE / "signals.csv", index=False, encoding="utf-8-sig")
print("\nsaved is_oos.csv / exit_decay.csv / signals.csv")
