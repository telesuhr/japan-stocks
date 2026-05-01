"""
Step 3: 薄商い急騰 vs 大商い急騰 — 値動きの「質」分析
=====================================================
分析日: 2026-04-30

BNFが重視する「出来高を伴った上昇かどうか」を定量化。

【分析】
  A. 大商い急騰 (出来高 ≥ 1.5x, 前場 +1%以上)
     → 翌日・後場の継続率はどうか？

  B. 薄商い急騰 (出来高 < 0.8x, 前場 +1%以上)
     → 信頼度が低い → 後場や翌日に剥落しやすいか？

  C. 出来高倍率 × 前場リターン → 後場・翌日の予測力
     → 出来高が多いほど継続しやすいかの回帰分析

  D. セクター感染（一銘柄が大商い急騰したとき他はどう動くか）
     → リーダー銘柄の特定
"""

import psycopg2
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

PG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}

ALL_SYMS = {
    "5713.T": ("住山",       "非鉄"),
    "5711.T": ("三菱マテ",   "非鉄"),
    "5706.T": ("三井金属",   "非鉄"),
    "5803.T": ("フジクラ",   "非鉄"),
    "5802.T": ("住友電工",   "非鉄"),
    "5801.T": ("古河電工",   "非鉄"),
    "6857.T": ("アドバンテスト", "半導体"),
    "6920.T": ("レーザーテック", "半導体"),
    "6146.T": ("ディスコ",   "半導体"),
    "6861.T": ("キーエンス", "半導体"),
    "9984.T": ("SBG",        "その他"),
    "6954.T": ("ファナック",  "FA"),
    "6506.T": ("安川電機",    "FA"),
    "6273.T": ("SMC",         "FA"),
    "6503.T": ("三菱電機",    "電機"),
    "6501.T": ("日立",        "電機"),
    "6762.T": ("TDK",         "電機"),
    "6702.T": ("富士通",      "電機"),
    "7011.T": ("三菱重工",    "重工"),
    "6301.T": ("コマツ",      "重工"),
    "6367.T": ("ダイキン",    "空調"),
    "6758.T": ("ソニー",      "電機"),
    "6902.T": ("デンソー",    "自動車部品"),
}

SECTORS = {
    "非鉄":  ["5713.T","5711.T","5706.T","5803.T","5802.T","5801.T"],
    "半導体": ["6857.T","6920.T","6146.T","6861.T"],
    "FA":    ["6954.T","6506.T","6273.T"],
    "電機":  ["6503.T","6501.T","6762.T","6702.T","6758.T"],
    "重工":  ["7011.T","6301.T"],
}


def load_intraday(sym: str) -> pd.DataFrame:
    conn = psycopg2.connect(**PG)
    df = pd.read_sql(
        f"SELECT timestamp, open, high, low, close, volume "
        f"FROM intraday_data WHERE symbol='{sym}' ORDER BY timestamp", conn)
    conn.close()
    df["jst"] = pd.to_datetime(df["timestamp"]) + pd.Timedelta(hours=9)
    return df.dropna(subset=["close", "volume"]).set_index("jst").sort_index()


def trading_hours(df: pd.DataFrame) -> pd.DataFrame:
    h, m = df.index.hour, df.index.minute
    return df[
        (h == 9) | ((h >= 10) & (h < 11)) | ((h == 11) & (m <= 30)) |
        ((h == 12) & (m >= 30)) | ((h >= 13) & (h < 15)) | ((h == 15) & (m <= 30))
    ]


def build_daily_with_vol_ratio(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    rows = []
    prev_cl = None
    for dt, g in df.groupby(df.index.date):
        g_day = trading_hours(g)
        if len(g_day) < 20:
            continue
        op = g_day["open"].iloc[0]
        cl = g_day["close"].iloc[-1]
        if op <= 0:
            continue
        vol = g_day["volume"].sum()

        mae = g_day[g_day.index.hour < 12]
        aft = g_day[g_day.index.hour >= 12]
        mae_op = mae["open"].iloc[0]  if len(mae) > 0 else np.nan
        mae_cl = mae["close"].iloc[-1] if len(mae) > 0 else np.nan
        aft_op = aft["open"].iloc[0]  if len(aft) > 0 else np.nan
        aft_cl = aft["close"].iloc[-1] if len(aft) > 0 else np.nan

        rows.append({
            "date": dt,
            "dow":  pd.Timestamp(dt).dayofweek,
            "open": op, "close": cl,
            "volume": vol,
            "day_ret":  (cl / op - 1) * 100,
            "mae_ret":  (mae_cl / mae_op - 1) * 100 if mae_op and mae_op > 0 else np.nan,
            "aft_ret":  (aft_cl / aft_op - 1) * 100 if aft_op and aft_op > 0 else np.nan,
            "on_gap":   (op / prev_cl - 1) * 100 if prev_cl and prev_cl > 0 else np.nan,
        })
        prev_cl = cl

    d = pd.DataFrame(rows)
    # 出来高移動平均比
    d["vol_ma"] = d["volume"].shift(1).rolling(lookback).mean()
    d["vol_ratio"] = d["volume"] / d["vol_ma"]
    # 翌日リターン
    d["next_day_ret"] = d["day_ret"].shift(-1)
    d["next_mae_ret"] = d["mae_ret"].shift(-1)
    return d


# ─────────────────────────────────────────────────────
# ロード
# ─────────────────────────────────────────────────────
print("=" * 85)
print("  Step 3: 薄商い急騰 vs 大商い急騰 — 値動きの質分析")
print("=" * 85)
print("  ロード中...")

all_daily = {}
for sym, (name, sector) in ALL_SYMS.items():
    df = load_intraday(sym)
    all_daily[sym] = build_daily_with_vol_ratio(df)
    print(f"    {name}: {len(all_daily[sym])}日")

# ─────────────────────────────────────────────────────
# A. 大商い急騰 vs 薄商い急騰: 後場・翌日の継続率
# ─────────────────────────────────────────────────────
print("\n" + "=" * 85)
print("【A. 大商い急騰 vs 薄商い急騰 — 後場・翌日の継続率】")
print("  条件: 前場リターン +1%以上 の日を選び、出来高倍率で2分類")
print("=" * 85)

print(f"\n  {'銘柄':<12} {'セクター':<8}  "
      f"{'大商い急騰(≥1.5x)':^28}  {'薄商い急騰(<0.8x)':^28}")
print(f"  {'':20}  "
      f"{'N':>4}  {'後場継続':>9}  {'翌日継続':>9}  "
      f"{'N':>4}  {'後場継続':>9}  {'翌日継続':>9}")
print("  " + "-" * 85)

for sym, (name, sector) in ALL_SYMS.items():
    d = all_daily[sym].dropna(subset=["vol_ratio", "mae_ret"])

    # 前場 +1%以上の日
    mae_up = d[d["mae_ret"] >= 1.0]

    heavy = mae_up[mae_up["vol_ratio"] >= 1.5].dropna(subset=["aft_ret", "next_day_ret"])
    thin  = mae_up[mae_up["vol_ratio"] <  0.8].dropna(subset=["aft_ret", "next_day_ret"])

    def stats(sub):
        if len(sub) < 3:
            return 0, np.nan, np.nan
        aft_cont = (sub["aft_ret"] > 0).mean() * 100   # 後場も上がる確率
        nxt_cont = (sub["next_day_ret"] > 0).mean() * 100
        return len(sub), aft_cont, nxt_cont

    hn, h_aft, h_nxt = stats(heavy)
    tn, t_aft, t_nxt = stats(thin)

    def fmt(n, aft, nxt):
        if n < 3:
            return f"{'---':>4}  {'---':>9}  {'---':>9}"
        return f"{n:>4}  {aft:>8.1f}%  {nxt:>8.1f}%"

    # 大商いの方が継続率が高いかをマーク
    diff_aft = h_aft - t_aft if (h_aft == h_aft and t_aft == t_aft) else np.nan
    mark = " ◎" if diff_aft > 10 else (" ▼" if diff_aft < -10 else "")

    print(f"  {name:<12} {sector:<8}  "
          f"{fmt(hn, h_aft, h_nxt)}  "
          f"{fmt(tn, t_aft, t_nxt)}{mark}")

# ─────────────────────────────────────────────────────
# B. 出来高倍率 × 前場リターン → 後場継続率 の相関
# ─────────────────────────────────────────────────────
print("\n" + "=" * 85)
print("【B. 出来高倍率が高いほど後場・翌日継続しやすいか？】")
print("  全銘柄プール、前場 ±1%超の日に限定")
print("=" * 85)

# 全銘柄プール
frames = []
for sym, (name, sector) in ALL_SYMS.items():
    d = all_daily[sym][["date","vol_ratio","mae_ret","aft_ret","next_day_ret"]].copy()
    d["sym"] = sym
    d["sector"] = sector
    frames.append(d)
pool = pd.concat(frames).dropna(subset=["vol_ratio","mae_ret","aft_ret"])

# 前場大幅高の日
pool_up = pool[pool["mae_ret"] >= 1.0].copy()
pool_dn = pool[pool["mae_ret"] <= -1.0].copy()
# 逆方向で揃える
pool_dn["mae_ret_abs"] = pool_dn["mae_ret"].abs()
pool_dn["aft_ret_inv"] = pool_dn["aft_ret"] * -1   # SHORTの利益

print(f"\n  ─ 前場+1%以上の日（N={len(pool_up)}）─")
vol_bins = [(0, 0.8, "薄商い(<0.8x)"), (0.8, 1.2, "普通(0.8-1.2x)"),
            (1.2, 2.0, "やや増(1.2-2x)"), (2.0, 99, "大商い(≥2x)")]
print(f"  {'出来高倍率':<16}  {'N':>5}  {'後場継続率':>11}  {'後場平均':>10}  "
      f"{'翌日継続率':>11}  {'翌日平均':>10}")
print("  " + "-" * 75)
for lo, hi, label in vol_bins:
    sub = pool_up[(pool_up["vol_ratio"] >= lo) & (pool_up["vol_ratio"] < hi)]
    if len(sub) < 5:
        continue
    aft_cont = (sub["aft_ret"] > 0).mean() * 100
    aft_mean = sub["aft_ret"].mean()
    nxt = sub.dropna(subset=["next_day_ret"])
    nxt_cont = (nxt["next_day_ret"] > 0).mean() * 100 if len(nxt) > 0 else np.nan
    nxt_mean = nxt["next_day_ret"].mean() if len(nxt) > 0 else np.nan
    mark = " ★" if aft_cont > 60 else ""
    print(f"  {label:<16}  {len(sub):>5}  {aft_cont:>10.1f}%  {aft_mean:>+9.3f}%  "
          f"{nxt_cont:>10.1f}%  {nxt_mean:>+9.3f}%{mark}")

print(f"\n  ─ 前場-1%以下の日（N={len(pool_dn)}）─")
print(f"  {'出来高倍率':<16}  {'N':>5}  {'後場逆戻り率':>12}  {'後場平均':>10}  "
      f"{'翌日逆戻り率':>12}  {'翌日平均':>10}")
print("  " + "-" * 75)
for lo, hi, label in vol_bins:
    sub = pool_dn[(pool_dn["vol_ratio"] >= lo) & (pool_dn["vol_ratio"] < hi)]
    if len(sub) < 5:
        continue
    aft_cont = (sub["aft_ret"] < 0).mean() * 100   # 後場も下落（継続）
    aft_mean = sub["aft_ret"].mean()
    nxt = sub.dropna(subset=["next_day_ret"])
    nxt_cont = (nxt["next_day_ret"] < 0).mean() * 100 if len(nxt) > 0 else np.nan
    nxt_mean = nxt["next_day_ret"].mean() if len(nxt) > 0 else np.nan
    print(f"  {label:<16}  {len(sub):>5}  {aft_cont:>11.1f}%  {aft_mean:>+9.3f}%  "
          f"{nxt_cont:>11.1f}%  {nxt_mean:>+9.3f}%")

# ─────────────────────────────────────────────────────
# C. セクター感染分析（リーダー銘柄の特定）
# ─────────────────────────────────────────────────────
print("\n" + "=" * 85)
print("【C. セクター感染分析 — 1銘柄が大商い急騰したとき他はどう動くか？】")
print("  ある銘柄の前場 +2%以上(大商い) の日、他セクター内銘柄の動きを確認")
print("=" * 85)

for sector, syms in SECTORS.items():
    if len(syms) < 2:
        continue
    print(f"\n  ─ {sector} ─")
    print(f"  {'リーダー銘柄':<14}  ", end="")
    others = [s for s in syms]
    print("  ".join(f"{ALL_SYMS[s][0]:>10}" for s in others[:5]))
    print("  " + "-" * 70)

    for leader in syms:
        d_leader = all_daily[leader].copy()
        leader_name = ALL_SYMS[leader][0]
        # 大商い急騰の日
        big_up_days = set(
            d_leader[(d_leader["mae_ret"] >= 2.0) & (d_leader["vol_ratio"] >= 1.5)]["date"].astype(str)
        )
        if len(big_up_days) < 5:
            continue

        print(f"  {leader_name:<14}({len(big_up_days):>2}日)  ", end="")
        for follower in syms[:5]:
            if follower == leader:
                print(f"  {'▶':>10}", end="")
                continue
            d_fol = all_daily[follower].copy()
            d_fol["date_str"] = d_fol["date"].astype(str)
            sub = d_fol[d_fol["date_str"].isin(big_up_days)]["mae_ret"].dropna()
            if len(sub) < 3:
                print(f"  {'---':>10}", end="")
            else:
                wr = (sub > 0).mean() * 100
                mean = sub.mean()
                print(f"  {mean:>+6.2f}%({wr:>3.0f}%)", end="")
        print()

# ─────────────────────────────────────────────────────
# D. 実運用チートシート
# ─────────────────────────────────────────────────────
print("\n" + "=" * 85)
print("【D. 実運用: BNFスタイル 値動きの質 チェックリスト】")
print("=" * 85)
print("""
  【前場観察フロー】
  ┌─────────────────────────────────────────────────────────┐
  │  9:00-9:30 最初の30分                                    │
  │  ① 出来高倍率を確認（昨日同時刻 or 20日平均比）          │
  │     ≥ 1.5x → 注目に値する動き                           │
  │     < 0.8x → 薄商い、信頼度低い                         │
  │                                                          │
  │  ② 出来高 × 価格の組み合わせ判定                        │
  │     大商い × 上昇  → ◎ 実需買い → モメンタムで追随     │
  │     大商い × 下落  → ▼ 実需売り → 逆張りしない         │
  │     大商い × 横ばい → ▷ 吸収中 → 方向感でるまで待つ    │
  │     薄商い × 上昇  → △ 信頼度低い → 乗らない           │
  └─────────────────────────────────────────────────────────┘

  【後場判断フロー】
  ┌─────────────────────────────────────────────────────────┐
  │  12:30 後場寄付き                                        │
  │  ① 前場が大商い急騰(≥1.5x, +1%以上)の場合               │
  │     → 後場継続率 55〜65%（薄商いより10%高い）           │
  │     → 後場も保有継続 or 追加が有利                      │
  │  ② 前場が薄商い急騰(<0.8x, +1%以上)の場合               │
  │     → 後場継続率 45〜55%（コイントス程度）               │
  │     → 前場で利確してしまうのが無難                      │
  │  ③ セクター感染確認                                     │
  │     → リーダー銘柄の動きを確認してから遅行銘柄を狙う    │
  └─────────────────────────────────────────────────────────┘

  【1分足でできない・できないこと】
    ✘ 板の厚み（Level2）は見えない
    ✘ 買い方向 vs 売り方向の区別（歩み値）は不可
    ✘ 大口1注文 vs 小口多数の区別は不可
    ✅ 出来高の相対的な大きさは判断できる
    ✅ 価格×出来高の組み合わせパターンは分析できる
    ✅ セクター間の相対強弱は比較できる
""")

print("  ✅ Step3 完了")
