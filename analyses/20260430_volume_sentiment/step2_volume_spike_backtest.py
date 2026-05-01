"""
Step 2: 出来高スパイク後の値動きパターン バックテスト（高速版）
=====================================================
分析日: 2026-04-30

出来高が時間帯平均の N倍を超えた1分足バーの後、価格はどう動くか？
  - 上昇スパイク → LONG追随
  - 下落スパイク → SHORT追随
  - 横ばいスパイク（吸収）→ 次の方向待ち

高速化: pandas shift() で前向きリターンをベクトル計算
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

HORIZONS   = [1, 5, 15, 30]
THRESHOLDS = [2.0, 3.0, 5.0]
COST_PCT   = 0.04


def load_and_prepare(sym: str) -> pd.DataFrame:
    conn = psycopg2.connect(**PG)
    df = pd.read_sql(
        f"SELECT timestamp, open, close, volume "
        f"FROM intraday_data WHERE symbol='{sym}' ORDER BY timestamp", conn)
    conn.close()
    df["jst"] = pd.to_datetime(df["timestamp"]) + pd.Timedelta(hours=9)
    df = df.dropna(subset=["close", "volume"]).set_index("jst").sort_index()

    # 取引時間のみ
    h, m = df.index.hour, df.index.minute
    df = df[(h == 9) | ((h >= 10) & (h < 11)) | ((h == 11) & (m <= 30)) |
            ((h == 12) & (m >= 30)) | ((h >= 13) & (h < 15)) | ((h == 15) & (m <= 30))].copy()

    # 時間帯別出来高平均（全期間）
    df["time_slot"] = df.index.time
    tavg = df.groupby("time_slot")["volume"].transform("mean")
    df["vol_ratio"] = df["volume"] / tavg.clip(lower=1)

    # バーのリターン
    df["bar_ret"] = (df["close"] / df["open"] - 1) * 100

    # 日付跨ぎを防ぐため、日付ラベルを付与
    df["date"] = df.index.date

    # 前向きリターン（shift）— 日付跨ぎはNaNにする
    for h in HORIZONS:
        future_close = df["close"].shift(-h)
        future_date  = df["date"].shift(-h)
        same_day = df["date"] == future_date
        df[f"fwd_{h}m"] = np.where(
            same_day,
            (future_close / df["close"] - 1) * 100,
            np.nan
        )

    return df


def spike_stats(df: pd.DataFrame, threshold: float) -> dict:
    """スパイク条件別の統計を返す"""
    spikes = df[df["vol_ratio"] >= threshold].copy()

    stats = {"n_total": len(spikes)}
    for direction, mask in [
        ("up",   spikes["bar_ret"] > 0.1),
        ("dn",   spikes["bar_ret"] < -0.1),
        ("flat", spikes["bar_ret"].between(-0.1, 0.1)),
    ]:
        sub = spikes[mask]
        stats[f"{direction}_n"] = len(sub)
        for h in HORIZONS:
            col = f"fwd_{h}m"
            vals = sub[col].dropna()
            if len(vals) < 5:
                stats[f"{direction}_{h}m_mean"] = np.nan
                stats[f"{direction}_{h}m_wr"]   = np.nan
            else:
                stats[f"{direction}_{h}m_mean"] = vals.mean()
                stats[f"{direction}_{h}m_wr"]   = (
                    (vals > 0).mean() * 100 if direction == "up" else
                    (vals < 0).mean() * 100 if direction == "dn" else
                    (vals.abs() < 0.2).mean() * 100
                )
    return stats


# ─────────────────────────────────────────────────────
print("=" * 85)
print("  Step 2: 出来高スパイク後の値動きパターン バックテスト")
print("=" * 85)
print("  ロード・計算中...")

pool: dict[float, dict] = {thr: {"up": [], "dn": [], "flat": []} for thr in THRESHOLDS}
sym_results = []

for sym, (name, sector) in ALL_SYMS.items():
    print(f"    {name}...", end=" ", flush=True)
    df = load_and_prepare(sym)

    for thr in THRESHOLDS:
        st = spike_stats(df, thr)
        spikes = df[df["vol_ratio"] >= thr]
        for direction, bar_mask in [
            ("up",   spikes["bar_ret"] > 0.1),
            ("dn",   spikes["bar_ret"] < -0.1),
            ("flat", spikes["bar_ret"].between(-0.1, 0.1)),
        ]:
            sub = spikes[bar_mask]
            for h in HORIZONS:
                vals = sub[f"fwd_{h}m"].dropna().tolist()
                pool[thr][direction].extend(vals)

        sym_results.append({
            "sym": sym, "name": name, "sector": sector, "thr": thr, **st
        })

    print("OK")

# ─────────────────────────────────────────────────────
# 1. 閾値別・全銘柄プール
# ─────────────────────────────────────────────────────
print("\n" + "=" * 85)
print("【全銘柄プール: 出来高スパイク後のリターン — 閾値別】")
print("  LONG = 上昇スパイク追随   SHORT = 下落スパイク追随")
print("=" * 85)

for thr in THRESHOLDS:
    p = pool[thr]
    total = sum(len(v) for v in p.values())
    print(f"\n  ─ 出来高スパイク ≥ {thr:.0f}x  (総イベント数: {total:,}) ─")
    print(f"  {'タイプ':<16}  N     " +
          "  ".join(f"{'↑'+str(h)+'m avg':>12}  {'勝率':>6}" for h in HORIZONS))
    print("  " + "-" * 85)

    for label, direction, sign in [
        ("上昇スパイク(LONG)",   "up",   1),
        ("下落スパイク(SHORT)",  "dn",  -1),
        ("横ばいスパイク(吸収)", "flat", 0),
    ]:
        vals_15 = np.array(p[direction]) if p[direction] else np.array([])
        n = len(vals_15)
        line = f"  {label:<16}  {n:>5}"
        for h in HORIZONS:
            # 15分だけ詳細計算（他はpool構造の都合でskip）
            pass
        # 各ホライズン別に再計算
        hz_stats = []
        for h in HORIZONS:
            all_r = []
            for sym_res in sym_results:
                if sym_res["thr"] != thr:
                    continue
                v = sym_res.get(f"{direction}_{h}m_mean", np.nan)
                n_r = sym_res.get(f"{direction}_n", 0)
                if v == v and n_r:
                    all_r.extend([v] * int(n_r))
            if all_r:
                arr = np.array(all_r)
                mean_v = arr.mean()
                wr_v   = (arr > 0).mean() * 100 if sign >= 0 else (arr < 0).mean() * 100
                hz_stats.append(f"  {mean_v:>+9.3f}%  {wr_v:>5.1f}%")
            else:
                hz_stats.append(f"  {'---':>10}  {'---':>5}")
        line += "".join(hz_stats)
        print(line)

# ─────────────────────────────────────────────────────
# 2. 銘柄別 3x スパイク後15分
# ─────────────────────────────────────────────────────
print("\n" + "=" * 85)
print("【銘柄別: 出来高3x超 スパイク後15分リターン】")
print("  上昇スパイク追随(LONG) / 下落スパイク追随(SHORT)")
print("=" * 85)
print(f"  {'銘柄':<12} {'セクター':<8}  "
      f"{'上N':>5}  {'上15m':>8}  {'勝率':>6}  "
      f"{'下N':>5}  {'下15m':>8}  {'逆勝率':>6}  "
      f"{'横N':>5}  {'横15m':>8}")
print("  " + "-" * 85)

tgt = 3.0
for sym, (name, sector) in ALL_SYMS.items():
    r = next((x for x in sym_results if x["sym"] == sym and x["thr"] == tgt), None)
    if not r:
        continue
    up_n  = r.get("up_n", 0)
    up_m  = r.get("up_15m_mean", np.nan)
    up_wr = r.get("up_15m_wr", np.nan)
    dn_n  = r.get("dn_n", 0)
    dn_m  = r.get("dn_15m_mean", np.nan)
    dn_wr = r.get("dn_15m_wr", np.nan)
    fl_n  = r.get("flat_n", 0)
    fl_m  = r.get("flat_15m_mean", np.nan)

    fmt_m  = lambda v: f"{v:>+7.3f}%" if v == v else "    ---"
    fmt_wr = lambda v: f"{v:>5.1f}%" if v == v else "  ---"

    up_mark = " ◎" if (up_m == up_m and up_m > 0.1 and up_wr and up_wr > 55) else ""
    dn_mark = " ◎" if (dn_m == dn_m and dn_m < -0.1 and dn_wr and dn_wr > 55) else ""

    print(f"  {name:<12} {sector:<8}  "
          f"{up_n:>5}  {fmt_m(up_m)}  {fmt_wr(up_wr)}  "
          f"{dn_n:>5}  {fmt_m(dn_m)}  {fmt_wr(dn_wr)}  "
          f"{fl_n:>5}  {fmt_m(fl_m)}"
          f"{up_mark}{dn_mark}")

# ─────────────────────────────────────────────────────
# 3. 時間帯別スパイク特性（寄付直後 vs 午後）
# ─────────────────────────────────────────────────────
print("\n" + "=" * 85)
print("【時間帯別スパイク特性（3x超）】")
print("  寄付直後（9:00-9:30）は最もスパイクが多く、その後はどう動くか？")
print("=" * 85)

time_buckets = [
    ("9:00-9:30",  (9,  0), (9, 30)),
    ("9:30-10:30", (9, 30), (10,30)),
    ("10:30-11:30",(10,30), (11,30)),
    ("12:30-13:30",(12,30), (13,30)),
    ("13:30-15:00",(13,30), (15, 0)),
    ("15:00-15:30",(15, 0), (15,30)),
]

print(f"  {'時間帯':<14}  {'N':>6}  {'上昇比率':>9}  {'上昇後15m':>10}  {'下落後15m':>10}")
print("  " + "-" * 60)

# 全銘柄の1分足を再ロードして時間帯別集計
all_up_by_time   = {b[0]: [] for b in time_buckets}
all_dn_by_time   = {b[0]: [] for b in time_buckets}
all_n_by_time    = {b[0]: 0  for b in time_buckets}

for sym in list(ALL_SYMS.keys())[:8]:   # 代表8銘柄で高速化
    df = load_and_prepare(sym)
    spikes = df[df["vol_ratio"] >= 3.0].copy()
    for label, (sh, sm), (eh, em) in time_buckets:
        mask = (
            ((spikes.index.hour > sh) | ((spikes.index.hour == sh) & (spikes.index.minute >= sm))) &
            ((spikes.index.hour < eh) | ((spikes.index.hour == eh) & (spikes.index.minute < em)))
        )
        sub = spikes[mask]
        all_n_by_time[label] += len(sub)
        up = sub[sub["bar_ret"] > 0.1]["fwd_15m"].dropna().tolist()
        dn = sub[sub["bar_ret"] < -0.1]["fwd_15m"].dropna().tolist()
        all_up_by_time[label].extend(up)
        all_dn_by_time[label].extend(dn)

for label, _, _ in time_buckets:
    n = all_n_by_time[label]
    up_arr = np.array(all_up_by_time[label]) if all_up_by_time[label] else np.array([])
    dn_arr = np.array(all_dn_by_time[label]) if all_dn_by_time[label] else np.array([])
    up_pct = len(up_arr) / n * 100 if n > 0 else np.nan
    up_m   = up_arr.mean() if len(up_arr) > 5 else np.nan
    dn_m   = dn_arr.mean() if len(dn_arr) > 5 else np.nan
    fmt = lambda v: f"{v:>+9.3f}%" if v == v else "       ---"
    print(f"  {label:<14}  {n:>6}  {up_pct:>8.1f}%  {fmt(up_m)}  {fmt(dn_m)}")

# ─────────────────────────────────────────────────────
# 4. 総括・実運用ガイド
# ─────────────────────────────────────────────────────
print("\n" + "=" * 85)
print("【総括: 出来高スパイク戦略の有効性】")
print("=" * 85)

# 全閾値での上昇スパイク15分リターン
for thr in THRESHOLDS:
    vals = []
    for sr in sym_results:
        if sr["thr"] != thr:
            continue
        v = sr.get("up_15m_mean", np.nan)
        n = sr.get("up_n", 0)
        if v == v and n > 0:
            vals.extend([v] * int(n))
    if vals:
        arr = np.array(vals)
        net = arr - COST_PCT
        print(f"  {thr:.0f}x閾値 上昇スパイク追随: N={len(arr):>6,}  "
              f"gross平均={arr.mean():>+7.3f}%  net={net.mean():>+7.3f}%  "
              f"勝率={(arr>0).mean()*100:>5.1f}%")

print()
print("  ─ 結論 ─")
print("  1. 1分足スパイク単発への機械的追随は コスト後ほぼゼロ〜マイナス")
print("  2. 出来高スパイクは「情報」だが即エントリーのシグナルではない")
print("  3. BNFが板読みで『スパイクの質』を判断するのはこのため")
print("     → 1分足だけでは買い吸収 vs 売り吸収が判別できない")
print("  4. 有効な活用法: 大商い急騰の『翌日継続』（Step3参照）")
print("     → 当日スパイクより翌日の方向確認に使う")

print("\n  ✅ Step2 完了")
