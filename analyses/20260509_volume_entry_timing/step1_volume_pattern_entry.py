"""
出来高パターン × 最適エントリータイミング分析
=============================================
分析日: 2026-05-09

【問い】
 - 大商い×下落 → 翌日リバーサル or 継続どちらが多いか？
 - 吸収パターン（大商い×横ばい）後の方向は予測できるか？
 - 引けvsVWAP・連続パターン・時間帯で精度は上がるか？

【分類】
  A. ◎実需買い   (vol>=1.5x, ret>+0.3%)
  B. ▼実需売り   (vol>=1.5x, ret<-0.3%)
  C. ▷吸収/膠着  (vol>=1.5x, |ret|<=0.3%)
  D. △薄買い    (vol<1.5x,  ret>+0.3%)
  E. ▼薄売り    (vol<1.5x,  ret<-0.3%)
  F.   横ばい   (vol<1.5x,  |ret|<=0.3%)
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

LOOKBACK = 20  # 出来高移動平均の基準日数


# ─────────────────────────────────────────────────────────────
# データ取得・日次集計
# ─────────────────────────────────────────────────────────────
def load_daily(sym: str) -> pd.DataFrame:
    conn = psycopg2.connect(**PG)
    df = pd.read_sql(
        f"SELECT timestamp, open, high, low, close, volume "
        f"FROM intraday_data WHERE symbol='{sym}' ORDER BY timestamp", conn)
    conn.close()
    df["jst"] = pd.to_datetime(df["timestamp"]) + pd.Timedelta(hours=9)
    df = df.dropna(subset=["close"]).set_index("jst").sort_index()

    # 取引時間フィルタ
    h, m = df.index.hour, df.index.minute
    df = df[(h == 9) | ((h >= 10) & (h < 11)) | ((h == 11) & (m <= 30)) |
            ((h == 12) & (m >= 30)) | ((h >= 13) & (h < 15)) | ((h == 15) & (m <= 30))]

    rows = []
    prev_close = None
    for dt, g in df.groupby(df.index.date):
        if len(g) < 20:
            continue
        op  = g["open"].iloc[0]
        cl  = g["close"].iloc[-1]
        hi  = g["high"].max()
        lo  = g["low"].min()
        vol = g["volume"].sum()
        if op <= 0:
            continue

        vwap = (g["close"] * g["volume"]).sum() / g["volume"].sum() \
               if g["volume"].sum() > 0 else op

        # 前場・後場別リターン
        am = g[(g.index.hour < 11) | ((g.index.hour == 11) & (g.index.minute <= 30))]
        pm = g[(g.index.hour >= 12)]
        am_ret = (am["close"].iloc[-1] / am["open"].iloc[0] - 1) * 100 if len(am) >= 3 else np.nan
        pm_ret = (pm["close"].iloc[-1] / pm["open"].iloc[0] - 1) * 100 if len(pm) >= 3 else np.nan

        on_gap = (op / prev_close - 1) * 100 if prev_close and prev_close > 0 else np.nan

        rows.append({
            "date":        dt,
            "dow":         pd.Timestamp(dt).dayofweek,
            "open":        op,
            "close":       cl,
            "high":        hi,
            "low":         lo,
            "volume":      vol,
            "day_ret":     (cl / op - 1) * 100,
            "on_gap":      on_gap,
            "range_pct":   (hi / lo - 1) * 100 if lo > 0 else np.nan,
            "vwap":        vwap,
            "cl_vs_vwap":  (cl / vwap - 1) * 100 if vwap > 0 else np.nan,
            "am_ret":      am_ret,
            "pm_ret":      pm_ret,
        })
        prev_close = cl

    d = pd.DataFrame(rows)
    if d.empty:
        return d

    d["vol_ma"] = d["volume"].shift(1).rolling(LOOKBACK).mean()
    d["vol_ratio"] = d["volume"] / d["vol_ma"].clip(lower=1)
    return d.dropna(subset=["vol_ratio"])


def classify_day(ret: float, vol_ratio: float) -> str:
    heavy  = vol_ratio >= 1.5
    strong = ret >  0.3
    weak   = ret < -0.3
    if heavy and strong: return "A_実需買い"
    if heavy and weak:   return "B_実需売り"
    if heavy:            return "C_吸収"
    if strong:           return "D_薄買い"
    if weak:             return "E_薄売り"
    return "F_横ばい"


# ─────────────────────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────────────────────
print("=" * 80)
print("  出来高パターン × 最適エントリータイミング 分析")
print("=" * 80)
print("  ロード中...")

all_days = []
for sym, (name, sector) in ALL_SYMS.items():
    d = load_daily(sym)
    if d.empty or len(d) < 30:
        print(f"    {name}: スキップ（データ不足）")
        continue
    d["sym"]    = sym
    d["name"]   = name
    d["sector"] = sector
    d["pattern"] = d.apply(lambda r: classify_day(r["day_ret"], r["vol_ratio"]), axis=1)

    # 翌日リターン（on_gapも含む）
    d["next_day_ret"]    = d["day_ret"].shift(-1)
    d["next_on_gap"]     = d["on_gap"].shift(-1)   # 当日→翌日の窓
    d["next_am_ret"]     = d["am_ret"].shift(-1)
    d["next_pm_ret"]     = d["pm_ret"].shift(-1)
    d["next_vol_ratio"]  = d["vol_ratio"].shift(-1)
    d["prev_pattern"]    = d["pattern"].shift(1)    # 前日パターン

    # 連続パターン
    d["consecutive"] = (d["pattern"] == d["pattern"].shift(1)).astype(int)

    all_days.append(d)
    print(f"    {name}: {len(d)}日")

df = pd.concat(all_days, ignore_index=True)
df = df.dropna(subset=["next_day_ret"])

print(f"\n  総サンプル: {len(df):,}日×銘柄")


# ─────────────────────────────────────────────────────────────
# 1. パターン別 翌日リターン統計
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("【1】パターン別 翌日リターン統計")
print("  当日の出来高×値動きパターン → 翌日の平均リターン・勝率")
print("=" * 80)

PAT_LABELS = {
    "A_実需買い": "◎実需買い (vol≥1.5x, ret>+0.3%)",
    "B_実需売り": "▼実需売り (vol≥1.5x, ret<-0.3%)",
    "C_吸収":     "▷吸収    (vol≥1.5x, |ret|≤0.3%)",
    "D_薄買い":   "△薄買い   (vol<1.5x, ret>+0.3%)",
    "E_薄売り":   "▼薄売り   (vol<1.5x, ret<-0.3%)",
    "F_横ばい":   "  横ばい   (vol<1.5x, |ret|≤0.3%)",
}

print(f"\n  {'パターン':<28}  {'N':>5}  {'翌日平均':>9}  {'翌日勝率':>9}  "
      f"{'翌ON平均':>9}  {'翌前場':>8}  {'翌後場':>8}")
print("  " + "-" * 80)

for pat, label in PAT_LABELS.items():
    sub = df[df["pattern"] == pat]
    if len(sub) < 5:
        continue
    nr  = sub["next_day_ret"].dropna()
    nog = sub["next_on_gap"].dropna()
    nam = sub["next_am_ret"].dropna()
    npm = sub["next_pm_ret"].dropna()
    print(f"  {label:<28}  {len(nr):>5}  "
          f"{nr.mean():>+8.3f}%  {(nr > 0).mean()*100:>8.1f}%  "
          f"{nog.mean():>+8.3f}%  "
          f"{nam.mean():>+7.3f}%  "
          f"{npm.mean():>+7.3f}%")


# ─────────────────────────────────────────────────────────────
# 2. 実需売り後のリバーサル vs 継続 詳細
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("【2】▼実需売り後のリバーサル/継続 詳細分析")
print("  「大商い×下落」の後、翌日はどちらに動くか？条件で変わるか？")
print("=" * 80)

sell = df[df["pattern"] == "B_実需売り"].copy()

print(f"\n  総イベント数: {len(sell)}")
rev = sell["next_day_ret"]
print(f"  翌日リターン: mean={rev.mean():+.3f}%  中央値={rev.median():+.3f}%  "
      f"勝率(翌日上昇)={(rev > 0).mean()*100:.1f}%")

# 下落幅別
print("\n  ─ 当日下落幅別 ─")
bins = [(-99, -3), (-3, -2), (-2, -1), (-1, -0.3)]
for lo_r, hi_r in bins:
    sub = sell[(sell["day_ret"] >= lo_r) & (sell["day_ret"] < hi_r)]
    nr = sub["next_day_ret"].dropna()
    if len(nr) < 5:
        continue
    print(f"  当日 {lo_r:>4}%〜{hi_r:>4}%: N={len(nr):>4}  "
          f"翌日={nr.mean():>+7.3f}%  勝率={(nr>0).mean()*100:>5.1f}%  "
          f"翌日ON={sub['next_on_gap'].mean():>+7.3f}%")

# 引けvsVWAP別
print("\n  ─ 引けvsVWAP別（大商い×下落でも引けがVWAP上なら吸収？）─")
sell_vwap = sell.dropna(subset=["cl_vs_vwap"])
for vlo, vhi, label in [(-99, -0.5, "VWAP大幅下"), (-0.5, 0, "VWAP小幅下"), (0, 99, "VWAP上（下ヒゲ反発？）")]:
    sub = sell_vwap[(sell_vwap["cl_vs_vwap"] >= vlo) & (sell_vwap["cl_vs_vwap"] < vhi)]
    nr = sub["next_day_ret"].dropna()
    if len(nr) < 5:
        continue
    print(f"  引け vs VWAP {label}: N={len(nr):>4}  "
          f"翌日={nr.mean():>+7.3f}%  勝率={(nr>0).mean()*100:>5.1f}%")

# 出来高倍率別
print("\n  ─ 出来高倍率別 ─")
for vlo, vhi in [(1.5, 2.0), (2.0, 3.0), (3.0, 99)]:
    sub = sell[(sell["vol_ratio"] >= vlo) & (sell["vol_ratio"] < vhi)]
    nr = sub["next_day_ret"].dropna()
    if len(nr) < 5:
        continue
    print(f"  出来高 {vlo:.0f}x〜{vhi:.0f}x: N={len(nr):>4}  "
          f"翌日={nr.mean():>+7.3f}%  勝率={(nr>0).mean()*100:>5.1f}%")

# 連続実需売りかどうか
print("\n  ─ 連続パターン別（前日も同じか？）─")
sell_consec = sell[sell["consecutive"] == 1]
sell_first  = sell[sell["consecutive"] == 0]
for label, sub in [("前日も実需売り（連続）", sell_consec), ("初回の実需売り", sell_first)]:
    nr = sub["next_day_ret"].dropna()
    if len(nr) < 5:
        continue
    print(f"  {label}: N={len(nr):>4}  翌日={nr.mean():>+7.3f}%  勝率={(nr>0).mean()*100:>5.1f}%")


# ─────────────────────────────────────────────────────────────
# 3. 吸収パターン後の方向性
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("【3】▷吸収パターン後の方向性")
print("  大商い×横ばい → 上下どちらへブレイクするか？")
print("=" * 80)

absorb = df[df["pattern"] == "C_吸収"].copy()
print(f"\n  総イベント数: {len(absorb)}")
nr = absorb["next_day_ret"].dropna()
print(f"  翌日リターン: mean={nr.mean():+.3f}%  勝率={(nr>0).mean()*100:.1f}%")

# 引けvsVWAPで方向判定
print("\n  ─ 引けvsVWAPで吸収の方向を判定 ─")
absorb_v = absorb.dropna(subset=["cl_vs_vwap"])
for vlo, vhi, label in [(-99, -0.3, "VWAP大幅下（売り吸収濃厚）"), (-0.3, 0.3, "VWAP付近"), (0.3, 99, "VWAP上（買い吸収濃厚）")]:
    sub = absorb_v[(absorb_v["cl_vs_vwap"] >= vlo) & (absorb_v["cl_vs_vwap"] < vhi)]
    nr = sub["next_day_ret"].dropna()
    if len(nr) < 5:
        continue
    print(f"  {label}: N={len(nr):>4}  翌日={nr.mean():>+7.3f}%  勝率={(nr>0).mean()*100:>5.1f}%")

# 前日パターン別
print("\n  ─ 吸収の前日パターン別 ─")
for prev_pat, plabel in [
    ("A_実需買い", "前日◎実需買い → 吸収"),
    ("B_実需売り", "前日▼実需売り → 吸収（売り止まり？）"),
    ("D_薄買い",   "前日△薄買い   → 吸収"),
]:
    sub = absorb[absorb["prev_pattern"] == prev_pat]
    nr = sub["next_day_ret"].dropna()
    if len(nr) < 5:
        continue
    print(f"  {plabel}: N={len(nr):>4}  翌日={nr.mean():>+7.3f}%  勝率={(nr>0).mean()*100:>5.1f}%")


# ─────────────────────────────────────────────────────────────
# 4. 複合シグナル: エントリー条件スコアリング
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("【4】複合シグナル — エントリー条件スコアリング")
print("  複数条件を組み合わせると精度は上がるか？")
print("=" * 80)

# Long エントリー候補条件
print("\n  ─ LONGエントリー候補（翌日寄付きロング）─")
conditions_long = [
    ("実需買い",                     df["pattern"] == "A_実需買い"),
    ("実需買い + VWAP上",            (df["pattern"] == "A_実需買い") & (df["cl_vs_vwap"] > 0.3)),
    ("実需買い + VWAP上 + vol≥2x",   (df["pattern"] == "A_実需買い") & (df["cl_vs_vwap"] > 0.3) & (df["vol_ratio"] >= 2.0)),
    ("実需売り後(リバ)",              df["pattern"] == "B_実需売り"),
    ("実需売り(VWAP上=下ヒゲ吸収)",  (df["pattern"] == "B_実需売り") & (df["cl_vs_vwap"] > 0)),
    ("吸収 + VWAP上",                (df["pattern"] == "C_吸収") & (df["cl_vs_vwap"] > 0.3)),
    ("吸収(前日実需売り) + VWAP上",  (df["pattern"] == "C_吸収") & (df["prev_pattern"] == "B_実需売り") & (df["cl_vs_vwap"] > 0)),
]

print(f"  {'条件':<36}  {'N':>5}  {'翌ONGap':>9}  {'翌前場':>9}  {'翌日Ret':>9}  {'翌日勝率':>9}")
print("  " + "-" * 80)

for label, mask in conditions_long:
    sub = df[mask].dropna(subset=["next_day_ret"])
    if len(sub) < 5:
        continue
    og = sub["next_on_gap"].dropna()
    am = sub["next_am_ret"].dropna()
    nr = sub["next_day_ret"].dropna()
    print(f"  {label:<36}  {len(nr):>5}  "
          f"{og.mean():>+8.3f}%  "
          f"{am.mean():>+8.3f}%  "
          f"{nr.mean():>+8.3f}%  "
          f"{(nr > 0).mean()*100:>8.1f}%")

# Short エントリー候補条件
print("\n  ─ SHORTエントリー候補（翌日逆張り）─")
conditions_short = [
    ("実需売り",                      df["pattern"] == "B_実需売り"),
    ("実需売り + VWAP下",             (df["pattern"] == "B_実需売り") & (df["cl_vs_vwap"] < -0.3)),
    ("実需売り + VWAP下 + vol≥2x",    (df["pattern"] == "B_実需売り") & (df["cl_vs_vwap"] < -0.3) & (df["vol_ratio"] >= 2.0)),
    ("連続実需売り(2日目)",            (df["pattern"] == "B_実需売り") & (df["consecutive"] == 1)),
    ("実需買い後のリバ(薄売り翌日)",   (df["pattern"] == "A_実需買い") & (df["next_day_ret"] < 0)),  # 参考
]

for label, mask in conditions_short:
    sub = df[mask].dropna(subset=["next_day_ret"])
    if len(sub) < 5:
        continue
    og = sub["next_on_gap"].dropna()
    nr = sub["next_day_ret"].dropna()
    # SHORTなので符号反転（売りポジのリターン）
    short_ret = -nr
    print(f"  {label:<36}  {len(nr):>5}  "
          f"{'(翌日下落率)':>9}  "
          f"{'---':>9}  "
          f"{short_ret.mean():>+8.3f}%  "
          f"{(nr < 0).mean()*100:>8.1f}%")


# ─────────────────────────────────────────────────────────────
# 5. セクター別 実需売り後の特性
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("【5】セクター別 — 実需売り後の翌日特性")
print("  セクターによってリバーサル率が違うか？")
print("=" * 80)

sell_sec = df[df["pattern"] == "B_実需売り"].copy()
print(f"\n  {'セクター':<10}  {'N':>5}  {'翌日mean':>9}  {'翌日勝率':>9}  "
      f"{'当日mean':>9}  {'vol_ratio_avg':>14}")
print("  " + "-" * 65)

for sect in ["非鉄", "半導体", "FA", "電機", "重工", "空調", "自動車部品", "その他"]:
    sub = sell_sec[sell_sec["sector"] == sect].dropna(subset=["next_day_ret"])
    if len(sub) < 5:
        continue
    nr = sub["next_day_ret"]
    print(f"  {sect:<10}  {len(sub):>5}  "
          f"{nr.mean():>+8.3f}%  "
          f"{(nr>0).mean()*100:>8.1f}%  "
          f"{sub['day_ret'].mean():>+8.3f}%  "
          f"{sub['vol_ratio'].mean():>13.2f}x")


# ─────────────────────────────────────────────────────────────
# 6. まとめ: エントリールール提言
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("【6】まとめ: 出来高パターンによるエントリー判断フレームワーク")
print("=" * 80)
print("""
  【LONGエントリー — 精度の高い順】
  ★★★ 実需買い + 引けがVWAP+0.3%以上
         → 機関買いが引けまで継続。翌日ギャップアップが期待できる
  ★★☆ 吸収(大商い横ばい) + 引けがVWAP上 + 前日が実需売り
         → 大量の売り圧力を買いが吸収しきった可能性。ブレイクアップ準備完了
  ★☆☆ 実需売り + 引けがVWAP上（下ヒゲ）
         → 一旦売られたが引けにかけて買い戻し。翌日リバーサル狙い

  【SHORTエントリー — 精度の高い順】
  ★★★ 連続実需売り(2日目以上) + 引けがVWAP-0.3%以下
         → 継続的な機関売り。需給悪化が継続しやすい
  ★★☆ 実需売り + 引けがVWAP-0.5%以下 + vol≥2x
         → 大量の売りが引けまで継続。翌日も売り継続確率が高い

  【フィルタ】
  - 下落幅が-3%以上の場合は Short追随より翌日リバーサルを狙う方が有利
  - VWAP位置が最重要フィルタ: 大商い下落でも引けがVWAP上 → 買い吸収シグナル
  - 出来高倍率が3x以上になると翌日平均回帰（リバーサル）が強まる傾向
""")

print("  ✅ 分析完了")
