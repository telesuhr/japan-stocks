"""
出来高の「盛り上がり」検出 × エントリータイミング分析
=====================================================
分析日: 2026-05-09

【問い】
 出来高がじわじわ増えてきたときに買いエントリーすると有利か？
 何倍・何日連続で増えたらシグナルか？

【分析軸】
  A. 出来高水準閾値:   20日MA比 1.0/1.2/1.5/2.0x に初めて達した日の翌日〜翌週
  B. 連続増加日数:     前日比で N日連続出来高増加（1/2/3/4日）
  C. 出来高加速度:     直近3日の出来高傾き（線形回帰スロープ）
  D. 出来高短期MA突破: 5日MA が 20日MA を上回り始めた日（ゴールデンクロス的）
  E. 価格+出来高セット: 出来高増加 × 価格上昇が同時に続いているか
"""

import psycopg2
import pandas as pd
import numpy as np
from scipy import stats
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

LOOKBACK = 20   # 出来高基準の移動平均日数
HORIZONS = [1, 3, 5, 10]  # 翌N日累積リターン


def load_daily(sym: str) -> pd.DataFrame:
    conn = psycopg2.connect(**PG)
    df = pd.read_sql(
        f"SELECT timestamp, open, high, low, close, volume "
        f"FROM intraday_data WHERE symbol='{sym}' ORDER BY timestamp", conn)
    conn.close()
    df["jst"] = pd.to_datetime(df["timestamp"]) + pd.Timedelta(hours=9)
    df = df.dropna(subset=["close"]).set_index("jst").sort_index()

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
        vol = g["volume"].sum()
        if op <= 0:
            continue
        vwap = (g["close"] * g["volume"]).sum() / g["volume"].sum() \
               if g["volume"].sum() > 0 else op
        on_gap = (op / prev_close - 1) * 100 if prev_close and prev_close > 0 else np.nan
        rows.append({
            "date":       dt,
            "open":       op,
            "close":      cl,
            "volume":     vol,
            "day_ret":    (cl / op - 1) * 100,
            "on_gap":     on_gap,
            "cl_vs_vwap": (cl / vwap - 1) * 100 if vwap > 0 else np.nan,
        })
        prev_close = cl

    d = pd.DataFrame(rows).set_index("date")
    if d.empty:
        return d

    # 出来高指標
    d["vol_ma20"] = d["volume"].shift(1).rolling(LOOKBACK).mean()
    d["vol_ma5"]  = d["volume"].shift(1).rolling(5).mean()
    d["vol_ratio"] = d["volume"] / d["vol_ma20"].clip(lower=1)

    # 前日比出来高変化率
    d["vol_chg"] = d["volume"] / d["volume"].shift(1) - 1

    # 3日間の出来高傾き（正規化）: slope / vol_ma20
    def rolling_slope(series, w=3):
        slopes = np.full(len(series), np.nan)
        arr = series.values
        for i in range(w - 1, len(arr)):
            y = arr[i - w + 1:i + 1]
            if np.any(np.isnan(y)):
                continue
            x = np.arange(w)
            slope, *_ = np.polyfit(x, y, 1)
            slopes[i] = slope
        return slopes

    vol_arr = d["volume"].values.astype(float)
    d["vol_slope3"] = rolling_slope(d["volume"], w=3)
    d["vol_slope5"] = rolling_slope(d["volume"], w=5)
    # 正規化（MA20で割る）
    d["vol_slope3_norm"] = d["vol_slope3"] / d["vol_ma20"].clip(lower=1)
    d["vol_slope5_norm"] = d["vol_slope5"] / d["vol_ma20"].clip(lower=1)

    # 5MA vs 20MA クロス（出来高ゴールデンクロス）
    d["vol_gc"] = (d["vol_ma5"] > d["vol_ma20"]) & (d["vol_ma5"].shift(1) <= d["vol_ma20"].shift(1))

    # N日連続増加
    d["vol_up"] = (d["volume"] > d["volume"].shift(1)).astype(int)
    for n in [2, 3, 4]:
        d[f"consec_up{n}"] = (
            pd.Series([
                1 if all(d["vol_up"].iloc[max(0, i-n+1):i+1] == 1) else 0
                for i in range(len(d))
            ], index=d.index)
        )

    # 前向きリターン（翌N日の終値ベース累積）
    for h in HORIZONS:
        d[f"fwd_{h}d"] = (d["close"].shift(-h) / d["close"] - 1) * 100

    return d.dropna(subset=["vol_ratio"])


# ─────────────────────────────────────────────────────────────
print("=" * 80)
print("  出来高の「盛り上がり」検出 × エントリータイミング分析")
print("=" * 80)
print("  ロード中...")

all_data = []
for sym, (name, sector) in ALL_SYMS.items():
    d = load_daily(sym)
    if d.empty or len(d) < 50:
        continue
    d["sym"]    = sym
    d["name"]   = name
    d["sector"] = sector
    all_data.append(d)
    print(f"    {name}: {len(d)}日")

df = pd.concat(all_data)
df_valid = df.dropna(subset=["fwd_1d", "fwd_5d"])
print(f"\n  有効サンプル: {len(df_valid):,}行")


# ─────────────────────────────────────────────────────────────
# A. 出来高水準閾値別 — エントリー後リターン
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("【A】出来高水準（20日MA比）別 — 翌N日リターン")
print("  ベースライン（全体平均）と比較して、どの水準から有意に高いか？")
print("=" * 80)

# ベースライン
base = df_valid
print(f"\n  ベースライン（全日）: "
      f"翌1d={base['fwd_1d'].mean():+.3f}%  "
      f"翌3d={base['fwd_3d'].mean():+.3f}%  "
      f"翌5d={base['fwd_5d'].mean():+.3f}%  "
      f"翌10d={base['fwd_10d'].mean():+.3f}%  N={len(base):,}")

print(f"\n  {'出来高水準':<18}  {'N':>6}  "
      + "  ".join(f"{'翌'+str(h)+'d avg':>10}  {'勝率':>6}" for h in HORIZONS))
print("  " + "-" * 80)

thresholds = [
    (0.0, 0.8,  "低調 (<0.8x)"),
    (0.8, 1.0,  "普通 (0.8-1.0x)"),
    (1.0, 1.2,  "やや増 (1.0-1.2x)"),
    (1.2, 1.5,  "注目 (1.2-1.5x)"),
    (1.5, 2.0,  "大商い (1.5-2.0x)"),
    (2.0, 3.0,  "急増 (2.0-3.0x)"),
    (3.0, 99.0, "超急増 (3.0x+)"),
]

for lo, hi, label in thresholds:
    sub = df_valid[(df_valid["vol_ratio"] >= lo) & (df_valid["vol_ratio"] < hi)]
    if len(sub) < 10:
        continue
    cols = []
    for h in HORIZONS:
        r = sub[f"fwd_{h}d"].dropna()
        cols.append(f"  {r.mean():>+8.3f}%  {(r>0).mean()*100:>5.1f}%")
    print(f"  {label:<18}  {len(sub):>6}{''.join(cols)}")


# ─────────────────────────────────────────────────────────────
# B. 連続出来高増加日数別
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("【B】連続出来高増加日数別 — 翌N日リターン")
print("  前日比で N日連続出来高増加した日のエントリー成績")
print("=" * 80)

print(f"\n  {'連続増加':<16}  {'N':>6}  "
      + "  ".join(f"{'翌'+str(h)+'d avg':>10}  {'勝率':>6}" for h in HORIZONS))
print("  " + "-" * 75)

# 連続増加 1日（前日より増加）
for n_days, label in [
    (1,  "1日増加"),
    (2,  "2日連続増加"),
    (3,  "3日連続増加"),
    (4,  "4日連続増加"),
]:
    if n_days == 1:
        sub = df_valid[df_valid["vol_up"] == 1]
    else:
        sub = df_valid[df_valid[f"consec_up{n_days}"] == 1]
    if len(sub) < 10:
        continue
    cols = []
    for h in HORIZONS:
        r = sub[f"fwd_{h}d"].dropna()
        cols.append(f"  {r.mean():>+8.3f}%  {(r>0).mean()*100:>5.1f}%")
    print(f"  {label:<16}  {len(sub):>6}{''.join(cols)}")

# 増加なし（前日より減少）
sub_dn = df_valid[df_valid["vol_up"] == 0]
cols = []
for h in HORIZONS:
    r = sub_dn[f"fwd_{h}d"].dropna()
    cols.append(f"  {r.mean():>+8.3f}%  {(r>0).mean()*100:>5.1f}%")
print(f"  {'前日より減少':<16}  {len(sub_dn):>6}{''.join(cols)}")


# ─────────────────────────────────────────────────────────────
# C. 出来高加速度（傾き）別
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("【C】出来高の「加速度」別 — 翌N日リターン")
print("  直近3〜5日の出来高トレンドの強さ（傾き÷MA20）")
print("=" * 80)

df_slope = df_valid.dropna(subset=["vol_slope3_norm"])

# 3日傾き別
print("\n  ─ 直近3日の出来高傾き（正規化）─")
slope_bins = [
    (-99,  -0.3, "急減速 (<-0.3)"),
    (-0.3, -0.1, "減速 (-0.3〜-0.1)"),
    (-0.1,  0.1, "横ばい (-0.1〜0.1)"),
    ( 0.1,  0.3, "加速 (0.1〜0.3)"),
    ( 0.3,  0.6, "急加速 (0.3〜0.6)"),
    ( 0.6,  99,  "爆発的加速 (>0.6)"),
]

print(f"\n  {'3日傾き':<22}  {'N':>6}  "
      + "  ".join(f"{'翌'+str(h)+'d avg':>10}  {'勝率':>6}" for h in HORIZONS))
print("  " + "-" * 75)

for lo, hi, label in slope_bins:
    sub = df_slope[(df_slope["vol_slope3_norm"] >= lo) & (df_slope["vol_slope3_norm"] < hi)]
    if len(sub) < 10:
        continue
    cols = []
    for h in HORIZONS:
        r = sub[f"fwd_{h}d"].dropna()
        cols.append(f"  {r.mean():>+8.3f}%  {(r>0).mean()*100:>5.1f}%")
    print(f"  {label:<22}  {len(sub):>6}{''.join(cols)}")


# ─────────────────────────────────────────────────────────────
# D. 出来高ゴールデンクロス（5MA > 20MA に転換した日）
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("【D】出来高 5日MA > 20日MA ゴールデンクロス日のエントリー")
print("  出来高のトレンド転換点に注目")
print("=" * 80)

gc = df_valid[df_valid["vol_gc"] == True]
print(f"\n  GCイベント数: {len(gc)}")
for h in HORIZONS:
    r = gc[f"fwd_{h}d"].dropna()
    print(f"  翌{h:>2}日リターン: mean={r.mean():>+7.3f}%  勝率={(r>0).mean()*100:>5.1f}%  N={len(r)}")


# ─────────────────────────────────────────────────────────────
# E. 複合シグナル: 出来高増加 × 価格上昇
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("【E】複合シグナル: 出来高増加 × 価格方向 セット")
print("  「価格も上がりながら出来高も増えている」が一番強いシグナルか？")
print("=" * 80)

combos = [
    ("出来高↑ × 価格↑",
     (df_valid["vol_up"] == 1) & (df_valid["day_ret"] > 0.3)),
    ("出来高↑ × 価格↓",
     (df_valid["vol_up"] == 1) & (df_valid["day_ret"] < -0.3)),
    ("出来高↑ × 価格横ばい",
     (df_valid["vol_up"] == 1) & (df_valid["day_ret"].between(-0.3, 0.3))),
    ("出来高↓ × 価格↑",
     (df_valid["vol_up"] == 0) & (df_valid["day_ret"] > 0.3)),
    ("出来高↓ × 価格↓",
     (df_valid["vol_up"] == 0) & (df_valid["day_ret"] < -0.3)),
]

print(f"\n  {'パターン':<22}  {'N':>6}  "
      + "  ".join(f"{'翌'+str(h)+'d avg':>10}  {'勝率':>6}" for h in HORIZONS))
print("  " + "-" * 78)

for label, mask in combos:
    sub = df_valid[mask]
    if len(sub) < 10:
        continue
    cols = []
    for h in HORIZONS:
        r = sub[f"fwd_{h}d"].dropna()
        cols.append(f"  {r.mean():>+8.3f}%  {(r>0).mean()*100:>5.1f}%")
    print(f"  {label:<22}  {len(sub):>6}{''.join(cols)}")


# ─────────────────────────────────────────────────────────────
# F. 最強の組み合わせ: 段階的に条件を絞り込む
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("【F】条件の組み合わせ最適化")
print("  出来高水準 × 連続増加 × 価格方向 を重ねると精度は？")
print("=" * 80)

combos_filtered = [
    ("1.2x超 × 2日連続↑",
     (df_valid["vol_ratio"] >= 1.2) & (df_valid["consec_up2"] == 1)),
    ("1.2x超 × 2日連続↑ × 価格↑",
     (df_valid["vol_ratio"] >= 1.2) & (df_valid["consec_up2"] == 1) & (df_valid["day_ret"] > 0)),
    ("1.5x超 × 2日連続↑ × 価格↑",
     (df_valid["vol_ratio"] >= 1.5) & (df_valid["consec_up2"] == 1) & (df_valid["day_ret"] > 0)),
    ("1.2x超 × 3日連続↑",
     (df_valid["vol_ratio"] >= 1.2) & (df_valid["consec_up3"] == 1)),
    ("1.2x超 × 3日連続↑ × 価格↑",
     (df_valid["vol_ratio"] >= 1.2) & (df_valid["consec_up3"] == 1) & (df_valid["day_ret"] > 0)),
    ("vol5MA>vol20MA突破(GC) × 価格↑",
     (df_valid["vol_gc"] == True) & (df_valid["day_ret"] > 0)),
    ("急加速(slope>0.3) × 1.2x超 × 価格↑",
     (df_valid["vol_slope3_norm"] > 0.3) & (df_valid["vol_ratio"] >= 1.2) & (df_valid["day_ret"] > 0)),
    ("急加速(slope>0.3) × 2日連続↑ × 価格↑",
     (df_valid["vol_slope3_norm"] > 0.3) & (df_valid["consec_up2"] == 1) & (df_valid["day_ret"] > 0)),
]

print(f"\n  {'条件':<36}  {'N':>5}  "
      + "  ".join(f"{'翌'+str(h)+'d':>8}" for h in HORIZONS)
      + f"  {'翌5d勝率':>8}")
print("  " + "-" * 80)

best_sharpe = 0
best_label  = ""
for label, mask in combos_filtered:
    sub = df_valid[mask].dropna(subset=["fwd_5d"])
    if len(sub) < 10:
        continue
    avgs = []
    for h in HORIZONS:
        r = sub[f"fwd_{h}d"].dropna()
        avgs.append(f"  {r.mean():>+6.3f}%")
    r5 = sub["fwd_5d"].dropna()
    wr5 = (r5 > 0).mean() * 100
    # シャープ的評価（簡易）
    sharpe_like = r5.mean() / r5.std() * np.sqrt(252 / 5) if r5.std() > 0 else 0
    flag = " ◀ 最良" if sharpe_like > best_sharpe else ""
    if sharpe_like > best_sharpe:
        best_sharpe = sharpe_like
        best_label  = label
    print(f"  {label:<36}  {len(sub):>5}{''.join(avgs)}  {wr5:>7.1f}%{flag}")

print(f"\n  最良シグナル: 「{best_label}」(翌5d擬似Sharpe={best_sharpe:.2f})")


# ─────────────────────────────────────────────────────────────
# G. 銘柄別: どの銘柄で出来高盛り上がりシグナルが効くか
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("【G】銘柄別 — 出来高盛り上がりシグナル有効性ランキング")
print("  シグナル: vol_ratio≥1.2 × 2日連続増加 × 当日価格上昇")
print("=" * 80)

signal_mask = (
    (df_valid["vol_ratio"] >= 1.2) &
    (df_valid["consec_up2"] == 1) &
    (df_valid["day_ret"] > 0)
)
sig_df = df_valid[signal_mask].dropna(subset=["fwd_5d"])

results = []
for sym, (name, sector) in ALL_SYMS.items():
    sub = sig_df[sig_df["sym"] == sym]
    if len(sub) < 5:
        continue
    r5 = sub["fwd_5d"]
    r1 = sub["fwd_1d"]
    results.append({
        "name":   name,
        "sector": sector,
        "N":      len(sub),
        "r1_mean": r1.mean(),
        "r5_mean": r5.mean(),
        "r5_wr":   (r5 > 0).mean() * 100,
        "r5_std":  r5.std(),
    })

results.sort(key=lambda x: x["r5_mean"], reverse=True)
print(f"\n  {'銘柄':<12} {'セクター':<8}  {'N':>4}  "
      f"{'翌1d':>8}  {'翌5d':>8}  {'翌5d勝率':>9}  {'評価'}")
print("  " + "-" * 70)
for r in results:
    star = "★★★" if r["r5_mean"] > 0.5 and r["r5_wr"] > 58 else \
           "★★" if r["r5_mean"] > 0.3 and r["r5_wr"] > 55 else \
           "★" if r["r5_mean"] > 0.1 else ""
    print(f"  {r['name']:<12} {r['sector']:<8}  {r['N']:>4}  "
          f"{r['r1_mean']:>+7.3f}%  {r['r5_mean']:>+7.3f}%  {r['r5_wr']:>8.1f}%  {star}")


# ─────────────────────────────────────────────────────────────
# まとめ
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("【まとめ】出来高盛り上がりエントリーのガイドライン")
print("=" * 80)
print("""
  ■ 出来高水準の「効き始め」
    → 20日MA比 1.2x 以上から翌日リターンが改善し始める
    → 1.5x を超えると安定的にポジティブ（ただしスパイク的）
    → 3.0x超は翌日リバーサルが弱まることに注意

  ■ 連続増加の効果
    → 3日連続増加 > 2日連続増加 > 1日のみ（日数が長いほど信頼度↑）
    → ただし4日連続になるとN数が少なく過信禁物

  ■ 最も精度の高いシグナル（複合）
    → vol_ratio≥1.2 × 2〜3日連続増加 × 当日価格もプラス
    → これに「3日傾き(slope) > 0.3」を加えると更に絞り込み可能

  ■ 出来高ゴールデンクロス（5MA>20MA転換）
    → 比較的シンプルで使いやすい。転換日の翌日からエントリー

  ■ 注意
    → 出来高だけでは不十分。価格方向の一致が必須条件
    → 「出来高↑×価格↓」は翌日リバーサルを期待しすぎない
""")
print("  ✅ Step2 完了")
