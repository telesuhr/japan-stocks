"""
Step 1: 出来高倍率 × 相対強弱スキャナー
==========================================
分析日: 2026-04-30

今週（直近5日）の各銘柄について：
  - 出来高倍率（過去20日平均比）
  - 相対リターン（セクター内での強弱）
  - 出来高急増 × 価格変動のパターン分類

BNFが「どこに資金が入っているか」を判断する際の定量版。
"""

import psycopg2
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from datetime import date, timedelta

PG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}

ALL_SYMS = {
    # 既存銘柄
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
    # FA・電機
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


def load_intraday(sym: str) -> pd.DataFrame:
    conn = psycopg2.connect(**PG)
    df = pd.read_sql(
        f"SELECT timestamp, open, high, low, close, volume "
        f"FROM intraday_data WHERE symbol='{sym}' ORDER BY timestamp", conn)
    conn.close()
    df["jst"] = pd.to_datetime(df["timestamp"]) + pd.Timedelta(hours=9)
    return df.dropna(subset=["close"]).set_index("jst").sort_index()


def trading_hours(df: pd.DataFrame) -> pd.DataFrame:
    h, m = df.index.hour, df.index.minute
    return df[
        (h == 9) | ((h >= 10) & (h < 11)) | ((h == 11) & (m <= 30)) |
        ((h == 12) & (m >= 30)) | ((h >= 13) & (h < 15)) | ((h == 15) & (m <= 30))
    ]


def build_daily_vol(df: pd.DataFrame) -> pd.DataFrame:
    """日次サマリー（出来高付き）"""
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
        hi = g_day["high"].max()
        lo = g_day["low"].min()

        # VWAP
        vwap = (g_day["close"] * g_day["volume"]).sum() / g_day["volume"].sum() \
            if g_day["volume"].sum() > 0 else op

        on_gap = (op / prev_cl - 1) * 100 if prev_cl and prev_cl > 0 else np.nan

        rows.append({
            "date": dt,
            "dow": pd.Timestamp(dt).dayofweek,
            "open": op, "close": cl, "high": hi, "low": lo,
            "volume": vol,
            "day_ret": (cl / op - 1) * 100,
            "on_gap":  on_gap,
            "range_pct": (hi / lo - 1) * 100 if lo > 0 else np.nan,
            "vwap": vwap,
            "close_vs_vwap": (cl / vwap - 1) * 100 if vwap > 0 else np.nan,
        })
        prev_cl = cl
    return pd.DataFrame(rows)


def classify_bar(ret: float, vol_ratio: float) -> str:
    """
    出来高倍率 × 価格変動でバーを分類
    BNF的な「どういう動きか」の定性判定
    """
    heavy = vol_ratio >= 1.5   # 出来高1.5倍以上
    strong = ret > 0.3          # 価格+0.3%超
    weak   = ret < -0.3         # 価格-0.3%超

    if heavy and strong:
        return "◎実需買い"    # 大商いの上昇 → 本物の資金流入
    elif heavy and weak:
        return "▼実需売り"    # 大商いの下落 → 本物の売り圧力
    elif heavy:
        return "▷吸収/膠着"  # 大商いでも動かない → 上下どちらかの吸収
    elif strong:
        return "△薄商い急騰"  # 小出来高の上昇 → 信頼度低い
    elif weak:
        return "▼薄商い急落"  # 小出来高の下落
    else:
        return "  横ばい"


# ─────────────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────────────
print("=" * 80)
print("  Step 1: 出来高倍率 × 相対強弱スキャナー")
print("=" * 80)
print("  ロード中...")

all_daily = {}
for sym, (name, sector) in ALL_SYMS.items():
    df = load_intraday(sym)
    all_daily[sym] = build_daily_vol(df)
    print(f"    {name}: {len(all_daily[sym])}日")

# ─────────────────────────────────────────────────────
# 1. 直近5日間の出来高倍率と価格変動
# ─────────────────────────────────────────────────────
LOOKBACK_VOL = 20   # 出来高の基準期間（20日移動平均）
RECENT_DAYS  = 5    # 直近5日間を表示

print("\n" + "=" * 80)
print("【直近5日間 出来高倍率 × 価格変動 スキャン】")
print("  出来高倍率 = 当日出来高 / 過去20日平均出来高")
print("=" * 80)

# セクター別に表示
sectors_order = ["非鉄", "半導体", "FA", "電機", "重工", "空調", "自動車部品", "その他"]
sector_syms: dict[str, list[str]] = {}
for sym, (name, sect) in ALL_SYMS.items():
    sector_syms.setdefault(sect, []).append(sym)

for sect in sectors_order:
    syms = sector_syms.get(sect, [])
    if not syms:
        continue

    print(f"\n  ─ {sect} ─")

    for sym in syms:
        d = all_daily[sym].copy()
        name = ALL_SYMS[sym][0]
        if len(d) < LOOKBACK_VOL + RECENT_DAYS:
            continue

        # 出来高移動平均（過去20日）
        d["vol_ma20"] = d["volume"].shift(1).rolling(LOOKBACK_VOL).mean()
        d["vol_ratio"] = d["volume"] / d["vol_ma20"]

        recent = d.tail(RECENT_DAYS)

        print(f"\n    {name}  (直近{RECENT_DAYS}日)")
        print(f"    {'日付':<12} {'曜':<3} {'日次Ret':>8}  {'出来高倍率':>10}  {'引けvsVWAP':>11}  {'判定':<12}")
        print("    " + "-" * 60)
        for _, row in recent.iterrows():
            if np.isnan(row["vol_ratio"]):
                continue
            label = classify_bar(row["day_ret"], row["vol_ratio"])
            dow_str = "月火水木金"[int(row["dow"])]
            cv = row["close_vs_vwap"] if not np.isnan(row["close_vs_vwap"]) else 0
            print(f"    {str(row['date']):<12} {dow_str}曜  "
                  f"{row['day_ret']:>+7.2f}%  "
                  f"{row['vol_ratio']:>9.2f}x  "
                  f"{cv:>+10.2f}%  "
                  f"{label}")

# ─────────────────────────────────────────────────────
# 2. 週次出来高倍率ランキング（資金流入検知）
# ─────────────────────────────────────────────────────
print("\n\n" + "=" * 80)
print("【週次 出来高倍率ランキング】  ← どのセクター・銘柄に資金が入っているか")
print("=" * 80)

rows_rank = []
for sym, (name, sect) in ALL_SYMS.items():
    d = all_daily[sym].copy()
    if len(d) < LOOKBACK_VOL + RECENT_DAYS:
        continue
    d["vol_ma20"] = d["volume"].shift(1).rolling(LOOKBACK_VOL).mean()
    d["vol_ratio"] = d["volume"] / d["vol_ma20"]
    recent = d.tail(RECENT_DAYS)

    avg_ratio  = recent["vol_ratio"].mean()
    avg_ret    = recent["day_ret"].mean()
    cum_ret    = recent["day_ret"].sum()
    max_ratio  = recent["vol_ratio"].max()
    heavy_days = (recent["vol_ratio"] >= 1.5).sum()

    rows_rank.append({
        "sym": sym, "name": name, "sector": sect,
        "avg_ratio": avg_ratio,
        "max_ratio": max_ratio,
        "heavy_days": heavy_days,
        "avg_ret": avg_ret,
        "cum_ret": cum_ret,
    })

rows_rank.sort(key=lambda x: x["avg_ratio"], reverse=True)

print(f"\n  {'順':>2}  {'銘柄':<12} {'セクター':<8}  "
      f"{'平均倍率':>8}  {'最大倍率':>8}  {'大商い日数':>10}  "
      f"{'週累積Ret':>10}  シグナル")
print("  " + "-" * 80)

for i, r in enumerate(rows_rank, 1):
    # 資金流入シグナル
    if r["avg_ratio"] >= 1.5 and r["avg_ret"] > 0:
        sig = "🔥資金流入"
    elif r["avg_ratio"] >= 1.5 and r["avg_ret"] < 0:
        sig = "⚠️  大量売り"
    elif r["avg_ratio"] >= 1.2 and r["avg_ret"] > 0:
        sig = "↑注目"
    elif r["avg_ratio"] < 0.8:
        sig = "  閑散"
    else:
        sig = ""

    print(f"  {i:>2}  {r['name']:<12} {r['sector']:<8}  "
          f"{r['avg_ratio']:>7.2f}x  "
          f"{r['max_ratio']:>7.2f}x  "
          f"{r['heavy_days']:>9}日  "
          f"{r['cum_ret']:>+9.2f}%  {sig}")

# ─────────────────────────────────────────────────────
# 3. セクター別まとめ
# ─────────────────────────────────────────────────────
print("\n\n" + "=" * 80)
print("【セクター別まとめ — 週次出来高倍率 × 累積リターン】")
print("=" * 80)
print(f"\n  {'セクター':<8}  {'平均出来高倍率':>14}  {'週累積Ret':>10}  {'大商い銘柄'}")
print("  " + "-" * 60)

for sect in sectors_order:
    syms = sector_syms.get(sect, [])
    sub = [r for r in rows_rank if r["sector"] == sect]
    if not sub:
        continue
    avg_ratio = np.mean([r["avg_ratio"] for r in sub])
    avg_ret   = np.mean([r["cum_ret"]   for r in sub])
    heavy = [r["name"] for r in sub if r["avg_ratio"] >= 1.5]
    heavy_str = "・".join(heavy) if heavy else "なし"

    arrow = "↑" if avg_ret > 0.5 else ("↓" if avg_ret < -0.5 else "→")
    print(f"  {sect:<8}  {avg_ratio:>13.2f}x  {avg_ret:>+9.2f}%  {arrow} {heavy_str}")

print("\n  ✅ Step1 完了")
