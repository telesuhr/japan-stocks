"""
Step 2: 出来高スパイク後の値動きパターン バックテスト
=====================================================
分析日: 2026-04-30

出来高が過去平均の N倍を超えた瞬間（1分足）に何が起きるか？

【分析軸】
  - スパイク閾値: 2x / 3x / 5x（過去20日の同時間帯平均比）
  - 価格変動の向き: 上昇バー vs 下落バー vs 変化なし
  - 次の 1 / 5 / 15 / 30 分のリターン

  BNF的解釈:
    出来高大 × 上昇 → 実需の買い → 追随（モメンタム）
    出来高大 × 下落 → 実需の売り → 逃げる or 逆張りNG
    出来高大 × 小動き → 吸収中 → どちらかが溜まっている
"""

import psycopg2
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

PG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}

# 分析銘柄（FA・電機・既存）
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

HORIZONS = [1, 5, 15, 30]   # 何分後を見るか
THRESHOLDS = [2.0, 3.0, 5.0]  # 出来高スパイク閾値（倍率）
LOOKBACK = 20 * 240           # 過去20日分の1分足（概算）
COST_PCT = 0.04               # 往復コスト（%）


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


def compute_spike_events(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    1分足データから出来高スパイクイベントを検出。
    同一時間帯（±30分窓）の過去20日平均と比較。
    """
    df_th = trading_hours(df).copy()
    df_th["time_slot"] = df_th.index.time

    # 時間帯別の出来高平均（過去全データ）を計算
    time_avg = df_th.groupby("time_slot")["volume"].mean().to_dict()
    df_th["vol_avg"] = df_th["time_slot"].map(time_avg)
    df_th["vol_ratio"] = np.where(
        df_th["vol_avg"] > 0,
        df_th["volume"] / df_th["vol_avg"],
        np.nan
    )

    # バーの方向
    df_th["bar_ret"] = (df_th["close"] / df_th["open"] - 1) * 100

    # スパイクフラグ
    df_th["is_spike"] = df_th["vol_ratio"] >= threshold

    return df_th


def compute_forward_returns(df_th: pd.DataFrame, horizons: list) -> pd.DataFrame:
    """スパイク後の N分後リターンを計算"""
    df_th = df_th.copy()
    df_th = df_th.sort_index()

    # 全インデックスリスト
    idx_list = df_th.index.tolist()
    idx_pos  = {t: i for i, t in enumerate(idx_list)}

    for h in horizons:
        fwd_ret = []
        for i, (ts, row) in enumerate(df_th.iterrows()):
            pos = idx_pos[ts]
            if pos + h < len(idx_list):
                future_price  = df_th.iloc[pos + h]["close"]
                current_price = row["close"]
                if current_price > 0:
                    fwd_ret.append((future_price / current_price - 1) * 100)
                else:
                    fwd_ret.append(np.nan)
            else:
                fwd_ret.append(np.nan)
        df_th[f"fwd_{h}m"] = fwd_ret

    return df_th


def analyze_spikes(sym: str, df_th: pd.DataFrame, threshold: float) -> dict:
    """スパイクイベントを方向別に分類して統計を出す"""
    spikes = df_th[df_th["is_spike"]].dropna(subset=["bar_ret"])
    if len(spikes) == 0:
        return {}

    # 方向分類
    up   = spikes[spikes["bar_ret"] > 0.1]   # 上昇スパイク
    dn   = spikes[spikes["bar_ret"] < -0.1]  # 下落スパイク
    flat = spikes[spikes["bar_ret"].between(-0.1, 0.1)]  # 横ばいスパイク

    result = {
        "sym": sym,
        "threshold": threshold,
        "n_total": len(spikes),
        "n_up": len(up),
        "n_dn": len(dn),
        "n_flat": len(flat),
    }

    for label, subset in [("up", up), ("dn", dn), ("flat", flat)]:
        for h in HORIZONS:
            col = f"fwd_{h}m"
            if col not in subset.columns:
                continue
            vals = subset[col].dropna()
            if len(vals) < 5:
                result[f"{label}_{h}m_mean"] = np.nan
                result[f"{label}_{h}m_wr"]   = np.nan
                result[f"{label}_{h}m_n"]    = len(vals)
                continue

            # LONGの場合（上昇スパイクを追随 or 下落スパイクを逆張り）
            if label == "up":
                ret = vals            # 上昇スパイク → LONG → そのままのリターン
            elif label == "dn":
                ret = vals            # 下落スパイク → SHORT → マイナスが利益（後で符号反転）
            else:
                ret = vals

            net = ret - COST_PCT if label in ("up",) else ret + COST_PCT
            result[f"{label}_{h}m_mean"] = ret.mean()
            result[f"{label}_{h}m_net"]  = net.mean()
            result[f"{label}_{h}m_wr"]   = (ret > 0).mean() * 100 if label == "up" \
                                           else (ret < 0).mean() * 100 if label == "dn" \
                                           else (ret.abs() < 0.1).mean() * 100
            result[f"{label}_{h}m_n"]    = len(vals)

    return result


# ─────────────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────────────
print("=" * 85)
print("  Step 2: 出来高スパイク後の値動きパターン バックテスト")
print("=" * 85)
print("  ロード中（1分足 × 全銘柄）...")

all_results = []
for sym, (name, sector) in ALL_SYMS.items():
    print(f"    {name}...", end=" ")
    df = load_intraday(sym)
    for thr in THRESHOLDS:
        df_th = compute_spike_events(df, threshold=thr)
        df_th = compute_forward_returns(df_th, HORIZONS)
        res = analyze_spikes(sym, df_th, thr)
        if res:
            res["name"] = name
            res["sector"] = sector
            all_results.append(res)
    print("OK")

# ─────────────────────────────────────────────────────
# 出力 1: 閾値別サマリー（全銘柄プール）
# ─────────────────────────────────────────────────────
print("\n" + "=" * 85)
print("【全銘柄プール: 出来高スパイク後のリターン — 閾値別】")
print("  LONG = 上昇スパイクに追随   SHORT = 下落スパイクに追随（逆方向）")
print("=" * 85)

for thr in THRESHOLDS:
    subset = [r for r in all_results if r["threshold"] == thr]
    if not subset:
        continue

    print(f"\n  ─ 出来高スパイク ≥ {thr:.0f}x 平均 ─")
    print(f"  {'タイプ':<14}  N    " +
          "  ".join(f"{'fwd+'+str(h)+'m':>10}" for h in HORIZONS))
    print("  " + "-" * 65)

    for label, direction in [
        ("上昇スパイク(LONG)", "up"),
        ("下落スパイク(SHORT)", "dn"),
        ("横ばいスパイク", "flat"),
    ]:
        ns    = [r.get(f"{direction}_{h}m_n", 0) or 0 for h in HORIZONS]
        means = [r.get(f"{direction}_{h}m_mean", np.nan) for h in HORIZONS for r in subset]
        # 全銘柄の平均を集計
        for h in HORIZONS:
            col_mean = f"{direction}_{h}m_mean"
            col_wr   = f"{direction}_{h}m_wr"
            col_n    = f"{direction}_{h}m_n"
            all_vals = []
            for r in subset:
                v = r.get(col_mean, np.nan)
                if not np.isnan(v) if v == v else False:
                    n = r.get(col_n, 0) or 0
                    all_vals.extend([v] * max(1, int(n)))
            pass

        # 銘柄横断で集計
        pooled = {h: [] for h in HORIZONS}
        for r in subset:
            for h in HORIZONS:
                v = r.get(f"{direction}_{h}m_mean", np.nan)
                n = r.get(f"{direction}_{h}m_n", 0) or 0
                if v == v and n > 0:  # not nan
                    pooled[h].extend([v] * int(n))

        total_n = sum(len(pooled[HORIZONS[0]]) for _ in [1])
        line = f"  {label:<14}  {len(pooled[HORIZONS[0]]):>4}"
        for h in HORIZONS:
            if pooled[h]:
                arr = np.array(pooled[h])
                wr = (arr > 0).mean() * 100 if direction == "up" \
                     else (arr < 0).mean() * 100 if direction == "dn" \
                     else (arr.abs() < 0.3).mean() * 100
                mean = arr.mean()
                line += f"  {mean:>+6.3f}%({wr:>4.0f}%)"
            else:
                line += f"  {'---':>10}"
        print(line)

# ─────────────────────────────────────────────────────
# 出力 2: 銘柄別 出来高3x スパイク後15分リターン
# ─────────────────────────────────────────────────────
print("\n" + "=" * 85)
print("【銘柄別: 出来高3x スパイク後15分リターン】")
print("  LONG = 上昇スパイク追随  /  SHORT = 下落スパイク追随の期待値")
print("=" * 85)
print(f"  {'銘柄':<12} {'セクター':<8}  "
      f"{'上昇N':>6}  {'上昇後15m':>10}  {'勝率':>7}  "
      f"{'下落N':>6}  {'下落後15m':>10}  {'逆勝率':>7}")
print("  " + "-" * 80)

thr_target = 3.0
subset_3x = [r for r in all_results if r["threshold"] == thr_target]
subset_3x.sort(key=lambda x: x.get("up_15m_mean", -99) or -99, reverse=True)

for r in subset_3x:
    up_n    = r.get("up_15m_n", 0) or 0
    up_mean = r.get("up_15m_mean", np.nan)
    up_wr   = r.get("up_15m_wr", np.nan)
    dn_n    = r.get("dn_15m_n", 0) or 0
    dn_mean = r.get("dn_15m_mean", np.nan)
    dn_wr   = r.get("dn_15m_wr", np.nan)

    up_str = f"{up_mean:>+8.3f}%  {up_wr:>6.1f}%" if up_mean == up_mean else "  ---"
    dn_str = f"{dn_mean:>+8.3f}%  {dn_wr:>6.1f}%" if dn_mean == dn_mean else "  ---"

    # マーカー
    up_mark = " ◎" if (up_mean == up_mean and up_mean > 0.15 and up_wr and up_wr > 55) else ""
    dn_mark = " ◎" if (dn_mean == dn_mean and dn_mean < -0.15 and dn_wr and dn_wr > 55) else ""

    print(f"  {r['name']:<12} {r['sector']:<8}  "
          f"{up_n:>6}  {up_str}{up_mark}  "
          f"{dn_n:>6}  {dn_str}{dn_mark}")

# ─────────────────────────────────────────────────────
# 出力 3: 吸収パターンの解釈（大商い × 小動き）
# ─────────────────────────────────────────────────────
print("\n" + "=" * 85)
print("【吸収パターン（出来高3x超 × 価格±0.1%以内）の後は？】")
print("  大量の売買が出たのに価格が動かない → 強い買い/売りが反対サイドを吸収中")
print("=" * 85)
print(f"  {'銘柄':<12} {'セクター':<8}  {'N':>5}  "
      f"{'5分後':>10}  {'15分後':>10}  {'30分後':>10}")
print("  " + "-" * 65)

for r in sorted(subset_3x, key=lambda x: x.get("flat_15m_mean", 0) or 0, reverse=True):
    flat_5  = r.get("flat_5m_mean", np.nan)
    flat_15 = r.get("flat_15m_mean", np.nan)
    flat_30 = r.get("flat_30m_mean", np.nan)
    flat_n  = r.get("flat_15m_n", 0) or 0

    def fmt(v):
        return f"{v:>+9.3f}%" if v == v else "       ---"

    print(f"  {r['name']:<12} {r['sector']:<8}  {flat_n:>5}  "
          f"{fmt(flat_5)}  {fmt(flat_15)}  {fmt(flat_30)}")

print("\n  ✅ Step2 完了")
print("  ヒント: 吸収後にどちらに動くかは銘柄・時間帯に依存する。")
print("  吸収が上なら（買い圧力が大きい）→ その後上昇しやすい。")
print("  吸収が下なら（売り圧力が大きい）→ その後下落しやすい。")
print("  ※ 1分足レベルでは板読みなしで吸収方向を判断するのは困難。")
