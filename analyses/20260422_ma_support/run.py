"""
移動平均のサポートライン検証

仮説: 下落局面で価格が MA に接近した時、サポートとして機能し反発するか?

アプローチ:
1. 主要銘柄の日足 (daily_stats) を使用
2. MA(5, 10, 20, 25, 50, 75, 100, 200) を計算
3. Touch event: 日中Low が MA の ±1.0% 以内 (下から接触)
4. 条件: 直近20日高値から -5% 以上 (下落局面に限定)
5. 評価: 接触日終値から t+1, t+5, t+10, t+20 の終値リターン
6. ベースライン: 接触なし日の同期間フォワードリターン比較

層別:
- MA傾き (+/-)  → 上昇MA のみ真のサポートか
- 接触回数 (初回 / 繰返し) → 減衰するか
"""
import psycopg2, pandas as pd, numpy as np
from pathlib import Path

PG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}
OUT = Path(__file__).parent

# 主要銘柄 (セクター分散)
SYMBOLS = [
    # 半導体
    "8035.T", "6146.T", "6920.T", "6857.T", "6963.T", "6762.T", "6861.T", "6954.T",
    # 非鉄
    "5711.T", "5713.T", "5706.T",
    # 自動車
    "7203.T", "7267.T", "7201.T",
    # 商社
    "8058.T", "8031.T", "8053.T", "8002.T", "8001.T",
    # 電機・重機
    "6503.T", "7011.T", "7012.T", "7013.T",
    # エネルギー
    "5020.T", "5016.T", "1605.T",
    # 通信・IT
    "9432.T", "9433.T", "9984.T", "6758.T",
    # 海運
    "9101.T", "9104.T",
    # 鉄鋼
    "5401.T", "5802.T",
    # ETF
    "1306.T", "1321.T",
]

MAS = [5, 10, 20, 25, 50, 75, 100, 200]
TOUCH_TOL_PCT = 1.0   # MA の ±1.0% 以内
DOWNTREND_DROP = 5.0  # 直近20日高値から -5% 以上
FWD_DAYS = [1, 5, 10, 20]
COST = 0.0  # 評価は純粋な予測力 (ポジション/コストは別途)


def load_daily(sym):
    conn = psycopg2.connect(**PG)
    df = pd.read_sql(
        "SELECT trade_date, open, high, low, close, volume FROM daily_stats "
        "WHERE symbol=%s ORDER BY trade_date",
        conn, params=(sym,))
    conn.close()
    if df.empty: return None
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.set_index("trade_date").sort_index()
    df = df.astype({c: float for c in ["open", "high", "low", "close"]})
    return df


def annotate(df):
    """MA, 傾き, ドローダウン, フォワードリターン を追加"""
    for n in MAS:
        df[f"ma{n}"] = df["close"].rolling(n).mean()
        df[f"ma{n}_slope"] = df[f"ma{n}"].diff(5)  # 5日変化
    # 20日高値からのドローダウン
    df["hh20"] = df["close"].rolling(20).max()
    df["dd20_pct"] = (df["close"] / df["hh20"] - 1) * 100
    # フォワードリターン
    for d in FWD_DAYS:
        df[f"fwd{d}_pct"] = df["close"].shift(-d) / df["close"] - 1
    return df


def find_touches(df, ma_col):
    """価格 (Low) が MA の ±TOUCH_TOL_PCT% 以内に接触した日を返す"""
    ma = df[ma_col]
    lo, hi = df["low"], df["high"]
    # Low が MA を下に割り込む or touched から TOL%内
    lower_band = ma * (1 - TOUCH_TOL_PCT / 100)
    upper_band = ma * (1 + TOUCH_TOL_PCT / 100)
    # 接触 = 当日レンジが MA を含む or MA±TOL に入る
    touched = ((lo <= upper_band) & (hi >= lower_band))
    return touched


def stats_on_returns(arr):
    arr = np.asarray(arr); arr = arr[~np.isnan(arr)]
    if len(arr) < 3:
        return dict(n=len(arr), mean=np.nan, median=np.nan, wr=np.nan, tstat=np.nan)
    return dict(n=len(arr), mean=arr.mean() * 100, median=np.median(arr) * 100,
                wr=(arr > 0).mean() * 100,
                tstat=arr.mean() / (arr.std(ddof=1) / np.sqrt(len(arr))))


def analyze_symbol(sym):
    df = load_daily(sym)
    if df is None or len(df) < 250:
        return None
    df = annotate(df)
    results = []
    for n in MAS:
        ma_col = f"ma{n}"
        slope_col = f"ma{n}_slope"
        if df[ma_col].isna().all(): continue

        # 下落局面 + MA接触日
        touched = find_touches(df, ma_col)
        downtrend = df["dd20_pct"] <= -DOWNTREND_DROP
        event = touched & downtrend & df[ma_col].notna()

        # 全体
        for fwd in FWD_DAYS:
            ret = df.loc[event, f"fwd{fwd}_pct"].dropna()
            s = stats_on_returns(ret.values)
            # Baseline: 接触なし + 下落局面
            base_mask = (~touched) & downtrend & df[ma_col].notna()
            base = df.loc[base_mask, f"fwd{fwd}_pct"].dropna()
            sb = stats_on_returns(base.values)
            results.append({
                "sym": sym, "ma": n, "slope_filter": "all", "fwd": fwd,
                "n_event": s["n"], "event_mean_pct": s["mean"],
                "event_wr": s["wr"], "event_t": s["tstat"],
                "n_base": sb["n"], "base_mean_pct": sb["mean"], "base_wr": sb["wr"],
                "edge_pct": (s["mean"] - sb["mean"]) if not np.isnan(s["mean"]) else np.nan,
            })
        # MA 上昇中 (slope>0) のみ
        for fwd in FWD_DAYS:
            mask = event & (df[slope_col] > 0)
            ret = df.loc[mask, f"fwd{fwd}_pct"].dropna()
            s = stats_on_returns(ret.values)
            base_mask = (~touched) & downtrend & (df[slope_col] > 0) & df[ma_col].notna()
            base = df.loc[base_mask, f"fwd{fwd}_pct"].dropna()
            sb = stats_on_returns(base.values)
            results.append({
                "sym": sym, "ma": n, "slope_filter": "up", "fwd": fwd,
                "n_event": s["n"], "event_mean_pct": s["mean"],
                "event_wr": s["wr"], "event_t": s["tstat"],
                "n_base": sb["n"], "base_mean_pct": sb["mean"], "base_wr": sb["wr"],
                "edge_pct": (s["mean"] - sb["mean"]) if not np.isnan(s["mean"]) else np.nan,
            })
    return pd.DataFrame(results)


def main():
    print("=" * 110)
    print(f"MA サポート検証 - {len(SYMBOLS)}銘柄, MA={MAS}, touch tol ±{TOUCH_TOL_PCT}%, downtrend dd20 ≤ -{DOWNTREND_DROP}%")
    print("=" * 110)

    all_res = []
    for sym in SYMBOLS:
        r = analyze_symbol(sym)
        if r is not None: all_res.append(r)
    df = pd.concat(all_res, ignore_index=True)
    df.to_csv(OUT / "by_symbol_ma.csv", index=False)

    # =========================================================================
    # 集計1: MA×slope×fwd の pooled 統計
    # =========================================================================
    print("\n[Pooled] MA × slope × fwd")
    print("-" * 110)
    # 各 (ma, slope, fwd) で N加重平均的に集約: event 全件を再計算したほうが正確
    # → 上の per-symbol results では N と mean 持つので、n加重で mean を合算
    rows = []
    for ma in MAS:
        for sl in ["all", "up"]:
            for fwd in FWD_DAYS:
                sub = df[(df["ma"] == ma) & (df["slope_filter"] == sl) & (df["fwd"] == fwd)]
                n_tot = sub["n_event"].sum()
                if n_tot == 0: continue
                w_mean = (sub["event_mean_pct"] * sub["n_event"]).sum() / n_tot
                nb_tot = sub["n_base"].sum()
                if nb_tot == 0: continue
                w_base = (sub["base_mean_pct"] * sub["n_base"]).sum() / nb_tot
                # weighted WR
                w_wr = (sub["event_wr"] * sub["n_event"]).sum() / n_tot
                w_base_wr = (sub["base_wr"] * sub["n_base"]).sum() / nb_tot
                rows.append({"ma": ma, "slope": sl, "fwd_days": fwd,
                             "n_event": n_tot, "event_mean%": w_mean, "event_wr%": w_wr,
                             "n_base": nb_tot, "base_mean%": w_base, "base_wr%": w_base_wr,
                             "edge%": w_mean - w_base, "wr_edge%": w_wr - w_base_wr})
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    summary.to_csv(OUT / "pooled_summary.csv", index=False)

    # =========================================================================
    # 集計2: 銘柄別トップ (edge が大きいもの)
    # =========================================================================
    print("\n[Top銘柄] MA=25 slope=up fwd=5 (ベース設定) の銘柄別 edge ランキング")
    print("-" * 110)
    base_setting = df[(df["ma"] == 25) & (df["slope_filter"] == "up") & (df["fwd"] == 5)]
    base_setting = base_setting.sort_values("edge_pct", ascending=False).head(15)
    print(base_setting[["sym", "n_event", "event_mean_pct", "event_wr", "event_t", "base_mean_pct", "edge_pct"]]
          .to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    # =========================================================================
    # 集計3: MA 別の「接触→反発 (fwd5 > 0)」確率を見る (pooled)
    # =========================================================================
    print("\n[Rebound Rate] MA 別 反発率 (fwd=5 pooled, slope=up vs all)")
    print("-" * 110)
    for sl in ["all", "up"]:
        sub = summary[(summary["fwd_days"] == 5) & (summary["slope"] == sl)]
        if sub.empty: continue
        print(f"slope={sl}")
        print(sub[["ma", "n_event", "event_mean%", "event_wr%", "base_mean%", "base_wr%", "edge%", "wr_edge%"]]
              .to_string(index=False, float_format=lambda x: f"{x:.2f}"))
        print()

    # =========================================================================
    # 集計4: 最強設定での t-stat ランキング
    # =========================================================================
    print("\n[有意性] edge が統計的に有意な (ma, slope, fwd) トップ")
    print("-" * 110)
    # 全event を pool して再計算
    rows = []
    for ma in MAS:
        for sl in ["all", "up"]:
            for fwd in FWD_DAYS:
                sub = df[(df["ma"] == ma) & (df["slope_filter"] == sl) & (df["fwd"] == fwd)]
                if sub["n_event"].sum() < 30: continue
                # per-symbol average of means (equal-weight) for stability
                mu = sub["event_mean_pct"].mean()
                # Overall t via combined: use weighted n
                w_mean = (sub["event_mean_pct"] * sub["n_event"]).sum() / sub["n_event"].sum()
                w_base = (sub["base_mean_pct"] * sub["n_base"]).sum() / sub["n_base"].sum()
                rows.append({"ma": ma, "slope": sl, "fwd": fwd,
                             "n_tot": sub["n_event"].sum(),
                             "per_sym_mean%": mu, "pooled_mean%": w_mean,
                             "pooled_base%": w_base, "edge%": w_mean - w_base})
    rank = pd.DataFrame(rows).sort_values("edge%", ascending=False).head(15)
    print(rank.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    rank.to_csv(OUT / "rank.csv", index=False)

    print("\nDONE")


if __name__ == "__main__":
    main()
