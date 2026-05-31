"""
auKabu 板imbalanceエッジ検証
================================================================
第一弾(2026-05-30)でナイーブシグナル(出来高スパイク/VWAP乖離)は全滅。
「エッジがあるとすれば板情報(imbalance/pressure=約定前需給)のみ」という示唆を検証。

データ: aukabu.snapshots_5sec (5秒板スナップショット)
        aukabu.bars_1min (1分足 + imbalance特徴量の1分平均)

特徴量:
  l1_imb       : L1板の買/売不均衡 (-1〜+1, 正=買い板優勢)
  depth10_imb  : 10段階板の不均衡
  w_imb        : 加重不均衡
  over_under_imb: 大口注文の不均衡
  vwap_dev_pct : VWAP乖離率(%)

検証軸:
  A. 時系列予測: 各銘柄の imbalance_t → 次の5秒/1分のリターン
  B. クロスセクション: 各タイムスタンプで imbalance ランク → 次1分の相対リターン

注意: 現時点でデータが1週間しかない。統計的確度は低いが、
      方向性確認と月次再実行フレームワーク確立が目的。
      毎月データが増えるたびに本スクリプトを再実行して更新する。
"""
import sys; sys.stdout.reconfigure(line_buffering=True)
import os
from pathlib import Path
import numpy as np, pandas as pd
import psycopg2

OUT = Path(__file__).parent
PG = dict(
    host=os.environ.get("PGHOST", "localhost"),
    port=int(os.environ.get("PGPORT", 5432)),
    user=os.environ.get("PGUSER", "postgres"),
    password=os.environ.get("PGPASSWORD", "postgres"),
    dbname=os.environ.get("PGDATABASE", "market_data"),
)

COST_BPS = 5.0   # 1分足の片道コスト(bps) — 板監視なら成行ではなく指値前提


def load_bars(conn):
    """1分足 + imbalance特徴量"""
    df = pd.read_sql(
        """SELECT symbol, bucket_ts, open, high, low, close, volume,
                  avg_l1_imb, avg_depth10_imb, avg_w_imb, avg_vwap_dev_pct, sample_n
           FROM aukabu.bars_1min
           WHERE sample_n >= 5          -- スナップショット5件以上ある分のみ
           ORDER BY symbol, bucket_ts""",
        conn
    )
    df["bucket_ts"] = pd.to_datetime(df["bucket_ts"], utc=True).dt.tz_convert("Asia/Tokyo")
    for c in ["open", "high", "low", "close", "volume",
              "avg_l1_imb", "avg_depth10_imb", "avg_w_imb", "avg_vwap_dev_pct"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_snaps(conn):
    """5秒スナップショット（最新データのみ）"""
    df = pd.read_sql(
        """SELECT symbol, bucket_ts, price, l1_imb, depth10_imb, w_imb, vwap_dev_pct
           FROM aukabu.snapshots_5sec
           WHERE l1_imb IS NOT NULL
           ORDER BY symbol, bucket_ts""",
        conn
    )
    df["bucket_ts"] = pd.to_datetime(df["bucket_ts"], utc=True).dt.tz_convert("Asia/Tokyo")
    for c in ["price", "l1_imb", "depth10_imb", "w_imb", "vwap_dev_pct"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ─── 検証A: 時系列予測 ────────────────────────────────────────────────────────

def test_timeseries(df, feature_col, fwd_bars=1, data_label="bars"):
    """
    各銘柄個別の時系列: feature_t → fwd_bars分後リターン
    IC (情報係数 = スピアマン相関) を銘柄間で集計
    """
    rows = []
    for sym, g in df.groupby("symbol"):
        g = g.sort_values("bucket_ts").reset_index(drop=True)
        g["fwd_ret"] = g["close"].shift(-fwd_bars) / g["close"] - 1
        g = g.dropna(subset=[feature_col, "fwd_ret"])
        if len(g) < 50:
            continue
        ic = g[[feature_col, "fwd_ret"]].corr(method="spearman").iloc[0, 1]
        t_stat = ic * np.sqrt(len(g) - 2) / np.sqrt(1 - ic**2) if abs(ic) < 1 else np.nan
        rows.append({"symbol": sym, "n": len(g), "IC": round(ic, 4), "t_stat": round(t_stat, 2)})
    return pd.DataFrame(rows)


# ─── 検証B: クロスセクション ─────────────────────────────────────────────────

def test_crosssection(df, feature_col, fwd_bars=1):
    """
    各タイムスタンプで全銘柄を feature でランク付け
    → 上位群(Long) - 下位群(Short) の次N分リターン
    コスト控除後のSharpeとt値
    """
    df = df.copy().sort_values(["bucket_ts", "symbol"])
    df["fwd_ret"] = df.groupby("symbol")["close"].transform(lambda s: s.shift(-fwd_bars) / s - 1)
    df = df.dropna(subset=[feature_col, "fwd_ret"])

    ls_rows = []
    for ts, x in df.groupby("bucket_ts"):
        x = x.dropna(subset=[feature_col])
        if len(x) < 6:
            continue
        q = x[feature_col].rank(pct=True)
        long_ret = x[q >= 0.7]["fwd_ret"].mean()
        short_ret = x[q <= 0.3]["fwd_ret"].mean()
        if pd.notna(long_ret) and pd.notna(short_ret):
            ls_rows.append({"ts": ts, "ls": long_ret - short_ret})

    if not ls_rows:
        return {}
    ls = pd.DataFrame(ls_rows).set_index("ts")["ls"]
    net = ls - COST_BPS / 1e4
    n = len(net)
    if n < 20 or net.std() == 0:
        return {}
    # 1分足なら1日≒330分、年率換算は√(330*245)
    bars_per_year = 330 * 245 / fwd_bars
    ann_sharpe = net.mean() / net.std() * np.sqrt(bars_per_year)
    t_stat = net.mean() / (net.std() / np.sqrt(n))
    return {
        "feature": feature_col,
        "fwd_bars": fwd_bars,
        "n_bars": n,
        "gross_bps": round(ls.mean() * 1e4, 2),
        "net_bps": round(net.mean() * 1e4, 2),
        "win_rate": round((net > 0).mean(), 3),
        "ann_sharpe": round(ann_sharpe, 2),
        "t_stat": round(t_stat, 2),
    }


# ─── 検証C: imbalance絶対値×方向フィルター ──────────────────────────────────

def test_threshold(df, feature_col, thresh=0.5, fwd_bars=1):
    """
    |feature| >= thresh の確信度が高いシグナルのみ発動
    Long: feature >= thresh, Short: feature <= -thresh
    """
    df = df.copy()
    df["fwd_ret"] = df.groupby("symbol")["close"].transform(lambda s: s.shift(-fwd_bars) / s - 1)
    df["signal"] = 0
    df.loc[df[feature_col] >= thresh, "signal"] = 1
    df.loc[df[feature_col] <= -thresh, "signal"] = -1
    df = df[df["signal"] != 0].dropna(subset=["fwd_ret"])

    if len(df) < 50:
        return {}

    ls_rows = []
    for ts, x in df.groupby("bucket_ts"):
        long_ret = x[x["signal"] == 1]["fwd_ret"].mean()
        short_ret = x[x["signal"] == -1]["fwd_ret"].mean()
        if pd.notna(long_ret) and pd.notna(short_ret):
            ls_rows.append({"ts": ts, "ls": long_ret - short_ret})

    if len(ls_rows) < 20:
        return {}
    ls = pd.DataFrame(ls_rows).set_index("ts")["ls"]
    net = ls - COST_BPS / 1e4
    n = len(net)
    bars_per_year = 330 * 245 / fwd_bars
    ann_sharpe = net.mean() / net.std() * np.sqrt(bars_per_year) if net.std() > 0 else np.nan
    t_stat = net.mean() / (net.std() / np.sqrt(n)) if net.std() > 0 else np.nan
    return {
        "feature": feature_col,
        "thresh": thresh,
        "fwd_bars": fwd_bars,
        "n_signals": len(df),
        "n_ts": n,
        "gross_bps": round(ls.mean() * 1e4, 2),
        "net_bps": round(net.mean() * 1e4, 2),
        "win_rate": round((net > 0).mean(), 3),
        "ann_sharpe": round(ann_sharpe, 2),
        "t_stat": round(t_stat, 2),
    }


def main():
    print("[RUN] auKabu 板imbalanceエッジ検証")
    conn = psycopg2.connect(**PG)
    bars = load_bars(conn)
    snaps = load_snaps(conn)
    conn.close()

    n_days = bars["bucket_ts"].dt.date.nunique()
    print(f"  bars: {len(bars):,}行 {n_days}営業日 {bars['symbol'].nunique()}銘柄")
    print(f"  snaps: {len(snaps):,}行 {snaps['symbol'].nunique()}銘柄")
    print(f"  ※ データ{n_days}日分: 統計パワーは低い。方向確認が目的。\n")

    FEATURES = ["avg_l1_imb", "avg_depth10_imb", "avg_w_imb", "avg_vwap_dev_pct"]
    SNAP_FEATURES = ["l1_imb", "depth10_imb", "w_imb"]

    # ─── A. 時系列IC (1分足)
    print("===== A. 時系列IC: imbalance_t → 次1分リターン =====")
    ts_rows = []
    for feat in FEATURES:
        r = test_timeseries(bars, feat, fwd_bars=1)
        if not r.empty:
            mean_ic = r["IC"].mean()
            mean_t = r["t_stat"].mean()
            sig = r[r["t_stat"].abs() >= 1.5]
            print(f"  {feat}: IC平均={mean_ic:.4f} t平均={mean_t:.2f} |t|≥1.5銘柄={len(sig)}/{len(r)}")
            r["feature"] = feat
            ts_rows.append(r)
    if ts_rows:
        ts_df = pd.concat(ts_rows)
        ts_df.to_csv(OUT / "timeseries_ic.csv", index=False)

    # ─── B. クロスセクション (1分足)
    print("\n===== B. クロスセクションL/S (1分足, コスト5bps後) =====")
    cs_rows = []
    for feat in FEATURES:
        for fwd in [1, 3, 5]:
            r = test_crosssection(bars, feat, fwd_bars=fwd)
            if r:
                cs_rows.append(r)
                print(f"  {feat} fwd={fwd}分: gross={r['gross_bps']:.2f}bps net={r['net_bps']:.2f}bps "
                      f"Sharpe={r['ann_sharpe']:.2f} t={r['t_stat']:.2f} n={r['n_bars']}")
    if cs_rows:
        pd.DataFrame(cs_rows).to_csv(OUT / "crosssection.csv", index=False)

    # ─── C. 閾値フィルター (1分足, l1_imb)
    print("\n===== C. 閾値フィルター: |l1_imb|≥thresh のみ (1分足) =====")
    thresh_rows = []
    for thresh in [0.3, 0.5, 0.7, 1.0]:
        r = test_threshold(bars, "avg_l1_imb", thresh=thresh, fwd_bars=1)
        if r:
            thresh_rows.append(r)
            print(f"  thresh={thresh}: n_sig={r['n_signals']} gross={r['gross_bps']:.2f}bps "
                  f"Sharpe={r['ann_sharpe']:.2f} t={r['t_stat']:.2f}")
    if thresh_rows:
        pd.DataFrame(thresh_rows).to_csv(OUT / "threshold.csv", index=False)

    # ─── D. 5秒データでのクロスセクション
    print("\n===== D. クロスセクションL/S (5秒足) =====")
    # 5秒足は1分足の代わりに使う
    snaps_r = snaps.rename(columns={"price": "close"})
    snap_rows = []
    for feat in SNAP_FEATURES:
        for fwd in [1, 3, 6, 12]:  # 5秒×1/3/6/12 = 5/15/30/60秒後
            r = test_crosssection(snaps_r, feat, fwd_bars=fwd)
            if r:
                snap_rows.append(r)
                print(f"  {feat} fwd={fwd*5}秒: gross={r['gross_bps']:.2f}bps net={r['net_bps']:.2f}bps "
                      f"Sharpe={r['ann_sharpe']:.2f} t={r['t_stat']:.2f} n={r['n_bars']}")
    if snap_rows:
        pd.DataFrame(snap_rows).to_csv(OUT / "snaps_crosssection.csv", index=False)

    # ─── サマリー
    print("\n===== サマリー: データ期間・統計パワーの評価 =====")
    print(f"  データ: {n_days}営業日 (目標: ≥60日で統計パワー確保)")
    print(f"  現時点のt値はN={n_days}日 × 22銘柄の制約で参考程度")
    print(f"  月次再実行スケジュール: 毎月末に本スクリプトを再実行し結果を更新")
    print(f"  昇格基準目安: クロスセクションSharpe≥1.0 かつ t≥2.0 かつ OOS確認")
    print("[DONE]")


if __name__ == "__main__":
    main()
