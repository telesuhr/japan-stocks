"""
auカブコム板シグナル（volumeSpike / vwapDeviation）のイントラエッジ検証
================================================================
データソース: public.stocks_intraday（1分足・確定データ・PostgreSQL@Omen）

目的:
  auカブコムのリアルタイム板監視で発火する4シグナルのうち、約定ベースで再現可能な
  2つ（出来高スパイク・VWAP乖離）について、発火後の短期リターンにエッジがあるかを
  長期1分足（2024/05〜）で検証する。板情報依存の imbalanceShift / marketPressure は対象外。

シグナル定義（auKabu/aukabu/strategy.py のロジックを1分足で近似再現）:
  - volumeSpike   : その分の出来高 / 手前N分平均 >= VOLSPIKE_RATIO
  - vwapDeviation : |close - 当日VWAP累計| / VWAP * 100 >= VWAPDEV_PCT

評価:
  - 各シグナルについて「順張り(momentum)」「逆張り(reversion)」の両方向を同時評価
  - シグナル発火後 5/15/30分の方向別フォワードリターン（同一営業日内）
  - 往復コスト COST_BPS 控除後、mean_bps / 勝率 / t統計量 / per-trade Sharpe
  - IS(〜2025/06) / OOS(2025/07〜) 分割で過学習チェック
  - 銘柄別・セクター別の内訳

注意:
  これは「採用判定」ではなく「エッジ探索＋検証パイプライン構築」。
  OOSで生き残ったシグナル/方向のみ次段（auカブコム実データ・板特徴量込み）へ進める。

実行:
  python run.py                  # 全22銘柄・全期間
  python run.py --smoke          # 1銘柄・1ヶ月の動作確認
  python run.py --codes 69200 80350 --start 2025-01-01 --end 2025-03-31
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import argparse
import os

import numpy as np
import pandas as pd
import psycopg2

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------
from pathlib import Path
OUTDIR = Path(__file__).parent

PG_CONFIG = {
    "host": os.environ.get("PGHOST", "localhost"),
    "port": int(os.environ.get("PGPORT", 5432)),
    "user": os.environ.get("PGUSER", "postgres"),
    "password": os.environ.get("PGPASSWORD", "postgres"),
    "dbname": os.environ.get("PGDATABASE", "market_data"),
}

# auカブコム監視22銘柄（5桁コード, name, sector）
SYMBOLS = {
    # 非鉄金属
    "57130": ("住友金属鉱山", "非鉄"), "57110": ("三菱マテリアル", "非鉄"),
    "57060": ("三井金属",     "非鉄"), "57140": ("DOWA",         "非鉄"),
    "50160": ("JX金属",       "非鉄"), "58010": ("古河電工",     "非鉄"),
    "58020": ("住友電工",     "非鉄"), "58030": ("フジクラ",     "非鉄"),
    # 半導体
    "80350": ("東京エレクトロン", "半導体"), "68570": ("アドバンテスト", "半導体"),
    "69200": ("レーザーテック",   "半導体"), "61460": ("ディスコ",       "半導体"),
    "77350": ("SCREEN",          "半導体"), "40630": ("信越化学",       "半導体"),
    "34360": ("SUMCO",           "半導体"), "77410": ("HOYA",           "半導体"),
    "69630": ("ローム",          "半導体"), "65260": ("ソシオネクスト", "半導体"),
    "99840": ("ソフトバンクG",   "半導体"), "40620": ("イビデン",       "半導体"),
    "67230": ("ルネサス",        "半導体"), "285A0": ("キオクシア",     "半導体"),
}

# シグナル閾値（strategy.py のデフォルトに準拠、1分足に合わせ調整）
VOLSPIKE_RATIO = 5.0       # 出来高スパイク倍率
VOLSPIKE_LOOKBACK = 20     # 平均を取る手前バー数（分）
VWAPDEV_PCT = 0.5          # VWAP乖離閾値(%)

HORIZONS = [5, 15, 30]     # フォワードリターン評価（分）
COST_BPS = 8.0             # 往復コスト(bps)
OOS_START = "2025-07-01"   # IS/OOS分割
COOLDOWN_MIN = 30          # 同一銘柄・同一シグナルのクールダウン(分)

DATA_START = "2024-05-01"
DATA_END = "2026-05-29"


# ---------------------------------------------------------------------------
# データ取得
# ---------------------------------------------------------------------------
def load_minute(conn, code: str, start: str, end: str) -> pd.DataFrame:
    """1分足を取得。当日VWAP累計も付与。index=ts(JST naive)。"""
    sql = """
        SELECT ts, open, high, low, close, volume, turnover_value
        FROM stocks_intraday
        WHERE code = %s AND ts >= %s AND ts < (%s::date + interval '1 day')
        ORDER BY ts
    """
    df = pd.read_sql(sql, conn, params=(code, start, end))
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.set_index("ts")
    for c in ["open", "high", "low", "close", "volume", "turnover_value"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["date"] = df.index.normalize()
    # 当日累計VWAP（turnover_value が約定代金。無ければ close*volume で代用）
    tv = df["turnover_value"].fillna(df["close"] * df["volume"])
    df["cum_tv"] = tv.groupby(df["date"]).cumsum()
    df["cum_vol"] = df["volume"].groupby(df["date"]).cumsum()
    df["vwap"] = df["cum_tv"] / df["cum_vol"].replace(0, np.nan)
    return df


# ---------------------------------------------------------------------------
# シグナル生成
# ---------------------------------------------------------------------------
def detect_signals(bars: pd.DataFrame) -> pd.DataFrame:
    """1分足からシグナル発火点を抽出。
    戻り: DataFrame[ts, kind, up_dir, ref_price]
      up_dir = +1: 発火時点で「上昇側」の事象（出来高陽線 / 上方VWAP乖離）
               -1: 「下落側」の事象（出来高陰線 / 下方VWAP乖離）
    順張り/逆張りは集計側で up_dir から導出する。"""
    if bars.empty:
        return pd.DataFrame(columns=["ts", "kind", "up_dir", "ref_price"])
    out = []

    # volumeSpike: cur / trailing-mean >= ratio（手前 lookback 分の平均、当日内限定）
    vols = bars["volume"]
    trail = vols.shift(1).rolling(VOLSPIKE_LOOKBACK, min_periods=VOLSPIKE_LOOKBACK).mean()
    # 当日境界をまたぐ平均を避けるため、その分の date と lookback本前の date が同じ行のみ採用
    same_day = bars["date"].values == bars["date"].shift(VOLSPIKE_LOOKBACK).values
    spike = ((vols / trail) >= VOLSPIKE_RATIO) & same_day
    for ts in bars.index[spike.fillna(False)]:
        row = bars.loc[ts]
        up = 1 if row["close"] >= row["open"] else -1
        out.append((ts, "volumeSpike", up, float(row["close"])))

    # vwapDeviation: |close/vwap-1|*100 >= thr
    dev_pct = (bars["close"] / bars["vwap"] - 1.0) * 100
    vd = dev_pct.abs() >= VWAPDEV_PCT
    for ts in bars.index[vd.fillna(False)]:
        up = 1 if dev_pct.loc[ts] > 0 else -1
        out.append((ts, "vwapDeviation", up, float(bars.loc[ts, "close"])))

    sig = pd.DataFrame(out, columns=["ts", "kind", "up_dir", "ref_price"])
    if sig.empty:
        return sig
    sig = sig.sort_values("ts").reset_index(drop=True)
    # 同一 kind 内でクールダウン
    keep, last_ts = [], {}
    for _, r in sig.iterrows():
        k = r["kind"]
        if k in last_ts and (r["ts"] - last_ts[k]).total_seconds() < COOLDOWN_MIN * 60:
            continue
        keep.append(r.name)
        last_ts[k] = r["ts"]
    return sig.loc[keep].reset_index(drop=True)


def forward_returns(bars: pd.DataFrame, sig: pd.DataFrame) -> pd.DataFrame:
    """各シグナルの発火後フォワードリターン（生の符号付き=上昇でプラス, bps, コスト前）。
    同一営業日内に限定。"""
    if sig.empty:
        return sig
    close = bars["close"]
    norm_idx = close.index.normalize()
    recs = []
    for _, r in sig.iterrows():
        t0, p0 = r["ts"], r["ref_price"]
        sess = t0.normalize()
        rec = {"ts": t0, "kind": r["kind"], "up_dir": int(r["up_dir"]),
               "ref_price": p0, "date": sess}
        for h in HORIZONS:
            t1 = t0 + pd.Timedelta(minutes=h)
            future = close[(close.index >= t1) & (norm_idx == sess)]
            rec[f"raw{h}"] = (float(future.iloc[0]) / p0 - 1.0) * 10000 if len(future) else np.nan
        recs.append(rec)
    return pd.DataFrame(recs)


# ---------------------------------------------------------------------------
# 集計（順張り/逆張り両方を評価）
# ---------------------------------------------------------------------------
def summarize(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """kind × direction(momentum/reversion) × horizon で集計。
    momentum  : up_dir 方向にエントリー → pnl = up_dir * raw - cost
    reversion : up_dir の逆方向        → pnl = -up_dir * raw - cost"""
    rows = []
    for kind in sorted(df["kind"].unique()):
        sub = df[df["kind"] == kind]
        for direction, sign in [("momentum", 1), ("reversion", -1)]:
            for h in HORIZONS:
                raw = sub[f"raw{h}"].dropna()
                if len(raw) < 2:
                    continue
                idx = raw.index
                pnl = sign * sub.loc[idx, "up_dir"] * raw - COST_BPS
                mean, sd, n = pnl.mean(), pnl.std(ddof=1), len(pnl)
                t = mean / (sd / np.sqrt(n)) if sd > 0 else np.nan
                rows.append({
                    "label": label, "kind": kind, "direction": direction,
                    "horizon_min": h, "n": n,
                    "mean_bps": round(mean, 2), "sd_bps": round(sd, 2),
                    "win_rate": round((pnl > 0).mean(), 3),
                    "t_stat": round(t, 2),
                    "sharpe_per_trade": round(mean / sd, 3) if sd > 0 else np.nan,
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--codes", nargs="*", default=None)
    ap.add_argument("--start", default=DATA_START)
    ap.add_argument("--end", default=DATA_END)
    args = ap.parse_args()

    if args.smoke:
        codes, start, end = ["69200"], "2025-01-01", "2025-01-31"
        print(f"[SMOKE] {codes} {start}..{end}")
    else:
        codes = args.codes or list(SYMBOLS.keys())
        start, end = args.start, args.end

    conn = psycopg2.connect(**PG_CONFIG)
    all_fwd = []
    for code in codes:
        name, sector = SYMBOLS.get(code, (code, "?"))
        bars = load_minute(conn, code, start, end)
        if bars.empty:
            print(f"  {code} {name}: 1分足なし（未上場/欠損）")
            continue
        sig = detect_signals(bars)
        fwd = forward_returns(bars, sig)
        if not fwd.empty:
            fwd["code"], fwd["name"], fwd["sector"] = code, name, sector
            all_fwd.append(fwd)
        n_vs = int((sig["kind"] == "volumeSpike").sum()) if not sig.empty else 0
        n_vd = int((sig["kind"] == "vwapDeviation").sum()) if not sig.empty else 0
        print(f"  {code} {name:<12} bars={len(bars):>7} volSpike={n_vs:>5} vwapDev={n_vd:>5}")
    conn.close()

    if not all_fwd:
        print("シグナル0件。")
        return

    fwd_all = pd.concat(all_fwd, ignore_index=True)
    fwd_all.to_csv(OUTDIR / "signals_forward_returns.csv", index=False)

    summaries = [summarize(fwd_all, "ALL")]
    is_mask = fwd_all["date"] < pd.Timestamp(OOS_START)
    summaries.append(summarize(fwd_all[is_mask], "IS"))
    summaries.append(summarize(fwd_all[~is_mask], "OOS"))
    for sec in sorted(fwd_all["sector"].unique()):
        summaries.append(summarize(fwd_all[fwd_all["sector"] == sec], f"sector:{sec}"))
    summ = pd.concat(summaries, ignore_index=True)
    summ.to_csv(OUTDIR / "edge_summary.csv", index=False)

    print("\n===== エッジサマリ（コスト8bps控除後, 単位bps） =====")
    with pd.option_context("display.width", 220, "display.max_rows", None):
        # 主要ビュー: ALL/IS/OOS の有意なものを上に
        view = summ[summ["label"].isin(["ALL", "IS", "OOS"])].copy()
        print(view.to_string(index=False))


if __name__ == "__main__":
    import traceback
    try:
        main()
        print("[DONE]")
    except Exception:
        with open(OUTDIR / "error.log", "w") as f:
            f.write(traceback.format_exc())
        traceback.print_exc()
        raise
