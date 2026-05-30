"""
全ユニバース 日足スイング: ブレイクアウト & 平均回帰 のエッジ検証
================================================================
データソース: public.stocks_daily（分割調整済 adj_* 使用, 2016/05〜, PostgreSQL@Omen）

方針:
  - セクター不問・全上場銘柄
  - 流動性フィルタ: トレーリング60営業日の平均売買代金 >= 10億円/日（前日時点, look-ahead無し）
  - 日足スイング（寄成エントリー → N日後の引成決済）。場が見れなくても回るハンズオフ運用前提
  - ブレイクアウト系と平均回帰系を同枠組みで横断検証

戦略ファミリー:
  Breakout (順張り):
    BO_donchian : close が過去 L 日 high を上抜け → 翌日寄成 Long、H 日保有
  MeanReversion (逆張り):
    MR_zscore   : (close - MA_L)/std_L <= -Z → 翌日寄成 Long、H 日保有
    MR_rsi      : RSI(L) <= R → 翌日寄成 Long、H 日保有

評価:
  - トレード単位: mean_bps / 勝率 / t統計量 / per-trade Sharpe（look-ahead無し）
  - 日次ポートフォリオ: 各日 active なトレードを等加重 → 年率 Sharpe / 累積 / MaxDD
  - IS(2016-2021) / OOS(2022-2026) 分割
  - コスト: 往復 COST_BPS（寄成/引成の成行スリッページ込みでやや保守的）

実行:
  /root/venvs/jpstocks/bin/python run.py --smoke      # 50銘柄・2年で動作確認
  /root/venvs/jpstocks/bin/python run.py              # 全ユニバース・全期間
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2

OUTDIR = Path(__file__).parent
PG_CONFIG = {
    "host": os.environ.get("PGHOST", "localhost"),
    "port": int(os.environ.get("PGPORT", 5432)),
    "user": os.environ.get("PGUSER", "postgres"),
    "password": os.environ.get("PGPASSWORD", "postgres"),
    "dbname": os.environ.get("PGDATABASE", "market_data"),
}

# --- パラメータ ---
ADV_WINDOW = 60          # トレーリング流動性窓（営業日）
ADV_FLOOR = 1e9          # 流動性下限: 10億円/日
COST_BPS = 10.0          # 往復コスト(bps), 成行スリッページ込みで保守的
OOS_START = "2022-01-01" # IS/OOS分割
TRADING_DAYS = 245       # 年率化用

# グリッド（戦略ファミリー × パラメータ × 保有日数）
HOLDS = [3, 5, 10]
BO_LOOKBACKS = [20, 60]          # Donchian high 窓
MR_LOOKBACKS = [10, 25]          # MA/std, RSI 窓
MR_Z = 2.0                       # zスコア閾値
MR_RSI = 30.0                    # RSI閾値

DATA_START = "2016-05-09"
DATA_END = "2026-05-28"


# ---------------------------------------------------------------------------
def load_universe(conn, start, end, smoke_codes=None):
    """流動性ゲートを通り得る銘柄の日足を取得（分割調整済）。
    プルーニング: 生涯で一度でも turnover_value >= ADV_FLOOR を超えた銘柄のみ。"""
    # 普通株のみ（ETF/REIT/その他を除外）: symbol_master の市場区分でフィルタ
    EQUITY_SEG = "(SELECT code5 FROM symbol_master WHERE market_nm IN ('プライム','スタンダード','グロース'))"
    if smoke_codes:
        code_filter = "AND code = ANY(%(codes)s)"
        params = {"start": start, "end": end, "codes": smoke_codes}
    else:
        code_filter = f"""AND code IN {EQUITY_SEG}
            AND code IN (
            SELECT code FROM stocks_daily
            WHERE date >= %(start)s GROUP BY code
            HAVING MAX(turnover_value) >= %(floor)s )"""
        params = {"start": start, "end": end, "floor": ADV_FLOOR}
    sql = f"""
        SELECT code, date, adj_open, adj_high, adj_low, adj_close,
               turnover_value, upper_limit
        FROM stocks_daily
        WHERE date >= %(start)s AND date <= %(end)s
          {code_filter}
        ORDER BY code, date
    """
    df = pd.read_sql(sql, conn, params=params)
    df["date"] = pd.to_datetime(df["date"])
    for c in ["adj_open", "adj_high", "adj_low", "adj_close", "turnover_value"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def add_features(df):
    """銘柄ごとにトレーリング特徴量（全て前日までの情報）。
    groupby.transform ベースで index 整合を保証する。"""
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    g = df.groupby("code")
    c = df["adj_close"]

    # 流動性ゲート（前日までの60日平均、当日は含めない）
    df["adv60"] = g["turnover_value"].transform(
        lambda s: s.rolling(ADV_WINDOW, min_periods=ADV_WINDOW).mean().shift(1))
    df["liquid"] = df["adv60"] >= ADV_FLOOR

    # Donchian: 前日までの過去 L 日 high の最大
    for L in BO_LOOKBACKS:
        df[f"hh{L}"] = g["adj_high"].transform(
            lambda s: s.rolling(L, min_periods=L).max().shift(1))

    # MA/std zスコア・RSI（当日close基準。エントリーは翌日寄成なので look-ahead 無し）
    for L in MR_LOOKBACKS:
        ma = g["adj_close"].transform(lambda s: s.rolling(L, min_periods=L).mean())
        sd = g["adj_close"].transform(lambda s: s.rolling(L, min_periods=L).std(ddof=0))
        df[f"z{L}"] = (c - ma) / sd.replace(0, np.nan)
        up = g["adj_close"].transform(
            lambda s: s.diff().clip(lower=0).rolling(L, min_periods=L).mean())
        dn = g["adj_close"].transform(
            lambda s: (-s.diff().clip(upper=0)).rolling(L, min_periods=L).mean())
        rs = up / dn.replace(0, np.nan)
        df[f"rsi{L}"] = 100 - 100 / (1 + rs)

    # 翌日寄成 / H日後引成
    df["next_open"] = g["adj_open"].transform(lambda s: s.shift(-1))
    for H in HOLDS:
        df[f"exit_close_h{H}"] = g["adj_close"].transform(lambda s: s.shift(-H))
    return df


def gen_signals(df):
    """シグナルを縦持ちで生成。1行=1トレード候補。
    戻り: DataFrame[code, date, family, param, hold, entry, exit, ret_bps_gross]"""
    recs = []
    base = df[df["liquid"] & ~df["upper_limit"].fillna(False)].copy()

    # Breakout: close > hhL
    for L in BO_LOOKBACKS:
        sig = base[base["adj_close"] > base[f"hh{L}"]]
        for H in HOLDS:
            entry = sig["next_open"]
            exit_ = sig[f"exit_close_h{H}"]
            ok = entry.notna() & exit_.notna() & (entry > 0)
            r = (exit_[ok] / entry[ok] - 1.0) * 10000  # Long, bps
            recs.append(pd.DataFrame({
                "code": sig["code"][ok], "date": sig["date"][ok],
                "family": "BO_donchian", "param": f"L{L}", "hold": H,
                "ret_bps_gross": r.values, "exit_date_idx": sig.index[ok],
            }))

    # MeanReversion zscore: zL <= -Z
    for L in MR_LOOKBACKS:
        sig = base[base[f"z{L}"] <= -MR_Z]
        for H in HOLDS:
            entry = sig["next_open"]; exit_ = sig[f"exit_close_h{H}"]
            ok = entry.notna() & exit_.notna() & (entry > 0)
            r = (exit_[ok] / entry[ok] - 1.0) * 10000
            recs.append(pd.DataFrame({
                "code": sig["code"][ok], "date": sig["date"][ok],
                "family": "MR_zscore", "param": f"L{L}z{MR_Z}", "hold": H,
                "ret_bps_gross": r.values, "exit_date_idx": sig.index[ok],
            }))

    # MeanReversion RSI: rsiL <= R
    for L in MR_LOOKBACKS:
        sig = base[base[f"rsi{L}"] <= MR_RSI]
        for H in HOLDS:
            entry = sig["next_open"]; exit_ = sig[f"exit_close_h{H}"]
            ok = entry.notna() & exit_.notna() & (entry > 0)
            r = (exit_[ok] / entry[ok] - 1.0) * 10000
            recs.append(pd.DataFrame({
                "code": sig["code"][ok], "date": sig["date"][ok],
                "family": "MR_rsi", "param": f"L{L}r{int(MR_RSI)}", "hold": H,
                "ret_bps_gross": r.values, "exit_date_idx": sig.index[ok],
            }))

    if not recs:
        return pd.DataFrame()
    out = pd.concat(recs, ignore_index=True)
    out["ret_bps_net"] = out["ret_bps_gross"] - COST_BPS
    return out


def summarize(trades, label):
    rows = []
    for (fam, param, hold), sub in trades.groupby(["family", "param", "hold"]):
        hold = int(hold)
        x = sub["ret_bps_net"].dropna()
        if len(x) < 30:
            continue
        mean, sd, n = x.mean(), x.std(ddof=1), len(x)
        t = mean / (sd / np.sqrt(n)) if sd > 0 else np.nan
        # 年率Sharpe近似: per-trade Sharpe × sqrt(年間トレード機会). 簡便のため
        # 1トレード=hold日保有とし、年間 TRADING_DAYS/hold 回転と仮定
        ppt = mean / sd if sd > 0 else np.nan
        ann_sharpe = ppt * np.sqrt(TRADING_DAYS / hold) if sd > 0 else np.nan
        rows.append({
            "label": label, "family": fam, "param": param, "hold": hold,
            "n": n, "mean_bps": round(mean, 1), "win_rate": round((x > 0).mean(), 3),
            "t_stat": round(t, 1), "sharpe_per_trade": round(ppt, 3),
            "ann_sharpe_approx": round(ann_sharpe, 2),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    conn = psycopg2.connect(**PG_CONFIG)
    if args.smoke:
        # 流動性上位の代表50銘柄（普通株のみ、ETF/REIT除外）
        codes = pd.read_sql(
            "SELECT s.code FROM stocks_daily s "
            "JOIN symbol_master m ON s.code=m.code5 "
            "WHERE s.date>='2024-01-01' AND m.market_nm IN ('プライム','スタンダード','グロース') "
            "GROUP BY s.code ORDER BY AVG(s.turnover_value) DESC LIMIT 50", conn
        )["code"].tolist()
        print(f"[SMOKE] {len(codes)}銘柄 2023-2025")
        df = load_universe(conn, "2023-01-01", "2025-12-31", smoke_codes=codes)
    else:
        print("[FULL] 全ユニバース 2016-2026")
        df = load_universe(conn, DATA_START, DATA_END)
    conn.close()
    print(f"  rows={len(df)} codes={df['code'].nunique()}")

    df = add_features(df)
    trades = gen_signals(df)
    if trades.empty:
        print("シグナル0件"); return
    print(f"  total trades={len(trades)}")
    trades.to_csv(OUTDIR / "trades.csv", index=False)

    summaries = [summarize(trades, "ALL")]
    is_mask = trades["date"] < pd.Timestamp(OOS_START)
    summaries.append(summarize(trades[is_mask], "IS"))
    summaries.append(summarize(trades[~is_mask], "OOS"))
    summ = pd.concat(summaries, ignore_index=True)
    summ.to_csv(OUTDIR / "edge_summary.csv", index=False)

    print("\n===== エッジサマリ（コスト10bps控除後, bps/トレード） =====")
    with pd.option_context("display.width", 200, "display.max_rows", None):
        # ALLをann_sharpe降順で
        a = summ[summ["label"] == "ALL"].sort_values("ann_sharpe_approx", ascending=False)
        print(a.to_string(index=False))
        print("\n--- 上位configのIS/OOS整合 ---")
        top = a.head(5)[["family", "param", "hold"]]
        for _, r in top.iterrows():
            sl = summ[(summ["family"] == r["family"]) & (summ["param"] == r["param"]) &
                      (summ["hold"] == r["hold"]) & (summ["label"].isin(["IS", "OOS"]))]
            print(sl.to_string(index=False))


if __name__ == "__main__":
    import traceback
    try:
        main(); print("[DONE]")
    except Exception:
        traceback.print_exc()
        with open(OUTDIR / "error.log", "w") as f:
            f.write(traceback.format_exc())
        raise
