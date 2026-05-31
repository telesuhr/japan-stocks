"""
OBVダイバージェンス 定量検証
================================================================
PDFの主張: 「中小型株でOBVが価格より先行してシグナルを出す」
  - 価格横ばい or 下落中にOBVが上昇 → 買い集め(アキュミュレーション)→後に上昇
  - 価格横ばい or 上昇中にOBVが下落 → 分配(ディストリビューション)→後に下落

OBV = cumsum(sign(close_t - close_{t-1}) * volume_t)
ダイバージェンス = N日間の価格変化とN日間のOBV変化の符号が逆のケース

設計:
  ユニバース: 中小型株（scale_cat=1,2）かつ流動性 1~50億円/日（流動性ありかつ大型除外）
  ダイバージェンスシグナル:
    Bullish: price_chg_N < 0 かつ obv_chg_N > 0 → Long
    Bearish: price_chg_N > 0 かつ obv_chg_N < 0 → Short
  評価:
    - クロスセクションL/S（各週で均等）
    - フォワードリターン H=5/10/20 営業日
    - セクター中立も確認
    - IS: 2021-2023 / OOS: 2024-2026
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

ADV_LO = 1e9     # 1億円以上（流動性あり）
ADV_HI = 5e10    # 500億円未満（大型除外）
OBS_LOOKBACK = 20  # OBVダイバージェンス観測期間（営業日）
FWD_HOLDS = [5, 10, 20]
COST_BPS = 20.0
OOS_START = "2024-01-01"
START = "2020-01-01"  # OBV計算のための余裕を持たせる


def load(conn):
    print("  loading daily (mid-small caps)...")
    df = pd.read_sql(
        """SELECT d.code, d.date, d.adj_close, d.adj_volume, d.turnover_value,
                  s.sector33_nm, s.scale_cat
           FROM stocks_daily d
           JOIN symbol_master s ON s.code5 = d.code
           WHERE d.date >= %(s)s
             AND s.market_nm IN ('プライム', 'スタンダード', 'グロース')
             AND s.scale_cat IN ('TOPIX Small 1', 'TOPIX Small 2', 'TOPIX Mid400')
           ORDER BY d.code, d.date""",
        conn, params={"s": START}
    )
    df["date"] = pd.to_datetime(df["date"])
    for c in ["adj_close", "adj_volume", "turnover_value"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    tp = pd.read_sql("SELECT date, close FROM index_daily WHERE code='0000' ORDER BY date", conn)
    tp["date"] = pd.to_datetime(tp["date"])
    return df, tp


def build_obv(df):
    """OBV と ダイバージェンスシグナルを計算"""
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    g = df.groupby("code")

    # OBV: sign(close変化) * volume を累積
    df["close_chg"] = g["adj_close"].transform(lambda s: s.diff())
    df["obv_sign"] = np.sign(df["close_chg"].fillna(0))
    df["obv_increment"] = df["obv_sign"] * df["adj_volume"].fillna(0)
    df["obv"] = g["obv_increment"].transform(lambda s: s.cumsum())

    # N日間の変化率
    N = OBS_LOOKBACK
    df["price_chg_n"] = g["adj_close"].transform(lambda s: s.pct_change(N))
    df["obv_chg_n"] = g["obv"].transform(lambda s: s.diff(N) / (s.abs().rolling(N).mean() + 1))

    # 流動性フィルター
    df["adv60"] = g["turnover_value"].transform(lambda s: s.rolling(60, min_periods=40).mean())
    df = df[(df["adv60"] >= ADV_LO) & (df["adv60"] <= ADV_HI)]

    # ダイバージェンスシグナル
    df["bullish"] = (df["price_chg_n"] < -0.02) & (df["obv_chg_n"] > 0)  # 価格下落 + OBV上昇
    df["bearish"] = (df["price_chg_n"] > 0.02) & (df["obv_chg_n"] < 0)   # 価格上昇 + OBV下落
    df["signal"] = 0
    df.loc[df["bullish"], "signal"] = 1   # Long
    df.loc[df["bearish"], "signal"] = -1  # Short

    print(f"  bullish signals: {df['bullish'].sum():,}")
    print(f"  bearish signals: {df['bearish'].sum():,}")
    return df.dropna(subset=["price_chg_n", "obv_chg_n"])


def add_fwd_returns(df, tp):
    """フォワードリターン（TOPIX超過）を追加"""
    tp = tp.set_index("date")["close"]
    tp_fwd = {}
    for H in FWD_HOLDS:
        tp_fwd[H] = tp.shift(-H) / tp - 1

    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    g = df.groupby("code")

    for H in FWD_HOLDS:
        df[f"stock_fwd{H}"] = g["adj_close"].transform(lambda s: s.shift(-H)) / df["adj_close"] - 1
        df[f"tp_fwd{H}"] = df["date"].map(tp_fwd[H])
        df[f"fwd{H}"] = (df[f"stock_fwd{H}"] - df[f"tp_fwd{H}"]) * 1e4  # bps

    return df


def eval_signal(df, label, sector_neutral=False):
    """
    シグナル評価: bullish(+1) と bearish(-1) を weekly抽出してL/S評価
    """
    # 週次シグナル: 月/水/金で重複を避けるため週単位で最初の日を使う
    df = df.copy()
    df["week"] = df["date"].dt.to_period("W")
    # 各銘柄・週で最初のシグナルのみ
    df_sig = df[df["signal"] != 0].copy()
    df_sig = df_sig.sort_values(["code", "date"]).drop_duplicates(["code", "week"])

    if len(df_sig) < 100:
        return pd.DataFrame()

    out = []
    for H in FWD_HOLDS:
        col = f"fwd{H}"
        sub = df_sig.dropna(subset=[col]).copy()

        if sector_neutral:
            # セクター内でzスコア化
            sub["fz"] = sub.groupby(["date", "sector33_nm"])["signal"].transform(
                lambda s: s  # シグナルはすでに±1なのでそのまま
            )
            rank_col = "fz"
        else:
            rank_col = "signal"

        # 日次集計: bullish群(signal=+1)の平均 - bearish群(signal=-1)の平均
        wk_rows = []
        for d, x in sub.groupby("date"):
            bull = x[x["signal"] == 1][col].mean()
            bear = x[x["signal"] == -1][col].mean()
            if pd.notna(bull) and pd.notna(bear):
                wk_rows.append({"date": d, "ls": bull - bear})
        if not wk_rows:
            continue

        ls_df = pd.DataFrame(wk_rows).set_index("date")["ls"]
        # 週次にリサンプリング（各週1点）
        ls_wk = ls_df.resample("W").mean().dropna()
        if len(ls_wk) < 20:
            continue

        net = ls_wk - COST_BPS
        ann = net.mean() / net.std() * np.sqrt(52) if net.std() > 0 else np.nan
        t_stat = net.mean() / (net.std() / np.sqrt(len(net))) if net.std() > 0 else np.nan
        n_sig = len(sub)
        out.append({
            "label": label,
            "hold": H,
            "n_signal": n_sig,
            "n_weeks": len(ls_wk),
            "bull_avg_bps": round(df_sig[df_sig["signal"] == 1][col].mean(), 1),
            "bear_avg_bps": round(df_sig[df_sig["signal"] == -1][col].mean(), 1),
            "gross_bps": round(ls_wk.mean(), 1),
            "net_bps": round(net.mean(), 1),
            "win_rate": round((net > 0).mean(), 3),
            "sharpe": round(ann, 2),
            "t_stat": round(t_stat, 2),
        })
    return pd.DataFrame(out)


def main():
    print("[RUN] OBVダイバージェンス検証 (中小型株)")
    conn = psycopg2.connect(**PG)
    df, tp = load(conn)
    conn.close()
    print(f"  raw rows={len(df):,} codes={df['code'].nunique()}")

    df = build_obv(df)
    df = add_fwd_returns(df, tp)
    # 2021年以降のシグナルのみ評価
    df = df[df["date"] >= "2021-01-01"]
    df.to_csv(OUT / "signals.csv", index=False)
    print(f"  signals saved: {len(df):,} rows, bullish={df['bullish'].sum()}, bearish={df['bearish'].sum()}")

    rows = []
    for (label, sub), sn in [
        (("ALL", df), False),
        (("IS", df[df["date"] < OOS_START]), False),
        (("OOS", df[df["date"] >= OOS_START]), False),
        (("ALL_SN", df), True),
        (("IS_SN", df[df["date"] < OOS_START]), True),
        (("OOS_SN", df[df["date"] >= OOS_START]), True),
    ]:
        r = eval_signal(sub, label, sector_neutral=sn)
        if not r.empty:
            rows.append(r)

    results = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    results.to_csv(OUT / "results.csv", index=False)

    print("\n===== OBVダイバージェンス L/S結果（bullish群 - bearish群） =====")
    print("(コスト20bps後、週次Sharpe√52換算)")
    print(results.to_string(index=False))

    # 参考: 分位別平均リターン（シグナル強度確認）
    print("\n===== 参考: シグナル別平均20日後リターン（bps, TOPIX超過） =====")
    for sig_label, sig_val in [("Bullish(+1)", 1), ("Bearish(-1)", -1), ("NoSignal(0)", 0)]:
        sub = df[df["signal"] == sig_val]["fwd20"].dropna()
        print(f"  {sig_label}: n={len(sub):,} mean={sub.mean():.1f}bps t={sub.mean()/(sub.std()/len(sub)**0.5):.2f}")

    print("[DONE]")


if __name__ == "__main__":
    main()
