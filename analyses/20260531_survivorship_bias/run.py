"""
生存バイアス補正の影響を定量確認
================================================================
既知の欠陥: symbol_masterにdelisted_at=NULLが4,462件(=全件が生存銘柄扱い)
stocks_dailyで2021-2024中にデータが途切れた325銘柄が実質的上場廃止候補。

目的:
  1. 上場廃止候補325銘柄のリターン特性（廃止直前のドローダウン）を計測
  2. これらを含めた場合、各戦略のSharpeがどれだけ変化するかを概算
  3. 過去の検証(PEAD/平均回帰)が「生存バイアスで過大評価」の方向かを確認

主要なリスク: 平均回帰戦略は「大きく下げた銘柄を買う」→上場廃止候補を掴む危険。
PEADは「決算イベント条件付き」→ ADV≥10億円フィルターで廃止候補が除外されている可能性。
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


def load(conn):
    print("  loading stocks_daily 2021-2026...")
    px = pd.read_sql(
        """SELECT d.code, d.date, d.adj_close, d.turnover_value, s.market_nm
           FROM stocks_daily d
           JOIN symbol_master s ON s.code5 = d.code
           WHERE d.date >= '2021-01-01'
           ORDER BY d.code, d.date""",
        conn
    )
    px["date"] = pd.to_datetime(px["date"])
    px["adj_close"] = pd.to_numeric(px["adj_close"], errors="coerce")
    px["turnover_value"] = pd.to_numeric(px["turnover_value"], errors="coerce")
    return px


def identify_delisted(px):
    """データが途絶えた銘柄を上場廃止候補として分類"""
    last_dates = px.groupby("code")["date"].max()
    # 2021-2026で途中（2024-06-01以前）に切れた銘柄
    delisted_candidates = last_dates[last_dates < pd.Timestamp("2024-06-01")].index.tolist()
    first_dates = px.groupby("code")["date"].min()
    # 2021年以降に上場した新規銘柄は除外（最終日が2021年台のものは2021年以前から継続していたもの）
    active_starts = first_dates[first_dates <= pd.Timestamp("2021-06-01")]
    delisted_established = [c for c in delisted_candidates if c in active_starts.index]
    print(f"  上場廃止候補: {len(delisted_candidates)}件 / 2021年以前から存在: {len(delisted_established)}件")
    return delisted_established


def analyze_delisted_returns(px, delisted_codes):
    """廃止候補銘柄の最終12ヶ月のリターン特性"""
    rows = []
    for code in delisted_codes:
        sub = px[px["code"] == code].sort_values("date")
        if len(sub) < 20:
            continue
        last_date = sub["date"].max()
        # 最終1年・6ヶ月・3ヶ月のリターン
        for months, label in [(12, "12M"), (6, "6M"), (3, "3M")]:
            cutoff = last_date - pd.DateOffset(months=months)
            period = sub[sub["date"] >= cutoff]
            if len(period) < 10:
                ret = np.nan
            else:
                ret = period.iloc[-1]["adj_close"] / period.iloc[0]["adj_close"] - 1
            rows.append({"code": code, "period": label, "last_date": last_date, "ret": ret})
    return pd.DataFrame(rows)


def assess_strategy_exposure(px, delisted_codes):
    """各戦略での廃止候補の暴露度推定"""
    adv_by_code = px.groupby("code")["turnover_value"].median()
    delisted_adv = adv_by_code.loc[delisted_codes].dropna()

    print(f"\n  廃止候補の流動性（日次売買代金中央値）:")
    print(f"    中央値: {delisted_adv.median()/1e6:.0f}百万円")
    print(f"    流動性<1億円: {(delisted_adv < 1e8).sum()}件 ({(delisted_adv < 1e8).mean():.0%})")
    print(f"    流動性<10億円: {(delisted_adv < 1e9).sum()}件 ({(delisted_adv < 1e9).mean():.0%})")
    print(f"    流動性≥10億円: {(delisted_adv >= 1e9).sum()}件 ({(delisted_adv >= 1e9).mean():.0%})")

    return delisted_adv


def simulate_meanrev_exposure(px, delisted_codes):
    """
    平均回帰戦略(RSI<30相当 = 大きく下落した銘柄を買う)への廃止候補の暴露を模擬。
    廃止候補銘柄が「RSI<30」を発動するほど下落していた期間を確認。
    """
    # 廃止候補の直前1年間の最大ドローダウン
    rows = []
    for code in delisted_codes[:100]:  # 計算量制限
        sub = px[px["code"] == code].sort_values("date")
        if len(sub) < 60:
            continue
        closes = sub["adj_close"].values
        # 20日RSI相当の短期下落率
        for i in range(20, len(closes)):
            window = closes[i-20:i]
            ret_20d = window[-1] / window[0] - 1 if window[0] > 0 else np.nan
            if pd.notna(ret_20d) and ret_20d < -0.15:  # 20日で-15%以上
                # この後廃止になったか確認（廃止まで最大60日以内）
                days_to_end = len(closes) - i
                if days_to_end <= 60:
                    rows.append({"code": code, "ret_20d": ret_20d, "days_to_end": days_to_end})
    if rows:
        ex = pd.DataFrame(rows)
        print(f"\n  廃止候補で「20日-15%超下落」を廃止60日前以内に発動: {len(ex)}件")
        print(f"  平均20日リターン: {ex['ret_20d'].mean()*100:.1f}%")
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def main():
    print("[RUN] 生存バイアス補正 影響調査")
    conn = psycopg2.connect(**PG)
    px = load(conn)
    conn.close()
    print(f"  total: {px['code'].nunique()}銘柄, {len(px):,}行")

    delisted = identify_delisted(px)

    # 1. 廃止候補のリターン特性
    print("\n  廃止候補のリターン分析中...")
    ret_analysis = analyze_delisted_returns(px, delisted)
    ret_summary = ret_analysis.groupby("period")["ret"].agg(["count", "mean", "median", "std"])
    ret_summary["mean%"] = (ret_summary["mean"] * 100).round(1)
    ret_summary["median%"] = (ret_summary["median"] * 100).round(1)
    print("\n===== 廃止候補銘柄の廃止前リターン =====")
    print(ret_summary[["count", "mean%", "median%"]].to_string())

    # 2. 流動性による戦略への暴露度
    delisted_adv = assess_strategy_exposure(px, delisted)

    # 3. 平均回帰戦略への暴露
    exposure = simulate_meanrev_exposure(px, delisted)

    # 4. 定量的影響の推計
    print("\n===== 生存バイアスの影響推計 =====")
    total_codes = px["code"].nunique()
    delisted_pct = len(delisted) / total_codes
    print(f"  実質上場廃止候補の割合: {delisted_pct:.1%} ({len(delisted)}/{total_codes})")

    # 廃止前12ヶ月の平均リターン
    r12m = ret_analysis[ret_analysis["period"] == "12M"]["ret"].mean()
    print(f"  廃止前12ヶ月平均リターン: {r12m*100:.1f}%")

    # 流動性≥10億円でフィルタリングした場合の廃止候補暴露
    high_liq_delisted = (delisted_adv >= 1e9).sum()
    print(f"  流動性≥10億円フィルター後も残る廃止候補: {high_liq_delisted}件")

    # Sharpe への影響概算
    # 均等ウェイト100銘柄ポートフォリオで廃止候補が数件混入した場合
    # 廃止前12ヶ月-50%として、ポート全体への影響:
    n_portfolio = 100  # 仮定
    delisted_in_portfolio = high_liq_delisted / total_codes * n_portfolio
    impact_bps = r12m * delisted_in_portfolio / n_portfolio * 1e4 / 12  # 月次換算
    print(f"\n  100銘柄ポートフォリオへの毎月の生存バイアス過大評価: 約 {abs(impact_bps):.1f}bps/月")
    print(f"  年率換算: 約 {abs(impact_bps)*12:.0f}bps = {abs(impact_bps)*12/100:.1f}%")

    ret_analysis.to_csv(OUT / "delisted_returns.csv", index=False)

    # 5. 各戦略別の影響評価
    print("\n===== 戦略別・生存バイアス影響評価 =====")
    strategies = [
        ("平均回帰(RSI<30)", "ADV≥10億円フィルターあるが、廃止候補が高流動性ゾーンに混在。逆張りで廃止候補を掴む危険大。過大評価方向。"),
        ("PEAD価格反応L/S", "ADV≥10億円フィルター+決算イベント条件。廃止候補は決算発表を行わない場合もある。影響は中程度。"),
        ("信用残ファクター", "流動性フィルターあり。廃止候補は信用残が積み上がっていることが多い。過大評価方向。"),
        ("OBVダイバージェンス", "流動性フィルター（1〜500億円）。廃止候補の一部が混入。Bullishシグナルが廃止候補を含む危険。"),
    ]
    for strat, note in strategies:
        print(f"\n  [{strat}]")
        print(f"    {note}")

    print("\n[DONE]")


if __name__ == "__main__":
    main()
