"""
信用残ファクター 月次リバランス版
================================================================
前段(20260530_jquants_margin_short_factors)の課題:
  - 週次リバランス(5営業日): コスト40bps/週 > 生スプレッド12bps/週 → 全滅
  - margin_ratio のコスト前Sharpe 0.41 = 弱い予測力は存在する

本検証の仮説:
  月次リバランス(20営業日)にすれば:
    コスト ≒ 10bps/月 (往復) ← 週次の1/4
    生スプレッドが4週分累積 ≒ 4×12bps = 48bps/月（楽観的上限）
  → コスト前Sharpeが0.41で残るなら20日後の累積も正?

追加: サイズ・セクター中立化でα純粋性を確認

コスト仮定:
  - 月次入替: 往復 10bps (流動性≥10億円なら現実的)
  - ショートコスト: 最低限のみ（月次なら無視できる水準）
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

ADV_FLOOR = 1e9
N_Q = 5           # 月次はサンプル少ないので5分位
FWD_DAYS = 20     # 約1ヶ月
COST_BPS = 10.0   # 月次往復コスト
OOS_START = "2024-01-01"
START = "2021-01-01"
END = "2026-05-29"


def load(conn):
    print("  loading margin (monthly end)...")
    mg = pd.read_sql(
        """SELECT code, date, shrt_vol, long_vol
           FROM jquants_margin_interest
           WHERE date >= %(s)s ORDER BY code, date""",
        conn, params={"s": START}
    )
    mg["date"] = pd.to_datetime(mg["date"])
    for c in ["shrt_vol", "long_vol"]:
        mg[c] = pd.to_numeric(mg[c], errors="coerce")

    print("  loading daily...")
    px = pd.read_sql(
        """SELECT s.code5 AS code, d.date, d.adj_close, d.turnover_value,
                  s.sector33_nm, s.scale_cat
           FROM stocks_daily d
           JOIN symbol_master s ON s.code5 = d.code
           WHERE d.date >= %(s)s AND d.date <= %(e)s
             AND s.market_nm IN ('プライム', 'スタンダード', 'グロース')
           ORDER BY d.code, d.date""",
        conn, params={"s": START, "e": END}
    )
    px["date"] = pd.to_datetime(px["date"])
    for c in ["adj_close", "turnover_value"]:
        px[c] = pd.to_numeric(px[c], errors="coerce")

    print("  loading TOPIX...")
    tp = pd.read_sql(
        "SELECT date, close FROM index_daily WHERE code='0000' ORDER BY date",
        conn
    )
    tp["date"] = pd.to_datetime(tp["date"])
    return mg, px, tp


def build(mg, px, tp):
    px = px.sort_values(["code", "date"]).reset_index(drop=True)
    g = px.groupby("code")
    px["adv60"] = g["turnover_value"].transform(lambda s: s.rolling(60, min_periods=40).mean())

    # TOPIX超過 20日後リターン
    tp = tp.set_index("date")["close"]
    tp_fwd = tp.shift(-FWD_DAYS) / tp - 1
    px["tp_ret20"] = px["date"].map(tp_fwd)
    px["fwd20"] = g["adj_close"].transform(lambda s: s.shift(-FWD_DAYS)) / px["adj_close"] - 1
    px["fwd20_xs"] = px["fwd20"] - px["tp_ret20"]  # TOPIX超過

    # 月次日付リスト（月末の営業日）
    all_dates = px["date"].sort_values().unique()
    month_ends = pd.to_datetime(pd.Series(all_dates)).groupby(
        pd.to_datetime(pd.Series(all_dates)).dt.to_period("M")
    ).last().values
    month_ends = pd.DatetimeIndex(month_ends)

    # margin を月末にサンプリング（margin は週次なので month_end と最寄り結合）
    mg = mg.sort_values(["code", "date"]).reset_index(drop=True)
    mg["margin_ratio"] = mg["long_vol"] / mg["shrt_vol"].replace(0, np.nan)
    mg["long_chg"] = mg.groupby("code")["long_vol"].transform(lambda s: s.pct_change())
    mg["short_chg"] = mg.groupby("code")["shrt_vol"].transform(lambda s: s.pct_change())
    mg["net_chg"] = mg["long_chg"] - mg["short_chg"]

    # 月末日を基準に margin を merge_asof (backward, 最大14日)
    panel_rows = []
    px_snap = px[px["date"].isin(month_ends)].copy()
    px_snap = px_snap[px_snap["adv60"] >= ADV_FLOOR]

    for me in month_ends:
        px_me = px_snap[px_snap["date"] == me][
            ["code", "fwd20_xs", "sector33_nm", "scale_cat"]
        ].copy()
        if len(px_me) < 50:
            continue
        # margin 最新値（≤ month_end, 最大14日前まで）
        mg_me = mg[(mg["date"] <= me) & (mg["date"] >= me - pd.Timedelta(days=14))]
        mg_me = mg_me.sort_values("date").groupby("code").last().reset_index()
        mg_me = mg_me[["code", "margin_ratio", "long_chg", "short_chg", "net_chg"]]
        merged = px_me.merge(mg_me, on="code", how="inner")
        merged["month_end"] = me
        panel_rows.append(merged)

    panel = pd.concat(panel_rows, ignore_index=True)
    print(f"  monthly panel: {len(panel):,} rows, {panel['month_end'].nunique()} months, {panel['code'].nunique()} codes")
    return panel


def ls_eval(panel, label, factor, neutral_sector=False, neutral_size=False):
    """月次L/S: 高分位 - 低分位"""
    sub = panel.dropna(subset=[factor, "fwd20_xs"]).copy()
    if neutral_sector or neutral_size:
        group_cols = []
        if neutral_sector:
            group_cols.append("sector33_nm")
        if neutral_size:
            group_cols.append("scale_cat")
        # グループ内でzスコア化してクロスセクションを平坦化
        sub["factor_z"] = sub.groupby(["month_end"] + group_cols)[factor].transform(
            lambda s: (s - s.mean()) / s.std() if s.std() > 0 else 0.0
        )
        rank_col = "factor_z"
    else:
        rank_col = factor

    results = []
    for me, x in sub.groupby("month_end"):
        if len(x) < N_Q * 4:
            continue
        x = x.copy()
        x["q"] = pd.qcut(x[rank_col].rank(method="first"), N_Q, labels=False)
        long_ret = x[x["q"] == N_Q - 1]["fwd20_xs"].mean()
        short_ret = x[x["q"] == 0]["fwd20_xs"].mean()
        ls = long_ret - short_ret
        results.append({"month_end": me, "ls_xs": ls})

    if not results:
        return {}
    df_r = pd.DataFrame(results).set_index("month_end")["ls_xs"]
    net = df_r - COST_BPS / 1e4
    n = len(net)
    if n < 10 or net.std() == 0:
        return {}
    ann = net.mean() / net.std() * np.sqrt(12)  # 月次なので√12
    t_stat = net.mean() / (net.std() / np.sqrt(n))
    return {
        "label": label, "factor": factor,
        "n_months": n,
        "gross_bps": round(df_r.mean() * 1e4, 1),
        "net_bps": round(net.mean() * 1e4, 1),
        "win_rate": round((net > 0).mean(), 3),
        "sharpe": round(ann, 2),
        "t_stat": round(t_stat, 2),
    }


def main():
    print("[RUN] 信用残ファクター 月次リバランス検証")
    conn = psycopg2.connect(**PG)
    mg, px, tp = load(conn)
    conn.close()
    panel = build(mg, px, tp)
    panel.to_csv(OUT / "panel.csv", index=False)

    factors = ["margin_ratio", "long_chg", "short_chg", "net_chg"]
    neutral_modes = [
        ("mixed", False, False),
        ("sector_neutral", True, False),
        ("size_neutral", False, True),
        ("sector_size_neutral", True, True),
    ]

    rows = []
    for split, sub in [
        ("ALL", panel),
        ("IS", panel[panel["month_end"] < OOS_START]),
        ("OOS", panel[panel["month_end"] >= OOS_START]),
    ]:
        for mode_name, ns, nz in neutral_modes:
            for f in factors:
                r = ls_eval(sub, split, f, neutral_sector=ns, neutral_size=nz)
                if r:
                    r["mode"] = mode_name
                    rows.append(r)

    results = pd.DataFrame(rows)
    results.to_csv(OUT / "results.csv", index=False)

    # サマリー表示: margin_ratio（最有望ファクター）のみ
    print("\n===== margin_ratio 月次L/S (コスト10bps後) =====")
    mr = results[results["factor"] == "margin_ratio"]
    print(mr[["mode", "label", "n_months", "gross_bps", "net_bps", "win_rate", "sharpe", "t_stat"]].to_string(index=False))

    print("\n===== 全ファクター × mode × ALL =====")
    all_res = results[results["label"] == "ALL"]
    print(all_res[["mode", "factor", "n_months", "gross_bps", "net_bps", "sharpe", "t_stat"]].to_string(index=False))
    print("[DONE]")


if __name__ == "__main__":
    main()
