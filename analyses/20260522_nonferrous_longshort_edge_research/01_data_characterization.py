"""
非鉄金属8銘柄のクロスセクション特性を把握する。

目的:
1. 8銘柄を「銅鉱山系(5)」「電線系(3)」に分けて、サブグループ間の相関構造を見る
2. 日中リターン(寄→引、前場、後場)のクロスセクション分散を測る
   → 分散が大きいほど L/S にエッジが乗りやすい
3. 前場リターンと後場リターンの相関 → モメンタム/リバーサル判定
4. ペア銘柄の共和分傾向(電線2 vs 銅鉱山3) → ペアトレード可否
"""

import os
import psycopg2
import pandas as pd
import numpy as np

PG_CONFIG = {
    "host": os.environ.get("PGHOST", "omen"),
    "port": int(os.environ.get("PGPORT", 5432)),
    "user": os.environ.get("PGUSER", "postgres"),
    "dbname": os.environ.get("PGDATABASE", "market_data"),
}

# 8銘柄: code5 → (name, group)
SYMBOLS = {
    "57060": ("三井金属", "miner"),
    "57110": ("三菱マテリアル", "miner"),
    "57130": ("住友金属鉱山", "miner"),
    "57140": ("DOWA HD", "miner"),
    "50160": ("JX金属", "miner"),
    "58010": ("古河電工", "wire"),
    "58020": ("住友電工", "wire"),
    "58030": ("フジクラ", "wire"),
}

# JX金属が始まる日に揃える
START = "2025-04-01"
END = "2026-05-21"


def fetch_minute(code: str) -> pd.DataFrame:
    conn = psycopg2.connect(**PG_CONFIG)
    sql = """
        SELECT ts, open, high, low, close, volume, turnover_value
        FROM stocks_intraday
        WHERE code=%s AND ts>=%s AND ts<=%s
        ORDER BY ts
    """
    df = pd.read_sql(sql, conn, params=(code, START, END + " 23:59:59"))
    conn.close()
    df["ts"] = pd.to_datetime(df["ts"])
    return df.set_index("ts")


def build_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    """1分足から日次特徴量を作る。
    時間帯定義 (JST naive):
      AM: 09:00-11:30   PM: 12:30-15:30
      open=9:00 1分足のopen、close=15:30直近の1分足close
    """
    df = df.copy()
    df["date"] = df.index.normalize()
    grouped = df.groupby("date")

    rows = []
    for d, g in grouped:
        if len(g) < 30:
            continue
        am = g.between_time("09:00", "11:30")
        pm = g.between_time("12:30", "15:30")
        if len(am) < 10 or len(pm) < 10:
            continue
        o = am.iloc[0]["open"]
        am_close = am.iloc[-1]["close"]
        pm_open = pm.iloc[0]["open"]
        c = pm.iloc[-1]["close"]
        vol = g["volume"].sum()
        tv = g["turnover_value"].sum()
        rows.append({
            "date": d.normalize(),
            "open": o,
            "am_close": am_close,
            "pm_open": pm_open,
            "close": c,
            "volume": vol,
            "turnover": tv,
        })
    out = pd.DataFrame(rows).set_index("date")
    out["r_open_to_amclose"] = np.log(out["am_close"] / out["open"])
    out["r_pm"] = np.log(out["close"] / out["pm_open"])
    out["r_day"] = np.log(out["close"] / out["open"])
    out["r_lunch_gap"] = np.log(out["pm_open"] / out["am_close"])
    return out


def main():
    print(f"期間: {START} 〜 {END}")
    daily = {}
    for code, (name, grp) in SYMBOLS.items():
        df = fetch_minute(code)
        if df.empty:
            print(f"  {code} {name}: データなし")
            continue
        d = build_daily_features(df)
        daily[code] = d
        print(f"  {code} {name:12s} ({grp:5s}): {len(d)}日, "
              f"avg_turnover={d['turnover'].mean()/1e8:.1f}億円")

    # ---------- (1) 前場リターンのクロスセクション分散 ----------
    am = pd.DataFrame({c: d["r_open_to_amclose"] for c, d in daily.items()})
    pm = pd.DataFrame({c: d["r_pm"] for c, d in daily.items()})
    day = pd.DataFrame({c: d["r_day"] for c, d in daily.items()})

    # 共通日付に揃える
    common = am.dropna().index.intersection(pm.dropna().index)
    am = am.loc[common]
    pm = pm.loc[common]
    day = day.loc[common]

    print(f"\n共通日数: {len(common)}")
    print(f"\n--- 前場リターン (open→11:30) クロスセクション統計 ---")
    cs_std_am = am.std(axis=1)
    cs_range_am = am.max(axis=1) - am.min(axis=1)
    print(f"日次のクロスセクション std:   平均 {cs_std_am.mean()*100:.2f}% "
          f"(中央値 {cs_std_am.median()*100:.2f}%)")
    print(f"日次の最大 - 最小 リターン:  平均 {cs_range_am.mean()*100:.2f}% "
          f"(中央値 {cs_range_am.median()*100:.2f}%)")

    print(f"\n--- 後場リターン (12:30→15:30) クロスセクション統計 ---")
    cs_std_pm = pm.std(axis=1)
    cs_range_pm = pm.max(axis=1) - pm.min(axis=1)
    print(f"日次のクロスセクション std:   平均 {cs_std_pm.mean()*100:.2f}%")
    print(f"日次の最大 - 最小 リターン:  平均 {cs_range_pm.mean()*100:.2f}%")

    # ---------- (2) 前場 → 後場 のリードラグ (クロスセクション) ----------
    # 各日に前場リターンを銘柄横断でデモシードランク化 → 後場リターンとのスピアマン相関
    from scipy.stats import spearmanr
    rho_list = []
    for d in common:
        a = am.loc[d].dropna()
        p = pm.loc[d].reindex(a.index).dropna()
        if len(a) < 4:
            continue
        r, _ = spearmanr(a.loc[p.index], p)
        rho_list.append(r)
    rho_series = pd.Series(rho_list)
    print(f"\n--- 前場ランク vs 後場リターン スピアマン相関 ---")
    print(f"日次相関の平均: {rho_series.mean():+.3f}  "
          f"(>0 → モメンタム継続, <0 → リバーサル)")
    print(f"日次相関の中央値: {rho_series.median():+.3f}")
    print(f"正の日: {(rho_series>0).sum()}, 負の日: {(rho_series<0).sum()}")
    # t統計量 (1サンプル, mean=0 帰無)
    from scipy.stats import ttest_1samp
    t, p_val = ttest_1samp(rho_series.dropna(), 0)
    print(f"t統計量 (H0: mean=0): t={t:.2f}, p={p_val:.4f}")

    # ---------- (3) サブグループ間の β (miner vs wire) ----------
    print(f"\n--- グループ平均リターン (寄→引) ---")
    miners = [c for c, (_, g) in SYMBOLS.items() if g == "miner" and c in day.columns]
    wires = [c for c, (_, g) in SYMBOLS.items() if g == "wire" and c in day.columns]
    g_miner = day[miners].mean(axis=1)
    g_wire = day[wires].mean(axis=1)
    spread = g_wire - g_miner
    print(f"miner (n={len(miners)}) avg日次: {g_miner.mean()*100:+.3f}%, std {g_miner.std()*100:.2f}%")
    print(f"wire  (n={len(wires)}) avg日次: {g_wire.mean()*100:+.3f}%, std {g_wire.std()*100:.2f}%")
    print(f"スプレッド(wire-miner): 平均 {spread.mean()*100:+.3f}%/日, std {spread.std()*100:.2f}%")
    print(f"  → wire ロング・miner ショート naive Sharpe ≈ "
          f"{spread.mean()/spread.std()*np.sqrt(245):.2f}")

    # ---------- (4) 銘柄ペア相関 ----------
    corr = day.corr()
    print(f"\n--- 寄→引リターン 相関行列 ---")
    print(corr.round(2).to_string())

    # 保存
    out_dir = os.path.dirname(os.path.abspath(__file__))
    am.to_csv(os.path.join(out_dir, "am_returns.csv"))
    pm.to_csv(os.path.join(out_dir, "pm_returns.csv"))
    day.to_csv(os.path.join(out_dir, "day_returns.csv"))
    print(f"\n保存: {out_dir}/am_returns.csv, pm_returns.csv, day_returns.csv")


if __name__ == "__main__":
    main()
