"""
PEAD拡張検証: セクター中立化 + サプライズフィルター + 極端分位集中
==============================================================================
前段(20260530_pead_price_reaction)が示したこと:
  - 全決算L/S PEAD: Sharpe ~0.38。「絞れば効く」が示唆された。
  - 生存エッジの弱さは「全決算を平等に10分位化」することで薄まるため。

本分析の改良3軸:
  A. 極端フィルター: |car0|≥ threshold（大きい驚きだけ）
  B. セクター中立化: 同業種内でランク→L/S（業種ベータを除去）
  C. サプライズ一致フィルター: OP実績 vs 前期FOP予想の符号がcar0と一致するもの

ベースデータ: pead_obs.csv（既存）+ symbol_master(sector) + fin_summary(OP/FOP)
"""
import sys; sys.stdout.reconfigure(line_buffering=True)
import os
from pathlib import Path
import numpy as np, pandas as pd
import psycopg2

BASE_OBS = Path(__file__).parent.parent / "20260530_pead_price_reaction" / "pead_obs.csv"
OUT = Path(__file__).parent
COST_BPS = 20.0
OOS_START = "2024-01-01"

PG = dict(
    host=os.environ.get("PGHOST", "localhost"),
    port=int(os.environ.get("PGPORT", 5432)),
    user=os.environ.get("PGUSER", "postgres"),
    password=os.environ.get("PGPASSWORD", "postgres"),
    dbname=os.environ.get("PGDATABASE", "market_data"),
)


def load_base():
    d = pd.read_csv(BASE_OBS)
    d["entry_date"] = pd.to_datetime(d["entry_date"])
    print(f"base obs={len(d):,} codes={d['code'].nunique()} days={d['entry_date'].nunique()}")
    return d


def load_sector(conn):
    sm = pd.read_sql(
        "SELECT code5 AS code, sector33_nm FROM symbol_master WHERE sector33_nm IS NOT NULL",
        conn
    )
    return sm.set_index("code")["sector33_nm"].to_dict()


def load_surprise(conn):
    """
    各(code, disc_date)の営業利益サプライズ = (OP_current - FOP_prev) / |FOP_prev|
    OP_prev は直前の決算発表時に示したFOP(次期会社予想)。
    fin_summary を時系列順にlagged joinして計算。
    """
    print("  loading fin_summary for surprise...")
    df = pd.read_sql(
        """SELECT code, disc_date,
                  NULLIF(payload->>'OP', '')::float  AS op_actual,
                  NULLIF(payload->>'FOP', '')::float AS fop_forecast
           FROM fin_summary
           WHERE disc_date >= '2019-01-01'
             AND payload->>'OP' NOT IN ('', 'null', 'None')
             AND payload->>'FOP' NOT IN ('', 'null', 'None')
           ORDER BY code, disc_date""",
        conn
    )
    df["disc_date"] = pd.to_datetime(df["disc_date"])
    # 前回発表のFOPをラグで取得
    df = df.sort_values(["code", "disc_date"]).reset_index(drop=True)
    df["prev_fop"] = df.groupby("code")["fop_forecast"].shift(1)
    # サプライズ: 前期予想比
    df["surprise"] = np.where(
        (df["prev_fop"].notna()) & (df["prev_fop"].abs() > 0),
        (df["op_actual"] - df["prev_fop"]) / df["prev_fop"].abs(),
        np.nan
    )
    return df[["code", "disc_date", "surprise"]].dropna()


def enrich(base, sector_map, surprise_df):
    """base に sector と surprise を結合。entry_dateからdisc_dateを逆算して照合。"""
    base = base.copy()
    # sector
    base["sector"] = base["code"].map(sector_map).fillna("不明")

    # surprise の結合: entry_date = react+1 = disc_date + ~2営業日
    # 厳密には disc_date で結合する必要があるが base に disc_date が無い。
    # 近似: entry_date-3 ~ entry_date-1 の間に disc_date が来る → entry_date-2 を結合キーとして試みる。
    # より頑健に: entry_date と disc_date の差分が 1~3 営業日のものを選ぶ。
    # ここでは entry_date を基準に 1~4 カレンダー日前の disc_date を候補とし、最新1件をマッチ。
    surp_lookup = surprise_df.copy()
    surp_lookup["entry_approx"] = surp_lookup["disc_date"] + pd.tseries.offsets.BusinessDay(2)
    # code + entry_approx でマージ（最寄り一致）
    base = base.merge(
        surp_lookup[["code", "entry_approx", "surprise"]].rename(columns={"entry_approx": "entry_date"}),
        on=["code", "entry_date"],
        how="left"
    )
    print(f"  surprise match rate: {base['surprise'].notna().mean():.1%}")
    return base


# ─── 評価関数群 ────────────────────────────────────────────────────────────────

def ls_eval(df, label, sector_neutral=False, car0_thresh=None, surprise_align=False):
    """
    L/S評価。各エントリー日で:
      - (sector_neutral=True) → 業種内ランク付け後、全業種のL/S集約
      - (car0_thresh) → |car0| < thresh を除外
      - (surprise_align=True) → surprise符号がcar0符号と一致するものに限定
    """
    df = df.copy()
    if car0_thresh is not None:
        df = df[df["car0"].abs() >= car0_thresh]
    if surprise_align:
        df = df[df["surprise"].notna()]
        # 符号一致: car0 > 0 かつ surprise > 0、または car0 < 0 かつ surprise < 0
        df = df[(df["car0"] * df["surprise"]) > 0]

    out = []
    for H in [5, 10, 20]:
        col = f"d{H}"
        sub = df.dropna(subset=[col]).copy()
        if len(sub) < 100:
            continue

        if sector_neutral:
            # 業種内で各エントリー日のcar0ランクをつけ、top/bottom群を決定
            def rank_sector(x):
                if len(x) < 4:
                    return x.assign(q=np.nan)
                x = x.copy()
                x["q"] = x["car0"].rank(pct=True)  # 0~1
                return x
            sub = sub.groupby(["entry_date", "sector"], group_keys=False).apply(rank_sector)
            sub = sub.dropna(subset=["q"])
            # top20%/bottom20%のみ
            long_grp = sub[sub["q"] >= 0.8]
            short_grp = sub[sub["q"] <= 0.2]
            # 各日のL-S
            long_ret = long_grp.groupby("entry_date")[col].mean()
            short_ret = short_grp.groupby("entry_date")[col].mean()
            common = long_ret.index.intersection(short_ret.index)
            if len(common) < 20:
                continue
            ls = (long_ret.loc[common] - short_ret.loc[common]) / 1e4
        else:
            # 全銘柄一緒に10分位
            parts = []
            for d, x in sub.groupby("entry_date"):
                if len(x) < 20:
                    continue
                x = x.copy()
                x["q"] = pd.qcut(x["car0"].rank(method="first"), 10, labels=False)
                parts.append(x)
            if not parts:
                continue
            sub2 = pd.concat(parts)
            wk = sub2.groupby(["entry_date", "q"])[col].mean().unstack("q")
            wk.columns = [int(c) for c in wk.columns]
            if 0 not in wk.columns or 9 not in wk.columns:
                continue
            ls = (wk[9] - wk[0]).dropna() / 1e4

        net = ls - COST_BPS / 1e4
        n_days = len(net)
        ann = net.mean() / net.std() * np.sqrt(245 / H) if net.std() > 0 else np.nan
        t_stat = net.mean() / (net.std() / np.sqrt(n_days)) if net.std() > 0 else np.nan
        n_obs = len(sub)
        out.append({
            "label": label, "hold": H, "n_days": n_days, "n_obs": n_obs,
            "gross_bps": round(ls.mean() * 1e4, 1),
            "net_bps": round(net.mean() * 1e4, 1),
            "win_rate": round((net > 0).mean(), 3),
            "sharpe": round(ann, 2),
            "t_stat": round(t_stat, 2),
        })
    return pd.DataFrame(out)


def run_scenario(df, name, **kwargs):
    splits = {"ALL": df, "IS": df[df["entry_date"] < OOS_START], "OOS": df[df["entry_date"] >= OOS_START]}
    rows = []
    for split, sub in splits.items():
        r = ls_eval(sub, split, **kwargs)
        if not r.empty:
            r.insert(0, "scenario", name)
            rows.append(r)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main():
    print("[RUN] PEAD enhanced analysis")
    base = load_base()
    conn = psycopg2.connect(**PG)
    sector_map = load_sector(conn)
    surprise_df = load_surprise(conn)
    conn.close()
    df = enrich(base, sector_map, surprise_df)
    df.to_csv(OUT / "pead_enriched.csv", index=False)
    print(f"  enriched: sector={df['sector'].notna().sum()} surprise={df['surprise'].notna().sum()}")

    results = []

    # ─── シナリオA: ベースライン（前段と同一）
    print("\n[A] ベースライン（全決算、業種混合、10分位L/S）")
    r = run_scenario(df, "A_baseline")
    results.append(r)
    print(r.to_string(index=False))

    # ─── シナリオB: 極端フィルターのみ（|car0|≥3%）
    print("\n[B] |car0|≥3% 極端フィルター")
    r = run_scenario(df, "B_car0_3pct", car0_thresh=0.03)
    results.append(r)
    print(r.to_string(index=False))

    # ─── シナリオC: セクター中立化（上下20%）
    print("\n[C] セクター中立化（業種内top/bottom20%）")
    r = run_scenario(df, "C_sector_neutral", sector_neutral=True)
    results.append(r)
    print(r.to_string(index=False))

    # ─── シナリオD: セクター中立 + 極端フィルター
    print("\n[D] セクター中立 + |car0|≥3%")
    r = run_scenario(df, "D_sector_extreme", sector_neutral=True, car0_thresh=0.03)
    results.append(r)
    print(r.to_string(index=False))

    # ─── シナリオE: セクター中立 + |car0|≥3% + サプライズ一致
    print("\n[E] セクター中立 + |car0|≥3% + サプライズ一致")
    r = run_scenario(df, "E_all_filters", sector_neutral=True, car0_thresh=0.03, surprise_align=True)
    results.append(r)
    print(r.to_string(index=False))

    all_results = pd.concat(results, ignore_index=True)
    all_results.to_csv(OUT / "results.csv", index=False)
    print("\n[DONE] results.csv 保存")
    return all_results


if __name__ == "__main__":
    main()
