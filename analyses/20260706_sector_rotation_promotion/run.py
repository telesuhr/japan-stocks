"""セクターローテーション MOM L3K3 昇格検証。

前回(20260603)の4つの残課題を決着させる:
  (1) cap-weight化で等加重の幻が消えないか
  (2) 33業種精密化で改善するか
  (3) 下落局面耐性 (long-onlyフルベータの生命線)
  (4) 既存バスケットとの相関 (低相関寄与が新戦略の価値)
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
import os

import numpy as np
import pandas as pd

from jstock import db

HERE = os.path.dirname(os.path.abspath(__file__))
ADV_MIN = 1e8
MIN_MEMBERS = 5
IS_END = "2021-06"
EXCLUDE = {"その他", "", None, "－"}
COST_BPS = 10.0


# ---------------- データ取得 ----------------

def load():
    sm = db.read_sql("""
        SELECT code5 AS code, sector17_nm AS sec17, sector33_nm AS sec33
        FROM symbol_master
        WHERE market_nm IN ('プライム','スタンダード','グロース')
    """)
    print("月次集計をDBから取得中...")
    md = db.read_sql("""
        WITH m AS (
          SELECT code, date_trunc('month',date)::date AS ym, adj_close, close,
                 ROW_NUMBER() OVER (PARTITION BY code, date_trunc('month',date)
                                    ORDER BY date DESC) AS rn,
                 AVG(turnover_value) OVER (PARTITION BY code, date_trunc('month',date)) AS adv_m
          FROM stocks_daily
        )
        SELECT code, ym, adj_close, close, adv_m FROM m WHERE rn=1
    """)
    topix = db.read_sql("""
        WITH m AS (
          SELECT date_trunc('month',date)::date AS ym, close,
                 ROW_NUMBER() OVER (PARTITION BY date_trunc('month',date)
                                    ORDER BY date DESC) AS rn
          FROM index_daily WHERE code='0000'
        )
        SELECT ym, close FROM m WHERE rn=1 ORDER BY ym
    """)
    shares = db.read_sql("""
        SELECT code, disc_date,
               (payload->>'ShOutFY')::numeric
               - COALESCE(CASE WHEN (payload->>'TrShFY') ~ '^[0-9.]+$'
                               THEN (payload->>'TrShFY')::numeric END, 0) AS sh
        FROM fin_summary
        WHERE cur_per_type='FY' AND (payload->>'ShOutFY') ~ '^[0-9.]+$'
    """)
    md = md.merge(sm, on="code", how="inner")
    md["ym"] = pd.to_datetime(md["ym"]).dt.to_period("M")
    topix["ym"] = pd.to_datetime(topix["ym"]).dt.to_period("M")
    return md, topix.set_index("ym")["close"], shares


def attach_mcap(md, shares):
    """前月末時点 as-of の発行済株式数(自己株控除)×未調整closeで時価総額を張る。"""
    sh = shares.dropna().sort_values("disc_date")
    sh = sh[sh["sh"] > 0]
    md = md.sort_values("ym")
    md["ym_end"] = md["ym"].dt.to_timestamp(how="end").dt.normalize()
    sh["disc_date"] = pd.to_datetime(sh["disc_date"])
    merged = pd.merge_asof(
        md.sort_values("ym_end"), sh.rename(columns={"disc_date": "ym_end"}).sort_values("ym_end"),
        on="ym_end", by="code")
    merged["mcap"] = merged["close"] * merged["sh"]
    return merged.drop(columns=["ym_end"])


def build_sector_returns(md, sec_col, adv_min=ADV_MIN, weight=None):
    """業種月次リターン (index=ym)。weight=None: 等加重 / 'mcap': 前月末時価総額加重。"""
    df = md.dropna(subset=[sec_col]).copy()
    df = df[~df[sec_col].isin(EXCLUDE)]
    df = df.sort_values(["code", "ym"])
    df["ret"] = df.groupby("code")["adj_close"].pct_change()
    if weight == "mcap":
        df["w"] = df.groupby("code")["mcap"].shift(1)  # 前月末cap (lookahead回避)
    df = df[np.isfinite(df["ret"]) & (df["adv_m"] >= adv_min) & (df["ret"].abs() <= 1.0)]

    if weight == "mcap":
        df = df[np.isfinite(df["w"]) & (df["w"] > 0)]
        num = df.groupby(["ym", sec_col]).apply(
            lambda g: np.average(g["ret"], weights=g["w"]))
        sec_ret = num.unstack()
        n_mem = df.groupby(["ym", sec_col]).size().unstack()
    else:
        g = df.groupby(["ym", sec_col])["ret"]
        sec_ret = g.mean().unstack()
        n_mem = g.size().unstack()
    return sec_ret.where(n_mem >= MIN_MEMBERS).sort_index()


# ---------------- 評価 ----------------

def ann_sharpe(x):
    x = pd.Series(x).dropna()
    if len(x) < 6 or x.std() == 0:
        return np.nan, np.nan, len(x)
    return (float(x.mean() / x.std() * np.sqrt(12)),
            float(x.mean() / (x.std() / np.sqrt(len(x)))), len(x))


def split_line(name, s):
    idx = s.index.astype(str)
    a = ann_sharpe(s); i = ann_sharpe(s[idx <= IS_END]); o = ann_sharpe(s[idx > IS_END])
    print(f"  {name:34} 全{a[0]:+5.2f}(t{a[1]:+5.2f},N{a[2]:3}) "
          f"IS{i[0]:+5.2f} OOS{o[0]:+5.2f} 月{s.dropna().mean()*1e4:+5.0f}bps")
    return dict(name=name, sh=a[0], t=a[1], n=a[2], sh_is=i[0], sh_oos=o[0],
                bps=s.dropna().mean() * 1e4)


def run_mom(sec_ret, topix_ret, L=3, K=3, cost_bps=COST_BPS, excess=True):
    cum = (1 + sec_ret).rolling(L).apply(np.prod, raw=True) - 1
    months = sec_ret.index
    rets, idx_out, prev = [], [], set()
    for i in range(L, len(months) - 1):
        sig = cum.loc[months[i]].dropna()
        if len(sig) < 2 * K:
            continue
        nxt = months[i + 1]
        top = list(sig.sort_values(ascending=False).index[:K])
        r = sec_ret.loc[nxt, top].mean()
        r -= (len(set(top) - prev) / K) * 2 * cost_bps / 1e4
        prev = set(top)
        if excess:
            r -= topix_ret.get(nxt, np.nan)
        rets.append(r); idx_out.append(nxt)
    return pd.Series(rets, index=pd.PeriodIndex(idx_out, freq="M"))


def max_drawdown(monthly):
    eq = (1 + monthly.fillna(0)).cumprod()
    dd = eq / eq.cummax() - 1
    trough = dd.idxmin()
    peak_eq = eq.cummax()
    rec = eq[eq.index > trough]
    peak_val = peak_eq[trough]
    recovered = rec[rec >= peak_val]
    rec_months = (recovered.index[0] - trough).n if len(recovered) else None
    return float(dd.min()), trough, rec_months


def main():
    md, topix_close, shares = load()
    md = attach_mcap(md, shares)
    topix_ret_full = topix_close.pct_change()
    print(f"銘柄×月: {len(md):,}  時価総額付与率: {md['mcap'].notna().mean():.1%}")

    rows = []
    # ── (0) 前回再現: 17業種 等加重 ──
    s17_eq = build_sector_returns(md, "sec17")
    tp = topix_ret_full.reindex(s17_eq.index)
    print("\n=== (0) 前回再現 + (1) cap-weight + (2) 33業種 (TOPIX超過, コスト10bps) ===")
    ex = run_mom(s17_eq, tp)
    rows.append(split_line("17業種 等加重 MOM L3K3 (前回再現)", ex))

    # ── (1) cap-weight ──
    s17_cap = build_sector_returns(md, "sec17", weight="mcap")
    rows.append(split_line("17業種 cap-weight MOM L3K3", run_mom(s17_cap, tp)))
    s17_cap_hi = build_sector_returns(md, "sec17", adv_min=1e9, weight="mcap")
    rows.append(split_line("17業種 cap-weight ADV≥10億", run_mom(s17_cap_hi, tp)))

    # ── (2) 33業種 ──
    s33_eq = build_sector_returns(md, "sec33")
    rows.append(split_line("33業種 等加重 MOM L3K3", run_mom(s33_eq, tp)))
    rows.append(split_line("33業種 等加重 MOM L3K5", run_mom(s33_eq, tp, K=5)))
    s33_cap = build_sector_returns(md, "sec33", weight="mcap")
    rows.append(split_line("33業種 cap-weight MOM L3K3", run_mom(s33_cap, tp)))

    # 等加重 高流動性 (前回のベスト)
    s17_hi = build_sector_returns(md, "sec17", adv_min=1e9)
    rows.append(split_line("17業種 等加重 ADV≥10億 (前回2.09)", run_mom(s17_hi, tp)))

    pd.DataFrame(rows).to_csv(os.path.join(HERE, "results.csv"), index=False)

    # ── (3) 下落局面耐性 (最重要: long-only絶対リターンで測る) ──
    print("\n=== (3) 下落局面耐性: 17業種 等加重 MOM L3K3 絶対リターン ===")
    ab = run_mom(s17_eq, tp, excess=False)
    bench = tp.reindex(ab.index)
    sh_a = ann_sharpe(ab)
    print(f"  絶対 Sharpe{sh_a[0]:+.2f} (t{sh_a[1]:+.2f})  "
          f"TOPIX Sharpe{ann_sharpe(bench)[0]:+.2f}")
    mdd, trough, rec = max_drawdown(ab)
    mdd_b, trough_b, rec_b = max_drawdown(bench.dropna())
    print(f"  最大DD: 戦略{mdd:.1%} (底={trough}, 回復{rec}ヶ月) / "
          f"TOPIX {mdd_b:.1%} (底={trough_b}, 回復{rec_b}ヶ月)")
    down = bench < 0
    print(f"  TOPIX下落月({int(down.sum())}ヶ月): 戦略平均{ab[down].mean()*1e4:+.0f}bps "
          f"vs TOPIX{bench[down].mean()*1e4:+.0f}bps  勝率(戦略>TOPIX) "
          f"{(ab[down] > bench[down]).mean():.0%}")
    reg = (1 + topix_ret_full).rolling(12).apply(np.prod, raw=True) - 1
    reg = reg.reindex(ab.index)
    for nm, mask in [("強気(TOPIX 12M+)", reg > 0), ("弱気(TOPIX 12M-)", reg <= 0)]:
        seg, seg_b = ab[mask], bench[mask]
        s = ann_sharpe(seg)
        print(f"  {nm}: N{s[2]:3} 絶対Sh{s[0]:+.2f} 月{seg.mean()*1e4:+.0f}bps "
              f"(TOPIX月{seg_b.mean()*1e4:+.0f}bps)")
    worst = ab.nsmallest(5)
    print("  最悪5ヶ月:", ", ".join(f"{i}:{v:.1%}" for i, v in worst.items()))

    # ── (4) 既存バスケットとの相関 (共通期間2024-11〜, 月次) ──
    print("\n=== (4) 既存バスケットとの相関 (月次, 共通期間短い点に注意) ===")
    sl_path = os.path.join(HERE, "..", "20260531_portfolio_daily_sharpe")
    sleeves = pd.read_csv(os.path.join(sl_path, "sleeve_daily_returns.csv"),
                          index_col=0, parse_dates=True)
    basket = pd.read_csv(os.path.join(sl_path, "basket_daily.csv"),
                         index_col=0, parse_dates=True)["basket"]
    mon = (1 + pd.concat([sleeves, basket.rename("basket")], axis=1)) \
        .resample("ME").prod() - 1
    mon.index = mon.index.to_period("M")
    common = mon.index.intersection(ab.index)
    strat_m = ab.reindex(common)
    print(f"  共通期間: {common.min()}〜{common.max()} ({len(common)}ヶ月)")
    for col in mon.columns:
        c = strat_m.corr(mon.loc[common, col])
        print(f"  vs {col:26} 相関 {c:+.2f}")
    print(f"  vs TOPIX                      相関 {strat_m.corr(bench.reindex(common)):+.2f}"
          f"  ← long-onlyフルベータなので高くて当然。バスケット寄与は超過リターン側で評価")
    exm = ex.reindex(common)
    for col in mon.columns:
        c = exm.corr(mon.loc[common, col])
        print(f"  [TOPIX超過側] vs {col:12} 相関 {c:+.2f}")

    # ── 現在の33業種ランキング (アクショナブル) ──
    cum3 = (1 + s33_eq).rolling(3).apply(np.prod, raw=True) - 1
    latest = cum3.iloc[-1].dropna().sort_values(ascending=False)
    print(f"\n=== 現在の trailing-3m 33業種ランキング ({s33_eq.index[-1]}時点, 上位8) ===")
    for i, (s, v) in enumerate(latest.head(8).items(), 1):
        print(f"  {i:2}. {s:14} {v*100:+6.1f}%{' ★保有(上位3)' if i <= 3 else ''}")

    ab.to_csv(os.path.join(HERE, "strategy_monthly_abs.csv"))
    ex.to_csv(os.path.join(HERE, "strategy_monthly_excess.csv"))
    print("\nsaved results.csv, strategy_monthly_abs.csv, strategy_monthly_excess.csv")


if __name__ == "__main__":
    main()
