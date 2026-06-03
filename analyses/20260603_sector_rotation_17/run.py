"""Phase1: 17業種セクターローテーション検証。

問い: 「資金循環を見極めて利確→次のセクター」は取引可能なエッジか?
2つの正反対の仮説を同じ厳密さで決着させる:
  (A) モメンタム継続 = 強い業種を持ち続け、上位から外れたら利確
  (B) 資金循環(平均回帰) = 過熱業種で利確し出遅れ業種を拾う

手法:
- 17業種の等加重(流動性フィルタ済)月次リターン系列を stocks_daily から構築
- 月次非重複リバランス(H=1ヶ月) → √12 年率化、オーバーラップの幻なし
- trailing L=1/3/6/12ヶ月の業種リターンで順位付け
- MOM: 上位K業種ロング / REV: 下位K業種ロング / L/S: 上位K−下位K
- 規範IC: trailing-L順位 vs 翌月業種リターンのクロスセクション順位相関(17業種/月)
       正IC=モメンタム、負IC=資金循環(平均回帰)
- 評価: コスト後 net Sharpe・IS/OOS一貫・t値・TOPIX買い持ち超過
"""
import sys, os
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
import pandas as pd
import psycopg2

PG = dict(host=os.environ.get("PGHOST", "localhost"), port=5432,
          user="postgres", password="postgres", dbname="market_data")

ADV_MIN = 1e8         # 月平均売買代金 ≥ 1億円 (流動性フィルタ)
MIN_MEMBERS = 5       # 業種あたり最低有効銘柄数
IS_END = "2021-06"    # IS: 〜2021-06, OOS: 2021-07〜 (約半々)
EXCLUDE = {"その他", None, ""}

# ---------------- データ取得 ----------------

def load():
    conn = psycopg2.connect(**PG)
    sm = pd.read_sql("""
        SELECT code5 AS code, sector17_nm AS sec, market_nm
        FROM symbol_master
        WHERE market_nm IN ('プライム','スタンダード','グロース')
    """, conn)
    sm = sm[~sm["sec"].isin(EXCLUDE)].dropna(subset=["sec"])
    sec_of = dict(zip(sm["code"], sm["sec"]))

    print("月次集計をDBから取得中...")
    md = pd.read_sql("""
        WITH m AS (
          SELECT code, date_trunc('month',date)::date AS ym, date, adj_close,
                 ROW_NUMBER() OVER (PARTITION BY code, date_trunc('month',date) ORDER BY date DESC) AS rn,
                 AVG(turnover_value) OVER (PARTITION BY code, date_trunc('month',date)) AS adv_m
          FROM stocks_daily
        )
        SELECT code, ym, adj_close, adv_m FROM m WHERE rn=1
    """, conn)
    topix = pd.read_sql("""
        WITH m AS (
          SELECT date_trunc('month',date)::date AS ym, date, close,
                 ROW_NUMBER() OVER (PARTITION BY date_trunc('month',date) ORDER BY date DESC) AS rn
          FROM index_daily WHERE code='0000'
        )
        SELECT ym, close FROM m WHERE rn=1 ORDER BY ym
    """, conn)
    conn.close()

    md = md[md["code"].isin(sec_of)].copy()
    md["sec"] = md["code"].map(sec_of)
    md["ym"] = pd.to_datetime(md["ym"]).dt.to_period("M")
    topix["ym"] = pd.to_datetime(topix["ym"]).dt.to_period("M")
    return md, topix.set_index("ym")["close"]


def build_sector_returns(md, adv_min=ADV_MIN):
    """流動性フィルタ後の等加重 業種月次リターン系列を返す (index=ym, cols=17業種)。"""
    md = md.sort_values(["code", "ym"])
    md["ret"] = md.groupby("code")["adj_close"].pct_change()
    # 翌月リターンを当月に張るのではなく、当月リターンは当月adv基準で採用
    # 流動性: 当月の adv_m ≥ adv_min かつ リターンが有限
    md = md[np.isfinite(md["ret"])]
    md = md[md["adv_m"] >= adv_min]
    # 極端な分割未調整等を除外 (|月次| > 100%)
    md = md[md["ret"].abs() <= 1.0]

    g = md.groupby(["ym", "sec"])["ret"]
    sec_ret = g.mean().unstack()
    n_mem = g.size().unstack()
    sec_ret = sec_ret.where(n_mem >= MIN_MEMBERS)  # 銘柄少ない業種月はNaN
    sec_ret = sec_ret.sort_index()
    return sec_ret


# ---------------- バックテスト ----------------

def ann_sharpe(x):
    x = x.dropna()
    if len(x) < 6 or x.std() == 0:
        return np.nan, np.nan, len(x)
    sh = x.mean() / x.std() * np.sqrt(12)
    t = x.mean() / (x.std() / np.sqrt(len(x)))
    return sh, t, len(x)


def split_stats(excess):
    """全/IS/OOS の (Sharpe,t,N) と 月平均bps。"""
    idx = excess.index.astype(str)
    is_mask = idx <= IS_END
    out = {}
    for name, m in [("全", slice(None)), ("IS", is_mask), ("OOS", ~is_mask)]:
        seg = excess[m] if name != "全" else excess
        sh, t, n = ann_sharpe(seg)
        out[name] = (sh, t, n, seg.dropna().mean() * 1e4)
    return out


def run_strategy(sec_ret, L, K, mode, topix_ret, cost_bps=10.0, excess=True):
    """mode: 'MOM'(上位Kロング) / 'REV'(下位Kロング) / 'LS'(上位−下位).
    戻り: TOPIX超過(LSは常に絶対)月次系列。excess=False で long-only の絶対リターン。"""
    cum = (1 + sec_ret).rolling(L).apply(np.prod, raw=True) - 1  # trailing L ヶ月
    months = sec_ret.index
    rets, prev_hold = [], set()
    idx_out = []
    for i in range(L, len(months) - 1):
        t = months[i]
        sig = cum.loc[t].dropna()
        if len(sig) < 2 * K:
            continue
        nxt = months[i + 1]
        fwd = sec_ret.loc[nxt]
        ranked = sig.sort_values(ascending=False)
        top = list(ranked.index[:K]); bot = list(ranked.index[-K:])
        if mode == "MOM":
            hold = top; r = fwd[top].mean()
        elif mode == "REV":
            hold = bot; r = fwd[bot].mean()
        else:  # LS
            hold = top + bot
            r = fwd[top].mean() - fwd[bot].mean()
        # コスト: 入替分のみ。long-only=入替率×2×片側、LSは両サイド
        if mode in ("MOM", "REV"):
            turn = len(set(hold) - prev_hold) / max(K, 1)
            r -= turn * 2 * cost_bps / 1e4
        else:
            turn = len(set(hold) - prev_hold) / max(2 * K, 1)
            r -= turn * 2 * cost_bps / 1e4
        prev_hold = set(hold)
        if mode in ("MOM", "REV") and excess:
            r = r - topix_ret.get(nxt, np.nan)  # TOPIX超過
        rets.append(r); idx_out.append(nxt)
    return pd.Series(rets, index=pd.PeriodIndex(idx_out, freq="M"))


def rank_ic(sec_ret, L):
    """trailing-L順位 vs 翌月リターン順位 のクロスセクション相関(月次)平均。
    正=モメンタム、負=資金循環(平均回帰)。"""
    cum = (1 + sec_ret).rolling(L).apply(np.prod, raw=True) - 1
    months = sec_ret.index
    ics = []
    for i in range(L, len(months) - 1):
        s = cum.loc[months[i]].dropna()
        f = sec_ret.loc[months[i + 1]].dropna()
        common = s.index.intersection(f.index)
        if len(common) >= 8:
            ics.append(s[common].rank().corr(f[common].rank()))
    ics = pd.Series(ics)
    return ics.mean(), ics.mean() / (ics.std() / np.sqrt(len(ics))), len(ics)


def main():
    md, topix_close = load()
    sec_ret = build_sector_returns(md)
    topix_ret = topix_close.pct_change().reindex(sec_ret.index)
    print(f"\n業種リターン系列: {sec_ret.shape[0]}ヶ月 × {sec_ret.shape[1]}業種 "
          f"({sec_ret.index.min()}〜{sec_ret.index.max()})")
    print(f"業種: {list(sec_ret.columns)}")

    # TOPIX買い持ちベンチ
    bsh, bt, bn = ann_sharpe(topix_ret)
    print(f"\n[ベンチ] TOPIX買い持ち: Sharpe {bsh:+.2f} (t{bt:+.2f}, N={bn}, "
          f"月平均{topix_ret.mean()*1e4:+.0f}bps)")

    print("\n=== 規範IC: trailing-L順位 vs 翌月業種リターン (正=モメンタム/負=資金循環) ===")
    for L in (1, 3, 6, 12):
        ic, t, n = rank_ic(sec_ret, L)
        tag = "→モメンタム" if ic > 0 else "→資金循環(逆張り)"
        print(f"  L={L:2}ヶ月: IC{ic:+.3f} (t{t:+.2f}, N={n}) {tag}")

    print("\n=== 戦略別 (コスト10bps/片側, TOPIX超過[LSは絶対]) ===")
    print(f"{'mode':4} {'L':>2} {'K':>2} | {'全Sh':>6} {'IS':>6} {'OOS':>6} {'t全':>5} {'月bps':>6} N")
    rows = []
    for mode in ("MOM", "REV", "LS"):
        for L in (1, 3, 6, 12):
            for K in (3, 5):
                ex = run_strategy(sec_ret, L, K, mode, topix_ret, cost_bps=10.0)
                st = split_stats(ex)
                a, i_, o = st["全"], st["IS"], st["OOS"]
                print(f"{mode:4} {L:2} {K:2} | {a[0]:+6.2f} {i_[0]:+6.2f} {o[0]:+6.2f} "
                      f"{a[1]:+5.2f} {a[3]:+6.0f} {a[2]}")
                rows.append(dict(mode=mode, L=L, K=K, sh_all=a[0], sh_is=i_[0],
                                 sh_oos=o[0], t_all=a[1], bps=a[3], n=a[2]))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(os.path.dirname(__file__), "results.csv"), index=False)

    # 最良候補のコスト感度
    best = df.sort_values("sh_all", ascending=False).iloc[0]
    print(f"\n[最良] {best['mode']} L={int(best['L'])} K={int(best['K'])} "
          f"全Sharpe{best['sh_all']:+.2f}")
    print("  コスト感度:")
    for c in (0, 5, 10, 20):
        ex = run_strategy(sec_ret, int(best["L"]), int(best["K"]),
                          best["mode"], topix_ret, cost_bps=c)
        sh, t, n = ann_sharpe(ex)
        print(f"    {c:2}bps: Sharpe{sh:+.2f} (t{t:+.2f})")

    # ── 頑健性チェック (最良 MOM L=3 K=3) ──
    L, K = 3, 3
    print("\n=== 頑健性: MOM L=3 K=3 ===")
    # (1) 絶対Sharpe (TOPIX超過でない素のlong-only)
    ex = run_strategy(sec_ret, L, K, "MOM", topix_ret, 10.0, excess=True)
    ab = run_strategy(sec_ret, L, K, "MOM", topix_ret, 10.0, excess=False)
    sh_e, t_e, _ = ann_sharpe(ex); sh_a, t_a, _ = ann_sharpe(ab)
    print(f"  TOPIX超過 Sharpe{sh_e:+.2f}(t{t_e:+.2f}) / 絶対(素) Sharpe{sh_a:+.2f}(t{t_a:+.2f}) "
          f"月平均{ab.mean()*1e4:+.0f}bps")

    # (2) 年別 TOPIX超過 Sharpe (レジーム依存チェック)
    print("  年別 TOPIX超過 Sharpe:")
    yr = ex.groupby(ex.index.year)
    line = "   "
    for y, seg in yr:
        s, _, n = ann_sharpe(seg)
        line += f" {y}:{s:+.2f}({n})"
    print(line)

    # (3) 高流動性ユニバース (ADV≥10億) で小型株モメンタムの化身か検証
    print("\n=== 頑健性: 高流動性 ADV≥10億 で再構築 ===")
    sec_ret_hi = build_sector_returns(md, adv_min=1e9)
    topix_hi = topix_close.pct_change().reindex(sec_ret_hi.index)
    ic, t, n = rank_ic(sec_ret_hi, 3)
    print(f"  規範IC L=3: {ic:+.3f} (t{t:+.2f}, N={n})")
    for KK in (3, 5):
        ex_hi = run_strategy(sec_ret_hi, 3, KK, "MOM", topix_hi, 10.0)
        st = split_stats(ex_hi)
        print(f"  MOM L3 K{KK} 超過: 全{st['全'][0]:+.2f} IS{st['IS'][0]:+.2f} "
              f"OOS{st['OOS'][0]:+.2f} (t{st['全'][1]:+.2f}, 月{st['全'][3]:+.0f}bps)")

    # 非鉄(鉄鋼・非鉄)・半導体(電機・精密) の相対強さ推移を保存用に書き出し
    rs = sec_ret.sub(topix_ret, axis=0)  # 各業種のTOPIX超過月次
    rs_cum = (1 + rs.fillna(0)).cumprod()
    rs_cum.to_csv(os.path.join(os.path.dirname(__file__), "sector_relative_cum.csv"))
    sec_ret.to_csv(os.path.join(os.path.dirname(__file__), "sector_returns.csv"))

    # ── 現在の trailing-3m ランキング (アクショナブル) ──
    cum3 = (1 + sec_ret).rolling(3).apply(np.prod, raw=True) - 1
    latest = cum3.iloc[-1].dropna().sort_values(ascending=False)
    print(f"\n=== 現在の trailing-3m 業種ランキング ({sec_ret.index[-1]}時点) ===")
    for i, (s, v) in enumerate(latest.items(), 1):
        mark = " ★保有(上位3)" if i <= 3 else ""
        watch = "  ← 注目" if s in ("鉄鋼・非鉄", "電機・精密") else ""
        print(f"  {i:2}. {s:20} {v*100:+6.1f}%{mark}{watch}")
    print("\nsaved results.csv, sector_relative_cum.csv, sector_returns.csv")


if __name__ == "__main__":
    main()
