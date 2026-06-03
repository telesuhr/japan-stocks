"""
AI半導体サプライチェーン リードラグ分析
4層（上流材料→製造装置→デバイス部品→AI/DC下流）の日足リードラグ・循環検証

前回(20260421_semiconductor_leadlag): 1分足では共通因子で同時に動く→エッジなし
今回: 日足スケールで上流→下流の「資金循環」の規則性を探る
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import os, warnings
import numpy as np
import pandas as pd
import psycopg2

# scipy.stats がない場合の軽量代替
try:
    from scipy import stats
except ModuleNotFoundError:
    class stats:
        @staticmethod
        def pearsonr(x, y):
            r = np.corrcoef(x, y)[0, 1]
            n = len(x)
            t = r * np.sqrt(n - 2) / np.sqrt(1 - r**2 + 1e-15)
            from scipy.special import betainc  # fallback p-value略
            return r, np.nan
        @staticmethod
        def spearmanr(x, y):
            rx = pd.Series(x).rank(); ry = pd.Series(y).rank()
            r = np.corrcoef(rx, ry)[0, 1]
            return r, np.nan

warnings.filterwarnings("ignore")

PG = os.getenv("DATABASE_URL", "postgresql://postgres@localhost/market_data")

# ── ユニバース定義 ──────────────────────────────────────────────────────
LAYERS = {
    "①上流(材料)":    ["40630", "34360", "41860", "40620"],  # 信越化学/SUMCO/東京応化/イビデン
    "②製造装置":      ["80350", "68570", "69200", "61460"],  # 東エレ/アドテスト/レーザーテック/DISCO
    "③デバイス部品":  ["67230", "69630", "69810", "67620", "69710"],  # ルネサス/ローム/村田/TDK/京セラ
    "④AI/DC下流":     ["99840", "94340", "94320", "94330", "37780"],  # SBG/SB/NTT/KDDI/さくら
}
LAYER_SHORT = {"①上流(材料)":"上流", "②製造装置":"装置", "③デバイス部品":"部品", "④AI/DC下流":"AI下流"}
ALL_CODES = [c for codes in LAYERS.values() for c in codes]

# SBG単独でも見る
SBG_CODE = "99840"

START_DATE = "2018-01-01"
IS_END     = "2021-12-31"   # IS: 2018-2021, OOS: 2022-2026


def load(conn):
    codes_str = ",".join(f"'{c}'" for c in ALL_CODES)
    df = pd.read_sql(f"""
        SELECT d.date, d.code, d.adj_close
        FROM stocks_daily d
        WHERE d.code IN ({codes_str})
          AND d.date >= '{START_DATE}'
        ORDER BY d.date, d.code
    """, conn)
    # TOPIX
    topix = pd.read_sql("""
        SELECT date, close as topix
        FROM index_daily WHERE code='0000' AND date >= '2018-01-01'
        ORDER BY date
    """, conn)

    # 名前マップ
    names = pd.read_sql(f"""
        SELECT code5, name_ja FROM symbol_master WHERE code5 IN ({codes_str})
    """, conn)
    name_map = dict(zip(names.code5, names.name_ja))

    return df, topix, name_map


def build_price_matrix(df):
    """code × date の終値行列"""
    px = df.pivot(index="date", columns="code", values="adj_close")
    px.index = pd.to_datetime(px.index)
    px = px.sort_index()
    return px


def build_layer_returns(px):
    """各層の等ウェイト日次リターン"""
    ret = px.pct_change().replace([np.inf, -np.inf], np.nan)
    # |ret|>50%はデータ異常として除外
    ret = ret.where(ret.abs() <= 0.5)

    layer_ret = {}
    for layer, codes in LAYERS.items():
        valid = [c for c in codes if c in ret.columns]
        layer_ret[layer] = ret[valid].mean(axis=1)
    return pd.DataFrame(layer_ret), ret


def cross_correlation(x, y, max_lag=10):
    """lag=-max_lag..+max_lag の相関を計算。正のlagはxがyに先行"""
    results = {}
    for lag in range(-max_lag, max_lag + 1):
        if lag > 0:
            xi, yi = x.iloc[:-lag], y.iloc[lag:]
        elif lag < 0:
            xi, yi = x.iloc[-lag:], y.iloc[:lag]
        else:
            xi, yi = x, y
        mask = xi.notna() & yi.notna()
        if mask.sum() < 30:
            results[lag] = np.nan
        else:
            r, p = stats.pearsonr(xi[mask], yi[mask])
            results[lag] = r
    return results


def analyze_leadlag(layer_ret, period_name, date_mask):
    """全層ペアのクロス相関を表示"""
    lr = layer_ret[date_mask].dropna(how="all")
    layers = list(LAYERS.keys())
    print(f"\n{'='*60}")
    print(f"リードラグ分析 [{period_name}] (lag=+N: 先行層がN日先行)")
    print(f"期間: {lr.index[0].date()} 〜 {lr.index[-1].date()}, N={len(lr)}")
    print(f"{'='*60}")

    # 全ペアのlag -3〜+3を表示
    results = {}
    for i, lead in enumerate(layers):
        for j, follow in enumerate(layers):
            if i == j:
                continue
            key = f"{LAYER_SHORT[lead]}→{LAYER_SHORT[follow]}"
            cc = cross_correlation(lr[lead], lr[follow], max_lag=10)
            results[key] = cc

    # リードラグの要約: 各ペアのlag+1,+2,+3の平均相関
    print(f"\n【先行相関サマリー (lag=+1〜+3の平均)】")
    print(f"{'ペア':<22} {'lag+1':>7} {'lag+2':>7} {'lag+3':>7} {'avg':>7}")
    print("-"*52)
    summary = []
    for key, cc in results.items():
        l1 = cc.get(1, np.nan)
        l2 = cc.get(2, np.nan)
        l3 = cc.get(3, np.nan)
        avg = np.nanmean([l1, l2, l3])
        summary.append((key, l1, l2, l3, avg))
    summary.sort(key=lambda x: -x[4])
    for row in summary:
        key, l1, l2, l3, avg = row
        mark = " ◀" if abs(avg) > 0.04 else ""
        print(f"  {key:<20} {l1:>7.4f} {l2:>7.4f} {l3:>7.4f} {avg:>7.4f}{mark}")

    return results


def rank_ic_analysis(layer_ret, date_mask):
    """層間のランク相関IC: 今日の層リターン vs 翌日の別層リターン"""
    print(f"\n【翌日予測力 (Rank IC: 今日の層リターン→翌日の別層リターン)】")
    lr = layer_ret[date_mask].dropna(how="all")
    layers = list(LAYERS.keys())
    rows = []
    for lead in layers:
        for follow in layers:
            if lead == follow:
                continue
            x = lr[lead]
            y = lr[follow].shift(-1)
            mask = x.notna() & y.notna()
            if mask.sum() < 30:
                continue
            ic, pval = stats.spearmanr(x[mask], y[mask])
            t = ic * np.sqrt(mask.sum() - 2) / np.sqrt(1 - ic**2 + 1e-12)
            rows.append({
                "先行層": LAYER_SHORT[lead],
                "後続層": LAYER_SHORT[follow],
                "IC": ic,
                "t値": t,
                "N": mask.sum(),
            })
    rdf = pd.DataFrame(rows).sort_values("IC", ascending=False)
    print(rdf.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    return rdf


def sbg_special(px, layer_ret, date_mask):
    """SBG(9984)単独: 各層との先行・追随を詳しく確認"""
    print(f"\n{'='*60}")
    print("SBG(9984) 単独分析")
    print(f"{'='*60}")
    lr = layer_ret[date_mask]
    if SBG_CODE not in px.columns:
        print("SBG データなし"); return
    sbg_ret = px[SBG_CODE].pct_change().replace([np.inf, -np.inf], np.nan)
    sbg_ret = sbg_ret.where(sbg_ret.abs() <= 0.5)[date_mask]

    print(f"\n【SBG vs 各層 クロス相関 (lag: SBGがN日先行)】")
    print(f"{'層':<22} {'lag-3':>7} {'lag-2':>7} {'lag-1':>7} {'0':>7} {'lag+1':>7} {'lag+2':>7} {'lag+3':>7}")
    print("-"*65)
    for layer in LAYERS:
        y = lr[layer]
        cc = cross_correlation(sbg_ret, y, max_lag=5)
        vals = [cc.get(k, np.nan) for k in [-3,-2,-1,0,1,2,3]]
        best_lag = max(range(-3,4), key=lambda k: abs(cc.get(k, 0)))
        line = f"  {LAYER_SHORT[layer]:<20}"
        for v in vals:
            mark = "★" if abs(v)==abs(cc.get(best_lag,0)) and abs(v)>0.04 else " "
            line += f" {v:>6.4f}{mark}"
        print(line)


def regime_split(layer_ret, date_mask_is, date_mask_oos):
    """IS/OOS比較 + AI相場(2023-) vs それ以前"""
    print(f"\n{'='*60}")
    print("期間別検証 (安定性確認)")
    print(f"{'='*60}")
    ai_mask = layer_ret.index >= "2023-01-01"
    pre_ai_mask = (layer_ret.index >= "2018-01-01") & (layer_ret.index < "2023-01-01")

    for period_name, mask in [
        ("IS (2018-2021)", date_mask_is),
        ("OOS (2022-2026)", date_mask_oos),
        ("AI相場以前 (2018-2022)", pre_ai_mask),
        ("AI相場 (2023-2026)", ai_mask),
    ]:
        lr = layer_ret[mask].dropna(how="all")
        if len(lr) < 60:
            continue
        # 装置→AI下流 の lag+1 相関だけ追う（最も興味深いペア）
        lead, follow = "②製造装置", "④AI/DC下流"
        cc = cross_correlation(lr[lead], lr[follow], max_lag=5)
        l1 = cc.get(1, np.nan)
        l2 = cc.get(2, np.nan)
        r0 = cc.get(0, np.nan)
        print(f"  {period_name:<25} N={len(lr):4d} | 装置→AI下流 lag0={r0:+.4f} lag+1={l1:+.4f} lag+2={l2:+.4f}")


def rolling_lead(layer_ret, lead, follow, window=120):
    """ローリング相関 (lag+1) — 規則性がどの時期に強いか"""
    lr = layer_ret[[lead, follow]].dropna()
    x = lr[lead]
    y = lr[follow].shift(-1)
    both = pd.DataFrame({"x": x, "y": y}).dropna()
    rolling_r = both["x"].rolling(window).corr(both["y"])
    return rolling_r


def main():
    print("Loading data...")
    conn = psycopg2.connect(PG)
    df, topix, name_map = load(conn)
    conn.close()
    print(f"  {len(df)} rows, {df.code.nunique()} codes, {df.date.min()} to {df.date.max()}")

    px = build_price_matrix(df)
    layer_ret, stock_ret = build_layer_returns(px)

    is_mask  = layer_ret.index <= IS_END
    oos_mask = layer_ret.index >  IS_END

    # ── 全期間リードラグ ──
    all_mask = pd.Series(True, index=layer_ret.index)
    cc_all = analyze_leadlag(layer_ret, "全期間 2018-2026", all_mask)

    # ── IS / OOS ──
    cc_is  = analyze_leadlag(layer_ret, "IS 2018-2021", is_mask)
    cc_oos = analyze_leadlag(layer_ret, "OOS 2022-2026", oos_mask)

    # ── 翌日予測 IC ──
    print(f"\n{'='*60}\n翌日予測力 Rank IC — 全期間")
    rank_ic_analysis(layer_ret, all_mask)

    # ── SBG特別分析 ──
    sbg_special(px, layer_ret, all_mask)

    # ── 期間別安定性 ──
    regime_split(layer_ret, is_mask, oos_mask)

    # ── ローリング相関の計算・保存 ──
    print("\n\nローリング相関 (装置→AI下流, 120日) を計算中...")
    rolling = rolling_lead(layer_ret, "②製造装置", "④AI/DC下流", window=120)
    rolling.name = "r_装置→AI下流_lag1"

    # 結果保存
    layer_ret.to_csv("layer_returns.csv")
    rolling.to_csv("rolling_corr.csv")

    # 各層間の全ペアのlag+1相関を一覧保存
    rows = []
    for period_name, mask in [("all", all_mask), ("IS", is_mask), ("OOS", oos_mask)]:
        lr = layer_ret[mask].dropna(how="all")
        for lead in LAYERS:
            for follow in LAYERS:
                if lead == follow: continue
                cc = cross_correlation(lr[lead], lr[follow], max_lag=10)
                for lag in range(-5, 6):
                    rows.append({"period": period_name, "lead": LAYER_SHORT[lead],
                                 "follow": LAYER_SHORT[follow], "lag": lag, "r": cc.get(lag)})
    pd.DataFrame(rows).to_csv("crosscorr_all.csv", index=False)
    print("\n保存完了: layer_returns.csv, rolling_corr.csv, crosscorr_all.csv")


if __name__ == "__main__":
    main()
