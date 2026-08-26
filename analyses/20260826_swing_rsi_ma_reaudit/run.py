"""RSI<30反発 / MA25-75順張り の再監査 — 「OOS Sharpe +6.15」は本物か。

`20260522_strategy_true_oos_validation` は2本を「昇格基準クリア」と判定したが1年以上
strategies/ に実装されないまま放置されている。低タッチ・スイングの第2エンジン候補として
拾い直すにあたり、当時の測定に2つの疑いがあるので事前に潰す。

疑い1（Sharpe の年率化）: 元コードの metrics() は pnl を **決済日だけ** で groupby して
  √245 を掛けている。年30回しか決済しないなら30個の観測を245日分として年率化することになる。
  `20260531_strategy_sharpe_audit` で「per-trade×√252 は約5倍の過大評価」と既に判明した型。
  → 全営業日を並べた日次ポートフォリオ系列で測り直す（無ポジ日は0%として含める）。

疑い2（分割）: 元コードは `open`/`close` の**生値**を使っており調整していない。分割を跨ぐ
  トレードは偽の巨大損益になる。ユニバース21銘柄にはトヨタ(2021 5:1)・古河電工(2026 1:10)・
  住友電工(2026 1:4)等が含まれる。
  → `adj_factor` の将来累積から調整値を自前復元して使う（DBの adj_close は2026-06以降の
     35銘柄で遡及調整が未修復なため、そのままでは信用できない。20260822 参照）。

疑い3（βか固有αか）: 主軸 pre_earnings_drift は「相場βに乗るエンジン」と判明済み(20260815)。
  Long-only の順張り/逆張りも同じ穴に落ちうる。TOPIXへの回帰で α を分離して確かめる。

銘柄・戦略の割当は当時のIS(2016-2020)判定を**そのまま凍結して使う**（再選別しない＝真のOOS）。
OOSは当時の 2026-05-22 から今日まで3ヶ月延長され、その分だけ新規の未使用データが入る。
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
from pathlib import Path
import numpy as np
import pandas as pd
from jstock import db

HERE = Path(__file__).resolve().parent
IS_START, IS_END = "2016-01-01", "2020-12-31"
OOS_START = "2021-01-01"
WARMUP = "2015-06-01"
COST_BPS = 8.0          # 片道 8bps（元検証と同値。感度も後段で見る）
ANN = 252

UNIVERSE = {
    "57060": ("三井金属", "非鉄"), "57110": ("三菱マテリアル", "非鉄"),
    "57130": ("住友金属鉱山", "非鉄"), "57140": ("DOWA HD", "非鉄"),
    "58010": ("古河電工", "非鉄"), "58020": ("住友電工", "非鉄"),
    "58030": ("フジクラ", "非鉄"),
    "83060": ("三菱UFJ", "銀行"), "83160": ("三井住友", "銀行"),
    "84110": ("みずほ", "銀行"),
    "68570": ("アドバンテスト", "半導体"), "69200": ("レーザーテック", "半導体"),
    "80350": ("東京エレクトロン", "半導体"),
    "80010": ("伊藤忠商事", "商社"), "80310": ("三井物産", "商社"),
    "80580": ("三菱商事", "商社"),
    "33820": ("セブン&アイ", "小売"), "99830": ("ファストリテ", "小売"),
    "72030": ("トヨタ自動車", "自動車"), "72670": ("ホンダ", "自動車"),
    "79740": ("任天堂", "その他"),
}
CODES = list(UNIVERSE)

# ---------------------------------------------------------------- データ
raw = db.read_sql("""
    SELECT code, date, open, close, adj_factor
    FROM stocks_daily
    WHERE code = ANY(%s) AND date >= %s AND close > 0 AND open > 0
    ORDER BY code, date
""", (CODES, WARMUP))
raw["date"] = pd.to_datetime(raw["date"])
for c in ("open", "close", "adj_factor"):
    raw[c] = raw[c].astype(float)
raw["adj_factor"] = raw["adj_factor"].fillna(1.0).replace(0, 1.0)


def back_adjust(g):
    """生値 × 将来の adj_factor の累積 = 分割調整済み系列。

    JQuants の adj_factor は分割実施日に 1/n が立つ。その日より前の価格に将来分の
    factor を全部掛けると、現在基準に揃った連続系列になる。DBの adj_close を信用せず
    ここで作り直すのが要点（2026-06以降の35銘柄は adj_close が未修復）。
    """
    g = g.sort_values("date").copy()
    fwd = g["adj_factor"].shift(-1)[::-1].cumprod()[::-1].fillna(1.0)
    g["ac"] = g["close"] * fwd
    g["ao"] = g["open"] * fwd
    return g


raw = raw.groupby("code", group_keys=False).apply(back_adjust)
AC = raw.pivot(index="date", columns="code", values="ac").sort_index()
AO = raw.pivot(index="date", columns="code", values="ao").sort_index()
cal = AC.index
print(f"データ: {cal[0].date()}〜{cal[-1].date()}  {len(cal)}営業日 × {len(CODES)}銘柄")

# 復元の検算: 分割していない銘柄では DB の adj_close と一致するはず
chk = db.read_sql("SELECT code,date,adj_close FROM stocks_daily WHERE code='72670' AND date>=%s", (WARMUP,))
chk["date"] = pd.to_datetime(chk["date"])
m = chk.set_index("date")["adj_close"].astype(float).reindex(cal).dropna()
dev = (AC.loc[m.index, "72670"] / m - 1).abs().max()
print(f"検算(ホンダ): 自前復元 vs DB adj_close 最大乖離 {dev:.2e}")


def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def pos_ma(c, fast=25, slow=75):
    return (c.rolling(fast).mean() > c.rolling(slow).mean()).astype(int)


def pos_rsi(c, lo=30, hi=50):
    r = rsi(c)
    out = np.zeros(len(c), dtype=int)
    st = 0
    for i, v in enumerate(r.values):
        if np.isfinite(v):
            if st == 0 and v < lo:
                st = 1
            elif st == 1 and v > hi:
                st = 0
        out[i] = st
    return pd.Series(out, index=c.index)


def daily_pnl(code, sig):
    """シグナル(日T終値時点) → 翌営業日(T+1)の寄成で建/落。日次リターン系列を返す。

    寄成で入るので、建てた日の取り分は 引け/寄り、外した日は 寄り/前日引け。
    保有中の平常日は 引け/前日引け。無ポジ日は 0（＝資金は寝ている）。
    低タッチ制約に完全に合致する執行（前夜にMOOを置くだけ）。
    """
    ac, ao = AC[code], AO[code]
    held = sig.shift(1).fillna(0).astype(int)          # T+1寄りから建つ
    prev = held.shift(1).fillna(0).astype(int)
    r = pd.Series(0.0, index=cal)
    c2c = ac / ac.shift(1) - 1
    o2c = ac / ao - 1
    c2o = ao / ac.shift(1) - 1
    ent = (held == 1) & (prev == 0)
    hold = (held == 1) & (prev == 1)
    ext = (held == 0) & (prev == 1)
    r[ent] = o2c[ent]
    r[hold] = c2c[hold]
    r[ext] = c2o[ext]
    cost = (ent.astype(float) + ext.astype(float)) * COST_BPS / 1e4
    return (r.fillna(0) - cost), int(ent.sum())


SIG = {}
for code in CODES:
    c = AC[code].dropna()
    SIG[code] = {"MA": pos_ma(c).reindex(cal).fillna(0).astype(int),
                 "RSI": pos_rsi(c).reindex(cal).fillna(0).astype(int)}


def sharpe(s):
    s = s.dropna()
    return s.mean() / s.std() * np.sqrt(ANN) if len(s) > 2 and s.std() > 0 else 0.0


def report(s, label):
    s = s.dropna()
    if len(s) < 20:
        return None
    eq = (1 + s).cumprod()
    yrs = len(s) / ANN
    return {"構成": label, "年率%": (eq.iloc[-1] ** (1 / yrs) - 1) * 100,
            "Sharpe": sharpe(s), "MDD%": (eq / eq.cummax() - 1).min() * 100,
            "勝率%": (s > 0).mean() * 100, "N日": len(s)}


# ------------------------------------------------- STEP1: IS選別（当時の再現）
sl = slice(IS_START, IS_END)
sel = []
for code in CODES:
    row = {"code": code, "name": UNIVERSE[code][0], "sector": UNIVERSE[code][1]}
    for st in ("MA", "RSI"):
        p, n = daily_pnl(code, SIG[code][st])
        row[f"{st}_Sh"] = sharpe(p[sl])
        row[f"{st}_N"] = int(SIG[code][st][sl].diff().eq(1).sum())
    row["best"] = "MA" if row["MA_Sh"] > row["RSI_Sh"] else "RSI"
    row["best_Sh"] = max(row["MA_Sh"], row["RSI_Sh"])
    sel.append(row)
S = pd.DataFrame(sel)
print("\n" + "=" * 92)
print("STEP1: IS(2016-2020)で銘柄×戦略を選別 ※日次PF系列の正しいSharpeで")
print("=" * 92)
print(S[["name", "sector", "MA_Sh", "RSI_Sh", "best", "best_Sh"]]
      .sort_values("best_Sh", ascending=False)
      .to_string(index=False, float_format=lambda v: f"{v:6.2f}"))

for THR in (2.0, 0.5):
    ad = S[S["best_Sh"] >= THR]
    print(f"  IS Sharpe ≥ {THR}: {len(ad)}銘柄採用 "
          f"(MA {int((ad['best']=='MA').sum())} / RSI {int((ad['best']=='RSI').sum())})")

# 当時の基準2.0は「膨らんだSharpe」前提の閾値。正しい尺度では通る銘柄が激減するため、
# 選別ルール自体は当時と同じ「各銘柄で良い方の戦略・上位N銘柄」に読み替えて凍結する。
ADOPT = S.sort_values("best_Sh", ascending=False).head(15).reset_index(drop=True)
print(f"\n  → 当時と同じ15銘柄枠で凍結（IS上位15）: "
      f"MA {int((ADOPT['best']=='MA').sum())} / RSI {int((ADOPT['best']=='RSI').sum())}")


def basket(df, period):
    """等額バスケットの日次リターン。資金は常に1/Nずつ張り付け、無ポジ日は0（現金）。"""
    if len(df) == 0:
        return pd.Series(dtype=float)
    ps = [daily_pnl(r["code"], SIG[r["code"]][r["best"]])[0] for _, r in df.iterrows()]
    return pd.concat(ps, axis=1).mean(axis=1)[period]


OOS = slice(OOS_START, None)
rows = []
for lb, df in [("RSI<30反発", ADOPT[ADOPT["best"] == "RSI"]),
               ("MA25/75順張り", ADOPT[ADOPT["best"] == "MA"]),
               ("混合15銘柄", ADOPT)]:
    for pl, per in [("IS", sl), ("OOS", OOS)]:
        r = report(basket(df, per), f"{lb} ({pl}, {len(df)}銘柄)")
        if r:
            rows.append(r)
print("\n" + "=" * 92)
print(f"STEP2: バスケット成績（日次PF系列・√252・往復{COST_BPS*2:.0f}bps）")
print("=" * 92)
print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:8.2f}"))

# ------------------------------------------------- 当時の測り方との比較
print("\n" + "=" * 92)
print("STEP3: 当時の『決済日だけ並べて√245』との比較 — 膨張率")
print("=" * 92)
cmp = []
for lb, df in [("RSI<30反発", ADOPT[ADOPT["best"] == "RSI"]),
               ("MA25/75順張り", ADOPT[ADOPT["best"] == "MA"])]:
    b = basket(df, OOS)
    proper = sharpe(b)
    exitonly = b[b != 0]                      # 決済日/稼働日だけ拾う旧方式の近似
    infl = sharpe(exitonly) * np.sqrt(245 / ANN)
    cmp.append({"構成": lb, "正しいSharpe": proper, "旧方式(近似)": infl,
                "膨張率": infl / proper if proper != 0 else np.nan,
                "稼働日割合%": (b != 0).mean() * 100})
print(pd.DataFrame(cmp).to_string(index=False, float_format=lambda v: f"{v:8.2f}"))

# ------------------------------------------------- βか固有αか
tpx = db.read_sql("SELECT date, close FROM index_daily WHERE code='0000' AND date>=%s", (WARMUP,))
tpx["date"] = pd.to_datetime(tpx["date"])
mkt = tpx.set_index("date")["close"].astype(float).reindex(cal).ffill().pct_change()
print("\n" + "=" * 92)
print("STEP4: TOPIXへの回帰でαを分離（主軸pre_earnings_driftはβエンジンと判明済み）")
print("=" * 92)
al = []
for lb, df in [("RSI<30反発", ADOPT[ADOPT["best"] == "RSI"]),
               ("MA25/75順張り", ADOPT[ADOPT["best"] == "MA"]),
               ("混合15銘柄", ADOPT)]:
    b = basket(df, OOS)
    x = pd.concat([b.rename("y"), mkt.rename("m")], axis=1).dropna()
    if len(x) < 100:
        continue
    beta, alpha = np.polyfit(x["m"], x["y"], 1)
    resid = x["y"] - (alpha + beta * x["m"])
    t_a = alpha / (resid.std() / np.sqrt(len(x)))
    al.append({"構成": lb, "β": beta, "α年率%": alpha * ANN * 100, "t(α)": t_a,
               "残差Sharpe": sharpe(resid), "R²": np.corrcoef(x["m"], x["y"])[0, 1] ** 2})
print(pd.DataFrame(al).to_string(index=False, float_format=lambda v: f"{v:8.2f}"))
print(f"  参考: TOPIX単純保有(OOS) 年率 "
      f"{((1+mkt[OOS].dropna()).prod()**(ANN/len(mkt[OOS].dropna()))-1)*100:.2f}% / "
      f"Sharpe {sharpe(mkt[OOS]):.2f}")

# ------------------------------------------------- コスト感度
print("\n" + "=" * 92)
print("STEP5: コスト感度（片道bps）")
print("=" * 92)
base = COST_BPS
sens = []
for cb in (2.0, 5.0, 8.0, 15.0):
    COST_BPS = cb
    row = {"片道bps": cb}
    for lb, df in [("RSI", ADOPT[ADOPT["best"] == "RSI"]), ("MA", ADOPT[ADOPT["best"] == "MA"])]:
        row[f"{lb}_Sh"] = sharpe(basket(df, OOS))
    sens.append(row)
COST_BPS = base
print(pd.DataFrame(sens).to_string(index=False, float_format=lambda v: f"{v:8.2f}"))

ADOPT.to_csv(HERE / "is_selection.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(rows).to_csv(HERE / "basket_results.csv", index=False, encoding="utf-8-sig")
for lb, df in [("rsi", ADOPT[ADOPT["best"] == "RSI"]), ("ma", ADOPT[ADOPT["best"] == "MA"])]:
    basket(df, OOS).to_csv(HERE / f"daily_{lb}_oos.csv", encoding="utf-8-sig")
mkt.to_csv(HERE / "topix_daily.csv", encoding="utf-8-sig")
print("\nsaved is_selection.csv / basket_results.csv / daily_*_oos.csv / topix_daily.csv")
