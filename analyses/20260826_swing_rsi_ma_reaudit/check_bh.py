"""決定的な検算: MA25/75 は「タイミングのα」か「銘柄を持っていただけ」か。

run.py で MA バスケットは OOS 年率36.2%・Sharpe1.43・TOPIX比α20%(t=2.68) と出た。
しかし**稼働日割合が97.6%＝ほぼ常時ロング**であり、構成8銘柄は2021以降に暴騰した
半導体・非鉄・銀行に偏る。この条件では TOPIX を benchmark にするのは不当で、
正しい対照は **同じ8銘柄を単純に持ち続けた場合（buy&hold）**。

MAフィルタの価値 = 戦略リターン − 同銘柄BHリターン。これが正でなければ、
「シグナルに従って売買した意味はゼロ（むしろコストと機会損失で負け）」と結論する。
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
from pathlib import Path
import numpy as np
import pandas as pd
from jstock import db

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run as R   # noqa: E402  データ・シグナル・関数を再利用

ANN, OOS = R.ANN, slice(R.OOS_START, None)
ADOPT, AC, cal = R.ADOPT, R.AC, R.cal


def bh(codes, period):
    """等額バイ&ホールド（日次リバランスなしの近似=等額平均日次リターン）。"""
    r = (AC[codes] / AC[codes].shift(1) - 1)[period]
    return r.mean(axis=1)


def rep(s, lb):
    s = s.dropna()
    eq = (1 + s).cumprod()
    yrs = len(s) / ANN
    return {"構成": lb, "年率%": (eq.iloc[-1] ** (1 / yrs) - 1) * 100,
            "Sharpe": R.sharpe(s), "MDD%": (eq / eq.cummax() - 1).min() * 100, "N日": len(s)}


print("=" * 100)
print("決定的検算: 戦略 vs 同じ銘柄のバイ&ホールド（OOS 2021-01〜2026-08）")
print("=" * 100)
rows, diffs = [], {}
for lb, df in [("RSI<30反発", ADOPT[ADOPT["best"] == "RSI"]),
               ("MA25/75順張り", ADOPT[ADOPT["best"] == "MA"]),
               ("混合15銘柄", ADOPT)]:
    codes = list(df["code"])
    st = R.basket(df, OOS)
    hold = bh(codes, OOS)
    x = pd.concat([st.rename("s"), hold.rename("h")], axis=1).dropna()
    d = x["s"] - x["h"]
    rows.append(rep(x["s"], f"{lb} 戦略"))
    rows.append(rep(x["h"], f"{lb} 同銘柄BH"))
    t = d.mean() / d.std() * np.sqrt(len(d))
    diffs[lb] = {"構成": lb, "戦略−BH 年率pt": (d.mean() * ANN) * 100,
                 "t(差)": t, "差のSharpe": R.sharpe(d),
                 "戦略が勝った日%": (d > 0).mean() * 100}
print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:8.2f}"))
print("\n" + "=" * 100)
print("MAフィルタの正味の価値（戦略 − 同銘柄BH）")
print("=" * 100)
print(pd.DataFrame(diffs.values()).to_string(index=False, float_format=lambda v: f"{v:8.2f}"))

# 退出タイミングは価値を生んだか: ノーポジ日にBHはどう動いたか
print("\n" + "=" * 100)
print("『降りていた日』は本当に悪い日だったか")
print("=" * 100)
out = []
for lb, df in [("RSI<30反発", ADOPT[ADOPT["best"] == "RSI"]),
               ("MA25/75順張り", ADOPT[ADOPT["best"] == "MA"])]:
    codes = list(df["code"])
    # 銘柄ごとに「保有していない日」の当該銘柄リターンを集める
    inn, offn = [], []
    for c in codes:
        sig = R.SIG[c][df.set_index("code").loc[c, "best"]]
        held = sig.shift(1).fillna(0).astype(int)[OOS]
        r = (AC[c] / AC[c].shift(1) - 1)[OOS]
        inn.append(r[held == 1].mean())
        offn.append(r[held == 0].mean())
    out.append({"構成": lb, "保有日の平均日次%": np.nanmean(inn) * 100,
                "非保有日の平均日次%": np.nanmean(offn) * 100,
                "差(bp/日)": (np.nanmean(inn) - np.nanmean(offn)) * 1e4})
print(pd.DataFrame(out).to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
print("  ※非保有日の平均がプラスなら、降りている間に上げを取り逃している")

# 弱気局面での防御力（本来の売り: MDD抑制）
print("\n" + "=" * 100)
print("下落局面での防御力（TOPIX下落日/暴落年）")
print("=" * 100)
mkt = R.mkt[OOS]
dn = []
for lb, df in [("MA25/75順張り", ADOPT[ADOPT["best"] == "MA"]),
               ("混合15銘柄", ADOPT)]:
    codes = list(df["code"])
    st, hold = R.basket(df, OOS), bh(codes, OOS)
    x = pd.concat([st.rename("s"), hold.rename("h"), mkt.rename("m")], axis=1).dropna()
    bad = x[x["m"] < -0.01]
    dn.append({"構成": lb, "TOPIX-1%超下落日 戦略%": bad["s"].mean() * 100,
               "同 BH%": bad["h"].mean() * 100, "N日": len(bad)})
print(pd.DataFrame(dn).to_string(index=False, float_format=lambda v: f"{v:8.3f}"))

yr = []
for lb, df in [("MA25/75順張り", ADOPT[ADOPT["best"] == "MA"]),
               ("RSI<30反発", ADOPT[ADOPT["best"] == "RSI"])]:
    codes = list(df["code"])
    st, hold = R.basket(df, OOS), bh(codes, OOS)
    x = pd.concat([st.rename("戦略"), hold.rename("BH")], axis=1).dropna()
    g = x.groupby(x.index.year).apply(lambda d: ((1 + d).prod() - 1) * 100)
    g["差"] = g["戦略"] - g["BH"]
    g.insert(0, "構成", lb)
    yr.append(g)
Y = pd.concat(yr)
print("\n年次（%）:")
print(Y.to_string(float_format=lambda v: f"{v:8.2f}"))
Y.to_csv(HERE / "yearly_vs_bh.csv", encoding="utf-8-sig")
pd.DataFrame(rows).to_csv(HERE / "vs_buyhold.csv", index=False, encoding="utf-8-sig")
print("\nsaved vs_buyhold.csv / yearly_vs_bh.csv")
