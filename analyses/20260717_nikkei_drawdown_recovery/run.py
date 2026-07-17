"""
日経225 ドローダウン→回復期間・回復後リターンの過去傾向

問い: これだけ下落した後、過去は回復までどれくらいかかったか / 下落後の先行きは?
判断材料: ポジションをキープすべきか（レバは50%下落に耐える水準）。

手法:
  - N225日足(index_daily, 2016-05〜)。全期間ランニングピーク→DD。
  - (A) 閾値クロス回復: DDが初めて-D%を割った日から、元のピークに戻る(DD=0)までの営業日数。
  - (B) エピソード別: ピーク→回復の水面下期間と最大DD深度、トラフ→回復日数。
  - (C) 条件付き先行リターン: 「ピーク比-X%の状態」/「直近10日で-Y%下落」後の
        5/20/60/120/250営業日先リターン（平均・中央値・プラス率）。
注意: JQuants由来で履歴2016〜の約10年。深いDD(-20%+)のエピソードは少数(N明記)。
"""
import sys; sys.stdout.reconfigure(line_buffering=True)
from pathlib import Path
import numpy as np, pandas as pd
from jstock import db

HERE = Path(__file__).resolve().parent

d = db.read_sql("SELECT date, close FROM index_daily WHERE code='N225' ORDER BY date", [])
d["date"] = pd.to_datetime(d["date"])
d = d.reset_index(drop=True)
d["peak"] = d["close"].cummax()
d["dd"] = d["close"] / d["peak"] - 1
N = len(d)
close = d["close"].values
dd = d["dd"].values

print(f"N225 {d['date'].iloc[0].date()}〜{d['date'].iloc[-1].date()}  {N}営業日")
cur = d.iloc[-1]
print(f"現状(7/16): 終値{cur['close']:.0f} / ピーク比 {cur['dd']*100:.2f}% "
      f"/ 直近10日 {(close[-1]/close[-11]-1)*100:.2f}%\n")

# ---------- (A) 閾値クロス→ピーク復帰 ----------
print("="*72)
print("(A) ピーク比 -D% に到達した後、元のピークに戻るまでの営業日数")
print("="*72)
rowsA = []
for D in [0.03, 0.05, 0.07, 0.10, 0.15, 0.20]:
    recover_days = []
    ongoing = 0
    i = 0
    while i < N:
        if dd[i] <= -D:
            # このピーク水準
            peak_level = d["peak"].values[i]
            # 元のピークに戻る日を探す
            j = i
            while j < N and close[j] < peak_level:
                j += 1
            if j < N:
                recover_days.append(j - i)  # -D%到達からピーク復帰まで
            else:
                ongoing += 1
            # 次: このピークを超えた後の新たな局面へ（回復後 or 末尾）
            i = j + 1 if j < N else N
        else:
            i += 1
    if recover_days:
        arr = np.array(recover_days)
        rowsA.append({"深度": f"-{int(D*100)}%", "エピソード数": len(arr),
                      "中央値(日)": int(np.median(arr)), "平均(日)": int(arr.mean()),
                      "最短": int(arr.min()), "最長": int(arr.max()),
                      "未回復中": ongoing})
    else:
        rowsA.append({"深度": f"-{int(D*100)}%", "エピソード数": 0, "中央値(日)": None,
                      "平均(日)": None, "最短": None, "最長": None, "未回復中": ongoing})
A = pd.DataFrame(rowsA)
print(A.to_string(index=False))
print("※営業日。20営業日≒1ヶ月, 250≒1年。『未回復中』は末尾で水面下のまま=右打ち切り")

# ---------- (B) 主要ドローダウン・エピソード一覧 ----------
print("\n" + "="*72)
print("(B) 最大DD -7%以上の主要エピソード（ピーク→トラフ→回復）")
print("="*72)
episodes = []
i = 0
while i < N:
    if dd[i] < 0:
        peak_level = d["peak"].values[i]
        peak_date = d["date"].values[i-1] if i > 0 else d["date"].values[0]
        j = i
        trough_idx = i
        while j < N and close[j] < peak_level:
            if dd[j] < dd[trough_idx]:
                trough_idx = j
            j += 1
        maxdd = dd[trough_idx]
        if maxdd <= -0.07:
            rec = (d["date"].values[j] if j < N else None)
            episodes.append({
                "ピーク": pd.Timestamp(peak_date).date(),
                "トラフ": pd.Timestamp(d["date"].values[trough_idx]).date(),
                "最大DD": f"{maxdd*100:.1f}%",
                "下落日数": trough_idx - i + 1,
                "回復日": (pd.Timestamp(rec).date() if rec is not None else "未回復"),
                "トラフ→回復": (j - trough_idx if j < N else None),
                "全体(ピーク→回復)": (j - i + 1 if j < N else None),
            })
        i = j + 1 if j < N else N
    else:
        i += 1
B = pd.DataFrame(episodes)
print(B.to_string(index=False))

# ---------- (C) 条件付き 先行リターン ----------
print("\n" + "="*72)
print("(C) 条件付き先行リターン（プラス率 / 中央値%）  ※重複日含む・傾向把握用")
print("="*72)
hz = [5, 20, 60, 120, 250]
def fwd_stats(mask, label):
    idx = np.where(mask)[0]
    idx = idx[idx < N - 1]
    out = {"条件": label, "該当日数": len(idx)}
    for h in hz:
        v = [close[k+h]/close[k]-1 for k in idx if k+h < N]
        if v:
            v = np.array(v)
            out[f"+{h}d"] = f"{(v>0).mean()*100:.0f}%/{np.median(v)*100:+.1f}"
        else:
            out[f"+{h}d"] = "—"
    return out
rowsC = []
for X in [0.05, 0.07, 0.10, 0.15]:
    rowsC.append(fwd_stats(dd <= -X, f"ピーク比≤-{int(X*100)}%"))
# 直近10日下落
ret10 = np.concatenate([np.full(10, np.nan), close[10:]/close[:-10]-1])
for Y in [0.05, 0.07, 0.10]:
    rowsC.append(fwd_stats(ret10 <= -Y, f"直近10日≤-{int(Y*100)}%"))
# ベースライン
rowsC.append(fwd_stats(np.ones(N, bool), "無条件(ベース)"))
C = pd.DataFrame(rowsC)
print(C.to_string(index=False))
print("\n読み: 各セル = 先行リターンのプラス率% / 中央値%。ベースと比べ「押し目で買い有利か」を見る")

# 保存
A.to_csv(HERE/"recovery_by_depth.csv", index=False)
B.to_csv(HERE/"episodes.csv", index=False)
C.to_csv(HERE/"conditional_forward.csv", index=False)
d[["date","close","dd"]].to_csv(HERE/"n225_dd.csv", index=False)
print(f"\n保存: recovery_by_depth.csv / episodes.csv / conditional_forward.csv")
