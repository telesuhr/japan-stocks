"""
#13 6Mモメンタムが元検証(Sh+0.71/+0.71)から再現時(Sh+0.04)へ落ちた原因の分解。
2つの変更を1つずつ入れ、どちらが効いているかを特定する。
  (A) ユニバース: 全期間固定ADV+存続銘柄のみ(元) → point-in-time(修正)
  (B) 執行:      月末終値→翌月末終値(元)        → 翌月初寄成→月末引成(修正)
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
import pandas as pd
from jstock import db

COST_LS, ADV_MIN, ADV_WIN, Q_LS = 0.0008, 5e8, 60, 0.20
IS_END = pd.Timestamp("2021-06-30")

raw = db.read_sql("""
    SELECT code, date, adj_open, adj_close, turnover_value
    FROM stocks_daily WHERE date >= '2015-01-01' AND adj_close > 0 AND adj_open > 0
""")
raw["date"] = pd.to_datetime(raw["date"])
AO = raw.pivot(index="date", columns="code", values="adj_open").sort_index()
AC = raw.pivot(index="date", columns="code", values="adj_close").sort_index()
TV = raw.pivot(index="date", columns="code", values="turnover_value").sort_index()
cal = AC.index

# 元検証のユニバース（全期間平均ADV≥5億・上場維持のみ = 未来情報）
surv = db.read_sql("""
    SELECT sd.code FROM stocks_daily sd JOIN symbol_master sm ON sm.code5 = sd.code
    WHERE sm.delisted_at IS NULL AND sd.date BETWEEN '2015-01-01' AND '2026-06-30'
    GROUP BY sd.code HAVING AVG(sd.turnover_value) >= 5e8 AND COUNT(*) >= 500
""")["code"].tolist()
surv = [c for c in surv if c in AC.columns]
UNIV_PIT = TV.rolling(ADV_WIN, min_periods=40).mean().shift(1) >= ADV_MIN
UNIV_ORIG = pd.DataFrame(False, index=AC.index, columns=AC.columns)
UNIV_ORIG[surv] = True
print(f"元ユニバース {len(surv)}銘柄 / PIT平均 {UNIV_PIT.sum(axis=1).mean():.0f}銘柄")

me = pd.DatetimeIndex(pd.Series(cal, index=cal).groupby([cal.year, cal.month]).last().values)
ms = pd.DatetimeIndex(pd.Series(cal, index=cal).groupby([cal.year, cal.month]).first().values)


def run(univ, exec_open):
    out = []
    for i in range(len(me) - 1):
        sd = me[i]
        j = np.searchsorted(cal, sd); k = j - 120
        if k < 0:
            continue
        entry, exitd = ms[i + 1], me[i + 1]
        if entry <= sd:
            continue
        m6 = AC.loc[sd] / AC.iloc[k] - 1.0
        u = univ.loc[sd] & m6.notna() & AC.loc[exitd].notna()
        u &= AO.loc[entry].notna() if exec_open else AC.loc[sd].notna()
        m6 = m6[u]
        if len(m6) < 50:
            continue
        p0 = AO.loc[entry, m6.index] if exec_open else AC.loc[sd, m6.index]
        fwd = AC.loc[exitd, m6.index] / p0 - 1.0
        n = max(3, int(len(m6) * Q_LS))
        o = m6.sort_values().index
        out.append(dict(date=exitd, ret=fwd[o[-n:]].mean() - fwd[o[:n]].mean() - COST_LS))
    s = pd.DataFrame(out).set_index("date")["ret"]
    return s[s.index >= "2016-01-01"]


def sh(s, ann=12):
    return float(s.mean() / s.std() * np.sqrt(ann)) if s.std() > 0 else np.nan


print("\n" + "=" * 78)
print("#13 6Mモメンタム 分解（月次Sharpe・コスト後）")
print("=" * 78)
print(f"{'設定':<44}{'N':>5}{'IS':>8}{'OOS':>8}{'ALL':>8}{'年率%':>8}")
rows=[]
for lbl, univ, eo in [
    ("(元) 存続ADVユニバース × 月末終値→終値", UNIV_ORIG, False),
    ("(A) PITユニバース    × 月末終値→終値", UNIV_PIT, False),
    ("(B) 存続ADVユニバース × 寄成→引成", UNIV_ORIG, True),
    ("(A+B) PIT × 寄成→引成 【本分析の正】", UNIV_PIT, True),
]:
    s = run(univ, eo)
    rows.append(dict(spec=lbl, N=len(s), IS=sh(s[s.index<=IS_END]), OOS=sh(s[s.index>IS_END]),
                     ALL=sh(s), ann=s.mean()*12*100))
    print(f"{lbl:<44}{len(s):>5}{sh(s[s.index<=IS_END]):>8.2f}{sh(s[s.index>IS_END]):>8.2f}"
          f"{sh(s):>8.2f}{s.mean()*12*100:>8.1f}")
pd.DataFrame(rows).to_csv("mom_decomp.csv", index=False)
