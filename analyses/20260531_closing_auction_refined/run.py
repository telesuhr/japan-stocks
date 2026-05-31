"""
クロージング・オークション反転 — 精密版

初版 (20260531_closing_auction_reversion) の知見:
  - 反転は実在 (負IC、IS/OOS一貫、OOS強化)
  - 月末5営業日で増幅 (L/S net Sh 1.64)
  - 下落側 (bottom jump→翌朝反発Long) が非対称に強い
  - テイカー往復10bpsでgross Sharpe3.64→net0.51
  - overnight に重大なデータ異常 (max+98957%) → 要クレンジング

本精密版:
  1. データクレンジング (|overnight|<=10% で異常gap/分割未調整を除去)
  2. jump magnitude バケット (大overshootほど反転強いか)
  3. 下落側Long-only / 上昇側Short-only の非対称性
  4. 月末集中
  5. パッシブ執行コストモデル (メイカー: 往復0-5bps)
  6. 最良サブセット (月末×大下落jump Long-only) の IS/OOS

データ: 初版の observations.csv を再利用 (close_jump, overnight 計算済み)
"""
from __future__ import annotations

import os, sys
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

sys.stdout.reconfigure(line_buffering=True)
HERE = os.path.dirname(__file__)
OBS = os.path.join(HERE, '..', '20260531_closing_auction_reversion', 'observations.csv')
OOS_START = pd.Timestamp('2025-08-05')  # 初版の分割日

df = pd.read_csv(OBS)
df['date'] = pd.to_datetime(df['date'])
n0 = len(df)

# 1. クレンジング: overnight 異常値除去 (分割未調整/価格エラー)
df = df[df['overnight'].abs() <= 0.10].copy()       # ±10%以内
df = df[df['close_jump'].abs() <= 0.05].copy()       # ±5%以内 (現実的なauction jump)
print("=" * 76)
print("クロージング・オークション反転 — 精密版")
print("=" * 76)
print(f"\n[クレンジング] {n0:,} → {len(df):,} (overnight異常値・極端jump除去)")

# 月末フラグ (月内で末尾から5営業日)
df['month'] = df['date'].dt.to_period('M')
df['rank_from_end'] = df.groupby('month')['date'].rank(ascending=False, method='dense')
df['monthend'] = df['rank_from_end'] <= 5
df['period'] = np.where(df['date'] >= OOS_START, 'OOS', 'IS')

# bps
df['on_bps'] = df['overnight'] * 1e4
df['jump_bps'] = df['close_jump'] * 1e4


def sharpe(x, ann=np.sqrt(252)):
    x = pd.Series(x).dropna()
    if len(x) < 10 or x.std() == 0:
        return float('nan')
    return float(x.mean() / x.std() * ann)


# ============================================
# A. jump magnitude バケット別 反転 (符号反転後の翌朝リターン)
# ============================================
print("\n" + "=" * 76)
print("A. close_jump の大きさ別 反転リターン (overshootに逆張りした時の翌朝bps)")
print("=" * 76)
print(f"\n  {'jumpバケット':<18} {'n':>7} {'jump平均bps':>11} {'翌朝on_bps':>11} {'逆張りリターン':>13} {'勝率%':>7}")
print("  " + "-" * 70)
# 逆張りリターン = -sign(jump) * overnight (jump上昇ならShort, 下落ならLong)
df['rev_ret'] = -np.sign(df['close_jump']) * df['on_bps']
buckets = [(-1e9,-100),(-100,-50),(-50,-20),(-20,20),(20,50),(50,100),(100,1e9)]
blabels = ['<-100','-100〜-50','-50〜-20','-20〜20','20〜50','50〜100','>100']
for (lo,hi),lab in zip(buckets, blabels):
    g = df[(df['jump_bps']>=lo)&(df['jump_bps']<hi)]
    if len(g)==0: continue
    print(f"  {lab:<18} {len(g):>7} {g['jump_bps'].mean():>11.1f} {g['on_bps'].mean():>11.1f} "
          f"{g['rev_ret'].mean():>13.1f} {(g['rev_ret']>0).mean()*100:>7.1f}")

# ============================================
# B. 下落側Long / 上昇側Short の非対称性
# ============================================
print("\n" + "=" * 76)
print("B. 下落側Long-only vs 上昇側Short-only (|jump|>=50bps, 逆張り翌朝決済)")
print("=" * 76)
print(f"\n  {'戦略':<24} {'期間':<8} {'n':>6} {'rev平均bps':>11} {'Sharpe':>8} {'勝率%':>7}")
print("  " + "-" * 64)
down = df[df['jump_bps'] <= -50]   # 大きく下げて引けた → 翌朝Long(反発)
up   = df[df['jump_bps'] >= 50]    # 大きく上げて引けた → 翌朝Short(反落)
for name, sub, side in [('下落側Long(jump<=-50)', down, 1), ('上昇側Short(jump>=50)', up, -1)]:
    for label, s in [('全期間', sub), ('IS', sub[sub.period=='IS']), ('OOS', sub[sub.period=='OOS'])]:
        ret = side * (s['on_bps']) if side==1 else -s['on_bps']  # Long: +on, Short: -on
        # 日次系列でSharpe
        daily = s.assign(r=side*s['on_bps']*(1 if side==1 else 1)).groupby('date')['on_bps'].mean()*side
        sh = sharpe(daily)
        print(f"  {name:<24} {label:<8} {len(s):>6} {ret.mean():>11.1f} {sh:>8.2f} {(ret>0).mean()*100:>7.1f}")

# ============================================
# C. 月末集中 × クロスセクションL/S
# ============================================
print("\n" + "=" * 76)
print("C. 月末集中 L/S (top10%上昇Short / bottom10%下落Long, 翌朝決済)")
print("=" * 76)

def ls_daily(sub, q=0.1):
    out=[]
    for dt,g in sub.groupby('date'):
        g=g.dropna(subset=['close_jump','on_bps'])
        if len(g)<20: continue
        k=max(1,int(len(g)*q))
        r=g.sort_values('close_jump')
        out.append({'date':dt,'ls': r.head(k)['on_bps'].mean()-r.tail(k)['on_bps'].mean()})
    return pd.DataFrame(out)

print(f"\n  {'サブセット':<16} {'期間':<8} {'n日':>5} {'gross/日bps':>11} {'net(3bps)':>10} {'Sh net':>8}")
print("  " + "-"*60)
COST = 3.0  # パッシブ(メイカー)往復3bps想定
for setname, dfsub in [('月末5営業日', df[df.monthend]), ('全日', df)]:
    for label, s in [('全期間', dfsub),('IS', dfsub[dfsub.period=='IS']),('OOS', dfsub[dfsub.period=='OOS'])]:
        ld=ls_daily(s)
        if len(ld)<10:
            print(f"  {setname:<16} {label:<8} {'n/a(<10日)':>5}"); continue
        net=ld['ls']-COST
        print(f"  {setname:<16} {label:<8} {len(ld):>5} {ld['ls'].mean():>11.1f} {net.mean():>10.1f} {sharpe(net):>8.2f}")

# ============================================
# D. 最良サブセット: 月末 × 下落側Long-only + パッシブコスト感度
# ============================================
print("\n" + "=" * 76)
print("D. 最良サブセット: 月末5営業日 × 大下落jump(<=-50bps) Long-only")
print("=" * 76)
best = df[df.monthend & (df.jump_bps <= -50)]
print(f"\n  {'期間':<8} {'n':>5} {'翌朝on平均bps':>13} {'勝率%':>7} {'t値':>7}")
print("  " + "-"*46)
for label, s in [('全期間', best),('IS', best[best.period=='IS']),('OOS', best[best.period=='OOS'])]:
    d=s['on_bps']
    t=d.mean()/d.std()*np.sqrt(len(d)) if d.std()>0 and len(d)>1 else 0
    print(f"  {label:<8} {len(s):>5} {d.mean():>13.1f} {(d>0).mean()*100:>7.1f} {t:>7.1f}")

print("\n  パッシブ執行コスト感度 (全期間 Long-only 平均bps):")
d=best['on_bps']
for c in [0,3,5,10]:
    print(f"    片道{c:>2}bps(往復{c*2}): net平均={d.mean()-c*2:+.1f}bps  発火/月≈{len(best)/18:.1f}回")

df.to_csv(os.path.join(HERE,'cleaned_obs.csv'), index=False)
print(f"\n  保存: cleaned_obs.csv")
print("\n完了")
