"""
既存6戦略 Sharpe測定法の監査 — √252 過大評価のチェック

背景: V6スコア研究 (20260531_overlap_inflation_check) で
「保有H日のリターンを日次サンプルとして √252 で年率化すると √H 倍盛れる」
を確認した。同じ問題が昇格済み6戦略にも無いかを監査する。

各戦略の Sharpe 計算式 (ソース確認済み):
  topix/eneos/vwap_meanrevert/lasertec/bank_absorption/earnings_pead
      → per_trade.mean()/std() × √252   (= 年間252トレードを仮定)
  pre_earnings_drift
      → per_trade.mean()/std() × √(252/4)  (= 4日保有を考慮、より保守的)

問題: √252 は「年間252回の独立トレード」を前提とする。
実際のトレード頻度がこれより低い戦略は、standalone (単独運用) の
年率化Sharpeを過大評価する。

正しい関係 (非重複・独立トレードの場合):
  年率Sharpe(standalone) = per_trade_Sharpe × √(実トレード数/年)
  報告Sharpe = per_trade_Sharpe × √252
  → 補正 = 報告Sharpe × √(実トレード数/年 ÷ 252)

注意: 同時保有・クラスタ発火する戦略 (earnings系・bank) では
トレードが独立でないため、上式は overstate 寄りの近似。
厳密には日次ポートフォリオ収益系列のSharpe (earnings_pead検証 line340 の
sharpe_d が該当) が gold standard。
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd

sys.stdout.reconfigure(line_buffering=True)

# 戦略パラメータ (検証実行 20260511 の全期間値 + ソース確認した年率化係数)
# reported: 報告Sharpe, factor: 使用した年率化係数, N: 全期間トレード数, years: 検証年数
# hold: 保有営業日, cluster: True=同時保有/クラスタ発火 (独立性低)
STRATS = [
    # name, reported, ann_factor_used, N, years, hold_days, intraday, cluster
    dict(name="vwap_morning_meanrevert", reported=4.46, factor=np.sqrt(252), N=80,  years=2.0, hold=0, intraday=True,  cluster=False),
    dict(name="eneos_vwap_trend",        reported=1.81, factor=np.sqrt(252), N=104, years=2.0, hold=0, intraday=True,  cluster=False),
    dict(name="lasertec_ma25_support",   reported=2.69, factor=np.sqrt(252), N=39,  years=5.0, hold=10, intraday=False, cluster=False),
    dict(name="bank_absorption",         reported=3.94, factor=np.sqrt(252), N=70,  years=5.0, hold=5,  intraday=False, cluster=True),
    dict(name="earnings_pead",           reported=2.19, factor=np.sqrt(252), N=216, years=5.0, hold=5,  intraday=False, cluster=True),
    dict(name="pre_earnings_drift",      reported=2.07, factor=np.sqrt(252/4), N=1007, years=5.0, hold=4, intraday=False, cluster=True),
]

print("=" * 90)
print("既存6戦略 Sharpe測定法の監査 — √252 年率換算の影響")
print("=" * 90)
print("""
報告Sharpe = per_trade_Sharpe × (使用係数)
  ・ほぼ全戦略が ×√252 (=15.87) を使用 → 年間252トレードを暗黙に仮定
  ・pre_earnings_drift のみ ×√(252/4)=7.94 (4日保有を考慮、保守的)

standalone年率Sharpe = per_trade_Sharpe × √(実トレード数/年)
""")

print(f"{'戦略':<26} {'報告Sh':>7} {'係数':>7} {'N':>5} {'年':>4} {'回/年':>6} {'/trade Sh':>10} {'補正Sh':>8} {'判定'}")
print("-" * 90)

rows = []
for s in STRATS:
    per_trade = s['reported'] / s['factor']
    trades_per_year = s['N'] / s['years']
    corrected = per_trade * np.sqrt(trades_per_year)
    verdict = "○2.0+" if corrected >= 2.0 else ("△1.0+" if corrected >= 1.0 else "✗<1.0")
    flag = "" if not s['cluster'] else " *"  # クラスタ戦略は近似注意
    print(f"{s['name']:<26} {s['reported']:>7.2f} {s['factor']:>7.2f} {s['N']:>5} "
          f"{s['years']:>4.1f} {trades_per_year:>6.1f} {per_trade:>10.3f} {corrected:>8.2f} {verdict}{flag}")
    rows.append(dict(strategy=s['name'], reported=s['reported'], per_trade_sharpe=round(per_trade,3),
                     trades_per_year=round(trades_per_year,1), corrected_sharpe=round(corrected,2),
                     cluster=s['cluster']))

print("\n  * = 同時保有/クラスタ発火戦略。トレードが独立でないため補正Shは overstate 寄りの近似")
print("    (厳密には日次ポートフォリオ収益系列のSharpeが必要)")

print("\n" + "=" * 90)
print("解釈")
print("=" * 90)
print("""
[1] 報告された Sharpe 2.0+ は「per-trame品質を年間252トレードに引き伸ばした値」であり、
    standalone (その戦略だけを1年運用) の年率Sharpeとは別物。
    実トレード頻度が 8〜52回/年 の戦略は、standalone Sharpe が大きく下がる。

[2] ただし V6 の「オーバーラップの幻」とは性質が異なる:
    - V6: 日次リバランス basket の20日リターンに √252 → 同一メトリクスの水増し (実体なし寄り)
    - 6戦略: per-trade のエッジは実在 (正のリターン・良好な t値)。問題は「年率換算の前提」だけ。

[3] per-trade Sharpe (= 1トレードあたりの質) を見ると:
    vwap_morning 0.28 / pre_earnings 0.26 が高く、lasertec 0.17 / eneos 0.11 と続く。
    これは「1回のトレードがどれだけ良いか」であり、ここは本物。

[4] standalone年率では多くが 0.5〜1.8 で2.0未達。
    だが【重要】6戦略 + 候補を ポートフォリオ運用 すれば資金が年中稼働し、
    アグリゲートの年率Sharpeは個別より高くなる (これが実際の運用形態)。
    低頻度戦略を「単独で2.0」と期待するのが誤りで、戦略バスケットとして評価すべき。

[5] pre_earnings_drift は √(252/4) で測っており、最も誠実な年率化。
    クラスタ性を考えても比較的頑健。

【結論】
  ・既存戦略は「インチキ」ではない (per-trade エッジは実在、t値も良好)
  ・だが headline Sharpe(√252) は standalone 年率Sharpe を過大評価しており、
    市場Sharpeと直接比較できない。昇格バー≥2.0 は per-trade×√252 に適用されていた。
  ・正しい評価は「日次ポートフォリオ収益系列のSharpe」または「戦略バスケット全体のSharpe」。
  ・最優先の次アクション: 6戦略を1つの日次資金曲線に合成し、ポートフォリオ年率Sharpeを測る。
""")

out = os.path.join(os.path.dirname(__file__), "audit_results.csv")
pd.DataFrame(rows).to_csv(out, index=False)
print(f"保存: {out}")
print("\n完了")
