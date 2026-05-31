# 既存6戦略 Sharpe測定法の監査

## 分析の目的

V6スコア研究で「保有H日のリターンを日次サンプルとして √252 で年率化すると √H 倍盛れる」
([`20260531_overlap_inflation_check`](../20260531_overlap_inflation_check/)) を確認した。
**同じ過大評価が昇格済み6戦略 (Sharpe 2.0+) にも無いか**を監査する。

## 発見した測定法 (ソースコード確認済み)

| 戦略 | Sharpe計算式 | 含意 |
|---|---|---|
| eneos / vwap_morning / lasertec / bank_absorption / earnings_pead | `per_trade.mean()/std × √252` | 年間252トレードを暗黙に仮定 |
| pre_earnings_drift | `per_trade.mean()/std × √(252/4)` | 4日保有を考慮、より保守的 |

`√252` は「年間252回の独立トレード」前提。実トレード頻度がこれより低い戦略は
**standalone (単独運用) の年率Sharpeを過大評価**する。

## 補正方法

非重複・独立トレードの場合:
```
年率Sharpe(standalone) = per_trade_Sharpe × √(実トレード数/年)
報告Sharpe            = per_trade_Sharpe × √252
→ 補正 = 報告 × √(実トレード数/年 ÷ 252)
```

## 主要発見

| 戦略 | 報告Sh | 実回/年 | per-trade Sh | 補正Sh(standalone) | 判定 |
|---|---|---|---|---|---|
| vwap_morning_meanrevert | 4.46 | 40 | 0.281 | **1.78** | △ |
| pre_earnings_drift* | 2.07 | 201 | 0.261 | **3.70** | ○ |
| bank_absorption* | 3.94 | 14 | 0.248 | **0.93** | ✗ |
| lasertec_ma25_support | 2.69 | 7.8 | 0.169 | **0.47** | ✗ |
| earnings_pead* | 2.19 | 43 | 0.138 | **0.91** | ✗ |
| eneos_vwap_trend | 1.81 | 52 | 0.114 | **0.82** | ✗ |

\* 同時保有/クラスタ発火戦略 → トレードが独立でないため補正Shは overstate 寄りの近似
(厳密には日次ポートフォリオ収益系列のSharpeが必要)

数値は検証実行 [`20260511_strategy_validation_jquants`](../20260511_strategy_validation_jquants/) の
全期間N + ソース確認した年率化係数より。期間: イントラ2年(2024-05〜) / 日足5年(2021-05〜)。

## 解釈 (V6の幻とは性質が違う)

### V6 (オーバーラップの幻) との違い
- **V6**: 日次リバランスbasketの20日リターンに √252 → **同一メトリクスの水増し** (実体なし寄り)
- **6戦略**: per-trade のエッジは**実在** (正リターン・良好なt値)。問題は「年率換算の前提」だけ

### per-trade Sharpe (1トレードの質) は本物
vwap_morning 0.28 / pre_earnings 0.26 が高く、lasertec 0.17 / eneos 0.11 と続く。
ここは実在するエッジ。

### standalone年率では2.0未達が多いが、それは「単独運用」の話
低頻度戦略 (年8〜52回) は資金が年中遊ぶため standalone Sharpe が下がる。
**しかし実運用は6戦略+候補のポートフォリオ**。資金が年中稼働すればアグリゲートの
年率Sharpeは個別より高い。「1戦略で単独2.0」を期待するのが誤りで、バスケットで評価すべき。

### pre_earnings_drift は最も誠実
√(252/4) で測っており、クラスタ性を考えても比較的頑健 (補正後も高い)。

## 結論

1. **既存戦略は「インチキ」ではない** — per-trade エッジは実在、t値も良好。
   V6の「メトリクス自体が幻」とは根本的に異なる。
2. **だが headline Sharpe (√252) は standalone年率Sharpe を過大評価**しており、
   市場Sharpeや他戦略と直接比較できない。昇格バー≥2.0 は per-trade×√252 に適用されていた。
3. **正しい評価軸 = 日次ポートフォリオ収益系列のSharpe** (earnings_pead検証 line340 の `sharpe_d` が該当) または **6戦略バスケット全体の資金曲線Sharpe**。

## 限界・注意点

- 補正式 `per_trade×√(回/年)` はトレード独立前提。クラスタ戦略 (earnings系・bank) では
  同時保有でトレードが相関するため、実際の standalone Sharpe は補正値とずれる
  (overstate 寄り)。厳密には日次ポートフォリオ収益系列が必要。
- bank_absorption の報告Sharpeは資料間で不整合 (戦略マップ3.94 / 一覧表1.84)。3.94を採用。
- イントラ戦略のN・期間は2年 (intraday data が2024-05〜のため)。

## 次のアクション候補 (最優先)

1. **6戦略を1本の日次資金曲線に合成し、ポートフォリオ年率Sharpeを測る** ← 真の実力判定
2. earnings_pead検証 line340 の日次Sharpe `sharpe_d` を全戦略に展開 (gold standard)
3. 昇格基準を「per-trade×√252 ≥ 2.0」から「日次ポートフォリオSharpe ≥ X」に再定義
4. per-trade Sharpe (トレードの質) と 年率Sharpe (資金効率) を README で区別して併記
