# 採用戦略の足元エッジ確認 (90/180/365日窓)

## 分析の目的

現状の採用6戦略 + 新候補(closing_auction_rebound)が「足元でもエッジを保っているか」を、
直近90/180/365日の複数窓で確認する。低頻度戦略は90日では N が少なく判定不能のため
複数窓で robust に見る。

## データ・手法

- `Dashboard_CC/strategy_oos_monitor.py` の各戦略 oos_* 関数を再利用
- 年率化 = per-trade Sharpe × **√(245/hold_days)** (保有期間考慮の正しい換算)
- コスト往復20bps、最新営業日 2026-05-29 基準
- closing_auction_rebound は [20260531_closing_auction_exec](../20260531_closing_auction_exec/) の OOS net 値

## 主要発見

| 戦略 | 90日 | 180日 | 365日 | 判定 |
|---|---|---|---|---|
| **pre_earnings_drift** | 1.63 (n1023) | 1.67 (t9.6) | 1.56 (t12.9) | ✅ 最頑健・足元エッジ明確 |
| **earnings_pead** | 1.07 (n120) | 1.71 (t3.9) | 1.61 (t5.0) | ✅ 健在 |
| **vwap_morning_meanrevert** | 4.24 (n20) | 2.86 (n25) | 4.26 (n37) | ✅ 健在 (高Sharpe・勝率56-62%) |
| **lasertec_ma25_support** | N/A (n3) | 3.26 (n9, 勝率78%) | 2.97 (n17, t2.5) | ✅ 健在 (低頻度) |
| **bank_absorption** | 1.29 (n52) | 0.83 (n89) | 1.20 (n181, t2.3) | ⚠️ 弱体化 (正だが減衰) |
| **eneos_vwap_trend** | **-1.27** (n29) | **-1.21** (n44) | +0.24 (n72) | ⛔ **壊れている** |
| closing_auction_rebound (候補) | — | — | net 2.15 (OOS) | ✅ 健在 (新制度=全データ足元) |

## 判定サマリー

- **健在 (足元エッジ明確) 4戦略**: pre_earnings_drift / earnings_pead / vwap_morning_meanrevert /
  lasertec_ma25_support。全窓で正、Sharpe 1.5〜4.3、N十分なものは t も有意。
- **弱体化 1戦略**: bank_absorption。正(Sharpe 0.8〜1.3)だがIS比で大幅減衰。要監視・継続。
- **要注意・薄利 1戦略 (当初「壊れている」と判定→訂正)**: **eneos_vwap_trend**。
  詳細診断 [eneos_diagnosis.md](eneos_diagnosis.md) → **「CRITICAL -1.27」はモニタの一律往復20bps
  コスト前提によるアーティファクト。シグナルは反転しておらず gross は全窓プラス (Sharpe1.25-2.92)、
  戦略本来の4bpsでは net +6.8bps(90日)とプラス**。ただし直近の gross ~10.8bps は薄く、現実的
  コスト~10bpsではほぼトントン。**停止はしないが最低確信度・要コスト実測**。
- **新候補 健在**: closing_auction_rebound (net Sharpe 2.15, 新制度ならではの足元エッジ)。

## 重要な注意点 (IS基準との比較は割引いて解釈)

- 本確認の Sharpe は **√(245/hold) の正しい年率化**。一方、`strategies/README.md` の
  IS基準値 (bank 3.94 等) の多くは **√252 (年間252トレード仮定) の過大評価** (本セッション
  [strategy_sharpe_audit](../20260531_strategy_sharpe_audit/) 参照)。
- 例: bank IS 3.94 は √252 ベース。√(245/5)=√49 換算なら IS相当 ~1.7。よって bank の
  「0.83 vs 3.94 = ratio0.33」は方法論ミスマッチを含み、**実際の劣化はもっと軽い** (proper IS~1.7 vs 足元0.8-1.2)。
- pre_earnings_drift の IS 2.07 は √(252/4) の proper 値なので、足元1.6との比較は妥当 (軽微劣化)。
- → **oos_ratio による「WARNING」は割引いて読み、絶対Sharpe と sign/t-stat を一次情報とすべき**。

## 結論

**6戦略すべて足元でシグナルは生存** (gross正)。pre_earnings_drift/earnings_pead/vwap/lasertec が
明確に健在、bank は弱いが正、**eneos はシグナル健在だが薄利・コスト感度高 (当初「壊れ」判定は
コスト前提アーティファクトと判明し訂正)**。新候補 closing_auction_rebound も足元で健在。
恒久エンジンは機能している。要対応は bank の監視継続と eneos の実コスト実測・サイズ管理。

## 次のアクション候補

1. **eneos_vwap_trend を一時停止** (strategies/README に停止フラグ、_archive検討前に劣化原因調査)
2. eneos劣化原因調査 (ENEOS株価特性・VWAP回帰パターンの変化、原油レジーム?)
3. bank_absorption の弱体化を継続監視 (同時保有上限の最適化も)
4. OOSモニタを週次cron化し全戦略の足元エッジを自動追跡
5. closing_auction_rebound のライブ執行検証 → 昇格でバスケット強化
