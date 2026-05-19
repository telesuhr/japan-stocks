# 採用9戦略 直近6ヶ月 継続性チェック (2026-05-18)

## 分析の目的・仮説
採用中の9戦略が直近の相場 (2025-11-15〜2026-05-15) でも依然として機能しているかを確認し、
**Sharpe < 1.0 に大幅劣化した戦略を `_archive/` に退避** する判定を行う。

## データ概要
- 期間: 2025-11-15 〜 2026-05-15 (約6ヶ月)
- 既存5戦略 (`run_6months.py`): `stocks_intraday` / `stocks_daily` から直接シグナル再生成
- 新規4戦略 (`aggregate_recent.py`): 各 `analyses/202605xx_*_validation/trades.csv` から該当期間を抽出

## 手法
- 既存5戦略: `analyses/20260511_strategy_validation_jquants/run.py` を `INTRADAY_START`/`DAILY_START = 2025-11-15` に書き換えて再実行
- 新規4戦略: trades CSV から `disc_date` (or `exit_date`) で期間フィルタ、Sharpe概算 = mean/std × √(252/hold_days)
- コスト: 片道2bps × 往復 = 4bps

## 主要発見

### 9戦略の Sharpe 推移 (5年Base → 直近6M)

| 戦略 | 5年Base | 直近6M | Δ | N (6M) | WR | PF | 判定 |
|---|---:|---:|---:|---:|---:|---:|:---:|
| lasertec_ma25_support | +7.57 | **+6.96** | -0.61 | 5 | 60.0% | 2.54 | ✅継続 |
| vwap_morning_meanrevert | +6.76 | **+4.58** | -2.18 | 24 | 62.5% | 2.13 | ⚠️低下 (基準内) |
| bank_absorption | +1.84 | **+3.95** | **+2.11** | 70 | 61.4% | 1.85 | ✅継続 (Base超え) |
| pre_earnings_drift | +2.07 | **+2.45** | +0.38 | 1007 | 62.2% | 2.54 | ✅継続 (Base超え) |
| eneos_vwap_trend | +3.81 | **+2.37** | -1.44 | 43 | 51.2% | 1.44 | ✅継続 |
| earnings_pead | +2.19 | **+2.17** | -0.02 | 216 | 54.6% | 2.36 | ✅継続 |
| **orb_breakout_long** | +2.31 | **+0.15** | -2.16 | 116 | 50.0% | 1.03 | ❌劣化 |
| **large_cap_oversold_reversal** | +2.92 | **+0.13** | -2.79 | 127 | 50.4% | 1.05 | ❌劣化 |
| **oversold_ma25_reversal** | +3.21 | **-0.03** | **-3.24** | 299 | 49.2% | 0.99 | ❌劣化 |

### 銘柄別 (orb_breakout_long)
- 三井金属 (57060): N=67, WR 46.3%, Sharpe **-0.28** → 完全劣化
- ディスコ (61460): N=49, WR 55.1%, Sharpe **+1.33** → 残存

## 解釈・示唆

### 機能停止3戦略の共通点：**「反発系」が効かない相場**
- 2025-2026 は半導体・非鉄ともに強い上昇トレンドが継続
- 「MA25-20%乖離からの反発」「Core30大幅下落からの反発」というシグナルが **そもそも発生しにくい**
- 発生してもPF≒1.0勝率50%前後で機能停止

### 機能継続6戦略の共通点：**方向性/構造アノマリー型**
- bank_absorption: 金利上昇局面で銀行強含み (ベース+2.11改善)
- pre_earnings_drift / earnings_pead: 決算ドリフトは相場局面非依存
- lasertec_ma25_support: 銘柄固定型は地合いに左右されにくい
- vwap_meanrevert / eneos_vwap_trend: イントラの構造アノマリーは継続

## 退避判定 (2026-05-18 実施)

`strategies/` → `strategies/_archive/` に退避:
1. **oversold_ma25_reversal** (Sharpe -0.03)
2. **large_cap_oversold_reversal** (Sharpe +0.13)
3. **orb_breakout_long** (Sharpe +0.15)

採用戦略数: **9 → 6** に変更。

## 限界・注意点
- 既存5戦略は実シグナルを再計算しているが、新規4戦略は trades.csv 抽出 (シグナル再計算なし)
- 6ヶ月N=5〜1007と幅があり、特に lasertec (N=5) は統計的信頼性低い
- 「劣化判定」は Sharpe<1.0 で機械的に行ったが、レジーム特定の余地あり

## 次のアクション候補
1. **oversold系 v2**: VIX/日経MA200条件で発動制限する分析
2. **orb_breakout v2**: ディスコ単独化で 5年Sharpe を再計測
3. **bank_absorption サイズ調整**: ベース大幅超えのため資金配分見直し
4. **6戦略のポートフォリオ相関分析**: 同時保有時の分散効果検証

## ファイル
- `run_6months.py`: 既存5戦略の直近6ヶ月再実行スクリプト
- `aggregate_recent.py`: 新規4戦略を含む9戦略集計スクリプト
- `summary.csv`: 9戦略のサマリーデータ
- `result.png`: X投稿用グラフ
- `results.csv`: run_6months.py の結果CSV
