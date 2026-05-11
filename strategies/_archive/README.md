# _archive — 廃止された戦略

新規分析では参照しないこと。本ディレクトリは過去の参考用に保管。

## 廃止カテゴリA: LME/米マクロ依存 (2026-05-10 廃止)

データベース統合 (Refinitiv → JQuants 一本化、LME解約) に伴い**運用停止**。
詳細経緯: [DATA_SCHEMA.md](../../DATA_SCHEMA.md) / [trading-system-redesign](https://github.com/telesuhr/trading-system-redesign)

| 戦略 | 廃止理由 | 旧Sharpe |
|------|---------|---------|
| [lme_on_copper](lme_on_copper/) | LME銅 (CMCU3) 取得不可 + 2026-05-09検証で Sharpe -3.14 と機能停止 | +12.34 (旧) |
| [nonferrous_lme_link](nonferrous_lme_link/) | LME依存 | — |
| [sox_overnight_short](sox_overnight_short/) | 米国指数 (.SOX/ESc1/VXc1) NAS MariaDB凍結 | +2.11 |
| [semi_sox_fade](semi_sox_fade/) | .SOX依存 | — |

## 廃止カテゴリB: JQuants長期検証で機能消失 (2026-05-11 廃止)

`analyses/20260511_strategy_validation_jquants/` で5年データ再検証した結果、
**Sharpe < 採用基準2.0** となったため運用停止。

| 戦略 | 廃止時 Sharpe (5年) | 旧Sharpe (1年) | 廃止理由 |
|------|:---:|:---:|------|
| [topix_overnight](topix_overnight/) | **+0.58** (N=363) | +6.27 | 1年検証バイアス。5年では機能せず WR 55.6%/PF 1.16 のみ |
| [pair_portfolio](pair_portfolio/) | **+0.65** (N=447) | +1.37 | 18ペアEWで勝7敗7相殺。個別では半TEL-レーザー Sharpe+12.89 など有望ペアあり |

## 復活シナリオ

### カテゴリA (LME/米マクロ)
1. JQuants以外のデータソース (例: yfinance, FRED API) でデータパイプライン構築
2. または日本株のみで類似シグナルを生成 (例: SOX相関銘柄バスケットで .SOX を擬似化)

### カテゴリB (機能消失)
- **topix_overnight**: 期間限定で機能していた可能性 → どの市場レジームで有効かを `analyses/` で詳細化
- **pair_portfolio**: 勝ちペア7つのみ抽出した v2 を `analyses/` で検証 → 採用基準クリアなら復活
  - 推奨対象: 半TEL-レーザー(+12.89), 通NTT-KDDI(+7.13), 電機日立-三菱電(+5.47), 鉄JR東-JR東海(+4.65), 非鉄マテ-三井金(+3.80), 薬アステラス-大塚(+2.62), 電機ソニー-富士通(+2.52)

復活手順: `analyses/<候補名>/` で再検証 (Sharpe ≥ 2.0 & N ≥ 30 & t-stat ≥ 2.0) → OoS でも継続を確認 → `strategies/` に再昇格。
