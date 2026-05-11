# _archive — 廃止された戦略 (LME/米マクロ依存)

2026-05-10 のデータベース統合 (Refinitiv → JQuants 一本化、LME解約) に伴い、以下の戦略は**運用停止**された。
本ディレクトリは過去の参考用に保管するもので、新規分析では参照しないこと。

詳細経緯: [DATA_SCHEMA.md](../../DATA_SCHEMA.md) / [trading-system-redesign](https://github.com/telesuhr/trading-system-redesign)

## 廃止戦略一覧

| 戦略 | 廃止理由 | 旧Sharpe |
|------|---------|---------|
| [lme_on_copper](lme_on_copper/) | LME銅 (CMCU3) 取得不可 + 2026-05-09検証で Sharpe -3.14 と機能停止 | +12.34 (旧) |
| [nonferrous_lme_link](nonferrous_lme_link/) | LME依存 | — |
| [sox_overnight_short](sox_overnight_short/) | 米国指数 (.SOX/ESc1/VXc1) NAS MariaDB凍結 | +2.11 |
| [semi_sox_fade](semi_sox_fade/) | .SOX依存 | — |

## なぜ復活させないか
- **LME解約方針**: コスト削減のためRefinitive契約終了
- **米国指数データソース未確保**: JQuants は日本市場専用
- 2026-05-10時点で代替データ手段なし

## 将来の復活シナリオ
これらの戦略を復活させる場合:
1. JQuants以外のデータソース (例: yfinance, FRED API) で .SOX / ESc1 / LME銅 をフィードする独自パイプライン構築
2. または日本株のみで類似シグナルを生成 (例: SOX相関銘柄バスケットで .SOX を擬似化)

代替案検討 → `analyses/` で実証 → 採用判定 → `strategies/` 復活、の手順を踏む。
