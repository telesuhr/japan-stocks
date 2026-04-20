# Japan Stocks Trading

## プロジェクト概要
日本株のイントラデイ・スキャルピング戦略の分析・バックテスト。

## 主な分析対象
- 半導体・電機セクター
- ペアトレード（統計的アービトラージ）
- ORB（オープンレンジブレイクアウト）

## 共通知識
`~/.claude/rules/intraday-trading.md` を参照。
タイムスタンプは必ず +9h でJST変換。

## DB接続
```python
PG_CONFIG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}
```
