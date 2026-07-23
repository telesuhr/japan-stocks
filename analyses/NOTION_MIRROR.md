# Notion 閲覧ミラー（人間の閲覧用・派生ビュー）

分析レポートの**正本(source of truth)は常にこの Git リポジトリ**。Notion はスマホ/ブラウザ閲覧・
共有用の**派生ビュー**であり、**git → Notion の一方向同期のみ**。Notion 側の編集は Git に戻さない
（次回同期で上書きされる）。コード(`run.py`)・データ(`csv`)・全文 README は Git に置き、
Notion 各行は **GitHubリンク**で参照する。

> 背景: 旧「ANALYSIS NOTEBOOK」(LME時代) は 2026-04 で更新停止＝二重管理でドリフトした前例あり。
> それを繰り返さないため、同期先IDをここに固定し、要約だけ・一方向・仕組みで追従する。

## 同期先（2026-07-23 構築）

| 対象 | Notion | ID |
|---|---|---|
| ハブページ | 🇯🇵 日本株トレード研究 (Git連携ミラー) | `3a627144-a4f0-81d9-a720-c4ce2178ba03` |
| 戦略DB | 採用/候補/一時停止/却下 | data_source `648ce331-fb51-47e7-a98f-0d20ba00e1db` |
| 分析ログDB | 各分析の索引 | data_source `5d20cdea-aa74-4b1c-8f3d-456c47a2e73a` |

- ハブ: https://app.notion.com/p/3a627144a4f081d9a720c4ce2178ba03

## 何をミラーするか（要約レイヤーのみ）

- **戦略DB** ← `SUMMARY.md`（実運用推奨セット＋LME系＋却下済み）。
  プロパティ: 戦略名 / 状態(採用・候補・一時停止・却下) / 種別 / Sharpe / OOS状態 / 備考 / GitHub。
- **分析ログDB** ← `analyses/README.md` 索引。1分析1行。
  プロパティ: 分析(フォルダ名) / 日付 / 判定(⭐⭐⭐〜✗却下・🔧ツール) / セクター / 一言 / GitHub。

**載せないもの**: `run.py` / `csv` / result.png / 全文README（全部 Git 側。Notion行の GitHub リンクで飛ぶ）。

## 同期手順（対話セッションからのみ。cron禁止）

Notion API は不安定（タイムアウトあり）。**自動バッチ(cron)には絶対入れない**。
更新は対話セッションで、Notion MCP ツールを使って upsert する:

1. `SUMMARY.md` / `analyses/README.md` を読む。
2. 既存 Notion 行を `notion-fetch`（data_source）で取得し、**戦略名/分析(フォルダ名)をキーに突合**。
3. 新規は `notion-create-pages`、変更は `notion-update-page` で upsert（title一致で重複を避ける）。
4. 迷ったら「Git が正・Notion は追従」。Notion 側の手編集は尊重しない。

将来 `/notion-sync` スキルに落とす想定。それまでは本ファイルの手順で手動同期。

## バックフィル状況

- 戦略DB: 現行ロスター（採用/候補/一時停止＋主要却下）を投入済み（22件）。
- 分析ログDB: **2026-07 の全分析＋主要フラッグシップのみ投入（10件）**。
  2026-05〜06 の約50フォルダは未バックフィル。次回同期で追加してよい（優先度低）。

## 旧 Notion 資産

- 「ANALYSIS NOTEBOOK」/「LME」配下ページ群（〜2026-04）は**別系統・非同期**。
  参照は可だが本ミラーとは混同しない。整理（アーカイブ）は任意の後続作業。
