# DATA SCHEMA — 分析用データベース構成

このリポジトリの分析・バックテストが参照する PostgreSQL `market_data` のスキーマ案内。
データ取得パイプラインは別リポ ([trading-system-redesign](https://github.com/telesuhr/trading-system-redesign)) と Mac ローカル `~/claude-code/DataFetcher/` で管理。

## 接続情報

```python
PG_CONFIG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}
```

NAS MariaDB (`100.92.181.92:3306`) は **2026-05 を最後にトレーディング用途では使わない**。
NAS は今後バックアップ受け・長期保管のみ。

## 2026-05-10 時点の重要変更

Refinitiv/Bloomberg 解約方針に伴い、データソースを **JQuants 一本化**。
旧 Refinitiv 由来テーブルは `archive` スキーマに退避し、新 JQuants 由来テーブルが `public` に並ぶ。

詳細経緯: [trading-system-redesign/MIGRATION_LOG.md](https://github.com/telesuhr/trading-system-redesign/blob/main/MIGRATION_LOG.md)

---

## スキーマ全体

### `public` — 現行データ（自動更新）

| テーブル | 行数 | 内容 | 更新頻度 |
|---|---|---|---|
| `symbol_master` | 4,449 | 銘柄マスタ (5桁/4桁/RIC/ScaleCat/Mrgn) | 週次 |
| `stocks_daily` | 1,007万 | 日足 OHLCV + 調整値（10年分） | 日次 |
| `stocks_intraday` | (進行中→2.5億) | 1分足（過去2年）。月次パーティション | 日次 |
| `index_daily` | 29K | TOPIX + 主要指数日足 | 日次 |
| `index_master` | 12 | 指数マスタ | 必要時 |
| `investor_types` | 2,395 | 投資部門別売買（週次） | 週次 |
| `fin_summary` | 19万 | 財務サマリ (JSONB) | 日次 |
| `earnings_calendar` | snapshot | 決算発表予定 | 日次 |
| `jquants_margin_interest` | 1.10M | 信用残（週次） | 週次 |
| `jquants_margin_alert` | 256K | 信用規制（日次） | 日次 |
| `jquants_short_ratio` | 44K | 業種別空売り比率 | 日次 |
| `jquants_short_sale_report` | 893K | 空売り残高報告 | 日次 |
| `sectors_33` / `sectors_17` / `market_segments` / `trading_calendar` | — | 静的マスタ | 月次 |

### `archive` — 旧データ（凍結、Refinitiv 解約後は更新停止）

| テーブル | 行数 | 旧内容 |
|---|---|---|
| `archive.intraday_data` | 14.5M | Refinitiv 1分足 (122銘柄、`symbol`=RIC, `timestamp`=UTC, 2024-11〜2026-05) |
| `archive.daily_stats` | 39K | Refinitiv 派生指標 (`symbol`=RIC, `trade_date`) |
| `archive.data_fetch_log` | 713K | Refinitiv 取得ログ |
| `archive.fins_statements` | 0 | 旧設計（fin_summary に統合） |

`archive.*` は `public.*` 名前で参照しても **`relation "..." does not exist`** で即エラーになる。
これは「死んだデータを誤って使うリスク」を最小化するための意図的設計。

---

## 銘柄コードの正規化

| 形式 | 例 (トヨタ) | どこで使う |
|---|---|---|
| **5桁** | `72030` | JQuants 由来テーブル全般 (`stocks_daily.code`, `stocks_intraday.code`, `jquants_*.code`, `symbol_master.code5`) |
| 4桁 | `7203` | 東証一般表記。`symbol_master.code4` (生成カラム) |
| RIC | `7203.T` | 旧 Refinitiv (`archive.intraday_data.symbol` 等)。`symbol_master.ric` |

JQuants 由来は **5桁が canonical**。RIC/4桁から変換するなら：

```sql
-- RIC → 5桁
SELECT s.code5 FROM symbol_master s WHERE s.ric = '7203.T';

-- 4桁数字 → 5桁（普通株）
SELECT '7203' || '0' AS code5;
```

---

## 主要クエリのパターン

### 1分足を取得する

```python
# 現行: stocks_intraday (5桁)
sql = """
SELECT ts, open, high, low, close, volume
FROM stocks_intraday
WHERE code = %s AND ts >= %s AND ts < %s
ORDER BY ts
"""
df = pd.read_sql(sql, conn, params=('72030', '2026-04-01', '2026-05-01'))
# ts は JST 直接（タイムゾーン変換不要）
```

```python
# 旧 Refinitiv (archive 経由、2026-05-08で凍結):
sql = """
SELECT timestamp, open, high, low, close
FROM archive.intraday_data
WHERE symbol = '7203.T' AND timestamp BETWEEN ... 
"""
# timestamp は UTC、JST に直すには +9h
```

### 日足を取得する

```python
sql = """
SELECT date, open, high, low, close, volume,
       adj_open, adj_high, adj_low, adj_close, adj_volume
FROM stocks_daily WHERE code = %s ORDER BY date
"""
```

### ティック（生約定データ）

DuckDB 経由で CSV.gz を直接クエリ（PG投入なし）。
データは `~/Data/jquants_trades/`、ヘルパは `~/claude-code/DataFetcher/src/ticks.py`。

```python
import sys; sys.path.insert(0, '/Users/Yusuke/claude-code/DataFetcher')
from src.ticks import TickQuery
tq = TickQuery()
df = tq.bars(code='72030', start='2025-01-01', end='2025-01-31', freq='5min')
```

---

## 既存戦略コードの状態（2026-05-10）

`strategies/*/signal_check.py` は **未移行**で、旧テーブル `intraday_data` / `daily_stats` を直接参照したまま（実DBには存在せず、`archive.*` に移動）。

⚠ **これらのコードを新規分析の参考にしない。**
古い戦略コードを動かしたい場合は手動で `archive.*` に書き換えるか、新テーブルへ書き直す。新規分析は必ず `stocks_intraday` / `stocks_daily` を使う（[.claude/CLAUDE.md](.claude/CLAUDE.md) のテンプレ参照）。

| 戦略 | 旧テーブル参照 | 新テーブル移行 |
|---|---|---|
| `lme_on_copper`, `nonferrous_lme_link` | LME銘柄依存（`CMCU3` 等） | **不可**（LME解約方針）→ archive 候補 |
| `orb_breakout_long`, `vwap_morning_meanrevert`, `topix_overnight`, `eneos_vwap_trend` | `intraday_data` (RIC) | `stocks_intraday` (5桁、JST) に要書き換え |
| `lasertec_ma25_support` | `daily_stats` (RIC) | `stocks_daily` (5桁) に要書き換え |
| `sox_overnight_short`, `semi_sox_fade`, `pair_portfolio` | 未調査 | 別途確認 |
