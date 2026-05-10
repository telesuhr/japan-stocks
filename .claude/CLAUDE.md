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

**新規分析は必ず以下のテーブルを使うこと。** DB の全体像・カラム詳細は [DATA_SCHEMA.md](../DATA_SCHEMA.md) 参照。

### 必ず使うテーブル（canonical, 自動更新中）

| 用途 | テーブル | キー | timestamp |
|---|---|---|---|
| 1分足 | **`stocks_intraday`** | `(code, ts)` | JST 直接（変換不要） |
| 日足 | **`stocks_daily`** | `(code, date)` | DATE |
| 銘柄マスタ | `symbol_master` | `code5` | — |
| 指数日足 | `index_daily` | `(code, date)` | DATE |
| 信用残（週次）| `jquants_margin_interest` | `(date, code)` | — |
| 信用規制 | `jquants_margin_alert` | `(pub_date, code)` | — |
| 業種別空売り | `jquants_short_ratio` | `(date, s33)` | — |
| 空売り報告 | `jquants_short_sale_report` | `(disc_date, code, ss_name)` | — |
| 投資部門別 | `investor_types` | `(pub_date, section)` | — |
| 財務サマリ | `fin_summary` | `(code, disc_no)` | — |
| ティック | (DBなし) | — | DuckDB 経由 |

ティックは `~/claude-code/DataFetcher/src/ticks.py` の `TickQuery` を使う。

### 銘柄コードは必ず JQuants 5桁

```python
# トヨタ → '72030'  (4桁 7203 + 普通株 0)
# 5桁が canonical。RIC ('7203.T') や4桁 ('7203') を貰ったら必ず変換
SELECT code5 FROM symbol_master WHERE ric = '7203.T';
```

### 使ってはいけないテーブル（archive、Refinitiv解約で凍結）

⚠ 以下は新規分析で **絶対に使わない**。誤って使うと `relation does not exist` エラー or 古いデータで分析事故。

- `intraday_data` ／ `archive.intraday_data` ← 122銘柄のみ・2026-05-08 で凍結
- `daily_stats` ／ `archive.daily_stats` ← Refinitiv 派生指標・凍結
- `data_fetch_log` ／ `archive.data_fetch_log`
- `fins_statements` ／ `archive.fins_statements`

過去戦略 (`strategies/*/signal_check.py`) はまだ旧テーブル名を参照しているが、これは未移行コードなので新規分析の参考にしないこと。

### 新規分析テンプレ（コピペ用）

```python
import psycopg2, pandas as pd

PG_CONFIG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}

def load_minute(code5: str, start: str, end: str) -> pd.DataFrame:
    """5桁コード・JST文字列で 1分足を取得。返り値は ts インデックス（JST naive）"""
    conn = psycopg2.connect(**PG_CONFIG)
    sql = """
        SELECT ts, open, high, low, close, volume, turnover_value
        FROM stocks_intraday
        WHERE code = %s AND ts >= %s AND ts < %s
        ORDER BY ts
    """
    df = pd.read_sql(sql, conn, params=(code5, start, end))
    conn.close()
    return df.set_index('ts')

def load_daily(code5: str, start: str, end: str) -> pd.DataFrame:
    conn = psycopg2.connect(**PG_CONFIG)
    sql = """
        SELECT date, open, high, low, close, volume,
               adj_open, adj_high, adj_low, adj_close, adj_volume
        FROM stocks_daily
        WHERE code = %s AND date BETWEEN %s AND %s ORDER BY date
    """
    df = pd.read_sql(sql, conn, params=(code5, start, end))
    conn.close()
    return df.set_index('date')

def ric_to_code5(ric: str) -> str:
    """'7203.T' → '72030' （symbol_master 経由）"""
    conn = psycopg2.connect(**PG_CONFIG)
    cur = conn.cursor()
    cur.execute("SELECT code5 FROM symbol_master WHERE ric=%s", (ric,))
    r = cur.fetchone()
    conn.close()
    return r[0] if r else None
```

### ティック取得テンプレ

```python
import sys; sys.path.insert(0, '/Users/Yusuke/claude-code/DataFetcher')
from src.ticks import TickQuery
tq = TickQuery()
bars = tq.bars(code='72030', start='2025-01-01', end='2025-01-31', freq='5min')
```

## ファイル管理
- 分析スクリプトは必ずセッションごとのフォルダに格納する
- フォルダ名: `analyses/YYYYMMDD_分析内容/`
- 例: `analyses/20260421_semiconductor_leadlag/`
- セッション開始時に必ずフォルダを作成してから作業を始める

## Git・Push ルール

### コミット粒度
- **セッション単位でまとめてコミット** (複数分析を1コミットに集約)
- コミットメッセージ形式: `analyses: テーマA・テーマB・テーマC (YYYYMMDD)`
- 発見の数値はメッセージに含めない (README に書く)

### Push タイミング
- コミット後は**即座に Push** する

### セッション末尾の必須作業 (コミット前に必ず実施)
1. **`analyses/README.md` の目次を更新する**
   - 新規分析フォルダを大分類の適切な位置に追記
   - テーマ・主要発見・Sharpe (あれば) を記載
   - 「最新分析」セクションを当日分に差し替える
2. 上記を含めてまとめてコミット → 即Push

### 各分析フォルダの必須ファイル
```
analyses/YYYYMMDD_テーマ/
├── run.py       # 分析スクリプト
├── README.md    # 発見・解釈・結論 (詳細、後述)
├── result.png   # X投稿用グラフ (後述)
└── *.csv        # 集計データ (あれば)
```

### README.md 記載項目 (できる限り詳細に)
1. **分析の目的・仮説**
2. **データ概要** (期間・銘柄・件数)
3. **手法・パラメータ**
4. **主要発見** (数値・t統計量・N を明記)
5. **解釈・示唆** (なぜそうなるか)
6. **限界・注意点** (N不足・過学習リスク等)
7. **次のアクション候補**

## X (Twitter) 投稿用グラフ規則

`result.png` はそのままXに投稿できるスタンドアロン画像として作成する。

### サイズ・レイアウト
- **サイズ**: 1200×675px (16:9, Twitter推奨) を基本
- **タイトル** をグラフ上部に大きく表示 (日本語OK)
- **キー発見・数値** をグラフ内またはサブタイトルに注記
- **ソース・期間** をフッターに小さく記載 (例: `データ: 2025-04-01〜2026-04-21 / 日本株1分足`)

### デザイン
- 背景: 白 (#ffffff) または濃紺 (#0d1117) のどちらか (統一)
- フォント: 日本語対応 (`IPAexGothic` or `Noto Sans CJK JP` or `sans-serif` フォールバック)
- 色数: 3色以内を基本、アクセントカラー1色
- 軸ラベル・凡例は日本語で記載
- グリッドは薄く (alpha=0.3程度)

### matplotlib 設定テンプレート
```python
fig = plt.figure(figsize=(12, 6.75), facecolor='white')  # 1200x675px @100dpi
plt.rcParams.update({
    'font.family': ['IPAexGothic', 'Noto Sans CJK JP', 'sans-serif'],
    'axes.unicode_minus': False,
    'figure.facecolor': 'white',
    'axes.facecolor': '#f8f9fa',
    'grid.alpha': 0.3,
})
# フッター
fig.text(0.99, 0.01, 'データ: 2025-04-01〜2026-04-21 / 日本株1分足 (Refinitiv)',
         ha='right', va='bottom', fontsize=8, color='gray')
plt.savefig('result.png', dpi=100, bbox_inches='tight', facecolor='white')
```
