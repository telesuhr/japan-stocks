# 市場データ スキーマ仕様

## データソース
- **ソースDB**: NAS MariaDB (`refinitiv_news` 経由で Refinitiv Eikon)
- **分析DB**: ローカル PostgreSQL (`market_data`)
- **同期**: `RfNews/sync_mariadb_to_postgres.py` で差分同期
- **ライセンス**: Refinitiv Eikon契約下、**再配布禁止**

## PostgreSQL接続
```python
PG_CONFIG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}
```

## メインテーブル: `intraday_data`

| 列名 | 型 | 内容 | 備考 |
|------|---|------|------|
| `symbol` | VARCHAR | シンボル (例: `5711.T`, `CMCU3`, `.TOPX`) | RIC形式 |
| `timestamp` | TIMESTAMP | 時刻 | **UTC保存**, JSTは `+9h` |
| `interval` | VARCHAR | 足種別 | `'1min'` 固定 |
| `open` | NUMERIC | 始値 | |
| `high` | NUMERIC | 高値 | |
| `low` | NUMERIC | 安値 | |
| `close` | NUMERIC | 終値 | |
| `volume` | NUMERIC | 出来高 | 指数系は NULL or 0 |

主キー: `(symbol, timestamp)`
インデックス: `symbol`, `timestamp`

---

## ⚠ 最重要ルール: タイムスタンプ

**全データ UTC で保存されている。JST取引時間と対応させるには必ず +9h する。**

```python
df['jst'] = df['timestamp'] + pd.Timedelta(hours=9)
```

よくある間違い:
- `+18h` で変換 → 翌日になってしまう
- 時差を `+9h` ではなく `+8h` (CST) と混同

---

## 取引時間 (JST)

### 日本株 (.T suffix)
- **前場**: 9:00-11:30 (UTC 0:00-2:30)
- **後場**: 12:30-15:30 (UTC 3:30-6:30)
- 昼休み中はデータなし

### LME銅 (CMCU3)
- **アジア**: 9:00-16:00 JST (UTC 0:00-7:00)
- **ロンドン**: 16:00-翌1:00 JST (UTC 7:00-16:00)
- **クローズ**: JST 3:00頃 (日本株寄付の約6時間前)
- **BST期間中**: オープン時刻が1時間前倒し (9:00 → 10:00)

### 日経先物 (JNIc1) / TOPIX先物
- ほぼ24時間(日中/夜間セッション)
- 夜間セッションは前日15:15-翌6:00 頃

### WTI原油 (CLc1) / Brent原油 (LCOc1)
- ほぼ24時間 (NY時間ベース)

### TOPIX現物 (.TOPX)
- 9:00-11:30, 12:30-15:30 JST (日本株と同じ)

---

## BST期間 (夏時間) の扱い

LME/欧州市場は3月末〜10月末に夏時間 (BST/CEST)。
`mdutil.is_bst(date)` で判定。BST中はLMEオープンが1時間前倒し。

```python
BST_PERIODS = [
    (date(2024, 3, 31), date(2024, 10, 27)),
    (date(2025, 3, 30), date(2025, 10, 26)),
    (date(2026, 3, 29), date(2026, 10, 25)),
]
```

---

## シンボル一覧 (本プロジェクトで使用)

### 日本株 (`.T` suffix = Tokyo)

#### 景気敏感コア5 (CORE5)
| シンボル | 銘柄名 | セクター |
|---|---|---|
| 5711.T | 三菱マテリアル | 非鉄金属 |
| 6501.T | 日立製作所 | 総合電機 |
| 7011.T | 三菱重工業 | 重機・防衛 |
| 5016.T | 出光興産 | 石油 |
| 4502.T | 武田薬品 | 医薬品 (ディフェンシブ) |

#### 非鉄 (NONFERROUS)
| 5706.T | 三井金属 |
| 5713.T | 住友金属鉱山 |

#### 半導体 (SEMICON)
| 8035.T | 東京エレクトロン (TEL) |
| 6857.T | アドバンテスト |
| 6146.T | ディスコ |
| 4063.T | 信越化学 |
| 6963.T | ローム |

#### 海運 (SHIPPING)
| 9101.T | 日本郵船 |
| 9104.T | 商船三井 |
| 9107.T | 川崎汽船 |

#### エネルギー (ENERGY)
| 1605.T | INPEX |
| 5020.T | ENEOS |

#### 内需 (LS Short leg用, DOMESTIC_SHORT)
| 8267.T | イオン |
| 9020.T | JR東日本 |
| 7974.T | 任天堂 |
| 6758.T | ソニー |
| 8411.T | みずほ |

### 外部指標
| シンボル | 内容 |
|---|---|
| `.TOPX` | TOPIX現物指数 |
| `JNIc1` | 日経225先物 Generic 1st |
| `CMCU3` | LME銅 Cash |
| `CLc1` | WTI原油 Generic 1st |
| `LCOc1` | Brent原油 Generic 1st |

---

## データ取得の標準パターン

### 1分足 全期間
```python
import mdutil as U
df = U.fetch_intraday('5711.T')  # jst index で返る
```

### 日次OHLC (9:00寄り/15:20-30引けから合成)
```python
jp = U.load_jp_daily('5711.T')  # date index, open/close列
```

### 前場データ抽出
```python
morning = df[(df.index.hour >= 9) &
             ((df.index.hour < 11) | ((df.index.hour == 11) & (df.index.minute <= 30)))]
```

---

## データ品質の既知の注意点

1. **出来高ゼロ分足の混入**: 流動性低い時間帯はvolume=0 だが価格は前足引継ぎ
2. **指数系のvolume = NULL**: `.TOPX` などはNULL、dropna時に注意
3. **LME BST切替日**: 開始/終了日は1時間ずれる可能性
4. **株式分割異常値**: `|日次return| > 15%` を外れ値として除外 (`U.OUTLIER_PCT = 15.0`)
5. **新規上場/期中データ**: 5713.T は 2024-12-02 から、6963.T は 2025-03-25 からのデータ

---

## コスト前提 (全分析共通)

```python
COST_BPS = 4.0       # 片側2bps × 往復 (アウトライト)
COST_BPS_LS = 8.0    # LS戦略: 片側2bps × 往復 × 2銘柄
```

実運用では更に以下を考慮:
- 滑り (成行発注時の板ズレ)
- 信用金利 (Short側)
- 消費税・配当落ち

---

## 生データ vs Git管理

| 種別 | Git管理 | 備考 |
|---|---|---|
| テーブルスキーマ定義 | ✓ | 本ファイル |
| fingerprint.json | ✓ | 各シンボル件数・ハッシュ |
| サンプルCSV (5日×27銘柄, <3MB) | ✓ | `data/sample/` |
| 集計アウトプット (CSV, PNG) | ✓ | `analyses/*/` |
| 生1分足 全期間データ | ✗ | Refinitivライセンス違反 |
| DB接続情報 | ✗ | コード内の `PG_CONFIG` はローカル前提 |

---

## 参照

- `fingerprint.json`: 各シンボルの件数・期間・md5ハッシュ
- `SAMPLE_SUMMARY.md`: サンプルCSVの概要
- `generate_artifacts.py`: 本データ生成スクリプト
- `../analyses/20260421_common/mdutil.py`: データロード実装
