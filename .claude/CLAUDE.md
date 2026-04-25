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
