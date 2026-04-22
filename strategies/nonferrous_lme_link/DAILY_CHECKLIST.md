# 毎営業日チェックリスト — nonferrous_lme_link

## 寄付前 (08:55 〜 08:59 JST)

### 1. シグナル判定
```bash
cd ~/claude-code/japan-stocks/strategies/nonferrous_lme_link
python3 signal_check.py
```

- [ ] LME ON 変化値を確認
- [ ] TIER1 / TIER2 / 無シグナルを判定
- [ ] 方向 (Long / Short) を確認

### 2. 発注 (シグナル発動時)
- [ ] 8 銘柄全て寄成で発注 (Long または Short 一括)
  - 5711.T 三菱マテ
  - 5706.T 三井金
  - 5713.T 住友金鉱 ★最強
  - 5714.T DOWA
  - 5016.T JX金属
  - 5801.T 古河電
  - 5802.T 住友電工
  - 5803.T フジクラ
- [ ] 各銘柄 1/8 ロット (個別ウェイト調整適用なら指示通り)
- [ ] 決済時刻メモ (TIER1=15:30 / TIER2=11:00)

### 3. 寄付後監視 (09:00 〜 09:30)
- [ ] 含み損が個別銘柄で -150bps 超 → 該当銘柄のみクローズ
- [ ] 8銘柄合計 -300bps 超 → 全クローズ

### 4. 決済発注
- TIER1: **15:25 引け成行で反対売買**
- TIER2: **11:00 寄成で反対売買**
- [ ] 全 8 銘柄同時に発注 (片側残し禁止)

### 5. 記録
- [ ] `trade_log.csv` に記録 (entry_date, on_bps, tier, direction, 各銘柄 pnl_bps)

## 異常時フロー
- **LME データ未到着 (08:55 時点で >12h 欠損)**: 当日エントリ見送り
- **MariaDB→PostgreSQL 同期遅延**: `cd ~/RfNews && python3 sync_mariadb_to_postgres.py` で確認
- **個別銘柄が異常気配 (M&A 報道・決算リーク)**: その銘柄を除外、残り 7 銘柄のみ発注

## トラブルシュート
- `psycopg2` 未インストール → `pip install psycopg2-binary`
- LME データが古い場合 → IntraTrading 配下の同期スクリプトを実行
