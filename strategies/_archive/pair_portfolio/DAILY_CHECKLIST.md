# 毎営業日チェックリスト — pair_portfolio

## 引け後 (15:35 〜) または 翌朝 (07:00 〜 08:30)

### 1. シグナル判定
```bash
cd ~/claude-code/japan-stocks/strategies/pair_portfolio
python3 signal_check.py
```

- [ ] `🔔 【新規エントリー候補】` を確認
- [ ] `🔻 【エグジット対象】` を確認
- [ ] `📊 【保有中ポジション】` の現在 Z を確認

### 2. 新規エントリー発注 (あれば)
各候補ペアについて:
- [ ] 既存ポジがないことを確認
- [ ] 発注方向を決定 (Z の符号と反対側に建てる)
- [ ] p1 / p2 のサイジングを計算 (1 : |β|)
- [ ] 証券会社で **寄成** 注文 (両銘柄同時)
- [ ] `positions.json` にエントリを追加

### 3. エグジット発注 (あれば)
各 exit 対象について:
- [ ] reason (MR / STOP / TIME) を確認
- [ ] 証券会社で **寄成** 反対売買 (両銘柄同時)
- [ ] `positions.json` から該当エントリを削除
- [ ] `trade_log.csv` に記録 (entry_date, exit_date, pair, entry_z, exit_z, direction, pnl_bps, reason)

### 4. リスクチェック
- [ ] 当月累計 drawdown < 500 bps
- [ ] 各ペアの連敗 3 未満 / 累計 -300 bps 未満
- [ ] VIX < 40
- [ ] 保有ペアに重大イベント (M&A、決算サプライズ、TOB) なし

### 5. 記録
- [ ] `trade_log.csv` 更新
- [ ] 月末は月次レビュー実施 (`RULES.md` 参照)

## 異常時フロー
- **MariaDB 接続失敗**: `python3 signal_check.py --verify-db` で原因特定
- **シグナルが全く出ない**: データ最新日 (MAX trade_date) を確認。NAS 同期遅延の可能性
- **想定外の PnL 変動**: `positions.json` と実際の口座残高を照合

## トラブルシュート
- `pymysql` / `statsmodels` 未インストール → `pip install pymysql statsmodels`
- MariaDB Tailscale 接続不可 → LAN fallback 192.168.0.250 に自動切替される
