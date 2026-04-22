# 毎日のオペレーション — orb_breakout_long

## 🕘 09:00 (寄付)

- [ ] FOMC/日銀/SQ/対象2銘柄 (6146, 5706) の決算発表日でないか確認
- [ ] 前日 .SOX ≤ -2% かを確認 → 該当時は本日スキップ (sox_overnight_short 優先)
- [ ] 対象銘柄が特別気配継続でないか確認

## 🕧 09:30 — 三井金属 OR確定 / 監視開始

- [ ] 5706.T の 9:00-9:30 High / Low を記録
- [ ] 9:31 以降、1分バーごとに High > OR High を監視
- [ ] ブレイク発生 → **Long 成行**、Stop = OR Low を板に置く
- [ ] `trade_log.csv` に記録 (date / sym / or_high / or_low / entry_time / entry_price)

## 🕙 10:00 — ディスコ OR確定 / 監視開始

- [ ] vwap_morning_meanrevert が既に発動していれば ORB はスキップ
- [ ] 6146.T の 9:00-10:00 High / Low を記録
- [ ] 10:01 以降、1分バーごとに High > OR High を監視
- [ ] ブレイク発生 → **Long 成行**、Stop = OR Low を板に置く
- [ ] `trade_log.csv` に記録

## 🕧 10:00-15:25 — 建玉監視

- [ ] Stop 発動確認 (OR Low ヒット)
- [ ] Stop 約定したら `stop_hit=True` で記録

## 🕒 15:22-15:25 — 引け決済

- [ ] 建玉残がある銘柄に **引成** 決済注文
- [ ] 15:25 約定確認
- [ ] `exit_time / exit_price / gross_bps / net_bps` を追記

## 🌙 取引終了後

- [ ] 本日の P&L 集計
- [ ] 週末: 週次集計 → 想定Sharpe+2.15 との乖離確認
- [ ] 月末: 月次 Sharpe / WR / 取引数 / Stop Hit率 を記録

---

## 🚨 異常時対応

| 状況 | 対応 |
|---|---|
| ブレイク検知スクリプト停止 | 手動で OR High/Low を記録、目視監視 |
| 寄付遅延 (特別気配継続) | その銘柄は当日スキップ |
| ブレイク直後にギャップダウン (Stop 即時ヒット) | 記録は通常通り、最低1トレード挙動として計上 |
| サーキットブレーカー | 全建玉を速やかに決済 |

---

## 📊 週次/月次レビュー

### 毎週金曜 16:00
- [ ] 週次トレード数 (目安 5-6件)
- [ ] 銘柄別の寄与度 (ディスコ vs 三井金属)
- [ ] Stop Hit 率 (目安 30-40%)

### 毎月末
- [ ] 月次 Sharpe / WR / 取引数
- [ ] H1/H2 バックテスト値 (Sharpe+2.0) との比較
- [ ] 破綻条件チェック (3ヶ月連続マイナス)

---

## 📝 便利コマンド

```bash
# 過去日のシグナル検証
python3 strategies/orb_breakout_long/signal_check.py --date 2026-04-15

# 本日のリアルタイム監視
python3 strategies/orb_breakout_long/signal_check.py --live
```
