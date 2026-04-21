# 日次チェックリスト — ENEOS VWAP Trend

---

## 9:00 — 寄付き確認

- [ ] ENEOSが通常通り寄り付いたか確認
- [ ] 本日の決算発表・重大ニュースがないか確認
- [ ] VWAPトラッキング開始（ツール or 手動）

---

## 9:30 — シグナル判定

補助スクリプト:
```bash
cd ~/claude-code/japan-stocks/strategies/eneos_vwap_trend
python3 signal_check.py
```

手動計算の場合:
```
9:30時点の close 価格: ¥______
9:30時点の VWAP:       ¥______
乖離(dev) = (close / VWAP - 1) × 10000 = ______bps
```

- [ ] dev ≥ +50bps → **Long エントリー予定**
- [ ] dev ≤ -50bps → **Short エントリー予定**
- [ ] |dev| < 50bps → **本日スキップ**

---

## 9:31〜 — エントリー発注

```
発注種別: 成行
銘柄:     5020.T ENEOS
方向:     Long（buy） or Short（sell）
数量:     ¥[ポジションサイズ] ÷ 現値 ≒ _____ 株（100株単位）
```

- [ ] 発注完了
- [ ] 約定価格を記録: ¥______
- [ ] 約定時刻を記録: ______

---

## 日中ホールド — 任意モニタリング

- [ ] 大きな値動き（±3%超）がないか確認（11:00 / 13:00）
- [ ] 急変時は裁量で早期決済を検討

---

## 15:25-15:29 — 引成決済発注

```
発注種別: 引成 (CLO)
銘柄:     5020.T ENEOS
方向:     Sell（Longの場合）or Buy（Shortの場合）
数量:     保有全量
```

- [ ] 引成発注完了

---

## 15:30 — 約定確認・記録

- [ ] 決済約定価格を確認: ¥______
- [ ] P&L計算
  - Gross P&L (bps): ______
  - コスト 4bps を差引
  - Net P&L (bps): ______
  - Net P&L (円): ¥______
- [ ] trade_log.csv に記入

---

## スキップ理由コード

- `dev_below_threshold`: |dev| < 50bps
- `earnings`: 決算日
- `market_panic`: 市場急変動（±3%超）
- `discretionary`: 手動裁量スキップ
