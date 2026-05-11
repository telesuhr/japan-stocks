# 📋 日次オペレーションチェックリスト — .SOX急落 TOPIX日中Short戦略

所要時間: 朝 07:00-08:55 に15分 + Day N 15:20-15:35 に10分。

---

## 🌅 Day N 07:00-08:30 — 前日の米国市場確認

### 1. 米国指数 日次リターン記録

- [ ] Bloomberg/investing.com/TradingView で前日終値を確認

```
.SOX (Philadelphia Semiconductor):  前日 Ret = ______ %
ESc1 (S&P500 futures):              前日 Ret = ______ %
NQc1 (Nasdaq futures):              前日 Ret = ______ %
VXc1 / ^VIX:                        終値    = ______
```

### 2. シグナル判定

- **.SOX ≤ -2.0%** → 発動候補 ▶ 次へ
- **.SOX > -2.0%** → スキップ (signal_not_fired)

### 3. 補助条件チェック

- [ ] ESc1 ≤ -1.0% か? (推奨AND条件)
  - Yes → 強いシグナル (Sharpe+2.83)
  - No → サイズ半減 or 見送り
- [ ] VIX 終値
  - VIX < 15 → スキップ (低ボラは不発)
  - VIX 15-35 → OK
  - VIX ≥ 35 → スキップ (パニックは反発リスク)

補助スクリプト:
```bash
cd ~/claude-code/japan-stocks/strategies/sox_overnight_short
python3 signal_check.py
```

---

## 🔍 Day N 08:30-08:55 — 寄付直前確認

### 4. 市場環境

- [ ] 日経225先物 (大阪夜間、JST 08:45頃) の現在値
  - 通常範囲 → OK
  - 既に -3% 超 → サイズ半減 (底抜け中のShort は危険)
- [ ] CME N225 先物 
- [ ] USD/JPY: 急変動なし
- [ ] 日銀報道で緊急声明なし

### 5. 曜日チェック

- [ ] Day N 曜日: ______ 
  - **火曜日 → スキップ推奨** (バックテストで劣化)

### 6. 除外条件

- [ ] topix_overnight (Long) シグナル発動確認
  - 発動している場合 → Long 側キャンセル、SOX Short を優先実行
- [ ] FOMC/日銀会合/日本重要指標 (GDP/CPI等) 当日発表なし
  - あり → サイズ半減 or 決定前決済
- [ ] 1306.T 配当落ち日でない (3月末・9月末に注意)

---

## 🔔 Day N 08:57-08:59 — 寄成Short発注

### 7. 株数計算

```
想定資金: ¥______万 (初期 ¥1,000万推奨)
前日 1306 終値: ¥______
株数 = 資金 / 前日終値 (100株単位で切り下げ)
株数: ______株
```

### 8. 寄成 (OPG) 売建て発注

```
取引区分: 信用売り (新規) or 現物売建
銘柄: 1306.T (NEXT FUNDS TOPIX連動型ETF)
数量: [株数]
価格: 成行
執行条件: 寄成 (OPG)
```

- [ ] 発注完了
- [ ] 09:00 約定価格を記録: ¥______

---

## 📈 Day N 09:00-15:25 — 保有中モニタリング

- [ ] 日中 +1.5% 超の反発 → 部分決済検討
- [ ] 米先物急反発 (+2%超) → 早期決済検討
- [ ] 日銀緊急発表 (ETF買入再開等) → 即時成行決済
- [ ] 大幅逆行時の最大含み損幅を記録

---

## 🔔 Day N 15:27-15:29 — 引成決済発注

```
取引区分: 信用買戻し (決済) or 現物買戻
価格: 成行
執行条件: 引成 (CLO)
```

- [ ] 発注完了
- [ ] 15:30 約定価格を記録: ¥______

---

## 💰 Day N 15:30-15:40 — P&L確定

```
Gross P&L = (entry_price - exit_price) × 株数
手数料 = ¥______ (Short側2bps + Cover側2bps = 4bps ≒ ¥4,000 @¥1,000万)
Net P&L = ¥______
リターン (bps) = Net P&L / 資金 × 10000 = ______ bps
```

- [ ] trade_log_template.csv に記録
- [ ] 日中最大逆行幅 (adverse_excursion) を記録

---

## 🚫 スキップ理由コード

- `signal_not_fired`: .SOX > -2%
- `es_not_confirmed`: .SOX OK だが ESc1 > -1% で AND条件不発
- `vix_too_low`: VIX < 15
- `vix_panic`: VIX ≥ 35 (反発リスク)
- `tuesday_entry`: 火曜エントリー除外
- `jp_futures_crash`: 日経先物既に -3% 超
- `topix_overnight_conflict`: (本来は SOX Short優先なので基本発生しない)
- `macro_event`: FOMC/日銀会合直前
- `dividend_day`: 1306 配当落ち日
- `system_failure`: 証券会社システム障害
- `discretionary`: 手動裁量スキップ
