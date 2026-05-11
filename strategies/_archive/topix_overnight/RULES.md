# 意思決定フローチャート — TOPIX夜間ONホールド戦略

---

## 🟢 Day N 09:05 — トレードするか?

```
START (Day N 09:05)
  │
  ▼
① lme_on_copper シグナルが本日発動予定か?
   (LME 東京セッション 9:00 開始直後の動きから推測)
  │
  ├── Yes → 本戦略はスキップ (重複回避 skipped_reason=lme_on_copper_priority)
  │
  └── No/未定 ▼

② TOPIX 夜間変化率 (前日終値→当日09:00) ≥ +0.3% か?
  │
  ├── No  → スキップ (skipped_reason=topix_below_threshold)
  │
  └── Yes ▼

③ Day N は木曜日か?
  │
  ├── Yes → スキップ (skipped_reason=thursday)
  │
  └── No  ▼

④ Day N / Day N+1 に FOMC/米雇用統計/日銀会合が控えているか?
  │
  ├── Yes → サイズ半減 (各¥500万) or スキップ
  │
  └── No  ▼

⑤ 決算発表日該当銘柄があるか?
  │
  ├── Yes → その銘柄のみ除外
  │         除外後 < 3 銘柄 → 全体スキップ
  │
  └── No  ▼

⑥ Day N+1 配当落ち該当銘柄があるか?
  │
  ├── Yes → その銘柄のみ除外
  │
  └── No  ▼

⑦ 日経225先物・CMEが急落中 (-2%超) か?
  │
  ├── Yes → スキップ (skipped_reason=panic)
  │
  └── No  ▼

⚠️ Day N 15:15 — lme_on_copper シグナル最終確認
  │
  ├── lme_on_copper 発動確定 → 本戦略は取消 (lme_on_copper優先)
  │
  └── 未発動 ▼

✅ Day N 15:27-15:29: 5銘柄 引成 Long 発注
   Day N 15:30 大引け: 約定
```

---

## 🌙 Day N 夜間

lme_on_copper と同じ — 通常放置、-3%超時のみ翌朝検討。

---

## 🌅 Day N+1 08:45 — 寄付前

```
⑧ 個別銘柄に重大ニュース (不祥事等) があるか?
  │
  ├── Yes → 該当銘柄は指値で対応
  │
  └── No  ▼

✅ Day N+1 08:55-08:59: 5銘柄 寄成 Sell 発注
   Day N+1 09:00 寄付: 約定 → P&L確定
END
```

---

## 📊 ポジションサイジング

lme_on_copper と同じルール。

```
基準: 1銘柄 = ¥1,000万, バスケット合計 = ¥5,000万
```

---

## 🔒 リスク上限

- 1トレード: Gross -2% (-¥100万) 超で翌寄付即決済
- 月次: Net -¥200万 到達で月内新規停止
- 年次: Net -¥500万 到達で戦略停止

---

## 🔁 継続運用判断

| 指標 | 良好 | 注意 | 撤退 |
|---|---|---|---|
| 勝率 (直近3ヶ月) | ≥60% | 50-60% | <50% |
| 平均リターン/トレード | ≥+30bps | +15〜30 | <+15 |
| Sharpe (直近12トレード) | ≥+3 | +2〜3 | <+2 |
| 実現 MaxDD | <-3% | -3〜-5% | <-5% |

---

## 🔗 lme_on_copper との併用ルール

| 状況 | 対応 |
|---|---|
| 両方発動 | lme_on_copper のみ実行 (Sharpeが高い) |
| lme_on_copper のみ | lme_on_copper 実行 |
| topix のみ | topix_overnight 実行 |
| 両方不発 | ノーポジ |

**ダブルベット禁止** (同じバスケットに重ねない)

---

## 💡 裁量介入ルール

lme_on_copper と同じ。年間5回以内の裁量スキップ許容。

---

## 📝 記録必須事項

- signal_fired (Yes/No), skipped_reason (該当時)
- topix_prev_close, topix_open_day_n, topix_gap_pct
- 5銘柄 entry/exit/ret_bps
- 手数料, Gross/Net P&L
- lme_on_copper同日シグナル有無 (competing_signal列)
