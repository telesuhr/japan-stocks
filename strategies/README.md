# 採用戦略一覧 (日本株専用 · 6戦略)

実運用ドキュメント。**バックテストで採用判定された戦略のみ**を格納する。
研究・検証段階の分析は [`../analyses/`](../analyses/) を参照。

> **2026-05-18 検証**: JQuants新DBで直近6ヶ月の継続性チェックを実施 ([`analyses/20260518_strategy_current_check/`](../analyses/20260518_strategy_current_check/))。
> 直近6ヶ月で Sharpe < 1.0 に大幅劣化した **oversold_ma25_reversal (-0.03)** / **large_cap_oversold_reversal (+0.13)** / **orb_breakout_long (+0.15)** の3戦略は [`_archive/`](_archive/) に退避し、`analyses/` での再検証フェーズに戻した。
>
> **2026-05-11**: JQuants新DBで5年データ再検証 ([`analyses/20260511_strategy_validation_jquants/`](../analyses/20260511_strategy_validation_jquants/))。**topix_overnight (Sharpe +0.58)** / **pair_portfolio (Sharpe +0.65)** は基準未達で `_archive/` へ。
>
> **2026-05-10**: データソースが Refinitiv → JQuants に統合。LMEや米国指数データは利用不可となり、関連戦略 (lme_on_copper, nonferrous_lme_link, sox_overnight_short, semi_sox_fade) も `_archive/` へ退避済み。
> 詳細: [DATA_SCHEMA.md](../DATA_SCHEMA.md)

**運用方針**: 初期分析は `analyses/` で実施し、その中から実際のトレーディング戦略として使えそうなもの (Sharpe ≥ 2.0 & N ≥ 30 & t-stat ≥ 2.0) を `strategies/` にオペレーショナルドキュメント付きで昇格させる。
**継続採用基準**: 5年検証で Sharpe ≥ 2.0 を維持していること。

---

## 戦略マップ (6戦略)

```
┌──────────────────── スイング戦略 (4) ────────────────────┐
│  bank_absorption (5営業日保有)                          │
│  銀行22銘柄ホワイトリスト                               │
│  シグナル: 出来高≥1.5×平均 + 陰線 + 売買代金≥10億       │
│  Day N+1 09:00 寄成 Long → Day N+5 15:30 引成           │
│  5年Sharpe +3.94 / 直近6M Sharpe +3.95 ✅               │
│                                                          │
│  lasertec_ma25_support (10営業日保有)                   │
│  6920.T レーザーテック / dd20≤-5% + MA25接触            │
│  10日クールダウン適用                                    │
│  5年Sharpe +2.95 / 直近6M Sharpe +6.96 ✅               │
│                                                          │
│  earnings_pead (5営業日保有)                            │
│  良決算+大窓銘柄を翌日寄成 Long                          │
│  5年Sharpe +2.19 / 直近6M Sharpe +2.17 ✅               │
│                                                          │
│  pre_earnings_drift (3-5営業日保有)                     │
│  決算前2〜4日の値動きを Long で取る                      │
│  5年Sharpe +2.07 / 直近6M Sharpe +2.45 ✅               │
└──────────────────────────────────────────────────────────┘

┌──────────────────── イントラ戦略 (2) ────────────────────┐
│  vwap_morning_meanrevert (両方向)                       │
│  TEL/ディスコ/レーザー / 10:00-11:30 VWAP乖離           │
│  |dev|≥275bps → 反転エントリー                          │
│  5年Sharpe +6.76 / 直近6M Sharpe +4.58 ✅               │
│                                                          │
│  eneos_vwap_trend (Long/Short)                          │
│  5020.T ENEOS / 9:30 VWAP乖離 ≥ ±50bps                 │
│  9:31〜成行 → 15:30 引成                                │
│  5年Sharpe +3.81 / 直近6M Sharpe +2.37 ✅               │
└──────────────────────────────────────────────────────────┘
```

---

## 戦略一覧 (6戦略)

| # | フォルダ | 型 | 発動頻度 | 5年Sharpe | 直近6M | N | 想定資金 |
|:-:|---------|:--:|---------|:--:|:--:|:--:|---------|
| 1 | [lasertec_ma25_support](lasertec_ma25_support/) | スイング Long (10営業日) | 月1-2回 | +2.95 | **+6.96** | 5 | ¥500-1,000万 |
| 2 | [vwap_morning_meanrevert](vwap_morning_meanrevert/) | イントラ両方向 | 月2-5回 | **+6.76** | **+4.58** | 24 | ¥900-1,500万 |
| 3 | [bank_absorption](bank_absorption/) | スイング Long (5営業日) | 月10-20回 | +1.84 | **+3.95** | 70 | ¥300万 (¥100万×3銘柄) |
| 4 | [pre_earnings_drift](pre_earnings_drift/) | スイング Long (3-5営業日) | 月50-200回 (決算期集中) | +2.07 | **+2.45** | 1007 | ¥1,500万 (¥100万×15銘柄) |
| 5 | [eneos_vwap_trend](eneos_vwap_trend/) | イントラ両方向 | 月5-6回 | +3.81 | **+2.37** | 43 | ¥1,000-3,000万 |
| 6 | [earnings_pead](earnings_pead/) | スイング Long (5営業日) | 月10-40回 (決算期集中) | +2.19 | **+2.17** | 216 | ¥1,000万 (¥100万×10銘柄) |

**直近6M Sharpe は [analyses/20260518_strategy_current_check/](../analyses/20260518_strategy_current_check/) で算出 (2025-11-15〜2026-05-15)**
**5年Sharpe は [analyses/20260511_strategy_validation_jquants/](../analyses/20260511_strategy_validation_jquants/) のJQuants長期検証値**

**コード状態**: 全6戦略が ✅ 新DB対応済 (`stocks_intraday`/`stocks_daily`/`index_daily` + 5桁code + JST naive)

---

## 同日発動時のルール

### lasertec_ma25_support × vwap_morning_meanrevert (レーザーテック共通)
- 時間軸が完全に独立 (日足スイング vs イントラ日中)
- **両方発動可、独立管理**
- スイング枠 ¥500-1,000万 とイントラ枠は別資金として運用

### lasertec_ma25_support: クールダウン
- 前回エントリーから **10営業日以内は再エントリー禁止**
- 連続シグナルでの重複エントリーを防ぐ (signal_check.py 内蔵)

### bank_absorption
- 銀行セクター22銘柄に閉じているため、他戦略との重複なし
- ただし日次同時保有は最大3銘柄 (vol_ratio降順で選定)

---

## 戦略追加ワークフロー

新戦略を採用する場合の手順:

1. `analyses/` で初期検証 (Sharpe ≥ 2.0 & N ≥ 30 & t-stat ≥ 2.0)
2. **5年検証で Sharpe ≥ 2.0 を維持していることを確認** (1年バイアスを排除)
3. `analyses/<name>_oos/` で Out-of-Sample 検証 (H1/H2 分割等)
4. OoSでも劣化なしを確認 → `strategies/<name>/` ディレクトリを作成
5. **必須**: 新DBテーブル (`stocks_intraday` / `stocks_daily`) + 5桁コードで実装
6. 最低限のファイルを揃える:
   - `README.md` — 戦略概要・バックテスト結果・オペレーション
   - `RULES.md` — 意思決定フローチャート
   - `DAILY_CHECKLIST.md` — 日次チェックリスト
   - `signal_check.py` — 発動判定スクリプト
   - `trade_log_template.csv` — 記録テンプレート
7. 本ファイルの戦略一覧に追記

---

## 廃止戦略 ([_archive/](_archive/))

### カテゴリA: LME/米マクロ依存 (2026-05-10廃止)
| 戦略 | 廃止理由 |
|------|---------|
| lme_on_copper | LME銅取得不可 + 検証で機能停止 (Sharpe -3.14) |
| nonferrous_lme_link | LME依存 |
| sox_overnight_short | 米国指数 (.SOX/ESc1/VXc1) 取得不可 |
| semi_sox_fade | .SOX依存 |

### カテゴリB: 5年検証で機能消失 (2026-05-11廃止)
| 戦略 | 5年Sharpe | 廃止理由 |
|------|:---:|---------|
| topix_overnight | **+0.58** | 1年検証バイアス。5年では機能せず |
| pair_portfolio | **+0.65** | 18ペアEWで勝7敗7相殺。半TEL-レーザー +12.89 等 個別有望ペアあり → v2 検討 |

### カテゴリC: 直近6ヶ月で機能停止 (2026-05-18 退避)
| 戦略 | 5年Base | 直近6M | 退避理由 |
|------|:---:|:---:|---------|
| oversold_ma25_reversal | +3.21 | **-0.03** | MA25-20%乖離戦略。上昇トレンドで過売りが発生せず、シグナル発動するもPF≒1.0勝率49.2%でほぼ機能停止 |
| large_cap_oversold_reversal | +2.92 | **+0.13** | Core30+Large70限定版も同様に機能停止。N=127/WR50.4%/PF1.05 |
| orb_breakout_long | +2.31 | **+0.15** | 三井金属Sharpe-0.28・ディスコ+1.33。3戦略中で最も復活余地あり (ディスコ単独化で再昇格検討) |

詳細は [_archive/README.md](_archive/README.md) と [analyses/20260518_strategy_current_check/](../analyses/20260518_strategy_current_check/) 参照。

---

## 候補戦略 (analyses/ で検証中)

| 候補 | Sharpe | N | 根拠分析 | 課題 |
|------|--------|---|---------|------|
| 9101 日本郵船 VWAP Breakout | +4.18 | 99 | [`vwap_comprehensive/`](../analyses/20260422_vwap_comprehensive/) | 銘柄集中リスク + 5年再検証必要 |
| 1605 INPEX VWAP Trend | +15.56 | — | 同上 | eneos_vwap_trend との相関リスク確認 + 5年再検証必要 |
| **pair_portfolio v2 (勝ち7ペアのみ)** | 推定+5前後 | — | [20260511_strategy_validation_jquants](../analyses/20260511_strategy_validation_jquants/) | 半TEL-レーザー +12.89, 通NTT-KDDI +7.13 など7ペアに絞り込み |

新たに JQuants 固有データ (信用残・空売り比率・投資部門別) を活用した戦略候補も今後 `analyses/` で検討予定。
