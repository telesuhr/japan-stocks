# 採用戦略一覧 (日本株専用 · 5戦略)

実運用ドキュメント。**バックテストで採用判定された戦略のみ**を格納する。
研究・検証段階の分析は [`../analyses/`](../analyses/) を参照。

> **2026-05-11 検証**: JQuants新DBで5年データ再検証を実施 ([`analyses/20260511_strategy_validation_jquants/`](../analyses/20260511_strategy_validation_jquants/))。
> 採用基準 (Sharpe ≥ 2.0) を下回った **topix_overnight (Sharpe +0.58)** と **pair_portfolio (Sharpe +0.65)** は [`_archive/`](_archive/) に退避済み。
>
> **2026-05-10**: データソースが Refinitiv → JQuants に統合。LMEや米国指数データは利用不可となり、関連戦略 (lme_on_copper, nonferrous_lme_link, sox_overnight_short, semi_sox_fade) も `_archive/` へ退避済み。
> 詳細: [DATA_SCHEMA.md](../DATA_SCHEMA.md)

**運用方針**: 初期分析は `analyses/` で実施し、その中から実際のトレーディング戦略として使えそうなもの (Sharpe ≥ 2.0 & N ≥ 30 & t-stat ≥ 2.0) を `strategies/` にオペレーショナルドキュメント付きで昇格させる。
**継続採用基準**: 5年検証で Sharpe ≥ 2.0 を維持していること。

---

## 戦略マップ (5戦略)

```
┌──────────────────── スイング戦略 (2) ────────────────────┐
│  bank_absorption (5営業日保有)                          │
│  銀行22銘柄ホワイトリスト                               │
│  シグナル: 出来高≥1.5×平均 + 陰線 + 売買代金≥10億       │
│  Day N+1 09:00 寄成 Long → Day N+5 15:30 引成           │
│  Sharpe: +3.94 (5年/N=908)                              │
│                                                          │
│  lasertec_ma25_support (10営業日保有)                   │
│  6920.T レーザーテック / dd20≤-5% + MA25接触            │
│  10日クールダウン適用                                    │
│  Sharpe: +2.95 (5年/N=39)                               │
└──────────────────────────────────────────────────────────┘

┌──────────────────── イントラ戦略 (3) ────────────────────┐
│  eneos_vwap_trend (Long/Short)                          │
│  5020.T ENEOS / 9:30 VWAP乖離 ≥ ±50bps                 │
│  9:31〜成行 → 15:30 引成                                │
│  Sharpe: +2.97 (2年/Long+3.97 / Short+2.18)             │
│                                                          │
│  vwap_morning_meanrevert (両方向)                       │
│  ディスコ/レーザー / 10:00-11:30 VWAP乖離               │
│  |dev|≥275bps → 反転エントリー                          │
│  Sharpe: +4.81 (2年/N=77)                               │
│  ※TEL は Sharpe -0.65 で除外推奨                        │
│                                                          │
│  orb_breakout_long (Long専用)                           │
│  ディスコ60分OR / 三井金属30分OR ブレイク               │
│  Sharpe: +2.19 (2年/N=469)                              │
└──────────────────────────────────────────────────────────┘
```

---

## 戦略一覧 (6戦略)

| # | フォルダ | 型 | 発動頻度 | 直近Sharpe | N | 想定資金 |
|:-:|---------|:--:|---------|:--:|:--:|---------|
| 1 | [bank_absorption](bank_absorption/) | スイング Long (5営業日) | 月10-20回 | **+3.94** | 908 | ¥300万 (¥100万×3銘柄) |
| 2 | [vwap_morning_meanrevert](vwap_morning_meanrevert/) | イントラ両方向 | 月2-5回 | **+4.81** | 77 | ¥900-1,500万 |
| 3 | [eneos_vwap_trend](eneos_vwap_trend/) | イントラ両方向 | 月5-6回 | **+2.97** | 97 | ¥1,000-3,000万 |
| 4 | [lasertec_ma25_support](lasertec_ma25_support/) | スイング Long (10営業日) | 月1-2回 | **+2.95** | 39 | ¥500-1,000万 |
| 5 | [orb_breakout_long](orb_breakout_long/) | イントラ Long | 月20-25回 | **+2.19** | 469 | ¥1,000-2,000万 |
| 6 | [earnings_pead](earnings_pead/) | スイング Long (5営業日) | 月10-40回 (決算期集中) | **+2.19** | 1244 | ¥1,000万 (¥100万×10銘柄) |

**Sharpe は [analyses/20260511_strategy_validation_jquants/](../analyses/20260511_strategy_validation_jquants/) の JQuants長期検証値**。
イントラ系は2年検証 (2024-05〜2026-05)、日足系は5年検証 (2021-05〜2026-05)。

**コード状態**: 全5戦略が ✅ 新DB対応済 (`stocks_intraday`/`stocks_daily`/`index_daily` + 5桁code + JST naive)

---

## 同日発動時のルール

### orb_breakout_long × vwap_morning_meanrevert (ディスコ共通)
- 両戦略とも **ディスコ (61460/6146.T)** 監視対象
- 同日両発動時は **vwap_morning_meanrevert を優先** (Sharpe +4.81 > +2.19)
- 三井金属 (57060/5706.T) は ORB 単独発動なので影響なし

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

詳細は [_archive/README.md](_archive/README.md) 参照。

---

## 候補戦略 (analyses/ で検証中)

| 候補 | Sharpe | N | 根拠分析 | 課題 |
|------|--------|---|---------|------|
| 9101 日本郵船 VWAP Breakout | +4.18 | 99 | [`vwap_comprehensive/`](../analyses/20260422_vwap_comprehensive/) | 銘柄集中リスク + 5年再検証必要 |
| 1605 INPEX VWAP Trend | +15.56 | — | 同上 | eneos_vwap_trend との相関リスク確認 + 5年再検証必要 |
| **pair_portfolio v2 (勝ち7ペアのみ)** | 推定+5前後 | — | [20260511_strategy_validation_jquants](../analyses/20260511_strategy_validation_jquants/) | 半TEL-レーザー +12.89, 通NTT-KDDI +7.13 など7ペアに絞り込み |

新たに JQuants 固有データ (信用残・空売り比率・投資部門別) を活用した戦略候補も今後 `analyses/` で検討予定。
