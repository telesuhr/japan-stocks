# 採用戦略一覧 (日本株専用)

実運用ドキュメント。**バックテストで採用判定された戦略のみ**を格納する。
研究・検証段階の分析は [`../analyses/`](../analyses/) を参照。

> **2026-05-10 重要変更**: データソースが Refinitiv → JQuants に統合され、LMEや米国指数データは利用不可となった。
> LME / 米マクロ依存の戦略は [`_archive/`](_archive/) に退避。本一覧は **JQuants日本株データのみで稼働可能な戦略** に絞っている。
> 詳細: [DATA_SCHEMA.md](../DATA_SCHEMA.md)

**運用方針**: 初期分析は `analyses/` で実施し、その中から実際のトレーディング戦略として使えそうなもの (Sharpe ≥ 2.0 & N ≥ 30 & t-stat ≥ 2.0) を `strategies/` にオペレーショナルドキュメント付きで昇格させる。

---

## 戦略マップ (日本株オンリー)

```
┌──────────────────── ON戦略 ────────────────────┐
│  topix_overnight                              │
│  CORE5 (5711/6501/7011/5016/4502)             │
│  シグナル: TOPIX 前日終値→当日9:00 ≥ +0.3%     │
│  Day N 15:30 引成 Long → Day N+1 09:00 寄成   │
│  発動: 月4-5回 / Sharpe: +6.27 (2026-05検証)  │
└────────────────────────────────────────────────┘

┌──────────────────── スイング戦略 ──────────────────┐
│  bank_absorption (5営業日保有)                    │
│  銀行22銘柄ホワイトリスト                         │
│  シグナル: 出来高≥1.5×平均 + 陰線 + 売買代金≥10億 │
│  Day N+1 09:00 寄成 Long → Day N+5 15:30 引成     │
│  Sharpe: +1.84 (test +1.59)                       │
│                                                    │
│  lasertec_ma25_support (10営業日保有)             │
│  6920.T レーザーテック / dd20≤-5% + MA25接触      │
│  ※10日クールダウン適用後 Sharpe +7.57            │
└────────────────────────────────────────────────────┘

┌──────────────────── イントラ戦略 ──────────────────┐
│  eneos_vwap_trend (Long/Short)                    │
│  5020.T ENEOS / 9:30 VWAP乖離 ≥ ±50bps           │
│  9:31〜成行 → 15:30 引成                          │
│  Sharpe: +3.81 (Long +4.73 / Short +3.26)         │
│                                                    │
│  vwap_morning_meanrevert (両方向)                 │
│  TEL/ディスコ/レーザー / 10:00-11:30 VWAP乖離     │
│  |dev|≥275bps → 反転エントリー                    │
│  Sharpe: +6.76                                    │
│                                                    │
│  orb_breakout_long (Long専用)                     │
│  ディスコ60分OR / 三井金属30分OR ブレイク         │
│  Sharpe: +2.31                                    │
└────────────────────────────────────────────────────┘

┌──────────────────── 統計裁定 ──────────────────────┐
│  pair_portfolio (18ペア)                          │
│  日本株18ペア Z-score平均回帰 EWポートフォリオ    │
│  Sharpe: +1.37 (年率)                             │
│  ⚠ コードはMariaDB依存のままJQuants化要書換え     │
└────────────────────────────────────────────────────┘
```

---

## 戦略一覧 (日本株のみ・7戦略)

| # | フォルダ | 型 | 発動頻度 | Sharpe | 想定資金 | コード状態 |
|:-:|---------|:--:|---------|--------|---------|----------|
| 1 | [topix_overnight](topix_overnight/) | ON Long | 月4-5回 | **+6.27** | ¥5,000万 (ON枠) | 🟡 旧テーブル参照 |
| 2 | [eneos_vwap_trend](eneos_vwap_trend/) | イントラ両方向 | 月5-6回 | **+3.81** | ¥1,000-3,000万 | 🟡 旧テーブル参照 |
| 3 | [vwap_morning_meanrevert](vwap_morning_meanrevert/) | イントラ両方向 | 月2-5回 | **+6.76** | ¥900-1,500万 | 🟡 旧テーブル参照 |
| 4 | [orb_breakout_long](orb_breakout_long/) | イントラ Long | 月20-25回 | **+2.31** | ¥1,000-2,000万 | 🟡 旧テーブル参照 |
| 5 | [lasertec_ma25_support](lasertec_ma25_support/) | スイング Long (10営業日) | 月1-2回 | **+7.57** | ¥500-1,000万 | 🟡 旧テーブル参照 |
| 6 | [bank_absorption](bank_absorption/) | スイング Long (5営業日) | 月10-20回 | **+1.84** | ¥300万 (¥100万×3銘柄) | ✅ 新DB対応済 |
| 7 | [pair_portfolio](pair_portfolio/) | 統計裁定 LS | 〜1,400往復/年 | **+1.37** | ¥3,000万 | 🔴 MariaDB依存 (要書換) |

**Sharpeは [analyses/20260509_strategy_validation/](../analyses/20260509_strategy_validation/) の最新検証値**

**コード状態**:
- ✅ 新DB対応済: `stocks_intraday`/`stocks_daily` (5桁code, JST) を使用
- 🟡 旧テーブル参照: `intraday_data`/`daily_stats` (RIC, UTC) のまま → `archive.*` に退避済みなので動かない、書き換え必要
- 🔴 MariaDB依存: NAS MariaDB を直接参照、PostgreSQL 移行が必要

---

## 同日発動時のルール

### topix_overnight (Long) と各イントラ戦略
- ON戦略 × イントラ戦略は **時間帯独立**、ダブルベット対象外
- ただし注意: bank_absorption (Day N+1 寄成 Long) と topix_overnight (Day N+1 09:00 寄成決済) は同じ寄り付きで反対売買が混在する場合あり
  - bank_absorption は買い、topix_overnight は売り (ON決済)
  - 同一時刻なので相殺される可能性あり、約定タイミングに注意

### orb_breakout_long × vwap_morning_meanrevert (ディスコ共通)
- 両戦略とも **ディスコ (6146.T)** 監視対象
- 同日両発動時は **vwap_morning_meanrevert を優先** (Sharpe +6.76 > +2.31)
- 三井金属 (5706.T) は ORB 単独発動なので影響なし

### lasertec_ma25_support × vwap_morning_meanrevert (レーザーテック共通)
- 時間軸が完全に独立 (日足スイング vs イントラ日中)
- **両方発動可、独立管理**
- スイング枠 ¥500-1,000万 とイントラ枠は別資金として運用

### lasertec_ma25_support 内: クールダウン
- 前回エントリーから **10営業日以内は再エントリー禁止**
- 連続シグナルでの重複エントリーを防ぐ (signal_check.py 内蔵)

### pair_portfolio
- 18ペアの中に **6920.T (レーザーテック)** が含まれる場合は lasertec_ma25_support と方向衝突に注意
- 銘柄重複を確認の上、片方を優先 (要追加検討)

---

## 戦略追加ワークフロー

新戦略を採用する場合の手順:

1. `analyses/` で初期検証 (Sharpe ≥ 2.0 & N ≥ 30 & t-stat ≥ 2.0)
2. `analyses/<name>_oos/` で Out-of-Sample 検証 (H1/H2 分割等)
3. OoSでも劣化なしを確認 → `strategies/<name>/` ディレクトリを作成
4. **必須**: 新DBテーブル (`stocks_intraday` / `stocks_daily`) + 5桁コードで実装
5. 最低限のファイルを揃える:
   - `README.md` — 戦略概要・バックテスト結果・オペレーション
   - `RULES.md` — 意思決定フローチャート
   - `DAILY_CHECKLIST.md` — 日次チェックリスト
   - `signal_check.py` — 発動判定スクリプト
   - `trade_log_template.csv` — 記録テンプレート
6. 本ファイルの戦略一覧に追記

---

## 廃止戦略 ([_archive/](_archive/))

| 戦略 | 廃止理由 |
|------|---------|
| lme_on_copper | LME銅取得不可 + 2026-05-09検証で機能停止 (Sharpe -3.14) |
| nonferrous_lme_link | LME依存 |
| sox_overnight_short | 米国指数 (.SOX/ESc1/VXc1) 取得不可 |
| semi_sox_fade | .SOX依存 |

詳細は [_archive/README.md](_archive/README.md) 参照。

---

## 既存戦略コードの書き換え TODO

### 優先度: 🔴 高 (今月中)

新DBで動かすため、各 `signal_check.py` を以下の方針で書き換える:

```python
# 旧 (RIC + UTC)
PG_CONFIG = {...}
sql = "SELECT timestamp, close FROM intraday_data WHERE symbol = '6920.T' ..."
# timestamp = UTC、JST は +9h

# 新 (5桁code + JST直接)
sql = "SELECT ts, close FROM stocks_intraday WHERE code = '69200' ..."
# ts = JST 直接、変換不要
```

書換え対象 (5戦略):
- topix_overnight, eneos_vwap_trend, vwap_morning_meanrevert, orb_breakout_long, lasertec_ma25_support

### 優先度: 🔴 緊急 (今月中)
- pair_portfolio: NAS MariaDB → PostgreSQL `stocks_daily` への完全書き換え

---

## 候補戦略 (analyses/ で検証中)

| 候補 | Sharpe | N | 根拠分析 | 課題 |
|------|--------|---|---------|------|
| 9101 日本郵船 VWAP Breakout | +4.18 | 99 | [`vwap_comprehensive/`](../analyses/20260422_vwap_comprehensive/) | 銘柄集中リスク |
| 1605 INPEX VWAP Trend | +15.56 | — | 同上 | eneos_vwap_trend との相関リスク確認 |

新たに JQuants 固有データ (信用残・空売り比率・投資部門別) を活用した戦略候補は今後 `analyses/` で検討予定。
