# イントラデイ・エントリー方向の定量化フレームワーク

**目的**: "Long か Short か" をその場の勘でなく、**事前に定式化した数値シグナル**で決める体系を構築する。
**対象期間**: 2025-04 〜 2026-04 (258営業日), 非鉄3 + 半導体5 = 8銘柄
**動機**: これまでの分析で「シグナルはあるが方向が銘柄依存」と判明。方向決定を**属人化しない**のが目標。

---

## 0. 問題の分解

エントリー方向を決める問題は、以下 **3つの独立した意思決定** に分解できる。

```
(A) そもそも今日エントリーすべきか?          ← トリガー判定
(B) エントリーする場合、方向は Long/Short?   ← 方向判定 ← 本題
(C) サイズと決済水準は?                      ← リスク管理
```

本書は **(B) の方向判定** に焦点を当て、利用可能なシグナルを網羅的に列挙・比較する。

---

## 1. 方向シグナルの分類 (11カテゴリ)

### カテゴリ別の整理

| # | カテゴリ | 参照データ | 確定時刻 | 予測力の性質 |
|:-:|---|---|---|---|
| 1 | **静的特性** | 過去1年の日足 | 事前 | ベースレート (unconditional) |
| 2 | **レジームフィルター** | マクロ | 事前/当日朝 | 戦争・FOMC等の期間除外 |
| 3 | **オーバーナイトマクロ** | LME/先物/FX/原油 | JST 9:00前 | 強い方向性、本プロジェクト確立済み |
| 4 | **プリオープン** | PTS/板寄せ | 8:59 | 情報漏洩の織り込み |
| 5 | **寄付ギャップ** | 前日終値→今日始値 | 9:00 | サイズ依存 |
| 6 | **寄付レンジ動学** | 9:00-9:30 OHLCV | 9:30 | 銘柄特性で方向決定 |
| 7 | **クロスセクション** | 同セクター他銘柄 | 随時 | ペア乖離、LS |
| 8 | **テクニカル** | 過去足の派生指標 | 随時 | VWAP/MA/BB/RSI |
| 9 | **マイクロ構造** | 約定歩み値・板 | 随時 | ティック非対称、OF |
| 10 | **カレンダー/イベント** | 曜日/月末/決算 | 事前 | 曜日アノマリー等 |
| 11 | **アンサンブル/ML** | 上記の合成 | 随時 | ロジスティック/GBT |

以下、各カテゴリを「何を測るか」「どう定式化するか」「このプロジェクトでの所見」で深掘り。

---

### カテゴリ 1: 静的特性 (Unconditional Bias)

**概念**: 何もシグナルがなくても、銘柄固有に Long/Short どちらが勝ちやすいかのベースレート。

| 指標 | 定義 | 使い方 |
|---|---|---|
| `EOD_drift` | 過去1年の 9:00→15:30 平均リターン (bps) | これが+20bps以上なら**デフォルトLong** |
| `skew_daily` | 日次リターン分布の歪度 | 負の歪度 = 急落リスク大 → Longバイアスを減らす |
| `overnight_drift` | 前日15:30→当日9:00 平均 | ギャップ期待値 |
| `intraday_drift` | 9:00→15:30 のみ | 日中バイアス (本命) |

**本プロジェクトの観測**:
- 8銘柄すべて intraday_drift はプラス (+7〜+47bps)
- ただし **WR=50%前後** → 平均値は少数の勝ち日が牽引 → "とりあえずLong"は非効率

**使いどころ**: タイブレーカー・最終判断のプライア。単独では弱い。

---

### カテゴリ 2: レジームフィルター

**概念**: 「そもそもエントリーしない日」を先に除外し、残った日でのみ方向判定する。

| フィルター | ルール | 本プロジェクトの根拠 |
|---|---|---|
| 戦争期間除外 | 2026-02-28〜2026-03-13 のエントリー禁止 | dayofweek_iran2026: -792bps |
| 木曜除外 | 曜日 == Thu なら順張りLong禁止 | dayofweek_lme: Thu Sharpe -1.31 |
| FOMC/日銀会合 | 政策決定当日は禁止 | マクロイベントはノイズ極大 |
| 決算前日/当日 | 個別要因優位で方向予測困難 | (要実装) 決算カレンダー参照 |
| 半日立会 | 年末・祝前日など | 流動性異常 |

**効果**: 勝率を数%押し上げる「下駄」。方向判定とは直交するため、併用必須。

---

### カテゴリ 3: オーバーナイトマクロ (✅ 本プロジェクト確立済み)

**概念**: JST 9:00の寄付前に**確定している**海外市場の変化率で方向決定。

| シグナル | 確定時刻 | 対象銘柄 | 既知の効果 |
|---|---|---|---|
| LME銅 ON変化率 | JST 3:00頃 (冬)/4:00 (BST) | 非鉄, コア5 | **+1% → Long Sharpe+11.5** |
| LME累積 LB=10 | 同上 | コア5 | +3% → Long Sharpe+7.5 |
| 日経先物 夜間 | JST 6:00 | コア5, 半導体 | +0.3% → Sharpe+4.7 |
| TOPIX先物 夜間 | JST 6:00 | 全般 | +0.3% → Sharpe+4.8 |
| WTI原油 ON | JST 6:00 | エネルギー | +0.5% → Sharpe+2.7 |
| Brent ON | JST 6:00 | 海運 | +2% → Sharpe+3.6 |
| USDJPY ON変化 | JST 6:00 | 輸出株(半導体) | **要検証** |
| SOX指数 US close | JST 6:00 | 半導体 | **要検証** — 日本の半導体はSOX追随性高いはず |
| S&P500 US close | JST 6:00 | 全般 | **要検証** |

**方向判定ルール (定式化)**:
```python
def direction_from_overnight(sym, signals):
    # 1. LMEがプライマリ (非鉄/コア5)
    if sym in NONFERROUS + CORE5:
        if signals['lme_on'] >= 0.01: return 'long', priority=1
        if signals['lme_on'] <= -0.01: return 'short', priority=1
    # 2. セクター別フォロー
    if sym in SEMICON and signals['sox_on'] >= 0.005:
        return 'long', priority=2
    if sym in ENERGY and signals['wti_on'] >= 0.005:
        return 'long', priority=2
    # 3. 市場全体
    if signals['topix_fut_on'] >= 0.003: return 'long', priority=3
    return None  # シグナルなし、他カテゴリへ
```

**メリット**: 確立済み。最強エッジ。
**弱点**: シグナル発生が月1-2回 → 残り95%の日はどうする?

---

### カテゴリ 4: プリオープン (板寄せ)

**概念**: 8:00〜9:00の板寄せ気配値で、機関の需要超過 or 供給超過が見える。

| 指標 | 計算 |
|---|---|
| `preopen_bias_bps` | 寄前最終気配 / 前日終値 - 1 |
| `preopen_volume_imbalance` | 指値買数量 vs 売数量の比率 (取得可能なら) |
| `preopen_range_vs_prev_range` | 気配値変動幅 / 前日日中レンジ |

**このプロジェクトでの限界**:
- PostgreSQL には板寄せデータは保存されていない (1分足のみ)
- Refinitiv では LCMM 気配値を別途取得可能
- 9:00寄付値 ≈ 板寄せ結果 なので、寄付値を「プリオープンシグナル」として扱うことは可能

**代替**: **寄付ギャップ (カテゴリ5)** がプリオープン情報の集約値。

---

### カテゴリ 5: 寄付ギャップ

**概念**: 前日終値 → 今日始値の変化を、**サイズ別 × 方向** で評価。

| 指標 | 定義 | 方向判定への寄与 |
|---|---|---|
| `gap_pct` | (open/prev_close - 1) × 100 | シグナル本体 |
| `gap_z` | gap_pct / 直近60日ギャップstd | 標準化 |
| `gap_vs_fut` | gap_pct - 日経先物夜間変化率 | 個別株超過ギャップ |

**ギャップと日中方向の経験則** (本プロジェクトの観測):

| ギャップ方向 | サイズ | 非鉄 | 半導体 |
|---|---|---|---|
| Up-gap | 小 (0.3〜1%) | フィル率46% → 五分 | フィル率43% → 持続 |
| Up-gap | 大 (>2%) | 未検証 | 未検証 |
| Down-gap | 小 | フィル率54% → **Long有利** | フィル率49% → 五分 |
| Down-gap | 大 (>-2%) | 未検証 | 6963は59% → Long有利 |

**定式化**:
```python
def direction_from_gap(sym, gap_pct, gap_z):
    if sym in NONFERROUS and gap_pct < -0.5:
        return 'long'  # down-gap fill bias
    if sym == '6963.T' and gap_pct < -0.5:
        return 'long'  # fill率59%
    if abs(gap_z) > 2.0:
        # 大ギャップは個別要因 → 様子見推奨
        return None
    ...
```

**TODO 検証**: gap_pct × gap_z × sector のクロス集計で、WRのコントラストマップを作る。

---

### カテゴリ 6: 寄付レンジ動学 (OR Dynamics)

**概念**: 9:00-9:30(orそれ以降)の挙動から、その日のキャラクターを推定。

| 指標 | 定義 | 予測対象 |
|---|---|---|
| `first_N_ret_bps` | 9:00-9:Nmin のリターン | 残り時間の方向 |
| `or_range_bps` | (OR_high - OR_low) / OR_open | ボラ水準 (サイズ調整に) |
| `or_vol_ratio` | 当日OR出来高 / 20日中央値 | ブレイク信頼度 |
| `or_position` | (first_N_close - OR_low) / (OR_high - OR_low) | レンジ内の相対位置 |
| `or_skew_time` | 高値到達分/OR_minutes と安値到達分/OR_minutes の差 | 方向性の強さ |

**本プロジェクトの観測** (20260422_*_intraday_patterns):

| 銘柄 | first30 特性 | vol_ratio >=1.3 適用後 |
|---|---|---|
| 5711 | 弱モメンタム | **Long Sharpe+5.83 (t=+2.30)** |
| 5706 | リバーサル | Long (vr=2.0) Sharpe+7.58 |
| 5713 | 横ばい | エッジなし |
| 8035 | 弱モメンタム | Long Sharpe+2.54 |
| 6857 | **双方向+** | Long (vr=1.5) Sharpe+7.56 |
| 6146 | **午後リバーサル** | **Short** Sharpe+10.18 |
| 4063 | **強モメンタム** | **Long Sharpe+6.04 (t=+2.01)** |
| 6963 | 弱リバーサル | Short Sharpe+2.56 |

**定式化 (銘柄別方向マップ)**:
```python
# 既に確立済み: ORB volume 分析の結果をハードコード
FIRST30_DIR_MAP = {
    '5711.T': ('long',  +50),   # first30 > +50 → Long
    '5706.T': ('long',  -50),   # first30 < -50 → Long (reversal)
    '6857.T': ('long',  -50),   # first30 < -50 → Long (WR 65%!)
    '4063.T': ('long',  +50),   # first30 > +50 → Long (momentum)
    '6146.T': ('short', +50),   # first30 > +50 → Short (reversal)
    '6963.T': ('short', +50),   # first30 > +50 → Short (reversal)
    # 5713, 8035 はシグナル弱
}
```

**これが現時点で最も再現性の高い方向シグナル**。カテゴリ3 (マクロ) と独立して機能する。

---

### カテゴリ 7: クロスセクション

**概念**: 銘柄単独ではなく、同セクター内での**相対位置**で方向を決める。

| 指標 | 計算 | 方向判定 |
|---|---|---|
| `sector_rank_first30` | セクター内 first30リターン順位 | 最弱Short / 最強Long or 逆 |
| `pair_zscore` | (ret_A - ret_B) の rolling Z | |Z|>2 で回帰方向 |
| `sector_mean_deviation` | sym_ret - sector_mean | 乖離の大きい銘柄の収れん方向 |
| `spread_vs_historical` | LS スプレッドの位置 (0-100 percentile) | extreme なら逆張り |

**本プロジェクトの観測 (`nonfer_ls`)**:
- **寄付ダイバージェンス LS**: 9:00 ON差 >=1.0% → gainer Short / laggard Long → 15:30決済
- Z-score 平均回帰はすべて Sharpe マイナス (失敗)
- → **ON時点の差**は有効、**intradayの累積差**は有効でない

**定式化**:
```python
def pair_divergence_direction(sym_a, sym_b, on_ret_a, on_ret_b, threshold=0.01):
    diff = on_ret_a - on_ret_b
    if diff >= threshold:   # a が b より強くON した
        return {'sym_a': 'short', 'sym_b': 'long'}
    if diff <= -threshold:
        return {'sym_a': 'long',  'sym_b': 'short'}
    return None
```

**応用**: 半導体5銘柄でも同じロジックが通用するか要検証。6857×6146 or 4063×6963 などが候補。

---

### カテゴリ 8: テクニカル指標

**概念**: 過去足から派生した古典的指標。イントラデイでは短期パラメータが使われる。

| 指標 | 計算 | 方向シグナル |
|---|---|---|
| `ret_above_vwap` | close > VWAP かどうか | Long傾向 |
| `vwap_dev_bps` | (close - VWAP) / VWAP × 10000 | -100bps超で逆張りLong |
| `ema_fast_slow` | EMA(5m) vs EMA(20m) | クロス方向 |
| `rsi_14_5min` | 14期間RSI on 5分足 | >70=逆張りShort, <30=逆張りLong |
| `bb_position` | (close - MA20)/(2*std20) | ±2σで逆張り |
| `macd_signal` | MACDヒストグラム符号 | 順張り |

**本プロジェクトの先行研究**:
- LME銅の5分足テクニカル (lme-trading/06) で **VWAP Reversion のみ** Sharpe+2.0、他は全滅
- 日本株でも同様の結果が出る可能性が高い (イントラデイアルファは薄い)

**定式化**:
```python
# VWAP Reversionのみ採用
def direction_from_vwap(close, vwap, threshold_bps=100):
    dev = (close / vwap - 1) * 10000
    if dev <= -threshold_bps: return 'long'   # 下に乖離 → 戻り期待
    if dev >= +threshold_bps: return 'short'
    return None
```

**注意**: 単独では Sharpe 2.0前後。**フィルターとしての使用**が適切 (他シグナルのbooster)。

---

### カテゴリ 9: マイクロ構造

**概念**: 1分足より細かい約定データ (ティック) / 板情報から方向バイアスを読む。

| 指標 | 定義 |
|---|---|
| `uptick_ratio` | 上昇ティック数 / 全ティック数 (直近N分) |
| `order_flow_imbalance` | (買出来高 - 売出来高) / 総出来高 |
| `buy_volume_bias` | 大口ブロック約定の方向 |
| `spread_widening` | bid-ask スプレッドの拡大 → 流動性低下 = 方向性強化 |

**本プロジェクトの制約**:
- ティックデータは保存されていない (1分OHLCVのみ)
- 1分足の `close > open` を uptick 代理 にはできるが粒度が粗い

**代替指標 (1分足から可能)**:
```python
# 過去N分の up-bar 比率
up_bar_ratio = (df['close'] > df['open']).rolling(N).mean()
# ≥0.65 で強いLong地合い、≤0.35 で強いShort
```

**優先度**: 低 (データ制約)。将来的に歩み値を取得する場合に再訪。

---

### カテゴリ 10: カレンダー/イベント

**概念**: 時間的構造から生じるバイアス。独立だが弱いシグナル。

| ファクター | 効果 | 対応 |
|---|---|---|
| 曜日 | 木曜 Sharpe-1.31, 火曜+13 | 木曜Longを除外 |
| 月末/月初 | リバランス買い/新規資金流入 | 月末1-2日はLong傾向 |
| メジャーSQ週 | 先物限月終了の週 | ボラ増、方向性低下 |
| 配当落ち日 | 機械的下落 | 直前/当日除外 |
| 日銀会合当日 | 正午決定でジャンプ | 後場禁止 |
| 権利付き最終日 | クロス取引 | 大引け前活発 |

**定式化**:
```python
def calendar_veto(date, sym):
    wd = date.weekday()
    if wd == 3: return 'no_long'   # 木曜ロング禁止
    if is_bojmeeting(date): return 'no_afternoon'
    if is_ex_dividend(date, sym): return 'no_entry'
    if is_sq_week(date): return 'reduce_size'
    return None
```

---

### カテゴリ 11: アンサンブル / ML

**概念**: 上記 1-10 の特徴量を合成して、**Long確率 P(up)** を推定する。

#### 11.1 ロジスティック回帰 (推奨スタート地点)

**目的変数**: `y = 1 if 9:30→15:30 ret > 0 else 0`
**特徴量** (例):
- `gap_pct` (カテゴリ5)
- `first30_ret_bps` (カテゴリ6)
- `or_vol_ratio` (カテゴリ6)
- `lme_on_pct` (カテゴリ3)
- `topix_fut_on` (カテゴリ3)
- `sector_rank_first30` (カテゴリ7)
- `vwap_dev_930` (カテゴリ8)
- `weekday` one-hot (カテゴリ10)
- `sym` one-hot × 上記との交互作用

**学習**:
- Walk-forward: 9か月train / 3か月test を月単位でロール
- Regularization: L2 (Ridge) で過学習抑制
- Class balance: up/down 約50/50 なのでそのまま

**出力**: `P(up | features)` ∈ [0,1]
- P > 0.6 → Long
- P < 0.4 → Short
- 0.4 ≤ P ≤ 0.6 → 見送り

**期待効果**: 単一シグナル Sharpe 2-6 → 合成で Sharpe 4-8 程度 (独立性次第)

#### 11.2 グラディエントブースティング (LightGBM)

**特徴**: 非線形、交互作用を自動検出
**リスク**: イントラデイ N=258日 × 8銘柄 = 2000サンプル → **過学習しやすい**
**対策**:
- max_depth=3, n_estimators=100 に厳しく制約
- SHAP で解釈性確認
- Bagging (50% 再サンプリング) で分散低減

#### 11.3 ニューラルネット

**非推奨**: データ量2000サンプルに対して過剰。ロジスティック/GBTで十分。

---

## 2. シグナル統合の設計パターン

### パターンA: 優先度ベース (現実的スタート地点)

```
IF overnight_macro_signal STRONG:
    direction = overnight_macro_direction   # 最優先
ELIF first30_signal STRONG:
    direction = first30_direction           # 次点
ELIF gap_fill_signal:
    direction = gap_fill_direction
ELIF pair_divergence_signal:
    direction = pair_direction
ELSE:
    no_trade
```

**長所**: 解釈可能、デバッグ容易
**短所**: シグナル間の相互作用を無視、閾値感応度高い

### パターンB: 加重投票

各カテゴリが -1 (Short) / 0 / +1 (Long) を投票、重み付け和の符号で決定。

```python
score = (
    1.5 * sign(overnight_macro) +
    1.0 * sign(first30_direction_mapped) +
    0.5 * sign(gap_fill) +
    0.5 * sign(pair_div) +
    0.3 * sign(vwap_dev)
)
if score >= 1.0: direction = 'long'
elif score <= -1.0: direction = 'short'
```

**長所**: 簡潔、重みは分析的に設定可能
**短所**: 重みの設定が恣意的

### パターンC: ロジスティック回帰 (✅ 推奨本命)

全特徴量を線形合成、係数は walk-forward で学習。

**長所**: 最適な重みを自動で求まる、確率出力で size scaling 可能
**短所**: 解釈にひと手間、実装量多め

### パターンD: 階層ベイズ

銘柄ごとのパラメータを階層的にプール。N=258の問題を緩和。

**長所**: 統計的に正しい、銘柄間で情報共有
**短所**: 実装コスト高、収束確認が必要

---

## 3. 評価フレームワーク (厳格)

### 3.1 Cross-Validation 設計

```
[2025-04 ------ 2026-03] = training window (12か月)
         ↓ walk-forward
[2025-07 ------ 2026-03][2026-04]  = test (1か月)
         ↓
[2025-08 ------ 2026-04][2026-05]  = next
```

- **必須**: 時系列ブロック分割、シャッフル禁止
- **推奨**: Purged CV (訓練と検証の間にembargo 5日を入れて leak 防止)

### 3.2 評価指標

| 指標 | 閾値 |
|---|---|
| Sharpe | >= 2.0 (cost後) |
| t-stat | >= 2.0 |
| N/年 | >= 30 |
| MaxDD | >= -3% (資産比) |
| **OOS劣化率** | **in-sample Sharpe の 70% 以上を OOS で保つ** |
| 日次PNLの分布 | 歪度 > -1, 尖度 < 10 |

### 3.3 過学習検出

1. **パラメータ感応度**: 閾値を ±10% ずらしても Sharpe が 80% 維持されるか
2. **銘柄ランダム化**: 銘柄ラベルをシャッフルして学習 → 元のSharpeより有意に低いか (未達ならリーク疑惑)
3. **レジーム別**: 戦争前/戦争中/戦争後で分割し、各期間で Sharpe > 0 か

### 3.4 特徴量の貢献度分解

- ロジスティック: 標準化係数
- GBT: SHAP values
- どの特徴量が Sharpe をどれだけ押し上げているかを**常に把握する**

---

## 4. 具体的な実装ロードマップ

### Phase 1: シグナル辞書の構築 (基礎固め)

```python
# 各シグナルを統一IF で実装
def signal_overnight_lme(date, sym) -> (direction, strength) | None: ...
def signal_first30(date, sym) -> (direction, strength) | None: ...
def signal_gap_fill(date, sym) -> (direction, strength) | None: ...
def signal_pair_divergence(date, sym) -> (direction, strength) | None: ...
def signal_vwap_reversion(timestamp, sym) -> (direction, strength) | None: ...
```

- 入力は**その時点までに確定している情報のみ** (look-ahead厳禁)
- 出力は (direction ∈ {-1, +1}, strength ∈ [0, 1])
- 発火しない場合は None

### Phase 2: 個別シグナル検証

各シグナル単独で、8銘柄 × 258日をバックテスト。
- Sharpe, WR, N, 銘柄別内訳
- 既存結果 (LME Sharpe+11.5 等) と一致することを確認 (回帰テスト)

### Phase 3: 独立性確認

- シグナル同士のピアソン相関行列
- 発火日の重複率
- 互いに独立なシグナルを優先的にアンサンブル

### Phase 4: アンサンブル構築

- **MVP**: パターンA (優先度ベース) で3-5シグナル組み合わせ
- **次版**: パターンC (ロジスティック) で確率出力化

### Phase 5: Walk-Forward OOS 検証

- 6か月 in-sample → 1か月 OOS を12回ロール
- 各OOS月の Sharpe 分布を可視化
- 月次 Sharpe の IQR が [0.5, 4.0] に収まるか

### Phase 6: 実運用

- 日次ジョブ: 朝8:30 に当日のシグナルを計算し、推奨ポジションを Slack/メール通知
- 発注は手動 (マニュアル発注前提)
- 約定後の実績を毎日記録、想定値との乖離をモニター

---

## 5. よくある落とし穴

### 5.1 リーク

- `rolling(N).mean()` が当日値を含む → 必ず `.shift(1)` か、**その時刻までのデータのみ**で計算
- 9:30の first30 シグナルで、9:30 の close 値を使っているか、9:29 までか?
  - 1分足の9:30バーは「9:30:00〜9:30:59」なので 9:30 の close は 9:30:59 の値 = 使って良い

### 5.2 生存バイアス

- 期中上場 (6963 ローム 2025-03-25) や 銘柄入れ替えの扱い
- 現状8銘柄は全期間で存在するため問題なし、だが今後拡張時に注意

### 5.3 取引コスト過小評価

- 現在 4bps (片側2bps × 往復)
- **現実**: 滑り 1-2bps + 金利 (short側) 1bps → 実質 6-8bps
- **バックテストは 8bps で再評価**して耐性を見る

### 5.4 重複シグナル

- LME+1% の日は先物夜間+0.3% も発生しやすい → 相関ほぼ1
- 合算すると"2倍効く"と誤認
- **直交化** (例: 先物夜間 - LME残差) をして独立成分に分ける

### 5.5 目的変数の選び方

- `sign(EOD return)` だけでなく、`sign(VWAP再帰後 return)` や `sign(ATR超過)` なども候補
- **"儲かったか"ではなく"方向が合ったか"** を予測するのが本質

---

## 6. 最終推奨: 段階的実装プラン

### Step 1 (今すぐ): 現行シグナルを統一IF に揃える
既存分析 (LME/先物/first30/gap) を `signals/` モジュールにまとめる。
所要: 0.5日

### Step 2 (1週間): 優先度ベース統合 (パターンA)
```
priority 1: LME銅 ON  |lme| >= 1% → direction = sign(lme)
priority 2: first30  (銘柄別マップに従う)
priority 3: gap fill (down-gapで非鉄/6963)
priority 4: 先物夜間 (|fut| >= 0.3%)
else: no trade
```
OOS 3か月で Sharpe を測定。**期待値: Sharpe 3-5, N ~120/年**

### Step 3 (2週間): ロジスティック回帰 (パターンC)
10特徴量程度で学習。確率 P(up) 出力。
サイズを |P - 0.5| に比例させる (Kelly 近似)。
**期待値: Sharpe 4-6, MaxDD 半減**

### Step 4 (1か月): カテゴリ拡張
- SOX指数、USDJPY ON、S&P500 ON の取得
- クロスセクション (半導体ペア) の実装
- VWAP deviation 戦略の単独検証

### Step 5 (3か月): 実運用 + 毎週の再学習

---

## 7. 評価ゲート (Go/No-Go 基準)

各ステップを次に進める前に確認:

| ゲート | 基準 |
|---|---|
| Step 1 → 2 | 既存全シグナルが統一IFから呼び出せる、旧コードの数値と一致 |
| Step 2 → 3 | 優先度ベースで OOS 3か月 Sharpe >= 2.0 |
| Step 3 → 4 | ロジスティックで OOS Sharpe が優先度ベースを上回る (>= 0.5 差) |
| Step 4 → 5 | 追加特徴で Sharpe + 0.5 以上改善 OR Sharpe同等で N+30% |
| Step 5 → 本運用 | 3か月の紙トレードで Sharpe 2.0 維持、日次乖離 |想定-実績| < 20bps |

---

## 8. シグナル独立性の期待表 (事前仮説)

各シグナル間の相関予想 (Phase 3で検証予定):

|  | LME | 先物夜間 | first30 | gap | pair | VWAP | 曜日 |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| LME | 1.00 | 0.70 | 0.10 | 0.30 | 0.05 | 0.02 | 0 |
| 先物夜間 | | 1.00 | 0.15 | 0.35 | 0.05 | 0.02 | 0 |
| first30 | | | 1.00 | 0.50 | 0.40 | 0.30 | 0 |
| gap | | | | 1.00 | 0.25 | 0.15 | 0 |
| pair | | | | | 1.00 | 0.20 | 0 |
| VWAP | | | | | | 1.00 | 0 |

**独立性が高い組み合わせ**:
- LME × first30 (0.10) ✅
- LME × pair (0.05) ✅
- 曜日 × 全て (0.00) ✅ → カテゴリ2フィルターとして独立使用

**重複が強いもの**:
- LME × 先物夜間 (0.70) → 合算ではなく択一 or 直交化

---

## 9. この先の深掘りテーマ

1. **個別銘柄 × 時間帯 × 出来高** の 3-way interaction の網羅的探索
2. **5分足アラートシステム**: 条件発火時に Slack 通知
3. **サイズ決定**: Volatility-targeting (年率10%ボラで固定) の導入
4. **マルチペリオドポートフォリオ**: 同時に複数銘柄のシグナルが発火した時の配分
5. **リスクパリティ**: 各シグナルのリスク寄与を均等化
6. **Regime-Switching Model**: HMMで "リスクオン/オフ" を推定、モデル切替

---

## 10. サマリー: 5つの行動指針

1. **方向判定は11カテゴリに分類可能** — 漏らさず検討すること
2. **銘柄特性に応じた方向マップを持つ** — 一律ルールは非効率
3. **カテゴリ10 (カレンダー) はフィルターとして独立使用** — 方向判定と直交
4. **アンサンブルはロジスティック回帰で十分** — NN/GBTは過学習リスク高
5. **OOS 検証を必ず通す** — in-sample Sharpe の70%を死守

---

## 付録: 現状シグナル一覧 (本リポジトリ分析結果)

| カテゴリ | シグナル | 対応フォルダ | Sharpe | N/年 |
|---|---|---|:-:|:-:|
| 3 | LME銅 +1% → コア5 Long | `20260421_lme_copper_overnight` | +11.50 | 19 |
| 3 | LME累積 LB=10 +3% | `20260422_lme_momentum` | +7.55 | 37 |
| 3 | TOPIX夜間 +0.3% → Long | `20260422_topix_futures_gap` | +4.79 | 100 |
| 3 | 日経夜間 +0.3% → Long | `20260422_nikkei_futures_gap` | +4.68 | 89 |
| 3 | WTI +0.5% → エネルギー | `20260422_wti_energy` | +2.74 | 89 |
| 3 | Brent +2% → 海運 | `20260422_brent_shipping` | +3.56 | 30 |
| 6 | first30 rev/mom 銘柄別 | `20260422_orb_volume` | +2.5〜+10.2 | 20〜57 |
| 7 | 非鉄 寄付ダイバージェンスLS | `20260422_nonfer_ls` | +3.66 | 83 |
| 10 | 木曜除外フィルター | `20260422_dayofweek_lme` | 除外で Sharpe向上 | — |

**これらを統合すると、理論上 N=200+/年, Sharpe 4-6 のパイプラインが構築可能**。
