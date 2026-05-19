"""
曜日別ON ＆ 当日セッション分解
- 月曜ON: 金曜引け→月曜寄付
- 火曜ON: 月曜引け→火曜寄付
- 水曜ON: 火曜引け→水曜寄付
- 木曜ON: 水曜引け→木曜寄付
- 金曜ON: 木曜引け→金曜寄付

各曜日ON保有後の：
- 当日寄付クローズ
- 当日引けクローズ
の累積効果を比較
"""
import sys, os, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import psycopg2
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

PG_CONFIG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}

SEMI = {
    '69200': 'レーザーテック', '80350': '東京エレクトロン', '68570': 'アドバンテスト',
    '61460': 'ディスコ', '40630': '信越化学', '69630': 'ローム',
    '77350': 'SCREEN', '34360': 'SUMCO', '65260': 'ソシオネクスト', '99840': 'SBG',
}
NONFER = {
    '57130': '住友金属鉱山', '57110': '三菱マテリアル', '57060': '三井金属',
    '57140': 'DOWA', '57150': '古河機械金属', '57270': '東邦チタニウム',
}
ALL = {**SEMI, **NONFER}
START = '2025-11-01'
END   = '2026-05-12'

DOW_LABELS = {0:'月曜', 1:'火曜', 2:'水曜', 3:'木曜', 4:'金曜'}

# ── データ取得 ──────────────────────────
conn = psycopg2.connect(**PG_CONFIG)
placeholders = ','.join(f"'{c}'" for c in ALL)
daily = pd.read_sql(f"""
    SELECT code, date, open, close
    FROM stocks_daily
    WHERE code IN ({placeholders})
      AND date >= '{START}' AND date <= '{END}'
    ORDER BY code, date
""", conn)
conn.close()

daily['date'] = pd.to_datetime(daily['date']).dt.tz_localize(None)

# ── ON & 当日リターン計算 ─────────────────
def compute(g):
    g = g.sort_values('date').reset_index(drop=True)
    g['prev_close'] = g['close'].shift(1)
    g['on_gap']     = (g['open']  / g['prev_close'] - 1) * 100  # 前日引け→当日寄付
    g['day_open_close'] = (g['close'] / g['open'] - 1) * 100    # 当日寄付→当日引け
    g['full_day']   = (g['close'] / g['prev_close'] - 1) * 100  # 前日引け→当日引け
    g['dow']        = g['date'].dt.dayofweek
    g['sector']     = 'semi' if g['code'].iloc[0] in SEMI else 'nonfer'
    return g

result = daily.groupby('code', group_keys=False).apply(compute)
result = result.dropna(subset=['on_gap'])
# 平日のみ
result = result[result['dow'] <= 4]

# ===========================================================
# 集計1: 曜日別 ON / 当日セッション分解 (セクター別)
# ===========================================================
print("="*75)
print("【曜日別ON × セクター】 N=銘柄数×週数")
print("="*75)
print(f"\n  {'曜日':6} {'セクター':8} {'ON gap':>10} {'寄→引':>10} {'当日全日':>10} {'ON勝率':>7} {'N':>5}")
print("-"*75)
for dow in range(5):
    for sec_code, sec_name in [('semi','半導体AI'), ('nonfer','非鉄金属')]:
        sub = result[(result['dow'] == dow) & (result['sector'] == sec_code)]
        if len(sub) == 0: continue
        on    = sub['on_gap'].mean()
        sess  = sub['day_open_close'].mean()
        full  = sub['full_day'].mean()
        wr_on = (sub['on_gap'] > 0).mean() * 100
        print(f"  {DOW_LABELS[dow]:6} {sec_name:8} {on:>+9.3f}% {sess:>+9.3f}% {full:>+9.3f}%  {wr_on:>5.0f}%  {len(sub):>4}")

# ===========================================================
# 集計2: 銘柄別 金曜ON (木曜引け→金曜寄付) ランキング
# ===========================================================
print("\n" + "="*75)
print("【金曜ON 銘柄別ランキング】 (木曜引け→金曜寄付)")
print("="*75)
fri = result[result['dow'] == 4].copy()
rank_fri = fri.groupby('code').agg(
    on_mean=('on_gap','mean'), on_wr=('on_gap', lambda x: (x>0).mean()*100),
    sess_mean=('day_open_close','mean'),
    full_mean=('full_day','mean'),
    n=('on_gap','count')
).sort_values('on_mean', ascending=False)

print(f"\n  {'銘柄':14} {'セク':6} {'ON':>8} {'ON勝率':>7} {'寄→引':>8} {'金曜全日':>8}  N")
for code, row in rank_fri.iterrows():
    sec = '半導体' if code in SEMI else '非鉄'
    print(f"  {ALL[code]:14}{sec:6} {row['on_mean']:>+7.2f}% {row['on_wr']:>5.0f}%  "
          f"{row['sess_mean']:>+7.2f}% {row['full_mean']:>+7.2f}%  {int(row['n'])}")

# ===========================================================
# 集計3: 各曜日 → 当日寄付クローズ戦略の累積効果 (セクター)
# ===========================================================
print("\n" + "="*75)
print("【各曜日ON のみ保有戦略の累積効果】")
print("（前日引け買い→当日寄付売り、コスト4bps/トレード）")
print("="*75)
print(f"\n  {'曜日':6} {'セクター':8} {'累積':>9} {'平均':>9} {'勝率':>7} {'Sharpe':>8} {'N':>5}")
COST_BPS = 4.0
for dow in range(5):
    for sec_code, sec_name in [('semi','半導体AI'), ('nonfer','非鉄金属')]:
        sub = result[(result['dow'] == dow) & (result['sector'] == sec_code)]
        if len(sub) == 0: continue
        # 各銘柄の各日のON gapを取り、銘柄平均→日付集約
        daily_pnl = sub.groupby('date')['on_gap'].mean() - COST_BPS/100  # コスト引き
        cum = ((1 + daily_pnl/100).cumprod().iloc[-1] - 1) * 100
        wr  = (daily_pnl > 0).mean() * 100
        avg = daily_pnl.mean()
        sharpe = daily_pnl.mean() / daily_pnl.std() * np.sqrt(252/5) if daily_pnl.std()>0 else 0
        print(f"  {DOW_LABELS[dow]:6} {sec_name:8} {cum:>+8.2f}% {avg:>+8.3f}%  "
              f"{wr:>5.0f}%  {sharpe:>7.2f}  {len(daily_pnl):>4}")

# 保存
with open(os.path.join(os.path.dirname(__file__), 'results.pkl'), 'wb') as f:
    pickle.dump(dict(
        result=result, rank_fri=rank_fri,
        ALL=ALL, SEMI=SEMI, NONFER=NONFER,
        START=START, END=END,
    ), f)
print("\n→ results.pkl 保存完了")
