"""V6b L/S 累積パフォーマンス X投稿用グラフ生成 (1200x675)。"""
from __future__ import annotations

import os
import sys
import psycopg2
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(line_buffering=True)
DB_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost/market_data")

CODES4 = [
    '5713','5711','5706','5714','5016','5801','5802','5803',
    '8035','6857','6920','6146','7735','4063','3436','7741','6963','6526','9984','4062','6723','285A','6525',
    '8306','8316','8411','7011','7013','7012','6503','6501','6758','7203','7267','8058','8031',
    '6981','6762','6971','6976','4004','8766','1605','6861','6954','9432','7974','9983','6098','9433',
]
CODES5 = [c + '0' for c in CODES4]
CODE_LIST = ','.join(f"'{c}'" for c in CODES5)
HOLD = 20
N_SIDE = 8
IS_START = pd.Timestamp("2022-01-01")
IS_END = pd.Timestamp("2023-12-31")


def fetch(sql):
    conn = psycopg2.connect(DB_URL)
    df = pd.read_sql(sql, conn)
    conn.close()
    return df


prices = fetch(f"""
    SELECT LEFT(code,4) c, date, adj_close::float ac
    FROM stocks_daily WHERE code IN ({CODE_LIST})
      AND date >= '2020-07-01' AND adj_close > 0 ORDER BY code, date
""")
prices['date'] = pd.to_datetime(prices['date'])

# コード別に (date配列, ac配列) を事前計算して高速化
by_code = {}
for code, g in prices.groupby('c'):
    g = g.sort_values('date')
    by_code[code] = (g['date'].values, g['ac'].values)

rows = []
start = np.datetime64('2021-07-01')
for code5 in CODES5:
    code = code5[:4]
    if code not in by_code:
        continue
    dates, ac = by_code[code]
    n = len(ac)
    for i in range(n):
        if dates[i] < start:
            continue
        if i < 89 or i + HOLD >= n:
            continue
        last = ac[i]
        r20 = last / ac[i - 20] - 1
        ma75 = ac[i - 74:i + 1].mean()
        d75 = last / ma75 - 1
        daily = ac[i - 19:i + 1] / ac[i - 20:i] - 1
        vol20 = float(np.std(daily, ddof=1) * np.sqrt(252))
        if vol20 <= 0:
            continue
        v6b = r20 / vol20 + 0.5 * d75
        fwd = ac[i + HOLD] / last - 1
        rows.append({'date': dates[i], 'code': code, 'v6b': v6b, 'fwd': fwd})

df = pd.DataFrame(rows)
df['date'] = pd.to_datetime(df['date'])

daily_ls = []
for dt, g in df.groupby('date'):
    if len(g) < 2 * N_SIDE:
        continue
    r = g.sort_values('v6b', ascending=False)
    daily_ls.append({'date': dt,
                     'ls': r.head(N_SIDE)['fwd'].mean() - r.tail(N_SIDE)['fwd'].mean()})
ls = pd.DataFrame(daily_ls).set_index('date').sort_index()
ls['contrib'] = ls['ls'] / HOLD
ls['cum'] = (1 + ls['contrib']).cumprod()

# Sharpe
def sharpe(s):
    return s.mean() / s.std() * np.sqrt(252)

is_mask = (ls.index >= IS_START) & (ls.index <= IS_END)
is_sh = sharpe(ls[is_mask]['contrib'])
oos_sh = sharpe(ls[ls.index > IS_END]['contrib'])

# --- 描画 ---
import matplotlib.font_manager as fm
_fp = "/root/.fonts/NotoSansJP.ttf"
if os.path.exists(_fp):
    fm.fontManager.addfont(_fp)
    jp_font = fm.FontProperties(fname=_fp).get_name()
else:
    jp_font = 'sans-serif'
plt.rcParams.update({
    'font.family': [jp_font, 'sans-serif'],
    'axes.unicode_minus': False,
    'figure.facecolor': 'white',
    'axes.facecolor': '#f8f9fa',
    'grid.alpha': 0.3,
})
fig, ax = plt.subplots(figsize=(12, 6.75), facecolor='white')

ax.plot(ls.index, (ls['cum'] - 1) * 100, color='#1f77b4', linewidth=2,
        label='V6b L/S 累積リターン')
ax.axvline(IS_END, color='#d62728', linestyle='--', alpha=0.7)
ax.text(IS_END, ax.get_ylim()[1] * 0.05, ' OOS開始 (2024-01)',
        color='#d62728', fontsize=10, va='bottom')

ax.set_title('V6スコア (ボラ調整モメンタム + 長期トレンド) — 50銘柄ロング/ショート',
             fontsize=16, fontweight='bold', pad=14)
ax.set_ylabel('累積リターン (%)', fontsize=12)
ax.grid(True, alpha=0.3)
ax.legend(loc='upper left', fontsize=11)

txt = (f"V6b = r20÷ボラ + 0.5×MA75乖離\n"
       f"top8 Long / bottom8 Short・保有20日・日次\n"
       f"IS Sharpe={is_sh:.2f}  OOS Sharpe={oos_sh:.2f}")
ax.text(0.99, 0.04, txt, transform=ax.transAxes, ha='right', va='bottom',
        fontsize=11, bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

fig.text(0.99, 0.01,
         'データ: 2021-07〜2026-05 / 日本株日足 (JQuants) / 50銘柄ユニバース',
         ha='right', va='bottom', fontsize=8, color='gray')

out = os.path.join(os.path.dirname(__file__), "result.png")
plt.savefig(out, dpi=100, bbox_inches='tight', facecolor='white')
print(f"保存: {out}")
print(f"IS Sharpe={is_sh:.2f}, OOS Sharpe={oos_sh:.2f}, 最終累積={(ls['cum'].iloc[-1]-1)*100:.1f}%")
