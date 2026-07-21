import sys
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, "/mnt/d/Root/ClaudeCode/01_Trading/japan-stocks")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from jstock import db

# ------------------------------------------------------------------
# 非鉄・半導体セクターの押し目エントリー・スクリーニング (2026-07-21 時点)
# 検証済みエッジに整合: oversold_rsi_reversal(RSI<30反発,OOS Sh+6.15) /
#   semi_gapdn(半導体下窓,net Sh1.8) / lasertec_ma25_support(MA25サポート)
# 分割ロバスト: 直近40営業日で|日次|>35%の段差を分割とみなし以降のみで指標算出
# ------------------------------------------------------------------
SEMI = {'80350':'東エレク','68570':'アドテスト','69200':'レーザーテック','77350':'SCREEN',
        '61460':'ディスコ','77290':'東京精密','40630':'信越化学','34360':'SUMCO',
        '40620':'イビデン','67230':'ルネサス','69630':'ローム'}


def screen():
    nf = db.read_sql("SELECT code5,name_ja FROM symbol_master WHERE sector33_nm='非鉄金属'", [])
    codes = list(SEMI.keys()) + list(nf['code5'])
    namemap = {**SEMI, **dict(zip(nf['code5'], nf['name_ja']))}
    px = db.read_sql(
        "SELECT code,date,close,turnover_value FROM stocks_daily "
        "WHERE code = ANY(%s) AND date>='2026-02-01' ORDER BY code,date", [codes])
    rows = []
    for c, g in px.groupby('code'):
        g = g.sort_values('date').reset_index(drop=True)
        if len(g) < 30:
            continue
        a = g['close'].values.astype(float)
        r = np.diff(a) / a[:-1]
        look = min(40, len(r))
        jumps = np.where(np.abs(r[-look:]) > 0.35)[0]
        note = ''
        s = a
        if len(jumps):
            cut = len(a) - look + jumps[-1] + 1
            s = a[cut:]
            note = '分割補正'
        d = np.diff(s)
        up = np.where(d > 0, d, 0)
        dn = np.where(d < 0, -d, 0)
        n = min(14, len(d))
        if n < 5:
            continue
        rs = up[-n:].mean() / (dn[-n:].mean() + 1e-9)
        rsi = 100 - 100 / (1 + rs)
        m = min(25, len(s))
        ma = s[-m:].mean()
        last = s[-1]
        dist = (last / ma - 1) * 100
        h = min(20, len(s))
        dd = (last / s[-h:].max() - 1) * 100
        r5 = (s[-1] / s[-6] - 1) * 100 if len(s) > 6 else np.nan
        r20 = (s[-1] / s[-min(20, len(s))] - 1) * 100
        adv = g['turnover_value'].tail(20).mean() / 1e8
        sec = '半導体' if c in SEMI else '非鉄'
        rows.append([c, namemap.get(c, c), sec, round(last), round(rsi, 1),
                     round(dist, 1), round(dd, 1), round(r5, 1), round(r20, 1),
                     round(adv), note])
    df = pd.DataFrame(rows, columns=['code', 'name', 'sec', 'close', 'RSI14',
                                     'vsMA25%', 'DD20H%', 'r5%', 'r20%', 'ADV億', 'note'])
    df = df[df['ADV億'] >= 10].sort_values(['sec', 'RSI14']).reset_index(drop=True)
    return df


if __name__ == '__main__':
    df = screen()
    df.to_csv('candidates.csv', index=False, encoding='utf-8-sig')
    print(df.to_string(index=False))
    print('\nsaved candidates.csv')

    # --- 可視化: RSI vs 20日DD 散布図 (押し目度マップ) ---
    try:
        fm.fontManager.addfont('/root/.fonts/NotoSansJP.ttf')
        plt.rcParams['font.family'] = 'Noto Sans JP'
    except Exception:
        pass
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(figsize=(12, 6.75), facecolor='white')
    for sec, col in [('半導体', '#d62728'), ('非鉄', '#1f77b4')]:
        d = df[df['sec'] == sec]
        ax.scatter(d['RSI14'], d['DD20H%'], s=(d['ADV億'] ** 0.5) * 6,
                   c=col, alpha=0.6, label=sec, edgecolors='white')
        for _, r in d.iterrows():
            ax.annotate(r['name'], (r['RSI14'], r['DD20H%']), fontsize=8,
                        xytext=(3, 3), textcoords='offset points')
    ax.axvline(30, ls='--', color='gray', alpha=0.6)
    ax.text(30.3, ax.get_ylim()[0] + 1, 'RSI30 (売られ過ぎ域)', fontsize=8, color='gray')
    ax.set_xlabel('RSI(14)  ←売られ過ぎ')
    ax.set_ylabel('20日高値からの下落率 %')
    ax.set_title('非鉄・半導体 押し目マップ (2026-07-17時点, バブル=売買代金)', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.text(0.99, 0.01, 'データ: stocks_daily 〜2026-07-17 / 分割ロバスト指標',
             ha='right', va='bottom', fontsize=8, color='gray')
    fig.savefig('result.png', dpi=100, bbox_inches='tight', facecolor='white')
    print('saved result.png')
