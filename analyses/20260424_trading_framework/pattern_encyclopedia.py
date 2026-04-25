"""
パターン辞典 — 全銘柄 × 全パターンの体系的スキャン
出力: patterns_result.csv（シグナル生成スクリプトが読み込む）
"""
import psycopg2, pandas as pd, numpy as np, warnings, sys
warnings.filterwarnings('ignore')

PG = {"host":"localhost","port":5432,"user":"postgres","dbname":"market_data"}

SYMS = {
    "5713.T":"住山",  "5711.T":"三菱マテ","5706.T":"三井金属",
    "5803.T":"フジクラ","5802.T":"住友電工","5801.T":"古河電工",
    "6857.T":"アドバンテスト","6920.T":"レーザーテック",
    "6146.T":"ディスコ","6861.T":"キーエンス","9984.T":"SBG",
}
SECTOR = {
    "5713.T":"非鉄","5711.T":"非鉄","5706.T":"非鉄",
    "5803.T":"非鉄","5802.T":"非鉄","5801.T":"非鉄",
    "6857.T":"半導体","6920.T":"半導体","6146.T":"半導体",
    "6861.T":"半導体","9984.T":"その他",
}
DAY = {0:'月',1:'火',2:'水',3:'木',4:'金'}

# ── ローダー ──────────────────────────────
def load(sym):
    conn = psycopg2.connect(**PG)
    df = pd.read_sql(
        f"SELECT timestamp,open,high,low,close,volume FROM intraday_data "
        f"WHERE symbol='{sym}' ORDER BY timestamp", conn)
    conn.close()
    df['jst'] = pd.to_datetime(df['timestamp']) + pd.Timedelta(hours=9)
    return df.dropna(subset=['close']).set_index('jst').sort_index()

def thr(df):
    h,m = df.index.hour, df.index.minute
    return df[(h==9)|((h>=10)&(h<11))|((h==11)&(m<=30))|
              ((h==12)&(m>=30))|((h>=13)&(h<15))|((h==15)&(m<=30))]
def mae(df):
    h,m=df.index.hour,df.index.minute
    return df[(h==9)|((h>=10)&(h<11))|((h==11)&(m<=30))]
def aft(df):
    h,m=df.index.hour,df.index.minute
    return df[((h==12)&(m>=30))|((h>=13)&(h<15))|((h==15)&(m<=30))]

# ── 統計ユーティリティ ────────────────────
def stats(arr):
    a = np.array(arr, dtype=float)
    a = a[~np.isnan(a)]
    if len(a) < 5:
        return dict(n=len(a), mean=np.nan, med=np.nan, std=np.nan,
                    wr=np.nan, sharpe=np.nan)
    wr = (a>0).mean()*100
    sh = a.mean()/a.std()*np.sqrt(252) if a.std()>0 else 0
    return dict(n=len(a), mean=a.mean(), med=np.median(a),
                std=a.std(), wr=wr, sharpe=sh)

def record(rows, sym, pattern, cond, pnl_arr, cost_pct=0.04):
    """パターン結果を rows に追記"""
    s = stats(pnl_arr)
    if s['n'] < 5: return
    a = np.array(pnl_arr, dtype=float)
    a = a[~np.isnan(a)]
    net = a - cost_pct
    rows.append({
        'sym': sym, 'name': SYMS[sym], 'sector': SECTOR[sym],
        'pattern': pattern, 'condition': cond,
        **{f'gross_{k}': v for k,v in s.items()},
        'net_mean': net.mean(), 'net_wr': (net>0).mean()*100,
        'net_sharpe': net.mean()/net.std()*np.sqrt(252) if net.std()>0 else 0,
    })

# ── 日次データ構築 ─────────────────────────
def build_daily(df):
    rows = []
    prev_cl = None; prev_dow = None; prev_ret = None
    for dt, g in df.groupby(df.index.date):
        ga = thr(g); gm = mae(g); go = aft(g)
        if len(ga)<20 or len(gm)<8 or len(go)<8:
            prev_cl=None; continue
        dow = pd.Timestamp(dt).dayofweek
        op  = float(ga['open'].iloc[0])
        cl  = float(ga['close'].iloc[-1])
        mae_op = float(gm['open'].iloc[0])
        mae_cl = float(gm['close'].iloc[-1])
        aft_op = float(go['open'].iloc[0])
        aft_cl = float(go['close'].iloc[-1])
        hi = float(ga['high'].max()); lo = float(ga['low'].min())
        if op<=0 or mae_op<=0 or aft_op<=0: prev_cl=None; continue

        # ORB(最初15分の高値/安値)
        orb = ga[(ga.index.hour==9)&(ga.index.minute<=15)]
        orb_hi = float(orb['high'].max()) if len(orb)>0 else np.nan
        orb_lo = float(orb['low'].min())  if len(orb)>0 else np.nan

        # 後場寄後5分
        aft_5m = go[go.index.hour==12]
        aft_5m_cl = float(aft_5m['close'].iloc[-1]) if len(aft_5m)>0 else np.nan

        # チェックポイント
        def cp(hh,mm,base=op):
            b=ga[(ga.index.hour==hh)&(ga.index.minute==mm)]
            return (float(b['close'].iloc[-1])/base-1)*100 if len(b)>0 and base>0 else np.nan

        row = {
            'date':dt,'dow':dow,
            'open':op,'close':cl,'high':hi,'low':lo,
            'mae_op':mae_op,'mae_cl':mae_cl,
            'aft_op':aft_op,'aft_cl':aft_cl,
            'day_ret':  (cl/op-1)*100,
            'mae_ret':  (mae_cl/mae_op-1)*100,
            'aft_ret':  (aft_cl/aft_op-1)*100,
            'gap_noon': (aft_op/mae_cl-1)*100,
            'gap_day':  (op/prev_cl-1)*100 if prev_cl else np.nan,  # 前日引け→当日寄付
            'prev_ret': prev_ret,
            'orb_hi':orb_hi,'orb_lo':orb_lo,
            'aft_5m_cl':aft_5m_cl,
            'aft_5m_ret': (aft_5m_cl/aft_op-1)*100 if aft_5m_cl and aft_op>0 else np.nan,
            'cp0930': cp(9,30), 'cp1000':cp(10,0),
            'cp1130': cp(11,30),'cp1300':cp(13,0),
            'cp1430': cp(14,30),'cp1500':cp(15,0),'cp1530':cp(15,30),
        }
        rows.append(row)
        prev_cl = cl; prev_dow = dow; prev_ret = row['day_ret']
    return pd.DataFrame(rows)

# ── パターン集 ────────────────────────────
def scan_patterns(sym, d):
    rows = []

    # ═══════════════════════════════════════
    # 1. 曜日効果
    # ═══════════════════════════════════════
    for dow in range(5):
        sub = d[d['dow']==dow]['day_ret'].dropna()
        record(rows, sym, '曜日効果_日次', f'{DAY[dow]}曜_日次', sub)
        # 前場
        record(rows, sym, '曜日効果_前場', f'{DAY[dow]}曜_前場', d[d['dow']==dow]['mae_ret'].dropna())
        # 後場
        record(rows, sym, '曜日効果_後場', f'{DAY[dow]}曜_後場', d[d['dow']==dow]['aft_ret'].dropna())

    # ═══════════════════════════════════════
    # 2. 前日→当日 モメンタム/リバーサル
    # ═══════════════════════════════════════
    d2 = d.dropna(subset=['prev_ret','day_ret'])
    d2 = d2.copy()
    d2['prev_q'] = pd.qcut(d2['prev_ret'], 4,
        labels=['前日大幅安','前日小幅安','前日小幅高','前日大幅高'])
    for q in ['前日大幅安','前日小幅安','前日小幅高','前日大幅高']:
        sub = d2[d2['prev_q']==q]['day_ret'].dropna()
        record(rows, sym, '前日→翌日', q, sub)

    # ═══════════════════════════════════════
    # 3. 寄付ギャップ → 当日リターン
    # ═══════════════════════════════════════
    dg = d.dropna(subset=['gap_day'])
    for label, mask in [
        ('GU大(>+1%)',  dg['gap_day']> 1.0),
        ('GU小(0〜1%)', (dg['gap_day']>0)&(dg['gap_day']<=1)),
        ('GD小(0〜-1%)',(dg['gap_day']<0)&(dg['gap_day']>=-1)),
        ('GD大(<-1%)',  dg['gap_day']<-1.0),
    ]:
        # ギャップ方向への追随（モメンタム）
        dir_ = 1 if 'GU' in label else -1
        sub_day  = dg[mask]['day_ret'].dropna() * dir_
        sub_mae  = dg[mask]['mae_ret'].dropna() * dir_
        record(rows, sym, '寄付ギャップ_日次モメンタム', label, sub_day)
        record(rows, sym, '寄付ギャップ_前場モメンタム', label, sub_mae)
        # ギャップフィル（逆張り）
        record(rows, sym, '寄付ギャップ_逆張り', label, dg[mask]['day_ret'].dropna() * -dir_)

    # ═══════════════════════════════════════
    # 4. ORBブレイクアウト
    # ═══════════════════════════════════════
    do = d.dropna(subset=['orb_hi','orb_lo','cp1530'])
    for orb_min in [15, 30]:
        # 30分ORBは cp0930で代用
        cp_col = 'cp0930' if orb_min==15 else 'cp1000'
        do2 = d.dropna(subset=[cp_col,'cp1530'])
        # ORB上抜け（前場ORB後の動き）
        # 簡易: 寄から15分で+0.3%以上 → 前場引けまで継続
        up = do2[do2[cp_col]> 0.3]['mae_ret'].dropna()
        dn = do2[do2[cp_col]<-0.3]['mae_ret'].dropna() * -1
        record(rows, sym, f'ORB{orb_min}分_上抜け→前場', f'前場+0.3%以上', up)
        record(rows, sym, f'ORB{orb_min}分_下抜け→前場', f'前場-0.3%以下', dn)
        # 引けまで
        up2 = do2[do2[cp_col]> 0.3]['cp1530'].dropna()
        dn2 = do2[do2[cp_col]<-0.3]['cp1530'].dropna() * -1
        record(rows, sym, f'ORB{orb_min}分_上抜け→引け', f'前場+0.3%以上', up2)
        record(rows, sym, f'ORB{orb_min}分_下抜け→引け', f'前場-0.3%以下', dn2)

    # ═══════════════════════════════════════
    # 5. 前場方向 → 後場
    # ═══════════════════════════════════════
    for label, mask in [
        ('前場大幅高(>+0.5%)', d['mae_ret']> 0.5),
        ('前場小幅高(0〜+0.5%)',(d['mae_ret']>0)&(d['mae_ret']<=0.5)),
        ('前場小幅安(0〜-0.5%)',(d['mae_ret']<0)&(d['mae_ret']>=-0.5)),
        ('前場大幅安(<-0.5%)', d['mae_ret']<-0.5),
    ]:
        record(rows, sym, '前場→後場', label, d[mask]['aft_ret'].dropna())
        # 曜日別
        for dow in range(5):
            sub = d[mask & (d['dow']==dow)]['aft_ret'].dropna()
            record(rows, sym, f'前場×曜日→後場', f'{label}_{DAY[dow]}曜',
                   sub, cost_pct=0.04)

    # ═══════════════════════════════════════
    # 6. 昼間ギャップ → 後場
    # ═══════════════════════════════════════
    for label, mask in [
        ('昼GU(>+0.2%)',  d['gap_noon']> 0.2),
        ('昼フラット',     d['gap_noon'].abs()<=0.2),
        ('昼GD(<-0.2%)',  d['gap_noon']<-0.2),
    ]:
        record(rows, sym, '昼間ギャップ→後場', label, d[mask]['aft_ret'].dropna())

    # ═══════════════════════════════════════
    # 7. 後場寄直後（12:30台）→ 引け
    # ═══════════════════════════════════════
    da = d.dropna(subset=['aft_5m_ret','aft_ret'])
    for label, mask in [
        ('後場寄後上昇(>+0.1%)', da['aft_5m_ret']> 0.1),
        ('後場寄後フラット',      da['aft_5m_ret'].abs()<=0.1),
        ('後場寄後下落(<-0.1%)', da['aft_5m_ret']<-0.1),
    ]:
        sub = da[mask]['aft_ret'].dropna()
        record(rows, sym, '後場寄直後→引け', label, sub, cost_pct=0.04)

    # ═══════════════════════════════════════
    # 8. 引け前30分（15:00→15:30）
    # ═══════════════════════════════════════
    d['last30'] = d['cp1530'] - d['cp1500']
    for dow in range(5):
        sub = d[d['dow']==dow]['last30'].dropna()
        record(rows, sym, '引け前30分_曜日別', f'{DAY[dow]}曜', sub, cost_pct=0.04)
    # 前場×後場パターン別
    for label, mask in [
        ('前高後高',  (d['mae_ret']>0)&(d['aft_ret']>0)),
        ('前高後安',  (d['mae_ret']>0)&(d['aft_ret']<0)),
        ('前安後高',  (d['mae_ret']<0)&(d['aft_ret']>0)),
        ('前安後安',  (d['mae_ret']<0)&(d['aft_ret']<0)),
    ]:
        sub = d[mask]['last30'].dropna()
        record(rows, sym, '引け前30分_パターン別', label, sub, cost_pct=0.04)

    # ═══════════════════════════════════════
    # 9. 月曜ON（週末ギャップ）
    # ═══════════════════════════════════════
    # 金曜分位別の翌月曜ギャップ
    d_fri = d[d['dow']==4][['date','day_ret']].copy()
    d_mon = d[d['dow']==0][['date','gap_day']].copy()
    mon_gaps = []
    for _, row in d_mon.iterrows():
        # 直前の金曜を探す
        cands = d_fri[pd.to_datetime(d_fri['date']) < pd.to_datetime(row['date'])]
        if len(cands)==0: continue
        fri_ret = cands.iloc[-1]['day_ret']
        mon_gaps.append({'fri_ret':fri_ret, 'mon_gap':row['gap_day']})
    if len(mon_gaps)>=10:
        mg = pd.DataFrame(mon_gaps)
        mg['fq'] = pd.qcut(mg['fri_ret'],4,
            labels=['金大幅安','金小幅安','金小幅高','金大幅高'])
        for q in ['金大幅安','金小幅安','金小幅高','金大幅高']:
            sub = mg[mg['fq']==q]['mon_gap'].dropna()
            record(rows, sym, '月曜ON_金曜分位別gap', q, sub, cost_pct=0.0)
        record(rows, sym, '月曜ON_全体gap', '全体', mg['mon_gap'].dropna(), cost_pct=0.0)

    return rows


# ── メイン ───────────────────────────────
if __name__ == '__main__':
    all_rows = []
    daily_cache = {}

    print("=== パターンスキャン開始 ===")
    for sym, name in SYMS.items():
        print(f"  {name}...", end='', flush=True)
        df = load(sym)
        d  = build_daily(df)
        daily_cache[sym] = d
        rows = scan_patterns(sym, d)
        all_rows.extend(rows)
        print(f" {len(rows)}パターン")

    result = pd.DataFrame(all_rows)
    result.to_csv('analyses/20260424_trading_framework/patterns_result.csv',
                  index=False, encoding='utf-8-sig')
    print(f"\n合計 {len(result)} パターン → patterns_result.csv に保存")

    # ─────────────────────────────────────
    # ランキング出力
    # ─────────────────────────────────────
    print("\n" + "="*72)
    print("【パターン信頼度ランキング Top30（net_sharpe 順）】")
    print("="*72)
    top = (result
           .dropna(subset=['net_sharpe'])
           .query('gross_n >= 20')
           .sort_values('net_sharpe', ascending=False)
           .head(30))
    print(f"{'順':>2} {'銘柄':<10} {'パターン':<22} {'条件':<26} {'N':>4} "
          f"{'総平均':>7} {'勝率':>6} {'Sharpe':>7} {'net_Sharpe':>10}")
    print("-"*105)
    for i, (_, r) in enumerate(top.iterrows(), 1):
        print(f"{i:>2} {r['name']:<10} {r['pattern']:<22} {r['condition']:<26} "
              f"{int(r['gross_n']):>4} {r['gross_mean']:>+6.3f}% {r['gross_wr']:>5.1f}% "
              f"{r['gross_sharpe']:>+6.2f}  {r['net_sharpe']:>+9.2f}")

    print("\n" + "="*72)
    print("【ワースト Bottom10（避けるべきパターン）】")
    print("="*72)
    bot = (result
           .dropna(subset=['net_sharpe'])
           .query('gross_n >= 20')
           .sort_values('net_sharpe')
           .head(10))
    for i, (_, r) in enumerate(bot.iterrows(), 1):
        print(f"{i:>2} {r['name']:<10} {r['pattern']:<22} {r['condition']:<26} "
              f"{int(r['gross_n']):>4} {r['gross_mean']:>+6.3f}% {r['gross_wr']:>5.1f}% "
              f"{r['gross_sharpe']:>+6.2f}  {r['net_sharpe']:>+9.2f}")

    # セクター別サマリー
    print("\n" + "="*72)
    print("【セクター別 各パターンの平均net_Sharpe】")
    print("="*72)
    pat_grp = (result
               .dropna(subset=['net_sharpe'])
               .query('gross_n >= 15')
               .groupby(['sector','pattern'])['net_sharpe']
               .mean()
               .unstack('sector')
               .sort_values('非鉄', ascending=False, na_position='last'))
    pd.set_option('display.max_colwidth', 30)
    print(pat_grp.round(2).to_string())
