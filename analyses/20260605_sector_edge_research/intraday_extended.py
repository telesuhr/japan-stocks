"""
イントラデイ戦略 拡張検証 (ORB条件付け + 新パターン)
"""
import sys; sys.stdout.reconfigure(line_buffering=True)
import pandas as pd, numpy as np, psycopg2
from scipy import stats
import warnings; warnings.filterwarnings('ignore')

DB_URL  = "postgresql://postgres@localhost/market_data"
OUT_DIR = "/mnt/d/Root/ClaudeCode/01_Trading/japan-stocks/analyses/20260605_sector_edge_research"
COST    = 0.001

CODES = ['80350','68570','69200','61460','99840',
         '57130','57110','69810','83060','70110']

def get_conn(): return psycopg2.connect(DB_URL)

def load_intraday():
    months = ['202405','202406','202407','202408','202409','202410',
              '202411','202412','202501','202502','202503','202504',
              '202505','202506']
    cl = ",".join(f"'{c}'" for c in CODES)
    unions = " UNION ALL ".join(
        f"SELECT code,ts,open,high,low,close,volume FROM public.stocks_intraday_{m} WHERE code IN ({cl})"
        for m in months)
    with get_conn() as cn: df = pd.read_sql(f"SELECT * FROM ({unions}) t ORDER BY code,ts", cn, parse_dates=['ts'])
    df[['open','high','low','close','volume']] = df[['open','high','low','close','volume']].astype(float)
    return df

def load_macro():
    sql = """SELECT symbol, trade_date AS date, close
             FROM macro.daily_ohlcv
             WHERE symbol IN ('.SOX','NVDA','HGc1','JPY=','VXc1')
             AND trade_date >= '2024-04-01' ORDER BY symbol, trade_date"""
    with get_conn() as cn: return pd.read_sql(sql, cn, parse_dates=['date'])

def load_daily_close():
    cl = ",".join(f"'{c}'" for c in CODES)
    sql = f"SELECT code,date,adj_close AS prev_close FROM public.stocks_daily WHERE code IN ({cl}) AND date >= '2024-04-01' ORDER BY code,date"
    with get_conn() as cn: df = pd.read_sql(sql, cn, parse_dates=['date'])
    df['prev_close'] = df.groupby('code')['prev_close'].shift(1)
    return df.dropna()

print("ロード中...")
intra = load_intraday()
macro_raw = load_macro()
macro = macro_raw.pivot(index='date',columns='symbol',values='close').ffill()
daily_cl = load_daily_close()

intra['date']   = intra['ts'].dt.date
intra['minute'] = intra['ts'].dt.hour*60 + intra['ts'].dt.minute
intra['date_dt']= pd.to_datetime(intra['date'])
daily_cl['date_dt'] = pd.to_datetime(daily_cl['date'].dt.date) if hasattr(daily_cl['date'],'dt') else pd.to_datetime(daily_cl['date'])
intra = intra.merge(daily_cl[['code','date_dt','prev_close']], on=['code','date_dt'], how='left')

# SOX前日リターンを日付→翌日にマップ
sox_ret = macro['.SOX'].pct_change() if '.SOX' in macro.columns else None
sox_sign_map = {}
if sox_ret is not None:
    for dt, v in sox_ret.items():
        next_dt = dt + pd.Timedelta(days=1)
        sox_sign_map[next_dt.date()] = np.sign(v) if pd.notna(v) else 0

print(f"  1分足:{len(intra):,}行 期間:{intra['ts'].min().date()}~{intra['ts'].max().date()}")

# ─── 統計 ─────────────────────────────────────────────────────────────────
def cstats(rets, name):
    r = pd.Series(rets).dropna() - COST
    n = len(r)
    if n < 15: return {'strategy':name,'n':n,'edge_class':'insufficient','t_stat':np.nan,'mean_net':np.nan,'win_rate':np.nan,'sharpe_annual':np.nan,'max_dd':np.nan}
    wr = (r>0).mean(); mu = r.mean(); sd = r.std()
    t,p = stats.ttest_1samp(r,0)
    sh = mu/sd*np.sqrt(252*5) if sd>0 else np.nan
    cum = (1+r).cumprod(); mdd = (cum/cum.cummax()-1).min()
    ec = 'strong_pos' if t>2.5 and mu>0 else 'weak_pos' if t>1.8 and mu>0 else \
         'strong_neg' if t<-2.5 and mu<0 else 'weak_neg' if t<-1.8 and mu<0 else 'noise'
    return {'strategy':name,'n':n,'win_rate':wr,'mean_net':mu,'sharpe_annual':sh,'t_stat':t,'p_value':p,'max_dd':mdd,'edge_class':ec}

results = []

# ─── ORB基本 (再掲) ─────────────────────────────────────────────────────────
def orb_core(df, orb_end, exit_end, long_only=False, short_only=False,
             sox_filter=None, min_range=0.0, max_range=999.0,
             vol_filter=False, gap_align=False):
    """
    sox_filter: +1=SOX陽線の日のみ, -1=SOX陰線の日のみ, None=全日
    min_range/max_range: ORB値幅フィルター (対open%)
    vol_filter: 当日9:00-9:05の出来高が20日平均の1.5倍以上の日のみ
    gap_align: ギャップ方向と同方向のブレイクのみ
    """
    # 銘柄別に20日平均出来高を事前計算
    vol20 = {}
    if vol_filter:
        for (code, date_val), gdf in df.groupby(['code','date']):
            orb_bars = gdf[(gdf['minute'] >= 540) & (gdf['minute'] < 540+orb_end)]
            if len(orb_bars) > 0:
                vol20.setdefault(code, []).append((date_val, orb_bars['volume'].sum()))
        for code in vol20:
            arr = sorted(vol20[code])
            vol20[code] = {d: np.mean([v for _,v in arr[max(0,i-20):i]]) if i>0 else 0
                           for i,(d,_) in enumerate(arr)}

    rets = []
    for (code, date_val), gdf in df.groupby(['code','date']):
        gdf = gdf.sort_values('minute')
        # SOXフィルター
        if sox_filter is not None:
            s = sox_sign_map.get(date_val, 0)
            if sox_filter == 1 and s <= 0: continue
            if sox_filter == -1 and s >= 0: continue

        orb_bars = gdf[(gdf['minute'] >= 540) & (gdf['minute'] < 540+orb_end)]
        if len(orb_bars) < 2: continue
        orb_high = orb_bars['high'].max()
        orb_low  = orb_bars['low'].min()
        orb_range = orb_high - orb_low
        if orb_bars['open'].iloc[0] <= 0: continue
        range_pct = orb_range / orb_bars['open'].iloc[0]
        if range_pct < min_range or range_pct > max_range: continue

        # 出来高フィルター
        if vol_filter:
            avg_vol = vol20.get(code, {}).get(date_val, 0)
            cur_vol = orb_bars['volume'].sum()
            if avg_vol > 0 and cur_vol < avg_vol * 1.5: continue

        # ギャップ
        pc = gdf['prev_close'].iloc[0] if 'prev_close' in gdf.columns else np.nan
        gap_dir = np.sign(orb_bars['open'].iloc[0] - pc) if pd.notna(pc) and pc > 0 else 0

        post = gdf[(gdf['minute'] >= 540+orb_end) & (gdf['minute'] <= exit_end)]
        if len(post) < 2: continue

        entered = False; entry, direction, stop = None, None, None
        for _, bar in post.iterrows():
            if not entered:
                if not short_only and bar['high'] > orb_high:
                    if gap_align and gap_dir < 0: continue
                    entry, direction, stop = orb_high, +1, orb_high - orb_range * 1.5
                    entered = True
                elif not long_only and bar['low'] < orb_low:
                    if gap_align and gap_dir > 0: continue
                    entry, direction, stop = orb_low, -1, orb_low + orb_range * 1.5
                    entered = True
            else:
                if (direction==+1 and bar['low']<stop) or (direction==-1 and bar['high']>stop):
                    rets.append((stop-entry)/entry * direction if direction==1 else (entry-stop)/entry)
                    break
        else:
            if entered and entry:
                ep = post.iloc[-1]['close']
                rets.append((ep-entry)/entry * direction if direction==1 else (entry-ep)/entry)
    return rets

print("\n[ORB拡張]")
# 4. ORB3 (3分)
r = orb_core(intra, 3, 890); results.append(cstats(r,'ORB3  (9:03→14:50)'))
# 5. ORB5 SOX陽線の日のみ
r = orb_core(intra, 5, 890, sox_filter=+1); results.append(cstats(r,'ORB5+SOX陽線フィルター→14:50'))
# 6. ORB5 SOX陰線の日のみ
r = orb_core(intra, 5, 890, sox_filter=-1); results.append(cstats(r,'ORB5+SOX陰線フィルター→14:50'))
# 7. ORB5 ロングのみ (市場の上昇バイアス活用)
r = orb_core(intra, 5, 890, long_only=True); results.append(cstats(r,'ORB5ロングのみ→14:50'))
# 8. ORB5 タイトレンジのみ (0.3%-0.8%) → より精度高いブレイク
r = orb_core(intra, 5, 890, min_range=0.003, max_range=0.008); results.append(cstats(r,'ORB5タイトレンジ(0.3-0.8%)→14:50'))
# 9. ORB5 ワイドレンジのみ (>1%) → 大きな動き
r = orb_core(intra, 5, 890, min_range=0.010); results.append(cstats(r,'ORB5ワイドレンジ(>1%)→14:50'))
# 10. ORB5 出来高1.5倍フィルター
r = orb_core(intra, 5, 890, vol_filter=True); results.append(cstats(r,'ORB5+出来高1.5倍フィルター→14:50'))
# 11. ORB5 ギャップ方向整合のみ
r = orb_core(intra, 5, 890, gap_align=True); results.append(cstats(r,'ORB5+ギャップ方向整合→14:50'))
# 12. ORB10 (10分)
r = orb_core(intra, 10, 890); results.append(cstats(r,'ORB10 (9:10→14:50)'))
# 13. ORB5 前引け出口
r = orb_core(intra, 5, 685); results.append(cstats(r,'ORB5前引け出口(→11:25)'))
print(f"  ORB拡張完了")

# ─── VWAP 再取得 (Reclaim) ─────────────────────────────────────────────────
print("\n[VWAP Reclaim]")
def vwap_reclaim(df, hold=20):
    """
    VWAPを下回った後に再びVWAPを上抜けた瞬間にロング
    下回ったと確認してから上回った時点でエントリー
    """
    rets = []
    for (code, date_val), gdf in df.groupby(['code','date']):
        gdf = gdf.sort_values('minute').copy()
        gdf['tp']     = (gdf['high']+gdf['low']+gdf['close'])/3
        gdf['cum_pv'] = (gdf['tp']*gdf['volume']).cumsum()
        gdf['cumv']   = gdf['volume'].cumsum()
        gdf['vwap']   = gdf['cum_pv']/gdf['cumv'].replace(0,np.nan)
        gdf = gdf.reset_index(drop=True)
        below = False
        for i, row in gdf.iterrows():
            if not (550 <= row['minute'] <= 860): continue
            if pd.isna(row['vwap']): continue
            if not below:
                if row['close'] < row['vwap'] * 0.998: below = True
            else:
                if row['close'] > row['vwap']:  # 再取得
                    entry = row['close']
                    exit_i = min(i+hold, len(gdf)-1)
                    exit_p = gdf.loc[exit_i,'close']
                    rets.append((exit_p-entry)/entry)
                    below = False
    return rets

r = vwap_reclaim(intra, 20); results.append(cstats(r,'VWAP_RECLAIM (下→上抜け→20分)'))
r = vwap_reclaim(intra, 30); results.append(cstats(r,'VWAP_RECLAIM (下→上抜け→30分)'))
print(f"  VWAP Reclaim完了")

# ─── 前場高値/安値の後場ブレイク ───────────────────────────────────────────
print("\n[前場H/L ブレイク]")
def am_hl_break(df, hold=30):
    """前場(9:00-11:30)の高値/安値を後場でブレイクした方向"""
    rets = []
    for (code, date_val), gdf in df.groupby(['code','date']):
        gdf = gdf.sort_values('minute')
        am = gdf[gdf['minute'] < 690]
        pm = gdf[gdf['minute'] >= 750]
        if len(am)<5 or len(pm)<5: continue
        am_high = am['high'].max(); am_low = am['low'].min()
        entered = False
        for _, bar in pm.iterrows():
            if not entered:
                if bar['high'] > am_high:
                    entry = am_high; direction = +1; entered = True
                elif bar['low'] < am_low:
                    entry = am_low;  direction = -1; entered = True
            else:
                exit_p = bar['close']
                ret = direction*(exit_p-entry)/entry if direction==1 else (entry-exit_p)/entry
                rets.append(ret); break
        else:
            if entered:
                exit_p = pm.iloc[-1]['close']
                rets.append(direction*(exit_p-entry)/entry if direction==1 else (entry-exit_p)/entry)
    return rets

r = am_hl_break(intra); results.append(cstats(r,'前場H/L→後場ブレイク方向'))
print(f"  AM H/L Break完了")

# ─── セクターリーダー→フォロワー ─────────────────────────────────────────
print("\n[セクターリプル]")
def sector_ripple(df, leader='80350', followers=None, lag_min=5, hold=15):
    """
    leader銘柄がORB5高値をブレイクした lag_min 分後に
    まだブレイクしていない follower 銘柄でブレイク方向にエントリー
    """
    if followers is None:
        followers = ['68570','69200','61460']
    rets = []
    dates = df['date'].unique()
    for d in dates:
        dg = df[df['date']==d]
        lg = dg[dg['code']==leader].sort_values('minute')
        if len(lg) < 6: continue
        orb_bars = lg[(lg['minute']>=540)&(lg['minute']<545)]
        if len(orb_bars)<2: continue
        orb_h = orb_bars['high'].max(); orb_l = orb_bars['low'].min()
        # リーダーのブレイク時刻
        break_min, break_dir = None, None
        for _, bar in lg[lg['minute']>=545].iterrows():
            if bar['high']>orb_h:  break_min=bar['minute']; break_dir=+1; break
            if bar['low'] <orb_l:  break_min=bar['minute']; break_dir=-1; break
        if break_min is None: continue
        # フォロワー: lag分後にエントリー
        entry_min = break_min + lag_min
        for fc in followers:
            fg = dg[dg['code']==fc].sort_values('minute')
            entry_bars = fg[fg['minute']>=entry_min]
            if len(entry_bars)<2: continue
            ep = entry_bars.iloc[0]['open']
            exit_bars = fg[fg['minute']>=entry_min+hold]
            xp = exit_bars.iloc[0]['close'] if len(exit_bars)>0 else fg.iloc[-1]['close']
            ret = break_dir*(xp-ep)/ep if break_dir==1 else (ep-xp)/ep
            rets.append(ret)
    return rets

r = sector_ripple(intra, lag_min=5,  hold=15); results.append(cstats(r,'セクターリプル 8035→他3銘柄(lag5分,15分保有)'))
r = sector_ripple(intra, lag_min=10, hold=20); results.append(cstats(r,'セクターリプル 8035→他3銘柄(lag10分,20分保有)'))
print(f"  セクターリプル完了")

# ─── 前場強さ→後場キャリー ────────────────────────────────────────────────
print("\n[前場強→後場キャリー]")
def am_strong_pm(df, am_threshold=0.005, hold=60):
    """前場+0.5%以上の銘柄を後場12:30エントリー"""
    rets = []
    for (code, date_val), gdf in df.groupby(['code','date']):
        gdf = gdf.sort_values('minute')
        am = gdf[(gdf['minute']>=540)&(gdf['minute']<690)]
        pm = gdf[gdf['minute']>=750]
        if len(am)<5 or len(pm)<3: continue
        am_ret = (am.iloc[-1]['close']-am.iloc[0]['open'])/am.iloc[0]['open']
        if abs(am_ret) < am_threshold: continue
        direction = np.sign(am_ret)
        ep = pm.iloc[0]['open']
        exit_bars = pm[pm['minute']>=750+hold]
        xp = exit_bars.iloc[0]['close'] if len(exit_bars)>0 else pm.iloc[-1]['close']
        rets.append(direction*(xp-ep)/ep if direction==1 else (ep-xp)/ep)
    return rets

r = am_strong_pm(intra, 0.005, 60); results.append(cstats(r,'前場+0.5%→後場60分キャリー'))
r = am_strong_pm(intra, 0.010, 60); results.append(cstats(r,'前場+1.0%→後場60分キャリー'))
print(f"  前場強→後場完了")

# ─── 分結果表示 ─────────────────────────────────────────────────────────────
MARKER = {'strong_pos':'★★強↑','weak_pos':'★弱↑','noise':'  中立',
          'weak_neg':'▼弱↓','strong_neg':'▼▼強↓','insufficient':'N不足'}

print("\n" + "="*88)
print("  イントラデイ拡張戦略 結果")
print("="*88)
print(f"  {'戦略':<46}  {'N':>5}  {'勝率':>6}  {'期待値':>8}  {'t値':>5}  {'Sharpe':>6}  判定")
print("-"*88)
for r in results:
    if r.get('edge_class')=='insufficient': continue
    n  = int(r['n'])
    wr = f"{r['win_rate']*100:5.1f}%" if pd.notna(r.get('win_rate')) else '  –'
    mn = f"{r['mean_net']*100:+.3f}%" if pd.notna(r.get('mean_net')) else '   –'
    t  = f"{r['t_stat']:+.2f}"        if pd.notna(r.get('t_stat'))  else '  –'
    sh = f"{r['sharpe_annual']:+.2f}" if pd.notna(r.get('sharpe_annual')) else '  –'
    ec = MARKER.get(r['edge_class'],'')
    print(f"  {r['strategy']:<46}  {n:>5}  {wr}  {mn:>8}  {t:>5}  {sh:>6}  {ec}")

pos = [r for r in results if r.get('edge_class') in ('strong_pos','weak_pos') and pd.notna(r.get('t_stat'))]
print(f"\n  ✅ プラスエッジ: {len(pos)}個")
for r in sorted(pos, key=lambda x: x['t_stat'], reverse=True):
    print(f"     [{r['t_stat']:+.2f}] {r['strategy']}  勝率{r['win_rate']*100:.1f}%  期待値{r['mean_net']*100:+.3f}%")

pd.DataFrame(results).to_csv(f"{OUT_DIR}/intraday_extended.csv", index=False)
print(f"\n保存: {OUT_DIR}/intraday_extended.csv  完了")
