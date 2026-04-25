"""
金曜イントラデイ：非鉄 vs 半導体 強弱ランキング
分析日: 2026-04-22
"""

import psycopg2
import pandas as pd
import numpy as np

PG_CONFIG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}

ALL_SYMS = {
    "5713.T": "住友鉱山",
    "5711.T": "三菱マテ",
    "5706.T": "三井金属",
    "5803.T": "フジクラ",
    "5802.T": "住友電工",
    "5801.T": "古河電工",
    "6857.T": "アドバンテスト",
    "6920.T": "レーザーテック",
    "6146.T": "ディスコ",
    "6861.T": "キーエンス",
}
SECTOR = {
    "5713.T": "非鉄", "5711.T": "非鉄", "5706.T": "非鉄",
    "5803.T": "非鉄", "5802.T": "非鉄", "5801.T": "非鉄",
    "6857.T": "半導体", "6920.T": "半導体", "6146.T": "半導体", "6861.T": "半導体",
}


def load(sym):
    conn = psycopg2.connect(**PG_CONFIG)
    df = pd.read_sql(
        f"SELECT timestamp,open,high,low,close FROM intraday_data WHERE symbol='{sym}' ORDER BY timestamp",
        conn)
    conn.close()
    df['jst'] = pd.to_datetime(df['timestamp']) + pd.Timedelta(hours=9)
    return df.dropna(subset=['close']).set_index('jst').sort_index()


def thr(df):
    h, m = df.index.hour, df.index.minute
    return df[(h==9)|((h>=10)&(h<11))|((h==11)&(m<=30))|
              ((h==12)&(m>=30))|((h>=13)&(h<15))|((h==15)&(m<=30))]


def get_stats(df):
    rows = []
    for dt, g in df.groupby(df.index.date):
        g = thr(g)
        if len(g) < 20: continue
        dow = pd.Timestamp(dt).dayofweek
        op = g['open'].iloc[0]
        cl = g['close'].iloc[-1]
        if op <= 0: continue
        mae = g[g.index.hour < 12]
        aft = g[g.index.hour >= 12]
        mae_cl = mae['close'].iloc[-1] if len(mae) > 0 else np.nan
        # チェックポイント別累積リターン
        def cp(hh, mm):
            b = g[(g.index.hour==hh)&(g.index.minute==mm)]
            return (b['close'].iloc[-1]/op-1)*100 if len(b)>0 and op>0 else np.nan
        rows.append({
            'date': dt, 'dow': dow,
            'day_ret':  (cl/op-1)*100,
            'mae_ret':  (mae_cl/op-1)*100 if not np.isnan(mae_cl) else np.nan,
            'aft_ret':  (cl/mae_cl-1)*100 if (not np.isnan(mae_cl) and mae_cl>0) else np.nan,
            'cp0930':   cp(9,30),
            'cp1130':   cp(11,30),
            'cp1300':   cp(13,0),
            'cp1430':   cp(14,30),
            'cp1500':   cp(15,0),
            'cp1530':   cp(15,30),
            'last30':   cp(15,30) - cp(15,0) if (cp(15,30) is not None and cp(15,0) is not None
                        and not np.isnan(cp(15,30)) and not np.isnan(cp(15,0))) else np.nan,
        })
    return pd.DataFrame(rows)


if __name__ == '__main__':
    print("データロード中...")
    stats = {}
    for sym, name in ALL_SYMS.items():
        df = load(sym)
        stats[sym] = get_stats(df)
        print(f"  {name}: 金曜{(stats[sym]['dow']==4).sum()}日")

    # ── 金曜集計 ──────────────────────────────
    fri = {sym: d[d['dow']==4] for sym, d in stats.items()}
    non = {sym: d[d['dow']!=4] for sym, d in stats.items()}

    # ─────────────────────────────────────────
    # 1. 金曜強弱ランキング（総合）
    # ─────────────────────────────────────────
    print("\n" + "="*72)
    print("【金曜 強弱ランキング】  ← 平均/中央値/勝率/対月〜木差/Sharpe")
    print("="*72)
    print(f"{'順':>2} {'銘柄':<12} {'セクター':>5}  {'平均':>7}  {'中央値':>7}  {'勝率':>6}  {'vs月〜木':>8}  {'Sharpe':>7}")
    print("-"*70)

    rows = []
    for sym, name in ALL_SYMS.items():
        f = fri[sym]['day_ret'].dropna()
        n = non[sym]['day_ret'].dropna()
        if len(f) < 5: continue
        vs = f.mean() - n.mean()
        sharpe = f.mean() / f.std() * np.sqrt(52) if f.std()>0 else 0
        rows.append({'sym':sym,'name':name,'mean':f.mean(),'med':f.median(),
                     'wr':(f>0).mean()*100,'vs':vs,'sharpe':sharpe,'sector':SECTOR[sym]})
    rows.sort(key=lambda r: r['sharpe'], reverse=True)

    for i, r in enumerate(rows, 1):
        print(f"{i:>2} {r['name']:<12} {r['sector']:>5}  {r['mean']:>+6.3f}%  {r['med']:>+6.3f}%  {r['wr']:>5.1f}%  {r['vs']:>+7.3f}%  {r['sharpe']:>+6.2f}")

    # ─────────────────────────────────────────
    # 2. 時間帯別累積リターン（金曜 vs 月〜木）
    # ─────────────────────────────────────────
    print("\n" + "="*72)
    print("【金曜 時間帯別累積リターン】  (寄=0% / カッコ内は月〜木差)")
    print("="*72)

    cps = [('09:30','cp0930'),('11:30','cp1130'),('13:00','cp1300'),
           ('14:30','cp1430'),('15:00','cp1500'),('15:30','cp1530')]

    # セクター別に出力
    for sect, syms_in_sect in [("非鉄", [s for s in ALL_SYMS if SECTOR[s]=="非鉄"]),
                                ("半導体",[s for s in ALL_SYMS if SECTOR[s]=="半導体"])]:
        print(f"\n  ─ {sect} ─")
        print(f"  {'銘柄':<12}", end="")
        for label, _ in cps:
            print(f"  {label}", end="")
        print()
        print("  " + "-"*60)
        for sym in syms_in_sect:
            name = ALL_SYMS[sym]
            print(f"  {name:<12}", end="")
            for label, col in cps:
                fv = fri[sym][col].mean()
                nv = non[sym][col].mean()
                diff = fv - nv
                print(f"  {fv:>+5.3f}({diff:>+4.2f})", end="")
            print()

    # ─────────────────────────────────────────
    # 3. 引け前30分（15:00→15:30）ランキング
    # ─────────────────────────────────────────
    print("\n" + "="*72)
    print("【金曜 引け前30分（15:00→15:30）】  持ち越すか手仕舞うか")
    print("="*72)
    print(f"  {'銘柄':<12} {'セクター':>5}  {'平均':>7}  {'σ':>6}  {'勝率':>6}  {'vs月〜木':>8}")
    print("  " + "-"*55)

    last_rows = []
    for sym, name in ALL_SYMS.items():
        f = fri[sym]['last30'].dropna()
        n = non[sym]['last30'].dropna()
        if len(f) < 5: continue
        last_rows.append({'name':name,'sect':SECTOR[sym],'mean':f.mean(),
                          'std':f.std(),'wr':(f>0).mean()*100,'vs':f.mean()-n.mean()})
    last_rows.sort(key=lambda r: r['mean'], reverse=True)

    for r in last_rows:
        marker = "▲買い継続" if r['mean']>0.05 else ("▼手仕舞い" if r['mean']<-0.05 else "  中立")
        print(f"  {r['name']:<12} {r['sect']:>5}  {r['mean']:>+6.3f}%  {r['std']:>6.3f}%  {r['wr']:>5.1f}%  {r['vs']:>+7.3f}%  {marker}")

    # ─────────────────────────────────────────
    # 4. 金曜の前場/後場パターン分布
    # ─────────────────────────────────────────
    print("\n" + "="*72)
    print("【金曜 前場/後場パターン分布】")
    print("="*72)
    print(f"  {'銘柄':<12}  {'前高後高':>10}  {'前高後安':>10}  {'前安後高':>10}  {'前安後安':>10}")
    print("  " + "-"*58)

    for sym, name in ALL_SYMS.items():
        f = fri[sym].dropna(subset=['mae_ret','aft_ret'])
        if len(f) < 5: continue
        hh = (f['mae_ret']>0)&(f['aft_ret']>0)
        hl = (f['mae_ret']>0)&(f['aft_ret']<0)
        lh = (f['mae_ret']<0)&(f['aft_ret']>0)
        ll = (f['mae_ret']<0)&(f['aft_ret']<0)
        def fmt(mask):
            n = mask.sum()
            avg = f.loc[mask,'day_ret'].mean() if n>0 else np.nan
            return f"{n}件{avg:>+5.2f}%" if n>0 and not np.isnan(avg) else "  0件"
        print(f"  {name:<12}  {fmt(hh):>10}  {fmt(hl):>10}  {fmt(lh):>10}  {fmt(ll):>10}")

    print("\n分析完了!")
