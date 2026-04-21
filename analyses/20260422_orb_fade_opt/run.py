#!/usr/bin/env python3
"""
ORB Fade (半導体) 条件最適化

ベース戦略: 9:00-9:30 の High/Low ブレイクを「逆張り」、指定保有期間で決済。
既存検証 (semi_intraday_battery) で Pooled Sharpe+1.01 / t+2.15 判明。
採用基準 Sharpe≥2 に乗せるためのパラメータ探索:

  (A) Opening Range 幅層別     (OR幅 bps ≥ 閾値)
  (B) 時間帯フィルタ            (9:30〜X時 のブレイクのみ)
  (C) 保有期間                  (+30min / +60min / +90min / 引けまで)
  (D) H1/H2 OoS検証 (最良config)

採用基準: Sharpe≥2.0 & N≥30 & t-stat≥2.0
コスト: 4bps
"""
import sys, os
from datetime import time as dtime
import numpy as np
import pandas as pd
import psycopg2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['Hiragino Sans','Arial Unicode MS','sans-serif']

PG = dict(host='localhost', port=5432, user='postgres', dbname='market_data')
SEMI = ['6857.T','6920.T','6146.T','6503.T','8035.T']
COST_BPS = 4.0
OUT = os.path.dirname(os.path.abspath(__file__))

def fetch(sym):
    conn = psycopg2.connect(**PG)
    df = pd.read_sql("SELECT timestamp, open, high, low, close, volume FROM intraday_data WHERE symbol=%s ORDER BY timestamp",
                     conn, params=[sym])
    conn.close()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['jst'] = df['timestamp'] + pd.Timedelta(hours=9)
    df = df.set_index('jst').sort_index()
    df['date'] = df.index.date
    return df


def stats(r):
    r = np.asarray(r); r = r[~np.isnan(r)]; n = len(r)
    if n < 2: return dict(n=n, sharpe=np.nan, tstat=np.nan, mean=np.nan, wr=np.nan)
    m, s = r.mean(), r.std()
    return dict(n=n, mean=m, std=s, wr=(r>0).mean()*100,
                sharpe=(m/s)*np.sqrt(252) if s>0 else 0,
                tstat=(m/(s/np.sqrt(n))) if s>0 else 0)


def session_bars(df, d):
    day = df[df['date'] == d]
    h = day.index.hour; mm = day.index.minute
    mask = ((h==9) | (h==10) | ((h==11)&(mm<=30)) | ((h==12)&(mm>=30)) |
            (h==13) | (h==14) | ((h==15)&(mm<=30)))
    return day[mask]


def build_trades(df):
    """全日分の ORB Fade トレード候補を生成。
    各日: (date, or_high, or_low, or_range_bps, break_time, direction, entry_price, bar_series)
    direction: +1 なら上方ブレイク (fade→ Short), -1 なら下方ブレイク (fade→ Long)
    """
    trades = []
    dates = sorted(df['date'].unique())
    for d in dates:
        day = session_bars(df, d)
        if len(day) < 60: continue
        open_range = day[day.index < (pd.Timestamp(d) + pd.Timedelta(hours=9, minutes=30))]
        if len(open_range) < 5: continue
        or_high = open_range['high'].max()
        or_low = open_range['low'].min()
        or_mid = (or_high + or_low) / 2
        or_range_bps = (or_high / or_low - 1) * 10000
        rest = day[day.index >= (pd.Timestamp(d) + pd.Timedelta(hours=9, minutes=30))]
        if len(rest) < 20: continue
        # 最初のブレイク検出
        break_ts = None; direction = 0; entry_price = np.nan
        for ts, row in rest.iterrows():
            if row['high'] > or_high:
                break_ts = ts; direction = +1; entry_price = or_high; break
            if row['low'] < or_low:
                break_ts = ts; direction = -1; entry_price = or_low; break
        if break_ts is None: continue
        trades.append(dict(date=d, or_high=or_high, or_low=or_low,
                           or_range_bps=or_range_bps, break_ts=break_ts,
                           direction=direction, entry_price=entry_price,
                           bars=rest))
    return trades


def eval_trades(trades, or_range_min_bps=0, break_cutoff_hour=15, hold_minutes=None,
                use_tp=True, stop_mult=1.0):
    """
    Fade戦略の正しい実装:
    - Up breakout (direction=+1) → Short at or_high
        PT = or_low (反対OR端 = 戻り狙い)
        Stop = or_high + stop_mult × or_width
    - Down breakout (direction=-1) → Long at or_low
        PT = or_high
        Stop = or_low - stop_mult × or_width
    - Time exit: hold_minutes 経過 or 引け
    - イベント優先順位: TP → Stop → Time
    """
    rets = []
    for t in trades:
        if t['or_range_bps'] < or_range_min_bps: continue
        if t['break_ts'].hour >= break_cutoff_hour: continue
        direction = t['direction']  # +1=上ブレイク, -1=下ブレイク
        fade_dir = -direction
        entry = t['entry_price']
        or_high, or_low = t['or_high'], t['or_low']
        or_width = or_high - or_low
        if direction == 1:  # fade Short
            tp = or_low
            stop = or_high + stop_mult * or_width
        else:               # fade Long
            tp = or_high
            stop = or_low - stop_mult * or_width

        bars = t['bars']
        after = bars[bars.index >= t['break_ts']].copy()
        if after.empty: continue
        time_limit = (t['break_ts'] + pd.Timedelta(minutes=hold_minutes)) if hold_minutes else after.index[-1]
        exit_p = None
        for ts, row in after.iterrows():
            if ts > time_limit: break
            # Check TP then stop (fade Short: tp is BELOW entry, stop ABOVE)
            if fade_dir == -1:  # Short
                if use_tp and row['low'] <= tp:  exit_p = tp; break
                if row['high'] >= stop:           exit_p = stop; break
            else:               # Long
                if use_tp and row['high'] >= tp: exit_p = tp; break
                if row['low'] <= stop:           exit_p = stop; break
        if exit_p is None:
            # time exit
            within = after[after.index <= time_limit]
            exit_p = float(within['close'].iloc[-1]) if not within.empty else float(after['close'].iloc[0])
        ret_bps = fade_dir * (exit_p/entry - 1) * 10000 - COST_BPS
        rets.append((t['date'], ret_bps))
    return rets


def run_grid(stock_trades):
    """パラメータグリッド探索"""
    or_min_grid = [0, 50, 100, 150, 200]        # OR幅 bps 最小
    cutoff_grid = [11, 13, 15]                  # ブレイク受付 ≤ X時
    hold_grid = [30, 60, 90, None]              # 分 / None=引けまで
    rows = []
    for om in or_min_grid:
        for ch in cutoff_grid:
            for hm in hold_grid:
                pooled = []
                per_stock = {}
                for sym, tr in stock_trades.items():
                    rs = eval_trades(tr, or_range_min_bps=om, break_cutoff_hour=ch, hold_minutes=hm)
                    arr = np.array([r[1] for r in rs])
                    s = stats(arr); per_stock[sym] = s
                    pooled.extend(arr)
                ps = stats(np.array(pooled))
                rows.append(dict(or_min=om, cutoff=ch, hold=(hm if hm else 'close'),
                                 n=ps['n'], sharpe=ps['sharpe'], tstat=ps['tstat'],
                                 mean=ps['mean'], wr=ps['wr']))
    return pd.DataFrame(rows)


def run_h1h2(stock_trades, or_min, cutoff, hold):
    """最良configで H1/H2 時系列OoS"""
    all_rows = []
    for sym, tr in stock_trades.items():
        rs = eval_trades(tr, or_range_min_bps=or_min, break_cutoff_hour=cutoff, hold_minutes=hold)
        for d, r in rs:
            all_rows.append((sym, d, r))
    df = pd.DataFrame(all_rows, columns=['sym','date','ret_bps']).sort_values('date')
    n = len(df); mid = n // 2
    h1 = df.iloc[:mid]['ret_bps'].values
    h2 = df.iloc[mid:]['ret_bps'].values
    full = df['ret_bps'].values
    return df, stats(full), stats(h1), stats(h2)


def main():
    print("="*90)
    print("ORB Fade (半導体) 条件最適化 + OoS検証")
    print("="*90)
    stock_trades = {}
    for sym in SEMI:
        df = fetch(sym)
        tr = build_trades(df)
        print(f"  {sym}: {len(tr)} ORB break days")
        stock_trades[sym] = tr

    # Grid search
    print("\n[Grid search] OR幅 × ブレイク時刻上限 × 保有期間")
    grid = run_grid(stock_trades)
    grid = grid.sort_values('sharpe', ascending=False).reset_index(drop=True)
    grid.to_csv(os.path.join(OUT, 'grid.csv'), index=False)

    print("\nTop 15 configs (by Sharpe):")
    print(grid.head(15).to_string(index=False))

    # 採用基準をクリアするconfig抽出
    passed = grid[(grid['sharpe'] >= 2) & (grid['n'] >= 30) & (grid['tstat'] >= 2)]
    print(f"\n採用基準クリア config数: {len(passed)}")
    if not passed.empty:
        print(passed.head(10).to_string(index=False))

    # Top config で OoS検証
    best = grid.iloc[0]
    or_min = int(best['or_min']); cutoff = int(best['cutoff'])
    hold = None if best['hold']=='close' else int(best['hold'])
    print(f"\n" + "="*90)
    print(f"[OoS検証] Top config: OR≥{or_min}bps, Cutoff≤{cutoff}時, Hold={best['hold']}")
    print("="*90)
    df_all, sf, sh1, sh2 = run_h1h2(stock_trades, or_min, cutoff, hold)
    def ptr(label, s):
        pass_ = (s['sharpe']>=2 and s['n']>=30 and s['tstat']>=2)
        print(f"  {label:20} N={s['n']:>4}  Sharpe={s['sharpe']:+5.2f}  t={s['tstat']:+5.2f}  "
              f"mean={s['mean']:+6.2f}bp  WR={s['wr']:>5.1f}%  → {'✅ PASS' if pass_ else '❌'}")
    ptr("Full", sf); ptr("H1 (前半)", sh1); ptr("H2 (後半=OoS)", sh2)

    # 銘柄別成績 (best config)
    print(f"\n[銘柄別 best config 成績]")
    for sym, tr in stock_trades.items():
        rs = eval_trades(tr, or_range_min_bps=or_min, break_cutoff_hour=cutoff, hold_minutes=hold)
        s = stats(np.array([r[1] for r in rs]))
        pass_ = (s['sharpe']>=2 and s['n']>=30 and s['tstat']>=2)
        print(f"  {sym}: N={s['n']:>4}  Sharpe={s['sharpe']:+5.2f}  t={s['tstat']:+5.2f}  "
              f"mean={s['mean']:+6.2f}bp  {'⭐PASS' if pass_ else ''}")

    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    # (A) heatmap: sharpe by (or_min, hold) at cutoff=15
    sub = grid[grid['cutoff'] == 15].copy()
    pivot = sub.pivot(index='or_min', columns='hold', values='sharpe')
    cols_order = [30, 60, 90, 'close']
    pivot = pivot[[c for c in cols_order if c in pivot.columns]]
    im = axes[0,0].imshow(pivot.values, aspect='auto', cmap='RdBu_r', vmin=-2, vmax=2)
    axes[0,0].set_xticks(range(len(pivot.columns))); axes[0,0].set_xticklabels(pivot.columns)
    axes[0,0].set_yticks(range(len(pivot.index))); axes[0,0].set_yticklabels(pivot.index)
    axes[0,0].set_xlabel('Hold'); axes[0,0].set_ylabel('OR_min (bps)')
    axes[0,0].set_title('Pooled Sharpe (cutoff=15)')
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i,j]
            axes[0,0].text(j, i, f"{v:+.2f}", ha='center', va='center', fontsize=9)
    plt.colorbar(im, ax=axes[0,0])

    # (B) cumulative by best config (pooled)
    cum = np.cumsum(df_all.sort_values('date')['ret_bps'].values)
    axes[0,1].plot(cum, color='navy')
    axes[0,1].axhline(0, color='black', lw=0.5)
    axes[0,1].set_title(f"Cumulative (best config) N={len(df_all)}")
    axes[0,1].set_xlabel('Trade #'); axes[0,1].set_ylabel('Cumulative bps')
    axes[0,1].grid(alpha=0.3)

    # (C) per-stock sharpe at best config
    per_stock = {}
    for sym, tr in stock_trades.items():
        rs = eval_trades(tr, or_range_min_bps=or_min, break_cutoff_hour=cutoff, hold_minutes=hold)
        per_stock[sym] = stats(np.array([r[1] for r in rs]))
    names = list(per_stock.keys()); sharpes = [per_stock[n]['sharpe'] for n in names]
    colors = ['green' if s>=2 else 'skyblue' if s>0 else 'salmon' for s in sharpes]
    axes[1,0].bar(names, sharpes, color=colors)
    axes[1,0].axhline(2, color='red', ls='--', label='採用基準+2')
    axes[1,0].set_ylabel('Sharpe'); axes[1,0].set_title('銘柄別 Sharpe (best config)')
    for i, s in enumerate(sharpes):
        axes[1,0].text(i, s+0.05, f"{s:+.2f}\nN={per_stock[names[i]]['n']}", ha='center', fontsize=8)
    axes[1,0].legend()

    # (D) H1/H2
    h1c = np.cumsum(df_all.sort_values('date').iloc[:len(df_all)//2]['ret_bps'].values)
    h2c = np.cumsum(df_all.sort_values('date').iloc[len(df_all)//2:]['ret_bps'].values)
    axes[1,1].plot(h1c, label=f'H1 Sharpe={sh1["sharpe"]:+.2f}', color='orange')
    axes[1,1].plot(h2c, label=f'H2 Sharpe={sh2["sharpe"]:+.2f}', color='navy')
    axes[1,1].axhline(0, color='black', lw=0.5)
    axes[1,1].set_title(f'H1/H2 OoS split (best config)')
    axes[1,1].set_xlabel('Trade #'); axes[1,1].set_ylabel('Cumulative bps')
    axes[1,1].legend(); axes[1,1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'opt_result.png'), dpi=120, bbox_inches='tight')
    print(f"\nSaved: {os.path.join(OUT, 'opt_result.png')}")


if __name__ == '__main__':
    main()
