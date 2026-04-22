#!/usr/bin/env python3
"""
13戦略の一括バックテスト
- 対象: 非鉄 8 + 半導体 9 + 通信 2 = 19 銘柄
- 1分足、17ヶ月 (2024-11 ~ 2026-04)
- コスト: 片側 4 bps (single-stock 往復 8 bps / pair 16 bps)
"""
import warnings, sys
import numpy as np
import pandas as pd
from lib_data import load_all, ALL_SYMBOLS, SECTOR, NONFERROUS, SEMI, perf, print_perf

warnings.filterwarnings("ignore")

COST_SINGLE = 8.0   # bps, 往復
COST_PAIR = 16.0

# ---------- 共通ユーティリティ ----------
def resample_1min_to_daily_sessions(df):
    """日次にグループ化し、各日の重要バーを抽出"""
    g = df.groupby(df.index.date)
    return g


def intraday_slice(day_df, start_h, start_m, end_h, end_m):
    idx = day_df.index
    h, m = idx.hour, idx.minute
    mask = ((h > start_h) | ((h == start_h) & (m >= start_m))) & \
           ((h < end_h) | ((h == end_h) & (m <= end_m)))
    return day_df[mask]


# ---------- 戦略 1-3: ORB Breakout ----------
def strat_orb(df, minutes, side="both"):
    """寄付Nminレンジブレイクアウト → 引けまで保有"""
    pnls = []
    for dt, g in df.groupby(df.index.date):
        if len(g) < 50: continue
        open_t = g.index[0]
        range_end = open_t + pd.Timedelta(minutes=minutes)
        range_df = g[g.index < range_end]
        if len(range_df) < 3: continue
        rh, rl = range_df['high'].max(), range_df['low'].min()
        after = g[g.index >= range_end]
        if len(after) < 5: continue
        close = after['close'].iloc[-1]
        # どちらが先にブレイクしたか
        hi_break = after[after['high'] > rh]
        lo_break = after[after['low'] < rl]
        first_hi = hi_break.index[0] if len(hi_break) else None
        first_lo = lo_break.index[0] if len(lo_break) else None
        if first_hi and (not first_lo or first_hi < first_lo):
            if side in ("both", "long"):
                entry = rh
                pnls.append((close/entry - 1)*10000 - COST_SINGLE)
        elif first_lo and (not first_hi or first_lo < first_hi):
            if side in ("both", "short"):
                entry = rl
                pnls.append((entry/close - 1)*10000 - COST_SINGLE)
    return np.array(pnls)


# ---------- 戦略 4: Gap Fade ----------
def strat_gap_fade(df, thresh_bps=100, hold_min=30):
    """寄付ギャップ > Xbps → 逆方向に寄成エントリ、N分後決済"""
    pnls = []
    prev_close = None
    for dt, g in df.groupby(df.index.date):
        if len(g) < 30: continue
        open_p = g['open'].iloc[0]
        if prev_close is not None:
            gap_bps = (open_p/prev_close - 1) * 10000
            if abs(gap_bps) > thresh_bps:
                exit_t = g.index[0] + pd.Timedelta(minutes=hold_min)
                exit_df = g[g.index >= exit_t]
                if len(exit_df):
                    exit_p = exit_df['close'].iloc[0]
                    # gap positive → fade short
                    pnl = (open_p/exit_p - 1)*10000 if gap_bps > 0 else (exit_p/open_p - 1)*10000
                    pnls.append(pnl - COST_SINGLE)
        prev_close = g['close'].iloc[-1]
    return np.array(pnls)


# ---------- 戦略 5: Opening Spike Fade ----------
def strat_opening_spike_fade(df, spike_min=15, thresh_bps=80, hold_min=60):
    """寄付後Nminの動き > Xbps → 逆方向に、M分後決済"""
    pnls = []
    for dt, g in df.groupby(df.index.date):
        if len(g) < 60: continue
        open_p = g['open'].iloc[0]
        spike_end = g.index[0] + pd.Timedelta(minutes=spike_min)
        spike_df = g[g.index < spike_end]
        if len(spike_df) < 3: continue
        spike_p = spike_df['close'].iloc[-1]
        spike_ret = (spike_p/open_p - 1) * 10000
        if abs(spike_ret) < thresh_bps: continue
        exit_t = g.index[0] + pd.Timedelta(minutes=spike_min + hold_min)
        exit_df = g[g.index >= exit_t]
        if not len(exit_df): continue
        exit_p = exit_df['close'].iloc[0]
        pnl = (spike_p/exit_p - 1)*10000 if spike_ret > 0 else (exit_p/spike_p - 1)*10000
        pnls.append(pnl - COST_SINGLE)
    return np.array(pnls)


# ---------- 戦略 6: Lunch Reversal ----------
def strat_lunch_reversal(df, thresh_bps=30):
    """前場終値 → 後場寄付のギャップをフェード、後場引けまで"""
    pnls = []
    for dt, g in df.groupby(df.index.date):
        morning = g[(g.index.hour < 11) | ((g.index.hour == 11) & (g.index.minute <= 30))]
        afternoon = g[g.index.hour >= 12]
        if len(morning) < 20 or len(afternoon) < 20: continue
        m_close = morning['close'].iloc[-1]
        a_open = afternoon['open'].iloc[0]
        a_close = afternoon['close'].iloc[-1]
        lunch_gap = (a_open/m_close - 1) * 10000
        if abs(lunch_gap) < thresh_bps: continue
        pnl = (a_open/a_close - 1)*10000 if lunch_gap > 0 else (a_close/a_open - 1)*10000
        pnls.append(pnl - COST_SINGLE)
    return np.array(pnls)


# ---------- 戦略 7/8: Morning → Afternoon ----------
def strat_morning_afternoon(df, thresh_bps=50, follow=True):
    """前場リターン > X → 後場を順張り(follow)/逆張り"""
    pnls = []
    for dt, g in df.groupby(df.index.date):
        morning = g[(g.index.hour < 11) | ((g.index.hour == 11) & (g.index.minute <= 30))]
        afternoon = g[g.index.hour >= 12]
        if len(morning) < 20 or len(afternoon) < 20: continue
        m_open = morning['open'].iloc[0]
        m_close = morning['close'].iloc[-1]
        a_open = afternoon['open'].iloc[0]
        a_close = afternoon['close'].iloc[-1]
        m_ret = (m_close/m_open - 1) * 10000
        if abs(m_ret) < thresh_bps: continue
        a_ret = (a_close/a_open - 1) * 10000
        if follow:
            pnl = a_ret if m_ret > 0 else -a_ret
        else:
            pnl = -a_ret if m_ret > 0 else a_ret
        pnls.append(pnl - COST_SINGLE)
    return np.array(pnls)


# ---------- 戦略 9/10: Last Hour ----------
def strat_last_hour(df, thresh_bps=50, follow=True):
    """14:30 時点当日リターン → 引けまで順/逆"""
    pnls = []
    for dt, g in df.groupby(df.index.date):
        if len(g) < 100: continue
        open_p = g['open'].iloc[0]
        close_p = g['close'].iloc[-1]
        # 14:30 近辺の値
        target = [(h, m) for (h, m) in [(14, 30), (14, 31), (14, 32)]]
        p1430 = None
        for h, m in target:
            sel = g[(g.index.hour == h) & (g.index.minute == m)]
            if len(sel):
                p1430 = sel['close'].iloc[0]
                break
        if p1430 is None: continue
        day_ret = (p1430/open_p - 1) * 10000
        if abs(day_ret) < thresh_bps: continue
        last_ret = (close_p/p1430 - 1) * 10000
        if follow:
            pnl = last_ret if day_ret > 0 else -last_ret
        else:
            pnl = -last_ret if day_ret > 0 else last_ret
        pnls.append(pnl - COST_SINGLE)
    return np.array(pnls)


# ---------- 戦略 11: VWAP Mean Reversion ----------
def strat_vwap_mr(df, check_hours=[10, 11, 13, 14], thresh_sigma=1.5, hold_min=30):
    """指定時刻で VWAP 乖離 > σ×K → 逆張り、Nmin保有"""
    pnls = []
    for dt, g in df.groupby(df.index.date):
        if len(g) < 60: continue
        tp = (g['high'] + g['low'] + g['close']) / 3
        vwap = (tp * g['volume']).cumsum() / g['volume'].cumsum().replace(0, np.nan)
        dev = g['close'] - vwap
        roll_sd = dev.rolling(30, min_periods=10).std()
        for ch in check_hours:
            sel = g[(g.index.hour == ch) & (g.index.minute == 0)]
            if not len(sel): continue
            ts = sel.index[0]
            cp = sel['close'].iloc[0]
            vp = vwap.loc[ts]
            sd = roll_sd.loc[ts]
            if pd.isna(sd) or sd == 0: continue
            z = (cp - vp) / sd
            if abs(z) < thresh_sigma: continue
            exit_t = ts + pd.Timedelta(minutes=hold_min)
            ex = g[g.index >= exit_t]
            if not len(ex): continue
            ep = ex['close'].iloc[0]
            # 正乖離なら short (下落で利益)
            pnl = (cp/ep - 1)*10000 if z > 0 else (ep/cp - 1)*10000
            pnls.append(pnl - COST_SINGLE)
    return np.array(pnls)


# ---------- 戦略 12: Intraday Z Reversal (5分リサンプル) ----------
def strat_intraday_z(df, zwin=30, z_thresh=2.0, hold_min=15):
    """5分足 Z-score (対当日リターン) > K → 逆張り、Nmin保有"""
    pnls = []
    for dt, g in df.groupby(df.index.date):
        if len(g) < 80: continue
        r5 = g['close'].resample('5T').last().dropna()
        if len(r5) < zwin + 5: continue
        ret = r5.pct_change()
        z = (ret - ret.rolling(zwin).mean()) / ret.rolling(zwin).std()
        signals = z[z.abs() > z_thresh]
        hold_bars = max(1, hold_min // 5)
        for ts, zv in signals.items():
            try:
                i = r5.index.get_loc(ts)
            except KeyError:
                continue
            if i + hold_bars >= len(r5): continue
            ep = r5.iloc[i]
            xp = r5.iloc[i + hold_bars]
            pnl = (ep/xp - 1)*10000 if zv > 0 else (xp/ep - 1)*10000
            pnls.append(pnl - COST_SINGLE)
    return np.array(pnls)


# ---------- 戦略 13: Sector Dispersion Catch-up ----------
def strat_sector_dispersion(data_dict, sector_name, check_time=(11, 0),
                             thresh_bps=100, hold_to_close=True):
    """指定時刻でセクター平均 vs 個別銘柄の乖離 > X → 平均に収束する方向"""
    members = [s for s, sec in SECTOR.items() if sec == sector_name and s in data_dict]
    if len(members) < 3: return np.array([])
    pnls = []
    # 全日付 union
    all_dates = set()
    for s in members:
        all_dates.update(data_dict[s].index.date)
    for dt in sorted(all_dates):
        rets_at_check = {}
        price_at_check = {}
        close_prices = {}
        for s in members:
            g = data_dict[s][data_dict[s].index.date == dt]
            if len(g) < 80: continue
            op = g['open'].iloc[0]
            sel = g[(g.index.hour == check_time[0]) & (g.index.minute == check_time[1])]
            if not len(sel): continue
            cp = sel['close'].iloc[0]
            rets_at_check[s] = (cp/op - 1) * 10000
            price_at_check[s] = cp
            close_prices[s] = g['close'].iloc[-1]
        if len(rets_at_check) < 3: continue
        mean_ret = np.mean(list(rets_at_check.values()))
        for s, r in rets_at_check.items():
            dev = r - mean_ret
            if abs(dev) < thresh_bps: continue
            ep = price_at_check[s]
            xp = close_prices[s]
            # 平均より大きく買われている → short で収束狙い
            pnl = (ep/xp - 1)*10000 if dev > 0 else (xp/ep - 1)*10000
            pnls.append(pnl - COST_SINGLE)
    return np.array(pnls)


# ---------- メイン ----------
def main():
    print("=" * 130)
    print(f"非鉄・半導体・通信 イントラ戦略バックテスト   対象: {len(ALL_SYMBOLS)} 銘柄")
    print("=" * 130)
    print("データロード中 ...")
    data = load_all(ALL_SYMBOLS)
    for s in ALL_SYMBOLS:
        print(f"  {s} ({SECTOR[s]:<10}): {len(data[s]):>7} bars  "
              f"{data[s].index.min().date()} ~ {data[s].index.max().date()}")

    rows = []

    def run_single(strategy_fn, label, **kwargs):
        """各銘柄で戦略を回し、全プール"""
        all_pnl = []
        for s in ALL_SYMBOLS:
            p = strategy_fn(data[s], **kwargs)
            if len(p): all_pnl.extend(p.tolist())
        rows.append(perf(np.array(all_pnl), label=label))

    print("\n[1] ORB 15min (both sides) ...")
    run_single(strat_orb, "ORB15 both", minutes=15, side="both")
    print("[2] ORB 30min (both sides) ...")
    run_single(strat_orb, "ORB30 both", minutes=30, side="both")
    print("[3] ORB 60min (both sides) ...")
    run_single(strat_orb, "ORB60 both", minutes=60, side="both")
    print("[4] Gap Fade (>100bps, 30min hold) ...")
    run_single(strat_gap_fade, "Gap Fade 100bps/30m", thresh_bps=100, hold_min=30)
    print("[4b] Gap Fade (>50bps, 60min hold) ...")
    run_single(strat_gap_fade, "Gap Fade 50bps/60m", thresh_bps=50, hold_min=60)
    print("[5] Opening Spike Fade (15min/80bps/60m) ...")
    run_single(strat_opening_spike_fade, "OpenSpikeFade 15/80/60",
               spike_min=15, thresh_bps=80, hold_min=60)
    print("[6] Lunch Reversal ...")
    run_single(strat_lunch_reversal, "Lunch Reversal 30bps", thresh_bps=30)
    print("[7] Morning→Afternoon FOLLOW ...")
    run_single(strat_morning_afternoon, "M→A Follow 50bps",
               thresh_bps=50, follow=True)
    print("[8] Morning→Afternoon REVERSAL ...")
    run_single(strat_morning_afternoon, "M→A Reverse 50bps",
               thresh_bps=50, follow=False)
    print("[9] Last Hour Momentum ...")
    run_single(strat_last_hour, "LastHour Follow 50bps",
               thresh_bps=50, follow=True)
    print("[10] Last Hour Reversal ...")
    run_single(strat_last_hour, "LastHour Reverse 50bps",
               thresh_bps=50, follow=False)
    print("[11] VWAP Mean Reversion ...")
    run_single(strat_vwap_mr, "VWAP MR σ1.5/30m",
               check_hours=[10, 11, 13, 14], thresh_sigma=1.5, hold_min=30)
    print("[12] Intraday Z Reversal (5min) ...")
    run_single(strat_intraday_z, "Intraday Z5m >2σ/15m",
               zwin=30, z_thresh=2.0, hold_min=15)
    print("[13a] Sector Dispersion (Semi, 11:00) ...")
    p = strat_sector_dispersion(data, "semi", check_time=(11, 0), thresh_bps=100)
    rows.append(perf(p, label="Disp Semi 11:00/100bps"))
    print("[13b] Sector Dispersion (NonFerrous, 11:00) ...")
    p = strat_sector_dispersion(data, "nonferrous", check_time=(11, 0), thresh_bps=100)
    rows.append(perf(p, label="Disp NonFerrous 11:00/100bps"))

    print("\n" + "=" * 130)
    print("結果サマリ (コスト差引後 bps, 全銘柄プール)")
    print("=" * 130)
    print_perf(rows)

    # 上位のみハイライト
    print("\n" + "=" * 130)
    print("Sharpe 上位 5")
    print("=" * 130)
    valid = [r for r in rows if not np.isnan(r["sharpe"])]
    top = sorted(valid, key=lambda r: r["sharpe"], reverse=True)[:5]
    print_perf(top)

    # 保存
    df_out = pd.DataFrame(rows)
    df_out.to_csv("strategies_summary.csv", index=False)
    print("\n→ strategies_summary.csv 保存")


if __name__ == "__main__":
    main()
