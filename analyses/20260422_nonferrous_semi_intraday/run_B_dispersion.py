#!/usr/bin/env python3
"""
B. 非鉄セクター分散取引 深掘り (最適化版)
 - 各銘柄の日次 OHLC + 指定時刻での価格を事前計算
 - パラメータスキャン時は pandas 上でベクトル演算
"""
import warnings
import numpy as np
import pandas as pd
from lib_data import load_all, NONFERROUS, perf, print_perf

warnings.filterwarnings("ignore")
COST = 8.0


def build_daily_grid(data, members, check_times, extra_exits):
    """date × symbol × field の panel を構築"""
    frames = []
    for s in members:
        df = data[s]
        by = df.groupby(df.index.date)
        rec = {}
        for dt, g in by:
            if len(g) < 80: continue
            row = {"open": g['open'].iloc[0], "close": g['close'].iloc[-1]}
            for ct in check_times:
                sel = g[(g.index.hour == ct[0]) & (g.index.minute == ct[1])]
                row[f"p_{ct[0]:02d}{ct[1]:02d}"] = sel['close'].iloc[0] if len(sel) else np.nan
            for et in extra_exits:
                if et is None: continue
                sel = g[(g.index.hour == et[0]) & (g.index.minute == et[1])]
                row[f"x_{et[0]:02d}{et[1]:02d}"] = sel['close'].iloc[0] if len(sel) else np.nan
            rec[dt] = row
        f = pd.DataFrame(rec).T
        f.columns = pd.MultiIndex.from_product([[s], f.columns])
        frames.append(f)
    return pd.concat(frames, axis=1).sort_index()


def run_dispersion(panel, members, check_time, thresh_bps, exit_col="close"):
    ct_col = f"p_{check_time[0]:02d}{check_time[1]:02d}"
    opens = pd.concat([panel[(s, "open")] for s in members], axis=1); opens.columns = members
    checks = pd.concat([panel[(s, ct_col)] for s in members], axis=1); checks.columns = members
    exits = pd.concat([panel[(s, exit_col)] for s in members], axis=1); exits.columns = members

    rets = (checks / opens - 1) * 10000   # bps at check time
    mean_ret = rets.mean(axis=1, skipna=True)
    dev = rets.sub(mean_ret, axis=0)
    # シグナル: |dev| > thresh
    mask = dev.abs() > thresh_bps
    # P&L: 上に乖離 (dev>0) ならshort → (check/exit - 1). 下ならlong → (exit/check - 1)
    pnl_short = (checks / exits - 1) * 10000
    pnl_long = (exits / checks - 1) * 10000
    pnl = pnl_short.where(dev > 0, pnl_long)
    pnl = pnl.where(mask)
    # コスト
    pnl = pnl - COST
    arr = pnl.stack().dropna().values
    return arr


def main():
    print("=" * 120)
    print("B. 非鉄分散取引 深掘り (最適化)")
    print("=" * 120)
    print("ロード中 ...")
    data = load_all(NONFERROUS)
    for s in NONFERROUS:
        print(f"  {s}: {len(data[s])} bars")

    check_times = [(10, 0), (10, 30), (11, 0), (11, 30), (13, 0), (13, 30), (14, 0)]
    extra_exits = [(11, 30), (13, 0), (13, 30), (14, 0), (14, 30), (15, 29)]
    print("\n日次グリッド構築中 ...")
    panel = build_daily_grid(data, NONFERROUS, check_times, extra_exits)
    print(f"  日数 = {len(panel)}")

    # ---- スキャン1 ----
    print("\n【時刻 × 乖離閾値スキャン (引けまで保有)】")
    rows = []
    for ct in check_times:
        for th in [50, 80, 100, 150, 200]:
            arr = run_dispersion(panel, NONFERROUS, ct, th, exit_col="close")
            rows.append(perf(arr, label=f"{ct[0]:02d}:{ct[1]:02d}  thr={th:>3}bps"))
    print_perf(rows)

    print("\n【上位 10 (N≥200)】")
    valid = [r for r in rows if not np.isnan(r["sharpe"]) and r["N"] >= 200]
    top = sorted(valid, key=lambda r: r["sharpe"], reverse=True)[:10]
    print_perf(top)

    # ---- スキャン2: 保有時間 ----
    print("\n【保有時間バリエーション (ct=最良戦略)】")
    best = top[0] if top else None
    if best:
        ct_str = best["label"].split()[0]
        th = int(best["label"].split("=")[1].replace("bps", "").strip())
        h, m = int(ct_str.split(":")[0]), int(ct_str.split(":")[1])
        rows2 = []
        for et in extra_exits + [None]:
            exit_col = "close" if et is None else f"x_{et[0]:02d}{et[1]:02d}"
            label = "close" if et is None else f"{et[0]:02d}:{et[1]:02d}"
            arr = run_dispersion(panel, NONFERROUS, (h, m), th, exit_col=exit_col)
            rows2.append(perf(arr, label=f"{h:02d}:{m:02d} → {label}"))
        print_perf(rows2)

        # 銘柄別寄与
        print("\n【銘柄別寄与 (最良戦略)】")
        ct_col = f"p_{h:02d}{m:02d}"
        opens = pd.concat([panel[(s, "open")] for s in NONFERROUS], axis=1); opens.columns = NONFERROUS
        checks = pd.concat([panel[(s, ct_col)] for s in NONFERROUS], axis=1); checks.columns = NONFERROUS
        exits = pd.concat([panel[(s, "close")] for s in NONFERROUS], axis=1); exits.columns = NONFERROUS
        rets = (checks/opens - 1)*10000
        mean_ret = rets.mean(axis=1, skipna=True)
        dev = rets.sub(mean_ret, axis=0)
        pnl_short = (checks/exits - 1)*10000
        pnl_long = (exits/checks - 1)*10000
        pnl = pnl_short.where(dev > 0, pnl_long)
        pnl = pnl.where(dev.abs() > th) - COST
        for s in NONFERROUS:
            x = pnl[s].dropna().values
            if len(x) < 5:
                print(f"  {s}: N={len(x)}")
                continue
            sh = x.mean()/x.std()*np.sqrt(252) if x.std()>0 else 0
            print(f"  {s}: N={len(x):4d}  mean={x.mean():+6.1f}  sum={x.sum():+7.0f}  Sharpe={sh:+.2f}")

    # ---- 複数時刻併用 ----
    print("\n【複数検査時刻併用 (thr=100bps、各時刻独立エントリ→引け)】")
    all_pnl = []
    for ct in [(10, 0), (11, 0), (13, 30), (14, 0)]:
        all_pnl.extend(run_dispersion(panel, NONFERROUS, ct, 100, exit_col="close").tolist())
    print_perf([perf(np.array(all_pnl), label="4時刻併用 thr=100")])
    # thr 低めで量を稼ぐ
    all_pnl = []
    for ct in [(10, 0), (11, 0), (13, 30), (14, 0)]:
        all_pnl.extend(run_dispersion(panel, NONFERROUS, ct, 80, exit_col="close").tolist())
    print_perf([perf(np.array(all_pnl), label="4時刻併用 thr=80")])


if __name__ == "__main__":
    main()
