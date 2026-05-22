"""
寄付ギャップフェード 深堀り

検証項目:
  1. ギャップ閾値スイープ (0.5%, 1.0%, 1.5%, 2.0%, 2.5%, 3.0%)
  2. 銘柄選別 (Top5 銘柄のみ採用) でポートフォリオSharpe向上するか
  3. 半期ごと安定性 (2025H1 / 2025H2 / 2026H1)
  4. 月次PnL分布
  5. PL画像 (1200x675px) 作成
"""

import os
import psycopg2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

PG_CONFIG = {
    "host": os.environ.get("PGHOST", "omen"),
    "port": int(os.environ.get("PGPORT", 5432)),
    "user": os.environ.get("PGUSER", "postgres"),
    "dbname": os.environ.get("PGDATABASE", "market_data"),
}

SYMBOLS = {
    "57060": "三井金属", "57110": "三菱マテリアル", "57130": "住友金属鉱山",
    "57140": "DOWA HD", "50160": "JX金属",
    "58010": "古河電工", "58020": "住友電工", "58030": "フジクラ",
}
START = "2025-04-01"
END = "2026-05-21"


def fetch_minute(code):
    conn = psycopg2.connect(**PG_CONFIG)
    sql = "SELECT ts, open, close FROM stocks_intraday WHERE code=%s AND ts>=%s AND ts<=%s ORDER BY ts"
    df = pd.read_sql(sql, conn, params=(code, START, END + " 23:59:59"))
    conn.close()
    df["ts"] = pd.to_datetime(df["ts"])
    return df.set_index("ts")


def fetch_daily_close(code):
    conn = psycopg2.connect(**PG_CONFIG)
    sql = "SELECT date, close FROM stocks_daily WHERE code=%s AND date BETWEEN %s AND %s ORDER BY date"
    df = pd.read_sql(sql, conn, params=(code, START, END))
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["close"].rename(code)


def gap_fade_trades(minute_df, prev_close, code, gap_thr=0.015, cost_oneway=0.0008):
    df = minute_df.copy()
    df["date"] = df.index.normalize()
    trades = []
    for d, g in df.groupby("date"):
        if d not in prev_close.index or pd.isna(prev_close.loc[d]):
            continue
        pc = prev_close.loc[d]
        am = g.between_time("09:00", "09:01")
        if len(am) == 0:
            continue
        open_px = am.iloc[0]["open"]
        gap = np.log(open_px / pc)
        if abs(gap) < gap_thr:
            continue
        side = -np.sign(gap)
        last = g.between_time("15:25", "15:30")
        if len(last) == 0:
            continue
        close_px = last.iloc[-1]["close"]
        r = np.log(close_px / open_px)
        pnl = r * side - cost_oneway * 2
        trades.append({"date": d, "code": code, "side": int(side), "gap_pct": gap * 100,
                       "entry": open_px, "exit": close_px, "pnl": pnl})
    return pd.DataFrame(trades)


def metrics(daily_series, ann=245):
    s = daily_series.dropna()
    if len(s) < 5:
        return {}
    mu, sd = s.mean(), s.std()
    sharpe = mu / sd * np.sqrt(ann) if sd > 0 else 0
    eq = s.cumsum()
    dd = (eq - eq.cummax()).min()
    wr = (s > 0).mean() * 100
    pf = (s[s > 0].sum() / -s[s < 0].sum()) if (s < 0).any() else np.inf
    return {"N": len(s), "mu%": round(mu*100, 4), "Sharpe": round(sharpe, 2),
            "WR%": round(wr, 1), "PF": round(pf, 2),
            "Cum%": round(eq.iloc[-1]*100, 1), "DD%": round(dd*100, 1)}


def pool_daily_pnl(trades_by_code):
    """銘柄別 trades を集約: 同日複数銘柄ある場合は均等分散"""
    pool = pd.concat(trades_by_code.values(), ignore_index=True)
    if len(pool) == 0:
        return pd.Series(dtype=float)
    pool["pnl_scaled"] = pool.groupby("date")["pnl"].transform(
        lambda s: s / max(len(s), 1))
    return pool.groupby("date")["pnl_scaled"].sum()


def main():
    print("=== 寄付ギャップフェード 深堀り ===\n")
    minutes = {c: fetch_minute(c) for c in SYMBOLS}
    daily_closes = {c: fetch_daily_close(c) for c in SYMBOLS}
    prev_closes = {c: daily_closes[c].shift(1) for c in SYMBOLS}

    # ============ (1) ギャップ閾値スイープ ============
    print("--- (1) ギャップ閾値スイープ (8 bps片道) ---")
    rows = []
    for thr in [0.005, 0.010, 0.015, 0.020, 0.025, 0.030]:
        trades_dict = {}
        for c in SYMBOLS:
            t = gap_fade_trades(minutes[c], prev_closes[c], c, gap_thr=thr, cost_oneway=0.0008)
            trades_dict[c] = t
        daily = pool_daily_pnl(trades_dict)
        m = metrics(daily)
        n_trades = sum(len(t) for t in trades_dict.values())
        rows.append({"gap_thr_%": thr*100, "N_trades": n_trades, "N_days": m.get("N", 0),
                     **m})
        print(f"  thr={thr*100:.1f}%: N={n_trades:4d}, Sharpe={m.get('Sharpe',0):+.2f}, "
              f"Cum={m.get('Cum%',0):+.1f}%, DD={m.get('DD%',0):.1f}%, "
              f"WR={m.get('WR%',0):.1f}%, PF={m.get('PF',0):.2f}")
    df_thr = pd.DataFrame(rows)

    # ============ (2) Top5銘柄選別 ============
    print("\n--- (2) Top5銘柄選別 (Sharpe上位のみ) ---")
    # まず銘柄別 Sharpe を計算 (thr=1.5%)
    by_symbol = {}
    for c in SYMBOLS:
        t = gap_fade_trades(minutes[c], prev_closes[c], c, gap_thr=0.015, cost_oneway=0.0008)
        m = metrics(t.set_index("date")["pnl"]) if len(t) > 0 else {}
        by_symbol[c] = {"trades": t, "metrics": m}
    ranking = sorted(by_symbol.items(),
                     key=lambda kv: kv[1]["metrics"].get("Sharpe", -99), reverse=True)
    print("  銘柄ランキング (thr=1.5%):")
    for c, info in ranking:
        m = info["metrics"]
        print(f"    {c} {SYMBOLS[c]:10s}: N={m.get('N',0):3d}, Sharpe={m.get('Sharpe',0):+.2f}, "
              f"Cum={m.get('Cum%',0):+.1f}%")

    # Top5 だけでプール
    top5_codes = [c for c, _ in ranking[:5]]
    print(f"\n  Top5: {[SYMBOLS[c] for c in top5_codes]}")
    top5_trades = {c: by_symbol[c]["trades"] for c in top5_codes}
    top5_daily = pool_daily_pnl(top5_trades)
    print(f"  Top5プール: {metrics(top5_daily)}")

    # 全8銘柄プール (比較用)
    all_trades = {c: by_symbol[c]["trades"] for c in SYMBOLS}
    all_daily = pool_daily_pnl(all_trades)
    print(f"  全8銘柄プール: {metrics(all_daily)}")

    # ============ (3) 半期安定性 (Top5) ============
    print("\n--- (3) 半期ごと安定性 (Top5プール, thr=1.5%, 8bps) ---")
    top5_daily.index = pd.to_datetime(top5_daily.index)
    periods = [
        ("2025H1 (4-9)", "2025-04-01", "2025-09-30"),
        ("2025H2 (10-3)", "2025-10-01", "2026-03-31"),
        ("2026H1 (4-5)", "2026-04-01", "2026-05-21"),
    ]
    for label, s, e in periods:
        sub = top5_daily[(top5_daily.index >= s) & (top5_daily.index <= e)]
        m = metrics(sub)
        print(f"  {label}: N={m.get('N',0)}, Sharpe={m.get('Sharpe',0):+.2f}, "
              f"Cum={m.get('Cum%',0):+.1f}%, DD={m.get('DD%',0):.1f}%")

    # ============ (4) コスト感度 (Top5) ============
    print("\n--- (4) コスト感度 (Top5プール, thr=1.5%) ---")
    for c_bps in [3, 5, 8, 10, 12, 15, 20]:
        cost = c_bps / 10000
        td = {c: gap_fade_trades(minutes[c], prev_closes[c], c, gap_thr=0.015,
                                  cost_oneway=cost) for c in top5_codes}
        daily = pool_daily_pnl(td)
        m = metrics(daily)
        print(f"  cost={c_bps:2d} bps: Sharpe={m.get('Sharpe',0):+.2f}, "
              f"Cum={m.get('Cum%',0):+.1f}%, DD={m.get('DD%',0):.1f}%")

    # ============ 保存 ============
    out_dir = os.path.dirname(os.path.abspath(__file__))
    df_thr.to_csv(os.path.join(out_dir, "threshold_sweep.csv"), index=False)
    pd.DataFrame({"top5": top5_daily, "all8": all_daily}).to_csv(
        os.path.join(out_dir, "daily_pnl.csv"))
    pd.DataFrame([{"code": c, "name": SYMBOLS[c], **info["metrics"]}
                  for c, info in ranking]).to_csv(
        os.path.join(out_dir, "by_symbol_ranking.csv"), index=False)

    # 個別 trades の集約 (top5)
    top5_all_trades = pd.concat([by_symbol[c]["trades"] for c in top5_codes],
                                  ignore_index=True)
    top5_all_trades.to_csv(os.path.join(out_dir, "top5_trades.csv"), index=False)
    print(f"\n保存: threshold_sweep.csv, daily_pnl.csv, by_symbol_ranking.csv, top5_trades.csv")

    # ============ (5) PL画像作成 ============
    print("\n--- (5) PL画像作成 ---")
    plt.rcParams.update({
        'font.family': ['Hiragino Sans', 'Hiragino Maru Gothic Pro',
                        'IPAexGothic', 'sans-serif'],
        'axes.unicode_minus': False,
        'figure.facecolor': 'white',
        'axes.facecolor': '#f8f9fa',
        'grid.alpha': 0.3,
    })
    ACCENT = '#d4243a'
    NEUTRAL = '#2e4a7d'

    fig = plt.figure(figsize=(12, 6.75), facecolor='white')
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.25,
                          left=0.06, right=0.97, top=0.87, bottom=0.10)

    m_top5 = metrics(top5_daily)
    fig.suptitle("非鉄8銘柄 寄付ギャップフェード戦略 (個別エントリ)",
                 fontsize=15, fontweight='bold', y=0.965)
    fig.text(0.5, 0.92,
             f"Top5銘柄 / |ギャップ|≥1.5% / 8bps片道 / Sharpe {m_top5['Sharpe']:+.2f} / "
             f"累積 {m_top5['Cum%']:+.1f}% / DD {m_top5['DD%']:.1f}%",
             ha='center', fontsize=10, color='#333')

    # 左上: 累積PnL (Top5 vs All8)
    ax1 = fig.add_subplot(gs[0, 0])
    eq_top5 = top5_daily.cumsum() * 100
    eq_all = all_daily.cumsum() * 100
    ax1.plot(eq_all.index, eq_all.values, color='#888', linewidth=1.0,
             alpha=0.7, label=f'全8銘柄 ({metrics(all_daily)["Sharpe"]:+.2f})')
    ax1.plot(eq_top5.index, eq_top5.values, color=ACCENT, linewidth=2.0,
             label=f'Top5 ({m_top5["Sharpe"]:+.2f})')
    ax1.axhline(0, color='gray', linewidth=0.5)
    ax1.set_title("累積PnL (%, コスト後)", fontsize=11, loc='left')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%y/%m'))
    ax1.tick_params(labelsize=8)

    # 右上: ドローダウン
    ax2 = fig.add_subplot(gs[0, 1])
    eq = top5_daily.cumsum()
    dd_series = (eq - eq.cummax()) * 100
    ax2.fill_between(dd_series.index, dd_series.values, 0, color=ACCENT, alpha=0.4)
    ax2.plot(dd_series.index, dd_series.values, color=ACCENT, linewidth=1)
    ax2.axhline(0, color='gray', linewidth=0.5)
    ax2.set_title("ドローダウン (%, Top5)", fontsize=11, loc='left')
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%y/%m'))
    ax2.tick_params(labelsize=8)

    # 左下: 銘柄別Sharpe
    ax3 = fig.add_subplot(gs[1, 0])
    codes_sorted = [c for c, _ in ranking]
    sharpes = [by_symbol[c]["metrics"].get("Sharpe", 0) for c in codes_sorted]
    names = [SYMBOLS[c] for c in codes_sorted]
    colors = [ACCENT if s > 0 else NEUTRAL for s in sharpes]
    bars = ax3.barh(range(len(sharpes)), sharpes, color=colors, alpha=0.85)
    ax3.set_yticks(range(len(names)))
    ax3.set_yticklabels(names, fontsize=9)
    ax3.invert_yaxis()
    ax3.axvline(0, color='gray', linewidth=0.5)
    ax3.set_title("銘柄別 Sharpe (thr=1.5%, 8bps)", fontsize=11, loc='left')
    ax3.grid(True, alpha=0.3, axis='x')
    ax3.tick_params(labelsize=8)
    for i, s in enumerate(sharpes):
        ax3.text(s + (0.1 if s > 0 else -0.1), i, f"{s:+.2f}",
                 va='center', ha='left' if s > 0 else 'right', fontsize=8)

    # 右下: 統計
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    text = (
        f"対象: 非鉄金属8銘柄 (Top5採用)\n\n"
        f"戦略  : 寄付ギャップフェード (個別)\n"
        f"条件  : |寄付ギャップ| ≥ 1.5%\n"
        f"エントリ: 寄付 (9:00) ・サイド= -sign(gap)\n"
        f"エグジット: 引け (15:30)\n"
        f"コスト : 片道 8 bps\n\n"
        f"期間   : {top5_daily.index.min():%Y-%m-%d} 〜 {top5_daily.index.max():%Y-%m-%d}\n"
        f"取引数 : {len(top5_all_trades)}\n"
        f"発火日数: {m_top5['N']}\n"
        f"Sharpe : {m_top5['Sharpe']:+.2f}\n"
        f"勝率   : {m_top5['WR%']:.1f} %\n"
        f"PF     : {m_top5['PF']:.2f}\n"
        f"累積PnL: {m_top5['Cum%']:+.1f} %\n"
        f"最大DD : {m_top5['DD%']:.1f} %\n"
    )
    ax4.text(0.0, 1.0, text, transform=ax4.transAxes, fontsize=10, va='top', ha='left')

    fig.text(0.99, 0.01,
             f"データ: {top5_daily.index.min():%Y-%m-%d}〜{top5_daily.index.max():%Y-%m-%d} / "
             f"日本株1分足 (JQuants) / OMEN PostgreSQL",
             ha='right', va='bottom', fontsize=7, color='gray')

    out_path = os.path.join(out_dir, "result.png")
    plt.savefig(out_path, dpi=100, bbox_inches='tight', facecolor='white')
    print(f"  保存: {out_path}")


if __name__ == "__main__":
    main()
