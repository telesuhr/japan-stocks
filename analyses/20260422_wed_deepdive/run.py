"""
水曜半導体Long 戦略の徹底解剖

既発見 (dayofweek_nonfer_semi):
  - 水曜全日 半導体バスケット Long: +332bps/日 (t=+3.85, N=25)
  - 水曜ON (火引→水寄): +256bps (t=+3.22)
  - 水曜日中 (寄→引): +76bps (t弱)

検証項目:
  1. 時間帯別リターン分解 (寄り10分 / 前場残 / 前引→後寄 / 後場前半 / 引け15分)
  2. 銘柄別寄与度 (5銘柄のうちドライバは?)
  3. SOX 前日リターンでの層別 (上昇日 / 下落日)
  4. USDJPY 夜間変化での層別
  5. Walk-forward 再現性 (月次 rolling)
  6. H1/H2 OoS
"""

import psycopg2, pymysql, pandas as pd, numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}
MARIA = {"host": "100.92.181.92", "port": 3306, "user": "rfnews",
         "password": "Bleach@924", "database": "refinitive_news",
         "connect_timeout": 5}
MARIA_LAN = {**MARIA, "host": "192.168.0.250"}

OUT = Path(__file__).parent
COST_BPS = 4.0

SEMI = {
    "8035.T": "TEL",
    "6857.T": "アドバン",
    "6146.T": "ディスコ",
    "6920.T": "レーザー",
    "6503.T": "三菱電機",
}

def load_intraday(sym):
    conn = psycopg2.connect(**PG)
    df = pd.read_sql(
        f"SELECT timestamp, open, high, low, close, volume FROM intraday_data "
        f"WHERE symbol='{sym}' ORDER BY timestamp",
        conn,
    )
    conn.close()
    if df.empty:
        return None
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["jst"] = df["timestamp"] + pd.Timedelta(hours=9)
    df = df.dropna(subset=["open", "close"]).set_index("jst").sort_index()
    h, m = df.index.hour, df.index.minute
    morning = (h == 9) | (h == 10) | ((h == 11) & (m <= 30))
    afternoon = ((h == 12) & (m >= 30)) | (h == 13) | (h == 14) | ((h == 15) & (m <= 30))
    df = df[morning | afternoon].copy()
    return df


def load_sox_daily():
    """NAS MariaDB から .SOX 日次"""
    for cfg in (MARIA, MARIA_LAN):
        try:
            conn = pymysql.connect(**cfg)
            df = pd.read_sql(
                "SELECT trade_date, close FROM daily_data WHERE ric='.SOX' ORDER BY trade_date",
                conn,
            )
            conn.close()
            df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
            df["ret"] = df["close"].pct_change() * 100
            return df
        except Exception as e:
            print(f"  MariaDB {cfg['host']} fail: {e}")
    return None


def session_segments():
    """水曜セッション内を以下に分解:
       S1: 寄 (9:00 open) → 9:10 close
       S2: 9:10 → 11:25
       S3: 11:25 → 12:35 (前場末→後場頭)
       S4: 12:35 → 14:55
       S5: 14:55 → 15:25 (引け直前)
    """
    return ["S1_open10", "S2_morning", "S3_lunchgap", "S4_afternoon", "S5_closing"]


def decompose_wednesday(df, sym):
    """水曜日の時間帯別リターンを分解"""
    rows = []
    for date, g in df.groupby(df.index.date):
        if pd.Timestamp(date).dayofweek != 2:  # Wed=2
            continue
        g = g.sort_index()
        # 指定時刻に最も近いバーを取る (分ずれ許容 ±5min)
        def pick(h, m, col="close", tol=5):
            target_min = h * 60 + m
            bar_min = g.index.hour * 60 + g.index.minute
            diff = np.abs(bar_min - target_min)
            idx = diff.argmin()
            if diff[idx] > tol:
                return None
            return g.iloc[idx][col]

        try:
            p_open = pick(9, 0, "open")
            p_0910 = pick(9, 10)
            p_1125 = pick(11, 25)
            p_1235 = pick(12, 35)
            p_1455 = pick(14, 55)
            p_1525 = pick(15, 25)
            if any(x is None for x in [p_open, p_0910, p_1125, p_1235, p_1455, p_1525]):
                continue
            rows.append({
                "date": date,
                "sym": sym,
                "S1_open10": (p_0910 / p_open - 1) * 10000,
                "S2_morning": (p_1125 / p_0910 - 1) * 10000,
                "S3_lunchgap": (p_1235 / p_1125 - 1) * 10000,
                "S4_afternoon": (p_1455 / p_1235 - 1) * 10000,
                "S5_closing": (p_1525 / p_1455 - 1) * 10000,
                "full_day": (p_1525 / p_open - 1) * 10000,
            })
        except Exception:
            continue
    return pd.DataFrame(rows)


def stats(arr):
    arr = np.asarray(arr)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 3:
        return dict(n=len(arr), mean=np.nan, sharpe=np.nan, tstat=np.nan, wr=np.nan)
    mean = arr.mean()
    std = arr.std(ddof=1)
    sharpe = mean / std * np.sqrt(52) if std > 0 else np.nan  # 週次 (水曜のみ)
    tstat = mean / (std / np.sqrt(len(arr))) if std > 0 else np.nan
    wr = (arr > 0).mean() * 100
    return dict(n=len(arr), mean=mean, sharpe=sharpe, tstat=tstat, wr=wr)


def main():
    print("=" * 80)
    print("水曜半導体Long 徹底解剖")
    print("=" * 80)

    # 1) 各銘柄の水曜セグメント別リターン取得
    all_seg = []
    for sym in SEMI:
        df = load_intraday(sym)
        if df is None:
            print(f"  SKIP {sym}")
            continue
        seg = decompose_wednesday(df, sym)
        print(f"  {sym} {SEMI[sym]}: 水曜 N={len(seg)}")
        all_seg.append(seg)
    seg_df = pd.concat(all_seg, ignore_index=True) if all_seg else pd.DataFrame()
    seg_df.to_csv(OUT / "segments.csv", index=False)
    print()

    # 2) 時間帯別統計 (pooled + 銘柄別)
    segments = session_segments() + ["full_day"]
    print("=" * 80)
    print("(A) 時間帯別 Pooled 統計 (全半導体5銘柄)")
    print("=" * 80)
    rows = []
    for s in segments:
        st = stats(seg_df[s].values - COST_BPS)  # コスト差引後
        rows.append({"segment": s, **st})
    tab_a = pd.DataFrame(rows)
    print(tab_a.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    tab_a.to_csv(OUT / "pooled_by_segment.csv", index=False)
    print()

    # 3) 銘柄 × セグメント のマトリクス (mean bps, net)
    print("=" * 80)
    print("(B) 銘柄 × セグメント 平均リターン (bps, コスト控除後)")
    print("=" * 80)
    mat = seg_df.groupby("sym")[segments].mean() - COST_BPS
    mat["n"] = seg_df.groupby("sym").size()
    print(mat.round(1).to_string())
    mat.to_csv(OUT / "stock_x_segment.csv")
    print()

    # 4) 銘柄 × セグメント の t-stat マトリクス
    print("=" * 80)
    print("(C) 銘柄 × セグメント t-stat")
    print("=" * 80)
    t_rows = []
    for sym in seg_df["sym"].unique():
        sub = seg_df[seg_df["sym"] == sym]
        r = {"sym": sym}
        for s in segments:
            st = stats(sub[s].values - COST_BPS)
            r[s] = st["tstat"]
        t_rows.append(r)
    t_mat = pd.DataFrame(t_rows).set_index("sym")
    print(t_mat.round(2).to_string())
    t_mat.to_csv(OUT / "tstat_matrix.csv")
    print()

    # 5) SOX 前日リターン層別
    sox = load_sox_daily()
    if sox is not None:
        sox_map = dict(zip(sox["trade_date"], sox["ret"]))
        seg_df["sox_prev"] = seg_df["date"].map(
            lambda d: sox_map.get(max([k for k in sox_map if k < d], default=None))
        )
        seg_df["sox_sign"] = pd.cut(seg_df["sox_prev"], bins=[-np.inf, -0.5, 0.5, np.inf],
                                     labels=["SOX↓(<-0.5)", "flat", "SOX↑(>+0.5)"])
        print("=" * 80)
        print("(D) 前日.SOX 層別 pooled 全日リターン (bps, net)")
        print("=" * 80)
        rows = []
        for label, sub in seg_df.groupby("sox_sign"):
            st = stats(sub["full_day"].values - COST_BPS)
            rows.append({"sox_prev": label, **st})
        tab_d = pd.DataFrame(rows)
        print(tab_d.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
        tab_d.to_csv(OUT / "sox_stratified.csv", index=False)
        print()

    # 6) H1/H2 OoS (full_day pooled)
    print("=" * 80)
    print("(E) H1/H2 OoS — pooled 全日リターン")
    print("=" * 80)
    seg_df_sorted = seg_df.sort_values("date").reset_index(drop=True)
    mid = len(seg_df_sorted) // 2
    for label, sub in [
        ("Full", seg_df_sorted),
        ("H1", seg_df_sorted.iloc[:mid]),
        ("H2", seg_df_sorted.iloc[mid:]),
    ]:
        st = stats(sub["full_day"].values - COST_BPS)
        dr = f"{sub['date'].min()} ~ {sub['date'].max()}" if len(sub) else ""
        print(f"  {label:5s} N={st['n']:4d} Sharpe={st['sharpe']:+.2f} t={st['tstat']:+.2f} "
              f"mean={st['mean']:+.1f}bp WR={st['wr']:.1f}% [{dr}]")
    print()

    # 7) 月次 walk-forward
    print("=" * 80)
    print("(F) 月次 walk-forward (pooled 全日リターン)")
    print("=" * 80)
    seg_df_sorted["ym"] = pd.to_datetime(seg_df_sorted["date"]).dt.to_period("M")
    rows = []
    for ym, sub in seg_df_sorted.groupby("ym"):
        if len(sub) < 3:
            continue
        st = stats(sub["full_day"].values - COST_BPS)
        rows.append({"ym": str(ym), "n": st["n"], "mean_bp": st["mean"], "tstat": st["tstat"]})
    wf = pd.DataFrame(rows)
    print(wf.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    wf.to_csv(OUT / "monthly_wf.csv", index=False)
    print()

    # --- Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    mat_plot = mat[segments[:-1]]  # drop full_day
    im = ax.imshow(mat_plot.values, aspect="auto", cmap="RdYlGn", vmin=-30, vmax=30)
    ax.set_yticks(range(len(mat_plot.index)))
    ax.set_yticklabels([f"{s}\n{SEMI.get(s,s)}" for s in mat_plot.index])
    ax.set_xticks(range(len(segments[:-1])))
    ax.set_xticklabels(segments[:-1], rotation=30, ha="right")
    ax.set_title("銘柄×セグメント 平均bps (net)")
    plt.colorbar(im, ax=ax)
    for i in range(mat_plot.shape[0]):
        for j in range(mat_plot.shape[1]):
            ax.text(j, i, f"{mat_plot.values[i,j]:.0f}", ha="center", va="center", fontsize=8)

    ax = axes[0, 1]
    pooled_means = [stats(seg_df[s].values - COST_BPS)["mean"] for s in segments[:-1]]
    pooled_t = [stats(seg_df[s].values - COST_BPS)["tstat"] for s in segments[:-1]]
    colors = ["g" if t > 2 else "r" if t < -2 else "gray" for t in pooled_t]
    ax.bar(range(len(segments[:-1])), pooled_means, color=colors)
    ax.set_xticks(range(len(segments[:-1])))
    ax.set_xticklabels(segments[:-1], rotation=30, ha="right")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title("Pooled セグメント別 mean bp (緑=t>+2)")
    ax.set_ylabel("bps")
    for i, (m, t) in enumerate(zip(pooled_means, pooled_t)):
        ax.text(i, m, f"t={t:.2f}", ha="center",
                va="bottom" if m >= 0 else "top", fontsize=9)

    ax = axes[1, 0]
    if not wf.empty:
        ax.bar(range(len(wf)), wf["mean_bp"], color=["g" if t > 0 else "r" for t in wf["mean_bp"]])
        ax.set_xticks(range(len(wf)))
        ax.set_xticklabels(wf["ym"], rotation=45, ha="right", fontsize=8)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title("月次 water-fall (pooled 全日 mean bp net)")
        ax.set_ylabel("bps")

    ax = axes[1, 1]
    cum = seg_df_sorted.groupby("date")["full_day"].mean().cumsum() - COST_BPS * np.arange(1, len(seg_df_sorted["date"].unique())+1)
    cum.plot(ax=ax)
    ax.set_title("pooled 全日 cum ret (net)")
    ax.set_ylabel("bps")
    ax.axhline(0, color="k", lw=0.5)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT / "result.png", dpi=100)
    print(f"Saved: {OUT/'result.png'}")


if __name__ == "__main__":
    main()
