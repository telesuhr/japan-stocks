"""
銘柄価格特性 総合分析
======================
分析日: 2026-04-25

各銘柄の「値動きそのもの」の特性を定量化する。

【分析項目】
  1. 基本統計 — 日次リターン分布（平均/σ/歪度/尖度/VaR95）
  2. ボラティリティ特性
     - 時間帯別ボラ（イントラデイU字カーブ）
     - 日次ボラの自己相関（ARCH効果）
     - 1分足リターンの分布形状
  3. モメンタム vs 平均回帰
     - 1分足リターンの自己相関（ラグ1〜10）
     - 日次リターンの自己相関（ラグ1〜5）
     - 当日前場 → 後場の引き継ぎ傾向
  4. ギャップ特性
     - オーバーナイトギャップの分布
     - ギャップフィル率（サイズ別）
     - 前場寄付直後（最初30分）の動き
  5. セッション別寄与度
     - 前場 / 後場 / オーバーナイト の日次リターンへの寄与
     - 各セッションのボラ比較
  6. 銘柄間特性比較マトリクス
     - ボラ / モメンタム / ギャップフィル率 のランキング
"""

import psycopg2
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

PG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}

SYMS = {
    "5713.T": "住山",
    "5711.T": "三菱マテ",
    "5706.T": "三井金属",
    "5803.T": "フジクラ",
    "5802.T": "住友電工",
    "5801.T": "古河電工",
    "6857.T": "アドバンテスト",
    "6920.T": "レーザーテック",
    "6146.T": "ディスコ",
    "6861.T": "キーエンス",
    "9984.T": "SBG",
}
SECTOR = {
    "5713.T": "非鉄", "5711.T": "非鉄", "5706.T": "非鉄",
    "5803.T": "非鉄", "5802.T": "非鉄", "5801.T": "非鉄",
    "6857.T": "半導体", "6920.T": "半導体", "6146.T": "半導体",
    "6861.T": "半導体", "9984.T": "その他",
}


def load_intraday(sym: str) -> pd.DataFrame:
    conn = psycopg2.connect(**PG)
    df = pd.read_sql(
        f"SELECT timestamp, open, high, low, close, volume "
        f"FROM intraday_data WHERE symbol='{sym}' ORDER BY timestamp", conn
    )
    conn.close()
    df["jst"] = pd.to_datetime(df["timestamp"]) + pd.Timedelta(hours=9)
    return df.dropna(subset=["close"]).set_index("jst").sort_index()


def trading_hours(df: pd.DataFrame) -> pd.DataFrame:
    h, m = df.index.hour, df.index.minute
    return df[
        (h == 9) | ((h >= 10) & (h < 11)) | ((h == 11) & (m <= 30)) |
        ((h == 12) & (m >= 30)) | ((h >= 13) & (h < 15)) | ((h == 15) & (m <= 30))
    ]


def build_daily(df: pd.DataFrame) -> pd.DataFrame:
    """日次サマリーを構築"""
    rows = []
    prev_cl = None
    for dt, g in df.groupby(df.index.date):
        g_day = trading_hours(g)
        if len(g_day) < 20:
            continue
        op = g_day["open"].iloc[0]
        cl = g_day["close"].iloc[-1]
        hi = g_day["high"].max()
        lo = g_day["low"].min()
        if op <= 0:
            continue

        mae = g_day[g_day.index.hour < 12]
        aft = g_day[g_day.index.hour >= 12]
        mae_op = mae["open"].iloc[0] if len(mae) > 0 else np.nan
        mae_cl = mae["close"].iloc[-1] if len(mae) > 0 else np.nan
        aft_op = aft["open"].iloc[0] if len(aft) > 0 else np.nan
        aft_cl = aft["close"].iloc[-1] if len(aft) > 0 else np.nan

        on_gap = (op / prev_cl - 1) * 100 if prev_cl and prev_cl > 0 else np.nan

        rows.append({
            "date": dt,
            "dow": pd.Timestamp(dt).dayofweek,
            "open": op, "close": cl, "high": hi, "low": lo,
            "mae_op": mae_op, "mae_cl": mae_cl,
            "aft_op": aft_op, "aft_cl": aft_cl,
            "day_ret":  (cl / op - 1) * 100,
            "on_gap":   on_gap,
            "mae_ret":  (mae_cl / mae_op - 1) * 100 if mae_op and mae_op > 0 else np.nan,
            "aft_ret":  (aft_cl / aft_op - 1) * 100 if aft_op and aft_op > 0 else np.nan,
            "noon_gap": (aft_op / mae_cl - 1) * 100 if mae_cl and mae_cl > 0 else np.nan,
            "range_pct": (hi / lo - 1) * 100 if lo > 0 else np.nan,
            "high_time": g_day["close"].idxmax().hour if len(g_day) > 0 else np.nan,
            "vol_day":   g_day["volume"].sum() if "volume" in g_day else np.nan,
        })
        prev_cl = cl

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════
# 1. 基本統計
# ═══════════════════════════════════════════════════════
def section1_basic_stats(all_daily: dict):
    print("\n" + "=" * 90)
    print("1. 日次リターン 基本統計")
    print("=" * 90)
    print(f"  {'銘柄':<12} {'セクター':<6} {'N':>4}  {'平均':>7}  {'σ':>6}  "
          f"{'歪度':>6}  {'尖度':>6}  {'VaR95':>7}  {'最大1日':>8}  {'最小1日':>8}")
    print("  " + "-" * 82)

    stats_list = []
    for sym, name in SYMS.items():
        d = all_daily[sym]["day_ret"].dropna()
        if len(d) < 20:
            continue
        mean = d.mean()
        std = d.std()
        skew = float(stats.skew(d))
        kurt = float(stats.kurtosis(d))  # excess kurtosis
        var95 = float(np.percentile(d, 5))
        dmax = d.max()
        dmin = d.min()
        sector = SECTOR[sym]

        # 正規分布からの乖離（尖度が大きいほど厚い裾野）
        fat_tail = "◆" if abs(kurt) > 3 else ("▲" if abs(kurt) > 1.5 else " ")
        stats_list.append({
            "sym": sym, "name": name, "sector": sector,
            "n": len(d), "mean": mean, "std": std,
            "skew": skew, "kurt": kurt, "var95": var95,
            "dmax": dmax, "dmin": dmin,
        })
        print(f"  {name:<12} {sector:<6} {len(d):>4}  {mean:>+6.3f}%  {std:>5.3f}%  "
              f"  {skew:>+5.2f}  {kurt:>5.2f}{fat_tail}  {var95:>+6.2f}%  "
              f"{dmax:>+7.2f}%  {dmin:>+7.2f}%")

    print()
    print("  ◆ = 尖度|k|>3（超ファットテール）  ▲ = 尖度|k|>1.5（ファットテール）")
    return stats_list


# ═══════════════════════════════════════════════════════
# 2. ボラティリティ特性
# ═══════════════════════════════════════════════════════
def section2_volatility(all_intraday: dict, all_daily: dict):
    print("\n" + "=" * 90)
    print("2. ボラティリティ特性")
    print("=" * 90)

    # 2a. 時間帯別ボラ（1分足標準偏差）
    print("\n  ─ 2a. 時間帯別ボラティリティ（1分足リターンの絶対値平均%）─")
    time_slots = [
        ("9:00-9:15", (9, 0, 9, 15)),
        ("9:15-9:30", (9, 15, 9, 30)),
        ("9:30-10:00", (9, 30, 10, 0)),
        ("10:00-11:00", (10, 0, 11, 0)),
        ("11:00-11:30", (11, 0, 11, 30)),
        ("12:30-13:00", (12, 30, 13, 0)),
        ("13:00-14:00", (13, 0, 14, 0)),
        ("14:00-14:30", (14, 0, 14, 30)),
        ("14:30-15:00", (14, 30, 15, 0)),
        ("15:00-15:30", (15, 0, 15, 30)),
    ]

    print(f"  {'銘柄':<12}", end="")
    for label, _ in time_slots:
        print(f"  {label}", end="")
    print()
    print("  " + "-" * 130)

    for sym, name in SYMS.items():
        df = trading_hours(all_intraday[sym]).copy()
        df["ret1m"] = df["close"].pct_change().abs() * 100
        print(f"  {name:<12}", end="")
        for label, (h1, m1, h2, m2) in time_slots:
            mask = (
                ((df.index.hour == h1) & (df.index.minute >= m1)) |
                ((df.index.hour > h1) & (df.index.hour < h2)) |
                ((df.index.hour == h2) & (df.index.minute < m2))
            ) if h1 != h2 else (
                (df.index.hour == h1) & (df.index.minute >= m1) & (df.index.minute < m2)
            )
            v = df[mask]["ret1m"].mean()
            print(f"  {v:>8.4f}%" if not np.isnan(v) else f"  {'---':>8}", end="")
        print()

    # 2b. ボラティリティクラスタリング（日次ボラの自己相関）
    print("\n  ─ 2b. ボラティリティクラスタリング（日次|リターン|のACF ラグ1〜5）─")
    print(f"  {'銘柄':<12}  {'ラグ1':>8}  {'ラグ2':>8}  {'ラグ3':>8}  {'ラグ4':>8}  {'ラグ5':>8}  判定")
    print("  " + "-" * 75)
    for sym, name in SYMS.items():
        abs_ret = all_daily[sym]["day_ret"].dropna().abs()
        acfs = [abs_ret.autocorr(lag=l) for l in range(1, 6)]
        sig = "◆ARCH" if acfs[0] > 0.1 else ("▲弱" if acfs[0] > 0.05 else "　なし")
        print(f"  {name:<12}  " + "  ".join(f"{a:>+7.3f}" for a in acfs) + f"  {sig}")

    # 2c. 日次レンジ（H-L）特性
    print("\n  ─ 2c. 日次レンジ（H-L/Open %）特性 ─")
    print(f"  {'銘柄':<12} {'平均レンジ':>10}  {'σレンジ':>9}  "
          f"{'最大レンジ':>10}  {'レンジ>3%頻度':>14}")
    print("  " + "-" * 65)
    for sym, name in SYMS.items():
        r = all_daily[sym]["range_pct"].dropna()
        print(f"  {name:<12} {r.mean():>9.3f}%  {r.std():>8.3f}%  "
              f"{r.max():>9.3f}%  {(r>3.0).mean()*100:>12.1f}%")


# ═══════════════════════════════════════════════════════
# 3. モメンタム vs 平均回帰
# ═══════════════════════════════════════════════════════
def section3_momentum_mr(all_intraday: dict, all_daily: dict):
    print("\n" + "=" * 90)
    print("3. モメンタム vs 平均回帰")
    print("=" * 90)

    # 3a. 1分足リターン自己相関
    print("\n  ─ 3a. 1分足リターン自己相関（ACF ラグ1〜10）─")
    print("  正 = モメンタム（トレンド追従）  負 = 平均回帰（ビッドアスクバウンス）")
    print(f"  {'銘柄':<12}  " + "  ".join(f"{'L'+str(l):>7}" for l in range(1, 11)) + "  判定")
    print("  " + "-" * 100)

    for sym, name in SYMS.items():
        df = trading_hours(all_intraday[sym]).copy()
        df["ret1m"] = df["close"].pct_change() * 100
        # 日をまたがない（各日の最初のリターンを除外）
        df["date"] = df.index.date
        df["first"] = df.groupby("date")["ret1m"].transform(lambda x: x.index == x.index[0])
        ret = df[~df["first"]]["ret1m"].dropna()

        acfs = [ret.autocorr(lag=l) for l in range(1, 11)]
        if acfs[0] < -0.05:
            judge = "◆平均回帰"
        elif acfs[0] > 0.05:
            judge = "▲モメンタム"
        else:
            judge = "　中立"
        print(f"  {name:<12}  " + "  ".join(f"{a:>+6.3f}" for a in acfs) + f"  {judge}")

    # 3b. 日次リターン自己相関
    print("\n  ─ 3b. 日次リターン自己相関（ACF ラグ1〜5）─")
    print(f"  {'銘柄':<12}  {'ラグ1':>8}  {'ラグ2':>8}  {'ラグ3':>8}  {'ラグ4':>8}  {'ラグ5':>8}  判定")
    print("  " + "-" * 75)
    for sym, name in SYMS.items():
        d = all_daily[sym]["day_ret"].dropna()
        acfs = [d.autocorr(lag=l) for l in range(1, 6)]
        if acfs[0] < -0.05:
            judge = "◆日次平均回帰"
        elif acfs[0] > 0.05:
            judge = "▲日次モメンタム"
        else:
            judge = "　中立"
        print(f"  {name:<12}  " + "  ".join(f"{a:>+7.3f}" for a in acfs) + f"  {judge}")

    # 3c. 前場 → 後場 引き継ぎ特性
    print("\n  ─ 3c. 前場方向 → 後場の引き継ぎ特性 ─")
    print("  正相関 = モメンタム引き継ぎ（順張り）  負相関 = リバーサル（前場高→後場安）")
    print(f"  {'銘柄':<12}  {'相関':>8}  {'前高後高':>10}  {'前高後安':>10}  "
          f"{'前安後高':>10}  {'前安後安':>10}  判定")
    print("  " + "-" * 85)
    for sym, name in SYMS.items():
        d = all_daily[sym].dropna(subset=["mae_ret", "aft_ret"])
        corr = d["mae_ret"].corr(d["aft_ret"])
        hh = ((d["mae_ret"] > 0) & (d["aft_ret"] > 0)).mean() * 100
        hl = ((d["mae_ret"] > 0) & (d["aft_ret"] < 0)).mean() * 100
        lh = ((d["mae_ret"] < 0) & (d["aft_ret"] > 0)).mean() * 100
        ll = ((d["mae_ret"] < 0) & (d["aft_ret"] < 0)).mean() * 100
        judge = "◆リバーサル" if corr < -0.1 else ("▲モメンタム" if corr > 0.1 else "　弱い相関")
        print(f"  {name:<12}  {corr:>+7.3f}  {hh:>9.1f}%  {hl:>9.1f}%  "
              f"  {lh:>9.1f}%  {ll:>9.1f}%  {judge}")


# ═══════════════════════════════════════════════════════
# 4. ギャップ特性
# ═══════════════════════════════════════════════════════
def section4_gap(all_intraday: dict, all_daily: dict):
    print("\n" + "=" * 90)
    print("4. ギャップ特性")
    print("=" * 90)

    # 4a. オーバーナイトギャップ分布
    print("\n  ─ 4a. オーバーナイトギャップ（前日引け→当日寄付）分布 ─")
    print(f"  {'銘柄':<12} {'平均ON':>8}  {'σ':>6}  {'GU>1%率':>9}  {'GD>1%率':>9}  "
          f"{'最大GU':>8}  {'最大GD':>8}")
    print("  " + "-" * 75)
    for sym, name in SYMS.items():
        g = all_daily[sym]["on_gap"].dropna()
        if len(g) < 10:
            continue
        mean = g.mean()
        std = g.std()
        gu1 = (g > 1.0).mean() * 100
        gd1 = (g < -1.0).mean() * 100
        maxgu = g.max()
        maxgd = g.min()
        print(f"  {name:<12} {mean:>+7.3f}%  {std:>5.3f}%  {gu1:>8.1f}%  {gd1:>8.1f}%  "
              f"{maxgu:>+7.2f}%  {maxgd:>+7.2f}%")

    # 4b. ギャップフィル率（GU後に当日中に寄付価格を下回る確率）
    print("\n  ─ 4b. ギャップフィル率（当日中にギャップを埋めるか）─")
    print(f"  {'銘柄':<12}  {'GU大(>1%)':<16}  {'GU小(0-1%)':<16}  "
          f"{'GD小(0-1%)':<16}  {'GD大(<-1%)':<16}")
    print("  " + "-" * 80)

    for sym, name in SYMS.items():
        d = all_daily[sym].dropna(subset=["on_gap"])
        df_raw = all_intraday[sym]
        fill_rates = []

        for gap_label, mask in [
            ("GU大(>1%)", d["on_gap"] > 1.0),
            ("GU小(0-1%)", (d["on_gap"] > 0) & (d["on_gap"] <= 1.0)),
            ("GD小(0-1%)", (d["on_gap"] < 0) & (d["on_gap"] >= -1.0)),
            ("GD大(<-1%)", d["on_gap"] < -1.0),
        ]:
            sub = d[mask]
            fills = 0
            for _, row in sub.iterrows():
                g_day = trading_hours(df_raw[df_raw.index.date == row["date"]])
                if len(g_day) == 0:
                    continue
                op = row["open"]
                if row["on_gap"] > 0:  # GU: 寄付が前日より高い → 当日中に前日終値以下になるか
                    prev_cl = op / (1 + row["on_gap"] / 100)
                    if g_day["low"].min() <= prev_cl:
                        fills += 1
                else:  # GD: 寄付が前日より低い → 当日中に前日終値以上になるか
                    prev_cl = op / (1 + row["on_gap"] / 100)
                    if g_day["high"].max() >= prev_cl:
                        fills += 1
            fill_rate = fills / len(sub) * 100 if len(sub) > 0 else np.nan
            fill_rates.append(f"{fill_rate:>5.0f}%({len(sub):>3}件)" if not np.isnan(fill_rate) else "  ---")

        print(f"  {name:<12}  " + "  ".join(f"{fr:<16}" for fr in fill_rates))

    # 4c. 寄付直後30分の動き特性（9:30時点の方向性）
    print("\n  ─ 4c. 寄付直後30分（9:30）の動きが引けまで続く確率 ─")
    print(f"  {'銘柄':<12}  {'9:30方向=引け方向':>18}  {'9:30上昇→引け上昇':>18}  "
          f"{'9:30下落→引け下落':>18}")
    print("  " + "-" * 75)
    for sym, name in SYMS.items():
        d = all_daily[sym].copy()
        df_raw = all_intraday[sym]

        # 9:30時点の寄比
        cp930_list = []
        for _, row in d.iterrows():
            g_day = trading_hours(df_raw[df_raw.index.date == row["date"]])
            if len(g_day) == 0:
                cp930_list.append(np.nan)
                continue
            op = g_day["open"].iloc[0]
            cp930_bars = g_day[(g_day.index.hour == 9) & (g_day.index.minute == 30)]
            if len(cp930_bars) > 0 and op > 0:
                cp930_list.append((cp930_bars["close"].iloc[-1] / op - 1) * 100)
            else:
                cp930_list.append(np.nan)
        d["cp930"] = cp930_list

        d2 = d.dropna(subset=["cp930", "day_ret"])
        same = (np.sign(d2["cp930"]) == np.sign(d2["day_ret"])).mean() * 100
        up_up = ((d2["cp930"] > 0) & (d2["day_ret"] > 0)).sum()
        up_total = (d2["cp930"] > 0).sum()
        dn_dn = ((d2["cp930"] < 0) & (d2["day_ret"] < 0)).sum()
        dn_total = (d2["cp930"] < 0).sum()
        up_rate = up_up / up_total * 100 if up_total > 0 else np.nan
        dn_rate = dn_dn / dn_total * 100 if dn_total > 0 else np.nan
        print(f"  {name:<12}  {same:>17.1f}%  {up_rate:>17.1f}%  {dn_rate:>17.1f}%")


# ═══════════════════════════════════════════════════════
# 5. セッション別寄与度
# ═══════════════════════════════════════════════════════
def section5_session(all_daily: dict):
    print("\n" + "=" * 90)
    print("5. セッション別リターン寄与度")
    print("=" * 90)

    print(f"\n  {'銘柄':<12}  {'ON寄与':>9}  {'前場寄与':>9}  {'昼間ギャップ':>12}  "
          f"{'後場寄与':>9}  {'前場σ':>8}  {'後場σ':>8}  {'ON σ':>8}")
    print("  " + "-" * 90)

    for sym, name in SYMS.items():
        d = all_daily[sym].dropna(subset=["mae_ret", "aft_ret"])
        on = d["on_gap"].mean() if "on_gap" in d else np.nan
        mae = d["mae_ret"].mean()
        noon = d["noon_gap"].mean() if "noon_gap" in d else np.nan
        aft = d["aft_ret"].mean()
        mae_std = d["mae_ret"].std()
        aft_std = d["aft_ret"].std()
        on_std = d["on_gap"].std() if "on_gap" in d else np.nan

        print(f"  {name:<12}  {on:>+8.3f}%  {mae:>+8.3f}%  {noon:>+11.3f}%  "
              f"{aft:>+8.3f}%  {mae_std:>7.3f}%  {aft_std:>7.3f}%  {on_std:>7.3f}%")

    print()
    print("  ─ セッション間の相関（前場×後場）─")
    print(f"  {'銘柄':<12}  {'前場×後場相関':>14}  {'ON×前場相関':>14}  {'ON×当日相関':>14}")
    print("  " + "-" * 65)
    for sym, name in SYMS.items():
        d = all_daily[sym].dropna(subset=["mae_ret", "aft_ret"])
        c1 = d["mae_ret"].corr(d["aft_ret"])
        c2 = d["on_gap"].corr(d["mae_ret"]) if "on_gap" in d else np.nan
        c3 = d["on_gap"].corr(d["day_ret"]) if "on_gap" in d else np.nan
        print(f"  {name:<12}  {c1:>+13.3f}  {c2:>+13.3f}  {c3:>+13.3f}")


# ═══════════════════════════════════════════════════════
# 6. 銘柄間比較マトリクス
# ═══════════════════════════════════════════════════════
def section6_comparison(all_intraday: dict, all_daily: dict):
    print("\n" + "=" * 90)
    print("6. 銘柄間特性比較マトリクス（ランキング）")
    print("=" * 90)

    summary = []
    for sym, name in SYMS.items():
        df = trading_hours(all_intraday[sym]).copy()
        df["ret1m"] = df["close"].pct_change() * 100
        df["date_col"] = df.index.date
        df["is_first"] = df.groupby("date_col")["ret1m"].transform(lambda x: x.index == x.index[0])
        ret1m = df[~df["is_first"]]["ret1m"].dropna()

        d = all_daily[sym]
        day_ret = d["day_ret"].dropna()
        mae_ret = d["mae_ret"].dropna()
        aft_ret = d["aft_ret"].dropna()
        on_gap = d["on_gap"].dropna()

        # 1分足自己相関（ラグ1）
        acf1m = ret1m.autocorr(lag=1)
        # 日次自己相関（ラグ1）
        acf1d = day_ret.autocorr(lag=1)
        # 前場×後場相関
        d2 = d.dropna(subset=["mae_ret", "aft_ret"])
        mae_aft_corr = d2["mae_ret"].corr(d2["aft_ret"])
        # 日次ボラ
        day_vol = day_ret.std()
        # ONギャップσ
        on_vol = on_gap.std()
        # ギャップフィル率（GU>1%のみ簡易計算）
        on_pos = on_gap[on_gap > 1.0]
        # ファットテール（尖度）
        kurt = float(stats.kurtosis(day_ret))

        summary.append({
            "sym": sym, "name": name, "sector": SECTOR[sym],
            "day_vol": day_vol,          # 日次ボラ（大→リスク高）
            "on_vol": on_vol,            # ONギャップのσ
            "acf1m": acf1m,             # 1分自己相関（負→平均回帰）
            "acf1d": acf1d,             # 日次自己相関
            "mae_aft": mae_aft_corr,    # 前場→後場（負→リバーサル）
            "kurt": kurt,               # 尖度（大→ファットテール）
            "n_gaps_gu1": len(on_pos),
        })

    df_sum = pd.DataFrame(summary)

    print(f"\n  {'銘柄':<12} {'セクター':<6} "
          f"{'日次σ':>7}  {'ONσ':>6}  "
          f"{'1分ACF':>8}  {'日次ACF':>8}  "
          f"{'前後相関':>9}  {'尖度':>6}")
    print("  " + "-" * 80)

    for _, r in df_sum.iterrows():
        acf1m_mark = "▼" if r["acf1m"] < -0.05 else ("▲" if r["acf1m"] > 0.05 else " ")
        mae_aft_mark = "◆" if r["mae_aft"] < -0.1 else ("▲" if r["mae_aft"] > 0.1 else " ")
        print(f"  {r['name']:<12} {r['sector']:<6} "
              f"{r['day_vol']:>6.3f}%  {r['on_vol']:>5.3f}%  "
              f"{r['acf1m']:>+7.3f}{acf1m_mark}  {r['acf1d']:>+7.3f}  "
              f"  {r['mae_aft']:>+8.3f}{mae_aft_mark}  {r['kurt']:>5.2f}")

    print()
    print("  ─ 特性別ランキング ─")
    print()

    # ボラランク
    vol_rank = df_sum.nlargest(11, "day_vol")
    print(f"  【日次ボラ ランキング（大→小）】")
    for i, (_, r) in enumerate(vol_rank.iterrows(), 1):
        print(f"    {i}. {r['name']:<12} {r['day_vol']:>6.3f}% σ/日")

    print()
    # 1分足平均回帰ランク
    mr_rank = df_sum.nsmallest(11, "acf1m")
    print(f"  【1分足 平均回帰強度ランキング（最も負→最も弱い）】")
    for i, (_, r) in enumerate(mr_rank.iterrows(), 1):
        mark = "◆強い平均回帰" if r["acf1m"] < -0.05 else ""
        print(f"    {i}. {r['name']:<12} ACF={r['acf1m']:>+7.4f}  {mark}")

    print()
    # セッションリバーサルランク
    rev_rank = df_sum.nsmallest(11, "mae_aft")
    print(f"  【前場→後場リバーサル強度ランキング（最も負→最も弱い）】")
    for i, (_, r) in enumerate(rev_rank.iterrows(), 1):
        mark = "◆強いリバーサル" if r["mae_aft"] < -0.1 else ("▲モメンタム" if r["mae_aft"] > 0.1 else "")
        print(f"    {i}. {r['name']:<12} 前場×後場相関={r['mae_aft']:>+7.3f}  {mark}")

    return df_sum


# ═══════════════════════════════════════════════════════
# メイン
# ═══════════════════════════════════════════════════════
def main():
    print("=" * 90)
    print("  銘柄価格特性 総合分析")
    print("=" * 90)
    print("  データロード中...")

    all_intraday = {}
    all_daily = {}
    for sym, name in SYMS.items():
        df = load_intraday(sym)
        all_intraday[sym] = df
        all_daily[sym] = build_daily(df)
        print(f"    {name}: {len(df)}バー  /  {len(all_daily[sym])}日")

    # 各セクションを実行
    section1_basic_stats(all_daily)
    section2_volatility(all_intraday, all_daily)
    section3_momentum_mr(all_intraday, all_daily)
    section4_gap(all_intraday, all_daily)
    section5_session(all_daily)
    df_sum = section6_comparison(all_intraday, all_daily)

    print("\n" + "=" * 90)
    print("  ✅ 分析完了")
    print("=" * 90)


if __name__ == "__main__":
    main()
