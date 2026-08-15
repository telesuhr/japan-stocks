"""
sector_mom_topix_ls — 健全性チェック用サイクル再構成

採用戦略 #7（2026-07-11採用・配分20%）。月次リバランス。
- 前月末のセクター17分類等加重リターン上位3セクター × ADV上位10 = 30銘柄 LONG（等加重）
- 1306 TOPIX ETF 等額 SHORT
- 月初第1営業日の寄成でエントリー → 月末引成で決済

本モジュールは各月サイクルの実現L/Sリターンを stocks_daily から再構成する。
Dashboard_CC/sector_mom_ls_signal.py の compute_signal ロジックを再利用。

コスト（教訓2）: L/S往復8bps + 1306 ETF借入 ≈ 10bps/月 = フルサイクル約18bps。
"""
import sys
from pathlib import Path

import pandas as pd
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # health/ を通して _lib パッケージを解決
sys.path.insert(0, "/mnt/d/Root/ClaudeCode/01_Trading/Dashboard_CC")
from _lib import get_conn  # noqa: E402
from sector_mom_ls_signal import get_sector_returns  # noqa: E402

from datetime import date  # noqa: E402

LS_ROUNDTRIP_BPS = 8.0      # L/S 往復
ETF_BORROW_BPS_M = 10.0     # 1306 借入 月率
TOPIX_ETF = "13060"         # 1306 5桁


def _basket_from_ranking(month_end: date, prev_month_end: date):
    """指定月末ランキングから top3セクター×ADV上位10 = LONGバスケット(code5)を返す"""
    df = get_sector_returns(month_end, prev_month_end)
    if df.empty:
        return [], []
    sec_rank = df.groupby("sector")["ret"].mean().sort_values(ascending=False)
    top3 = sec_rank.head(3).index.tolist()
    longs = (
        df[df["sector"].isin(top3)]
        .sort_values(["sector", "adv"], ascending=[True, False])
        .groupby("sector").head(10)
    )
    # code は stocks_daily の5桁
    return top3, longs["code"].tolist()


def _basket_return(codes, entry_date, exit_date):
    """等加重バスケットの entry寄成→exit引成 リターン (小数)"""
    conn = get_conn()
    q = """
        SELECT code,
            MAX(CASE WHEN date = %(e)s THEN adj_open  END) AS p_in,
            MAX(CASE WHEN date = %(x)s THEN adj_close END) AS p_out
        FROM stocks_daily
        WHERE code = ANY(%(codes)s) AND date IN (%(e)s, %(x)s)
        GROUP BY code
    """
    df = pd.read_sql(q, conn, params={"e": str(entry_date), "x": str(exit_date),
                                      "codes": list(codes)})
    conn.close()
    df = df.dropna()
    df = df[(df["p_in"] > 0) & (df["p_out"] > 0)]
    if df.empty:
        return None, 0
    rets = df["p_out"] / df["p_in"] - 1.0
    return float(rets.mean()), len(df)


def _cycle(label, rank_month_end, rank_prev_month_end, entry_date, exit_date, complete):
    top3, longs = _basket_from_ranking(rank_month_end, rank_prev_month_end)
    long_ret, n = _basket_return(longs, entry_date, exit_date)
    short_ret, _ = _basket_return([TOPIX_ETF], entry_date, exit_date)
    if long_ret is None or short_ret is None:
        return None
    gross = long_ret - short_ret  # ロング - TOPIXショート
    # コスト
    if complete:
        cost = (LS_ROUNDTRIP_BPS + ETF_BORROW_BPS_M) / 10000.0
    else:
        # 未決済: エントリー側4bps + 経過日数按分の借入
        cost = (4.0 + ETF_BORROW_BPS_M * 0.5) / 10000.0
    net = gross - cost
    return {
        "label": label, "top3": top3, "n_long": n,
        "entry": str(entry_date), "exit": str(exit_date),
        "long_ret_pct": round(long_ret * 100, 2),
        "topix_ret_pct": round(short_ret * 100, 2),
        "gross_pct": round(gross * 100, 2),
        "net_pct": round(net * 100, 2),
        "complete": complete,
    }


def health():
    """採用後の各サイクルを再構成して返す"""
    cycles = []
    # 7月サイクル: 6月末ランキング(5/31→6/30) → 7/1寄成 → 7/31引成 (完了)
    c1 = _cycle("2026-07", date(2026, 6, 30), date(2026, 5, 31),
                date(2026, 7, 1), date(2026, 7, 31), complete=True)
    if c1:
        cycles.append(c1)
    # 8月サイクル: 7月末ランキング(6/30→7/31) → 8/3寄成 → 8/14引成 (進行中MTM)
    c2 = _cycle("2026-08", date(2026, 7, 31), date(2026, 6, 30),
                date(2026, 8, 3), date(2026, 8, 14), complete=False)
    if c2:
        cycles.append(c2)
    return cycles


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    for c in health():
        print(c)
