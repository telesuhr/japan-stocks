#!/usr/bin/env python3
"""
pair_portfolio — 18ペア統合ペアトレード シグナルチェック

毎営業日 15:30 引け後 (or 翌朝 07:00-08:30) に実行:
  python3 signal_check.py                # 本日のシグナル判定
  python3 signal_check.py --date 2026-04-22
  python3 signal_check.py --verify-db    # MariaDB 接続・依存性確認
  python3 signal_check.py --full-report  # 全18ペアの Z/β 詳細

エントリー: |Z|≥entry_z (ペアごと設定) で翌寄成発注
  - Z > +entry_z → Spread Short = p1 Short + β×p2 Long
  - Z < -entry_z → Spread Long  = p1 Long  + β×p2 Short
エグジット (毎日引け後チェック):
  - |Z|<0.5 → MR exit
  - |Z|>4.0 → Stop exit
  - MaxHold 経過 → Time exit

依存:
  - NAS MariaDB 100.92.181.92 / fallback 192.168.0.250 (database=refinitiv_news)
  - テーブル: daily_data (symbol, trade_date, close)
"""
import argparse
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import pymysql
    import statsmodels.api as sm
except ImportError as e:
    print(f"❌ 依存パッケージ不足: {e}")
    sys.exit(1)

MARIA = dict(host='100.92.181.92', port=3306, user='rfnews',
             password='Bleach@924', database='refinitiv_news')
LAN_FALLBACK = '192.168.0.250'

# ペア定義: (p1, p2, ラベル, entry_z, z_window, max_hold, exit_z, stop_z)
PAIRS = [
    ("7011.T", "7013.T", "重工MHI-IHI",        2.0, 20, 30, 0.5, 4.0),
    ("8306.T", "8411.T", "銀MUFG-みずほ",       2.0, 60, 30, 0.5, 4.0),
    ("8306.T", "8316.T", "銀MUFG-SMFG",       2.0, 60, 20, 0.5, 4.0),
    ("6146.T", "6323.T", "半ディスコ-ローツェ",    2.0, 40, 10, 0.5, 4.0),
    ("9432.T", "9433.T", "通NTT-KDDI",        2.5, 60, 10, 0.6, 4.0),
    ("8002.T", "8031.T", "商丸紅-三井物産",      2.5, 60, 10, 0.6, 4.0),
    ("5711.T", "5713.T", "非鉄マテ-住友金鉱",    2.5, 20, 30, 0.6, 4.0),
    ("5711.T", "5706.T", "非鉄マテ-三井金",      2.5, 20, 30, 0.6, 4.0),
    ("7203.T", "7267.T", "車トヨタ-ホンダ",       1.5, 20, 30, 0.4, 4.0),
    ("6501.T", "6503.T", "電機日立-三菱電",      2.5, 20, 10, 0.6, 4.0),
    ("7270.T", "7269.T", "車スバル-スズキ",      2.5, 40, 10, 0.6, 4.0),
    ("9020.T", "9022.T", "鉄JR東-JR東海",      2.5, 40, 30, 0.6, 4.0),
    ("4503.T", "4578.T", "薬アステラス-大塚",    2.5, 40, 10, 0.6, 4.0),
    ("6758.T", "6702.T", "電機ソニー-富士通",    2.0, 40, 10, 0.5, 4.0),
    ("5802.T", "5801.T", "電線住電-古河",        1.5, 40, 20, 0.4, 4.0),
    ("6920.T", "6857.T", "半レーザー-アドバン",   2.5, 40, 10, 0.6, 4.0),
    ("8035.T", "6920.T", "半TEL-レーザー",       2.5, 40, 10, 0.6, 4.0),
    ("1605.T", "5020.T", "エINPEX-ENEOS",      2.5, 20, 30, 0.6, 4.0),
]

BETA_WINDOW = 60
COST_BPS = 8.0

# ポジション記録 (オープンポジを JSON で永続化)
STATE_FILE = Path(__file__).parent / "positions.json"


def _connect_maria():
    try:
        return pymysql.connect(**MARIA, connect_timeout=5)
    except Exception:
        return pymysql.connect(**{**MARIA, 'host': LAN_FALLBACK}, connect_timeout=5)


def verify_db():
    print("=" * 70)
    print("pair_portfolio — MariaDB 依存性検証")
    print("=" * 70)
    try:
        conn = _connect_maria()
    except Exception as e:
        print(f"❌ MariaDB 接続失敗: {e}")
        return 1
    cur = conn.cursor()
    cur.execute("SHOW TABLES LIKE 'daily_data'")
    if not cur.fetchone():
        print("❌ daily_data テーブルなし"); return 1
    print("✓ daily_data テーブル存在")
    needed = sorted({s for p1, p2, *_ in PAIRS for s in (p1, p2)})
    placeholders = ",".join(["%s"] * len(needed))
    cur.execute(f"SELECT symbol, COUNT(*), MAX(trade_date) FROM daily_data "
                f"WHERE symbol IN ({placeholders}) GROUP BY symbol", tuple(needed))
    rows = {r[0]: (r[1], r[2]) for r in cur.fetchall()}
    missing = [s for s in needed if s not in rows]
    if missing:
        print(f"❌ 銘柄不足: {missing}"); return 1
    oldest_last = min(rows.values(), key=lambda x: x[1])
    print(f"✓ 45 銘柄全て存在、最古 last_date = {oldest_last[1]}")
    print(f"  レコード数範囲: {min(r[0] for r in rows.values())} 〜 "
          f"{max(r[0] for r in rows.values())}")
    conn.close()
    return 0


def load_closes(symbols, start_date, end_date):
    conn = _connect_maria()
    placeholders = ",".join(["%s"] * len(symbols))
    q = f"""SELECT symbol, trade_date, close FROM daily_data
            WHERE symbol IN ({placeholders})
              AND trade_date BETWEEN %s AND %s
            ORDER BY symbol, trade_date"""
    params = tuple(symbols) + (start_date, end_date)
    df = pd.read_sql(q, conn, params=params)
    conn.close()
    if df.empty:
        return pd.DataFrame()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.pivot(index="trade_date", columns="symbol",
                    values="close").astype(float)


def compute_pair_signal(p1, p2, px, beta_window=BETA_WINDOW, z_window=40):
    """最新日の Z と β を返す"""
    lp = np.log(px[[p1, p2]].dropna())
    if len(lp) < beta_window + z_window + 5:
        return None
    y = lp[p1].iloc[-beta_window:]
    x = lp[p2].iloc[-beta_window:]
    X = sm.add_constant(x)
    try:
        beta = sm.OLS(y, X).fit().params.iloc[1]
    except Exception:
        return None
    spread_hist = lp[p1].iloc[-z_window:] - beta * lp[p2].iloc[-z_window:]
    mu = spread_hist.mean()
    sd = spread_hist.std()
    if sd == 0 or pd.isna(sd):
        return None
    spread_now = lp[p1].iloc[-1] - beta * lp[p2].iloc[-1]
    z = (spread_now - mu) / sd
    return dict(beta=beta, spread=spread_now, mu=mu, sd=sd, z=z,
                last_date=lp.index[-1].date(),
                p1_price=float(np.exp(lp[p1].iloc[-1])),
                p2_price=float(np.exp(lp[p2].iloc[-1])))


def load_positions():
    if STATE_FILE.exists():
        with open(STATE_FILE) as f: return json.load(f)
    return {}


def save_positions(pos):
    with open(STATE_FILE, "w") as f:
        json.dump(pos, f, indent=2, default=str)


def check_exits(positions, signals, today):
    """既存ポジの exit 判定"""
    exits = []
    for pair_key, pos in list(positions.items()):
        sig = signals.get(pair_key)
        if sig is None:
            continue
        cfg = next((p for p in PAIRS if f"{p[0]}-{p[1]}" == pair_key), None)
        if cfg is None: continue
        _, _, lbl, _, _, max_hold, exit_z, stop_z = cfg
        z = sig["z"]
        entry_dt = datetime.strptime(pos["entry_date"], "%Y-%m-%d").date()
        hold_days = (today - entry_dt).days
        # 営業日カウントは簡易に自然日で代用 (日次MRでは大差ない)
        reason = None
        if abs(z) < exit_z: reason = "MR"
        elif abs(z) > stop_z: reason = "STOP"
        elif hold_days >= max_hold * 1.4: reason = "TIME"  # 自然日/営業日補正
        if reason:
            exits.append(dict(pair=lbl, key=pair_key, reason=reason,
                             z=z, hold_days=hold_days,
                             entry_z=pos["entry_z"], entry_date=pos["entry_date"],
                             direction=pos["direction"]))
    return exits


def run(target_date=None, full_report=False):
    target_date = target_date or date.today()
    print("=" * 80)
    print(f"pair_portfolio シグナルチェック  {target_date}")
    print("=" * 80)

    syms = sorted({s for p1, p2, *_ in PAIRS for s in (p1, p2)})
    start = target_date - timedelta(days=200)
    px = load_closes(syms, start, target_date)
    if px.empty:
        print("❌ データなし"); return 1
    px = px[px.index.date <= target_date]
    last_bar = px.index[-1].date()
    print(f"最新バー: {last_bar}  / 全銘柄揃い日数: {len(px)}\n")

    signals = {}
    new_entries = []
    for cfg in PAIRS:
        p1, p2, lbl, ez, zw, mh, exit_z, stop_z = cfg
        if p1 not in px.columns or p2 not in px.columns:
            continue
        sig = compute_pair_signal(p1, p2, px, z_window=zw)
        if sig is None: continue
        key = f"{p1}-{p2}"
        signals[key] = sig
        # エントリー候補
        if abs(sig["z"]) >= ez:
            direction = "SHORT_SPREAD" if sig["z"] > 0 else "LONG_SPREAD"
            new_entries.append(dict(
                pair=lbl, p1=p1, p2=p2, z=sig["z"], beta=sig["beta"],
                direction=direction, entry_z_thresh=ez,
                p1_price=sig["p1_price"], p2_price=sig["p2_price"], key=key))

    positions = load_positions()

    # --- 出口判定 ---
    exits = check_exits(positions, signals, last_bar)

    # --- 表示 ---
    if new_entries:
        print("🔔 【新規エントリー候補】")
        for e in new_entries:
            side1 = "SHORT" if e["direction"] == "SHORT_SPREAD" else "LONG"
            side2 = "LONG"  if e["direction"] == "SHORT_SPREAD" else "SHORT"
            print(f"  {e['pair']:<22}  Z={e['z']:+.2f}  β={e['beta']:+.3f}")
            print(f"     {side1} {e['p1']} @{e['p1_price']:.1f}  "
                  f"/ {side2} {e['p2']} @{e['p2_price']:.1f}  (notional β:1)")
            if e["key"] in positions:
                print(f"     ⚠ 既存ポジあり。追加しない")
        print()
    else:
        print("新規エントリー: なし\n")

    if exits:
        print("🔻 【エグジット対象】")
        for x in exits:
            print(f"  {x['pair']:<22}  {x['reason']:<4}  Z={x['z']:+.2f}  "
                  f"hold={x['hold_days']}日  entry_z={x['entry_z']:+.2f}  "
                  f"({x['entry_date']})")
        print()
    else:
        print("エグジット対象: なし\n")

    if positions:
        print("📊 【保有中ポジション】")
        for key, pos in positions.items():
            sig = signals.get(key, {})
            z_now = sig.get("z", float("nan"))
            lbl = next((p[2] for p in PAIRS if f"{p[0]}-{p[1]}" == key), key)
            print(f"  {lbl:<22}  entry {pos['entry_date']} Z={pos['entry_z']:+.2f}  "
                  f"→ 現在 Z={z_now:+.2f}  {pos['direction']}")
        print()

    if full_report:
        print("=" * 80)
        print("【全ペア Z/β 詳細】")
        print("=" * 80)
        print(f"{'ペア':<22} {'Z':>7} {'β':>7} {'entry_z':>8} {'status':<10}")
        print("-" * 60)
        for cfg in PAIRS:
            p1, p2, lbl, ez, *_ = cfg
            sig = signals.get(f"{p1}-{p2}")
            if sig is None:
                print(f"{lbl:<22}  データ不足"); continue
            if abs(sig["z"]) >= ez:
                st = "⚠ ENTRY"
            elif abs(sig["z"]) >= ez * 0.75:
                st = "接近"
            else:
                st = "-"
            print(f"{lbl:<22} {sig['z']:>+7.2f} {sig['beta']:>+7.3f} "
                  f"±{ez:>5.2f}   {st}")

    print("=" * 80)
    print("ポジション更新: positions.json を手動編集するか、")
    print("  python3 -c \"from signal_check import *; ...\" で更新。")
    print("=" * 80)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", type=str, default=None, help="YYYY-MM-DD")
    ap.add_argument("--verify-db", action="store_true")
    ap.add_argument("--full-report", action="store_true")
    args = ap.parse_args()
    if args.verify_db:
        sys.exit(verify_db())
    d = datetime.strptime(args.date, "%Y-%m-%d").date() if args.date else None
    sys.exit(run(d, args.full_report))


if __name__ == "__main__":
    main()
