"""
ティックVolume Profile (VPVR) 試作 — 東京エレクトロン(8035/80350)

約定ティックを価格帯ごとに積算し、出来高が厚い壁(HVN)・薄い帯(LVN)を可視化。
- POC : Point of Control = 出来高最大の価格 (最も止まりやすい価格)
- VA  : Value Area = 出来高70%が収まる帯 (VAH上限/VAL下限)
- HVN : High Volume Node = 反発/反落しやすい壁
- LVN : Low Volume Node = 素通りしやすい谷
現在値の上下にある壁を出して「どこで止まりやすいか」を確認する。
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, "/mnt/d/Root/ClaudeCode/DataFetcher")

from datetime import date, timedelta
import numpy as np
from src.ticks import TickQuery

CODE = "80350"     # 東京エレクトロン
NAME = "東京エレクトロン"
START = "2026-05-25"
END = "2026-05-29"
BIN_YEN = 50       # 価格ビン幅(円)。値がさ株なので50円刻み

def main():
    tq = TickQuery()
    df = tq.raw(code=CODE, start=START, end=END)
    if df.empty:
        print("ティックデータなし")
        return

    px = df["Price"].to_numpy(dtype=float)
    vol = df["TradingVolume"].to_numpy(dtype=float)

    lo = np.floor(px.min() / BIN_YEN) * BIN_YEN
    hi = np.ceil(px.max() / BIN_YEN) * BIN_YEN
    edges = np.arange(lo, hi + BIN_YEN, BIN_YEN)
    centers = edges[:-1] + BIN_YEN / 2

    idx = np.clip(((px - lo) / BIN_YEN).astype(int), 0, len(centers) - 1)
    profile = np.zeros(len(centers))
    np.add.at(profile, idx, vol)

    total = profile.sum()
    poc_i = int(profile.argmax())
    poc = centers[poc_i]

    # Value Area: POCから出来高の多い順に積み上げ70%まで
    order = np.argsort(profile)[::-1]
    cum, va_mask = 0.0, np.zeros(len(centers), dtype=bool)
    for i in order:
        va_mask[i] = True
        cum += profile[i]
        if cum >= total * 0.70:
            break
    va_prices = centers[va_mask]
    vah, val = va_prices.max(), va_prices.min()

    cur = float(px[-1])  # 直近約定価格

    # HVN/LVN: プロファイルの局所極大/極小
    hvn, lvn = [], []
    for i in range(1, len(centers) - 1):
        if profile[i] >= profile[i-1] and profile[i] >= profile[i+1] and profile[i] > total*0.015:
            hvn.append((centers[i], profile[i]))
        if profile[i] <= profile[i-1] and profile[i] <= profile[i+1] and profile[i] < total*0.003:
            lvn.append((centers[i], profile[i]))

    print(f"=== {NAME}({CODE}) VPVR  {START}〜{END} ===")
    print(f"期間レンジ: {lo:,.0f}〜{hi:,.0f}円 / 総出来高 {total:,.0f}株 / 直近約定 {cur:,.0f}円\n")
    print(f"POC (最も出来高が厚い=止まりやすい価格): {poc:,.0f}円")
    print(f"Value Area (出来高70%帯): {val:,.0f} 〜 {vah:,.0f}円")
    print(f"  → 現在値 {cur:,.0f}円 は VA{'内' if val<=cur<=vah else '外(' + ('上' if cur>vah else '下') + ')'}\n")

    # ASCIIヒストグラム
    maxv = profile.max()
    print("価格帯別 出来高プロファイル (★=POC ▲=VAH ▼=VAL ◀=現在値):")
    for i in range(len(centers) - 1, -1, -1):
        c = centers[i]
        bar = "█" * int(profile[i] / maxv * 50)
        marks = ""
        if i == poc_i: marks += "★"
        if abs(c - vah) < BIN_YEN/2: marks += "▲"
        if abs(c - val) < BIN_YEN/2: marks += "▼"
        if val <= c <= vah and marks == "": marks = "·"
        if abs(c - (np.floor(cur/BIN_YEN)*BIN_YEN + BIN_YEN/2)) < 1: marks += "◀今"
        print(f"{c:>8,.0f} |{bar:<50}| {profile[i]/total*100:4.1f}% {marks}")

    # 現在値の上下の壁
    print("\n--- 現在値の上下にある壁(HVN) ---")
    up = sorted([h for h in hvn if h[0] > cur], key=lambda x: x[0])[:3]
    dn = sorted([h for h in hvn if h[0] < cur], key=lambda x: -x[0])[:3]
    print("上の抵抗(反落しやすい):", " / ".join(f"{p:,.0f}円({v/total*100:.1f}%)" for p,v in up) or "なし")
    print("下の支持(反発しやすい):", " / ".join(f"{p:,.0f}円({v/total*100:.1f}%)" for p,v in dn) or "なし")
    if lvn:
        print("素通り帯(LVN):", " / ".join(f"{p:,.0f}円" for p,_ in lvn[:5]))

if __name__ == "__main__":
    main()
