"""自社株買い「決定」発表後の超過リターン(ドリフト)— サイズ調整ベンチで予備検証。

仮説(教訓5): 日本では自社株買い発表後に正のドリフトが残る(米より非効率)という既知アノマリ。
発表は引け後が多い→**翌営業日寄りで買えるか**=取引可能な形で、サイズ交絡を排除して測る。

データ: TDnet `public.tdnet_disclosures` 31日分(2026-05-14〜06-12, 3月決算後の発表ピーク期)。
  - イベント = title に「自己株式の取得」を含み、結果/終了/状況/消却/報告/訂正/変更 を含まない=取得"決定"
  - 価格 = `public.stocks_daily` の adj_open/adj_close(分割調整済)
  - ベンチ = TOPIXサイズ別指数 index_daily(0040 Core30 / 0041 Large70 / 0043 Mid400 / 0045 Small)
  - サイズ = symbol_master.scale_cat
手法: entry=発表翌営業日の寄り(adj_open)。h∈{1,3,5,10}営業日 hold→ close。
  abnormal = 個別リターン − 同区間の所属サイズ指数リターン。t値・勝率・コスト後をサイズ別に。
注意: **単一窓・小N(〜71件)=予備的read**。IS/OOS・複数レジームは無い。正式版は poller 前進蓄積後。
"""
import os, sys
sys.stdout.reconfigure(line_buffering=True)
import psycopg2, pandas as pd, numpy as np
import matplotlib.pyplot as plt

PG_CONFIG = {
    "host": os.environ.get("PGHOST", "localhost"),
    "port": int(os.environ.get("PGPORT", 5432)),
    "user": os.environ.get("PGUSER", "postgres"),
    "dbname": os.environ.get("PGDATABASE", "market_data"),
    "password": os.environ.get("PGPASSWORD", "postgres"),
}
conn = psycopg2.connect(**PG_CONFIG)

HORIZONS = [1, 3, 5, 10]
COST_ONEWAY_BP = 5.0   # 片道(手数料+スリッページ) 概算。往復=寄り買い+引け売りで2回 → 10bp
RT_COST = 2 * COST_ONEWAY_BP / 1e4

SIZE_IDX = {"TOPIX Core30": "0040", "TOPIX Large70": "0041", "TOPIX Mid400": "0043",
            "TOPIX Small 1": "0045", "TOPIX Small 2": "0045"}
DEFAULT_IDX = "0045"  # 非TOPIX(Growth等 '-')は Small で近似

# --- 1) イベント(自社株買い決定) ---
ev = pd.read_sql("""
    SELECT DISTINCT ON (code, disclosed_at::date) code, disclosed_at::date AS disc_date, title
    FROM public.tdnet_disclosures
    WHERE title LIKE '%%自己株式の取得%%'
      AND title !~ '結果|終了|状況|消却|報告|訂正|変更'
    ORDER BY code, disclosed_at::date, disclosed_at
""", conn)
print(f"自社株買い決定イベント: {len(ev)}件 ({ev.disc_date.min()}〜{ev.disc_date.max()})")

# --- 2) 取引日カレンダー(TOPIX 0000 の日付) ---
cal = pd.read_sql("SELECT DISTINCT date FROM public.index_daily WHERE code='0000' ORDER BY date", conn)
trad = list(cal["date"])
idx_of = {d: i for i, d in enumerate(trad)}

def next_trading(d):
    # d より後の最初の取引日
    import bisect
    i = bisect.bisect_right(trad, d)
    return trad[i] if i < len(trad) else None

# --- 3) サイズ ---
sm = pd.read_sql("SELECT code5, scale_cat FROM public.symbol_master", conn)
scat = dict(zip(sm.code5, sm.scale_cat))

# --- 4) 価格(関与銘柄) ---
codes = tuple(sorted(set(ev.code)))
px = pd.read_sql("""SELECT code, date, adj_open, adj_close FROM public.stocks_daily
                    WHERE code IN %s AND date >= %s""", conn, params=[codes, ev.disc_date.min()])
px = px.dropna(subset=["adj_open", "adj_close"])
pxg = {c: g.set_index("date").sort_index() for c, g in px.groupby("code")}

# --- 5) サイズ指数 ---
idx = pd.read_sql("""SELECT code, date, open, close FROM public.index_daily
                     WHERE code IN ('0040','0041','0043','0045') AND date >= %s""",
                  conn, params=[ev.disc_date.min()])
idxg = {c: g.set_index("date").sort_index() for c, g in idx.groupby("code")}

def fwd_return(series_df, entry_date, h, col_open="adj_open", col_close="adj_close"):
    """entry_date の寄り → entry から h-1 先の取引日の引け。無ければ None。"""
    if entry_date not in idx_of:
        return None
    ei = idx_of[entry_date]
    xi = ei + (h - 1)
    if xi >= len(trad):
        return None
    exit_date = trad[xi]
    if entry_date not in series_df.index or exit_date not in series_df.index:
        return None
    o = series_df.at[entry_date, col_open]
    c = series_df.at[exit_date, col_close]
    if o is None or c is None or o == 0:
        return None
    return float(c) / float(o) - 1.0

# --- 6) イベントごとに各horizonの abnormal を計算 ---
rec = []
n_unmatched = 0
for _, r in ev.iterrows():
    code, dd = r.code, r.disc_date
    if code not in pxg:
        n_unmatched += 1
        continue
    entry = next_trading(dd)
    if entry is None:
        continue
    sc = scat.get(code, "-")
    ic = SIZE_IDX.get(sc, DEFAULT_IDX)
    big = sc in ("TOPIX Core30", "TOPIX Large70", "TOPIX Mid400")
    row = {"code": code, "disc_date": dd, "entry": entry, "scale": sc,
           "size_bucket": "大型(Core30-Mid400)" if big else "小型(Small/非TOPIX)"}
    for h in HORIZONS:
        sr = fwd_return(pxg[code], entry, h)
        br = fwd_return(idxg[ic], entry, h, "open", "close")
        row[f"raw_{h}"] = sr
        row[f"abn_{h}"] = (sr - br) if (sr is not None and br is not None) else None
    rec.append(row)

df = pd.DataFrame(rec)
print(f"価格マッチ: {len(df)}件 / コード未マッチ {n_unmatched}件")

# --- 7) 集計 ---
def summarize(sub, label):
    out = []
    for h in HORIZONS:
        a = sub[f"abn_{h}"].dropna().values
        rawm = np.nanmean(sub[f"raw_{h}"].dropna().values) if sub[f"raw_{h}"].notna().any() else np.nan
        if len(a) < 3:
            out.append((label, h, len(a), np.nan, np.nan, np.nan, np.nan, np.nan)); continue
        m = a.mean(); sd = a.std(ddof=1); t = m / (sd / np.sqrt(len(a))) if sd > 0 else np.nan
        hit = (a > 0).mean()
        net = m - RT_COST  # 往復コスト後の abnormal
        out.append((label, h, len(a), rawm * 1e4, m * 1e4, t, hit, net * 1e4))
    return out

rows = []
rows += summarize(df, "全体")
for b, sub in df.groupby("size_bucket"):
    rows += summarize(sub, b)

res = pd.DataFrame(rows, columns=["group", "h(d)", "n", "raw_bp", "abn_bp", "t", "hit", "net_abn_bp"])
pd.set_option("display.width", 160, "display.max_columns", 20)
print("\n=== 自社株買い決定→翌寄りエントリー abnormal(サイズ調整) ===")
print("raw_bp=素リターン平均, abn_bp=超過(対サイズ指数), net_abn_bp=往復コスト10bp控除後, hit=超過>0比率")
print(res.to_string(index=False, float_format=lambda x: f"{x:.1f}" if pd.notna(x) else "—"))

# --- 8) 可視化 ---
try:
    import matplotlib.font_manager as fm
    fm.fontManager.addfont("/root/.fonts/NotoSansJP.ttf")
    plt.rcParams["font.family"] = "Noto Sans JP"
except Exception:
    pass
fig, ax = plt.subplots(figsize=(12, 6.75))
for label, mk in [("全体", "o-"), ("大型(Core30-Mid400)", "s--"), ("小型(Small/非TOPIX)", "^--")]:
    sub = res[res.group == label]
    ax.plot(sub["h(d)"], sub["abn_bp"], mk, label=f"{label} (n≈{int(sub['n'].iloc[0])})", linewidth=2, markersize=8)
ax.axhline(0, color="gray", lw=0.8)
ax.axhline(-COST_ONEWAY_BP*2, color="red", ls=":", lw=1, label="往復コスト(-10bp)")
ax.set_xlabel("保有営業日数"); ax.set_ylabel("abnormal CAR (bp, 対サイズ指数)")
ax.set_title("自社株買い『決定』発表後の超過リターン（翌寄りエントリー・サイズ調整）\n2026-05〜06 発表ピーク期・予備検証(単一窓・小N)")
ax.legend(); ax.grid(alpha=0.3)
fig.savefig("result.png", dpi=100, bbox_inches="tight")
print("\nsaved result.png")
conn.close()
