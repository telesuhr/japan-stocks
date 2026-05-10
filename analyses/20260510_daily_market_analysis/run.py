"""
包括的日次市場分析
=================
新DBスキーマ（stocks_daily / stocks_intraday / symbol_master 等）対応版

カバー範囲:
  1. 指数パフォーマンス（TOPIX / Core30 / Large70 / Growth250）
  2. セクター別パフォーマンス（sector33, 直近5日・1ヶ月）
  3. 出来高盛り上がりスキャン（プライム全銘柄、vol_ratio上位）
  4. 投資部門別動向（外国人・個人・信託銀行）
  5. 業種別空売り比率トレンド
  6. 信用残動向（信用倍率の変化）
  7. ストップ高・安・値上がり率ランキング
"""

import psycopg2
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

PG = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "market_data"}

def q(sql, params=None):
    conn = psycopg2.connect(**PG)
    df = pd.read_sql(sql, conn, params=params)
    conn.close()
    return df

# 最新営業日の確認
latest = q("SELECT MAX(date) as d FROM stocks_daily")["d"].iloc[0]
print(f"\n  最新営業日: {latest}")


# ════════════════════════════════════════════════════════════════
# 1. 指数パフォーマンス
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("【1】主要指数 パフォーマンス")
print("=" * 72)

idx_map = {
    "0000": "TOPIX",
    "0040": "TOPIX Core30",
    "0041": "TOPIX Large70",
    "0042": "TOPIX 100",
    "0045": "TOPIX Small",
    "0050": "グロース250",
    "0075": "JPXプライム150",
}

idx_df = q(f"""
    WITH base AS (
        SELECT code, date, close,
               LAG(close,  1) OVER (PARTITION BY code ORDER BY date) AS prev1,
               LAG(close,  5) OVER (PARTITION BY code ORDER BY date) AS prev5,
               LAG(close, 20) OVER (PARTITION BY code ORDER BY date) AS prev20,
               LAG(close, 60) OVER (PARTITION BY code ORDER BY date) AS prev60
        FROM index_daily
        WHERE code IN ({','.join("'" + c + "'" for c in idx_map)})
    )
    SELECT code, date, close, prev1, prev5, prev20, prev60
    FROM base
    WHERE date = '{latest}'
""")

print(f"\n  {'指数':<18}  {'現値':>10}  {'前日比':>8}  {'週比':>8}  {'月比':>8}  {'3月比':>8}")
print("  " + "-" * 68)
for _, r in idx_df.iterrows():
    name = idx_map.get(r["code"], r["code"])
    cl = float(r["close"])
    p1  = f"{(cl/float(r['prev1'])-1)*100:>+7.2f}%"  if r["prev1"]  else "    ---"
    p5  = f"{(cl/float(r['prev5'])-1)*100:>+7.2f}%"  if r["prev5"]  else "    ---"
    p20 = f"{(cl/float(r['prev20'])-1)*100:>+7.2f}%" if r["prev20"] else "    ---"
    p60 = f"{(cl/float(r['prev60'])-1)*100:>+7.2f}%" if r["prev60"] else "    ---"
    print(f"  {name:<18}  {cl:>10,.2f}  {p1}  {p5}  {p20}  {p60}")


# ════════════════════════════════════════════════════════════════
# 2. セクター別パフォーマンス（sector33、プライム）
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("【2】セクター別パフォーマンス（プライム市場）")
print("=" * 72)

sect_df = q(f"""
    WITH ranked AS (
        SELECT d.code, d.date, d.adj_close, s.sector33_nm,
               ROW_NUMBER() OVER (PARTITION BY d.code ORDER BY d.date DESC) AS rn
        FROM stocks_daily d
        JOIN symbol_master s ON s.code5 = d.code
        WHERE s.market = '0111'
          AND d.adj_close > 0
          AND d.date >= '{latest}'::date - INTERVAL '65 days'
    ),
    pivoted AS (
        SELECT code, sector33_nm,
               MAX(adj_close) FILTER (WHERE rn = 1)  AS c0,
               MAX(adj_close) FILTER (WHERE rn = 2)  AS c1,
               MAX(adj_close) FILTER (WHERE rn = 6)  AS c5,
               MAX(adj_close) FILTER (WHERE rn = 21) AS c20,
               MAX(adj_close) FILTER (WHERE rn = 61) AS c60
        FROM ranked GROUP BY code, sector33_nm
    )
    SELECT sector33_nm,
           COUNT(*) as n_stocks,
           AVG((c0/c1 - 1)*100)  as ret_1d,
           AVG((c0/c5 - 1)*100)  as ret_5d,
           AVG((c0/c20 - 1)*100) as ret_20d,
           AVG((c0/c60 - 1)*100) as ret_60d
    FROM pivoted
    WHERE c0 > 0 AND c1 > 0
    GROUP BY sector33_nm
    ORDER BY ret_5d DESC
""")

print(f"\n  {'セクター':<14}  {'銘柄数':>5}  {'前日比':>8}  {'週比(5d)':>9}  {'月比(20d)':>10}  {'3月比(60d)':>11}")
print("  " + "-" * 68)
for _, r in sect_df.iterrows():
    def fmt(v):
        if pd.isna(v): return "    ---"
        arrow = "↑" if v > 0 else "↓" if v < 0 else "→"
        return f"{arrow}{v:>+6.2f}%"
    print(f"  {str(r['sector33_nm']):<14}  {int(r['n_stocks']):>5}  "
          f"{fmt(r['ret_1d'])}  {fmt(r['ret_5d'])}   {fmt(r['ret_20d'])}    {fmt(r['ret_60d'])}")


# ════════════════════════════════════════════════════════════════
# 3. 出来高盛り上がりスキャン（プライム全銘柄）
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("【3】出来高盛り上がりスキャン（プライム全銘柄）")
print("  vol_ratio = 直近5日平均出来高 / 過去20日平均出来高")
print("=" * 72)

vol_df = q(f"""
    WITH vol_stats AS (
        SELECT d.code, s.name_ja, s.sector33_nm,
               -- 直近5日平均と過去20日平均（直近5日を除く）
               AVG(d.volume) FILTER (
                   WHERE d.date > '{latest}'::date - INTERVAL '7 days'
               ) AS vol_5d,
               AVG(d.volume) FILTER (
                   WHERE d.date BETWEEN '{latest}'::date - INTERVAL '30 days'
                         AND '{latest}'::date - INTERVAL '6 days'
               ) AS vol_base,
               -- 直近5日の累積リターン
               MAX(d.adj_close) FILTER (WHERE d.date = '{latest}') /
               NULLIF(MAX(d.adj_close) FILTER (
                   WHERE d.date = (
                       SELECT d2.date FROM stocks_daily d2
                       WHERE d2.code = d.code AND d2.date < '{latest}'::date - INTERVAL '4 days'
                       ORDER BY d2.date DESC LIMIT 1
                   )
               ), 0) - 1 AS ret_5d,
               -- 前日比
               MAX(d.adj_close) FILTER (WHERE d.date = '{latest}') /
               NULLIF(MAX(d.adj_close) FILTER (
                   WHERE d.date = (
                       SELECT d2.date FROM stocks_daily d2
                       WHERE d2.code = d.code AND d2.date < '{latest}'
                       ORDER BY d2.date DESC LIMIT 1
                   )
               ), 0) - 1 AS ret_1d,
               -- 直近終値（売買代金フィルタ用）
               MAX(d.turnover_value) FILTER (WHERE d.date = '{latest}') AS tv_latest
        FROM stocks_daily d
        JOIN symbol_master s ON s.code5 = d.code
        WHERE s.market = '0111'
          AND d.date >= '{latest}'::date - INTERVAL '35 days'
          AND d.adj_close > 0
        GROUP BY d.code, s.name_ja, s.sector33_nm
    )
    SELECT *,
           CASE WHEN vol_base > 0 THEN vol_5d / vol_base ELSE NULL END AS vol_ratio
    FROM vol_stats
    WHERE vol_base > 0 AND vol_5d > 0
      AND tv_latest > 500000000  -- 売買代金5億円以上
    ORDER BY vol_ratio DESC
""")

print(f"\n  ─ 出来高急増 上位20銘柄 ─")
print(f"  {'銘柄':<18}  {'セクター':<12}  {'出来高倍率':>10}  {'5日Ret':>8}  {'前日比':>8}")
print("  " + "-" * 65)
for _, r in vol_df.head(20).iterrows():
    vr = f"{float(r['vol_ratio']):.2f}x" if r['vol_ratio'] else "---"
    r5 = f"{float(r['ret_5d'])*100:>+7.2f}%" if r['ret_5d'] else "   ---"
    r1 = f"{float(r['ret_1d'])*100:>+7.2f}%" if r['ret_1d'] else "   ---"
    flag = " 🔥" if r['vol_ratio'] and r['vol_ratio'] > 2 and r['ret_5d'] and r['ret_5d'] > 0 else ""
    print(f"  {str(r['name_ja']):<18}  {str(r['sector33_nm']):<12}  {vr:>10}  {r5}  {r1}{flag}")

print(f"\n  ─ 出来高急減（閑散）下位10銘柄 ─")
print(f"  {'銘柄':<18}  {'セクター':<12}  {'出来高倍率':>10}  {'5日Ret':>8}")
print("  " + "-" * 55)
for _, r in vol_df.tail(10).iterrows():
    vr = f"{float(r['vol_ratio']):.2f}x" if r['vol_ratio'] else "---"
    r5 = f"{float(r['ret_5d'])*100:>+7.2f}%" if r['ret_5d'] else "   ---"
    print(f"  {str(r['name_ja']):<18}  {str(r['sector33_nm']):<12}  {vr:>10}  {r5}")


# ════════════════════════════════════════════════════════════════
# 4. 投資部門別動向
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("【4】投資部門別動向（TSEプライム）")
print("  FrgnBuy/Sell=外国人, IndBuy/Sell=個人, TrstBnkBuy/Sell=信託銀行")
print("=" * 72)

inv_df = q("""
    SELECT pub_date, st_date, en_date,
           (payload->>'FrgnBuy')::numeric  AS frgn_buy,
           (payload->>'FrgnSell')::numeric AS frgn_sell,
           (payload->>'IndBuy')::numeric   AS ind_buy,
           (payload->>'IndSell')::numeric  AS ind_sell,
           (payload->>'TrstBnkBuy')::numeric  AS tb_buy,
           (payload->>'TrstBnkSell')::numeric AS tb_sell,
           (payload->>'InvTrBuy')::numeric    AS invtr_buy,
           (payload->>'InvTrSell')::numeric   AS invtr_sell
    FROM investor_types
    WHERE section = 'TSEPrime'
    ORDER BY pub_date DESC
    LIMIT 8
""")

print(f"\n  {'週(公表日)':>12}  {'外国人(億)':>12}  {'個人(億)':>10}  {'信託銀行(億)':>13}  {'投信(億)':>10}")
print(f"  {'':>12}  {'買超(+)/売超(-)':>12}  {'':>10}  {'':>13}  {'':>10}")
print("  " + "-" * 65)
for _, r in inv_df.iterrows():
    def bal(buy, sell):
        if pd.isna(buy) or pd.isna(sell): return "     ---"
        b = (float(buy) - float(sell)) / 1e8
        return f"{b:>+8.1f}"
    print(f"  {str(r['pub_date']):>12}  {bal(r['frgn_buy'],r['frgn_sell']):>12}  "
          f"{bal(r['ind_buy'],r['ind_sell']):>10}  "
          f"{bal(r['tb_buy'],r['tb_sell']):>13}  "
          f"{bal(r['invtr_buy'],r['invtr_sell']):>10}")


# ════════════════════════════════════════════════════════════════
# 5. 業種別空売り比率
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("【5】業種別空売り比率（全市場、直近 vs 1ヶ月前）")
print("  空売り比率 = (制限なし空売り+残高あり空売り) / 総売買代金")
print("=" * 72)

short_df = q(f"""
    WITH latest2 AS (
        SELECT s33, date,
               shrt_with_res_va + shrt_no_res_va AS short_va,
               sell_ex_short_va AS total_va,
               ROW_NUMBER() OVER (PARTITION BY s33 ORDER BY date DESC) AS rn
        FROM jquants_short_ratio
        WHERE s33 = '9999'  -- 全市場
    )
    SELECT date,
           ROUND(short_va::numeric / NULLIF(total_va, 0) * 100, 2) AS short_ratio
    FROM latest2
    WHERE rn <= 20
    ORDER BY date DESC
""")

print(f"\n  全市場 空売り比率 推移:")
print(f"  {'日付':>12}  {'空売り比率':>12}")
for _, r in short_df.iterrows():
    print(f"  {str(r['date']):>12}  {float(r['short_ratio']):>10.2f}%")

# 業種別
short_sect = q(f"""
    WITH latest_date AS (SELECT MAX(date) as d FROM jquants_short_ratio WHERE s33 != '9999'),
         prev_date AS (
             SELECT MAX(date) as d FROM jquants_short_ratio
             WHERE s33 != '9999' AND date < (SELECT d FROM latest_date) - INTERVAL '25 days'
         ),
         latest AS (
             SELECT s33, shrt_with_res_va + shrt_no_res_va AS sv, sell_ex_short_va AS tv
             FROM jquants_short_ratio
             WHERE date = (SELECT d FROM latest_date) AND s33 != '9999'
         ),
         prev AS (
             SELECT s33, shrt_with_res_va + shrt_no_res_va AS sv, sell_ex_short_va AS tv
             FROM jquants_short_ratio
             WHERE date = (SELECT d FROM prev_date) AND s33 != '9999'
         )
    SELECT l.s33, ss.s33_nm,
           ROUND(l.sv::numeric / NULLIF(l.tv, 0) * 100, 2) AS short_now,
           ROUND(p.sv::numeric / NULLIF(p.tv, 0) * 100, 2) AS short_1m
    FROM latest l
    LEFT JOIN prev p ON p.s33 = l.s33
    LEFT JOIN sectors_33 ss ON ss.s33 = l.s33
    ORDER BY short_now DESC
    LIMIT 15
""")

print(f"\n  業種別空売り比率（上位15）:")
print(f"  {'業種':<14}  {'現在':>8}  {'1ヶ月前':>9}  {'変化':>8}")
print("  " + "-" * 45)
for _, r in short_sect.iterrows():
    now = float(r['short_now']) if r['short_now'] else 0
    prev = float(r['short_1m']) if r['short_1m'] else 0
    chg = f"{now-prev:>+6.2f}%" if prev else "    ---"
    arrow = "↑" if now > prev else "↓"
    print(f"  {str(r['s33_nm']):<14}  {now:>7.2f}%  {prev:>8.2f}%  {arrow}{chg}")


# ════════════════════════════════════════════════════════════════
# 6. 信用残動向（プライム銘柄、信用倍率ランキング）
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("【6】信用残動向（プライム市場）")
print("  信用倍率 = 買残 / 売残、高いほど売り方優位になりにくい")
print("=" * 72)

margin_df = q(f"""
    WITH latest_date AS (SELECT MAX(date) as d FROM jquants_margin_interest),
         prev_date AS (
             SELECT MAX(date) as d FROM jquants_margin_interest
             WHERE date < (SELECT d - INTERVAL '6 days' FROM latest_date)
         ),
         latest AS (
             SELECT code, long_vol, shrt_vol
             FROM jquants_margin_interest
             WHERE date = (SELECT d FROM latest_date)
         ),
         prev AS (
             SELECT code, long_vol, shrt_vol
             FROM jquants_margin_interest
             WHERE date = (SELECT d FROM prev_date)
         )
    SELECT l.code, s.name_ja, s.sector33_nm,
           l.long_vol, l.shrt_vol,
           ROUND(l.long_vol::numeric / NULLIF(l.shrt_vol, 0), 2) AS margin_ratio,
           l.long_vol - p.long_vol AS long_chg,
           l.shrt_vol - p.shrt_vol AS shrt_chg
    FROM latest l
    JOIN symbol_master s ON s.code5 = l.code
    LEFT JOIN prev p ON p.code = l.code
    WHERE s.market = '0111'
      AND l.long_vol > 0 AND l.shrt_vol > 0
    ORDER BY margin_ratio DESC
""")

print(f"\n  ─ 信用倍率 高位（買残過多）上位10 ─")
print(f"  {'銘柄':<16}  {'セクター':<12}  {'買残':>10}  {'売残':>8}  {'倍率':>7}  {'買残変化':>10}")
print("  " + "-" * 70)
for _, r in margin_df.head(10).iterrows():
    lc = f"{int(r['long_chg']):>+10,}" if r['long_chg'] else "         -"
    print(f"  {str(r['name_ja']):<16}  {str(r['sector33_nm']):<12}  "
          f"{int(r['long_vol']):>10,}  {int(r['shrt_vol']):>8,}  "
          f"{float(r['margin_ratio']):>6.1f}x  {lc}")

print(f"\n  ─ 信用倍率 低位（売残過多）下位10 ─")
print(f"  {'銘柄':<16}  {'セクター':<12}  {'買残':>10}  {'売残':>8}  {'倍率':>7}  {'売残変化':>10}")
print("  " + "-" * 70)
for _, r in margin_df[margin_df['margin_ratio'] > 0].tail(10).iterrows():
    sc = f"{int(r['shrt_chg']):>+10,}" if r['shrt_chg'] else "         -"
    print(f"  {str(r['name_ja']):<16}  {str(r['sector33_nm']):<12}  "
          f"{int(r['long_vol']):>10,}  {int(r['shrt_vol']):>8,}  "
          f"{float(r['margin_ratio']):>6.1f}x  {sc}")


# ════════════════════════════════════════════════════════════════
# 7. 値上がり/値下がりランキング & ストップ高・安
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("【7】直近日 値上がり/値下がりランキング（プライム、売買代金5億以上）")
print("=" * 72)

rank_df = q(f"""
    WITH prev_close AS (
        SELECT DISTINCT ON (d.code) d.code, d.adj_close
        FROM stocks_daily d
        WHERE d.date < '{latest}'
        ORDER BY d.code, d.date DESC
    )
    SELECT d.code, s.name_ja, s.sector33_nm,
           d.adj_close AS cl,
           p.adj_close AS prev_cl,
           (d.adj_close / NULLIF(p.adj_close, 0) - 1) * 100 AS ret,
           d.turnover_value,
           d.upper_limit,
           d.lower_limit,
           d.morning_close,
           d.afternoon_close
    FROM stocks_daily d
    JOIN symbol_master s ON s.code5 = d.code
    JOIN prev_close p ON p.code = d.code
    WHERE d.date = '{latest}'
      AND s.market = '0111'
      AND d.turnover_value >= 500000000
      AND d.adj_close > 0 AND p.adj_close > 0
""")

print(f"\n  ─ 値上がり上位10 ─")
top = rank_df.nlargest(10, 'ret')
print(f"  {'銘柄':<18}  {'セクター':<12}  {'前日比':>8}  {'売買代金(億)':>12}  {'前場/後場'}")
print("  " + "-" * 68)
for _, r in top.iterrows():
    tv = f"{float(r['turnover_value'])/1e8:.1f}億" if r['turnover_value'] else "  ---"
    sl = "★ストップ高" if r['upper_limit'] else ""
    mc = f"{float(r['morning_close']):.1f}" if r['morning_close'] else "---"
    ac = f"{float(r['afternoon_close']):.1f}" if r['afternoon_close'] else "---"
    print(f"  {str(r['name_ja']):<18}  {str(r['sector33_nm']):<12}  "
          f"{float(r['ret']):>+7.2f}%  {tv:>12}  {mc}/{ac} {sl}")

print(f"\n  ─ 値下がり下位10 ─")
bot = rank_df.nsmallest(10, 'ret')
print(f"  {'銘柄':<18}  {'セクター':<12}  {'前日比':>8}  {'売買代金(億)':>12}")
print("  " + "-" * 55)
for _, r in bot.iterrows():
    tv = f"{float(r['turnover_value'])/1e8:.1f}億" if r['turnover_value'] else "  ---"
    sl = "★ストップ安" if r['lower_limit'] else ""
    print(f"  {str(r['name_ja']):<18}  {str(r['sector33_nm']):<12}  "
          f"{float(r['ret']):>+7.2f}%  {tv:>12}  {sl}")

# ストップ高・安集計
stop_up = int(rank_df['upper_limit'].sum())
stop_dn = int(rank_df['lower_limit'].sum())
adv = int((rank_df['ret'] > 0).sum())
dec = int((rank_df['ret'] < 0).sum())
print(f"\n  ─ 市場全体（売買代金5億以上プライム）─")
print(f"  値上がり: {adv}銘柄  値下がり: {dec}銘柄  "
      f"ストップ高: {stop_up}  ストップ安: {stop_dn}")
print(f"  騰落レシオ(概算): {adv/(adv+dec)*100:.1f}%")


# ════════════════════════════════════════════════════════════════
# 8. 出来高×価格上昇 複合シグナル（買いエントリー候補）
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("【8】買いエントリー候補（出来高1.2x超 × 2日連続増加 × 価格上昇）")
print("  ← 前回分析の最良シグナルをプライム全銘柄に適用")
print("=" * 72)

signal_df = q(f"""
    WITH daily_vol AS (
        SELECT d.code, s.name_ja, s.sector33_nm, d.date,
               d.adj_close, d.volume, d.turnover_value,
               LAG(d.volume, 1) OVER (PARTITION BY d.code ORDER BY d.date) AS vol_1,
               LAG(d.volume, 2) OVER (PARTITION BY d.code ORDER BY d.date) AS vol_2,
               AVG(d.volume) OVER (
                   PARTITION BY d.code
                   ORDER BY d.date
                   ROWS BETWEEN 21 PRECEDING AND 2 PRECEDING
               ) AS vol_ma20,
               LAG(d.adj_close, 1) OVER (PARTITION BY d.code ORDER BY d.date) AS prev_cl
        FROM stocks_daily d
        JOIN symbol_master s ON s.code5 = d.code
        WHERE s.market = '0111'
          AND d.date >= '{latest}'::date - INTERVAL '30 days'
          AND d.adj_close > 0
    )
    SELECT code, name_ja, sector33_nm, date,
           adj_close, prev_cl,
           volume, vol_1, vol_2, vol_ma20,
           ROUND(volume::numeric / NULLIF(vol_ma20, 0), 2) AS vol_ratio,
           ROUND((adj_close / NULLIF(prev_cl, 0) - 1) * 100, 2) AS ret_1d,
           turnover_value
    FROM daily_vol
    WHERE date = '{latest}'
      AND volume > vol_1          -- 前日より出来高増加
      AND vol_1 > vol_2           -- 前前日より前日も増加（2日連続）
      AND vol_ma20 > 0
      AND volume::numeric / NULLIF(vol_ma20, 0) >= 1.2   -- 20日MA比1.2x以上
      AND adj_close > prev_cl     -- 価格も上昇
      AND turnover_value >= 500000000  -- 売買代金5億以上
    ORDER BY vol_ratio DESC
""")

print(f"\n  シグナル発生銘柄数: {len(signal_df)}")
print(f"  {'銘柄':<18}  {'セクター':<12}  {'出来高倍率':>10}  {'前日比':>8}  {'売買代金(億)':>12}")
print("  " + "-" * 68)
for _, r in signal_df.iterrows():
    tv = f"{float(r['turnover_value'])/1e8:.1f}億" if r['turnover_value'] else "  ---"
    vr = f"{float(r['vol_ratio']):.2f}x"
    ret = f"{float(r['ret_1d']):>+7.2f}%"
    flag = " 🔥" if float(r['vol_ratio']) >= 2.0 else ""
    print(f"  {str(r['name_ja']):<18}  {str(r['sector33_nm']):<12}  "
          f"{vr:>10}  {ret}  {tv:>12}{flag}")


print("\n  ✅ 包括的日次市場分析 完了")
print(f"  集計日: {latest}")
