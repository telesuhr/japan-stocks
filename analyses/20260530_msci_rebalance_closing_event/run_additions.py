"""
MSCI採用銘柄の「公表→発効」リターン検証（Web収集した実採用リスト使用）
================================================================
仮説: MSCIスタンダード指数の新規採用銘柄は、公表日(announce)から発効日(effective)
      にかけてパッシブファンドの買いを織り込んで上昇する。公表後に買い、発効日の
      引け（パッシブの強制買い）の手前で売り抜ければ取れるのではないか。

データ: Web収集した確実な採用銘柄（複数ソースで裏取り）+ stocks_daily + index_daily(TOPIX)

採用リスト（announce=公表日, effective=発効=リバランス実施日）:
  2026-05 (a:2026-05-12, e:2026-05-29): 古河電工58010, 三井金属57060
  2025-11 (a:2025-11-06, e:2025-11-21): 荏原63610, JX金属50160, キオクシア285A0, 西武90240
  2025-08 (a:2025-08-08, e:2025-08-26): 川崎重工70120, 良品計画74530
  2025-02 (a:2025-02-11, e:2025-02-28): 東京メトロ90230
※レゾナック40040は公表前のDBデータ欠損のため除外

計測（全てTOPIX超過 = 個別リターン − TOPIXリターン）:
  R1 公表ドリフト : 公表日翌営業日 寄り → 発効日 前営業日 引け（パッシブ買いに先回り）
  R2 公表→発効引け: 公表日翌営業日 寄り → 発効日 引け（引けスパイクも取りに行く）
  R3 発効日当日   : 発効日 寄り → 発効日 引け（引けスパイクのみ）
  R4 発効翌日反転 : 発効日 引け → 発効翌営業日 引け（需給一巡後の反落？）
"""
import sys; sys.stdout.reconfigure(line_buffering=True)
import os
from pathlib import Path
import numpy as np, pandas as pd
import psycopg2

OUT=Path(__file__).parent
PG=dict(host=os.environ.get("PGHOST","localhost"),port=int(os.environ.get("PGPORT",5432)),
        user=os.environ.get("PGUSER","postgres"),password=os.environ.get("PGPASSWORD","postgres"),
        dbname=os.environ.get("PGDATABASE","market_data"))

ADDITIONS=[
    ("2026-05","2026-05-12","2026-05-29","58010","古河電工"),
    ("2026-05","2026-05-12","2026-05-29","57060","三井金属"),
    ("2025-11","2025-11-06","2025-11-21","63610","荏原"),
    ("2025-11","2025-11-06","2025-11-21","50160","JX金属"),
    ("2025-11","2025-11-06","2025-11-21","285A0","キオクシア"),
    ("2025-11","2025-11-06","2025-11-21","90240","西武HD"),
    ("2025-08","2025-08-08","2025-08-26","70120","川崎重工"),
    ("2025-08","2025-08-08","2025-08-26","74530","良品計画"),
    ("2025-02","2025-02-11","2025-02-28","90230","東京メトロ"),
]

def load(conn,code):
    df=pd.read_sql("SELECT date,open,close FROM stocks_daily WHERE code=%s ORDER BY date",conn,params=(code,))
    df["date"]=pd.to_datetime(df["date"]); return df.set_index("date")

def main():
    conn=psycopg2.connect(**PG)
    topix=pd.read_sql("SELECT date,open,close FROM index_daily WHERE code='0000' ORDER BY date",conn)
    topix["date"]=pd.to_datetime(topix["date"]); topix=topix.set_index("date")
    rows=[]
    for grp,a,e,code,name in ADDITIONS:
        px=load(conn,code)
        if px.empty: print(f"  {name}: データ無"); continue
        dts=px.index
        a=pd.Timestamp(a); e=pd.Timestamp(e)
        # 公表翌営業日, 発効前営業日, 発効翌営業日
        after_a=dts[dts>a]; before_e=dts[dts<e]; after_e=dts[dts>e]
        if len(after_a)==0 or len(before_e)==0: continue
        d_a1=after_a[0]; d_e=dts[dts<=e][-1]; d_em1=before_e[-1]
        d_e1=after_e[0] if len(after_e)>0 else None
        # 公表日以前の営業日（公表織り込み窓の起点）
        upto_a=dts[dts<=a]
        d_a=upto_a[-1] if len(upto_a)>0 else None           # 公表当日(以前最新)
        d_pre20=upto_a[-21] if len(upto_a)>=21 else None    # 公表20営業日前
        d_pre10=upto_a[-11] if len(upto_a)>=11 else None    # 公表10営業日前
        def ret(p0_dt,p0_fld,p1_dt,p1_fld):
            try:
                if p0_dt is None or p1_dt is None: return np.nan
                s=px.loc[p0_dt,p0_fld]; t=px.loc[p1_dt,p1_fld]
                ts=topix.loc[p0_dt,p0_fld]; tt=topix.loc[p1_dt,p1_fld]
                return (t/s-1)*100-(tt/ts-1)*100   # TOPIX超過(%)
            except Exception: return np.nan
        # 予想織り込み局面（公表"前"のドリフト）
        p20=ret(d_pre20,"close",d_a,"close")  # 公表20営業日前→公表日
        p10=ret(d_pre10,"close",d_a,"close")  # 公表10営業日前→公表日
        # 公表後
        r1=ret(d_a1,"open",d_em1,"close")   # 公表翌→発効前日(先回り)
        r2=ret(d_a1,"open",d_e,"close")     # 公表翌→発効引け
        r3=ret(d_e,"open",d_e,"close")      # 発効当日
        r4=ret(d_e,"close",d_e1,"close") if d_e1 is not None else np.nan  # 発効翌日反転
        rows.append({"grp":grp,"code":code,"name":name,
            "P20_公表前20d":round(p20,1),"P10_公表前10d":round(p10,1),
            "R1_公表後ドリフト":round(r1,1),"R2_公表→発効引":round(r2,1),
            "R3_発効当日":round(r3,1),"R4_発効翌日":round(r4,1)})
    conn.close()
    res=pd.DataFrame(rows); res.to_csv(OUT/"additions_returns.csv",index=False)
    print("\n===== MSCI採用銘柄 TOPIX超過リターン(%) =====")
    print(res.to_string(index=False))
    print("\n===== 平均 (n={}) =====".format(len(res)))
    for c in ["P20_公表前20d","P10_公表前10d","R1_公表後ドリフト","R2_公表→発効引","R3_発効当日","R4_発効翌日"]:
        v=res[c].dropna()
        print(f"  {c:16s} 平均{v.mean():6.1f}%  中央{v.median():6.1f}%  勝率{(v>0).mean():.0%}  (n={len(v)})")

if __name__=="__main__":
    import traceback
    try: main(); print("[DONE]")
    except Exception:
        traceback.print_exc()
        with open(OUT/"error_add.log","w") as f: f.write(traceback.format_exc())
        raise
