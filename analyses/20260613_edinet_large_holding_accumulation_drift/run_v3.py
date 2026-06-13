"""v3: 大手運用(global_am)の新規5%超ドリフトを passive(index) vs active(stock-picker) に分解。
passiveは定義上αを持てない→passiveも同程度に上がる=資金フロー/index効果。activeだけ上がる=α(情報)。
モメンタム制御＋月次de-mean。350初回限定。
"""
import os, sys, re
sys.stdout.reconfigure(line_buffering=True)
import numpy as np, pandas as pd, psycopg2

PG={"host":os.environ.get("PGHOST","localhost"),"port":int(os.environ.get("PGPORT",5432)),
    "user":os.environ.get("PGUSER","postgres"),"password":os.environ.get("PGPASSWORD","postgres"),
    "dbname":os.environ.get("PGDATABASE","market_data")}
HORIZONS=[5,10,20,40]; COST_RT=0.0020; OOS=pd.Timestamp("2025-03-01"); HOLD=40

conn=psycopg2.connect(**PG)
ev=pd.read_sql("""SELECT issuer_code5 code, submit_date, filer_name, COALESCE(purpose,'') purpose
                  FROM public.edinet_large_holdings
                  WHERE prev_holding_ratio IS NULL AND doc_type_code='350'
                    AND submit_date BETWEEN '2024-06-01' AND '2025-10-31' AND issuer_code5 IS NOT NULL""",conn)
sd=pd.read_sql("SELECT code,date,adj_open,adj_close FROM stocks_daily WHERE code=ANY(%s) AND date>='2024-02-01'",conn,params=[list(ev.code.unique())])
idx=pd.read_sql("SELECT code,date,open,close FROM index_daily WHERE code IN ('0040','0041','0043','0045') AND date>='2024-02-01'",conn)
wt=pd.read_sql("SELECT code5,size_class FROM public.topix_weights WHERE ref_date='2026-04-30'",conn)
conn.close()

# 5分類 (active系を先にチェック=フィデリティ投信をactiveに)
ACTIVE_GLOBAL=(r"フィデリティ|ＦＭＲ|FMR|エフエムアール|キャピタル・(リサーチ|グループ|インターナショナル|マネージメント)|"
               r"ウエリントン|ベイリー・?ギフォード|ティー・?ロウ|シュローダー|インベスコ|アライアンス・?バーンスタイン|"
               r"ヌビーン|アムンディ|ピクテ|マネジメント・アンド・リサーチ")
PASSIVE_INDEX=(r"ブラックロック|ステート・?ストリート|ＳＳＧＡ|グローバル・アドバイザーズ|バンガード|ディメンショナル|"
               r"野村アセット|大和アセット|三井住友.{0,4}アセット|三菱ＵＦＪ.*投信|ニッセイ|アセット.?マネジ|アセットマネジ|投信|インベストメント")
def classify(filer,purpose):
    if re.search(r"商品在庫|ディーリング|一時保有|証券業務", purpose) or "証券" in filer: return "dealer"
    if "政策投資" in purpose or re.search(r"銀行|フィナンシャル・?グループ", filer): return "policy"
    if re.search(ACTIVE_GLOBAL, filer): return "active_global"
    if re.search(PASSIVE_INDEX, filer): return "passive_index"
    return "activist"   # アクティビスト/戦略/個人/集中
ev["cls"]=[classify(f,p) for f,p in zip(ev.filer_name,ev.purpose)]
ev=ev.drop_duplicates(["code","submit_date","cls"])
print("=== 新規5%超(350) クラス別件数 ===")
print(ev.cls.value_counts().to_string())

SIZE_MAP={"TOPIX Core30":"0040","TOPIX Large70":"0041","TOPIX Mid400":"0043","TOPIX Small 1":"0045","TOPIX Small 2":"0045"}
code_size={r.code5:SIZE_MAP.get(r.size_class,"0045") for r in wt.itertuples()}
idx["date"]=pd.to_datetime(idx["date"]); IDX={}
for c,g in idx.groupby("code"):
    g=g.sort_values("date")
    IDX[c]=({d:i for i,d in enumerate(g["date"].values.astype("datetime64[D]"))},g["open"].astype(float).values,g["close"].astype(float).values)
cal=IDX["0045"][0]
sd["date"]=pd.to_datetime(sd["date"]); sd=sd.sort_values(["code","date"])
carr={c:(g["date"].values.astype("datetime64[D]"),g["adj_open"].astype(float).values,g["adj_close"].astype(float).values) for c,g in sd.groupby("code")}
def idx_ret(code,ed,xd):
    m,o,c=IDX[code]; ti=m.get(ed); xi=m.get(xd)
    return (c[xi]/o[ti]-1) if (ti is not None and xi is not None) else None

ev["submit_date"]=pd.to_datetime(ev["submit_date"]); rows=[]
for r in ev.itertuples():
    a=carr.get(r.code)
    if a is None: continue
    cd,co,cc=a
    ei=np.searchsorted(cd,np.datetime64(r.submit_date.date(),"D"),side="right")
    if ei<61 or ei>=len(cd): continue
    ed=cd[ei]
    if ed not in cal or co[ei]<=0: continue
    szc=code_size.get(r.code,"0045"); eo=co[ei]
    pm=idx_ret(szc,cd[ei-20],cd[ei-1]); mom=(cc[ei-1]/co[ei-20]-1)-pm if pm is not None else np.nan
    rec={"code":r.code,"entry_date":pd.Timestamp(ed),"cls":r.cls,"mom":mom,"month":pd.Timestamp(ed).to_period("M")}
    for h in HORIZONS:
        xi=ei+h
        if xi>=len(cd): rec[f"d{h}"]=np.nan; continue
        szi=idx_ret(szc,ed,cd[xi])
        rec[f"d{h}"]=(cc[xi]/eo-1)-szi if szi is not None else np.nan
    rows.append(rec)
df=pd.DataFrame(rows).dropna(subset=["d40"])
for h in HORIZONS:
    df[f"dm{h}"]=df[f"d{h}"]-df.groupby("month")[f"d{h}"].transform("mean")
print(f"\nevents n={len(df)}")

print("\n=== クラス別 サイズ調整ドリフト(bp) ／ 月次de-mean後 ===")
g=df.groupby("cls")[[f"d{h}" for h in HORIZONS]].mean()*1e4; g["n"]=df.groupby("cls").size()
gd=df.groupby("cls")[[f"dm{h}" for h in HORIZONS]].mean()*1e4
out=g.join(gd)
print(out.round(1).to_string())

print("\n=== ★核心: passive_index vs active_global (αかフローか) ===")
for c in ["passive_index","active_global"]:
    s=df[df.cls==c]
    print(f"  {c:14} n={len(s):4d}  d40={s.d40.mean()*1e4:7.1f}bp  de-mean d40={s.dm40.mean()*1e4:7.1f}bp")
print("  → passiveも同程度に上がる=フロー/index効果。activeだけ上がる=α(情報)。")

# モメンタム制御(各バケツ)
def z(s): return (s-s.mean())/s.std()
def ols(y,X):
    X=np.column_stack([np.ones(len(X)),X]); b,*_=np.linalg.lstsq(X,y,rcond=None)
    e=y-X@b; se=np.sqrt(np.diag((e@e)/(len(y)-2)*np.linalg.inv(X.T@X)))
    return b[0]*1e4, b[1]*1e4, b[1]/se[1]
print("\n=== d40 ~ z(prior_mom20) バケツ別 [モメンタム代理か] ===")
for c in ["passive_index","active_global","activist"]:
    s=df[df.cls==c].dropna(subset=["mom"])
    if len(s)>30:
        cst,zc,zt=ols(s.d40.values, z(s.mom).values)
        print(f"  {c:14} const={cst:7.1f}bp  zmom係数={zc:7.1f}bp t={zt:+.2f}  (n={len(s)})")

# 可視化
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
try:
    import matplotlib.font_manager as fm
    fm.fontManager.addfont("/root/.fonts/NotoSansJP.ttf"); plt.rcParams["font.family"]="Noto Sans JP"; plt.rcParams["axes.unicode_minus"]=False
except Exception: pass
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(13.5,5.2))
order=["active_global","passive_index","activist","dealer","policy"]
col={"active_global":"#27ae60","passive_index":"#2980b9","activist":"#e67e22","dealer":"#c0392b","policy":"#7f8c8d"}
for c in order:
    if c in g.index: ax1.plot(HORIZONS,[g.loc[c,f"d{h}"] for h in HORIZONS],"o-",label=f"{c}(n={int(g.loc[c,'n'])})",color=col[c],lw=1.8)
ax1.axhline(0,color="k",lw=.8); ax1.set_xlabel("保有日数"); ax1.set_ylabel("サイズ調整ドリフト(bp)")
ax1.set_title("大手運用の新規5%超: passive vs active 分解\npassiveも上がる=フロー/activeだけ=α"); ax1.legend(fontsize=8)
# de-mean棒
cats=["active_global","passive_index","activist","dealer","policy"]
vals=[gd.loc[c,"dm40"] if c in gd.index else 0 for c in cats]
ax2.bar(cats,vals,color=[col[c] for c in cats]); ax2.axhline(0,color="k",lw=.8)
ax2.set_ylabel("de-mean d40 (bp)"); ax2.set_title("月次de-mean後 d40 (共通フロー除去)"); ax2.tick_params(axis="x",rotation=30,labelsize=8)
fig.tight_layout(); fig.savefig(os.path.join(os.path.dirname(__file__),"result_v3.png"),dpi=100,bbox_inches="tight")
print("\nsaved result_v3.png")
