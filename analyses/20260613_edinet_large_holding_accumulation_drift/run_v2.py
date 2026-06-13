"""v2: 新規5%超を純化(アクティブ機関のみ)＋モメンタム制御＋月次de-mean。
v1で新規5%超が+394bp/40dだったが、filerの大半は証券ディーラー/銀行政策投資/パッシブAM=非情報。
purpose+filer名で分類し、アクティブ(純投資/物言う株主)に絞って本物か確認。
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
                  WHERE prev_holding_ratio IS NULL AND doc_type_code='350'  -- 350初回のみ(360訂正を除外)
                    AND submit_date BETWEEN '2024-06-01' AND '2025-10-31'
                    AND issuer_code5 IS NOT NULL""",conn)
sd=pd.read_sql("SELECT code,date,adj_open,adj_close FROM stocks_daily WHERE code=ANY(%s) AND date>='2024-02-01'",conn,params=[list(ev.code.unique())])
idx=pd.read_sql("SELECT code,date,open,close FROM index_daily WHERE code IN ('0040','0041','0043','0045') AND date>='2024-02-01'",conn)
wt=pd.read_sql("SELECT code5,size_class FROM public.topix_weights WHERE ref_date='2026-04-30'",conn)
conn.close()

# filer分類
GLOBAL_AM=(r"アセット.?マネジ|アセットマネジ|投信|インベストメント|マネジメント・アンド・リサーチ|"
           r"ブラックロック|フィデリティ|ＦＭＲ|FMR|エフエムアール|バンガード|ステート・?ストリート|"
           r"キャピタル・(リサーチ|グループ|インターナショナル|マネージメント)|ウエリントン|ベイリー・?ギフォード|シュローダー|"
           r"インベスコ|アムンディ|ニッセイ|ピクテ|ディメンショナル|ティー・?ロウ|ヌビーン|アライアンス・?バーンスタイン|"
           r"ＳＳＧＡ|グローバル・アドバイザーズ|野村アセット|大和アセット|三井住友")
def classify(filer,purpose):
    if re.search(r"商品在庫|ディーリング|一時保有|証券業務", purpose) or "証券" in filer: return "dealer"
    if "政策投資" in purpose or re.search(r"銀行|フィナンシャル・?グループ", filer): return "policy"
    if re.search(GLOBAL_AM, filer): return "global_am"   # 大手分散運用(パッシブ/active問わず=高確信でない)
    return "active"   # 残り=アクティビスト/戦略/個人/集中ファンド
ev["cls"]=[classify(f,p) for f,p in zip(ev.filer_name,ev.purpose)]
ev=ev.drop_duplicates(["code","submit_date","cls"])   # 同日同銘柄同クラスは1件
print("=== 新規5%超 クラス別件数 ===")
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
print(f"\nevents n={len(df)}")

print("\n=== クラス別 サイズ調整ドリフト(bp) ===")
g=df.groupby("cls")[[f"d{h}" for h in HORIZONS]].mean()*1e4; g["n"]=df.groupby("cls").size()
print(g.round(1).to_string())

# 月次de-mean(全新規5%の同月平均を引く=活動の共通bull分を除去)
for h in HORIZONS:
    df[f"dm{h}"]=df[f"d{h}"]-df.groupby("month")[f"d{h}"].transform("mean")
print("\n=== クラス別 月次de-mean後ドリフト(bp) [全新規5%平均との差=活動共通分を除去] ===")
gd=df.groupby("cls")[[f"dm{h}" for h in HORIZONS]].mean()*1e4
print(gd.round(1).to_string())

# active のモメンタム制御
act=df[df.cls=="active"].dropna(subset=["mom"])
def z(s): return (s-s.mean())/s.std()
def ols(y,X,names):
    X=np.column_stack([np.ones(len(X))]+[X[c].values for c in X.columns]); names=["const"]+names
    b,*_=np.linalg.lstsq(X,y,rcond=None); e=y-X@b; se=np.sqrt(np.diag((e@e)/(len(y)-X.shape[1])*np.linalg.inv(X.T@X)))
    return pd.DataFrame({"coef_bp":b*1e4,"t":b/se},index=names)
print(f"\n=== active新規5%超({len(act)}件) d40 ~ z(prior_mom20) [モメンタム代理か] ===")
print(ols(act["d40"].values, act.assign(zm=z(act.mom))[["zm"]].rename(columns={"zm":"zmom"}), ["zmom"]).round(2).to_string())
print("  prior_mom 三分位別 active d40(bp):")
act2=act.assign(mt=pd.qcut(act.mom,3,labels=["低","中","高"]))
print("   "+act2.groupby("mt")["d40"].agg(lambda s:round(s.mean()*1e4,1)).to_string().replace("\n","  "))

# active long-only 戦略
print(f"\n=== active新規5% ロングオンリー(翌寄り{HOLD}日・往復{COST_RT*1e4:.0f}bp・サイズ中立) ===")
for label,d in [("全",act),("IS(〜2025-02)",act[act.entry_date<OOS]),("OOS(2025-03〜)",act[act.entry_date>=OOS])]:
    p=d[f"d{HOLD}"]-COST_RT
    print(f"  {label:13} n={len(p):4d} net={p.mean()*1e4:7.1f}bp 勝率{(p>0).mean()*100:.0f}% Sh={p.mean()/p.std()*np.sqrt(252/HOLD):+.2f}")

# 可視化
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
try:
    import matplotlib.font_manager as fm
    fm.fontManager.addfont("/root/.fonts/NotoSansJP.ttf"); plt.rcParams["font.family"]="Noto Sans JP"; plt.rcParams["axes.unicode_minus"]=False
except Exception: pass
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(13.5,5.2))
order=["active","global_am","policy","dealer"]; col={"active":"#27ae60","global_am":"#95a5a6","policy":"#e67e22","dealer":"#c0392b"}
for c in order:
    if c in g.index: ax1.plot(HORIZONS,[g.loc[c,f"d{h}"] for h in HORIZONS],"o-",label=f"{c}(n={int(g.loc[c,'n'])})",color=col[c],lw=1.8)
ax1.axhline(0,color="k",lw=.8); ax1.set_xlabel("保有日数"); ax1.set_ylabel("サイズ調整ドリフト(bp)")
ax1.set_title("新規5%超 filerクラス別ドリフト\nアクティブだけ効くか(非情報と分離)"); ax1.legend(fontsize=8)
ac=act.sort_values("entry_date"); ac["p"]=ac[f"d{HOLD}"]-COST_RT
ax2.plot(ac["entry_date"],ac["p"].cumsum()*100,color="#27ae60",lw=1.3,label=f"active新規5% long(n={len(ac)})")
ax2.axvline(OOS,color="red",ls="--",lw=1,alpha=.7); ax2.axhline(0,color="k",lw=.8)
ax2.set_ylabel("累積ネット(%)"); ax2.set_title(f"active新規5% ロングオンリー P&L\n翌寄り{HOLD}日・往復20bp"); ax2.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(os.path.dirname(__file__),"result_v2.png"),dpi=100,bbox_inches="tight")
print("\nsaved result_v2.png")
