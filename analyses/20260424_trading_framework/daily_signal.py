"""
日次シグナル生成スクリプト  — 当日の取引チートシート
使い方:
  python daily_signal.py              # 今日のシグナル（市場前・市場中・市場後）
  python daily_signal.py --date 2026-04-25  # 特定日

出力する情報:
  [朝] 前日の動き → 当日の期待値・注意点
  [前場] 前場リターン → 後場期待値（10分毎に更新可能）
  [後場] 後場寄直後の方向 → 引けまでの推奨アクション
  [引け前] 15:00時点のパターン → 15:30まで保有すべきか

ORB戦略（最重要ルール）:
  ✔ 9:30時点で±0.3%超 → その方向に乗る（前場モメンタム）
  ✔ 10:00時点でも同方向なら前場引けまでホールド
  ✔ 方向に逆らわない
"""
import psycopg2, pandas as pd, numpy as np, argparse
from datetime import date, datetime, timedelta
import warnings; warnings.filterwarnings('ignore')

PG = {"host":"localhost","port":5432,"user":"postgres","dbname":"market_data"}

SYMS = {
    "5713.T":"住山","5711.T":"三菱マテ","5706.T":"三井金属",
    "5803.T":"フジクラ","5802.T":"住友電工","5801.T":"古河電工",
    "6857.T":"アドバンテスト","6920.T":"レーザーテック",
    "6146.T":"ディスコ","6861.T":"キーエンス","9984.T":"SBG",
}
SECTOR={"5713.T":"非鉄","5711.T":"非鉄","5706.T":"非鉄",
        "5803.T":"非鉄","5802.T":"非鉄","5801.T":"非鉄",
        "6857.T":"半導体","6920.T":"半導体","6146.T":"半導体",
        "6861.T":"半導体","9984.T":"その他"}
DAY={0:'月',1:'火',2:'水',3:'木',4:'金',5:'土',6:'日'}

# ── パターン期待値テーブル（pattern_encyclopedia の結果から）──
# ORB30分モメンタム：net_sharpe順上位の期待値（%）
ORB30_EXPECTED = {  # (sym, direction) -> (mean, wr, net_sharpe)
    "5713.T": {"up":(1.30,88.8,14.50), "dn":(1.30,88.8,14.50)},
    "5711.T": {"up":(1.45,91.9,18.51), "dn":(1.35,85.1,14.82)},
    "5706.T": {"up":(2.27,89.2,14.49), "dn":(2.04,85.7,15.13)},  # 古河電工→三井金属
    "5803.T": {"up":(2.23,87.9,14.87), "dn":(1.77,83.8,14.65)},
    "5802.T": {"up":(1.75,82.8,15.00), "dn":(1.72,86.8,14.07)},
    "5801.T": {"up":(2.34,85.7,13.27), "dn":(2.04,85.7,15.13)},
    "6857.T": {"up":(1.71,84.9,14.25), "dn":(1.52,86.7,14.37)},
    "6920.T": {"up":(2.09,88.0,16.03), "dn":(1.58,89.8,15.21)},
    "6146.T": {"up":(1.42,87.1,15.28), "dn":(1.88,93.2,18.75)},
    "6861.T": {"up":(1.19,91.2,18.07), "dn":(0.93,87.7,16.99)},
    "9984.T": {"up":(2.08,85.7,13.68), "dn":(2.08,85.7,12.91)},
}

# 曜日別期待値（後場）- 意識すべき曜日パターン
# (mean_aft_ret, wr)
DOW_AFT_EXPECTED = {
    "5713.T": {0:(+0.179,48.4),1:(-0.173,44.1),2:(+0.118,51.4),3:(-0.002,44.3),4:(+0.172,44.9)},
    "5803.T": {0:(+0.024,53.1),1:(-1.573,47.8),2:(+0.578,65.3),3:(+0.021,52.8),4:(+0.200,66.7)},
}


def load_recent(sym, days=5):
    conn = psycopg2.connect(**PG)
    df = pd.read_sql(
        f"SELECT timestamp,open,high,low,close,volume FROM intraday_data "
        f"WHERE symbol='{sym}' AND timestamp >= NOW() - INTERVAL '{days} days' "
        f"ORDER BY timestamp", conn)
    conn.close()
    if len(df)==0: return pd.DataFrame()
    df['jst'] = pd.to_datetime(df['timestamp']) + pd.Timedelta(hours=9)
    return df.dropna(subset=['close']).set_index('jst').sort_index()


def load_date(sym, target_date):
    conn = psycopg2.connect(**PG)
    df = pd.read_sql(
        f"SELECT timestamp,open,high,low,close,volume FROM intraday_data "
        f"WHERE symbol='{sym}' "
        f"AND timestamp >= '{target_date} 00:00:00' "
        f"AND timestamp <  '{target_date} 23:59:59' "
        f"ORDER BY timestamp", conn)
    conn.close()
    if len(df)==0: return pd.DataFrame()
    df['jst'] = pd.to_datetime(df['timestamp']) + pd.Timedelta(hours=9)
    return df.dropna(subset=['close']).set_index('jst').sort_index()


def thr(df):
    h,m=df.index.hour,df.index.minute
    return df[(h==9)|((h>=10)&(h<11))|((h==11)&(m<=30))|
              ((h==12)&(m>=30))|((h>=13)&(h<15))|((h==15)&(m<=30))]
def mae_sess(df):
    h,m=df.index.hour,df.index.minute
    return df[(h==9)|((h>=10)&(h<11))|((h==11)&(m<=30))]
def aft_sess(df):
    h,m=df.index.hour,df.index.minute
    return df[((h==12)&(m>=30))|((h>=13)&(h<15))|((h==15)&(m<=30))]

# ── シグナル評価関数 ──────────────────────

def signal_label(val, thresholds=((0.5,'◎強い買い'),(0.2,'○買い'),(-0.2,'△中立'),(-0.5,'▽売り'),)):
    for thr, label in thresholds:
        if val >= thr: return label
    return '▼強い売り'

def orb_signal(ret_at_checkpoint):
    """ORBシグナル: checkpointでの寄付比リターン → 方向"""
    if ret_at_checkpoint > 0.3:
        return ('LONG', f'+{ret_at_checkpoint:.2f}% → 前場モメンタム買い')
    elif ret_at_checkpoint < -0.3:
        return ('SHORT', f'{ret_at_checkpoint:.2f}% → 前場モメンタム売り')
    else:
        return ('FLAT', f'{ret_at_checkpoint:+.2f}% → 様子見')


def print_separator(title='', char='─', width=70):
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{'─'*pad} {title} {'─'*pad}")
    else:
        print('─'*width)


def generate_signal(target_date_str=None):
    if target_date_str:
        target_date = datetime.strptime(target_date_str, '%Y-%m-%d').date()
    else:
        target_date = date.today()

    dow = target_date.weekday()
    prev_date = target_date - timedelta(days=1)
    while prev_date.weekday() >= 5:
        prev_date -= timedelta(days=1)

    print(f"\n{'='*70}")
    print(f"  📋 日次トレーディングシグナル   {target_date} ({DAY[dow]}曜日)")
    print(f"{'='*70}")

    # ────────────────────────────────────────
    # SECTION 1: 朝の準備（前日分析）
    # ────────────────────────────────────────
    print_separator("① 朝の準備 — 前日リターンと当日バイアス")
    print(f"{'銘柄':<12} {'前日ret':>7}  {'当日バイアス':>10}  {'ORBで確認ポイント':>24}  {'注意'}")
    print("─"*80)

    sym_data = {}
    for sym, name in SYMS.items():
        # 前日データ
        prev_df = load_date(sym, str(prev_date))
        today_df = load_date(sym, str(target_date))

        if len(prev_df) == 0:
            continue
        prev_g = thr(prev_df)
        if len(prev_g) < 10: continue

        prev_op = prev_g['open'].iloc[0]; prev_cl = prev_g['close'].iloc[-1]
        prev_ret = (prev_cl/prev_op-1)*100 if prev_op>0 else np.nan

        # 前日分位によるバイアス
        if   prev_ret >  1.0: bias = '↑モメンタム注意'
        elif prev_ret >  0.3: bias = '↑弱モメンタム'
        elif prev_ret < -1.0: bias = '↓モメンタム注意'
        elif prev_ret < -0.3: bias = '↓弱モメンタム'
        else:                  bias = '→中立'

        # 月曜の場合は週末ON注意
        on_note = ''
        if dow == 0: on_note = '⚠️週末ON注意'
        # 火曜フジクラは特別警告
        if dow == 1 and sym == '5803.T': on_note = '🚨火曜後場危険'
        # 金曜住山の前場高パターン
        if dow == 4 and sym == '5713.T': on_note = '⭐金曜前場高→後場+0.6%'

        orb_hint = '9:30で±0.3%確認'
        print(f"{name:<12} {prev_ret:>+6.2f}%  {bias:>12}  {orb_hint:>26}  {on_note}")

        sym_data[sym] = {
            'prev_ret': prev_ret, 'today_df': today_df,
            'prev_df': prev_df
        }

    # ────────────────────────────────────────
    # SECTION 2: 前場ORBシグナル
    # ────────────────────────────────────────
    print_separator("② 前場 ORBシグナル — 9:30 / 10:00 時点")

    has_today = any(len(v['today_df'])>0 for v in sym_data.values())
    if not has_today:
        print("  ※ 当日データ未取得（市場前または休場）")
        print("\n  【ORBルール（暗記）】")
        print("  ┌─────────────────────────────────────────────────────────┐")
        print("  │ 9:30で寄比 >+0.3% → LONG  前場引けまで保有 (勝率87〜93%)│")
        print("  │ 9:30で寄比 <-0.3% → SHORT 前場引けまで保有 (勝率83〜93%)│")
        print("  │ ±0.3%以内      → 様子見  無理にエントリーしない    │")
        print("  │ 10:00でも同方向 → 確信度UP、ホールド継続           │")
        print("  │ 10:00で逆転     → 即クローズ                       │")
        print("  └─────────────────────────────────────────────────────────┘")
    else:
        print(f"{'銘柄':<12} {'9:30寄比':>9} {'シグナル':>8}  {'期待リターン':>12}  {'勝率':>6}  {'10:00寄比':>9}")
        print("─"*70)
        for sym, name in SYMS.items():
            if sym not in sym_data or len(sym_data[sym]['today_df'])==0: continue
            g = thr(sym_data[sym]['today_df'])
            if len(g)==0: continue
            op = g['open'].iloc[0]
            cp930  = g[(g.index.hour==9)&(g.index.minute==30)]
            cp1000 = g[(g.index.hour==10)&(g.index.minute==0)]
            ret930  = (cp930['close'].iloc[-1]/op-1)*100  if (len(cp930)>0  and op>0) else np.nan
            ret1000 = (cp1000['close'].iloc[-1]/op-1)*100 if (len(cp1000)>0 and op>0) else np.nan

            if np.isnan(ret930): continue
            sig, msg = orb_signal(ret930)
            exp = ORB30_EXPECTED.get(sym, {})
            if sig=='LONG' and 'up' in exp:
                mean_, wr_, sh_ = exp['up']
                exp_str = f'+{mean_:.2f}%'
                wr_str  = f'{wr_:.0f}%'
            elif sig=='SHORT' and 'dn' in exp:
                mean_, wr_, sh_ = exp['dn']
                exp_str = f'+{mean_:.2f}%'
                wr_str  = f'{wr_:.0f}%'
            else:
                exp_str = '---'; wr_str = '---'

            sig_icon = {'LONG':'🔼','SHORT':'🔽','FLAT':'⏸'}[sig]
            ret1000_str = f'{ret1000:>+6.2f}%' if not np.isnan(ret1000) else '未確定'
            print(f"{name:<12} {ret930:>+8.2f}%  {sig_icon}{sig:<6}  {exp_str:>12}  {wr_str:>6}  {ret1000_str:>9}")

    # ────────────────────────────────────────
    # SECTION 3: 後場シグナル（12:30）
    # ────────────────────────────────────────
    print_separator("③ 後場シグナル — 12:30 寄直後の方向確認")
    print("  【ルール】後場寄付後5〜10分の動きが引けまでの方向を決める")
    print(f"{'銘柄':<12} {'昼ギャップ':>10} {'後場寄5分':>10}  {'推奨':>20}  {'期待/勝率':>12}")
    print("─"*72)

    for sym, name in SYMS.items():
        if sym not in sym_data or len(sym_data[sym]['today_df'])==0: continue
        g = thr(sym_data[sym]['today_df'])
        gm = mae_sess(sym_data[sym]['today_df'])
        go = aft_sess(sym_data[sym]['today_df'])
        if len(gm)<5 or len(go)<3: continue

        mae_cl  = gm['close'].iloc[-1]
        aft_op  = go['open'].iloc[0]
        aft_5m  = go[go.index.hour==12]
        aft_5m_cl = aft_5m['close'].iloc[-1] if len(aft_5m)>0 else np.nan

        gap_noon = (aft_op/mae_cl-1)*100 if mae_cl>0 else np.nan
        aft_5m_ret = (aft_5m_cl/aft_op-1)*100 if (not np.isnan(aft_5m_cl) and aft_op>0) else np.nan

        if np.isnan(gap_noon): continue

        # 昼ギャップシグナル
        if gap_noon > 0.2:
            gap_sig = f'↑GU {gap_noon:+.2f}%'
        elif gap_noon < -0.2:
            gap_sig = f'↓GD {gap_noon:+.2f}%'
        else:
            gap_sig = f'→フラット {gap_noon:+.2f}%'

        # 後場5分シグナル
        if np.isnan(aft_5m_ret):
            aft_sig = '未確定'; action = '12:30台を待て'
            exp_str = ''
        elif aft_5m_ret > 0.1:
            aft_sig = f'↑{aft_5m_ret:+.2f}%'
            # 銘柄別期待値
            if sym == '5713.T':   action='LONG保有(勝率73%)'; exp_str='+0.66%/73%'
            elif sym == '5803.T': action='LONG保有(勝率83%)'; exp_str='+1.03%/83%'
            else:                 action='LONG方向で保有'  ; exp_str='~+0.5〜1%'
        elif aft_5m_ret < -0.1:
            aft_sig = f'↓{aft_5m_ret:+.2f}%'
            if sym == '5713.T':   action='SHORT/撤退(勝率76%)'; exp_str='-0.48%/24%'
            elif sym == '5803.T': action='SHORT/撤退(勝率67%)'; exp_str='-0.73%/34%'
            else:                 action='SHORT方向で保有'   ; exp_str='~-0.5%'
        else:
            aft_sig = f'→{aft_5m_ret:+.2f}%'; action='様子見'; exp_str=''

        print(f"{name:<12} {gap_sig:>10} {aft_sig:>10}  {action:>20}  {exp_str:>12}")

    # ────────────────────────────────────────
    # SECTION 4: 引け前チェック（15:00）
    # ────────────────────────────────────────
    print_separator("④ 引け前チェック — 15:00 時点でポジションを継続するか")
    print("  【ルール】15:00の段階でパターンを確認。引け30分は方向が継続しやすい")
    print(f"{'銘柄':<12} {'前場方向':>8} {'後場方向':>8} {'15:00寄比':>10}  {'推奨':>22}  {'勝率目安':>8}")
    print("─"*75)

    for sym, name in SYMS.items():
        if sym not in sym_data or len(sym_data[sym]['today_df'])==0: continue
        g   = thr(sym_data[sym]['today_df'])
        gm  = mae_sess(sym_data[sym]['today_df'])
        go  = aft_sess(sym_data[sym]['today_df'])
        if len(gm)<5 or len(go)<5: continue

        op      = g['open'].iloc[0] if len(g)>0 else np.nan
        mae_op  = gm['open'].iloc[0]; mae_cl = gm['close'].iloc[-1]
        aft_op  = go['open'].iloc[0]
        cp1500  = go[(go.index.hour==15)&(go.index.minute==0)]
        cp1500_cl = cp1500['close'].iloc[-1] if len(cp1500)>0 else np.nan

        if np.isnan(op) or op<=0: continue
        mae_ret = (mae_cl/mae_op-1)*100 if mae_op>0 else np.nan
        aft_ret_so_far = (cp1500_cl/aft_op-1)*100 if (not np.isnan(cp1500_cl) and aft_op>0) else np.nan
        cp1500_vs_open = (cp1500_cl/op-1)*100 if (not np.isnan(cp1500_cl) and op>0) else np.nan

        if np.isnan(mae_ret) or np.isnan(aft_ret_so_far): continue

        mae_dir = '↑前高' if mae_ret>0 else '↓前安'
        aft_dir = f'↑+{aft_ret_so_far:.2f}%' if aft_ret_so_far>0 else f'↓{aft_ret_so_far:.2f}%'
        cp1500_str = f'{cp1500_vs_open:+.2f}%' if not np.isnan(cp1500_vs_open) else '---'

        # 引け前推奨
        mae_up  = mae_ret > 0
        aft_pos = aft_ret_so_far > 0

        if mae_up and aft_pos:
            action='継続LONG(前高後高)'; wr='70%'
        elif not mae_up and aft_pos:
            action='継続LONG(V字回復)'; wr='67%'
        elif mae_up and not aft_pos:
            action='手仕舞い推奨(前高後安)'; wr='32%'
        else:
            if sym=='5803.T': action='⚠️即撤退(前安後安)'; wr='27%'
            else:             action='手仕舞い推奨(前安後安)';wr='30%'

        print(f"{name:<12} {mae_dir:>8} {aft_dir:>9} {cp1500_str:>10}  {action:>22}  {wr:>8}")

    # ────────────────────────────────────────
    # SECTION 5: 曜日別の今日の注意事項
    # ────────────────────────────────────────
    print_separator(f"⑤ 今日（{DAY[dow]}曜）の特別注意事項")
    rules = {
        0: [  # 月曜
            "・週末ONギャップ：非鉄平均-0.48%、半導体-0.37%。月曜は全体的に期待値低め",
            "・住山：月曜ONのGDは日中さらに続落（フィル率37%）。寄付直後の小反発は罠",
            "・SBG：月曜ONは全銘柄中最良（中央値+0.26%、勝率56%）",
            "・住山月曜：前場安→後場はV字の期待値あり(+0.385%)",
        ],
        1: [  # 火曜
            "・火曜は全銘柄最弱日。エントリー基準を高めに設定",
            "・🚨フジクラ火曜後場：期待値-1.57%（特に前場安×火曜後場は-2.60%）",
            "・アドバンテスト：火曜は前場高でも後場で売られる（前高後安23件が最多）",
            "・レーザーテック/ディスコ：前場強くても後場でクラッシュするトラップ多発",
            "・キーエンス：寄付直後からマイナス入りするパターンが多い",
        ],
        2: [  # 水曜
            "・水曜は全銘柄でプラス傾向（最強曜日）",
            "・フジクラ水曜後場：期待値+0.58%、勝率65%。後場の保有継続を推奨",
            "・アドバンテスト/キーエンス：水曜は引け直前に急伸するパターン（引け保有推奨）",
            "・住山水曜：前場安→後場V字(+0.289%)。前場安でもパニック売りしない",
        ],
        3: [  # 木曜
            "・木曜は概ね中立〜弱め。大きなバイアスなし",
            "・キーエンス木曜：前場高→後場-0.11%（前場高すぎたら後場手仕舞い検討）",
            "・ディスコ木曜：15:30に+0.530%の急伸パターンあり（木曜引けは保有推奨）",
        ],
        4: [  # 金曜
            "・金曜は非鉄全般が強い（三菱マテSharpe+1.87が最高）",
            "・⭐住山：金曜前場高→後場+0.60%（最強の組み合わせ）。後場保有継続",
            "・⭐三菱マテ：引け直前に跳ねるパターン（15:00→15:30で+0.13%/勝率63%）",
            "・🔴アドバンテスト：金曜引け前-0.10%（勝率48%）。15:00に手仕舞い推奨",
            "・レーザーテック/ディスコ：金曜引け前は継続LONG（勝率60〜63%）",
        ],
    }
    for rule in rules.get(dow, []):
        print(f"  {rule}")

    # ────────────────────────────────────────
    # SECTION 6: ORBチートシート（常時参照）
    # ────────────────────────────────────────
    print_separator("⑥ ORBチートシート（最重要 — 全日共通）")
    print("""
  【エントリー】
  ┌──────────────────────────────────────────────────────────────┐
  │ 9:30 寄比 > +0.3%  → LONG  前場引けを目標（期待値+1.3〜2.3%）│
  │ 9:30 寄比 < -0.3%  → SHORT 前場引けを目標（期待値+0.9〜2.0%）│
  │ 9:30 ±0.3%以内     → 10:00まで待つ（様子見）               │
  └──────────────────────────────────────────────────────────────┘

  【信頼度ランキング（net_Sharpe）】
   ①ディスコ下  Sharpe18.75  ②三菱マテ上  18.51  ③キーエンス上 18.07
   ④キーエンス下 16.99        ⑤レーザー上   16.03  ⑥住山下      14.50

  【エグジット】
   前場引け（11:30）でクローズ — 後場まで引っ張らない（コストが増える）
   10:00で逆転していたら即撤退（損切り最優先）

  【絶対やらないこと】
   ✘ 火曜にフジクラ後場LONG（期待値-1.57%）
   ✘ 前安後安パターンで引け前30分保有（Sharpe -10〜-14）
   ✘ 月曜GD後の寄付直後小反発で買い（罠、その後続落）
   ✘ 週末にポジションを持ち越す（非鉄平均-0.48%/week）
""")

    print(f"{'='*70}")
    print(f"  生成時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, default=None,
                        help='分析日 (YYYY-MM-DD) デフォルト: today')
    args = parser.parse_args()
    generate_signal(args.date)
