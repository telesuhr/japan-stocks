"""jstock — 日本株リサーチ共通ライブラリ。

analyses/ 配下の run.py はここから import する（PG_CONFIG のコピペ禁止）:

    from jstock import db, data, jcal, costs, stats

    df = data.load_daily(codes=["72030"], start="2025-01-01")
    net = costs.net_returns(gross_ret, ls=True)

環境変数は libpq 標準 (PGHOST / PGPORT / PGUSER / PGDATABASE) に統一。
OMEN上では未設定のまま動く（localhost / market_data がデフォルト）。
外部マシンからは PGHOST=omen (Tailscale) を設定。

未実装（必要になったら既存 analyses から移植して育てる）:
    universe: point-in-time 流動性ユニバース / size調整ベンチ
"""
from . import db, data, jcal, costs, stats  # noqa: F401
