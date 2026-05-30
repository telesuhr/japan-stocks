import pymysql
conn = pymysql.connect(host='100.92.181.92', port=3306, user='rfnews', password='Bleach@924', database='refinitiv_news')
cur = conn.cursor()

print("=== refinitiv_fetch_log (何を取得しているか) ===")
cur.execute("DESCRIBE refinitiv_fetch_log")
for row in cur.fetchall():
    print(f"  {row}")

cur.execute("SELECT * FROM refinitiv_fetch_log ORDER BY id DESC LIMIT 10")
for row in cur.fetchall():
    print(f"  {row}")

print()
print("=== external_fetch_log ===")
cur.execute("DESCRIBE external_fetch_log")
for row in cur.fetchall():
    print(f"  {row}")

cur.execute("SELECT * FROM external_fetch_log ORDER BY id DESC LIMIT 10")
for row in cur.fetchall():
    print(f"  {row}")

print()
print("=== daily_data: 半導体銘柄の最新データ確認 ===")
cur.execute("""
SELECT symbol, MIN(trade_date), MAX(trade_date), COUNT(*) as n
FROM daily_data
WHERE symbol IN ('8035.T','6857.T','6146.T','4063.T','6963.T','6920.T')
GROUP BY symbol ORDER BY symbol
""")
for row in cur.fetchall():
    print(f"  {row}")

conn.close()
