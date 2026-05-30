import pymysql
conn = pymysql.connect(host='100.92.181.92', port=3306, user='rfnews', password='Bleach@924', database='refinitiv_news')
cur = conn.cursor()

print("=== category一覧 ===")
cur.execute("SELECT DISTINCT category, COUNT(*) as n FROM refinitiv_news GROUP BY category ORDER BY n DESC")
for row in cur.fetchall():
    print(f"  {row[0]}: {row[1]}")

print()
print("=== Financial Results ニュース (日本株関連) ===")
cur.execute("""
SELECT published_at, headline, category
FROM refinitiv_news
WHERE (headline LIKE '%Financial Results%' OR headline LIKE '%Consolidated%' OR headline LIKE '%earnings%')
  AND (headline LIKE '%.T%' OR headline LIKE '%Japan%' OR headline LIKE '%Tokyo Electron%'
       OR headline LIKE '%Advantest%' OR headline LIKE '%Disco%' OR headline LIKE '%Shin-Etsu%'
       OR headline LIKE '%Lasertec%' OR headline LIKE '%SCREEN%' OR headline LIKE '%Keyence%')
ORDER BY published_at DESC
""")
rows = cur.fetchall()
print(f"hits: {len(rows)}")
for row in rows:
    print(f"  {row[0]}  {row[1][:80]}")

print()
print("=== topics 構造サンプル ===")
cur.execute("SELECT published_at, headline, topics FROM refinitiv_news WHERE topics != '[]' LIMIT 3")
for row in cur.fetchall():
    print(f"  [{row[0]}] {row[1][:50]}")
    print(f"    topics: {row[2][:200]}")

print()
print("=== 日本株決算ニュース期間 ===")
cur.execute("""
SELECT MIN(published_at), MAX(published_at), COUNT(*)
FROM refinitiv_news
WHERE headline LIKE '%Financial Results%' OR headline LIKE '%Consolidated%'
""")
row = cur.fetchone()
print(f"  {row[0]} ~ {row[1]}  N={row[2]}")

conn.close()
