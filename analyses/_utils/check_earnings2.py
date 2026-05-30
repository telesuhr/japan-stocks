import pymysql
conn = pymysql.connect(host='100.92.181.92', port=3306, user='rfnews', password='Bleach@924', database='refinitiv_news')
cur = conn.cursor()

print("=== japan_earnings カテゴリ: 半導体関連 ===")
cur.execute("""
SELECT published_at, headline
FROM refinitiv_news
WHERE category = 'japan_earnings'
  AND (headline LIKE '%Disco%' OR headline LIKE '%Tokyo Electron%' OR headline LIKE '%Advantest%'
       OR headline LIKE '%Shin-Etsu%' OR headline LIKE '%Lasertec%' OR headline LIKE '%SCREEN%'
       OR headline LIKE '%Keyence%' OR headline LIKE '%8035%' OR headline LIKE '%6857%'
       OR headline LIKE '%6146%' OR headline LIKE '%6920%')
ORDER BY published_at DESC
""")
rows = cur.fetchall()
print(f"hits: {len(rows)}")
for row in rows:
    print(f"  {row[0]}  {row[1][:90]}")

print()
print("=== japan_semiconductor カテゴリ: 決算関連 ===")
cur.execute("""
SELECT published_at, headline
FROM refinitiv_news
WHERE category = 'japan_semiconductor'
  AND (headline LIKE '%earnings%' OR headline LIKE '%results%' OR headline LIKE '%profit%'
       OR headline LIKE '%Financial%' OR headline LIKE '%forecast%' OR headline LIKE '%revenue%')
ORDER BY published_at DESC LIMIT 30
""")
rows = cur.fetchall()
print(f"hits: {len(rows)}")
for row in rows:
    print(f"  {row[0]}  {row[1][:90]}")

print()
print("=== DIARY-Japan corporate earnings 記事の内容 ===")
cur.execute("""
SELECT published_at, headline, body_text
FROM refinitiv_news
WHERE headline LIKE '%DIARY%Japan%earnings%'
ORDER BY published_at DESC LIMIT 3
""")
rows = cur.fetchall()
print(f"hits: {len(rows)}")
for row in rows:
    print(f"\n  [{row[0]}] {row[1]}")
    if row[2]:
        print(f"  body: {row[2][:500]}")

print()
print("=== ニュースDB 期間確認 ===")
cur.execute("SELECT MIN(published_at), MAX(published_at), COUNT(*) FROM refinitiv_news")
row = cur.fetchone()
print(f"  全体: {row[0]} ~ {row[1]}  N={row[2]}")

cur.execute("SELECT MIN(published_at), MAX(published_at), COUNT(*) FROM refinitiv_news WHERE category='japan_earnings'")
row = cur.fetchone()
print(f"  japan_earnings: {row[0]} ~ {row[1]}  N={row[2]}")

conn.close()
