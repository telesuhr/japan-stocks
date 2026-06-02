import sys
sys.stdout.reconfigure(line_buffering=True)
import psycopg2

conn = psycopg2.connect(host='192.168.0.118', port=5432, user='postgres', password='postgres', dbname='market_data')
cur = conn.cursor()

cur.execute("SELECT column_name, data_type FROM information_schema.columns WHERE table_schema='macro' AND table_name='daily_ohlcv' ORDER BY ordinal_position")
print('=== macro.daily_ohlcv columns ===')
for r in cur.fetchall():
    print(r)

cur.execute("SELECT DISTINCT symbol FROM macro.daily_ohlcv ORDER BY symbol")
print('\n=== macro symbols ===')
for r in cur.fetchall():
    print(r[0])

cur.execute("SELECT column_name FROM information_schema.columns WHERE table_schema='nas_archive' AND table_name='daily_data' ORDER BY ordinal_position")
print('\n=== nas_archive.daily_data columns ===')
for r in cur.fetchall():
    print(r)

cur.execute("SELECT DISTINCT symbol FROM nas_archive.daily_data WHERE symbol LIKE '%CU%' OR symbol LIKE '%CMCU%' ORDER BY symbol")
print('\n=== nas_archive.daily_data copper symbols ===')
for r in cur.fetchall():
    print(r[0])

cur.execute("SELECT DISTINCT symbol FROM nas_archive.intraday_data WHERE symbol LIKE '%CU%' OR symbol LIKE '%CMCU%' ORDER BY symbol")
print('\n=== nas_archive.intraday_data copper symbols ===')
for r in cur.fetchall():
    print(r[0])

cur.execute("SELECT column_name, data_type FROM information_schema.columns WHERE table_schema='nas_archive' AND table_name='intraday_data' ORDER BY ordinal_position")
print('\n=== nas_archive.intraday_data columns ===')
for r in cur.fetchall():
    print(r)

# intraday_data CMCU3 の日付範囲
cur.execute("SELECT MIN(timestamp), MAX(timestamp), COUNT(*) FROM nas_archive.intraday_data WHERE symbol='CMCU3'")
print('\n=== CMCU3 range ===')
for r in cur.fetchall():
    print(r)

conn.close()
print('\nDone')
