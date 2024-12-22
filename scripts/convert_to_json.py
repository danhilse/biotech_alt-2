import sqlite3
import json

def convert_db_to_json():
    conn = sqlite3.connect('stock_data.db')
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT symbol, timestamp, open, high, low, close, volume
        FROM historical_prices
        ORDER BY symbol, timestamp
    """)
    
    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    
    data = [
        dict(zip(columns, row))
        for row in rows
    ]
    
    with open('public/data/stock_data.json', 'w') as f:
        json.dump(data, f)
    
    conn.close()

if __name__ == '__main__':
    convert_db_to_json()