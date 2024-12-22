import sqlite3
from datetime import datetime, timedelta
import pandas as pd
import requests
import time

from dotenv import load_dotenv
import os

load_dotenv()
POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')

class StockDatabaseManager:
    def __init__(self, db_path, api_key):
        self.db_path = db_path
        self.api_key = api_key
        self.setup_database()

    def setup_database(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS historical_prices (
                symbol TEXT,
                timestamp DATETIME,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                PRIMARY KEY (symbol, timestamp)
            )
        """)
        conn.commit()
        conn.close()

    def check_ticker(self, symbol):
        """Check if ticker exists and get its latest date"""
        conn = sqlite3.connect(self.db_path)
        result = conn.execute("""
            SELECT MAX(timestamp) 
            FROM historical_prices 
            WHERE symbol = ?
        """, (symbol,)).fetchone()
        conn.close()
        
        if not result[0]:
            return False, None
        return True, pd.to_datetime(result[0])

    def download_full_history(self, symbol):
        start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        return self._fetch_and_store_data(symbol, start_date, end_date)

    def update_ticker(self, symbol):
        exists, latest_date = self.check_ticker(symbol)
        
        if not exists:
            return self.download_full_history(symbol)
            
        start_date = (latest_date + timedelta(days=1)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        if start_date >= end_date:
            return True  # Already up to date
            
        return self._fetch_and_store_data(symbol, start_date, end_date)

    def _fetch_and_store_data(self, symbol, start_date, end_date):
        try:
            url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}?apiKey={self.api_key}"
            response = requests.get(url)
            
            if response.status_code == 429:  # Rate limit
                time.sleep(60)  # Wait for rate limit reset
                return self._fetch_and_store_data(symbol, start_date, end_date)
                
            if response.status_code == 200:
                data = response.json()
                if 'results' in data and data['results']:
                    df = pd.DataFrame(data['results'])
                    df['symbol'] = symbol
                    df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
                    df = df.rename(columns={
                        'o': 'open', 'h': 'high', 
                        'l': 'low', 'c': 'close', 
                        'v': 'volume'
                    })
                    
                    conn = sqlite3.connect(self.db_path)
                    df[['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume']].to_sql(
                        'historical_prices',
                        conn,
                        if_exists='append',
                        index=False
                    )
                    conn.close()
                    return True
                    
            return False
            
        except Exception as e:
            print(f"Error processing {symbol}: {str(e)}")
            return False

    def update_all_tickers(self, symbols):
        results = {}
        for symbol in symbols:
            success = self.update_ticker(symbol)
            results[symbol] = success
        return results


def main():
    load_dotenv()
    POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')
    
    if not POLYGON_API_KEY:
        raise ValueError("POLYGON_API_KEY not found in .env file")

    db_path = os.path.join(os.path.dirname(__file__), "..", "biotech-dashboard-frontend", "public", "data", "stock_data.db")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    manager = StockDatabaseManager(db_path, POLYGON_API_KEY)
    symbols = ["VERA", "ANIX", "INMB", "GUTS", "MIRA", "EYPT", "GNPX", "GLUE", "TGTX", "AUPH", 
               "ITCI", "CLOV", "CERE", "IPHA", "MDWD", "ZYME", "ASMB", "JAZZ", "PRTA", "TMDX",
               "GILD", "NTNX", "INAB", "MNPR", "APVO", "HRMY", "BHC", "BCRX", "GRTX", "AXSM",
               "SMMT", "SAGE", "MYNZ", "GMAB", "LUMO", "NEO", "ARCT", "TEVA", "VMD", "VERU",
               "VRCA", "SIGA", "INMD", "EXEL", "CPRX", "HALO", "NVOS", "ATAI", "BNGO", "ENOV",
               "BIIB", "MIST", "ARDX", "CVM", "ACLS", "IDYA", "RYTM", "TWST", "STEM", "GERN",
               "VIR", "ALKS", "AMPH", "SVRA", "EVLO", "GH", "NTLA", "MRTX", "SRPT", "RARE",
               "TRVI", "PGEN", "EVH", "ARQT", "QNRX", "SYRS", "GTHX", "MNKD", "XERS", "SNDX",
               "PRTK", "PLRX", "MREO", "MDGL", "KZR", "GALT", "ETNB", "EPZM", "CMRX", "CDTX",
               "GYRE", "CBAY", "AGEN", "ABUS", "ABCL", "LOGC", "BLCM", "ADVM", "SNY", "MRSN",
               "TCRT", "ASRT", "ABBV", "ADMA", "RKLB"]  # Add your full list here
    
    print("Starting historical data download...")
    results = manager.update_all_tickers(symbols)
    
    for symbol, success in results.items():
        print(f"{symbol}: {'Success' if success else 'Failed'}")

if __name__ == "__main__":
    main()