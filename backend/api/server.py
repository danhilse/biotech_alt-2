# backend/api/server.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import os
from pathlib import Path

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get the absolute path to the database
BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "stock_data.db"

# Create tables if they don't exist
def initialize_database():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Create the historical_prices table if it doesn't exist
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS historical_prices (
            symbol TEXT,
            timestamp TEXT,
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
        print(f"Database initialized at {DB_PATH}")
    except Exception as e:
        print(f"Error initializing database: {e}")
        raise

def get_db_connection():
    """Create a database connection and handle errors"""
    if not DB_PATH.exists():
        raise HTTPException(
            status_code=500, 
            detail=f"Database file not found at {DB_PATH}. Please ensure the database file exists."
        )
    
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Database connection failed: {str(e)}"
        )

@app.on_event("startup")
async def startup_event():
    """Initialize the database on startup"""
    print("Initializing database...")
    initialize_database()

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchone()
        conn.close()
        return {
            "status": "healthy",
            "database": str(DB_PATH),
            "database_exists": DB_PATH.exists()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=str(e)
        )

@app.get("/api/stocks")
async def get_available_symbols():
    """Get list of all available stock symbols"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT symbol FROM historical_prices")
        symbols = [row[0] for row in cursor.fetchall()]
        conn.close()
        return symbols
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stocks/{symbol}")
async def get_stock_data(symbol: str, timeframe: str = "1Y"):
    """Get historical data for a specific symbol"""
    try:
        conn = get_db_connection()
        
        # Calculate date range
        end_date = datetime.now()
        if timeframe == "5Y":
            start_date = end_date - timedelta(days=1825)
        elif timeframe == "1Y":
            start_date = end_date - timedelta(days=365)
        else:  # Default to 1M
            start_date = end_date - timedelta(days=30)
        
        query = """
        SELECT 
            timestamp,
            open,
            high,
            low,
            close,
            volume
        FROM historical_prices
        WHERE symbol = ? AND timestamp >= ?
        ORDER BY timestamp
        """
        
        df = pd.read_sql_query(
            query,
            conn,
            params=(symbol, start_date.strftime('%Y-%m-%d'))
        )
        
        conn.close()
        
        # Convert timestamps to ISO format
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%dT%H:%M:%S')
        
        return df.to_dict(orient='records')
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print(f"Database path: {DB_PATH}")
    uvicorn.run(app, host="0.0.0.0", port=8000)