import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_insider_summary(ticker: yf.Ticker) -> Dict[str, Any]:
    """
    Get insider trading summary from recent transactions
    """
    try:
        transactions = ticker.insider_transactions
        if transactions is None or transactions.empty:
            return {
                "recent_trades": 0,
                "net_shares": 0,
                "notable_trades": []
            }
        
        # Get last 30 days of transactions
        recent_date = datetime.now() - timedelta(days=30)
        recent_trades = transactions[transactions.index >= recent_date]
        
        notable_trades = []
        for idx, trade in recent_trades.iterrows():
            if abs(trade.get('Shares', 0)) > 5000 or abs(trade.get('Value', 0)) > 100000:
                notable_trades.append({
                    "date": idx.strftime('%Y-%m-%d'),
                    "insider": trade.get('Insider', 'Unknown'),
                    "shares": int(trade.get('Shares', 0)),
                    "value": float(trade.get('Value', 0)),
                    "transaction": trade.get('Transaction', 'Unknown')
                })
        
        return {
            "recent_trades": len(recent_trades),
            "net_shares": int(recent_trades['Shares'].sum()),
            "notable_trades": notable_trades[:5]  # Limit to top 5 notable trades
        }
    except Exception as e:
        logger.error(f"Error fetching insider data: {str(e)}")
        return {
            "recent_trades": 0,
            "net_shares": 0,
            "notable_trades": []
        }

def get_news_summary(ticker: yf.Ticker) -> List[Dict[str, str]]:
    """
    Get recent news articles summary
    """
    try:
        news = ticker.news
        if not news:
            return []
        
        recent_news = []
        for article in news[:5]:  # Limit to 5 most recent articles
            recent_news.append({
                "title": article.get('title', ''),
                "publisher": article.get('publisher', ''),
                "timestamp": datetime.fromtimestamp(article.get('providerPublishTime', 0)).strftime('%Y-%m-%d %H:%M:%S'),
                "type": article.get('type', ''),
                "url": article.get('link', '')
            })
        return recent_news
    except Exception as e:
        logger.error(f"Error fetching news data: {str(e)}")
        return []

def fetch_market_data(tickers: List[str]) -> List[Dict[str, Any]]:
    """
    Fetch comprehensive market data for the given tickers using yfinance
    Returns data in the format needed for the dashboard
    """
    market_data = []
    
    def convert_to_native(obj):
        if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        if isinstance(obj, dict):
            return {key: convert_to_native(value) for key, value in obj.items()}
        return obj
    
    for symbol in tickers:
        try:
            logger.info(f"Fetching data for {symbol}")
            ticker = yf.Ticker(symbol)
            
            # Get company info and sector data
            info = ticker.info
            sector = info.get('sector', 'Unknown')
            industry = info.get('industry', 'Unknown')
            
            # Get volume and price data (existing code)
            volume_data = ticker.history(period="6mo", interval="1d")
            if volume_data.empty:
                logger.error(f"{symbol}: Volume data is empty")
                continue
            
            # Calculate volume metrics (existing code)
            available_days = len(volume_data)
            rolling_window = min(90, available_days)
            volume_series = volume_data['Volume']
            rolling_avg = volume_series.rolling(window=rolling_window, min_periods=1).mean().iloc[-1]
            recent_volumes = volume_series.tail(5).tolist()
            current_volume = recent_volumes[-1]
            prev_volume = recent_volumes[-2]
            
            volume_24h_change_pct = ((current_volume - prev_volume) / prev_volume) * 100
            volume_vs_avg_pct = ((current_volume - rolling_avg) / rolling_avg) * 100

            # Get today's price data (existing code)
            today_data = ticker.history(period="1d", interval="1h")
            if today_data.empty:
                logger.error(f"{symbol}: Today's data is empty")
                continue

            current_price = today_data['Close'].iloc[-1]
            open_price = today_data['Open'].iloc[0]
            days_high = today_data['High'].max()
            days_low = today_data['Low'].min()
            prev_close = volume_data['Close'].iloc[-2]
            price_change = current_price - prev_close
            price_change_pct = (price_change / prev_close) * 100

            # Get insider trading data
            insider_data = get_insider_summary(ticker)
            
            # Get news data
            news_data = get_news_summary(ticker)
            
            # Get analyst recommendations
            try:
                recommendations = ticker.recommendations_summary
                latest_rec = recommendations.iloc[-1] if not recommendations.empty else None
                analyst_rating = {
                    "rating": str(latest_rec.get('To Grade', 'N/A')) if latest_rec is not None else 'N/A',
                    "firm": str(latest_rec.get('Firm', 'N/A')) if latest_rec is not None else 'N/A',
                    "date": latest_rec.name.strftime('%Y-%m-%d') if latest_rec is not None else 'N/A'
                }
            except Exception as e:
                logger.error(f"Error fetching analyst data: {str(e)}")
                analyst_rating = {"rating": "N/A", "firm": "N/A", "date": "N/A"}

            # Calculate alerts including new criteria
            alerts = sum([
                abs(price_change_pct) > 5,     # 5% price movement
                volume_24h_change_pct >= 10,    # 10% volume spike
                volume_24h_change_pct >= 20,    # 20% volume spike
                volume_vs_avg_pct > 50,         # Significantly above average volume
                insider_data['recent_trades'] > 0,  # Recent insider activity
                len(news_data) > 0              # Recent news
            ])
    
            market_data.append(convert_to_native({
                "symbol": symbol,
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                # Company info
                "sector": sector,
                "industry": industry,
                # Price metrics (existing)
                "price": round(float(current_price), 2),
                "priceChange": round(float(price_change_pct), 2),
                "openPrice": round(float(open_price), 2),
                "prevClose": round(float(prev_close), 2),
                "dayHigh": round(float(days_high), 2),
                "dayLow": round(float(days_low), 2),
                # Volume metrics (existing)
                "volume": int(current_volume),
                "prevVolume": int(prev_volume),
                "averageVolume": float(rolling_avg),
                "volumeChange": round(float(volume_24h_change_pct), 2),
                "volumeVsAvg": round(float(volume_vs_avg_pct), 2),
                "recentVolumes": [int(vol) for vol in recent_volumes],
                # Market metrics (existing)
                "marketCap": int(info.get('marketCap', 0)),
                "fiftyTwoWeekHigh": round(float(info.get('fiftyTwoWeekHigh', 0)), 2),
                "fiftyTwoWeekLow": round(float(info.get('fiftyTwoWeekLow', 0)), 2),
                # New data
                "insiderActivity": insider_data,
                "recentNews": news_data,
                "analystRating": analyst_rating,
                # Alert info (updated)
                "alerts": int(alerts),
                "alertDetails": {
                    "priceAlert": bool(abs(price_change_pct) > 5),
                    "volumeSpike10": bool(volume_24h_change_pct >= 10),
                    "volumeSpike20": bool(volume_24h_change_pct >= 20),
                    "highVolume": bool(volume_vs_avg_pct > 50),
                    "insiderAlert": bool(insider_data['recent_trades'] > 0),
                    "newsAlert": bool(len(news_data) > 0)
                }
            }))
                
            logger.info(f"Successfully processed {symbol}")
            
        except Exception as e:
            logger.error(f"Unexpected error processing {symbol}: {str(e)}")
            continue
    
    # Save to JSON file with verification
    try:
        json_str = json.dumps(market_data, indent=2)
        with open('market_data.json', 'w') as f:
            f.write(json_str)
        logger.info(f"Saved data for {len(market_data)} stocks")
        
        with open('market_data.json', 'r') as f:
            verification_data = json.load(f)
        if len(verification_data) == len(market_data):
            logger.info("JSON file verification successful")
        else:
            logger.error(f"JSON file verification failed: Expected {len(market_data)} stocks, found {len(verification_data)}")
    except Exception as e:
        logger.error(f"Error saving to JSON: {str(e)}")

    return market_data

def main():
    # Your existing list of tickers here
    tickers = ["VERA", "ANIX", "INMB", "GUTS", "MIRA", "EYPT", "GNPX", "GLUE", "TGTX", "AUPH", 
               "ITCI", "CLOV", "CERE", "IPHA", "MDWD", "ZYME", "ASMB", "JAZZ", "PRTA", "TMDX",
               "GILD", "NTNX", "INAB", "MNPR", "APVO", "HRMY", "BHC", "BCRX", "GRTX", "AXSM",
               "SMMT", "SAGE", "MYNZ", "GMAB", "LUMO", "NEO", "ARCT", "TEVA", "VMD", "VERU",
               "VRCA", "SIGA", "INMD", "EXEL", "CPRX", "HALO", "NVOS", "ATAI", "BNGO", "ENOV",
               "BIIB", "MIST", "ARDX", "CVM", "ACLS", "IDYA", "RYTM", "TWST", "STEM", "GERN",
               "VIR", "ALKS", "AMPH", "SVRA", "EVLO", "GH", "NTLA", "MRTX", "SRPT", "RARE",
               "TRVI", "PGEN", "EVH", "ARQT", "QNRX", "SYRS", "GTHX", "MNKD", "XERS", "SNDX",
               "PRTK", "PLRX", "MREO", "MDGL", "KZR", "GALT", "ETNB", "EPZM", "CMRX", "CDTX",
               "GYRE", "CBAY", "AGEN", "ABUS", "ABCL", "LOGC", "BLCM", "ADVM", "SNY", "MRSN",
               "TCRT", "ASRT", "ABBV", "ADMA", "RKLB"]

    # Fetch and save data
    market_data = fetch_market_data(tickers)
    
    # Print summary of interesting stocks
    print("\nStocks with alerts:")
    alerts = [stock for stock in market_data if stock['alerts'] > 0]
    alerts.sort(key=lambda x: x['alerts'], reverse=True)
    
    for stock in alerts:
        print(f"\n{stock['symbol']} ({stock['sector']}): {stock['alerts']} alerts")
        print(f"  Price: ${stock['price']} ({stock['priceChange']}%)")
        print(f"  Volume: {stock['volumeChange']}% daily change")
        print(f"  Volume vs Avg: {stock['volumeVsAvg']}%")
        
        if stock['alertDetails']['priceAlert']:
            print("  - Major price movement")
        if stock['alertDetails']['volumeSpike20']:
            print("  - Major volume spike")
        elif stock['alertDetails']['volumeSpike10']:
            print("  - Notable volume increase")
        if stock['alertDetails']['highVolume']:
            print("  - Significantly above average volume")
        if stock['alertDetails']['insiderAlert']:
            print(f"  - Recent insider activity: {stock['insiderActivity']['recent_trades']} trades")
        if stock['alertDetails']['newsAlert']:
            print(f"  - Recent news: {len(stock['recentNews'])} articles")
            
        # Print latest news if available
        if stock['recentNews']:
            print("\n  Latest news:")
            for article in stock['recentNews'][:2]:  # Show only 2 most recent
                print(f"    - {article['title']} ({article['publisher']})")

if __name__ == "__main__":
    main()
