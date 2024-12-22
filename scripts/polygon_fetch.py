import requests
import yfinance as yf
from datetime import datetime, timedelta
import time
import json
import logging
from typing import Dict, List, Any
from ratelimit import limits, sleep_and_retry
import pandas as pd

from dotenv import load_dotenv
import os

load_dotenv()
POLYGON_KEY = os.getenv('POLYGON_API_KEY')


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridDataFetcher:
    def __init__(self, polygon_key: str):
        self.polygon_key = polygon_key
        self.base_url = "https://api.polygon.io"
        
    @sleep_and_retry
    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """
        Make rate-limited API request to Polygon.io
        """
        if params is None:
            params = {}
            
        # Add API key to parameters
        params['apiKey'] = self.polygon_key
        
        try:
            response = requests.get(f"{self.base_url}{endpoint}", params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            if response.status_code == 429:
                logger.warning("Rate limit hit, sleeping for 60 seconds...")
                time.sleep(60)
            return None
        
    def get_detailed_financials(self, symbol: str) -> Dict[str, Any]:
        try:
            ticker = yf.Ticker(symbol)
            
            # Get statements and log what we receive
            income_stmt = ticker.income_stmt
            cashflow = ticker.cashflow
            
            # Convert to DataFrame if not already
            quarterly_income = pd.DataFrame(income_stmt)
            quarterly_cashflow = pd.DataFrame(cashflow)
            
            financials = {
                "revenue": [],
                "revenueChanges": [],
                "netIncome": [],
                "netIncomeChanges": [],
                "operatingMargin": [],  # Change from gross margin to operating margin
                "operatingMarginChanges": [],
                "freeCashFlow": [],
                "freeCashFlowChanges": [],
                "dates": []
            }
            
            if not quarterly_income.empty:
                revenue = quarterly_income.loc['Total Revenue'] if 'Total Revenue' in quarterly_income.index else None
                net_income = quarterly_income.loc['Net Income'] if 'Net Income' in quarterly_income.index else None
                operating_income = quarterly_income.loc['Operating Income'] if 'Operating Income' in quarterly_income.index else None
                
                if revenue is not None:
                    for date in revenue.index:
                        # Add date
                        financials["dates"].append(date.strftime('%Y-%m-%d'))
                        
                        # Process Revenue
                        current_revenue = float(revenue[date])
                        financials["revenue"].append(current_revenue)
                        
                        # Calculate YoY revenue change
                        if date.year > revenue.index[0].year:
                            prev_year_value = revenue[revenue.index[revenue.index.year == date.year - 1][0]]
                            revenue_change = ((current_revenue - prev_year_value) / prev_year_value * 100 if prev_year_value != 0 else 0)
                            financials["revenueChanges"].append(round(revenue_change, 2))
                        else:
                            financials["revenueChanges"].append(None)
                        
                        # Process Net Income
                        if net_income is not None and date in net_income.index:
                            current_net_income = float(net_income[date])
                            financials["netIncome"].append(current_net_income)
                            
                            if date.year > net_income.index[0].year:
                                prev_year_value = net_income[net_income.index[net_income.index.year == date.year - 1][0]]
                                net_income_change = ((current_net_income - prev_year_value) / abs(prev_year_value) * 100 if prev_year_value != 0 else 0)
                                financials["netIncomeChanges"].append(round(net_income_change, 2))
                            else:
                                financials["netIncomeChanges"].append(None)
                        else:
                            financials["netIncome"].append(None)
                            financials["netIncomeChanges"].append(None)
                        
                        # Process Operating Margin
                        if operating_income is not None and date in operating_income.index:
                            current_operating_income = float(operating_income[date])
                            current_operating_margin = (current_operating_income / current_revenue * 100) if current_revenue != 0 else 0
                            financials["operatingMargin"].append(round(current_operating_margin, 2))
                            
                            if date.year > operating_income.index[0].year:
                                prev_year_revenue = revenue[revenue.index[revenue.index.year == date.year - 1][0]]
                                prev_year_operating_income = operating_income[operating_income.index[operating_income.index.year == date.year - 1][0]]
                                prev_operating_margin = (prev_year_operating_income / prev_year_revenue * 100) if prev_year_revenue != 0 else 0
                                margin_change = current_operating_margin - prev_operating_margin
                                financials["operatingMarginChanges"].append(round(margin_change, 2))
                            else:
                                financials["operatingMarginChanges"].append(None)
                        else:
                            financials["operatingMargin"].append(None)
                            financials["operatingMarginChanges"].append(None)
            
            return financials
            
        except Exception as e:
            logger.error(f"Error fetching detailed financials for {symbol}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())  # Add stack trace for debugging
            return {
                "revenue": [],
                "revenueChanges": [],
                "netIncome": [],
                "netIncomeChanges": [],
                "operatingMargin": [],
                "operatingMarginChanges": [],
                "freeCashFlow": [],
                "freeCashFlowChanges": [],
                "dates": []
            }
    def get_financials_metrics(self, symbol: str) -> Dict[str, Any]:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            income_stmt = ticker.income_stmt  # Use this instead of quarterly_earnings
            
            
            # Get PE Ratios
            pe_ratios = {
                "trailingPE": info.get('trailingPE'),
                "forwardPE": info.get('forwardPE'),
            }
            
            # Get Estimates
            estimates = {
                "revenue": {
                    "nextQuarter": info.get('revenueEstimateNextQuarter'),
                    "currentYear": info.get('revenueEstimateCurrentYear'),
                    "nextYear": info.get('revenueEstimateNextYear'),
                },
                "earnings": {
                    "nextQuarter": info.get('estimateNextQuarter'),
                    "currentYear": info.get('estimateCurrentYear'),
                    "nextYear": info.get('estimateNextYear'),
                }
            }
            
            # Get quarterly data
            quarterly_revenue = pd.DataFrame(ticker.quarterly_earnings)
            quarterly_balance = pd.DataFrame(ticker.quarterly_balance_sheet)
            quarterly_cashflow = pd.DataFrame(ticker.quarterly_cashflow)
            
            # Process quarterly data
            quarterly_metrics = {
                "revenue": [],
                "revenueChange": [],
                "periods": []
            }
            
            if not quarterly_revenue.empty:
                for date, row in quarterly_revenue.iterrows():
                    quarterly_metrics["periods"].append(date.strftime('%Y-%m-%d'))
                    quarterly_metrics["revenue"].append(float(row['Revenue']))
                    # Calculate YoY change
                    prev_year = date - pd.DateOffset(years=1)
                    if prev_year in quarterly_revenue.index:
                        change = ((row['Revenue'] - quarterly_revenue.loc[prev_year, 'Revenue']) / 
                                quarterly_revenue.loc[prev_year, 'Revenue'] * 100)
                        quarterly_metrics["revenueChange"].append(round(change, 2))
                    else:
                        quarterly_metrics["revenueChange"].append(None)
            
            metrics = {
                "peRatios": pe_ratios,
                "estimates": estimates,
                "quarterly": quarterly_metrics
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error fetching financial metrics for {symbol}: {str(e)}")
            return {
                "peRatios": {"trailingPE": None, "forwardPE": None},
                "estimates": {
                    "revenue": {"nextQuarter": None, "currentYear": None, "nextYear": None},
                    "earnings": {"nextQuarter": None, "currentYear": None, "nextYear": None}
                },
                "quarterly": {
                    "revenue": [],
                    "revenueChange": [],
                    "periods": []
                }
            }

    def get_insider_data(self, symbol: str, months_lookback: int = 3) -> Dict[str, Any]:
        """
        Get enhanced insider trading data from yfinance with fixed transaction categorization
        """
        try:
            logger.info(f"Fetching insider data for {symbol} from yfinance")
            ticker = yf.Ticker(symbol)
            transactions = ticker.insider_transactions
            
            if transactions is None or transactions.empty:
                return {
                    "recent_trades": 0,
                    "net_shares": 0,
                    "notable_trades": [],
                    "summary": {
                        "total_sales": 0,
                        "total_purchases": 0,
                        "total_awards": 0,
                        "sales_count": 0,
                        "purchases_count": 0,
                        "awards_count": 0
                    }
                }
            
            # Convert dates and ensure numeric values
            transactions['Start Date'] = pd.to_datetime(transactions['Start Date'])
            transactions['Value'] = pd.to_numeric(transactions['Value'], errors='coerce')
            transactions['Shares'] = pd.to_numeric(transactions['Shares'], errors='coerce')
            
            # Filter for recent transactions
            cutoff_date = datetime.now() - timedelta(days=30 * months_lookback)
            recent_transactions = transactions[transactions['Start Date'] >= cutoff_date].copy()
            
            if recent_transactions.empty:
                return {
                    "recent_trades": 0,
                    "net_shares": 0,
                    "notable_trades": [],
                    "summary": {
                        "total_sales": 0,
                        "total_purchases": 0,
                        "total_awards": 0,
                        "sales_count": 0,
                        "purchases_count": 0,
                        "awards_count": 0
                    }
                }
            
            # Categorize transactions based on price per share and value
            def categorize_transaction(row):
                if pd.isna(row['Transaction']) or row['Transaction'] == '':
                    # If transaction type is empty, categorize based on price and value
                    if row['Value'] > 0:
                        price_per_share = row['Value'] / row['Shares']
                        if price_per_share < 20:  # Threshold for identifying awards/grants
                            return 'Stock Award'
                        else:
                            return 'Sale'  # High value per share typically indicates a sale
                return row['Transaction']
            
            recent_transactions['Transaction_Category'] = recent_transactions.apply(categorize_transaction, axis=1)
            
            # Categorize based on the enhanced transaction type
            sales = recent_transactions[recent_transactions['Transaction_Category'] == 'Sale']
            awards = recent_transactions[recent_transactions['Transaction_Category'] == 'Stock Award']
            purchases = recent_transactions[
                ~recent_transactions.index.isin(sales.index) & 
                ~recent_transactions.index.isin(awards.index)
            ]
            
            # Calculate net shares (sales are negative, awards and purchases are positive)
            net_shares = (
                purchases['Shares'].sum() + 
                awards['Shares'].sum() - 
                sales['Shares'].sum()
            )
            
            # Format notable trades (any transaction over $100,000)
            notable_trades = []
            significant_transactions = recent_transactions[recent_transactions['Value'] >= 100000]
            
            for _, trade in significant_transactions.iterrows():
                price_per_share = float(trade['Value'] / trade['Shares']) if trade['Value'] > 0 and trade['Shares'] != 0 else 0.0
                transaction_type = trade['Transaction_Category']
                
                notable_trades.append({
                    "date": trade['Start Date'].strftime('%Y-%m-%d'),
                    "insider": trade['Insider'],
                    "position": trade['Position'],
                    "type": transaction_type,
                    "shares": int(trade['Shares']),
                    "value": float(trade['Value']),
                    "price_per_share": price_per_share
                })
            
            # Sort notable trades by date (most recent first)
            notable_trades.sort(key=lambda x: x['date'], reverse=True)
            
            summary = {
                "total_sales": int(abs(sales['Shares'].sum()) if not sales.empty else 0),
                "total_purchases": int(purchases['Shares'].sum() if not purchases.empty else 0),
                "total_awards": int(awards['Shares'].sum() if not awards.empty else 0),
                "sales_count": len(sales),
                "purchases_count": len(purchases),
                "awards_count": len(awards),
                "total_value": {
                    "sales": float(sales['Value'].sum() if not sales.empty else 0),
                    "purchases": float(purchases['Value'].sum() if not purchases.empty else 0),
                    "awards": float(awards['Value'].sum() if not awards.empty else 0)
                }
            }
            
            return {
                "recent_trades": len(recent_transactions),
                "net_shares": int(net_shares),
                "notable_trades": notable_trades,
                "summary": summary,
                "latest_date": recent_transactions['Start Date'].max().strftime('%Y-%m-%d') if not recent_transactions.empty else None
            }
            
        except Exception as e:
            logger.error(f"Error fetching insider data for {symbol}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "recent_trades": 0,
                "net_shares": 0,
                "notable_trades": [],
                "summary": {
                    "total_sales": 0,
                    "total_purchases": 0,
                    "total_awards": 0,
                    "sales_count": 0,
                    "purchases_count": 0,
                    "awards_count": 0,
                    "total_value": {
                        "sales": 0,
                        "purchases": 0,
                        "awards": 0
                    }
                },
                "latest_date": None
            }
            
    def get_investing_url(self, symbol: str, polygon_name: str = None, yf_name: str = None) -> str:
        """
        Generate Investing.com URL using Polygon.io name as primary source
        """
        try:
            # Common URL pattern transformations
            common_mappings = {
                "VERA": "vera-therapeutics",
                "BIIB": "biogen-idec",
                "GILD": "gilead-sciences",
                "ABBV": "abbvie"
            }
            
            # Check if we have a direct mapping
            if symbol in common_mappings:
                base_name = common_mappings[symbol]
            else:
                # Try to create URL-friendly name from Polygon.io name first
                if polygon_name:
                    base_name = polygon_name.lower()
                    suffixes = [" inc", " inc.", " corporation", " corp", " corp.", 
                            " ltd", " ltd.", " limited", " llc", " llc.", 
                            " plc", " plc.", ", inc", ", inc."]
                    for suffix in suffixes:
                        base_name = base_name.replace(suffix, "")
                    
                    # Clean up special characters
                    base_name = base_name.replace("&", "and")
                    base_name = base_name.replace(".", "")
                    base_name = base_name.replace(",", "")
                    base_name = base_name.replace("'", "")
                    base_name = base_name.replace('"', "")
                    base_name = base_name.replace('/', "-")
                    base_name = base_name.replace(" & ", "-")
                    base_name = base_name.replace(" ", "-")
                    base_name = base_name.strip("-")
                elif yf_name:
                    # Fallback to yfinance name if Polygon name is not available
                    base_name = yf_name.lower()
                    # Apply same cleaning rules as above
                else:
                    # Last resort: use symbol
                    base_name = symbol.lower()
            
            return f"https://www.investing.com/equities/{base_name}"
        
        except Exception as e:
            logger.error(f"Error generating Investing.com URL for {symbol}: {str(e)}")
            return f"https://www.investing.com/search/?q={symbol}"

    def get_company_details(self, symbol: str) -> Dict[str, Any]:
        """
        Get company details with Polygon.io name as primary source
        """
        try:
            company_details = {
                "sector": "Unknown",
                "industry": "Unknown",
                "description": None,
                "icon_url": None,
                "logo_url": None,
                "investing_url": None
            }
            
            # Get Polygon.io data first
            logger.info(f"Fetching data for {symbol} from Polygon.io")
            try:
                endpoint = f"/v3/reference/tickers/{symbol}"
                polygon_data = self._make_request(endpoint)
                
                if polygon_data and 'results' in polygon_data:
                    results = polygon_data['results']
                    branding = results.get('branding', {})
                    
                    # Get the official name from Polygon
                    polygon_name = results.get('name')
                    
                    icon_url = branding.get('icon_url')
                    logo_url = branding.get('logo_url')
                    
                    if icon_url:
                        icon_url = f"{icon_url}?apiKey={self.polygon_key}"
                    if logo_url:
                        logo_url = f"{logo_url}?apiKey={self.polygon_key}"
                    
                    company_details.update({
                        "name": polygon_name,
                        "description": results.get('description'),
                        "icon_url": icon_url,
                        "logo_url": logo_url,
                        "sector": results.get('sic_sector', 'Unknown'),
                        "industry": results.get('sic_industry', 'Unknown')
                    })
            except Exception as e:
                logger.error(f"Error fetching Polygon.io company data: {str(e)}")
                polygon_name = None

            # Get additional data from yfinance
            logger.info(f"Fetching additional data for {symbol} from yfinance")
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                yf_name = info.get('longName')
                
                # Update with yfinance data only if Polygon didn't provide it
                if company_details["sector"] == "Unknown":
                    company_details["sector"] = info.get('sector', 'Unknown')
                if company_details["industry"] == "Unknown":
                    company_details["industry"] = info.get('industry', 'Unknown')
                
                # Store both names if available
                company_details.update({
                    "names": {
                        "polygon": polygon_name,
                        "yfinance": yf_name,
                        "short": info.get('shortName', symbol)
                    }
                })
            except Exception as e:
                logger.error(f"Error fetching yfinance company data: {str(e)}")
                yf_name = None

            # Generate Investing.com URL using Polygon name as primary source
            company_details["investing_url"] = self.get_investing_url(
                symbol, 
                polygon_name=polygon_name,
                yf_name=yf_name
            )
            
            return company_details
        
        except Exception as e:
            logger.error(f"Error in get_company_details: {str(e)}")
            return {
                "sector": "Unknown",
                "industry": "Unknown",
                "description": None,
                "icon_url": None,
                "logo_url": None,
                "investing_url": self.get_investing_url(symbol),
                "names": {
                    "polygon": None,
                    "yfinance": None,
                    "short": symbol
                }
            }
    def get_aggregates(self, symbol: str, days: int = 90) -> Dict:
        """
        Get daily aggregates from Polygon.io
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        endpoint = f"/v2/aggs/ticker/{symbol}/range/1/day/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
        return self._make_request(endpoint)

    def get_ticker_details(self, symbol: str) -> Dict:
        """
        Get company details from Polygon.io
        """
        endpoint = f"/v3/reference/tickers/{symbol}"
        return self._make_request(endpoint)

    def get_sma(self, symbol: str, window: int = 90) -> Dict:
        """
        Calculate SMA from Polygon.io data
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=window + 10)
        
        endpoint = f"/v2/aggs/ticker/{symbol}/range/1/day/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
        data = self._make_request(endpoint)
        
        if not data or 'results' not in data:
            return None
            
        volumes = [bar['v'] for bar in data['results']]
        if len(volumes) < window:
            return None
            
        sma = sum(volumes[-window:]) / window
        return {'value': sma}

    def get_rsi(self, symbol: str) -> Dict:
        """
        Calculate RSI from Polygon.io data
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=15)
        
        endpoint = f"/v2/aggs/ticker/{symbol}/range/1/day/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
        data = self._make_request(endpoint)
        
        if not data or 'results' not in data:
            return None
            
        closes = [bar['c'] for bar in data['results']]
        if len(closes) < 14:
            return None
            
        gains = []
        losses = []
        for i in range(1, len(closes)):
            change = closes[i] - closes[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
                
        avg_gain = sum(gains) / len(gains)
        avg_loss = sum(losses) / len(losses)
        
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
        return {'value': rsi}
    def get_volume_history(self, symbol: str, days: int = 5) -> Dict:
        """
        Get detailed volume history from Polygon.io
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days + 1)  # Add buffer day
            
            endpoint = f"/v2/aggs/ticker/{symbol}/range/1/day/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
            data = self._make_request(endpoint)
            
            if not data or 'results' not in data or not data['results']:
                logger.error(f"No volume history data available for {symbol}")
                return {
                    'volumes': [],
                    'dates': [],
                    'volume_changes': []
                }
                
            # Get volume history
            volume_data = data['results']
            volumes = [bar['v'] for bar in volume_data]
            dates = [datetime.fromtimestamp(bar['t']/1000).strftime('%Y-%m-%d') for bar in volume_data]
            
            # Calculate daily volume changes
            volume_changes = []
            for i in range(1, len(volumes)):
                if volumes[i-1] > 0:
                    change = ((volumes[i] - volumes[i-1]) / volumes[i-1]) * 100
                    volume_changes.append(round(change, 2))
                else:
                    volume_changes.append(0.0)
                    
            return {
                'volumes': volumes,
                'dates': dates,
                'volume_changes': volume_changes
            }
        except Exception as e:
            logger.error(f"Error getting volume history for {symbol}: {str(e)}")
            return {
                'volumes': [],
                'dates': [],
                'volume_changes': []
            }

    def get_volume_metrics(self, symbol: str) -> Dict:
        """
        Get comprehensive volume metrics using both Polygon and yfinance data
        """
        try:
            # Get recent daily volumes from Polygon (more reliable for recent data)
            volume_history = self.get_volume_history(symbol, days=5)
            recent_volumes = volume_history['volumes']
            
            if not recent_volumes:
                logger.error(f"No recent volume data from Polygon for {symbol}")
                return self._empty_volume_metrics()
            
            # Get 90-day average volume from Polygon
            end_date = datetime.now()
            start_date = end_date - timedelta(days=90)
            
            aggs_endpoint = f"/v2/aggs/ticker/{symbol}/range/1/day/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
            data = self._make_request(aggs_endpoint)
            
            if not data or 'results' not in data or not data['results']:
                logger.error(f"No aggregate data available for {symbol}")
                return self._empty_volume_metrics()
            
            volumes = [bar['v'] for bar in data['results']]
            if len(volumes) < 5:
                logger.error(f"Insufficient data points for {symbol}")
                return self._empty_volume_metrics()
            
            # Calculate metrics
            current_volume = recent_volumes[-1]
            prev_volume = recent_volumes[-2]
            volume_sma = sum(volumes) / len(volumes)
            
            # Calculate changes
            volume_24h_change = ((current_volume - prev_volume) / prev_volume * 100) if prev_volume > 0 else 0
            volume_vs_avg = ((current_volume - volume_sma) / volume_sma * 100) if volume_sma > 0 else 0
            
            metrics = {
                'current_volume': current_volume,
                'prev_volume': prev_volume,
                'sma': volume_sma,
                'recent_volumes': recent_volumes[-5:],  # Last 5 days
                'volume_dates': volume_history['dates'][-5:],  # Dates for the volumes
                'daily_changes': volume_history['volume_changes'][-4:],  # Last 4 daily changes
                'current_ratio': current_volume / volume_sma if volume_sma > 0 else None,
                'volume_24h_change': round(volume_24h_change, 2),
                'volume_vs_avg': round(volume_vs_avg, 2),
                'days_included': len(volumes)
            }
            
            logger.info(f"Calculated volume metrics for {symbol}: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting volume metrics for {symbol}: {str(e)}")
            return self._empty_volume_metrics()

    def _empty_volume_metrics(self) -> Dict:
        """Helper method to return empty metrics structure"""
        return {
            'current_volume': 0,
            'prev_volume': 0,
            'sma': None,
            'recent_volumes': [],
            'volume_dates': [],
            'daily_changes': [],
            'current_ratio': None,
            'volume_24h_change': 0,
            'volume_vs_avg': 0,
            'days_included': 0
        }

    def get_news(self, symbol: str, limit: int = 5) -> Dict:
        """
        Get news from Polygon.io
        """
        endpoint = f"/v2/reference/news"
        params = {
            'ticker': symbol,
            'limit': limit,
            'sort': 'published_utc',
            'order': 'desc'
        }
        return self._make_request(endpoint, params)

    def fetch_market_data(self, symbols: List[str]) -> List[Dict]:
        """
        Fetch comprehensive market data using both Polygon.io and yfinance
        """
        market_data = []
        
        for symbol in symbols:
            try:
                logger.info(f"Fetching data for {symbol}")
                
                # Get enhanced company details
                company_details = self.get_company_details(symbol)
                if not company_details:
                    logger.error(f"Failed to get company details for {symbol}")
                    continue
                
                # Get ticker details from Polygon.io
                details = self.get_ticker_details(symbol)
                if not details or 'results' not in details:
                    logger.error(f"Failed to get details for {symbol}")
                    continue
                
                ticker_info = details['results']
                
                # Get aggregates with explicit error handling
                aggs = self.get_aggregates(symbol, days=5)
                if not aggs or 'results' not in aggs or not aggs['results']:
                    logger.error(f"Failed to get aggregates for {symbol}")
                    continue
                
                results = aggs['results']
                latest = results[-1]
                prev_day = results[-2] if len(results) > 1 else latest
                
                # Calculate basic metrics
                price_change = latest['c'] - prev_day['c']
                price_change_pct = (price_change / prev_day['c']) * 100
                
                # Get volume metrics using Polygon's native SMA
                logger.info(f"Getting volume metrics for {symbol}")

                
                            # Get comprehensive volume metrics
                logger.info(f"Getting volume metrics for {symbol}")
                volume_metrics = self.get_volume_metrics(symbol)
                logger.info(f"Volume metrics for {symbol}: {volume_metrics}")
                
                # Get 52-week high/low from yfinance
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    fifty_two_week_high = info.get('fiftyTwoWeekHigh')
                    fifty_two_week_low = info.get('fiftyTwoWeekLow')
                    
                    # Calculate proximity to 52-week high
                    current_price = latest['c']
                    high_proximity_pct = None
                    if fifty_two_week_high and current_price:
                        high_proximity_pct = ((fifty_two_week_high - current_price) / fifty_two_week_high) * 100
                    
                    near_high_alert = high_proximity_pct is not None and high_proximity_pct <= 5
                    
                except Exception as e:
                    logger.error(f"Error fetching yfinance data for {symbol}: {str(e)}")
                    fifty_two_week_high = None
                    fifty_two_week_low = None
                    high_proximity_pct = None
                    near_high_alert = False
                
                # Update volume alerts based on comprehensive metrics
                volume_alerts = {
                    "volumeSpike10": volume_metrics['volume_24h_change'] >= 10,
                    "volumeSpike20": volume_metrics['volume_24h_change'] >= 20,
                    "highVolume": volume_metrics['volume_vs_avg'] > 50
                }
                
                # Get 52-week high/low and additional data from yfinance
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    
                    # Get 52-week high/low data
                    fifty_two_week_high = info.get('fiftyTwoWeekHigh')
                    fifty_two_week_low = info.get('fiftyTwoWeekLow')
                    
                    # Calculate proximity to 52-week high
                    current_price = latest['c']
                    high_proximity_pct = None
                    if fifty_two_week_high and current_price:
                        high_proximity_pct = ((fifty_two_week_high - current_price) / fifty_two_week_high) * 100
                    
                    # Add new alert for proximity to 52-week high
                    near_high_alert = high_proximity_pct is not None and high_proximity_pct <= 5  # Within 5% of 52-week high
                    
                except Exception as e:
                    logger.error(f"Error fetching yfinance data for {symbol}: {str(e)}")
                    fifty_two_week_high = None
                    fifty_two_week_low = None
                    high_proximity_pct = None
                    near_high_alert = False
                
                # Get insider trades from yfinance
                insider_summary = self.get_insider_data(symbol)
                
                # Get news from Polygon.io
                news_data = self.get_news(symbol)
                
                try:
                    financial_metrics = self.get_financials_metrics(symbol)
                    detailed_financials = self.get_detailed_financials(symbol)
                except Exception as e:
                    logger.error(f"Error getting financial metrics for {symbol}: {str(e)}")
                    financial_metrics = {}
                    detailed_financials = {}

                
                # Process news data
                news_summary = [
                    {
                        "title": article.get('title'),
                        "publisher": article.get('publisher', {}).get('name'),
                        "timestamp": article.get('published_utc'),
                        "url": article.get('article_url'),
                        "type": article.get('tickers', [None])[0]
                    }
                    for article in (news_data.get('results', []) if news_data else [])
                ]
                
                # Get RSI data
                rsi_data = self.get_rsi(symbol)

                # Calculate alerts
                alerts = sum([
                    abs(price_change_pct) > 5,  # 5% price movement
                    volume_metrics['volume_24h_change'] >= 10,  # 10% volume spike
                    volume_metrics['volume_24h_change'] >= 20,  # 20% volume spike
                    bool(insider_summary['notable_trades']),  # Insider activity
                    bool(news_summary),  # Recent news
                    near_high_alert  # Near 52-week high
                ])
                
                market_data.append({
                    "symbol": symbol,
                    "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    # Enhanced company info
                    "sector": company_details['sector'],
                    "industry": company_details['industry'],
                    "description": company_details['description'],
                    "branding": {
                        "icon_url": company_details['icon_url'],
                        "logo_url": company_details['logo_url']
                    },
                    # Price data
                    "price": latest['c'],
                    "priceChange": price_change_pct,
                    "openPrice": latest['o'],
                    "prevClose": prev_day['c'],
                    "dayHigh": latest['h'],
                    "dayLow": latest['l'],
                    # Volume data
                    "volume": volume_metrics['current_volume'],
                    "prevVolume": volume_metrics['prev_volume'],
                    "volumeMetrics": {
                        "recentVolumes": volume_metrics['recent_volumes'],
                        "volumeDates": volume_metrics['volume_dates'],
                        "dailyChanges": volume_metrics['daily_changes'],
                        "averageVolume": volume_metrics['sma'],
                        "volumeChange": volume_metrics['volume_24h_change'],
                        "volumeVsAvg": volume_metrics['volume_vs_avg']
                    },
                    # Technical indicators
                    "technicals": {
                        "rsi": rsi_data['value'] if rsi_data else None,
                        "volumeSMA": volume_metrics['sma']
                    },
                    # 52-week data
                    "fiftyTwoWeekHigh": fifty_two_week_high,
                    "fiftyTwoWeekLow": fifty_two_week_low,
                    "highProximityPct": high_proximity_pct,
                    # Market metrics
                    "marketCap": ticker_info.get('market_cap', 0),
                    # Additional data
                    "insiderActivity": insider_summary,
                    "recentNews": news_summary,
                    # Add new metrics section

                    # In the market_data.append, update the fundamentals section:
                    "fundamentals": {
                        "peRatios": financial_metrics.get("peRatios", {"trailingPE": None, "forwardPE": None}),
                        "estimates": financial_metrics.get("estimates", {
                            "revenue": {"nextQuarter": None, "currentYear": None, "nextYear": None},
                            "earnings": {"nextQuarter": None, "currentYear": None, "nextYear": None}
                        }),
                        "quarterlyMetrics": financial_metrics.get("quarterly", {
                            "revenue": [],
                            "revenueChange": [],
                            "periods": []
                        }),
                        "detailedQuarterly": {
                            "revenue": detailed_financials.get("revenue", []),
                            "revenueChanges": detailed_financials.get("revenueChanges", []),
                            "netIncome": detailed_financials.get("netIncome", []),
                            "netIncomeChanges": detailed_financials.get("netIncomeChanges", []),
                            "operatingMargin": detailed_financials.get("operatingMargin", []),
                            "operatingMarginChanges": detailed_financials.get("operatingMarginChanges", []),
                            "freeCashFlow": detailed_financials.get("freeCashFlow", []),
                            "freeCashFlowChanges": detailed_financials.get("freeCashFlowChanges", []),
                            "dates": detailed_financials.get("dates", [])
                        }
                    },
                    # Alert info
                    "alerts": alerts,
                    "alertDetails": {
                        "priceAlert": abs(price_change_pct) > 5,
                        "volumeSpike10": volume_metrics['volume_24h_change'] >= 10,
                        "volumeSpike20": volume_metrics['volume_24h_change'] >= 20,
                        "highVolume": volume_metrics['volume_vs_avg'] > 50,
                        "insiderAlert": bool(insider_summary['notable_trades']),
                        "newsAlert": bool(news_summary),
                        "technicalAlert": (rsi_data and (rsi_data['value'] > 70 or rsi_data['value'] < 30)),
                        "nearHighAlert": near_high_alert
                    }
                })
                
                logger.info(f"Successfully processed {symbol}")
                
            except Exception as e:
                logger.error(f"Unexpected error processing {symbol}: {str(e)}")
                continue
        
        # Save to JSON file
        try:
            with open('market_data.json', 'w') as f:
                json.dump(market_data, f, indent=2)
            logger.info(f"Saved data for {len(market_data)} stocks")
        except Exception as e:
            logger.error(f"Error saving to JSON: {str(e)}")
        
        return market_data

def main():
    POLYGON_KEY = "WL7rkKMQGlEpcRy_GogYH6hyclU8sOnq"
    
    # Test with a smaller list first
    # tickers = ["VERA", "ANIX", "INMB", "GUTS"]  # Add your full list here    
    
    tickers = ["VERA"]  # Add your full list here    
    # tickers = ["VERA", "ANIX", "INMB", "GUTS", "MIRA", "EYPT", "GNPX", "GLUE", "TGTX", "AUPH", 
    #            "ITCI", "CLOV", "CERE", "IPHA", "MDWD", "ZYME", "ASMB", "JAZZ", "PRTA", "TMDX",
    #            "GILD", "NTNX", "INAB", "MNPR", "APVO", "HRMY", "BHC", "BCRX", "GRTX", "AXSM",
    #            "SMMT", "SAGE", "MYNZ", "GMAB", "LUMO", "NEO", "ARCT", "TEVA", "VMD", "VERU",
    #            "VRCA", "SIGA", "INMD", "EXEL", "CPRX", "HALO", "NVOS", "ATAI", "BNGO", "ENOV",
    #            "BIIB", "MIST", "ARDX", "CVM", "ACLS", "IDYA", "RYTM", "TWST", "STEM", "GERN",
    #            "VIR", "ALKS", "AMPH", "SVRA", "EVLO", "GH", "NTLA", "MRTX", "SRPT", "RARE",
    #            "TRVI", "PGEN", "EVH", "ARQT", "QNRX", "SYRS", "GTHX", "MNKD", "XERS", "SNDX",
    #            "PRTK", "PLRX", "MREO", "MDGL", "KZR", "GALT", "ETNB", "EPZM", "CMRX", "CDTX",
    #            "GYRE", "CBAY", "AGEN", "ABUS", "ABCL", "LOGC", "BLCM", "ADVM", "SNY", "MRSN",
    #            "TCRT", "ASRT", "ABBV", "ADMA", "RKLB"]  # Add your full list here
    
    fetcher = HybridDataFetcher(POLYGON_KEY)
    market_data = fetcher.fetch_market_data(tickers)
    
    print("\nStocks with alerts:")
    alerts = [stock for stock in market_data if stock['alerts'] > 0]
    alerts.sort(key=lambda x: x['alerts'], reverse=True)
    
    for stock in alerts:
        print(f"\n{stock['symbol']} ({stock['sector']}): {stock['alerts']} alerts")
        print(f"  Price: ${stock['price']:.2f} ({stock['priceChange']:.2f}%)")
        if stock['technicals']['rsi']:
            print(f"  RSI: {stock['technicals']['rsi']:.2f}")
        
        if stock['alertDetails']['priceAlert']:
            print("  - Major price movement")
        if stock['alertDetails']['volumeSpike20']:
            print("  - Major volume spike")
        if stock['alertDetails']['insiderAlert']:
            print(f"  - Recent insider activity: {len(stock['insiderActivity']['notable_trades'])} notable trades")
        if stock['alertDetails']['newsAlert']:
            print("  - Recent news:")
            for article in stock['recentNews'][:2]:
                print(f"    * {article['title']}")
        if stock['alertDetails']['technicalAlert']:
            print(f"  - Technical alert (RSI: {stock['technicals']['rsi']:.2f})")

if __name__ == "__main__":
    main()