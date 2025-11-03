"""
Data fetcher for Yahoo Finance: OHLCV, Fundamentals, and News
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarketDataFetcher:
    """
    Fetches and preprocesses market data from Yahoo Finance
    """
    
    def __init__(self, ticker: str, lookback_days: int = 365):
        """
        Args:
            ticker: Stock ticker symbol
            lookback_days: Number of days of historical data to fetch
        """
        self.ticker = ticker
        self.lookback_days = lookback_days
        self.yf_ticker = yf.Ticker(ticker)
        
    def fetch_ohlcv(self, start_date: Optional[str] = None, 
                     end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch OHLCV data
        
        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume
        """
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=self.lookback_days)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        try:
            df = self.yf_ticker.history(start=start_date, end=end_date)
            
            if df.empty:
                logger.error(f"No data fetched for {self.ticker}")
                return pd.DataFrame()
            
            # Keep only essential columns
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            
            # Handle missing values
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            logger.info(f"Fetched {len(df)} days of OHLCV data for {self.ticker}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching OHLCV data: {e}")
            return pd.DataFrame()
    
    def fetch_fundamentals(self) -> Dict[str, float]:
        """
        Fetch fundamental data
        
        Returns:
            Dictionary with fundamental metrics
        """
        try:
            info = self.yf_ticker.info
            
            fundamentals = {
                'pe_ratio': info.get('trailingPE', np.nan),
                'forward_pe': info.get('forwardPE', np.nan),
                'peg_ratio': info.get('pegRatio', np.nan),
                'price_to_book': info.get('priceToBook', np.nan),
                'debt_to_equity': info.get('debtToEquity', np.nan),
                'roe': info.get('returnOnEquity', np.nan),
                'roa': info.get('returnOnAssets', np.nan),
                'profit_margin': info.get('profitMargins', np.nan),
                'operating_margin': info.get('operatingMargins', np.nan),
                'revenue_growth': info.get('revenueGrowth', np.nan),
                'earnings_growth': info.get('earningsGrowth', np.nan),
                'current_ratio': info.get('currentRatio', np.nan),
                'quick_ratio': info.get('quickRatio', np.nan),
                'market_cap': info.get('marketCap', np.nan),
                'beta': info.get('beta', np.nan),
            }
            
            # Handle missing values - replace with 0 or median
            for key, value in fundamentals.items():
                if pd.isna(value) or value is None:
                    fundamentals[key] = 0.0
                    
            logger.info(f"Fetched {len(fundamentals)} fundamental metrics for {self.ticker}")
            return fundamentals
            
        except Exception as e:
            logger.error(f"Error fetching fundamentals: {e}")
            return {}
    
    def fetch_news(self, max_news: int = 20) -> List[Dict[str, str]]:
        """
        Fetch recent news headlines
        
        Args:
            max_news: Maximum number of news items to fetch
            
        Returns:
            List of news items with title and summary
        """
        try:
            news = self.yf_ticker.news[:max_news]
            
            news_list = []
            for item in news:
                news_item = {
                    'title': item.get('title', ''),
                    'publisher': item.get('publisher', ''),
                    'link': item.get('link', ''),
                    'timestamp': item.get('providerPublishTime', 0),
                }
                news_list.append(news_item)
            
            logger.info(f"Fetched {len(news_list)} news items for {self.ticker}")
            return news_list
            
        except Exception as e:
            logger.error(f"Error fetching news: {e}")
            return []
    
    def fetch_all_data(self) -> Tuple[pd.DataFrame, Dict[str, float], List[Dict]]:
        """
        Fetch all data: OHLCV, fundamentals, and news
        
        Returns:
            Tuple of (ohlcv_df, fundamentals_dict, news_list)
        """
        ohlcv = self.fetch_ohlcv()
        fundamentals = self.fetch_fundamentals()
        news = self.fetch_news()
        
        return ohlcv, fundamentals, news


class TechnicalIndicators:
    """
    Calculate technical indicators for the OHLCV data
    """
    
    @staticmethod
    def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add common technical indicators to OHLCV dataframe
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with additional technical indicator columns
        """
        df = df.copy()
        
        # Returns
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Moving Averages
        df['sma_5'] = df['Close'].rolling(window=5).mean()
        df['sma_20'] = df['Close'].rolling(window=20).mean()
        df['sma_50'] = df['Close'].rolling(window=50).mean()
        
        # Exponential Moving Averages
        df['ema_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_diff'] = df['macd'] - df['macd_signal']
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # ATR (Average True Range)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['atr'] = true_range.rolling(14).mean()
        
        # Volume indicators
        df['volume_sma'] = df['Volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma']
        
        # Fill NaN values
        df = df.fillna(method='bfill').fillna(0)
        
        return df


if __name__ == "__main__":
    # Test the data fetcher
    fetcher = MarketDataFetcher("AAPL")
    ohlcv, fundamentals, news = fetcher.fetch_all_data()
    
    print(f"\nOHLCV Data Shape: {ohlcv.shape}")
    print(ohlcv.head())
    
    print(f"\nFundamentals: {fundamentals}")
    
    print(f"\nNews Items: {len(news)}")
    if news:
        print(f"Latest: {news[0]['title']}")
    
    # Add technical indicators
    ohlcv_with_indicators = TechnicalIndicators.add_technical_indicators(ohlcv)
    print(f"\nOHLCV with Indicators Shape: {ohlcv_with_indicators.shape}")
    print(ohlcv_with_indicators.tail())

