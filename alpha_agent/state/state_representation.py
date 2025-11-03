"""
State Representation: Combines price history (GAF), fundamentals, and sentiment
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from pyts.image import GramianAngularField
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GramianAngularFieldTransformer:
    """
    Transform time series price data into Gramian Angular Field images
    """
    
    def __init__(self, image_size: int = 30, method: str = 'summation'):
        """
        Args:
            image_size: Size of the resulting GAF image (image_size x image_size)
            method: 'summation' (GASF) or 'difference' (GADF)
        """
        self.image_size = image_size
        self.method = method
        self.gaf = GramianAngularField(image_size=image_size, method=method)
        logger.info(f"Initialized GAF transformer: size={image_size}, method={method}")
    
    def transform_series(self, series: np.ndarray) -> np.ndarray:
        """
        Transform a single time series into GAF image
        
        Args:
            series: 1D array of time series data
            
        Returns:
            2D GAF image (image_size x image_size)
        """
        try:
            if len(series) < self.image_size:
                # Pad if too short
                series = np.pad(series, (0, self.image_size - len(series)), mode='edge')
            elif len(series) > self.image_size:
                # Sample if too long
                indices = np.linspace(0, len(series) - 1, self.image_size, dtype=int)
                series = series[indices]
            
            # Normalize to [-1, 1] for GAF
            series_normalized = 2 * (series - series.min()) / (series.max() - series.min() + 1e-8) - 1
            
            # Transform
            gaf_image = self.gaf.fit_transform(series_normalized.reshape(1, -1))[0]
            
            return gaf_image
            
        except Exception as e:
            logger.error(f"Error transforming series to GAF: {e}")
            return np.zeros((self.image_size, self.image_size))
    
    def transform_ohlcv(self, ohlcv_df: pd.DataFrame, lookback: int = 30) -> Dict[str, np.ndarray]:
        """
        Transform OHLCV data into multiple GAF images
        
        Args:
            ohlcv_df: DataFrame with OHLCV columns
            lookback: Number of days to look back
            
        Returns:
            Dictionary with GAF images for each price type
        """
        if len(ohlcv_df) < lookback:
            logger.warning(f"Insufficient data: {len(ohlcv_df)} < {lookback}")
            lookback = len(ohlcv_df)
        
        # Get recent data
        recent_data = ohlcv_df.tail(lookback)
        
        gaf_images = {}
        for column in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if column in recent_data.columns:
                series = recent_data[column].values
                gaf_images[column.lower()] = self.transform_series(series)
        
        return gaf_images


class FundamentalEncoder:
    """
    Encode and normalize fundamental data
    """
    
    def __init__(self):
        """Initialize scalers"""
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # Define expected fundamental features
        self.features = [
            'pe_ratio', 'forward_pe', 'peg_ratio', 'price_to_book',
            'debt_to_equity', 'roe', 'roa', 'profit_margin',
            'operating_margin', 'revenue_growth', 'earnings_growth',
            'current_ratio', 'quick_ratio', 'beta'
        ]
    
    def encode(self, fundamentals: Dict[str, float]) -> np.ndarray:
        """
        Encode fundamental data into a normalized vector
        
        Args:
            fundamentals: Dictionary of fundamental metrics
            
        Returns:
            Normalized vector of fundamentals
        """
        # Extract features in order
        feature_vector = []
        for feature in self.features:
            value = fundamentals.get(feature, 0.0)
            
            # Handle missing or invalid values
            if pd.isna(value) or value is None or np.isinf(value):
                value = 0.0
            
            feature_vector.append(value)
        
        feature_vector = np.array(feature_vector).reshape(1, -1)
        
        # Normalize (use simple clipping if not fitted)
        if not self.is_fitted:
            # Clip extreme values for robustness
            feature_vector = np.clip(feature_vector, -100, 100)
            # Simple normalization
            feature_vector = feature_vector / (np.abs(feature_vector).max() + 1e-8)
        else:
            feature_vector = self.scaler.transform(feature_vector)
        
        return feature_vector.flatten()
    
    def fit(self, fundamentals_list: list):
        """
        Fit the scaler on historical fundamental data
        
        Args:
            fundamentals_list: List of fundamental dictionaries
        """
        feature_matrix = []
        for fundamentals in fundamentals_list:
            vector = self.encode(fundamentals)
            feature_matrix.append(vector)
        
        if feature_matrix:
            self.scaler.fit(np.array(feature_matrix))
            self.is_fitted = True
            logger.info("Fundamental encoder fitted")


class SentimentEncoder:
    """
    Encode sentiment data (FinBERT embeddings + GPT scores)
    """
    
    def __init__(self, embedding_dim: int = 768):
        """
        Args:
            embedding_dim: Dimension of FinBERT embeddings
        """
        self.embedding_dim = embedding_dim
    
    def encode(self, sentiment_data: Dict) -> np.ndarray:
        """
        Encode sentiment data into a fixed-size vector
        
        Args:
            sentiment_data: Output from MultiModalSentimentAnalyzer
            
        Returns:
            Encoded sentiment vector
        """
        # Extract components
        finbert_sentiment = sentiment_data.get('finbert', {})
        gpt_sentiment = sentiment_data.get('gpt', {})
        embedding = sentiment_data.get('embedding', np.zeros(self.embedding_dim))
        
        # Sentiment scores (4 from FinBERT + 2 from GPT)
        sentiment_scores = np.array([
            finbert_sentiment.get('positive', 0.0),
            finbert_sentiment.get('negative', 0.0),
            finbert_sentiment.get('neutral', 0.0),
            finbert_sentiment.get('compound', 0.0),
            gpt_sentiment.get('sentiment_score', 0.0),
            gpt_sentiment.get('confidence', 0.0),
        ])
        
        # For the full state, we might want to reduce embedding dimensionality
        # Use PCA or take a subset (for simplicity, take first 32 dims)
        embedding_reduced = embedding[:32] if len(embedding) >= 32 else np.pad(embedding, (0, 32 - len(embedding)))
        
        # Combine
        encoded = np.concatenate([sentiment_scores, embedding_reduced])
        
        return encoded


class CompleteStateRepresentation:
    """
    Combines all state components: GAF images, fundamentals, sentiment
    """
    
    def __init__(self, 
                 gaf_size: int = 30,
                 lookback_days: int = 30,
                 use_gaf: bool = True,
                 flatten_output: bool = True):
        """
        Args:
            gaf_size: Size of GAF images
            lookback_days: Number of days for price history
            use_gaf: Whether to use GAF transformation (if False, use raw prices)
            flatten_output: Whether to flatten the state to 1D vector
        """
        self.gaf_size = gaf_size
        self.lookback_days = lookback_days
        self.use_gaf = use_gaf
        self.flatten_output = flatten_output
        
        if use_gaf:
            self.gaf_transformer = GramianAngularFieldTransformer(image_size=gaf_size)
        
        self.fundamental_encoder = FundamentalEncoder()
        self.sentiment_encoder = SentimentEncoder()
        
        logger.info(f"Initialized CompleteStateRepresentation: GAF={use_gaf}, lookback={lookback_days}")
    
    def create_state(self,
                     ohlcv_df: pd.DataFrame,
                     fundamentals: Dict[str, float],
                     sentiment_data: Dict,
                     technical_indicators: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Create complete state representation
        
        Args:
            ohlcv_df: Historical OHLCV data
            fundamentals: Fundamental metrics
            sentiment_data: Sentiment analysis output
            technical_indicators: Optional technical indicator data
            
        Returns:
            Complete state vector/tensor
        """
        state_components = []
        
        # 1. Price representation
        if self.use_gaf:
            # GAF images for OHLCV
            gaf_images = self.gaf_transformer.transform_ohlcv(ohlcv_df, self.lookback_days)
            
            # Stack GAF images (5 channels: OHLCV)
            gaf_stack = np.stack([
                gaf_images.get('close', np.zeros((self.gaf_size, self.gaf_size))),
                gaf_images.get('open', np.zeros((self.gaf_size, self.gaf_size))),
                gaf_images.get('high', np.zeros((self.gaf_size, self.gaf_size))),
                gaf_images.get('low', np.zeros((self.gaf_size, self.gaf_size))),
                gaf_images.get('volume', np.zeros((self.gaf_size, self.gaf_size))),
            ], axis=0)  # Shape: (5, gaf_size, gaf_size)
            
            if self.flatten_output:
                state_components.append(gaf_stack.flatten())
            else:
                state_components.append(gaf_stack)
        else:
            # Use raw normalized prices
            recent_data = ohlcv_df.tail(self.lookback_days)
            price_features = []
            
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col in recent_data.columns:
                    values = recent_data[col].values
                    # Normalize
                    values_norm = (values - values.mean()) / (values.std() + 1e-8)
                    price_features.append(values_norm)
            
            price_array = np.concatenate(price_features)
            state_components.append(price_array)
        
        # 2. Fundamental data
        fundamental_vector = self.fundamental_encoder.encode(fundamentals)
        state_components.append(fundamental_vector)
        
        # 3. Sentiment data
        sentiment_vector = self.sentiment_encoder.encode(sentiment_data)
        state_components.append(sentiment_vector)
        
        # 4. Technical indicators (optional)
        if technical_indicators is not None and len(technical_indicators) > 0:
            recent_ti = technical_indicators.tail(1)
            ti_features = []
            
            for col in ['rsi', 'macd', 'bb_width', 'atr', 'volume_ratio']:
                if col in recent_ti.columns:
                    value = recent_ti[col].values[0]
                    if not pd.isna(value):
                        ti_features.append(value)
            
            if ti_features:
                ti_array = np.array(ti_features)
                # Normalize
                ti_array = np.clip(ti_array, -10, 10) / 10.0
                state_components.append(ti_array)
        
        # Combine all components
        if self.flatten_output:
            state = np.concatenate([comp.flatten() for comp in state_components])
        else:
            # Return as dictionary for complex networks
            state = {
                'price': state_components[0],
                'fundamentals': fundamental_vector,
                'sentiment': sentiment_vector,
            }
        
        return state
    
    def get_state_shape(self) -> Tuple:
        """
        Get the shape of the state representation
        
        Returns:
            Tuple representing state shape
        """
        if self.use_gaf:
            if self.flatten_output:
                gaf_size = 5 * self.gaf_size * self.gaf_size  # 5 channels
                fundamental_size = len(self.fundamental_encoder.features)
                sentiment_size = 38  # 6 scores + 32 embedding dims
                ti_size = 5  # approximate
                total_size = gaf_size + fundamental_size + sentiment_size + ti_size
                return (total_size,)
            else:
                return (5, self.gaf_size, self.gaf_size)
        else:
            price_size = 5 * self.lookback_days
            fundamental_size = len(self.fundamental_encoder.features)
            sentiment_size = 38
            ti_size = 5
            total_size = price_size + fundamental_size + sentiment_size + ti_size
            return (total_size,)


if __name__ == "__main__":
    # Test state representation
    from alpha_agent.data import MarketDataFetcher, TechnicalIndicators
    from alpha_agent.sentiment import MultiModalSentimentAnalyzer
    
    # Fetch data
    fetcher = MarketDataFetcher("AAPL", lookback_days=60)
    ohlcv, fundamentals, news = fetcher.fetch_all_data()
    
    # Add technical indicators
    ohlcv_ti = TechnicalIndicators.add_technical_indicators(ohlcv)
    
    # Analyze sentiment
    sentiment_analyzer = MultiModalSentimentAnalyzer()
    sentiment_data = sentiment_analyzer.analyze_comprehensive(news, "AAPL")
    
    # Create state representation
    state_repr = CompleteStateRepresentation(
        gaf_size=30,
        lookback_days=30,
        use_gaf=True,
        flatten_output=True
    )
    
    state = state_repr.create_state(ohlcv, fundamentals, sentiment_data, ohlcv_ti)
    
    print(f"State shape: {state.shape}")
    print(f"State range: [{state.min():.3f}, {state.max():.3f}]")
    print(f"Expected shape: {state_repr.get_state_shape()}")

