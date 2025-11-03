"""
Market Regime Detection System
Identifies bull, bear, sideways, and high volatility market conditions
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from enum import Enum
import logging
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime types"""
    BULL = "bull"  # Strong uptrend
    BEAR = "bear"  # Strong downtrend
    SIDEWAYS = "sideways"  # Range-bound
    HIGH_VOLATILITY = "high_volatility"  # Crisis mode
    UNCERTAIN = "uncertain"  # Unclear


class RegimeDetector:
    """
    Detects current market regime using multiple indicators
    """
    
    def __init__(self, lookback: int = 50):
        """
        Args:
            lookback: Number of periods for regime analysis
        """
        self.lookback = lookback
        self.scaler = StandardScaler()
        self.gmm = GaussianMixture(n_components=4, random_state=42)
        self.is_fitted = False
        
        logger.info(f"RegimeDetector initialized with lookback={lookback}")
    
    def detect_regime(self, ohlcv: pd.DataFrame) -> Tuple[MarketRegime, float]:
        """
        Detect current market regime
        
        Args:
            ohlcv: DataFrame with OHLCV data
            
        Returns:
            (regime, confidence)
        """
        if len(ohlcv) < self.lookback:
            return MarketRegime.UNCERTAIN, 0.0
        
        # Extract features
        features = self._extract_regime_features(ohlcv)
        
        # Rule-based detection (fast and interpretable)
        regime_rule, confidence_rule = self._rule_based_detection(features)
        
        # ML-based detection (if fitted)
        if self.is_fitted:
            regime_ml, confidence_ml = self._ml_based_detection(features)
            # Combine both approaches
            if confidence_ml > 0.7:
                return regime_ml, confidence_ml
        
        return regime_rule, confidence_rule
    
    def _extract_regime_features(self, ohlcv: pd.DataFrame) -> Dict[str, float]:
        """
        Extract features for regime detection
        
        Returns:
            Dictionary of features
        """
        recent = ohlcv.tail(self.lookback)
        close = recent['Close'].values
        
        # Trend features
        returns = np.diff(close) / close[:-1]
        cumulative_return = (close[-1] - close[0]) / close[0]
        
        # Moving averages
        sma_20 = close[-20:].mean() if len(close) >= 20 else close.mean()
        sma_50 = close[-50:].mean() if len(close) >= 50 else close.mean()
        price_vs_sma20 = (close[-1] - sma_20) / sma_20
        price_vs_sma50 = (close[-1] - sma_50) / sma_50
        ma_trend = (sma_20 - sma_50) / sma_50 if sma_50 > 0 else 0
        
        # Volatility features
        volatility = np.std(returns)
        volatility_percentile = self._calculate_percentile(volatility, returns)
        
        # Trend strength (ADX-like)
        positive_moves = np.sum(returns > 0) / len(returns)
        trend_consistency = abs(2 * positive_moves - 1)  # 0 = random, 1 = strong trend
        
        # Drawdown
        peak = np.maximum.accumulate(close)
        drawdown = (peak - close) / peak
        max_drawdown = np.max(drawdown)
        
        # Volume (if available)
        if 'Volume' in recent.columns:
            volume_trend = recent['Volume'].tail(10).mean() / recent['Volume'].mean()
        else:
            volume_trend = 1.0
        
        return {
            'cumulative_return': cumulative_return,
            'volatility': volatility,
            'volatility_percentile': volatility_percentile,
            'price_vs_sma20': price_vs_sma20,
            'price_vs_sma50': price_vs_sma50,
            'ma_trend': ma_trend,
            'trend_consistency': trend_consistency,
            'max_drawdown': max_drawdown,
            'volume_trend': volume_trend,
            'positive_rate': positive_moves
        }
    
    def _rule_based_detection(self, features: Dict) -> Tuple[MarketRegime, float]:
        """
        Rule-based regime detection
        
        Returns:
            (regime, confidence)
        """
        # Check for high volatility first (crisis mode)
        if features['volatility'] > 0.04 or features['volatility_percentile'] > 0.9:
            confidence = min(1.0, features['volatility'] / 0.04)
            return MarketRegime.HIGH_VOLATILITY, confidence
        
        # Check for bull market
        bull_score = 0.0
        if features['cumulative_return'] > 0.05:  # 5%+ gain
            bull_score += 0.3
        if features['price_vs_sma20'] > 0.02:  # Above 20-day MA
            bull_score += 0.2
        if features['price_vs_sma50'] > 0.03:  # Above 50-day MA
            bull_score += 0.2
        if features['ma_trend'] > 0.01:  # MAs trending up
            bull_score += 0.15
        if features['trend_consistency'] > 0.6:  # Consistent trend
            bull_score += 0.15
        
        # Check for bear market
        bear_score = 0.0
        if features['cumulative_return'] < -0.05:  # 5%+ loss
            bear_score += 0.3
        if features['price_vs_sma20'] < -0.02:  # Below 20-day MA
            bear_score += 0.2
        if features['price_vs_sma50'] < -0.03:  # Below 50-day MA
            bear_score += 0.2
        if features['ma_trend'] < -0.01:  # MAs trending down
            bear_score += 0.15
        if features['max_drawdown'] > 0.10:  # Large drawdown
            bear_score += 0.15
        
        # Determine regime
        if bull_score > 0.6:
            return MarketRegime.BULL, bull_score
        elif bear_score > 0.6:
            return MarketRegime.BEAR, bear_score
        elif features['volatility'] < 0.015 and abs(features['cumulative_return']) < 0.03:
            # Low vol + low returns = sideways
            confidence = 1.0 - features['volatility'] / 0.015
            return MarketRegime.SIDEWAYS, confidence
        else:
            return MarketRegime.UNCERTAIN, 0.5
    
    def _ml_based_detection(self, features: Dict) -> Tuple[MarketRegime, float]:
        """
        ML-based regime detection using Gaussian Mixture Model
        
        Returns:
            (regime, confidence)
        """
        # Convert features to array
        feature_vector = np.array([
            features['cumulative_return'],
            features['volatility'],
            features['ma_trend'],
            features['trend_consistency']
        ]).reshape(1, -1)
        
        # Normalize
        feature_vector = self.scaler.transform(feature_vector)
        
        # Predict cluster
        cluster = self.gmm.predict(feature_vector)[0]
        probabilities = self.gmm.predict_proba(feature_vector)[0]
        confidence = probabilities[cluster]
        
        # Map cluster to regime (learned from data)
        regime_map = {
            0: MarketRegime.BULL,
            1: MarketRegime.BEAR,
            2: MarketRegime.SIDEWAYS,
            3: MarketRegime.HIGH_VOLATILITY
        }
        
        regime = regime_map.get(cluster, MarketRegime.UNCERTAIN)
        return regime, float(confidence)
    
    def fit(self, historical_data: List[pd.DataFrame]):
        """
        Fit ML model on historical data
        
        Args:
            historical_data: List of OHLCV dataframes
        """
        all_features = []
        
        for ohlcv in historical_data:
            if len(ohlcv) >= self.lookback:
                features = self._extract_regime_features(ohlcv)
                feature_vector = [
                    features['cumulative_return'],
                    features['volatility'],
                    features['ma_trend'],
                    features['trend_consistency']
                ]
                all_features.append(feature_vector)
        
        if len(all_features) > 10:
            X = np.array(all_features)
            X = self.scaler.fit_transform(X)
            self.gmm.fit(X)
            self.is_fitted = True
            logger.info(f"RegimeDetector fitted on {len(all_features)} samples")
    
    def _calculate_percentile(self, value: float, data: np.ndarray) -> float:
        """Calculate percentile of value in data"""
        return np.sum(data < value) / len(data)


class RegimeAdaptiveStrategy:
    """
    Adapts trading strategy based on market regime
    """
    
    def __init__(self):
        self.detector = RegimeDetector()
        
        # Regime-specific parameters
        self.regime_params = {
            MarketRegime.BULL: {
                'position_bias': 0.6,  # Bias towards long positions
                'risk_tolerance': 1.2,  # Higher risk ok
                'holding_period': 'long'
            },
            MarketRegime.BEAR: {
                'position_bias': -0.4,  # Bias towards short/cash
                'risk_tolerance': 0.6,  # Lower risk
                'holding_period': 'short'
            },
            MarketRegime.SIDEWAYS: {
                'position_bias': 0.0,  # No bias
                'risk_tolerance': 1.0,  # Normal risk
                'holding_period': 'short'
            },
            MarketRegime.HIGH_VOLATILITY: {
                'position_bias': -0.8,  # Strong defensive
                'risk_tolerance': 0.3,  # Very low risk
                'holding_period': 'very_short'
            },
            MarketRegime.UNCERTAIN: {
                'position_bias': 0.0,
                'risk_tolerance': 0.8,
                'holding_period': 'medium'
            }
        }
    
    def adapt_action(self, 
                     raw_action: np.ndarray, 
                     ohlcv: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
        """
        Adapt action based on current market regime
        
        Args:
            raw_action: Action from base agent
            ohlcv: Market data
            
        Returns:
            (adapted_action, regime_info)
        """
        # Detect regime
        regime, confidence = self.detector.detect_regime(ohlcv)
        
        # Get regime parameters
        params = self.regime_params[regime]
        
        # Adapt action
        adapted_action = raw_action.copy()
        
        # Apply position bias
        adapted_action[0] = (adapted_action[0] + params['position_bias']) / 2
        
        # Apply risk adjustment
        adapted_action[0] *= params['risk_tolerance']
        
        # Clip to valid range
        adapted_action[0] = np.clip(adapted_action[0], -1.0, 1.0)
        
        regime_info = {
            'regime': regime.value,
            'confidence': confidence,
            'position_bias': params['position_bias'],
            'risk_tolerance': params['risk_tolerance'],
            'holding_period': params['holding_period'],
            'adaptation_applied': True
        }
        
        return adapted_action, regime_info


if __name__ == "__main__":
    # Test regime detector
    from alpha_agent.data import MarketDataFetcher
    
    # Fetch real data
    fetcher = MarketDataFetcher("AAPL", lookback_days=100)
    ohlcv, _, _ = fetcher.fetch_all_data()
    
    # Detect regime
    detector = RegimeDetector(lookback=50)
    regime, confidence = detector.detect_regime(ohlcv)
    
    print(f"\nCurrent Market Regime: {regime.value}")
    print(f"Confidence: {confidence:.2%}")
    
    # Test adaptive strategy
    strategy = RegimeAdaptiveStrategy()
    raw_action = np.array([0.5, 0.8])
    adapted_action, info = strategy.adapt_action(raw_action, ohlcv)
    
    print(f"\nRaw Action: {raw_action}")
    print(f"Adapted Action: {adapted_action}")
    print(f"Regime Info: {info}")

