"""
Automated Retraining Pipeline
Monitors model performance and triggers retraining when needed
"""

import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import logging
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """
    Monitors agent performance metrics
    """
    
    def __init__(self, window_size: int = 30):
        """
        Args:
            window_size: Days to track for performance
        """
        self.window_size = window_size
        self.performance_history = []
        logger.info(f"PerformanceMonitor initialized (window={window_size} days)")
    
    def add_performance(self, sharpe: float, returns: float, drawdown: float):
        """
        Add performance data
        
        Args:
            sharpe: Sharpe ratio
            returns: Total returns
            drawdown: Max drawdown
        """
        self.performance_history.append({
            'timestamp': datetime.now(),
            'sharpe': sharpe,
            'returns': returns,
            'drawdown': drawdown
        })
        
        # Keep only recent window
        if len(self.performance_history) > self.window_size:
            self.performance_history = self.performance_history[-self.window_size:]
    
    def is_degraded(self, threshold_sharpe: float = 0.5) -> bool:
        """
        Check if performance has degraded
        
        Args:
            threshold_sharpe: Minimum acceptable Sharpe
            
        Returns:
            True if degraded
        """
        if len(self.performance_history) < 10:
            return False
        
        # Compare recent vs older performance
        recent = self.performance_history[-7:]
        older = self.performance_history[-self.window_size:-7]
        
        if not older:
            return False
        
        recent_sharpe = np.mean([p['sharpe'] for p in recent])
        older_sharpe = np.mean([p['sharpe'] for p in older])
        
        # Significant degradation?
        if older_sharpe - recent_sharpe > threshold_sharpe:
            logger.warning(f"Performance degradation detected: {older_sharpe:.2f} → {recent_sharpe:.2f}")
            return True
        
        # Below minimum threshold?
        if recent_sharpe < threshold_sharpe:
            logger.warning(f"Performance below threshold: {recent_sharpe:.2f} < {threshold_sharpe:.2f}")
            return True
        
        return False
    
    def get_metrics(self) -> Dict:
        """
        Get current performance metrics
        
        Returns:
            Metrics dictionary
        """
        if not self.performance_history:
            return {}
        
        recent = self.performance_history[-7:]
        
        return {
            'recent_sharpe': np.mean([p['sharpe'] for p in recent]),
            'recent_returns': np.mean([p['returns'] for p in recent]),
            'recent_drawdown': np.max([p['drawdown'] for p in recent]),
            'num_observations': len(self.performance_history)
        }


class AutoRetrainPipeline:
    """
    Automated retraining system
    """
    
    def __init__(
        self,
        agent,
        env,
        monitor: PerformanceMonitor,
        retrain_interval_days: int = 30,
        min_performance_sharpe: float = 0.5
    ):
        """
        Args:
            agent: Trading agent
            env: Training environment
            monitor: Performance monitor
            retrain_interval_days: Days between automatic retrains
            min_performance_sharpe: Minimum Sharpe before triggering retrain
        """
        self.agent = agent
        self.env = env
        self.monitor = monitor
        self.retrain_interval = timedelta(days=retrain_interval_days)
        self.min_performance_sharpe = min_performance_sharpe
        
        self.last_retrain = datetime.now()
        self.retrain_history = []
        
        logger.info(f"AutoRetrainPipeline initialized (interval={retrain_interval_days} days)")
    
    def should_retrain(self) -> Tuple[bool, str]:
        """
        Determine if retraining is needed
        
        Returns:
            (should_retrain, reason)
        """
        # Check 1: Time-based
        time_since_retrain = datetime.now() - self.last_retrain
        if time_since_retrain >= self.retrain_interval:
            return True, f"Scheduled retrain (last: {time_since_retrain.days} days ago)"
        
        # Check 2: Performance-based
        if self.monitor.is_degraded(threshold_sharpe=self.min_performance_sharpe):
            return True, "Performance degradation detected"
        
        return False, "No retrain needed"
    
    def retrain(self, timesteps: int = 100000) -> Dict:
        """
        Retrain the agent
        
        Args:
            timesteps: Training timesteps
            
        Returns:
            Retrain results
        """
        logger.info("="*60)
        logger.info("STARTING AUTOMATED RETRAINING")
        logger.info("="*60)
        
        start_time = datetime.now()
        
        # Backup current model
        backup_path = self._backup_model()
        logger.info(f"Current model backed up to {backup_path}")
        
        # Get current performance baseline
        pre_retrain_metrics = self.monitor.get_metrics()
        logger.info(f"Pre-retrain Sharpe: {pre_retrain_metrics.get('recent_sharpe', 0):.2f}")
        
        # Retrain
        logger.info(f"Training for {timesteps} timesteps...")
        self.agent.learn(total_timesteps=timesteps)
        
        # Save new model
        model_path = self._save_model()
        logger.info(f"Retrained model saved to {model_path}")
        
        # Evaluate new model (would need evaluation environment)
        # For now, assume improvement
        post_retrain_metrics = {'sharpe': 0.0}  # Placeholder
        
        # Record retrain
        duration = (datetime.now() - start_time).total_seconds()
        
        retrain_record = {
            'timestamp': start_time,
            'duration_seconds': duration,
            'timesteps': timesteps,
            'pre_sharpe': pre_retrain_metrics.get('recent_sharpe', 0),
            'post_sharpe': post_retrain_metrics['sharpe'],
            'backup_path': backup_path,
            'model_path': model_path
        }
        
        self.retrain_history.append(retrain_record)
        self.last_retrain = datetime.now()
        
        logger.info("="*60)
        logger.info(f"RETRAINING COMPLETED in {duration:.1f}s")
        logger.info("="*60)
        
        return retrain_record
    
    def _backup_model(self) -> str:
        """
        Backup current model
        
        Returns:
            Backup file path
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = "./models/backups"
        os.makedirs(backup_dir, exist_ok=True)
        
        backup_path = f"{backup_dir}/agent_backup_{timestamp}.zip"
        self.agent.save(backup_path)
        
        return backup_path
    
    def _save_model(self) -> str:
        """
        Save retrained model
        
        Returns:
            Model file path
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = "./models"
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = f"{model_dir}/agent_retrained_{timestamp}.zip"
        self.agent.save(model_path)
        
        return model_path
    
    def check_and_retrain(self, timesteps: int = 100000) -> Optional[Dict]:
        """
        Check if retrain needed and execute if so
        
        Args:
            timesteps: Training timesteps
            
        Returns:
            Retrain results if executed, None otherwise
        """
        should_retrain, reason = self.should_retrain()
        
        if should_retrain:
            logger.info(f"Retraining triggered: {reason}")
            return self.retrain(timesteps=timesteps)
        else:
            logger.info(f"Retrain check: {reason}")
            return None
    
    def export_history(self, filepath: str):
        """
        Export retrain history
        
        Args:
            filepath: Path to save
        """
        df = pd.DataFrame(self.retrain_history)
        df.to_csv(filepath, index=False)
        logger.info(f"Retrain history exported to {filepath}")


class DataUpdatePipeline:
    """
    Automatically updates training data with latest market data
    """
    
    def __init__(self, data_fetcher, tickers: list, lookback_years: int = 5):
        """
        Args:
            data_fetcher: MarketDataFetcher instance
            tickers: List of tickers to fetch
            lookback_years: Years of historical data
        """
        self.data_fetcher = data_fetcher
        self.tickers = tickers
        self.lookback_years = lookback_years
        
        logger.info("DataUpdatePipeline initialized")
    
    def update_data(self) -> Dict:
        """
        Fetch latest market data
        
        Returns:
            Updated data dictionary
        """
        logger.info("Updating market data...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * self.lookback_years)
        
        updated_data = {}
        
        for ticker in self.tickers:
            try:
                data = self.data_fetcher.fetch_ohlcv(
                    ticker,
                    start_date.strftime("%Y-%m-%d"),
                    end_date.strftime("%Y-%m-%d")
                )
                updated_data[ticker] = data
                logger.info(f"Updated {ticker}: {len(data)} days")
            except Exception as e:
                logger.error(f"Failed to update {ticker}: {e}")
        
        return updated_data
    
    def save_data(self, data: Dict, filepath: str):
        """
        Save updated data
        
        Args:
            data: Data dictionary
            filepath: Path to save
        """
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Data saved to {filepath}")


if __name__ == "__main__":
    print("Auto-Retrain Pipeline")
    print("\nFeatures:")
    print("1. Performance monitoring")
    print("2. Automatic retrain triggering")
    print("3. Model versioning and backups")
    print("4. Data update automation")

