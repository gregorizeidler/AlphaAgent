"""
Helper functions for AlphaAgent
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_portfolio_metrics(portfolio_values: List[float], 
                                returns: List[float],
                                risk_free_rate: float = 0.02) -> Dict[str, float]:
    """
    Calculate comprehensive portfolio metrics
    
    Args:
        portfolio_values: List of portfolio values
        returns: List of returns
        risk_free_rate: Annual risk-free rate
        
    Returns:
        Dictionary of metrics
    """
    portfolio_values = np.array(portfolio_values)
    returns = np.array(returns)
    
    # Total return
    total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
    
    # Annualized return
    num_periods = len(portfolio_values)
    years = num_periods / 252  # Assuming daily data
    annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    
    # Volatility
    volatility = np.std(returns) * np.sqrt(252)
    
    # Sharpe ratio
    daily_rf = risk_free_rate / 252
    excess_returns = returns - daily_rf
    sharpe = (np.mean(excess_returns) / (np.std(returns) + 1e-8)) * np.sqrt(252)
    
    # Sortino ratio
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 0:
        sortino = (np.mean(excess_returns) / (np.std(downside_returns) + 1e-8)) * np.sqrt(252)
    else:
        sortino = np.inf
    
    # Maximum drawdown
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (peak - portfolio_values) / peak
    max_drawdown = np.max(drawdown)
    
    # Calmar ratio
    calmar = annualized_return / max_drawdown if max_drawdown > 0 else np.inf
    
    # Win rate
    positive_returns = np.sum(returns > 0)
    win_rate = positive_returns / len(returns) if len(returns) > 0 else 0
    
    # Average win/loss
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    avg_win = np.mean(wins) if len(wins) > 0 else 0
    avg_loss = np.mean(losses) if len(losses) > 0 else 0
    
    # Profit factor
    total_wins = np.sum(wins) if len(wins) > 0 else 0
    total_losses = abs(np.sum(losses)) if len(losses) > 0 else 0
    profit_factor = total_wins / total_losses if total_losses > 0 else np.inf
    
    metrics = {
        'total_return': float(total_return),
        'annualized_return': float(annualized_return),
        'volatility': float(volatility),
        'sharpe_ratio': float(sharpe),
        'sortino_ratio': float(sortino),
        'max_drawdown': float(max_drawdown),
        'calmar_ratio': float(calmar),
        'win_rate': float(win_rate),
        'avg_win': float(avg_win),
        'avg_loss': float(avg_loss),
        'profit_factor': float(profit_factor),
    }
    
    return metrics


def save_results_to_json(results: Dict, filepath: str):
    """
    Save results to JSON file
    
    Args:
        results: Results dictionary
        filepath: Path to save JSON file
    """
    # Convert numpy arrays to lists for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    serializable_results = convert_to_serializable(results)
    
    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"Results saved to {filepath}")


def load_results_from_json(filepath: str) -> Dict:
    """
    Load results from JSON file
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Results dictionary
    """
    with open(filepath, 'r') as f:
        results = json.load(f)
    
    logger.info(f"Results loaded from {filepath}")
    return results


def split_train_test_dates(start_date: str, end_date: str, 
                           train_ratio: float = 0.8) -> Tuple[str, str, str, str]:
    """
    Split date range into train and test periods
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        train_ratio: Ratio of training data
        
    Returns:
        train_start, train_end, test_start, test_end
    """
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    total_days = (end - start).days
    train_days = int(total_days * train_ratio)
    
    train_start = start
    train_end = start + pd.Timedelta(days=train_days)
    test_start = train_end + pd.Timedelta(days=1)
    test_end = end
    
    return (
        train_start.strftime('%Y-%m-%d'),
        train_end.strftime('%Y-%m-%d'),
        test_start.strftime('%Y-%m-%d'),
        test_end.strftime('%Y-%m-%d')
    )


def format_large_number(number: float) -> str:
    """
    Format large numbers with K, M, B suffixes
    
    Args:
        number: Number to format
        
    Returns:
        Formatted string
    """
    if abs(number) >= 1e9:
        return f"${number/1e9:.2f}B"
    elif abs(number) >= 1e6:
        return f"${number/1e6:.2f}M"
    elif abs(number) >= 1e3:
        return f"${number/1e3:.2f}K"
    else:
        return f"${number:.2f}"


def print_metrics_table(metrics: Dict):
    """
    Print metrics in a formatted table
    
    Args:
        metrics: Metrics dictionary
    """
    print("\n" + "="*60)
    print("PERFORMANCE METRICS")
    print("="*60)
    
    metric_formats = {
        'total_return': ('Total Return', lambda x: f"{x*100:.2f}%"),
        'annualized_return': ('Annualized Return', lambda x: f"{x*100:.2f}%"),
        'volatility': ('Volatility', lambda x: f"{x*100:.2f}%"),
        'sharpe_ratio': ('Sharpe Ratio', lambda x: f"{x:.3f}"),
        'sortino_ratio': ('Sortino Ratio', lambda x: f"{x:.3f}"),
        'max_drawdown': ('Max Drawdown', lambda x: f"{x*100:.2f}%"),
        'calmar_ratio': ('Calmar Ratio', lambda x: f"{x:.3f}"),
        'win_rate': ('Win Rate', lambda x: f"{x*100:.2f}%"),
        'profit_factor': ('Profit Factor', lambda x: f"{x:.3f}"),
    }
    
    for key, (label, formatter) in metric_formats.items():
        if key in metrics:
            value = metrics[key]
            if np.isinf(value):
                formatted = "∞"
            else:
                formatted = formatter(value)
            print(f"{label:25s}: {formatted}")
    
    print("="*60)


if __name__ == "__main__":
    # Test helper functions
    np.random.seed(42)
    
    # Generate dummy data
    portfolio_values = [10000 * (1.0005)**i + np.random.randn()*20 for i in range(252)]
    returns = [(portfolio_values[i] - portfolio_values[i-1])/portfolio_values[i-1] 
               for i in range(1, len(portfolio_values))]
    
    # Calculate metrics
    metrics = calculate_portfolio_metrics(portfolio_values, returns)
    
    # Print metrics
    print_metrics_table(metrics)
    
    # Test date splitting
    train_start, train_end, test_start, test_end = split_train_test_dates(
        '2020-01-01', '2024-12-31', train_ratio=0.8
    )
    print(f"\nTrain: {train_start} to {train_end}")
    print(f"Test: {test_start} to {test_end}")

