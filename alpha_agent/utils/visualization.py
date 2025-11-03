"""
Visualization utilities for trading results
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sns.set_style("darkgrid")


def plot_training_progress(log_file: str, save_path: str = None):
    """
    Plot training progress from log file
    
    Args:
        log_file: Path to training log file
        save_path: Optional path to save plot
    """
    # This would parse TensorBoard logs
    # For now, placeholder
    logger.info(f"Plotting training progress from {log_file}")
    pass


def plot_portfolio_comparison(results_list: List[Dict], labels: List[str], save_path: str = None):
    """
    Compare multiple portfolio results
    
    Args:
        results_list: List of result dictionaries
        labels: Labels for each result
        save_path: Optional path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Portfolio values
    ax = axes[0, 0]
    for results, label in zip(results_list, labels):
        ax.plot(results['portfolio_history'], label=label, alpha=0.7)
    ax.set_title('Portfolio Value Comparison')
    ax.set_xlabel('Step')
    ax.set_ylabel('Value ($)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Returns distribution
    ax = axes[0, 1]
    for results, label in zip(results_list, labels):
        returns = results['returns_history']
        ax.hist(returns, bins=30, alpha=0.5, label=label)
    ax.set_title('Returns Distribution')
    ax.set_xlabel('Return')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Cumulative returns
    ax = axes[1, 0]
    for results, label in zip(results_list, labels):
        portfolio = np.array(results['portfolio_history'])
        cum_returns = (portfolio / portfolio[0] - 1) * 100
        ax.plot(cum_returns, label=label, alpha=0.7)
    ax.set_title('Cumulative Returns')
    ax.set_xlabel('Step')
    ax.set_ylabel('Return (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Metrics table
    ax = axes[1, 1]
    ax.axis('off')
    
    metrics_data = []
    for results, label in zip(results_list, labels):
        portfolio = np.array(results['portfolio_history'])
        total_return = (portfolio[-1] / portfolio[0] - 1) * 100
        returns = results['returns_history']
        sharpe = (np.mean(returns) / (np.std(returns) + 1e-8)) * np.sqrt(252)
        
        metrics_data.append([label, f"{total_return:.2f}%", f"{sharpe:.2f}"])
    
    table = ax.table(
        cellText=metrics_data,
        colLabels=['Strategy', 'Total Return', 'Sharpe'],
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Comparison plot saved to {save_path}")
    
    plt.show()


def plot_attention_heatmap(attention_weights: np.ndarray, labels: List[str] = None, save_path: str = None):
    """
    Plot attention weights as heatmap
    
    Args:
        attention_weights: Attention weight matrix
        labels: Optional labels for rows/columns
        save_path: Optional path to save plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(
        attention_weights,
        annot=False,
        cmap='viridis',
        ax=ax,
        xticklabels=labels if labels else False,
        yticklabels=labels if labels else False
    )
    
    ax.set_title('Attention Weights Heatmap')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Attention heatmap saved to {save_path}")
    
    plt.show()


def plot_action_distribution(actions: np.ndarray, save_path: str = None):
    """
    Plot distribution of agent actions
    
    Args:
        actions: Array of actions taken
        save_path: Optional path to save plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Histogram
    ax = axes[0]
    ax.hist(actions, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax.set_title('Action Distribution')
    ax.set_xlabel('Action (Position Change)')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)
    
    # Time series
    ax = axes[1]
    ax.plot(actions, alpha=0.7, linewidth=0.5)
    ax.axhline(y=0, color='r', linestyle='--', linewidth=1)
    ax.set_title('Actions Over Time')
    ax.set_xlabel('Step')
    ax.set_ylabel('Action')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Action distribution plot saved to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # Test visualization functions
    np.random.seed(42)
    
    # Create dummy results
    results = {
        'portfolio_history': [10000 * (1 + 0.001)**i + np.random.randn()*50 for i in range(100)],
        'returns_history': np.random.normal(0.001, 0.02, 100).tolist()
    }
    
    # Test action distribution
    actions = np.random.normal(0, 0.3, 100)
    plot_action_distribution(actions)

