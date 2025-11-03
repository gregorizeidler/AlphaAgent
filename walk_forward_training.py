"""
Walk-Forward Training and Analysis
Prevents overfitting by training on past and testing on future data
"""

import argparse
import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

from alpha_agent.environment import TradingEnvironment
from alpha_agent.agents import TradingPPOAgent, TradingCallback
from alpha_agent.risk import RiskManager
from alpha_agent.market_regime import RegimeAdaptiveStrategy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sns.set_style("darkgrid")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Walk-Forward Training')
    
    parser.add_argument('--ticker', type=str, default='AAPL', help='Stock ticker')
    parser.add_argument('--start-year', type=int, default=2018, help='Start year')
    parser.add_argument('--end-year', type=int, default=2024, help='End year')
    parser.add_argument('--train-window', type=int, default=2, help='Training window in years')
    parser.add_argument('--test-window', type=int, default=1, help='Test window in years')
    parser.add_argument('--timesteps', type=int, default=50000, help='Timesteps per training')
    parser.add_argument('--initial-balance', type=float, default=10000.0, help='Initial balance')
    parser.add_argument('--use-risk-manager', action='store_true', help='Use risk management')
    parser.add_argument('--use-regime-adaptation', action='store_true', help='Use regime adaptation')
    parser.add_argument('--output-dir', type=str, default='./walk_forward_results/', help='Output directory')
    
    return parser.parse_args()


def create_date_windows(start_year: int, end_year: int, 
                        train_window: int, test_window: int):
    """
    Create train/test date windows for walk-forward analysis
    
    Args:
        start_year: Starting year
        end_year: Ending year
        train_window: Training window in years
        test_window: Test window in years
        
    Returns:
        List of (train_start, train_end, test_start, test_end) tuples
    """
    windows = []
    current_year = start_year
    
    while current_year + train_window + test_window <= end_year:
        train_start = f"{current_year}-01-01"
        train_end = f"{current_year + train_window}-12-31"
        test_start = f"{current_year + train_window + 1}-01-01"
        test_end = f"{current_year + train_window + test_window}-12-31"
        
        windows.append((train_start, train_end, test_start, test_end))
        
        # Move forward by test_window years
        current_year += test_window
    
    return windows


def train_and_test_window(env_train, env_test, args, window_id: int):
    """
    Train on train window and test on test window
    
    Args:
        env_train: Training environment
        env_test: Test environment
        args: Arguments
        window_id: Window identifier
        
    Returns:
        Dictionary with results
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Window {window_id}: Training and Testing")
    logger.info(f"{'='*60}")
    
    # Create agent
    agent = TradingPPOAgent(
        env=env_train,
        learning_rate=3e-4,
        use_attention=False,
        verbose=1
    )
    
    # Train
    logger.info(f"Training on {env_train.max_steps} days...")
    callback = TradingCallback(
        check_freq=5000,
        save_path=f"{args.output_dir}/window_{window_id}",
        verbose=0
    )
    agent.train(total_timesteps=args.timesteps, callback=callback)
    
    # Test on unseen data
    logger.info(f"Testing on {env_test.max_steps} days (NEVER SEEN BEFORE)...")
    
    # Optional: Add risk management and regime adaptation
    if args.use_risk_manager:
        risk_manager = RiskManager(
            max_drawdown=0.15,
            max_daily_loss=0.03,
            max_position_size=0.30
        )
    else:
        risk_manager = None
    
    if args.use_regime_adaptation:
        regime_strategy = RegimeAdaptiveStrategy()
    else:
        regime_strategy = None
    
    # Run test episode
    obs, info = env_test.reset()
    done = False
    test_rewards = []
    test_actions = []
    
    while not done:
        # Get agent action
        action, _states = agent.predict(obs, deterministic=True)
        
        # Apply regime adaptation if enabled
        if regime_strategy:
            action, regime_info = regime_strategy.adapt_action(action, env_test.ohlcv_data)
        
        # Apply risk management if enabled
        if risk_manager:
            portfolio_state = {
                'portfolio_value': env_test.portfolio_value,
                'shares_held': env_test.shares_held,
                'position_ratio': (env_test.shares_held * env_test.ohlcv_data.iloc[env_test.lookback_days + env_test.current_step]['Close']) / env_test.portfolio_value,
                'entry_price': env_test.ohlcv_data.iloc[env_test.lookback_days + env_test.current_step]['Close'],
                'volatility': np.std(env_test.returns_history[-20:]) if len(env_test.returns_history) >= 20 else 0.02
            }
            current_price = env_test.ohlcv_data.iloc[env_test.lookback_days + env_test.current_step]['Close']
            action, risk_info = risk_manager.check_action_safety(action, portfolio_state, current_price)
        
        # Execute action
        obs, reward, terminated, truncated, info = env_test.step(action)
        test_rewards.append(reward)
        test_actions.append(action[0])
        done = terminated or truncated
    
    # Calculate metrics
    returns = np.array(env_test.returns_history)
    portfolio_values = np.array(env_test.portfolio_history)
    
    total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
    sharpe = (np.mean(returns) / (np.std(returns) + 1e-8)) * np.sqrt(252)
    
    # Sortino
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 0:
        sortino = (np.mean(returns) / (np.std(downside_returns) + 1e-8)) * np.sqrt(252)
    else:
        sortino = np.inf
    
    # Max drawdown
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (peak - portfolio_values) / peak
    max_drawdown = np.max(drawdown)
    
    # Win rate
    positive_returns = np.sum(returns > 0)
    win_rate = positive_returns / len(returns) if len(returns) > 0 else 0
    
    results = {
        'window_id': window_id,
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'final_value': portfolio_values[-1],
        'num_trades': len(env_test.trades_history),
        'portfolio_history': portfolio_values,
        'returns': returns,
        'actions': test_actions
    }
    
    logger.info(f"\nWindow {window_id} Results:")
    logger.info(f"  Total Return: {total_return*100:.2f}%")
    logger.info(f"  Sharpe Ratio: {sharpe:.2f}")
    logger.info(f"  Sortino Ratio: {sortino:.2f}")
    logger.info(f"  Max Drawdown: {max_drawdown*100:.2f}%")
    logger.info(f"  Win Rate: {win_rate*100:.2f}%")
    
    # Save model
    agent.save(f"{args.output_dir}/window_{window_id}/model.zip")
    
    return results


def plot_walk_forward_results(all_results: list, windows: list, save_path: str):
    """
    Plot walk-forward analysis results
    
    Args:
        all_results: List of result dictionaries
        windows: List of date windows
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Walk-Forward Analysis Results', fontsize=16, fontweight='bold')
    
    # Extract metrics
    window_ids = [r['window_id'] for r in all_results]
    returns = [r['total_return'] * 100 for r in all_results]
    sharpes = [r['sharpe_ratio'] for r in all_results]
    sortinos = [min(r['sortino_ratio'], 5) for r in all_results]  # Cap for visualization
    drawdowns = [r['max_drawdown'] * 100 for r in all_results]
    win_rates = [r['win_rate'] * 100 for r in all_results]
    
    # Plot 1: Returns per window
    ax = axes[0, 0]
    colors = ['green' if r > 0 else 'red' for r in returns]
    ax.bar(window_ids, returns, color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.set_title('Returns per Window (Out-of-Sample)')
    ax.set_xlabel('Window')
    ax.set_ylabel('Return (%)')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Sharpe Ratios
    ax = axes[0, 1]
    ax.plot(window_ids, sharpes, marker='o', linewidth=2)
    ax.axhline(y=1.0, color='orange', linestyle='--', label='Target (1.0)', alpha=0.7)
    ax.set_title('Sharpe Ratio per Window')
    ax.set_xlabel('Window')
    ax.set_ylabel('Sharpe Ratio')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Cumulative Performance
    ax = axes[0, 2]
    cumulative_returns = np.cumprod([1 + r/100 for r in returns]) - 1
    ax.plot(window_ids, cumulative_returns * 100, marker='o', linewidth=2, color='blue')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.set_title('Cumulative Returns')
    ax.set_xlabel('Window')
    ax.set_ylabel('Cumulative Return (%)')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Drawdowns
    ax = axes[1, 0]
    ax.bar(window_ids, drawdowns, color='red', alpha=0.7)
    ax.axhline(y=15, color='orange', linestyle='--', label='Limit (15%)', alpha=0.7)
    ax.set_title('Maximum Drawdown per Window')
    ax.set_xlabel('Window')
    ax.set_ylabel('Max Drawdown (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Win Rates
    ax = axes[1, 1]
    ax.plot(window_ids, win_rates, marker='o', linewidth=2, color='green')
    ax.axhline(y=50, color='gray', linestyle='--', label='Random (50%)', alpha=0.7)
    ax.set_title('Win Rate per Window')
    ax.set_xlabel('Window')
    ax.set_ylabel('Win Rate (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Statistics Summary
    ax = axes[1, 2]
    ax.axis('off')
    
    summary_text = f"""
    Walk-Forward Analysis Summary
    
    Number of Windows: {len(all_results)}
    
    Average Return: {np.mean(returns):.2f}%
    Std Dev Returns: {np.std(returns):.2f}%
    
    Average Sharpe: {np.mean(sharpes):.2f}
    Average Sortino: {np.mean(sortinos):.2f}
    
    Average Max DD: {np.mean(drawdowns):.2f}%
    Average Win Rate: {np.mean(win_rates):.2f}%
    
    Positive Windows: {sum(1 for r in returns if r > 0)}/{len(returns)}
    
    Final Cumulative: {cumulative_returns[-1]*100:.2f}%
    """
    
    ax.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
            fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Walk-forward plot saved to {save_path}")
    plt.show()


def main():
    """Main walk-forward training function"""
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("="*60)
    logger.info("Walk-Forward Training and Analysis")
    logger.info("="*60)
    logger.info(f"Ticker: {args.ticker}")
    logger.info(f"Period: {args.start_year} - {args.end_year}")
    logger.info(f"Train Window: {args.train_window} years")
    logger.info(f"Test Window: {args.test_window} year(s)")
    logger.info(f"Risk Management: {args.use_risk_manager}")
    logger.info(f"Regime Adaptation: {args.use_regime_adaptation}")
    logger.info("="*60)
    
    # Create date windows
    windows = create_date_windows(
        args.start_year,
        args.end_year,
        args.train_window,
        args.test_window
    )
    
    logger.info(f"\nCreated {len(windows)} walk-forward windows:")
    for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
        logger.info(f"  Window {i+1}:")
        logger.info(f"    Train: {train_start} to {train_end}")
        logger.info(f"    Test: {test_start} to {test_end}")
    
    # Run walk-forward analysis
    all_results = []
    
    for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
        # Create training environment
        env_train = TradingEnvironment(
            ticker=args.ticker,
            start_date=train_start,
            end_date=train_end,
            initial_balance=args.initial_balance,
            reward_type='sharpe',
            use_gaf=False
        )
        
        # Create test environment
        env_test = TradingEnvironment(
            ticker=args.ticker,
            start_date=test_start,
            end_date=test_end,
            initial_balance=args.initial_balance,
            reward_type='sharpe',
            use_gaf=False
        )
        
        # Train and test
        results = train_and_test_window(env_train, env_test, args, i+1)
        all_results.append(results)
        
        # Clean up
        env_train.close()
        env_test.close()
    
    # Plot results
    plot_path = f"{args.output_dir}/walk_forward_analysis.png"
    plot_walk_forward_results(all_results, windows, plot_path)
    
    # Save results to CSV
    results_df = pd.DataFrame(all_results)
    csv_path = f"{args.output_dir}/walk_forward_results.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info(f"\nResults saved to {csv_path}")
    
    logger.info("\n✓ Walk-Forward Analysis Completed!")


if __name__ == "__main__":
    main()

