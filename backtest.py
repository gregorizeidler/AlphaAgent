"""
Backtesting Script for AlphaAgent
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging

from alpha_agent.environment import TradingEnvironment
from alpha_agent.agents import TradingPPOAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sns.set_style("darkgrid")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Backtest AlphaAgent')
    
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--ticker', type=str, default='AAPL', help='Stock ticker')
    parser.add_argument('--start-date', type=str, required=True, help='Backtest start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None, help='Backtest end date (YYYY-MM-DD)')
    parser.add_argument('--initial-balance', type=float, default=10000.0, help='Initial balance')
    parser.add_argument('--compare-buyhold', action='store_true', help='Compare with buy-and-hold strategy')
    parser.add_argument('--output-dir', type=str, default='./backtest_results/', help='Output directory')
    
    return parser.parse_args()


def run_backtest(agent, env):
    """
    Run backtest on environment
    
    Args:
        agent: Trained agent
        env: Trading environment
        
    Returns:
        Backtest results dictionary
    """
    logger.info("Running backtest...")
    
    obs, info = env.reset()
    done = False
    
    results = {
        'dates': [],
        'portfolio_values': [],
        'cash_balances': [],
        'positions': [],
        'prices': [],
        'actions': [],
        'rewards': [],
    }
    
    step = 0
    while not done:
        action, _states = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Record data
        current_idx = env.lookback_days + env.current_step
        date = env.ohlcv_data.index[current_idx]
        
        results['dates'].append(date)
        results['portfolio_values'].append(info['portfolio_value'])
        results['cash_balances'].append(info['balance'])
        results['positions'].append(info['shares_held'])
        results['prices'].append(info['current_price'])
        results['actions'].append(action[0])
        results['rewards'].append(reward)
        
        step += 1
        if step % 50 == 0:
            logger.info(f"Step {step}: Portfolio=${info['portfolio_value']:.2f}, Return={info['total_return']*100:.2f}%")
    
    logger.info(f"Backtest completed: {step} steps")
    
    return results


def calculate_metrics(results, initial_balance):
    """
    Calculate performance metrics
    
    Args:
        results: Backtest results
        initial_balance: Initial balance
        
    Returns:
        Metrics dictionary
    """
    portfolio_values = np.array(results['portfolio_values'])
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    
    total_return = (portfolio_values[-1] - initial_balance) / initial_balance
    
    # Sharpe ratio
    sharpe = (np.mean(returns) / (np.std(returns) + 1e-8)) * np.sqrt(252)
    
    # Sortino ratio
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 0:
        sortino = (np.mean(returns) / (np.std(downside_returns) + 1e-8)) * np.sqrt(252)
    else:
        sortino = np.inf
    
    # Maximum drawdown
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (peak - portfolio_values) / peak
    max_drawdown = np.max(drawdown)
    
    # Win rate
    positive_returns = np.sum(returns > 0)
    win_rate = positive_returns / len(returns) if len(returns) > 0 else 0
    
    # Profit factor
    gains = np.sum(returns[returns > 0])
    losses = abs(np.sum(returns[returns < 0]))
    profit_factor = gains / losses if losses > 0 else np.inf
    
    metrics = {
        'total_return': total_return,
        'annualized_return': total_return / (len(portfolio_values) / 252),
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'final_portfolio_value': portfolio_values[-1],
    }
    
    return metrics


def buy_and_hold_benchmark(prices, initial_balance):
    """
    Calculate buy-and-hold benchmark
    
    Args:
        prices: Array of prices
        initial_balance: Initial balance
        
    Returns:
        Buy-and-hold portfolio values
    """
    shares = initial_balance / prices[0]
    portfolio_values = shares * prices
    return portfolio_values


def plot_backtest_results(results, metrics, ticker, buyhold_values=None, save_path=None):
    """
    Plot backtest results
    
    Args:
        results: Backtest results
        metrics: Performance metrics
        ticker: Stock ticker
        buyhold_values: Optional buy-and-hold values
        save_path: Path to save plot
    """
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)
    
    fig.suptitle(f'AlphaAgent Backtest Results - {ticker}', fontsize=16, fontweight='bold')
    
    # Portfolio value
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(results['dates'], results['portfolio_values'], label='AlphaAgent', linewidth=2)
    if buyhold_values is not None:
        ax1.plot(results['dates'], buyhold_values, label='Buy & Hold', linewidth=2, linestyle='--', alpha=0.7)
    ax1.axhline(y=results['portfolio_values'][0], color='gray', linestyle=':', alpha=0.5)
    ax1.set_title('Portfolio Value Over Time', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Drawdown
    ax2 = fig.add_subplot(gs[1, 0])
    portfolio_values = np.array(results['portfolio_values'])
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (peak - portfolio_values) / peak * 100
    ax2.fill_between(results['dates'], drawdown, alpha=0.3, color='red')
    ax2.plot(results['dates'], drawdown, color='red', linewidth=1)
    ax2.set_title('Drawdown', fontsize=12)
    ax2.set_ylabel('Drawdown (%)')
    ax2.grid(True, alpha=0.3)
    
    # Cumulative returns
    ax3 = fig.add_subplot(gs[1, 1])
    cumulative_returns = (portfolio_values / portfolio_values[0] - 1) * 100
    ax3.plot(results['dates'], cumulative_returns, linewidth=2)
    ax3.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax3.set_title('Cumulative Returns', fontsize=12)
    ax3.set_ylabel('Return (%)')
    ax3.grid(True, alpha=0.3)
    
    # Positions
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(results['dates'], results['positions'], linewidth=1)
    ax4.set_title('Position Size (Shares)', fontsize=12)
    ax4.set_ylabel('Shares Held')
    ax4.grid(True, alpha=0.3)
    
    # Actions distribution
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.hist(results['actions'], bins=50, alpha=0.7, edgecolor='black')
    ax5.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax5.set_title('Actions Distribution', fontsize=12)
    ax5.set_xlabel('Action')
    ax5.set_ylabel('Frequency')
    ax5.grid(True, alpha=0.3)
    
    # Price
    ax6 = fig.add_subplot(gs[3, 0])
    ax6.plot(results['dates'], results['prices'], linewidth=1, color='orange')
    ax6.set_title(f'{ticker} Price', fontsize=12)
    ax6.set_ylabel('Price ($)')
    ax6.grid(True, alpha=0.3)
    
    # Metrics table
    ax7 = fig.add_subplot(gs[3, 1])
    ax7.axis('off')
    
    metrics_text = f"""
    Performance Metrics:
    
    Total Return: {metrics['total_return']*100:.2f}%
    Annualized Return: {metrics['annualized_return']*100:.2f}%
    Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
    Sortino Ratio: {metrics['sortino_ratio']:.2f}
    Max Drawdown: {metrics['max_drawdown']*100:.2f}%
    Win Rate: {metrics['win_rate']*100:.2f}%
    Profit Factor: {metrics['profit_factor']:.2f}
    
    Final Value: ${metrics['final_portfolio_value']:.2f}
    """
    
    ax7.text(0.1, 0.5, metrics_text, fontsize=11, verticalalignment='center',
             fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
    
    plt.show()


def main():
    """Main backtesting function"""
    args = parse_args()
    
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("="*60)
    logger.info("AlphaAgent Backtesting")
    logger.info("="*60)
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Ticker: {args.ticker}")
    logger.info(f"Period: {args.start_date} to {args.end_date or 'present'}")
    logger.info("="*60)
    
    # Create environment
    env = TradingEnvironment(
        ticker=args.ticker,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_balance=args.initial_balance,
        reward_type='sharpe',
        use_gaf=False,
    )
    
    # Load agent
    logger.info(f"Loading model from {args.model_path}")
    agent = TradingPPOAgent(env=env)
    agent.load(args.model_path)
    
    # Run backtest
    results = run_backtest(agent, env)
    
    # Calculate metrics
    metrics = calculate_metrics(results, args.initial_balance)
    
    # Print metrics
    logger.info("\n" + "="*60)
    logger.info("BACKTEST RESULTS")
    logger.info("="*60)
    logger.info(f"Total Return: {metrics['total_return']*100:.2f}%")
    logger.info(f"Annualized Return: {metrics['annualized_return']*100:.2f}%")
    logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    logger.info(f"Sortino Ratio: {metrics['sortino_ratio']:.2f}")
    logger.info(f"Maximum Drawdown: {metrics['max_drawdown']*100:.2f}%")
    logger.info(f"Win Rate: {metrics['win_rate']*100:.2f}%")
    logger.info(f"Profit Factor: {metrics['profit_factor']:.2f}")
    logger.info(f"Final Portfolio Value: ${metrics['final_portfolio_value']:.2f}")
    
    # Buy-and-hold comparison
    buyhold_values = None
    if args.compare_buyhold:
        buyhold_values = buy_and_hold_benchmark(
            np.array(results['prices']),
            args.initial_balance
        )
        buyhold_return = (buyhold_values[-1] - args.initial_balance) / args.initial_balance
        logger.info(f"\nBuy & Hold Return: {buyhold_return*100:.2f}%")
        logger.info(f"Alpha (outperformance): {(metrics['total_return'] - buyhold_return)*100:.2f}%")
    
    logger.info("="*60)
    
    # Plot results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = f"{args.output_dir}/backtest_{args.ticker}_{timestamp}.png"
    plot_backtest_results(results, metrics, args.ticker, buyhold_values, save_path=plot_path)
    
    # Save results to CSV
    results_df = pd.DataFrame({
        'date': results['dates'],
        'portfolio_value': results['portfolio_values'],
        'cash_balance': results['cash_balances'],
        'position': results['positions'],
        'price': results['prices'],
        'action': results['actions'],
        'reward': results['rewards'],
    })
    
    if buyhold_values is not None:
        results_df['buyhold_value'] = buyhold_values
    
    csv_path = f"{args.output_dir}/backtest_{args.ticker}_{timestamp}.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info(f"\nResults saved to {csv_path}")
    
    env.close()
    logger.info("\n✓ Backtesting completed!")


if __name__ == "__main__":
    main()

