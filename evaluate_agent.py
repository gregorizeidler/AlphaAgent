"""
Evaluation Script for AlphaAgent
With SHAP and LIME Explainability Analysis
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
from alpha_agent.explainability import (
    FeatureAttributor,
    LIMEExplainer,
    CombinedExplainer
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set plotting style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (14, 8)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate AlphaAgent')
    
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--ticker', type=str, default='AAPL', help='Stock ticker')
    parser.add_argument('--start-date', type=str, default=None, help='Start date')
    parser.add_argument('--end-date', type=str, default=None, help='End date')
    parser.add_argument('--initial-balance', type=float, default=10000.0, help='Initial balance')
    parser.add_argument('--n-episodes', type=int, default=10, help='Number of episodes')
    parser.add_argument('--deterministic', action='store_true', help='Use deterministic policy')
    parser.add_argument('--render', action='store_true', help='Render environment')
    parser.add_argument('--save-plots', action='store_true', help='Save plots')
    parser.add_argument('--output-dir', type=str, default='./results/', help='Output directory')
    parser.add_argument('--explainability', action='store_true', help='Run SHAP + LIME analysis')
    parser.add_argument('--n-explain', type=int, default=5, help='Number of decisions to explain')
    
    return parser.parse_args()


def evaluate_agent(agent, env, n_episodes: int = 10, deterministic: bool = True, render: bool = False):
    """
    Evaluate agent and collect metrics
    
    Args:
        agent: Trained agent
        env: Trading environment
        n_episodes: Number of episodes
        deterministic: Use deterministic policy
        render: Render environment
        
    Returns:
        Dictionary with evaluation results
    """
    logger.info(f"Evaluating agent for {n_episodes} episodes")
    
    results = {
        'episode_rewards': [],
        'episode_returns': [],
        'episode_sharpes': [],
        'episode_sortinos': [],
        'episode_max_drawdowns': [],
        'episode_num_trades': [],
        'portfolio_histories': [],
        'returns_histories': [],
    }
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        logger.info(f"\nEpisode {episode + 1}/{n_episodes}")
        
        while not done:
            action, _states = agent.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            step += 1
            
            if render and step % 10 == 0:
                env.render()
        
        # Collect episode metrics
        results['episode_rewards'].append(episode_reward)
        results['episode_returns'].append(info['total_return'])
        results['episode_num_trades'].append(info['num_trades'])
        results['portfolio_histories'].append(env.portfolio_history.copy())
        results['returns_histories'].append(env.returns_history.copy())
        
        # Calculate Sharpe ratio
        if len(env.returns_history) > 1:
            mean_return = np.mean(env.returns_history)
            std_return = np.std(env.returns_history)
            sharpe = (mean_return / (std_return + 1e-8)) * np.sqrt(252)  # Annualized
            results['episode_sharpes'].append(sharpe)
        
        # Calculate Sortino ratio
        downside_returns = [r for r in env.returns_history if r < 0]
        if len(downside_returns) > 1:
            downside_std = np.sqrt(np.mean(np.square(downside_returns)))
            sortino = (np.mean(env.returns_history) / (downside_std + 1e-8)) * np.sqrt(252)
            results['episode_sortinos'].append(sortino)
        
        # Calculate maximum drawdown
        portfolio_values = np.array(env.portfolio_history)
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        max_drawdown = np.max(drawdown)
        results['episode_max_drawdowns'].append(max_drawdown)
        
        logger.info(f"  Reward: {episode_reward:.2f}")
        logger.info(f"  Return: {info['total_return']*100:.2f}%")
        logger.info(f"  Sharpe: {sharpe:.2f}" if len(env.returns_history) > 1 else "  Sharpe: N/A")
        logger.info(f"  Max Drawdown: {max_drawdown*100:.2f}%")
        logger.info(f"  Num Trades: {info['num_trades']}")
    
    return results


def print_summary_statistics(results):
    """
    Print summary statistics
    
    Args:
        results: Evaluation results dictionary
    """
    logger.info("\n" + "="*60)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*60)
    
    metrics = {
        'Total Return': (results['episode_returns'], 100, '%'),
        'Sharpe Ratio': (results['episode_sharpes'], 1, ''),
        'Sortino Ratio': (results['episode_sortinos'], 1, ''),
        'Max Drawdown': (results['episode_max_drawdowns'], 100, '%'),
        'Num Trades': (results['episode_num_trades'], 1, ''),
        'Episode Reward': (results['episode_rewards'], 1, ''),
    }
    
    for name, (values, scale, unit) in metrics.items():
        if values:
            mean_val = np.mean(values) * scale
            std_val = np.std(values) * scale
            min_val = np.min(values) * scale
            max_val = np.max(values) * scale
            
            logger.info(f"\n{name}:")
            logger.info(f"  Mean: {mean_val:.2f}{unit} ± {std_val:.2f}{unit}")
            logger.info(f"  Range: [{min_val:.2f}{unit}, {max_val:.2f}{unit}]")


def plot_results(results, ticker: str, save_path: str = None):
    """
    Plot evaluation results
    
    Args:
        results: Evaluation results dictionary
        ticker: Stock ticker
        save_path: Path to save plots
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'AlphaAgent Evaluation Results - {ticker}', fontsize=16, fontweight='bold')
    
    # Portfolio value evolution
    ax = axes[0, 0]
    for i, portfolio_history in enumerate(results['portfolio_histories']):
        ax.plot(portfolio_history, alpha=0.6, label=f'Episode {i+1}')
    ax.axhline(y=results['portfolio_histories'][0][0], color='r', linestyle='--', label='Initial Balance')
    ax.set_title('Portfolio Value Evolution')
    ax.set_xlabel('Step')
    ax.set_ylabel('Portfolio Value ($)')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Returns distribution
    ax = axes[0, 1]
    all_returns = [r for returns in results['returns_histories'] for r in returns]
    ax.hist(all_returns, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax.set_title('Returns Distribution')
    ax.set_xlabel('Return')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)
    
    # Episode returns
    ax = axes[0, 2]
    episode_returns_pct = [r * 100 for r in results['episode_returns']]
    ax.bar(range(1, len(episode_returns_pct) + 1), episode_returns_pct, alpha=0.7)
    ax.axhline(y=0, color='r', linestyle='--')
    ax.set_title('Total Return by Episode')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Return (%)')
    ax.grid(True, alpha=0.3)
    
    # Sharpe ratios
    ax = axes[1, 0]
    if results['episode_sharpes']:
        ax.bar(range(1, len(results['episode_sharpes']) + 1), results['episode_sharpes'], alpha=0.7)
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_title('Sharpe Ratio by Episode')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Sharpe Ratio')
        ax.grid(True, alpha=0.3)
    
    # Max drawdowns
    ax = axes[1, 1]
    max_drawdowns_pct = [dd * 100 for dd in results['episode_max_drawdowns']]
    ax.bar(range(1, len(max_drawdowns_pct) + 1), max_drawdowns_pct, alpha=0.7, color='red')
    ax.set_title('Maximum Drawdown by Episode')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Max Drawdown (%)')
    ax.grid(True, alpha=0.3)
    
    # Number of trades
    ax = axes[1, 2]
    ax.bar(range(1, len(results['episode_num_trades']) + 1), results['episode_num_trades'], alpha=0.7, color='green')
    ax.set_title('Number of Trades by Episode')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Num Trades')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
    
    plt.show()


def run_explainability_analysis(agent, env, n_decisions: int = 5, output_dir: str = './results/'):
    """
    Run SHAP and LIME explainability analysis
    
    Args:
        agent: Trained agent
        env: Trading environment
        n_decisions: Number of decisions to explain
        output_dir: Output directory for plots
    """
    logger.info("\n" + "="*60)
    logger.info("🔍 EXPLAINABILITY ANALYSIS (SHAP + LIME)")
    logger.info("="*60)
    
    # Collect observations
    logger.info("Collecting observations for analysis...")
    observations = []
    obs, _ = env.reset()
    
    for _ in range(100):  # Collect 100 observations for background
        observations.append(obs.copy())
        action, _ = agent.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
    
    observations = np.array(observations)
    
    # Create REAL feature names
    from alpha_agent.explainability.feature_attribution import generate_feature_names
    feature_names = generate_feature_names(
        state_size=observations.shape[1],
        lookback_days=30,
        use_gaf=False
    )
    logger.info(f"Using {len(feature_names)} real feature names")
    
    # Initialize combined explainer
    logger.info("Initializing SHAP + LIME explainers...")
    explainer = CombinedExplainer(
        agent=agent,
        feature_names=feature_names,
        background_data=observations
    )
    
    # Explain random decisions
    logger.info(f"\nExplaining {n_decisions} random decisions...")
    
    for i in range(n_decisions):
        idx = np.random.randint(0, len(observations))
        obs = observations[idx]
        
        logger.info(f"\nDecision {i+1}/{n_decisions}:")
        
        # Get combined explanation
        explanation = explainer.explain_action(obs)
        
        action = explanation['action']
        decision = "BUY" if action > 0.1 else "SELL" if action < -0.1 else "HOLD"
        
        logger.info(f"  Action: {decision} ({action:.3f})")
        logger.info(f"  Agreement Score: {explanation['agreement_score']:.2%}")
        
        logger.info("  Top SHAP features:")
        for feat in explanation['shap_explanation']['top_features'][:3]:
            logger.info(f"    • {feat['feature']}: {feat['shap_value']:+.4f}")
        
        logger.info("  Top LIME features:")
        for feat in explanation['lime_explanation']['top_features'][:3]:
            logger.info(f"    • {feat['feature']}: {feat['importance']:+.4f}")
    
    # Generate SHAP classic plots (beeswarm + bar)
    logger.info("\nGenerating SHAP classic plots (beeswarm + bar)...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Combined SHAP plot (Global + Local like the classic image)
    shap_combined_path = f"{output_dir}/shap_combined_{timestamp}.png"
    explainer.shap_explainer.plot_combined_summary(
        observations[:50],  # Use subset for speed
        max_display=15,
        save_path=shap_combined_path
    )
    
    # SHAP vs LIME comparison
    logger.info("\nGenerating SHAP vs LIME comparison...")
    comparison_path = f"{output_dir}/shap_lime_comparison_{timestamp}.png"
    random_idx = np.random.randint(0, len(observations))
    explainer.plot_comparison(observations[random_idx], save_path=comparison_path)
    
    logger.info(f"✓ SHAP classic plot saved: {shap_combined_path}")
    logger.info(f"✓ SHAP vs LIME comparison saved: {comparison_path}")
    logger.info("="*60)


def main():
    """Main evaluation function"""
    args = parse_args()
    
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("="*60)
    logger.info("AlphaAgent Evaluation")
    logger.info("="*60)
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Ticker: {args.ticker}")
    logger.info(f"Episodes: {args.n_episodes}")
    logger.info(f"Explainability: {'YES' if args.explainability else 'NO'}")
    logger.info("="*60)
    
    # Create environment
    env = TradingEnvironment(
        ticker=args.ticker,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_balance=args.initial_balance,
        reward_type='sharpe',
        use_gaf=False,  # Use False for faster evaluation
    )
    
    # Load agent
    logger.info(f"Loading model from {args.model_path}")
    agent = TradingPPOAgent(env=env)
    agent.load(args.model_path)
    
    # Evaluate
    results = evaluate_agent(
        agent=agent,
        env=env,
        n_episodes=args.n_episodes,
        deterministic=args.deterministic,
        render=args.render
    )
    
    # Print summary
    print_summary_statistics(results)
    
    # Plot results
    if args.save_plots:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = f"{args.output_dir}/evaluation_{args.ticker}_{timestamp}.png"
    else:
        plot_path = None
    
    plot_results(results, args.ticker, save_path=plot_path)
    
    # Save results to CSV
    results_df = pd.DataFrame({
        'episode': range(1, args.n_episodes + 1),
        'reward': results['episode_rewards'],
        'return': results['episode_returns'],
        'sharpe': results['episode_sharpes'] if results['episode_sharpes'] else [None] * args.n_episodes,
        'max_drawdown': results['episode_max_drawdowns'],
        'num_trades': results['episode_num_trades'],
    })
    
    csv_path = f"{args.output_dir}/evaluation_{args.ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info(f"\nResults saved to {csv_path}")
    
    # Run explainability analysis if requested
    if args.explainability:
        try:
            run_explainability_analysis(
                agent=agent,
                env=env,
                n_decisions=args.n_explain,
                output_dir=args.output_dir
            )
        except Exception as e:
            logger.error(f"Explainability analysis failed: {e}")
            logger.error("Continuing with evaluation...")
    
    env.close()
    logger.info("\n✓ Evaluation completed!")


if __name__ == "__main__":
    main()

