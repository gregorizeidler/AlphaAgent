"""
Training Script for AlphaAgent with Multi-Horizon Support
"""

import argparse
import os
from datetime import datetime, timedelta
import numpy as np
import logging

from alpha_agent.environment import TradingEnvironment
from alpha_agent.agents import TradingPPOAgent, TradingCallback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train AlphaAgent Trading Bot')
    
    # Environment parameters
    parser.add_argument('--ticker', type=str, default='AAPL', help='Stock ticker symbol')
    parser.add_argument('--start-date', type=str, default=None, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None, help='End date (YYYY-MM-DD)')
    parser.add_argument('--initial-balance', type=float, default=10000.0, help='Initial balance')
    parser.add_argument('--transaction-cost', type=float, default=0.001, help='Transaction cost rate')
    parser.add_argument('--slippage', type=float, default=0.0005, help='Slippage rate')
    parser.add_argument('--lookback-days', type=int, default=30, help='Lookback window in days')
    parser.add_argument('--reward-type', type=str, default='sharpe', 
                       choices=['sharpe', 'sortino', 'composite'], help='Reward function type')
    parser.add_argument('--use-gaf', action='store_true', help='Use GAF transformation for prices')
    parser.add_argument('--enable-shorting', action='store_true', help='Enable short selling')
    
    # Agent parameters
    parser.add_argument('--learning-rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--n-steps', type=int, default=2048, help='Steps per update')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--n-epochs', type=int, default=10, help='Epochs per update')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--use-attention', action='store_true', help='Use attention mechanism')
    
    # Training parameters
    parser.add_argument('--total-timesteps', type=int, default=100000, help='Total training timesteps')
    parser.add_argument('--save-freq', type=int, default=5000, help='Model save frequency')
    parser.add_argument('--log-dir', type=str, default='./logs/', help='Log directory')
    parser.add_argument('--model-dir', type=str, default='./models/', help='Model save directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Multi-horizon training
    parser.add_argument('--multi-horizon', action='store_true', help='Enable multi-horizon training')
    parser.add_argument('--horizons', type=str, default='1,5,20', help='Comma-separated horizons (days)')
    
    return parser.parse_args()


def create_environment(args):
    """
    Create trading environment
    
    Args:
        args: Command line arguments
        
    Returns:
        Trading environment
    """
    logger.info(f"Creating environment for {args.ticker}")
    
    env = TradingEnvironment(
        ticker=args.ticker,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_balance=args.initial_balance,
        transaction_cost=args.transaction_cost,
        slippage=args.slippage,
        lookback_days=args.lookback_days,
        reward_type=args.reward_type,
        use_gaf=args.use_gaf,
        enable_shorting=args.enable_shorting,
    )
    
    return env


def train_single_horizon(args):
    """
    Train agent on single horizon
    
    Args:
        args: Command line arguments
    """
    # Create directories
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Create environment
    env = create_environment(args)
    
    logger.info(f"Environment created: {env.max_steps} trading days")
    logger.info(f"Observation space: {env.observation_space.shape}")
    logger.info(f"Action space: {env.action_space.shape}")
    
    # Create agent
    agent = TradingPPOAgent(
        env=env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        use_attention=args.use_attention,
        verbose=1
    )
    
    # Create callback
    callback = TradingCallback(
        check_freq=args.save_freq,
        save_path=args.model_dir,
        verbose=1
    )
    
    # Train
    logger.info(f"Starting training for {args.total_timesteps} timesteps")
    agent.train(total_timesteps=args.total_timesteps, callback=callback)
    
    # Save final model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"{args.model_dir}/ppo_agent_{args.ticker}_{timestamp}.zip"
    agent.save(model_path)
    
    logger.info(f"Training completed! Model saved to {model_path}")
    
    # Evaluate
    logger.info("Evaluating trained agent...")
    evaluate_agent(agent, env, n_episodes=5)
    
    env.close()


def train_multi_horizon(args):
    """
    Train agent with multiple horizons
    
    Args:
        args: Command line arguments
    """
    horizons = [int(h) for h in args.horizons.split(',')]
    logger.info(f"Multi-horizon training with horizons: {horizons}")
    
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    np.random.seed(args.seed)
    
    # Train on each horizon
    for horizon in horizons:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training on {horizon}-day horizon")
        logger.info(f"{'='*60}\n")
        
        # Adjust lookback for this horizon
        args.lookback_days = min(horizon, 30)
        
        # Create environment
        env = create_environment(args)
        
        # Create agent
        agent = TradingPPOAgent(
            env=env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            use_attention=args.use_attention,
            verbose=1
        )
        
        # Adjust timesteps based on horizon
        timesteps_for_horizon = args.total_timesteps // len(horizons)
        
        # Create callback
        callback = TradingCallback(
            check_freq=args.save_freq,
            save_path=f"{args.model_dir}/horizon_{horizon}",
            verbose=1
        )
        
        # Train
        logger.info(f"Training for {timesteps_for_horizon} timesteps")
        agent.train(total_timesteps=timesteps_for_horizon, callback=callback)
        
        # Save model for this horizon
        model_path = f"{args.model_dir}/ppo_agent_{args.ticker}_h{horizon}.zip"
        agent.save(model_path)
        
        logger.info(f"Horizon {horizon} training completed!")
        
        # Evaluate
        evaluate_agent(agent, env, n_episodes=3)
        
        env.close()
    
    logger.info("\nMulti-horizon training completed!")


def evaluate_agent(agent, env, n_episodes: int = 5):
    """
    Evaluate trained agent
    
    Args:
        agent: Trained agent
        env: Trading environment
        n_episodes: Number of episodes to evaluate
    """
    logger.info(f"Evaluating agent for {n_episodes} episodes")
    
    episode_rewards = []
    episode_returns = []
    episode_sharpes = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, _states = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        episode_rewards.append(episode_reward)
        episode_returns.append(info['total_return'])
        
        # Calculate Sharpe ratio for episode
        if len(env.returns_history) > 1:
            sharpe = np.mean(env.returns_history) / (np.std(env.returns_history) + 1e-8)
            episode_sharpes.append(sharpe * np.sqrt(252))  # Annualized
        
        logger.info(f"Episode {episode + 1}: Reward={episode_reward:.2f}, Return={info['total_return']*100:.2f}%, Trades={info['num_trades']}")
    
    logger.info(f"\nEvaluation Results:")
    logger.info(f"  Mean Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    logger.info(f"  Mean Return: {np.mean(episode_returns)*100:.2f}% ± {np.std(episode_returns)*100:.2f}%")
    if episode_sharpes:
        logger.info(f"  Mean Sharpe Ratio: {np.mean(episode_sharpes):.2f}")


def main():
    """Main training function"""
    args = parse_args()
    
    logger.info("="*60)
    logger.info("AlphaAgent Training")
    logger.info("="*60)
    logger.info(f"Ticker: {args.ticker}")
    logger.info(f"Reward Type: {args.reward_type}")
    logger.info(f"Total Timesteps: {args.total_timesteps}")
    logger.info(f"Use GAF: {args.use_gaf}")
    logger.info(f"Use Attention: {args.use_attention}")
    logger.info("="*60)
    
    if args.multi_horizon:
        train_multi_horizon(args)
    else:
        train_single_horizon(args)
    
    logger.info("\n✓ Training completed successfully!")


if __name__ == "__main__":
    main()

