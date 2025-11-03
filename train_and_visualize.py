"""
Training Visualization Script
Trains the agent and generates comprehensive training plots
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from stable_baselines3.common.callbacks import BaseCallback
import argparse

print("="*70)
print("🎓 AlphaAgent - TRAINING WITH VISUALIZATION")
print("="*70)
print()

from alpha_agent.data.data_fetcher import MarketDataFetcher
from alpha_agent.environment.trading_env import TradingEnvironment
from alpha_agent.agents.ppo_agent import TradingPPOAgent


class TrainingPlotter(BaseCallback):
    """
    Custom callback to track and plot training metrics
    """
    
    def __init__(self, verbose=0, plot_freq=1000):
        super().__init__(verbose)
        self.plot_freq = plot_freq
        
        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
        self.learning_rates = []
        self.policy_losses = []
        self.value_losses = []
        self.entropy_losses = []
        self.timesteps = []
        
        # Episode tracking
        self.current_episode_reward = 0
        self.current_episode_length = 0
    
    def _on_step(self) -> bool:
        # Track episode progress
        self.current_episode_reward += self.locals['rewards'][0]
        self.current_episode_length += 1
        
        # Check if episode ended
        if self.locals['dones'][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.timesteps.append(self.num_timesteps)
            
            self.current_episode_reward = 0
            self.current_episode_length = 0
            
            if len(self.episode_rewards) % 10 == 0:
                print(f"  Episode {len(self.episode_rewards)}: "
                      f"Reward={self.episode_rewards[-1]:.2f}, "
                      f"Length={self.episode_lengths[-1]}")
        
        return True
    
    def _on_training_end(self) -> None:
        print("\n✓ Training completed! Generating plots...")


def plot_training_progress(plotter, save_dir='./plots'):
    """Generate comprehensive training plots"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot 1: Episode Rewards Over Time
    print("\nGenerating training plots...")
    print("  1. Episode rewards...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Rewards
    ax = axes[0, 0]
    if len(plotter.episode_rewards) > 0:
        rewards = np.array(plotter.episode_rewards)
        episodes = np.arange(len(rewards))
        
        ax.plot(episodes, rewards, alpha=0.3, color='gray', label='Raw')
        
        # Moving average
        if len(rewards) >= 10:
            window = min(20, len(rewards) // 5)
            ma = pd.Series(rewards).rolling(window=window).mean()
            ax.plot(episodes, ma, linewidth=2, color='#2ecc71', label=f'MA({window})')
        
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('Episode', fontsize=11)
        ax.set_ylabel('Total Reward', fontsize=11)
        ax.set_title('Episode Rewards', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Episode lengths
    ax = axes[0, 1]
    if len(plotter.episode_lengths) > 0:
        lengths = np.array(plotter.episode_lengths)
        episodes = np.arange(len(lengths))
        
        ax.plot(episodes, lengths, alpha=0.4, color='gray')
        
        if len(lengths) >= 10:
            window = min(20, len(lengths) // 5)
            ma = pd.Series(lengths).rolling(window=window).mean()
            ax.plot(episodes, ma, linewidth=2, color='#3498db', label=f'MA({window})')
        
        ax.set_xlabel('Episode', fontsize=11)
        ax.set_ylabel('Episode Length', fontsize=11)
        ax.set_title('Episode Lengths', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Cumulative rewards
    ax = axes[1, 0]
    if len(plotter.episode_rewards) > 0:
        cumulative = np.cumsum(plotter.episode_rewards)
        episodes = np.arange(len(cumulative))
        
        ax.plot(episodes, cumulative, linewidth=2, color='#9b59b6')
        ax.fill_between(episodes, 0, cumulative, alpha=0.3, color='#9b59b6')
        ax.set_xlabel('Episode', fontsize=11)
        ax.set_ylabel('Cumulative Reward', fontsize=11)
        ax.set_title('Cumulative Rewards', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # Reward distribution
    ax = axes[1, 1]
    if len(plotter.episode_rewards) > 0:
        rewards = np.array(plotter.episode_rewards)
        ax.hist(rewards, bins=30, alpha=0.7, color='#e74c3c', edgecolor='black')
        ax.axvline(x=np.mean(rewards), color='green', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(rewards):.2f}')
        ax.axvline(x=np.median(rewards), color='blue', linestyle='--', 
                   linewidth=2, label=f'Median: {np.median(rewards):.2f}')
        ax.set_xlabel('Reward', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Reward Distribution', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('📊 Training Progress', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/11_training_progress.png', dpi=300, bbox_inches='tight')
    print(f"    ✓ Saved: {save_dir}/11_training_progress.png")
    plt.close()
    
    # Plot 2: Learning Curve Analysis
    print("  2. Learning curves...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    if len(plotter.episode_rewards) > 0:
        rewards = np.array(plotter.episode_rewards)
        episodes = np.arange(len(rewards))
        
        # Plot 1: Reward improvement
        ax = axes[0, 0]
        if len(rewards) >= 50:
            # Split into quartiles
            q1_end = len(rewards) // 4
            q2_end = len(rewards) // 2
            q3_end = 3 * len(rewards) // 4
            
            quartiles = [
                ('Q1 (Early)', rewards[:q1_end], '#e74c3c'),
                ('Q2', rewards[q1_end:q2_end], '#f39c12'),
                ('Q3', rewards[q2_end:q3_end], '#3498db'),
                ('Q4 (Late)', rewards[q3_end:], '#2ecc71')
            ]
            
            positions = [1, 2, 3, 4]
            data = [q[1] for q in quartiles]
            labels = [q[0] for q in quartiles]
            colors = [q[2] for q in quartiles]
            
            bp = ax.boxplot(data, positions=positions, labels=labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_ylabel('Reward', fontsize=11)
            ax.set_title('Reward Improvement Across Training', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Best reward over time
        ax = axes[0, 1]
        best_rewards = np.maximum.accumulate(rewards)
        ax.plot(episodes, best_rewards, linewidth=2, color='#2ecc71')
        ax.fill_between(episodes, 0, best_rewards, alpha=0.3, color='#2ecc71')
        ax.scatter([np.argmax(rewards)], [np.max(rewards)], 
                  s=200, color='yellow', edgecolors='black', linewidth=2, zorder=5)
        ax.annotate(f'Best: {np.max(rewards):.2f}',
                   xy=(np.argmax(rewards), np.max(rewards)),
                   xytext=(20, 20), textcoords='offset points',
                   fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        ax.set_xlabel('Episode', fontsize=11)
        ax.set_ylabel('Best Reward So Far', fontsize=11)
        ax.set_title('Best Reward Progress', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Success rate (positive rewards)
        ax = axes[1, 0]
        if len(rewards) >= 20:
            window = 20
            success_rate = []
            for i in range(window, len(rewards)):
                window_rewards = rewards[i-window:i]
                success_rate.append(np.sum(window_rewards > 0) / window * 100)
            
            ax.plot(range(window, len(rewards)), success_rate, 
                   linewidth=2, color='#3498db')
            ax.fill_between(range(window, len(rewards)), 0, success_rate, 
                           alpha=0.3, color='#3498db')
            ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50%')
            ax.set_xlabel('Episode', fontsize=11)
            ax.set_ylabel('Success Rate (%)', fontsize=11)
            ax.set_title(f'Success Rate (Rolling {window} episodes)', 
                        fontsize=13, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot 4: Reward variance
        ax = axes[1, 1]
        if len(rewards) >= 20:
            window = 20
            variance = []
            for i in range(window, len(rewards)):
                window_rewards = rewards[i-window:i]
                variance.append(np.std(window_rewards))
            
            ax.plot(range(window, len(rewards)), variance, 
                   linewidth=2, color='#9b59b6')
            ax.fill_between(range(window, len(rewards)), 0, variance, 
                           alpha=0.3, color='#9b59b6')
            ax.set_xlabel('Episode', fontsize=11)
            ax.set_ylabel('Std Dev', fontsize=11)
            ax.set_title(f'Reward Stability (Rolling {window} episodes)', 
                        fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
    
    plt.suptitle('📈 Learning Curve Analysis', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/12_learning_curves.png', dpi=300, bbox_inches='tight')
    print(f"    ✓ Saved: {save_dir}/12_learning_curves.png")
    plt.close()
    
    # Plot 3: Training Statistics Summary
    print("  3. Training statistics...")
    
    if len(plotter.episode_rewards) > 0:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        rewards = np.array(plotter.episode_rewards)
        
        stats_text = f"""
        TRAINING STATISTICS SUMMARY
        {'='*50}
        
        Total Episodes:           {len(rewards)}
        Total Timesteps:          {plotter.timesteps[-1] if plotter.timesteps else 0:,}
        
        REWARDS:
        ├─ Mean:                  {np.mean(rewards):>10.2f}
        ├─ Median:                {np.median(rewards):>10.2f}
        ├─ Std Dev:               {np.std(rewards):>10.2f}
        ├─ Min:                   {np.min(rewards):>10.2f}
        ├─ Max:                   {np.max(rewards):>10.2f}
        └─ Final (last 10 avg):   {np.mean(rewards[-10:]):>10.2f}
        
        PERFORMANCE:
        ├─ Successful Episodes:   {np.sum(rewards > 0)} ({np.sum(rewards > 0)/len(rewards)*100:.1f}%)
        ├─ Failed Episodes:       {np.sum(rewards <= 0)} ({np.sum(rewards <= 0)/len(rewards)*100:.1f}%)
        └─ Improvement:           {((np.mean(rewards[-20:]) - np.mean(rewards[:20])) / abs(np.mean(rewards[:20])) * 100):.1f}%
        
        EPISODE LENGTH:
        ├─ Mean:                  {np.mean(plotter.episode_lengths):.1f}
        ├─ Min:                   {np.min(plotter.episode_lengths)}
        └─ Max:                   {np.max(plotter.episode_lengths)}
        
        {'='*50}
        """
        
        ax.text(0.1, 0.5, stats_text, transform=ax.transAxes,
               fontsize=12, verticalalignment='center',
               family='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.axis('off')
        
        plt.title('📋 Training Statistics', fontsize=18, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/13_training_stats.png', dpi=300, bbox_inches='tight')
        print(f"    ✓ Saved: {save_dir}/13_training_stats.png")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train agent with visualization')
    parser.add_argument('--ticker', type=str, default='AAPL', help='Stock ticker')
    parser.add_argument('--timesteps', type=int, default=50000, help='Training timesteps')
    parser.add_argument('--save-freq', type=int, default=10000, help='Save frequency')
    args = parser.parse_args()
    
    print(f"Ticker: {args.ticker}")
    print(f"Training timesteps: {args.timesteps:,}")
    print()
    
    # Fetch data
    print("📊 Fetching data...")
    data_fetcher = MarketDataFetcher(ticker=args.ticker)
    ohlcv = data_fetcher.fetch_ohlcv("2022-01-01", "2024-01-01")
    
    if ohlcv.empty:
        print(f"❌ No data for {args.ticker}")
        return
    
    print(f"  ✓ Loaded {len(ohlcv)} days of data")
    
    # Create environment
    print("\n🏗️  Creating environment...")
    env = TradingEnvironment(
        ticker=args.ticker,
        start_date="2022-01-01",
        end_date="2024-01-01",
        initial_balance=10000.0,
        use_gaf=False  # Faster training
    )
    print("  ✓ Environment ready")
    
    # Create agent
    print("\n🤖 Creating agent...")
    agent = TradingPPOAgent(env, learning_rate=3e-4, verbose=1)
    print("  ✓ Agent initialized")
    
    # Create callback
    print("\n🎓 Starting training...")
    print("="*70)
    callback = TrainingPlotter(verbose=1)
    
    # Train
    agent.train(total_timesteps=args.timesteps, callback=callback)
    
    print("="*70)
    print("✓ Training completed!")
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"./models/ppo_agent_{args.ticker}_{timestamp}.zip"
    os.makedirs("./models", exist_ok=True)
    agent.save(model_path)
    print(f"\n💾 Model saved: {model_path}")
    
    # Generate plots
    print("\n📊 Generating training visualizations...")
    plot_training_progress(callback)
    
    print("\n" + "="*70)
    print("✅ TRAINING AND VISUALIZATION COMPLETE!")
    print("="*70)
    print(f"\n📁 Model: {model_path}")
    print(f"📁 Plots: ./plots/11_training_progress.png")
    print(f"         ./plots/12_learning_curves.png")
    print(f"         ./plots/13_training_stats.png")
    print("\n🎉 All done!")


if __name__ == "__main__":
    main()

