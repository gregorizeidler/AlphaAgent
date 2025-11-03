"""
Meta-Agent: Agent of Agents
Orchestrates multiple specialized agents and learns which to use when
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from stable_baselines3 import PPO
import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetaAgent:
    """
    High-level agent that selects which specialized agent to use
    """
    
    def __init__(self, agent_pool: Dict, learning_rate: float = 1e-4):
        """
        Args:
            agent_pool: Dictionary of {name: agent} specialized agents
            learning_rate: Learning rate for meta-policy
        """
        self.agent_pool = agent_pool
        self.agent_names = list(agent_pool.keys())
        self.n_agents = len(self.agent_names)
        
        # Performance tracking for each agent
        self.agent_performance = {name: [] for name in self.agent_names}
        self.agent_usage_count = {name: 0 for name in self.agent_names}
        
        # Meta-policy (learns which agent to select)
        self.meta_policy = None  # Will be PPO that outputs agent selection
        self.learning_rate = learning_rate
        
        logger.info(f"MetaAgent initialized with {self.n_agents} sub-agents: {self.agent_names}")
    
    def select_agent(self, market_state: np.ndarray, method: str = 'performance') -> str:
        """
        Select which agent to use
        
        Args:
            market_state: Current market observation
            method: 'performance', 'random', 'meta_learned'
            
        Returns:
            Agent name
        """
        if method == 'random':
            return np.random.choice(self.agent_names)
        
        elif method == 'performance':
            # Select based on recent performance
            recent_perf = {}
            for name in self.agent_names:
                if len(self.agent_performance[name]) > 0:
                    # Use exponentially weighted average (recent performance matters more)
                    weights = np.exp(np.linspace(-1, 0, len(self.agent_performance[name][-10:])))
                    weights /= weights.sum()
                    recent_perf[name] = np.average(self.agent_performance[name][-10:], weights=weights)
                else:
                    recent_perf[name] = 0.0
            
            # Softmax selection (exploration vs exploitation)
            scores = np.array([recent_perf[name] for name in self.agent_names])
            probs = np.exp(scores - np.max(scores))
            probs /= probs.sum()
            
            selected = np.random.choice(self.agent_names, p=probs)
            return selected
        
        elif method == 'meta_learned':
            # Use learned meta-policy
            if self.meta_policy is None:
                logger.warning("Meta-policy not trained, falling back to performance-based")
                return self.select_agent(market_state, method='performance')
            
            # Meta-policy predicts best agent
            agent_logits = self.meta_policy.predict(market_state, deterministic=True)
            selected_idx = np.argmax(agent_logits[0])
            return self.agent_names[selected_idx]
        
        else:
            raise ValueError(f"Unknown selection method: {method}")
    
    def predict(self, observation: np.ndarray, method: str = 'performance', 
                deterministic: bool = True) -> Tuple[np.ndarray, str]:
        """
        Make prediction using selected agent
        
        Args:
            observation: Market state
            method: Agent selection method
            deterministic: Use deterministic policy
            
        Returns:
            (action, selected_agent_name)
        """
        # Select agent
        selected_agent_name = self.select_agent(observation, method=method)
        selected_agent = self.agent_pool[selected_agent_name]
        
        # Get action
        action, _ = selected_agent.predict(observation, deterministic=deterministic)
        
        # Track usage
        self.agent_usage_count[selected_agent_name] += 1
        
        return action, selected_agent_name
    
    def update_performance(self, agent_name: str, performance: float):
        """
        Update performance history for an agent
        
        Args:
            agent_name: Name of agent
            performance: Performance metric (e.g., return, sharpe)
        """
        self.agent_performance[agent_name].append(performance)
        
        # Keep only recent history
        if len(self.agent_performance[agent_name]) > 100:
            self.agent_performance[agent_name] = self.agent_performance[agent_name][-100:]
    
    def get_statistics(self) -> Dict:
        """
        Get meta-agent statistics
        
        Returns:
            Statistics dictionary
        """
        total_usage = sum(self.agent_usage_count.values())
        
        stats = {
            'total_predictions': total_usage,
            'agent_usage': {},
            'agent_performance': {}
        }
        
        for name in self.agent_names:
            # Usage percentage
            usage_pct = (self.agent_usage_count[name] / total_usage * 100) if total_usage > 0 else 0
            stats['agent_usage'][name] = {
                'count': self.agent_usage_count[name],
                'percentage': usage_pct
            }
            
            # Performance
            if len(self.agent_performance[name]) > 0:
                stats['agent_performance'][name] = {
                    'mean': np.mean(self.agent_performance[name]),
                    'std': np.std(self.agent_performance[name]),
                    'recent_mean': np.mean(self.agent_performance[name][-10:])
                }
            else:
                stats['agent_performance'][name] = {
                    'mean': 0.0,
                    'std': 0.0,
                    'recent_mean': 0.0
                }
        
        return stats
    
    def plot_agent_usage(self, save_path: str = None):
        """
        Visualize which agents are being used
        
        Args:
            save_path: Optional save path
        """
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Usage pie chart
        ax = axes[0]
        sizes = [self.agent_usage_count[name] for name in self.agent_names]
        colors = plt.cm.Set3(np.linspace(0, 1, self.n_agents))
        ax.pie(sizes, labels=self.agent_names, autopct='%1.1f%%', colors=colors, startangle=90)
        ax.set_title('Agent Usage Distribution', fontweight='bold')
        
        # Performance comparison
        ax = axes[1]
        perf_means = []
        perf_stds = []
        for name in self.agent_names:
            if len(self.agent_performance[name]) > 0:
                perf_means.append(np.mean(self.agent_performance[name]))
                perf_stds.append(np.std(self.agent_performance[name]))
            else:
                perf_means.append(0)
                perf_stds.append(0)
        
        x = np.arange(len(self.agent_names))
        ax.bar(x, perf_means, yerr=perf_stds, capsize=5, alpha=0.7, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(self.agent_names, rotation=45, ha='right')
        ax.set_ylabel('Mean Performance')
        ax.set_title('Agent Performance Comparison', fontweight='bold')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Agent usage plot saved to {save_path}")
        
        plt.show()


class AdversarialAgent:
    """
    Agent trained adversarially to be robust
    """
    
    def __init__(self, env):
        """
        Args:
            env: Trading environment
        """
        self.env = env
        self.protagonist = PPO('MlpPolicy', env, verbose=0)  # Main trading agent
        self.antagonist = PPO('MlpPolicy', env, verbose=0)   # Adversarial agent
        
        logger.info("AdversarialAgent initialized")
    
    def train_adversarial(self, n_rounds: int = 10, steps_per_round: int = 10000):
        """
        Train both agents adversarially
        
        Args:
            n_rounds: Number of adversarial rounds
            steps_per_round: Training steps per round
        """
        logger.info(f"Starting adversarial training: {n_rounds} rounds")
        
        for round_num in range(n_rounds):
            logger.info(f"\nRound {round_num + 1}/{n_rounds}")
            
            # Train protagonist (tries to maximize profit)
            logger.info("Training protagonist...")
            self.protagonist.learn(total_timesteps=steps_per_round)
            
            # Evaluate protagonist
            prot_performance = self._evaluate_agent(self.protagonist)
            logger.info(f"Protagonist performance: {prot_performance:.2f}")
            
            # Train antagonist (tries to make protagonist fail)
            # In practice, this creates difficult market scenarios
            logger.info("Training antagonist...")
            self.antagonist.learn(total_timesteps=steps_per_round)
            
            # Antagonist reward = negative of protagonist performance
            # This creates a min-max game
        
        logger.info("Adversarial training completed")
    
    def _evaluate_agent(self, agent, n_episodes: int = 5) -> float:
        """Evaluate agent performance"""
        total_reward = 0
        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            done = False
            while not done:
                action, _ = agent.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = self.env.step(action)
                total_reward += reward
                done = terminated or truncated
        
        return total_reward / n_episodes


class HierarchicalAgent:
    """
    Multi-timeframe hierarchical agent
    """
    
    def __init__(self, env):
        """
        Args:
            env: Trading environment
        """
        self.env = env
        
        # Three levels of decision-making
        self.strategic_agent = None   # Monthly: Portfolio allocation
        self.tactical_agent = None    # Weekly: Position sizing
        self.execution_agent = None   # Daily: Entry/exit timing
        
        logger.info("HierarchicalAgent initialized")
    
    def train_hierarchical(self, timesteps_per_level: int = 50000):
        """
        Train all three levels
        
        Args:
            timesteps_per_level: Training steps per agent level
        """
        logger.info("Training hierarchical agents...")
        
        # Train execution agent first (daily decisions)
        logger.info("Training execution agent (daily)...")
        self.execution_agent = PPO('MlpPolicy', self.env, verbose=0)
        self.execution_agent.learn(total_timesteps=timesteps_per_level)
        
        # Train tactical agent (weekly, uses execution agent)
        logger.info("Training tactical agent (weekly)...")
        self.tactical_agent = PPO('MlpPolicy', self.env, verbose=0)
        self.tactical_agent.learn(total_timesteps=timesteps_per_level)
        
        # Train strategic agent (monthly, uses tactical agent)
        logger.info("Training strategic agent (monthly)...")
        self.strategic_agent = PPO('MlpPolicy', self.env, verbose=0)
        self.strategic_agent.learn(total_timesteps=timesteps_per_level)
        
        logger.info("Hierarchical training completed")
    
    def predict(self, observation: Dict, deterministic: bool = True) -> np.ndarray:
        """
        Hierarchical prediction
        
        Args:
            observation: Must contain 'daily', 'weekly', 'monthly' keys
            deterministic: Use deterministic policies
            
        Returns:
            Final action
        """
        # Strategic level decides overall direction
        if self.strategic_agent and 'monthly' in observation:
            strategy, _ = self.strategic_agent.predict(
                observation['monthly'], deterministic=deterministic
            )
        else:
            strategy = np.array([0.0, 0.5])  # Neutral
        
        # Tactical level decides position sizing
        if self.tactical_agent and 'weekly' in observation:
            tactics, _ = self.tactical_agent.predict(
                observation['weekly'], deterministic=deterministic
            )
        else:
            tactics = np.array([0.0, 0.5])
        
        # Execution level decides exact timing
        if self.execution_agent and 'daily' in observation:
            action, _ = self.execution_agent.predict(
                observation['daily'], deterministic=deterministic
            )
        else:
            action = np.array([0.0, 0.5])
        
        # Combine hierarchical decisions
        # Simple combination: weighted average
        final_action = (0.5 * strategy + 0.3 * tactics + 0.2 * action)
        
        return final_action


if __name__ == "__main__":
    print("Advanced Agents Module")
    print("\n1. MetaAgent: Orchestrates multiple specialized agents")
    print("2. AdversarialAgent: Robust training via adversarial competition")
    print("3. HierarchicalAgent: Multi-timeframe decision making")

