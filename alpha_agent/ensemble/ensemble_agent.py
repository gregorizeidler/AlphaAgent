"""
Ensemble Agent System
Combines multiple specialized agents for different market conditions
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from stable_baselines3 import PPO

from alpha_agent.market_regime import RegimeDetector, MarketRegime
from alpha_agent.agents import TradingPPOAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnsembleAgent:
    """
    Ensemble of specialized agents for different market regimes
    """
    
    def __init__(self, env, learning_rate: float = 3e-4):
        """
        Args:
            env: Trading environment
            learning_rate: Learning rate for agents
        """
        self.env = env
        self.learning_rate = learning_rate
        self.regime_detector = RegimeDetector()
        
        # Create specialized agents
        self.agents = {
            MarketRegime.BULL: None,  # Will be trained for bull markets
            MarketRegime.BEAR: None,  # Will be trained for bear markets
            MarketRegime.SIDEWAYS: None,  # Will be trained for sideways
            MarketRegime.HIGH_VOLATILITY: None,  # Will be trained for high vol
        }
        
        # Performance tracking
        self.agent_performance = {regime: [] for regime in self.agents.keys()}
        
        logger.info("EnsembleAgent initialized with 4 specialized agents")
    
    def train_specialized_agents(self,
                                 historical_data: Dict[MarketRegime, List],
                                 timesteps_per_agent: int = 50000):
        """
        Train specialized agents on regime-specific data
        
        Args:
            historical_data: Dict mapping regime to list of data samples
            timesteps_per_agent: Training timesteps per agent
        """
        for regime, data_samples in historical_data.items():
            if not data_samples:
                logger.warning(f"No data for {regime.value}, skipping")
                continue
            
            logger.info(f"Training {regime.value} agent on {len(data_samples)} samples")
            
            # Create agent for this regime
            agent = TradingPPOAgent(
                env=self.env,
                learning_rate=self.learning_rate,
                use_attention=False,
                verbose=0
            )
            
            # Train on regime-specific data
            # In practice, you'd create a custom env with filtered data
            agent.train(total_timesteps=timesteps_per_agent)
            
            self.agents[regime] = agent
            logger.info(f"{regime.value} agent training completed")
    
    def predict(self, observation, ohlcv_data, deterministic: bool = True):
        """
        Predict action using ensemble
        
        Args:
            observation: Current state
            ohlcv_data: Historical price data for regime detection
            deterministic: Use deterministic policy
            
        Returns:
            action, states
        """
        # Detect current regime
        regime, confidence = self.regime_detector.detect_regime(ohlcv_data)
        
        # Get predictions from all agents
        predictions = {}
        for reg, agent in self.agents.items():
            if agent is not None:
                action, _ = agent.predict(observation, deterministic=deterministic)
                predictions[reg] = action
        
        # Weighted combination based on regime confidence
        if regime in predictions and confidence > 0.6:
            # High confidence: use specialized agent
            return predictions[regime], regime
        else:
            # Low confidence: average predictions
            actions = list(predictions.values())
            avg_action = np.mean(actions, axis=0)
            return avg_action, MarketRegime.UNCERTAIN
    
    def save(self, path_prefix: str):
        """Save all agents"""
        for regime, agent in self.agents.items():
            if agent is not None:
                agent.save(f"{path_prefix}_{regime.value}")
        logger.info(f"Ensemble agents saved with prefix: {path_prefix}")
    
    def load(self, path_prefix: str):
        """Load all agents"""
        for regime in self.agents.keys():
            try:
                agent = TradingPPOAgent(env=self.env)
                agent.load(f"{path_prefix}_{regime.value}")
                self.agents[regime] = agent
                logger.info(f"Loaded {regime.value} agent")
            except Exception as e:
                logger.warning(f"Could not load {regime.value} agent: {e}")


class WeightedEnsemble:
    """
    Weighted ensemble using past performance
    """
    
    def __init__(self, agents: List[TradingPPOAgent], window: int = 20):
        """
        Args:
            agents: List of trading agents
            window: Performance tracking window
        """
        self.agents = agents
        self.window = window
        self.performance_history = [[] for _ in agents]
        self.weights = np.ones(len(agents)) / len(agents)  # Start equal
    
    def predict(self, observation, deterministic: bool = True):
        """
        Predict using weighted combination
        
        Args:
            observation: Current state
            deterministic: Use deterministic policy
            
        Returns:
            Weighted action
        """
        actions = []
        for agent in self.agents:
            action, _ = agent.predict(observation, deterministic=deterministic)
            actions.append(action)
        
        # Weighted average
        weighted_action = np.average(actions, axis=0, weights=self.weights)
        return weighted_action, None
    
    def update_weights(self, agent_idx: int, performance: float):
        """
        Update agent weights based on performance
        
        Args:
            agent_idx: Index of agent to update
            performance: Performance metric (e.g., return)
        """
        self.performance_history[agent_idx].append(performance)
        
        # Keep only recent history
        if len(self.performance_history[agent_idx]) > self.window:
            self.performance_history[agent_idx].pop(0)
        
        # Recalculate weights based on recent performance
        recent_perf = []
        for hist in self.performance_history:
            if hist:
                recent_perf.append(np.mean(hist))
            else:
                recent_perf.append(0.0)
        
        # Softmax weighting (better performers get more weight)
        exp_perf = np.exp(np.array(recent_perf) - np.max(recent_perf))
        self.weights = exp_perf / np.sum(exp_perf)
        
        logger.debug(f"Updated weights: {self.weights}")


if __name__ == "__main__":
    from alpha_agent.environment import TradingEnvironment
    
    # Create environment
    env = TradingEnvironment(
        ticker="AAPL",
        initial_balance=10000.0,
        reward_type='sharpe',
        use_gaf=False
    )
    
    # Create ensemble
    ensemble = EnsembleAgent(env)
    
    print("Ensemble agent created with 4 specialized sub-agents:")
    for regime in ensemble.agents.keys():
        print(f"  - {regime.value} specialist")
    
    print("\nTo train: ensemble.train_specialized_agents(data, timesteps=50000)")
    print("To predict: action = ensemble.predict(obs, ohlcv_data)")
    
    env.close()

