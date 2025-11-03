"""
Advanced reward functions for trading: Sharpe Ratio, Sortino Ratio, etc.
"""

import numpy as np
from typing import List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RewardCalculator:
    """
    Base class for reward calculation
    """
    
    def __init__(self, window_size: int = 30, risk_free_rate: float = 0.02):
        """
        Args:
            window_size: Number of periods for rolling calculations
            risk_free_rate: Annual risk-free rate (default 2%)
        """
        self.window_size = window_size
        self.risk_free_rate = risk_free_rate / 252  # Daily risk-free rate
        self.returns_history = []
        
    def reset(self):
        """Reset the reward calculator"""
        self.returns_history = []
    
    def calculate(self, returns: List[float]) -> float:
        """
        Calculate reward - to be implemented by subclasses
        
        Args:
            returns: List of returns
            
        Returns:
            Reward value
        """
        raise NotImplementedError


class SimpleReturnReward(RewardCalculator):
    """
    Simple return-based reward (baseline)
    """
    
    def calculate(self, returns: List[float]) -> float:
        """
        Calculate simple return reward
        
        Args:
            returns: List of returns
            
        Returns:
            Latest return
        """
        if not returns:
            return 0.0
        
        return returns[-1] * 100  # Scale up for learning


class SharpeRatioReward(RewardCalculator):
    """
    Sharpe Ratio: (mean_return - risk_free_rate) / std_return
    Measures risk-adjusted returns
    """
    
    def __init__(self, window_size: int = 30, risk_free_rate: float = 0.02, 
                 scale_factor: float = 10.0):
        """
        Args:
            window_size: Rolling window for Sharpe calculation
            risk_free_rate: Annual risk-free rate
            scale_factor: Scaling factor for reward magnitude
        """
        super().__init__(window_size, risk_free_rate)
        self.scale_factor = scale_factor
    
    def calculate(self, returns: List[float]) -> float:
        """
        Calculate Sharpe Ratio reward
        
        Args:
            returns: List of returns
            
        Returns:
            Sharpe ratio (scaled)
        """
        if len(returns) < 2:
            return 0.0
        
        # Use recent window
        recent_returns = returns[-self.window_size:] if len(returns) > self.window_size else returns
        
        mean_return = np.mean(recent_returns)
        std_return = np.std(recent_returns)
        
        if std_return < 1e-6:
            # If no volatility, return based on sign of returns
            return self.scale_factor * np.sign(mean_return)
        
        # Sharpe ratio
        sharpe = (mean_return - self.risk_free_rate) / std_return
        
        # Scale for RL learning
        reward = sharpe * self.scale_factor
        
        return float(reward)
    
    def calculate_incremental(self, new_return: float) -> float:
        """
        Calculate incremental Sharpe ratio with new return
        
        Args:
            new_return: New return to add
            
        Returns:
            Sharpe ratio reward
        """
        self.returns_history.append(new_return)
        return self.calculate(self.returns_history)


class SortinoRatioReward(RewardCalculator):
    """
    Sortino Ratio: (mean_return - risk_free_rate) / downside_deviation
    Focuses on downside risk (punishes negative volatility more)
    """
    
    def __init__(self, window_size: int = 30, risk_free_rate: float = 0.02,
                 scale_factor: float = 10.0, target_return: float = 0.0):
        """
        Args:
            window_size: Rolling window for calculation
            risk_free_rate: Annual risk-free rate
            scale_factor: Scaling factor for reward
            target_return: Minimum acceptable return (MAR)
        """
        super().__init__(window_size, risk_free_rate)
        self.scale_factor = scale_factor
        self.target_return = target_return
    
    def calculate(self, returns: List[float]) -> float:
        """
        Calculate Sortino Ratio reward
        
        Args:
            returns: List of returns
            
        Returns:
            Sortino ratio (scaled)
        """
        if len(returns) < 2:
            return 0.0
        
        # Use recent window
        recent_returns = returns[-self.window_size:] if len(returns) > self.window_size else returns
        
        mean_return = np.mean(recent_returns)
        
        # Downside deviation (only negative returns)
        downside_returns = [r - self.target_return for r in recent_returns if r < self.target_return]
        
        if len(downside_returns) == 0:
            # No downside, high reward
            return self.scale_factor * 2.0
        
        downside_std = np.sqrt(np.mean(np.square(downside_returns)))
        
        if downside_std < 1e-6:
            return self.scale_factor * np.sign(mean_return)
        
        # Sortino ratio
        sortino = (mean_return - self.risk_free_rate) / downside_std
        
        # Scale for RL learning
        reward = sortino * self.scale_factor
        
        return float(reward)
    
    def calculate_incremental(self, new_return: float) -> float:
        """
        Calculate incremental Sortino ratio
        
        Args:
            new_return: New return to add
            
        Returns:
            Sortino ratio reward
        """
        self.returns_history.append(new_return)
        return self.calculate(self.returns_history)


class CompositeReward(RewardCalculator):
    """
    Composite reward combining multiple objectives
    """
    
    def __init__(self, 
                 window_size: int = 30,
                 risk_free_rate: float = 0.02,
                 return_weight: float = 0.3,
                 sharpe_weight: float = 0.4,
                 sortino_weight: float = 0.3):
        """
        Args:
            window_size: Rolling window
            risk_free_rate: Risk-free rate
            return_weight: Weight for simple returns
            sharpe_weight: Weight for Sharpe ratio
            sortino_weight: Weight for Sortino ratio
        """
        super().__init__(window_size, risk_free_rate)
        
        self.return_weight = return_weight
        self.sharpe_weight = sharpe_weight
        self.sortino_weight = sortino_weight
        
        # Initialize sub-calculators
        self.simple_reward = SimpleReturnReward(window_size, risk_free_rate)
        self.sharpe_reward = SharpeRatioReward(window_size, risk_free_rate, scale_factor=1.0)
        self.sortino_reward = SortinoRatioReward(window_size, risk_free_rate, scale_factor=1.0)
        
        # Normalize weights
        total_weight = return_weight + sharpe_weight + sortino_weight
        self.return_weight /= total_weight
        self.sharpe_weight /= total_weight
        self.sortino_weight /= total_weight
    
    def calculate(self, returns: List[float]) -> float:
        """
        Calculate composite reward
        
        Args:
            returns: List of returns
            
        Returns:
            Weighted combination of rewards
        """
        if not returns:
            return 0.0
        
        # Calculate individual rewards
        simple_r = self.simple_reward.calculate(returns)
        sharpe_r = self.sharpe_reward.calculate(returns)
        sortino_r = self.sortino_reward.calculate(returns)
        
        # Combine
        composite = (
            self.return_weight * simple_r +
            self.sharpe_weight * sharpe_r +
            self.sortino_weight * sortino_r
        )
        
        return float(composite)
    
    def calculate_incremental(self, new_return: float) -> float:
        """
        Calculate incremental composite reward
        
        Args:
            new_return: New return to add
            
        Returns:
            Composite reward
        """
        self.returns_history.append(new_return)
        return self.calculate(self.returns_history)


class RiskAdjustedPnLReward(RewardCalculator):
    """
    Risk-adjusted P&L reward with drawdown penalty
    """
    
    def __init__(self,
                 window_size: int = 30,
                 risk_free_rate: float = 0.02,
                 drawdown_penalty: float = 2.0,
                 volatility_penalty: float = 0.5):
        """
        Args:
            window_size: Rolling window
            risk_free_rate: Risk-free rate
            drawdown_penalty: Penalty multiplier for drawdowns
            volatility_penalty: Penalty for high volatility
        """
        super().__init__(window_size, risk_free_rate)
        self.drawdown_penalty = drawdown_penalty
        self.volatility_penalty = volatility_penalty
        self.peak_value = 0.0
    
    def calculate(self, returns: List[float], portfolio_values: Optional[List[float]] = None) -> float:
        """
        Calculate risk-adjusted P&L reward
        
        Args:
            returns: List of returns
            portfolio_values: Optional list of portfolio values for drawdown calculation
            
        Returns:
            Risk-adjusted reward
        """
        if not returns:
            return 0.0
        
        recent_returns = returns[-self.window_size:] if len(returns) > self.window_size else returns
        
        # Base reward: cumulative return
        cumulative_return = np.sum(recent_returns)
        
        # Volatility penalty
        volatility = np.std(recent_returns) if len(recent_returns) > 1 else 0.0
        volatility_cost = self.volatility_penalty * volatility
        
        # Drawdown penalty
        drawdown_cost = 0.0
        if portfolio_values and len(portfolio_values) > 0:
            recent_values = portfolio_values[-self.window_size:] if len(portfolio_values) > self.window_size else portfolio_values
            peak = np.maximum.accumulate(recent_values)
            drawdown = (peak - recent_values) / (peak + 1e-8)
            max_drawdown = np.max(drawdown)
            drawdown_cost = self.drawdown_penalty * max_drawdown
        
        # Composite reward
        reward = cumulative_return - volatility_cost - drawdown_cost
        
        return float(reward * 100)  # Scale up


class TransactionCostAwareReward(RewardCalculator):
    """
    Reward that accounts for transaction costs
    """
    
    def __init__(self,
                 base_reward_calculator: RewardCalculator,
                 transaction_cost_rate: float = 0.001,
                 penalty_weight: float = 1.0):
        """
        Args:
            base_reward_calculator: Base reward calculator to wrap
            transaction_cost_rate: Transaction cost rate (e.g., 0.001 = 0.1%)
            penalty_weight: Weight for transaction cost penalty
        """
        super().__init__()
        self.base_calculator = base_reward_calculator
        self.transaction_cost_rate = transaction_cost_rate
        self.penalty_weight = penalty_weight
    
    def calculate(self, returns: List[float], transaction_costs: List[float]) -> float:
        """
        Calculate reward with transaction cost penalty
        
        Args:
            returns: List of returns
            transaction_costs: List of transaction costs incurred
            
        Returns:
            Adjusted reward
        """
        # Base reward
        base_reward = self.base_calculator.calculate(returns)
        
        # Transaction cost penalty
        if transaction_costs:
            total_tc = np.sum(transaction_costs[-self.base_calculator.window_size:])
            tc_penalty = self.penalty_weight * total_tc * 100
        else:
            tc_penalty = 0.0
        
        return base_reward - tc_penalty


if __name__ == "__main__":
    # Test reward functions
    np.random.seed(42)
    
    # Simulate returns
    returns = np.random.normal(0.001, 0.02, 100).tolist()  # Daily returns
    
    print("Testing Reward Functions\n")
    
    # Simple returns
    simple = SimpleReturnReward()
    print(f"Simple Return Reward: {simple.calculate(returns):.4f}")
    
    # Sharpe ratio
    sharpe = SharpeRatioReward(window_size=30)
    print(f"Sharpe Ratio Reward: {sharpe.calculate(returns):.4f}")
    
    # Sortino ratio
    sortino = SortinoRatioReward(window_size=30)
    print(f"Sortino Ratio Reward: {sortino.calculate(returns):.4f}")
    
    # Composite
    composite = CompositeReward(window_size=30)
    print(f"Composite Reward: {composite.calculate(returns):.4f}")
    
    # Risk-adjusted P&L
    portfolio_values = [10000 * (1 + r) for r in np.cumsum(returns)]
    risk_adj = RiskAdjustedPnLReward(window_size=30)
    print(f"Risk-Adjusted P&L Reward: {risk_adj.calculate(returns, portfolio_values):.4f}")

