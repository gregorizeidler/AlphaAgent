"""
Gymnasium Trading Environment with Transaction Costs, Slippage, and Market Impact
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any
import logging

from alpha_agent.data import MarketDataFetcher, TechnicalIndicators
from alpha_agent.sentiment import MultiModalSentimentAnalyzer
from alpha_agent.state import CompleteStateRepresentation
from alpha_agent.rewards import SharpeRatioReward, SortinoRatioReward, CompositeReward

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TradingEnvironment(gym.Env):
    """
    Advanced Trading Environment for DRL
    
    Features:
    - Realistic transaction costs, slippage, market impact
    - Multi-modal state space (GAF, fundamentals, sentiment)
    - Sophisticated reward functions (Sharpe/Sortino)
    - Continuous action space (position sizing)
    """
    
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(self,
                 ticker: str,
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None,
                 initial_balance: float = 10000.0,
                 transaction_cost: float = 0.001,  # 0.1%
                 slippage: float = 0.0005,  # 0.05%
                 market_impact_factor: float = 0.0001,  # 0.01%
                 lookback_days: int = 30,
                 reward_type: str = 'sharpe',  # 'sharpe', 'sortino', 'composite'
                 use_gaf: bool = True,
                 max_position: float = 1.0,
                 enable_shorting: bool = False,
                 render_mode: Optional[str] = None):
        """
        Args:
            ticker: Stock ticker symbol
            start_date: Start date for data
            end_date: End date for data
            initial_balance: Initial cash balance
            transaction_cost: Transaction cost rate
            slippage: Slippage rate
            market_impact_factor: Market impact factor (based on order size)
            lookback_days: Days of price history in state
            reward_type: Type of reward function
            use_gaf: Whether to use GAF transformation
            max_position: Maximum position as fraction of portfolio
            enable_shorting: Whether to allow short positions
            render_mode: Rendering mode
        """
        super().__init__()
        
        self.ticker = ticker
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.market_impact_factor = market_impact_factor
        self.lookback_days = lookback_days
        self.use_gaf = use_gaf
        self.max_position = max_position
        self.enable_shorting = enable_shorting
        self.render_mode = render_mode
        
        # Fetch data
        logger.info(f"Initializing environment for {ticker}")
        self.data_fetcher = MarketDataFetcher(ticker, lookback_days=365)
        self.ohlcv_data, self.fundamentals, self.news = self.data_fetcher.fetch_all_data()
        
        if self.ohlcv_data.empty:
            raise ValueError(f"No data available for {ticker}")
        
        # Add technical indicators
        self.ohlcv_data = TechnicalIndicators.add_technical_indicators(self.ohlcv_data)
        
        # Sentiment analysis
        self.sentiment_analyzer = MultiModalSentimentAnalyzer()
        self.sentiment_data = self.sentiment_analyzer.analyze_comprehensive(self.news, ticker)
        
        # State representation
        self.state_repr = CompleteStateRepresentation(
            gaf_size=30,
            lookback_days=lookback_days,
            use_gaf=use_gaf,
            flatten_output=True
        )
        
        # Define observation space (state space)
        state_shape = self.state_repr.get_state_shape()
        # Add current position and balance to state
        additional_features = 3  # position, balance_ratio, portfolio_value_change
        obs_dim = state_shape[0] + additional_features
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Define action space
        # Action: [position_change, confidence]
        # position_change: -1 (full sell) to +1 (full buy)
        # confidence: 0 to 1 (not used yet, but for future advanced strategies)
        if enable_shorting:
            action_low = np.array([-2.0, 0.0], dtype=np.float32)  # Can go -100% (short) to +100% (long)
            action_high = np.array([2.0, 1.0], dtype=np.float32)
        else:
            action_low = np.array([-1.0, 0.0], dtype=np.float32)  # 0% to 100% long only
            action_high = np.array([1.0, 1.0], dtype=np.float32)
        
        self.action_space = spaces.Box(
            low=action_low,
            high=action_high,
            dtype=np.float32
        )
        
        # Reward function
        if reward_type == 'sharpe':
            self.reward_calculator = SharpeRatioReward(window_size=30, scale_factor=10.0)
        elif reward_type == 'sortino':
            self.reward_calculator = SortinoRatioReward(window_size=30, scale_factor=10.0)
        elif reward_type == 'composite':
            self.reward_calculator = CompositeReward(window_size=30)
        else:
            raise ValueError(f"Unknown reward type: {reward_type}")
        
        # Trading state
        self.current_step = 0
        self.max_steps = len(self.ohlcv_data) - lookback_days - 1
        
        self.balance = initial_balance
        self.shares_held = 0.0
        self.portfolio_value = initial_balance
        self.previous_portfolio_value = initial_balance
        
        # History tracking
        self.returns_history = []
        self.portfolio_history = [initial_balance]
        self.trades_history = []
        self.transaction_costs_paid = []
        
        logger.info(f"Environment initialized: {self.max_steps} trading days available")
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment
        
        Returns:
            Initial observation and info dict
        """
        super().reset(seed=seed)
        
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0.0
        self.portfolio_value = self.initial_balance
        self.previous_portfolio_value = self.initial_balance
        
        self.returns_history = []
        self.portfolio_history = [self.initial_balance]
        self.trades_history = []
        self.transaction_costs_paid = []
        
        self.reward_calculator.reset()
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment
        
        Args:
            action: [position_change, confidence]
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Parse action
        position_change = float(action[0])
        confidence = float(action[1]) if len(action) > 1 else 1.0
        
        # Clip to valid range
        position_change = np.clip(position_change, -1.0, 1.0)
        
        # Get current price
        current_idx = self.lookback_days + self.current_step
        current_price = self.ohlcv_data.iloc[current_idx]['Close']
        
        # Calculate desired position
        current_position_value = self.shares_held * current_price
        current_position_ratio = current_position_value / (self.portfolio_value + 1e-8)
        
        # Target position (as ratio of portfolio value)
        target_position_ratio = np.clip(
            current_position_ratio + position_change * self.max_position,
            -self.max_position if self.enable_shorting else 0.0,
            self.max_position
        )
        
        # Execute trade
        trade_info = self._execute_trade(current_price, target_position_ratio)
        
        # Move to next step
        self.current_step += 1
        next_idx = self.lookback_days + self.current_step
        next_price = self.ohlcv_data.iloc[next_idx]['Close']
        
        # Update portfolio value
        self.portfolio_value = self.balance + self.shares_held * next_price
        
        # Calculate return
        step_return = (self.portfolio_value - self.previous_portfolio_value) / (self.previous_portfolio_value + 1e-8)
        self.returns_history.append(step_return)
        self.portfolio_history.append(self.portfolio_value)
        self.previous_portfolio_value = self.portfolio_value
        
        # Calculate reward
        reward = self.reward_calculator.calculate_incremental(step_return)
        
        # Check if episode is done
        terminated = self.portfolio_value <= 0  # Bankruptcy
        truncated = self.current_step >= self.max_steps - 1  # End of data
        
        # Get next observation
        observation = self._get_observation()
        info = self._get_info()
        info['trade'] = trade_info
        
        return observation, float(reward), terminated, truncated, info
    
    def _execute_trade(self, current_price: float, target_position_ratio: float) -> Dict:
        """
        Execute a trade with transaction costs, slippage, and market impact
        
        Args:
            current_price: Current asset price
            target_position_ratio: Target position as ratio of portfolio value
            
        Returns:
            Trade information dictionary
        """
        # Calculate target shares
        target_value = self.portfolio_value * target_position_ratio
        target_shares = target_value / current_price
        
        shares_to_trade = target_shares - self.shares_held
        
        if abs(shares_to_trade) < 1e-6:
            # No trade
            return {
                'shares_traded': 0.0,
                'trade_value': 0.0,
                'transaction_cost': 0.0,
                'slippage_cost': 0.0,
                'market_impact': 0.0,
                'execution_price': current_price
            }
        
        # Calculate market impact (proportional to order size)
        order_size_ratio = abs(shares_to_trade * current_price) / (self.portfolio_value + 1e-8)
        market_impact = self.market_impact_factor * order_size_ratio
        
        # Calculate execution price with slippage and market impact
        if shares_to_trade > 0:  # Buying
            execution_price = current_price * (1 + self.slippage + market_impact)
        else:  # Selling
            execution_price = current_price * (1 - self.slippage - market_impact)
        
        # Trade value
        trade_value = abs(shares_to_trade * execution_price)
        
        # Transaction cost
        transaction_cost = trade_value * self.transaction_cost
        
        # Check if we have enough balance for buying
        if shares_to_trade > 0:
            total_cost = trade_value + transaction_cost
            if total_cost > self.balance:
                # Can't afford full trade, scale down
                affordable_value = self.balance * 0.99  # Leave small buffer
                shares_to_trade = affordable_value / (execution_price * (1 + self.transaction_cost))
                trade_value = shares_to_trade * execution_price
                transaction_cost = trade_value * self.transaction_cost
        
        # Execute trade
        if shares_to_trade > 0:  # Buying
            self.balance -= (trade_value + transaction_cost)
            self.shares_held += shares_to_trade
        else:  # Selling
            self.balance += (trade_value - transaction_cost)
            self.shares_held += shares_to_trade  # shares_to_trade is negative
        
        # Track costs
        self.transaction_costs_paid.append(transaction_cost)
        
        # Record trade
        trade_info = {
            'shares_traded': float(shares_to_trade),
            'trade_value': float(trade_value),
            'transaction_cost': float(transaction_cost),
            'slippage_cost': float(trade_value * self.slippage),
            'market_impact': float(trade_value * market_impact),
            'execution_price': float(execution_price),
        }
        
        self.trades_history.append(trade_info)
        
        return trade_info
    
    def _get_observation(self) -> np.ndarray:
        """
        Get current observation (state)
        
        Returns:
            State vector
        """
        # Get data up to current step
        current_idx = self.lookback_days + self.current_step
        historical_ohlcv = self.ohlcv_data.iloc[:current_idx]
        
        # Create state representation
        state = self.state_repr.create_state(
            ohlcv_df=historical_ohlcv,
            fundamentals=self.fundamentals,
            sentiment_data=self.sentiment_data,
            technical_indicators=historical_ohlcv
        )
        
        # Add current position and balance info
        current_price = self.ohlcv_data.iloc[current_idx]['Close']
        position_ratio = (self.shares_held * current_price) / (self.portfolio_value + 1e-8)
        balance_ratio = self.balance / (self.portfolio_value + 1e-8)
        portfolio_change = (self.portfolio_value - self.initial_balance) / self.initial_balance
        
        additional_features = np.array([
            position_ratio,
            balance_ratio,
            portfolio_change
        ], dtype=np.float32)
        
        observation = np.concatenate([state, additional_features]).astype(np.float32)
        
        return observation
    
    def _get_info(self) -> Dict[str, Any]:
        """
        Get additional info
        
        Returns:
            Info dictionary
        """
        current_idx = self.lookback_days + self.current_step
        current_price = self.ohlcv_data.iloc[current_idx]['Close']
        
        info = {
            'step': self.current_step,
            'portfolio_value': self.portfolio_value,
            'balance': self.balance,
            'shares_held': self.shares_held,
            'current_price': current_price,
            'total_return': (self.portfolio_value - self.initial_balance) / self.initial_balance,
            'num_trades': len(self.trades_history),
            'total_transaction_costs': sum(self.transaction_costs_paid),
        }
        
        return info
    
    def render(self):
        """
        Render the environment
        """
        if self.render_mode == 'human':
            info = self._get_info()
            print(f"\n--- Step {info['step']} ---")
            print(f"Portfolio Value: ${info['portfolio_value']:.2f}")
            print(f"Cash Balance: ${info['balance']:.2f}")
            print(f"Shares Held: {info['shares_held']:.2f}")
            print(f"Current Price: ${info['current_price']:.2f}")
            print(f"Total Return: {info['total_return']*100:.2f}%")
            print(f"Number of Trades: {info['num_trades']}")
    
    def close(self):
        """
        Clean up resources
        """
        pass


if __name__ == "__main__":
    # Test the environment
    env = TradingEnvironment(
        ticker="AAPL",
        initial_balance=10000.0,
        reward_type='sharpe',
        use_gaf=False,  # Use False for faster testing
    )
    
    print(f"Observation Space: {env.observation_space}")
    print(f"Action Space: {env.action_space}")
    
    # Test episode
    obs, info = env.reset()
    print(f"\nInitial Observation Shape: {obs.shape}")
    print(f"Initial Info: {info}")
    
    # Run a few steps with random actions
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"\nStep {i+1}:")
        print(f"  Action: {action}")
        print(f"  Reward: {reward:.4f}")
        print(f"  Portfolio Value: ${info['portfolio_value']:.2f}")
        print(f"  Total Return: {info['total_return']*100:.2f}%")
        
        if terminated or truncated:
            break
    
    env.close()
    print("\nEnvironment test completed!")

