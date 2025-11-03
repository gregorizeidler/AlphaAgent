"""
Multi-Asset Trading Environment
Enables portfolio optimization across multiple assets
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging

from alpha_agent.data import MarketDataFetcher, TechnicalIndicators
from alpha_agent.sentiment import MultiModalSentimentAnalyzer
from alpha_agent.state import CompleteStateRepresentation
from alpha_agent.rewards import SharpeRatioReward
from alpha_agent.risk import RiskManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiAssetTradingEnvironment(gym.Env):
    """
    Trading environment for multiple assets with portfolio optimization
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self,
                 tickers: List[str],
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None,
                 initial_balance: float = 10000.0,
                 transaction_cost: float = 0.001,
                 slippage: float = 0.0005,
                 lookback_days: int = 30,
                 max_concentration: float = 0.40,  # Max 40% in single asset
                 use_risk_manager: bool = True):
        """
        Args:
            tickers: List of stock tickers
            start_date: Start date
            end_date: End date
            initial_balance: Initial cash
            transaction_cost: Transaction cost rate
            slippage: Slippage rate
            lookback_days: Days of history in state
            max_concentration: Maximum position in single asset
            use_risk_manager: Enable risk management
        """
        super().__init__()
        
        self.tickers = tickers
        self.n_assets = len(tickers)
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.lookback_days = lookback_days
        self.max_concentration = max_concentration
        
        # Fetch data for all assets
        logger.info(f"Initializing multi-asset environment for {self.n_assets} assets")
        self.data_fetchers = {}
        self.ohlcv_data = {}
        self.fundamentals = {}
        self.sentiment_data = {}
        
        for ticker in tickers:
            fetcher = MarketDataFetcher(ticker, lookback_days=365)
            ohlcv, fundamentals, news = fetcher.fetch_all_data()
            
            if ohlcv.empty:
                raise ValueError(f"No data for {ticker}")
            
            # Add technical indicators
            ohlcv = TechnicalIndicators.add_technical_indicators(ohlcv)
            
            # Sentiment analysis
            sentiment_analyzer = MultiModalSentimentAnalyzer()
            sentiment = sentiment_analyzer.analyze_comprehensive(news, ticker)
            
            self.data_fetchers[ticker] = fetcher
            self.ohlcv_data[ticker] = ohlcv
            self.fundamentals[ticker] = fundamentals
            self.sentiment_data[ticker] = sentiment
        
        # Find common date range
        self._align_data()
        
        # State representation
        self.state_repr = CompleteStateRepresentation(
            gaf_size=30,
            lookback_days=lookback_days,
            use_gaf=False,
            flatten_output=True
        )
        
        # Observation space: state for each asset + portfolio info
        single_asset_dim = self.state_repr.get_state_shape()[0]
        portfolio_dim = self.n_assets + 3  # weights + cash_ratio + total_value + diversity_score
        obs_dim = single_asset_dim * self.n_assets + portfolio_dim
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Action space: target weight for each asset (sums to <=1)
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.n_assets,),
            dtype=np.float32
        )
        
        # Reward function
        self.reward_calculator = SharpeRatioReward(window_size=30, scale_factor=10.0)
        
        # Risk manager
        if use_risk_manager:
            self.risk_manager = RiskManager(
                max_drawdown=0.15,
                max_daily_loss=0.03,
                max_position_size=max_concentration
            )
        else:
            self.risk_manager = None
        
        # Trading state
        self.current_step = 0
        self.max_steps = self.common_length - lookback_days - 1
        
        self.balance = initial_balance
        self.shares_held = {ticker: 0.0 for ticker in tickers}
        self.portfolio_value = initial_balance
        self.previous_portfolio_value = initial_balance
        
        # History
        self.returns_history = []
        self.portfolio_history = [initial_balance]
        self.weights_history = []
        self.trades_history = []
        
        logger.info(f"Multi-asset environment initialized: {self.max_steps} trading days")
    
    def _align_data(self):
        """Align data to common date range"""
        # Find common dates
        all_indices = [df.index for df in self.ohlcv_data.values()]
        common_dates = all_indices[0]
        for idx in all_indices[1:]:
            common_dates = common_dates.intersection(idx)
        
        # Filter to common dates
        for ticker in self.tickers:
            self.ohlcv_data[ticker] = self.ohlcv_data[ticker].loc[common_dates]
        
        self.common_length = len(common_dates)
        logger.info(f"Aligned data: {self.common_length} common trading days")
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset environment"""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = {ticker: 0.0 for ticker in self.tickers}
        self.portfolio_value = self.initial_balance
        self.previous_portfolio_value = self.initial_balance
        
        self.returns_history = []
        self.portfolio_history = [self.initial_balance]
        self.weights_history = []
        self.trades_history = []
        
        if self.risk_manager:
            self.risk_manager.peak_portfolio_value = self.initial_balance
        
        self.reward_calculator.reset()
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: np.ndarray):
        """Execute one step"""
        # Normalize action to sum to <=1
        target_weights = action / (action.sum() + 1e-8)
        target_weights = np.clip(target_weights, 0, self.max_concentration)
        
        # Rebalance portfolio
        trades_info = self._rebalance_portfolio(target_weights)
        
        # Move to next step
        self.current_step += 1
        
        # Update portfolio value
        self.portfolio_value = self._calculate_portfolio_value()
        
        # Calculate return
        step_return = (self.portfolio_value - self.previous_portfolio_value) / (self.previous_portfolio_value + 1e-8)
        self.returns_history.append(step_return)
        self.portfolio_history.append(self.portfolio_value)
        self.previous_portfolio_value = self.portfolio_value
        
        # Record weights
        self.weights_history.append(target_weights.copy())
        
        # Calculate reward
        reward = self.reward_calculator.calculate_incremental(step_return)
        
        # Check termination
        terminated = self.portfolio_value <= self.initial_balance * 0.5  # 50% loss
        truncated = self.current_step >= self.max_steps - 1
        
        observation = self._get_observation()
        info = self._get_info()
        info['trades'] = trades_info
        
        return observation, float(reward), terminated, truncated, info
    
    def _rebalance_portfolio(self, target_weights: np.ndarray) -> List[Dict]:
        """
        Rebalance portfolio to target weights
        
        Args:
            target_weights: Target weight for each asset
            
        Returns:
            List of trade information
        """
        trades = []
        current_prices = self._get_current_prices()
        
        # Calculate current weights
        current_values = {ticker: self.shares_held[ticker] * current_prices[ticker] 
                         for ticker in self.tickers}
        total_value = sum(current_values.values()) + self.balance
        current_weights = {ticker: current_values[ticker] / total_value 
                          for ticker in self.tickers}
        
        # Execute trades
        for i, ticker in enumerate(self.tickers):
            target_value = total_value * target_weights[i]
            current_value = current_values[ticker]
            trade_value = target_value - current_value
            
            if abs(trade_value) < 10:  # Skip small trades
                continue
            
            # Calculate shares to trade
            price = current_prices[ticker]
            shares_to_trade = trade_value / price
            
            # Apply costs
            execution_price = price * (1 + self.slippage if shares_to_trade > 0 else 1 - self.slippage)
            cost = abs(trade_value) * self.transaction_cost
            
            # Execute
            if shares_to_trade > 0:  # Buy
                if self.balance >= abs(trade_value) + cost:
                    self.shares_held[ticker] += shares_to_trade
                    self.balance -= (abs(trade_value) + cost)
                    trades.append({'ticker': ticker, 'shares': shares_to_trade, 'value': trade_value, 'cost': cost})
            else:  # Sell
                self.shares_held[ticker] += shares_to_trade
                self.balance += (abs(trade_value) - cost)
                trades.append({'ticker': ticker, 'shares': shares_to_trade, 'value': trade_value, 'cost': cost})
        
        return trades
    
    def _get_current_prices(self) -> Dict[str, float]:
        """Get current prices for all assets"""
        current_idx = self.lookback_days + self.current_step
        prices = {}
        for ticker in self.tickers:
            prices[ticker] = self.ohlcv_data[ticker].iloc[current_idx]['Close']
        return prices
    
    def _calculate_portfolio_value(self) -> float:
        """Calculate total portfolio value"""
        current_prices = self._get_current_prices()
        holdings_value = sum(self.shares_held[ticker] * current_prices[ticker] 
                            for ticker in self.tickers)
        return self.balance + holdings_value
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        current_idx = self.lookback_days + self.current_step
        
        # Get state for each asset
        asset_states = []
        for ticker in self.tickers:
            historical_ohlcv = self.ohlcv_data[ticker].iloc[:current_idx]
            state = self.state_repr.create_state(
                ohlcv_df=historical_ohlcv,
                fundamentals=self.fundamentals[ticker],
                sentiment_data=self.sentiment_data[ticker],
                technical_indicators=historical_ohlcv
            )
            asset_states.append(state)
        
        # Portfolio info
        current_prices = self._get_current_prices()
        total_value = self._calculate_portfolio_value()
        
        weights = np.array([
            (self.shares_held[ticker] * current_prices[ticker]) / total_value
            for ticker in self.tickers
        ])
        
        cash_ratio = self.balance / total_value
        value_change = (total_value - self.initial_balance) / self.initial_balance
        
        # Diversity score (Herfindahl index)
        diversity = 1 - np.sum(weights ** 2)
        
        portfolio_info = np.concatenate([
            weights,
            [cash_ratio, value_change, diversity]
        ])
        
        # Combine all
        observation = np.concatenate(asset_states + [portfolio_info]).astype(np.float32)
        
        return observation
    
    def _get_info(self) -> Dict[str, Any]:
        """Get info dictionary"""
        current_prices = self._get_current_prices()
        total_value = self._calculate_portfolio_value()
        
        weights = {ticker: (self.shares_held[ticker] * current_prices[ticker]) / total_value
                  for ticker in self.tickers}
        
        return {
            'step': self.current_step,
            'portfolio_value': total_value,
            'balance': self.balance,
            'weights': weights,
            'total_return': (total_value - self.initial_balance) / self.initial_balance,
            'num_trades': len(self.trades_history)
        }
    
    def render(self):
        """Render environment"""
        info = self._get_info()
        print(f"\n--- Step {info['step']} ---")
        print(f"Portfolio Value: ${info['portfolio_value']:.2f}")
        print(f"Cash: ${info['balance']:.2f}")
        print("Weights:")
        for ticker, weight in info['weights'].items():
            print(f"  {ticker}: {weight*100:.1f}%")
        print(f"Total Return: {info['total_return']*100:.2f}%")
    
    def close(self):
        """Clean up"""
        pass


if __name__ == "__main__":
    # Test multi-asset environment
    tickers = ["AAPL", "GOOGL", "MSFT"]
    
    env = MultiAssetTradingEnvironment(
        tickers=tickers,
        initial_balance=10000.0,
        use_risk_manager=True
    )
    
    print(f"Multi-Asset Environment:")
    print(f"  Assets: {env.n_assets}")
    print(f"  Observation Space: {env.observation_space.shape}")
    print(f"  Action Space: {env.action_space.shape}")
    print(f"  Trading Days: {env.max_steps}")
    
    # Test episode
    obs, info = env.reset()
    print(f"\nInitial state shape: {obs.shape}")
    
    for i in range(3):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"\nStep {i+1}:")
        print(f"  Portfolio: ${info['portfolio_value']:.2f}")
        print(f"  Return: {info['total_return']*100:.2f}%")
        
        if terminated or truncated:
            break
    
    env.close()

