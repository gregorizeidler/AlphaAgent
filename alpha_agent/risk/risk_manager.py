"""
Advanced Risk Management System
Protects against catastrophic losses and enforces trading limits
"""

import numpy as np
from typing import Dict, Optional, Tuple
import logging
from datetime import datetime, timedelta
from collections import deque

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskManager:
    """
    Comprehensive risk management system with multiple safety mechanisms
    """
    
    def __init__(self,
                 max_drawdown: float = 0.15,
                 max_daily_loss: float = 0.03,
                 max_position_size: float = 0.30,
                 stop_loss_pct: float = 0.02,
                 max_leverage: float = 1.0,
                 daily_trade_limit: int = 10,
                 cooling_period_hours: int = 24):
        """
        Args:
            max_drawdown: Maximum allowed drawdown (15% default)
            max_daily_loss: Maximum loss per day (3% default)
            max_position_size: Maximum position as fraction of portfolio (30%)
            stop_loss_pct: Stop-loss per trade (2%)
            max_leverage: Maximum leverage allowed (1.0 = no leverage)
            daily_trade_limit: Maximum trades per day
            cooling_period_hours: Hours to wait after max drawdown hit
        """
        self.max_drawdown = max_drawdown
        self.max_daily_loss = max_daily_loss
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct
        self.max_leverage = max_leverage
        self.daily_trade_limit = daily_trade_limit
        self.cooling_period = timedelta(hours=cooling_period_hours)
        
        # State tracking
        self.peak_portfolio_value = 0.0
        self.daily_start_value = 0.0
        self.daily_trades = 0
        self.last_reset_date = None
        self.in_cooling_period = False
        self.cooling_period_start = None
        self.trade_history = deque(maxlen=100)
        self.violation_count = 0
        
        logger.info("RiskManager initialized with strict limits")
    
    def reset_daily_counters(self):
        """Reset daily tracking counters"""
        self.daily_trades = 0
        self.last_reset_date = datetime.now().date()
    
    def check_action_safety(self, 
                           action: np.ndarray,
                           portfolio_state: Dict,
                           current_price: float) -> Tuple[np.ndarray, Dict]:
        """
        Validate and potentially modify action based on risk limits
        
        Args:
            action: Proposed action [position_change, confidence]
            portfolio_state: Current portfolio state
            current_price: Current asset price
            
        Returns:
            (modified_action, info_dict)
        """
        info = {
            'original_action': action.copy(),
            'modifications': [],
            'risk_level': 'normal',
            'warnings': []
        }
        
        # Check if cooling period is active
        if self.in_cooling_period:
            if datetime.now() - self.cooling_period_start < self.cooling_period:
                info['modifications'].append('cooling_period_active')
                info['risk_level'] = 'critical'
                return np.array([0.0, 0.0]), info  # Force hold
            else:
                self.in_cooling_period = False
                logger.info("Cooling period ended, resuming trading")
        
        # Reset daily counters if new day
        current_date = datetime.now().date()
        if self.last_reset_date != current_date:
            self.reset_daily_counters()
            self.daily_start_value = portfolio_state['portfolio_value']
        
        # Check daily trade limit
        if self.daily_trades >= self.daily_trade_limit:
            info['modifications'].append('daily_trade_limit_reached')
            info['warnings'].append(f"Daily trade limit ({self.daily_trade_limit}) reached")
            return np.array([0.0, 0.0]), info
        
        # Update peak portfolio value
        current_value = portfolio_state['portfolio_value']
        if current_value > self.peak_portfolio_value:
            self.peak_portfolio_value = current_value
        
        # Calculate current drawdown
        if self.peak_portfolio_value > 0:
            current_drawdown = (self.peak_portfolio_value - current_value) / self.peak_portfolio_value
        else:
            current_drawdown = 0.0
        
        # CHECK 1: Maximum Drawdown
        if current_drawdown >= self.max_drawdown:
            logger.warning(f"⚠️ MAX DRAWDOWN TRIGGERED: {current_drawdown*100:.2f}%")
            info['modifications'].append('max_drawdown_breach')
            info['risk_level'] = 'critical'
            info['warnings'].append(f"Drawdown {current_drawdown*100:.2f}% exceeds limit")
            
            # Enter cooling period and close all positions
            self.in_cooling_period = True
            self.cooling_period_start = datetime.now()
            self.violation_count += 1
            
            # Force close positions
            if portfolio_state.get('shares_held', 0) != 0:
                return np.array([-1.0, 1.0]), info  # Sell everything
            else:
                return np.array([0.0, 0.0]), info
        
        # CHECK 2: Daily Loss Limit
        if self.daily_start_value > 0:
            daily_loss = (self.daily_start_value - current_value) / self.daily_start_value
            if daily_loss >= self.max_daily_loss:
                logger.warning(f"⚠️ DAILY LOSS LIMIT: {daily_loss*100:.2f}%")
                info['modifications'].append('daily_loss_limit')
                info['risk_level'] = 'high'
                info['warnings'].append(f"Daily loss {daily_loss*100:.2f}% exceeds {self.max_daily_loss*100}%")
                return np.array([0.0, 0.0]), info  # Stop trading for the day
        
        # CHECK 3: Position Size Limit
        position_change = float(action[0])
        current_position_ratio = portfolio_state.get('position_ratio', 0.0)
        target_position = current_position_ratio + position_change * self.max_position_size
        
        # Limit position size
        if abs(target_position) > self.max_position_size:
            original_position = target_position
            target_position = np.clip(target_position, -self.max_position_size, self.max_position_size)
            adjusted_change = (target_position - current_position_ratio) / self.max_position_size
            action[0] = adjusted_change
            info['modifications'].append('position_size_limited')
            info['warnings'].append(f"Position limited from {original_position:.2%} to {target_position:.2%}")
        
        # CHECK 4: Stop-Loss per Position
        if portfolio_state.get('shares_held', 0) != 0:
            entry_price = portfolio_state.get('entry_price', current_price)
            unrealized_pnl = (current_price - entry_price) / entry_price
            
            if unrealized_pnl <= -self.stop_loss_pct:
                logger.warning(f"⚠️ STOP-LOSS TRIGGERED: {unrealized_pnl*100:.2f}%")
                info['modifications'].append('stop_loss_triggered')
                info['risk_level'] = 'high'
                info['warnings'].append(f"Stop-loss at {unrealized_pnl*100:.2f}%")
                return np.array([-1.0, 1.0]), info  # Close position immediately
        
        # CHECK 5: Prevent excessive leverage
        portfolio_value = portfolio_state['portfolio_value']
        position_value = abs(portfolio_state.get('shares_held', 0) * current_price)
        current_leverage = position_value / portfolio_value if portfolio_value > 0 else 0
        
        if current_leverage > self.max_leverage:
            info['modifications'].append('leverage_reduced')
            info['warnings'].append(f"Leverage {current_leverage:.2f}x exceeds limit")
            # Reduce position
            scale_factor = self.max_leverage / current_leverage
            action[0] *= scale_factor
        
        # CHECK 6: Volatility-based position sizing
        volatility = portfolio_state.get('volatility', 0.02)
        if volatility > 0.05:  # High volatility (5%+)
            vol_scale = 0.02 / volatility  # Reduce position in high vol
            action[0] *= vol_scale
            info['modifications'].append('volatility_adjustment')
            info['warnings'].append(f"Position reduced due to high volatility ({volatility*100:.1f}%)")
        
        # Set risk level
        if current_drawdown > self.max_drawdown * 0.75:
            info['risk_level'] = 'high'
        elif current_drawdown > self.max_drawdown * 0.50:
            info['risk_level'] = 'medium'
        
        return action, info
    
    def update_trade_history(self, trade_info: Dict):
        """
        Update trade history for analysis
        
        Args:
            trade_info: Trade execution details
        """
        self.trade_history.append({
            'timestamp': datetime.now(),
            'trade_info': trade_info
        })
        self.daily_trades += 1
    
    def get_risk_metrics(self, portfolio_state: Dict) -> Dict:
        """
        Calculate comprehensive risk metrics
        
        Args:
            portfolio_state: Current portfolio state
            
        Returns:
            Dictionary of risk metrics
        """
        current_value = portfolio_state['portfolio_value']
        
        # Drawdown
        if self.peak_portfolio_value > 0:
            current_drawdown = (self.peak_portfolio_value - current_value) / self.peak_portfolio_value
        else:
            current_drawdown = 0.0
        
        # Daily P&L
        if self.daily_start_value > 0:
            daily_pnl = (current_value - self.daily_start_value) / self.daily_start_value
        else:
            daily_pnl = 0.0
        
        # Position concentration
        position_ratio = portfolio_state.get('position_ratio', 0.0)
        
        # Trade frequency
        recent_trades = len([t for t in self.trade_history 
                           if datetime.now() - t['timestamp'] < timedelta(hours=1)])
        
        return {
            'current_drawdown': current_drawdown,
            'max_drawdown_limit': self.max_drawdown,
            'drawdown_utilization': current_drawdown / self.max_drawdown if self.max_drawdown > 0 else 0,
            'daily_pnl': daily_pnl,
            'daily_loss_limit': self.max_daily_loss,
            'position_concentration': abs(position_ratio),
            'max_position_limit': self.max_position_size,
            'daily_trades': self.daily_trades,
            'daily_trade_limit': self.daily_trade_limit,
            'recent_trade_rate': recent_trades,
            'in_cooling_period': self.in_cooling_period,
            'violation_count': self.violation_count,
            'risk_score': self._calculate_risk_score(current_drawdown, daily_pnl, position_ratio)
        }
    
    def _calculate_risk_score(self, drawdown: float, daily_pnl: float, position_ratio: float) -> float:
        """
        Calculate overall risk score (0-100, higher = more risky)
        
        Returns:
            Risk score
        """
        # Weighted risk components
        drawdown_risk = (drawdown / self.max_drawdown) * 40  # 40% weight
        loss_risk = max(0, (-daily_pnl / self.max_daily_loss)) * 30  # 30% weight
        position_risk = (abs(position_ratio) / self.max_position_size) * 30  # 30% weight
        
        total_risk = min(100, drawdown_risk + loss_risk + position_risk)
        return total_risk


class KellyPositionSizer:
    """
    Dynamic position sizing using Kelly Criterion
    """
    
    def __init__(self, lookback: int = 30, kelly_fraction: float = 0.25):
        """
        Args:
            lookback: Number of trades to analyze
            kelly_fraction: Fraction of Kelly to use (0.25 = quarter Kelly)
        """
        self.lookback = lookback
        self.kelly_fraction = kelly_fraction
        self.trade_results = deque(maxlen=lookback)
    
    def update(self, trade_return: float):
        """Update with new trade result"""
        self.trade_results.append(trade_return)
    
    def calculate_position_size(self) -> float:
        """
        Calculate optimal position size using Kelly Criterion
        
        Returns:
            Position size as fraction of portfolio (0.0 to 1.0)
        """
        if len(self.trade_results) < 10:
            return 0.10  # Conservative until enough data
        
        results = np.array(self.trade_results)
        wins = results[results > 0]
        losses = results[results < 0]
        
        if len(wins) == 0 or len(losses) == 0:
            return 0.10  # Conservative if no wins or losses
        
        # Calculate Kelly parameters
        win_rate = len(wins) / len(results)
        avg_win = np.mean(wins)
        avg_loss = abs(np.mean(losses))
        
        # Kelly formula: f = (p*b - q) / b
        # where p = win rate, q = loss rate, b = avg_win/avg_loss
        if avg_loss > 0:
            b = avg_win / avg_loss
            kelly = (win_rate * b - (1 - win_rate)) / b
        else:
            kelly = 0.10
        
        # Apply fraction and limits
        position_size = max(0.05, min(0.30, kelly * self.kelly_fraction))
        
        return position_size


if __name__ == "__main__":
    # Test risk manager
    risk_manager = RiskManager(
        max_drawdown=0.15,
        max_daily_loss=0.03,
        max_position_size=0.30
    )
    
    # Simulate portfolio state
    portfolio_state = {
        'portfolio_value': 9000,  # Lost $1000 from $10000
        'shares_held': 10,
        'position_ratio': 0.25,
        'entry_price': 100,
        'volatility': 0.02
    }
    
    # Test action
    action = np.array([0.5, 0.8])  # Try to increase position
    current_price = 95  # Price went down
    
    # Check safety
    safe_action, info = risk_manager.check_action_safety(
        action, portfolio_state, current_price
    )
    
    print("Original action:", action)
    print("Safe action:", safe_action)
    print("Modifications:", info['modifications'])
    print("Warnings:", info['warnings'])
    print("Risk level:", info['risk_level'])
    
    # Get risk metrics
    metrics = risk_manager.get_risk_metrics(portfolio_state)
    print("\nRisk Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

