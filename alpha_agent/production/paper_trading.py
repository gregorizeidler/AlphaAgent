"""
Paper Trading Integration
Live trading simulation without real money
"""

import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PaperTradingBroker:
    """
    Simulates real trading using live market data
    Integrates with Alpaca Paper Trading API or similar
    """
    
    def __init__(self, initial_balance: float = 10000.0, api_key: Optional[str] = None):
        """
        Args:
            initial_balance: Starting capital
            api_key: Optional API key for broker integration
        """
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions = {}
        self.orders = []
        self.trade_history = []
        self.api_key = api_key
        
        # Mock live data (in production, use real API)
        self.live_prices = {}
        
        logger.info(f"PaperTradingBroker initialized with ${initial_balance:,.2f}")
    
    def get_live_price(self, ticker: str) -> float:
        """
        Get current live price (mock or real API)
        
        Args:
            ticker: Stock ticker
            
        Returns:
            Current price
        """
        # In production, fetch from real API
        # For now, simulate with random walk
        if ticker not in self.live_prices:
            self.live_prices[ticker] = 150.0  # Initial price
        
        # Simulate price movement
        self.live_prices[ticker] *= (1 + np.random.randn() * 0.001)
        
        return self.live_prices[ticker]
    
    def place_order(
        self,
        ticker: str,
        quantity: int,
        side: str,  # 'buy' or 'sell'
        order_type: str = 'market'
    ) -> Dict:
        """
        Place a paper trading order
        
        Args:
            ticker: Stock ticker
            quantity: Number of shares
            side: 'buy' or 'sell'
            order_type: 'market' or 'limit'
            
        Returns:
            Order confirmation
        """
        timestamp = datetime.now()
        price = self.get_live_price(ticker)
        
        # Calculate order value
        order_value = quantity * price
        
        # Simulate transaction cost (0.1%)
        transaction_cost = order_value * 0.001
        
        if side == 'buy':
            # Check if enough cash
            total_cost = order_value + transaction_cost
            if total_cost > self.balance:
                logger.warning(f"Insufficient funds: need ${total_cost:.2f}, have ${self.balance:.2f}")
                return {'status': 'rejected', 'reason': 'insufficient_funds'}
            
            # Execute buy
            self.balance -= total_cost
            
            if ticker in self.positions:
                self.positions[ticker]['shares'] += quantity
                self.positions[ticker]['value'] = self.positions[ticker]['shares'] * price
            else:
                self.positions[ticker] = {
                    'shares': quantity,
                    'avg_price': price,
                    'value': order_value
                }
            
            logger.info(f"BUY {quantity} {ticker} @ ${price:.2f}")
        
        elif side == 'sell':
            # Check if enough shares
            if ticker not in self.positions or self.positions[ticker]['shares'] < quantity:
                logger.warning(f"Insufficient shares to sell: {ticker}")
                return {'status': 'rejected', 'reason': 'insufficient_shares'}
            
            # Execute sell
            self.balance += (order_value - transaction_cost)
            self.positions[ticker]['shares'] -= quantity
            
            if self.positions[ticker]['shares'] == 0:
                del self.positions[ticker]
            else:
                self.positions[ticker]['value'] = self.positions[ticker]['shares'] * price
            
            logger.info(f"SELL {quantity} {ticker} @ ${price:.2f}")
        
        # Record order
        order = {
            'timestamp': timestamp,
            'ticker': ticker,
            'quantity': quantity,
            'side': side,
            'price': price,
            'value': order_value,
            'cost': transaction_cost,
            'status': 'filled'
        }
        
        self.orders.append(order)
        self.trade_history.append(order)
        
        return order
    
    def get_portfolio_value(self) -> float:
        """
        Calculate total portfolio value
        
        Returns:
            Total value
        """
        holdings_value = 0.0
        for ticker, position in self.positions.items():
            current_price = self.get_live_price(ticker)
            holdings_value += position['shares'] * current_price
        
        total_value = self.balance + holdings_value
        return total_value
    
    def get_portfolio_summary(self) -> Dict:
        """
        Get portfolio summary
        
        Returns:
            Summary dictionary
        """
        total_value = self.get_portfolio_value()
        total_return = ((total_value - self.initial_balance) / self.initial_balance) * 100
        
        summary = {
            'timestamp': datetime.now(),
            'total_value': total_value,
            'cash_balance': self.balance,
            'holdings_value': total_value - self.balance,
            'initial_balance': self.initial_balance,
            'total_return_pct': total_return,
            'total_return_dollar': total_value - self.initial_balance,
            'num_positions': len(self.positions),
            'num_trades': len(self.trade_history),
            'positions': dict(self.positions)
        }
        
        return summary
    
    def export_trades(self, filepath: str):
        """
        Export trade history to CSV
        
        Args:
            filepath: Path to save CSV
        """
        if not self.trade_history:
            logger.warning("No trades to export")
            return
        
        df = pd.DataFrame(self.trade_history)
        df.to_csv(filepath, index=False)
        logger.info(f"Trade history exported to {filepath}")


class LiveTradingSession:
    """
    Manages a live paper trading session with an agent
    """
    
    def __init__(
        self,
        agent,
        broker: PaperTradingBroker,
        tickers: list,
        update_interval: int = 60
    ):
        """
        Args:
            agent: Trained trading agent
            broker: Paper trading broker
            tickers: List of tickers to trade
            update_interval: Seconds between updates
        """
        self.agent = agent
        self.broker = broker
        self.tickers = tickers
        self.update_interval = update_interval
        
        self.is_running = False
        self.session_log = []
        
        logger.info(f"LiveTradingSession initialized for {tickers}")
    
    def start(self, duration_hours: Optional[int] = None):
        """
        Start live trading session
        
        Args:
            duration_hours: Optional duration (None = run indefinitely)
        """
        logger.info("Starting live trading session...")
        self.is_running = True
        
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=duration_hours) if duration_hours else None
        
        iteration = 0
        
        try:
            while self.is_running:
                iteration += 1
                current_time = datetime.now()
                
                # Check if duration reached
                if end_time and current_time >= end_time:
                    logger.info("Session duration reached")
                    break
                
                logger.info(f"\n{'='*60}")
                logger.info(f"Iteration {iteration} | {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
                logger.info(f"{'='*60}")
                
                # Get portfolio state
                portfolio = self.broker.get_portfolio_summary()
                logger.info(f"Portfolio Value: ${portfolio['total_value']:,.2f} ({portfolio['total_return_pct']:+.2f}%)")
                
                # Get market data (mock for now)
                market_data = self._get_market_data()
                
                # Agent decides
                action, _ = self.agent.predict(market_data, deterministic=True)
                
                # Execute trades based on action
                self._execute_action(action)
                
                # Log session data
                self.session_log.append({
                    'timestamp': current_time,
                    'portfolio_value': portfolio['total_value'],
                    'action': float(action[0]),
                    'num_positions': portfolio['num_positions']
                })
                
                # Wait for next iteration
                time.sleep(self.update_interval)
                
        except KeyboardInterrupt:
            logger.info("\nSession interrupted by user")
        finally:
            self.stop()
    
    def stop(self):
        """Stop trading session"""
        self.is_running = False
        
        # Final summary
        final_portfolio = self.broker.get_portfolio_summary()
        
        logger.info("\n" + "="*60)
        logger.info("SESSION SUMMARY")
        logger.info("="*60)
        logger.info(f"Final Portfolio Value: ${final_portfolio['total_value']:,.2f}")
        logger.info(f"Total Return: {final_portfolio['total_return_pct']:+.2f}% (${final_portfolio['total_return_dollar']:+,.2f})")
        logger.info(f"Number of Trades: {final_portfolio['num_trades']}")
        logger.info(f"Number of Positions: {final_portfolio['num_positions']}")
        logger.info("="*60)
        
        # Export data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.broker.export_trades(f"./paper_trading_history_{timestamp}.csv")
        
        # Export session log
        session_df = pd.DataFrame(self.session_log)
        session_df.to_csv(f"./session_log_{timestamp}.csv", index=False)
        logger.info(f"Session log exported")
    
    def _get_market_data(self) -> np.ndarray:
        """
        Get current market data for agent
        (Mock implementation - replace with real data pipeline)
        
        Returns:
            Market state observation
        """
        # In production, this would fetch real-time data
        # For now, return mock observation
        return np.random.randn(210)
    
    def _execute_action(self, action: np.ndarray):
        """
        Execute agent's action
        
        Args:
            action: Action from agent [position_change, confidence]
        """
        position_change = float(action[0])
        
        # Simple execution: trade first ticker in list
        ticker = self.tickers[0]
        
        # Determine trade size (simple logic)
        portfolio_value = self.broker.get_portfolio_value()
        trade_size = int(abs(position_change) * portfolio_value / self.broker.get_live_price(ticker) / 10)
        
        if trade_size == 0:
            logger.info("No action (trade size too small)")
            return
        
        if position_change > 0.1:
            # Buy signal
            self.broker.place_order(ticker, trade_size, 'buy')
        elif position_change < -0.1:
            # Sell signal
            if ticker in self.broker.positions:
                shares_owned = self.broker.positions[ticker]['shares']
                trade_size = min(trade_size, shares_owned)
                if trade_size > 0:
                    self.broker.place_order(ticker, trade_size, 'sell')
        else:
            logger.info("Hold (no strong signal)")


if __name__ == "__main__":
    print("Paper Trading Module")
    print("\nUsage:")
    print("1. Create PaperTradingBroker")
    print("2. Create LiveTradingSession with your agent")
    print("3. Start session and let it run")
    print("\nNo real money at risk!")

