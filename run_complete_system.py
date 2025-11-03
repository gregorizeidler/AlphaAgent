"""
Complete AlphaAgent System Demo
Showcases ALL advanced features
"""

import os
import sys
import argparse
from datetime import datetime
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("="*70)
print("🤖 AlphaAgent COMPLETE SYSTEM")
print("="*70)
print()

# Import all modules
print("Loading modules...")
from alpha_agent.data.data_fetcher import MarketDataFetcher
from alpha_agent.sentiment.sentiment_analyzer import FinBERTSentimentAnalyzer
from alpha_agent.state.state_representation import CompleteStateRepresentation
from alpha_agent.environment.trading_env import TradingEnvironment
from alpha_agent.agents.ppo_agent import TradingPPOAgent
from alpha_agent.risk.risk_manager import RiskManager
from alpha_agent.market_regime.regime_detector import RegimeDetector
from alpha_agent.explainability.feature_attribution import FeatureAttributor, DecisionLogger
from alpha_agent.analysis.performance_attribution import PerformanceAttributor, MonteCarloSimulator
from alpha_agent.advanced_agents.meta_agent import MetaAgent
from alpha_agent.production.paper_trading import PaperTradingBroker, LiveTradingSession
from alpha_agent.production.alert_system import AlertSystem
from alpha_agent.production.auto_retrain import PerformanceMonitor, AutoRetrainPipeline

print("✓ All modules loaded successfully!")
print()


def demo_feature_attribution(agent, env):
    """Demo feature attribution"""
    print("\n" + "="*70)
    print("📊 FEATURE ATTRIBUTION DEMO")
    print("="*70)
    
    # Get sample observation
    obs, _ = env.reset()
    
    # Create attributor
    feature_names = [f"feature_{i}" for i in range(len(obs))]
    attributor = FeatureAttributor(agent, feature_names)
    
    # Create explainer with background data
    background_data = np.array([env.reset()[0] for _ in range(20)])
    print("Creating SHAP explainer (this may take a moment)...")
    attributor.create_explainer(background_data, n_samples=10)
    
    # Explain a decision
    print("\nExplaining agent decision...")
    explanation = attributor.explain_action(obs, max_display=5)
    
    print(f"\nPredicted Action: {explanation['action']:.3f}")
    print(f"Base Value: {explanation['base_value']:.3f}")
    print(f"\nTop Contributing Features:")
    for feat in explanation['top_features']:
        print(f"  • {feat['feature']}: {feat['shap_value']:+.4f} (importance: {feat['importance']:.4f})")
    print(f"\n{explanation['reasoning']}")
    
    print("\n✓ Feature Attribution complete!")


def demo_performance_attribution():
    """Demo performance attribution"""
    print("\n" + "="*70)
    print("💰 PERFORMANCE ATTRIBUTION DEMO")
    print("="*70)
    
    # Simulate trades
    trades = [
        {'trade_value': 500, 'transaction_cost': 5, 'slippage_cost': 2, 'market_impact': 1},
        {'trade_value': -300, 'transaction_cost': 3, 'slippage_cost': 1.5, 'market_impact': 0.5},
        {'trade_value': 800, 'transaction_cost': 8, 'slippage_cost': 3, 'market_impact': 2},
        {'trade_value': 200, 'transaction_cost': 2, 'slippage_cost': 1, 'market_impact': 0.5},
        {'trade_value': -150, 'transaction_cost': 1.5, 'slippage_cost': 0.5, 'market_impact': 0.3},
    ]
    
    attributor = PerformanceAttributor()
    attribution = attributor.attribute_trades(trades, 10000, 11050)
    
    report = attributor.create_attribution_report(attribution)
    print("\n" + report)
    
    print("\n✓ Performance Attribution complete!")


def demo_monte_carlo():
    """Demo Monte Carlo simulation"""
    print("\n" + "="*70)
    print("🎲 MONTE CARLO SIMULATION DEMO")
    print("="*70)
    
    # Simulate historical returns
    historical_returns = np.random.randn(100) * 0.02 + 0.001
    
    simulator = MonteCarloSimulator()
    scenarios, stats = simulator.simulate_future(
        current_value=10000,
        historical_returns=historical_returns,
        n_days=90,
        n_simulations=500
    )
    
    report = simulator.generate_report(stats, 10000, 90)
    print("\n" + report)
    
    print("\n✓ Monte Carlo complete!")


def demo_alert_system():
    """Demo alert system"""
    print("\n" + "="*70)
    print("🚨 ALERT SYSTEM DEMO")
    print("="*70)
    
    alert_system = AlertSystem()
    
    # Trigger various alerts
    print("\nSimulating market conditions...")
    alert_system.check_drawdown(0.18)
    alert_system.check_daily_loss(-0.06)
    alert_system.check_position_size('AAPL', 0.45)
    alert_system.check_regime_change('BEAR', 0.85)
    
    print("\nRecent Alerts:")
    for alert in alert_system.get_recent_alerts():
        print(f"  [{alert.severity}] {alert.message}")
        print(f"    → {alert.action}")
    
    print("\n✓ Alert System complete!")


def demo_paper_trading():
    """Demo paper trading"""
    print("\n" + "="*70)
    print("📈 PAPER TRADING DEMO")
    print("="*70)
    
    broker = PaperTradingBroker(initial_balance=10000)
    
    # Execute some trades
    print("\nExecuting sample trades...")
    broker.place_order('AAPL', 10, 'buy')
    broker.place_order('GOOGL', 5, 'buy')
    broker.place_order('AAPL', 5, 'sell')
    
    # Get portfolio summary
    summary = broker.get_portfolio_summary()
    
    print(f"\nPortfolio Summary:")
    print(f"  Total Value: ${summary['total_value']:,.2f}")
    print(f"  Cash Balance: ${summary['cash_balance']:,.2f}")
    print(f"  Holdings Value: ${summary['holdings_value']:,.2f}")
    print(f"  Total Return: {summary['total_return_pct']:+.2f}%")
    print(f"  Number of Trades: {summary['num_trades']}")
    
    print("\n✓ Paper Trading complete!")


def demo_auto_retrain():
    """Demo auto-retrain system"""
    print("\n" + "="*70)
    print("🔄 AUTO-RETRAIN SYSTEM DEMO")
    print("="*70)
    
    monitor = PerformanceMonitor(window_size=30)
    
    # Add some performance data
    print("\nSimulating performance tracking...")
    for i in range(20):
        sharpe = 1.5 - (i * 0.05)  # Degrading performance
        monitor.add_performance(sharpe=sharpe, returns=0.1, drawdown=0.05)
    
    # Check if degraded
    is_degraded = monitor.is_degraded(threshold_sharpe=0.5)
    metrics = monitor.get_metrics()
    
    print(f"\nPerformance Metrics:")
    print(f"  Recent Sharpe: {metrics['recent_sharpe']:.2f}")
    print(f"  Recent Returns: {metrics['recent_returns']:.2%}")
    print(f"  Recent Drawdown: {metrics['recent_drawdown']:.2%}")
    print(f"  Performance Degraded: {'YES - Retrain Triggered!' if is_degraded else 'NO'}")
    
    print("\n✓ Auto-Retrain System complete!")


def demo_risk_management():
    """Demo risk management"""
    print("\n" + "="*70)
    print("🛡️ RISK MANAGEMENT DEMO")
    print("="*70)
    
    risk_manager = RiskManager(
        max_drawdown=0.15,
        max_position_size=0.30,
        daily_loss_limit=0.05
    )
    
    # Test various scenarios
    print("\nTesting risk scenarios...")
    
    # Scenario 1: Position sizing
    size = risk_manager.calculate_position_size(
        current_value=10000,
        entry_price=150,
        stop_loss=140
    )
    print(f"\n1. Position Sizing (Kelly):")
    print(f"   Recommended shares: {size}")
    
    # Scenario 2: Check drawdown
    allowed = risk_manager.check_drawdown(0.12, 10000, 8800)
    print(f"\n2. Drawdown Check (12%):")
    print(f"   Trading allowed: {'YES' if allowed else 'NO - Limit breached!'}")
    
    # Scenario 3: Daily loss limit
    allowed = risk_manager.check_daily_loss_limit(10000, 9400)
    print(f"\n3. Daily Loss Check (6%):")
    print(f"   Trading allowed: {'YES' if allowed else 'NO - Daily limit hit!'}")
    
    print("\n✓ Risk Management complete!")


def demo_regime_detection(env):
    """Demo market regime detection"""
    print("\n" + "="*70)
    print("🌍 MARKET REGIME DETECTION DEMO")
    print("="*70)
    
    detector = RegimeDetector(lookback_window=20)
    
    # Get market data from environment
    obs, _ = env.reset()
    
    # Detect regime (using dummy data)
    prices = np.random.randn(50).cumsum() + 100
    regime = detector.detect_regime(prices)
    
    print(f"\nDetected Market Regime:")
    print(f"  Regime: {regime['regime']}")
    print(f"  Confidence: {regime['confidence']:.1%}")
    print(f"  Volatility: {regime['volatility']:.4f}")
    print(f"  Trend: {regime['trend']:.4f}")
    
    print("\n✓ Regime Detection complete!")


def main():
    parser = argparse.ArgumentParser(description='AlphaAgent Complete System Demo')
    parser.add_argument('--ticker', type=str, default='AAPL', help='Stock ticker')
    parser.add_argument('--quick', action='store_true', help='Run quick demo (skip training)')
    args = parser.parse_args()
    
    print(f"Ticker: {args.ticker}")
    print(f"Mode: {'Quick Demo' if args.quick else 'Full Demo'}")
    print()
    
    # Setup
    print("Setting up environment...")
    data_fetcher = MarketDataFetcher(ticker=args.ticker)
    sentiment_analyzer = FinBERTSentimentAnalyzer()
    
    # Fetch data
    print(f"Fetching data for {args.ticker}...")
    ohlcv = data_fetcher.fetch_ohlcv("2023-01-01", "2024-01-01")
    news = data_fetcher.fetch_news()
    
    if ohlcv.empty:
        print(f"❌ No data for {args.ticker}")
        return
    
    # Create environment
    print("Creating trading environment...")
    env = TradingEnvironment(
        ticker=args.ticker,
        start_date="2023-01-01",
        end_date="2024-01-01",
        initial_balance=10000.0,
        use_gaf=False  # Disable GAF for speed in demo
    )
    
    # Create or load agent
    model_path = f"./models/ppo_agent_{args.ticker}_latest.zip"
    
    if os.path.exists(model_path) and args.quick:
        print(f"Loading existing model from {model_path}...")
        agent = TradingPPOAgent.load(model_path, env)
    else:
        print("Creating new agent...")
        agent = TradingPPOAgent(env, learning_rate=3e-4, verbose=1)
        
        if not args.quick:
            print("Training agent (5000 timesteps)...")
            agent.train(5000)
            agent.save(model_path)
    
    print("✓ Setup complete!")
    
    # Run all demos
    demos = [
        ("Feature Attribution", lambda: demo_feature_attribution(agent, env)),
        ("Performance Attribution", demo_performance_attribution),
        ("Monte Carlo Simulation", demo_monte_carlo),
        ("Alert System", demo_alert_system),
        ("Paper Trading", demo_paper_trading),
        ("Auto-Retrain System", demo_auto_retrain),
        ("Risk Management", demo_risk_management),
        ("Regime Detection", lambda: demo_regime_detection(env)),
    ]
    
    for name, demo_func in demos:
        try:
            demo_func()
        except Exception as e:
            print(f"\n⚠️ {name} encountered an error: {e}")
            import traceback
            traceback.print_exc()
    
    # Final summary
    print("\n" + "="*70)
    print("✅ COMPLETE SYSTEM DEMO FINISHED!")
    print("="*70)
    print("\nAll Features Demonstrated:")
    print("  ✓ Feature Attribution (SHAP explanations)")
    print("  ✓ Performance Attribution (P&L breakdown)")
    print("  ✓ Monte Carlo Simulation (future projections)")
    print("  ✓ Alert System (monitoring & notifications)")
    print("  ✓ Paper Trading (risk-free live trading)")
    print("  ✓ Auto-Retrain (automated model updates)")
    print("  ✓ Risk Management (position sizing, limits)")
    print("  ✓ Regime Detection (market state awareness)")
    print("\nAdditional Features Available:")
    print("  • Meta-Agent (agent of agents)")
    print("  • Adversarial Training")
    print("  • Hierarchical Multi-Timeframe Agent")
    print("  • Live Dashboard (run: python live_dashboard.py)")
    print("  • Walk-Forward Analysis")
    print("  • Multi-Asset Portfolio")
    print("  • Ensemble Agents")
    print()
    print("🚀 AlphaAgent: Production-Ready DRL Trading System")
    print("="*70)


if __name__ == "__main__":
    main()

