"""
Quick Demo of AlphaAgent System
"""

import logging
from alpha_agent.data import MarketDataFetcher, TechnicalIndicators
from alpha_agent.sentiment import MultiModalSentimentAnalyzer
from alpha_agent.state import CompleteStateRepresentation
from alpha_agent.environment import TradingEnvironment
from alpha_agent.agents import TradingPPOAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_data_fetching():
    """Demo data fetching capabilities"""
    print("\n" + "="*60)
    print("DEMO 1: Data Fetching")
    print("="*60)
    
    ticker = "AAPL"
    logger.info(f"Fetching data for {ticker}...")
    
    fetcher = MarketDataFetcher(ticker, lookback_days=60)
    ohlcv, fundamentals, news = fetcher.fetch_all_data()
    
    print(f"\n✓ Fetched {len(ohlcv)} days of OHLCV data")
    print(f"✓ Fetched {len(fundamentals)} fundamental metrics")
    print(f"✓ Fetched {len(news)} news items")
    
    print(f"\nLatest OHLCV:")
    print(ohlcv.tail(3))
    
    print(f"\nKey Fundamentals:")
    for key in ['pe_ratio', 'debt_to_equity', 'roe']:
        print(f"  {key}: {fundamentals.get(key, 'N/A')}")
    
    if news:
        print(f"\nLatest News:")
        print(f"  {news[0]['title']}")


def demo_sentiment_analysis():
    """Demo sentiment analysis"""
    print("\n" + "="*60)
    print("DEMO 2: Sentiment Analysis")
    print("="*60)
    
    ticker = "AAPL"
    fetcher = MarketDataFetcher(ticker, lookback_days=30)
    _, _, news = fetcher.fetch_all_data()
    
    logger.info("Analyzing sentiment...")
    analyzer = MultiModalSentimentAnalyzer()
    sentiment_data = analyzer.analyze_comprehensive(news, ticker)
    
    print(f"\n✓ Sentiment Analysis Complete")
    print(f"\nFinBERT Sentiment:")
    print(f"  Positive: {sentiment_data['finbert']['positive']:.3f}")
    print(f"  Negative: {sentiment_data['finbert']['negative']:.3f}")
    print(f"  Neutral: {sentiment_data['finbert']['neutral']:.3f}")
    print(f"  Compound: {sentiment_data['finbert']['compound']:.3f}")
    
    print(f"\nGPT Analysis:")
    print(f"  Sentiment Score: {sentiment_data['gpt']['sentiment_score']:.3f}")
    print(f"  Confidence: {sentiment_data['gpt']['confidence']:.3f}")
    print(f"  Key Topics: {', '.join(sentiment_data['gpt']['key_topics'][:3])}")
    
    print(f"\nCombined Sentiment: {sentiment_data['combined_sentiment']:.3f}")


def demo_state_representation():
    """Demo state representation"""
    print("\n" + "="*60)
    print("DEMO 3: State Representation")
    print("="*60)
    
    ticker = "AAPL"
    
    # Fetch data
    fetcher = MarketDataFetcher(ticker, lookback_days=60)
    ohlcv, fundamentals, news = fetcher.fetch_all_data()
    ohlcv = TechnicalIndicators.add_technical_indicators(ohlcv)
    
    # Sentiment
    analyzer = MultiModalSentimentAnalyzer()
    sentiment_data = analyzer.analyze_comprehensive(news, ticker)
    
    # Create state
    logger.info("Creating state representation...")
    state_repr = CompleteStateRepresentation(
        gaf_size=30,
        lookback_days=30,
        use_gaf=False,  # Faster without GAF
        flatten_output=True
    )
    
    state = state_repr.create_state(ohlcv, fundamentals, sentiment_data, ohlcv)
    
    print(f"\n✓ State Created")
    print(f"  Shape: {state.shape}")
    print(f"  Range: [{state.min():.3f}, {state.max():.3f}]")
    print(f"  Components: Price history + Fundamentals + Sentiment + Technical Indicators")


def demo_environment():
    """Demo trading environment"""
    print("\n" + "="*60)
    print("DEMO 4: Trading Environment")
    print("="*60)
    
    ticker = "AAPL"
    logger.info(f"Creating trading environment for {ticker}...")
    
    env = TradingEnvironment(
        ticker=ticker,
        initial_balance=10000.0,
        reward_type='sharpe',
        use_gaf=False,
    )
    
    print(f"\n✓ Environment Created")
    print(f"  Observation Space: {env.observation_space.shape}")
    print(f"  Action Space: {env.action_space.shape}")
    print(f"  Max Steps: {env.max_steps}")
    
    # Run a few steps
    print(f"\nRunning 5 random steps...")
    obs, info = env.reset()
    
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"  Step {i+1}: Portfolio=${info['portfolio_value']:.2f}, Reward={reward:.2f}")
        
        if terminated or truncated:
            break
    
    env.close()


def demo_agent():
    """Demo PPO agent"""
    print("\n" + "="*60)
    print("DEMO 5: PPO Agent")
    print("="*60)
    
    ticker = "AAPL"
    
    # Create environment
    env = TradingEnvironment(
        ticker=ticker,
        initial_balance=10000.0,
        reward_type='sharpe',
        use_gaf=False,
    )
    
    # Create agent
    logger.info("Creating PPO agent...")
    agent = TradingPPOAgent(
        env=env,
        learning_rate=3e-4,
        use_attention=False,
        verbose=0
    )
    
    print(f"\n✓ Agent Created")
    print(f"  Policy Architecture: Actor-Critic")
    print(f"  Algorithm: PPO (Proximal Policy Optimization)")
    
    # Test prediction
    obs, info = env.reset()
    action, _states = agent.predict(obs, deterministic=True)
    
    print(f"\n✓ Test Prediction:")
    print(f"  Observation shape: {obs.shape}")
    print(f"  Action: {action}")
    print(f"  Action[0] (position change): {action[0]:.3f}")
    print(f"  Action[1] (confidence): {action[1]:.3f}")
    
    env.close()


def main():
    """Run all demos"""
    print("\n" + "="*80)
    print(" " * 20 + "AlphaAgent System Demo")
    print("="*80)
    
    try:
        demo_data_fetching()
        demo_sentiment_analysis()
        demo_state_representation()
        demo_environment()
        demo_agent()
        
        print("\n" + "="*80)
        print(" " * 25 + "All Demos Completed!")
        print("="*80)
        
        print("\n📚 Next Steps:")
        print("  1. Train an agent: python train_agent.py --ticker AAPL --total-timesteps 50000")
        print("  2. Evaluate agent: python evaluate_agent.py --model-path models/best_model.zip --ticker AAPL")
        print("  3. Backtest: python backtest.py --model-path models/best_model.zip --ticker AAPL --start-date 2023-01-01")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    main()

