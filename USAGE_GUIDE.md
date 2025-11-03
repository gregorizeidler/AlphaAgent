# AlphaAgent Usage Guide

## Quick Start

### 1. Installation

```bash
# Clone or download the project
cd yaho2

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Setup API Keys

Create a `.env` file (copy from `.env.example`):

```bash
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:
```
OPENAI_API_KEY=sk-your-key-here
```

**Note**: If you don't have an OpenAI API key, the system will still work using only FinBERT for sentiment analysis.

### 3. Run Demo

Test that everything is working:

```bash
python demo.py
```

This will demonstrate:
- Data fetching from Yahoo Finance
- Sentiment analysis (FinBERT + GPT)
- State representation (GAF, fundamentals, sentiment)
- Trading environment
- PPO agent

## Training an Agent

### Basic Training

Train an agent on Apple stock:

```bash
python train_agent.py --ticker AAPL --total-timesteps 100000
```

### Advanced Training Options

```bash
python train_agent.py \
  --ticker AAPL \
  --total-timesteps 200000 \
  --reward-type sharpe \
  --use-gaf \
  --use-attention \
  --learning-rate 0.0003 \
  --initial-balance 10000
```

### Multi-Horizon Training

Train on multiple time horizons (1-day, 5-day, 20-day):

```bash
python train_agent.py \
  --ticker AAPL \
  --multi-horizon \
  --horizons 1,5,20 \
  --total-timesteps 150000
```

### Training Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--ticker` | Stock ticker symbol | AAPL |
| `--total-timesteps` | Total training steps | 100000 |
| `--reward-type` | Reward function (sharpe/sortino/composite) | sharpe |
| `--use-gaf` | Use Gramian Angular Fields | False |
| `--use-attention` | Use attention mechanism | False |
| `--learning-rate` | Learning rate | 3e-4 |
| `--initial-balance` | Initial portfolio balance | 10000 |
| `--enable-shorting` | Allow short selling | False |
| `--transaction-cost` | Transaction cost rate | 0.001 |
| `--slippage` | Slippage rate | 0.0005 |

## Evaluating an Agent

### Basic Evaluation

```bash
python evaluate_agent.py \
  --model-path models/best_model.zip \
  --ticker AAPL \
  --n-episodes 10
```

### With Plots

```bash
python evaluate_agent.py \
  --model-path models/ppo_agent_AAPL_20250101_120000.zip \
  --ticker AAPL \
  --n-episodes 20 \
  --save-plots \
  --output-dir ./results/
```

### Evaluation Output

The evaluation script provides:
- Episode-by-episode performance
- Mean reward and returns
- Sharpe and Sortino ratios
- Maximum drawdown
- Number of trades
- Visualization plots
- CSV results file

## Backtesting

### Run Backtest

```bash
python backtest.py \
  --model-path models/best_model.zip \
  --ticker AAPL \
  --start-date 2023-01-01 \
  --end-date 2024-12-31
```

### Compare with Buy-and-Hold

```bash
python backtest.py \
  --model-path models/best_model.zip \
  --ticker AAPL \
  --start-date 2023-01-01 \
  --compare-buyhold
```

### Backtest Output

- Comprehensive performance metrics
- Comparison with buy-and-hold strategy
- Detailed plots (portfolio value, drawdown, positions, etc.)
- CSV file with daily data

## Understanding the System

### State Space

The agent observes a complex state composed of:

1. **Price History** (30 days)
   - OHLCV data
   - Optional GAF transformation
   - Technical indicators (RSI, MACD, Bollinger Bands, etc.)

2. **Fundamental Data**
   - P/E ratio, Forward P/E
   - Debt/Equity ratio
   - ROE, ROA
   - Profit margins
   - Revenue/earnings growth
   - Beta

3. **Sentiment Data**
   - FinBERT sentiment scores (positive, negative, neutral)
   - GPT analysis (sentiment score, confidence, key topics)
   - Sentiment embeddings (768-dimensional)

4. **Portfolio State**
   - Current position ratio
   - Cash balance ratio
   - Portfolio value change

### Action Space

The agent outputs continuous actions:
- `action[0]`: Position change (-1 to +1)
  - -1 = Sell all / Go short
  - 0 = Hold
  - +1 = Buy / Go long
- `action[1]`: Confidence (0 to 1) - reserved for future use

### Reward Functions

#### Sharpe Ratio (Default)
```
reward = (mean_return - risk_free_rate) / std_return
```
Balances returns and volatility.

#### Sortino Ratio
```
reward = (mean_return - risk_free_rate) / downside_std
```
Focuses on downside risk only.

#### Composite Reward
Weighted combination of simple returns, Sharpe, and Sortino.

### Market Simulation

The environment simulates realistic trading:
- **Transaction Costs**: 0.1% per trade
- **Slippage**: 0.05% (price moves against you)
- **Market Impact**: Proportional to order size

## Advanced Usage

### Custom Ticker Lists

Train on multiple tickers:

```bash
for ticker in AAPL GOOGL MSFT TSLA; do
  python train_agent.py --ticker $ticker --total-timesteps 100000
done
```

### Hyperparameter Tuning

Experiment with different configurations:

```python
# Create a custom training script
configs = [
    {'learning_rate': 1e-4, 'gamma': 0.99},
    {'learning_rate': 3e-4, 'gamma': 0.95},
    {'learning_rate': 5e-4, 'gamma': 0.99},
]

for config in configs:
    # Train with each config
    ...
```

### Using GAF Transformation

Gramian Angular Fields convert time series into images:

```bash
python train_agent.py \
  --ticker AAPL \
  --use-gaf \
  --total-timesteps 150000
```

This is more computationally expensive but can capture temporal patterns better.

### Using Attention Mechanism

Enable attention-based feature extraction:

```bash
python train_agent.py \
  --ticker AAPL \
  --use-attention \
  --total-timesteps 150000
```

## Performance Tips

### For Faster Training

1. **Disable GAF**: Use raw prices instead
   ```bash
   --use-gaf  # Don't use this flag
   ```

2. **Reduce Timesteps**: Start with fewer timesteps
   ```bash
   --total-timesteps 50000
   ```

3. **Smaller Lookback**: Reduce lookback window
   ```bash
   --lookback-days 15
   ```

### For Better Performance

1. **Use GAF**: Better pattern recognition
2. **Use Attention**: Better feature extraction
3. **Multi-Horizon**: Train on multiple horizons
4. **More Timesteps**: 200k-500k timesteps
5. **Experiment with Rewards**: Try different reward functions

## Monitoring Training

### TensorBoard

Training logs are saved to `./tensorboard_logs/`. View them with:

```bash
tensorboard --logdir=./tensorboard_logs/
```

Then open http://localhost:6006 in your browser.

### Training Progress

The training script prints:
- Timestep count
- Mean reward
- Best model saves

Models are saved:
- **Best model**: `models/best_model.zip`
- **Final model**: `models/ppo_agent_TICKER_timestamp.zip`

## Troubleshooting

### Issue: Out of Memory

**Solution**: 
- Disable GAF: Remove `--use-gaf`
- Reduce batch size: Add `--batch-size 32`
- Reduce n_steps: Add `--n-steps 1024`

### Issue: Training is Slow

**Solution**:
- Disable GAF
- Disable attention
- Use GPU if available (PyTorch will auto-detect)

### Issue: Poor Performance

**Solution**:
- Train longer: Increase `--total-timesteps`
- Try different reward functions: `--reward-type sortino`
- Enable advanced features: `--use-gaf --use-attention`
- Adjust hyperparameters: `--learning-rate 1e-4`

### Issue: No OpenAI API Key

**Solution**:
The system will work with just FinBERT. Simply don't set `OPENAI_API_KEY` in `.env`, or leave it empty.

## Examples

### Example 1: Quick Test

```bash
# Fast training for testing
python train_agent.py \
  --ticker AAPL \
  --total-timesteps 10000 \
  --save-freq 2000
```

### Example 2: Production Training

```bash
# High-quality training
python train_agent.py \
  --ticker AAPL \
  --total-timesteps 300000 \
  --use-gaf \
  --use-attention \
  --reward-type composite \
  --learning-rate 0.0001
```

### Example 3: Full Pipeline

```bash
# 1. Train
python train_agent.py --ticker AAPL --total-timesteps 100000

# 2. Evaluate
python evaluate_agent.py \
  --model-path models/best_model.zip \
  --ticker AAPL \
  --n-episodes 20 \
  --save-plots

# 3. Backtest
python backtest.py \
  --model-path models/best_model.zip \
  --ticker AAPL \
  --start-date 2023-01-01 \
  --compare-buyhold
```

## Research Directions

This is a research-grade system. Areas for improvement:

1. **Ensemble Methods**: Combine multiple agents
2. **Transfer Learning**: Train on one ticker, fine-tune on others
3. **Market Regime Detection**: Adapt strategy to market conditions
4. **Portfolio Optimization**: Multi-asset trading
5. **Risk Management**: Dynamic position sizing based on volatility
6. **Alternative Data**: Integrate more data sources (social media, options flow, etc.)

## Citation

If you use this system in your research, please cite:

```
AlphaAgent: Deep Reinforcement Learning Trading System
Built with: PPO, FinBERT, GPT, Gramian Angular Fields
Technology: Python, PyTorch, Stable-Baselines3, Yahoo Finance
```

## Support

For issues, questions, or contributions:
- Check the code comments for implementation details
- Review the research papers on PPO and DRL for trading
- Experiment with different configurations

Good luck with your trading agent! 🚀📈

