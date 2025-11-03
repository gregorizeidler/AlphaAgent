"""
Performance Attribution Analysis
Decomposes returns into contributing factors
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceAttributor:
    """
    Analyzes where profits/losses come from
    """
    
    def __init__(self):
        self.attributions = []
        logger.info("PerformanceAttributor initialized")
    
    def attribute_trades(self, trades_history: List[Dict], 
                        initial_balance: float,
                        final_balance: float) -> Dict:
        """
        Decompose total P&L into components
        
        Args:
            trades_history: List of executed trades
            initial_balance: Starting capital
            final_balance: Ending capital
            
        Returns:
            Attribution dictionary
        """
        total_pnl = final_balance - initial_balance
        
        attribution = {
            'total_pnl': total_pnl,
            'gross_profit': 0.0,
            'gross_loss': 0.0,
            'transaction_costs': 0.0,
            'slippage_costs': 0.0,
            'market_impact_costs': 0.0,
            'winning_trades': 0,
            'losing_trades': 0,
            'breakeven_trades': 0
        }
        
        for trade in trades_history:
            # Extract trade P&L
            trade_value = trade.get('trade_value', 0)
            transaction_cost = trade.get('transaction_cost', 0)
            slippage_cost = trade.get('slippage_cost', 0)
            market_impact = trade.get('market_impact', 0)
            
            # Categorize
            if trade_value > 0:
                attribution['gross_profit'] += trade_value
                attribution['winning_trades'] += 1
            elif trade_value < 0:
                attribution['gross_loss'] += abs(trade_value)
                attribution['losing_trades'] += 1
            else:
                attribution['breakeven_trades'] += 1
            
            # Costs
            attribution['transaction_costs'] += transaction_cost
            attribution['slippage_costs'] += slippage_cost
            attribution['market_impact_costs'] += market_impact
        
        # Calculate derived metrics
        attribution['net_profit'] = attribution['gross_profit'] - attribution['gross_loss']
        attribution['total_costs'] = (attribution['transaction_costs'] + 
                                     attribution['slippage_costs'] + 
                                     attribution['market_impact_costs'])
        
        attribution['profit_factor'] = (attribution['gross_profit'] / 
                                       (attribution['gross_loss'] + 1e-8))
        
        total_trades = (attribution['winning_trades'] + 
                       attribution['losing_trades'] + 
                       attribution['breakeven_trades'])
        attribution['win_rate'] = (attribution['winning_trades'] / 
                                  (total_trades + 1e-8))
        
        logger.info(f"Performance attributed: Total P&L=${total_pnl:.2f}")
        
        return attribution
    
    def plot_waterfall(self, attribution: Dict, save_path: str = None):
        """
        Create waterfall chart of P&L attribution
        
        Args:
            attribution: Attribution dictionary
            save_path: Optional save path
        """
        # Components for waterfall
        initial = 0
        components = [
            ('Gross Profit', attribution['gross_profit']),
            ('Gross Loss', -attribution['gross_loss']),
            ('Transaction Costs', -attribution['transaction_costs']),
            ('Slippage', -attribution['slippage_costs']),
            ('Market Impact', -attribution['market_impact_costs'])
        ]
        
        # Calculate cumulative
        values = [initial]
        labels = ['Initial']
        colors = ['gray']
        
        cumulative = initial
        for label, value in components:
            values.append(value)
            labels.append(label)
            colors.append('#27ae60' if value > 0 else '#e74c3c')
            cumulative += value
        
        # Final
        values.append(cumulative)
        labels.append('Final P&L')
        colors.append('#2c3e50')
        
        # Create waterfall plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Calculate positions
        positions = np.arange(len(labels))
        bottoms = [0]
        running_total = initial
        
        for i, val in enumerate(values[1:-1]):
            bottoms.append(running_total)
            running_total += val
        bottoms.append(0)
        
        # Plot bars
        for i, (label, value, color, bottom) in enumerate(zip(labels, values, colors, bottoms)):
            if label == 'Initial' or label == 'Final P&L':
                ax.bar(i, running_total if label == 'Final P&L' else initial, 
                      color=color, alpha=0.7, width=0.6)
            else:
                ax.bar(i, value, bottom=bottom, color=color, alpha=0.7, width=0.6)
        
        # Connect bars
        for i in range(len(positions) - 1):
            if i == 0:
                start_y = initial
            else:
                start_y = bottoms[i] + values[i]
            end_y = bottoms[i + 1]
            ax.plot([i + 0.3, i + 0.7], [start_y, end_y], 'k--', alpha=0.3, linewidth=1)
        
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('P&L ($)')
        ax.set_title('Performance Attribution Waterfall', fontsize=14, fontweight='bold')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Waterfall chart saved to {save_path}")
        
        plt.show()
    
    def create_attribution_report(self, attribution: Dict) -> str:
        """
        Generate text report of attribution
        
        Args:
            attribution: Attribution dictionary
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 60)
        report.append("PERFORMANCE ATTRIBUTION REPORT")
        report.append("=" * 60)
        report.append("")
        
        # P&L Breakdown
        report.append("P&L Breakdown:")
        report.append(f"  Gross Profit:        ${attribution['gross_profit']:>12,.2f}")
        report.append(f"  Gross Loss:          ${-attribution['gross_loss']:>12,.2f}")
        report.append(f"  ─────────────────────────────────────")
        report.append(f"  Net Trading P&L:     ${attribution['net_profit']:>12,.2f}")
        report.append("")
        
        # Costs
        report.append("Costs:")
        report.append(f"  Transaction Costs:   ${-attribution['transaction_costs']:>12,.2f}")
        report.append(f"  Slippage:            ${-attribution['slippage_costs']:>12,.2f}")
        report.append(f"  Market Impact:       ${-attribution['market_impact_costs']:>12,.2f}")
        report.append(f"  ─────────────────────────────────────")
        report.append(f"  Total Costs:         ${-attribution['total_costs']:>12,.2f}")
        report.append("")
        
        # Final
        report.append(f"  TOTAL P&L:           ${attribution['total_pnl']:>12,.2f}")
        report.append("")
        
        # Statistics
        report.append("Trade Statistics:")
        report.append(f"  Winning Trades:      {attribution['winning_trades']:>12}")
        report.append(f"  Losing Trades:       {attribution['losing_trades']:>12}")
        report.append(f"  Breakeven Trades:    {attribution['breakeven_trades']:>12}")
        report.append(f"  Win Rate:            {attribution['win_rate']*100:>11.1f}%")
        report.append(f"  Profit Factor:       {attribution['profit_factor']:>12.2f}")
        report.append("")
        
        # Cost Analysis
        if attribution['net_profit'] > 0:
            cost_percentage = (attribution['total_costs'] / attribution['net_profit']) * 100
            report.append(f"Costs consumed {cost_percentage:.1f}% of gross profit")
        
        report.append("=" * 60)
        
        return "\n".join(report)


class MonteCarloSimulator:
    """
    Monte Carlo simulation for future portfolio projections
    """
    
    def __init__(self, seed: int = 42):
        """
        Args:
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)
        logger.info("MonteCarloSimulator initialized")
    
    def simulate_future(
        self,
        current_value: float,
        historical_returns: np.ndarray,
        n_days: int = 90,
        n_simulations: int = 1000
    ) -> Tuple[np.ndarray, Dict]:
        """
        Simulate future portfolio scenarios
        
        Args:
            current_value: Current portfolio value
            historical_returns: Historical daily returns
            n_days: Days to simulate
            n_simulations: Number of scenarios
            
        Returns:
            (scenarios array, statistics dict)
        """
        logger.info(f"Running {n_simulations} Monte Carlo simulations for {n_days} days")
        
        # Estimate parameters from historical data
        mean_return = np.mean(historical_returns)
        std_return = np.std(historical_returns)
        
        # Generate scenarios
        scenarios = np.zeros((n_simulations, n_days + 1))
        scenarios[:, 0] = current_value
        
        for i in range(n_simulations):
            for day in range(1, n_days + 1):
                # Generate random return
                daily_return = np.random.normal(mean_return, std_return)
                scenarios[i, day] = scenarios[i, day - 1] * (1 + daily_return)
        
        # Calculate statistics
        final_values = scenarios[:, -1]
        percentiles = np.percentile(final_values, [5, 25, 50, 75, 95])
        
        stats = {
            'percentile_5': percentiles[0],
            'percentile_25': percentiles[1],
            'median': percentiles[2],
            'percentile_75': percentiles[3],
            'percentile_95': percentiles[4],
            'mean': np.mean(final_values),
            'std': np.std(final_values),
            'prob_profit': np.mean(final_values > current_value),
            'prob_loss': np.mean(final_values < current_value),
            'var_95': np.percentile(final_values - current_value, 5),  # Value at Risk
            'cvar_95': np.mean((final_values - current_value)[final_values < percentiles[0]])  # Conditional VaR
        }
        
        logger.info(f"Monte Carlo complete. Median final value: ${stats['median']:.2f}")
        
        return scenarios, stats
    
    def plot_scenarios(
        self,
        scenarios: np.ndarray,
        stats: Dict,
        current_value: float,
        save_path: str = None
    ):
        """
        Plot Monte Carlo scenarios
        
        Args:
            scenarios: Simulated scenarios
            stats: Statistics dictionary
            current_value: Current portfolio value
            save_path: Optional save path
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Left plot: Scenario paths
        ax = axes[0]
        n_days = scenarios.shape[1]
        days = np.arange(n_days)
        
        # Plot percentile bands
        percentiles = np.percentile(scenarios, [5, 25, 50, 75, 95], axis=0)
        
        ax.fill_between(days, percentiles[0], percentiles[4], 
                        alpha=0.2, color='#3498db', label='5th-95th percentile')
        ax.fill_between(days, percentiles[1], percentiles[3], 
                        alpha=0.3, color='#3498db', label='25th-75th percentile')
        ax.plot(days, percentiles[2], color='#2c3e50', linewidth=2, label='Median')
        ax.axhline(y=current_value, color='red', linestyle='--', 
                  linewidth=2, label='Current Value')
        
        ax.set_xlabel('Days')
        ax.set_ylabel('Portfolio Value ($)')
        ax.set_title('Monte Carlo Future Scenarios', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Right plot: Final value distribution
        ax = axes[1]
        final_values = scenarios[:, -1]
        ax.hist(final_values, bins=50, alpha=0.7, color='#9b59b6', edgecolor='black')
        ax.axvline(x=current_value, color='red', linestyle='--', 
                  linewidth=2, label='Current Value')
        ax.axvline(x=stats['median'], color='#2c3e50', linestyle='-', 
                  linewidth=2, label='Median')
        ax.axvline(x=stats['percentile_5'], color='#e74c3c', linestyle=':', 
                  linewidth=2, label='5th percentile (VaR)')
        
        ax.set_xlabel('Final Portfolio Value ($)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Final Values', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Monte Carlo plot saved to {save_path}")
        
        plt.show()
    
    def generate_report(self, stats: Dict, current_value: float, n_days: int) -> str:
        """
        Generate Monte Carlo report
        
        Args:
            stats: Statistics dictionary
            current_value: Current value
            n_days: Forecast horizon
            
        Returns:
            Formatted report
        """
        report = []
        report.append("=" * 60)
        report.append(f"MONTE CARLO FORECAST ({n_days} days)")
        report.append("=" * 60)
        report.append("")
        report.append(f"Current Value:          ${current_value:,.2f}")
        report.append("")
        report.append("Projected Final Values:")
        report.append(f"  5th Percentile (Worst): ${stats['percentile_5']:,.2f}")
        report.append(f"  25th Percentile:        ${stats['percentile_25']:,.2f}")
        report.append(f"  50th Percentile (Median): ${stats['median']:,.2f}")
        report.append(f"  75th Percentile:        ${stats['percentile_75']:,.2f}")
        report.append(f"  95th Percentile (Best): ${stats['percentile_95']:,.2f}")
        report.append("")
        report.append(f"  Mean:                   ${stats['mean']:,.2f}")
        report.append(f"  Std Dev:                ${stats['std']:,.2f}")
        report.append("")
        report.append("Probabilities:")
        report.append(f"  Probability of Profit:  {stats['prob_profit']*100:.1f}%")
        report.append(f"  Probability of Loss:    {stats['prob_loss']*100:.1f}%")
        report.append("")
        report.append("Risk Metrics:")
        report.append(f"  Value at Risk (95%):    ${stats['var_95']:,.2f}")
        report.append(f"  Conditional VaR (95%):  ${stats['cvar_95']:,.2f}")
        report.append("=" * 60)
        
        return "\n".join(report)


if __name__ == "__main__":
    print("Performance Attribution & Monte Carlo Modules")
    print("\nUsage:")
    print("1. Performance Attribution: Analyze where P&L comes from")
    print("2. Monte Carlo: Simulate future portfolio scenarios")

