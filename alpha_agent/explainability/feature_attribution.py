"""
Feature Attribution & Explainability using SHAP and LIME
Explains WHY the agent makes each decision using multiple methods
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import logging
import shap
from lime import lime_tabular

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_feature_names(state_size: int, lookback_days: int = 30, use_gaf: bool = False) -> List[str]:
    """
    Generate real feature names based on state composition
    
    Args:
        state_size: Total size of state vector
        lookback_days: Number of lookback days
        use_gaf: Whether GAF is used
        
    Returns:
        List of feature names
    """
    feature_names = []
    
    if use_gaf:
        # GAF features: 5 channels × 30×30 = 4500 features
        for channel in ['Open', 'High', 'Low', 'Close', 'Volume']:
            for i in range(900):  # 30×30 = 900 per channel
                feature_names.append(f'GAF_{channel}_{i}')
    else:
        # Raw price features: lookback_days × 5 OHLCV
        for day in range(lookback_days):
            feature_names.extend([
                f'Open_t-{day}',
                f'High_t-{day}',
                f'Low_t-{day}',
                f'Close_t-{day}',
                f'Volume_t-{day}'
            ])
    
    # Fundamental features (14)
    fundamentals = [
        'PE_Ratio',
        'PB_Ratio',
        'PS_Ratio',
        'Dividend_Yield',
        'ROE',
        'ROA',
        'Debt_to_Equity',
        'Current_Ratio',
        'Quick_Ratio',
        'Gross_Margin',
        'Operating_Margin',
        'Net_Margin',
        'Revenue_Growth',
        'Earnings_Growth'
    ]
    feature_names.extend(fundamentals)
    
    # Sentiment features (38)
    sentiment_features = [
        'Sentiment_Mean',
        'Sentiment_Std',
        'Sentiment_Positive',
        'Sentiment_Negative',
        'Sentiment_Neutral',
        'Sentiment_Score'
    ]
    feature_names.extend(sentiment_features)
    
    # Sentiment embedding (32 dimensions)
    for i in range(32):
        feature_names.append(f'Sentiment_Embed_{i}')
    
    # Technical indicators (5)
    technical_indicators = [
        'RSI',
        'MACD',
        'Bollinger_Bands',
        'ATR',
        'Volume_Ratio'
    ]
    feature_names.extend(technical_indicators)
    
    # Portfolio state (3)
    portfolio_features = [
        'Current_Position',
        'Cash_Balance',
        'Portfolio_Value'
    ]
    feature_names.extend(portfolio_features)
    
    # If we have more features than names, pad with generic names
    if len(feature_names) < state_size:
        for i in range(len(feature_names), state_size):
            feature_names.append(f'Feature_{i}')
    
    # If we have too many names, truncate
    return feature_names[:state_size]


class FeatureAttributor:
    """
    Explains agent decisions using SHAP (SHapley Additive exPlanations)
    """
    
    def __init__(self, agent, feature_names: List[str]):
        """
        Args:
            agent: Trained trading agent
            feature_names: Names of all features in state
        """
        self.agent = agent
        self.feature_names = feature_names
        self.explainer = None
        
        logger.info(f"FeatureAttributor initialized with {len(feature_names)} features")
    
    def create_explainer(self, background_data: np.ndarray, n_samples: int = 100):
        """
        Create SHAP explainer with background data
        
        Args:
            background_data: Sample of states for SHAP baseline
            n_samples: Number of background samples
        """
        # Use subset for efficiency
        background_subset = background_data[:n_samples]
        
        # Create explainer
        def model_predict(x):
            """Wrapper for agent prediction"""
            actions = []
            for obs in x:
                action, _ = self.agent.predict(obs, deterministic=True)
                actions.append(action[0])  # Position change
            return np.array(actions)
        
        self.explainer = shap.KernelExplainer(model_predict, background_subset)
        logger.info("SHAP explainer created")
    
    def explain_action(self, observation: np.ndarray, max_display: int = 10) -> Dict:
        """
        Explain a single action
        
        Args:
            observation: State observation
            max_display: Number of top features to show
            
        Returns:
            Dictionary with explanation
        """
        if self.explainer is None:
            raise ValueError("Explainer not created. Call create_explainer() first")
        
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(observation.reshape(1, -1))
        
        # Get base value (expected output)
        base_value = self.explainer.expected_value
        
        # Sort features by importance
        importance = np.abs(shap_values[0])
        top_indices = np.argsort(importance)[::-1][:max_display]
        
        # Create explanation
        top_features = []
        for idx in top_indices:
            feature_name = self.feature_names[idx] if idx < len(self.feature_names) else f"feature_{idx}"
            top_features.append({
                'feature': feature_name,
                'value': float(observation[idx]),
                'shap_value': float(shap_values[0][idx]),
                'importance': float(importance[idx])
            })
        
        # Predict action
        action, _ = self.agent.predict(observation, deterministic=True)
        
        explanation = {
            'action': float(action[0]),
            'base_value': float(base_value),
            'top_features': top_features,
            'total_shap_effect': float(np.sum(shap_values[0])),
            'reasoning': self._generate_reasoning(top_features, action[0])
        }
        
        return explanation
    
    def _generate_reasoning(self, top_features: List[Dict], action: float) -> str:
        """
        Generate human-readable reasoning
        
        Args:
            top_features: Top contributing features
            action: Predicted action
            
        Returns:
            Reasoning text
        """
        decision = "BUY" if action > 0.1 else "SELL" if action < -0.1 else "HOLD"
        
        # Find main contributors
        main_contributors = [f for f in top_features[:3] if abs(f['shap_value']) > 0.01]
        
        if not main_contributors:
            return f"Decision: {decision} (weak signal)"
        
        reasoning_parts = [f"Decision: {decision} because:"]
        
        for feat in main_contributors:
            direction = "high" if feat['shap_value'] > 0 else "low"
            contribution = "encourages buying" if feat['shap_value'] > 0 else "encourages selling"
            reasoning_parts.append(
                f"  • {feat['feature']} is {direction} ({feat['value']:.3f}), which {contribution}"
            )
        
        return "\n".join(reasoning_parts)
    
    def plot_waterfall(self, observation: np.ndarray, save_path: Optional[str] = None):
        """
        Plot SHAP waterfall chart
        
        Args:
            observation: State observation
            save_path: Optional path to save plot
        """
        if self.explainer is None:
            raise ValueError("Explainer not created")
        
        shap_values = self.explainer.shap_values(observation.reshape(1, -1))
        
        # Create waterfall plot
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[0],
                base_values=self.explainer.expected_value,
                data=observation,
                feature_names=self.feature_names[:len(observation)]
            )
        )
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Waterfall plot saved to {save_path}")
        
        plt.show()
    
    def plot_summary(self, observations: np.ndarray, max_display: int = 20, save_path: Optional[str] = None):
        """
        Plot SHAP summary (beeswarm plot) - CLASSIC BLUE/RED STYLE
        
        Args:
            observations: Multiple observations
            max_display: Number of features to display
            save_path: Optional path to save plot
        """
        if self.explainer is None:
            raise ValueError("Explainer not created")
        
        logger.info("Generating SHAP summary plot (this may take a moment)...")
        
        # Calculate SHAP values for multiple observations
        shap_values = self.explainer.shap_values(observations)
        
        # Create classic beeswarm plot with blue/red colors
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values, 
            observations,
            feature_names=self.feature_names[:observations.shape[1]],
            max_display=max_display,
            show=False
        )
        
        plt.title('SHAP Feature Importance (Beeswarm)', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"SHAP summary plot saved to {save_path}")
        
        plt.show()
    
    def plot_bar(self, observations: np.ndarray, max_display: int = 20, save_path: Optional[str] = None):
        """
        Plot SHAP bar plot - Global feature importance
        
        Args:
            observations: Multiple observations
            max_display: Number of features to display
            save_path: Optional path to save plot
        """
        if self.explainer is None:
            raise ValueError("Explainer not created")
        
        logger.info("Generating SHAP bar plot...")
        
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(observations)
        
        # Create bar plot
        plt.figure(figsize=(10, 8))
        shap.plots.bar(
            shap.Explanation(
                values=shap_values,
                base_values=self.explainer.expected_value,
                data=observations,
                feature_names=self.feature_names[:observations.shape[1]]
            ),
            max_display=max_display,
            show=False
        )
        
        plt.title('Global Feature Importance (SHAP)', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"SHAP bar plot saved to {save_path}")
        
        plt.show()
    
    def plot_combined_summary(self, observations: np.ndarray, max_display: int = 15, save_path: Optional[str] = None):
        """
        Plot combined SHAP visualization: Bar + Beeswarm (like the classic SHAP image)
        
        Args:
            observations: Multiple observations
            max_display: Number of features to display
            save_path: Optional path to save plot
        """
        if self.explainer is None:
            raise ValueError("Explainer not created")
        
        logger.info("Generating combined SHAP visualization...")
        
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(observations)
        
        # Get feature names matching observation size
        feature_names_to_use = self.feature_names[:observations.shape[1]]
        logger.info(f"Using {len(feature_names_to_use)} feature names for {observations.shape[1]} features")
        logger.info(f"Sample names: {feature_names_to_use[:5]}")
        
        # Create figure with 2 subplots side by side
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # Left: Bar plot (Global importance) - usando shap.summary_plot com plot_type='bar'
        plt.sca(axes[0])
        shap.summary_plot(
            shap_values,
            observations,
            feature_names=feature_names_to_use,
            plot_type='bar',
            max_display=max_display,
            show=False
        )
        axes[0].set_title('Global Feature Importance', fontsize=13, fontweight='bold')
        
        # Right: Beeswarm plot (Local explanations with feature values)
        plt.sca(axes[1])
        shap.summary_plot(
            shap_values,
            observations,
            feature_names=feature_names_to_use,
            max_display=max_display,
            show=False
        )
        axes[1].set_title('Local Explanation Summary', fontsize=13, fontweight='bold')
        
        # Main title
        fig.suptitle('SHAP Analysis: Global Importance & Local Explanations', 
                     fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Combined SHAP plot saved to {save_path}")
        
        plt.show()
    
    def plot_force(self, observation: np.ndarray, save_path: Optional[str] = None):
        """
        Plot SHAP force plot
        
        Args:
            observation: State observation
            save_path: Optional path to save plot
        """
        if self.explainer is None:
            raise ValueError("Explainer not created")
        
        shap_values = self.explainer.shap_values(observation.reshape(1, -1))
        
        # Create force plot
        shap.force_plot(
            self.explainer.expected_value,
            shap_values[0],
            observation,
            feature_names=self.feature_names[:len(observation)],
            matplotlib=True,
            show=False
        )
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Force plot saved to {save_path}")
        
        plt.show()
    
    def analyze_feature_importance_over_time(
        self, 
        observations: np.ndarray,
        window_size: int = 20
    ) -> pd.DataFrame:
        """
        Analyze how feature importance changes over time
        
        Args:
            observations: Sequence of observations
            window_size: Window for aggregating importance
            
        Returns:
            DataFrame with feature importance over time
        """
        if self.explainer is None:
            raise ValueError("Explainer not created")
        
        importance_history = []
        
        for i in range(0, len(observations), window_size):
            window = observations[i:i+window_size]
            if len(window) == 0:
                continue
            
            # Calculate SHAP for window
            shap_values = self.explainer.shap_values(window)
            
            # Average absolute importance
            avg_importance = np.mean(np.abs(shap_values), axis=0)
            
            importance_history.append(avg_importance)
        
        # Convert to DataFrame
        df = pd.DataFrame(
            importance_history,
            columns=self.feature_names[:observations.shape[1]]
        )
        
        return df
    
    def plot_feature_importance_heatmap(
        self,
        importance_df: pd.DataFrame,
        top_n: int = 15,
        save_path: Optional[str] = None
    ):
        """
        Plot heatmap of feature importance over time
        
        Args:
            importance_df: DataFrame from analyze_feature_importance_over_time
            top_n: Number of top features to show
            save_path: Optional save path
        """
        # Select top N most important features overall
        overall_importance = importance_df.mean(axis=0)
        top_features = overall_importance.nlargest(top_n).index
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            importance_df[top_features].T,
            cmap='YlOrRd',
            cbar_kws={'label': 'Importance (|SHAP|)'},
            xticklabels=False
        )
        
        plt.title('Feature Importance Over Time', fontsize=14, fontweight='bold')
        plt.xlabel('Time Window')
        plt.ylabel('Feature')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Heatmap saved to {save_path}")
        
        plt.show()


class DecisionLogger:
    """
    Logs and analyzes agent decisions for explainability
    """
    
    def __init__(self):
        self.decisions = []
    
    def log_decision(
        self,
        timestamp,
        observation: np.ndarray,
        action: float,
        explanation: Dict,
        result: Optional[float] = None
    ):
        """
        Log a decision with explanation
        
        Args:
            timestamp: When decision was made
            observation: State
            action: Action taken
            explanation: Explanation from FeatureAttributor
            result: Optional outcome (profit/loss)
        """
        self.decisions.append({
            'timestamp': timestamp,
            'observation': observation,
            'action': action,
            'explanation': explanation,
            'result': result
        })
    
    def analyze_good_vs_bad_decisions(self) -> Dict:
        """
        Compare features of profitable vs unprofitable decisions
        
        Returns:
            Analysis dictionary
        """
        if not self.decisions:
            return {}
        
        # Filter decisions with results
        decisions_with_results = [d for d in self.decisions if d['result'] is not None]
        
        if not decisions_with_results:
            return {}
        
        # Separate good and bad decisions
        good_decisions = [d for d in decisions_with_results if d['result'] > 0]
        bad_decisions = [d for d in decisions_with_results if d['result'] <= 0]
        
        def extract_top_features(decisions):
            """Extract most common top features"""
            feature_counts = {}
            for d in decisions:
                for feat in d['explanation']['top_features'][:3]:
                    name = feat['feature']
                    feature_counts[name] = feature_counts.get(name, 0) + 1
            return feature_counts
        
        analysis = {
            'num_good': len(good_decisions),
            'num_bad': len(bad_decisions),
            'good_decision_features': extract_top_features(good_decisions),
            'bad_decision_features': extract_top_features(bad_decisions),
            'win_rate': len(good_decisions) / len(decisions_with_results)
        }
        
        return analysis
    
    def export_decisions(self, filepath: str):
        """Export decisions to JSON"""
        import json
        
        # Convert to serializable format
        export_data = []
        for d in self.decisions:
            export_data.append({
                'timestamp': str(d['timestamp']),
                'action': float(d['action']),
                'explanation': d['explanation'],
                'result': float(d['result']) if d['result'] is not None else None
            })
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Decisions exported to {filepath}")


class LIMEExplainer:
    """
    Explains agent decisions using LIME (Local Interpretable Model-agnostic Explanations)
    """
    
    def __init__(self, agent, feature_names: List[str], training_data: np.ndarray):
        """
        Args:
            agent: Trained trading agent
            feature_names: Names of all features
            training_data: Training data for LIME baseline
        """
        self.agent = agent
        self.feature_names = feature_names
        
        # Create LIME explainer
        self.explainer = lime_tabular.LimeTabularExplainer(
            training_data=training_data,
            feature_names=feature_names,
            mode='regression',
            verbose=False
        )
        
        logger.info(f"LIMEExplainer initialized with {len(feature_names)} features")
    
    def explain_action(self, observation: np.ndarray, num_features: int = 10) -> Dict:
        """
        Explain a single action using LIME
        
        Args:
            observation: State observation
            num_features: Number of top features to show
            
        Returns:
            Dictionary with explanation
        """
        # Define prediction function
        def predict_fn(x):
            """Wrapper for agent prediction"""
            actions = []
            for obs in x:
                action, _ = self.agent.predict(obs, deterministic=True)
                actions.append(action[0])  # Position change
            return np.array(actions)
        
        # Generate LIME explanation
        exp = self.explainer.explain_instance(
            observation,
            predict_fn,
            num_features=num_features
        )
        
        # Get action
        action, _ = self.agent.predict(observation, deterministic=True)
        
        # Parse explanation
        feature_importance = exp.as_list()
        
        explanation = {
            'action': float(action[0]),
            'prediction_score': exp.score,
            'top_features': [
                {
                    'feature': feat[0],
                    'importance': feat[1]
                }
                for feat in feature_importance
            ],
            'reasoning': self._generate_reasoning(feature_importance, action[0])
        }
        
        return explanation
    
    def _generate_reasoning(self, feature_importance: List[Tuple], action: float) -> str:
        """
        Generate human-readable reasoning
        
        Args:
            feature_importance: List of (feature, importance) tuples
            action: Predicted action
            
        Returns:
            Reasoning text
        """
        decision = "BUY" if action > 0.1 else "SELL" if action < -0.1 else "HOLD"
        
        reasoning_parts = [f"Decision: {decision} based on:"]
        
        for feat_desc, importance in feature_importance[:3]:
            direction = "positive" if importance > 0 else "negative"
            reasoning_parts.append(
                f"  • {feat_desc} has {direction} influence ({importance:+.4f})"
            )
        
        return "\n".join(reasoning_parts)
    
    def plot_explanation(self, observation: np.ndarray, save_path: Optional[str] = None):
        """
        Plot LIME explanation
        
        Args:
            observation: State observation
            save_path: Optional path to save plot
        """
        def predict_fn(x):
            actions = []
            for obs in x:
                action, _ = self.agent.predict(obs, deterministic=True)
                actions.append(action[0])
            return np.array(actions)
        
        exp = self.explainer.explain_instance(
            observation,
            predict_fn,
            num_features=15
        )
        
        # Create plot
        fig = exp.as_pyplot_figure()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"LIME plot saved to {save_path}")
        
        plt.show()


class CombinedExplainer:
    """
    Combines SHAP and LIME for comprehensive explanations
    """
    
    def __init__(
        self,
        agent,
        feature_names: List[str],
        background_data: np.ndarray
    ):
        """
        Args:
            agent: Trained trading agent
            feature_names: Names of all features
            background_data: Background data for both explainers
        """
        self.agent = agent
        self.feature_names = feature_names
        
        # Initialize both explainers
        self.shap_explainer = FeatureAttributor(agent, feature_names)
        self.shap_explainer.create_explainer(background_data, n_samples=50)
        
        self.lime_explainer = LIMEExplainer(agent, feature_names, background_data)
        
        logger.info("CombinedExplainer initialized with SHAP and LIME")
    
    def explain_action(self, observation: np.ndarray) -> Dict:
        """
        Get explanations from both SHAP and LIME
        
        Args:
            observation: State observation
            
        Returns:
            Combined explanation dictionary
        """
        # Get SHAP explanation
        shap_exp = self.shap_explainer.explain_action(observation, max_display=10)
        
        # Get LIME explanation
        lime_exp = self.lime_explainer.explain_action(observation, num_features=10)
        
        # Combine
        combined = {
            'action': shap_exp['action'],
            'shap_explanation': shap_exp,
            'lime_explanation': lime_exp,
            'agreement_score': self._calculate_agreement(shap_exp, lime_exp)
        }
        
        return combined
    
    def _calculate_agreement(self, shap_exp: Dict, lime_exp: Dict) -> float:
        """
        Calculate agreement between SHAP and LIME
        
        Args:
            shap_exp: SHAP explanation
            lime_exp: LIME explanation
            
        Returns:
            Agreement score (0-1)
        """
        # Get top features from each
        shap_features = {f['feature']: f['shap_value'] 
                        for f in shap_exp['top_features'][:5]}
        
        lime_features = {f['feature'].split()[0]: f['importance'] 
                        for f in lime_exp['top_features'][:5]}
        
        # Calculate correlation of importances
        common_features = set(shap_features.keys()) & set(lime_features.keys())
        
        if len(common_features) < 2:
            return 0.0
        
        shap_vals = [shap_features[f] for f in common_features]
        lime_vals = [lime_features[f] for f in common_features]
        
        # Pearson correlation
        correlation = np.corrcoef(shap_vals, lime_vals)[0, 1]
        
        return abs(correlation) if not np.isnan(correlation) else 0.0
    
    def plot_comparison(self, observation: np.ndarray, save_path: Optional[str] = None):
        """
        Plot side-by-side comparison of SHAP and LIME
        
        Args:
            observation: State observation
            save_path: Optional save path
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Get explanations
        exp = self.explain_action(observation)
        
        # Plot SHAP
        ax = axes[0]
        shap_features = exp['shap_explanation']['top_features'][:10]
        features = [f['feature'] for f in shap_features]
        values = [f['shap_value'] for f in shap_features]
        
        colors = ['red' if v < 0 else 'green' for v in values]
        ax.barh(features, values, color=colors, alpha=0.7)
        ax.set_xlabel('SHAP Value', fontsize=11)
        ax.set_title('SHAP Feature Attribution', fontsize=13, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Plot LIME
        ax = axes[1]
        lime_features = exp['lime_explanation']['top_features'][:10]
        features = [f['feature'].split()[0] for f in lime_features]
        values = [f['importance'] for f in lime_features]
        
        colors = ['red' if v < 0 else 'green' for v in values]
        ax.barh(features, values, color=colors, alpha=0.7)
        ax.set_xlabel('LIME Importance', fontsize=11)
        ax.set_title('LIME Feature Attribution', fontsize=13, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add agreement score
        agreement = exp['agreement_score']
        fig.suptitle(f'Explainability Comparison (Agreement: {agreement:.2%})', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Comparison plot saved to {save_path}")
        
        plt.show()


if __name__ == "__main__":
    # Demo
    print("Feature Attribution Module")
    print("Supports SHAP and LIME explainability")
    print("\nUsage:")
    print("1. FeatureAttributor - SHAP explanations")
    print("2. LIMEExplainer - LIME explanations")
    print("3. CombinedExplainer - Both methods with comparison")
    print("\nFeatures:")
    print("  • Waterfall plots")
    print("  • Force plots")
    print("  • Feature importance over time")
    print("  • Side-by-side comparison")

