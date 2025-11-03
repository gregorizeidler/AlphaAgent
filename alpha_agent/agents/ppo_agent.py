"""
PPO Agent with Custom Actor-Critic Networks for Trading
"""

import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
from typing import Dict, Type, Optional
import gymnasium as gym
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TradingFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for trading state
    
    Processes the complex state representation with multiple components
    """
    
    def __init__(self, observation_space: gym.Space, features_dim: int = 256):
        """
        Args:
            observation_space: Observation space
            features_dim: Dimension of the features output
        """
        super().__init__(observation_space, features_dim)
        
        n_input = observation_space.shape[0]
        
        # Multi-layer perceptron for feature extraction
        self.feature_net = nn.Sequential(
            nn.Linear(n_input, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, features_dim),
            nn.ReLU()
        )
        
        logger.info(f"Initialized TradingFeatureExtractor: input={n_input}, output={features_dim}")
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            observations: Input observations
            
        Returns:
            Extracted features
        """
        return self.feature_net(observations)


class AttentionTradingFeatureExtractor(BaseFeaturesExtractor):
    """
    Advanced feature extractor with attention mechanism
    """
    
    def __init__(self, observation_space: gym.Space, features_dim: int = 256, num_heads: int = 4):
        """
        Args:
            observation_space: Observation space
            features_dim: Dimension of the features output
            num_heads: Number of attention heads
        """
        super().__init__(observation_space, features_dim)
        
        n_input = observation_space.shape[0]
        
        # Initial projection
        self.input_projection = nn.Linear(n_input, 512)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=512,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, features_dim),
            nn.ReLU()
        )
        
        logger.info(f"Initialized AttentionTradingFeatureExtractor: input={n_input}, output={features_dim}")
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with attention
        
        Args:
            observations: Input observations
            
        Returns:
            Extracted features
        """
        # Project input
        x = self.input_projection(observations)
        
        # Add sequence dimension for attention
        x = x.unsqueeze(1)  # (batch, 1, features)
        
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        
        # Remove sequence dimension
        attn_out = attn_out.squeeze(1)
        
        # Feed-forward
        features = self.ffn(attn_out)
        
        return features


class TradingActorCriticPolicy(ActorCriticPolicy):
    """
    Custom Actor-Critic policy for trading
    """
    
    def __init__(self,
                 observation_space: gym.Space,
                 action_space: gym.Space,
                 lr_schedule,
                 use_attention: bool = False,
                 *args,
                 **kwargs):
        """
        Args:
            observation_space: Observation space
            action_space: Action space
            lr_schedule: Learning rate schedule
            use_attention: Whether to use attention mechanism
        """
        # Set custom feature extractor
        if use_attention:
            kwargs['features_extractor_class'] = AttentionTradingFeatureExtractor
        else:
            kwargs['features_extractor_class'] = TradingFeatureExtractor
        
        kwargs['features_extractor_kwargs'] = {'features_dim': 256}
        
        # Set network architecture for actor and critic
        kwargs['net_arch'] = dict(
            pi=[256, 128],  # Actor network
            vf=[256, 128]   # Critic (Value function) network
        )
        
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs
        )


class TradingPPOAgent:
    """
    PPO Agent specifically designed for trading
    """
    
    def __init__(self,
                 env: gym.Env,
                 learning_rate: float = 3e-4,
                 n_steps: int = 2048,
                 batch_size: int = 64,
                 n_epochs: int = 10,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_range: float = 0.2,
                 ent_coef: float = 0.01,
                 vf_coef: float = 0.5,
                 max_grad_norm: float = 0.5,
                 use_attention: bool = False,
                 device: str = 'auto',
                 verbose: int = 1):
        """
        Initialize PPO agent with custom policy
        
        Args:
            env: Trading environment
            learning_rate: Learning rate
            n_steps: Number of steps per update
            batch_size: Batch size
            n_epochs: Number of epochs per update
            gamma: Discount factor
            gae_lambda: GAE lambda
            clip_range: PPO clipping range
            ent_coef: Entropy coefficient
            vf_coef: Value function coefficient
            max_grad_norm: Max gradient norm
            use_attention: Use attention-based feature extractor
            device: Device to use
            verbose: Verbosity level
        """
        self.env = env
        
        # Create policy kwargs
        policy_kwargs = {
            'use_attention': use_attention,
        }
        
        # Initialize PPO agent
        self.model = PPO(
            policy=TradingActorCriticPolicy,
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            policy_kwargs=policy_kwargs,
            device=device,
            verbose=verbose,
            tensorboard_log="./tensorboard_logs/"
        )
        
        logger.info("Initialized TradingPPOAgent")
        logger.info(f"  Learning Rate: {learning_rate}")
        logger.info(f"  Attention: {use_attention}")
        logger.info(f"  Device: {device}")
    
    def train(self, total_timesteps: int, callback: Optional[BaseCallback] = None):
        """
        Train the agent
        
        Args:
            total_timesteps: Total training timesteps
            callback: Optional callback
        """
        logger.info(f"Starting training for {total_timesteps} timesteps")
        
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True
        )
        
        logger.info("Training completed")
    
    def predict(self, observation, deterministic: bool = True):
        """
        Predict action given observation
        
        Args:
            observation: Current observation
            deterministic: Whether to use deterministic policy
            
        Returns:
            Action and state
        """
        return self.model.predict(observation, deterministic=deterministic)
    
    def save(self, path: str):
        """
        Save the model
        
        Args:
            path: Path to save the model
        """
        self.model.save(path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """
        Load the model
        
        Args:
            path: Path to load the model from
        """
        self.model = PPO.load(path, env=self.env)
        logger.info(f"Model loaded from {path}")


class TradingCallback(BaseCallback):
    """
    Custom callback for tracking training progress
    """
    
    def __init__(self, check_freq: int = 1000, save_path: str = "./models/", verbose: int = 1):
        """
        Args:
            check_freq: Frequency of checks
            save_path: Path to save best model
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.best_mean_reward = -np.inf
        
    def _init_callback(self) -> None:
        """Initialize callback"""
        import os
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
    
    def _on_step(self) -> bool:
        """
        Called at every step
        
        Returns:
            True to continue training
        """
        if self.n_calls % self.check_freq == 0:
            # Get recent episode rewards
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = np.mean([ep_info['r'] for ep_info in self.model.ep_info_buffer])
                
                if self.verbose > 0:
                    print(f"Timestep: {self.num_timesteps}")
                    print(f"Mean Reward: {mean_reward:.2f}")
                
                # Save best model
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    if self.save_path is not None:
                        save_file = f"{self.save_path}/best_model.zip"
                        self.model.save(save_file)
                        if self.verbose > 0:
                            print(f"New best model saved! Reward: {mean_reward:.2f}")
        
        return True


if __name__ == "__main__":
    # Test the agent
    from alpha_agent.environment import TradingEnvironment
    
    # Create environment
    env = TradingEnvironment(
        ticker="AAPL",
        initial_balance=10000.0,
        reward_type='sharpe',
        use_gaf=False  # Faster for testing
    )
    
    # Create agent
    agent = TradingPPOAgent(
        env=env,
        learning_rate=3e-4,
        use_attention=False,
        verbose=1
    )
    
    print("Agent created successfully!")
    print(f"Policy: {agent.model.policy}")
    
    # Test prediction
    obs, info = env.reset()
    action, _states = agent.predict(obs)
    print(f"\nTest Prediction:")
    print(f"  Observation shape: {obs.shape}")
    print(f"  Action: {action}")
    
    env.close()

