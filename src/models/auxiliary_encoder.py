"""
Custom encoder with auxiliary price prediction head for multi-task learning.

Architecture:
- Shared trunk: Extracts features from observations
- Policy head: Used by IQL/SAC for action selection
- Auxiliary head: Predicts future price direction (supervised task)

Benefits:
- Shared representations learn to predict future returns
- These representations help the policy make better decisions
- Multi-task learning improves generalization
"""

import torch
import torch.nn as nn
from typing import Sequence
from d3rlpy.models.encoders import EncoderFactory, Encoder


class AuxiliaryPredictionEncoder(Encoder):
    """
    Encoder with auxiliary price direction prediction head.
    
    Architecture:
        obs → [shared_trunk] → features
                                  ├─→ [policy_head] → policy_features (for actor/critic)
                                  └─→ [aux_head] → price_direction_logits
    """
    
    def __init__(
        self,
        observation_shape: Sequence[int],
        feature_size: int,
        hidden_units: Sequence[int] = (256, 256),
        activation: nn.Module = nn.ReLU,
        use_batch_norm: bool = False,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self._observation_shape = observation_shape
        self._feature_size = feature_size
        
        # Shared trunk (extracts general features)
        trunk_layers = []
        in_size = observation_shape[0]
        
        for hidden_size in hidden_units:
            trunk_layers.append(nn.Linear(in_size, hidden_size))
            if use_batch_norm:
                trunk_layers.append(nn.BatchNorm1d(hidden_size))
            trunk_layers.append(activation())
            if dropout_rate > 0:
                trunk_layers.append(nn.Dropout(dropout_rate))
            in_size = hidden_size
        
        self.trunk = nn.Sequential(*trunk_layers)
        
        # Policy head (for RL)
        self.policy_head = nn.Linear(in_size, feature_size)
        
        # Auxiliary prediction head (for supervised learning)
        # Predicts 3 classes: DOWN (-1), NEUTRAL (0), UP (+1)
        self.aux_head = nn.Sequential(
            nn.Linear(in_size, 128),
            activation(),
            nn.Dropout(0.2),
            nn.Linear(128, 3)  # 3-class classification
        )
        
        # Track auxiliary predictions for analysis
        self._last_aux_logits = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for policy (used by IQL/SAC).
        
        Args:
            x: Observations [batch, obs_dim]
            
        Returns:
            Policy features [batch, feature_size]
        """
        h = self.trunk(x)
        policy_features = self.policy_head(h)
        
        # Also compute auxiliary predictions (for monitoring)
        with torch.no_grad():
            self._last_aux_logits = self.aux_head(h)
        
        return policy_features
    
    def forward_with_aux(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with auxiliary predictions (for training).
        
        Args:
            x: Observations [batch, obs_dim]
            
        Returns:
            policy_features: [batch, feature_size]
            aux_logits: [batch, 3] (class logits for DOWN/NEUTRAL/UP)
        """
        h = self.trunk(x)
        policy_features = self.policy_head(h)
        aux_logits = self.aux_head(h)
        return policy_features, aux_logits
    
    @property
    def observation_shape(self) -> Sequence[int]:
        return self._observation_shape
    
    @property
    def feature_size(self) -> int:
        return self._feature_size
    
    def get_aux_predictions(self) -> torch.Tensor:
        """Get last auxiliary predictions (for logging)."""
        return self._last_aux_logits


class AuxiliaryEncoderFactory(EncoderFactory):
    """Factory for creating AuxiliaryPredictionEncoder."""
    
    def __init__(
        self,
        feature_size: int = 256,
        hidden_units: Sequence[int] = (256, 256),
        activation: str = "relu",
        use_batch_norm: bool = False,
        dropout_rate: float = 0.0,
    ):
        self._feature_size = feature_size
        self._hidden_units = hidden_units
        self._activation = activation
        self._use_batch_norm = use_batch_norm
        self._dropout_rate = dropout_rate
    
    def create(self, observation_shape: Sequence[int]) -> AuxiliaryPredictionEncoder:
        """Create encoder instance."""
        activation_fn = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "swish": nn.SiLU,
        }.get(self._activation.lower(), nn.ReLU)
        
        return AuxiliaryPredictionEncoder(
            observation_shape=observation_shape,
            feature_size=self._feature_size,
            hidden_units=self._hidden_units,
            activation=activation_fn,
            use_batch_norm=self._use_batch_norm,
            dropout_rate=self._dropout_rate,
        )
    
    @staticmethod
    def get_type() -> str:
        return "auxiliary"


# Register factory for d3rlpy
def create_auxiliary_encoder_factory(
    feature_size: int = 256,
    hidden_units: Sequence[int] = (256, 256),
    **kwargs
) -> AuxiliaryEncoderFactory:
    """
    Helper function to create auxiliary encoder factory.
    
    Usage in config:
        actor_encoder_factory:
          type: auxiliary
          params:
            feature_size: 256
            hidden_units: [256, 256]
            dropout_rate: 0.1
    """
    return AuxiliaryEncoderFactory(
        feature_size=feature_size,
        hidden_units=hidden_units,
        **kwargs
    )
