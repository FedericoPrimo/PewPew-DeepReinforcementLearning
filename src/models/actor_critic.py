"""
Actor-Critic network per PPO su Atari.

Architettura (backbone CNN condiviso, paper PPO Schulman et al. 2017):
  CNNBackbone(in_channels=4) → feature_dim
  → Policy head: Linear(feature_dim, n_actions)  — logits distribuzione π(a|s)
  → Value  head: Linear(feature_dim, 1)           — stima scalare V(s)

Inizializzazione ortogonale standard PPO:
  backbone: gain=sqrt(2)  (default ortogonale)
  policy head: gain=0.01  (outputs quasi-uniformi all'inizio)
  value head:  gain=1.0
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical

from src.models.cnn_backbone import CNNBackbone


class ActorCriticNet(nn.Module):
    """
    Rete Actor-Critic con backbone CNN condiviso tra policy e value.

    Input:  (B, 4, 84, 84) float32 in [0, 1]
    Policy: distribuzione categorica π(a|s) su n_actions
    Value:  scalare V(s) per ogni elemento del batch
    """

    def __init__(self, n_actions: int, feature_dim: int = 512):
        super().__init__()
        self.backbone = CNNBackbone(in_channels=4, feature_dim=feature_dim)
        self.policy_head = nn.Linear(feature_dim, n_actions)
        self.value_head = nn.Linear(feature_dim, 1)

        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, 4, 84, 84) float32
        Returns:
            logits: (B, n_actions), value: (B,)
        """
        features = self.backbone(x)
        return self.policy_head(features), self.value_head(features).squeeze(-1)

    def get_action_and_value(self, x: torch.Tensor, action: torch.Tensor = None):
        """
        Campiona azione (o usa quella fornita) e restituisce log_prob, entropy, value.

        Args:
            x:      (B, 4, 84, 84) float32
            action: (B,) long — se fornita, calcola log_prob per queste azioni (update)

        Returns:
            action (B,), log_prob (B,), entropy (B,), value (B,)
        """
        logits, value = self(x)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """Stima V(s) senza campionare azioni. Usato per bootstrapping GAE."""
        features = self.backbone(x)
        return self.value_head(features).squeeze(-1)
