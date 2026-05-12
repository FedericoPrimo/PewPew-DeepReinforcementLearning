"""
DQN Agent — agente value-based con Q-learning profondo.

Architettura (Nature DQN, Mnih et al. 2015):
  CNNBackbone(in_channels=4) → Linear(512, n_actions)

Differenze chiave DQN vs PPO:
  - Value-based (solo Q-network) vs Actor-Critic (policy + value head condivisi)
  - Off-policy con replay buffer vs on-policy senza buffer
  - Esplorazione epsilon-greedy (soglia deterministica) vs distribuzione stocastica
  - Target network separata (stabilità training) vs Critic head (baseline per vantaggio)
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from src.models.cnn_backbone import CNNBackbone
from src.agents.base_agent import BaseAgent


class QNetwork(nn.Module):
    """
    Rete Q: backbone CNN + head lineare sulle azioni.

    Input:  (B, 4, 84, 84) float32 in [0, 1]
    Output: (B, n_actions) float32  — Q-value per ogni azione
    """

    def __init__(self, n_actions: int, feature_dim: int = 512):
        super().__init__()
        # in_channels=4: 4 frame grayscale impilati (standard Atari DQN)
        self.backbone = CNNBackbone(in_channels=4, feature_dim=feature_dim)
        self.head = nn.Linear(feature_dim, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)


class DQNAgent(BaseAgent):
    """
    Agente DQN con epsilon-greedy e target network.

    Durante training: usa epsilon-greedy per esplorazione.
    Durante evaluation: policy greedy (argmax Q).
    """

    def __init__(
        self,
        n_actions: int,
        feature_dim: int = 512,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay_steps: int = 500_000,
        device: str = "cpu",
    ):
        super().__init__(action_space_size=n_actions, name="DQNAgent")

        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.device = torch.device(device)
        self._training = True

        self.q_network = QNetwork(n_actions, feature_dim).to(self.device)
        # Target network: copia della Q-network, aggiornata periodicamente (hard update).
        # Stabilizza il training evitando che target e predizioni cambino insieme.
        self.target_network = QNetwork(n_actions, feature_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)

    def act(self, observation: np.ndarray) -> int:
        """
        Sceglie azione con epsilon-greedy (training) o greedy (eval).

        Args:
            observation: (4, 84, 84) uint8 da env con wrapper Atari.
        """
        if self._training and random.random() < self.epsilon:
            return random.randrange(self.action_space_size)

        obs = torch.from_numpy(np.array(observation)).float().unsqueeze(0).to(self.device) / 255.0
        with torch.no_grad():
            q_values = self.q_network(obs)
        return int(q_values.argmax(dim=1).item())

    def reset(self) -> None:
        pass

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Aggiorna Q-network su un batch dal replay buffer.

        Target Q = r + γ * max_a' Q_target(s', a')  se not done
        Target Q = r                                  se done

        Loss: Huber (smooth_l1) — più robusta di MSE per valori outlier in RL.

        Returns:
            Dict con 'loss' per logging TensorBoard.
        """
        obs = batch["obs"]
        next_obs = batch["next_obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        dones = batch["dones"]

        # Q-value predetto per le azioni effettivamente eseguite
        q_pred = self.q_network(obs).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q: calcolato con target network (no gradient)
        with torch.no_grad():
            q_next = self.target_network(next_obs).max(dim=1)[0]
            q_target = rewards + self.gamma * q_next * (1.0 - dones)

        loss = F.smooth_l1_loss(q_pred, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping per stabilità (comune in DQN su Atari)
        nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()

        return {"loss": loss.item()}

    def update_target_network(self) -> None:
        """Hard update: copia pesi da Q-network a target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self, step: int) -> None:
        """Decadimento lineare di epsilon in base allo step corrente."""
        fraction = min(1.0, step / self.epsilon_decay_steps)
        self.epsilon = self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)

    def set_training_mode(self, training: bool) -> None:
        self._training = training
        if training:
            self.q_network.train()
        else:
            self.q_network.eval()

    def save(self, path: str) -> None:
        torch.save({
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
        }, path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
