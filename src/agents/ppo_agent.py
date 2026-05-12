"""
PPO Agent — wrapper di stable-baselines3 compatibile con BaseAgent.

Architettura (SB3 CnnPolicy = NatureCNN):
  Conv2d(4,32,k=8,s=4) → Conv2d(32,64,k=4,s=2) → Conv2d(64,64,k=3,s=1)
  → Flatten → Linear(3136, 512)
  → Actor head: Linear(512, n_actions)   — distribuzione categorica
  → Critic head: Linear(512, 1)          — stima V(s)

Differenze chiave PPO vs DQN:
  - Actor-Critic (policy + value condivisi) vs solo Q-network
  - On-policy: usa dati freschi, nessun replay buffer
  - Esplorazione stocastica: campiona dalla distribuzione categorica
  - Clip del ratio π/π_old (PPO-Clip) per aggiornamenti stabili
  - GAE (Generalized Advantage Estimation) come baseline per il vantaggio
"""

import numpy as np
from stable_baselines3 import PPO

from src.agents.base_agent import BaseAgent


class PPOAgent(BaseAgent):
    """
    Wrapper sottile di SB3 PPO per compatibilità con BaseAgent.

    Il training è delegato completamente a SB3 via scripts/train_ppo.py.
    Questo wrapper è usato solo durante l'evaluation post-training.
    """

    def __init__(self, n_actions: int, sb3_model: PPO):
        """
        Args:
            n_actions: Dimensione action space (per compatibilità BaseAgent).
            sb3_model: Istanza SB3 PPO già addestrata.
        """
        super().__init__(action_space_size=n_actions, name="PPOAgent")
        self.model = sb3_model

    def act(self, observation: np.ndarray) -> int:
        """
        Azione deterministica (greedy) dalla policy addestrata.

        Args:
            observation: (4, 84, 84) uint8 — stesso formato DQN per comparabilità.
        """
        action, _ = self.model.predict(observation, deterministic=True)
        return int(action)

    def reset(self) -> None:
        pass

    def save(self, path: str) -> None:
        self.model.save(path)

    def load(self, path: str) -> None:
        self.model = PPO.load(path)

    @classmethod
    def from_checkpoint(cls, path: str, env, n_actions: int) -> "PPOAgent":
        """Carica un agente PPO da checkpoint SB3."""
        model = PPO.load(path, env=env)
        return cls(n_actions=n_actions, sb3_model=model)
