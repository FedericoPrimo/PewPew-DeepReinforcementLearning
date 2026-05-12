"""
Agente che implementa algoritmo DQN
"""

from abc import ABC, abstractmethod
import numpy as np
from .base_agent import BaseAgent

from typing import Optional

import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

from src.models.DQN import DQN
from src.models.cnn_backbone import CNNBackbone

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 2500
TAU = 0.005
LR = 3e-4


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQNAgent(BaseAgent):
    """
    Classe base astratta per gli agenti.

    Definisce il contratto che ogni agente deve rispettare,
    garantendo compatibilità con il loop di training/evaluation.
    """

    def __init__(
        self,
        action_space_size: int,
        observation_space_size: int,
        device: str = "cpu",
        seed: Optional[int] = None,
        cnn: CNNBackbone = None
    ):
        """
        Args:
            action_space_size: Numero di azioni discrete disponibili.
            name: Nome identificativo dell'agente.
        """
        super().__init__(action_space_size=action_space_size, name="DQNAgent")

        self._action_space_size = action_space_size
        self._observation_space_size = observation_space_size

        self.device = device
        self.cnn = cnn.to(device)

        feature_dim = cnn.get_feature_dim()

        self.dqn = DQN(n_observations=feature_dim, n_actions=action_space_size)

        self._rng = np.random.default_rng(seed)
        self._seed = seed

        self.steps_done = 0

    def act(self, observation: np.ndarray) -> int:
        """
        Sceglie un'azione casuale ignorando l'osservazione.

        Args:
            observation: Osservazione preprocessata (ignorata).

        Returns:
            Azione casuale intera in [0, action_space_size).
        """
        obs_tensor = torch.from_numpy(observation).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
          features = self.cnn(obs_tensor)
          q_values = self.dqn(features)
    
        # Explore the actiosn
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)
        if random.random() > eps_threshold:
            with torch.no_grad():
                return int(self._rng.integers(0, self._action_space_size))
        else:
            return int(q_values.argmax(dim=1).item())

    def reset(self) -> None:
        """
        Reimposta il generatore casuale al seed iniziale
        per garantire riproducibilità tra episodi con lo stesso seed.
        """
        self._rng = np.random.default_rng(self._seed)

    def update(self, *args, **kwargs) -> None:
        """
        Aggiorna i parametri dell'agente (es. pesi della rete).
        Placeholder per futuri agenti allenabili.
        Per il DummyAgent non fa nulla.
        """
        pass

    def save(self, path: str) -> None:
        """Salva il modello su disco. Placeholder."""
        pass

    def load(self, path: str) -> None:
        """Carica il modello da disco. Placeholder."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(action_space_size={self.action_space_size})"
