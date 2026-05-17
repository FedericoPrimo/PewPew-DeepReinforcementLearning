"""
Replay buffer uniforme per DQN.

Storage uint8 per ridurre memoria: 100k frame (C, 84, 84) con C configurabile.
Normalizzazione a float32 avviene al momento del sample, non dello storage.
"""

import numpy as np
import torch
from typing import Dict


class ReplayBuffer:
    """
    Buffer circolare con campionamento uniforme.

    Differenza DQN vs PPO: DQN è off-policy e richiede replay buffer per
    rompere le correlazioni temporali tra campioni. PPO è on-policy e usa
    i dati raccolti direttamente senza buffer.
    """

    def __init__(
        self,
        capacity: int,
        obs_shape: tuple = (12, 84, 84),
        device: str = "cpu",
    ):
        """
        Args:
            capacity: Numero massimo di transizioni memorizzate.
            obs_shape: Shape dell'osservazione (C, H, W).
            device: Device torch per il sample.
        """
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.device = torch.device(device)

        self.obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=bool)

        self.ptr = 0
        self.size = 0

    def push(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Inserisce una transizione nel buffer (sovrascrive se pieno)."""
        self.obs[self.ptr] = obs
        self.next_obs[self.ptr] = next_obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Campionamento uniforme senza rimpiazzo.

        Returns:
            Dict con chiavi: obs, next_obs, actions, rewards, dones.
            Tutti su self.device. obs/next_obs normalizzati in [0, 1].
        """
        idxs = np.random.randint(0, self.size, size=batch_size)

        return {
            "obs": torch.from_numpy(self.obs[idxs]).float().to(self.device) / 255.0,
            "next_obs": torch.from_numpy(self.next_obs[idxs]).float().to(self.device) / 255.0,
            "actions": torch.from_numpy(self.actions[idxs]).to(self.device),
            "rewards": torch.from_numpy(self.rewards[idxs]).to(self.device),
            "dones": torch.from_numpy(self.dones[idxs].astype(np.float32)).to(self.device),
        }

    def __len__(self) -> int:
        return self.size
