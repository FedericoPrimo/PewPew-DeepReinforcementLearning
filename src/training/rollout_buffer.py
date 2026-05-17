"""
Rollout buffer on-policy per PPO.

Accumula n_steps × n_envs transizioni, poi calcola GAE e genera
minibatch per l'update. I dati vengono scartati dopo ogni update (on-policy).

GAE (Generalized Advantage Estimation, Schulman et al. 2016):
  δ(t)   = r(t) + γ * V(t+1) * (1 - done(t)) - V(t)
  A(t)   = δ(t) + (γλ) * A(t+1) * (1 - done(t))
  R(t)   = A(t) + V(t)   — discounted return usato come target value
"""

import numpy as np
import torch
from typing import Iterator, Dict


class RolloutBuffer:
    """
    Buffer a capacità fissa n_steps × n_envs.

    Differenze vs ReplayBuffer DQN:
    - Nessun campionamento casuale: i dati sono usati tutti e poi scartati
    - Memorizza log_prob e values (necessari per PPO-Clip e GAE)
    - compute_gae() chiamato una volta dopo ogni rollout completo
    """

    def __init__(
        self,
        n_steps: int,
        n_envs: int,
        obs_shape: tuple = (12, 84, 84),
        device: str = "cpu",
    ):
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.obs_shape = obs_shape
        self.device = torch.device(device)
        self.ptr = 0

        self.obs = np.zeros((n_steps, n_envs, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros((n_steps, n_envs), dtype=np.int64)
        self.rewards = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.dones = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.values = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.log_probs = np.zeros((n_steps, n_envs), dtype=np.float32)

        self.advantages = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.returns = np.zeros((n_steps, n_envs), dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        value: np.ndarray,
        log_prob: np.ndarray,
    ) -> None:
        """Inserisce uno step simultaneo da tutti gli n_envs."""
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.ptr += 1

    def compute_gae(
        self,
        last_values: np.ndarray,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        """
        Calcola GAE advantages e discounted returns in-place.

        Args:
            last_values: (n_envs,) V(s_T) — bootstrapping sull'ultimo stato osservato.
            gamma: fattore di sconto.
            gae_lambda: parametro λ per GAE (0=solo TD, 1=Monte Carlo).
        """
        gae = np.zeros(self.n_envs, dtype=np.float32)
        for t in reversed(range(self.n_steps)):
            next_non_terminal = 1.0 - self.dones[t]
            next_values = last_values if t == self.n_steps - 1 else self.values[t + 1]
            delta = self.rewards[t] + gamma * next_values * next_non_terminal - self.values[t]
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            self.advantages[t] = gae
        self.returns = self.advantages + self.values

    def get_minibatches(self, batch_size: int) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Flatten (n_steps × n_envs), shuffle, itera su minibatch di batch_size.

        Advantages normalizzati a media 0 e std 1 per ridurre varianza durante update.
        """
        total = self.n_steps * self.n_envs
        indices = np.random.permutation(total)

        flat_adv = self.advantages.reshape(-1)
        flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)

        flat_obs = self.obs.reshape(total, *self.obs_shape)
        flat_actions = self.actions.reshape(total)
        flat_returns = self.returns.reshape(total)
        flat_log_probs = self.log_probs.reshape(total)

        for start in range(0, total, batch_size):
            idx = indices[start : start + batch_size]
            yield {
                "obs": torch.from_numpy(flat_obs[idx]).float().to(self.device) / 255.0,
                "actions": torch.from_numpy(flat_actions[idx]).long().to(self.device),
                "returns": torch.from_numpy(flat_returns[idx]).float().to(self.device),
                "advantages": torch.from_numpy(flat_adv[idx]).float().to(self.device),
                "old_log_probs": torch.from_numpy(flat_log_probs[idx]).float().to(self.device),
            }

    def reset(self) -> None:
        self.ptr = 0
