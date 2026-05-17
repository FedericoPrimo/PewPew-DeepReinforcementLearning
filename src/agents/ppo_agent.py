"""
PPO Agent — implementazione da zero (senza stable-baselines3).

Algoritmo PPO-Clip (Schulman et al. 2017):
  Loop:
    1. collect_rollouts(): n_steps × n_envs transizioni con π corrente
    2. compute_gae(): advantages GAE + discounted returns nel buffer
    3. update(): n_epochs × minibatch — ottimizza L_CLIP + L_VF + H(π)

  L_CLIP = E[ min(r_t * A_t,  clip(r_t, 1-ε, 1+ε) * A_t) ]
  L_VF   = 0.5 * SmoothL1(V(s_t), R_t)
  H(π)   = entropia distribuzione categorica (incoraggia esplorazione)
  r_t    = π(a|s) / π_old(a|s)  — probability ratio

Differenze chiave da DQN:
  - On-policy: dati usati n_epochs volte e poi scartati (nessun replay buffer)
  - Actor-Critic: policy head π(a|s) + value head V(s) con backbone condiviso
  - Esplorazione stocastica: campiona da distribuzione categorica
  - Parallelismo: n_envs ambienti simultanei (via SB3 VecEnv)
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple

from src.agents.base_agent import BaseAgent
from src.models.actor_critic import ActorCriticNet
from src.training.rollout_buffer import RolloutBuffer


class PPOAgent(BaseAgent):
    """
    PPO con Actor-Critic CNN, GAE e PPO-Clip.

    Training loop esterno (in train_ppo.py):
        obs = vec_env.reset()
        while step < total:
            obs, ep_rewards = agent.collect_rollouts(vec_env, obs)
            metrics = agent.update()
            log(metrics, ep_rewards)
    """

    def __init__(
        self,
        n_actions: int,
        n_envs: int,
        n_steps: int,
        n_epochs: int,
        batch_size: int,
        learning_rate: float,
        clip_range: float,
        gamma: float,
        gae_lambda: float,
        ent_coef: float,
        vf_coef: float,
        device: str = "cpu",
        feature_dim: int = 512,
        obs_shape: tuple[int, int, int] = (12, 84, 84),
    ):
        super().__init__(action_space_size=n_actions, name="PPOAgent")
        self.n_envs = n_envs
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.clip_range = clip_range
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.device = torch.device(device)
        self.obs_shape = obs_shape

        self.net = ActorCriticNet(
            n_actions=n_actions,
            feature_dim=feature_dim,
            in_channels=obs_shape[0],
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate, eps=1e-5)

        self.buffer = RolloutBuffer(
            n_steps=n_steps,
            n_envs=n_envs,
            obs_shape=obs_shape,
            device=device,
        )

    def _to_tensor(self, obs: np.ndarray) -> torch.Tensor:
        """(n_envs, C, 84, 84) uint8/float32 → float32 [0, 1] su device."""
        arr = np.asarray(obs)
        tensor = torch.from_numpy(arr).float().to(self.device)
        if arr.dtype == np.uint8 or tensor.max().item() > 1.0:
            tensor = tensor / 255.0
        return tensor

    @torch.no_grad()
    def collect_rollouts(
        self, vec_env, obs: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        """
        Raccoglie n_steps step da vec_env con la policy corrente.

        Args:
            vec_env: VecEnv custom con obs shape (n_envs, C, 84, 84)
            obs:     osservazione corrente (n_envs, C, 84, 84) uint8/float32

        Returns:
            obs: osservazione dopo l'ultimo step (per bootstrapping GAE)
            ep_rewards: reward totale di ogni episodio completato durante il rollout
        """
        self.buffer.reset()
        self.net.eval()
        ep_rewards: List[float] = []

        for _ in range(self.n_steps):
            obs_t = self._to_tensor(obs)
            actions, log_probs, _, values = self.net.get_action_and_value(obs_t)

            actions_np = actions.cpu().numpy()
            log_probs_np = log_probs.cpu().numpy()
            values_np = values.cpu().numpy()

            next_obs, rewards, dones, infos = vec_env.step(actions_np)

            # Episodi completati: SB3 Monitor aggiunge 'episode' a info quando done
            for i, (done, info) in enumerate(zip(dones, infos)):
                if done and "episode" in info:
                    ep_rewards.append(float(info["episode"]["r"]))

            self.buffer.add(
                obs=np.asarray(obs),
                action=actions_np,
                reward=np.asarray(rewards, dtype=np.float32),
                done=np.asarray(dones, dtype=np.float32),
                value=values_np,
                log_prob=log_probs_np,
            )
            obs = next_obs

        last_values = self.net.get_value(self._to_tensor(np.asarray(obs))).cpu().numpy()
        self.buffer.compute_gae(last_values, self.gamma, self.gae_lambda)

        return obs, ep_rewards

    def update(self) -> Dict[str, float]:
        """
        PPO-Clip update: n_epochs di minibatch sull'intero rollout corrente.

        Returns:
            Dict con metriche medie: policy_loss, value_loss, entropy, approx_kl.
        """
        self.net.train()
        metrics: Dict[str, List[float]] = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "approx_kl": [],
        }

        for _ in range(self.n_epochs):
            for batch in self.buffer.get_minibatches(self.batch_size):
                obs = batch["obs"]
                actions = batch["actions"]
                returns = batch["returns"]
                advantages = batch["advantages"]
                old_log_probs = batch["old_log_probs"]

                _, new_log_probs, entropy, new_values = self.net.get_action_and_value(obs, actions)

                log_ratio = new_log_probs - old_log_probs
                ratio = log_ratio.exp()

                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.functional.smooth_l1_loss(new_values, returns)

                entropy_loss = -entropy.mean()

                loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=0.5)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()

                metrics["policy_loss"].append(policy_loss.item())
                metrics["value_loss"].append(value_loss.item())
                metrics["entropy"].append(-entropy_loss.item())
                metrics["approx_kl"].append(approx_kl)

        return {k: float(np.mean(v)) for k, v in metrics.items()}

    def act(self, observation: np.ndarray) -> int:
        """Azione deterministica (argmax logits) per evaluation post-training."""
        obs_t = torch.from_numpy(np.asarray(observation)).float().unsqueeze(0).to(self.device) / 255.0
        with torch.no_grad():
            logits, _ = self.net(obs_t)
        return int(logits.argmax(dim=1).item())

    def reset(self) -> None:
        pass

    def save(self, path: str) -> None:
        torch.save({
            "net": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.net.load_state_dict(checkpoint["net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
