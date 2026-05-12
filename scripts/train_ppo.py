"""
Training PPO su Space Invaders (Atari) via stable-baselines3.

Uso:
  python scripts/train_ppo.py
  python scripts/train_ppo.py --timesteps 1000 --device cpu   # test rapido
  python scripts/train_ppo.py --config configs/config_ppo.yaml --common configs/config_common.yaml

PPO usa CnnPolicy di SB3 (NatureCNN equivalente a CNNBackbone):
  Conv2d(4,32) → Conv2d(32,64) → Conv2d(64,64) → Linear(512)
  → Actor head (azioni)  [distribuzione categorica stocastica]
  → Critic head (V(s))   [stima del valore dello stato]

Differenza da DQN: on-policy, nessun replay buffer, esplorazione stocastica.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import gymnasium as gym
import ale_py

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import BaseCallback

# Registra namespace ALE prima che SB3 chiami gym.make internamente
gym.register_envs(ale_py)

from src.utils.config import load_config
from src.utils.seeding import set_global_seed
from src.agents.ppo_agent import PPOAgent


class StandardMetricsCallback(BaseCallback):
    """
    Callback SB3 che remappa i tag interni nei tag standard del progetto:
      rollout/ep_rew_mean  → train/mean_reward
      rollout/ep_len_mean  → train/episode_length
      train/policy_loss    → train/policy_loss   (invariato)
      train/value_loss     → train/value_loss    (invariato)
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._max_reward = float("-inf")

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        logs = self.logger.name_to_value
        step = self.num_timesteps

        if "rollout/ep_rew_mean" in logs:
            mean_rew = logs["rollout/ep_rew_mean"]
            self._max_reward = max(self._max_reward, mean_rew)
            self.logger.record("train/mean_reward", mean_rew)
            self.logger.record("train/max_reward", self._max_reward)

        if "rollout/ep_len_mean" in logs:
            self.logger.record("train/episode_length", logs["rollout/ep_len_mean"])


def make_ppo_env(env_id: str, n_envs: int, seed: int):
    """
    Crea VecEnv con wrapper Atari + FrameStack(4) via SB3.

    Stesso preprocessing di DQN (AtariWrapper applicato internamente da make_atari_env):
      NoopReset → MaxAndSkip(4) → EpisodicLife → FireReset
      → WarpFrame(84x84 gray) → ClipReward → FrameStack(4)
    """
    vec_env = make_atari_env(env_id, n_envs=n_envs, seed=seed)
    vec_env = VecFrameStack(vec_env, n_stack=4)
    return vec_env


def evaluate_greedy(model: PPO, env_id: str, seed: int, n_episodes: int) -> list:
    """Valuta policy PPO greedy su n_episodes episodi (env singolo)."""
    eval_env = make_ppo_env(env_id, n_envs=1, seed=seed)
    rewards = []
    for ep in range(n_episodes):
        obs = eval_env.reset()
        ep_reward = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done_arr, _ = eval_env.step(action)
            ep_reward += float(reward[0])
            done = bool(done_arr[0])
        rewards.append(ep_reward)
    eval_env.close()
    return rewards


def linear_schedule(initial_value: float):
    """Schedule lineare decrescente da initial_value a 0."""
    def schedule(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return schedule


def main():
    parser = argparse.ArgumentParser(description="Training PPO su Atari Space Invaders")
    parser.add_argument("--config", default="configs/config_ppo.yaml")
    parser.add_argument("--common", default="configs/config_common.yaml")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda", "mps"])
    parser.add_argument("--timesteps", type=int, default=None, help="Override total_timesteps")
    args = parser.parse_args()

    cfg_ppo = load_config(args.config)
    cfg_common = load_config(args.common)

    seed = cfg_common.seed
    env_id = cfg_common.env.id
    eval_episodes = cfg_common.eval.episodes
    results_dir = Path(cfg_common.eval.results_dir)
    tb_dir = cfg_common.logging.tensorboard_dir

    ppo = cfg_ppo.ppo
    total_timesteps = args.timesteps if args.timesteps is not None else ppo.total_timesteps
    device = args.device or cfg_ppo.model.device

    set_global_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"[PPO] Env: {env_id} | Seed: {seed} | Device: {device} | Steps: {total_timesteps:,}")
    print(f"[PPO] n_envs: {ppo.n_envs} | n_steps: {ppo.n_steps} | batch_size: {ppo.batch_size}")

    vec_env = make_ppo_env(env_id, n_envs=ppo.n_envs, seed=seed)

    lr = linear_schedule(ppo.learning_rate) if ppo.lr_schedule == "linear" else ppo.learning_rate

    model = PPO(
        policy="CnnPolicy",
        env=vec_env,
        n_steps=ppo.n_steps,
        n_epochs=ppo.n_epochs,
        batch_size=ppo.batch_size,
        learning_rate=lr,
        clip_range=ppo.clip_range,
        gamma=ppo.gamma,
        gae_lambda=ppo.gae_lambda,
        ent_coef=ppo.ent_coef,
        vf_coef=ppo.vf_coef,
        tensorboard_log=str(Path(tb_dir) / "ppo"),
        device=device,
        seed=seed,
        verbose=1,
    )

    callback = StandardMetricsCallback()

    start_time = time.time()
    print("[PPO] Inizio training...")

    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True,
    )

    training_time = time.time() - start_time
    vec_env.close()
    print(f"[PPO] Training completato in {training_time:.1f}s")

    # Salva modello
    model_path = results_dir / "ppo_model"
    model.save(str(model_path))
    print(f"[PPO] Modello salvato: {model_path}.zip")

    # Evaluation greedy post-training
    print(f"[PPO] Evaluation greedy su {eval_episodes} episodi...")
    eval_rewards = evaluate_greedy(model, env_id, seed=seed + 10000, n_episodes=eval_episodes)

    results = {
        "agent": "ppo",
        "mean_reward": float(np.mean(eval_rewards)),
        "std_reward": float(np.std(eval_rewards)),
        "max_reward": float(np.max(eval_rewards)),
        "total_timesteps": total_timesteps,
        "training_time_seconds": training_time,
        "eval_episodes": eval_episodes,
        "seed": seed,
    }

    results_path = results_dir / "ppo_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[PPO] Risultati salvati: {results_path}")
    print(f"[PPO] mean_reward={results['mean_reward']:.1f} ± {results['std_reward']:.1f}")


if __name__ == "__main__":
    main()
