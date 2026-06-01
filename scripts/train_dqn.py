"""
Training DQN su Space Invaders (Atari).

Uso:
  python scripts/train_dqn.py
  python scripts/train_dqn.py --timesteps 1000 --device cpu   # test rapido
  python scripts/train_dqn.py --config configs/config_dqn.yaml --common configs/config_common.yaml
"""

import argparse
import json
import sys
import time
import random
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import gymnasium as gym
import ale_py
from tqdm import tqdm

from src.utils.config import load_config
from src.utils.seeding import set_global_seed
from src.agents.dqn_agent import DQNAgent
from src.training.replay_buffer import ReplayBuffer
from src.training.tb_logger import TBLogger
from src.envs.atari_wrappers import make_atari_env

gym.register_envs(ale_py)


def obs_to_numpy(obs) -> np.ndarray:
    """Converte l'osservazione della pipeline unificata in np.ndarray."""
    return np.asarray(obs)


def evaluate_greedy(
    agent: DQNAgent,
    env_id: str,
    seed: int,
    n_episodes: int,
    preprocessing_kwargs: dict,
    masking_kwargs: dict,
) -> list:
    """Valuta agente greedy (no epsilon) per n_episodes episodi."""
    agent.set_training_mode(False)
    rewards = []
    for ep in range(n_episodes):
        env = make_atari_env(
            env_id,
            seed=seed + ep,
            preprocessing_kwargs=preprocessing_kwargs,
            masking_kwargs=masking_kwargs,
        )
        obs, _ = env.reset()
        ep_reward = 0.0
        done = False
        while not done:
            action = agent.act(obs_to_numpy(obs))
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        rewards.append(ep_reward)
        env.close()
    agent.set_training_mode(True)
    return rewards


def main():
    ## Load configs
    parser = argparse.ArgumentParser(description="Training DQN su Atari Space Invaders")
    parser.add_argument("--config", default="configs/config_dqn.yaml")
    parser.add_argument("--common", default="configs/config_common.yaml")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda", "mps"])
    parser.add_argument("--timesteps", type=int, default=None, help="Override total_timesteps")
    args = parser.parse_args()

    cfg_dqn = load_config(args.config)
    cfg_common = load_config(args.common)

    seed = cfg_common.seed
    env_id = cfg_common.env.id
    eval_episodes = cfg_common.eval.episodes
    results_dir = Path(cfg_common.eval.results_dir)
    tb_dir = cfg_common.logging.tensorboard_dir
    preprocessing_kwargs = cfg_common.preprocessing.to_dict()
    masking_kwargs = cfg_common.masking.to_dict()

    dqn = cfg_dqn.dqn
    total_timesteps = args.timesteps if args.timesteps is not None else dqn.total_timesteps
    device = args.device or cfg_dqn.model.device

    set_global_seed(seed)

    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"[DQN] Env: {env_id} | Seed: {seed} | Device: {device} | Steps: {total_timesteps:,}")

    # Start making env

    env = make_atari_env(
        env_id,
        seed=seed,
        preprocessing_kwargs=preprocessing_kwargs,
        masking_kwargs=masking_kwargs,
    )
    n_actions = env.action_space.n
    obs_shape = env.observation_space.shape

    agent = DQNAgent(
        n_actions=n_actions,
        feature_dim=cfg_dqn.model.feature_dim,
        learning_rate=dqn.learning_rate,
        gamma=dqn.gamma,
        epsilon_start=dqn.epsilon_start,
        epsilon_end=dqn.epsilon_end,
        epsilon_decay_steps=dqn.epsilon_decay_steps,
        device=device,
        in_channels=obs_shape[0],
    )

    buffer = ReplayBuffer(
        capacity=dqn.replay_buffer_size,
        obs_shape=obs_shape,
        device=device,
    )

    tb_logger = TBLogger(log_dir=str(Path(tb_dir) / "dqn"))

    obs, _ = env.reset(seed=seed)
    ep_reward = 0.0
    ep_length = 0
    ep_rewards_window = deque(maxlen=100)
    max_reward_seen = float("-inf")

    start_time = time.time()
    print("[DQN] Inizio training...")

    pbar = tqdm(
        range(1, total_timesteps + 1),
        desc="DQN",
        unit="step",
        dynamic_ncols=True,
        smoothing=0.05,
    )
    last_loss = float("nan")
    for step in pbar:
        obs_np = obs_to_numpy(obs)

        action = agent.act(obs_np)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        next_obs_np = obs_to_numpy(next_obs)
        buffer.push(obs_np, action, float(reward), next_obs_np, done)

        obs = next_obs
        ep_reward += reward
        ep_length += 1

        # Training
        if step >= dqn.learning_starts and step % dqn.train_freq == 0:
            if len(buffer) >= dqn.batch_size:
                losses = []
                gradient_steps = int(getattr(dqn, "gradient_steps", 1))
                for _ in range(gradient_steps):
                    batch = buffer.sample(dqn.batch_size)
                    info = agent.update(batch)
                    losses.append(info["loss"])
                last_loss = float(np.mean(losses))
                tb_logger.log_td_loss(step, last_loss)

        # Target network hard update
        if step % dqn.target_update_freq == 0:
            agent.update_target_network()

        # Epsilon decay
        agent.decay_epsilon(step)

        if step % 1000 == 0:
            tb_logger.log_epsilon(step, agent.epsilon)

        # Fine episodio
        if done:
            ep_rewards_window.append(ep_reward)
            mean_reward = float(np.mean(ep_rewards_window))
            max_reward_seen = max(max_reward_seen, ep_reward)

            tb_logger.log_episode(
                step=step,
                reward=mean_reward,
                length=ep_length,
                max_reward=max_reward_seen,
            )

            pbar.set_postfix(
                {
                    "mean100": f"{mean_reward:.1f}",
                    "max": f"{max_reward_seen:.0f}",
                    "eps": f"{agent.epsilon:.3f}",
                    "loss": f"{last_loss:.4f}" if last_loss == last_loss else "n/a",
                },
                refresh=False,
            )

            obs, _ = env.reset()
            ep_reward = 0.0
            ep_length = 0

    pbar.close()
    training_time = time.time() - start_time
    env.close()
    tb_logger.log_training_time(training_time)
    print(f"[DQN] Training completato in {training_time:.1f}s")

    # Salva modello
    model_path = results_dir / "dqn_model.pth"
    agent.save(str(model_path))
    print(f"[DQN] Modello salvato: {model_path}")

    # Evaluation greedy post-training
    print(f"[DQN] Evaluation greedy su {eval_episodes} episodi...")
    eval_rewards = evaluate_greedy(
        agent,
        env_id,
        seed=seed + 10000,
        n_episodes=eval_episodes,
        preprocessing_kwargs=preprocessing_kwargs,
        masking_kwargs=masking_kwargs,
    )

    results = {
        "agent": "dqn",
        "mean_reward": float(np.mean(eval_rewards)),
        "std_reward": float(np.std(eval_rewards)),
        "max_reward": float(np.max(eval_rewards)),
        "total_timesteps": total_timesteps,
        "training_time_seconds": training_time,
        "eval_episodes": eval_episodes,
        "seed": seed,
    }

    results_path = results_dir / "dqn_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    tb_logger.log_eval(
        step=total_timesteps,
        mean_reward=results["mean_reward"],
        std_reward=results["std_reward"],
        max_reward=results["max_reward"],
    )
    tb_logger.flush()
    tb_logger.close()

    print(f"[DQN] Risultati salvati: {results_path}")
    print(f"[DQN] mean_reward={results['mean_reward']:.1f} ± {results['std_reward']:.1f}")


if __name__ == "__main__":
    main()
