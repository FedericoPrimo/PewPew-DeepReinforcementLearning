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

from stable_baselines3.common.atari_wrappers import AtariWrapper

from src.utils.config import load_config, merge_configs
from src.utils.seeding import set_global_seed
from src.agents.dqn_agent import DQNAgent
from src.training.replay_buffer import ReplayBuffer
from src.training.tb_logger import TBLogger

gym.register_envs(ale_py)


def make_dqn_env(env_id: str, seed: int) -> gym.Env:
    """
    Crea env con wrapper SB3 Atari + FrameStack(4).

    Pipeline (identica per DQN e PPO, garantisce confronto equo):
      NoopReset → MaxAndSkip(4) → EpisodicLife → FireReset
      → WarpFrame(84x84 gray) → ClipReward → FrameStack(4)

    Output shape: (4, 84, 84) uint8 grayscale.
    """
    env = gym.make(env_id)
    env = AtariWrapper(env, clip_reward=True)
    # gymnasium 1.x: FrameStack rinominato FrameStackObservation
    env = gym.wrappers.FrameStackObservation(env, 4)
    return env


def obs_to_numpy(obs) -> np.ndarray:
    """Converte LazyFrames/array a (4, 84, 84) uint8."""
    arr = np.array(obs)
    # WarpFrame → (84,84,1), FrameStack → (4,84,84,1) o (4,84,84)
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr.squeeze(-1)   # (4,84,84)
    return arr.astype(np.uint8)


def evaluate_greedy(agent: DQNAgent, env_id: str, seed: int, n_episodes: int) -> list:
    """Valuta agente greedy (no epsilon) per n_episodes episodi."""
    agent.set_training_mode(False)
    rewards = []
    for ep in range(n_episodes):
        env = make_dqn_env(env_id, seed=seed + ep)
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

    dqn = cfg_dqn.dqn
    total_timesteps = args.timesteps if args.timesteps is not None else dqn.total_timesteps
    device = args.device or cfg_dqn.model.device

    set_global_seed(seed)

    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"[DQN] Env: {env_id} | Seed: {seed} | Device: {device} | Steps: {total_timesteps:,}")

    # Start making env

    env = make_dqn_env(env_id, seed)
    n_actions = env.action_space.n

    agent = DQNAgent(
        n_actions=n_actions,
        feature_dim=cfg_dqn.model.feature_dim,
        learning_rate=dqn.learning_rate,
        gamma=dqn.gamma,
        epsilon_start=dqn.epsilon_start,
        epsilon_end=dqn.epsilon_end,
        epsilon_decay_steps=dqn.epsilon_decay_steps,
        device=device,
    )

    buffer = ReplayBuffer(
        capacity=dqn.replay_buffer_size,
        obs_shape=(4, 84, 84),
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
        # Reduce the observation if it has the greyscale channel at the end which is useless (4x84x84x1)
        obs_np = obs_to_numpy(obs)
        # Obs now should be (4x84x84) which are 4 stacked frames reduced by the atari wrapper

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
                batch = buffer.sample(dqn.batch_size)
                info = agent.update(batch)
                last_loss = info["loss"]
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
    eval_rewards = evaluate_greedy(agent, env_id, seed=seed + 10000, n_episodes=eval_episodes)

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
