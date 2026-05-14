"""
Training PPO da zero su Space Invaders (Atari).

Uso:
  python scripts/train_ppo.py
  python scripts/train_ppo.py --timesteps 10000 --device cpu   # test rapido
  python scripts/train_ppo.py --config configs/config_ppo.yaml --common configs/config_common.yaml

Loop principale:
  obs = vec_env.reset()
  while step < total_timesteps:
    obs, ep_rewards = agent.collect_rollouts(vec_env, obs)   ← π_old, n_steps × n_envs
    metrics = agent.update()                                  ← n_epochs PPO-Clip
    aggiorna LR (schedule lineare)
    log metrics su TensorBoard

Preprocessing (identico a DQN per comparabilità):
  NoopReset → MaxAndSkip(4) → EpisodicLife → FireReset
  → WarpFrame(84×84 gray) → ClipReward → FrameStack(4) → transpose (C,H,W)
  Output shape: (n_envs, 4, 84, 84) uint8
VecEnv: custom DummyVecEnv (no SB3 dependency — sequential, same process).
"""

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

from src.utils.config import load_config
from src.utils.seeding import set_global_seed
from src.agents.ppo_agent import PPOAgent
from src.training.tb_logger import TBLogger
from src.envs.vec_env import make_ppo_vec_env, make_eval_vec_env


def evaluate_greedy(agent: PPOAgent, env_id: str, seed: int, n_episodes: int) -> list:
    """Valuta policy greedy (deterministica) su n_episodes episodi singoli."""
    eval_env = make_eval_vec_env(env_id, seed=seed)
    obs = eval_env.reset()
    rewards = []
    for _ in range(n_episodes):
        ep_reward = 0.0
        done = False
        while not done:
            action = agent.act(np.asarray(obs)[0])
            obs, reward, done_arr, _ = eval_env.step(np.array([action]))
            ep_reward += float(reward[0])
            done = bool(done_arr[0])
        rewards.append(ep_reward)
        # DummyVecEnv auto-resets on done; obs already holds first frame of next ep
    eval_env.close()
    return rewards


def main():
    parser = argparse.ArgumentParser(description="Training PPO da zero su Atari Space Invaders")
    parser.add_argument("--config", default="configs/config_ppo.yaml")
    parser.add_argument("--common", default="configs/config_common.yaml")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda", "mps"])
    parser.add_argument("--timesteps", type=int, default=None)
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
    feature_dim = getattr(cfg_ppo.model, "feature_dim", 512)

    set_global_seed(seed)
    results_dir.mkdir(parents=True, exist_ok=True)

    steps_per_iter = ppo.n_envs * ppo.n_steps
    print(f"[PPO] Env: {env_id} | Seed: {seed} | Device: {device} | Steps: {total_timesteps:,}")
    print(f"[PPO] n_envs={ppo.n_envs} n_steps={ppo.n_steps} → {steps_per_iter} step/iter")
    print(f"[PPO] n_epochs={ppo.n_epochs} batch_size={ppo.batch_size}")

    vec_env = make_ppo_vec_env(env_id, n_envs=ppo.n_envs, seed=seed)
    tb_logger = TBLogger(log_dir=str(Path(tb_dir) / "ppo"))

    agent = PPOAgent(
        n_actions=vec_env.action_space.n,
        n_envs=ppo.n_envs,
        n_steps=ppo.n_steps,
        n_epochs=ppo.n_epochs,
        batch_size=ppo.batch_size,
        learning_rate=ppo.learning_rate,
        clip_range=ppo.clip_range,
        gamma=ppo.gamma,
        gae_lambda=ppo.gae_lambda,
        ent_coef=ppo.ent_coef,
        vf_coef=ppo.vf_coef,
        device=device,
        feature_dim=feature_dim,
    )

    obs = vec_env.reset()
    total_steps = 0
    iteration = 0
    ep_rewards_window = deque(maxlen=100)
    max_reward_seen = float("-inf")

    start_time = time.time()
    print("[PPO] Inizio training...")

    pbar = tqdm(
        total=total_timesteps,
        desc="PPO",
        unit="step",
        dynamic_ncols=True,
        smoothing=0.05,
    )

    while total_steps < total_timesteps:
        obs, ep_rewards = agent.collect_rollouts(vec_env, obs)
        total_steps += steps_per_iter
        iteration += 1

        # Schedule LR lineare (decrescente a 0)
        if ppo.lr_schedule == "linear":
            frac = max(0.0, 1.0 - total_steps / total_timesteps)
            for pg in agent.optimizer.param_groups:
                pg["lr"] = ppo.learning_rate * frac

        metrics = agent.update()

        if ep_rewards:
            ep_rewards_window.extend(ep_rewards)
            max_reward_seen = max(max_reward_seen, max(ep_rewards))

        mean_reward = float(np.mean(ep_rewards_window)) if ep_rewards_window else float("nan")
        pbar.update(steps_per_iter)
        pbar.set_postfix(
            {
                "mean100": f"{mean_reward:.1f}",
                "max": f"{max_reward_seen:.0f}",
                "pi": f"{metrics['policy_loss']:.4f}",
                "vf": f"{metrics['value_loss']:.4f}",
                "ent": f"{metrics['entropy']:.3f}",
            },
            refresh=False,
        )

        if ep_rewards_window:
            mean_r = float(np.mean(ep_rewards_window))
            tb_logger.log_episode(step=total_steps, reward=mean_r, length=0, max_reward=max_reward_seen)

        tb_logger.log_policy_loss(total_steps, metrics["policy_loss"])
        tb_logger.log_value_loss(total_steps, metrics["value_loss"])
        tb_logger.writer.add_scalar("train/entropy", metrics["entropy"], total_steps)
        tb_logger.writer.add_scalar("train/approx_kl", metrics["approx_kl"], total_steps)

    pbar.close()
    vec_env.close()
    training_time = time.time() - start_time
    tb_logger.log_training_time(training_time)
    print(f"[PPO] Training completato in {training_time:.1f}s")

    model_path = results_dir / "ppo_model.pth"
    agent.save(str(model_path))
    print(f"[PPO] Modello salvato: {model_path}")

    print(f"[PPO] Evaluation greedy su {eval_episodes} episodi...")
    eval_rewards = evaluate_greedy(agent, env_id, seed=seed + 10000, n_episodes=eval_episodes)

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

    tb_logger.log_eval(
        step=total_timesteps,
        mean_reward=results["mean_reward"],
        std_reward=results["std_reward"],
        max_reward=results["max_reward"],
    )
    tb_logger.flush()
    tb_logger.close()

    print(f"[PPO] Risultati salvati: {results_path}")
    print(f"[PPO] mean_reward={results['mean_reward']:.1f} ± {results['std_reward']:.1f}")


if __name__ == "__main__":
    main()
