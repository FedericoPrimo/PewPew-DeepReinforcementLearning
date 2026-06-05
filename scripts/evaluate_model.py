r"""
Valuta un checkpoint DQN o PPO gia' addestrato su Space Invaders.

Esempi:
  python scripts/evaluate_model.py --checkpoint "models_checkpoints\Best-RGB-optimized\ppo\best_trial\ppo_model.pth"
  python scripts/evaluate_model.py --checkpoint results/dqn_model.pth --episodes 10
  python scripts/evaluate_model.py --checkpoint results/ppo_model.pth --episodes 10 --num-seeds 5
  python scripts/evaluate_model.py --model ppo --checkpoint results/ppo_model.pth --setting rgb
  python scripts/evaluate_model.py --checkpoint "models_checkpoints\Best-RGB-optimized\ppo\best_trial\ppo_model.pth" --episodes 30 --device cuda --num-seeds 10
  python scripts/evaluate_model.py --checkpoint "models_checkpoints\Best-RGB-optimized\dqn\best_trial\dqn_model.pth" --episodes 30 --device cuda --num-seeds 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import ale_py
import gymnasium as gym
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.dqn_agent import DQNAgent
from src.agents.ppo_agent import PPOAgent
from src.envs.atari_wrappers import make_atari_env
from src.envs.vec_env import make_eval_vec_env
from src.utils.config import load_config
from src.utils.seeding import set_global_seed

gym.register_envs(ale_py)

SETTINGS = ("rgb", "masked", "grayscale")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluation di checkpoint DQN/PPO.")
    parser.add_argument("--model", default=None, choices=["dqn", "ppo"], help="Se omesso, viene inferito dal path.")
    parser.add_argument("--checkpoint", required=True, help="Path del file .pth da testare.")
    parser.add_argument("--common", default="configs/config_common.yaml", help="Config base per env, seed e preprocessing.")
    parser.add_argument("--config-dqn", default="configs/config_dqn.yaml")
    parser.add_argument("--config-ppo", default="configs/config_ppo.yaml")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--device", default=None, choices=["cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=None, help="Seed iniziale. Default: common.seed + 10000.")
    parser.add_argument("--num-seeds", type=int, default=1, help="Numero di seed consecutivi da testare.")
    parser.add_argument(
        "--setting",
        default="all",
        choices=["all", *SETTINGS],
        help="Setting da testare. Default: all = rgb, masked, grayscale.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path JSON per un singolo setting, oppure cartella output quando --setting all.",
    )
    return parser.parse_args()


def infer_model(checkpoint: Path) -> str:
    parts = [part.lower() for part in checkpoint.parts]
    stem = checkpoint.stem.lower()
    if "dqn" in parts or "dqn" in stem:
        return "dqn"
    if "ppo" in parts or "ppo" in stem:
        return "ppo"
    raise ValueError("Non riesco a inferire il modello dal path. Usa --model dqn oppure --model ppo.")


def build_setting_kwargs(common_cfg, setting: str) -> tuple[dict, dict]:
    preprocessing_kwargs = common_cfg.preprocessing.to_dict()
    masking_kwargs = common_cfg.masking.to_dict()

    if setting == "rgb":
        preprocessing_kwargs["mode"] = "rgb"
        masking_kwargs["enabled"] = False
    elif setting == "masked":
        preprocessing_kwargs["mode"] = "rgb"
        masking_kwargs["enabled"] = True
    elif setting == "grayscale":
        preprocessing_kwargs["mode"] = "grayscale"
        masking_kwargs["enabled"] = False
    else:
        raise ValueError(f"Setting non valido: {setting}")

    return preprocessing_kwargs, masking_kwargs


def output_path_for(args: argparse.Namespace, checkpoint: Path, setting: str) -> Path:
    if args.output is None:
        return checkpoint.with_name(f"{checkpoint.stem}_eval_{setting}.json")

    output = Path(args.output)
    if args.setting == "all":
        return output / f"{checkpoint.stem}_eval_{setting}.json"
    return output


def make_single_env(env_id: str, seed: int, preprocessing_kwargs: dict, masking_kwargs: dict):
    return make_atari_env(
        env_id,
        seed=seed,
        preprocessing_kwargs=preprocessing_kwargs,
        masking_kwargs=masking_kwargs,
    )


def evaluate_dqn(
    args: argparse.Namespace,
    common_cfg,
    preprocessing_kwargs: dict,
    masking_kwargs: dict,
    eval_seed: int,
) -> list[float]:
    cfg_dqn = load_config(args.config_dqn)
    device = args.device or cfg_dqn.model.device

    probe_env = make_single_env(common_cfg.env.id, eval_seed, preprocessing_kwargs, masking_kwargs)
    obs_shape = probe_env.observation_space.shape
    n_actions = probe_env.action_space.n
    probe_env.close()

    agent = DQNAgent(
        n_actions=n_actions,
        feature_dim=cfg_dqn.model.feature_dim,
        learning_rate=cfg_dqn.dqn.learning_rate,
        gamma=cfg_dqn.dqn.gamma,
        epsilon_start=0.0,
        epsilon_end=0.0,
        epsilon_decay_steps=1,
        device=device,
        in_channels=obs_shape[0],
    )
    agent.load(args.checkpoint)
    agent.set_training_mode(False)

    rewards: list[float] = []
    env = make_single_env(common_cfg.env.id, eval_seed, preprocessing_kwargs, masking_kwargs)
    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action = agent.act(np.asarray(obs))
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += float(reward)
            done = terminated or truncated
        rewards.append(ep_reward)
        print(f"[eval][DQN][seed={eval_seed}] episode {ep + 1}/{args.episodes}: reward={ep_reward:.1f}")
    env.close()
    return rewards


def evaluate_ppo(
    args: argparse.Namespace,
    common_cfg,
    preprocessing_kwargs: dict,
    masking_kwargs: dict,
    eval_seed: int,
) -> list[float]:
    cfg_ppo = load_config(args.config_ppo)
    device = args.device or cfg_ppo.model.device

    eval_env = make_eval_vec_env(
        common_cfg.env.id,
        seed=eval_seed,
        preprocessing_kwargs=preprocessing_kwargs,
        masking_kwargs=masking_kwargs,
    )
    obs_shape = eval_env.observation_space.shape
    agent = PPOAgent(
        n_actions=eval_env.action_space.n,
        n_envs=1,
        n_steps=cfg_ppo.ppo.n_steps,
        n_epochs=cfg_ppo.ppo.n_epochs,
        batch_size=cfg_ppo.ppo.batch_size,
        learning_rate=cfg_ppo.ppo.learning_rate,
        clip_range=cfg_ppo.ppo.clip_range,
        gamma=cfg_ppo.ppo.gamma,
        gae_lambda=cfg_ppo.ppo.gae_lambda,
        ent_coef=cfg_ppo.ppo.ent_coef,
        vf_coef=cfg_ppo.ppo.vf_coef,
        clip_range_vf=getattr(cfg_ppo.ppo, "clip_range_vf", None),
        normalize_advantage=getattr(cfg_ppo.ppo, "normalize_advantage", True),
        max_grad_norm=getattr(cfg_ppo.ppo, "max_grad_norm", 0.5),
        device=device,
        feature_dim=getattr(cfg_ppo.model, "feature_dim", 512),
        obs_shape=obs_shape,
    )
    agent.load(args.checkpoint)
    agent.net.eval()

    obs = eval_env.reset()
    rewards: list[float] = []
    for ep in range(args.episodes):
        done = False
        ep_reward = 0.0
        while not done:
            action = agent.act(np.asarray(obs)[0])
            obs, reward, done_arr, _ = eval_env.step(np.array([action]))
            ep_reward += float(reward[0])
            done = bool(done_arr[0])
        rewards.append(ep_reward)
        print(f"[eval][PPO][seed={eval_seed}] episode {ep + 1}/{args.episodes}: reward={ep_reward:.1f}")
    eval_env.close()
    return rewards


def main() -> int:
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint non trovato: {checkpoint}")

    args.model = args.model or infer_model(checkpoint)
    common_cfg = load_config(args.common)
    args.episodes = args.episodes or common_cfg.eval.episodes
    seed_start = args.seed if args.seed is not None else common_cfg.seed + 10000
    if args.num_seeds < 1:
        raise ValueError("--num-seeds deve essere >= 1")
    eval_seeds = [seed_start + index for index in range(args.num_seeds)]
    settings = SETTINGS if args.setting == "all" else (args.setting,)
    all_results = []

    print(f"[eval] model={args.model} checkpoint={checkpoint}")
    print(f"[eval] settings={', '.join(settings)} episodes={args.episodes} seeds={eval_seeds}")

    for setting in settings:
        print("\n" + "=" * 70)
        print(f"[eval] setting: {setting}")
        print("=" * 70)
        preprocessing_kwargs, masking_kwargs = build_setting_kwargs(common_cfg, setting)
        per_seed_results = []
        all_rewards: list[float] = []

        for eval_seed in eval_seeds:
            print(f"\n[eval] seed: {eval_seed}")
            set_global_seed(eval_seed)
            if args.model == "dqn":
                rewards = evaluate_dqn(args, common_cfg, preprocessing_kwargs, masking_kwargs, eval_seed)
            else:
                rewards = evaluate_ppo(args, common_cfg, preprocessing_kwargs, masking_kwargs, eval_seed)

            all_rewards.extend(rewards)
            per_seed_results.append(
                {
                    "seed": eval_seed,
                    "episodes": args.episodes,
                    "mean_reward": float(np.mean(rewards)),
                    "std_reward": float(np.std(rewards)),
                    "min_reward": float(np.min(rewards)),
                    "max_reward": float(np.max(rewards)),
                    "episode_rewards": rewards,
                }
            )

        result = {
            "agent": args.model,
            "setting": setting,
            "checkpoint": str(checkpoint),
            "common_config": args.common,
            "preprocessing": preprocessing_kwargs,
            "masking": masking_kwargs,
            "episodes_per_seed": args.episodes,
            "num_seeds": args.num_seeds,
            "seeds": eval_seeds,
            "total_episodes": len(all_rewards),
            "mean_reward": float(np.mean(all_rewards)),
            "std_reward": float(np.std(all_rewards)),
            "min_reward": float(np.min(all_rewards)),
            "max_reward": float(np.max(all_rewards)),
            "episode_rewards": all_rewards,
            "per_seed": per_seed_results,
        }

        output = output_path_for(args, checkpoint, setting)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)

        all_results.append((setting, result, output))
        print(f"[eval] {setting}: mean_reward={result['mean_reward']:.1f} +/- {result['std_reward']:.1f}")
        print(f"[eval] {setting}: max_reward={result['max_reward']:.1f}")
        print(f"[eval] {setting}: risultati={output}")

    print("\n" + "=" * 70)
    print("[eval] riepilogo")
    print("=" * 70)
    for setting, result, output in all_results:
        print(
            f"{setting:<10} mean={result['mean_reward']:>7.1f} "
            f"std={result['std_reward']:>7.1f} max={result['max_reward']:>7.1f} "
            f"seeds={result['num_seeds']} episodes={result['total_episodes']} -> {output}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
