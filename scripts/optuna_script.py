"""
Hyperparameter tuning con Optuna per DQN e PPO.

Esempi:
  python scripts/optuna_script.py --model dqn --n-trials 20 --timesteps 500000 --device cpu
  python scripts/optuna_script.py --model ppo --n-trials 20 --timesteps 500000 --study-name ppo_v1
  python scripts/optuna_script.py --model dqn --n-trials 10 --dry-run

Lo script riusa i trainer esistenti:
  scripts/train_dqn.py
  scripts/train_ppo.py

Search space ispirato a:
  Eimer, Lindauer, Raileanu (2023), "Hyperparameters in Reinforcement
  Learning and How To Tune Them", Appendix D.3, Table 4.

Per ogni trial crea:
  results/optuna/<model>/<study_name>/trial_XXXX/config.yaml
  results/optuna/<model>/<study_name>/trial_XXXX/common_resolved.yaml
  logs/optuna/<model>/<study_name>/trial_XXXX/

La metrica ottimizzata e' mean_reward letta dal JSON finale del trainer.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config
import optuna


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ottimizza iperparametri DQN/PPO con Optuna.")
    parser.add_argument("--model", required=True, choices=["dqn", "ppo"])
    parser.add_argument("--base-config", default=None, help="Config modello di partenza.")
    parser.add_argument("--common", default="configs/config_common.yaml", help="Config comune di base.")
    parser.add_argument("--study-name", default=None, help="Nome study. Default: <model>_optuna.")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument(
        "--timesteps",
        type=int,
        default=500000,
        help="Override total_timesteps per ogni trial. Default: 500000.",
    )
    parser.add_argument("--device", default=None, choices=["cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=42, help="Seed del sampler Optuna.")
    parser.add_argument("--timeout", type=int, default=None, help="Timeout complessivo in secondi.")
    parser.add_argument("--metric", default="mean_reward", choices=["mean_reward", "max_reward"])
    parser.add_argument("--direction", default="maximize", choices=["maximize", "minimize"])
    parser.add_argument("--resume", action="store_true", help="Riusa lo study SQLite se esiste.")
    parser.add_argument("--dry-run", action="store_true", help="Genera/stampa un trial senza lanciare training.")
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_yaml_dict(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"YAML non valido: {path}")
    return data


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def build_common(base_common: dict[str, Any], results_dir: Path, tensorboard_dir: Path) -> dict[str, Any]:
    common = copy.deepcopy(base_common)
    common["eval"] = dict(common.get("eval", {}))
    common["logging"] = dict(common.get("logging", {}))
    common["eval"]["results_dir"] = str(results_dir)
    common["logging"]["tensorboard_dir"] = str(tensorboard_dir)
    return common


def resolve_total_timesteps(model_config: dict[str, Any], override: int | None) -> int:
    if override is not None:
        return override
    return int(model_config["total_timesteps"])


def suggest_dqn_config(
    trial: Any,
    base_config: dict[str, Any],
    timesteps: int | None,
) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    dqn = dict(config.get("dqn", {}))
    total_timesteps = resolve_total_timesteps(dqn, timesteps)

    dqn["learning_rate"] = trial.suggest_float("learning_rate", 1e-6, 1e-1, log=True)
    dqn["batch_size"] = trial.suggest_categorical("batch_size", [4, 8, 16, 32])
    dqn["learning_starts"] = trial.suggest_int("learning_starts", 0, 10000)
    dqn["train_freq"] = trial.suggest_int("train_freq", 1, 1000)
    dqn["gradient_steps"] = trial.suggest_int("gradient_steps", 1, 10)
    dqn["epsilon_start"] = trial.suggest_float("exploration_initial_eps", 0.5, 1.0)
    dqn["epsilon_end"] = trial.suggest_float("exploration_final_eps", 0.001, 0.2)
    # Il paper usa un range fino a 5e7, ma questo ReplayBuffer prealloca
    # obs e next_obs in RAM. Sopra 100k diventa facilmente ingestibile.
    dqn["replay_buffer_size"] = trial.suggest_int("buffer_size", 5000, 100000, log=True)

    exploration_fraction = trial.suggest_float("exploration_fraction", 0.005, 0.5)
    dqn["epsilon_decay_steps"] = max(1, int(total_timesteps * exploration_fraction))

    config["dqn"] = dqn
    return config


def suggest_ppo_config(trial: Any, base_config: dict[str, Any]) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    ppo = dict(config.get("ppo", {}))
    n_envs = int(ppo.get("n_envs", 1))

    n_steps = trial.suggest_categorical("n_steps", [256, 512, 1024, 2048, 4096])
    rollout_size = n_envs * int(n_steps)
    valid_batch_sizes = [value for value in [16, 32, 64, 128] if value <= rollout_size]

    ppo["n_steps"] = n_steps
    ppo["batch_size"] = trial.suggest_categorical("batch_size", valid_batch_sizes)
    ppo["n_epochs"] = trial.suggest_int("n_epochs", 5, 20)
    ppo["learning_rate"] = trial.suggest_float("learning_rate", 1e-6, 1e-1, log=True)
    ppo["ent_coef"] = trial.suggest_float("ent_coef", 0.0, 0.5)
    ppo["gae_lambda"] = trial.suggest_float("gae_lambda", 0.8, 0.9999)
    ppo["clip_range"] = trial.suggest_float("clip_range", 0.0, 0.5)
    ppo["clip_range_vf"] = trial.suggest_float("clip_range_vf", 0.0, 0.5)
    ppo["normalize_advantage"] = trial.suggest_categorical("normalize_advantage", [True, False])
    ppo["vf_coef"] = trial.suggest_float("vf_coef", 0.0, 1.0)
    ppo["max_grad_norm"] = trial.suggest_float("max_grad_norm", 0.0, 1.0)

    config["ppo"] = ppo
    return config


def run_training(
    model: str,
    config_path: Path,
    common_path: Path,
    device: str | None,
    timesteps: int | None,
    dry_run: bool,
) -> int:
    train_script = PROJECT_ROOT / "scripts" / f"train_{model}.py"
    command = [
        sys.executable,
        str(train_script),
        "--config",
        str(config_path),
        "--common",
        str(common_path),
    ]
    if device is not None:
        command.extend(["--device", device])
    if timesteps is not None:
        command.extend(["--timesteps", str(timesteps)])

    print(f"[OPTUNA] Command: {' '.join(command)}")
    if dry_run:
        return 0

    completed = subprocess.run(command, cwd=PROJECT_ROOT)
    return completed.returncode


def read_metric(model: str, results_dir: Path, metric: str) -> float:
    results_path = results_dir / f"{model}_results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"Risultati non trovati: {results_path}")

    with results_path.open("r", encoding="utf-8") as handle:
        results = json.load(handle)

    if metric not in results:
        raise KeyError(f"Metrica '{metric}' assente in {results_path}")
    return float(results[metric])


def make_objective(
    model: str,
    base_config: dict[str, Any],
    base_common: dict[str, Any],
    study_results_root: Path,
    study_logs_root: Path,
    device: str | None,
    timesteps: int | None,
    metric: str,
    dry_run: bool,
):
    def objective(trial: Any) -> float:
        trial_name = f"trial_{trial.number:04d}"
        trial_results_dir = study_results_root / trial_name
        trial_logs_dir = study_logs_root / trial_name

        if trial_results_dir.exists():
            shutil.rmtree(trial_results_dir)
        if trial_logs_dir.exists():
            shutil.rmtree(trial_logs_dir)
        trial_results_dir.mkdir(parents=True, exist_ok=False)
        trial_logs_dir.mkdir(parents=True, exist_ok=False)

        if model == "dqn":
            trial_config = suggest_dqn_config(trial, base_config, timesteps)
        else:
            trial_config = suggest_ppo_config(trial, base_config)

        if timesteps is not None:
            trial_config[model]["total_timesteps"] = timesteps

        if device is not None:
            trial_config.setdefault("model", {})
            trial_config["model"]["device"] = device

        trial_common = build_common(base_common, trial_results_dir, trial_logs_dir)
        trial_config_path = trial_results_dir / "config.yaml"
        trial_common_path = trial_results_dir / "common_resolved.yaml"
        write_yaml(trial_config_path, trial_config)
        write_yaml(trial_common_path, trial_common)

        print(f"[OPTUNA] Trial {trial.number}: {trial.params}")
        print(f"[OPTUNA] Results dir: {trial_results_dir}")

        return_code = run_training(
            model=model,
            config_path=trial_config_path,
            common_path=trial_common_path,
            device=device,
            timesteps=timesteps,
            dry_run=dry_run,
        )
        if return_code != 0:
            raise RuntimeError(f"Training fallito nel trial {trial.number} con exit code {return_code}")

        if dry_run:
            print("[OPTUNA] Dry-run completato. Ritorno metrica fittizia 0.0.")
            return 0.0

        value = read_metric(model, trial_results_dir, metric)
        trial.set_user_attr("results_dir", str(trial_results_dir))
        trial.set_user_attr("logs_dir", str(trial_logs_dir))
        print(f"[OPTUNA] Trial {trial.number} -> {metric}={value:.4f}")
        return value

    return objective


def default_base_config(model: str) -> Path:
    return PROJECT_ROOT / "configs" / f"config_{model}.yaml"


def main() -> int:
    args = parse_args()

    base_config_path = resolve_path(args.base_config) if args.base_config else default_base_config(args.model)
    common_path = resolve_path(args.common)
    if not base_config_path.exists():
        raise FileNotFoundError(f"Config modello non trovata: {base_config_path}")
    if not common_path.exists():
        raise FileNotFoundError(f"Config comune non trovata: {common_path}")

    base_config = load_yaml_dict(base_config_path)
    base_common = load_yaml_dict(common_path)
    common_cfg = load_config(common_path)

    study_name = args.study_name or f"{args.model}_optuna"
    study_results_root = PROJECT_ROOT / common_cfg.eval.results_dir / "optuna" / args.model / study_name
    study_logs_root = PROJECT_ROOT / common_cfg.logging.tensorboard_dir / "optuna" / args.model / study_name
    study_results_root.mkdir(parents=True, exist_ok=True)
    study_logs_root.mkdir(parents=True, exist_ok=True)

    storage_path = study_results_root / "study.db"
    storage_url = f"sqlite:///{storage_path.as_posix()}"

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(
        study_name=study_name,
        direction=args.direction,
        sampler=sampler,
        storage=storage_url,
        load_if_exists=args.resume,
    )

    print(f"[OPTUNA] Model: {args.model}")
    print(f"[OPTUNA] Base config: {base_config_path}")
    print(f"[OPTUNA] Study: {study_name}")
    print(f"[OPTUNA] Storage: {storage_path}")
    print(f"[OPTUNA] Trials richiesti: {args.n_trials}")

    objective = make_objective(
        model=args.model,
        base_config=base_config,
        base_common=base_common,
        study_results_root=study_results_root,
        study_logs_root=study_logs_root,
        device=args.device,
        timesteps=args.timesteps,
        metric=args.metric,
        dry_run=args.dry_run,
    )

    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout, catch=(RuntimeError,))

    print("[OPTUNA] Ottimizzazione completata.")
    print(f"[OPTUNA] Best value: {study.best_value:.4f}")
    print(f"[OPTUNA] Best trial: {study.best_trial.number}")
    print(f"[OPTUNA] Best params: {study.best_params}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
