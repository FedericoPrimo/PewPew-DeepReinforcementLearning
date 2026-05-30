"""
Orchestratore per lanciare sweep di configurazioni DQN o PPO.

Uso:
  python scripts/orchestrator.py --model dqn --grid-dir configs/grid_dqn
  python scripts/orchestrator.py --model ppo --grid-dir configs/ppo --device cpu
  python scripts/orchestrator.py --model dqn --grid-dir configs/grid_dqn --timesteps 10000 --dry-run

Per evitare conflitti tra artefatti, ogni sweep crea:
  results/sweeps/<model>/<grid_name>/<timestamp>/<config_name>/
  logs/sweeps/<model>/<grid_name>/<timestamp>/<config_name>/

In ogni run viene scritto anche un file `common_resolved.yaml` con i path
effettivi usati dal trainer.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lancia una sweep di configurazioni DQN o PPO.")
    parser.add_argument("--model", required=True, choices=["dqn", "ppo"])
    parser.add_argument("--grid-dir", required=True, help="Cartella con i file YAML della grid search.")
    parser.add_argument("--common", default="configs/config_common.yaml", help="Config comune di base.")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda", "mps"])
    parser.add_argument("--timesteps", type=int, default=None, help="Override total_timesteps per tutte le run.")
    parser.add_argument(
        "--sweep-name",
        default=None,
        help="Nome opzionale della sweep. Se assente usa il nome della cartella grid.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Stampa i comandi e i path risolti senza eseguire il training.",
    )
    return parser.parse_args()


def load_common_dict(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config comune non valida: {path}")
    return data


def build_run_common(
    base_common: dict,
    results_dir: Path,
    tensorboard_dir: Path,
) -> dict:
    common = dict(base_common)
    common["eval"] = dict(base_common.get("eval", {}))
    common["logging"] = dict(base_common.get("logging", {}))
    common["eval"]["results_dir"] = str(results_dir)
    common["logging"]["tensorboard_dir"] = str(tensorboard_dir)
    return common


def ensure_yaml_files(grid_dir: Path) -> list[Path]:
    configs = sorted(p for p in grid_dir.glob("*.yaml") if p.is_file())
    if not configs:
        raise FileNotFoundError(f"Nessun file YAML trovato in {grid_dir}")
    return configs


def run_config(
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

    print(f"[ORCH] Run: {config_path.stem}")
    print(f"[ORCH] Command: {' '.join(command)}")
    if dry_run:
        print(f"[ORCH] Dry-run: comando validato per {config_path.stem}")
        return 0

    completed = subprocess.run(command, cwd=PROJECT_ROOT)
    return completed.returncode


def main() -> int:
    args = parse_args()

    grid_dir = (PROJECT_ROOT / args.grid_dir).resolve() if not Path(args.grid_dir).is_absolute() else Path(args.grid_dir)
    common_path = (PROJECT_ROOT / args.common).resolve() if not Path(args.common).is_absolute() else Path(args.common)

    if not grid_dir.exists():
        raise FileNotFoundError(f"Grid directory non trovata: {grid_dir}")
    if not common_path.exists():
        raise FileNotFoundError(f"Config comune non trovata: {common_path}")

    config_files = ensure_yaml_files(grid_dir)
    base_common = load_common_dict(common_path)
    common_cfg = load_config(common_path)

    sweep_name = args.sweep_name or grid_dir.name
    sweep_root_results = PROJECT_ROOT / common_cfg.eval.results_dir / "sweeps" / args.model / sweep_name
    sweep_root_logs = PROJECT_ROOT / common_cfg.logging.tensorboard_dir / "sweeps" / args.model / sweep_name
    if not args.dry_run:
        sweep_root_results.mkdir(parents=True, exist_ok=True)
        sweep_root_logs.mkdir(parents=True, exist_ok=True)

    print(f"[ORCH] Model: {args.model}")
    print(f"[ORCH] Grid: {grid_dir}")
    print(f"[ORCH] Sweep root results: {sweep_root_results}")
    print(f"[ORCH] Sweep root logs: {sweep_root_logs}")
    print(f"[ORCH] Config trovate: {len(config_files)}")

    failures: list[tuple[str, int]] = []

    for index, config_path in enumerate(config_files, start=1):
        run_name = f"{index:02d}_{config_path.stem}"
        run_results_dir = sweep_root_results / run_name
        run_logs_dir = sweep_root_logs / run_name
        if not args.dry_run:
            if run_results_dir.exists():
                shutil.rmtree(run_results_dir)
            if run_logs_dir.exists():
                shutil.rmtree(run_logs_dir)
            run_results_dir.mkdir(parents=True, exist_ok=False)
            run_logs_dir.mkdir(parents=True, exist_ok=False)

        run_common = build_run_common(
            base_common=base_common,
            results_dir=run_results_dir,
            tensorboard_dir=run_logs_dir,
        )
        run_common_path = run_results_dir / "common_resolved.yaml"
        if not args.dry_run:
            with run_common_path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(run_common, handle, sort_keys=False)

        print(f"[ORCH] ({index}/{len(config_files)}) {config_path.name}")
        print(f"[ORCH] Results dir: {run_results_dir}")
        print(f"[ORCH] Logs dir: {run_logs_dir}")

        return_code = run_config(
            model=args.model,
            config_path=config_path,
            common_path=run_common_path,
            device=args.device,
            timesteps=args.timesteps,
            dry_run=args.dry_run,
        )
        if return_code != 0:
            failures.append((config_path.name, return_code))
            print(f"[ORCH] Run fallita: {config_path.name} (exit code {return_code})")
        else:
            if args.dry_run:
                print(f"[ORCH] Run validata: {config_path.name}")
            else:
                print(f"[ORCH] Run completata: {config_path.name}")

    print("[ORCH] Sweep completata.")
    print(f"[ORCH] Run totali: {len(config_files)} | Fallite: {len(failures)}")

    if failures:
        for config_name, code in failures:
            print(f"[ORCH] Failure -> {config_name}: exit code {code}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
