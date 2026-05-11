"""
Script principale: valutazione del CNNDummyAgent su Space Invaders.

Uso:
    python scripts/run_dummy.py
    python scripts/run_dummy.py --render
    python scripts/run_dummy.py --agent dummy          # usa DummyAgent senza CNN
    python scripts/run_dummy.py --agent cnn-dummy      # usa CNNDummyAgent (default)
    python scripts/run_dummy.py --episodes 5 --seeds 42 123
    python scripts/run_dummy.py --mode grayscale --masking
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config, merge_configs
from src.agents.dummy_agent import DummyAgent
from src.agents.cnn_dummy_agent import CNNDummyAgent
from src.models.cnn_backbone import build_cnn_from_config
from src.envs.make_env import make_env_from_config, get_action_space_size
from src.loops.evaluate import evaluate_agent
from src.logging.metrics_logger import MetricsLogger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Valuta un agente su Atari Space Invaders."
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--agent", type=str, choices=["dummy", "cnn-dummy"], default="cnn-dummy",
                        help="Agente da usare: dummy (senza CNN) o cnn-dummy (con CNN)")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--mode", type=str, choices=["rgb", "grayscale"], default=None)
    parser.add_argument("--masking", action="store_true", default=None)
    parser.add_argument("--output-format", type=str, choices=["csv", "json"], default=None)
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--render", action="store_true", help="Mostra il gioco a schermo")
    parser.add_argument("--log-level", type=str, choices=["DEBUG", "INFO", "WARNING"], default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("run_dummy")

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"File di configurazione non trovato: {config_path}")
        sys.exit(1)

    config = load_config(config_path)

    overrides = {}
    if args.episodes is not None:
        overrides.setdefault("evaluation", {})["num_episodes"] = args.episodes
    if args.seeds is not None:
        overrides.setdefault("evaluation", {})["seeds"] = args.seeds
    if args.mode is not None:
        overrides.setdefault("preprocessing", {})["mode"] = args.mode
    if args.masking:
        overrides.setdefault("masking", {})["enabled"] = True
    if args.output_format is not None:
        overrides.setdefault("evaluation", {})["results_format"] = args.output_format
    if args.results_dir is not None:
        overrides.setdefault("evaluation", {})["results_dir"] = args.results_dir
    if args.render:
        overrides.setdefault("env", {})["render_mode"] = "human"
    if overrides:
        config = merge_configs(config, overrides)

    # Ambiente temporaneo per lo spazio azioni
    logger.info("Inizializzazione ambiente...")
    tmp_env = make_env_from_config(config)
    action_space_size = get_action_space_size(tmp_env)
    tmp_env.close()
    logger.info(f"Azioni disponibili: {action_space_size}")

    # Costruisce l'agente scelto
    if args.agent == "cnn-dummy":
        logger.info("Costruzione CNN backbone...")
        cnn = build_cnn_from_config(config)
        logger.info(f"\n{cnn.summary()}\n")
        device = config.model.device
        agent = CNNDummyAgent(action_space_size=action_space_size, cnn=cnn, device=device)
    else:
        agent = DummyAgent(action_space_size=action_space_size)

    logger.info(f"Agente: {agent.name}")

    # Nome della run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or (
        f"{agent.name.lower()}_{config.preprocessing.mode}"
        f"_mask{int(config.masking.enabled)}"
        f"_{timestamp}"
    )

    results_dir = config.evaluation.results_dir
    results_format = config.evaluation.results_format
    logger.info(f"Risultati → {results_dir}/{run_name}.{results_format}")

    with MetricsLogger(output_dir=results_dir, run_name=run_name, format=results_format) as metrics_logger:
        results = evaluate_agent(agent=agent, config=config, metrics_logger=metrics_logger)

    rewards = [r.total_reward for r in results]
    lengths = [r.episode_length for r in results]
    print("\n" + "=" * 50)
    print(f"  RUN COMPLETATA: {run_name}")
    print("=" * 50)
    print(f"  Agente:           {agent.name}")
    print(f"  Episodi totali:   {len(results)}")
    print(f"  Reward medio:     {sum(rewards)/len(rewards):.2f}")
    print(f"  Reward mediano:   {sorted(rewards)[len(rewards)//2]:.2f}")
    print(f"  Reward min/max:   {min(rewards):.1f} / {max(rewards):.1f}")
    print(f"  Lunghezza media:  {sum(lengths)/len(lengths):.1f} step")
    print("=" * 50)


if __name__ == "__main__":
    main()
