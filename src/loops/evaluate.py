"""
Valutazione di un agente su più episodi e più seed.

Compatibile con futuri agenti allenabili:
    - evaluate_agent(...)  → valutazione senza training
    - train_agent(...)     → placeholder per il training
"""

import logging
from typing import List, Optional

from src.agents.base_agent import BaseAgent
from src.envs.make_env import make_env_from_config
from src.preprocessing.image_preprocessor import build_preprocessor_from_config
from src.preprocessing.masking import build_masker_from_config
from src.loops.run_episode import run_episode, EpisodeResult
from src.logging.metrics_logger import MetricsLogger
from src.utils.config import Config
from src.utils.seeding import set_global_seed, get_env_seed

logger = logging.getLogger(__name__)


def evaluate_agent(
    agent: BaseAgent,
    config: Config,
    seeds: Optional[List[int]] = None,
    num_episodes: Optional[int] = None,
    metrics_logger: Optional[MetricsLogger] = None,
    verbose: bool = True,
) -> List[EpisodeResult]:
    """
    Valuta un agente su più episodi e più seed.

    Args:
        agent: Agente da valutare.
        config: Configurazione completa.
        seeds: Lista di seed (sovrascrive config.evaluation.seeds se fornita).
        num_episodes: Episodi per seed (sovrascrive config se fornito).
        metrics_logger: Logger per salvare le metriche (opzionale).
        verbose: Se True, stampa progressi.

    Returns:
        Lista di EpisodeResult, uno per ogni episodio completato.
    """
    seeds = seeds or list(config.evaluation.seeds)
    num_episodes = num_episodes or config.evaluation.num_episodes

    preprocessor = build_preprocessor_from_config(config)
    masker = build_masker_from_config(config)

    all_results: List[EpisodeResult] = []
    total_episodes = len(seeds) * num_episodes

    if verbose:
        logger.info(
            f"Avvio valutazione: agente={agent.name}, "
            f"seed={seeds}, episodi/seed={num_episodes}, "
            f"preprocessing={config.preprocessing.mode}, "
            f"masking={config.masking.enabled}"
        )

    episode_count = 0

    for seed in seeds:
        set_global_seed(seed)
        env = make_env_from_config(config, seed=seed)

        try:
            for ep_idx in range(num_episodes):
                env_seed = get_env_seed(base_seed=seed, episode=ep_idx)
                result = run_episode(
                    env=env,
                    agent=agent,
                    preprocessor=preprocessor,
                    masker=masker,
                    seed=env_seed,
                    episode_index=ep_idx,
                )
                all_results.append(result)

                if metrics_logger:
                    metrics_logger.log(result)

                episode_count += 1
                log_interval = getattr(config.logging, "log_every_n_episodes", 1)
                if verbose and episode_count % log_interval == 0:
                    logger.info(
                        f"  [{episode_count}/{total_episodes}] "
                        f"seed={seed} ep={ep_idx} "
                        f"reward={result.total_reward:.1f} "
                        f"length={result.episode_length} "
                        f"time={result.duration_seconds:.2f}s"
                    )
        finally:
            env.close()

    if metrics_logger:
        metrics_logger.flush()

    if verbose:
        rewards = [r.total_reward for r in all_results]
        logger.info(
            f"Valutazione completata: "
            f"reward medio={sum(rewards)/len(rewards):.2f}, "
            f"min={min(rewards):.1f}, max={max(rewards):.1f}"
        )

    return all_results


def train_agent(
    agent: BaseAgent,
    config: Config,
    **kwargs,
) -> None:
    """
    Placeholder per il training di futuri agenti RL.
    
    Per la Milestone 1 non è implementato.
    Verrà completato nelle milestone successive con agenti DQN/PPO.

    Args:
        agent: Agente da addestrare.
        config: Configurazione completa.
    """
    logger.warning(
        f"train_agent() è un placeholder nella Milestone 1. "
        f"L'agente '{agent.name}' non supporta il training."
    )
