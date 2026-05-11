"""
Esecuzione di un singolo episodio completo.

Il ciclo è:
    reset ambiente
    → preprocessing osservazione
    → agente sceglie azione
    → env.step(action)
    → logging reward e done
    → ripetizione fino a done/truncated
"""

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import gymnasium as gym

from src.agents.base_agent import BaseAgent
from src.preprocessing.image_preprocessor import ImagePreprocessor
from src.preprocessing.masking import RandomScreenMasker


@dataclass
class EpisodeResult:
    """Metriche raccolte durante un episodio."""
    seed: int
    agent_name: str
    preprocessing_mode: str
    episode_index: int
    total_reward: float
    episode_length: int
    masking_enabled: bool
    duration_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "agent_name": self.agent_name,
            "preprocessing_mode": self.preprocessing_mode,
            "episode_index": self.episode_index,
            "total_reward": self.total_reward,
            "episode_length": self.episode_length,
            "masking_enabled": self.masking_enabled,
            "duration_seconds": round(self.duration_seconds, 4),
        }


def run_episode(
    env: gym.Env,
    agent: BaseAgent,
    preprocessor: ImagePreprocessor,
    masker: RandomScreenMasker,
    seed: int,
    episode_index: int = 0,
    render: bool = False,
) -> EpisodeResult:
    """
    Esegue un singolo episodio e restituisce le metriche.

    Args:
        env: Ambiente Gymnasium già creato.
        agent: Agente con interfaccia act/reset.
        preprocessor: Preprocessore immagini.
        masker: Masker (può essere disabilitato).
        seed: Seed usato per questo episodio.
        episode_index: Indice dell'episodio nella run.
        render: Se True, mostra il rendering a schermo.

    Returns:
        EpisodeResult con le metriche dell'episodio.
    """
    t_start = time.perf_counter()

    # Reset
    obs_raw, _ = env.reset(seed=seed)
    agent.reset()
    preprocessor.reset()
    masker.reset()

    total_reward = 0.0
    step = 0
    done = False
    truncated = False

    while not (done or truncated):
        # Preprocessing
        obs_processed = preprocessor.process(obs_raw)

        # Masking opzionale
        obs_processed = masker.apply(obs_processed)

        # Azione dell'agente
        action = agent.act(obs_processed)

        # Step nell'ambiente
        obs_raw, reward, done, truncated, info = env.step(action)

        total_reward += float(reward)
        step += 1

    duration = time.perf_counter() - t_start

    return EpisodeResult(
        seed=seed,
        agent_name=agent.name,
        preprocessing_mode=preprocessor.mode,
        episode_index=episode_index,
        total_reward=total_reward,
        episode_length=step,
        masking_enabled=masker.enabled,
        duration_seconds=duration,
    )
