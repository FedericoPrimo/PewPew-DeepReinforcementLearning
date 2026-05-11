"""
Creazione e configurazione dell'ambiente Gymnasium per Atari Space Invaders.
"""

import gymnasium as gym
import ale_py  # registra il namespace ALE in Gymnasium
from typing import Optional
from src.utils.config import Config

# Registra automaticamente gli ambienti ALE (necessario con gymnasium >= 0.26)
gym.register_envs(ale_py)


def make_env(
    env_id: str = "ALE/SpaceInvaders-v5",
    render_mode: Optional[str] = None,
    repeat_action_probability: float = 0.0,
    seed: Optional[int] = None,
) -> gym.Env:
    """
    Crea e restituisce un'istanza dell'ambiente Gymnasium.

    Args:
        env_id: Identificatore dell'ambiente (es. "ALE/SpaceInvaders-v5").
        render_mode: Modalità di rendering ("human", "rgb_array", o None).
        repeat_action_probability: Probabilità di ripetere l'ultima azione
                                   (sticky actions). 0.0 = deterministico.
        seed: Seed per la riproducibilità.

    Returns:
        Istanza dell'ambiente Gymnasium.
    """
    env = gym.make(
        env_id,
        render_mode=render_mode,
        repeat_action_probability=repeat_action_probability,
    )

    if seed is not None:
        env.reset(seed=seed)

    return env


def make_env_from_config(config: Config, seed: Optional[int] = None) -> gym.Env:
    """
    Crea l'ambiente usando i parametri della configurazione.

    Args:
        config: Oggetto Config con sezione `env`.
        seed: Seed opzionale (sovrascrive quello in config se presente).

    Returns:
        Istanza dell'ambiente Gymnasium.
    """
    render_mode = config.env.render_mode if config.env.render_mode != "null" else None

    return make_env(
        env_id=config.env.id,
        render_mode=render_mode,
        repeat_action_probability=config.env.repeat_action_probability,
        seed=seed,
    )


def get_action_space_size(env: gym.Env) -> int:
    """
    Restituisce il numero di azioni disponibili nell'ambiente.

    Args:
        env: Istanza dell'ambiente Gymnasium.

    Returns:
        Numero intero di azioni discrete.
    """
    return int(env.action_space.n)
