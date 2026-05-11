"""
Test per la creazione e il funzionamento dell'ambiente Gymnasium.
"""

import pytest
import numpy as np


@pytest.fixture
def config():
    """Configurazione minimale per i test (senza ambiente Atari reale)."""
    from src.utils.config import Config
    return Config({
        "env": {
            "id": "ALE/SpaceInvaders-v5",
            "render_mode": None,
            "repeat_action_probability": 0.0,
        },
        "preprocessing": {
            "mode": "rgb",
            "image_size": 84,
            "frame_stack": 1,
            "normalize": True,
        },
        "masking": {
            "enabled": False,
            "num_masks": 2,
            "mask_size": [20, 20],
            "mask_duration": 5,
            "apply_probability": 0.3,
            "mask_value": "zero",
        },
        "evaluation": {
            "num_episodes": 1,
            "seeds": [42],
            "agent": "DummyAgent",
            "save_results": False,
            "results_dir": "results",
            "results_format": "csv",
        },
        "logging": {
            "level": "INFO",
            "log_every_n_episodes": 1,
        },
        "analysis": {
            "confidence_level": 0.95,
            "plot_format": "png",
            "plots_dir": "results/plots",
        },
    })


def test_make_env(config):
    """Verifica che l'ambiente si crei senza errori."""
    pytest.importorskip("gymnasium")
    pytest.importorskip("ale_py")
    from src.envs.make_env import make_env_from_config
    env = make_env_from_config(config, seed=42)
    assert env is not None
    env.close()


def test_env_reset_returns_obs(config):
    """Verifica che reset() restituisca un'osservazione valida."""
    pytest.importorskip("gymnasium")
    pytest.importorskip("ale_py")
    from src.envs.make_env import make_env_from_config
    env = make_env_from_config(config, seed=42)
    obs, info = env.reset(seed=42)
    assert obs is not None
    assert isinstance(obs, np.ndarray)
    assert obs.ndim == 3  # HxWxC
    env.close()


def test_env_step(config):
    """Verifica che step() funzioni e restituisca i valori attesi."""
    pytest.importorskip("gymnasium")
    pytest.importorskip("ale_py")
    from src.envs.make_env import make_env_from_config, get_action_space_size
    env = make_env_from_config(config, seed=42)
    env.reset(seed=42)
    n_actions = get_action_space_size(env)
    assert n_actions > 0
    obs, reward, done, truncated, info = env.step(0)
    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, (int, float))
    assert isinstance(done, bool)
    assert isinstance(truncated, bool)
    env.close()


def test_action_space_size(config):
    """Verifica che l'action space sia correttamente rilevato."""
    pytest.importorskip("gymnasium")
    pytest.importorskip("ale_py")
    from src.envs.make_env import make_env_from_config, get_action_space_size
    env = make_env_from_config(config, seed=42)
    n = get_action_space_size(env)
    assert n > 0
    assert isinstance(n, int)
    env.close()
