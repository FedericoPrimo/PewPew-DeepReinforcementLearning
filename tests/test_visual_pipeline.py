"""
Smoke test per la pipeline visiva unificata.

Questi test non addestrano nulla: verificano solo che preprocessing,
masking, wrapper e forward degli agenti non vadano in errore.
"""

import numpy as np
import pytest

from src.agents.dqn_agent import DQNAgent
from src.agents.ppo_agent import PPOAgent
from src.preprocessing.image_preprocessor import ImagePreprocessor
from src.preprocessing.masking import RandomScreenMasker

try:
    import gymnasium as gym
    from src.preprocessing.image_preprocessor import ImagePreprocessingWrapper
except ModuleNotFoundError:  # pragma: no cover - dipendenza opzionale
    gym = None


if gym is not None:
    class FakeRGBEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self):
            super().__init__()
            self.observation_space = gym.spaces.Box(
                low=0,
                high=255,
                shape=(210, 160, 3),
                dtype=np.uint8,
            )
            self.action_space = gym.spaces.Discrete(6)
            self._step_count = 0
            self._rng = np.random.default_rng(123)

        def _make_obs(self) -> np.ndarray:
            return self._rng.integers(0, 256, size=(210, 160, 3), dtype=np.uint8)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self._step_count = 0
            return self._make_obs(), {}

        def step(self, action):
            self._step_count += 1
            terminated = self._step_count >= 3
            truncated = False
            return self._make_obs(), 1.0, terminated, truncated, {}


    def test_unified_wrapper_rgb_smoke():
        env = ImagePreprocessingWrapper(
            FakeRGBEnv(),
            preprocessor=ImagePreprocessor(mode="rgb", image_size=84, frame_stack=4, normalize=False),
            masker=RandomScreenMasker(enabled=False),
        )
        obs, _ = env.reset()
        assert obs.shape == (12, 84, 84)
        assert obs.dtype == np.uint8

        next_obs, reward, terminated, truncated, _ = env.step(0)
        assert next_obs.shape == (12, 84, 84)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)


    def test_unified_wrapper_grayscale_with_masking_smoke():
        env = ImagePreprocessingWrapper(
            FakeRGBEnv(),
            preprocessor=ImagePreprocessor(mode="grayscale", image_size=84, frame_stack=2, normalize=False),
            masker=RandomScreenMasker(
                enabled=True,
                apply_probability=1.0,
                num_masks=1,
                mask_size=(8, 8),
                mask_value="zero",
            ),
        )
        obs, _ = env.reset()
        assert obs.shape == (6, 84, 84)
        assert obs.dtype == np.uint8
        assert obs.min() == 0


def test_dqn_agent_forward_smoke():
    preprocessor = ImagePreprocessor(mode="rgb", image_size=84, frame_stack=4, normalize=False)
    frame = np.random.default_rng(0).integers(0, 256, size=(210, 160, 3), dtype=np.uint8)
    obs = preprocessor.process(frame)

    agent = DQNAgent(n_actions=6, in_channels=obs.shape[0], device="cpu")
    action = agent.act(obs)
    assert isinstance(action, int)
    assert 0 <= action < 6


def test_ppo_agent_forward_smoke():
    preprocessor = ImagePreprocessor(mode="grayscale", image_size=84, frame_stack=3, normalize=False)
    frame = np.random.default_rng(1).integers(0, 256, size=(210, 160, 3), dtype=np.uint8)
    obs = preprocessor.process(frame)
    batch_obs = np.stack([obs, obs], axis=0)

    agent = PPOAgent(
        n_actions=6,
        n_envs=2,
        n_steps=2,
        n_epochs=1,
        batch_size=2,
        learning_rate=2.5e-4,
        clip_range=0.1,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        vf_coef=0.5,
        device="cpu",
        feature_dim=128,
        obs_shape=obs.shape,
    )

    obs_t = agent._to_tensor(batch_obs)
    actions, log_probs, entropy, values = agent.net.get_action_and_value(obs_t)
    assert actions.shape == (2,)
    assert log_probs.shape == (2,)
    assert entropy.shape == (2,)
    assert values.shape == (2,)
