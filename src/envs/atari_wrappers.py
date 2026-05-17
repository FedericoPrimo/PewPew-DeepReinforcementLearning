"""
Atari environment wrappers con preprocessing visivo unificato.

Stack order (outer → inner):
  ClipReward → ImagePreprocessingWrapper → FireReset
  → EpisodicLife → MonitorWrapper → MaxAndSkip(4) → NoopReset → raw_env

MonitorWrapper placed before EpisodicLife so episode reward accumulates
across life-loss sub-episodes and reports only on true game over.
"""

import numpy as np
import gymnasium as gym

from src.preprocessing.image_preprocessor import ImagePreprocessor, ImagePreprocessingWrapper
from src.preprocessing.masking import RandomScreenMasker


class NoopResetEnv(gym.Wrapper):
    """Random no-ops on reset to sample diverse start states."""

    def __init__(self, env: gym.Env, noop_max: int = 30):
        super().__init__(env)
        self.noop_max = noop_max

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        n_noops = np.random.randint(1, self.noop_max + 1)
        for _ in range(n_noops):
            obs, _, terminated, truncated, info = self.env.step(0)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info


class MaxAndSkipEnv(gym.Wrapper):
    """Return every skip-th frame; max-pool last two raw frames to remove flicker."""

    def __init__(self, env: gym.Env, skip: int = 4):
        super().__init__(env)
        self._skip = skip
        self._buf = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)

    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False
        info = {}
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            if i == self._skip - 2:
                self._buf[0] = obs
            if i == self._skip - 1:
                self._buf[1] = obs
            total_reward += reward
            if terminated or truncated:
                break
        return self._buf.max(axis=0), total_reward, terminated, truncated, info


class MonitorWrapper(gym.Wrapper):
    """Accumulate episode reward/length; add info['episode'] on game over."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._ep_reward = 0.0
        self._ep_length = 0

    def reset(self, **kwargs):
        self._ep_reward = 0.0
        self._ep_length = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._ep_reward += reward
        self._ep_length += 1
        if terminated or truncated:
            info["episode"] = {"r": self._ep_reward, "l": self._ep_length}
        return obs, reward, terminated, truncated, info


class EpisodicLifeEnv(gym.Wrapper):
    """Treat life loss as episode end; reset only on true game over."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated or truncated
        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            terminated = True
        self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            obs, _, _, _, info = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info


class FireResetEnv(gym.Wrapper):
    """Press FIRE on reset for envs that need it to begin play."""

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs, _, terminated, truncated, _ = self.env.step(1)
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        return obs, info


class ClipRewardEnv(gym.RewardWrapper):
    """Clip reward to {-1, 0, +1}."""

    def reward(self, reward: float) -> float:
        return float(np.sign(reward))


def make_atari_env(
    env_id: str,
    seed: int = 0,
    preprocessing_kwargs: dict | None = None,
    masking_kwargs: dict | None = None,
) -> gym.Env:
    """
    Single Atari env con pipeline comune DQN/PPO.

    preprocessing_kwargs controlla:
      - mode: rgb | grayscale
      - image_size
      - frame_stack
      - normalize

    masking_kwargs controlla il masking opzionale.
    """
    import ale_py
    gym.register_envs(ale_py)
    env = gym.make(env_id)
    env.reset(seed=seed)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = MonitorWrapper(env)
    env = EpisodicLifeEnv(env)
    env = FireResetEnv(env)
    preprocessor = ImagePreprocessor(**(preprocessing_kwargs or {}))
    masker = RandomScreenMasker(**(masking_kwargs or {}))
    env = ImagePreprocessingWrapper(env, preprocessor=preprocessor, masker=masker)
    env = ClipRewardEnv(env)
    return env
