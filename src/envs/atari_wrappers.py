"""
Atari preprocessing wrappers — standalone, no SB3 dependency.

Stack order (outer → inner):
  FrameStack(4) → ClipReward → WarpFrame → FireReset
  → EpisodicLife → MonitorWrapper → MaxAndSkip(4) → NoopReset → raw_env

MonitorWrapper placed before EpisodicLife so episode reward accumulates
across life-loss sub-episodes and reports only on true game over.
"""

import numpy as np
import cv2
import gymnasium as gym
from collections import deque


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


class WarpFrame(gym.ObservationWrapper):
    """Grayscale + resize to 84×84 → output (84, 84, 1) uint8."""

    def __init__(self, env: gym.Env, width: int = 84, height: int = 84):
        super().__init__(env)
        self.width = width
        self.height = height
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(height, width, 1), dtype=np.uint8
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return resized[:, :, np.newaxis]


class ClipRewardEnv(gym.RewardWrapper):
    """Clip reward to {-1, 0, +1}."""

    def reward(self, reward: float) -> float:
        return float(np.sign(reward))


class FrameStack(gym.Wrapper):
    """Stack last n_stack frames along channel axis → (H, W, n_stack) uint8."""

    def __init__(self, env: gym.Env, n_stack: int = 4):
        super().__init__(env)
        self.n_stack = n_stack
        self._frames: deque = deque(maxlen=n_stack)
        h, w, c = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(h, w, n_stack * c), dtype=np.uint8
        )

    def _obs(self) -> np.ndarray:
        return np.concatenate(list(self._frames), axis=-1)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.n_stack):
            self._frames.append(obs)
        return self._obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._frames.append(obs)
        return self._obs(), reward, terminated, truncated, info


def make_atari_env(env_id: str, seed: int = 0) -> gym.Env:
    """Single Atari env with full preprocessing stack matching SB3's make_atari_env."""
    import ale_py
    gym.register_envs(ale_py)
    env = gym.make(env_id)
    env.reset(seed=seed)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = MonitorWrapper(env)
    env = EpisodicLifeEnv(env)
    env = FireResetEnv(env)
    env = WarpFrame(env)
    env = ClipRewardEnv(env)
    env = FrameStack(env, n_stack=4)
    return env
