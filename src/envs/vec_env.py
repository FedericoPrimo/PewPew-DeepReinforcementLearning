"""
Vectorized environments — custom, no SB3 dependency.

DummyVecEnv:   n_envs envs sequential in same process. Used for evaluation.
SubprocVecEnv: n_envs envs each in a separate subprocess (true parallelism).
               Used for PPO training. Each env runs in its own Process;
               parent ↔ worker communication via multiprocessing.Pipe.

Interface (both classes):
  reset() → np.ndarray (n_envs, C, H, W) uint8
  step(actions) → (obs, rewards, dones, infos)

Observations transposed (H, W, C) → (C, H, W) for channels-first CNN input.
Auto-reset on done: returned obs is already the first frame of the new episode.

macOS note: default start method is 'spawn' (not 'fork'), so:
  - _worker must be a top-level picklable function
  - env factory must be a picklable callable class, not a lambda/closure
  - ale_py registered inside make_atari_env (each worker starts fresh)
"""

import multiprocessing as mp
import numpy as np
import gymnasium as gym
from typing import Callable, List, Tuple


# ---------------------------------------------------------------------------
# Worker (runs in subprocess)
# ---------------------------------------------------------------------------

def _worker(conn: mp.connection.Connection, env_fn: Callable[[], gym.Env]) -> None:
    """
    Subprocess worker. Receives (cmd, data) from parent via Pipe, sends back results.

    Commands:
      ('reset', None)   → sends obs (H, W, C) uint8
      ('step', action)  → sends (obs, reward, done, info); auto-resets on done
      ('get_spaces', None) → sends (observation_space, action_space)
      ('close', None)   → closes env and exits
    """
    env = env_fn()
    try:
        while True:
            cmd, data = conn.recv()
            if cmd == "reset":
                obs, info = env.reset()
                conn.send(obs)
            elif cmd == "step":
                obs, reward, terminated, truncated, info = env.step(data)
                done = terminated or truncated
                if done:
                    obs, _ = env.reset()
                conn.send((obs, float(reward), done, info))
            elif cmd == "get_spaces":
                conn.send((env.observation_space, env.action_space))
            elif cmd == "close":
                env.close()
                break
            else:
                raise ValueError(f"Unknown command: {cmd}")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Picklable env factory (required for multiprocessing spawn)
# ---------------------------------------------------------------------------

class _AtariEnvFactory:
    """Picklable factory — lambdas with closures are not picklable under spawn."""

    def __init__(self, env_id: str, seed: int):
        self.env_id = env_id
        self.seed = seed

    def __call__(self) -> gym.Env:
        from src.envs.atari_wrappers import make_atari_env
        return make_atari_env(self.env_id, self.seed)


# ---------------------------------------------------------------------------
# DummyVecEnv — sequential, same process
# ---------------------------------------------------------------------------

class DummyVecEnv:
    """Sequential vectorized environment. Lightweight; used for evaluation."""

    def __init__(self, env_fns: List[Callable[[], gym.Env]]):
        self.envs = [fn() for fn in env_fns]
        self.n_envs = len(self.envs)
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

    def reset(self) -> np.ndarray:
        return np.stack([env.reset()[0].transpose(2, 0, 1) for env in self.envs])

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
        obs_list, rewards, dones, infos = [], [], [], []
        for env, action in zip(self.envs, actions):
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            if done:
                obs, _ = env.reset()
            obs_list.append(obs.transpose(2, 0, 1))
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        return (
            np.stack(obs_list),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=bool),
            infos,
        )

    def close(self) -> None:
        for env in self.envs:
            env.close()


# ---------------------------------------------------------------------------
# SubprocVecEnv — one subprocess per env, true parallelism
# ---------------------------------------------------------------------------

class SubprocVecEnv:
    """
    Vectorized environment using one subprocess per env.

    Each env runs in a dedicated Process. Parent and worker communicate
    via bidirectional Pipe. Auto-resets on episode end inside the worker.
    """

    def __init__(self, env_fns: List[Callable[[], gym.Env]]):
        self.n_envs = len(env_fns)

        # Create one Pipe per env: (parent_conn, child_conn)
        pairs = [mp.Pipe() for _ in env_fns]
        self._parent_conns = [p for p, _ in pairs]
        child_conns = [c for _, c in pairs]

        # Start worker processes
        self._procs: List[mp.Process] = []
        for child_conn, env_fn in zip(child_conns, env_fns):
            proc = mp.Process(target=_worker, args=(child_conn, env_fn), daemon=True)
            proc.start()
            self._procs.append(proc)

        # Close child-side connections in parent (only worker needs them)
        for conn in child_conns:
            conn.close()

        # Retrieve spaces from worker 0
        self._parent_conns[0].send(("get_spaces", None))
        obs_space, act_space = self._parent_conns[0].recv()
        self.observation_space = obs_space
        self.action_space = act_space

    def reset(self) -> np.ndarray:
        for conn in self._parent_conns:
            conn.send(("reset", None))
        obs_list = [conn.recv() for conn in self._parent_conns]
        return np.stack([obs.transpose(2, 0, 1) for obs in obs_list])

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
        # Send all actions first (async), then collect results
        for conn, action in zip(self._parent_conns, actions):
            conn.send(("step", int(action)))
        results = [conn.recv() for conn in self._parent_conns]
        obs_list, rewards, dones, infos = zip(*results)
        return (
            np.stack([obs.transpose(2, 0, 1) for obs in obs_list]),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=bool),
            list(infos),
        )

    def close(self) -> None:
        for conn in self._parent_conns:
            try:
                conn.send(("close", None))
            except BrokenPipeError:
                pass
        for proc in self._procs:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_ppo_vec_env(env_id: str, n_envs: int, seed: int) -> SubprocVecEnv:
    """
    Build a SubprocVecEnv of n_envs Atari envs with full preprocessing.
    Each env gets seed + i for diverse initial states.
    """
    env_fns = [_AtariEnvFactory(env_id, seed + i) for i in range(n_envs)]
    return SubprocVecEnv(env_fns)


def make_eval_vec_env(env_id: str, seed: int) -> DummyVecEnv:
    """Single-env DummyVecEnv for greedy evaluation."""
    return DummyVecEnv([_AtariEnvFactory(env_id, seed)])
