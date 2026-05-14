# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt
autorom --accept-license   # install Atari ROMs (required once)

# Train agents
python scripts/train_dqn.py                          # DQN training
python scripts/train_ppo.py                          # PPO training
python scripts/train_dqn.py --timesteps 50000 --device mps   # quick test

# Analyze results
python scripts/compare.py
python scripts/compare_results.py --results-dir results/ --plot-format pdf

# TensorBoard
tensorboard --logdir logs/ --port 6006

# Tests
python -m pytest
python -m pytest tests/test_cnn.py -v
```

## Architecture

**Observation flow (both agents):**
`raw ALE env` → preprocessing wrappers → `(4, 84, 84) uint8` → `agent.act()` → action

Both agents receive **4 stacked grayscale frames at (4, 84, 84)**. Preprocessing is identical.

**DQN** — off-policy, sequential single env, epsilon-greedy, target network, replay buffer.
**PPO** — on-policy, 8 parallel envs via custom `SubprocVecEnv`, Actor-Critic, GAE, PPO-Clip.

**Key modules:**

- [src/agents/dqn_agent.py](src/agents/dqn_agent.py) — `QNetwork` (CNNBackbone + linear head) + `DQNAgent` (Huber loss, Adam, grad clip, target net). Off-policy.
- [src/agents/ppo_agent.py](src/agents/ppo_agent.py) — `PPOAgent`: `collect_rollouts` + PPO-Clip `update`. On-policy, manual implementation.
- [src/models/actor_critic.py](src/models/actor_critic.py) — `ActorCriticNet`: shared CNNBackbone + policy head + value head. Orthogonal init. Used by PPO only.
- [src/models/cnn_backbone.py](src/models/cnn_backbone.py) — `CNNBackbone`: Nature DQN (Mnih 2015), three Conv2d + ReLU + Linear → feature_dim. Shared by DQN and PPO.
- [src/training/replay_buffer.py](src/training/replay_buffer.py) — experience replay for DQN.
- [src/training/rollout_buffer.py](src/training/rollout_buffer.py) — on-policy buffer for PPO: GAE computation + shuffled minibatch iterator.
- [src/training/tb_logger.py](src/training/tb_logger.py) — TensorBoard logging wrapper.
- [src/envs/atari_wrappers.py](src/envs/atari_wrappers.py) — standalone Atari wrappers (no SB3): `NoopReset`, `MaxAndSkip`, `MonitorWrapper`, `EpisodicLife`, `FireReset`, `WarpFrame`, `ClipReward`, `FrameStack`. `make_atari_env()` factory.
- [src/envs/vec_env.py](src/envs/vec_env.py) — `SubprocVecEnv` (training, one subprocess per env via `mp.Pipe`) + `DummyVecEnv` (evaluation, sequential). `make_ppo_vec_env()` → SubprocVecEnv; `make_eval_vec_env()` → DummyVecEnv. No SB3 dependency.
- [src/envs/make_env.py](src/envs/make_env.py) — raw Gymnasium env factory (used by DQN).
- [src/agents/base_agent.py](src/agents/base_agent.py) — abstract `BaseAgent`.
- [src/preprocessing/image_preprocessor.py](src/preprocessing/image_preprocessor.py) — custom preprocessor (not used in training; `atari_wrappers.py` supersedes it).
- [src/preprocessing/masking.py](src/preprocessing/masking.py) — rectangular masks. Milestone 6, not yet wired.
- [src/analysis/](src/analysis/) — `compare_runs.py` + `plots.py` for `compare_results.py`.
- [src/utils/config.py](src/utils/config.py) — loads YAML configs into nested `Config` object.

## Configuration

| File | Used by |
|------|---------|
| `configs/config_common.yaml` | both — seed, env id, eval settings, logging dir |
| `configs/config_dqn.yaml` | `train_dqn.py` — lr, gamma, buffer size, epsilon schedule |
| `configs/config_ppo.yaml` | `train_ppo.py` — n_envs, n_steps, clip_range, GAE λ |
| `configs/default.yaml` | legacy dummy scripts only |

`model.device`: `"cpu"` | `"cuda"` | `"mps"`.

## Preprocessing stack (both agents identical)

```
NoopReset(noop_max=30) → MaxAndSkip(skip=4) → MonitorWrapper
→ EpisodicLife → FireReset → WarpFrame(84×84 gray) → ClipReward → FrameStack(n=4)
→ transpose (H,W,C) → (C,H,W)   [SubprocVecEnv for PPO training, DummyVecEnv for eval, manual in DQN loop]
```

`MonitorWrapper` before `EpisodicLife` ensures `info["episode"]["r"]` reports full-game reward (not per-life reward).

## Known issues / open TODOs

- Custom `ImagePreprocessor` + `RandomScreenMasker` not yet wired into training loops.
- Masking experiments (milestone 6) pending integration.
- `model.device` hardcoded in YAML; auto-detect not implemented.

## Milestone status

| Milestone | Status | Description |
|-----------|--------|-------------|
| 1 | Done | Pipeline + DummyAgent + preprocessing + masking |
| 2 | Done | CNN backbone (Nature DQN) + CNNDummyAgent |
| 3 | Done | DQN agent (QNetwork, target net, replay buffer, epsilon-greedy) |
| 4 | Done | PPO agent — manual implementation, SubprocVecEnv + DummyVecEnv (no SB3) |
| 5 | Next | RGB vs. grayscale statistical comparison |
| 6 | Planned | Random screen masking experiments |
