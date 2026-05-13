"""
Logger TensorBoard condiviso per DQN e PPO.

Schema unificato dei tag:
  train/mean_reward
  train/episode_length
  train/max_reward
  train/loss/td
  train/loss/policy
  train/loss/value
  train/exploration/epsilon
  eval/mean_reward
  eval/std_reward
  eval/max_reward
  meta/training_seconds
"""

from typing import Optional

from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter


class TBLogger:
  """Wrapper sottile su SummaryWriter con helper per metriche RL standard."""

  def __init__(self, log_dir: str):
    self.writer = SummaryWriter(log_dir=log_dir)

  def log_episode(self, step: int, reward: float, length: int, max_reward: Optional[float] = None) -> None:
    self.writer.add_scalar("train/mean_reward", reward, step)
    self.writer.add_scalar("train/episode_length", length, step)
    if max_reward is not None:
      self.writer.add_scalar("train/max_reward", max_reward, step)

  def log_loss(self, step: int, loss_name: str, loss_value: float) -> None:
    self.writer.add_scalar(f"train/loss/{loss_name}", loss_value, step)

  def log_td_loss(self, step: int, loss: float) -> None:
    self.log_loss(step, "td", loss)

  def log_policy_loss(self, step: int, loss: float) -> None:
    self.log_loss(step, "policy", loss)

  def log_value_loss(self, step: int, loss: float) -> None:
    self.log_loss(step, "value", loss)

  def log_epsilon(self, step: int, epsilon: float) -> None:
    self.writer.add_scalar("train/exploration/epsilon", epsilon, step)

  def log_eval(self, step: int, mean_reward: float, std_reward: float, max_reward: float) -> None:
    self.writer.add_scalar("eval/mean_reward", mean_reward, step)
    self.writer.add_scalar("eval/std_reward", std_reward, step)
    self.writer.add_scalar("eval/max_reward", max_reward, step)

  def log_training_time(self, seconds: float, step: int = 0) -> None:
    self.writer.add_scalar("meta/training_seconds", seconds, step)

  def flush(self) -> None:
    self.writer.flush()

  def close(self) -> None:
    self.writer.close()


class PPOTrainingMetricsCallback(BaseCallback):
  """Callback SB3 che copia le metriche PPO nel logger TensorBoard condiviso."""

  def __init__(self, tb_logger: TBLogger, verbose: int = 0):
    super().__init__(verbose)
    self.tb_logger = tb_logger
    self._max_reward = float("-inf")

  def _on_step(self) -> bool:
    return True

  def _on_rollout_end(self) -> None:
    logs = self.logger.name_to_value
    step = self.num_timesteps

    if "rollout/ep_rew_mean" in logs:
      mean_reward = float(logs["rollout/ep_rew_mean"])
      self._max_reward = max(self._max_reward, mean_reward)
      episode_length = int(logs.get("rollout/ep_len_mean", 0) or 0)
      self.tb_logger.log_episode(
        step=step,
        reward=mean_reward,
        length=episode_length,
        max_reward=self._max_reward,
      )

    if "train/policy_loss" in logs:
      self.tb_logger.log_policy_loss(step, float(logs["train/policy_loss"]))

    if "train/value_loss" in logs:
      self.tb_logger.log_value_loss(step, float(logs["train/value_loss"]))
