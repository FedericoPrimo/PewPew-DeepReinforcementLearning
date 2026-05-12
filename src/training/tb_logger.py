"""
Wrapper TensorBoard con tag standardizzati per DQN e PPO.

Tag condivisi (comparabili tra agenti):
  train/mean_reward    — reward episodica media (finestra mobile)
  train/max_reward     — reward massima episodica
  train/episode_length — lunghezza media episodi

Tag specifici DQN:
  train/td_loss        — Huber loss tra Q predetto e Q target
  train/epsilon        — valore corrente di epsilon

Tag specifici PPO (scritti da SB3, remappati dal callback):
  train/policy_loss    — policy gradient loss
  train/value_loss     — critic MSE loss
"""

from torch.utils.tensorboard import SummaryWriter


class TBLogger:
    """Wrapper sottile su SummaryWriter con helper per metriche RL standard."""

    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir=log_dir)

    def log_episode(self, reward: float, length: int, step: int) -> None:
        self.writer.add_scalar("train/mean_reward", reward, step)
        self.writer.add_scalar("train/episode_length", length, step)

    def log_max_reward(self, reward: float, step: int) -> None:
        self.writer.add_scalar("train/max_reward", reward, step)

    def log_td_loss(self, loss: float, step: int) -> None:
        self.writer.add_scalar("train/td_loss", loss, step)

    def log_epsilon(self, epsilon: float, step: int) -> None:
        self.writer.add_scalar("train/epsilon", epsilon, step)

    def close(self) -> None:
        self.writer.close()
