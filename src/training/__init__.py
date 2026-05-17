from .replay_buffer import ReplayBuffer
from .rollout_buffer import RolloutBuffer

__all__ = ["ReplayBuffer", "RolloutBuffer", "TBLogger", "PPOTrainingMetricsCallback"]


def __getattr__(name: str):
    if name == "TBLogger":
        from .tb_logger import TBLogger
        return TBLogger
    if name == "PPOTrainingMetricsCallback":
        from .tb_logger import PPOTrainingMetricsCallback
        return PPOTrainingMetricsCallback
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
