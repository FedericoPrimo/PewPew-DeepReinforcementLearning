from .base_agent import BaseAgent

__all__ = ["BaseAgent", "DQNAgent", "PPOAgent"]


def __getattr__(name: str):
    if name == "DQNAgent":
        from .dqn_agent import DQNAgent
        return DQNAgent
    if name == "PPOAgent":
        from .ppo_agent import PPOAgent
        return PPOAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
