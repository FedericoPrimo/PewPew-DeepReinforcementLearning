"""
Interfaccia base per tutti gli agenti.

Tutti i futuri agenti RL (DQN, PPO, ecc.) devono ereditare da BaseAgent
e implementare i metodi `act` e `reset`.
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseAgent(ABC):
    """
    Classe base astratta per gli agenti.

    Definisce il contratto che ogni agente deve rispettare,
    garantendo compatibilità con il loop di training/evaluation.
    """

    def __init__(self, action_space_size: int, name: str = "BaseAgent"):
        """
        Args:
            action_space_size: Numero di azioni discrete disponibili.
            name: Nome identificativo dell'agente.
        """
        self.action_space_size = action_space_size
        self.name = name

    @abstractmethod
    def act(self, observation: np.ndarray) -> int:
        """
        Sceglie un'azione dato lo stato corrente.

        Args:
            observation: Osservazione preprocessata (C, H, W) float32.

        Returns:
            Indice dell'azione scelta (intero in [0, action_space_size)).
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """
        Reimposta lo stato interno dell'agente.
        Da chiamare all'inizio di ogni episodio.
        """
        ...

    def update(self, *args, **kwargs) -> None:
        """
        Aggiorna i parametri dell'agente (es. pesi della rete).
        Placeholder per futuri agenti allenabili.
        Per il DummyAgent non fa nulla.
        """
        pass

    def save(self, path: str) -> None:
        """Salva il modello su disco. Placeholder."""
        pass

    def load(self, path: str) -> None:
        """Carica il modello da disco. Placeholder."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(action_space_size={self.action_space_size})"
