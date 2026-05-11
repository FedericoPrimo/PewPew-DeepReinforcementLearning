"""
DummyAgent — agente che sceglie azioni casuali uniformi.

Utile come baseline e per testare la pipeline senza
dover addestrare un vero agente RL.
"""

import numpy as np
from .base_agent import BaseAgent


class DummyAgent(BaseAgent):
    """
    Agente che campiona azioni casuali uniformemente.

    Rispetta la stessa interfaccia che avranno i futuri agenti RL:
        act(observation) -> action
        reset()          -> None

    Il seed è opzionale e garantisce riproducibilità.
    """

    def __init__(self, action_space_size: int, seed: int | None = None):
        """
        Args:
            action_space_size: Numero di azioni discrete.
            seed: Seed per il generatore casuale interno (opzionale).
        """
        super().__init__(action_space_size=action_space_size, name="DummyAgent")
        self._rng = np.random.default_rng(seed)
        self._seed = seed

    def act(self, observation: np.ndarray) -> int:
        """
        Sceglie un'azione casuale ignorando l'osservazione.

        Args:
            observation: Osservazione preprocessata (ignorata).

        Returns:
            Azione casuale intera in [0, action_space_size).
        """
        return int(self._rng.integers(0, self.action_space_size))

    def reset(self) -> None:
        """
        Reimposta il generatore casuale al seed iniziale
        per garantire riproducibilità tra episodi con lo stesso seed.
        """
        self._rng = np.random.default_rng(self._seed)
