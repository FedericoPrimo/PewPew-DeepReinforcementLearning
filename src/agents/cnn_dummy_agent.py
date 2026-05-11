"""
CNNDummyAgent — come il DummyAgent ma passa l'osservazione
attraverso la CNN prima di scegliere l'azione.

Scopo della Milestone 2:
    - Verificare che la CNN riceva input della shape corretta
    - Verificare che il forward pass funzioni senza errori
    - Esporre il vettore di features prodotto dalla CNN
    - Mantenere la stessa interfaccia degli agenti futuri (DQN, PPO)

L'azione rimane casuale: la CNN estrae features ma non le usa
per decidere. Sarà compito degli agenti delle milestone successive
collegare le features a una policy.
"""

import numpy as np
import torch
from typing import Optional

from src.agents.base_agent import BaseAgent
from src.models.cnn_backbone import CNNBackbone


class CNNDummyAgent(BaseAgent):
    """
    Agente che esegue il forward pass della CNN e sceglie azioni casuali.

    Interfaccia identica a DummyAgent e ai futuri agenti RL:
        act(observation) -> action
        reset()          -> None

    In più espone:
        last_features    -> vettore di features dell'ultimo forward pass
    """

    def __init__(
        self,
        action_space_size: int,
        cnn: CNNBackbone,
        device: str = "cpu",
        seed: Optional[int] = None,
    ):
        """
        Args:
            action_space_size: Numero di azioni discrete.
            cnn: Istanza di CNNBackbone già costruita.
            device: "cpu" o "cuda".
            seed: Seed per il generatore casuale interno.
        """
        super().__init__(action_space_size=action_space_size, name="CNNDummyAgent")

        self.cnn = cnn.to(device)
        self.cnn.eval()   # Nessun training in questa milestone
        self.device = device

        self._rng = np.random.default_rng(seed)
        self._seed = seed
        self.last_features: Optional[torch.Tensor] = None

    def act(self, observation: np.ndarray) -> int:
        """
        Esegue il forward pass della CNN, poi sceglie un'azione casuale.

        Args:
            observation: Array (C, H, W) float32 — osservazione preprocessata.

        Returns:
            Azione casuale intera in [0, action_space_size).
        """
        # Converti osservazione in tensore PyTorch: (C,H,W) → (1,C,H,W)
        obs_tensor = torch.from_numpy(observation).unsqueeze(0).to(self.device)

        # Forward pass CNN — nessun gradiente necessario
        with torch.no_grad():
            features = self.cnn(obs_tensor)   # (1, feature_dim)

        # Salva per ispezione/debug
        self.last_features = features

        # Azione casuale (le features non vengono usate)
        return int(self._rng.integers(0, self.action_space_size))

    def reset(self) -> None:
        """Reimposta il generatore casuale e pulisce le features salvate."""
        self._rng = np.random.default_rng(self._seed)
        self.last_features = None

    @property
    def feature_dim(self) -> int:
        """Dimensione del vettore di features prodotto dalla CNN."""
        return self.cnn.feature_dim
