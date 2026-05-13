"""
Gestione dei seed per la riproducibilità degli esperimenti.
"""

import random
import numpy as np
import torch
from typing import Optional


def set_global_seed(seed: int) -> None:
    """
    Imposta il seed globale per Python random, NumPy e PyTorch.

    Args:
        seed: Valore intero del seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_env_seed(base_seed: int, episode: int = 0) -> int:
    """
    Genera un seed deterministico per un episodio specifico,
    derivato dal seed base. Garantisce diversità tra episodi
    mantenendo la riproducibilità.

    Args:
        base_seed: Seed base dell'esperimento.
        episode: Numero dell'episodio corrente.

    Returns:
        Seed derivato per l'ambiente.
    """
    return (base_seed * 1000 + episode) % (2**31 - 1)
