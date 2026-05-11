"""
Random Screen Masking — trasformazione opzionale che oscura
sezioni casuali dello schermo per esperimenti di occlusione.

Funziona sia su frame RGB sia su grayscale replicato.
Il formato atteso è (C, H, W) float32 (post-preprocessing).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class MaskState:
    """Stato di una singola maschera attiva (posizione + durata residua)."""
    y: int
    x: int
    height: int
    width: int
    frames_left: int


class RandomScreenMasker:
    """
    Applica maschere casuali a frame preprocessati in formato (C, H, W).

    Parametri:
        enabled: se False, il masker è un no-op
        num_masks: numero di maschere da applicare contemporaneamente
        mask_size: (height, width) di ogni maschera
        mask_duration: durata in frame di ogni maschera
        apply_probability: probabilità di generare nuove maschere ad ogni step
        mask_value: "zero" → pixel a 0.0; "mean" → media del frame
    """

    VALID_MASK_VALUES = ("zero", "mean")

    def __init__(
        self,
        enabled: bool = False,
        num_masks: int = 2,
        mask_size: Tuple[int, int] = (20, 20),
        mask_duration: int = 5,
        apply_probability: float = 0.3,
        mask_value: str = "zero",
    ):
        if mask_value not in self.VALID_MASK_VALUES:
            raise ValueError(
                f"mask_value non valido: '{mask_value}'. Scegli tra {self.VALID_MASK_VALUES}"
            )

        self.enabled = enabled
        self.num_masks = num_masks
        self.mask_size = tuple(mask_size)   # (h, w)
        self.mask_duration = mask_duration
        self.apply_probability = apply_probability
        self.mask_value = mask_value

        self._active_masks: List[MaskState] = []

    def reset(self) -> None:
        """Rimuove tutte le maschere attive. Da chiamare all'inizio di ogni episodio."""
        self._active_masks.clear()

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """
        Applica le maschere al frame.

        Args:
            frame: Array (C, H, W) float32.

        Returns:
            Frame con maschere applicate (stesso dtype e shape).
        """
        if not self.enabled:
            return frame

        frame = frame.copy()
        _, h, w = frame.shape

        # Decrementa la durata delle maschere esistenti e rimuove quelle scadute
        self._active_masks = [
            m for m in self._active_masks if m.frames_left > 0
        ]
        for m in self._active_masks:
            m.frames_left -= 1

        # Genera nuove maschere con probabilità apply_probability
        if np.random.random() < self.apply_probability:
            mask_h, mask_w = self.mask_size
            for _ in range(self.num_masks):
                max_y = max(0, h - mask_h)
                max_x = max(0, w - mask_w)
                y = np.random.randint(0, max_y + 1)
                x = np.random.randint(0, max_x + 1)
                self._active_masks.append(
                    MaskState(
                        y=y, x=x,
                        height=mask_h, width=mask_w,
                        frames_left=self.mask_duration,
                    )
                )

        # Applica tutte le maschere attive
        for m in self._active_masks:
            fill_value = self._compute_fill_value(frame, m)
            frame[:, m.y:m.y + m.height, m.x:m.x + m.width] = fill_value

        return frame

    def _compute_fill_value(self, frame: np.ndarray, mask: MaskState) -> float:
        """Calcola il valore con cui riempire la maschera."""
        if self.mask_value == "mean":
            return float(frame.mean())
        return 0.0


def build_masker_from_config(config) -> RandomScreenMasker:
    """
    Costruisce un RandomScreenMasker dalla sezione `masking` della config.

    Args:
        config: Oggetto Config con sezione `masking`.

    Returns:
        Istanza di RandomScreenMasker configurata.
    """
    m = config.masking
    return RandomScreenMasker(
        enabled=m.enabled,
        num_masks=m.num_masks,
        mask_size=tuple(m.mask_size),
        mask_duration=m.mask_duration,
        apply_probability=m.apply_probability,
        mask_value=m.mask_value,
    )
