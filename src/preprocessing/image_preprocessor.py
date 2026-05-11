"""
Preprocessing delle osservazioni visive dell'ambiente.

Supporta due modalità:
- RGB: resize + normalizzazione
- Grayscale: conversione BN + resize + normalizzazione + replica su 3 canali

Il formato finale è sempre: (C, H, W) con C=3 (o C=3*frame_stack)
così da mantenere invariata l'architettura della CNN.
"""

import numpy as np
import cv2
from collections import deque
from typing import Tuple, Optional


class ImagePreprocessor:
    """
    Preprocessa osservazioni dell'ambiente Atari.

    Parametri:
        mode: "rgb" o "grayscale"
        image_size: dimensione target (es. 84 → 84x84)
        frame_stack: numero di frame da stackare (1 = nessuno stacking)
        normalize: se True, normalizza i pixel in [0.0, 1.0]
    """

    VALID_MODES = ("rgb", "grayscale")

    def __init__(
        self,
        mode: str = "rgb",
        image_size: int = 84,
        frame_stack: int = 1,
        normalize: bool = True,
    ):
        if mode not in self.VALID_MODES:
            raise ValueError(f"Modalità non valida: '{mode}'. Scegli tra {self.VALID_MODES}")
        if image_size <= 0:
            raise ValueError(f"image_size deve essere positivo, ricevuto: {image_size}")
        if frame_stack < 1:
            raise ValueError(f"frame_stack deve essere >= 1, ricevuto: {frame_stack}")

        self.mode = mode
        self.image_size = image_size
        self.frame_stack = frame_stack
        self.normalize = normalize

        # Buffer circolare per il frame stacking
        self._frame_buffer: deque = deque(maxlen=frame_stack)

        # Shape di un singolo frame preprocessato (C, H, W)
        self._single_frame_shape: Tuple[int, int, int] = (3, image_size, image_size)

    @property
    def observation_shape(self) -> Tuple[int, ...]:
        """
        Restituisce la shape dell'osservazione finale.
        Con frame_stack=1: (3, H, W)
        Con frame_stack=k: (3*k, H, W)
        """
        channels = 3 * self.frame_stack
        return (channels, self.image_size, self.image_size)

    def reset(self) -> None:
        """Svuota il buffer dei frame. Da chiamare all'inizio di ogni episodio."""
        self._frame_buffer.clear()

    def process(self, observation: np.ndarray) -> np.ndarray:
        """
        Preprocessa una singola osservazione (frame).

        Args:
            observation: Array NumPy HxWx3 (RGB uint8) dall'ambiente.

        Returns:
            Array NumPy preprocessato. Se frame_stack=1: (3, H, W).
            Se frame_stack>1: (3*frame_stack, H, W).
        """
        frame = self._preprocess_single_frame(observation)
        self._frame_buffer.append(frame)

        # Riempie il buffer con il primo frame se non è ancora pieno
        while len(self._frame_buffer) < self.frame_stack:
            self._frame_buffer.append(frame)

        # Stacking lungo l'asse dei canali: (3*k, H, W)
        stacked = np.concatenate(list(self._frame_buffer), axis=0)
        return stacked

    def _preprocess_single_frame(self, observation: np.ndarray) -> np.ndarray:
        """
        Applica il preprocessing a un singolo frame.

        Args:
            observation: Array HxWxC uint8.

        Returns:
            Array (3, H, W) float32.
        """
        if self.mode == "grayscale":
            frame = self._to_grayscale(observation)
        else:
            frame = observation.copy()

        # Resize: OpenCV vuole (W, H)
        frame = cv2.resize(
            frame,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_AREA,
        )

        # Normalizzazione
        if self.normalize:
            frame = frame.astype(np.float32) / 255.0
        else:
            frame = frame.astype(np.float32)

        # Porta in formato (C, H, W)
        if frame.ndim == 2:
            # Grayscale → (H, W) → replica su 3 canali → (3, H, W)
            frame = np.stack([frame, frame, frame], axis=0)
        else:
            # RGB: (H, W, C) → (C, H, W)
            frame = np.transpose(frame, (2, 0, 1))

        return frame

    @staticmethod
    def _to_grayscale(observation: np.ndarray) -> np.ndarray:
        """
        Converte un frame RGB in grayscale usando OpenCV.

        Args:
            observation: Array HxWx3 uint8.

        Returns:
            Array HxW uint8.
        """
        # OpenCV si aspetta BGR, ma l'immagine Gymnasium è RGB
        # Usiamo la formula luminance-aware di cv2.cvtColor
        gray = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        return gray


def build_preprocessor_from_config(config) -> ImagePreprocessor:
    """
    Costruisce un ImagePreprocessor dalla sezione `preprocessing` della config.

    Args:
        config: Oggetto Config con attributi mode, image_size, frame_stack, normalize.

    Returns:
        Istanza di ImagePreprocessor configurata.
    """
    return ImagePreprocessor(
        mode=config.preprocessing.mode,
        image_size=config.preprocessing.image_size,
        frame_stack=config.preprocessing.frame_stack,
        normalize=config.preprocessing.normalize,
    )
