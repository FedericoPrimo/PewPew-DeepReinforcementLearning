"""
Preprocessing unificato delle osservazioni visive.

Pipeline:
  frame RGB grezzo -> resize -> conversione opzionale in grayscale
  -> replica a 3 canali se grayscale -> frame stack -> masking opzionale
  -> output finale (C, H, W) con C = 3 * frame_stack
"""

from collections import deque
from typing import Tuple

import numpy as np
from PIL import Image

from src.preprocessing.masking import RandomScreenMasker

try:
    import gymnasium as gym
except ModuleNotFoundError:  # pragma: no cover - dipendenza opzionale per i wrapper
    gym = None


class ImagePreprocessor:
    """
    Preprocessa osservazioni dell'ambiente Atari.

    Parametri:
        mode: "rgb" o "grayscale"
        image_size: dimensione target (es. 84 → 84x84)
        frame_stack: numero di frame da stackare (1 = nessuno stacking)
        normalize: se True, normalizza i pixel in [0.0, 1.0], altrimenti mantiene uint8
    """

    VALID_MODES = ("rgb", "grayscale")

    def __init__(
        self,
        mode: str = "rgb",
        image_size: int = 84,
        frame_stack: int = 4,
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

    @property
    def output_dtype(self):
        """Dtype finale dell'osservazione preprocessata."""
        return np.float32 if self.normalize else np.uint8

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

        stacked = np.concatenate(list(self._frame_buffer), axis=0)
        return stacked

    def _preprocess_single_frame(self, observation: np.ndarray) -> np.ndarray:
        """
        Applica il preprocessing a un singolo frame.

        Args:
            observation: Array HxWxC uint8.

        Returns:
            Array (3, H, W) uint8 o float32.
        """
        if self.mode == "grayscale":
            frame = self._to_grayscale(observation)
        else:
            frame = observation.copy()

        frame = self._resize_frame(frame)

        # Porta in formato (C, H, W) con contratto fisso a 3 canali per frame
        if frame.ndim == 2:
            frame = np.stack([frame, frame, frame], axis=0)
        else:
            frame = np.transpose(frame, (2, 0, 1))

        if self.normalize:
            frame = frame.astype(np.float32) / 255.0
        else:
            frame = frame.astype(np.uint8)

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
        return np.asarray(Image.fromarray(observation, mode="RGB").convert("L"), dtype=np.uint8)

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize via Pillow per ridurre dipendenze runtime."""
        image = Image.fromarray(frame)
        resized = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        return np.asarray(resized)


if gym is not None:
    class ImagePreprocessingWrapper(gym.Wrapper):
        """
        Wrapper Gym che applica l'intera pipeline visiva unificata.

        La logica di preprocessing e masking vive qui in modo condiviso tra DQN e PPO.
        """

        def __init__(
            self,
            env: gym.Env,
            preprocessor: ImagePreprocessor,
            masker: RandomScreenMasker | None = None,
        ):
            super().__init__(env)
            self.preprocessor = preprocessor
            self.masker = masker or RandomScreenMasker(enabled=False)
            dtype = self.preprocessor.output_dtype
            high = 1.0 if dtype == np.float32 else 255
            self.observation_space = gym.spaces.Box(
                low=0,
                high=high,
                shape=self.preprocessor.observation_shape,
                dtype=dtype,
            )

        def _process(self, observation: np.ndarray) -> np.ndarray:
            processed = self.preprocessor.process(observation)
            return self.masker.apply(processed)

        def reset(self, **kwargs):
            self.preprocessor.reset()
            self.masker.reset()
            observation, info = self.env.reset(**kwargs)
            return self._process(observation), info

        def step(self, action):
            observation, reward, terminated, truncated, info = self.env.step(action)
            return self._process(observation), reward, terminated, truncated, info
else:
    class ImagePreprocessingWrapper:  # pragma: no cover - fallback per import senza gymnasium
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError("gymnasium is required to use ImagePreprocessingWrapper")


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
