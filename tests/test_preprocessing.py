"""
Test per il preprocessing delle osservazioni.

Verifica che:
- RGB e Grayscale producano la stessa shape finale
- Il frame stacking funzioni correttamente
- Il masking applichi le maschere correttamente
"""

import pytest
import numpy as np

from src.preprocessing.image_preprocessor import ImagePreprocessor
from src.preprocessing.masking import RandomScreenMasker


# Frame sintetico: simula un'osservazione Atari (210x160x3, uint8)
@pytest.fixture
def fake_obs():
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(210, 160, 3), dtype=np.uint8)


class TestImagePreprocessor:

    def test_rgb_output_shape(self, fake_obs):
        """RGB produce shape (3, 84, 84) con image_size=84."""
        prep = ImagePreprocessor(mode="rgb", image_size=84, frame_stack=1)
        result = prep.process(fake_obs)
        assert result.shape == (3, 84, 84), f"Shape attesa (3,84,84), ottenuta {result.shape}"

    def test_grayscale_output_shape(self, fake_obs):
        """Grayscale produce shape (3, 84, 84) — stesso di RGB."""
        prep = ImagePreprocessor(mode="grayscale", image_size=84, frame_stack=1)
        result = prep.process(fake_obs)
        assert result.shape == (3, 84, 84), f"Shape attesa (3,84,84), ottenuta {result.shape}"

    def test_rgb_grayscale_same_shape(self, fake_obs):
        """RGB e Grayscale producono la stessa shape finale (requisito critico)."""
        prep_rgb = ImagePreprocessor(mode="rgb", image_size=84, frame_stack=1)
        prep_gray = ImagePreprocessor(mode="grayscale", image_size=84, frame_stack=1)
        rgb_out = prep_rgb.process(fake_obs)
        gray_out = prep_gray.process(fake_obs)
        assert rgb_out.shape == gray_out.shape, (
            f"Shape non coincidono: RGB={rgb_out.shape} vs Grayscale={gray_out.shape}"
        )

    def test_frame_stack_rgb(self, fake_obs):
        """Frame stacking con k=4 produce shape (12, 84, 84)."""
        prep = ImagePreprocessor(mode="rgb", image_size=84, frame_stack=4)
        result = prep.process(fake_obs)
        assert result.shape == (12, 84, 84), f"Shape attesa (12,84,84), ottenuta {result.shape}"

    def test_frame_stack_grayscale(self, fake_obs):
        """Frame stacking grayscale con k=4 produce shape (12, 84, 84)."""
        prep = ImagePreprocessor(mode="grayscale", image_size=84, frame_stack=4)
        result = prep.process(fake_obs)
        assert result.shape == (12, 84, 84), f"Shape attesa (12,84,84), ottenuta {result.shape}"

    def test_normalization_range(self, fake_obs):
        """I valori normalizzati sono in [0.0, 1.0]."""
        prep = ImagePreprocessor(mode="rgb", image_size=84, normalize=True)
        result = prep.process(fake_obs)
        assert result.min() >= 0.0, f"Minimo fuori range: {result.min()}"
        assert result.max() <= 1.0, f"Massimo fuori range: {result.max()}"

    def test_no_normalization(self, fake_obs):
        """Senza normalizzazione i valori sono uint8 in [0, 255]."""
        prep = ImagePreprocessor(mode="rgb", image_size=84, normalize=False)
        result = prep.process(fake_obs)
        assert result.max() > 1.0, "Senza normalizzazione i valori devono essere > 1"
        assert result.dtype == np.uint8

    def test_custom_image_size(self, fake_obs):
        """Funziona con dimensioni diverse da 84."""
        prep = ImagePreprocessor(mode="rgb", image_size=64)
        result = prep.process(fake_obs)
        assert result.shape == (3, 64, 64)

    def test_reset_clears_buffer(self, fake_obs):
        """Reset svuota il buffer del frame stacking."""
        prep = ImagePreprocessor(mode="rgb", image_size=84, frame_stack=4)
        prep.process(fake_obs)
        prep.process(fake_obs)
        prep.reset()
        assert len(prep._frame_buffer) == 0

    def test_observation_shape_property(self):
        """La property observation_shape corrisponde all'output reale."""
        prep = ImagePreprocessor(mode="rgb", image_size=84, frame_stack=4)
        assert prep.observation_shape == (12, 84, 84)

    def test_invalid_mode_raises(self):
        """Una modalità non valida solleva ValueError."""
        with pytest.raises(ValueError):
            ImagePreprocessor(mode="invalid_mode")

    def test_invalid_image_size_raises(self):
        """image_size <= 0 solleva ValueError."""
        with pytest.raises(ValueError):
            ImagePreprocessor(image_size=0)

    def test_invalid_frame_stack_raises(self):
        """frame_stack < 1 solleva ValueError."""
        with pytest.raises(ValueError):
            ImagePreprocessor(frame_stack=0)

    def test_dtype_float32_when_normalized(self, fake_obs):
        """Con normalizzazione attiva l'output è float32."""
        prep = ImagePreprocessor(mode="rgb", image_size=84)
        result = prep.process(fake_obs)
        assert result.dtype == np.float32


class TestRandomScreenMasker:

    @pytest.fixture
    def sample_frame(self):
        """Frame sintetico in formato (3, 84, 84) float32."""
        rng = np.random.default_rng(0)
        return rng.random((3, 84, 84)).astype(np.float32)

    def test_disabled_masker_is_noop(self, sample_frame):
        """Con masker disabilitato il frame non cambia."""
        masker = RandomScreenMasker(enabled=False)
        result = masker.apply(sample_frame)
        np.testing.assert_array_equal(result, sample_frame)

    def test_masker_preserves_shape(self, sample_frame):
        """Il masker non cambia la shape del frame."""
        masker = RandomScreenMasker(enabled=True, apply_probability=1.0)
        result = masker.apply(sample_frame)
        assert result.shape == sample_frame.shape

    def test_masker_preserves_dtype(self, sample_frame):
        """Il masker non cambia il dtype del frame."""
        masker = RandomScreenMasker(enabled=True, apply_probability=1.0)
        result = masker.apply(sample_frame)
        assert result.dtype == sample_frame.dtype

    def test_masker_applies_zeros(self, sample_frame):
        """Con mask_value='zero' e probabilità 1, almeno un pixel diventa 0."""
        masker = RandomScreenMasker(
            enabled=True,
            apply_probability=1.0,
            num_masks=1,
            mask_size=(20, 20),
            mask_value="zero",
        )
        result = masker.apply(sample_frame)
        assert result.min() == 0.0

    def test_masker_reset_clears_masks(self, sample_frame):
        """Reset svuota le maschere attive."""
        masker = RandomScreenMasker(enabled=True, apply_probability=1.0)
        masker.apply(sample_frame)
        masker.reset()
        assert len(masker._active_masks) == 0

    def test_invalid_mask_value_raises(self):
        """mask_value non valido solleva ValueError."""
        with pytest.raises(ValueError):
            RandomScreenMasker(enabled=True, mask_value="invalid")

    def test_masker_does_not_modify_original(self, sample_frame):
        """Il masker non modifica il frame originale (copia interna)."""
        original = sample_frame.copy()
        masker = RandomScreenMasker(enabled=True, apply_probability=1.0)
        masker.apply(sample_frame)
        np.testing.assert_array_equal(sample_frame, original)
