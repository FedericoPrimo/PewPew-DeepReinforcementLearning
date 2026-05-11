"""
Test per il DummyAgent e per la pipeline completa
observation → preprocessing → action.
"""

import pytest
import numpy as np

from src.agents.dummy_agent import DummyAgent
from src.agents.base_agent import BaseAgent
from src.preprocessing.image_preprocessor import ImagePreprocessor
from src.preprocessing.masking import RandomScreenMasker


@pytest.fixture
def fake_obs():
    """Frame sintetico Atari: (210, 160, 3) uint8."""
    rng = np.random.default_rng(1)
    return rng.integers(0, 256, size=(210, 160, 3), dtype=np.uint8)


@pytest.fixture
def dummy_agent():
    return DummyAgent(action_space_size=6, seed=42)


class TestDummyAgent:

    def test_inherits_base_agent(self, dummy_agent):
        """DummyAgent deve ereditare da BaseAgent."""
        assert isinstance(dummy_agent, BaseAgent)

    def test_act_returns_int(self, dummy_agent, fake_obs):
        """act() deve restituire un intero."""
        preprocessor = ImagePreprocessor(mode="rgb", image_size=84)
        obs = preprocessor.process(fake_obs)
        action = dummy_agent.act(obs)
        assert isinstance(action, int), f"Tipo atteso int, ottenuto {type(action)}"

    def test_act_within_action_space(self, dummy_agent, fake_obs):
        """L'azione scelta deve essere in [0, action_space_size)."""
        preprocessor = ImagePreprocessor(mode="rgb", image_size=84)
        obs = preprocessor.process(fake_obs)
        for _ in range(50):
            action = dummy_agent.act(obs)
            assert 0 <= action < dummy_agent.action_space_size, (
                f"Azione {action} fuori range [0, {dummy_agent.action_space_size})"
            )

    def test_reset_reproducibility(self, fake_obs):
        """Dopo reset, la sequenza di azioni è identica."""
        preprocessor = ImagePreprocessor(mode="rgb", image_size=84)
        obs = preprocessor.process(fake_obs)

        agent = DummyAgent(action_space_size=6, seed=99)
        actions_before = [agent.act(obs) for _ in range(10)]
        agent.reset()
        actions_after = [agent.act(obs) for _ in range(10)]
        assert actions_before == actions_after

    def test_different_seeds_different_actions(self, fake_obs):
        """Seed diversi producono sequenze di azioni diverse."""
        preprocessor = ImagePreprocessor(mode="rgb", image_size=84)
        obs = preprocessor.process(fake_obs)

        agent1 = DummyAgent(action_space_size=6, seed=1)
        agent2 = DummyAgent(action_space_size=6, seed=2)
        actions1 = [agent1.act(obs) for _ in range(20)]
        actions2 = [agent2.act(obs) for _ in range(20)]
        assert actions1 != actions2, "Seed diversi devono produrre azioni diverse"

    def test_name_attribute(self, dummy_agent):
        """Il nome dell'agente deve essere 'DummyAgent'."""
        assert dummy_agent.name == "DummyAgent"

    def test_action_space_size_attribute(self, dummy_agent):
        """action_space_size deve essere corretto."""
        assert dummy_agent.action_space_size == 6


class TestPipeline:
    """
    Test end-to-end della pipeline:
    observation → preprocessing (→ masking) → action
    """

    def test_full_pipeline_rgb(self, fake_obs):
        """Pipeline completa con preprocessing RGB."""
        preprocessor = ImagePreprocessor(mode="rgb", image_size=84)
        masker = RandomScreenMasker(enabled=False)
        agent = DummyAgent(action_space_size=6, seed=42)

        obs_processed = preprocessor.process(fake_obs)
        obs_masked = masker.apply(obs_processed)
        action = agent.act(obs_masked)

        assert obs_processed.shape == (3, 84, 84)
        assert obs_masked.shape == (3, 84, 84)
        assert 0 <= action < 6

    def test_full_pipeline_grayscale(self, fake_obs):
        """Pipeline completa con preprocessing Grayscale."""
        preprocessor = ImagePreprocessor(mode="grayscale", image_size=84)
        masker = RandomScreenMasker(enabled=False)
        agent = DummyAgent(action_space_size=6, seed=42)

        obs_processed = preprocessor.process(fake_obs)
        obs_masked = masker.apply(obs_processed)
        action = agent.act(obs_masked)

        assert obs_processed.shape == (3, 84, 84)
        assert obs_masked.shape == (3, 84, 84)
        assert 0 <= action < 6

    def test_pipeline_rgb_grayscale_same_output_shape(self, fake_obs):
        """RGB e Grayscale producono la stessa shape: requisito critico della milestone."""
        prep_rgb = ImagePreprocessor(mode="rgb", image_size=84, frame_stack=1)
        prep_gray = ImagePreprocessor(mode="grayscale", image_size=84, frame_stack=1)

        rgb_out = prep_rgb.process(fake_obs)
        gray_out = prep_gray.process(fake_obs)

        assert rgb_out.shape == gray_out.shape, (
            f"CRITICO: shape diversa tra RGB {rgb_out.shape} e Grayscale {gray_out.shape}"
        )

    def test_pipeline_with_masking_enabled(self, fake_obs):
        """Pipeline con masking abilitato: shape invariata."""
        preprocessor = ImagePreprocessor(mode="rgb", image_size=84)
        masker = RandomScreenMasker(
            enabled=True,
            apply_probability=1.0,
            num_masks=2,
            mask_size=(10, 10),
        )
        agent = DummyAgent(action_space_size=6, seed=42)

        obs_processed = preprocessor.process(fake_obs)
        obs_masked = masker.apply(obs_processed)
        action = agent.act(obs_masked)

        assert obs_masked.shape == obs_processed.shape
        assert 0 <= action < 6

    def test_pipeline_frame_stack_4(self, fake_obs):
        """Pipeline con frame stacking a 4: shape (12, 84, 84)."""
        preprocessor = ImagePreprocessor(mode="rgb", image_size=84, frame_stack=4)
        masker = RandomScreenMasker(enabled=False)
        agent = DummyAgent(action_space_size=6, seed=42)

        for _ in range(4):
            obs = preprocessor.process(fake_obs)

        obs_masked = masker.apply(obs)
        action = agent.act(obs_masked)

        assert obs.shape == (12, 84, 84)
        assert 0 <= action < 6
