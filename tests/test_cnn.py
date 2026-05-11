"""
Test per la CNN backbone e il CNNDummyAgent.
"""

import pytest
import numpy as np
import torch

from src.models.cnn_backbone import CNNBackbone
from src.agents.cnn_dummy_agent import CNNDummyAgent
from src.preprocessing.image_preprocessor import ImagePreprocessor


@pytest.fixture
def fake_obs():
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(210, 160, 3), dtype=np.uint8)


@pytest.fixture
def cnn():
    return CNNBackbone(in_channels=3, feature_dim=512)


@pytest.fixture
def cnn_agent(cnn):
    return CNNDummyAgent(action_space_size=6, cnn=cnn, device="cpu", seed=42)


class TestCNNBackbone:

    def test_output_shape(self, cnn):
        """Forward pass: (B, 3, 84, 84) → (B, 512)."""
        x = torch.zeros(1, 3, 84, 84)
        out = cnn(x)
        assert out.shape == (1, 512), f"Shape attesa (1, 512), ottenuta {out.shape}"

    def test_batch_output_shape(self, cnn):
        """Funziona con batch size > 1."""
        x = torch.zeros(4, 3, 84, 84)
        out = cnn(x)
        assert out.shape == (4, 512)

    def test_output_dtype(self, cnn):
        """L'output è float32."""
        x = torch.zeros(1, 3, 84, 84)
        out = cnn(x)
        assert out.dtype == torch.float32

    def test_custom_feature_dim(self):
        """Funziona con feature_dim diverso da 512."""
        cnn = CNNBackbone(in_channels=3, feature_dim=256)
        x = torch.zeros(1, 3, 84, 84)
        out = cnn(x)
        assert out.shape == (1, 256)

    def test_frame_stack_4_channels(self):
        """Funziona con frame_stack=4 → in_channels=12."""
        cnn = CNNBackbone(in_channels=12, feature_dim=512)
        x = torch.zeros(1, 12, 84, 84)
        out = cnn(x)
        assert out.shape == (1, 512)

    def test_custom_conv_layers(self):
        """Funziona con architettura CNN personalizzata (2 layer)."""
        cnn = CNNBackbone(
            in_channels=3,
            feature_dim=256,
            conv_configs=[
                {"out_channels": 32, "kernel_size": 8, "stride": 4},
                {"out_channels": 64, "kernel_size": 4, "stride": 2},
            ],
        )
        x = torch.zeros(1, 3, 84, 84)
        out = cnn(x)
        assert out.shape == (1, 256)

    def test_has_parameters(self, cnn):
        """La CNN ha parametri allenabili."""
        n_params = sum(p.numel() for p in cnn.parameters())
        assert n_params > 0

    def test_summary_string(self, cnn):
        """summary() restituisce una stringa non vuota."""
        s = cnn.summary()
        assert isinstance(s, str) and len(s) > 0
        assert "CNNBackbone" in s
        assert "512" in s

    def test_no_nan_in_output(self):
        """L'output non contiene NaN con input casuali."""
        cnn = CNNBackbone(in_channels=3, feature_dim=512)
        x = torch.rand(2, 3, 84, 84)
        out = cnn(x)
        assert not torch.isnan(out).any()

    def test_eval_mode_no_grad(self, cnn):
        """In eval mode, torch.no_grad() non solleva eccezioni."""
        cnn.eval()
        x = torch.rand(1, 3, 84, 84)
        with torch.no_grad():
            out = cnn(x)
        assert out.shape == (1, 512)


class TestCNNDummyAgent:

    def test_act_returns_valid_action(self, cnn_agent, fake_obs):
        """act() restituisce un'azione valida."""
        prep = ImagePreprocessor(mode="rgb", image_size=84)
        obs = prep.process(fake_obs)
        action = cnn_agent.act(obs)
        assert isinstance(action, int)
        assert 0 <= action < 6

    def test_last_features_populated(self, cnn_agent, fake_obs):
        """Dopo act(), last_features è un tensore (1, feature_dim)."""
        prep = ImagePreprocessor(mode="rgb", image_size=84)
        obs = prep.process(fake_obs)
        cnn_agent.act(obs)
        assert cnn_agent.last_features is not None
        assert cnn_agent.last_features.shape == (1, 512)

    def test_reset_clears_features(self, cnn_agent, fake_obs):
        """Dopo reset(), last_features è None."""
        prep = ImagePreprocessor(mode="rgb", image_size=84)
        obs = prep.process(fake_obs)
        cnn_agent.act(obs)
        cnn_agent.reset()
        assert cnn_agent.last_features is None

    def test_reproducibility(self, cnn, fake_obs):
        """Stessa sequenza di azioni dopo reset con stesso seed."""
        prep = ImagePreprocessor(mode="rgb", image_size=84)
        obs = prep.process(fake_obs)
        agent = CNNDummyAgent(action_space_size=6, cnn=cnn, device="cpu", seed=99)
        acts1 = [agent.act(obs) for _ in range(10)]
        agent.reset()
        acts2 = [agent.act(obs) for _ in range(10)]
        assert acts1 == acts2

    def test_feature_dim_property(self, cnn_agent):
        """feature_dim corrisponde a quello della CNN."""
        assert cnn_agent.feature_dim == 512

    def test_full_pipeline_cnn_rgb(self, cnn, fake_obs):
        """Pipeline completa con CNN e preprocessing RGB."""
        prep = ImagePreprocessor(mode="rgb", image_size=84)
        agent = CNNDummyAgent(action_space_size=6, cnn=cnn, device="cpu")
        obs = prep.process(fake_obs)
        action = agent.act(obs)
        assert obs.shape == (3, 84, 84)
        assert agent.last_features.shape == (1, 512)
        assert 0 <= action < 6

    def test_full_pipeline_cnn_grayscale(self, cnn, fake_obs):
        """Pipeline completa con CNN e preprocessing Grayscale."""
        prep = ImagePreprocessor(mode="grayscale", image_size=84)
        agent = CNNDummyAgent(action_space_size=6, cnn=cnn, device="cpu")
        obs = prep.process(fake_obs)
        action = agent.act(obs)
        assert obs.shape == (3, 84, 84)
        assert agent.last_features.shape == (1, 512)
        assert 0 <= action < 6
