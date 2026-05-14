"""
CNN Backbone — estrattore di features visive per agenti RL su Atari.

Architettura default: Nature DQN (Mnih et al., 2015)
    Conv2d(in_channels, 32, kernel=8, stride=4)  → pattern grossolani
    Conv2d(32, 64, kernel=4, stride=2)            → pattern medi
    Conv2d(64, 64, kernel=3, stride=1)            → pattern fini
    Flatten
    Linear(conv_out, feature_dim)                 → feature vector

Il modulo è indipendente dall'agente: riceve un'osservazione
preprocessata (B, C, H, W) e restituisce un vettore di features (B, feature_dim).
Gli agenti futuri (DQN, PPO, ecc.) useranno questo backbone come encoder.
"""

import torch
import torch.nn as nn
from typing import List, Tuple


class CNNBackbone(nn.Module):
    """
    CNN estrattore di features per osservazioni visive Atari.

    Input:  (B, C, H, W)  float32  valori in [0, 1]
    Output: (B, feature_dim)  float32

    Parametri configurabili via YAML:
        in_channels:   canali input  (default 3, o 12 con frame_stack=4)
        feature_dim:   dimensione feature vector output (default 512)
        conv_layers:   lista di dict con keys [out_channels, kernel_size, stride]
    """

    def __init__(
        self,
        in_channels: int = 3,
        feature_dim: int = 512,
        conv_configs: List[dict] = None,
    ):
        """
        Args:
            in_channels: Canali dell'osservazione input (C in C x H x W).
            feature_dim: Dimensione del vettore di features in uscita.
            conv_configs: Lista di configurazioni per i layer convoluzionali.
                          Ogni elemento è un dict con chiavi:
                            - out_channels (int)
                            - kernel_size  (int)
                            - stride       (int)
                          Se None, usa l'architettura Nature DQN di default.
        """
        super().__init__()

        if conv_configs is None:
            # Architettura Nature DQN (Mnih et al., 2015)
            conv_configs = [
                {"out_channels": 32, "kernel_size": 8, "stride": 4},
                {"out_channels": 64, "kernel_size": 4, "stride": 2},
                {"out_channels": 64, "kernel_size": 3, "stride": 1},
            ]

        self.in_channels = in_channels
        self.feature_dim = feature_dim
        self.conv_configs = conv_configs

        # Costruisce i layer convoluzionali dinamicamente
        conv_layers = []
        current_channels = in_channels
        for cfg in conv_configs:
            conv_layers.append(
                nn.Conv2d(
                    in_channels=current_channels,
                    out_channels=cfg["out_channels"],
                    kernel_size=cfg["kernel_size"],
                    stride=cfg["stride"],
                )
            )
            conv_layers.append(nn.ReLU(inplace=True))
            current_channels = cfg["out_channels"]

        self.conv = nn.Sequential(*conv_layers)

        # Calcola la dimensione dell'output convoluzionale con un forward pass dummy
        self._conv_out_dim = self._compute_conv_out_dim()

        # Layer fully connected: conv_out → feature_dim
        self.fc = nn.Sequential(
            nn.Linear(self._conv_out_dim, feature_dim),
            nn.ReLU(inplace=True),
        )

    def _compute_conv_out_dim(self) -> int:
        """
        Calcola automaticamente la dimensione dell'output dei layer conv
        passando un tensore dummy. Funziona con qualsiasi configurazione.
        """
        with torch.no_grad():
            dummy = torch.zeros(1, self.in_channels, 84, 84)
            out = self.conv(dummy)
            return int(out.view(1, -1).shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Tensore (B, C, H, W) float32, valori in [0, 1].

        Returns:
            Tensore (B, feature_dim) float32.
        """
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)   # Flatten: (B, conv_out_dim)
        x = self.fc(x)               # (B, feature_dim)
        return x

    def summary(self) -> str:
        """Restituisce una stringa descrittiva dell'architettura."""
        lines = [
            f"CNNBackbone",
            f"  Input:       ({self.in_channels}, 84, 84)",
            f"  Conv layers: {len(self.conv_configs)}",
        ]
        ch = self.in_channels
        for i, cfg in enumerate(self.conv_configs):
            lines.append(
                f"    [{i+1}] Conv2d({ch}, {cfg['out_channels']}, "
                f"kernel={cfg['kernel_size']}, stride={cfg['stride']}) + ReLU"
            )
            ch = cfg["out_channels"]
        lines.append(f"  Flatten:     → {self._conv_out_dim}")
        lines.append(f"  FC:          {self._conv_out_dim} → {self.feature_dim} + ReLU")
        lines.append(f"  Output:      (B, {self.feature_dim})")
        n_params = sum(p.numel() for p in self.parameters())
        lines.append(f"  Parametri:   {n_params:,}")
        return "\n".join(lines)
    
    def get_feature_dim(self) -> int:
        return self.feature_dim


def build_cnn_from_config(config) -> CNNBackbone:
    """
    Costruisce un CNNBackbone dalla sezione `model` della config.

    Args:
        config: Oggetto Config con sezione `model`.

    Returns:
        Istanza di CNNBackbone configurata.
    """
    m = config.model

    # Calcola in_channels in base a preprocessing (frame_stack * 3)
    in_channels = config.preprocessing.frame_stack * 3

    conv_configs = None
    if hasattr(m, "conv_layers") and m.conv_layers is not None:
        conv_configs = [
            {
                "out_channels": layer["out_channels"] if isinstance(layer, dict) else layer.out_channels,
                "kernel_size": layer["kernel_size"] if isinstance(layer, dict) else layer.kernel_size,
                "stride": layer["stride"] if isinstance(layer, dict) else layer.stride,
            }
            for layer in m.conv_layers
        ]

    return CNNBackbone(
        in_channels=in_channels,
        feature_dim=m.feature_dim,
        conv_configs=conv_configs,
    )
