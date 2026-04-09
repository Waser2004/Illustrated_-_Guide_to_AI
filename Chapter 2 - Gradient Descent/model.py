"""Simple CNN model factories for cube visibility tasks."""

from __future__ import annotations

import torch
from torch import nn


class SimpleCNNEncoder(nn.Module):
    """Lightweight CNN encoder for small regression/classification heads."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return x.flatten(1)


class SimpleCNNRegressor(nn.Module):
    """Simple CNN regressor with optional bounded output in [0, 1]."""

    def __init__(self, bounded_output: bool = True, freeze_backbone: bool = False) -> None:
        super().__init__()
        self.encoder = SimpleCNNEncoder()
        self.regression_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(64, 1),
        )
        self.output_activation = nn.Sigmoid() if bounded_output else nn.Identity()

        if freeze_backbone:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.regression_head(x)
        return self.output_activation(x)


def build_simple_cnn_regressor(
    bounded_output: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Build a simple CNN regressor for cube-visibility prediction."""
    return SimpleCNNRegressor(
        bounded_output=bounded_output,
        freeze_backbone=freeze_backbone,
    )
