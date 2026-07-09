##
## This file is copied from week 4
##

from collections import OrderedDict

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """Small MLP mapping CartPole observations to action values."""

    def __init__(
        self, obs_dim: int, n_actions: int, hidden_dim: int = 64, num_layers: int = 2
    ) -> None:
        super().__init__()
        layers = OrderedDict()
        layers["fc1"] = nn.Linear(obs_dim, hidden_dim)
        layers["relu1"] = nn.ReLU()

        for i in range(2, num_layers + 1):
            layers[f"fc{i}"] = nn.Linear(hidden_dim, hidden_dim)
            layers[f"relu{i}"] = nn.ReLU()

        layers["out"] = nn.Linear(hidden_dim, n_actions)
        self.net = nn.Sequential(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
