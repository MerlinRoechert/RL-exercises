from collections import OrderedDict

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """
    A simple MLP mapping state → Q‐values for each action.

    Architecture:
      Input → Linear(obs_dim→hidden_dim) → ReLU
            → Linear(hidden_dim→hidden_dim) → ReLU
            → Linear(hidden_dim→n_actions)
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 64, num_layers: int = 2) -> None:
        """
        Parameters
        ----------
        obs_dim : int
            Dimensionality of observation space.
        n_actions : int
            Number of discrete actions.
        hidden_dim : int
            Hidden layer size.
        """
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
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Batch of states, shape (batch, obs_dim).

        Returns
        -------
        torch.Tensor
            Q‐values, shape (batch, n_actions).
        """
        return self.net(x)
