from __future__ import annotations

import torch as th
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CnnExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space: spaces.Box, features_dim: int = 128) -> None:
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = th.nn.Sequential(
            th.nn.Conv2d(n_input_channels, 16, kernel_size=3, padding=1),
            th.nn.ReLU(),
            th.nn.MaxPool2d(2),
            th.nn.Conv2d(16, 32, kernel_size=3, padding=1),
            th.nn.ReLU(),
            th.nn.MaxPool2d(2),
            th.nn.Flatten(),
        )
        with th.no_grad():
            dummy = th.zeros(1, *observation_space.shape)
            n_flatten = self.cnn(dummy).shape[1]
        self.linear = th.nn.Sequential(
            th.nn.Linear(n_flatten, features_dim),
            th.nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))
