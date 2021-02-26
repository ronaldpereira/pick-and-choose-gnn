import numpy as np
import torch
from torch import nn


class FullyConnectedDistanceNetwork(nn.Module):

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()

        self.fc = nn.Linear(in_dim, out_dim)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        x = self.activation(x)

        return x


class NeighborhoodSampler:

    def __init__(
        self, h_v_l: np.array, y: np.array, n_layers: int, rho_minority_class: float,
        rho_majority_class: float
    ):
        self.model = FullyConnectedDistanceNetwork(n_layers, 1)

        self._train_distance(h_v_l, y)

    def _train_distance(
        self, x: torch.tensor, y: torch.tensor, learning_rate: float = 1e-2, epochs: int = 100
    ):
        self.loss_fn = nn.CrossEntropyLoss(reduction='sum')

        for epoch in range(epochs):
            y_pred = self.model(x)

            loss = self.loss_fn(y_pred, y)
            print(epoch, loss.item())

            self.model.zero_grad()

            loss.backward()

            with torch.no_grad():
                for param in self.model.parameters():
                    param -= learning_rate * param.grad

    def _distance_function(self):
        pass

    def minority_class_sampling(self):
        pass
