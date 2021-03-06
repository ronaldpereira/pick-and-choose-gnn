import numpy as np
import torch
from torch import nn
from typing import List


class FullyConnectedDistanceNetwork(nn.Module):

    def __init__(self, n_layers: int, n_relations: int):
        super().__init__()

        self.n_layers = n_layers
        self.n_relations = n_relations

        self.fc = nn.Linear(n_layers, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        x = self.activation(x)

        return x

    def custom_cross_entropy_loss(
        self, target: torch.tensor, D_r: torch.tensor, h: np.array, n_nodes: List[int]
    ) -> float:
        result_along_layers = np.zeros(shape=[self.n_layers])
        for l in self.n_layers:
            for r in self.n_relations:
                for v in range(n_nodes):
                    result_along_layers.append(target[v] * np.log2(D_r[l] * h[v][r][l]))

        loss_dist = -torch.sum(result_along_layers)

        return loss_dist


class NeighborhoodSampler:

    def __init__(self, h: np.array, y: np.array, n_layers: int, n_nodes: int):
        self.model = FullyConnectedDistanceNetwork(n_layers)

        self.h = h
        self.n_nodes = n_nodes

        self._train_distance(self.h, y)

    def _train_distance(
        self, x: torch.tensor, y: torch.tensor, learning_rate: float = 1e-2, epochs: int = 100
    ):
        for epoch in range(epochs):
            y_pred = self.model(x)

            loss = self.model.custom_cross_entropy_loss(
                y, self.model[0].weight, self.h, self.n_nodes
            )
            print(epoch, loss)

            self.model.zero_grad()

            loss.backward()

            with torch.no_grad():
                for param in self.model.parameters():
                    param -= learning_rate * param.grad

    def _distance_function(self):
        pass

    def minority_class_sampling(self):
        pass

    def majority_class_sampling(self):
        pass
