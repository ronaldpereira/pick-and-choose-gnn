from typing import List

import networkx
import numpy as np
import torch
from torch import nn


class FullyConnectedDistanceNetwork(nn.Module):

    def __init__(self, n_layers: int, d_l: int, n_relations: int):
        super().__init__()

        self.n_layers = n_layers
        self.d_l = d_l
        self.n_relations = n_relations

        self.fc = nn.Linear(self.d_l, 1)
        nn.init.normal_(self.fc.weight)
        self.activation = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.fc(x)
        x = self.activation(x)
        x = self.softmax(x)

        return x


class NeighborhoodSampler:

    def __init__(
        self, G: networkx.graph, h: torch.tensor, y: np.array, n_relations: int, n_nodes: int
    ):

        self.G = G

        self.h = h
        self.n_nodes = n_nodes

        self.model = FullyConnectedDistanceNetwork(1, self.h.shape[2], n_relations)

        self._train_distance(self.h, torch.tensor([y]))

        self.rho_plus = self.minority_class_sampling()
        print(self.rho_plus)
        self.rho_minus = self.majority_class_sampling()
        print(self.rho_minus)

    def _train_distance(
        self, x: torch.tensor, y: torch.tensor, learning_rate: float = 0.1, epochs: int = 100
    ):
        for epoch in range(epochs):
            y_pred = self.model(x)

            loss = self.model.loss_fn(y_pred, y.view(-1, 1))
            print(epoch, loss.item())

            self.model.zero_grad()

            loss.backward()

            with torch.no_grad():
                for param in self.model.parameters():
                    param -= learning_rate * param.grad

    def _get_nodes_from_class(self, label: int):
        return list(filter(lambda x: self.G.nodes[x]['label'] == label, self.G.nodes))

    def minority_class_sampling(self):
        # calculates the rho_plus hyperparameter
        nodes_ids = self._get_nodes_from_class(1)

        _, degrees = zip(*self.G.degree(nodes_ids))

        mean_degree = np.mean(degrees)

        return 0.5 * mean_degree

    def majority_class_sampling(self):
        # calculates the rho_minus hyperparameter
        nodes_ids = self._get_nodes_from_class(0)

        _, degrees = zip(*self.G.degree(nodes_ids))

        median_degree = np.median(degrees)

        return median_degree
