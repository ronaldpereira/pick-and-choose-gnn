from itertools import combinations
from typing import List

import networkx
import numpy as np
from sklearn.metrics import f1_score
import torch
from torch import nn
from torch import optim


class FullyConnectedDistanceNetwork(nn.Module):

    def __init__(self, n_layers: int, d_l: int, n_relations: int):
        super().__init__()

        self.n_layers = n_layers
        self.d_l = d_l
        self.n_relations = n_relations

        self.fc = nn.Linear(self.d_l, 2)
        self.activation = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc(x)
        x = self.activation(x)
        x = self.softmax(x)

        return x


class NeighborhoodSampler:

    def __init__(self, G: networkx.graph, h: torch.tensor, y: np.array, n_relations: int):

        self.G = G

        self.h = h
        self.n_nodes = self.G.number_of_nodes()

        self.model = FullyConnectedDistanceNetwork(1, self.h.shape[1], n_relations)
        self._train_distance(self.h, torch.tensor(y))

    def _train_distance(
        self, x: torch.tensor, y: torch.tensor, learning_rate: float = 0.1, epochs: int = 100
    ):
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        for epoch in range(epochs):
            optimizer.zero_grad()

            y_pred = self.model(x)

            loss = loss_fn(y_pred, y)
            f1 = f1_score(
                y.detach().numpy(), torch.argmax(y_pred, dim=1).detach().numpy(), average='macro'
            )
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f'[{epoch}] loss={loss.item():.4f} f1={f1}')

            loss.backward()

            optimizer.step()

    def _get_nodes_from_class(self, label: int):
        return list(filter(lambda x: self.G.nodes[x]['label'] == label, self.G.nodes))

    def _calculate_rho_plus(self):
        nodes_ids = self._get_nodes_from_class(1)

        _, degrees = zip(*self.G.degree(nodes_ids))

        mean_degree = np.mean(degrees)

        distances = []
        for u, v in combinations(nodes_ids, r=2):
            distances.append(self._distance_function(u, v))

        top_k_distance = np.percentile(distances, 100 - 0.5 * mean_degree)

        return top_k_distance

    def _calculate_rho_minus(self, v: int, neighborhood: List[int]):
        distances = []
        for u in neighborhood:
            distances.append(self._distance_function(u, v))

        median_distance = np.percentile(distances, 50)

        return median_distance

    def _distance_function(self, u: int, v: int):

        u_y_pred = self.model(self.h[u].view(1, -1)).detach().numpy()
        v_y_pred = self.model(self.h[v].view(1, -1)).detach().numpy()
        distance = np.linalg.norm(v_y_pred - u_y_pred, ord=1)

        return distance

    def _oversample_neighborhood_function(self, v: int):
        sampled_G = networkx.Graph()
        sampled_G.add_node(v, **self.G.nodes[v])

        rho_plus = self._calculate_rho_plus()

        same_class_nodes = list(
            filter(lambda x: self.G.nodes[x]['label'] == self.G.nodes[v]['label'], self.G.nodes)
        )
        for u in same_class_nodes:
            if u != v:
                distance = self._distance_function(u, v)
                print(f'u={u} v={v} d(u,v)={distance} rho_plus={rho_plus}')
                if distance <= rho_plus:
                    sampled_G.add_node(u, **self.G.nodes[u])
                    sampled_G.add_edge(u, v)

        return sampled_G

    def _undersample_neighborhood_function(self, v: int):
        sampled_G = networkx.Graph()
        sampled_G.add_node(v, **self.G.nodes[v])

        neighborhood = list(self.G.adj[v].keys())
        rho_minus = self._calculate_rho_minus(v, neighborhood)

        for u in neighborhood:
            if u != v:
                distance = self._distance_function(u, v)
                print(f'u={u} v={v} d(u,v)={distance} rho_minus={rho_minus}')
                if distance <= rho_minus:
                    sampled_G.add_node(u, **self.G.nodes[u])
                    sampled_G.add_edge(u, v)

        return sampled_G

    def _majority_class_sampler(self, v: int):
        print('undersample')
        undersampled_G = self._undersample_neighborhood_function(v)
        print(undersampled_G.nodes)
        print(undersampled_G.edges)

        return undersampled_G

    def _minority_class_sampler(self, v: int):
        print('undersample')
        undersampled_G = self._undersample_neighborhood_function(v)
        print(undersampled_G.nodes)
        print(undersampled_G.edges)

        print('oversample')
        oversampled_G = self._oversample_neighborhood_function(v)
        print(oversampled_G.nodes)
        print(oversampled_G.edges)

        print('final graph')
        final_G = networkx.compose(undersampled_G, oversampled_G)
        print(final_G.nodes)
        print(final_G.edges)

        return final_G

    def node_sampler(self, v: int):
        if self.G.nodes[v]['label'] == 0:
            print('majority class')
            return self._majority_class_sampler(v)
        else:
            print('minority class')
            return self._minority_class_sampler(v)
