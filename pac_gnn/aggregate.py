import math
import random
from typing import List
from itertools import combinations

import networkx
import numpy as np

from pac_gnn.pick import LabelBalancedSampler

random.seed(1212)


class MessagePassing:

    def __init__(
        self, G: networkx.Graph, features: np.array, v_train: np.array, picks: int, epochs: int,
        batch_size: int, n_layers: int, dimension_size: int,
        label_balanced_sampler: LabelBalancedSampler
    ):
        """Messsage Passing class object.

        Args:
            G (networkx.Graph): Input graph.
            features (np.array): 2D array containing node features.
            v_train (np.array): 1D array containing train set node indexes.
            picks (int): Number of nodes to pick in each epoch.
            epochs (int): Number of total training epochs.
            batch_size (int): Number of training batch size.
            n_layers (int): Number of layers.
            dimension_size (int): Dimension size for the l-th layer.
            label_balanced_sampler (LabelBalancedSampler): LabelBalancedSampler object.
        """

        self.G = G
        self.features = features
        self.v_train = v_train
        self.picks = picks
        self.epochs = epochs
        self.batch_size = batch_size if batch_size else len(G.nodes)
        self.n_layers = n_layers
        self.dimension_size = dimension_size
        self.label_balanced_sampler = label_balanced_sampler

        self.h_v = np.zeros(shape=[n_layers, features.shape[0], features.shape[1]])
        self._init_h_v_0()

        self.weights = np.zeros(shape=[n_layers, features.shape[0], features.shape[1]])

    def _init_h_v_0(self):
        self.h_v[0] = self.features

    def _construct_subgraph(self, nodes_idx: List[int]) -> networkx.Graph:
        return self.G.subgraph(nodes_idx)

    def _generate_node_batches(self, nodes_idx: List[int], batch_size: int):
        for i in range(0, len(nodes_idx), batch_size):
            yield nodes_idx[i:i + batch_size]

    @staticmethod
    def _relu(matrix: np.array) -> np.array:
        return matrix * (matrix > 0)

    def _calculate_h_v_l(self, layer: int) -> np.array:
        return self._relu(
            self.weights[layer] @ np.concatenate(self.h_v[layer - 1], self.h_v[layer])
        )

    def execute(self):
        for epoch in range(self.epochs):
            V_picked = random.choices(
                self.G.nodes,
                weights=list(
                    map(lambda n: self.label_balanced_sampler.calculate_P(n), self.G.nodes)
                ),
                k=self.picks
            )

            batches = math.ceil(len(V_picked) / self.batch_size)

            # TODO: Continue implementing this part
            for batch in batches:
                batch_nodes = self._generate_node_batches(V_picked, self.batch_size)
                sub_graph = self._construct_subgraph(batch_nodes)
                for layer in range(1, self.n_layers + 1):
                    # TODO: Implement the choose step to calculate this part
                    #for relation in self.n_relations:
                    self.h_v[layer] = self._calculate_h_v_l(layer)
