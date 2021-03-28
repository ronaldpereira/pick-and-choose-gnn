import math
import random
from typing import List

import networkx
import numpy as np
import torch

from pac_gnn.choose import NeighborhoodSampler
from pac_gnn.pick import LabelBalancedSampler

random.seed(1212)


class MessagePassing:

    def __init__(
        self, G: networkx.Graph, features: torch.tensor, v_train: np.array, labels: np.array,
        epochs: int, picks: int, batch_size: int, n_layers: int, dimension_size: int,
        n_relations: int, label_balanced_sampler: LabelBalancedSampler,
        neighborhood_sampler: NeighborhoodSampler
    ):
        """Messsage Passing class object.

        Args:
            G (networkx.Graph): Input graph.
            features (torch.tensor): 2D tensor containing node features.
            v_train (np.array): 1D array containing train set node indexes.
            labels (np.array): Numpy array containing all labels.
            epochs (int): Number of total training epochs.
            picks (int): Number of nodes to pick in each epoch.
            batch_size (int): Number of training batch size.
            n_layers (int): Number of layers.
            dimension_size (int): Dimension size for the l-th layer.
            label_balanced_sampler (LabelBalancedSampler): LabelBalancedSampler object.
            neighborhood_sampler (NeighborhoodSampler): NeighborhoodSampler object.
        """

        self.G = G
        self.features = features.detach().numpy()
        self.v_train = v_train
        self.epochs = epochs
        self.picks = picks
        self.batch_size = batch_size if batch_size else G.number_of_nodes()
        self.n_layers = n_layers
        self.dimension_size = dimension_size
        self.n_relations = n_relations
        self.labels = labels

        self.label_balanced_sampler = label_balanced_sampler

        self.h_v_l = np.zeros(
            shape=[
                G.number_of_nodes(), n_layers + 1, self.features.shape[0], self.features.shape[1]
            ],
            dtype=float
        )
        self.h_v_l[:, 0] = self.features

        self.h_v_r_l = np.zeros(
            shape=[
                G.number_of_nodes(), n_relations + 1, n_layers + 1, self.features.shape[0],
                self.features.shape[1]
            ],
            dtype=float
        )
        self.h_v_r_l[:, :, 0] = self.features

        self.w = np.ones(shape=[n_layers + 1, features.shape[1], features.shape[0]])

        self.w_r = np.ones(
            shape=[n_relations + 1, n_layers + 1, features.shape[1], features.shape[0]]
        )

        self.neighborhood_sampler = neighborhood_sampler

    def _construct_subgraph(self, nodes_idx: List[int]) -> networkx.Graph:
        return self.G.subgraph(nodes_idx)

    @staticmethod
    def _generate_node_batches(nodes_idx: List[int], batch_size: int) -> List[int]:
        for i in range(0, len(nodes_idx), batch_size):
            yield nodes_idx[i:i + batch_size]

    @staticmethod
    def _relu(x: np.array) -> np.array:
        return x * (x > 0)

    def _update_h_v_r_l(
        self, v: int, v_graph: networkx.Graph, relation: int, layer: int
    ) -> np.array:
        mean_agg = np.mean(self.h_v_r_l[list(v_graph.neighbors(v)), relation, layer - 1], axis=0)

        self.h_v_r_l[v, relation, layer] = self._relu(
            np.dot(
                self.w_r[relation, layer],
                np.concatenate((self.h_v_r_l[v, relation, layer - 1], mean_agg), axis=1)
            )
        )

    def _update_h_v_l(self, batch_nodes: int, layer: int) -> np.array:
        for v in batch_nodes:
            concat_result = self.h_v[layer - 1]
            for relation in self.n_relations:
                concat_result = np.concatenate((concat_result, self.h_v_r_l[v, relation, layer]))

            self.h_v_l[v, layer] = self._relu(self.w[layer] @ concat_result)

    def execute(self):
        for epoch in range(self.epochs):
            probs = list(map(lambda n: self.label_balanced_sampler.calculate_P(n), self.G.nodes))
            probs /= np.sum(probs)
            V_picked = np.random.choice(list(self.G.nodes), size=self.picks, replace=False, p=probs)

            batches = math.ceil(len(V_picked) / self.batch_size)

            for batch in range(batches):
                batch_nodes = list(self._generate_node_batches(V_picked, self.batch_size))[batch]
                sub_graph = self._construct_subgraph(batch_nodes)
                for layer in range(1, self.n_layers + 1):
                    for relation in range(1, self.n_relations + 1):
                        for v in batch_nodes:
                            v_graph = self.neighborhood_sampler.node_sampler(v)
                            self._update_h_v_r_l(v, v_graph, relation, layer)
                    self._update_h_v_l(batch_nodes, layer)

        return self.h_v_l
