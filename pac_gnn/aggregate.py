import math
import random
from typing import List

import networkx
import numpy as np

from pac_gnn.pick import LabelBalancedSampler

random.seed(1212)


class MessagePassing:

    def __init__(
        self, G: networkx.Graph, embeddings: np.array, v_train: np.array, picks: int, epochs: int,
        batch_size: int, n_layers: int, dimension_size: int, n_relations: int,
        label_balanced_sampler: LabelBalancedSampler
    ):
        """Messsage Passing class object.

        Args:
            G (networkx.Graph): Input graph.
            embeddings (np.array): 2D array containing node embeddings.
            v_train (np.array): 1D array containing train set node indexes.
            picks (int): Number of nodes to pick in each epoch.
            epochs (int): Number of total training epochs.
            batch_size (int): Number of training batch size.
            n_layers (int): Number of layers.
            dimension_size (int): Dimension size for the l-th layer.
            label_balanced_sampler (LabelBalancedSampler): LabelBalancedSampler object.
        """

        self.G = G
        self.embeddings = embeddings
        self.v_train = v_train
        self.picks = picks
        self.epochs = epochs
        self.batch_size = batch_size if batch_size else G.number_of_nodes()
        self.n_layers = n_layers
        self.dimension_size = dimension_size
        self.n_relations = n_relations

        self.label_balanced_sampler = label_balanced_sampler

        self.h_v_l = np.zeros(
            shape=[
                G.number_of_nodes(), n_layers + 1,
                G.number_of_nodes(), embeddings.embedding_dim
            ],
            dtype=object
        )
        self.h_v_l[:, 0] = embeddings

        self.h_v_r_l = np.zeros(
            shape=[
                G.number_of_nodes(), n_relations + 1, n_layers + 1,
                G.number_of_nodes(), embeddings.embedding_dim
            ],
            dtype=object
        )
        self.h_v_r_l[:, 0, 0] = embeddings

        self.w = np.zeros(
            shape=[
                n_layers + 1, embeddings.embedding_dim, (n_relations + 1) * embeddings.embedding_dim
            ]
        )
        self.w_r = np.zeros(
            shape=[
                n_relations + 1, n_layers + 1, embeddings.embedding_dim, 2 *
                embeddings.embedding_dim
            ]
        )

    def _construct_subgraph(self, nodes_idx: List[int]) -> networkx.Graph:
        return self.G.subgraph(nodes_idx)

    @staticmethod
    def _generate_node_batches(nodes_idx: List[int], batch_size: int) -> List[int]:
        for i in range(0, len(nodes_idx), batch_size):
            yield nodes_idx[i:i + batch_size]

    @staticmethod
    def _relu(x: np.array) -> np.array:
        return x * (x > 0)

    def _update_h_v_r_l(self, batch_nodes: int, relation: int, layer: int) -> np.array:
        for v in batch_nodes:
            mean_agg = np.mean(self.h_v_r_l[list(self.G.neighbors(v)), relation, layer - 1], axis=0)

            self.h_v_r_l[v, relation, layer] = self._relu(
                np.dot(
                    self.w_r[relation, layer],
                    np.concatenate((self.h_v_r_l[v, relation, layer - 1], mean_agg), axis=0)
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
            V_picked = random.choices(
                list(self.G.nodes),
                weights=list(
                    map(lambda n: self.label_balanced_sampler.calculate_P(n), self.G.nodes)
                ),
                k=self.picks
            )

            batches = math.ceil(len(V_picked) / self.batch_size)

            for batch in range(batches):
                batch_nodes = list(self._generate_node_batches(V_picked, self.batch_size))[batch]
                sub_graph = self._construct_subgraph(batch_nodes)
                for layer in range(1, self.n_layers + 1):
                    for relation in range(1, self.n_relations + 1):
                        self._update_h_v_r_l(batch_nodes, relation, layer)
                    self._update_h_v_l(batch_nodes, layer)

        return self.h_v_l
