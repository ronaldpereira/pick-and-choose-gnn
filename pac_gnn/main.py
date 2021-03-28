from typing import Tuple

import networkx
from networkx.linalg.graphmatrix import adjacency_matrix
import numpy as np
import torch
from torch import nn

from pac_gnn.aggregate import MessagePassing
from pac_gnn.choose import NeighborhoodSampler
from pac_gnn.pick import LabelBalancedSampler

torch.random.manual_seed(1212)


def _create_graph() -> Tuple[networkx.Graph, np.array]:
    G = networkx.Graph()

    G.add_node(0, id='a', label=0)
    G.add_node(1, id='b', label=0)
    G.add_node(2, id='c', label=0)
    G.add_node(3, id='d', label=0)
    G.add_node(4, id='e', label=0)
    G.add_node(5, id='f', label=0)
    G.add_node(6, id='g', label=0)
    G.add_node(7, id='h', label=0)
    G.add_node(8, id='u', label=1)
    G.add_node(9, id='v', label=1)
    G.add_node(10, id='w', label=0)

    G.add_edge(0, 1)  # a -> b
    G.add_edge(0, 9)  # a -> v
    G.add_edge(1, 9)  # b -> v
    G.add_edge(9, 2)  # v -> c
    G.add_edge(9, 3)  # v -> d
    G.add_edge(2, 3)  # c -> d
    G.add_edge(3, 10)  # d -> w
    G.add_edge(3, 4)  # d -> e
    G.add_edge(10, 4)  # w -> e
    G.add_edge(10, 8)  # w -> u
    G.add_edge(4, 8)  # e -> u
    G.add_edge(8, 7)  # u -> h
    G.add_edge(8, 5)  # u -> f
    G.add_edge(8, 6)  # u -> g
    G.add_edge(5, 6)  # f -> g

    labels = np.array(list(map(lambda n: G.nodes[n]['label'], G.nodes)))

    return G, labels


def main():
    G, labels = _create_graph()

    label_balanced_sampler = LabelBalancedSampler(
        A=np.array(adjacency_matrix(G).todense()), labels=labels
    )

    for u in G.nodes:
        print(u, label_balanced_sampler.calculate_P(u))

    features = torch.randn(size=[G.number_of_nodes(), 10])

    neighborhood_sampler = NeighborhoodSampler(G=G, h=features, y=labels, n_relations=1)

    neighborhood_sampler.node_sampler(3)
    neighborhood_sampler.node_sampler(9)

    message_passing = MessagePassing(
        G=G,
        features=features,
        v_train=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        labels=labels,
        epochs=100,
        picks=11,
        batch_size=11,
        n_layers=5,
        dimension_size=2,
        n_relations=1,
        label_balanced_sampler=label_balanced_sampler,
        neighborhood_sampler=neighborhood_sampler
    )

    print(message_passing.execute())


if __name__ == '__main__':
    main()
