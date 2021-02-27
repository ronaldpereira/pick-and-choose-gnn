from typing import Tuple

import networkx
from networkx.linalg.graphmatrix import adjacency_matrix
import numpy as np
from torch import nn

from pac_gnn.aggregate import MessagePassing
from pac_gnn.pick import LabelBalancedSampler


def _create_graph() -> Tuple[networkx.Graph, np.array]:
    G = networkx.Graph()

    G.add_node(0, label=0)
    G.add_node(1, label=0)
    G.add_node(2, label=1)

    G.add_edges_from([[0, 1], [1, 2]])

    labels = np.array(list(map(lambda n: G.nodes[n]['label'], G.nodes)))

    return G, labels


def main():
    G, labels = _create_graph()

    label_balanced_sampler = LabelBalancedSampler(np.array(adjacency_matrix(G).todense()), labels)

    embeddings = nn.Embedding(G.number_of_nodes(), 5)

    message_passing = MessagePassing(
        G, embeddings, [0, 2], 2, 10, 3, 2, 2, 1, label_balanced_sampler
    )

    print(message_passing.execute())


if __name__ == '__main__':
    main()
