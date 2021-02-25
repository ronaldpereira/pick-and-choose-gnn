from pac_gnn.aggregate import MessagePassing
from typing import List, Tuple
import pytest
import networkx
import numpy as np


def _create_graph(nodes: List[int], edges: List[Tuple[int, int]]):
    G = networkx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    return G


@pytest.mark.parametrize(
    argnames=['graph', 'nodes_idx', 'expected_subgraph'],
    argvalues=[
        (
            _create_graph([0, 1, 2, 3, 4], [(0, 1), (1, 2), (1, 3), (2, 4),
                                            (3, 4)]), [0, 3, 4], _create_graph([0, 3, 4], [(3, 4)])
        ),
        (
            _create_graph([0, 1, 2, 3, 4], [(0, 1), (1, 2), (1, 3), (2, 4), (3, 4)]), [0, 1, 3, 4],
            _create_graph([0, 1, 3, 4], [(0, 1), (1, 3), (3, 4)])
        ),
        (
            _create_graph([0, 1, 2, 3, 4], [(0, 1), (1, 2), (1, 3), (2, 4),
                                            (3, 4)]), [0, 1, 4], _create_graph([0, 1, 4], [(0, 1)])
        )
    ]
)
def test_construct_subgraph(
    graph: networkx.Graph, nodes_idx: List[int], expected_subgraph: networkx.Graph
):
    message_passing = MessagePassing(graph, np.array([[]]), None, None, None, None, 1, None, None)

    sub_graph = message_passing._construct_subgraph(nodes_idx)

    assert expected_subgraph.nodes == sub_graph.nodes
    assert expected_subgraph.edges == sub_graph.edges
