from pac_gnn.aggregate import MessagePassing
from typing import List, Tuple
import pytest
import networkx
import numpy as np
from torch import nn
import torch


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
    message_passing = MessagePassing(
        graph, torch.randn(size=[graph.number_of_nodes(), 10]), None, None, 1, 1, 1, 1, 1, 1, None,
        None
    )

    sub_graph = message_passing._construct_subgraph(nodes_idx)

    assert expected_subgraph.nodes == sub_graph.nodes
    assert expected_subgraph.edges == sub_graph.edges


@pytest.mark.parametrize(
    argnames=['nodes_idx', 'batch_size', 'expected_nodes_idx_batches'],
    argvalues=[
        ([0, 1, 2, 3, 4], 2, [[0, 1], [2, 3], [4]]), ([0, 1, 2, 3, 4], 3, [[0, 1, 2], [3, 4]]),
        ([0, 1, 2, 3, 4], 5, [[0, 1, 2, 3, 4]]), ([0, 1, 2, 3, 4], 1, [[0], [1], [2], [3], [4]]),
        ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 5, [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10]]),
        ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 11, [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]),
        ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 12, [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    ]
)
def test_generate_node_batches(
    nodes_idx: List[int], batch_size: int, expected_nodes_idx_batches: List[List[int]]
):
    message_passing = MessagePassing(
        networkx.Graph(), torch.randn(size=[networkx.Graph().number_of_nodes(), 10]), None, None, 1,
        1, 1, 1, 1, 1, None, None
    )

    nodes_idx_batches = list(message_passing._generate_node_batches(nodes_idx, batch_size))

    assert expected_nodes_idx_batches == nodes_idx_batches
