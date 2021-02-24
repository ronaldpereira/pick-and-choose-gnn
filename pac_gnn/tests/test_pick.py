import numpy as np
from pac_gnn.pick import LabelBalancedSampler
import pytest


@pytest.mark.parametrize(
    argnames=['A', 'expected_D'],
    argvalues=[
        (np.array([[0, 1, 1], [1, 0, 1], [0, 1, 1]]), np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])),
        (np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]), np.array([[1, 0, 0], [0, 2, 0], [0, 0, 1]]))
    ]
)
def test_calculate_D(A: np.array, expected_D: np.array):
    D = LabelBalancedSampler(A, np.array([]))._calculate_D()

    assert np.allclose(expected_D, D)


@pytest.mark.parametrize(
    argnames=['A', 'expected_A_hat'],
    argvalues=[
        (
            np.array([[0, 1, 1], [1, 0, 1], [0, 1, 1]]),
            np.array([[0, 0.5, 0.5], [0.5, 0, 0.5], [0, 0.5, 0.5]]),
        ),
        (
            np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]),
            np.array([[0, 0.70710678, 0], [0.70710678, 0, 0.70710678], [0, 0.70710678, 0]]),
        )
    ]
)
def test_calculate_A_hat(A: np.array, expected_A_hat: np.array):
    A_hat = LabelBalancedSampler(A, np.array([]))._calculate_A_hat()

    assert np.allclose(expected_A_hat, A_hat)


@pytest.mark.parametrize(
    argnames=['A', 'labels', 'node_idx', 'expected_node_frequency'],
    argvalues=[
        (np.array([[0, 1, 1], [1, 0, 1], [0, 1, 1]]), np.array([0, 1, 0]), 0, 2 / 3),
        (np.array([[0, 1, 1], [1, 0, 1], [0, 1, 1]]), np.array([0, 1, 0]), 1, 1 / 3),
        (np.array([[0, 1, 1], [1, 0, 1], [0, 1, 1]]), np.array([0, 1, 0]), 2, 2 / 3),
        (np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]), np.array([0, 0, 1]), 0, 2 / 3),
        (np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]), np.array([0, 0, 1]), 1, 2 / 3),
        (np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]), np.array([0, 0, 1]), 2, 1 / 3),
    ]
)
def test_calculate_A_hat(
    A: np.array, labels: np.array, node_idx: int, expected_node_frequency: float
):
    node_frequency = LabelBalancedSampler(A, labels)._node_label_frequency(node_idx)

    assert expected_node_frequency == node_frequency


@pytest.mark.parametrize(
    argnames=['A', 'labels', 'node_idx', 'expected_P'],
    argvalues=[
        (np.array([[0, 1, 1], [1, 0, 1], [0, 1, 1]]), np.array([0, 1, 0]), 0, 0.24999999999999994),
        (np.array([[0, 1, 1], [1, 0, 1], [0, 1, 1]]), np.array([0, 1, 0]), 1, 0.7071067811865474),
        (np.array([[0, 1, 1], [1, 0, 1], [0, 1, 1]]), np.array([0, 1, 0]), 2, 0.43301270189221924),
        (np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]), np.array([0, 0, 1]), 0, 0.35355339059327373),
        (np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]), np.array([0, 0, 1]), 1, 0.49999999999999994),
        (np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]), np.array([0, 0, 1]), 2, 0.7071067811865475),
    ]
)
def test_calculate_A_hat(A: np.array, labels: np.array, node_idx: int, expected_P: float):
    P = LabelBalancedSampler(A, labels).calculate_P(node_idx)

    assert expected_P == P
