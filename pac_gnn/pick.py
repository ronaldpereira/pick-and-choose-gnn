import numpy as np
from collections import Counter


class LabelBalancedSampler:

    def __init__(self, A: np.array, labels: np.array):
        """Label Balanced Sampler object to pick phase.

        Args:
            A (np.array): 2D array graph adjacency matrix.
            labels (np.array): 1D array label vector for each node in the graph.
        """

        self.A = A
        self.D = self._calculate_D()
        self.A_hat = self._calculate_A_hat()

        self.labels = labels
        self.labels_frequency = Counter(self.labels)

    def _calculate_D(self) -> np.array:
        """Calculates D, which is a diagonal matrix with degree of each node as its element.

        Returns:
            np.array: Diagonal matrix of the graph.
        """

        D = np.diag(self.A.sum(axis=1))

        return D

    def _calculate_A_hat(self) -> np.array:
        """Calculated A_hat, which is the normalized adjancency matrix.

        Returns:
            np.array: A_hat matrix.
        """

        D_sqrt_inverse = np.diag(1 / np.sqrt(np.diag(self.D)))

        A_hat = np.array(D_sqrt_inverse @ self.A @ D_sqrt_inverse)

        return A_hat

    def _node_label_frequency(self, node_idx: int) -> float:
        node_label_count = self.labels_frequency[self.labels[node_idx]]

        return node_label_count

    def calculate_P(self, node_idx: int) -> float:
        prob = np.linalg.norm(self.A_hat[:, node_idx], ord=2) / self._node_label_frequency(node_idx)

        return prob
