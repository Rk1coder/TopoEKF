import numpy as np
from scipy.optimize import linear_sum_assignment


class HungarianAssociator:
    def __init__(self, max_cost: float = 9.21):
        self.max_cost = max_cost

    def associate(self, cost_matrix: np.ndarray) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if cost_matrix.size == 0:
            n_tracks = cost_matrix.shape[0] if cost_matrix.ndim == 2 else 0
            n_detections = cost_matrix.shape[1] if cost_matrix.ndim == 2 else 0
            return [], list(range(n_tracks)), list(range(n_detections))

        rows, cols = linear_sum_assignment(cost_matrix)
        matches: list[tuple[int, int]] = []
        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        for row, col in zip(rows, cols):
            if cost_matrix[row, col] <= self.max_cost:
                matches.append((int(row), int(col)))
                matched_tracks.add(int(row))
                matched_detections.add(int(col))

        unmatched_tracks = [idx for idx in range(cost_matrix.shape[0]) if idx not in matched_tracks]
        unmatched_detections = [idx for idx in range(cost_matrix.shape[1]) if idx not in matched_detections]
        return matches, unmatched_tracks, unmatched_detections
